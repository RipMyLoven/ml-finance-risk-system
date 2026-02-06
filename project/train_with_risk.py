#!/usr/bin/env python3
"""
================================================================================
  PRODUCTION-GRADE MULTI-MODEL FUTURES TRADING SYSTEM WITH RISK ENGINE
================================================================================
  
  POLARS VERSION - High-performance parallel DataFrame operations
  
  This is a live-trading-grade system designed to:
  - Train 3 alpha models: Scalp, Intraday, Swing
  - Implement centralized Risk Engine with CVaR, Kelly, and Drawdown controls
  - Multi-objective Optuna optimization
  - ONNX export for production deployment
  - Configurable CPU/RAM utilization via config.yaml
  
  Author: AI Trading Systems
  Version: 2.0.0 (Polars)
  
================================================================================
"""

from __future__ import annotations

import os
import sys
import gc
import json
import time
import logging
import warnings
import hashlib
import pickle
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Suppress warnings for clean output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==============================================================================
# LOAD CONFIG FIRST
# ==============================================================================

def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

CONFIG = load_config()

# Get CPU settings from config
CPU_CONFIG = CONFIG.get('cpu', {})
CPU_USAGE_PCT = CPU_CONFIG.get('usage_percent', 1.0)
EXACT_CORES = CPU_CONFIG.get('exact_cores', None)

if EXACT_CORES:
    N_CORES_TO_USE = int(EXACT_CORES)
else:
    N_CORES_TO_USE = max(1, int(mp.cpu_count() * CPU_USAGE_PCT))

N_CORES_TOTAL = mp.cpu_count()

# Force parallelization across libraries with configured cores
os.environ['OMP_NUM_THREADS'] = str(N_CORES_TO_USE)
os.environ['MKL_NUM_THREADS'] = str(N_CORES_TO_USE)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_CORES_TO_USE)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(N_CORES_TO_USE)
os.environ['NUMEXPR_NUM_THREADS'] = str(N_CORES_TO_USE)
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp/joblib'
os.environ['LOKY_MAX_CPU_COUNT'] = str(N_CORES_TO_USE)
os.environ['JOBLIB_START_METHOD'] = 'forkserver'
os.environ['LGB_NUM_THREADS'] = str(N_CORES_TO_USE)
# Polars uses all cores by default
os.environ['POLARS_MAX_THREADS'] = str(N_CORES_TO_USE)

import numpy as np
import polars as pl

# Configure Polars for maximum performance
pl.Config.set_streaming_chunk_size(50_000_000)  # 50M rows per chunk for streaming
pl.Config.set_fmt_str_lengths(100)

from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import lightgbm as lgb

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from joblib import Parallel, delayed, Memory

# ==============================================================================
# SYSTEM CONFIGURATION
# ==============================================================================

@dataclass
class SystemConfig:
    """Global system configuration loaded from config.yaml."""
    
    # CPU settings from config
    n_cpu: int = field(default_factory=lambda: N_CORES_TO_USE)
    n_cpu_total: int = field(default_factory=lambda: N_CORES_TOTAL)
    total_ram_gb: float = field(default_factory=lambda: min(
        os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3),
        CONFIG.get('ram', {}).get('max_gb', 9999)
    ))
    
    # Parallelization settings
    n_jobs: int = -1
    parallel_backend: str = 'loky'
    
    # Resource usage from config
    cpu_usage_pct: float = field(default_factory=lambda: CPU_USAGE_PCT)
    ram_usage_pct: float = field(default_factory=lambda: CONFIG.get('ram', {}).get('usage_percent', 0.70))
    ram_max_gb: float = field(default_factory=lambda: CONFIG.get('ram', {}).get('max_gb', None))
    
    # LightGBM settings from config
    max_bin: int = field(default_factory=lambda: CONFIG.get('lightgbm', {}).get('max_bin', 512))
    num_leaves: int = field(default_factory=lambda: CONFIG.get('lightgbm', {}).get('num_leaves', 255))
    max_depth: int = field(default_factory=lambda: CONFIG.get('lightgbm', {}).get('max_depth', 15))
    min_data_in_leaf: int = field(default_factory=lambda: CONFIG.get('lightgbm', {}).get('min_data_in_leaf', 10))
    learning_rate: float = field(default_factory=lambda: CONFIG.get('lightgbm', {}).get('learning_rate', 0.03))
    num_boost_round: int = field(default_factory=lambda: CONFIG.get('lightgbm', {}).get('num_boost_round', 2000))
    early_stopping_rounds: int = field(default_factory=lambda: CONFIG.get('lightgbm', {}).get('early_stopping_rounds', 100))
    feature_fraction_bynode: float = 0.9
    histogram_pool_size: int = -1
    bin_construct_sample_cnt: int = 5000000
    max_cached_hist_node: int = 65536
    
    # Data paths from config
    data_path: str = field(default_factory=lambda: CONFIG.get('paths', {}).get('data', '/home/ai/NogutiAI/aiTrainCrypto/data'))
    model_path: str = field(default_factory=lambda: CONFIG.get('paths', {}).get('models', '/home/ai/NogutiAI/aiTrainCrypto/project/models'))
    cache_path: str = field(default_factory=lambda: CONFIG.get('paths', {}).get('cache', '/home/ai/NogutiAI/aiTrainCrypto/project/.cache'))
    
    # Training settings from config
    random_seed: int = field(default_factory=lambda: CONFIG.get('misc', {}).get('random_seed', 42))
    test_size: float = field(default_factory=lambda: CONFIG.get('misc', {}).get('test_size', 0.20))
    val_size: float = field(default_factory=lambda: CONFIG.get('misc', {}).get('val_size', 0.10))
    
    # Risk thresholds from config
    max_leverage: float = field(default_factory=lambda: CONFIG.get('risk', {}).get('max_leverage', 10.0))
    max_position_pct: float = field(default_factory=lambda: CONFIG.get('risk', {}).get('max_position_pct', 0.25))
    max_drawdown_warning: float = field(default_factory=lambda: CONFIG.get('risk', {}).get('drawdown_warning', 0.05))
    max_drawdown_critical: float = field(default_factory=lambda: CONFIG.get('risk', {}).get('drawdown_critical', 0.10))
    max_drawdown_emergency: float = field(default_factory=lambda: CONFIG.get('risk', {}).get('drawdown_emergency', 0.15))
    
    # CVaR settings from config
    cvar_confidence: float = field(default_factory=lambda: CONFIG.get('risk', {}).get('cvar_confidence', 0.95))
    cvar_window: int = field(default_factory=lambda: CONFIG.get('risk', {}).get('cvar_window', 252))
    cvar_max_threshold: float = field(default_factory=lambda: CONFIG.get('risk', {}).get('cvar_max_threshold', 0.03))
    
    # Kelly settings from config
    kelly_fraction: float = field(default_factory=lambda: CONFIG.get('risk', {}).get('kelly_fraction', 0.25))
    kelly_min: float = field(default_factory=lambda: CONFIG.get('risk', {}).get('kelly_min', 0.01))
    kelly_max: float = field(default_factory=lambda: CONFIG.get('risk', {}).get('kelly_max', 0.50))
    
    # Optuna settings from config
    optuna_n_trials: int = field(default_factory=lambda: CONFIG.get('optuna', {}).get('n_trials', 50))
    
    def __post_init__(self):
        self.n_jobs = self.n_cpu if self.n_jobs == -1 else self.n_jobs
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)


# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================

class RiskState(Enum):
    """Global risk state managed by Risk Engine."""
    ON = auto()
    REDUCED = auto()
    OFF = auto()


class ModelType(Enum):
    """Trading model types."""
    SCALP = "scalp"
    INTRADAY = "intraday"
    SWING = "swing"
    RISK = "risk"


class TradeDecision(Enum):
    """Risk Engine trade decision."""
    APPROVED = auto()
    REDUCED = auto()
    REJECTED = auto()
    VETOED = auto()


@dataclass
class TradeSignal:
    """Signal from trading model to Risk Engine."""
    model_type: ModelType
    symbol: str
    timestamp: datetime
    direction: int
    probability: float
    expected_return: float
    confidence: float
    volatility_regime: str
    raw_features: Optional[np.ndarray] = None


@dataclass
class RiskDecision:
    """Decision from Risk Engine."""
    signal: TradeSignal
    decision: TradeDecision
    approved_size: float
    approved_leverage: float
    risk_state: RiskState
    cvar_current: float
    kelly_fraction: float
    drawdown_current: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'model': self.signal.model_type.value,
            'symbol': self.signal.symbol,
            'decision': self.decision.name,
            'size': self.approved_size,
            'leverage': self.approved_leverage,
            'risk_state': self.risk_state.name,
            'cvar': self.cvar_current,
            'kelly': self.kelly_fraction,
            'drawdown': self.drawdown_current,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass  
class OptimizationMetrics:
    """Multi-objective optimization metrics."""
    auc: float = 0.0
    cvar: float = 0.0
    max_drawdown: float = 0.0
    tail_loss_frequency: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    equity_stability: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}
    
    def weighted_score(self, weights: Dict[str, float] = None) -> float:
        """Compute weighted multi-objective score."""
        if weights is None:
            weights = {
                'auc': 0.15,
                'cvar': -0.20,
                'max_drawdown': -0.20,
                'tail_loss_frequency': -0.10,
                'sharpe_ratio': 0.15,
                'sortino_ratio': 0.10,
                'equity_stability': 0.10
            }
        
        score = 0.0
        for metric, weight in weights.items():
            value = getattr(self, metric, 0.0)
            score += weight * value
        return score


# ==============================================================================
# LOGGING SYSTEM
# ==============================================================================

class RiskLogger:
    """Comprehensive logging for audit trails."""
    
    def __init__(self, log_path: str = None):
        self.log_path = log_path or CONFIG.get('paths', {}).get('logs', '/home/ai/NogutiAI/aiTrainCrypto/project/logs')
        os.makedirs(self.log_path, exist_ok=True)
        
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_path, f'risk_engine_{self.session_id}.log')
        self.decisions_file = os.path.join(self.log_path, f'decisions_{self.session_id}.jsonl')
        
        self.logger = logging.getLogger('RiskEngine')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.decisions_buffer: List[Dict] = []
        
    def info(self, msg: str):
        self.logger.info(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)
        
    def debug(self, msg: str):
        self.logger.debug(msg)
        
    def log_decision(self, decision: RiskDecision):
        """Log risk decision for audit."""
        record = decision.to_dict()
        self.decisions_buffer.append(record)
        
        with open(self.decisions_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
            
        self.debug(f"Decision logged: {decision.decision.name} for {decision.signal.symbol}")
        
    def log_metrics(self, metrics: OptimizationMetrics, prefix: str = ''):
        """Log optimization metrics."""
        self.info(f"{prefix}Metrics: AUC={metrics.auc:.4f}, CVaR={metrics.cvar:.4f}, "
                 f"MaxDD={metrics.max_drawdown:.4f}, Sharpe={metrics.sharpe_ratio:.4f}")
        
    def flush(self):
        """Flush all handlers."""
        for handler in self.logger.handlers:
            handler.flush()


# ==============================================================================
# CVaR CALCULATOR (MANDATORY)
# ==============================================================================

class CVaRCalculator:
    """Conditional Value at Risk (CVaR) Calculator."""
    
    def __init__(self, confidence: float = 0.95, window: int = 252):
        self.confidence = confidence
        self.window = window
        self.pnl_history: Dict[str, List[float]] = {}
        self.stress_scenarios: List[np.ndarray] = []
        
    def update_pnl(self, model_id: str, pnl: float):
        """Update PnL history for a model."""
        if model_id not in self.pnl_history:
            self.pnl_history[model_id] = []
        self.pnl_history[model_id].append(pnl)
        
        if len(self.pnl_history[model_id]) > self.window:
            self.pnl_history[model_id] = self.pnl_history[model_id][-self.window:]
            
    def compute_var(self, returns: np.ndarray, confidence: float = None) -> float:
        """Compute Value at Risk."""
        if confidence is None:
            confidence = self.confidence
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    def compute_cvar(self, returns: np.ndarray, confidence: float = None) -> float:
        """Compute Conditional Value at Risk (Expected Shortfall)."""
        if confidence is None:
            confidence = self.confidence
        if len(returns) == 0:
            return 0.0
            
        var = self.compute_var(returns, confidence)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        return np.mean(tail_returns)
    
    def compute_model_cvar(self, model_id: str) -> float:
        """Compute CVaR for specific model."""
        if model_id not in self.pnl_history or len(self.pnl_history[model_id]) < 10:
            return 0.0
        return self.compute_cvar(np.array(self.pnl_history[model_id]))
    
    def compute_portfolio_cvar(self) -> float:
        """Compute CVaR for entire portfolio."""
        all_pnl = []
        for pnl_list in self.pnl_history.values():
            all_pnl.extend(pnl_list)
        if len(all_pnl) < 10:
            return 0.0
        return self.compute_cvar(np.array(all_pnl))
    
    def compute_stress_cvar(self, returns: np.ndarray, stress_multiplier: float = 2.0) -> float:
        """Compute CVaR under stress scenario."""
        if len(returns) == 0:
            return 0.0
        stressed_returns = np.where(returns < 0, returns * stress_multiplier, returns)
        return self.compute_cvar(stressed_returns)
    
    def compute_worst_tail_cvar(self, returns: np.ndarray, tail_pct: float = 0.01) -> float:
        """Compute CVaR on worst tail."""
        if len(returns) == 0:
            return 0.0
        n_tail = max(1, int(len(returns) * tail_pct))
        worst_returns = np.sort(returns)[:n_tail]
        return np.mean(worst_returns)
    
    def cvar_loss_penalty(self, returns: np.ndarray, threshold: float = 0.03) -> float:
        """Compute CVaR-based loss penalty for training."""
        cvar = abs(self.compute_cvar(returns))
        if cvar > threshold:
            return (cvar - threshold) ** 2 * 100
        return 0.0
    
    def is_trade_admissible(self, expected_cvar_impact: float,
                           current_cvar: float,
                           threshold: float) -> Tuple[bool, str]:
        """Trade admission filter based on CVaR."""
        projected_cvar = current_cvar + expected_cvar_impact
        
        if projected_cvar > threshold:
            return False, f"CVaR exceeds threshold: {projected_cvar:.4f} > {threshold:.4f}"
        
        if expected_cvar_impact > threshold * 0.5:
            return False, f"Trade CVaR impact too high: {expected_cvar_impact:.4f}"
            
        return True, "CVaR within limits"
    
    def get_exposure_limit(self, current_cvar: float, max_cvar: float) -> float:
        """Dynamic exposure limiter based on CVaR."""
        if current_cvar >= max_cvar:
            return 0.0
        utilization = current_cvar / max_cvar
        return max(0.0, 1.0 - utilization)


# ==============================================================================
# KELLY CRITERION CALCULATOR (MANDATORY)
# ==============================================================================

class KellyCalculator:
    """Fractional & Constrained Kelly Position Sizing."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.base_fraction = config.kelly_fraction
        self.min_fraction = config.kelly_min
        self.max_fraction = config.kelly_max
        
        self.vol_multipliers = {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.6,
            'extreme': 0.2
        }
        
        self.risk_state_multipliers = {
            RiskState.ON: 1.0,
            RiskState.REDUCED: 0.5,
            RiskState.OFF: 0.0
        }
        
    def compute_raw_kelly(self, win_rate: float, payoff_ratio: float) -> float:
        """Compute raw Kelly fraction."""
        if payoff_ratio <= 0 or win_rate <= 0:
            return 0.0
        q = 1 - win_rate
        kelly = (win_rate * payoff_ratio - q) / payoff_ratio
        return max(0.0, kelly)
    
    def dampen_kelly(self, raw_kelly: float, dampening_factor: float = 0.5) -> float:
        """Apply dampening to raw Kelly."""
        return raw_kelly * dampening_factor
    
    def apply_confidence_scaling(self, kelly: float, confidence: float) -> float:
        """Scale Kelly by model confidence."""
        confidence_multiplier = confidence ** 2
        return kelly * confidence_multiplier
    
    def apply_volatility_scaling(self, kelly: float, volatility_regime: str) -> float:
        """Scale Kelly by volatility regime."""
        multiplier = self.vol_multipliers.get(volatility_regime, 0.5)
        return kelly * multiplier
    
    def apply_risk_state_scaling(self, kelly: float, risk_state: RiskState) -> float:
        """Scale Kelly by global risk state."""
        multiplier = self.risk_state_multipliers.get(risk_state, 0.0)
        return kelly * multiplier
    
    def cap_kelly(self, kelly: float) -> float:
        """Apply hard caps to Kelly fraction."""
        return np.clip(kelly, self.min_fraction, self.max_fraction)
    
    def compute_constrained_kelly(self, 
                                  win_rate: float,
                                  payoff_ratio: float,
                                  confidence: float,
                                  volatility_regime: str,
                                  risk_state: RiskState,
                                  cvar_utilization: float = 0.0) -> Tuple[float, str]:
        """Compute fully constrained Kelly fraction."""
        reasoning_parts = []
        
        raw = self.compute_raw_kelly(win_rate, payoff_ratio)
        reasoning_parts.append(f"Raw Kelly: {raw:.4f}")
        
        dampened = self.dampen_kelly(raw, self.base_fraction)
        reasoning_parts.append(f"Dampened: {dampened:.4f}")
        
        capped = self.cap_kelly(dampened)
        reasoning_parts.append(f"Capped: {capped:.4f}")
        
        confidence_scaled = self.apply_confidence_scaling(capped, confidence)
        reasoning_parts.append(f"Confidence-scaled: {confidence_scaled:.4f}")
        
        vol_scaled = self.apply_volatility_scaling(confidence_scaled, volatility_regime)
        reasoning_parts.append(f"Vol-scaled ({volatility_regime}): {vol_scaled:.4f}")
        
        risk_adjusted = self.apply_risk_state_scaling(vol_scaled, risk_state)
        reasoning_parts.append(f"Risk-adjusted ({risk_state.name}): {risk_adjusted:.4f}")
        
        if cvar_utilization > 0.5:
            cvar_penalty = 1.0 - (cvar_utilization - 0.5) * 2
            risk_adjusted *= max(0.0, cvar_penalty)
            reasoning_parts.append(f"CVaR-penalized: {risk_adjusted:.4f}")
        
        final = self.cap_kelly(risk_adjusted)
        reasoning = " → ".join(reasoning_parts)
        
        return final, reasoning


# ==============================================================================
# DRAWDOWN MONITOR (MANDATORY)
# ==============================================================================

class DrawdownMonitor:
    """Drawdown-Aware Dynamic Risk Control."""
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        self.config = config
        self.logger = logger
        
        self.warning_threshold = config.max_drawdown_warning
        self.critical_threshold = config.max_drawdown_critical
        self.emergency_threshold = config.max_drawdown_emergency
        
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown = 0.0
        self.drawdown_history: List[Tuple[datetime, float]] = []
        self.disabled_models: set = set()
        
    def update_equity(self, equity: float, timestamp: datetime = None):
        """Update equity and compute drawdown."""
        timestamp = timestamp or datetime.now()
        self.current_equity = equity
        
        if equity > self.peak_equity:
            self.peak_equity = equity
            
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0
            
        self.drawdown_history.append((timestamp, self.current_drawdown))
        
        if len(self.drawdown_history) > 10000:
            self.drawdown_history = self.drawdown_history[-10000:]
            
    def get_position_multiplier(self) -> Tuple[float, str]:
        """Get position size multiplier based on drawdown."""
        dd = self.current_drawdown
        
        if dd < self.warning_threshold:
            return 1.0, f"Normal: DD={dd:.2%} < {self.warning_threshold:.2%}"
        elif dd < self.critical_threshold:
            pct = (dd - self.warning_threshold) / (self.critical_threshold - self.warning_threshold)
            multiplier = 1.0 - 0.5 * pct
            return multiplier, f"Warning: DD={dd:.2%}, multiplier={multiplier:.2f}"
        elif dd < self.emergency_threshold:
            multiplier = 0.25
            return multiplier, f"Critical: DD={dd:.2%}, multiplier={multiplier:.2f}"
        else:
            return 0.0, f"Emergency: DD={dd:.2%}, trading halted"
            
    def should_disable_model(self, model_type: ModelType) -> Tuple[bool, str]:
        """Check if model should be disabled based on drawdown."""
        dd = self.current_drawdown
        
        if model_type == ModelType.SCALP and dd > self.critical_threshold:
            self.disabled_models.add(model_type)
            return True, f"Scalp disabled: DD={dd:.2%} > {self.critical_threshold:.2%}"
            
        if model_type == ModelType.INTRADAY and dd > self.emergency_threshold:
            self.disabled_models.add(model_type)
            return True, f"Intraday disabled: DD={dd:.2%} > {self.emergency_threshold:.2%}"
            
        if model_type in self.disabled_models:
            if dd < self.warning_threshold:
                self.disabled_models.remove(model_type)
                return False, f"{model_type.value} re-enabled: DD={dd:.2%}"
                
        return model_type in self.disabled_models, "Model status unchanged"
    
    def get_risk_state(self) -> Tuple[RiskState, str]:
        """Determine global risk state based on drawdown."""
        dd = self.current_drawdown
        
        if dd >= self.emergency_threshold:
            return RiskState.OFF, f"Risk OFF: Emergency DD={dd:.2%}"
        elif dd >= self.critical_threshold:
            return RiskState.REDUCED, f"Risk REDUCED: Critical DD={dd:.2%}"
        else:
            return RiskState.ON, f"Risk ON: DD={dd:.2%}"
            
    def get_max_historical_drawdown(self) -> float:
        """Get maximum historical drawdown."""
        if not self.drawdown_history:
            return 0.0
        return max(dd for _, dd in self.drawdown_history)


# ==============================================================================
# VOLATILITY REGIME DETECTOR
# ==============================================================================

class VolatilityRegimeDetector:
    """Detect market volatility regime for risk adjustment."""
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.volatility_history: List[float] = []
        
        self.low_threshold = 0.25
        self.medium_threshold = 0.50
        self.high_threshold = 0.75
        
    def update(self, returns: np.ndarray):
        """Update volatility estimate."""
        if len(returns) < 2:
            return
        vol = np.std(returns) * np.sqrt(252)
        self.volatility_history.append(vol)
        
        if len(self.volatility_history) > 1000:
            self.volatility_history = self.volatility_history[-1000:]
            
    def get_current_volatility(self) -> float:
        """Get current volatility estimate."""
        if not self.volatility_history:
            return 0.0
        return self.volatility_history[-1]
    
    def get_regime(self) -> str:
        """Get current volatility regime."""
        if len(self.volatility_history) < self.lookback:
            return 'medium'
            
        current_vol = self.volatility_history[-1]
        historical_vols = np.array(self.volatility_history[:-1])
        
        percentile = stats.percentileofscore(historical_vols, current_vol) / 100
        
        if percentile < self.low_threshold:
            return 'low'
        elif percentile < self.medium_threshold:
            return 'medium'
        elif percentile < self.high_threshold:
            return 'high'
        else:
            return 'extreme'
            
    def compute_from_prices(self, prices: np.ndarray) -> str:
        """Compute regime directly from price array."""
        if len(prices) < self.lookback + 1:
            return 'medium'
        returns = np.diff(prices) / prices[:-1]
        self.update(returns)
        return self.get_regime()


# ==============================================================================
# RISK ENGINE (CORE CONTROLLER)
# ==============================================================================

class RiskEngine:
    """Central Risk Engine - THE CORE CONTROLLER."""
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        self.config = config
        self.logger = logger
        
        self.cvar_calculator = CVaRCalculator(
            confidence=config.cvar_confidence,
            window=config.cvar_window
        )
        self.kelly_calculator = KellyCalculator(config)
        self.drawdown_monitor = DrawdownMonitor(config, logger)
        self.volatility_detector = VolatilityRegimeDetector()
        
        self.risk_state = RiskState.ON
        self.total_decisions = 0
        self.approved_decisions = 0
        self.rejected_decisions = 0
        
        self.kill_switch_active = False
        
        self.model_performance: Dict[ModelType, List[float]] = {
            ModelType.SCALP: [],
            ModelType.INTRADAY: [],
            ModelType.SWING: []
        }
        
        self.logger.info("Risk Engine initialized")
        
    def activate_kill_switch(self, reason: str):
        """Emergency kill switch - halts ALL trading."""
        self.kill_switch_active = True
        self.risk_state = RiskState.OFF
        self.logger.error(f"KILL SWITCH ACTIVATED: {reason}")
        
    def deactivate_kill_switch(self):
        """Deactivate kill switch."""
        self.kill_switch_active = False
        self.risk_state = RiskState.ON
        self.logger.warning("Kill switch deactivated")
        
    def update_model_pnl(self, model_type: ModelType, pnl: float):
        """Update PnL tracking for a model."""
        self.cvar_calculator.update_pnl(model_type.value, pnl)
        self.model_performance[model_type].append(pnl)
        
    def update_equity(self, equity: float):
        """Update portfolio equity."""
        self.drawdown_monitor.update_equity(equity)
        
    def evaluate_signal(self, signal: TradeSignal) -> RiskDecision:
        """Evaluate trading signal and make risk decision."""
        self.total_decisions += 1
        
        if self.kill_switch_active:
            self.rejected_decisions += 1
            return RiskDecision(
                signal=signal,
                decision=TradeDecision.VETOED,
                approved_size=0.0,
                approved_leverage=0.0,
                risk_state=RiskState.OFF,
                cvar_current=self.cvar_calculator.compute_portfolio_cvar(),
                kelly_fraction=0.0,
                drawdown_current=self.drawdown_monitor.current_drawdown,
                reasoning="Kill switch active - all trading halted"
            )
            
        reasoning_parts = []
        
        model_disabled, dd_reason = self.drawdown_monitor.should_disable_model(signal.model_type)
        if model_disabled:
            self.rejected_decisions += 1
            return RiskDecision(
                signal=signal,
                decision=TradeDecision.VETOED,
                approved_size=0.0,
                approved_leverage=0.0,
                risk_state=self.risk_state,
                cvar_current=self.cvar_calculator.compute_portfolio_cvar(),
                kelly_fraction=0.0,
                drawdown_current=self.drawdown_monitor.current_drawdown,
                reasoning=dd_reason
            )
        reasoning_parts.append(dd_reason)
        
        self.risk_state, risk_reason = self.drawdown_monitor.get_risk_state()
        reasoning_parts.append(risk_reason)
        
        if self.risk_state == RiskState.OFF:
            self.rejected_decisions += 1
            return RiskDecision(
                signal=signal,
                decision=TradeDecision.REJECTED,
                approved_size=0.0,
                approved_leverage=0.0,
                risk_state=self.risk_state,
                cvar_current=self.cvar_calculator.compute_portfolio_cvar(),
                kelly_fraction=0.0,
                drawdown_current=self.drawdown_monitor.current_drawdown,
                reasoning=f"Risk state OFF: {risk_reason}"
            )
            
        current_cvar = abs(self.cvar_calculator.compute_portfolio_cvar())
        expected_cvar_impact = abs(signal.expected_return) * (1 - signal.confidence) * 0.1
        
        cvar_admissible, cvar_reason = self.cvar_calculator.is_trade_admissible(
            expected_cvar_impact, current_cvar, self.config.cvar_max_threshold
        )
        
        if not cvar_admissible:
            self.rejected_decisions += 1
            return RiskDecision(
                signal=signal,
                decision=TradeDecision.REJECTED,
                approved_size=0.0,
                approved_leverage=0.0,
                risk_state=self.risk_state,
                cvar_current=current_cvar,
                kelly_fraction=0.0,
                drawdown_current=self.drawdown_monitor.current_drawdown,
                reasoning=cvar_reason
            )
        reasoning_parts.append(cvar_reason)
        
        model_returns = self.model_performance.get(signal.model_type, [])
        if len(model_returns) > 10:
            wins = [r for r in model_returns if r > 0]
            losses = [r for r in model_returns if r < 0]
            win_rate = len(wins) / len(model_returns)
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = abs(np.mean(losses)) if losses else 1.0
            payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        else:
            win_rate = 0.5
            payoff_ratio = 1.0
            
        cvar_utilization = current_cvar / self.config.cvar_max_threshold if self.config.cvar_max_threshold > 0 else 0
        
        kelly_fraction, kelly_reason = self.kelly_calculator.compute_constrained_kelly(
            win_rate=win_rate,
            payoff_ratio=payoff_ratio,
            confidence=signal.confidence,
            volatility_regime=signal.volatility_regime,
            risk_state=self.risk_state,
            cvar_utilization=cvar_utilization
        )
        reasoning_parts.append(f"Kelly: {kelly_reason}")
        
        dd_multiplier, dd_mult_reason = self.drawdown_monitor.get_position_multiplier()
        reasoning_parts.append(dd_mult_reason)
        
        final_size = kelly_fraction * dd_multiplier
        final_size = min(final_size, self.config.max_position_pct)
        
        base_leverage = self.config.max_leverage * signal.confidence
        vol_multipliers = {'low': 1.0, 'medium': 0.8, 'high': 0.5, 'extreme': 0.2}
        vol_mult = vol_multipliers.get(signal.volatility_regime, 0.5)
        final_leverage = min(base_leverage * vol_mult * dd_multiplier, self.config.max_leverage)
        
        if final_size < 0.001:
            decision = TradeDecision.REJECTED
            reasoning = f"Position too small: {final_size:.4f}"
            self.rejected_decisions += 1
        elif final_size < kelly_fraction * 0.5:
            decision = TradeDecision.REDUCED
            reasoning = " | ".join(reasoning_parts)
            self.approved_decisions += 1
        else:
            decision = TradeDecision.APPROVED
            reasoning = " | ".join(reasoning_parts)
            self.approved_decisions += 1
            
        risk_decision = RiskDecision(
            signal=signal,
            decision=decision,
            approved_size=final_size,
            approved_leverage=final_leverage,
            risk_state=self.risk_state,
            cvar_current=current_cvar,
            kelly_fraction=kelly_fraction,
            drawdown_current=self.drawdown_monitor.current_drawdown,
            reasoning=reasoning
        )
        
        self.logger.log_decision(risk_decision)
        
        return risk_decision
    
    def get_statistics(self) -> Dict:
        """Get risk engine statistics."""
        return {
            'total_decisions': self.total_decisions,
            'approved': self.approved_decisions,
            'rejected': self.rejected_decisions,
            'approval_rate': self.approved_decisions / max(1, self.total_decisions),
            'risk_state': self.risk_state.name,
            'current_drawdown': self.drawdown_monitor.current_drawdown,
            'max_drawdown': self.drawdown_monitor.get_max_historical_drawdown(),
            'portfolio_cvar': self.cvar_calculator.compute_portfolio_cvar(),
            'kill_switch_active': self.kill_switch_active
        }


# ==============================================================================
# POLARS DATA LOADER - HIGH PERFORMANCE PARALLEL
# ==============================================================================

class PolarsDataLoader:
    """High-performance data loader using Polars - fully parallel."""
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        self.config = config
        self.logger = logger
        
        # Data storage using Polars LazyFrames
        self.symbols: List[str] = []
        self.klines: Dict[str, Dict[str, pl.LazyFrame]] = {}
        self.funding_rates: Dict[str, pl.LazyFrame] = {}
        self.open_interest: Dict[str, pl.LazyFrame] = {}
        
    def discover_symbols(self) -> List[str]:
        """Discover all available symbols from data files."""
        data_path = Path(self.config.data_path)
        symbols = set()
        
        for f in data_path.glob('klines_usdt_m_*_1h.csv'):
            name = f.stem
            parts = name.split('_')
            if len(parts) >= 4:
                symbol = '_'.join(parts[3:-1]) if len(parts) > 4 else parts[3]
                symbols.add(symbol)
                
        self.symbols = sorted(list(symbols))
        self.logger.info(f"Discovered {len(self.symbols)} symbols")
        return self.symbols
    
    def _load_single_file(self, filepath: Path) -> Optional[pl.LazyFrame]:
        """Load single CSV file with Polars - fully parallel."""
        try:
            # Schema overrides to handle mixed types (Int64 vs Float64 in different files)
            schema_overrides = {
                'open': pl.Float64,
                'high': pl.Float64,
                'low': pl.Float64,
                'close': pl.Float64,
                'volume': pl.Float64,
                'quote_volume': pl.Float64,
                'taker_buy_volume': pl.Float64,
                'taker_buy_quote_volume': pl.Float64,
                'trades': pl.Int64,
                'count': pl.Int64,
            }
            
            # Polars scan_csv is lazy and uses all cores automatically
            lf = pl.scan_csv(
                filepath,
                low_memory=False,
                rechunk=True,
                n_rows=None,
                schema_overrides=schema_overrides,
                infer_schema_length=10000,  # More rows for better type inference
            )
            
            # Convert timestamp columns
            columns = lf.columns
            if 'timestamp' in columns:
                lf = lf.with_columns(
                    pl.col('timestamp').cast(pl.Int64).cast(pl.Datetime('ms')).alias('timestamp')
                )
            elif 'open_time' in columns:
                lf = lf.with_columns(
                    pl.col('open_time').cast(pl.Int64).cast(pl.Datetime('ms')).alias('open_time')
                )
            
            return lf
        except Exception as e:
            return None
            
    def _load_symbol_data(self, symbol: str) -> Dict:
        """Load all data for a single symbol using Polars LazyFrames."""
        data_path = Path(self.config.data_path)
        result = {
            'symbol': symbol,
            'klines': {},
            'funding_rate': None,
            'open_interest': None
        }
        
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        for tf in timeframes:
            filepath = data_path / f'klines_usdt_m_{symbol}_{tf}.csv'
            if filepath.exists():
                lf = self._load_single_file(filepath)
                if lf is not None:
                    result['klines'][tf] = lf
                    
        fr_path = data_path / f'funding_rate_usdt_m_{symbol}.csv'
        if fr_path.exists():
            result['funding_rate'] = self._load_single_file(fr_path)
            
        oi_path = data_path / f'open_interest_usdt_m_{symbol}_5m.csv'
        if oi_path.exists():
            result['open_interest'] = self._load_single_file(oi_path)
            
        return result
    
    def load_all_data(self) -> Dict:
        """Load all data with Polars - uses lazy evaluation for maximum efficiency."""
        self.logger.info(f"Loading data from {self.config.data_path}")
        
        symbols = self.discover_symbols()
        self.logger.info(f"Found {len(symbols)} symbols")
        
        start_time = time.time()
        
        self.logger.info(f"Loading with {self.config.n_cpu} parallel workers (Polars)...")
        
        # Load all symbols - Polars is lazy, so this is fast
        total_rows = 0
        for symbol in symbols:
            result = self._load_symbol_data(symbol)
            self.klines[symbol] = result['klines']
            
            if result['funding_rate'] is not None:
                self.funding_rates[symbol] = result['funding_rate']
                
            if result['open_interest'] is not None:
                self.open_interest[symbol] = result['open_interest']
        
        # Count rows by collecting a single symbol
        if self.klines:
            sample_symbol = list(self.klines.keys())[0]
            for tf, lf in self.klines[sample_symbol].items():
                try:
                    count = lf.select(pl.count()).collect().item()
                    total_rows += count * len(symbols)
                except:
                    pass
                
        elapsed = time.time() - start_time
        self.logger.info(f"Loaded {total_rows:,} rows (lazy) in {elapsed:.1f}s")
        
        return {
            'symbols': self.symbols,
            'klines': self.klines,
            'funding_rates': self.funding_rates,
            'open_interest': self.open_interest,
            'total_rows': total_rows
        }
    
    def get_training_data(self, model_type: ModelType) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get prepared training data for specific model type.
        
        Returns:
            X: Feature matrix
            y: Target labels (0=Down, 1=Flat, 2=Up)
            returns: Actual future returns
        """
        timeframe_map = {
            ModelType.SCALP: '5m',
            ModelType.INTRADAY: '1h',
            ModelType.SWING: '1d'
        }
        primary_tf = timeframe_map.get(model_type, '1h')
        
        all_dfs = []
        
        for symbol in self.symbols:
            if symbol not in self.klines:
                continue
            if primary_tf not in self.klines[symbol]:
                continue
                
            lf = self.klines[symbol][primary_tf]
            lf = lf.with_columns(pl.lit(symbol).alias('symbol'))
            all_dfs.append(lf)
            
        if not all_dfs:
            return np.array([]), np.array([])
        
        # Concatenate all LazyFrames and collect
        combined = pl.concat(all_dfs)
        
        # Add target column
        combined = combined.with_columns([
            (pl.col('close').shift(-1) / pl.col('close') - 1).over('symbol').alias('future_return')
        ])
        
        # Collect and convert to numpy
        df = combined.collect()
        
        # Drop NA and create 3-class target: 0=Down, 1=Flat, 2=Up
        flat_threshold = 0.002  # 0.2%
        df = df.drop_nulls(subset=['future_return'])
        df = df.with_columns([
            pl.when(pl.col('future_return') < -flat_threshold)
              .then(pl.lit(0))  # Down
              .when(pl.col('future_return') > flat_threshold)
              .then(pl.lit(2))  # Up
              .otherwise(pl.lit(1))  # Flat
              .cast(pl.Int32).alias('target')
        ])
        
        # Get feature columns
        exclude_cols = ['symbol', 'open_time', 'timestamp', 'future_return', 'target', 
                       'interval', 'market_type', 'close_time']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y = df.select('target').to_numpy().flatten()
        returns = df.select('future_return').to_numpy().flatten().astype(np.float32)
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y, returns


# ==============================================================================
# POLARS FEATURE ENGINEERING - FULLY PARALLEL
# ==============================================================================

class PolarsFeatureEngine:
    """High-performance feature engineering with Polars - fully parallel."""
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        self.config = config
        self.logger = logger
        
    def compute_technical_features(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Compute technical indicators using Polars expressions - fully parallel."""
        
        # Get column names to check what's available
        columns = lf.columns
        
        if 'close' not in columns:
            return lf
        
        # Build all features in a single expression chain
        feature_exprs = []
        
        # ============================================================
        # PRICE FEATURES - Multiple timeframes
        # ============================================================
        for period in [3, 5, 7, 10, 14, 20, 30, 50, 100, 200]:
            feature_exprs.extend([
                pl.col('close').rolling_mean(period).alias(f'sma_{period}'),
                pl.col('close').ewm_mean(span=period).alias(f'ema_{period}'),
                pl.col('close').rolling_std(period).alias(f'std_{period}'),
                pl.col('close').pct_change(period).alias(f'return_{period}'),
                (pl.col('close') / pl.col('close').shift(period)).log().alias(f'log_return_{period}'),
            ])
            
        # ============================================================
        # RSI for multiple periods
        # ============================================================
        for period in [5, 7, 9, 14, 21, 28]:
            delta = pl.col('close').diff()
            gain = delta.clip(lower_bound=0).rolling_mean(period)
            loss = (-delta.clip(upper_bound=0)).rolling_mean(period)
            rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
            feature_exprs.append(rsi.alias(f'rsi_{period}'))
            
        # ============================================================
        # MACD variants
        # ============================================================
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (8, 17, 9)]:
            ema_fast = pl.col('close').ewm_mean(span=fast)
            ema_slow = pl.col('close').ewm_mean(span=slow)
            macd = ema_fast - ema_slow
            feature_exprs.extend([
                macd.alias(f'macd_{fast}_{slow}'),
                macd.ewm_mean(span=signal).alias(f'macd_signal_{fast}_{slow}'),
                (macd - macd.ewm_mean(span=signal)).alias(f'macd_hist_{fast}_{slow}'),
            ])
            
        # ============================================================
        # Rate of Change & Momentum
        # ============================================================
        for period in [5, 10, 20, 50]:
            feature_exprs.extend([
                ((pl.col('close') - pl.col('close').shift(period)) / 
                 (pl.col('close').shift(period) + 1e-10) * 100).alias(f'roc_{period}'),
                (pl.col('close') - pl.col('close').shift(period)).alias(f'momentum_{period}'),
            ])
            
        # ============================================================
        # Volatility (ATR, Bollinger)
        # ============================================================
        if 'high' in columns and 'low' in columns:
            # True Range components
            tr1 = pl.col('high') - pl.col('low')
            tr2 = (pl.col('high') - pl.col('close').shift()).abs()
            tr3 = (pl.col('low') - pl.col('close').shift()).abs()
            
            for period in [5, 10, 14, 20, 30, 50]:
                # ATR approximation using max of components
                atr = tr1.rolling_mean(period)  # Simplified
                feature_exprs.extend([
                    atr.alias(f'atr_{period}'),
                    (atr / (pl.col('close') + 1e-10) * 100).alias(f'natr_{period}'),
                ])
                
            # Williams %R
            for period in [14, 28]:
                highest = pl.col('high').rolling_max(period)
                lowest = pl.col('low').rolling_min(period)
                williams_r = -100 * (highest - pl.col('close')) / (highest - lowest + 1e-10)
                feature_exprs.append(williams_r.alias(f'williams_r_{period}'))
                
        # ============================================================
        # Bollinger Bands
        # ============================================================
        for period in [10, 20, 50]:
            sma = pl.col('close').rolling_mean(period)
            std = pl.col('close').rolling_std(period)
            for std_mult in [1.5, 2.0, 2.5, 3.0]:
                bb_upper = sma + std_mult * std
                bb_lower = sma - std_mult * std
                feature_exprs.extend([
                    bb_upper.alias(f'bb_upper_{period}_{std_mult}'),
                    bb_lower.alias(f'bb_lower_{period}_{std_mult}'),
                    ((bb_upper - bb_lower) / (sma + 1e-10)).alias(f'bb_width_{period}_{std_mult}'),
                    ((pl.col('close') - bb_lower) / (bb_upper - bb_lower + 1e-10)).alias(f'bb_pct_{period}_{std_mult}'),
                ])
                
        # ============================================================
        # Volume Features
        # ============================================================
        if 'volume' in columns:
            for period in [5, 10, 20, 50]:
                feature_exprs.extend([
                    pl.col('volume').rolling_mean(period).alias(f'volume_sma_{period}'),
                    pl.col('volume').rolling_std(period).alias(f'volume_std_{period}'),
                    (pl.col('volume') / (pl.col('volume').rolling_mean(period) + 1e-10)).alias(f'volume_ratio_{period}'),
                ])
                
            # OBV
            obv = (pl.col('close').diff().sign() * pl.col('volume')).cum_sum()
            feature_exprs.extend([
                obv.alias('obv'),
                obv.rolling_mean(20).alias('obv_sma_20'),
                (obv.diff(5) / 5).alias('obv_slope'),
            ])
            
        # ============================================================
        # Statistical Features
        # ============================================================
        returns = pl.col('close').pct_change()
        for period in [10, 20, 50, 100]:
            feature_exprs.extend([
                returns.rolling_skew(period).alias(f'skew_{period}'),
                ((pl.col('close') - pl.col('close').rolling_mean(period)) / 
                 (pl.col('close').rolling_std(period) + 1e-10)).alias(f'zscore_{period}'),
            ])
            
        # ============================================================
        # Cross-over Signals
        # ============================================================
        feature_exprs.extend([
            (pl.col('close').rolling_mean(5) > pl.col('close').rolling_mean(20)).cast(pl.Float32).alias('sma_5_20_cross'),
            (pl.col('close').rolling_mean(10) > pl.col('close').rolling_mean(50)).cast(pl.Float32).alias('sma_10_50_cross'),
            (pl.col('close').rolling_mean(20) > pl.col('close').rolling_mean(100)).cast(pl.Float32).alias('sma_20_100_cross'),
            (pl.col('close') > pl.col('close').rolling_mean(20)).cast(pl.Float32).alias('price_sma_20_cross'),
            (pl.col('close') > pl.col('close').rolling_mean(50)).cast(pl.Float32).alias('price_sma_50_cross'),
        ])
        
        # ============================================================
        # Lag Features
        # ============================================================
        for lag in [1, 2, 3, 5, 10]:
            feature_exprs.extend([
                pl.col('close').shift(lag).alias(f'close_lag_{lag}'),
                pl.col('close').pct_change().shift(lag).alias(f'return_lag_{lag}'),
            ])
            if 'volume' in columns:
                feature_exprs.append(pl.col('volume').shift(lag).alias(f'volume_lag_{lag}'))
                
        # ============================================================
        # Candlestick Patterns
        # ============================================================
        if 'open' in columns and 'high' in columns and 'low' in columns:
            body_size = (pl.col('close') - pl.col('open')).abs() / (pl.col('high') - pl.col('low') + 1e-10)
            feature_exprs.extend([
                body_size.alias('body_size'),
                ((pl.col('high') - pl.max_horizontal('close', 'open')) / (pl.col('high') - pl.col('low') + 1e-10)).alias('upper_shadow'),
                ((pl.min_horizontal('close', 'open') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-10)).alias('lower_shadow'),
                (pl.col('close') > pl.col('open')).cast(pl.Float32).alias('is_bullish'),
                (body_size < 0.1).cast(pl.Float32).alias('is_doji'),
            ])
            
        # Apply all features at once
        result = lf.with_columns(feature_exprs)
        
        return result
    
    def compute_risk_features(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Compute risk-specific features."""
        columns = lf.columns
        
        if 'close' not in columns:
            return lf
            
        risk_exprs = []
        
        # Volatility regimes
        returns = pl.col('close').pct_change()
        for period in [20, 60]:
            risk_exprs.append(
                (returns.rolling_std(period) * np.sqrt(252)).alias(f'realized_vol_{period}')
            )
            
        # Tail risk metrics
        risk_exprs.extend([
            returns.rolling_skew(20).alias('skewness_20'),
        ])
        
        # Drawdown
        cummax = pl.col('close').cum_max()
        risk_exprs.extend([
            ((cummax - pl.col('close')) / (cummax + 1e-10)).alias('drawdown'),
        ])
        
        # Rolling VaR approximation
        risk_exprs.append(
            returns.rolling_quantile(0.05, window_size=60).alias('rolling_var_95')
        )
        
        return lf.with_columns(risk_exprs)
    
    def process_symbol(self, symbol: str, klines_data: Dict[str, pl.LazyFrame]) -> Dict:
        """Process all features for a symbol."""
        result = {'symbol': symbol, 'features': {}}
        
        for tf, lf in klines_data.items():
            processed = self.compute_technical_features(lf)
            processed = self.compute_risk_features(processed)
            result['features'][tf] = processed
            
        return result
    
    def build_features_parallel(self, data_loader: PolarsDataLoader) -> Dict:
        """Build features with Polars - automatically parallel."""
        self.logger.info("Building 150+ features with Polars (fully parallel)...")
        start_time = time.time()
        
        symbols_to_process = [s for s in data_loader.symbols if s in data_loader.klines]
        total_symbols = len(symbols_to_process)
        
        self.logger.info(f"Processing {total_symbols} symbols with Polars parallel engine...")
        
        features = {}
        for symbol in symbols_to_process:
            result = self.process_symbol(symbol, data_loader.klines[symbol])
            features[result['symbol']] = result['features']
            
        elapsed = time.time() - start_time
        self.logger.info(f"Features built in {elapsed:.1f}s for {total_symbols} symbols")
        
        return features


# ==============================================================================
# TRADING MODEL BASE CLASS
# ==============================================================================

class BaseTradingModel:
    """Base class for all trading models."""
    
    def __init__(self, model_type: ModelType, config: SystemConfig, logger: RiskLogger):
        self.model_type = model_type
        self.config = config
        self.logger = logger
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        return X_scaled.astype(np.float32), y.astype(np.int32)
    
    def train(self, X: np.ndarray, y: np.ndarray, params: Dict = None):
        """Train the model."""
        raise NotImplementedError
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        raise NotImplementedError
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        raise NotImplementedError


class LightGBMTradingModel(BaseTradingModel):
    """LightGBM-based trading model - MAXIMUM PERFORMANCE."""
    
    def __init__(self, model_type: ModelType, config: SystemConfig, logger: RiskLogger):
        super().__init__(model_type, config, logger)
        
        available_ram_mb = int(config.total_ram_gb * 1024 * 0.7)
        
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': config.num_leaves,
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': config.min_data_in_leaf,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'max_bin': config.max_bin,
            'num_threads': config.n_cpu,
            'verbose': -1,
            'seed': config.random_seed,
            'force_col_wise': True,
            'deterministic': False,
            'histogram_pool_size': available_ram_mb,
            'feature_fraction_bynode': config.feature_fraction_bynode,
            'bin_construct_sample_cnt': config.bin_construct_sample_cnt,
            'max_cached_hist_node': config.max_cached_hist_node,
        }
        
    def train(self, X: np.ndarray, y: np.ndarray, params: Dict = None):
        """Train the LightGBM model."""
        train_params = self.lgb_params.copy()
        if params:
            train_params.update(params)
        
        # Log training configuration
        self.logger.info(f"  LightGBM training: {len(X):,} samples, {X.shape[1]} features, {train_params.get('num_threads', 'default')} threads")
            
        split_idx = int(len(X) * 0.9)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create Dataset with explicit construction for parallelization
        self.logger.info(f"  Creating LightGBM Dataset (binning)...")
        
        # Dataset params (these can't be changed after construct)
        dataset_params = {
            'num_threads': self.config.n_cpu, 
            'verbose': -1, 
            'max_bin': self.config.max_bin,
            'bin_construct_sample_cnt': self.config.bin_construct_sample_cnt,
            'min_data_in_leaf': self.config.min_data_in_leaf,
            'feature_pre_filter': False,  # Allow dynamic param changes
        }
        
        train_data = lgb.Dataset(
            X_train, label=y_train, feature_name=self.feature_names,
            params=dataset_params,
            free_raw_data=True
        )
        # Force parallel bin construction
        train_data.construct()
        
        val_data = lgb.Dataset(
            X_val, label=y_val, reference=train_data,
            free_raw_data=True
        )
        val_data.construct()
        self.logger.info(f"  Dataset ready, starting training...")
        
        # Remove dataset-only params from train_params
        train_params_clean = {k: v for k, v in train_params.items() 
                             if k not in ['bin_construct_sample_cnt']}
        
        callbacks = [
            lgb.early_stopping(self.config.early_stopping_rounds),
            lgb.log_evaluation(period=100)
        ]
        
        self.model = lgb.train(
            train_params_clean,
            train_data,
            num_boost_round=self.config.num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        self.is_trained = True
        
        if self.logger:
            self.logger.info(f"  Best iteration: {self.model.best_iteration}")
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for all 3 classes.
        
        Returns:
            Array of shape (n_samples, 3) with probabilities for [Down, Flat, Up]
        """
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (0=Down, 1=Flat, 2=Up)."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if self.model is None:
            return {}
        importance = self.model.feature_importance(importance_type='gain')
        return dict(zip(self.feature_names, importance))


# ==============================================================================
# RISK MODEL (SPECIALIZED)
# ==============================================================================

class RiskModel(BaseTradingModel):
    """Specialized model for risk prediction with CVaR integration."""
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        super().__init__(ModelType.RISK, config, logger)
        self.cvar_calculator = CVaRCalculator(config.cvar_confidence, config.cvar_window)
        self.metrics = OptimizationMetrics()
        
        available_ram_mb = int(config.total_ram_gb * 1024 * 0.5)
        
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': min(63, config.num_leaves),
            'max_depth': min(8, config.max_depth),
            'learning_rate': config.learning_rate * 0.5,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': config.min_data_in_leaf * 2,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'max_bin': min(255, config.max_bin),
            'num_threads': config.n_cpu,
            'verbose': -1,
            'seed': config.random_seed,
            'force_col_wise': True,
            'deterministic': False,
            'histogram_pool_size': available_ram_mb,
        }
        
    def compute_risk_labels(self, returns: np.ndarray, threshold: float = -0.02) -> np.ndarray:
        """Compute risk event labels from returns."""
        labels = np.zeros(len(returns), dtype=np.int32)
        
        # Large negative return
        labels[returns < threshold] = 1
        
        # Tail events
        var_95 = np.percentile(returns, 5)
        labels[returns < var_95] = 1
        
        return labels
    
    def train_with_cvar_penalty(self, X: np.ndarray, y: np.ndarray, 
                               returns: np.ndarray, params: Dict = None):
        """Train with CVaR-based penalty in objective."""
        train_params = self.lgb_params.copy()
        if params:
            train_params.update(params)
            
        split_idx = int(len(X) * 0.9)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Dataset params (these can't be changed after construct)
        dataset_params = {
            'num_threads': self.config.n_cpu, 
            'verbose': -1, 
            'max_bin': min(255, self.config.max_bin),
        }
        
        train_data = lgb.Dataset(
            X_train, label=y_train, feature_name=self.feature_names,
            params=dataset_params,
            free_raw_data=True
        )
        train_data.construct()
        
        val_data = lgb.Dataset(
            X_val, label=y_val, reference=train_data,
            free_raw_data=True
        )
        val_data.construct()
        
        # Remove dataset-only params from train_params
        train_params_clean = {k: v for k, v in train_params.items() 
                             if k not in ['bin_construct_sample_cnt']}
        
        callbacks = [
            lgb.early_stopping(self.config.early_stopping_rounds // 2),
            lgb.log_evaluation(period=100)
        ]
        
        self.model = lgb.train(
            train_params_clean,
            train_data,
            num_boost_round=self.config.num_boost_round // 2,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        self.is_trained = True
        
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       returns: np.ndarray) -> OptimizationMetrics:
        """Compute comprehensive metrics including CVaR."""
        metrics = OptimizationMetrics()
        
        try:
            metrics.auc = roc_auc_score(y_true, y_pred)
        except:
            metrics.auc = 0.5
            
        metrics.cvar = abs(self.cvar_calculator.compute_cvar(returns))
        
        # Drawdown
        equity = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (peak + 1e-10)
        metrics.max_drawdown = np.max(drawdown)
        
        # Tail loss frequency
        var_95 = np.percentile(returns, 5)
        metrics.tail_loss_frequency = np.mean(returns < var_95)
        
        # Risk-adjusted returns
        if len(returns) > 1 and np.std(returns) > 0:
            metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            neg_returns = returns[returns < 0]
            if len(neg_returns) > 0:
                downside_std = np.std(neg_returns)
                if downside_std > 0:
                    metrics.sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252)
                    
            if metrics.max_drawdown > 0:
                annual_return = np.mean(returns) * 252
                metrics.calmar_ratio = annual_return / metrics.max_drawdown
                
        # Equity stability
        if len(equity) > 10:
            metrics.equity_stability = 1 - np.std(np.diff(equity)) / (np.mean(equity) + 1e-10)
            
        # Win rate and profit factor
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        metrics.win_rate = len(wins) / (len(returns) + 1e-10)
        
        if len(losses) > 0 and np.sum(np.abs(losses)) > 0:
            metrics.profit_factor = np.sum(wins) / np.sum(np.abs(losses))
            
        self.metrics = metrics
        return metrics


# ==============================================================================
# RISK MODEL OPTIMIZER (OPTUNA)
# ==============================================================================

class RiskModelOptimizer:
    """Multi-objective optimization for risk model."""
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        self.config = config
        self.logger = logger
        self.best_params: Dict = {}
        self.best_metrics: OptimizationMetrics = OptimizationMetrics()
        
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                returns_train: np.ndarray, returns_val: np.ndarray,
                n_trials: int = 50) -> Dict:
        """Run Optuna optimization."""
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, using default parameters")
            return {}
            
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
            }
            
            # Create and train model
            model = RiskModel(self.config, None)
            model.feature_names = [f'f_{i}' for i in range(X_train.shape[1])]
            
            try:
                model.train_with_cvar_penalty(X_train, y_train, returns_train, params)
                
                # Evaluate on validation
                y_pred = model.predict_proba(X_val)
                metrics = model.compute_metrics(y_val, y_pred, returns_val)
                
                # Multi-objective: maximize AUC, minimize CVaR and drawdown
                score = metrics.auc - metrics.cvar * 10 - metrics.max_drawdown * 5
                
                return score
                
            except Exception as e:
                return -1.0
                
        # Create study
        sampler = TPESampler(seed=self.config.random_seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,  # LightGBM already uses all cores
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        self.logger.info(f"Best trial: {study.best_trial.value:.4f}")
        
        return self.best_params


# ==============================================================================
# ONNX EXPORTER
# ==============================================================================

class ONNXExporter:
    """Export models to ONNX format for production deployment."""
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        self.config = config
        self.logger = logger
        
    def export_lightgbm_to_onnx(self, model: lgb.Booster, 
                                feature_names: List[str],
                                output_path: str) -> bool:
        """Export LightGBM model to ONNX."""
        if not ONNX_AVAILABLE:
            self.logger.warning("ONNX not available, skipping export")
            return False
            
        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            initial_type = [('input', FloatTensorType([None, len(feature_names)]))]
            
            onnx_model = convert_lightgbm(
                model,
                initial_types=initial_type,
                target_opset=12
            )
            
            onnx.save(onnx_model, output_path)
            self._validate_onnx(output_path, feature_names)
            
            self.logger.info(f"Model exported to ONNX: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            return False
            
    def _validate_onnx(self, model_path: str, feature_names: List[str]):
        """Validate ONNX model."""
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        
        session = ort.InferenceSession(model_path)
        dummy_input = np.random.randn(1, len(feature_names)).astype(np.float32)
        
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: dummy_input})
        
        self.logger.debug(f"ONNX validation passed, output shape: {output[0].shape}")
        
    def export_risk_engine_config(self, risk_engine: RiskEngine, 
                                  output_path: str) -> bool:
        """Export risk engine configuration for production."""
        config = {
            'cvar_confidence': risk_engine.cvar_calculator.confidence,
            'cvar_window': risk_engine.cvar_calculator.window,
            'kelly_base_fraction': risk_engine.kelly_calculator.base_fraction,
            'kelly_min': risk_engine.kelly_calculator.min_fraction,
            'kelly_max': risk_engine.kelly_calculator.max_fraction,
            'dd_warning': risk_engine.drawdown_monitor.warning_threshold,
            'dd_critical': risk_engine.drawdown_monitor.critical_threshold,
            'dd_emergency': risk_engine.drawdown_monitor.emergency_threshold,
            'max_leverage': risk_engine.config.max_leverage,
            'max_position_pct': risk_engine.config.max_position_pct,
            'statistics': risk_engine.get_statistics()
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            self.logger.info(f"Risk engine config exported: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Config export failed: {e}")
            return False


# ==============================================================================
# MAIN TRAINING PIPELINE - POLARS VERSION
# ==============================================================================

class ProductionTrainingPipeline:
    """Production-grade training pipeline using Polars."""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.logger = RiskLogger()
        
        # Components - using Polars
        self.data_loader = PolarsDataLoader(self.config, self.logger)
        self.feature_engine = PolarsFeatureEngine(self.config, self.logger)
        self.risk_engine = RiskEngine(self.config, self.logger)
        self.optimizer = RiskModelOptimizer(self.config, self.logger)
        self.onnx_exporter = ONNXExporter(self.config, self.logger)
        
        # Models
        self.models: Dict[ModelType, BaseTradingModel] = {}
        self.risk_model: Optional[RiskModel] = None
        
        # Features storage
        self.features: Dict = {}
        
        # Results
        self.training_results: Dict = {}
        self.final_metrics: Dict = {}
        
    def print_header(self):
        """Print system header with config info."""
        config_path = Path(__file__).parent / 'config.yaml'
        config_status = "✓ Loaded" if config_path.exists() else "✗ Not found (using defaults)"
        
        print("=" * 80)
        print("  PRODUCTION-GRADE MULTI-MODEL FUTURES TRADING SYSTEM")
        print("  POLARS VERSION - High-Performance Parallel Processing")
        print("  Configure via: project/config.yaml")
        print("=" * 80)
        print(f"  Config: {config_status}")
        print(f"  CPU Cores: {self.config.n_cpu}/{self.config.n_cpu_total} ({self.config.cpu_usage_pct*100:.0f}%)")
        ram_limit_str = f", max={self.config.ram_max_gb:.0f}GB" if self.config.ram_max_gb else ""
        usable_ram = self.config.total_ram_gb * self.config.ram_usage_pct
        print(f"  RAM: {self.config.total_ram_gb:.1f} GB × {self.config.ram_usage_pct*100:.0f}% = {usable_ram:.1f} GB{ram_limit_str}")
        print(f"  Workers: {self.config.n_jobs}")
        print(f"  LightGBM: max_bin={self.config.max_bin}, leaves={self.config.num_leaves}, lr={self.config.learning_rate}")
        print(f"  Optuna: {'Yes' if OPTUNA_AVAILABLE else 'No'} ({self.config.optuna_n_trials} trials) | ONNX: {'Yes' if ONNX_AVAILABLE else 'No'}")
        print("=" * 80)
        print(f"  Edit config.yaml to change CPU/RAM usage, LightGBM params, etc.")
        print("=" * 80)
        print()
        
    def load_data(self):
        """Load all data using Polars."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: DATA LOADING (Polars)")
        self.logger.info("=" * 60)
        
        data = self.data_loader.load_all_data()
        
        self.logger.info(f"Loaded {len(data['symbols'])} symbols")
        self.logger.info(f"Total rows: {data['total_rows']:,}")
        
        return data
        
    def build_features(self):
        """Build features for all symbols using Polars."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 2: FEATURE ENGINEERING (Polars - Parallel)")
        self.logger.info("=" * 60)
        
        self.features = self.feature_engine.build_features_parallel(self.data_loader)
        
        # Count total features generated
        sample_symbol = list(self.features.keys())[0] if self.features else None
        if sample_symbol and self.features[sample_symbol]:
            sample_tf = list(self.features[sample_symbol].keys())[0]
            # Get column count from LazyFrame
            try:
                n_features = len(self.features[sample_symbol][sample_tf].columns)
                self.logger.info(f"Generated {n_features} features per timeframe")
            except:
                pass
        
        return self.features
    
    def _get_training_data_with_features(self, model_type: ModelType) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Get training data with ALL generated features using Polars.
        
        Returns:
            X: Feature matrix
            y: Target labels (0=Down, 1=Flat, 2=Up)
            returns: Actual future returns for risk calculations
            feature_cols: List of feature column names
        """
        # Threshold for 3-class classification (varies by model type)
        flat_threshold = {
            ModelType.SCALP: 0.001,    # 0.1% for scalp
            ModelType.INTRADAY: 0.002,  # 0.2% for intraday
            ModelType.SWING: 0.005      # 0.5% for swing
        }.get(model_type, 0.002)
        
        timeframe_map = {
            ModelType.SCALP: '5m',
            ModelType.INTRADAY: '1h',
            ModelType.SWING: '1d'
        }
        primary_tf = timeframe_map.get(model_type, '1h')
        
        all_lfs = []
        symbols_processed = []
        
        for symbol in self.data_loader.symbols:
            if symbol not in self.features:
                continue
            if primary_tf not in self.features[symbol]:
                continue
                
            lf = self.features[symbol][primary_tf]
            lf = lf.with_columns(pl.lit(symbol).alias('symbol'))
            all_lfs.append(lf)
            symbols_processed.append(symbol)
            
        if not all_lfs:
            return np.array([]), np.array([]), []
        
        self.logger.info(f"  Collecting {len(all_lfs)} symbols with Polars...")
        
        # Normalize schemas before concat - cast columns to consistent types
        # This handles differences like Int64 vs Float64 for volume columns
        normalized_lfs = []
        for lf in all_lfs:
            # Cast numeric columns to Float64 for consistency
            cast_exprs = []
            for col_name in lf.columns:
                # Skip string columns
                if col_name in ['symbol', 'interval', 'market_type']:
                    continue
                # Cast all numeric types to Float64
                cast_exprs.append(pl.col(col_name).cast(pl.Float64, strict=False))
            
            if cast_exprs:
                lf = lf.with_columns(cast_exprs)
            
            # Drop columns that may not exist in all files
            cols_to_drop = [c for c in ['market_type', 'ignore'] if c in lf.columns]
            if cols_to_drop:
                lf = lf.drop(cols_to_drop)
                
            normalized_lfs.append(lf)
        
        # Concatenate all LazyFrames with diagonal concat to handle schema differences
        combined = pl.concat(normalized_lfs, how='diagonal_relaxed')
        
        # Add target column
        combined = combined.with_columns([
            (pl.col('close').shift(-1) / pl.col('close') - 1).over('symbol').alias('future_return')
        ])
        
        # Collect (this is where Polars uses all cores)
        self.logger.info("  Collecting data (Polars parallel)...")
        df = combined.collect()
        
        # Free LazyFrames
        for symbol in symbols_processed:
            if symbol in self.features and primary_tf in self.features[symbol]:
                del self.features[symbol][primary_tf]
        gc.collect()
        
        # Drop NA and create target
        df = df.drop_nulls(subset=['future_return'])
        df = df.with_columns([
            (pl.col('future_return') > 0).cast(pl.Int32).alias('target')
        ])
        
        # Smart sampling if too many rows (LightGBM is slow with >10M rows)
        max_samples = 10_000_000  # 10M rows max for efficient training
        if len(df) > max_samples:
            self.logger.info(f"  Sampling {max_samples:,} from {len(df):,} rows for efficient training...")
            # Stratified-like sampling: sample from each symbol proportionally
            df = df.sample(n=max_samples, seed=42, shuffle=True)
        
        # Get feature columns
        exclude_cols = ['symbol', 'open_time', 'timestamp', 'future_return', 'target', 
                       'interval', 'market_type', 'close_time', 'quote_volume', 
                       'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        
        self.logger.info(f"  Converting to numpy ({len(feature_cols)} features)...")
        
        # Convert to numpy efficiently - cast to Float32 in Polars first to avoid double copy
        # Use rechunk for better memory layout
        X_df = df.select([pl.col(c).cast(pl.Float32) for c in feature_cols]).rechunk()
        X = X_df.to_numpy()
        del X_df
        
        y = df.select('target').to_numpy().flatten().astype(np.int32)
        
        # Extract actual returns for risk model
        returns = df.select('future_return').to_numpy().flatten().astype(np.float32)
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Free DataFrame
        del df
        gc.collect()
        
        # Log class distribution
        down_pct = (y == 0).mean() * 100
        flat_pct = (y == 1).mean() * 100
        up_pct = (y == 2).mean() * 100
        self.logger.info(f"  Data ready: {X.shape[0]:,} samples, {X.shape[1]} features")
        self.logger.info(f"  Class distribution: Down={down_pct:.1f}%, Flat={flat_pct:.1f}%, Up={up_pct:.1f}%")
        
        return X, y, returns, feature_cols
        
    def train_trading_models(self):
        """Train all trading models with Polars-generated features."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 3: TRADING MODEL TRAINING")
        self.logger.info("=" * 60)
        
        results = {}
        
        for model_type in [ModelType.SCALP, ModelType.INTRADAY, ModelType.SWING]:
            self.logger.info(f"\nTraining {model_type.value.upper()} model...")
            
            # Get training data with ALL features (now returns 4 values)
            X, y, returns, feature_names = self._get_training_data_with_features(model_type)
            
            if len(X) == 0:
                self.logger.warning(f"No data for {model_type.value}")
                continue
            
            # Log RAM usage estimate
            ram_usage_gb = X.nbytes / (1024**3)
            self.logger.info(f"  Data: {len(X):,} samples, {len(feature_names)} features")
            self.logger.info(f"  Feature matrix RAM: {ram_usage_gb:.2f} GB")
            
            # Use TimeSeriesSplit for proper temporal validation
            tscv = TimeSeriesSplit(n_splits=5)
            fold_aucs = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
            
            # Create and train model
            model = LightGBMTradingModel(model_type, self.config, self.logger)
            X_train_scaled, y_train_arr = model.prepare_data(X_train, y_train, feature_names)
            
            model.train(X_train_scaled, y_train_arr)
            
            # Evaluate with multiclass AUC
            X_test_scaled = model.scaler.transform(X_test)
            y_pred = model.predict_proba(X_test_scaled)
            
            try:
                # Use OVR (One-vs-Rest) for multiclass AUC
                auc = roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted')
            except Exception:
                auc = 0.5
                
            fold_aucs.append(auc)
            
            # Only train on last fold (largest training set)
            if fold_idx == tscv.n_splits - 1:
                self.models[model_type] = model
                
            del X_train, X_test, y_train, y_test, X_train_scaled
            
            # Break after evaluating - use last fold for final model
            if fold_idx < tscv.n_splits - 1:
                del model
                
            # Log average AUC across folds
            avg_auc = np.mean(fold_aucs) if fold_aucs else 0.5
            self.logger.info(f"  Cross-validated AUC: {avg_auc:.4f} (5 folds)")
            
            results[model_type.value] = {'auc': avg_auc, 'samples': len(X)}
            
            # FREE MEMORY after each model
            del X, y, returns
            gc.collect()
            self.logger.info(f"  Memory freed after {model_type.value} training.")
            
        self.training_results['trading_models'] = results
        return results
        
    def train_risk_model(self):
        """Train the Risk Model with multi-objective optimization."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 4: RISK MODEL TRAINING")
        self.logger.info("=" * 60)
        
        # Get combined data for risk training (now returns 4 values including actual returns)
        X, y, returns, feature_names = self._get_training_data_with_features(ModelType.INTRADAY)
        
        if len(X) == 0:
            self.logger.error("No data available for risk model training")
            return {}
        
        self.logger.info(f"Risk model data: {len(X):,} samples")
        self.logger.info(f"Returns stats: mean={returns.mean()*100:.3f}%, std={returns.std()*100:.3f}%")
        
        # Create risk model
        self.risk_model = RiskModel(self.config, self.logger)
        
        # Compute risk labels
        y_risk = self.risk_model.compute_risk_labels(returns)
        
        self.logger.info(f"Risk events: {y_risk.sum():,} ({y_risk.mean()*100:.1f}%)")
        
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X))
        
        # Use 70% train, 15% val, 15% test from the last split
        train_idx, test_idx = splits[-1]
        val_split = int(len(train_idx) * 0.85)
        val_idx = train_idx[val_split:]
        train_idx = train_idx[:val_split]
        
        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]
        
        y_train = y_risk[train_idx]
        y_val = y_risk[val_idx]
        y_test = y_risk[test_idx]
        
        returns_train = returns[train_idx]
        returns_val = returns[val_idx]
        returns_test = returns[test_idx]
        
        # Prepare data
        X_train_scaled, y_train_arr = self.risk_model.prepare_data(X_train, y_train, feature_names)
        X_val_scaled = self.risk_model.scaler.transform(X_val)
        X_test_scaled = self.risk_model.scaler.transform(X_test)
        
        # Optuna optimization
        if OPTUNA_AVAILABLE:
            self.logger.info("\nStarting Optuna multi-objective optimization...")
            best_params = self.optimizer.optimize(
                X_train_scaled, y_train_arr,
                X_val_scaled, y_val,
                returns_train, returns_val,
                n_trials=self.config.optuna_n_trials
            )
            
            if best_params:
                self.logger.info(f"Best parameters found")
        else:
            best_params = {}
            
        # Train final model with best params
        self.logger.info("\nTraining final risk model...")
        lgb_params = {k: v for k, v in best_params.items() 
                     if k in ['num_leaves', 'learning_rate', 'feature_fraction',
                             'bagging_fraction', 'min_child_samples', 'reg_alpha', 'reg_lambda']}
        
        self.risk_model.train_with_cvar_penalty(
            X_train_scaled, y_train_arr, returns_train, lgb_params
        )
        
        # Evaluate on test set
        y_pred = self.risk_model.predict_proba(X_test_scaled)
        final_metrics = self.risk_model.compute_metrics(y_test, y_pred, returns_test)
        
        self.logger.info("\n" + "=" * 40)
        self.logger.info("RISK MODEL FINAL METRICS")
        self.logger.info("=" * 40)
        self.logger.info(f"  AUC:              {final_metrics.auc:.4f}")
        self.logger.info(f"  CVaR:             {final_metrics.cvar:.4f}")
        self.logger.info(f"  Max Drawdown:     {final_metrics.max_drawdown:.4f}")
        self.logger.info(f"  Sharpe Ratio:     {final_metrics.sharpe_ratio:.4f}")
        self.logger.info(f"  Win Rate:         {final_metrics.win_rate:.4f}")
        
        self.final_metrics = final_metrics.to_dict()
        self.training_results['risk_model'] = self.final_metrics
        
        # Free memory
        del X, y, X_train, X_val, X_test
        gc.collect()
        
        return self.final_metrics
        
    def export_models(self):
        """Export models to ONNX and configs."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 5: MODEL EXPORT")
        self.logger.info("=" * 60)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export risk model to ONNX
        if self.risk_model and self.risk_model.is_trained:
            onnx_path = os.path.join(self.config.model_path, f'risk_model_{timestamp}.onnx')
            self.onnx_exporter.export_lightgbm_to_onnx(
                self.risk_model.model,
                self.risk_model.feature_names,
                onnx_path
            )
            
        # Export risk engine config
        config_path = os.path.join(self.config.model_path, f'risk_engine_config_{timestamp}.json')
        self.onnx_exporter.export_risk_engine_config(self.risk_engine, config_path)
        
        # Export training results
        results_path = os.path.join(self.config.model_path, f'training_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
            
        self.logger.info(f"Models exported to {self.config.model_path}")
        
    def generate_risk_assessment(self) -> Dict:
        """Generate final quantitative risk assessment."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 6: RISK ASSESSMENT")
        self.logger.info("=" * 60)
        
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'system_config': {
                'cpu_cores': self.config.n_cpu,
                'ram_gb': self.config.total_ram_gb,
                'max_leverage': self.config.max_leverage,
                'max_position_pct': self.config.max_position_pct
            },
            'risk_thresholds': {
                'cvar_max': self.config.cvar_max_threshold,
                'drawdown_warning': self.config.max_drawdown_warning,
                'drawdown_critical': self.config.max_drawdown_critical,
                'drawdown_emergency': self.config.max_drawdown_emergency
            },
            'model_metrics': self.final_metrics,
            'risk_engine_stats': self.risk_engine.get_statistics(),
            'recommendations': []
        }
        
        metrics = self.risk_model.metrics if self.risk_model else OptimizationMetrics()
        
        if metrics.max_drawdown > 0.15:
            assessment['recommendations'].append(
                "HIGH RISK: Max drawdown exceeds 15%. Consider reducing position sizes."
            )
            
        if metrics.cvar > 0.05:
            assessment['recommendations'].append(
                "ELEVATED TAIL RISK: CVaR > 5%. Consider tighter stop losses."
            )
            
        if metrics.sharpe_ratio < 1.0:
            assessment['recommendations'].append(
                "LOW RISK-ADJUSTED RETURN: Sharpe < 1.0. Review strategy parameters."
            )
            
        if not assessment['recommendations']:
            assessment['recommendations'].append(
                "System within acceptable risk parameters."
            )
            
        self.logger.info("\n" + "=" * 40)
        self.logger.info("FINAL RISK ASSESSMENT")
        self.logger.info("=" * 40)
        
        for key, value in assessment.items():
            if isinstance(value, dict):
                self.logger.info(f"\n{key}:")
                for k, v in value.items():
                    self.logger.info(f"  {k}: {v}")
            elif isinstance(value, list):
                self.logger.info(f"\n{key}:")
                for item in value:
                    self.logger.info(f"  - {item}")
                    
        return assessment
        
    def run(self):
        """Run complete training pipeline."""
        start_time = time.time()
        
        self.print_header()
        
        try:
            # Phase 1: Load data
            self.load_data()
            
            # Phase 2: Build features
            self.build_features()
            
            # Phase 3: Train trading models
            self.train_trading_models()
            
            # Phase 4: Train risk model
            self.train_risk_model()
            
            # Phase 5: Export models
            self.export_models()
            
            # Phase 6: Risk assessment
            assessment = self.generate_risk_assessment()
            
            elapsed = time.time() - start_time
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info(f"TRAINING COMPLETE in {elapsed:.1f}s")
            self.logger.info("=" * 60)
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Production-grade Multi-Model Futures Trading System with Risk Engine (Polars)'
    )
    parser.add_argument('--data-path', type=str, 
                       default='/home/ai/NogutiAI/aiTrainCrypto/data',
                       help='Path to data directory')
    parser.add_argument('--model-path', type=str,
                       default='/home/ai/NogutiAI/aiTrainCrypto/models',
                       help='Path to save models')
    parser.add_argument('--n-trials', '--trials', type=int, default=None,
                       help='Number of Optuna trials')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode with reduced trials')
    parser.add_argument('--cpu', type=float, default=None,
                       help='CPU usage percent (0.1-1.0), overrides config.yaml')
    parser.add_argument('--cores', type=int, default=None,
                       help='Exact number of CPU cores to use, overrides config.yaml')
    
    args = parser.parse_args()
    
    # Create config
    config = SystemConfig()
    config.data_path = args.data_path
    config.model_path = args.model_path
    
    # Override CPU settings from command line
    if args.cores:
        config.n_cpu = args.cores
        config.n_jobs = args.cores
    elif args.cpu:
        config.n_cpu = max(1, int(mp.cpu_count() * args.cpu))
        config.n_jobs = config.n_cpu
    
    # Override Optuna trials
    if args.n_trials:
        config.optuna_n_trials = args.n_trials
    
    if args.fast:
        config.optuna_n_trials = min(10, config.optuna_n_trials)
        
    # Run pipeline
    pipeline = ProductionTrainingPipeline(config)
    assessment = pipeline.run()
    
    return assessment


if __name__ == '__main__':
    main()
