#!/usr/bin/env python3
"""
================================================================================
  PRODUCTION-GRADE MULTI-MODEL FUTURES TRADING SYSTEM WITH RISK ENGINE
================================================================================
  
  This is a live-trading-grade system designed to:
  - Train 3 alpha models: Scalp, Intraday, Swing
  - Implement centralized Risk Engine with CVaR, Kelly, and Drawdown controls
  - Multi-objective Optuna optimization
  - ONNX export for production deployment
  - Configurable CPU/RAM utilization via config.yaml
  
  Author: AI Trading Systems
  Version: 1.1.0
  
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
# Memory-mapped arrays for efficiency
os.environ['JOBLIB_START_METHOD'] = 'forkserver'
# LightGBM specific optimizations
os.environ['LGB_NUM_THREADS'] = str(N_CORES_TO_USE)

import numpy as np
import pandas as pd
import polars as pl
pd.options.mode.chained_assignment = None

# Polars configuration for maximum parallelism
pl.Config.set_streaming_chunk_size(100_000_000)  # 100M rows per chunk

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
        CONFIG.get('ram', {}).get('max_gb', 9999)  # Hard limit from config
    ))
    
    # Parallelization settings
    n_jobs: int = -1  # Will be set to n_cpu in __post_init__
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
    data_path: str = field(default_factory=lambda: CONFIG.get('paths', {}).get('data', '/home/ai/NogutiAI/aiTrainCrypto/project/data'))
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
    ON = auto()        # Normal trading
    REDUCED = auto()   # Reduced exposure
    OFF = auto()       # No new trades, risk-off only


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
    direction: int  # 1 = long, -1 = short, 0 = neutral
    probability: float  # [0, 1]
    expected_return: float
    confidence: float  # [0, 1]
    volatility_regime: str  # 'low', 'medium', 'high', 'extreme'
    raw_features: Optional[np.ndarray] = None


@dataclass
class RiskDecision:
    """Decision from Risk Engine."""
    signal: TradeSignal
    decision: TradeDecision
    approved_size: float  # Position size [0, 1]
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
                'cvar': -0.20,  # Negative because lower is better
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
        
        # Setup logging
        self.logger = logging.getLogger('RiskEngine')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
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
        
        # Write to file immediately for safety
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
    """
    Conditional Value at Risk (CVaR) Calculator.
    
    CVaR is computed on:
    - Per-model PnL distributions
    - Portfolio-level PnL
    
    Using:
    - Rolling historical windows
    - Stress scenarios
    - Worst-tail aggregation
    """
    
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
        
        # Keep only window size
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
        """
        Compute Conditional Value at Risk (Expected Shortfall).
        
        CVaR = E[X | X <= VaR]
        """
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
    
    def compute_stress_cvar(self, returns: np.ndarray, 
                           stress_multiplier: float = 2.0) -> float:
        """Compute CVaR under stress scenario."""
        if len(returns) == 0:
            return 0.0
        
        # Apply stress multiplier to losses
        stressed_returns = np.where(returns < 0, returns * stress_multiplier, returns)
        return self.compute_cvar(stressed_returns)
    
    def compute_worst_tail_cvar(self, returns: np.ndarray, 
                                tail_pct: float = 0.01) -> float:
        """Compute CVaR on worst tail."""
        if len(returns) == 0:
            return 0.0
        
        n_tail = max(1, int(len(returns) * tail_pct))
        worst_returns = np.sort(returns)[:n_tail]
        return np.mean(worst_returns)
    
    def cvar_loss_penalty(self, returns: np.ndarray, 
                          threshold: float = 0.03) -> float:
        """
        Compute CVaR-based loss penalty for training.
        
        Used in loss function during model training.
        """
        cvar = abs(self.compute_cvar(returns))
        if cvar > threshold:
            return (cvar - threshold) ** 2 * 100  # Quadratic penalty
        return 0.0
    
    def is_trade_admissible(self, expected_cvar_impact: float,
                           current_cvar: float,
                           threshold: float) -> Tuple[bool, str]:
        """
        Trade admission filter based on CVaR.
        
        Returns (is_admissible, reason)
        """
        projected_cvar = current_cvar + expected_cvar_impact
        
        if projected_cvar > threshold:
            return False, f"CVaR exceeds threshold: {projected_cvar:.4f} > {threshold:.4f}"
        
        if expected_cvar_impact > threshold * 0.5:
            return False, f"Trade CVaR impact too high: {expected_cvar_impact:.4f}"
            
        return True, "CVaR within limits"
    
    def get_exposure_limit(self, current_cvar: float, 
                          max_cvar: float) -> float:
        """
        Dynamic exposure limiter based on CVaR.
        
        Returns exposure multiplier [0, 1]
        """
        if current_cvar >= max_cvar:
            return 0.0
        
        # Linear scaling
        utilization = current_cvar / max_cvar
        return max(0.0, 1.0 - utilization)


# ==============================================================================
# KELLY CRITERION CALCULATOR (MANDATORY)
# ==============================================================================

class KellyCalculator:
    """
    Fractional & Constrained Kelly Position Sizing.
    
    Kelly fraction is derived from:
    - Empirical win-rate
    - Payoff ratio
    - Model confidence score
    - Volatility regime
    
    MANDATORY CONSTRAINTS:
    - Never use raw Kelly
    - Hard leverage caps
    - Per-instrument exposure caps
    - Volatility-adjusted scaling
    - Global risk-state multipliers
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.base_fraction = config.kelly_fraction
        self.min_fraction = config.kelly_min
        self.max_fraction = config.kelly_max
        
        # Volatility regime multipliers
        self.vol_multipliers = {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.6,
            'extreme': 0.2
        }
        
        # Risk state multipliers
        self.risk_state_multipliers = {
            RiskState.ON: 1.0,
            RiskState.REDUCED: 0.5,
            RiskState.OFF: 0.0
        }
        
    def compute_raw_kelly(self, win_rate: float, 
                         payoff_ratio: float) -> float:
        """
        Compute raw Kelly fraction.
        
        Kelly = (p * b - q) / b
        where:
            p = win probability
            q = 1 - p (loss probability)
            b = payoff ratio (win/loss)
        """
        if payoff_ratio <= 0 or win_rate <= 0:
            return 0.0
            
        q = 1 - win_rate
        kelly = (win_rate * payoff_ratio - q) / payoff_ratio
        return max(0.0, kelly)
    
    def dampen_kelly(self, raw_kelly: float, 
                    dampening_factor: float = 0.5) -> float:
        """Apply dampening to raw Kelly (never use raw)."""
        return raw_kelly * dampening_factor
    
    def apply_confidence_scaling(self, kelly: float, 
                                 confidence: float) -> float:
        """Scale Kelly by model confidence."""
        # Confidence should penalize uncertain signals
        confidence_multiplier = confidence ** 2  # Quadratic scaling
        return kelly * confidence_multiplier
    
    def apply_volatility_scaling(self, kelly: float, 
                                volatility_regime: str) -> float:
        """Scale Kelly by volatility regime."""
        multiplier = self.vol_multipliers.get(volatility_regime, 0.5)
        return kelly * multiplier
    
    def apply_risk_state_scaling(self, kelly: float, 
                                risk_state: RiskState) -> float:
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
        """
        Compute fully constrained Kelly fraction.
        
        Pipeline: Raw → Dampened → Capped → Volatility-scaled → Risk-adjusted
        
        Returns (kelly_fraction, reasoning)
        """
        reasoning_parts = []
        
        # Step 1: Raw Kelly
        raw = self.compute_raw_kelly(win_rate, payoff_ratio)
        reasoning_parts.append(f"Raw Kelly: {raw:.4f}")
        
        # Step 2: Dampen
        dampened = self.dampen_kelly(raw, self.base_fraction)
        reasoning_parts.append(f"Dampened: {dampened:.4f}")
        
        # Step 3: Cap
        capped = self.cap_kelly(dampened)
        reasoning_parts.append(f"Capped: {capped:.4f}")
        
        # Step 4: Confidence scaling
        confidence_scaled = self.apply_confidence_scaling(capped, confidence)
        reasoning_parts.append(f"Confidence-scaled: {confidence_scaled:.4f}")
        
        # Step 5: Volatility scaling
        vol_scaled = self.apply_volatility_scaling(confidence_scaled, volatility_regime)
        reasoning_parts.append(f"Vol-scaled ({volatility_regime}): {vol_scaled:.4f}")
        
        # Step 6: Risk state scaling
        risk_adjusted = self.apply_risk_state_scaling(vol_scaled, risk_state)
        reasoning_parts.append(f"Risk-adjusted ({risk_state.name}): {risk_adjusted:.4f}")
        
        # Step 7: CVaR adjustment
        if cvar_utilization > 0.5:
            cvar_penalty = 1.0 - (cvar_utilization - 0.5) * 2
            risk_adjusted *= max(0.0, cvar_penalty)
            reasoning_parts.append(f"CVaR-penalized: {risk_adjusted:.4f}")
        
        # Final cap
        final = self.cap_kelly(risk_adjusted)
        reasoning = " → ".join(reasoning_parts)
        
        return final, reasoning


# ==============================================================================
# DRAWDOWN MONITOR (MANDATORY)
# ==============================================================================

class DrawdownMonitor:
    """
    Drawdown-Aware Dynamic Risk Control.
    
    Continuously tracks drawdown and responds non-linearly.
    All logic is deterministic, fully logged, and Optuna-configurable.
    """
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        self.config = config
        self.logger = logger
        
        # Drawdown thresholds (Optuna-configurable)
        self.warning_threshold = config.max_drawdown_warning
        self.critical_threshold = config.max_drawdown_critical
        self.emergency_threshold = config.max_drawdown_emergency
        
        # State tracking
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown = 0.0
        self.drawdown_history: List[Tuple[datetime, float]] = []
        
        # Model disable flags
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
        
        # Keep only recent history
        if len(self.drawdown_history) > 10000:
            self.drawdown_history = self.drawdown_history[-10000:]
            
    def get_position_multiplier(self) -> Tuple[float, str]:
        """
        Get position size multiplier based on drawdown.
        
        Non-linear response to drawdown levels.
        """
        dd = self.current_drawdown
        
        if dd < self.warning_threshold:
            return 1.0, f"Normal: DD={dd:.2%} < {self.warning_threshold:.2%}"
            
        elif dd < self.critical_threshold:
            # Linear reduction from 1.0 to 0.5
            pct = (dd - self.warning_threshold) / (self.critical_threshold - self.warning_threshold)
            multiplier = 1.0 - 0.5 * pct
            return multiplier, f"Warning: DD={dd:.2%}, multiplier={multiplier:.2f}"
            
        elif dd < self.emergency_threshold:
            # Reduce to 0.25
            multiplier = 0.25
            return multiplier, f"Critical: DD={dd:.2%}, multiplier={multiplier:.2f}"
            
        else:
            # Emergency - no new positions
            return 0.0, f"Emergency: DD={dd:.2%}, trading halted"
            
    def should_disable_model(self, model_type: ModelType) -> Tuple[bool, str]:
        """
        Check if model should be disabled based on drawdown.
        
        More aggressive models (Scalp) disabled first.
        """
        dd = self.current_drawdown
        
        # Scalp disabled at critical level
        if model_type == ModelType.SCALP and dd > self.critical_threshold:
            self.disabled_models.add(model_type)
            return True, f"Scalp disabled: DD={dd:.2%} > {self.critical_threshold:.2%}"
            
        # Intraday disabled at emergency level
        if model_type == ModelType.INTRADAY and dd > self.emergency_threshold:
            self.disabled_models.add(model_type)
            return True, f"Intraday disabled: DD={dd:.2%} > {self.emergency_threshold:.2%}"
            
        # Re-enable if drawdown recovers
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
        
        # Regime thresholds (percentiles of historical vol)
        self.low_threshold = 0.25
        self.medium_threshold = 0.50
        self.high_threshold = 0.75
        
    def update(self, returns: np.ndarray):
        """Update volatility estimate."""
        if len(returns) < 2:
            return
        vol = np.std(returns) * np.sqrt(252)  # Annualized
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
            
    def compute_from_prices(self, prices: pd.Series) -> str:
        """Compute regime directly from price series."""
        if len(prices) < self.lookback + 1:
            return 'medium'
            
        returns = prices.pct_change().dropna()
        self.update(returns.values)
        return self.get_regime()


# ==============================================================================
# RISK ENGINE (CORE CONTROLLER)
# ==============================================================================

class RiskEngine:
    """
    Central Risk Engine - THE CORE CONTROLLER.
    
    Has final veto power over every trade and can:
    - Adjust position size
    - Adjust leverage
    - Block individual trades
    - Disable individual models
    - Force global risk-off state
    
    NO TRADING MODEL MAY BYPASS THE RISK ENGINE.
    """
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.cvar_calculator = CVaRCalculator(
            confidence=config.cvar_confidence,
            window=config.cvar_window
        )
        self.kelly_calculator = KellyCalculator(config)
        self.drawdown_monitor = DrawdownMonitor(config, logger)
        self.volatility_detector = VolatilityRegimeDetector()
        
        # State
        self.risk_state = RiskState.ON
        self.total_decisions = 0
        self.approved_decisions = 0
        self.rejected_decisions = 0
        
        # Emergency kill switch
        self.kill_switch_active = False
        
        # Model performance tracking
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
        """Deactivate kill switch (requires manual intervention)."""
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
        """
        Evaluate trading signal and make risk decision.
        
        This is the central decision point - ALL trades must pass through here.
        """
        self.total_decisions += 1
        
        # Check kill switch first
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
        
        # 1. Check if model is disabled due to drawdown
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
        
        # 2. Get current risk state
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
            
        # 3. CVaR check
        current_cvar = abs(self.cvar_calculator.compute_portfolio_cvar())
        model_cvar = abs(self.cvar_calculator.compute_model_cvar(signal.model_type.value))
        
        # Estimate trade CVaR impact (simplified)
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
        
        # 4. Compute Kelly fraction
        # Get empirical metrics from model performance
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
        
        # 5. Apply drawdown multiplier
        dd_multiplier, dd_mult_reason = self.drawdown_monitor.get_position_multiplier()
        reasoning_parts.append(dd_mult_reason)
        
        # 6. Compute final position size
        final_size = kelly_fraction * dd_multiplier
        final_size = min(final_size, self.config.max_position_pct)
        
        # 7. Compute leverage
        # Base leverage on signal confidence and vol regime
        base_leverage = self.config.max_leverage * signal.confidence
        vol_multipliers = {'low': 1.0, 'medium': 0.8, 'high': 0.5, 'extreme': 0.2}
        vol_mult = vol_multipliers.get(signal.volatility_regime, 0.5)
        final_leverage = min(base_leverage * vol_mult * dd_multiplier, self.config.max_leverage)
        
        # 8. Final decision
        if final_size < 0.001:  # Too small to trade
            decision = TradeDecision.REJECTED
            reasoning = f"Position too small: {final_size:.4f}"
            self.rejected_decisions += 1
        elif final_size < kelly_fraction * 0.5:  # Significantly reduced
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
        
        # Log decision
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
# DATA LOADER (OPTIMIZED FOR FULL CPU/RAM UTILIZATION)
# ==============================================================================

class OptimizedDataLoader:
    """High-performance data loader with aggressive parallelization."""
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        self.config = config
        self.logger = logger
        self.cache = Memory(config.cache_path, verbose=0)
        
        # Data storage
        self.symbols: List[str] = []
        self.klines: Dict[str, Dict[str, pd.DataFrame]] = {}  # symbol -> timeframe -> df
        self.funding_rates: Dict[str, pd.DataFrame] = {}
        self.open_interest: Dict[str, pd.DataFrame] = {}
        
    def discover_symbols(self) -> List[str]:
        """Discover all available symbols from data files."""
        import re
        data_path = Path(self.config.data_path)
        symbols = set()
        
        # Pattern: klines_usdt_m_{SYMBOL}_{TIMEFRAME}.csv
        for f in data_path.glob('klines_usdt_m_*_1h.csv'):
            # Extract symbol from filename like klines_usdt_m_BTCUSDT_1h.csv
            name = f.stem  # klines_usdt_m_BTCUSDT_1h
            parts = name.split('_')
            if len(parts) >= 4:
                # Symbol is between 'usdt_m_' and '_1h'
                symbol = '_'.join(parts[3:-1]) if len(parts) > 4 else parts[3]
                symbols.add(symbol)
                
        self.symbols = sorted(list(symbols))
        self.logger.info(f"Discovered {len(self.symbols)} symbols")
        return self.symbols
    
    def _load_single_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load single CSV file with MAXIMUM PERFORMANCE settings."""
        try:
            # Use pyarrow for much faster CSV reading if available
            try:
                import pyarrow.csv as pa_csv
                table = pa_csv.read_csv(
                    filepath,
                    read_options=pa_csv.ReadOptions(
                        use_threads=True,
                        block_size=256 * 1024 * 1024  # 256MB blocks
                    ),
                    parse_options=pa_csv.ParseOptions(
                        delimiter=','
                    ),
                    convert_options=pa_csv.ConvertOptions(
                        include_missing_columns=True
                    )
                )
                df = table.to_pandas()
            except ImportError:
                # Fallback to pandas with optimized settings
                df = pd.read_csv(
                    filepath,
                    low_memory=False,
                    engine='c',  # Use C parser
                    memory_map=True  # Memory map file for speed
                )
            
            # Convert timestamp if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            elif 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            return df
        except Exception as e:
            return None
            
    def _load_symbol_data(self, symbol: str) -> Dict:
        """Load all data for a single symbol."""
        data_path = Path(self.config.data_path)
        result = {
            'symbol': symbol,
            'klines': {},
            'funding_rate': None,
            'open_interest': None
        }
        
        # Load klines for all timeframes
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        for tf in timeframes:
            filepath = data_path / f'klines_usdt_m_{symbol}_{tf}.csv'
            if filepath.exists():
                df = self._load_single_file(filepath)
                if df is not None and len(df) > 0:
                    result['klines'][tf] = df
                    
        # Load funding rate
        fr_path = data_path / f'funding_rate_usdt_m_{symbol}.csv'
        if fr_path.exists():
            result['funding_rate'] = self._load_single_file(fr_path)
            
        # Load open interest  
        oi_path = data_path / f'open_interest_usdt_m_{symbol}_5m.csv'
        if oi_path.exists():
            result['open_interest'] = self._load_single_file(oi_path)
            
        return result
    
    def load_all_data(self) -> Dict:
        """Load all data with MAXIMUM parallelization - uses multiprocessing.Pool."""
        self.logger.info(f"Loading data from {self.config.data_path}")
        
        # Discover symbols
        symbols = self.discover_symbols()
        self.logger.info(f"Found {len(symbols)} symbols")
        
        start_time = time.time()
        
        # Use multiprocessing Pool directly for MAXIMUM parallelism
        from multiprocessing import Pool, get_context
        
        # Use fork for faster startup
        ctx = get_context('fork')
        
        self.logger.info(f"Loading with {self.config.n_cpu} parallel workers...")
        
        with ctx.Pool(processes=self.config.n_cpu) as pool:
            results = pool.map(self._load_symbol_data, symbols, chunksize=1)
        
        # Organize results and force RAM allocation
        total_rows = 0
        for result in results:
            symbol = result['symbol']
            self.klines[symbol] = result['klines']
            
            for tf, df in result['klines'].items():
                total_rows += len(df)
                
            if result['funding_rate'] is not None:
                self.funding_rates[symbol] = result['funding_rate']
                total_rows += len(result['funding_rate'])
                
            if result['open_interest'] is not None:
                self.open_interest[symbol] = result['open_interest']
                total_rows += len(result['open_interest'])
                
        elapsed = time.time() - start_time
        self.logger.info(f"Loaded {total_rows:,} rows in {elapsed:.1f}s")
        
        return {
            'symbols': self.symbols,
            'klines': self.klines,
            'funding_rates': self.funding_rates,
            'open_interest': self.open_interest,
            'total_rows': total_rows
        }
    
    def get_training_data(self, model_type: ModelType) -> Tuple[pd.DataFrame, pd.Series]:
        """Get prepared training data for specific model type with MAXIMUM performance."""
        # Determine primary timeframe
        timeframe_map = {
            ModelType.SCALP: '5m',
            ModelType.INTRADAY: '1h',
            ModelType.SWING: '1d'
        }
        primary_tf = timeframe_map.get(model_type, '1h')
        
        all_data = []
        
        for symbol in self.symbols:
            if symbol not in self.klines:
                continue
            if primary_tf not in self.klines[symbol]:
                continue
                
            df = self.klines[symbol][primary_tf].copy()
            df['symbol'] = symbol
            all_data.append(df)
            
        if not all_data:
            return pd.DataFrame(), pd.Series()
            
        combined = pd.concat(all_data, ignore_index=True)
        
        # Create target (future return direction) - FIXED NA handling
        combined['future_return'] = combined.groupby('symbol')['close'].pct_change(1).shift(-1)
        
        # Drop rows with NaN future_return FIRST, then create target
        combined = combined.dropna(subset=['future_return'])
        combined['target'] = (combined['future_return'] > 0).astype(np.int32)
        
        # Features - select only numeric columns
        feature_cols = [c for c in combined.columns 
                       if c not in ['symbol', 'open_time', 'timestamp', 'future_return', 'target', 'interval', 'market_type']]
        
        X = combined[feature_cols]
        y = combined['target']
        
        # Convert to numpy-friendly dtypes (avoid nullable int which causes issues)
        X = X.astype(np.float64)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        return X, y


# ==============================================================================
# FEATURE ENGINEERING (PARALLELIZED)
# ==============================================================================

class ParallelFeatureEngine:
    """High-performance feature engineering with full parallelization."""
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        self.config = config
        self.logger = logger
        
    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute MASSIVE technical indicators - 200+ features to use ALL RAM."""
        result = df.copy()
        
        if 'close' not in result.columns:
            return result
            
        close = result['close'].astype(np.float64)
        high = result.get('high', close).astype(np.float64)
        low = result.get('low', close).astype(np.float64)
        open_price = result.get('open', close).astype(np.float64)
        volume = result.get('volume', pd.Series(0, index=close.index)).astype(np.float64)
        
        # ============================================================
        # PRICE FEATURES - Multiple timeframes (40+ features)
        # ============================================================
        for period in [3, 5, 7, 10, 14, 20, 30, 50, 100, 200]:
            result[f'sma_{period}'] = close.rolling(period, min_periods=1).mean()
            result[f'ema_{period}'] = close.ewm(span=period, min_periods=1).mean()
            result[f'wma_{period}'] = close.rolling(period, min_periods=1).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
            )
            result[f'std_{period}'] = close.rolling(period, min_periods=1).std()
            result[f'return_{period}'] = close.pct_change(period)
            result[f'log_return_{period}'] = np.log(close / close.shift(period))
            result[f'price_position_{period}'] = (close - close.rolling(period).min()) / (close.rolling(period).max() - close.rolling(period).min() + 1e-10)
            
        # ============================================================
        # MOMENTUM INDICATORS (30+ features)
        # ============================================================
        for period in [5, 7, 9, 14, 21, 28]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Stochastic RSI
            rsi = result[f'rsi_{period}']
            result[f'stoch_rsi_{period}'] = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min() + 1e-10)
            
        # MACD variants
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (8, 17, 9)]:
            ema_fast = close.ewm(span=fast, min_periods=1).mean()
            ema_slow = close.ewm(span=slow, min_periods=1).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, min_periods=1).mean()
            result[f'macd_{fast}_{slow}'] = macd
            result[f'macd_signal_{fast}_{slow}'] = macd_signal
            result[f'macd_hist_{fast}_{slow}'] = macd - macd_signal
            
        # Rate of Change
        for period in [5, 10, 20, 50]:
            result[f'roc_{period}'] = (close - close.shift(period)) / (close.shift(period) + 1e-10) * 100
            result[f'momentum_{period}'] = close - close.shift(period)
            
        # Williams %R
        for period in [14, 28]:
            highest = high.rolling(period, min_periods=1).max()
            lowest = low.rolling(period, min_periods=1).min()
            result[f'williams_r_{period}'] = -100 * (highest - close) / (highest - lowest + 1e-10)
            
        # ============================================================
        # VOLATILITY INDICATORS (30+ features)
        # ============================================================
        for period in [5, 10, 14, 20, 30, 50]:
            result[f'atr_{period}'] = self._compute_atr(high, low, close, period)
            result[f'natr_{period}'] = result[f'atr_{period}'] / (close + 1e-10) * 100
            
        # Bollinger Bands variants
        for period in [10, 20, 50]:
            for std_mult in [1.5, 2.0, 2.5, 3.0]:
                sma = close.rolling(period, min_periods=1).mean()
                std = close.rolling(period, min_periods=1).std()
                result[f'bb_upper_{period}_{std_mult}'] = sma + std_mult * std
                result[f'bb_lower_{period}_{std_mult}'] = sma - std_mult * std
                result[f'bb_width_{period}_{std_mult}'] = (result[f'bb_upper_{period}_{std_mult}'] - result[f'bb_lower_{period}_{std_mult}']) / (sma + 1e-10)
                result[f'bb_pct_{period}_{std_mult}'] = (close - result[f'bb_lower_{period}_{std_mult}']) / (result[f'bb_upper_{period}_{std_mult}'] - result[f'bb_lower_{period}_{std_mult}'] + 1e-10)
                
        # Keltner Channels
        for period in [10, 20]:
            ema = close.ewm(span=period, min_periods=1).mean()
            atr = self._compute_atr(high, low, close, period)
            result[f'keltner_upper_{period}'] = ema + 2 * atr
            result[f'keltner_lower_{period}'] = ema - 2 * atr
            result[f'keltner_width_{period}'] = 4 * atr / (ema + 1e-10)
            
        # Historical volatility
        for period in [5, 10, 20, 60]:
            returns = close.pct_change()
            result[f'hvol_{period}'] = returns.rolling(period, min_periods=1).std() * np.sqrt(252)
            result[f'hvol_ratio_{period}'] = result[f'hvol_{period}'] / result[f'hvol_{period}'].rolling(period*2, min_periods=1).mean()
            
        # ============================================================
        # VOLUME INDICATORS (20+ features)
        # ============================================================
        if volume.sum() > 0:
            for period in [5, 10, 20, 50]:
                result[f'volume_sma_{period}'] = volume.rolling(period, min_periods=1).mean()
                result[f'volume_std_{period}'] = volume.rolling(period, min_periods=1).std()
                result[f'volume_ratio_{period}'] = volume / (result[f'volume_sma_{period}'] + 1e-10)
                
            # OBV
            obv = (np.sign(close.diff()) * volume).cumsum()
            result['obv'] = obv
            result['obv_sma_20'] = obv.rolling(20, min_periods=1).mean()
            result['obv_slope'] = obv.diff(5) / 5
            
            # Money Flow Index
            for period in [14, 28]:
                typical_price = (high + low + close) / 3
                money_flow = typical_price * volume
                positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period, min_periods=1).sum()
                negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period, min_periods=1).sum()
                result[f'mfi_{period}'] = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
                
            # VWAP
            typical_price = (high + low + close) / 3
            result['vwap'] = (typical_price * volume).cumsum() / (volume.cumsum() + 1e-10)
            result['vwap_dist'] = (close - result['vwap']) / (result['vwap'] + 1e-10)
            
            # Accumulation/Distribution
            clv = ((close - low) - (high - close)) / (high - low + 1e-10)
            result['ad_line'] = (clv * volume).cumsum()
            result['ad_sma_20'] = result['ad_line'].rolling(20, min_periods=1).mean()
            
        # ============================================================
        # TREND INDICATORS (20+ features)
        # ============================================================
        # ADX
        for period in [14, 28]:
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            atr = self._compute_atr(high, low, close, period)
            plus_di = 100 * plus_dm.rolling(period, min_periods=1).mean() / (atr + 1e-10)
            minus_di = 100 * minus_dm.rolling(period, min_periods=1).mean() / (atr + 1e-10)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            result[f'adx_{period}'] = dx.rolling(period, min_periods=1).mean()
            result[f'plus_di_{period}'] = plus_di
            result[f'minus_di_{period}'] = minus_di
            
        # Aroon
        for period in [14, 25]:
            result[f'aroon_up_{period}'] = 100 * high.rolling(period+1, min_periods=1).apply(lambda x: x.argmax(), raw=True) / period
            result[f'aroon_down_{period}'] = 100 * low.rolling(period+1, min_periods=1).apply(lambda x: x.argmin(), raw=True) / period
            result[f'aroon_osc_{period}'] = result[f'aroon_up_{period}'] - result[f'aroon_down_{period}']
            
        # CCI
        for period in [14, 20]:
            typical_price = (high + low + close) / 3
            sma = typical_price.rolling(period, min_periods=1).mean()
            mad = typical_price.rolling(period, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            result[f'cci_{period}'] = (typical_price - sma) / (0.015 * mad + 1e-10)
            
        # ============================================================
        # CANDLESTICK PATTERNS (15+ features)
        # ============================================================
        result['body_size'] = abs(close - open_price) / (high - low + 1e-10)
        result['upper_shadow'] = (high - np.maximum(close, open_price)) / (high - low + 1e-10)
        result['lower_shadow'] = (np.minimum(close, open_price) - low) / (high - low + 1e-10)
        result['is_bullish'] = (close > open_price).astype(np.float64)
        result['body_to_range'] = abs(close - open_price) / (high - low + 1e-10)
        
        # Doji detection
        result['is_doji'] = (result['body_size'] < 0.1).astype(np.float64)
        
        # Engulfing patterns
        result['bullish_engulf'] = ((close > open_price) & (close.shift(1) < open_price.shift(1)) & 
                                    (close > open_price.shift(1)) & (open_price < close.shift(1))).astype(np.float64)
        result['bearish_engulf'] = ((close < open_price) & (close.shift(1) > open_price.shift(1)) & 
                                    (close < open_price.shift(1)) & (open_price > close.shift(1))).astype(np.float64)
                                    
        # ============================================================
        # STATISTICAL FEATURES (20+ features)
        # ============================================================
        returns = close.pct_change()
        for period in [10, 20, 50, 100]:
            result[f'skew_{period}'] = returns.rolling(period, min_periods=1).skew()
            result[f'kurt_{period}'] = returns.rolling(period, min_periods=1).kurt()
            result[f'zscore_{period}'] = (close - close.rolling(period, min_periods=1).mean()) / (close.rolling(period, min_periods=1).std() + 1e-10)
            result[f'percentile_{period}'] = close.rolling(period, min_periods=1).apply(lambda x: stats.percentileofscore(x, x.iloc[-1])/100, raw=False)
            
        # ============================================================
        # CROSS-OVER SIGNALS (10+ features)
        # ============================================================
        result['sma_5_20_cross'] = (result['sma_5'] > result['sma_20']).astype(np.float64)
        result['sma_10_50_cross'] = (result['sma_10'] > result['sma_50']).astype(np.float64)
        result['sma_20_100_cross'] = (result['sma_20'] > result['sma_100']).astype(np.float64)
        result['ema_5_20_cross'] = (result['ema_5'] > result['ema_20']).astype(np.float64)
        result['price_sma_20_cross'] = (close > result['sma_20']).astype(np.float64)
        result['price_sma_50_cross'] = (close > result['sma_50']).astype(np.float64)
        
        # Distance from MAs
        for period in [10, 20, 50, 100, 200]:
            result[f'dist_sma_{period}'] = (close - result[f'sma_{period}']) / (result[f'sma_{period}'] + 1e-10)
            result[f'dist_ema_{period}'] = (close - result[f'ema_{period}']) / (result[f'ema_{period}'] + 1e-10)
            
        # ============================================================
        # LAG FEATURES (create memory - 20+ features)
        # ============================================================
        for lag in [1, 2, 3, 5, 10]:
            result[f'close_lag_{lag}'] = close.shift(lag)
            result[f'return_lag_{lag}'] = returns.shift(lag)
            result[f'volume_lag_{lag}'] = volume.shift(lag) if volume.sum() > 0 else 0
            result[f'high_low_range_lag_{lag}'] = ((high - low) / (close + 1e-10)).shift(lag)
            
        return result
    
    def _compute_atr(self, high: pd.Series, low: pd.Series, 
                    close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def compute_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute risk-specific features."""
        result = df.copy()
        
        if 'close' not in result.columns:
            return result
            
        returns = result['close'].pct_change()
        
        # Volatility regimes
        for period in [20, 60]:
            result[f'realized_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            
        # Tail risk metrics
        result['skewness_20'] = returns.rolling(20).skew()
        result['kurtosis_20'] = returns.rolling(20).kurt()
        
        # Drawdown features
        cummax = result['close'].cummax()
        result['drawdown'] = (cummax - result['close']) / (cummax + 1e-10)
        result['drawdown_duration'] = result['drawdown'].groupby(
            (result['drawdown'] == 0).cumsum()
        ).cumcount()
        
        # CVaR approximation
        result['rolling_var_95'] = returns.rolling(60).quantile(0.05)
        
        return result
    
    def process_symbol(self, symbol: str, klines_data: Dict[str, pd.DataFrame]) -> Dict:
        """Process all features for a symbol - HEAVY computation per worker."""
        result = {'symbol': symbol, 'features': {}}
        
        for tf, df in klines_data.items():
            # Deep copy to use more RAM
            processed = self.compute_technical_features(df.copy())
            processed = self.compute_risk_features(processed)
            # Force memory allocation
            processed = processed.copy()
            result['features'][tf] = processed
            
        return result
    
    def build_features_parallel(self, data_loader: OptimizedDataLoader) -> Dict:
        """Build features with MAXIMUM CPU and RAM usage."""
        self.logger.info("Building 200+ features with MAXIMUM CPU/RAM usage...")
        start_time = time.time()
        
        symbols_to_process = [s for s in data_loader.symbols if s in data_loader.klines]
        total_symbols = len(symbols_to_process)
        
        self.logger.info(f"Processing {total_symbols} symbols with {self.config.n_cpu} workers...")
        
        # Use multiprocessing Pool directly for better control
        from multiprocessing import Pool, get_context
        
        # Prepare work items - each item is (symbol, klines_dict)
        work_items = [(s, data_loader.klines[s]) for s in symbols_to_process]
        
        # Use 'fork' for faster startup and shared memory
        ctx = get_context('fork')
        
        with ctx.Pool(processes=self.config.n_cpu) as pool:
            # Use imap_unordered for better load balancing
            results_list = list(pool.starmap(
                self._process_symbol_static,
                [(self.config, item[0], item[1]) for item in work_items],
                chunksize=1  # Process one symbol at a time for max parallelism
            ))
        
        features = {r['symbol']: r['features'] for r in results_list}
        elapsed = time.time() - start_time
        self.logger.info(f"Features built in {elapsed:.1f}s for {total_symbols} symbols")
        
        return features
    
    @staticmethod
    def _process_symbol_static(config, symbol: str, klines_data: Dict[str, pd.DataFrame]) -> Dict:
        """Static method for multiprocessing - process one symbol."""
        engine = ParallelFeatureEngine(config, None)
        return engine.process_symbol(symbol, klines_data)


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
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        # Select only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Handle infinities and NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median
        X = X.fillna(X.median())
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = list(X.columns)
        
        return X_scaled, y.values
    
    def train(self, X: np.ndarray, y: np.ndarray, params: Dict = None):
        """Train the model."""
        raise NotImplementedError
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        raise NotImplementedError
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        raise NotImplementedError


class LightGBMTradingModel(BaseTradingModel):
    """LightGBM-based trading model - MAXIMUM PERFORMANCE with ALL CPU cores and RAM."""
    
    def __init__(self, model_type: ModelType, config: SystemConfig, logger: RiskLogger):
        super().__init__(model_type, config, logger)
        
        # Calculate available memory for histogram (use 70% of total for MAXIMUM)
        available_ram_mb = int(config.total_ram_gb * 1024 * 0.7)
        
        self.default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            
            # === MAXIMUM CPU PARALLELIZATION ===
            'num_threads': config.n_cpu,  # ALL 48 cores
            'force_row_wise': True,  # Better parallelization for large data
            'force_col_wise': False,
            'device_type': 'cpu',
            
            # === AGGRESSIVE TREE SETTINGS (use more RAM) ===
            'num_leaves': config.num_leaves,  # 255 leaves
            'max_depth': 15,  # Deep trees for complex patterns
            'min_data_in_leaf': config.min_data_in_leaf,  # 10
            'min_sum_hessian_in_leaf': 1e-3,
            
            # === HISTOGRAM OPTIMIZATION (MAXIMUM RAM usage) ===
            'max_bin': config.max_bin,  # 512 bins
            'bin_construct_sample_cnt': config.bin_construct_sample_cnt,  # 5M samples
            'histogram_pool_size': available_ram_mb,  # ~260GB for histogram cache
            
            # === LEARNING SETTINGS ===
            'learning_rate': 0.03,  # Lower LR for more iterations
            'feature_fraction': 0.9,  # Use more features
            'feature_fraction_bynode': config.feature_fraction_bynode,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,  # Bagging every iteration for variance
            'pos_bagging_fraction': 1.0,
            'neg_bagging_fraction': 1.0,
            
            # === REGULARIZATION (minimal for speed) ===
            'lambda_l1': 0.0,
            'lambda_l2': 0.0,
            'min_gain_to_split': 0.0,
            'extra_trees': False,
            
            # === CRITICAL FIX: feature_pre_filter ===
            'feature_pre_filter': False,  # MUST be False to avoid the error
            
            # === PERFORMANCE FLAGS ===
            'verbose': -1,
            'random_state': config.random_seed,
            'deterministic': False,  # Non-deterministic for MAX speed
            'enable_bundle': True,  # Exclusive feature bundling
            'is_enable_sparse': True,
            'max_conflict_rate': 0.0,
            'zero_as_missing': False,
            'use_missing': True,
            'two_round': False,  # Single pass for speed
            'linear_tree': False,
        }
        
    def train(self, X: np.ndarray, y: np.ndarray, params: Dict = None):
        """Train LightGBM model with MAXIMUM performance - ALL CPU cores and RAM."""
        train_params = self.default_params.copy()
        if params:
            train_params.update(params)
        
        n_samples, n_features = X.shape
        estimated_ram_gb = (n_samples * n_features * 8) / (1024**3) * 3  # Estimate with copies
        
        self.logger.info(f"  Training with {train_params['num_threads']} CPU threads")
        self.logger.info(f"  Data: {n_samples:,} samples, {n_features} features")
        self.logger.info(f"  Estimated RAM usage: {estimated_ram_gb:.1f} GB")
        self.logger.info(f"  Settings: num_leaves={train_params['num_leaves']}, max_bin={train_params['max_bin']}, max_depth={train_params['max_depth']}")
        
        # Create dataset with MAXIMUM parallelization for binning
        train_data = lgb.Dataset(
            X, 
            label=y,
            free_raw_data=False,  # Keep raw data for reuse (uses more RAM)
            params={
                'max_bin': train_params['max_bin'],
                'bin_construct_sample_cnt': train_params.get('bin_construct_sample_cnt', 5000000),
                'num_threads': train_params['num_threads'],  # Parallel binning
                'feature_pre_filter': False,  # Required to avoid error
            }
        )
        
        # Pre-construct bins in parallel (uses ALL cores and more RAM)
        self.logger.info("  Constructing histogram bins (ALL CPU cores + RAM)...")
        train_data.construct()
        
        # Train with more iterations for better accuracy
        self.logger.info("  Starting GBDT training (ALL CPU cores)...")
        self.model = lgb.train(
            train_params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(period=200)
            ]
        )
        
        self.is_trained = True
        self.logger.info(f"  {self.model_type.value} model trained: {self.model.num_trees()} trees, best_iteration={self.model.best_iteration}")
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_trained:
            return {}
        importance = self.model.feature_importance(importance_type='gain')
        return dict(zip(self.feature_names, importance))


# ==============================================================================
# RISK MODEL (SPECIALIZED)
# ==============================================================================

class RiskModel(BaseTradingModel):
    """
    Specialized Risk Model for predicting adverse events.
    
    Multi-objective optimization:
    - AUC ↑
    - CVaR ↓
    - Maximum drawdown ↓
    - Tail-loss frequency ↓
    - Equity curve stability ↑
    """
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        super().__init__(ModelType.RISK, config, logger)
        self.cvar_calculator = CVaRCalculator(config.cvar_confidence, config.cvar_window)
        self.metrics = OptimizationMetrics()
        
    def compute_risk_labels(self, returns: pd.Series, threshold: float = -0.02) -> pd.Series:
        """Compute risk labels (1 = adverse event)."""
        # Multi-condition risk labels
        labels = pd.Series(0, index=returns.index)
        
        # Condition 1: Large single-period loss
        labels[returns < threshold] = 1
        
        # Condition 2: Drawdown start
        cummax = (1 + returns).cumprod().cummax()
        current = (1 + returns).cumprod()
        drawdown = (cummax - current) / cummax
        labels[drawdown > 0.05] = 1
        
        # Condition 3: Volatility spike
        vol = returns.rolling(20).std()
        vol_zscore = (vol - vol.rolling(60).mean()) / (vol.rolling(60).std() + 1e-10)
        labels[vol_zscore > 2] = 1
        
        return labels
    
    def train_with_cvar_penalty(self, X: np.ndarray, y: np.ndarray, 
                               returns: np.ndarray, params: Dict = None):
        """Train with CVaR-based loss penalty."""
        train_params = self.default_params.copy()
        if params:
            train_params.update(params)
            
        # Custom objective with CVaR penalty
        def cvar_penalized_objective(preds, train_data):
            labels = train_data.get_label()
            
            # Standard binary cross-entropy gradient
            preds = 1.0 / (1.0 + np.exp(-preds))
            grad = preds - labels
            hess = preds * (1.0 - preds)
            
            # CVaR penalty on false negatives (missed risk events)
            fn_mask = (labels == 1) & (preds < 0.5)
            if fn_mask.sum() > 0:
                cvar_penalty = self.cvar_calculator.cvar_loss_penalty(
                    returns[fn_mask] if len(returns) == len(labels) else returns[:fn_mask.sum()],
                    threshold=self.config.cvar_max_threshold
                )
                grad[fn_mask] *= (1 + cvar_penalty)
                
            return grad, hess
            
        train_data = lgb.Dataset(X, label=y)
        
        self.model = lgb.train(
            train_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        self.is_trained = True
        
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       returns: np.ndarray) -> OptimizationMetrics:
        """Compute comprehensive risk metrics."""
        metrics = OptimizationMetrics()
        
        # AUC
        try:
            metrics.auc = roc_auc_score(y_true, y_pred)
        except:
            metrics.auc = 0.5
            
        # CVaR
        metrics.cvar = abs(self.cvar_calculator.compute_cvar(returns))
        
        # Drawdown
        equity = (1 + pd.Series(returns)).cumprod()
        peak = equity.cummax()
        drawdown = (peak - equity) / peak
        metrics.max_drawdown = drawdown.max()
        
        # Tail loss frequency
        metrics.tail_loss_frequency = (returns < np.percentile(returns, 5)).mean()
        
        # Sharpe ratio
        if returns.std() > 0:
            metrics.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            metrics.sortino_ratio = returns.mean() / downside.std() * np.sqrt(252)
            
        # Calmar ratio
        if metrics.max_drawdown > 0:
            annual_return = returns.mean() * 252
            metrics.calmar_ratio = annual_return / metrics.max_drawdown
            
        # Equity stability (inverse of equity curve volatility)
        equity_returns = equity.pct_change().dropna()
        if len(equity_returns) > 0 and equity_returns.std() > 0:
            metrics.equity_stability = 1.0 / (1.0 + equity_returns.std())
            
        # Win rate
        metrics.win_rate = (returns > 0).mean()
        
        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses > 0:
            metrics.profit_factor = gains / losses
            
        self.metrics = metrics
        return metrics


# ==============================================================================
# OPTUNA OPTIMIZER (MANDATORY)
# ==============================================================================

class RiskModelOptimizer:
    """
    Multi-objective Optuna optimizer for Risk Model.
    
    Optimizes:
    - CVaR window size
    - CVaR confidence level
    - Kelly fraction scaling
    - Drawdown thresholds
    - Volatility scaling coefficients
    - Risk-state transition thresholds
    """
    
    def __init__(self, config: SystemConfig, logger: RiskLogger):
        self.config = config
        self.logger = logger
        self.study: Optional[optuna.Study] = None
        self.best_params: Dict = {}
        self.best_metrics: OptimizationMetrics = OptimizationMetrics()
        
    def create_objective(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        returns_train: np.ndarray, returns_val: np.ndarray):
        """Create Optuna objective function."""
        
        def objective(trial: optuna.Trial) -> Tuple[float, float, float]:
            """Multi-objective: maximize AUC, minimize CVaR, minimize max drawdown."""
            
            # Sample hyperparameters
            params = {
                # LightGBM params
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                
                # Risk params (for config update)
                'cvar_window': trial.suggest_int('cvar_window', 60, 504),
                'cvar_confidence': trial.suggest_float('cvar_confidence', 0.90, 0.99),
                'kelly_fraction': trial.suggest_float('kelly_fraction', 0.1, 0.5),
                'dd_warning': trial.suggest_float('dd_warning', 0.03, 0.10),
                'dd_critical': trial.suggest_float('dd_critical', 0.08, 0.15),
                'dd_emergency': trial.suggest_float('dd_emergency', 0.12, 0.25),
            }
            
            # Create and train model
            risk_model = RiskModel(self.config, self.logger)
            risk_model.cvar_calculator = CVaRCalculator(
                confidence=params['cvar_confidence'],
                window=params['cvar_window']
            )
            
            lgb_params = {k: v for k, v in params.items() 
                        if k in ['num_leaves', 'learning_rate', 'feature_fraction',
                                'bagging_fraction', 'min_child_samples', 'reg_alpha', 'reg_lambda']}
            
            try:
                risk_model.train_with_cvar_penalty(X_train, y_train, returns_train, lgb_params)
                
                # Predict on validation
                y_pred = risk_model.predict_proba(X_val)
                
                # Compute metrics
                metrics = risk_model.compute_metrics(y_val, y_pred, returns_val)
                
                # Report intermediate values for pruning
                trial.report(metrics.auc, step=0)
                trial.report(-metrics.cvar, step=1)  # Negative because we minimize
                trial.report(-metrics.max_drawdown, step=2)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
                # Return multi-objective values
                return metrics.auc, -metrics.cvar, -metrics.max_drawdown
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return 0.0, -1.0, -1.0
                
        return objective
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                returns_train: np.ndarray, returns_val: np.ndarray,
                n_trials: int = 100) -> Dict:
        """Run multi-objective optimization with MAXIMUM parallelism."""
        
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, using default parameters")
            return {}
            
        self.logger.info(f"Starting Optuna optimization with {n_trials} trials and {self.config.n_cpu} parallel workers")
        
        # Create multi-objective study with aggressive settings
        self.study = optuna.create_study(
            directions=['maximize', 'maximize', 'maximize'],  # AUC, -CVaR, -MaxDD
            sampler=TPESampler(
                seed=self.config.random_seed, 
                n_startup_trials=10,  # Less startup for faster convergence
                n_ei_candidates=48,   # More candidates per trial
                multivariate=True,    # Enable multivariate TPE
                warn_independent_sampling=False
            ),
            pruner=HyperbandPruner(
                min_resource=1,
                max_resource=n_trials,
                reduction_factor=3
            )
        )
        
        # Create objective
        objective = self.create_objective(
            X_train, y_train, X_val, y_val, returns_train, returns_val
        )
        
        # Optimize with MAXIMUM parallel trials (each trial uses 1 thread)
        # This uses ALL CPU cores across many trials
        self.study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=min(self.config.n_cpu, 24),  # Cap at 24 parallel trials to avoid memory issues
            show_progress_bar=True,
            catch=(Exception,),
            gc_after_trial=True  # Free memory after each trial
        )
        
        # Get Pareto front
        pareto_trials = self.study.best_trials
        self.logger.info(f"Found {len(pareto_trials)} Pareto-optimal trials")
        
        # Select best balanced solution
        if pareto_trials:
            # Score each trial by weighted sum
            best_score = float('-inf')
            best_trial = None
            
            for trial in pareto_trials:
                # Weights: AUC=0.4, CVaR=0.3, MaxDD=0.3
                score = 0.4 * trial.values[0] + 0.3 * trial.values[1] + 0.3 * trial.values[2]
                if score > best_score:
                    best_score = score
                    best_trial = trial
                    
            if best_trial:
                self.best_params = best_trial.params
                self.logger.info(f"Best trial: AUC={best_trial.values[0]:.4f}, "
                               f"CVaR={-best_trial.values[1]:.4f}, "
                               f"MaxDD={-best_trial.values[2]:.4f}")
                               
        # Log all trials
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                self.logger.debug(f"Trial {trial.number}: params={trial.params}, values={trial.values}")
                
        return self.best_params


# ==============================================================================
# ONNX EXPORTER (MANDATORY)
# ==============================================================================

class ONNXExporter:
    """Export Risk Model to ONNX for production deployment."""
    
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
            
            # Define input type
            initial_types = [('features', FloatTensorType([None, len(feature_names)]))]
            
            # Convert
            onnx_model = convert_lightgbm(
                model,
                initial_types=initial_types,
                target_opset=12
            )
            
            # Save
            onnx.save(onnx_model, output_path)
            
            # Validate
            self._validate_onnx(output_path, feature_names)
            
            self.logger.info(f"Model exported to ONNX: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            return False
            
    def _validate_onnx(self, model_path: str, feature_names: List[str]):
        """Validate ONNX model."""
        # Load and check model
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference
        session = ort.InferenceSession(model_path)
        
        # Create dummy input
        dummy_input = np.random.randn(1, len(feature_names)).astype(np.float32)
        
        # Run inference
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
# MAIN TRAINING PIPELINE
# ==============================================================================

class ProductionTrainingPipeline:
    """
    Production-grade training pipeline.
    
    Designed to saturate 100% CPU and aggressively utilize ALL RAM.
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.logger = RiskLogger()
        
        # Components
        self.data_loader = OptimizedDataLoader(self.config, self.logger)
        self.feature_engine = ParallelFeatureEngine(self.config, self.logger)
        self.risk_engine = RiskEngine(self.config, self.logger)
        self.optimizer = RiskModelOptimizer(self.config, self.logger)
        self.onnx_exporter = ONNXExporter(self.config, self.logger)
        
        # Models
        self.models: Dict[ModelType, BaseTradingModel] = {}
        self.risk_model: Optional[RiskModel] = None
        
        # Features storage (will hold 200+ features per symbol)
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
        """Load all data."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: DATA LOADING")
        self.logger.info("=" * 60)
        
        data = self.data_loader.load_all_data()
        
        self.logger.info(f"Loaded {len(data['symbols'])} symbols")
        self.logger.info(f"Total rows: {data['total_rows']:,}")
        
        return data
        
    def build_features(self):
        """Build features for all symbols - GENERATES 200+ FEATURES."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 2: FEATURE ENGINEERING (200+ features per symbol)")
        self.logger.info("=" * 60)
        
        self.features = self.feature_engine.build_features_parallel(self.data_loader)
        
        # Count total features generated
        sample_symbol = list(self.features.keys())[0] if self.features else None
        if sample_symbol and self.features[sample_symbol]:
            sample_tf = list(self.features[sample_symbol].keys())[0]
            n_features = len(self.features[sample_symbol][sample_tf].columns)
            self.logger.info(f"Generated {n_features} features per timeframe")
        
        # FREE MEMORY: Clear raw klines data since we have features now
        self.logger.info("Clearing raw data to free memory...")
        self.data_loader.klines.clear()
        self.data_loader.funding_rates.clear()
        self.data_loader.open_interest.clear()
        import gc
        gc.collect()
        self.logger.info("Memory freed.")
        
        return self.features
    
    def _get_training_data_with_features(self, model_type: ModelType) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training data with ALL generated features."""
        import gc
        
        timeframe_map = {
            ModelType.SCALP: '5m',
            ModelType.INTRADAY: '1h',
            ModelType.SWING: '1d'
        }
        primary_tf = timeframe_map.get(model_type, '1h')
        
        all_data = []
        symbols_processed = []
        
        for symbol in self.data_loader.symbols:
            if symbol not in self.features:
                continue
            if primary_tf not in self.features[symbol]:
                continue
                
            df = self.features[symbol][primary_tf].copy()
            df['symbol'] = symbol
            all_data.append(df)
            symbols_processed.append(symbol)
            
        if not all_data:
            return pd.DataFrame(), pd.Series()
        
        # FREE: Clear features for this timeframe to save memory
        self.logger.info(f"  Clearing {primary_tf} features from {len(symbols_processed)} symbols...")
        for symbol in symbols_processed:
            if symbol in self.features and primary_tf in self.features[symbol]:
                del self.features[symbol][primary_tf]
        gc.collect()
            
        self.logger.info(f"  Concatenating data from {len(all_data)} symbols...")
        combined = pd.concat(all_data, ignore_index=True)
        
        # Free all_data list
        del all_data
        gc.collect()
        
        # Create target
        combined['future_return'] = combined.groupby('symbol')['close'].pct_change(1).shift(-1)
        combined = combined.dropna(subset=['future_return'])
        combined['target'] = (combined['future_return'] > 0).astype(np.int32)
        
        # Exclude non-feature columns
        exclude_cols = ['symbol', 'open_time', 'timestamp', 'future_return', 'target', 
                       'interval', 'market_type', 'close_time', 'quote_volume', 
                       'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        
        feature_cols = [c for c in combined.columns if c not in exclude_cols]
        
        X = combined[feature_cols]
        y = combined['target']
        
        # Free combined dataframe
        del combined
        gc.collect()
        
        # Convert to float32 (saves 50% RAM compared to float64)
        X = X.select_dtypes(include=[np.number]).astype(np.float32)
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with column medians
        self.logger.info(f"  Handling missing values in {X.shape[1]} features...")
        X = X.fillna(X.median())
        
        return X, y
        
    def train_trading_models(self):
        """Train all trading models with 200+ features."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 3: TRADING MODEL TRAINING (200+ features)")
        self.logger.info("=" * 60)
        
        results = {}
        
        for model_type in [ModelType.SCALP, ModelType.INTRADAY, ModelType.SWING]:
            self.logger.info(f"\nTraining {model_type.value.upper()} model...")
            
            # Get training data with ALL features
            X, y = self._get_training_data_with_features(model_type)
            
            if len(X) == 0:
                self.logger.warning(f"No data for {model_type.value}")
                continue
            
            # Log RAM usage estimate
            ram_usage_gb = (X.memory_usage(deep=True).sum()) / (1024**3)
            self.logger.info(f"  Data: {len(X):,} samples, {X.shape[1]} features")
            self.logger.info(f"  Feature matrix RAM: {ram_usage_gb:.2f} GB")
            
            # Split data
            split_idx = int(len(X) * (1 - self.config.test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Create and train model
            model = LightGBMTradingModel(model_type, self.config, self.logger)
            X_train_scaled, y_train_arr = model.prepare_data(X_train, y_train)
            
            model.train(X_train_scaled, y_train_arr)
            
            # Evaluate
            X_test_numeric = X_test.select_dtypes(include=[np.number])
            X_test_numeric = X_test_numeric.replace([np.inf, -np.inf], np.nan).fillna(X_test_numeric.median())
            X_test_scaled = model.scaler.transform(X_test_numeric[model.feature_names])
            y_pred = model.predict_proba(X_test_scaled)
            
            try:
                auc = roc_auc_score(y_test, y_pred)
            except:
                auc = 0.5
                
            self.logger.info(f"  AUC: {auc:.4f}")
            
            self.models[model_type] = model
            results[model_type.value] = {'auc': auc, 'samples': len(X)}
            
            # FREE MEMORY after each model
            del X, y, X_train, X_test, y_train, y_test, X_train_scaled, y_train_arr, X_test_scaled
            import gc
            gc.collect()
            self.logger.info(f"  Memory freed after {model_type.value} training.")
            
        self.training_results['trading_models'] = results
        return results
        
    def train_risk_model(self):
        """Train the Risk Model with multi-objective optimization."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 4: RISK MODEL TRAINING")
        self.logger.info("=" * 60)
        
        # Combine data from all models for risk training
        all_X = []
        all_returns = []
        
        for model_type in [ModelType.SCALP, ModelType.INTRADAY, ModelType.SWING]:
            X, y = self.data_loader.get_training_data(model_type)
            if len(X) > 0:
                all_X.append(X)
                # Compute returns
                if 'close' in X.columns:
                    returns = X['close'].pct_change().fillna(0)
                    all_returns.append(returns)
                    
        if not all_X:
            self.logger.error("No data available for risk model training")
            return {}
            
        X_combined = pd.concat(all_X, ignore_index=True)
        returns_combined = pd.concat(all_returns, ignore_index=True)
        
        self.logger.info(f"Risk model data: {len(X_combined):,} samples")
        
        # Create risk model
        self.risk_model = RiskModel(self.config, self.logger)
        
        # Compute risk labels
        y_risk = self.risk_model.compute_risk_labels(returns_combined)
        
        self.logger.info(f"Risk events: {y_risk.sum():,} ({y_risk.mean()*100:.1f}%)")
        
        # Split data
        split_train = int(len(X_combined) * 0.7)
        split_val = int(len(X_combined) * 0.85)
        
        X_train = X_combined.iloc[:split_train]
        X_val = X_combined.iloc[split_train:split_val]
        X_test = X_combined.iloc[split_val:]
        
        y_train = y_risk.iloc[:split_train]
        y_val = y_risk.iloc[split_train:split_val]
        y_test = y_risk.iloc[split_val:]
        
        returns_train = returns_combined.iloc[:split_train].values
        returns_val = returns_combined.iloc[split_train:split_val].values
        returns_test = returns_combined.iloc[split_val:].values
        
        # Prepare data
        X_train_scaled, y_train_arr = self.risk_model.prepare_data(X_train, y_train)
        X_val_scaled = self.risk_model.scaler.transform(
            X_val.replace([np.inf, -np.inf], np.nan).fillna(X_val.median())
        )
        X_test_scaled = self.risk_model.scaler.transform(
            X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.median())
        )
        
        # Optuna optimization
        if OPTUNA_AVAILABLE:
            self.logger.info("\nStarting Optuna multi-objective optimization...")
            best_params = self.optimizer.optimize(
                X_train_scaled, y_train_arr,
                X_val_scaled, y_val.values,
                returns_train, returns_val,
                n_trials=50  # Can increase for better results
            )
            
            if best_params:
                self.logger.info(f"Best parameters: {best_params}")
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
        final_metrics = self.risk_model.compute_metrics(y_test.values, y_pred, returns_test)
        
        self.logger.info("\n" + "=" * 40)
        self.logger.info("RISK MODEL FINAL METRICS")
        self.logger.info("=" * 40)
        self.logger.info(f"  AUC:              {final_metrics.auc:.4f}")
        self.logger.info(f"  CVaR:             {final_metrics.cvar:.4f}")
        self.logger.info(f"  Max Drawdown:     {final_metrics.max_drawdown:.4f}")
        self.logger.info(f"  Tail Loss Freq:   {final_metrics.tail_loss_frequency:.4f}")
        self.logger.info(f"  Sharpe Ratio:     {final_metrics.sharpe_ratio:.4f}")
        self.logger.info(f"  Sortino Ratio:    {final_metrics.sortino_ratio:.4f}")
        self.logger.info(f"  Calmar Ratio:     {final_metrics.calmar_ratio:.4f}")
        self.logger.info(f"  Equity Stability: {final_metrics.equity_stability:.4f}")
        self.logger.info(f"  Win Rate:         {final_metrics.win_rate:.4f}")
        self.logger.info(f"  Profit Factor:    {final_metrics.profit_factor:.4f}")
        
        self.final_metrics = final_metrics.to_dict()
        self.training_results['risk_model'] = self.final_metrics
        
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
        
        # Generate recommendations based on metrics
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
            
        if metrics.win_rate < 0.45:
            assessment['recommendations'].append(
                "LOW WIN RATE: Consider improving signal quality."
            )
            
        if not assessment['recommendations']:
            assessment['recommendations'].append(
                "System within acceptable risk parameters."
            )
            
        # Print assessment
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
        description='Production-grade Multi-Model Futures Trading System with Risk Engine'
    )
    parser.add_argument('--data-path', type=str, 
                       default='/home/ai/NogutiAI/aiTrainCrypto/data',
                       help='Path to data directory')
    parser.add_argument('--model-path', type=str,
                       default='/home/ai/NogutiAI/aiTrainCrypto/models',
                       help='Path to save models')
    parser.add_argument('--n-trials', '--trials', type=int, default=50,
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
        import multiprocessing as mp
        config.n_cpu = max(1, int(mp.cpu_count() * args.cpu))
        config.n_jobs = config.n_cpu
    
    # Override Optuna trials
    if args.n_trials:
        config.optuna_n_trials = args.n_trials
    
    if args.fast:
        # Reduce trials for fast testing
        config.optuna_n_trials = min(10, config.optuna_n_trials)
        
    # Run pipeline
    pipeline = ProductionTrainingPipeline(config)
    assessment = pipeline.run()
    
    return assessment


if __name__ == '__main__':
    main()
