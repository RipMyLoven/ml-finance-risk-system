"""
CVaR (Conditional Value at Risk) Engine

Implements tail-risk estimation for:
- Per-model risk assessment
- Portfolio-level risk aggregation
- Stress scenario analysis

CVaR is used as:
- A penalty in the loss function
- A trade admission filter
- A position sizing constraint
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from scipy import stats


class RiskLevel(Enum):
    """Risk level classification for CVaR thresholds"""
    MINIMAL = "minimal"      # CVaR < 1%
    LOW = "low"              # CVaR 1-2%
    MODERATE = "moderate"    # CVaR 2-5%
    HIGH = "high"            # CVaR 5-10%
    EXTREME = "extreme"      # CVaR > 10%
    BLOCKED = "blocked"      # CVaR exceeds blocking threshold


@dataclass
class CVaRResult:
    """Container for CVaR calculation results"""
    var: float                      # Value at Risk
    cvar: float                     # Conditional VaR (Expected Shortfall)
    confidence_level: float         # Confidence level used
    window_size: int                # Window size used
    n_samples: int                  # Number of samples
    risk_level: RiskLevel           # Classified risk level
    tail_observations: int          # Number of tail observations
    max_loss: float                 # Maximum observed loss
    is_blocked: bool                # Whether trade should be blocked
    stress_cvar: Optional[float]    # CVaR under stress scenario
    
    def to_dict(self) -> Dict:
        return {
            'var': self.var,
            'cvar': self.cvar,
            'confidence_level': self.confidence_level,
            'window_size': self.window_size,
            'n_samples': self.n_samples,
            'risk_level': self.risk_level.value,
            'tail_observations': self.tail_observations,
            'max_loss': self.max_loss,
            'is_blocked': self.is_blocked,
            'stress_cvar': self.stress_cvar
        }


class CVaREngine:
    """
    Comprehensive CVaR Engine for risk estimation
    
    Implements:
    - Historical CVaR calculation
    - Rolling window CVaR
    - Stress scenario CVaR
    - Portfolio-level CVaR aggregation
    - Trade admission filtering
    
    Justification:
    - CVaR (Expected Shortfall) captures tail risk better than VaR
    - Historical approach is non-parametric and captures actual distribution
    - Stress scenarios account for regime changes
    - Per-model + portfolio aggregation enables layered risk control
    """
    
    # CVaR thresholds for risk classification (as percentage loss)
    RISK_THRESHOLDS = {
        RiskLevel.MINIMAL: 0.01,    # 1%
        RiskLevel.LOW: 0.02,        # 2%
        RiskLevel.MODERATE: 0.05,   # 5%
        RiskLevel.HIGH: 0.10,       # 10%
        RiskLevel.EXTREME: 0.15,    # 15%
    }
    
    # Default blocking threshold
    DEFAULT_BLOCK_THRESHOLD = 0.08  # Block trades when CVaR > 8%
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        default_window: int = 252,  # ~1 year of daily data
        min_samples: int = 30,
        stress_multiplier: float = 1.5,
        block_threshold: Optional[float] = None,
        model_cvar_limits: Optional[Dict[str, float]] = None
    ):
        """
        Initialize CVaR Engine
        
        Args:
            confidence_level: Confidence level for VaR/CVaR (0.95 = 95%)
            default_window: Default rolling window size
            min_samples: Minimum samples required for calculation
            stress_multiplier: Multiplier for stress scenario CVaR
            block_threshold: CVaR threshold for blocking trades
            model_cvar_limits: Per-model CVaR limits {model_type: threshold}
        """
        self.confidence_level = confidence_level
        self.default_window = default_window
        self.min_samples = min_samples
        self.stress_multiplier = stress_multiplier
        self.block_threshold = block_threshold or self.DEFAULT_BLOCK_THRESHOLD
        
        # Per-model limits (more aggressive for scalp, conservative for swing)
        self.model_cvar_limits = model_cvar_limits or {
            'scalp': 0.05,      # 5% CVaR limit for scalp
            'intraday': 0.07,   # 7% for intraday
            'swing': 0.10,      # 10% for swing
            'portfolio': 0.08   # 8% for portfolio
        }
        
        # Cache for rolling calculations
        self._pnl_history: Dict[str, List[float]] = {
            'scalp': [],
            'intraday': [],
            'swing': [],
            'portfolio': []
        }
        
        # Stress scenarios (historical crisis periods)
        self._stress_returns: Optional[np.ndarray] = None
        
    def calculate_cvar(
        self,
        returns: Union[np.ndarray, pd.Series, List[float]],
        confidence_level: Optional[float] = None,
        window_size: Optional[int] = None
    ) -> CVaRResult:
        """
        Calculate historical CVaR (Expected Shortfall)
        
        CVaR = E[Loss | Loss > VaR]
        
        Args:
            returns: Array of returns (can be PnL or percentage returns)
            confidence_level: Override confidence level
            window_size: Override window size
            
        Returns:
            CVaRResult with comprehensive risk metrics
            
        Justification:
            Historical CVaR uses actual distribution without parametric assumptions,
            capturing fat tails and skewness present in crypto markets.
        """
        # Convert to numpy array
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        elif isinstance(returns, list):
            returns = np.array(returns)
        
        returns = returns[~np.isnan(returns)]
        
        # Apply window if specified
        window = window_size or self.default_window
        if len(returns) > window:
            returns = returns[-window:]
        
        n_samples = len(returns)
        
        if n_samples < self.min_samples:
            # Insufficient data - return conservative estimate
            return CVaRResult(
                var=0.0,
                cvar=self.block_threshold,  # Assume worst case
                confidence_level=confidence_level or self.confidence_level,
                window_size=window,
                n_samples=n_samples,
                risk_level=RiskLevel.BLOCKED,
                tail_observations=0,
                max_loss=0.0,
                is_blocked=True,
                stress_cvar=self.block_threshold * self.stress_multiplier
            )
        
        conf = confidence_level or self.confidence_level
        
        # Calculate VaR (percentile of losses)
        # Note: We work with losses (negative returns), so we flip signs
        losses = -returns
        var_threshold = np.percentile(losses, conf * 100)
        
        # CVaR = Average of losses exceeding VaR
        tail_losses = losses[losses >= var_threshold]
        
        if len(tail_losses) == 0:
            # No tail observations - use VaR as estimate
            cvar = var_threshold
            tail_obs = 0
        else:
            cvar = np.mean(tail_losses)
            tail_obs = len(tail_losses)
        
        # Maximum observed loss
        max_loss = np.max(losses) if len(losses) > 0 else 0.0
        
        # Calculate stress CVaR
        stress_cvar = self._calculate_stress_cvar(returns, cvar)
        
        # Classify risk level
        risk_level = self._classify_risk(cvar)
        
        # Determine if trade should be blocked
        is_blocked = cvar > self.block_threshold or stress_cvar > (self.block_threshold * 1.5)
        
        return CVaRResult(
            var=float(var_threshold),
            cvar=float(cvar),
            confidence_level=conf,
            window_size=window,
            n_samples=n_samples,
            risk_level=risk_level,
            tail_observations=tail_obs,
            max_loss=float(max_loss),
            is_blocked=is_blocked,
            stress_cvar=float(stress_cvar)
        )
    
    def calculate_rolling_cvar(
        self,
        returns: Union[np.ndarray, pd.Series],
        window: int = 60
    ) -> pd.Series:
        """
        Calculate rolling CVaR
        
        Args:
            returns: Time series of returns
            window: Rolling window size
            
        Returns:
            Series of rolling CVaR values
            
        Justification:
            Rolling CVaR captures time-varying risk, essential for
            detecting regime changes in volatile crypto markets.
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        def _cvar_window(x):
            if len(x) < 10:
                return np.nan
            losses = -x
            var_threshold = np.percentile(losses, self.confidence_level * 100)
            tail_losses = losses[losses >= var_threshold]
            return np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
        
        return returns.rolling(window=window, min_periods=self.min_samples).apply(_cvar_window)
    
    def calculate_portfolio_cvar(
        self,
        model_returns: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> CVaRResult:
        """
        Calculate portfolio-level CVaR with correlation adjustment
        
        Args:
            model_returns: Dict of returns per model {model_type: returns}
            weights: Portfolio weights per model
            correlation_matrix: Correlation between models (if None, calculated)
            
        Returns:
            Portfolio CVaR result
            
        Justification:
            Portfolio CVaR accounts for diversification benefits and
            correlation risk. Critical for multi-model systems where
            models may be correlated during stress periods.
        """
        if weights is None:
            # Equal weights if not specified
            weights = {model: 1.0 / len(model_returns) for model in model_returns}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Stack returns and compute portfolio returns
        model_names = list(model_returns.keys())
        min_len = min(len(r) for r in model_returns.values())
        
        # Align returns
        aligned_returns = np.column_stack([
            model_returns[name][-min_len:] for name in model_names
        ])
        
        # Calculate weights array
        weight_array = np.array([weights.get(name, 0.0) for name in model_names])
        
        # Portfolio returns
        portfolio_returns = aligned_returns @ weight_array
        
        # Calculate portfolio CVaR
        result = self.calculate_cvar(portfolio_returns)
        
        # Adjust for correlation risk (conservative estimate during stress)
        if correlation_matrix is None:
            correlation_matrix = np.corrcoef(aligned_returns.T)
        
        # If high correlation detected, increase CVaR estimate
        avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        
        if avg_correlation > 0.7:
            # High correlation = diversification breakdown risk
            correlation_adjustment = 1.0 + (avg_correlation - 0.5)
            adjusted_cvar = result.cvar * correlation_adjustment
            
            return CVaRResult(
                var=result.var,
                cvar=adjusted_cvar,
                confidence_level=result.confidence_level,
                window_size=result.window_size,
                n_samples=result.n_samples,
                risk_level=self._classify_risk(adjusted_cvar),
                tail_observations=result.tail_observations,
                max_loss=result.max_loss,
                is_blocked=adjusted_cvar > self.model_cvar_limits.get('portfolio', self.block_threshold),
                stress_cvar=result.stress_cvar * correlation_adjustment if result.stress_cvar else None
            )
        
        return result
    
    def _calculate_stress_cvar(
        self,
        returns: np.ndarray,
        base_cvar: float
    ) -> float:
        """
        Calculate CVaR under stress scenario
        
        Justification:
            Stress CVaR accounts for tail risk amplification during
            market stress. Uses multiplier approach when historical
            stress data not available.
        """
        # Method 1: Historical stress if available
        if self._stress_returns is not None and len(self._stress_returns) >= self.min_samples:
            stress_result = self.calculate_cvar(self._stress_returns)
            return max(stress_result.cvar, base_cvar * self.stress_multiplier)
        
        # Method 2: Scaled CVaR with fat-tail adjustment
        # Assume stress scenario amplifies tail risk
        stress_cvar = base_cvar * self.stress_multiplier
        
        # Additional adjustment for extreme observations
        extreme_threshold = np.percentile(-returns, 99) if len(returns) > 0 else 0
        if extreme_threshold > base_cvar * 2:
            # Extreme events detected - increase stress estimate
            stress_cvar = max(stress_cvar, extreme_threshold * 1.2)
        
        return stress_cvar
    
    def _classify_risk(self, cvar: float) -> RiskLevel:
        """Classify risk level based on CVaR value"""
        if cvar <= self.RISK_THRESHOLDS[RiskLevel.MINIMAL]:
            return RiskLevel.MINIMAL
        elif cvar <= self.RISK_THRESHOLDS[RiskLevel.LOW]:
            return RiskLevel.LOW
        elif cvar <= self.RISK_THRESHOLDS[RiskLevel.MODERATE]:
            return RiskLevel.MODERATE
        elif cvar <= self.RISK_THRESHOLDS[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif cvar <= self.RISK_THRESHOLDS[RiskLevel.EXTREME]:
            return RiskLevel.EXTREME
        else:
            return RiskLevel.BLOCKED
    
    def check_trade_admission(
        self,
        model_type: str,
        recent_returns: np.ndarray,
        expected_return: float,
        position_size: float
    ) -> Tuple[bool, str]:
        """
        Check if trade should be admitted based on CVaR
        
        Args:
            model_type: Type of model (scalp/intraday/swing)
            recent_returns: Recent PnL history
            expected_return: Expected return of proposed trade
            position_size: Proposed position size
            
        Returns:
            (is_admitted, reason)
            
        Justification:
            Trade admission filter prevents entering trades when
            tail risk exceeds acceptable levels. Essential for
            capital preservation during volatile periods.
        """
        # Calculate current CVaR
        result = self.calculate_cvar(recent_returns)
        
        # Get model-specific limit
        cvar_limit = self.model_cvar_limits.get(model_type, self.block_threshold)
        
        # Check if CVaR exceeds limit
        if result.cvar > cvar_limit:
            return False, f"CVaR {result.cvar:.2%} exceeds {model_type} limit {cvar_limit:.2%}"
        
        # Check if stress CVaR is too high
        if result.stress_cvar and result.stress_cvar > cvar_limit * 1.5:
            return False, f"Stress CVaR {result.stress_cvar:.2%} exceeds stress limit"
        
        # Check expected return vs CVaR (reward-to-risk)
        if result.cvar > 0 and expected_return / result.cvar < 0.5:
            return False, f"Expected return/CVaR ratio {expected_return/result.cvar:.2f} too low"
        
        return True, "Trade admitted"
    
    def update_pnl_history(
        self,
        model_type: str,
        pnl: float,
        max_history: int = 1000
    ):
        """
        Update PnL history for a model
        
        Args:
            model_type: Model type
            pnl: Latest PnL value
            max_history: Maximum history to maintain
        """
        if model_type not in self._pnl_history:
            self._pnl_history[model_type] = []
        
        self._pnl_history[model_type].append(pnl)
        
        # Trim if exceeds max
        if len(self._pnl_history[model_type]) > max_history:
            self._pnl_history[model_type] = self._pnl_history[model_type][-max_history:]
    
    def get_model_cvar(self, model_type: str) -> CVaRResult:
        """Get current CVaR for a model from history"""
        if model_type not in self._pnl_history or len(self._pnl_history[model_type]) < self.min_samples:
            # Return conservative estimate
            return CVaRResult(
                var=0.0,
                cvar=self.model_cvar_limits.get(model_type, self.block_threshold),
                confidence_level=self.confidence_level,
                window_size=0,
                n_samples=len(self._pnl_history.get(model_type, [])),
                risk_level=RiskLevel.HIGH,
                tail_observations=0,
                max_loss=0.0,
                is_blocked=False,
                stress_cvar=None
            )
        
        return self.calculate_cvar(np.array(self._pnl_history[model_type]))
    
    def set_stress_scenarios(self, stress_returns: np.ndarray):
        """
        Set historical stress scenario returns for stress testing
        
        Args:
            stress_returns: Returns from historical stress period
                           (e.g., March 2020, May 2021 crypto crash)
        """
        self._stress_returns = stress_returns
    
    def get_cvar_penalty(
        self,
        returns: np.ndarray,
        penalty_weight: float = 0.1
    ) -> float:
        """
        Calculate CVaR penalty for loss function
        
        Args:
            returns: Model returns
            penalty_weight: Weight of CVaR penalty
            
        Returns:
            CVaR penalty value to add to loss
            
        Justification:
            Adding CVaR as a penalty encourages models to
            avoid strategies with high tail risk.
        """
        result = self.calculate_cvar(returns)
        return penalty_weight * result.cvar
    
    def to_features(self, returns: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """
        Convert CVaR metrics to features for ML model
        
        Args:
            returns: Return series
            prefix: Feature name prefix
            
        Returns:
            Dict of CVaR-derived features
        """
        result = self.calculate_cvar(returns)
        
        prefix = f"{prefix}_" if prefix else ""
        
        return {
            f'{prefix}cvar_95': result.cvar,
            f'{prefix}var_95': result.var,
            f'{prefix}max_loss': result.max_loss,
            f'{prefix}cvar_stress': result.stress_cvar or result.cvar * self.stress_multiplier,
            f'{prefix}risk_level': list(RiskLevel).index(result.risk_level),
            f'{prefix}tail_ratio': result.tail_observations / max(result.n_samples, 1)
        }
