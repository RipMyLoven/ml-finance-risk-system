"""
Kelly-Based Position Sizing Module

Implements institutional-grade position sizing with:
- Fractional Kelly criterion
- Constrained Kelly with maximum leverage limits
- Volatility-adjusted scaling
- Model confidence integration

IMPORTANT: Kelly is NEVER used in raw form - always capped and dampened.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class KellyMode(Enum):
    """Kelly calculation modes"""
    FULL = "full"               # Full Kelly (not recommended for live)
    HALF = "half"               # Half Kelly (common default)
    QUARTER = "quarter"         # Quarter Kelly (conservative)
    ADAPTIVE = "adaptive"       # Adaptive based on confidence


@dataclass
class KellyResult:
    """Container for Kelly calculation results"""
    raw_kelly: float            # Raw Kelly fraction (unbounded)
    kelly_fraction: float       # Applied Kelly fraction
    optimal_size: float         # Optimal position size
    final_size: float           # Final size after all constraints
    win_rate: float             # Win rate used
    payoff_ratio: float         # Payoff ratio used
    confidence_adjustment: float  # Confidence-based adjustment
    volatility_scaling: float   # Volatility scaling factor
    leverage_cap: float         # Maximum leverage applied
    constraints_applied: list   # List of constraints that modified size
    
    def to_dict(self) -> Dict:
        return {
            'raw_kelly': self.raw_kelly,
            'kelly_fraction': self.kelly_fraction,
            'optimal_size': self.optimal_size,
            'final_size': self.final_size,
            'win_rate': self.win_rate,
            'payoff_ratio': self.payoff_ratio,
            'confidence_adjustment': self.confidence_adjustment,
            'volatility_scaling': self.volatility_scaling,
            'leverage_cap': self.leverage_cap,
            'constraints_applied': self.constraints_applied
        }


class KellySizer:
    """
    Kelly Criterion Position Sizer with Institutional Constraints
    
    The Kelly Criterion calculates the optimal bet size to maximize
    logarithmic utility (long-term growth). However, raw Kelly is
    too aggressive for live trading due to:
    - Estimation error in win rate and payoff
    - Non-stationarity of market parameters
    - Psychological factors and drawdown tolerance
    
    This implementation ALWAYS applies:
    1. Fractional Kelly (typically 1/4 to 1/2 of optimal)
    2. Maximum leverage constraints
    3. Maximum exposure per instrument
    4. Volatility-adjusted scaling
    5. Model confidence dampening
    
    Justification:
    - Fractional Kelly reduces variance of returns by 75% at 50% Kelly
    - Leverage caps prevent catastrophic losses
    - Volatility scaling accounts for changing market conditions
    - Confidence dampening reduces size when model is uncertain
    """
    
    # Absolute maximum allowed values
    MAX_KELLY_FRACTION = 0.25   # Never use more than 25% of Kelly
    MAX_LEVERAGE = 10.0         # Maximum leverage regardless of Kelly
    MAX_SINGLE_POSITION = 0.15  # Max 15% of capital in single position
    MIN_POSITION_SIZE = 0.001   # Minimum meaningful position
    
    # Default fractional Kelly by mode
    KELLY_FRACTIONS = {
        KellyMode.FULL: 1.0,        # Never actually use this
        KellyMode.HALF: 0.5,
        KellyMode.QUARTER: 0.25,
        KellyMode.ADAPTIVE: None     # Calculated dynamically
    }
    
    def __init__(
        self,
        mode: KellyMode = KellyMode.QUARTER,
        max_leverage: float = 5.0,
        max_position_pct: float = 0.10,
        min_win_rate: float = 0.35,
        base_volatility: float = 0.02,
        model_leverage_limits: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Kelly Sizer
        
        Args:
            mode: Kelly fraction mode
            max_leverage: Maximum allowed leverage
            max_position_pct: Maximum position as % of capital
            min_win_rate: Minimum required win rate for positive Kelly
            base_volatility: Base volatility for scaling
            model_leverage_limits: Per-model leverage limits
        """
        self.mode = mode
        self.max_leverage = min(max_leverage, self.MAX_LEVERAGE)
        self.max_position_pct = min(max_position_pct, self.MAX_SINGLE_POSITION)
        self.min_win_rate = min_win_rate
        self.base_volatility = base_volatility
        
        # Per-model leverage limits (scalp can have higher leverage)
        self.model_leverage_limits = model_leverage_limits or {
            'scalp': 10.0,
            'intraday': 5.0,
            'swing': 3.0
        }
        
        # Historical stats for adaptive mode
        self._win_rate_history: Dict[str, list] = {}
        self._payoff_history: Dict[str, list] = {}
        
    def calculate_kelly(
        self,
        win_rate: float,
        payoff_ratio: float,
        confidence: float = 1.0,
        current_volatility: float = None,
        model_type: str = 'intraday',
        capital: float = 10000.0
    ) -> KellyResult:
        """
        Calculate Kelly-based position size with all constraints
        
        Kelly formula: f* = (p*b - q) / b
        where:
            p = probability of win
            q = probability of loss (1-p)
            b = payoff ratio (win amount / loss amount)
            f* = fraction of capital to bet
        
        Args:
            win_rate: Probability of winning trade
            payoff_ratio: Average win / average loss
            confidence: Model confidence score [0, 1]
            current_volatility: Current market volatility
            model_type: Type of trading model
            capital: Available capital
            
        Returns:
            KellyResult with all calculations and constraints
            
        Justification:
            This method implements a defensive Kelly with multiple
            layers of protection. Even with estimation errors,
            the position size will be bounded and reasonable.
        """
        constraints_applied = []
        
        # Validate inputs
        win_rate = np.clip(win_rate, 0.01, 0.99)
        payoff_ratio = max(payoff_ratio, 0.01)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Calculate raw Kelly
        q = 1 - win_rate
        raw_kelly = (win_rate * payoff_ratio - q) / payoff_ratio
        
        # Handle negative Kelly (don't trade)
        if raw_kelly <= 0:
            return KellyResult(
                raw_kelly=raw_kelly,
                kelly_fraction=0.0,
                optimal_size=0.0,
                final_size=0.0,
                win_rate=win_rate,
                payoff_ratio=payoff_ratio,
                confidence_adjustment=confidence,
                volatility_scaling=1.0,
                leverage_cap=0.0,
                constraints_applied=['negative_kelly']
            )
        
        # Check minimum win rate
        if win_rate < self.min_win_rate:
            constraints_applied.append('min_win_rate')
            raw_kelly *= 0.5  # Reduce significantly if low win rate
        
        # Apply Kelly fraction (NEVER use full Kelly)
        if self.mode == KellyMode.ADAPTIVE:
            # Adaptive: scale based on confidence
            kelly_fraction = self._adaptive_kelly_fraction(confidence, win_rate)
        else:
            kelly_fraction = self.KELLY_FRACTIONS.get(self.mode, 0.25)
        
        # Cap at maximum Kelly fraction
        kelly_fraction = min(kelly_fraction, self.MAX_KELLY_FRACTION)
        constraints_applied.append(f'kelly_mode_{self.mode.value}')
        
        # Apply fractional Kelly
        kelly_bet = raw_kelly * kelly_fraction
        
        # Apply confidence dampening
        # Low confidence = lower position size
        confidence_adjustment = self._confidence_dampening(confidence)
        kelly_bet *= confidence_adjustment
        if confidence < 0.7:
            constraints_applied.append('confidence_dampening')
        
        # Apply volatility scaling
        if current_volatility is not None and current_volatility > 0:
            vol_scaling = self._volatility_scaling(current_volatility)
            kelly_bet *= vol_scaling
            constraints_applied.append('volatility_scaling')
        else:
            vol_scaling = 1.0
        
        # Get model-specific leverage limit
        model_max_leverage = self.model_leverage_limits.get(
            model_type, self.max_leverage
        )
        effective_max_leverage = min(model_max_leverage, self.max_leverage)
        
        # Calculate optimal position size
        optimal_size = kelly_bet * capital
        
        # Apply maximum position constraint
        max_position_value = capital * self.max_position_pct
        if optimal_size > max_position_value:
            optimal_size = max_position_value
            constraints_applied.append('max_position_pct')
        
        # Calculate effective leverage
        effective_leverage = optimal_size / capital
        
        # Apply leverage cap
        if effective_leverage > effective_max_leverage:
            optimal_size = capital * effective_max_leverage
            constraints_applied.append('leverage_cap')
            effective_leverage = effective_max_leverage
        
        # Apply minimum position size
        if optimal_size < self.MIN_POSITION_SIZE * capital:
            optimal_size = 0.0
            constraints_applied.append('below_minimum')
        
        return KellyResult(
            raw_kelly=raw_kelly,
            kelly_fraction=kelly_fraction,
            optimal_size=optimal_size / capital,  # As fraction
            final_size=optimal_size,
            win_rate=win_rate,
            payoff_ratio=payoff_ratio,
            confidence_adjustment=confidence_adjustment,
            volatility_scaling=vol_scaling,
            leverage_cap=effective_max_leverage,
            constraints_applied=constraints_applied
        )
    
    def _adaptive_kelly_fraction(
        self,
        confidence: float,
        win_rate: float
    ) -> float:
        """
        Calculate adaptive Kelly fraction based on conditions
        
        Higher fraction when:
        - High confidence
        - Win rate close to historical average
        - Recent performance is good
        
        Justification:
            Adaptive Kelly accounts for uncertainty in estimates.
            When confidence is low, we want to bet less even if
            the raw Kelly suggests otherwise.
        """
        # Base adaptive fraction (quarter Kelly)
        base_fraction = 0.25
        
        # Confidence adjustment: scale from 0.15 to 0.25
        confidence_bonus = (confidence - 0.5) * 0.2
        adaptive = base_fraction + max(confidence_bonus, 0)
        
        # Win rate reliability: reduce if extreme
        if win_rate < 0.4 or win_rate > 0.8:
            # Extreme win rates are less reliable
            adaptive *= 0.8
        
        return min(adaptive, self.MAX_KELLY_FRACTION)
    
    def _confidence_dampening(self, confidence: float) -> float:
        """
        Calculate confidence-based dampening factor
        
        Uses a non-linear dampening curve:
        - confidence 1.0 → factor 1.0
        - confidence 0.7 → factor 0.8
        - confidence 0.5 → factor 0.5
        - confidence 0.3 → factor 0.2
        
        Justification:
            Non-linear dampening aggressively reduces position size
            when model confidence is low, providing extra protection
            during uncertain conditions.
        """
        if confidence >= 0.9:
            return 1.0
        elif confidence >= 0.7:
            return 0.7 + (confidence - 0.7) * 1.5
        elif confidence >= 0.5:
            return 0.4 + (confidence - 0.5) * 1.5
        else:
            return confidence * 0.8
    
    def _volatility_scaling(self, current_volatility: float) -> float:
        """
        Calculate volatility-based scaling factor
        
        Formula: scaling = base_volatility / current_volatility
        Capped between 0.3 and 1.5
        
        Justification:
            Position size should be inversely proportional to
            volatility to maintain consistent risk per trade.
            High volatility = smaller position.
        """
        if current_volatility <= 0:
            return 1.0
        
        scaling = self.base_volatility / current_volatility
        
        # Cap scaling to reasonable range
        return np.clip(scaling, 0.3, 1.5)
    
    def calculate_from_trades(
        self,
        trade_results: list,
        confidence: float = 0.8,
        current_volatility: float = None,
        model_type: str = 'intraday',
        capital: float = 10000.0,
        min_trades: int = 30
    ) -> KellyResult:
        """
        Calculate Kelly from historical trade results
        
        Args:
            trade_results: List of trade PnL values
            confidence: Current model confidence
            current_volatility: Current market volatility
            model_type: Trading model type
            capital: Available capital
            min_trades: Minimum trades required
            
        Returns:
            KellyResult with position sizing
        """
        trades = np.array(trade_results)
        
        if len(trades) < min_trades:
            # Insufficient data - return conservative estimate
            return KellyResult(
                raw_kelly=0.0,
                kelly_fraction=0.0,
                optimal_size=0.0,
                final_size=0.0,
                win_rate=0.5,
                payoff_ratio=1.0,
                confidence_adjustment=confidence,
                volatility_scaling=1.0,
                leverage_cap=self.model_leverage_limits.get(model_type, self.max_leverage),
                constraints_applied=['insufficient_data']
            )
        
        # Calculate win rate
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        
        win_rate = len(wins) / len(trades)
        
        # Calculate payoff ratio
        avg_win = np.mean(wins) if len(wins) > 0 else 0.01
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.01
        payoff_ratio = avg_win / max(avg_loss, 0.01)
        
        return self.calculate_kelly(
            win_rate=win_rate,
            payoff_ratio=payoff_ratio,
            confidence=confidence,
            current_volatility=current_volatility,
            model_type=model_type,
            capital=capital
        )
    
    def update_history(
        self,
        model_type: str,
        win: bool,
        pnl: float,
        max_history: int = 500
    ):
        """
        Update historical statistics for adaptive Kelly
        
        Args:
            model_type: Model type
            win: Whether trade was a win
            pnl: Trade PnL
            max_history: Maximum history to maintain
        """
        if model_type not in self._win_rate_history:
            self._win_rate_history[model_type] = []
            self._payoff_history[model_type] = []
        
        self._win_rate_history[model_type].append(1 if win else 0)
        self._payoff_history[model_type].append(pnl)
        
        # Trim history
        if len(self._win_rate_history[model_type]) > max_history:
            self._win_rate_history[model_type] = self._win_rate_history[model_type][-max_history:]
            self._payoff_history[model_type] = self._payoff_history[model_type][-max_history:]
    
    def get_historical_stats(
        self,
        model_type: str
    ) -> Tuple[float, float, int]:
        """
        Get historical win rate and payoff ratio
        
        Returns:
            (win_rate, payoff_ratio, n_trades)
        """
        if model_type not in self._win_rate_history:
            return 0.5, 1.0, 0
        
        wins = self._win_rate_history[model_type]
        pnls = self._payoff_history[model_type]
        
        if len(wins) == 0:
            return 0.5, 1.0, 0
        
        win_rate = np.mean(wins)
        
        # Calculate payoff ratio
        pnl_array = np.array(pnls)
        avg_win = np.mean(pnl_array[pnl_array > 0]) if np.any(pnl_array > 0) else 0.01
        avg_loss = abs(np.mean(pnl_array[pnl_array < 0])) if np.any(pnl_array < 0) else 0.01
        payoff_ratio = avg_win / max(avg_loss, 0.01)
        
        return win_rate, payoff_ratio, len(wins)
    
    def to_features(
        self,
        kelly_result: KellyResult,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Convert Kelly metrics to features for ML model
        
        Args:
            kelly_result: Kelly calculation result
            prefix: Feature name prefix
            
        Returns:
            Dict of Kelly-derived features
        """
        prefix = f"{prefix}_" if prefix else ""
        
        return {
            f'{prefix}kelly_raw': kelly_result.raw_kelly,
            f'{prefix}kelly_fraction': kelly_result.kelly_fraction,
            f'{prefix}kelly_optimal_size': kelly_result.optimal_size,
            f'{prefix}kelly_win_rate': kelly_result.win_rate,
            f'{prefix}kelly_payoff_ratio': kelly_result.payoff_ratio,
            f'{prefix}kelly_confidence_adj': kelly_result.confidence_adjustment,
            f'{prefix}kelly_vol_scaling': kelly_result.volatility_scaling
        }
