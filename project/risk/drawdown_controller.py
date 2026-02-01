"""
Drawdown-Aware Dynamic Position Sizing Controller

Implements non-linear drawdown response mechanisms:
- Dynamic risk multipliers
- Soft stop-loss regimes
- Model disable logic
- Automatic deleveraging

The controller continuously tracks drawdown and reacts
in a graduated, non-linear manner to protect capital.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging


class RiskState(Enum):
    """Global risk state controlled by drawdown"""
    FULL_RISK = "full_risk"          # Normal operations
    REDUCED_RISK = "reduced_risk"    # DD > threshold 1
    MINIMAL_RISK = "minimal_risk"    # DD > threshold 2
    RISK_OFF = "risk_off"            # DD > threshold 3, risk-off only
    EMERGENCY = "emergency"          # DD > emergency threshold


class ModelState(Enum):
    """Individual model state"""
    ENABLED = "enabled"
    REDUCED = "reduced"
    DISABLED = "disabled"


@dataclass
class DrawdownMetrics:
    """Container for drawdown metrics"""
    current_drawdown: float          # Current DD from peak
    max_drawdown: float              # Maximum historical DD
    drawdown_duration: int           # Periods in current DD
    recovery_ratio: float            # % recovered from max DD
    drawdown_velocity: float         # Speed of DD increase
    risk_multiplier: float           # Current risk multiplier
    global_risk_state: RiskState     # Current global state
    model_states: Dict[str, ModelState]  # Per-model states
    
    def to_dict(self) -> Dict:
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'drawdown_duration': self.drawdown_duration,
            'recovery_ratio': self.recovery_ratio,
            'drawdown_velocity': self.drawdown_velocity,
            'risk_multiplier': self.risk_multiplier,
            'global_risk_state': self.global_risk_state.value,
            'model_states': {k: v.value for k, v in self.model_states.items()}
        }


@dataclass
class DrawdownThresholds:
    """Configuration for drawdown thresholds"""
    # Position sizing reduction thresholds
    level_1: float = 0.05   # 5% DD → reduce to 75% size
    level_2: float = 0.10   # 10% DD → reduce to 50% size
    level_3: float = 0.15   # 15% DD → reduce to 25% size
    level_4: float = 0.20   # 20% DD → risk-off only
    emergency: float = 0.25  # 25% DD → emergency stop
    
    # Model-specific thresholds for disabling
    scalp_disable: float = 0.08    # Disable scalp at 8% DD
    intraday_disable: float = 0.12  # Disable intraday at 12% DD
    swing_disable: float = 0.18     # Disable swing at 18% DD
    
    # Recovery thresholds for re-enabling
    recovery_to_reduced: float = 0.5   # 50% recovery → reduced risk
    recovery_to_full: float = 0.8      # 80% recovery → full risk


@dataclass
class DrawdownEvent:
    """Record of a drawdown event"""
    timestamp: datetime
    drawdown_value: float
    risk_state: RiskState
    action_taken: str
    equity_value: float


class DrawdownController:
    """
    Drawdown-Aware Dynamic Position Sizing Controller
    
    This controller implements a graduated, non-linear response
    to drawdowns that:
    1. Reduces position sizes as drawdown increases
    2. Disables specific models at thresholds
    3. Switches to risk-off mode at extreme levels
    4. Triggers emergency stop at catastrophic levels
    
    Justification:
    - Non-linear response prevents small DDs from over-reducing size
    - Model-specific thresholds account for strategy characteristics
    - Graduated approach maintains some exposure during recoverable DDs
    - Emergency stop prevents complete account wipeout
    
    The controller maintains state and provides:
    - Real-time risk multipliers
    - Model enable/disable decisions
    - Trade approval/rejection
    - Automatic recovery monitoring
    """
    
    def __init__(
        self,
        initial_equity: float = 10000.0,
        thresholds: Optional[DrawdownThresholds] = None,
        recovery_periods: int = 20,
        enable_emergency_stop: bool = True,
        log_all_decisions: bool = True
    ):
        """
        Initialize Drawdown Controller
        
        Args:
            initial_equity: Starting equity value
            thresholds: Drawdown threshold configuration
            recovery_periods: Periods to track for recovery
            enable_emergency_stop: Whether to enable emergency stop
            log_all_decisions: Log all decisions to console
        """
        self.initial_equity = initial_equity
        self.thresholds = thresholds or DrawdownThresholds()
        self.recovery_periods = recovery_periods
        self.enable_emergency_stop = enable_emergency_stop
        self.log_all_decisions = log_all_decisions
        
        # State tracking
        self._equity_history: List[float] = [initial_equity]
        self._peak_equity: float = initial_equity
        self._current_drawdown: float = 0.0
        self._max_drawdown: float = 0.0
        self._drawdown_start_idx: Optional[int] = None
        
        # Risk state
        self._global_risk_state: RiskState = RiskState.FULL_RISK
        self._model_states: Dict[str, ModelState] = {
            'scalp': ModelState.ENABLED,
            'intraday': ModelState.ENABLED,
            'swing': ModelState.ENABLED
        }
        
        # Risk multiplier (1.0 = full size)
        self._risk_multiplier: float = 1.0
        
        # Event log
        self._events: List[DrawdownEvent] = []
        
        # Emergency flag
        self._emergency_triggered: bool = False
        
        # Logger
        self.logger = logging.getLogger('DrawdownController')
        if log_all_decisions:
            self.logger.setLevel(logging.INFO)
        
    def update(self, current_equity: float) -> DrawdownMetrics:
        """
        Update controller with new equity value
        
        This is the main update method that should be called
        after each trade or at regular intervals.
        
        Args:
            current_equity: Current portfolio equity
            
        Returns:
            DrawdownMetrics with current state
        """
        self._equity_history.append(current_equity)
        
        # Update peak equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            self._drawdown_start_idx = None
        
        # Calculate current drawdown
        if self._peak_equity > 0:
            self._current_drawdown = (self._peak_equity - current_equity) / self._peak_equity
        else:
            self._current_drawdown = 0.0
        
        # Update max drawdown
        if self._current_drawdown > self._max_drawdown:
            self._max_drawdown = self._current_drawdown
            if self._drawdown_start_idx is None:
                self._drawdown_start_idx = len(self._equity_history) - 1
        
        # Calculate drawdown duration
        if self._drawdown_start_idx is not None:
            drawdown_duration = len(self._equity_history) - self._drawdown_start_idx
        else:
            drawdown_duration = 0
        
        # Calculate recovery ratio
        if self._max_drawdown > 0:
            recovery_ratio = 1 - (self._current_drawdown / self._max_drawdown)
        else:
            recovery_ratio = 1.0
        
        # Calculate drawdown velocity (change in DD over recent periods)
        drawdown_velocity = self._calculate_drawdown_velocity()
        
        # Update risk state based on drawdown
        self._update_risk_state()
        
        # Update model states
        self._update_model_states()
        
        # Calculate risk multiplier
        self._risk_multiplier = self._calculate_risk_multiplier()
        
        # Check for emergency
        self._check_emergency()
        
        metrics = DrawdownMetrics(
            current_drawdown=self._current_drawdown,
            max_drawdown=self._max_drawdown,
            drawdown_duration=drawdown_duration,
            recovery_ratio=recovery_ratio,
            drawdown_velocity=drawdown_velocity,
            risk_multiplier=self._risk_multiplier,
            global_risk_state=self._global_risk_state,
            model_states=self._model_states.copy()
        )
        
        return metrics
    
    def _calculate_drawdown_velocity(self) -> float:
        """
        Calculate speed of drawdown increase
        
        Returns:
            Drawdown velocity (positive = increasing DD)
        """
        if len(self._equity_history) < 5:
            return 0.0
        
        recent_equity = np.array(self._equity_history[-5:])
        recent_dd = (self._peak_equity - recent_equity) / self._peak_equity
        
        # Velocity is the slope of recent drawdowns
        if len(recent_dd) > 1:
            return float(np.mean(np.diff(recent_dd)))
        return 0.0
    
    def _update_risk_state(self):
        """
        Update global risk state based on drawdown level
        
        Non-linear response with graduated thresholds.
        """
        prev_state = self._global_risk_state
        dd = self._current_drawdown
        
        if dd >= self.thresholds.emergency:
            self._global_risk_state = RiskState.EMERGENCY
        elif dd >= self.thresholds.level_4:
            self._global_risk_state = RiskState.RISK_OFF
        elif dd >= self.thresholds.level_3:
            self._global_risk_state = RiskState.MINIMAL_RISK
        elif dd >= self.thresholds.level_1:
            self._global_risk_state = RiskState.REDUCED_RISK
        else:
            self._global_risk_state = RiskState.FULL_RISK
        
        # Log state change
        if prev_state != self._global_risk_state:
            self._log_event(
                f"Risk state changed: {prev_state.value} → {self._global_risk_state.value}",
                dd
            )
    
    def _update_model_states(self):
        """
        Update individual model states based on drawdown
        
        Scalp is disabled first (most aggressive), then intraday,
        finally swing (most conservative).
        """
        dd = self._current_drawdown
        
        # Scalp model
        if dd >= self.thresholds.scalp_disable:
            if self._model_states['scalp'] != ModelState.DISABLED:
                self._model_states['scalp'] = ModelState.DISABLED
                self._log_event("SCALP model DISABLED", dd)
        elif dd >= self.thresholds.level_1:
            if self._model_states['scalp'] == ModelState.ENABLED:
                self._model_states['scalp'] = ModelState.REDUCED
                self._log_event("SCALP model REDUCED", dd)
        elif self._model_states['scalp'] != ModelState.ENABLED:
            # Recovery logic
            if self._current_drawdown < self.thresholds.level_1 * 0.5:
                self._model_states['scalp'] = ModelState.ENABLED
                self._log_event("SCALP model ENABLED (recovery)", dd)
        
        # Intraday model
        if dd >= self.thresholds.intraday_disable:
            if self._model_states['intraday'] != ModelState.DISABLED:
                self._model_states['intraday'] = ModelState.DISABLED
                self._log_event("INTRADAY model DISABLED", dd)
        elif dd >= self.thresholds.level_2:
            if self._model_states['intraday'] == ModelState.ENABLED:
                self._model_states['intraday'] = ModelState.REDUCED
                self._log_event("INTRADAY model REDUCED", dd)
        elif self._model_states['intraday'] != ModelState.ENABLED:
            if self._current_drawdown < self.thresholds.level_1 * 0.5:
                self._model_states['intraday'] = ModelState.ENABLED
                self._log_event("INTRADAY model ENABLED (recovery)", dd)
        
        # Swing model
        if dd >= self.thresholds.swing_disable:
            if self._model_states['swing'] != ModelState.DISABLED:
                self._model_states['swing'] = ModelState.DISABLED
                self._log_event("SWING model DISABLED", dd)
        elif dd >= self.thresholds.level_3:
            if self._model_states['swing'] == ModelState.ENABLED:
                self._model_states['swing'] = ModelState.REDUCED
                self._log_event("SWING model REDUCED", dd)
        elif self._model_states['swing'] != ModelState.ENABLED:
            if self._current_drawdown < self.thresholds.level_1 * 0.5:
                self._model_states['swing'] = ModelState.ENABLED
                self._log_event("SWING model ENABLED (recovery)", dd)
    
    def _calculate_risk_multiplier(self) -> float:
        """
        Calculate dynamic risk multiplier based on drawdown
        
        Non-linear response:
        - DD 0-5%: multiplier = 1.0
        - DD 5-10%: multiplier = 0.75
        - DD 10-15%: multiplier = 0.50
        - DD 15-20%: multiplier = 0.25
        - DD 20%+: multiplier = 0.0
        
        Justification:
            Non-linear reduces overreaction to small DDs while
            aggressively protecting against large DDs.
        """
        dd = self._current_drawdown
        
        if dd < self.thresholds.level_1:
            return 1.0
        elif dd < self.thresholds.level_2:
            # Linear interpolation from 1.0 to 0.75
            progress = (dd - self.thresholds.level_1) / (self.thresholds.level_2 - self.thresholds.level_1)
            return 1.0 - 0.25 * progress
        elif dd < self.thresholds.level_3:
            # Linear interpolation from 0.75 to 0.50
            progress = (dd - self.thresholds.level_2) / (self.thresholds.level_3 - self.thresholds.level_2)
            return 0.75 - 0.25 * progress
        elif dd < self.thresholds.level_4:
            # Linear interpolation from 0.50 to 0.25
            progress = (dd - self.thresholds.level_3) / (self.thresholds.level_4 - self.thresholds.level_3)
            return 0.50 - 0.25 * progress
        else:
            return 0.0  # Risk-off
    
    def _check_emergency(self):
        """Check and handle emergency conditions"""
        if self._current_drawdown >= self.thresholds.emergency and self.enable_emergency_stop:
            if not self._emergency_triggered:
                self._emergency_triggered = True
                self._log_event(
                    f"🚨 EMERGENCY STOP TRIGGERED! DD={self._current_drawdown:.1%}",
                    self._current_drawdown
                )
    
    def _log_event(self, action: str, drawdown: float):
        """Log a drawdown event"""
        event = DrawdownEvent(
            timestamp=datetime.now(),
            drawdown_value=drawdown,
            risk_state=self._global_risk_state,
            action_taken=action,
            equity_value=self._equity_history[-1] if self._equity_history else 0
        )
        self._events.append(event)
        
        if self.log_all_decisions:
            self.logger.info(f"[DD={drawdown:.1%}] {action}")
    
    def get_position_size_multiplier(
        self,
        model_type: str,
        base_size: float
    ) -> float:
        """
        Get adjusted position size based on drawdown
        
        Args:
            model_type: Type of trading model
            base_size: Base position size from Kelly
            
        Returns:
            Adjusted position size
        """
        # Check if model is enabled
        model_state = self._model_states.get(model_type, ModelState.DISABLED)
        
        if model_state == ModelState.DISABLED:
            return 0.0
        
        # Apply model state multiplier
        state_multiplier = {
            ModelState.ENABLED: 1.0,
            ModelState.REDUCED: 0.5,
            ModelState.DISABLED: 0.0
        }.get(model_state, 0.0)
        
        # Apply global risk multiplier
        return base_size * self._risk_multiplier * state_multiplier
    
    def is_trade_allowed(
        self,
        model_type: str,
        is_risk_off_trade: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if a trade is allowed given current drawdown state
        
        Args:
            model_type: Type of trading model
            is_risk_off_trade: Whether this is a risk-off trade
            
        Returns:
            (is_allowed, reason)
        """
        # Emergency stop
        if self._emergency_triggered:
            return False, "Emergency stop active - all trading halted"
        
        # Risk-off state
        if self._global_risk_state == RiskState.RISK_OFF:
            if is_risk_off_trade:
                return True, "Risk-off trade allowed"
            return False, f"Risk-off state - only risk-off trades allowed (DD={self._current_drawdown:.1%})"
        
        # Check model state
        model_state = self._model_states.get(model_type, ModelState.DISABLED)
        
        if model_state == ModelState.DISABLED:
            return False, f"{model_type} model disabled (DD={self._current_drawdown:.1%})"
        
        # Reduced state warning
        if model_state == ModelState.REDUCED:
            return True, f"{model_type} model in reduced state - size limited"
        
        return True, "Trade allowed"
    
    def get_metrics(self) -> DrawdownMetrics:
        """Get current drawdown metrics"""
        return DrawdownMetrics(
            current_drawdown=self._current_drawdown,
            max_drawdown=self._max_drawdown,
            drawdown_duration=len(self._equity_history) - (self._drawdown_start_idx or len(self._equity_history)),
            recovery_ratio=1 - (self._current_drawdown / max(self._max_drawdown, 0.001)),
            drawdown_velocity=self._calculate_drawdown_velocity(),
            risk_multiplier=self._risk_multiplier,
            global_risk_state=self._global_risk_state,
            model_states=self._model_states.copy()
        )
    
    def reset_emergency(self, new_equity: Optional[float] = None):
        """
        Reset emergency state (manual override)
        
        Should only be called after manual review.
        
        Args:
            new_equity: New starting equity (optional)
        """
        self._emergency_triggered = False
        self._global_risk_state = RiskState.REDUCED_RISK  # Start in reduced
        
        if new_equity is not None:
            self._peak_equity = new_equity
            self._equity_history = [new_equity]
            self._current_drawdown = 0.0
        
        self._log_event("Emergency state reset (manual)", self._current_drawdown)
    
    def get_events(self, last_n: int = 100) -> List[DrawdownEvent]:
        """Get recent drawdown events"""
        return self._events[-last_n:]
    
    def to_features(self, prefix: str = "") -> Dict[str, float]:
        """
        Convert drawdown metrics to features for ML model
        
        Args:
            prefix: Feature name prefix
            
        Returns:
            Dict of drawdown-derived features
        """
        prefix = f"{prefix}_" if prefix else ""
        
        return {
            f'{prefix}dd_current': self._current_drawdown,
            f'{prefix}dd_max': self._max_drawdown,
            f'{prefix}dd_velocity': self._calculate_drawdown_velocity(),
            f'{prefix}dd_risk_multiplier': self._risk_multiplier,
            f'{prefix}dd_risk_state': list(RiskState).index(self._global_risk_state),
            f'{prefix}dd_models_enabled': sum(1 for s in self._model_states.values() if s == ModelState.ENABLED),
            f'{prefix}dd_models_disabled': sum(1 for s in self._model_states.values() if s == ModelState.DISABLED)
        }
    
    @property
    def is_emergency(self) -> bool:
        """Check if emergency is triggered"""
        return self._emergency_triggered
    
    @property
    def risk_state(self) -> RiskState:
        """Get current global risk state"""
        return self._global_risk_state
    
    @property
    def risk_multiplier(self) -> float:
        """Get current risk multiplier"""
        return self._risk_multiplier
