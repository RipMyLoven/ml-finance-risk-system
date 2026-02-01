"""
Advanced Risk Model - Core Controller

This is the CENTRAL CONTROL UNIT for the multi-model trading system.
It has FINAL VETO POWER over all trades and authority to:
- Adjust position sizing
- Block trades
- Reduce exposure
- Disable individual trading models

The Risk Model integrates:
- CVaR (Conditional Value at Risk)
- Kelly-based position sizing
- Drawdown-aware dynamic sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
import json

from .cvar_engine import CVaREngine, CVaRResult, RiskLevel
from .kelly_sizing import KellySizer, KellyResult, KellyMode
from .drawdown_controller import DrawdownController, DrawdownMetrics, RiskState, ModelState


class TradeDecision(Enum):
    """Trade decision types"""
    APPROVED = "approved"
    REJECTED = "rejected"
    SIZE_REDUCED = "size_reduced"


@dataclass
class TradeProposal:
    """Input from trading models"""
    model_type: str             # scalp, intraday, swing
    symbol: str                 # Trading symbol
    direction: str              # long, short
    probability: float          # Trade probability (0-1)
    confidence: float           # Model confidence (0-1)
    expected_return: float      # Expected return
    volatility_regime: str      # low, normal, high
    proposed_size: float        # Proposed position size
    stop_loss: float            # Stop loss distance
    take_profit: float          # Take profit distance
    
    def to_dict(self) -> Dict:
        return {
            'model_type': self.model_type,
            'symbol': self.symbol,
            'direction': self.direction,
            'probability': self.probability,
            'confidence': self.confidence,
            'expected_return': self.expected_return,
            'volatility_regime': self.volatility_regime,
            'proposed_size': self.proposed_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }


@dataclass
class RiskDecision:
    """Output from Risk Model"""
    decision: TradeDecision     # Approve/Reject/Reduce
    approved_size: float        # Final approved position size
    adjusted_leverage: float    # Adjusted leverage
    reason: str                 # Decision explanation
    risk_state: RiskState       # Current global risk state
    cvar_result: Optional[CVaRResult]  # CVaR assessment
    kelly_result: Optional[KellyResult]  # Kelly calculation
    drawdown_metrics: Optional[DrawdownMetrics]  # DD metrics
    recommendations: List[str]  # Additional recommendations
    timestamp: datetime         # Decision timestamp
    
    def to_dict(self) -> Dict:
        return {
            'decision': self.decision.value,
            'approved_size': self.approved_size,
            'adjusted_leverage': self.adjusted_leverage,
            'reason': self.reason,
            'risk_state': self.risk_state.value,
            'cvar': self.cvar_result.to_dict() if self.cvar_result else None,
            'kelly': self.kelly_result.to_dict() if self.kelly_result else None,
            'drawdown': self.drawdown_metrics.to_dict() if self.drawdown_metrics else None,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }


class AdvancedRiskModel:
    """
    Advanced Risk Model - Core Controller
    
    This is the central risk management system with FINAL VETO POWER
    over all trading decisions. It integrates:
    
    1. CVaR Engine: Tail risk estimation and trade admission
    2. Kelly Sizer: Optimal position sizing with constraints
    3. Drawdown Controller: Dynamic sizing and model control
    
    The Risk Model receives proposals from trading models and returns
    final decisions on:
    - Trade approval/rejection
    - Position size
    - Leverage
    
    All decisions are logged and justified.
    """
    
    def __init__(
        self,
        initial_equity: float = 10000.0,
        cvar_confidence: float = 0.95,
        kelly_mode: KellyMode = KellyMode.QUARTER,
        max_leverage: float = 5.0,
        max_position_pct: float = 0.10,
        enable_kill_switch: bool = True,
        log_all_decisions: bool = True,
        cvar_block_threshold: float = 0.08,
        model_cvar_limits: Optional[Dict[str, float]] = None,
        drawdown_thresholds: Optional[Dict] = None
    ):
        """
        Initialize Advanced Risk Model
        
        Args:
            initial_equity: Starting equity
            cvar_confidence: CVaR confidence level
            kelly_mode: Kelly fraction mode
            max_leverage: Maximum allowed leverage
            max_position_pct: Maximum position as % of capital
            enable_kill_switch: Enable emergency kill switch
            log_all_decisions: Log all decisions to console
            cvar_block_threshold: CVaR threshold for blocking trades
            model_cvar_limits: Per-model CVaR limits
            drawdown_thresholds: Drawdown threshold configuration
        """
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.enable_kill_switch = enable_kill_switch
        self.log_all_decisions = log_all_decisions
        
        # Initialize CVaR Engine
        self.cvar_engine = CVaREngine(
            confidence_level=cvar_confidence,
            block_threshold=cvar_block_threshold,
            model_cvar_limits=model_cvar_limits
        )
        
        # Initialize Kelly Sizer
        self.kelly_sizer = KellySizer(
            mode=kelly_mode,
            max_leverage=max_leverage,
            max_position_pct=max_position_pct
        )
        
        # Initialize Drawdown Controller
        from .drawdown_controller import DrawdownThresholds
        dd_config = DrawdownThresholds(**drawdown_thresholds) if drawdown_thresholds else DrawdownThresholds()
        self.drawdown_controller = DrawdownController(
            initial_equity=initial_equity,
            thresholds=dd_config,
            enable_emergency_stop=enable_kill_switch,
            log_all_decisions=log_all_decisions
        )
        
        # PnL history for CVaR calculation
        self._pnl_history: Dict[str, List[float]] = {
            'scalp': [],
            'intraday': [],
            'swing': [],
            'portfolio': []
        }
        
        # Decision log
        self._decision_log: List[RiskDecision] = []
        
        # Kill switch state
        self._kill_switch_active: bool = False
        
        # Logger
        self.logger = logging.getLogger('AdvancedRiskModel')
        if log_all_decisions:
            logging.basicConfig(level=logging.INFO)
        
    def evaluate_trade(
        self,
        proposal: TradeProposal,
        recent_returns: Optional[np.ndarray] = None,
        current_volatility: Optional[float] = None
    ) -> RiskDecision:
        """
        Evaluate a trade proposal and return decision
        
        This is the main entry point for trade evaluation.
        The Risk Model has FINAL VETO POWER.
        
        Args:
            proposal: Trade proposal from trading model
            recent_returns: Recent PnL history for CVaR
            current_volatility: Current market volatility
            
        Returns:
            RiskDecision with approval/rejection and sizing
        """
        timestamp = datetime.now()
        recommendations = []
        
        # Log incoming proposal
        if self.log_all_decisions:
            self.logger.info(f"[PROPOSAL] {proposal.model_type} {proposal.direction} {proposal.symbol}")
        
        # =====================================================================
        # STEP 1: Check Kill Switch
        # =====================================================================
        if self._kill_switch_active:
            return self._create_rejection(
                proposal, "Kill switch active - all trading halted", timestamp
            )
        
        # =====================================================================
        # STEP 2: Check Drawdown State
        # =====================================================================
        dd_metrics = self.drawdown_controller.get_metrics()
        
        # Check if model is allowed to trade
        is_risk_off_trade = proposal.direction == 'close' or proposal.expected_return < 0
        trade_allowed, dd_reason = self.drawdown_controller.is_trade_allowed(
            proposal.model_type, is_risk_off_trade
        )
        
        if not trade_allowed:
            return self._create_rejection(proposal, dd_reason, timestamp, dd_metrics=dd_metrics)
        
        # =====================================================================
        # STEP 3: CVaR Check
        # =====================================================================
        if recent_returns is None:
            recent_returns = np.array(self._pnl_history.get(proposal.model_type, []))
        
        if len(recent_returns) >= 30:
            cvar_result = self.cvar_engine.calculate_cvar(recent_returns)
            
            # Check trade admission
            admitted, cvar_reason = self.cvar_engine.check_trade_admission(
                proposal.model_type,
                recent_returns,
                proposal.expected_return,
                proposal.proposed_size
            )
            
            if not admitted:
                return self._create_rejection(
                    proposal, cvar_reason, timestamp,
                    cvar_result=cvar_result, dd_metrics=dd_metrics
                )
            
            # Add CVaR recommendation
            if cvar_result.risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME]:
                recommendations.append(f"High CVaR ({cvar_result.cvar:.2%}) - consider reduced size")
        else:
            cvar_result = None
            recommendations.append("Insufficient data for CVaR - using conservative sizing")
        
        # =====================================================================
        # STEP 4: Calculate Kelly-Based Size
        # =====================================================================
        # Estimate win rate from proposal probability
        win_rate = proposal.probability
        
        # Estimate payoff ratio from SL/TP
        if proposal.stop_loss > 0:
            payoff_ratio = proposal.take_profit / proposal.stop_loss
        else:
            payoff_ratio = 1.5  # Default assumption
        
        kelly_result = self.kelly_sizer.calculate_kelly(
            win_rate=win_rate,
            payoff_ratio=payoff_ratio,
            confidence=proposal.confidence,
            current_volatility=current_volatility,
            model_type=proposal.model_type,
            capital=self.current_equity
        )
        
        if kelly_result.final_size == 0:
            return self._create_rejection(
                proposal, f"Kelly calculation suggests no position: {kelly_result.constraints_applied}",
                timestamp, cvar_result=cvar_result, kelly_result=kelly_result, dd_metrics=dd_metrics
            )
        
        # =====================================================================
        # STEP 5: Apply Drawdown Adjustment
        # =====================================================================
        dd_adjusted_size = self.drawdown_controller.get_position_size_multiplier(
            proposal.model_type, kelly_result.final_size
        )
        
        if dd_adjusted_size == 0:
            return self._create_rejection(
                proposal, f"Drawdown-adjusted size is zero (DD={dd_metrics.current_drawdown:.1%})",
                timestamp, cvar_result=cvar_result, kelly_result=kelly_result, dd_metrics=dd_metrics
            )
        
        # =====================================================================
        # STEP 6: Apply Volatility Regime Adjustment
        # =====================================================================
        vol_multiplier = self._get_volatility_multiplier(proposal.volatility_regime)
        final_size = dd_adjusted_size * vol_multiplier
        
        if proposal.volatility_regime == 'high':
            recommendations.append("High volatility - size reduced")
        
        # =====================================================================
        # STEP 7: Calculate Final Leverage
        # =====================================================================
        final_leverage = final_size / self.current_equity
        max_model_leverage = self.kelly_sizer.model_leverage_limits.get(
            proposal.model_type, self.kelly_sizer.max_leverage
        )
        
        if final_leverage > max_model_leverage:
            final_leverage = max_model_leverage
            final_size = self.current_equity * final_leverage
            recommendations.append(f"Leverage capped at {max_model_leverage}x")
        
        # =====================================================================
        # STEP 8: Determine Decision Type
        # =====================================================================
        if final_size >= proposal.proposed_size * 0.9:
            decision = TradeDecision.APPROVED
            reason = "Trade approved at requested size"
        else:
            decision = TradeDecision.SIZE_REDUCED
            reduction_pct = 1 - (final_size / proposal.proposed_size)
            reason = f"Trade approved with {reduction_pct:.0%} size reduction"
        
        # =====================================================================
        # STEP 9: Create and Log Decision
        # =====================================================================
        risk_decision = RiskDecision(
            decision=decision,
            approved_size=final_size,
            adjusted_leverage=final_leverage,
            reason=reason,
            risk_state=dd_metrics.global_risk_state,
            cvar_result=cvar_result,
            kelly_result=kelly_result,
            drawdown_metrics=dd_metrics,
            recommendations=recommendations,
            timestamp=timestamp
        )
        
        self._log_decision(proposal, risk_decision)
        
        return risk_decision
    
    def _create_rejection(
        self,
        proposal: TradeProposal,
        reason: str,
        timestamp: datetime,
        cvar_result: Optional[CVaRResult] = None,
        kelly_result: Optional[KellyResult] = None,
        dd_metrics: Optional[DrawdownMetrics] = None
    ) -> RiskDecision:
        """Create a rejection decision"""
        decision = RiskDecision(
            decision=TradeDecision.REJECTED,
            approved_size=0.0,
            adjusted_leverage=0.0,
            reason=reason,
            risk_state=dd_metrics.global_risk_state if dd_metrics else RiskState.RISK_OFF,
            cvar_result=cvar_result,
            kelly_result=kelly_result,
            drawdown_metrics=dd_metrics,
            recommendations=["Trade rejected - see reason"],
            timestamp=timestamp
        )
        
        self._log_decision(proposal, decision)
        return decision
    
    def _get_volatility_multiplier(self, volatility_regime: str) -> float:
        """
        Get position size multiplier based on volatility regime
        
        Justification:
            Higher volatility = higher uncertainty = smaller position
        """
        return {
            'low': 1.2,      # Can be slightly larger in low vol
            'normal': 1.0,   # Standard size
            'high': 0.6,     # Reduce significantly in high vol
            'extreme': 0.3   # Minimal size in extreme vol
        }.get(volatility_regime, 1.0)
    
    def _log_decision(self, proposal: TradeProposal, decision: RiskDecision):
        """Log decision to console and history"""
        self._decision_log.append(decision)
        
        if self.log_all_decisions:
            icon = "✅" if decision.decision == TradeDecision.APPROVED else \
                   "⚠️" if decision.decision == TradeDecision.SIZE_REDUCED else "❌"
            
            self.logger.info(
                f"[{icon} {decision.decision.value.upper()}] "
                f"{proposal.model_type} {proposal.symbol}: {decision.reason}"
            )
    
    def update_equity(self, new_equity: float):
        """
        Update current equity and trigger drawdown calculations
        
        Should be called after each trade or at regular intervals.
        """
        self.current_equity = new_equity
        dd_metrics = self.drawdown_controller.update(new_equity)
        
        # Check for emergency
        if self.enable_kill_switch and self.drawdown_controller.is_emergency:
            self._kill_switch_active = True
            self.logger.critical("🚨 KILL SWITCH ACTIVATED - ALL TRADING HALTED")
    
    def update_pnl(self, model_type: str, pnl: float):
        """
        Update PnL history for a model
        
        Args:
            model_type: Model type (scalp/intraday/swing)
            pnl: Trade PnL
        """
        if model_type not in self._pnl_history:
            self._pnl_history[model_type] = []
        
        self._pnl_history[model_type].append(pnl)
        self._pnl_history['portfolio'].append(pnl)
        
        # Update CVaR engine
        self.cvar_engine.update_pnl_history(model_type, pnl)
        
        # Update Kelly sizer
        self.kelly_sizer.update_history(model_type, pnl > 0, pnl)
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """
        Activate emergency kill switch
        
        All trading will be halted until manual reset.
        """
        self._kill_switch_active = True
        self.logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
    
    def reset_kill_switch(self, new_equity: Optional[float] = None):
        """
        Reset kill switch (requires manual intervention)
        
        Args:
            new_equity: Optional new starting equity
        """
        self._kill_switch_active = False
        self.drawdown_controller.reset_emergency(new_equity)
        
        if new_equity:
            self.current_equity = new_equity
        
        self.logger.info("✅ Kill switch reset - trading can resume in REDUCED state")
    
    def get_global_state(self) -> Dict:
        """Get current global risk state"""
        dd_metrics = self.drawdown_controller.get_metrics()
        
        return {
            'risk_state': dd_metrics.global_risk_state.value,
            'kill_switch_active': self._kill_switch_active,
            'current_equity': self.current_equity,
            'current_drawdown': dd_metrics.current_drawdown,
            'max_drawdown': dd_metrics.max_drawdown,
            'risk_multiplier': dd_metrics.risk_multiplier,
            'model_states': {k: v.value for k, v in dd_metrics.model_states.items()},
            'models_enabled': sum(1 for s in dd_metrics.model_states.values() if s == ModelState.ENABLED)
        }
    
    def get_model_status(self, model_type: str) -> Dict:
        """Get status for a specific model"""
        dd_metrics = self.drawdown_controller.get_metrics()
        model_state = dd_metrics.model_states.get(model_type, ModelState.DISABLED)
        
        # Get model-specific CVaR
        cvar_result = self.cvar_engine.get_model_cvar(model_type)
        
        # Get Kelly stats
        win_rate, payoff_ratio, n_trades = self.kelly_sizer.get_historical_stats(model_type)
        
        return {
            'model_type': model_type,
            'state': model_state.value,
            'is_enabled': model_state == ModelState.ENABLED,
            'is_reduced': model_state == ModelState.REDUCED,
            'cvar': cvar_result.cvar,
            'cvar_risk_level': cvar_result.risk_level.value,
            'win_rate': win_rate,
            'payoff_ratio': payoff_ratio,
            'n_trades': n_trades
        }
    
    def get_decision_log(self, last_n: int = 100) -> List[Dict]:
        """Get recent decision log"""
        return [d.to_dict() for d in self._decision_log[-last_n:]]
    
    def get_risk_features(self) -> Dict[str, float]:
        """
        Get current risk state as features for ML model
        
        These features can be used as inputs to other models.
        """
        dd_metrics = self.drawdown_controller.get_metrics()
        
        features = {}
        
        # Drawdown features
        features.update(self.drawdown_controller.to_features('dd'))
        
        # CVaR features for each model
        for model_type in ['scalp', 'intraday', 'swing']:
            if len(self._pnl_history.get(model_type, [])) >= 30:
                cvar_features = self.cvar_engine.to_features(
                    np.array(self._pnl_history[model_type]),
                    prefix=f'{model_type}_cvar'
                )
                features.update(cvar_features)
        
        # Global state features
        features['risk_state_encoded'] = list(RiskState).index(dd_metrics.global_risk_state)
        features['kill_switch'] = float(self._kill_switch_active)
        
        return features
    
    def to_onnx_input(self, proposal: TradeProposal) -> np.ndarray:
        """
        Convert proposal + risk state to ONNX input format
        
        This prepares the input for ONNX risk model inference.
        """
        # Get risk features
        risk_features = self.get_risk_features()
        
        # Proposal features
        proposal_features = {
            'probability': proposal.probability,
            'confidence': proposal.confidence,
            'expected_return': proposal.expected_return,
            'proposed_size': proposal.proposed_size,
            'stop_loss': proposal.stop_loss,
            'take_profit': proposal.take_profit,
            'is_scalp': float(proposal.model_type == 'scalp'),
            'is_intraday': float(proposal.model_type == 'intraday'),
            'is_swing': float(proposal.model_type == 'swing'),
            'is_long': float(proposal.direction == 'long'),
            'vol_low': float(proposal.volatility_regime == 'low'),
            'vol_normal': float(proposal.volatility_regime == 'normal'),
            'vol_high': float(proposal.volatility_regime == 'high')
        }
        
        # Combine features
        all_features = {**risk_features, **proposal_features}
        
        return np.array(list(all_features.values()), dtype=np.float32)


# =============================================================================
# OPTIMIZATION METRICS FOR RISK MODEL
# =============================================================================

def calculate_risk_model_metrics(
    decisions: List[RiskDecision],
    actual_returns: List[float]
) -> Dict[str, float]:
    """
    Calculate optimization metrics for Risk Model
    
    Multi-objective optimization targets:
    - AUC ↑
    - Maximum Drawdown ↓
    - CVaR ↓
    - Tail-loss frequency ↓
    - Equity curve stability ↑
    
    Args:
        decisions: List of risk decisions made
        actual_returns: Actual returns after each decision
        
    Returns:
        Dict of metrics
    """
    if len(decisions) == 0 or len(actual_returns) == 0:
        return {
            'approval_rate': 0.0,
            'rejection_rate': 0.0,
            'size_reduction_rate': 0.0,
            'mean_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'cvar_95': 0.0,
            'tail_loss_frequency': 0.0,
            'equity_stability': 0.0
        }
    
    returns = np.array(actual_returns)
    
    # Decision statistics
    approvals = sum(1 for d in decisions if d.decision == TradeDecision.APPROVED)
    rejections = sum(1 for d in decisions if d.decision == TradeDecision.REJECTED)
    reductions = sum(1 for d in decisions if d.decision == TradeDecision.SIZE_REDUCED)
    total = len(decisions)
    
    # Return statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = mean_return / (std_return + 1e-10) * np.sqrt(252)
    
    # Maximum drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-10 + abs(cumulative.min()))
    max_dd = abs(np.min(drawdown))
    
    # CVaR
    var_95 = np.percentile(-returns, 95)
    tail_losses = -returns[-returns <= -var_95]
    cvar_95 = np.mean(tail_losses) if len(tail_losses) > 0 else var_95
    
    # Tail loss frequency (losses > 2x std)
    tail_threshold = -2 * std_return
    tail_losses_count = np.sum(returns < tail_threshold)
    tail_loss_freq = tail_losses_count / len(returns)
    
    # Equity stability (inverse of return std)
    equity_stability = 1 / (1 + std_return * 100)
    
    return {
        'approval_rate': approvals / total,
        'rejection_rate': rejections / total,
        'size_reduction_rate': reductions / total,
        'mean_return': mean_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'cvar_95': cvar_95,
        'tail_loss_frequency': tail_loss_freq,
        'equity_stability': equity_stability
    }
