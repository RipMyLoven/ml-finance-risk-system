"""
Integration Test for Advanced Risk-Controlled Trading System

This module tests the complete integration of:
- CVaR Engine
- Kelly Position Sizing
- Drawdown Controller
- Advanced Risk Model
- Trade Flow Simulation

Run: python -m pytest tests/test_risk_integration.py -v
Or: python tests/test_risk_integration.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.cvar_engine import CVaREngine, RiskLevel
from risk.kelly_sizing import KellySizer, KellyMode
from risk.drawdown_controller import DrawdownController, RiskState, ModelState
from risk.risk_model import (
    AdvancedRiskModel, 
    TradeProposal, 
    TradeDecision,
    calculate_risk_model_metrics
)


class TradingSystemSimulator:
    """
    Simulates the complete trading system with risk controls
    
    This demonstrates how the components work together:
    1. Trading models generate proposals
    2. Risk model evaluates proposals
    3. Position sizing is calculated
    4. Trades are executed or rejected
    5. PnL is tracked
    6. Drawdown and CVaR are monitored
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """Initialize simulator"""
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Initialize risk model
        self.risk_model = AdvancedRiskModel(
            initial_equity=initial_capital,
            cvar_confidence=0.95,
            kelly_mode=KellyMode.QUARTER,
            max_leverage=5.0,
            max_position_pct=0.10,
            enable_kill_switch=True,
            log_all_decisions=False  # Quiet for tests
        )
        
        # Track history
        self.equity_history = [initial_capital]
        self.trade_history = []
        self.decision_history = []
        
    def simulate_trades(
        self,
        n_trades: int = 100,
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.015
    ) -> Dict:
        """
        Simulate a series of trades through the risk system
        
        Args:
            n_trades: Number of trades to simulate
            win_rate: Probability of winning
            avg_win: Average win percentage
            avg_loss: Average loss percentage
            
        Returns:
            Simulation results
        """
        np.random.seed(42)
        
        model_types = ['scalp', 'intraday', 'swing']
        
        for i in range(n_trades):
            # Generate random trade proposal
            model_type = np.random.choice(model_types)
            direction = np.random.choice(['long', 'short'])
            
            # Model confidence varies
            confidence = np.random.uniform(0.4, 0.9)
            probability = np.random.uniform(0.4, 0.7)
            
            proposal = TradeProposal(
                model_type=model_type,
                symbol='BTCUSDT',
                direction=direction,
                probability=probability,
                confidence=confidence,
                expected_return=np.random.uniform(0.01, 0.03),
                volatility_regime=np.random.choice(['low', 'normal', 'high']),
                proposed_size=self.capital * 0.10,  # 10% of capital
                stop_loss=0.02,
                take_profit=0.03
            )
            
            # Get risk model decision
            decision = self.risk_model.evaluate_trade(proposal)
            self.decision_history.append(decision)
            
            # If approved, simulate trade outcome
            if decision.decision != TradeDecision.REJECTED:
                # Determine if trade wins
                is_win = np.random.random() < win_rate
                
                if is_win:
                    pnl_pct = np.random.exponential(avg_win)
                else:
                    pnl_pct = -np.random.exponential(avg_loss)
                
                # Apply to approved size
                pnl = decision.approved_size * pnl_pct
                self.capital += pnl
                
                # Update risk model
                self.risk_model.update_equity(self.capital)
                self.risk_model.update_pnl(model_type, pnl)
                
                self.trade_history.append({
                    'trade_num': i,
                    'model_type': model_type,
                    'direction': direction,
                    'is_win': is_win,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'approved_size': decision.approved_size,
                    'risk_state': decision.risk_state.value
                })
            
            self.equity_history.append(self.capital)
            
            # Check for kill switch
            if self.risk_model.drawdown_controller.is_emergency:
                print(f"Emergency stop triggered at trade {i}!")
                break
        
        return self._compile_results()
    
    def _compile_results(self) -> Dict:
        """Compile simulation results"""
        equity = np.array(self.equity_history)
        
        # Calculate metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = abs(np.min(drawdown))
        
        # Trade stats
        n_approved = sum(1 for d in self.decision_history 
                        if d.decision != TradeDecision.REJECTED)
        n_rejected = sum(1 for d in self.decision_history 
                        if d.decision == TradeDecision.REJECTED)
        n_reduced = sum(1 for d in self.decision_history 
                       if d.decision == TradeDecision.SIZE_REDUCED)
        
        # Win rate of executed trades
        if self.trade_history:
            actual_win_rate = sum(1 for t in self.trade_history if t['is_win']) / len(self.trade_history)
        else:
            actual_win_rate = 0.0
        
        return {
            'final_capital': self.capital,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'n_trades_proposed': len(self.decision_history),
            'n_trades_approved': n_approved,
            'n_trades_rejected': n_rejected,
            'n_trades_reduced': n_reduced,
            'rejection_rate': n_rejected / max(len(self.decision_history), 1),
            'actual_win_rate': actual_win_rate,
            'n_trades_executed': len(self.trade_history),
            'risk_state': self.risk_model.get_global_state()
        }


def test_cvar_engine():
    """Test CVaR Engine functionality"""
    print("\n" + "="*50)
    print("TEST: CVaR Engine")
    print("="*50)
    
    engine = CVaREngine(confidence_level=0.95)
    
    # Generate test returns with fat tails
    np.random.seed(42)
    normal_returns = np.random.normal(0, 0.02, 200)
    
    # Add some tail events
    tail_events = np.random.normal(-0.10, 0.02, 10)
    returns = np.concatenate([normal_returns, tail_events])
    
    # Calculate CVaR
    result = engine.calculate_cvar(returns)
    
    print(f"  VaR (95%): {result.var:.4f}")
    print(f"  CVaR (95%): {result.cvar:.4f}")
    print(f"  Risk Level: {result.risk_level.value}")
    print(f"  Max Loss: {result.max_loss:.4f}")
    print(f"  Is Blocked: {result.is_blocked}")
    
    assert result.cvar > result.var, "CVaR should be >= VaR"
    assert result.risk_level in RiskLevel, "Should have valid risk level"
    
    print("  ✅ CVaR Engine test PASSED")
    return result


def test_kelly_sizer():
    """Test Kelly Position Sizing"""
    print("\n" + "="*50)
    print("TEST: Kelly Position Sizer")
    print("="*50)
    
    sizer = KellySizer(
        mode=KellyMode.QUARTER,
        max_leverage=5.0,
        max_position_pct=0.10
    )
    
    # Test with good edge
    result = sizer.calculate_kelly(
        win_rate=0.55,
        payoff_ratio=1.5,
        confidence=0.8,
        current_volatility=0.02,
        model_type='intraday',
        capital=10000.0
    )
    
    print(f"  Win Rate: {result.win_rate:.2f}")
    print(f"  Payoff Ratio: {result.payoff_ratio:.2f}")
    print(f"  Raw Kelly: {result.raw_kelly:.4f}")
    print(f"  Kelly Fraction Used: {result.kelly_fraction:.2f}")
    print(f"  Final Size: ${result.final_size:.2f}")
    print(f"  Confidence Adjustment: {result.confidence_adjustment:.2f}")
    print(f"  Constraints: {result.constraints_applied}")
    
    assert result.final_size > 0, "Should produce positive size with edge"
    assert result.kelly_fraction <= 0.25, "Should cap Kelly at 25%"
    
    # Test with negative edge
    result_neg = sizer.calculate_kelly(
        win_rate=0.40,
        payoff_ratio=0.8,
        confidence=0.5,
        model_type='scalp',
        capital=10000.0
    )
    
    print(f"\n  Negative Edge Case:")
    print(f"  Raw Kelly: {result_neg.raw_kelly:.4f}")
    print(f"  Final Size: ${result_neg.final_size:.2f}")
    
    assert result_neg.final_size == 0, "Should not trade with negative edge"
    
    print("  ✅ Kelly Sizer test PASSED")
    return result


def test_drawdown_controller():
    """Test Drawdown Controller"""
    print("\n" + "="*50)
    print("TEST: Drawdown Controller")
    print("="*50)
    
    controller = DrawdownController(
        initial_equity=10000.0,
        enable_emergency_stop=True,
        log_all_decisions=False
    )
    
    # Simulate drawdown
    equity_values = [
        10000, 9800, 9600, 9400, 9200,  # 8% DD
        9000, 8800, 8600, 8400, 8200,   # 18% DD
        8000, 7800, 7600                 # 24% DD
    ]
    
    print("  Simulating drawdown progression...")
    
    for i, equity in enumerate(equity_values):
        metrics = controller.update(equity)
        
        if i in [0, 4, 9, 12]:
            print(f"  Equity ${equity}: DD={metrics.current_drawdown:.1%}, "
                  f"State={metrics.global_risk_state.value}, "
                  f"Multiplier={metrics.risk_multiplier:.2f}")
    
    # Check final state
    final_metrics = controller.get_metrics()
    
    print(f"\n  Final Drawdown: {final_metrics.current_drawdown:.1%}")
    print(f"  Max Drawdown: {final_metrics.max_drawdown:.1%}")
    print(f"  Risk State: {final_metrics.global_risk_state.value}")
    print(f"  Model States: {dict(final_metrics.model_states)}")
    
    assert final_metrics.current_drawdown > 0.20, "Should detect significant DD"
    assert final_metrics.global_risk_state in [RiskState.RISK_OFF, RiskState.EMERGENCY]
    
    print("  ✅ Drawdown Controller test PASSED")
    return final_metrics


def test_risk_model_integration():
    """Test complete Risk Model integration"""
    print("\n" + "="*50)
    print("TEST: Risk Model Integration")
    print("="*50)
    
    risk_model = AdvancedRiskModel(
        initial_equity=10000.0,
        log_all_decisions=False
    )
    
    # Create trade proposal
    proposal = TradeProposal(
        model_type='intraday',
        symbol='ETHUSDT',
        direction='long',
        probability=0.60,
        confidence=0.75,
        expected_return=0.02,
        volatility_regime='normal',
        proposed_size=1000.0,
        stop_loss=0.02,
        take_profit=0.04
    )
    
    # Evaluate trade
    decision = risk_model.evaluate_trade(proposal)
    
    print(f"  Proposal: {proposal.model_type} {proposal.direction}")
    print(f"  Decision: {decision.decision.value}")
    print(f"  Approved Size: ${decision.approved_size:.2f}")
    print(f"  Adjusted Leverage: {decision.adjusted_leverage:.2f}x")
    print(f"  Risk State: {decision.risk_state.value}")
    print(f"  Reason: {decision.reason}")
    
    assert decision.decision != None, "Should produce decision"
    
    # Simulate some trades to build history
    print("\n  Simulating trades...")
    
    for i in range(20):
        pnl = np.random.normal(0, 50)  # Random PnL
        risk_model.update_pnl('intraday', pnl)
        risk_model.update_equity(10000 + np.sum([np.random.normal(0, 50) for _ in range(i)]))
    
    # Get global state
    state = risk_model.get_global_state()
    print(f"  Global State: {state}")
    
    print("  ✅ Risk Model Integration test PASSED")
    return decision


def test_trading_simulation():
    """Test full trading simulation"""
    print("\n" + "="*50)
    print("TEST: Full Trading Simulation")
    print("="*50)
    
    simulator = TradingSystemSimulator(initial_capital=10000.0)
    
    # Run simulation
    results = simulator.simulate_trades(
        n_trades=100,
        win_rate=0.52,  # Slight edge
        avg_win=0.02,
        avg_loss=0.015
    )
    
    print(f"  Final Capital: ${results['final_capital']:.2f}")
    print(f"  Total Return: {results['total_return']:.1%}")
    print(f"  Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"  Trades Proposed: {results['n_trades_proposed']}")
    print(f"  Trades Approved: {results['n_trades_approved']}")
    print(f"  Trades Rejected: {results['n_trades_rejected']}")
    print(f"  Rejection Rate: {results['rejection_rate']:.1%}")
    print(f"  Actual Win Rate: {results['actual_win_rate']:.1%}")
    
    # Risk should have prevented excessive loss
    assert results['max_drawdown'] < 0.50, "Risk controls should limit drawdown"
    
    print("  ✅ Trading Simulation test PASSED")
    return results


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("ADVANCED RISK SYSTEM INTEGRATION TESTS")
    print("="*60)
    print(f"Time: {datetime.now()}")
    
    results = {}
    
    # Run tests
    results['cvar'] = test_cvar_engine()
    results['kelly'] = test_kelly_sizer()
    results['drawdown'] = test_drawdown_controller()
    results['integration'] = test_risk_model_integration()
    results['simulation'] = test_trading_simulation()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✅")
    print("="*60)
    
    return results


if __name__ == "__main__":
    run_all_tests()
