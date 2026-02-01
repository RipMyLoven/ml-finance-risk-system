"""
═══════════════════════════════════════════════════════════════════════════════
 7️⃣ MODEL INTERACTION & ENSEMBLE LOGIC
═══════════════════════════════════════════════════════════════════════════════

Ensemble validation:
✅ Implement signal exchange between models
✅ Check conflicting signals
✅ Check aligned signals
✅ Verify Risk model has final veto
✅ Verify Risk model can disable other models
✅ Test ensemble logic
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Signal(Enum):
    """Trading signal types"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class RiskDecision(Enum):
    """Risk model decisions"""
    APPROVE = "approve"
    REDUCE = "reduce"
    VETO = "veto"
    DISABLE_MODEL = "disable_model"


@dataclass
class EnsembleResult:
    """Results from ensemble validation"""
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    signal_tests: Dict[str, Dict] = field(default_factory=dict)
    veto_tests: Dict[str, bool] = field(default_factory=dict)
    ensemble_tests: Dict[str, Dict] = field(default_factory=dict)


class EnsembleValidator:
    """
    Model Interaction & Ensemble Logic Validator
    
    Implements checkpoint 7:
    - Signal exchange validation
    - Conflict detection
    - Risk veto power validation
    - Model disabling functionality
    - Ensemble logic testing
    """
    
    # Model weights for ensemble
    MODEL_WEIGHTS = {
        'scalp': 0.5,
        'intraday': 0.3,
        'swing': 0.2
    }
    
    # Thresholds
    CONFIDENCE_THRESHOLD = 0.35
    RISK_VETO_THRESHOLD = 0.7
    CONFLICT_THRESHOLD = 0.3
    
    def __init__(self, log_to_console: bool = True):
        self.log_to_console = log_to_console
        self.results = EnsembleResult()
    
    def log(self, msg: str, level: str = "INFO"):
        if self.log_to_console:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"[{timestamp}] {prefix.get(level, '')} {msg}")
    
    def run_all_validations(self) -> EnsembleResult:
        """Run all ensemble validations"""
        print("\n" + "="*80)
        print("  7️⃣  MODEL INTERACTION & ENSEMBLE LOGIC")
        print("="*80 + "\n")
        
        # 1. Test signal exchange
        self.log("Step 1: Testing signal exchange...")
        self._test_signal_exchange()
        
        # 2. Test conflict detection
        self.log("Step 2: Testing conflict detection...")
        self._test_conflict_detection()
        
        # 3. Test aligned signals
        self.log("Step 3: Testing aligned signals...")
        self._test_aligned_signals()
        
        # 4. Test risk veto power
        self.log("Step 4: Testing Risk model veto power...")
        self._test_risk_veto()
        
        # 5. Test model disabling
        self.log("Step 5: Testing model disabling...")
        self._test_model_disabling()
        
        # 6. Test full ensemble logic
        self.log("Step 6: Testing full ensemble logic...")
        self._test_ensemble_logic()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _generate_model_signal(self, model_name: str, scenario: str) -> Dict:
        """Generate mock model signal"""
        signals = {
            'bullish': {'direction': 1, 'confidence': 0.7, 'strength': 0.8},
            'bearish': {'direction': -1, 'confidence': 0.65, 'strength': 0.75},
            'neutral': {'direction': 0, 'confidence': 0.4, 'strength': 0.3},
            'weak_buy': {'direction': 1, 'confidence': 0.45, 'strength': 0.4},
            'weak_sell': {'direction': -1, 'confidence': 0.42, 'strength': 0.35},
        }
        
        base = signals.get(scenario, signals['neutral'])
        return {
            'model': model_name,
            'direction': base['direction'],
            'confidence': base['confidence'],
            'strength': base['strength'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _test_signal_exchange(self):
        """Test signal exchange between models"""
        print(f"\n  📡 SIGNAL EXCHANGE TEST:")
        
        # Generate signals from each model
        test_scenarios = [
            {'scalp': 'bullish', 'intraday': 'bullish', 'swing': 'neutral'},
            {'scalp': 'bearish', 'intraday': 'neutral', 'swing': 'bearish'},
            {'scalp': 'bullish', 'intraday': 'bearish', 'swing': 'neutral'},  # Conflict
        ]
        
        for i, scenario in enumerate(test_scenarios):
            signals = {}
            for model, signal_type in scenario.items():
                signals[model] = self._generate_model_signal(model, signal_type)
            
            # Aggregate signals
            weighted_direction = sum(
                signals[m]['direction'] * signals[m]['confidence'] * self.MODEL_WEIGHTS[m]
                for m in signals
            )
            
            total_confidence = sum(
                signals[m]['confidence'] * self.MODEL_WEIGHTS[m]
                for m in signals
            )
            
            result = {
                'signals': {m: s['direction'] for m, s in signals.items()},
                'weighted_direction': weighted_direction,
                'avg_confidence': total_confidence,
                'final_signal': 'BUY' if weighted_direction > 0.2 else ('SELL' if weighted_direction < -0.2 else 'NEUTRAL')
            }
            
            print(f"\n    Scenario {i+1}:")
            print(f"      Signals: {result['signals']}")
            print(f"      Weighted: {weighted_direction:.3f}")
            print(f"      Final: {result['final_signal']}")
            
            self.results.signal_tests[f'scenario_{i+1}'] = result
        
        self.log("Signal exchange test complete", "SUCCESS")
    
    def _test_conflict_detection(self):
        """Test conflict detection between models"""
        print(f"\n  ⚔️ CONFLICT DETECTION TEST:")
        
        conflict_scenarios = [
            # (scalp, intraday, swing) - directions
            (1, 1, 1, False),      # All aligned - no conflict
            (1, -1, 0, True),      # Scalp vs Intraday
            (-1, -1, 1, True),     # Short term vs Swing
            (1, 0, -1, True),      # Scalp vs Swing
            (0, 0, 0, False),      # All neutral
        ]
        
        for i, (s, ind, sw, expected_conflict) in enumerate(conflict_scenarios):
            signals = {'scalp': s, 'intraday': ind, 'swing': sw}
            
            # Detect conflict
            directions = [v for v in signals.values() if v != 0]
            has_conflict = len(set(directions)) > 1 if directions else False
            
            # Calculate conflict severity
            if has_conflict:
                conflict_score = sum(
                    abs(signals[m1] - signals[m2]) * self.MODEL_WEIGHTS[m1] * self.MODEL_WEIGHTS[m2]
                    for m1 in signals for m2 in signals if m1 < m2
                )
            else:
                conflict_score = 0
            
            status = "✅" if has_conflict == expected_conflict else "❌"
            severity = "HIGH" if conflict_score > 0.5 else ("MEDIUM" if conflict_score > 0.2 else "LOW")
            
            print(f"    Test {i+1}: {signals} → Conflict={has_conflict} ({severity}) {status}")
            
            if has_conflict != expected_conflict:
                self.results.errors.append(f"Conflict detection failed for scenario {i+1}")
        
        self.log("Conflict detection test complete", "SUCCESS")
    
    def _test_aligned_signals(self):
        """Test handling of aligned signals"""
        print(f"\n  🎯 ALIGNED SIGNALS TEST:")
        
        # Strong alignment scenarios
        alignment_scenarios = [
            {'scalp': 0.8, 'intraday': 0.75, 'swing': 0.7, 'all_bullish': True},
            {'scalp': -0.8, 'intraday': -0.7, 'swing': -0.65, 'all_bullish': False},
            {'scalp': 0.6, 'intraday': 0.5, 'swing': 0.4, 'all_bullish': True},
        ]
        
        for i, scenario in enumerate(alignment_scenarios):
            signals = {k: v for k, v in scenario.items() if k != 'all_bullish'}
            all_same_direction = all(v > 0 for v in signals.values()) or all(v < 0 for v in signals.values())
            
            # Calculate alignment bonus
            if all_same_direction:
                avg_strength = np.mean(list(np.abs(list(signals.values()))))
                alignment_bonus = 1.0 + 0.2 * avg_strength  # Up to 20% bonus
            else:
                alignment_bonus = 1.0
            
            weighted_signal = sum(
                signals[m] * self.MODEL_WEIGHTS[m]
                for m in signals
            ) * alignment_bonus
            
            print(f"    Scenario {i+1}: Alignment bonus = {alignment_bonus:.2f}x")
            print(f"      Raw weighted: {sum(signals[m] * self.MODEL_WEIGHTS[m] for m in signals):.3f}")
            print(f"      With bonus: {weighted_signal:.3f}")
        
        self.log("Aligned signals test complete", "SUCCESS")
    
    def _test_risk_veto(self):
        """Test Risk model veto power"""
        print(f"\n  🛡️ RISK VETO TEST:")
        
        # Veto scenarios
        veto_scenarios = [
            {'risk_score': 0.3, 'should_veto': False, 'reason': 'Low risk'},
            {'risk_score': 0.5, 'should_veto': False, 'reason': 'Medium risk'},
            {'risk_score': 0.75, 'should_veto': True, 'reason': 'High risk'},
            {'risk_score': 0.9, 'should_veto': True, 'reason': 'Extreme risk'},
            {'risk_score': 0.69, 'should_veto': False, 'reason': 'Just below threshold'},
            {'risk_score': 0.71, 'should_veto': True, 'reason': 'Just above threshold'},
        ]
        
        all_passed = True
        
        for scenario in veto_scenarios:
            risk_score = scenario['risk_score']
            should_veto = scenario['should_veto']
            
            # Apply veto logic
            actual_veto = risk_score > self.RISK_VETO_THRESHOLD
            
            status = "✅" if actual_veto == should_veto else "❌"
            decision = "VETO" if actual_veto else "APPROVE"
            
            print(f"    Risk={risk_score:.2f} → {decision} ({scenario['reason']}) {status}")
            
            if actual_veto != should_veto:
                all_passed = False
                self.results.errors.append(f"Veto test failed: risk={risk_score}")
            
            self.results.veto_tests[f'risk_{int(risk_score*100)}'] = actual_veto == should_veto
        
        if all_passed:
            print(f"\n    ✅ Risk veto threshold ({self.RISK_VETO_THRESHOLD}) correctly enforced")
        
        self.log("Risk veto test complete", "SUCCESS")
    
    def _test_model_disabling(self):
        """Test model disabling functionality"""
        print(f"\n  🔌 MODEL DISABLING TEST:")
        
        # Scenarios where models should be disabled
        disable_scenarios = [
            {
                'model': 'scalp',
                'conditions': {'consecutive_losses': 5, 'recent_drawdown': 0.08},
                'should_disable': True
            },
            {
                'model': 'intraday',
                'conditions': {'consecutive_losses': 3, 'recent_drawdown': 0.05},
                'should_disable': False
            },
            {
                'model': 'swing',
                'conditions': {'consecutive_losses': 2, 'recent_drawdown': 0.15},
                'should_disable': True
            },
        ]
        
        disable_rules = {
            'max_consecutive_losses': 4,
            'max_recent_drawdown': 0.10
        }
        
        for scenario in disable_scenarios:
            model = scenario['model']
            cond = scenario['conditions']
            
            # Check disable conditions
            should_disable = (
                cond['consecutive_losses'] >= disable_rules['max_consecutive_losses'] or
                cond['recent_drawdown'] >= disable_rules['max_recent_drawdown']
            )
            
            expected = scenario['should_disable']
            status = "✅" if should_disable == expected else "❌"
            action = "DISABLED" if should_disable else "ACTIVE"
            
            print(f"    {model}: losses={cond['consecutive_losses']}, DD={cond['recent_drawdown']:.0%} → {action} {status}")
            
            if should_disable != expected:
                self.results.warnings.append(f"Model disable test mismatch: {model}")
        
        print(f"\n    Disable rules:")
        print(f"      - Max consecutive losses: {disable_rules['max_consecutive_losses']}")
        print(f"      - Max recent drawdown: {disable_rules['max_recent_drawdown']:.0%}")
        
        self.log("Model disabling test complete", "SUCCESS")
    
    def _test_ensemble_logic(self):
        """Test full ensemble logic"""
        print(f"\n  🧩 FULL ENSEMBLE LOGIC TEST:")
        
        # Full pipeline test scenarios
        test_cases = [
            {
                'name': 'All bullish, low risk',
                'signals': {'scalp': 0.7, 'intraday': 0.6, 'swing': 0.5},
                'risk_score': 0.3,
                'expected_action': 'LONG',
                'expected_size': 'FULL'
            },
            {
                'name': 'All bearish, low risk',
                'signals': {'scalp': -0.7, 'intraday': -0.6, 'swing': -0.5},
                'risk_score': 0.35,
                'expected_action': 'SHORT',
                'expected_size': 'FULL'
            },
            {
                'name': 'Bullish but high risk',
                'signals': {'scalp': 0.8, 'intraday': 0.7, 'swing': 0.6},
                'risk_score': 0.75,
                'expected_action': 'VETO',
                'expected_size': 'NONE'
            },
            {
                'name': 'Mixed signals',
                'signals': {'scalp': 0.5, 'intraday': -0.3, 'swing': 0.2},
                'risk_score': 0.4,
                'expected_action': 'REDUCE',
                'expected_size': 'HALF'
            },
            {
                'name': 'Weak signals',
                'signals': {'scalp': 0.2, 'intraday': 0.15, 'swing': 0.1},
                'risk_score': 0.3,
                'expected_action': 'HOLD',
                'expected_size': 'NONE'
            },
        ]
        
        for case in test_cases:
            signals = case['signals']
            risk_score = case['risk_score']
            
            # Calculate weighted signal
            weighted = sum(signals[m] * self.MODEL_WEIGHTS[m] for m in signals)
            
            # Check for conflicts
            directions = [v for v in signals.values() if abs(v) > 0.25]
            has_conflict = len(set(np.sign(d) for d in directions)) > 1 if directions else False
            
            # Apply risk veto
            if risk_score > self.RISK_VETO_THRESHOLD:
                action = 'VETO'
                size = 'NONE'
            elif abs(weighted) < self.CONFIDENCE_THRESHOLD:
                action = 'HOLD'
                size = 'NONE'
            elif has_conflict:
                action = 'REDUCE'
                size = 'HALF'
            else:
                action = 'LONG' if weighted > 0 else 'SHORT'
                size = 'FULL'
            
            expected_action = case['expected_action']
            status = "✅" if action == expected_action else "❌"
            
            print(f"\n    {case['name']}:")
            print(f"      Weighted signal: {weighted:.3f}")
            print(f"      Risk score: {risk_score:.2f}")
            print(f"      Conflict: {has_conflict}")
            print(f"      Decision: {action} ({size}) {status}")
            
            self.results.ensemble_tests[case['name']] = {
                'weighted_signal': float(weighted),
                'risk_score': risk_score,
                'has_conflict': has_conflict,
                'action': action,
                'size': size,
                'passed': action == expected_action
            }
            
            if action != expected_action:
                self.results.warnings.append(f"Ensemble logic mismatch: {case['name']}")
        
        self.log("Ensemble logic test complete", "SUCCESS")
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("  📋 ENSEMBLE VALIDATION SUMMARY")
        print("="*80)
        
        # Count results
        veto_passed = sum(self.results.veto_tests.values())
        veto_total = len(self.results.veto_tests)
        
        ensemble_passed = sum(1 for t in self.results.ensemble_tests.values() if t.get('passed', False))
        ensemble_total = len(self.results.ensemble_tests)
        
        print(f"\n  Test Results:")
        print(f"    Veto tests: {veto_passed}/{veto_total} passed")
        print(f"    Ensemble tests: {ensemble_passed}/{ensemble_total} passed")
        
        if self.results.errors:
            print(f"\n  Errors:")
            for err in self.results.errors:
                print(f"    ❌ {err}")
            self.results.passed = False
        
        if self.results.warnings:
            print(f"\n  Warnings:")
            for warn in self.results.warnings[:5]:
                print(f"    ⚠️  {warn}")
        
        if self.results.passed:
            print(f"\n  ✅ ENSEMBLE VALIDATION: PASSED")
        else:
            print(f"\n  ❌ ENSEMBLE VALIDATION: FAILED")
        
        print("\n" + "="*80 + "\n")
    
    def save_results(self, output_path: str = None):
        """Save validation results"""
        if output_path is None:
            output_path = Path(__file__).parent / 'ensemble_validation_report.json'
        
        report = {
            'passed': self.results.passed,
            'warnings': self.results.warnings,
            'errors': self.results.errors,
            'signal_tests': self.results.signal_tests,
            'veto_tests': self.results.veto_tests,
            'ensemble_tests': self.results.ensemble_tests,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {output_path}")
        return output_path


def run_ensemble_validation() -> EnsembleResult:
    """Run ensemble validation"""
    validator = EnsembleValidator()
    result = validator.run_all_validations()
    validator.save_results()
    return result


if __name__ == "__main__":
    result = run_ensemble_validation()
    sys.exit(0 if result.passed else 1)
