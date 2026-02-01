"""
═══════════════════════════════════════════════════════════════════════════════
 3️⃣ LABELING & TARGETS VALIDATION
═══════════════════════════════════════════════════════════════════════════════

Target validation:
✅ Define targets for each model explicitly
✅ Check prediction horizons correctness
✅ Verify no target leakage
✅ Verify correct shifts
✅ Log target distributions
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
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class LabelingResult:
    """Results from labeling validation"""
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    target_definitions: Dict[str, Dict] = field(default_factory=dict)
    distributions: Dict[str, Dict] = field(default_factory=dict)
    leakage_checks: Dict[str, Any] = field(default_factory=dict)


class LabelingValidator:
    """
    Labeling & Targets Validator
    
    Implements checkpoint 3:
    - Define targets for each model
    - Validate prediction horizons
    - Detect target leakage
    - Validate shift correctness
    - Log target distributions
    """
    
    # Target definitions
    TARGET_DEFINITIONS = {
        'scalp': {
            'horizon_bars': 12,  # 12 bars = 1 hour on 5m
            'threshold_pct': 0.3,  # 0.3% move
            'timeframe': '5m',
            'description': 'Short-term price direction prediction'
        },
        'intraday': {
            'horizon_bars': 6,  # 6 bars = 6 hours on 1h
            'threshold_pct': 1.0,  # 1% move
            'timeframe': '1h',
            'description': 'Medium-term trend prediction'
        },
        'swing': {
            'horizon_bars': 7,  # 7 bars = 7 days on 1d
            'threshold_pct': 3.0,  # 3% move
            'timeframe': '1d',
            'description': 'Long-term position prediction'
        },
        'risk': {
            'horizon_bars': 1,
            'target_type': 'regression',
            'description': 'Volatility/risk estimation'
        }
    }
    
    # Expected class distribution (for imbalance detection)
    MAX_CLASS_IMBALANCE = 5.0  # ratio
    
    def __init__(self, log_to_console: bool = True):
        self.log_to_console = log_to_console
        self.results = LabelingResult()
    
    def log(self, msg: str, level: str = "INFO"):
        if self.log_to_console:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"[{timestamp}] {prefix.get(level, '')} {msg}")
    
    def run_all_checks(self, df: pd.DataFrame = None) -> LabelingResult:
        """Run all labeling validation checks"""
        print("\n" + "="*80)
        print("  3️⃣  LABELING & TARGETS VALIDATION")
        print("="*80 + "\n")
        
        # Load data if not provided
        if df is None:
            df = self._load_sample_data()
        
        if df is None or len(df) == 0:
            self.results.errors.append("No data available for validation")
            self.results.passed = False
            return self.results
        
        # 1. Define targets explicitly
        self.log("Step 1: Defining targets for each model...")
        self._define_targets()
        
        # 2. Create and validate targets
        self.log("Step 2: Creating and validating targets...")
        targets_df = self._create_targets(df)
        
        # 3. Check horizons
        self.log("Step 3: Checking prediction horizons...")
        self._check_horizons(df, targets_df)
        
        # 4. Check for target leakage
        self.log("Step 4: Detecting target leakage...")
        self._check_target_leakage(df, targets_df)
        
        # 5. Validate shifts
        self.log("Step 5: Validating shift correctness...")
        self._validate_shifts(df, targets_df)
        
        # 6. Log distributions
        self.log("Step 6: Logging target distributions...")
        self._log_distributions(targets_df)
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _load_sample_data(self) -> Optional[pd.DataFrame]:
        """Load sample data"""
        data_dir = Path(__file__).parent.parent.parent / 'data'
        
        # Try 1h klines for balanced check
        klines = list(data_dir.glob('klines_*BTCUSDT_1h.csv'))
        if klines:
            try:
                return pd.read_csv(klines[0])
            except:
                pass
        
        # Fallback to any klines
        klines = list(data_dir.glob('klines_*.csv'))
        if klines:
            try:
                return pd.read_csv(klines[0])
            except:
                pass
        
        return None
    
    def _define_targets(self):
        """Define and log target definitions for each model"""
        print(f"\n  🎯 TARGET DEFINITIONS:")
        
        for model_name, definition in self.TARGET_DEFINITIONS.items():
            self.results.target_definitions[model_name] = definition
            
            print(f"\n    {model_name.upper()} Model:")
            print(f"      - Horizon: {definition.get('horizon_bars', 'N/A')} bars")
            
            if 'threshold_pct' in definition:
                print(f"      - Threshold: {definition['threshold_pct']}%")
                print(f"      - Classes: DOWN (0), NEUTRAL (1), UP (2)")
            elif definition.get('target_type') == 'regression':
                print(f"      - Type: Regression (continuous)")
            
            if 'timeframe' in definition:
                print(f"      - Timeframe: {definition['timeframe']}")
            
            print(f"      - Description: {definition['description']}")
        
        self.log("Target definitions complete", "SUCCESS")
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create targets for validation"""
        targets_df = df[['close']].copy() if 'close' in df.columns else pd.DataFrame()
        
        for model_name, definition in self.TARGET_DEFINITIONS.items():
            if 'horizon_bars' not in definition:
                continue
            
            horizon = definition['horizon_bars']
            
            # Calculate future returns
            future_return = df['close'].pct_change(horizon).shift(-horizon)
            
            if 'threshold_pct' in definition:
                # Classification target
                threshold = definition['threshold_pct'] / 100
                
                target = np.where(
                    future_return > threshold, 2,  # UP
                    np.where(future_return < -threshold, 0, 1)  # DOWN, NEUTRAL
                )
                targets_df[f'target_{model_name}'] = target
            else:
                # Regression target (for risk model)
                targets_df[f'target_{model_name}'] = future_return.rolling(horizon).std() * np.sqrt(252)
        
        return targets_df
    
    def _check_horizons(self, df: pd.DataFrame, targets_df: pd.DataFrame):
        """Validate prediction horizons"""
        print(f"\n  ⏱️ HORIZON VALIDATION:")
        
        issues = []
        
        for model_name, definition in self.TARGET_DEFINITIONS.items():
            horizon = definition.get('horizon_bars', 1)
            
            # Check if we have enough data
            min_required = horizon * 10
            actual_rows = len(df)
            
            if actual_rows < min_required:
                issues.append(f"{model_name}: Need {min_required} rows, have {actual_rows}")
                print(f"    ⚠️  {model_name}: Insufficient data for horizon {horizon}")
            else:
                print(f"    ✅ {model_name}: Horizon {horizon} bars validated")
            
            # Check NaN ratio in target
            target_col = f'target_{model_name}'
            if target_col in targets_df.columns:
                nan_pct = targets_df[target_col].isna().mean() * 100
                if nan_pct > 20:
                    issues.append(f"{model_name}: {nan_pct:.1f}% NaN in target")
                    print(f"    ⚠️  {model_name}: High NaN ratio ({nan_pct:.1f}%)")
        
        if issues:
            self.results.warnings.extend(issues)
        else:
            self.log("All horizons validated", "SUCCESS")
    
    def _check_target_leakage(self, df: pd.DataFrame, targets_df: pd.DataFrame):
        """Check for target leakage"""
        print(f"\n  🚨 TARGET LEAKAGE CHECK:")
        
        leakage_found = []
        
        for model_name in self.TARGET_DEFINITIONS.keys():
            target_col = f'target_{model_name}'
            
            if target_col not in targets_df.columns:
                continue
            
            target = targets_df[target_col].dropna()
            
            # Test 1: Check correlation with current/past price features
            for col in ['close', 'open', 'high', 'low', 'volume']:
                if col in df.columns:
                    # Current bar correlation
                    aligned_data = pd.concat([
                        df[col].iloc[target.index], 
                        target
                    ], axis=1).dropna()
                    
                    if len(aligned_data) > 100:
                        corr = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                        if abs(corr) > 0.7:
                            leakage_found.append(f"{model_name}: {col} highly correlated with target ({corr:.3f})")
            
            # Test 2: Train-test temporal check
            mid_point = len(target) // 2
            train_dist = Counter(target.iloc[:mid_point].astype(int))
            test_dist = Counter(target.iloc[mid_point:].astype(int))
            
            # Check if distributions are similar (good) or identical (suspicious)
            if train_dist == test_dist:
                leakage_found.append(f"{model_name}: Train/test distributions identical (suspicious)")
        
        if leakage_found:
            for leak in leakage_found[:5]:
                print(f"    ⚠️  {leak}")
            self.results.warnings.extend(leakage_found)
        else:
            print(f"    ✅ No obvious target leakage detected")
            self.log("Leakage check passed", "SUCCESS")
        
        self.results.leakage_checks = {'issues': leakage_found}
    
    def _validate_shifts(self, df: pd.DataFrame, targets_df: pd.DataFrame):
        """Validate that shifts are correct (no future data in features)"""
        print(f"\n  🔄 SHIFT VALIDATION:")
        
        shift_issues = []
        
        for model_name, definition in self.TARGET_DEFINITIONS.items():
            horizon = definition.get('horizon_bars', 1)
            target_col = f'target_{model_name}'
            
            if target_col not in targets_df.columns:
                continue
            
            # Check that last `horizon` rows should be NaN
            last_n = targets_df[target_col].iloc[-horizon:]
            nan_count = last_n.isna().sum()
            
            if nan_count != horizon:
                shift_issues.append(f"{model_name}: Last {horizon} rows should be NaN, but {nan_count} are")
                print(f"    ⚠️  {model_name}: Shift validation failed")
            else:
                print(f"    ✅ {model_name}: Shift correctly applied (-{horizon} bars)")
        
        if shift_issues:
            self.results.warnings.extend(shift_issues)
        else:
            self.log("All shifts validated", "SUCCESS")
    
    def _log_distributions(self, targets_df: pd.DataFrame):
        """Log target distributions"""
        print(f"\n  📊 TARGET DISTRIBUTIONS:")
        
        for model_name, definition in self.TARGET_DEFINITIONS.items():
            target_col = f'target_{model_name}'
            
            if target_col not in targets_df.columns:
                continue
            
            target = targets_df[target_col].dropna()
            
            if definition.get('target_type') == 'regression':
                # Regression stats
                stats = {
                    'count': len(target),
                    'mean': float(target.mean()),
                    'std': float(target.std()),
                    'min': float(target.min()),
                    'max': float(target.max()),
                    'median': float(target.median()),
                    'q25': float(target.quantile(0.25)),
                    'q75': float(target.quantile(0.75))
                }
                
                print(f"\n    {model_name.upper()} (Regression):")
                print(f"      - Mean: {stats['mean']:.4f}")
                print(f"      - Std: {stats['std']:.4f}")
                print(f"      - Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            else:
                # Classification distribution
                target_int = target.astype(int)
                value_counts = target_int.value_counts().sort_index()
                
                total = len(target)
                dist = {int(k): int(v) for k, v in value_counts.items()}
                pct = {int(k): round(v/total*100, 1) for k, v in value_counts.items()}
                
                print(f"\n    {model_name.upper()} (Classification):")
                print(f"      - Total samples: {total}")
                
                class_names = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}
                for cls in [0, 1, 2]:
                    cnt = dist.get(cls, 0)
                    pct_val = pct.get(cls, 0)
                    print(f"      - Class {cls} ({class_names.get(cls, '?')}): {cnt} ({pct_val}%)")
                
                # Check imbalance
                if dist:
                    max_class = max(dist.values())
                    min_class = min(dist.values()) or 1
                    imbalance = max_class / min_class
                    
                    if imbalance > self.MAX_CLASS_IMBALANCE:
                        self.results.warnings.append(f"{model_name}: Class imbalance ratio {imbalance:.1f}")
                        print(f"      ⚠️  Imbalance ratio: {imbalance:.1f}")
                    else:
                        print(f"      ✅ Balanced (ratio: {imbalance:.1f})")
                
                stats = {'distribution': dist, 'percentages': pct}
            
            self.results.distributions[model_name] = stats
        
        self.log("Distributions logged", "SUCCESS")
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("  📋 LABELING VALIDATION SUMMARY")
        print("="*80)
        
        print(f"\n  Total Warnings: {len(self.results.warnings)}")
        print(f"  Total Errors: {len(self.results.errors)}")
        
        if self.results.warnings:
            print(f"\n  Warnings:")
            for warn in self.results.warnings[:10]:
                print(f"    ⚠️  {warn}")
        
        if self.results.errors:
            print(f"\n  Errors:")
            for err in self.results.errors:
                print(f"    ❌ {err}")
            self.results.passed = False
        
        if self.results.passed:
            print(f"\n  ✅ LABELING VALIDATION: PASSED")
        else:
            print(f"\n  ❌ LABELING VALIDATION: FAILED")
        
        print("\n" + "="*80 + "\n")
    
    def save_report(self, output_path: str = None):
        """Save validation report"""
        if output_path is None:
            output_path = Path(__file__).parent / 'labeling_validation_report.json'
        
        report = {
            'passed': self.results.passed,
            'warnings': self.results.warnings,
            'errors': self.results.errors,
            'target_definitions': self.results.target_definitions,
            'distributions': self.results.distributions,
            'leakage_checks': self.results.leakage_checks,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {output_path}")
        return output_path


def run_labeling_validation(df: pd.DataFrame = None) -> LabelingResult:
    """Run complete labeling validation"""
    validator = LabelingValidator()
    result = validator.run_all_checks(df)
    validator.save_report()
    return result


if __name__ == "__main__":
    result = run_labeling_validation()
    sys.exit(0 if result.passed else 1)
