"""
═══════════════════════════════════════════════════════════════════════════════
 8️⃣ METRICS & OVERFITTING CONTROL
═══════════════════════════════════════════════════════════════════════════════

Overfitting detection:
✅ Check Train/Validation/Test metrics
✅ Compare error distributions
✅ Check stability over time
✅ Check degradation on OOS
✅ Record final metrics
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

warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class OverfittingResult:
    """Results from overfitting analysis"""
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics_by_split: Dict[str, Dict] = field(default_factory=dict)
    error_distributions: Dict[str, Dict] = field(default_factory=dict)
    time_stability: Dict[str, Dict] = field(default_factory=dict)
    oos_degradation: Dict[str, Dict] = field(default_factory=dict)
    final_metrics: Dict[str, Dict] = field(default_factory=dict)


class OverfittingController:
    """
    Metrics & Overfitting Controller
    
    Implements checkpoint 8:
    - Multi-split metrics analysis
    - Error distribution comparison
    - Time stability testing
    - OOS degradation detection
    """
    
    # Overfitting thresholds
    MAX_TRAIN_TEST_GAP = 0.15  # Max 15% gap between train and test
    MAX_TIME_DEGRADATION = 0.20  # Max 20% degradation over time
    MIN_CORRELATION_THRESHOLD = 0.5  # Minimum prediction correlation over time
    
    # Model configurations
    MODEL_CONFIGS = {
        'scalp': {'type': 'classification', 'horizon': 12, 'threshold': 0.003},
        'intraday': {'type': 'classification', 'horizon': 6, 'threshold': 0.01},
        'swing': {'type': 'classification', 'horizon': 7, 'threshold': 0.03},
        'risk': {'type': 'regression', 'horizon': 1}
    }
    
    def __init__(self, log_to_console: bool = True):
        self.log_to_console = log_to_console
        self.results = OverfittingResult()
    
    def log(self, msg: str, level: str = "INFO"):
        if self.log_to_console:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"[{timestamp}] {prefix.get(level, '')} {msg}")
    
    def run_all_checks(self, df: pd.DataFrame = None) -> OverfittingResult:
        """Run all overfitting checks"""
        print("\n" + "="*80)
        print("  8️⃣  METRICS & OVERFITTING CONTROL")
        print("="*80 + "\n")
        
        # Load data
        if df is None:
            df = self._load_data()
        
        if df is None or len(df) < 1000:
            self.results.errors.append("Insufficient data")
            self.results.passed = False
            return self.results
        
        # Build features
        self.log("Building features...")
        df = self._build_features(df)
        
        # Run checks for each model
        for model_name, config in self.MODEL_CONFIGS.items():
            print(f"\n  {'='*50}")
            print(f"  📊 Analyzing {model_name.upper()} Model")
            print(f"  {'='*50}")
            
            # 1. Train/Val/Test metrics
            self.log(f"Checking Train/Val/Test metrics for {model_name}...")
            self._check_split_metrics(df, model_name, config)
            
            # 2. Error distributions
            self.log(f"Comparing error distributions for {model_name}...")
            self._check_error_distributions(df, model_name, config)
            
            # 3. Time stability
            self.log(f"Checking time stability for {model_name}...")
            self._check_time_stability(df, model_name, config)
            
            # 4. OOS degradation
            self.log(f"Checking OOS degradation for {model_name}...")
            self._check_oos_degradation(df, model_name, config)
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _load_data(self) -> Optional[pd.DataFrame]:
        """Load training data"""
        data_dir = Path(__file__).parent.parent.parent / 'data'
        
        klines = list(data_dir.glob('klines_*BTCUSDT_1h.csv'))
        if klines:
            try:
                return pd.read_csv(klines[0])
            except:
                pass
        return None
    
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features"""
        df = df.copy()
        
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        df['volatility_10'] = df['log_return'].rolling(10).std()
        df['volatility_20'] = df['log_return'].rolling(20).std()
        
        df['volume_ratio_10'] = df['volume'] / (df['volume'].rolling(10).mean() + 1e-10)
        df['volume_ratio_20'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        
        for period in [10, 20, 50]:
            sma = df['close'].rolling(period).mean()
            df[f'sma_dev_{period}'] = (df['close'] - sma) / (sma + 1e-10)
        
        return df.dropna()
    
    def _prepare_data(self, df: pd.DataFrame, config: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare X, y arrays"""
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                       'close_time', 'quote_volume', 'trades', 'symbol', 
                       'interval', 'market_type', 'taker_buy_volume', 
                       'taker_buy_quote_volume']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols 
                       and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        horizon = config['horizon']
        
        if config['type'] == 'classification':
            threshold = config['threshold']
            future_return = df['close'].pct_change(horizon).shift(-horizon)
            y = np.where(future_return > threshold, 2,
                        np.where(future_return < -threshold, 0, 1))
        else:
            y = df['log_return'].rolling(horizon).std().shift(-horizon) * np.sqrt(252)
        
        X = df[feature_cols].values
        
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid_mask], y[valid_mask]
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y, feature_cols
    
    def _check_split_metrics(self, df: pd.DataFrame, model_name: str, config: Dict):
        """Check Train/Validation/Test metrics"""
        X, y, feature_cols = self._prepare_data(df, config)
        
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        # Train model
        if config['type'] == 'classification':
            model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
            model.fit(X_train, y_train.astype(int))
            
            train_pred = model.predict_proba(X_train)
            val_pred = model.predict_proba(X_val)
            test_pred = model.predict_proba(X_test)
            
            metrics = {
                'train': {
                    'accuracy': accuracy_score(y_train.astype(int), model.predict(X_train)),
                    'auc': roc_auc_score(y_train.astype(int), train_pred, multi_class='ovr'),
                    'log_loss': log_loss(y_train.astype(int), train_pred)
                },
                'val': {
                    'accuracy': accuracy_score(y_val.astype(int), model.predict(X_val)),
                    'auc': roc_auc_score(y_val.astype(int), val_pred, multi_class='ovr'),
                    'log_loss': log_loss(y_val.astype(int), val_pred)
                },
                'test': {
                    'accuracy': accuracy_score(y_test.astype(int), model.predict(X_test)),
                    'auc': roc_auc_score(y_test.astype(int), test_pred, multi_class='ovr'),
                    'log_loss': log_loss(y_test.astype(int), test_pred)
                }
            }
            primary_metric = 'auc'
        else:
            model = lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
            model.fit(X_train, y_train)
            
            metrics = {
                'train': {'rmse': np.sqrt(mean_squared_error(y_train, model.predict(X_train)))},
                'val': {'rmse': np.sqrt(mean_squared_error(y_val, model.predict(X_val)))},
                'test': {'rmse': np.sqrt(mean_squared_error(y_test, model.predict(X_test)))}
            }
            primary_metric = 'rmse'
        
        self.results.metrics_by_split[model_name] = metrics
        
        # Check for overfitting
        print(f"\n    📊 Split Metrics:")
        print(f"    {'Split':<10} {primary_metric.upper():<15}")
        print(f"    {'-'*25}")
        
        for split, m in metrics.items():
            print(f"    {split:<10} {m[primary_metric]:.4f}")
        
        # Calculate gaps
        if config['type'] == 'classification':
            train_val_gap = metrics['train']['auc'] - metrics['val']['auc']
            train_test_gap = metrics['train']['auc'] - metrics['test']['auc']
        else:
            train_val_gap = metrics['val']['rmse'] - metrics['train']['rmse']
            train_test_gap = metrics['test']['rmse'] - metrics['train']['rmse']
        
        if abs(train_test_gap) > self.MAX_TRAIN_TEST_GAP:
            self.results.warnings.append(f"{model_name}: Potential overfitting (gap={train_test_gap:.3f})")
            print(f"\n    ⚠️  Train-Test gap: {train_test_gap:.4f} (threshold: {self.MAX_TRAIN_TEST_GAP})")
        else:
            print(f"\n    ✅ Train-Test gap: {train_test_gap:.4f} (within threshold)")
    
    def _check_error_distributions(self, df: pd.DataFrame, model_name: str, config: Dict):
        """Compare error distributions across splits"""
        X, y, _ = self._prepare_data(df, config)
        
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        if config['type'] == 'classification':
            model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
            model.fit(X_train, y_train.astype(int))
            
            # Get probability errors
            train_proba = model.predict_proba(X_train)
            val_proba = model.predict_proba(X_val)
            test_proba = model.predict_proba(X_test)
            
            # Calculate cross-entropy errors per sample
            train_errors = -np.log(train_proba[np.arange(len(y_train)), y_train.astype(int)] + 1e-10)
            val_errors = -np.log(val_proba[np.arange(len(y_val)), y_val.astype(int)] + 1e-10)
            test_errors = -np.log(test_proba[np.arange(len(y_test)), y_test.astype(int)] + 1e-10)
        else:
            model = lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
            model.fit(X_train, y_train)
            
            train_errors = np.abs(y_train - model.predict(X_train))
            val_errors = np.abs(y_val - model.predict(X_val))
            test_errors = np.abs(y_test - model.predict(X_test))
        
        # Compare distributions using KS test
        ks_train_val = stats.ks_2samp(train_errors, val_errors)
        ks_train_test = stats.ks_2samp(train_errors, test_errors)
        ks_val_test = stats.ks_2samp(val_errors, test_errors)
        
        print(f"\n    📊 Error Distribution Comparison (KS test):")
        print(f"    {'Comparison':<20} {'Statistic':<12} {'p-value':<12}")
        print(f"    {'-'*44}")
        print(f"    {'Train vs Val':<20} {ks_train_val.statistic:.4f}{'':4} {ks_train_val.pvalue:.4f}")
        print(f"    {'Train vs Test':<20} {ks_train_test.statistic:.4f}{'':4} {ks_train_test.pvalue:.4f}")
        print(f"    {'Val vs Test':<20} {ks_val_test.statistic:.4f}{'':4} {ks_val_test.pvalue:.4f}")
        
        # Significant difference indicates distribution shift
        if ks_train_test.pvalue < 0.05:
            self.results.warnings.append(f"{model_name}: Significant error distribution shift (train vs test)")
            print(f"\n    ⚠️  Significant distribution shift detected")
        else:
            print(f"\n    ✅ Error distributions are similar")
        
        self.results.error_distributions[model_name] = {
            'train_val_ks': float(ks_train_val.statistic),
            'train_test_ks': float(ks_train_test.statistic),
            'val_test_ks': float(ks_val_test.statistic),
            'train_val_p': float(ks_train_val.pvalue),
            'train_test_p': float(ks_train_test.pvalue),
            'val_test_p': float(ks_val_test.pvalue)
        }
    
    def _check_time_stability(self, df: pd.DataFrame, model_name: str, config: Dict):
        """Check model stability over time"""
        X, y, _ = self._prepare_data(df, config)
        
        # Train on first 60%
        n = len(X)
        train_end = int(n * 0.6)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_test = X[train_end:]
        y_test = y[train_end:]
        
        if config['type'] == 'classification':
            model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
            model.fit(X_train, y_train.astype(int))
        else:
            model = lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
            model.fit(X_train, y_train)
        
        # Split test into time periods
        n_periods = 4
        period_size = len(X_test) // n_periods
        
        period_metrics = []
        
        for i in range(n_periods):
            start = i * period_size
            end = (i + 1) * period_size if i < n_periods - 1 else len(X_test)
            
            X_period = X_test[start:end]
            y_period = y_test[start:end]
            
            if config['type'] == 'classification':
                pred = model.predict_proba(X_period)
                try:
                    auc = roc_auc_score(y_period.astype(int), pred, multi_class='ovr')
                except:
                    auc = 0.5
                metric = auc
            else:
                pred = model.predict(X_period)
                rmse = np.sqrt(mean_squared_error(y_period, pred))
                metric = rmse
            
            period_metrics.append(metric)
        
        # Calculate degradation
        first_half_avg = np.mean(period_metrics[:2])
        second_half_avg = np.mean(period_metrics[2:])
        
        if config['type'] == 'classification':
            degradation = (first_half_avg - second_half_avg) / first_half_avg
        else:
            degradation = (second_half_avg - first_half_avg) / first_half_avg
        
        metric_name = 'AUC' if config['type'] == 'classification' else 'RMSE'
        
        print(f"\n    📊 Time Stability ({metric_name}):")
        print(f"    {'Period':<10} {metric_name:<15}")
        print(f"    {'-'*25}")
        for i, m in enumerate(period_metrics):
            print(f"    {f'Period {i+1}':<10} {m:.4f}")
        
        print(f"\n    Degradation: {degradation:.2%}")
        
        if abs(degradation) > self.MAX_TIME_DEGRADATION:
            self.results.warnings.append(f"{model_name}: Performance degrades over time ({degradation:.2%})")
            print(f"    ⚠️  Significant time degradation detected")
        else:
            print(f"    ✅ Performance stable over time")
        
        self.results.time_stability[model_name] = {
            'period_metrics': [float(m) for m in period_metrics],
            'degradation': float(degradation),
            'stable': abs(degradation) <= self.MAX_TIME_DEGRADATION
        }
    
    def _check_oos_degradation(self, df: pd.DataFrame, model_name: str, config: Dict):
        """Check out-of-sample degradation"""
        X, y, _ = self._prepare_data(df, config)
        
        # Walk-forward analysis
        n_folds = 5
        tscv = TimeSeriesSplit(n_splits=n_folds)
        
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if config['type'] == 'classification':
                model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
                model.fit(X_train, y_train.astype(int))
                
                pred = model.predict_proba(X_test)
                try:
                    metric = roc_auc_score(y_test.astype(int), pred, multi_class='ovr')
                except:
                    metric = 0.5
            else:
                model = lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
                model.fit(X_train, y_train)
                metric = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
            
            fold_metrics.append(metric)
        
        # Calculate OOS statistics
        mean_metric = np.mean(fold_metrics)
        std_metric = np.std(fold_metrics)
        cv = std_metric / (mean_metric + 1e-10)
        
        # Check for degradation trend
        trend = np.polyfit(range(n_folds), fold_metrics, 1)[0]
        
        metric_name = 'AUC' if config['type'] == 'classification' else 'RMSE'
        
        print(f"\n    📊 OOS Performance (Walk-Forward):")
        print(f"    Mean {metric_name}: {mean_metric:.4f} (±{std_metric:.4f})")
        print(f"    CV: {cv:.2%}")
        print(f"    Trend: {'Declining' if (trend < 0 and config['type'] == 'classification') or (trend > 0 and config['type'] != 'classification') else 'Stable/Improving'}")
        
        self.results.oos_degradation[model_name] = {
            'fold_metrics': [float(m) for m in fold_metrics],
            'mean': float(mean_metric),
            'std': float(std_metric),
            'cv': float(cv),
            'trend': float(trend)
        }
        
        # Final metrics
        self.results.final_metrics[model_name] = {
            'primary_metric': metric_name.lower(),
            'mean': float(mean_metric),
            'std': float(std_metric),
            'cv': float(cv)
        }
        
        if cv > 0.2:
            self.results.warnings.append(f"{model_name}: High OOS variance (CV={cv:.2%})")
            print(f"    ⚠️  High variance in OOS performance")
        else:
            print(f"    ✅ OOS performance is consistent")
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("  📋 OVERFITTING CONTROL SUMMARY")
        print("="*80)
        
        print(f"\n  📊 FINAL METRICS:")
        print(f"  " + "-"*60)
        print(f"  {'Model':<12} {'Metric':<10} {'Mean':<12} {'Std':<12} {'CV':<10}")
        print(f"  " + "-"*60)
        
        for model, metrics in self.results.final_metrics.items():
            print(f"  {model:<12} {metrics['primary_metric']:<10} "
                  f"{metrics['mean']:.4f}{'':4} {metrics['std']:.4f}{'':4} {metrics['cv']:.2%}")
        
        # Time stability summary
        print(f"\n  ⏰ TIME STABILITY:")
        for model, stability in self.results.time_stability.items():
            status = "✅" if stability.get('stable', False) else "⚠️"
            print(f"    {model}: {status} (degradation: {stability.get('degradation', 0):.2%})")
        
        if self.results.errors:
            print(f"\n  Errors:")
            for err in self.results.errors:
                print(f"    ❌ {err}")
            self.results.passed = False
        
        if self.results.warnings:
            print(f"\n  Warnings:")
            for warn in self.results.warnings[:10]:
                print(f"    ⚠️  {warn}")
        
        if self.results.passed:
            print(f"\n  ✅ OVERFITTING CHECK: PASSED")
        else:
            print(f"\n  ❌ OVERFITTING CHECK: FAILED")
        
        print("\n" + "="*80 + "\n")
    
    def save_results(self, output_path: str = None):
        """Save results"""
        if output_path is None:
            output_path = Path(__file__).parent / 'overfitting_report.json'
        
        report = {
            'passed': self.results.passed,
            'warnings': self.results.warnings,
            'errors': self.results.errors,
            'metrics_by_split': self.results.metrics_by_split,
            'error_distributions': self.results.error_distributions,
            'time_stability': self.results.time_stability,
            'oos_degradation': self.results.oos_degradation,
            'final_metrics': self.results.final_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {output_path}")
        return output_path


def run_overfitting_check(df: pd.DataFrame = None) -> OverfittingResult:
    """Run overfitting checks"""
    controller = OverfittingController()
    result = controller.run_all_checks(df)
    controller.save_results()
    return result


if __name__ == "__main__":
    result = run_overfitting_check()
    sys.exit(0 if result.passed else 1)
