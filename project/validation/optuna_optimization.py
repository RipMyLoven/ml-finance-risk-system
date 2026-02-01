"""
═══════════════════════════════════════════════════════════════════════════════
 5️⃣ OPTUNA OPTIMIZATION (Per Model)
═══════════════════════════════════════════════════════════════════════════════

Hyperparameter optimization:
✅ Create Optuna study for each model
✅ Define search space
✅ Run sufficient trials
✅ Use pruning (MedianPruner)
✅ Log each trial: params, metrics
✅ Record best params
✅ Check stability
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class OptunaResult:
    """Results from Optuna optimization"""
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    best_params: Dict[str, Dict] = field(default_factory=dict)
    best_metrics: Dict[str, Dict] = field(default_factory=dict)
    trial_histories: Dict[str, List[Dict]] = field(default_factory=dict)
    stability_checks: Dict[str, Dict] = field(default_factory=dict)


class OptunaOptimizer:
    """
    Optuna Hyperparameter Optimizer
    
    Implements checkpoint 5:
    - Per-model Optuna studies
    - Comprehensive search space
    - Pruning for efficiency
    - Trial logging
    - Stability validation
    """
    
    # Number of trials per model
    N_TRIALS = {
        'scalp': 50,
        'intraday': 50,
        'swing': 50,
        'risk': 30
    }
    
    # Model configurations
    MODEL_CONFIGS = {
        'scalp': {
            'type': 'classification',
            'horizon': 12,
            'threshold': 0.003,
            'metric': 'auc'
        },
        'intraday': {
            'type': 'classification',
            'horizon': 6,
            'threshold': 0.01,
            'metric': 'auc'
        },
        'swing': {
            'type': 'classification',
            'horizon': 7,
            'threshold': 0.03,
            'metric': 'auc'
        },
        'risk': {
            'type': 'regression',
            'horizon': 1,
            'metric': 'rmse'
        }
    }
    
    def __init__(self, log_to_console: bool = True, quick_mode: bool = False):
        """
        Args:
            log_to_console: Print logs
            quick_mode: Use fewer trials for testing
        """
        self.log_to_console = log_to_console
        self.quick_mode = quick_mode
        self.results = OptunaResult()
        
        if quick_mode:
            self.N_TRIALS = {k: min(10, v) for k, v in self.N_TRIALS.items()}
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install: pip install optuna")
    
    def log(self, msg: str, level: str = "INFO"):
        if self.log_to_console:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"[{timestamp}] {prefix.get(level, '')} {msg}")
    
    def run_all_optimizations(self, df: pd.DataFrame = None) -> OptunaResult:
        """Run Optuna optimization for all models"""
        print("\n" + "="*80)
        print("  5️⃣  OPTUNA OPTIMIZATION")
        print("="*80 + "\n")
        
        # Load data if not provided
        if df is None:
            df = self._load_data()
        
        if df is None or len(df) < 1000:
            self.results.errors.append("Insufficient data for optimization")
            self.results.passed = False
            return self.results
        
        # Build features
        self.log("Building features...")
        df = self._build_features(df)
        
        # Optimize each model
        for model_name, config in self.MODEL_CONFIGS.items():
            print(f"\n  {'='*60}")
            print(f"  🔬 OPTIMIZING {model_name.upper()} MODEL")
            print(f"  {'='*60}")
            
            self._optimize_model(df, model_name, config)
        
        # Check stability
        self.log("Checking optimization stability...")
        self._check_stability()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _load_data(self) -> Optional[pd.DataFrame]:
        """Load training data"""
        data_dir = Path(__file__).parent.parent.parent / 'data'
        
        klines = list(data_dir.glob('klines_*BTCUSDT_1h.csv'))
        if klines:
            try:
                df = pd.read_csv(klines[0])
                self.log(f"Loaded {len(df)} rows")
                return df
            except Exception as e:
                self.log(f"Error loading data: {e}", "ERROR")
        
        return None
    
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features for optimization"""
        df = df.copy()
        
        # Returns
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['volatility_10'] = df['log_return'].rolling(10).std()
        df['volatility_20'] = df['log_return'].rolling(20).std()
        
        # Volume
        df['volume_ratio_10'] = df['volume'] / (df['volume'].rolling(10).mean() + 1e-10)
        df['volume_ratio_20'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        
        # Moving averages
        for period in [10, 20, 50]:
            sma = df['close'].rolling(period).mean()
            df[f'sma_dev_{period}'] = (df['close'] - sma) / (sma + 1e-10)
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_norm'] = df['atr'] / df['close']
        
        # Bollinger
        bb_mean = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - bb_mean) / (2 * bb_std + 1e-10)
        
        return df.dropna()
    
    def _get_search_space(self, trial: 'optuna.Trial', model_type: str) -> Dict:
        """Define search space for LightGBM"""
        params = {
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42
        }
        
        if model_type == 'classification':
            params['objective'] = 'multiclass'
            params['num_class'] = 3
        else:
            params['objective'] = 'regression'
        
        return params
    
    def _optimize_model(self, df: pd.DataFrame, model_name: str, config: Dict):
        """Optimize a single model"""
        try:
            # Prepare data
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                           'close_time', 'quote_volume', 'trades', 'symbol', 
                           'interval', 'market_type', 'taker_buy_volume', 
                           'taker_buy_quote_volume']
            
            feature_cols = [c for c in df.columns if c not in exclude_cols 
                          and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
            
            # Create target
            horizon = config['horizon']
            
            if config['type'] == 'classification':
                threshold = config['threshold']
                future_return = df['close'].pct_change(horizon).shift(-horizon)
                target = np.where(
                    future_return > threshold, 2,
                    np.where(future_return < -threshold, 0, 1)
                )
            else:
                target = df['log_return'].rolling(horizon).std().shift(-horizon) * np.sqrt(252)
            
            # Prepare arrays
            X = df[feature_cols].values
            y = target
            
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Scale
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Trial history
            trial_history = []
            
            def objective(trial):
                params = self._get_search_space(trial, config['type'])
                
                fold_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    if config['type'] == 'classification':
                        model = lgb.LGBMClassifier(**params)
                        model.fit(X_train, y_train.astype(int),
                                 eval_set=[(X_val, y_val.astype(int))],
                                 callbacks=[lgb.early_stopping(10, verbose=False)])
                        
                        y_pred_proba = model.predict_proba(X_val)
                        
                        try:
                            score = roc_auc_score(y_val.astype(int), y_pred_proba, multi_class='ovr')
                        except:
                            score = 0.5
                    else:
                        model = lgb.LGBMRegressor(**params)
                        model.fit(X_train, y_train,
                                 eval_set=[(X_val, y_val)],
                                 callbacks=[lgb.early_stopping(10, verbose=False)])
                        
                        y_pred = model.predict(X_val)
                        score = -np.sqrt(mean_squared_error(y_val, y_pred))  # Negative for maximization
                    
                    fold_scores.append(score)
                    
                    # Pruning check
                    trial.report(np.mean(fold_scores), fold)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                avg_score = np.mean(fold_scores)
                
                # Log trial
                trial_info = {
                    'trial_number': trial.number,
                    'params': params.copy(),
                    'score': avg_score,
                    'fold_scores': fold_scores
                }
                # Remove non-serializable items
                trial_info['params'].pop('verbose', None)
                trial_info['params'].pop('n_jobs', None)
                trial_history.append(trial_info)
                
                return avg_score
            
            # Create and run study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1)
            )
            
            n_trials = self.N_TRIALS[model_name]
            
            print(f"\n  Running {n_trials} trials...")
            
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=True,
                callbacks=[lambda study, trial: self._trial_callback(study, trial, model_name)]
            )
            
            # Save results
            self.results.best_params[model_name] = study.best_params
            
            if config['type'] == 'classification':
                self.results.best_metrics[model_name] = {
                    'auc': study.best_value,
                    'n_trials': len(study.trials),
                    'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                }
            else:
                self.results.best_metrics[model_name] = {
                    'rmse': -study.best_value,  # Convert back
                    'n_trials': len(study.trials),
                    'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                }
            
            self.results.trial_histories[model_name] = trial_history
            
            # Log best params
            print(f"\n  ✅ Best {model_name.upper()} params:")
            for param, value in study.best_params.items():
                print(f"     {param}: {value}")
            
            metric_name = 'AUC' if config['type'] == 'classification' else 'RMSE'
            metric_val = study.best_value if config['type'] == 'classification' else -study.best_value
            print(f"\n  📊 Best {metric_name}: {metric_val:.4f}")
            
            # Save study
            self._save_study(study, model_name)
            
            self.log(f"{model_name} optimization complete", "SUCCESS")
            
        except Exception as e:
            self.results.errors.append(f"{model_name}: Optimization failed - {e}")
            self.log(f"{model_name} optimization failed: {e}", "ERROR")
    
    def _trial_callback(self, study: 'optuna.Study', trial: 'optuna.trial.FrozenTrial', model_name: str):
        """Callback after each trial"""
        if trial.number % 10 == 0:
            print(f"    Trial {trial.number}: {trial.value:.4f} (Best: {study.best_value:.4f})")
    
    def _check_stability(self):
        """Check optimization stability"""
        print(f"\n  🔍 STABILITY CHECK:")
        
        for model_name, history in self.results.trial_histories.items():
            if len(history) < 10:
                continue
            
            # Get scores from last 50% of trials
            n_trials = len(history)
            later_trials = history[n_trials//2:]
            scores = [t['score'] for t in later_trials]
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            cv = std_score / (abs(mean_score) + 1e-10)
            
            stability = {
                'mean': mean_score,
                'std': std_score,
                'cv': cv,
                'stable': cv < 0.1  # Less than 10% coefficient of variation
            }
            
            self.results.stability_checks[model_name] = stability
            
            if stability['stable']:
                print(f"    ✅ {model_name}: Stable (CV = {cv:.3f})")
            else:
                print(f"    ⚠️  {model_name}: Unstable (CV = {cv:.3f})")
                self.results.warnings.append(f"{model_name}: Optimization may be unstable")
    
    def _save_study(self, study: 'optuna.Study', model_name: str):
        """Save Optuna study"""
        output_dir = Path(__file__).parent.parent / 'optuna_studies'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle
        study_file = output_dir / f'{model_name}_study_validation.pkl'
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
    
    def _print_summary(self):
        """Print optimization summary"""
        print("\n" + "="*80)
        print("  📋 OPTUNA OPTIMIZATION SUMMARY")
        print("="*80)
        
        print(f"\n  📊 BEST PARAMETERS & METRICS:")
        print(f"  " + "-"*60)
        
        for model_name in self.MODEL_CONFIGS.keys():
            if model_name in self.results.best_metrics:
                metrics = self.results.best_metrics[model_name]
                
                print(f"\n  {model_name.upper()}:")
                
                if 'auc' in metrics:
                    print(f"    Best AUC: {metrics['auc']:.4f}")
                else:
                    print(f"    Best RMSE: {metrics['rmse']:.4f}")
                
                print(f"    Trials: {metrics['n_trials']} (Pruned: {metrics['n_pruned']})")
                
                # Show key params
                if model_name in self.results.best_params:
                    params = self.results.best_params[model_name]
                    print(f"    Key params:")
                    for key in ['n_estimators', 'num_leaves', 'learning_rate', 'max_depth']:
                        if key in params:
                            print(f"      - {key}: {params[key]}")
        
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
            print(f"\n  ✅ OPTUNA OPTIMIZATION: PASSED")
        else:
            print(f"\n  ❌ OPTUNA OPTIMIZATION: FAILED")
        
        print("\n" + "="*80 + "\n")
    
    def save_results(self, output_path: str = None):
        """Save optimization results"""
        if output_path is None:
            output_path = Path(__file__).parent / 'optuna_results.json'
        
        report = {
            'passed': self.results.passed,
            'warnings': self.results.warnings,
            'errors': self.results.errors,
            'best_params': self.results.best_params,
            'best_metrics': self.results.best_metrics,
            'stability_checks': self.results.stability_checks,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Results saved to: {output_path}")
        return output_path


def run_optuna_optimization(df: pd.DataFrame = None, quick_mode: bool = False) -> OptunaResult:
    """Run Optuna optimization for all models"""
    optimizer = OptunaOptimizer(quick_mode=quick_mode)
    result = optimizer.run_all_optimizations(df)
    optimizer.save_results()
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer trials')
    args = parser.parse_args()
    
    result = run_optuna_optimization(quick_mode=args.quick)
    sys.exit(0 if result.passed else 1)
