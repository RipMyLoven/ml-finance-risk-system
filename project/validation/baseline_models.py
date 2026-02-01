"""
═══════════════════════════════════════════════════════════════════════════════
 4️⃣ BASELINE MODELS
═══════════════════════════════════════════════════════════════════════════════

Baseline model training:
✅ Train baseline for each model
✅ Record baseline AUC
✅ Use baseline as reference
✅ Compare all improvements to baseline
"""

import os
import sys
import json
import pickle
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
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BaselineResult:
    """Results from baseline training"""
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    baseline_metrics: Dict[str, Dict] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)


class BaselineTrainer:
    """
    Baseline Model Trainer
    
    Implements checkpoint 4:
    - Train baseline model for each type
    - Record AUC, accuracy, log_loss
    - Save baselines for comparison
    """
    
    # Simple baseline LightGBM params
    BASELINE_PARAMS = {
        'objective': 'multiclass',
        'num_class': 3,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'n_estimators': 100,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
    
    BASELINE_PARAMS_REGRESSION = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'n_estimators': 100,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
    
    # Model configurations
    MODEL_CONFIGS = {
        'scalp': {
            'type': 'classification',
            'horizon': 12,
            'threshold': 0.003,
            'timeframe': '5m'
        },
        'intraday': {
            'type': 'classification',
            'horizon': 6,
            'threshold': 0.01,
            'timeframe': '1h'
        },
        'swing': {
            'type': 'classification',
            'horizon': 7,
            'threshold': 0.03,
            'timeframe': '1d'
        },
        'risk': {
            'type': 'regression',
            'horizon': 1,
            'timeframe': '1h'
        }
    }
    
    def __init__(self, log_to_console: bool = True):
        self.log_to_console = log_to_console
        self.results = BaselineResult()
        self.scalers = {}
    
    def log(self, msg: str, level: str = "INFO"):
        if self.log_to_console:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"[{timestamp}] {prefix.get(level, '')} {msg}")
    
    def run_all_baselines(self, df: pd.DataFrame = None) -> BaselineResult:
        """Train all baseline models"""
        print("\n" + "="*80)
        print("  4️⃣  BASELINE MODELS")
        print("="*80 + "\n")
        
        # Load data if not provided
        if df is None:
            df = self._load_data()
        
        if df is None or len(df) < 1000:
            self.results.errors.append("Insufficient data for baseline training")
            self.results.passed = False
            return self.results
        
        # Build features
        self.log("Building features...")
        df = self._build_features(df)
        
        # Train each baseline
        for model_name, config in self.MODEL_CONFIGS.items():
            self.log(f"Training {model_name.upper()} baseline...")
            self._train_baseline(df, model_name, config)
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _load_data(self) -> Optional[pd.DataFrame]:
        """Load training data"""
        data_dir = Path(__file__).parent.parent.parent / 'data'
        
        # Try to load 1h data
        klines = list(data_dir.glob('klines_*BTCUSDT_1h.csv'))
        if klines:
            try:
                df = pd.read_csv(klines[0])
                self.log(f"Loaded {len(df)} rows from {klines[0].name}")
                return df
            except Exception as e:
                self.log(f"Error loading data: {e}", "ERROR")
        
        return None
    
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build basic features for baseline"""
        df = df.copy()
        
        # Returns
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # Log returns
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
        
        # Moving average deviations
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
        
        # Bollinger Band position
        bb_mean = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - bb_mean) / (2 * bb_std + 1e-10)
        
        return df.dropna()
    
    def _train_baseline(self, df: pd.DataFrame, model_name: str, config: Dict):
        """Train a single baseline model"""
        try:
            # Get feature columns
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
                # Regression target: future volatility
                target = df['log_return'].rolling(horizon).std().shift(-horizon) * np.sqrt(252)
            
            # Prepare data
            X = df[feature_cols].values
            y = target
            
            # Remove NaN
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 1000:
                self.results.warnings.append(f"{model_name}: Insufficient data after cleaning")
                return
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[model_name] = scaler
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=5)
            
            metrics_list = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                if config['type'] == 'classification':
                    model = lgb.LGBMClassifier(**self.BASELINE_PARAMS)
                    model.fit(X_train, y_train.astype(int))
                    
                    y_pred = model.predict(X_val)
                    y_pred_proba = model.predict_proba(X_val)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_val.astype(int), y_pred)
                    logloss = log_loss(y_val.astype(int), y_pred_proba)
                    
                    # Multi-class AUC (one-vs-rest)
                    try:
                        auc = roc_auc_score(y_val.astype(int), y_pred_proba, multi_class='ovr')
                    except:
                        auc = 0.5
                    
                    metrics_list.append({
                        'accuracy': accuracy,
                        'log_loss': logloss,
                        'auc': auc
                    })
                else:
                    model = lgb.LGBMRegressor(**self.BASELINE_PARAMS_REGRESSION)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_val)
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    
                    metrics_list.append({
                        'rmse': rmse
                    })
            
            # Average metrics
            if config['type'] == 'classification':
                avg_metrics = {
                    'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
                    'accuracy_std': np.std([m['accuracy'] for m in metrics_list]),
                    'log_loss': np.mean([m['log_loss'] for m in metrics_list]),
                    'auc': np.mean([m['auc'] for m in metrics_list]),
                    'auc_std': np.std([m['auc'] for m in metrics_list])
                }
            else:
                avg_metrics = {
                    'rmse': np.mean([m['rmse'] for m in metrics_list]),
                    'rmse_std': np.std([m['rmse'] for m in metrics_list])
                }
            
            avg_metrics['n_samples'] = len(X)
            avg_metrics['n_features'] = len(feature_cols)
            avg_metrics['type'] = config['type']
            
            self.results.baseline_metrics[model_name] = avg_metrics
            
            # Train final model on all data
            if config['type'] == 'classification':
                final_model = lgb.LGBMClassifier(**self.BASELINE_PARAMS)
                final_model.fit(X_scaled, y.astype(int))
            else:
                final_model = lgb.LGBMRegressor(**self.BASELINE_PARAMS_REGRESSION)
                final_model.fit(X_scaled, y)
            
            self.results.models[model_name] = {
                'model': final_model,
                'scaler': scaler,
                'feature_cols': feature_cols
            }
            
            # Log results
            print(f"\n  📊 {model_name.upper()} BASELINE:")
            print(f"     Samples: {len(X)}, Features: {len(feature_cols)}")
            
            if config['type'] == 'classification':
                print(f"     Accuracy: {avg_metrics['accuracy']:.4f} (±{avg_metrics['accuracy_std']:.4f})")
                print(f"     AUC: {avg_metrics['auc']:.4f} (±{avg_metrics['auc_std']:.4f})")
                print(f"     Log Loss: {avg_metrics['log_loss']:.4f}")
            else:
                print(f"     RMSE: {avg_metrics['rmse']:.4f} (±{avg_metrics['rmse_std']:.4f})")
            
            self.log(f"{model_name} baseline complete", "SUCCESS")
            
        except Exception as e:
            self.results.errors.append(f"{model_name}: Training failed - {e}")
            self.log(f"{model_name} baseline failed: {e}", "ERROR")
    
    def _print_summary(self):
        """Print training summary"""
        print("\n" + "="*80)
        print("  📋 BASELINE TRAINING SUMMARY")
        print("="*80)
        
        print(f"\n  📊 BASELINE REFERENCE METRICS:")
        print(f"  " + "-"*60)
        print(f"  {'Model':<12} {'Type':<15} {'Main Metric':<20} {'Value':<15}")
        print(f"  " + "-"*60)
        
        for model_name, metrics in self.results.baseline_metrics.items():
            if metrics['type'] == 'classification':
                main_metric = 'AUC'
                value = f"{metrics['auc']:.4f}"
            else:
                main_metric = 'RMSE'
                value = f"{metrics['rmse']:.4f}"
            
            print(f"  {model_name:<12} {metrics['type']:<15} {main_metric:<20} {value:<15}")
        
        print(f"  " + "-"*60)
        
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
            print(f"\n  ✅ BASELINE TRAINING: PASSED")
        else:
            print(f"\n  ❌ BASELINE TRAINING: FAILED")
        
        print("\n" + "="*80 + "\n")
    
    def save_baselines(self, output_dir: str = None):
        """Save baseline models and metrics"""
        if output_dir is None:
            output_dir = Path(__file__).parent / 'baselines'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = output_dir / 'baseline_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.results.baseline_metrics, f, indent=2)
        
        # Save models
        for model_name, model_data in self.results.models.items():
            model_file = output_dir / f'{model_name}_baseline.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
        
        print(f"Baselines saved to: {output_dir}")
        return output_dir


def run_baseline_training(df: pd.DataFrame = None) -> BaselineResult:
    """Run baseline training for all models"""
    trainer = BaselineTrainer()
    result = trainer.run_all_baselines(df)
    trainer.save_baselines()
    return result


if __name__ == "__main__":
    result = run_baseline_training()
    sys.exit(0 if result.passed else 1)
