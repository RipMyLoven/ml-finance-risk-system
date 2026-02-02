#!/usr/bin/env python
"""
UNIFIED HIGH-PERFORMANCE AI TRAINING SYSTEM
============================================
Single entry point for all training operations.

Usage:
    python train.py                        # Full training (all models + risk)
    python train.py --fast                 # Fast training (no Optuna)
    python train.py --trials 100           # Optuna optimization
    python train.py --model scalp          # Train specific model
    python train.py --gpu                  # Use GPU
    python train.py --benchmark            # Run performance benchmark
    python train.py --diagnose             # System diagnostics

Features:
    ✓ Multi-threaded data loading
    ✓ Optimized feature engineering
    ✓ Integrated risk management (CVaR, Kelly, Drawdown)
    ✓ Optuna hyperparameter optimization
    ✓ ONNX export with validation
    ✓ Maximum CPU/GPU utilization
"""

# ============== EARLY OPTIMIZATION ==============
import os
import sys
import multiprocessing as mp

def _apply_early_optimizations():
    """Apply optimizations BEFORE importing heavy libraries"""
    n_cpus = mp.cpu_count()
    
    # Set threading environment
    os.environ['OMP_NUM_THREADS'] = str(n_cpus)
    os.environ['MKL_NUM_THREADS'] = str(n_cpus)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_cpus)
    os.environ['NUMBA_NUM_THREADS'] = str(n_cpus)
    os.environ['NUMBA_THREADING_LAYER'] = 'omp'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    
    return n_cpus

N_CPUS = _apply_early_optimizations()

# ============== IMPORTS ==============
import json
import pickle
import argparse
import warnings
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, root_mean_squared_error

# Suppress ALL warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

# Config imports
from config import (
    LGBM_MARKET_PARAMS, LGBM_RISK_PARAMS, MODEL_PATHS, RANDOM_STATE,
    PRICE_MOVE_THRESHOLD, PREDICTION_HORIZON,
    TRAIN_TEST_SPLIT, VALIDATION_SPLIT,
    CVAR_CONFIG, KELLY_CONFIG, DRAWDOWN_CONFIG, RISK_MODEL_PATHS
)

# Feature builders
from features.scalp_features import build_scalp_features
from features.intraday_features import build_intraday_features
from features.swing_features import build_swing_features
from training.train_risk import build_risk_features

# Data loader
try:
    from data.data_loader_v2 import prepare_training_data_v2 as prepare_training_data, BinanceDataLoader
    DATA_LOADER_V2 = True
except ImportError:
    try:
        from data.data_loader import prepare_training_data
        DATA_LOADER_V2 = False
    except ImportError:
        DATA_LOADER_V2 = False
        prepare_training_data = None

# Risk module
try:
    from risk import (
        CVaREngine, 
        KellySizer, 
        DrawdownController,
        AdvancedRiskModel,
        build_advanced_risk_features
    )
    from risk.train_risk_optuna import RiskModelOptimizer, run_risk_optimization
    RISK_MODULE_AVAILABLE = True
except ImportError:
    RISK_MODULE_AVAILABLE = False

# Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ONNX
try:
    import onnx
    import onnxruntime as ort
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# ============== CONSTANTS ==============
STUDY_DIR = "optuna_studies"
BEST_PARAMS_FILE = "best_params.json"


# ============== PARALLEL FEATURE BUILDERS (module-level for pickling) ==============

def _build_scalp_single(args):
    """Build scalp features for single symbol (module-level for ProcessPool)"""
    symbol, df_dict, horizon, threshold = args
    try:
        df = pd.DataFrame(df_dict)
        from features.scalp_features import build_scalp_features
        df, f_names = build_scalp_features(df, horizon=horizon, threshold=threshold)
        if len(df) > 100 and 'target' in df.columns:
            X = df[f_names].values
            y = df['target'].values
            valid_mask = ~np.isnan(y)
            return {
                'X': X[valid_mask].astype(np.float32),
                'y': y[valid_mask].astype(np.int32),
                'feature_names': f_names,
                'symbol': symbol
            }
    except Exception as e:
        pass
    return None


def _build_intraday_single(args):
    """Build intraday features for single symbol (module-level for ProcessPool)"""
    symbol, df_dict, horizon, threshold = args
    try:
        df = pd.DataFrame(df_dict)
        from features.intraday_features import build_intraday_features
        df, f_names = build_intraday_features(df, horizon=horizon, threshold=threshold)
        if len(df) > 50 and 'target' in df.columns:
            X = df[f_names].values
            y = df['target'].values
            valid_mask = ~np.isnan(y)
            return {
                'X': X[valid_mask].astype(np.float32),
                'y': y[valid_mask].astype(np.int32),
                'feature_names': f_names,
                'symbol': symbol
            }
    except Exception as e:
        pass
    return None


def _build_swing_single(args):
    """Build swing features for single symbol (module-level for ProcessPool)"""
    symbol, df_dict, horizon, threshold = args
    try:
        df = pd.DataFrame(df_dict)
        from features.swing_features import build_swing_features
        adaptive_horizon = min(horizon, max(1, len(df) // 10))
        df, f_names = build_swing_features(df, horizon=adaptive_horizon, threshold=threshold)
        if len(df) > 30 and 'target' in df.columns:
            X = df[f_names].values
            y = df['target'].values
            valid_mask = ~np.isnan(y)
            return {
                'X': X[valid_mask].astype(np.float32),
                'y': y[valid_mask].astype(np.int32),
                'feature_names': f_names,
                'symbol': symbol
            }
    except Exception as e:
        pass
    return None


def _build_risk_single(args):
    """Build risk features for single symbol (module-level for ProcessPool)"""
    symbol, df_dict, use_advanced = args
    try:
        df = pd.DataFrame(df_dict)
        if use_advanced:
            from risk import build_advanced_risk_features
            df, f_names = build_advanced_risk_features(df, market_predictions=None, include_derivatives=True)
            target_col = 'risk_target'
        else:
            from training.train_risk import build_risk_features
            df, f_names = build_risk_features(df)
            df['future_vol'] = df['close'].pct_change().rolling(20).std().shift(-20)
            target_col = 'future_vol'
        
        if len(df) > 100 and target_col in df.columns:
            X = df[f_names].values
            y = df[target_col].values
            valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
            if np.sum(valid_mask) > 50:
                return {
                    'X': X[valid_mask].astype(np.float32),
                    'y': y[valid_mask].astype(np.float32),
                    'feature_names': f_names,
                    'symbol': symbol
                }
    except Exception as e:
        pass
    return None


class UnifiedTrainer:
    """
    Unified High-Performance Trainer
    
    Combines:
    - Multi-model training (scalp, intraday, swing)
    - Advanced risk model with CVaR, Kelly, Drawdown
    - Optuna hyperparameter optimization
    - Maximum performance optimizations
    """
    
    def __init__(
        self,
        model_type: str = "all",
        use_gpu: bool = False,
        n_trials: int = 100,
        n_jobs: int = -1,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        continue_study: bool = False,
        verbose: bool = True
    ):
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.n_trials = n_trials
        self.n_jobs = N_CPUS if n_jobs == -1 else n_jobs
        self.timeout = timeout
        self.study_name = study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.continue_study = continue_study
        self.verbose = verbose
        
        # Data storage
        self.raw_data = {}
        self.data = {}
        self.best_params = {}
        self.models = {}
        self.metrics = {}
        self.feature_names = {}
        
        # Risk components
        if RISK_MODULE_AVAILABLE:
            self.cvar_engine = CVaREngine(**{
                'confidence_level': CVAR_CONFIG['confidence_level'],
                'default_window': CVAR_CONFIG['default_window'],
                'block_threshold': CVAR_CONFIG['block_threshold'],
                'model_cvar_limits': CVAR_CONFIG['model_limits']
            })
            self.kelly_sizer = KellySizer(
                max_leverage=KELLY_CONFIG['max_leverage'],
                max_position_pct=KELLY_CONFIG['max_position_pct'],
                model_leverage_limits=KELLY_CONFIG['model_leverage_limits']
            )
        
        # Create directories
        os.makedirs(STUDY_DIR, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        self._print_system_info()
    
    def _print_system_info(self):
        """Print system configuration"""
        if not self.verbose:
            return
        print("\n" + "="*70)
        print("  UNIFIED HIGH-PERFORMANCE AI TRAINER")
        print("="*70)
        print(f"  CPU Cores: {N_CPUS}")
        print(f"  Workers: {self.n_jobs}")
        print(f"  GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"  Optuna: {'Available' if OPTUNA_AVAILABLE else 'Not installed'}")
        print(f"  ONNX: {'Available' if ONNX_AVAILABLE else 'Not installed'}")
        print(f"  Risk Module: {'Available' if RISK_MODULE_AVAILABLE else 'Not installed'}")
        print(f"  Data Loader V2: {DATA_LOADER_V2}")
        print("="*70)
    
    # ==================== DATA LOADING ====================
    
    def load_data(self, data_path: Optional[str] = None) -> Dict:
        """Load and prepare all training data with parallel processing"""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        start_time = time.perf_counter()
        
        if data_path is None:
            data_path = str(PROJECT_ROOT.parent / 'data')
        
        print(f"Data path: {data_path}")
        
        if prepare_training_data is None:
            print("Warning: Data loader not available. Using synthetic data.")
            self._generate_synthetic_data()
        else:
            if DATA_LOADER_V2:
                self.raw_data = prepare_training_data(
                    data_dir=data_path,
                    timeframes=['5m', '15m', '1h', '4h', '1d'],
                    include_derivatives=True
                )
            else:
                self.raw_data = prepare_training_data(
                    data_dir=data_path,
                    timeframes=['5m', '15m', '1h', '4h', '1d']
                )
        
        if not self.raw_data:
            print("Warning: No data loaded. Using synthetic data.")
            self._generate_synthetic_data()
        
        n_symbols = len(self.raw_data)
        load_time = time.perf_counter() - start_time
        print(f"Loaded {n_symbols} symbols in {load_time:.2f}s")
        
        # Print detailed data statistics
        self._print_data_statistics()
        
        # Prepare data for all model types in parallel
        self._prepare_all_data_parallel()
        
        return self.data
    
    def _print_data_statistics(self):
        """Print detailed statistics about loaded data"""
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        
        total_rows = 0
        tf_stats = {}
        
        for symbol, tf_data in self.raw_data.items():
            for tf, df in tf_data.items():
                if tf not in tf_stats:
                    tf_stats[tf] = {'symbols': 0, 'rows': 0, 'cols': 0, 'date_min': None, 'date_max': None}
                
                tf_stats[tf]['symbols'] += 1
                tf_stats[tf]['rows'] += len(df)
                tf_stats[tf]['cols'] = len(df.columns)
                
                if len(df) > 0:
                    df_min = df.index.min()
                    df_max = df.index.max()
                    if tf_stats[tf]['date_min'] is None or df_min < tf_stats[tf]['date_min']:
                        tf_stats[tf]['date_min'] = df_min
                    if tf_stats[tf]['date_max'] is None or df_max > tf_stats[tf]['date_max']:
                        tf_stats[tf]['date_max'] = df_max
                
                total_rows += len(df)
        
        print(f"\n{'Timeframe':<10} {'Symbols':<10} {'Total Rows':<15} {'Columns':<10} {'Date Range'}")
        print("-" * 80)
        
        for tf in sorted(tf_stats.keys()):
            stats = tf_stats[tf]
            date_range = f"{stats['date_min'].strftime('%Y-%m-%d') if stats['date_min'] else 'N/A'} - {stats['date_max'].strftime('%Y-%m-%d') if stats['date_max'] else 'N/A'}"
            print(f"{tf:<10} {stats['symbols']:<10} {stats['rows']:>13,}   {stats['cols']:<10} {date_range}")
        
        print("-" * 80)
        print(f"{'TOTAL':<10} {len(self.raw_data):<10} {total_rows:>13,}")
        print("="*60 + "\n")
        
        return self.data
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing"""
        print("Generating synthetic data...")
        
        np.random.seed(42)
        n_samples = 10000
        
        for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
            self.raw_data[symbol] = {}
            
            for tf in ['5m', '15m', '1h', '4h', '1d']:
                dates = pd.date_range(start='2023-01-01', periods=n_samples, freq=tf)
                
                returns = np.random.normal(0, 0.02, n_samples)
                price = 100 * np.exp(np.cumsum(returns))
                
                df = pd.DataFrame({
                    'open': price * (1 + np.random.uniform(-0.01, 0.01, n_samples)),
                    'high': price * (1 + np.random.uniform(0, 0.02, n_samples)),
                    'low': price * (1 - np.random.uniform(0, 0.02, n_samples)),
                    'close': price,
                    'volume': np.random.exponential(1000000, n_samples),
                }, index=dates)
                
                self.raw_data[symbol][tf] = df
    
    def _prepare_all_data_parallel(self):
        """Prepare model data with parallel processing via joblib"""
        print(f"\nPreparing features with {self.n_jobs} workers...")
        start_time = time.perf_counter()
        
        # Build features for each model type in parallel
        self._prepare_scalp_data_parallel()
        self._prepare_intraday_data_parallel()
        self._prepare_swing_data_parallel()
        self._prepare_risk_data_parallel()
        
        total_time = time.perf_counter() - start_time
        print(f"\nFeatures prepared in {total_time:.2f}s")
    
    def _process_scalp_symbol(self, symbol: str, tf_data: Dict) -> Optional[Dict]:
        """Process single symbol for scalp features (for joblib)"""
        if '5m' not in tf_data:
            return None
        try:
            df = tf_data['5m'].copy()
            df, f_names = build_scalp_features(
                df, horizon=PREDICTION_HORIZON['scalp'],
                threshold=PRICE_MOVE_THRESHOLD['scalp'] / 100
            )
            if len(df) > 100 and 'target' in df.columns:
                X = df[f_names].values
                y = df['target'].values
                valid = ~np.isnan(y)
                return {
                    'X': X[valid].astype(np.float32),
                    'y': y[valid].astype(np.int32),
                    'feature_names': f_names
                }
        except:
            pass
        return None
    
    def _prepare_scalp_data_parallel(self):
        """Build scalp features with parallel processing"""
        print("  Building SCALP features...")
        t0 = time.perf_counter()
        
        # Parallel processing with joblib
        results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=0)(
            delayed(self._process_scalp_symbol)(symbol, tf_data) 
            for symbol, tf_data in self.raw_data.items()
        )
        
        # Aggregate results
        all_features, all_targets = [], []
        feature_names = None
        count = 0
        
        for r in results:
            if r is not None:
                if feature_names is None:
                    feature_names = r['feature_names']
                if len(r['feature_names']) == len(feature_names):
                    all_features.append(r['X'])
                    all_targets.append(r['y'])
                    count += 1
        
        if all_features:
            X = np.vstack(all_features)
            y = np.concatenate(all_targets)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            self.data['scalp'] = {'X': X, 'y': y, 'feature_names': feature_names}
            print(f"    SCALP: {X.shape[0]:,} samples, {X.shape[1]} features ({count} symbols) [{time.perf_counter()-t0:.1f}s]")
    
    def _process_intraday_symbol(self, symbol: str, tf_data: Dict) -> Optional[Dict]:
        """Process single symbol for intraday features (for joblib)"""
        if '1h' not in tf_data:
            return None
        try:
            df = tf_data['1h'].copy()
            df, f_names = build_intraday_features(
                df, horizon=PREDICTION_HORIZON['intraday'],
                threshold=PRICE_MOVE_THRESHOLD['intraday'] / 100
            )
            if len(df) > 50 and 'target' in df.columns:
                X = df[f_names].values
                y = df['target'].values
                valid = ~np.isnan(y)
                return {
                    'X': X[valid].astype(np.float32),
                    'y': y[valid].astype(np.int32),
                    'feature_names': f_names
                }
        except:
            pass
        return None
    
    def _prepare_intraday_data_parallel(self):
        """Build intraday features with parallel processing"""
        print("  Building INTRADAY features...")
        t0 = time.perf_counter()
        
        results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=0)(
            delayed(self._process_intraday_symbol)(symbol, tf_data) 
            for symbol, tf_data in self.raw_data.items()
        )
        
        all_features, all_targets = [], []
        feature_names = None
        count = 0
        
        for r in results:
            if r is not None:
                if feature_names is None:
                    feature_names = r['feature_names']
                if len(r['feature_names']) == len(feature_names):
                    all_features.append(r['X'])
                    all_targets.append(r['y'])
                    count += 1
        
        if all_features:
            X = np.vstack(all_features)
            y = np.concatenate(all_targets)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            self.data['intraday'] = {'X': X, 'y': y, 'feature_names': feature_names}
            print(f"    INTRADAY: {X.shape[0]:,} samples, {X.shape[1]} features ({count} symbols) [{time.perf_counter()-t0:.1f}s]")
    
    def _process_swing_symbol(self, symbol: str, tf_data: Dict) -> Optional[Dict]:
        """Process single symbol for swing features (for joblib)"""
        if '1d' not in tf_data:
            return None
        try:
            df = tf_data['1d'].copy()
            horizon = min(PREDICTION_HORIZON['swing'], max(1, len(df) // 10))
            df, f_names = build_swing_features(
                df, horizon=horizon,
                threshold=PRICE_MOVE_THRESHOLD['swing'] / 100
            )
            if len(df) > 30 and 'target' in df.columns:
                X = df[f_names].values
                y = df['target'].values
                valid = ~np.isnan(y)
                return {
                    'X': X[valid].astype(np.float32),
                    'y': y[valid].astype(np.int32),
                    'feature_names': f_names
                }
        except:
            pass
        return None
    
    def _prepare_swing_data_parallel(self):
        """Build swing features with parallel processing"""
        print("  Building SWING features...")
        t0 = time.perf_counter()
        
        results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=0)(
            delayed(self._process_swing_symbol)(symbol, tf_data) 
            for symbol, tf_data in self.raw_data.items()
        )
        
        all_features, all_targets = [], []
        feature_names = None
        count = 0
        
        for r in results:
            if r is not None:
                if feature_names is None:
                    feature_names = r['feature_names']
                if len(r['feature_names']) == len(feature_names):
                    all_features.append(r['X'])
                    all_targets.append(r['y'])
                    count += 1
        
        if all_features:
            X = np.vstack(all_features)
            y = np.concatenate(all_targets)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            self.data['swing'] = {'X': X, 'y': y, 'feature_names': feature_names}
            print(f"    SWING: {X.shape[0]:,} samples, {X.shape[1]} features ({count} symbols) [{time.perf_counter()-t0:.1f}s]")
    
    def _process_risk_symbol(self, symbol: str, tf_data: Dict, use_advanced: bool) -> Optional[Dict]:
        """Process single symbol for risk features (for joblib)"""
        if '5m' not in tf_data:
            return None
        try:
            df = tf_data['5m'].copy()
            if use_advanced:
                df, f_names = build_advanced_risk_features(df, market_predictions=None, include_derivatives=True)
                target_col = 'risk_target'
            else:
                df, f_names = build_risk_features(df)
                df['future_vol'] = df['close'].pct_change().rolling(20).std().shift(-20)
                target_col = 'future_vol'
            
            if len(df) > 100 and target_col in df.columns:
                X = df[f_names].values
                y = df[target_col].values
                valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
                if np.sum(valid) > 50:
                    return {
                        'X': X[valid].astype(np.float32),
                        'y': y[valid].astype(np.float32),
                        'feature_names': f_names
                    }
        except:
            pass
        return None
    
    def _prepare_risk_data_parallel(self):
        """Build risk features with parallel processing"""
        print("  Building RISK features...")
        t0 = time.perf_counter()
        use_advanced = RISK_MODULE_AVAILABLE
        
        results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=0)(
            delayed(self._process_risk_symbol)(symbol, tf_data, use_advanced) 
            for symbol, tf_data in self.raw_data.items()
        )
        
        all_features, all_targets = [], []
        feature_names = None
        count = 0
        
        for r in results:
            if r is not None:
                if feature_names is None:
                    feature_names = r['feature_names']
                if len(r['feature_names']) == len(feature_names):
                    all_features.append(r['X'])
                    all_targets.append(r['y'])
                    count += 1
        
        if all_features:
            X = np.vstack(all_features)
            y = np.concatenate(all_targets)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            self.data['risk'] = {'X': X, 'y': y, 'feature_names': feature_names}
            print(f"    RISK: {X.shape[0]:,} samples, {X.shape[1]} features ({count} symbols) [{time.perf_counter()-t0:.1f}s]")
    
    # ==================== OPTUNA OPTIMIZATION ====================
    
    def _create_objective(self, model_type: str) -> callable:
        """Create Optuna objective for model type"""
        is_classification = model_type != 'risk'
        data = self.data[model_type]
        X, y = data['X'], data['y']
        feature_names = data['feature_names']
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'num_leaves': trial.suggest_int('num_leaves', 16, 256),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'verbose': -1,
                'n_jobs': self.n_jobs,
                'random_state': RANDOM_STATE,
            }
            
            if is_classification:
                params['objective'] = 'multiclass'
                params['num_class'] = 3
                params['metric'] = 'multi_logloss'
            else:
                params['objective'] = 'regression'
                params['metric'] = 'rmse'
            
            if self.use_gpu:
                params['device'] = 'gpu'
                params['gpu_platform_id'] = 0
                params['gpu_device_id'] = 0
            
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    valid_names=['valid'],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )
                
                if is_classification:
                    y_pred = model.predict(X_val)
                    try:
                        score = log_loss(y_val, y_pred, labels=[0, 1, 2])
                    except ValueError:
                        score = 1.0
                else:
                    y_pred = model.predict(X_val)
                    score = root_mean_squared_error(y_val, y_pred)
                
                scores.append(score)
                
                trial.report(np.mean(scores), fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        return objective
    
    def optimize_with_optuna(self, model_type: str) -> Dict:
        """Run Optuna optimization"""
        if not OPTUNA_AVAILABLE:
            print("Optuna not available! Running fast training instead.")
            return self.train_fast(model_type)
        
        print(f"\n{'='*60}")
        print(f"OPTUNA OPTIMIZATION: {model_type.upper()}")
        print(f"{'='*60}")
        
        if model_type not in self.data:
            print(f"No data available for {model_type}")
            return {}
        
        # Special handling for risk model with multi-objective
        if model_type == 'risk' and RISK_MODULE_AVAILABLE:
            return self._optimize_risk_model()
        
        study_path = os.path.join(STUDY_DIR, f"{model_type}_{self.study_name}.db")
        storage = f"sqlite:///{study_path}"
        
        sampler = TPESampler(seed=RANDOM_STATE)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        
        if self.continue_study and os.path.exists(study_path):
            print(f"Continuing study from: {study_path}")
            study = optuna.load_study(
                study_name=f"{model_type}_study",
                storage=storage,
                sampler=sampler,
                pruner=pruner
            )
        else:
            study = optuna.create_study(
                study_name=f"{model_type}_study",
                storage=storage,
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=self.continue_study
            )
        
        objective = self._create_objective(model_type)
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=1,  # Sequential for stability
            timeout=self.timeout,
            show_progress_bar=True,
            gc_after_trial=True
        )
        
        print(f"\nBest trial:")
        print(f"  Value: {study.best_trial.value:.6f}")
        print(f"  Params: {study.best_trial.params}")
        
        self.best_params[model_type] = study.best_trial.params
        self._save_best_params()
        
        return self._train_with_params(model_type, study.best_trial.params)
    
    def _optimize_risk_model(self) -> Dict:
        """Optimize risk model with multi-objective optimization"""
        print("Running advanced risk model optimization...")
        
        X = self.data['risk']['X']
        y = self.data['risk']['y']
        feature_names = self.data['risk']['feature_names']
        
        results = run_risk_optimization(
            X=X,
            y=y,
            feature_names=feature_names,
            n_trials=self.n_trials,
            multi_objective=True,
            use_gpu=self.use_gpu
        )
        
        self.metrics['risk'] = results.get('metrics', {})
        
        model_path = os.path.join(STUDY_DIR, "risk_model_optimized.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
                self.models['risk'] = saved['model']
        
        return results
    
    # ==================== FAST TRAINING ====================
    
    def train_fast(self, model_type: str) -> Dict:
        """Fast training without Optuna"""
        print(f"\n{'='*60}")
        print(f"FAST TRAINING: {model_type.upper()}")
        print(f"{'='*60}")
        
        if model_type not in self.data:
            print(f"No data available for {model_type}")
            return {}
        
        is_classification = model_type != 'risk'
        params = LGBM_MARKET_PARAMS.copy() if is_classification else LGBM_RISK_PARAMS.copy()
        
        if self.use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
        
        params['n_jobs'] = self.n_jobs
        
        return self._train_with_params(model_type, params)
    
    def train_with_best(self, model_type: str) -> Dict:
        """Train with previously found best parameters"""
        params = self._load_best_params(model_type)
        
        if not params:
            print(f"No saved best params for {model_type}. Running optimization...")
            return self.optimize_with_optuna(model_type)
        
        print(f"\n{'='*60}")
        print(f"TRAINING WITH BEST PARAMS: {model_type.upper()}")
        print(f"{'='*60}")
        
        return self._train_with_params(model_type, params)
    
    def _train_with_params(self, model_type: str, params: Dict) -> Dict:
        """Train model with given parameters - OPTIMIZED for multi-core"""
        start_time = time.perf_counter()
        
        data = self.data[model_type]
        X, y = data['X'], data['y']
        feature_names = data['feature_names']
        
        is_classification = model_type != 'risk'
        
        # Optimized parameters for maximum performance
        full_params = params.copy()
        full_params['verbose'] = -1
        full_params['n_jobs'] = self.n_jobs
        full_params['random_state'] = RANDOM_STATE
        
        # Force histogram-based training for speed
        full_params['force_row_wise'] = False  # Let LightGBM choose
        full_params['num_threads'] = self.n_jobs
        
        if is_classification:
            full_params['objective'] = 'multiclass'
            full_params['num_class'] = 3
            full_params['metric'] = 'multi_logloss'
        else:
            full_params['objective'] = 'regression'
            full_params['metric'] = 'rmse'
        
        n_estimators = full_params.pop('n_estimators', 500)
        
        # Use fewer CV folds for large datasets (speed vs quality tradeoff)
        n_samples = len(X)
        if n_samples > 1_000_000:
            n_folds = 3
            print(f"\nUsing {n_folds} folds for {n_samples:,} samples (speed optimization)")
        else:
            n_folds = 5
        
        tscv = TimeSeriesSplit(n_splits=n_folds)
        cv_scores, cv_acc, best_iterations = [], [], []
        
        print(f"\nCross-validation training ({n_folds} folds, {self.n_jobs} threads)...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            fold_start = time.perf_counter()
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(
                X_train, label=y_train, 
                feature_name=feature_names,
                free_raw_data=True  # Free memory after construction
            )
            val_data = lgb.Dataset(
                X_val, label=y_val, 
                reference=train_data,
                free_raw_data=True
            )
            
            model = lgb.train(
                full_params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0 if not self.verbose else 100)
                ]
            )
            
            if is_classification:
                y_pred_proba = model.predict(X_val)
                y_pred = np.argmax(y_pred_proba, axis=1)
                try:
                    score = log_loss(y_val, y_pred_proba, labels=[0, 1, 2])
                except ValueError:
                    score = 1 - accuracy_score(y_val, y_pred)
                acc = accuracy_score(y_val, y_pred)
                cv_acc.append(acc)
            else:
                y_pred = model.predict(X_val)
                score = root_mean_squared_error(y_val, y_pred)
            
            cv_scores.append(score)
            best_iterations.append(model.best_iteration)
            
            fold_time = time.perf_counter() - fold_start
            print(f"  Fold {fold+1}: score={score:.4f}" + 
                  (f", acc={acc:.4f}" if is_classification else "") +
                  f" ({fold_time:.1f}s)")
        
        print(f"\nTraining final model on all data...")
        
        final_n_estimators = int(np.mean(best_iterations))
        full_data = lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=True)
        final_model = lgb.train(full_params, full_data, num_boost_round=final_n_estimators)
        
        train_time = time.perf_counter() - start_time
        
        metrics = {
            'model_type': model_type,
            'cv_score': float(np.mean(cv_scores)),
            'cv_score_std': float(np.std(cv_scores)),
            'best_iteration': final_n_estimators,
            'n_features': len(feature_names),
            'n_samples': len(X),
            'params': params,
            'train_time': train_time,
            'throughput': len(X) / train_time,
            'trained_at': datetime.now().isoformat()
        }
        
        if is_classification:
            metrics['cv_accuracy'] = float(np.mean(cv_acc))
            metrics['cv_accuracy_std'] = float(np.std(cv_acc))
        
        self.models[model_type] = final_model
        self.feature_names[model_type] = feature_names
        self.metrics[model_type] = metrics
        
        # Save model
        self._save_model(model_type, final_model, feature_names, metrics)
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE: {model_type.upper()}")
        print(f"{'='*60}")
        print(f"  CV Score: {metrics['cv_score']:.4f} ± {metrics['cv_score_std']:.4f}")
        if is_classification:
            print(f"  CV Accuracy: {metrics['cv_accuracy']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
        print(f"  Best Iteration: {final_n_estimators}")
        print(f"  Train Time: {train_time:.2f}s")
        print(f"  Throughput: {metrics['throughput']:,.0f} samples/sec")
        
        return metrics
    
    # ==================== MODEL SAVING ====================
    
    def _save_model(self, model_type: str, model: lgb.Booster, feature_names: List, metrics: Dict):
        """Save model in multiple formats"""
        if model_type == 'risk':
            model_path = RISK_MODEL_PATHS['pkl']
        else:
            model_path = MODEL_PATHS.get(model_type, f"models/{model_type}_lgbm.pkl")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save TXT
        txt_path = model_path.replace('.pkl', '.txt')
        model.save_model(txt_path)
        print(f"  Saved: {txt_path}")
        
        # Save PKL
        save_data = {
            'model': model,
            'feature_names': feature_names,
            'metrics': metrics
        }
        
        if model_type == 'risk':
            save_data.update({
                'cvar_config': CVAR_CONFIG,
                'kelly_config': KELLY_CONFIG,
                'drawdown_config': DRAWDOWN_CONFIG
            })
        
        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"  Saved: {model_path}")
        
        # Save feature names
        feature_path = model_path.replace('.pkl', '_features.json')
        with open(feature_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        # Save ONNX
        if ONNX_AVAILABLE:
            try:
                onnx_path = model_path.replace('.pkl', '.onnx')
                self._save_onnx(model, feature_names, onnx_path, model_type)
                print(f"  Saved: {onnx_path}")
                self._validate_onnx(model, onnx_path, feature_names, model_type)
            except Exception as e:
                print(f"  Warning: ONNX export failed: {e}")
    
    def _save_onnx(self, model: lgb.Booster, feature_names: List, onnx_path: str, model_type: str):
        """Convert and save to ONNX"""
        n_features = len(feature_names)
        is_classification = model_type != 'risk'
        
        initial_type = [('input', FloatTensorType([None, n_features]))]
        
        if is_classification:
            onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=15, zipmap=False)
        else:
            onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=15)
        
        # Add metadata
        meta = onnx_model.metadata_props.add()
        meta.key = "feature_names"
        meta.value = json.dumps(feature_names)
        
        meta = onnx_model.metadata_props.add()
        meta.key = "model_type"
        meta.value = model_type
        
        meta = onnx_model.metadata_props.add()
        meta.key = "created_at"
        meta.value = datetime.now().isoformat()
        
        onnx.save(onnx_model, onnx_path)
    
    def _validate_onnx(self, lgb_model: lgb.Booster, onnx_path: str, feature_names: List, model_type: str):
        """Validate ONNX export"""
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        
        X_test = self.data[model_type]['X'][:100].astype(np.float32)
        
        lgb_pred = lgb_model.predict(X_test)
        onnx_pred = session.run(None, {input_name: X_test})[0]
        
        if model_type != 'risk':
            onnx_pred = onnx_pred
        else:
            onnx_pred = onnx_pred.flatten()
            lgb_pred = lgb_pred.flatten()
        
        if isinstance(lgb_pred, np.ndarray) and lgb_pred.ndim > 1:
            max_diff = np.max(np.abs(lgb_pred - onnx_pred))
        else:
            max_diff = np.max(np.abs(np.array(lgb_pred).flatten() - np.array(onnx_pred).flatten()))
        
        status = "✅ PASSED" if max_diff < 1e-3 else "⚠️ WARNING"
        print(f"    ONNX validation: {status} (max diff: {max_diff:.6f})")
        
        # Latency test
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = session.run(None, {input_name: X_test[:1]})
            times.append((time.perf_counter() - start) * 1000)
        print(f"    ONNX latency: {np.mean(times):.2f}ms (±{np.std(times):.2f}ms)")
    
    def _save_best_params(self):
        """Save best parameters"""
        params_path = os.path.join(STUDY_DIR, BEST_PARAMS_FILE)
        
        all_params = {}
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                all_params = json.load(f)
        
        all_params.update(self.best_params)
        
        with open(params_path, 'w') as f:
            json.dump(all_params, f, indent=2)
    
    def _load_best_params(self, model_type: str) -> Optional[Dict]:
        """Load best parameters"""
        params_path = os.path.join(STUDY_DIR, BEST_PARAMS_FILE)
        
        if not os.path.exists(params_path):
            return None
        
        with open(params_path, 'r') as f:
            all_params = json.load(f)
        
        return all_params.get(model_type)
    
    # ==================== MAIN TRAINING ====================
    
    def train_all(self, mode: str = 'optuna') -> Dict:
        """Train all models"""
        results = {}
        
        if self.model_type == 'all':
            model_types = ['scalp', 'intraday', 'swing', 'risk']
        else:
            model_types = [self.model_type]
        
        for model_type in model_types:
            if model_type not in self.data:
                print(f"\nSkipping {model_type} - no data available")
                continue
            
            if mode == 'optuna':
                results[model_type] = self.optimize_with_optuna(model_type)
            elif mode == 'fast':
                results[model_type] = self.train_fast(model_type)
            elif mode == 'quality':
                results[model_type] = self.train_quality(model_type)
            elif mode == 'best':
                results[model_type] = self.train_with_best(model_type)
        
        return results
    
    def train_quality(self, model_type: str) -> Dict:
        """Quality-focused training with larger trees and more iterations"""
        print(f"\n{'='*60}")
        print(f"QUALITY TRAINING: {model_type.upper()}")
        print(f"{'='*60}")
        
        if model_type not in self.data:
            print(f"No data available for {model_type}")
            return {}
        
        is_classification = model_type != 'risk'
        
        # Quality-optimized parameters
        params = {
            'boosting_type': 'gbdt',
            'num_leaves': 127,           # Larger trees
            'max_depth': 12,             # Deeper
            'learning_rate': 0.02,       # Lower LR for better convergence
            'n_estimators': 2000,        # More iterations
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'feature_fraction_bynode': 0.8,  # Per-node feature sampling
            'path_smooth': 0.1,              # Regularization
        }
        
        if self.use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
        
        params['n_jobs'] = self.n_jobs
        
        return self._train_with_params(model_type, params)
    
    def get_risk_assessment(self) -> Dict:
        """Generate risk assessment report"""
        print("\n" + "="*60)
        print("RISK ASSESSMENT")
        print("="*60)
        
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'metrics': self.metrics,
            'cvar_config': CVAR_CONFIG if RISK_MODULE_AVAILABLE else {},
            'kelly_config': KELLY_CONFIG if RISK_MODULE_AVAILABLE else {},
            'drawdown_config': DRAWDOWN_CONFIG if RISK_MODULE_AVAILABLE else {},
        }
        
        # Save
        assessment_path = "models/training_assessment.json"
        with open(assessment_path, 'w') as f:
            json.dump(assessment, f, indent=2, default=str)
        print(f"Assessment saved: {assessment_path}")
        
        return assessment


# ==================== BENCHMARK ====================

def run_benchmark(data_path: str = None, n_symbols: int = 30):
    """Run performance benchmark"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    trainer = UnifiedTrainer(model_type='intraday', verbose=False)
    
    # Load data
    start = time.perf_counter()
    trainer.load_data(data_path)
    load_time = time.perf_counter() - start
    
    # Quick train
    if 'intraday' in trainer.data:
        X = trainer.data['intraday']['X']
        n_samples = len(X)
        
        start = time.perf_counter()
        trainer.train_fast('intraday')
        train_time = time.perf_counter() - start
        
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"  Data loading: {load_time:.2f}s")
        print(f"  Training: {train_time:.2f}s")
        print(f"  Total: {load_time + train_time:.2f}s")
        print(f"  Samples: {n_samples:,}")
        print(f"  Throughput: {n_samples / train_time:,.0f} samples/sec")


def run_diagnostics():
    """Run system diagnostics"""
    print("\n" + "="*60)
    print("SYSTEM DIAGNOSTICS")
    print("="*60)
    
    print(f"\n[System]")
    print(f"  CPU Cores: {N_CPUS}")
    print(f"  Python: {sys.version.split()[0]}")
    
    print(f"\n[Dependencies]")
    print(f"  NumPy: {np.__version__}")
    print(f"  Pandas: {pd.__version__}")
    print(f"  LightGBM: {lgb.__version__}")
    print(f"  Optuna: {'Available' if OPTUNA_AVAILABLE else 'Not installed'}")
    print(f"  ONNX: {'Available' if ONNX_AVAILABLE else 'Not installed'}")
    print(f"  Risk Module: {'Available' if RISK_MODULE_AVAILABLE else 'Not installed'}")
    
    # Check GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  GPU: Not available")
    except ImportError:
        print(f"  GPU: PyTorch not installed")
    
    print(f"\n[Paths]")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Data: {PROJECT_ROOT.parent / 'data'}")
    print(f"  Models: {PROJECT_ROOT / 'models'}")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description='Unified High-Performance AI Training System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py                        # Full training (all models)
    python train.py --fast                 # Fast training (no Optuna)
    python train.py --quality              # Quality-focused training (more iterations)
    python train.py --trials 100           # Optuna with 100 trials
    python train.py --model scalp          # Train only scalp
    python train.py --model risk           # Train only risk model
    python train.py --gpu                  # Use GPU
    python train.py --best                 # Use saved best params
    python train.py --benchmark            # Run benchmark
    python train.py --diagnose             # System diagnostics
        """
    )
    
    # Mode selection
    parser.add_argument('--fast', action='store_true', help='Fast training without Optuna')
    parser.add_argument('--quality', action='store_true', help='Quality-focused training (more iterations, larger trees)')
    parser.add_argument('--best', action='store_true', help='Train with best saved parameters')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--diagnose', action='store_true', help='Run system diagnostics')
    
    # Optuna settings
    parser.add_argument('--trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--continue-study', action='store_true', help='Continue previous study')
    parser.add_argument('--study-name', type=str, default=None, help='Optuna study name')
    
    # Training settings
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'scalp', 'intraday', 'swing', 'risk'],
                        help='Model to train')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--jobs', type=int, default=-1, help='Parallel jobs')
    
    # Data settings
    parser.add_argument('--data-path', type=str, default=None, help='Data directory')
    
    # Output
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Special modes
    if args.diagnose:
        run_diagnostics()
        return
    
    if args.benchmark:
        run_benchmark(args.data_path)
        return
    
    # Main training
    print("\n" + "="*70)
    print("  AI CRYPTO TRAINER")
    print("="*70)
    print(f"  Time: {datetime.now()}")
    print(f"  Model: {args.model}")
    print(f"  GPU: {args.gpu}")
    
    if args.fast:
        print("  Mode: FAST (no Optuna)")
        mode = 'fast'
    elif args.quality:
        print("  Mode: QUALITY (extended training)")
        mode = 'quality'
    elif args.best:
        print("  Mode: BEST (saved params)")
        mode = 'best'
    else:
        print(f"  Mode: OPTUNA ({args.trials} trials)")
        mode = 'optuna'
    
    if args.timeout:
        print(f"  Timeout: {args.timeout}s")
    print("="*70)
    
    # Initialize trainer
    trainer = UnifiedTrainer(
        model_type=args.model,
        use_gpu=args.gpu,
        n_trials=args.trials,
        n_jobs=args.jobs,
        timeout=args.timeout,
        study_name=args.study_name,
        continue_study=args.continue_study,
        verbose=not args.quiet
    )
    
    # Load data
    trainer.load_data(args.data_path)
    
    # Train
    start_time = time.perf_counter()
    results = trainer.train_all(mode=mode)
    total_time = time.perf_counter() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("  TRAINING SUMMARY")
    print("="*70)
    
    for model_type, metrics in results.items():
        if metrics:
            print(f"\n  {model_type.upper()}:")
            print(f"    CV Score: {metrics.get('cv_score', 'N/A'):.4f}")
            if 'cv_accuracy' in metrics:
                print(f"    CV Accuracy: {metrics['cv_accuracy']:.4f}")
            print(f"    Samples: {metrics.get('n_samples', 'N/A'):,}")
    
    print(f"\n  Total Time: {total_time:.2f}s")
    
    # Risk assessment
    trainer.get_risk_assessment()
    
    print("\n" + "="*70)
    print("  DONE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
