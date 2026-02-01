"""
HIGH-PERFORMANCE TRAINING SCRIPT
================================
Maximum CPU/RAM utilization with:
1. Parallel data loading
2. Vectorized feature engineering
3. Optimized LightGBM configuration
4. Multi-process cross-validation
5. Memory-efficient data handling
6. System-level optimizations

Run with:
    python train_optimized.py --fast      # Quick training
    python train_optimized.py --full      # Full optimization
    python train_optimized.py --benchmark # Run benchmarks
"""

import os
import sys
import gc
import json
import pickle
import argparse
import warnings
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Pool, shared_memory
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LGBM_MARKET_PARAMS, MODEL_PATHS, RANDOM_STATE,
    PRICE_MOVE_THRESHOLD, PREDICTION_HORIZON
)
from data.parallel_loader import ParallelDataLoader, preload_all_data
from features.fast_features import build_all_features_fast, build_features_parallel

# ============== SYSTEM CONFIGURATION ==============
NUM_CPUS = cpu_count()
PHYSICAL_CORES = max(1, NUM_CPUS // 2)  # Approximate physical cores
TOTAL_RAM_GB = 16  # Estimate, will be detected

try:
    import psutil
    TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
    AVAILABLE_RAM_GB = psutil.virtual_memory().available / (1024**3)
except ImportError:
    AVAILABLE_RAM_GB = TOTAL_RAM_GB * 0.7

print(f"[System] CPUs: {NUM_CPUS}, RAM: {TOTAL_RAM_GB:.1f}GB, Available: {AVAILABLE_RAM_GB:.1f}GB")


# ============== OPTIMIZED LIGHTGBM PARAMETERS ==============

def get_optimized_lgbm_params(
    n_samples: int,
    n_features: int,
    use_gpu: bool = False,
    objective: str = 'multiclass'
) -> Dict:
    """
    Get LightGBM parameters optimized for hardware.
    
    Tunes parameters based on:
    - Dataset size
    - Available CPU cores
    - Available RAM
    - Whether GPU is available
    """
    params = {
        'objective': objective,
        'boosting_type': 'gbdt',
        'metric': 'multi_logloss' if objective == 'multiclass' else 'rmse',
        
        # Core parameters
        'num_leaves': min(255, max(31, n_samples // 1000)),
        'max_depth': -1,  # Unlimited, controlled by num_leaves
        'learning_rate': 0.05,
        
        # Regularization
        'min_child_samples': max(20, n_samples // 10000),
        'min_child_weight': 1e-3,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        
        # Sampling for speed
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        
        # Performance tuning
        'n_jobs': NUM_CPUS,  # Use ALL cores
        'num_threads': NUM_CPUS,
        'verbose': -1,
        'seed': RANDOM_STATE,
        
        # Memory optimization
        'feature_pre_filter': False,
        'max_bin': 255,
        'bin_construct_sample_cnt': min(n_samples, 500000),
    }
    
    if objective == 'multiclass':
        params['num_class'] = 3
    
    # GPU settings
    if use_gpu:
        params['device'] = 'gpu'
        params['gpu_platform_id'] = 0
        params['gpu_device_id'] = 0
        params['gpu_use_dp'] = False  # Use float32 for speed
    
    # Large dataset optimizations
    if n_samples > 1_000_000:
        params['histogram_pool_size'] = 2048  # MB
        params['force_row_wise'] = True  # Better for large data
    
    return params


# ============== MEMORY-EFFICIENT DATA HANDLING ==============

class MemoryOptimizedDataset:
    """
    Memory-efficient dataset wrapper.
    
    Features:
    - Float32 precision (50% memory reduction)
    - Contiguous memory layout
    - Optional memory mapping
    - Pre-allocated buffers
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        # Ensure contiguous memory and float32
        self.X = np.ascontiguousarray(X.astype(np.float32))
        self.y = np.ascontiguousarray(y.astype(np.int32))
        self.feature_names = feature_names
        
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        
        # Calculate memory usage
        self.memory_mb = (self.X.nbytes + self.y.nbytes) / (1024 * 1024)
    
    def get_fold_data(
        self,
        train_idx: np.ndarray,
        val_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/val split with minimal copying"""
        return (
            self.X[train_idx],
            self.X[val_idx],
            self.y[train_idx],
            self.y[val_idx]
        )
    
    def create_lgb_dataset(
        self,
        indices: np.ndarray = None,
        reference: lgb.Dataset = None
    ) -> lgb.Dataset:
        """Create LightGBM dataset efficiently"""
        if indices is not None:
            X = self.X[indices]
            y = self.y[indices]
        else:
            X = self.X
            y = self.y
        
        return lgb.Dataset(
            X, label=y,
            feature_name=self.feature_names,
            reference=reference,
            free_raw_data=False  # Keep raw data for reuse
        )


# ============== PARALLEL CROSS-VALIDATION ==============

def _train_fold(args: Tuple) -> Dict:
    """Train a single CV fold - designed for parallel execution"""
    fold_idx, train_idx, val_idx, X, y, feature_names, params, num_boost_round = args
    
    try:
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            valid_names=['valid'],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Predictions
        y_pred_proba = model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return {
            'fold': fold_idx,
            'accuracy': accuracy_score(y_val, y_pred),
            'logloss': log_loss(y_val, y_pred_proba, labels=[0, 1, 2]),
            'best_iteration': model.best_iteration,
            'train_size': len(train_idx),
            'val_size': len(val_idx)
        }
        
    except Exception as e:
        return {'fold': fold_idx, 'error': str(e)}


class ParallelCrossValidator:
    """
    Parallel cross-validation for LightGBM.
    
    Runs CV folds in parallel using ProcessPoolExecutor.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_workers: int = None,
        purge_gap: int = 0
    ):
        self.n_splits = n_splits
        self.n_workers = n_workers or min(n_splits, NUM_CPUS // 2)
        self.purge_gap = purge_gap
    
    def cross_validate(
        self,
        dataset: MemoryOptimizedDataset,
        params: Dict,
        num_boost_round: int = 500
    ) -> Dict:
        """
        Run parallel cross-validation.
        
        Returns aggregated metrics across all folds.
        """
        print(f"\nRunning {self.n_splits}-fold CV with {self.n_workers} parallel workers...")
        
        # Generate fold indices
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        folds = list(tscv.split(dataset.X))
        
        # Apply purge gap if specified
        if self.purge_gap > 0:
            purged_folds = []
            for train_idx, val_idx in folds:
                # Remove samples from train that are within purge_gap of val
                max_train = train_idx.max()
                val_start = val_idx.min()
                
                if val_start - max_train < self.purge_gap:
                    cutoff = val_start - self.purge_gap
                    train_idx = train_idx[train_idx < cutoff]
                
                if len(train_idx) > 100:
                    purged_folds.append((train_idx, val_idx))
            
            folds = purged_folds
        
        # Prepare tasks
        tasks = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            tasks.append((
                fold_idx,
                train_idx,
                val_idx,
                dataset.X,
                dataset.y,
                dataset.feature_names,
                params,
                num_boost_round
            ))
        
        # Run in parallel
        start_time = time.time()
        results = []
        
        # For LightGBM, it's often faster to run sequentially due to internal parallelism
        # Only parallelize if we have many more cores than folds
        if self.n_workers > 1 and NUM_CPUS >= self.n_splits * 2:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(_train_fold, task): task[0] for task in tasks}
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            # Sequential execution with full CPU utilization per fold
            for task in tasks:
                results.append(_train_fold(task))
        
        elapsed = time.time() - start_time
        
        # Aggregate results
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'All folds failed'}
        
        metrics = {
            'cv_accuracy': np.mean([r['accuracy'] for r in valid_results]),
            'cv_accuracy_std': np.std([r['accuracy'] for r in valid_results]),
            'cv_logloss': np.mean([r['logloss'] for r in valid_results]),
            'cv_logloss_std': np.std([r['logloss'] for r in valid_results]),
            'best_iteration': int(np.median([r['best_iteration'] for r in valid_results])),
            'n_folds': len(valid_results),
            'cv_time_sec': elapsed,
            'fold_results': valid_results
        }
        
        print(f"CV completed in {elapsed:.2f}s")
        print(f"  Accuracy: {metrics['cv_accuracy']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
        print(f"  LogLoss:  {metrics['cv_logloss']:.4f} ± {metrics['cv_logloss_std']:.4f}")
        
        return metrics


# ============== MAIN TRAINING PIPELINE ==============

class OptimizedTrainer:
    """
    High-performance trainer with full system utilization.
    """
    
    def __init__(
        self,
        model_type: str = 'intraday',
        use_gpu: bool = False,
        n_workers: int = None
    ):
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.n_workers = n_workers or NUM_CPUS
        
        self.data = None
        self.dataset = None
        self.model = None
        self.metrics = {}
    
    def load_data(self, data_dir: str, max_symbols: int = None) -> None:
        """Load data using parallel loader"""
        print("\n" + "="*60)
        print("LOADING DATA (PARALLEL)")
        print("="*60)
        
        start_time = time.time()
        
        # Parallel loading
        loader = ParallelDataLoader(data_dir, n_workers=self.n_workers)
        
        # Determine timeframe based on model type
        timeframe_map = {
            'scalp': '5m',
            'intraday': '1h',
            'swing': '1d'
        }
        timeframe = timeframe_map.get(self.model_type, '1h')
        
        raw_data = loader.load_all_klines_parallel(
            timeframes=[timeframe],
            max_symbols=max_symbols
        )
        
        # Also load derivatives
        deriv_data = loader.load_derivatives_parallel(
            symbols=list(raw_data.keys())
        )
        
        # Merge derivatives
        for symbol in raw_data:
            if symbol in deriv_data:
                df = raw_data[symbol][timeframe]
                for dtype, deriv_df in deriv_data[symbol].items():
                    if deriv_df is not None and not deriv_df.empty:
                        try:
                            df = pd.merge_asof(
                                df.reset_index(),
                                deriv_df.reset_index(),
                                on='timestamp',
                                direction='backward'
                            ).set_index('timestamp')
                        except:
                            continue
                raw_data[symbol][timeframe] = df
        
        self.raw_data = raw_data
        
        elapsed = time.time() - start_time
        print(f"Data loaded in {elapsed:.2f}s")
    
    def prepare_features(self) -> None:
        """Build features using parallel processing"""
        print("\n" + "="*60)
        print("BUILDING FEATURES (PARALLEL)")
        print("="*60)
        
        start_time = time.time()
        
        timeframe_map = {
            'scalp': '5m',
            'intraday': '1h',
            'swing': '1d'
        }
        timeframe = timeframe_map.get(self.model_type, '1h')
        horizon = PREDICTION_HORIZON.get(self.model_type, 6)
        threshold = PRICE_MOVE_THRESHOLD.get(self.model_type, 1.0) / 100
        
        # Extract DataFrames
        symbol_dfs = {
            symbol: data[timeframe]
            for symbol, data in self.raw_data.items()
            if timeframe in data and len(data[timeframe]) > 200
        }
        
        # Build features in parallel
        results = build_features_parallel(
            symbol_dfs,
            horizon=horizon,
            threshold=threshold,
            n_workers=self.n_workers
        )
        
        # Combine all data
        all_X = []
        all_y = []
        feature_names = None
        
        for symbol, (df, f_names) in results.items():
            if feature_names is None:
                feature_names = f_names
            
            if len(f_names) != len(feature_names):
                continue
            
            X = df[feature_names].values
            y = df['target'].values
            
            valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
            all_X.append(X[valid_mask])
            all_y.append(y[valid_mask])
        
        X = np.vstack(all_X).astype(np.float32)
        y = np.concatenate(all_y).astype(np.int32)
        
        # Handle remaining NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create optimized dataset
        self.dataset = MemoryOptimizedDataset(X, y, feature_names)
        
        elapsed = time.time() - start_time
        print(f"Features built in {elapsed:.2f}s")
        print(f"  Samples: {self.dataset.n_samples:,}")
        print(f"  Features: {self.dataset.n_features}")
        print(f"  Memory: {self.dataset.memory_mb:.1f} MB")
        print(f"  Class distribution: {np.bincount(y)}")
    
    def train(
        self,
        num_boost_round: int = 500,
        n_cv_folds: int = 5,
        purge_gap: int = None
    ) -> Dict:
        """Train model with parallel CV"""
        print("\n" + "="*60)
        print(f"TRAINING {self.model_type.upper()} MODEL")
        print("="*60)
        
        if self.dataset is None:
            raise ValueError("No data prepared. Call prepare_features() first.")
        
        # Get optimized parameters
        params = get_optimized_lgbm_params(
            n_samples=self.dataset.n_samples,
            n_features=self.dataset.n_features,
            use_gpu=self.use_gpu,
            objective='multiclass'
        )
        
        print(f"\nLightGBM parameters:")
        print(f"  num_leaves: {params['num_leaves']}")
        print(f"  n_jobs: {params['n_jobs']}")
        print(f"  num_threads: {params['num_threads']}")
        
        # Determine purge gap
        if purge_gap is None:
            purge_gap = PREDICTION_HORIZON.get(self.model_type, 6) + 2
        
        # Parallel cross-validation
        cv = ParallelCrossValidator(
            n_splits=n_cv_folds,
            n_workers=min(n_cv_folds, self.n_workers // 2),
            purge_gap=purge_gap
        )
        
        cv_metrics = cv.cross_validate(
            self.dataset,
            params,
            num_boost_round=num_boost_round
        )
        
        # Train final model on all data
        print("\nTraining final model on all data...")
        start_time = time.time()
        
        full_dataset = self.dataset.create_lgb_dataset()
        
        self.model = lgb.train(
            params,
            full_dataset,
            num_boost_round=cv_metrics['best_iteration']
        )
        
        elapsed = time.time() - start_time
        print(f"Final model trained in {elapsed:.2f}s")
        
        # Compile metrics
        self.metrics = {
            'model_type': self.model_type,
            'cv_accuracy': cv_metrics['cv_accuracy'],
            'cv_accuracy_std': cv_metrics['cv_accuracy_std'],
            'cv_logloss': cv_metrics['cv_logloss'],
            'best_iteration': cv_metrics['best_iteration'],
            'n_samples': self.dataset.n_samples,
            'n_features': self.dataset.n_features,
            'cv_time_sec': cv_metrics['cv_time_sec'],
            'trained_at': datetime.now().isoformat()
        }
        
        return self.metrics
    
    def save_model(self, output_dir: str = 'models') -> None:
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model trained. Call train() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f'{self.model_type}_optimized.pkl')
        txt_path = model_path.replace('.pkl', '.txt')
        
        # Save LightGBM native format
        self.model.save_model(txt_path)
        
        # Save with pickle
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.dataset.feature_names,
                'metrics': self.metrics
            }, f)
        
        print(f"Model saved to {model_path}")


# ============== BENCHMARK UTILITIES ==============

def run_benchmark(data_dir: str, n_symbols: int = 30) -> Dict:
    """
    Run comprehensive performance benchmark.
    """
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    results = {}
    
    # Benchmark data loading
    print("\n[1/3] Data Loading Benchmark...")
    loader = ParallelDataLoader(data_dir)
    
    start = time.time()
    data = loader.load_all_klines_parallel(
        timeframes=['1h'],
        max_symbols=n_symbols
    )
    results['data_loading_sec'] = time.time() - start
    
    # Benchmark feature building
    print("\n[2/3] Feature Building Benchmark...")
    symbol_dfs = {s: d['1h'] for s, d in data.items() if '1h' in d}
    
    start = time.time()
    feature_results = build_features_parallel(
        symbol_dfs,
        horizon=6,
        threshold=0.01,
        n_workers=NUM_CPUS - 1
    )
    results['feature_building_sec'] = time.time() - start
    
    # Benchmark training
    print("\n[3/3] Training Benchmark...")
    
    all_X = []
    all_y = []
    feature_names = None
    
    for symbol, (df, f_names) in feature_results.items():
        if feature_names is None:
            feature_names = f_names
        X = df[f_names].values
        y = df['target'].values
        valid = ~np.isnan(y)
        all_X.append(X[valid])
        all_y.append(y[valid])
    
    X = np.vstack(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.int32)
    X = np.nan_to_num(X)
    
    dataset = MemoryOptimizedDataset(X, y, feature_names)
    
    params = get_optimized_lgbm_params(dataset.n_samples, dataset.n_features)
    
    cv = ParallelCrossValidator(n_splits=3)
    start = time.time()
    cv_metrics = cv.cross_validate(dataset, params, num_boost_round=100)
    results['cv_training_sec'] = time.time() - start
    
    # Summary
    total_time = sum([
        results['data_loading_sec'],
        results['feature_building_sec'],
        results['cv_training_sec']
    ])
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Data Loading:     {results['data_loading_sec']:.2f}s")
    print(f"Feature Building: {results['feature_building_sec']:.2f}s")
    print(f"CV Training:      {results['cv_training_sec']:.2f}s")
    print(f"TOTAL:            {total_time:.2f}s")
    print(f"\nThroughput: {dataset.n_samples / total_time:,.0f} samples/sec")
    
    return results


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(
        description='High-Performance AI Training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default='intraday',
                        choices=['scalp', 'intraday', 'swing'],
                        help='Model type to train')
    parser.add_argument('--fast', action='store_true',
                        help='Fast training with reduced iterations')
    parser.add_argument('--full', action='store_true',
                        help='Full training with all optimizations')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmarks')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    parser.add_argument('--symbols', type=int, default=None,
                        help='Max symbols to load')
    parser.add_argument('--iterations', type=int, default=500,
                        help='Max boosting iterations')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data directory')
    
    args = parser.parse_args()
    
    # Find data directory
    if args.data_path:
        data_dir = args.data_path
    else:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    
    print("\n" + "="*60)
    print("HIGH-PERFORMANCE AI TRAINER")
    print("="*60)
    print(f"Time: {datetime.now()}")
    print(f"Model: {args.model}")
    print(f"Data: {data_dir}")
    print(f"CPUs: {NUM_CPUS}")
    print(f"RAM: {TOTAL_RAM_GB:.1f}GB")
    print(f"GPU: {args.gpu}")
    
    if args.benchmark:
        run_benchmark(data_dir, n_symbols=30)
        return
    
    # Initialize trainer
    trainer = OptimizedTrainer(
        model_type=args.model,
        use_gpu=args.gpu,
        n_workers=args.workers
    )
    
    # Load data
    trainer.load_data(data_dir, max_symbols=args.symbols)
    
    # Build features
    trainer.prepare_features()
    
    # Train
    iterations = 200 if args.fast else args.iterations
    metrics = trainer.train(
        num_boost_round=iterations,
        n_cv_folds=5
    )
    
    # Save
    trainer.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"CV Accuracy: {metrics['cv_accuracy']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    print(f"Best Iteration: {metrics['best_iteration']}")


if __name__ == "__main__":
    # Enable multiprocessing on Windows
    mp.freeze_support()
    main()
