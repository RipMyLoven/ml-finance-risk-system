"""
HIGH-PERFORMANCE AI TRAINING LAUNCHER
======================================
Main entry point for optimized training pipeline.

Usage:
    python run_optimized.py --mode train          # Full training
    python run_optimized.py --mode benchmark      # Benchmark only
    python run_optimized.py --mode tune           # Optuna hyperparameter tuning
    python run_optimized.py --mode diagnose       # System diagnostics

Features:
    - Automatic hardware detection and optimization
    - Parallel data loading with memory mapping
    - Numba-JIT compiled feature engineering
    - LightGBM with full CPU utilization
    - Progress monitoring and profiling
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============== EARLY OPTIMIZATION ==============

def apply_early_optimizations():
    """Apply optimizations before importing heavy libraries"""
    
    # Get CPU count
    import multiprocessing as mp
    n_cpus = mp.cpu_count()
    
    # Set threading environment BEFORE importing numpy/pandas
    os.environ['OMP_NUM_THREADS'] = str(n_cpus)
    os.environ['MKL_NUM_THREADS'] = str(n_cpus)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_cpus)
    os.environ['NUMBA_NUM_THREADS'] = str(n_cpus)
    os.environ['NUMBA_THREADING_LAYER'] = 'omp'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    
    # Disable CUDA if not needed (faster startup)
    if '--no-gpu' in sys.argv:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    return n_cpus

N_CPUS = apply_early_optimizations()

# ============== IMPORTS ==============

import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='High-Performance AI Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_optimized.py --mode train --symbols BTCUSDT ETHUSDT
    python run_optimized.py --mode benchmark --verbose
    python run_optimized.py --mode tune --n-trials 100
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'benchmark', 'tune', 'diagnose'],
        default='train',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='Symbols to train on (default: all)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of Optuna trials for tuning mode'
    )
    
    parser.add_argument(
        '--n-workers',
        type=int,
        default=None,
        help=f'Number of worker processes (default: {N_CPUS})'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable profiling'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PROJECT_ROOT / 'models'),
        help='Output directory for models'
    )
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Execute mode
    if args.mode == 'diagnose':
        run_diagnostics()
    elif args.mode == 'benchmark':
        run_benchmark(args)
    elif args.mode == 'tune':
        run_hyperparameter_tuning(args)
    else:
        run_training(args)


def print_header():
    """Print startup header"""
    print("\n" + "="*70)
    print("  HIGH-PERFORMANCE AI TRAINING PIPELINE")
    print("  " + "="*66)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CPU Cores: {N_CPUS}")
    print("="*70 + "\n")


def run_diagnostics():
    """Run system diagnostics"""
    print("[1/3] Checking dependencies...")
    from training.system_config import check_dependencies
    deps = check_dependencies()
    
    print("\n[2/3] Detecting system...")
    from training.system_config import detect_system, get_optimal_config
    system = detect_system()
    config = get_optimal_config(system)
    
    print("\n[3/3] Printing tuning guide...")
    from training.system_config import print_tuning_guide
    print_tuning_guide()
    
    print("\n✓ Diagnostics complete")


def run_benchmark(args):
    """Run performance benchmark"""
    print("[BENCHMARK MODE]")
    print("-" * 50)
    
    # Import benchmark
    from training.train_optimized import run_benchmark as exec_benchmark
    
    # Run benchmark with data directory
    data_dir = str(PROJECT_ROOT / 'data')
    results = exec_benchmark(data_dir=data_dir, n_symbols=30)
    
    # Results are already printed by exec_benchmark
    print("\n✓ Benchmark complete")


def run_hyperparameter_tuning(args):
    """Run Optuna hyperparameter tuning"""
    print("[HYPERPARAMETER TUNING MODE]")
    print(f"  Trials: {args.n_trials}")
    print("-" * 50)
    
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("ERROR: Optuna not installed. Run: pip install optuna")
        return
    
    # Import optimized trainer
    from training.train_optimized import OptimizedTrainer
    
    # Load data
    print("\n[1/3] Loading data...")
    trainer = OptimizedTrainer(
        n_workers=args.n_workers or N_CPUS,
        verbose=args.verbose
    )
    
    data_dir = PROJECT_ROOT / 'data'
    X, y, meta = trainer.load_and_prepare_data(
        str(data_dir),
        symbols=args.symbols
    )
    
    print(f"  Loaded {len(X):,} samples with {X.shape[1]} features")
    
    # Define objective
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        
        # Run CV
        cv_results = trainer.train_with_cv(
            X, y, meta,
            lgbm_params=params,
            n_folds=5
        )
        
        return cv_results['mean_accuracy']
    
    # Create study
    print("\n[2/3] Running optimization...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=10)
    )
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=min(4, N_CPUS // 4),  # Parallel trials
        show_progress_bar=True
    )
    
    # Print results
    print("\n[3/3] Results")
    print("="*70)
    print(f"  Best accuracy: {study.best_value:.4f}")
    print(f"  Best params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # Save best params
    output_path = Path(args.output_dir) / 'best_params.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': args.n_trials,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n  Saved to: {output_path}")
    print("="*70)


def run_training(args):
    """Run full training pipeline"""
    print("[TRAINING MODE]")
    print("-" * 50)
    
    from training.train_optimized import OptimizedTrainer
    
    # Initialize trainer with correct API
    trainer = OptimizedTrainer(
        model_type='intraday',
        use_gpu=not args.no_gpu,
        n_workers=args.n_workers or N_CPUS
    )
    
    # Load data
    print("\n[1/3] Loading data...")
    start_time = time.time()
    
    data_dir = str(PROJECT_ROOT / 'data')
    trainer.load_data(data_dir, max_symbols=None if args.symbols is None else len(args.symbols))
    
    load_time = time.time() - start_time
    print(f"  ✓ Data loaded in {load_time:.1f}s")
    
    # Build features
    print("\n[2/3] Building features...")
    start_time = time.time()
    
    trainer.prepare_features()
    
    feature_time = time.time() - start_time
    print(f"  ✓ Features built in {feature_time:.1f}s")
    
    # Train with cross-validation (uses trainer.train() which does CV + final model)
    print("\n[3/3] Training with cross-validation...")
    start_time = time.time()
    
    metrics = trainer.train(
        num_boost_round=500,
        n_cv_folds=5
    )
    
    train_time = time.time() - start_time
    
    # Save model
    output_dir = str(Path(args.output_dir))
    trainer.save_model(output_dir)
    
    # Summary
    total_time = load_time + feature_time + train_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  CV Accuracy: {metrics['cv_accuracy']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    print(f"  Best Iteration: {metrics['best_iteration']}")
    print(f"  Samples: {metrics['n_samples']:,}")
    print(f"  Features: {metrics['n_features']}")
    print(f"  Total Time: {total_time:.1f}s")
    print(f"  Throughput: {metrics['n_samples'] / total_time:,.0f} samples/sec")
    print("="*70)


if __name__ == "__main__":
    main()
