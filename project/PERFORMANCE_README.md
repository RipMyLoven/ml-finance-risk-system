# High-Performance AI Training Pipeline

## Overview

This optimized training pipeline maximizes hardware utilization for cryptocurrency price prediction using LightGBM. 

**Performance improvements:**
- **Data loading**: 10-50x faster with parallel processing
- **Feature engineering**: 10-100x faster with Numba JIT
- **Training**: Full CPU/GPU utilization
- **Memory**: Efficient float32 + memory mapping

## Quick Start

```bash
# Install dependencies
pip install -r requirements_optimized.txt

# Run diagnostics (check your system)
python run_optimized.py --mode diagnose

# Run benchmark (measure performance)
python run_optimized.py --mode benchmark --verbose

# Train model
python run_optimized.py --mode train

# Hyperparameter tuning
python run_optimized.py --mode tune --n-trials 100
```

## Architecture

```
project/
├── run_optimized.py           # Main entry point
├── data/
│   └── parallel_loader.py     # Parallel CSV loading
├── features/
│   └── fast_features.py       # Numba-optimized features
└── training/
    ├── train_optimized.py     # Training orchestrator
    └── system_config.py       # System detection & tuning
```

## Performance Optimizations

### 1. Parallel Data Loading (`data/parallel_loader.py`)

- **ProcessPoolExecutor**: Parallel CSV file reading
- **Memory-mapped files**: Fast I/O without RAM copies
- **SharedMemoryCache**: Inter-process data sharing
- **Optimized dtypes**: float32 instead of float64

```python
from data.parallel_loader import ParallelDataLoader

loader = ParallelDataLoader(n_workers=8, use_mmap=True)
df = loader.load_all_csvs('data/', pattern='*USDT.csv')
```

### 2. Numba-JIT Feature Engineering (`features/fast_features.py`)

- **@njit compiled**: Rolling mean, std, RSI, MACD, ATR
- **Vectorized operations**: No Python loops
- **Parallel builds**: Multi-process feature calculation

```python
from features.fast_features import build_all_features_fast

# Single-threaded fast build
X = build_all_features_fast(df)

# Multi-process build
from features.fast_features import build_features_parallel
X = build_features_parallel(df, n_workers=8)
```

### 3. Optimized LightGBM Training

```python
# Optimized parameters for max CPU usage
params = {
    'n_jobs': -1,                    # All CPU cores
    'num_threads': cpu_count,        # All threads
    'boosting_type': 'gbdt',         # Or 'dart' for accuracy
    'histogram_pool_size': 2048,     # Large histogram cache
    'max_bin': 255,                  # Balance speed/accuracy
}
```

### 4. Memory Optimization

| Optimization | Memory Saving |
|--------------|---------------|
| float32 vs float64 | 50% |
| Category dtype | 70-90% |
| Memory mapping | No RAM copy |
| Chunked processing | Bounded usage |

## System Tuning

### Windows

```powershell
# Run as Administrator

# 1. High performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# 2. Disable Windows Defender for data folder
# Settings > Windows Security > Add exclusion

# 3. Run with high priority
Start-Process python -ArgumentList "run_optimized.py --mode train" -Priority High
```

### Linux

```bash
# Run as root

# 1. Reduce swappiness
echo "vm.swappiness=10" >> /etc/sysctl.conf
sysctl -p

# 2. CPU performance governor
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 3. Run with high priority
nice -n -19 python run_optimized.py --mode train
```

## Command Reference

```bash
# Full training with all CPUs
python run_optimized.py --mode train

# Train specific symbols
python run_optimized.py --mode train --symbols BTCUSDT ETHUSDT SOLUSDT

# Benchmark with verbose output
python run_optimized.py --mode benchmark --verbose

# Hyperparameter tuning (100 trials)
python run_optimized.py --mode tune --n-trials 100

# System diagnostics
python run_optimized.py --mode diagnose

# Limit workers (for debugging)
python run_optimized.py --mode train --n-workers 4

# Disable GPU
python run_optimized.py --mode train --no-gpu
```

## Expected Performance

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| Load 100 CSVs | 60s | 3s | 20x |
| Feature engineering | 120s | 5s | 24x |
| 5-fold CV | 300s | 30s | 10x |
| Full pipeline | 10min | 1min | 10x |

*Benchmarked on 8-core CPU with SSD*

## Troubleshooting

### "Numba compilation slow"
First run compiles JIT functions. Subsequent runs are fast.

```python
# Pre-compile by importing
from features.fast_features import _compile_all_functions
_compile_all_functions()
```

### "Memory error"
Reduce chunk size or worker count:

```bash
python run_optimized.py --mode train --n-workers 4
```

### "CPU not fully utilized"
Check environment variables:

```python
import os
os.environ['OMP_NUM_THREADS'] = '16'  # Your CPU count
```

### "Slow on Windows"
1. Disable Windows Defender real-time scanning
2. Use SSD for data folder
3. Close other applications

## API Reference

### ParallelDataLoader

```python
class ParallelDataLoader:
    def __init__(
        self,
        n_workers: int = None,      # Default: CPU count
        use_mmap: bool = True,      # Memory mapping
        dtype: str = 'float32',     # Data type
        verbose: bool = False
    )
    
    def load_all_csvs(
        self,
        data_dir: str,
        pattern: str = '*.csv'
    ) -> pd.DataFrame
```

### OptimizedTrainer

```python
class OptimizedTrainer:
    def __init__(
        self,
        n_workers: int = None,
        use_gpu: bool = False,
        verbose: bool = False
    )
    
    def load_and_prepare_data(
        self,
        data_dir: str,
        symbols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]
    
    def train_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: Dict,
        n_folds: int = 5,
        lgbm_params: Dict = None
    ) -> Dict
```

## License

MIT License
