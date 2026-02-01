"""
HIGH-PERFORMANCE DATA LOADER
============================
Optimizations:
1. Parallel CSV loading with ProcessPoolExecutor
2. Memory-mapped file reading for large datasets
3. Shared memory arrays to avoid data copying
4. Async I/O pipeline with prefetching
5. Columnar storage with efficient dtypes
6. Zero-copy data transfers where possible

Performance targets:
- 10-50x faster data loading vs sequential
- 50-70% memory reduction via efficient dtypes
- Full CPU core utilization during loading
"""

import os
import glob
import mmap
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Generator
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, shared_memory, Pool, Manager
import multiprocessing as mp
from functools import partial
import threading
import queue
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
# Detect optimal parallelism
NUM_CPUS = cpu_count()
NUM_WORKERS = max(1, NUM_CPUS - 1)  # Leave 1 core for main thread
CHUNK_SIZE = 100_000  # Rows per chunk for large files
PREFETCH_BUFFER = 4   # Number of files to prefetch
USE_MEMORY_MAP = True
DTYPE_OPTIMIZATION = True

# Optimized dtypes for memory efficiency
OPTIMIZED_DTYPES = {
    'open': np.float32,
    'high': np.float32,
    'low': np.float32,
    'close': np.float32,
    'volume': np.float64,
    'quote_volume': np.float64,
    'trades': np.int32,
    'taker_buy_volume': np.float64,
    'taker_buy_quote_volume': np.float64,
    'funding_rate': np.float32,
    'sum_open_interest': np.float64,
    'sum_open_interest_value': np.float64,
    'long_short_ratio': np.float32,
    'long_account': np.float32,
    'short_account': np.float32,
    'buy_sell_ratio': np.float32,
    'buy_vol': np.float64,
    'sell_vol': np.float64,
    'mark_close': np.float32,
    'premium_close': np.float32,
    'basis': np.float32,
    'buy_ratio': np.float32,
}


@dataclass
class LoadStats:
    """Statistics for data loading performance"""
    files_loaded: int = 0
    total_rows: int = 0
    total_bytes: int = 0
    load_time_sec: float = 0.0
    parse_time_sec: float = 0.0
    merge_time_sec: float = 0.0
    
    @property
    def throughput_mb_sec(self) -> float:
        if self.load_time_sec > 0:
            return (self.total_bytes / 1024 / 1024) / self.load_time_sec
        return 0.0
    
    @property
    def rows_per_sec(self) -> float:
        if self.load_time_sec > 0:
            return self.total_rows / self.load_time_sec
        return 0.0


def _load_csv_optimized(filepath: str, dtype_map: Dict = None) -> Optional[pd.DataFrame]:
    """
    Load a single CSV file with optimizations.
    
    Optimizations:
    - Low-level memory mapping for large files
    - Optimized dtype inference
    - Minimal memory allocation
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        file_size = os.path.getsize(filepath)
        
        # Use memory mapping for large files (>10MB)
        if USE_MEMORY_MAP and file_size > 10 * 1024 * 1024:
            df = pd.read_csv(
                filepath,
                engine='c',  # C parser is faster
                memory_map=True,
                low_memory=False,
                dtype=dtype_map
            )
        else:
            df = pd.read_csv(
                filepath,
                engine='c',
                low_memory=False,
                dtype=dtype_map
            )
        
        return df
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def _process_klines(filepath: str) -> Optional[Tuple[str, pd.DataFrame]]:
    """Process a single klines file - designed for parallel execution"""
    if not os.path.exists(filepath):
        return None
    
    try:
        # Extract symbol and timeframe from filename
        basename = os.path.basename(filepath)
        parts = basename.replace('.csv', '').split('_')
        if len(parts) >= 5:
            symbol = parts[3]
            timeframe = parts[4]
        else:
            return None
        
        # Load with optimized dtypes
        df = pd.read_csv(
            filepath,
            engine='c',
            memory_map=True,
            dtype={
                'open': np.float32,
                'high': np.float32,
                'low': np.float32,
                'close': np.float32,
                'volume': np.float64,
            }
        )
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
        
        # Calculate buy_ratio if columns exist
        if 'taker_buy_volume' in df.columns and 'volume' in df.columns:
            df['buy_ratio'] = (df['taker_buy_volume'] / (df['volume'] + 1e-10)).astype(np.float32)
        
        # Drop unnecessary columns
        drop_cols = ['close_time', 'symbol', 'interval', 'market_type']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        return (f"{symbol}_{timeframe}", df.sort_index())
        
    except Exception as e:
        return None


def _process_derivative_file(args: Tuple[str, str, str]) -> Optional[Tuple[str, str, pd.DataFrame]]:
    """Process derivative data files (funding, OI, etc.)"""
    filepath, symbol, data_type = args
    
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath, engine='c', memory_map=True)
        
        # Timestamp conversion based on data type
        if data_type == 'funding_rate':
            ts_col = 'funding_time' if 'funding_time' in df.columns else 'timestamp'
        else:
            ts_col = 'timestamp'
        
        if ts_col in df.columns:
            df['timestamp'] = pd.to_datetime(df[ts_col], unit='ms')
            df = df.set_index('timestamp')
        
        # Drop metadata columns
        drop_cols = ['symbol', 'market_type', 'period', 'interval', 'funding_time']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # Optimize dtypes
        for col in df.columns:
            if col in OPTIMIZED_DTYPES:
                df[col] = df[col].astype(OPTIMIZED_DTYPES[col])
        
        return (symbol, data_type, df.sort_index())
        
    except Exception as e:
        return None


class ParallelDataLoader:
    """
    High-performance parallel data loader.
    
    Features:
    - Multi-process CSV loading
    - Async prefetching
    - Memory-efficient storage
    - Progress tracking
    """
    
    def __init__(
        self,
        data_dir: str,
        n_workers: int = None,
        prefetch_size: int = PREFETCH_BUFFER
    ):
        self.data_dir = data_dir
        self.n_workers = n_workers or NUM_WORKERS
        self.prefetch_size = prefetch_size
        self.symbols = []
        self.stats = LoadStats()
        self._scan_symbols()
    
    def _scan_symbols(self):
        """Scan available symbols from klines files"""
        pattern = os.path.join(self.data_dir, 'klines_usdt_m_*_1h.csv')
        files = glob.glob(pattern)
        
        symbols = set()
        for f in files:
            basename = os.path.basename(f)
            parts = basename.replace('.csv', '').split('_')
            if len(parts) >= 4:
                symbols.add(parts[3])
        
        self.symbols = sorted(list(symbols))
        print(f"[ParallelLoader] Found {len(self.symbols)} symbols, using {self.n_workers} workers")
    
    def load_all_klines_parallel(
        self,
        timeframes: List[str] = ['1h'],
        symbols: List[str] = None,
        max_symbols: int = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load all klines data in parallel using ProcessPoolExecutor.
        
        Returns: Dict[symbol][timeframe] = DataFrame
        """
        start_time = time.time()
        
        symbols = symbols or self.symbols
        if max_symbols:
            symbols = symbols[:max_symbols]
        
        # Build file list
        file_tasks = []
        for symbol in symbols:
            for tf in timeframes:
                filepath = os.path.join(self.data_dir, f'klines_usdt_m_{symbol}_{tf}.csv')
                if os.path.exists(filepath):
                    file_tasks.append(filepath)
        
        print(f"[ParallelLoader] Loading {len(file_tasks)} klines files...")
        
        # Parallel loading with ProcessPoolExecutor
        results = {}
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(_process_klines, fp): fp for fp in file_tasks}
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    key, df = result
                    symbol, tf = key.rsplit('_', 1)
                    
                    if symbol not in results:
                        results[symbol] = {}
                    results[symbol][tf] = df
                    
                    self.stats.files_loaded += 1
                    self.stats.total_rows += len(df)
                
                completed += 1
                if completed % 50 == 0:
                    print(f"  Loaded {completed}/{len(file_tasks)} files...")
        
        self.stats.load_time_sec = time.time() - start_time
        print(f"[ParallelLoader] Loaded {len(results)} symbols in {self.stats.load_time_sec:.2f}s")
        print(f"  Throughput: {self.stats.rows_per_sec:,.0f} rows/sec")
        
        return results
    
    def load_derivatives_parallel(
        self,
        symbols: List[str] = None,
        data_types: List[str] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load derivative data (funding, OI, LS ratio, etc.) in parallel.
        
        Returns: Dict[symbol][data_type] = DataFrame
        """
        symbols = symbols or self.symbols
        data_types = data_types or ['funding_rate', 'open_interest', 'long_short_ratio', 'taker_volume']
        
        # Build task list
        tasks = []
        for symbol in symbols:
            for dtype in data_types:
                filepath = os.path.join(self.data_dir, f'{dtype}_usdt_m_{symbol}.csv')
                if os.path.exists(filepath):
                    tasks.append((filepath, symbol, dtype))
        
        print(f"[ParallelLoader] Loading {len(tasks)} derivative files...")
        start_time = time.time()
        
        results = {}
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(_process_derivative_file, task): task for task in tasks}
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    symbol, dtype, df = result
                    if symbol not in results:
                        results[symbol] = {}
                    results[symbol][dtype] = df
        
        elapsed = time.time() - start_time
        print(f"[ParallelLoader] Loaded derivatives in {elapsed:.2f}s")
        
        return results
    
    def load_symbol_complete(
        self,
        symbol: str,
        timeframe: str = '1h',
        include_derivatives: bool = True
    ) -> Optional[pd.DataFrame]:
        """Load complete data for a single symbol with all derivatives merged."""
        # Load klines
        klines_path = os.path.join(self.data_dir, f'klines_usdt_m_{symbol}_{timeframe}.csv')
        result = _process_klines(klines_path)
        
        if result is None:
            return None
        
        _, df = result
        
        if not include_derivatives:
            return df
        
        # Load and merge derivatives
        derivative_types = [
            ('funding_rate', ['funding_rate']),
            ('open_interest', ['sum_open_interest', 'sum_open_interest_value']),
            ('long_short_ratio', ['long_short_ratio', 'long_account', 'short_account']),
            ('taker_volume', ['buy_sell_ratio', 'buy_vol', 'sell_vol']),
        ]
        
        for dtype, cols in derivative_types:
            filepath = os.path.join(self.data_dir, f'{dtype}_usdt_m_{symbol}.csv')
            result = _process_derivative_file((filepath, symbol, dtype))
            
            if result is not None:
                _, _, deriv_df = result
                available_cols = [c for c in cols if c in deriv_df.columns]
                if available_cols:
                    df = pd.merge_asof(
                        df.reset_index(),
                        deriv_df[available_cols].reset_index(),
                        on='timestamp',
                        direction='backward'
                    ).set_index('timestamp')
                    
                    for col in available_cols:
                        if col in df.columns:
                            df[col] = df[col].ffill().fillna(0)
        
        return df


class AsyncDataPipeline:
    """
    Asynchronous data loading pipeline with prefetching.
    
    Uses producer-consumer pattern:
    - Producer threads load files in background
    - Consumer processes data as it arrives
    - Prefetch buffer ensures no I/O stalls
    """
    
    def __init__(self, data_dir: str, buffer_size: int = 8, n_workers: int = None):
        self.data_dir = data_dir
        self.buffer_size = buffer_size
        self.n_workers = n_workers or NUM_WORKERS
        self._queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
    
    def _producer(self, file_list: List[str]):
        """Background thread that loads files into queue"""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for filepath in file_list:
                if self._stop_event.is_set():
                    break
                
                future = executor.submit(_process_klines, filepath)
                result = future.result()
                
                if result is not None:
                    self._queue.put(result)
        
        # Signal end of data
        self._queue.put(None)
    
    def stream_data(self, file_list: List[str]) -> Generator[Tuple[str, pd.DataFrame], None, None]:
        """
        Stream data from files with prefetching.
        
        Yields: (key, DataFrame) tuples
        """
        # Start producer thread
        producer = threading.Thread(target=self._producer, args=(file_list,))
        producer.start()
        
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                yield item
        finally:
            self._stop_event.set()
            producer.join()


class SharedMemoryCache:
    """
    Shared memory cache for training data.
    
    Stores preprocessed numpy arrays in shared memory
    to avoid copying between processes.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._shm_blocks = {}
        self._shapes = {}
        self._dtypes = {}
    
    def store(self, key: str, array: np.ndarray) -> None:
        """Store numpy array in shared memory"""
        # Create shared memory block
        shm = shared_memory.SharedMemory(
            create=True,
            size=array.nbytes,
            name=f"{self.name}_{key}"
        )
        
        # Copy data to shared memory
        shared_array = np.ndarray(
            array.shape,
            dtype=array.dtype,
            buffer=shm.buf
        )
        shared_array[:] = array[:]
        
        self._shm_blocks[key] = shm
        self._shapes[key] = array.shape
        self._dtypes[key] = array.dtype
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get numpy array from shared memory (zero-copy)"""
        if key not in self._shm_blocks:
            return None
        
        shm = self._shm_blocks[key]
        return np.ndarray(
            self._shapes[key],
            dtype=self._dtypes[key],
            buffer=shm.buf
        )
    
    def close(self):
        """Close all shared memory blocks"""
        for shm in self._shm_blocks.values():
            shm.close()
    
    def unlink(self):
        """Unlink (delete) all shared memory blocks"""
        for shm in self._shm_blocks.values():
            try:
                shm.unlink()
            except:
                pass


def preload_all_data(
    data_dir: str,
    symbols: List[str] = None,
    timeframes: List[str] = ['1h'],
    include_derivatives: bool = True,
    n_workers: int = None
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    High-level function to preload all training data.
    
    Uses parallel loading and returns organized data structure.
    """
    loader = ParallelDataLoader(data_dir, n_workers=n_workers)
    
    # Load klines
    klines_data = loader.load_all_klines_parallel(
        timeframes=timeframes,
        symbols=symbols
    )
    
    if include_derivatives:
        # Load derivatives
        deriv_data = loader.load_derivatives_parallel(
            symbols=list(klines_data.keys())
        )
        
        # Merge derivatives into klines
        print("[ParallelLoader] Merging derivative data...")
        for symbol in klines_data:
            if symbol not in deriv_data:
                continue
            
            for tf in klines_data[symbol]:
                df = klines_data[symbol][tf]
                
                # Merge each derivative type
                for dtype, deriv_df in deriv_data.get(symbol, {}).items():
                    if deriv_df is None or deriv_df.empty:
                        continue
                    
                    try:
                        df = pd.merge_asof(
                            df.reset_index(),
                            deriv_df.reset_index(),
                            on='timestamp',
                            direction='backward'
                        ).set_index('timestamp')
                    except Exception:
                        continue
                
                klines_data[symbol][tf] = df
    
    return klines_data


# ============== BENCHMARK UTILITIES ==============

def benchmark_loader(data_dir: str, n_symbols: int = 20) -> Dict[str, float]:
    """
    Benchmark data loading performance.
    
    Returns timing comparison between sequential and parallel loading.
    """
    results = {}
    
    # Get symbol list
    loader = ParallelDataLoader(data_dir)
    symbols = loader.symbols[:n_symbols]
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Loading {n_symbols} symbols")
    print(f"{'='*60}")
    
    # Parallel loading
    start = time.time()
    parallel_data = loader.load_all_klines_parallel(
        timeframes=['1h'],
        symbols=symbols
    )
    results['parallel_sec'] = time.time() - start
    
    # Sequential loading (for comparison)
    print("\nSequential loading for comparison...")
    start = time.time()
    sequential_data = {}
    for symbol in symbols:
        filepath = os.path.join(data_dir, f'klines_usdt_m_{symbol}_1h.csv')
        result = _process_klines(filepath)
        if result:
            key, df = result
            sequential_data[symbol] = {'1h': df}
    results['sequential_sec'] = time.time() - start
    
    # Calculate speedup
    results['speedup'] = results['sequential_sec'] / results['parallel_sec']
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Sequential: {results['sequential_sec']:.2f}s")
    print(f"Parallel:   {results['parallel_sec']:.2f}s")
    print(f"Speedup:    {results['speedup']:.2f}x")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Find data directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    
    print(f"Data directory: {data_dir}")
    print(f"CPU cores: {NUM_CPUS}")
    print(f"Workers: {NUM_WORKERS}")
    
    # Run benchmark
    benchmark_loader(data_dir, n_symbols=30)
