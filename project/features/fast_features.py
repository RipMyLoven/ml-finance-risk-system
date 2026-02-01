"""
HIGH-PERFORMANCE FEATURE ENGINEERING
====================================
Optimizations:
1. Numba JIT compilation for rolling calculations
2. Vectorized operations (no Python loops)
3. In-place array operations
4. Parallel feature computation
5. Memory-efficient buffer reuse
6. SIMD-optimized NumPy operations

Performance targets:
- 10-100x faster feature computation
- Minimal memory allocation
- Full CPU vectorization
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import warnings

warnings.filterwarnings('ignore')

# Try importing Numba for JIT compilation
try:
    from numba import jit, prange, float32, float64, int32
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not installed. Install with: pip install numba")
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ============== NUMBA-OPTIMIZED ROLLING FUNCTIONS ==============

@jit(nopython=True, parallel=True, cache=True)
def rolling_mean_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Ultra-fast rolling mean using Numba"""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:window-1] = np.nan
    
    # Use parallel loop
    for i in prange(window-1, n):
        result[i] = np.mean(arr[i-window+1:i+1])
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def rolling_std_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Ultra-fast rolling std using Numba"""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:window-1] = np.nan
    
    for i in prange(window-1, n):
        result[i] = np.std(arr[i-window+1:i+1])
    
    return result


@jit(nopython=True, cache=True)
def rolling_sum_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Ultra-fast rolling sum with O(n) complexity"""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:window-1] = np.nan
    
    # Initial sum
    window_sum = 0.0
    for i in range(window):
        window_sum += arr[i]
    result[window-1] = window_sum
    
    # Sliding window - O(n) instead of O(n*window)
    for i in range(window, n):
        window_sum = window_sum - arr[i-window] + arr[i]
        result[i] = window_sum
    
    return result


@jit(nopython=True, cache=True)
def ema_numba(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average using Numba"""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    
    alpha = 2.0 / (span + 1)
    result[0] = arr[0]
    
    for i in range(1, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    
    return result


@jit(nopython=True, cache=True)
def rsi_numba(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI calculation optimized with Numba"""
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:period] = np.nan
    
    # Calculate price changes
    deltas = np.empty(n, dtype=np.float64)
    deltas[0] = 0.0
    for i in range(1, n):
        deltas[i] = close[i] - close[i-1]
    
    # Separate gains and losses
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if deltas[i] > 0:
            gains[i] = deltas[i]
        else:
            losses[i] = -deltas[i]
    
    # Calculate average gain/loss using EMA
    avg_gain = np.empty(n, dtype=np.float64)
    avg_loss = np.empty(n, dtype=np.float64)
    
    # First average
    avg_gain[period-1] = np.mean(gains[:period])
    avg_loss[period-1] = np.mean(losses[:period])
    
    # Smoothed averages
    alpha = 1.0 / period
    for i in range(period, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    # Calculate RSI
    for i in range(period-1, n):
        if avg_loss[i] == 0:
            result[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


@jit(nopython=True, cache=True)
def macd_numba(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD calculation optimized with Numba"""
    ema_fast = ema_numba(close, fast)
    ema_slow = ema_numba(close, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema_numba(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


@jit(nopython=True, cache=True)
def atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range optimized with Numba"""
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # ATR as EMA of TR
    atr = ema_numba(tr, period)
    
    return atr


@jit(nopython=True, cache=True)
def bollinger_bands_numba(close: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands optimized with Numba"""
    n = len(close)
    
    middle = rolling_mean_numba(close, window)
    std = rolling_std_numba(close, window)
    
    upper = np.empty(n, dtype=np.float64)
    lower = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        upper[i] = middle[i] + num_std * std[i]
        lower[i] = middle[i] - num_std * std[i]
    
    return upper, middle, lower


@jit(nopython=True, parallel=True, cache=True)
def trend_slope_numba(close: np.ndarray, window: int) -> np.ndarray:
    """Linear regression slope optimized with Numba"""
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:window-1] = 0.0
    
    # Pre-compute x values
    x = np.arange(window, dtype=np.float64)
    x_mean = np.mean(x)
    x_var = np.sum((x - x_mean) ** 2)
    
    for i in prange(window-1, n):
        y = close[i-window+1:i+1]
        y_mean = np.mean(y)
        
        # Slope formula
        numerator = 0.0
        for j in range(window):
            numerator += (x[j] - x_mean) * (y[j] - y_mean)
        
        slope = numerator / x_var
        # Normalize by current price
        result[i] = slope / close[i]
    
    return result


@jit(nopython=True, cache=True)
def rolling_corr_numba(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Rolling correlation optimized with Numba"""
    n = len(x)
    result = np.empty(n, dtype=np.float64)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        x_win = x[i-window+1:i+1]
        y_win = y[i-window+1:i+1]
        
        x_mean = np.mean(x_win)
        y_mean = np.mean(y_win)
        
        numerator = 0.0
        x_var = 0.0
        y_var = 0.0
        
        for j in range(window):
            x_diff = x_win[j] - x_mean
            y_diff = y_win[j] - y_mean
            numerator += x_diff * y_diff
            x_var += x_diff ** 2
            y_var += y_diff ** 2
        
        denom = np.sqrt(x_var * y_var)
        if denom > 1e-10:
            result[i] = numerator / denom
        else:
            result[i] = 0.0
    
    return result


# ============== VECTORIZED FEATURE BUILDERS ==============

def build_trend_features_vectorized(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Build trend features using vectorized operations.
    
    All calculations use NumPy vectorization or Numba JIT.
    """
    if not inplace:
        df = df.copy()
    
    close = df['close'].values.astype(np.float64)
    n = len(close)
    
    # SMA (vectorized via rolling)
    df['sma_10'] = rolling_mean_numba(close, 10) if NUMBA_AVAILABLE else df['close'].rolling(10).mean()
    df['sma_20'] = rolling_mean_numba(close, 20) if NUMBA_AVAILABLE else df['close'].rolling(20).mean()
    df['sma_50'] = rolling_mean_numba(close, 50) if NUMBA_AVAILABLE else df['close'].rolling(50).mean()
    
    # Price vs SMA (vectorized division)
    sma_10 = df['sma_10'].values
    sma_20 = df['sma_20'].values
    sma_50 = df['sma_50'].values
    
    df['price_vs_sma10'] = (close - sma_10) / (sma_10 + 1e-10)
    df['price_vs_sma20'] = (close - sma_20) / (sma_20 + 1e-10)
    df['price_vs_sma50'] = (close - sma_50) / (sma_50 + 1e-10)
    
    # SMA crossovers (vectorized)
    df['sma_cross_10_20'] = sma_10 / (sma_20 + 1e-10) - 1
    df['sma_cross_10_50'] = sma_10 / (sma_50 + 1e-10) - 1
    df['sma_cross_20_50'] = sma_20 / (sma_50 + 1e-10) - 1
    
    # Trend slope (Numba optimized)
    if NUMBA_AVAILABLE:
        df['trend_slope_10'] = trend_slope_numba(close, 10)
        df['trend_slope_20'] = trend_slope_numba(close, 20)
    else:
        # Fallback to pandas (slower)
        for window in [10, 20]:
            slopes = []
            for i in range(n):
                if i < window:
                    slopes.append(0)
                else:
                    y = close[i-window:i]
                    x = np.arange(window)
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope / close[i])
            df[f'trend_slope_{window}'] = slopes
    
    # EMA trend (Numba or pandas)
    if NUMBA_AVAILABLE:
        ema_12 = ema_numba(close, 12)
        ema_26 = ema_numba(close, 26)
    else:
        ema_12 = df['close'].ewm(span=12).mean().values
        ema_26 = df['close'].ewm(span=26).mean().values
    
    df['ema_trend'] = (ema_12 - ema_26) / (ema_26 + 1e-10)
    df['ema_trend_slope'] = np.concatenate([[0]*5, np.diff(df['ema_trend'].values, n=5)])
    
    return df


def build_volatility_features_vectorized(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """Build volatility features using vectorized operations."""
    if not inplace:
        df = df.copy()
    
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    
    # Returns (vectorized)
    returns = np.empty_like(close)
    returns[0] = 0.0
    returns[1:] = np.diff(close) / close[:-1]
    
    # Historical volatility (Numba)
    if NUMBA_AVAILABLE:
        df['volatility_10'] = rolling_std_numba(returns, 10)
        df['volatility_20'] = rolling_std_numba(returns, 20)
        df['volatility_50'] = rolling_std_numba(returns, 50)
    else:
        df['volatility_10'] = pd.Series(returns).rolling(10).std().values
        df['volatility_20'] = pd.Series(returns).rolling(20).std().values
        df['volatility_50'] = pd.Series(returns).rolling(50).std().values
    
    # Volatility ratio (vectorized)
    vol_10 = df['volatility_10'].values
    vol_50 = df['volatility_50'].values
    df['vol_ratio_10_50'] = vol_10 / (vol_50 + 1e-10)
    
    # ATR (Numba)
    if NUMBA_AVAILABLE:
        atr = atr_numba(high, low, close, 14)
    else:
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        atr = pd.Series(tr).rolling(14).mean().values
    
    df['atr_14'] = atr
    df['atr_norm'] = atr / (close + 1e-10)
    
    # Bollinger Bands (Numba)
    if NUMBA_AVAILABLE:
        bb_upper, bb_middle, bb_lower = bollinger_bands_numba(close, 20, 2.0)
    else:
        bb_middle = pd.Series(close).rolling(20).mean().values
        bb_std = pd.Series(close).rolling(20).std().values
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
    
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    return df


def build_momentum_features_vectorized(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """Build momentum features using vectorized operations."""
    if not inplace:
        df = df.copy()
    
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    
    # RSI (Numba)
    if NUMBA_AVAILABLE:
        rsi = rsi_numba(close, 14)
    else:
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.values
    
    df['rsi_14'] = rsi / 100 - 0.5  # Normalize to [-0.5, 0.5]
    
    # MACD (Numba)
    if NUMBA_AVAILABLE:
        macd_line, signal_line, histogram = macd_numba(close, 12, 26, 9)
    else:
        ema_12 = pd.Series(close).ewm(span=12).mean()
        ema_26 = pd.Series(close).ewm(span=26).mean()
        macd_line = (ema_12 - ema_26).values
        signal_line = pd.Series(macd_line).ewm(span=9).mean().values
        histogram = macd_line - signal_line
    
    df['macd_norm'] = macd_line / (close + 1e-10)
    df['macd_signal'] = histogram / (close + 1e-10)
    df['macd_hist_slope'] = np.concatenate([[0]*3, np.diff(df['macd_signal'].values, n=3)])
    
    # ADX (simplified vectorized version)
    high_diff = np.diff(high, prepend=high[0])
    low_diff = -np.diff(low, prepend=low[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    
    if NUMBA_AVAILABLE:
        atr = rolling_mean_numba(tr, 14)
        plus_di = 100 * rolling_mean_numba(plus_dm, 14) / (atr + 1e-10)
        minus_di = 100 * rolling_mean_numba(minus_dm, 14) / (atr + 1e-10)
    else:
        atr = pd.Series(tr).rolling(14).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean().values / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    if NUMBA_AVAILABLE:
        adx = rolling_mean_numba(dx, 14)
    else:
        adx = pd.Series(dx).rolling(14).mean().values
    
    df['adx'] = adx / 100  # Normalize to [0, 1]
    df['di_diff'] = (plus_di - minus_di) / 100
    
    return df


def build_volume_features_vectorized(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """Build volume features using vectorized operations."""
    if not inplace:
        df = df.copy()
    
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    
    # Volume ratios (vectorized)
    if NUMBA_AVAILABLE:
        vol_sma_10 = rolling_mean_numba(volume, 10)
        vol_sma_20 = rolling_mean_numba(volume, 20)
    else:
        vol_sma_10 = pd.Series(volume).rolling(10).mean().values
        vol_sma_20 = pd.Series(volume).rolling(20).mean().values
    
    df['volume_ratio'] = volume / (vol_sma_20 + 1e-10)
    df['volume_trend'] = vol_sma_10 / (vol_sma_20 + 1e-10) - 1
    
    # OBV (vectorized)
    price_direction = np.sign(np.diff(close, prepend=close[0]))
    obv = np.cumsum(price_direction * volume)
    
    if NUMBA_AVAILABLE:
        obv_sma = rolling_mean_numba(obv, 10)
    else:
        obv_sma = pd.Series(obv).rolling(10).mean().values
    
    df['obv_slope'] = np.diff(obv, n=10, prepend=[0]*10) / (obv_sma + 1e-10)
    
    # MFI (Money Flow Index)
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    tp_diff = np.diff(typical_price, prepend=typical_price[0])
    positive_flow = np.where(tp_diff > 0, money_flow, 0)
    negative_flow = np.where(tp_diff < 0, money_flow, 0)
    
    if NUMBA_AVAILABLE:
        pos_sum = rolling_sum_numba(positive_flow, 14)
        neg_sum = rolling_sum_numba(negative_flow, 14)
    else:
        pos_sum = pd.Series(positive_flow).rolling(14).sum().values
        neg_sum = pd.Series(negative_flow).rolling(14).sum().values
    
    df['mfi'] = (100 - 100 / (1 + pos_sum / (neg_sum + 1e-10))) / 100 - 0.5
    
    # VWAP
    tp_vol = typical_price * volume
    
    if NUMBA_AVAILABLE:
        vwap_10 = rolling_sum_numba(tp_vol, 10) / rolling_sum_numba(volume, 10)
        vwap_20 = rolling_sum_numba(tp_vol, 20) / rolling_sum_numba(volume, 20)
    else:
        vwap_10 = pd.Series(tp_vol).rolling(10).sum().values / pd.Series(volume).rolling(10).sum().values
        vwap_20 = pd.Series(tp_vol).rolling(20).sum().values / pd.Series(volume).rolling(20).sum().values
    
    df['vwap_dev_10'] = (close - vwap_10) / (vwap_10 + 1e-10)
    df['vwap_dev_20'] = (close - vwap_20) / (vwap_20 + 1e-10)
    
    # VWAP position
    if NUMBA_AVAILABLE:
        std_20 = rolling_std_numba(close, 20)
    else:
        std_20 = pd.Series(close).rolling(20).std().values
    
    vwap_upper = vwap_20 + 2 * std_20
    vwap_lower = vwap_20 - 2 * std_20
    df['vwap_position'] = (close - vwap_lower) / (vwap_upper - vwap_lower + 1e-10)
    
    return df


def build_all_features_fast(
    df: pd.DataFrame,
    horizon: int = 6,
    threshold: float = 0.01,
    inplace: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build all features using optimized vectorized operations.
    
    This is the main entry point for feature engineering.
    """
    if not inplace:
        df = df.copy()
    
    initial_len = len(df)
    
    # Build features (all vectorized)
    df = build_trend_features_vectorized(df, inplace=True)
    df = build_volatility_features_vectorized(df, inplace=True)
    df = build_momentum_features_vectorized(df, inplace=True)
    df = build_volume_features_vectorized(df, inplace=True)
    
    # Add funding features if available
    if 'funding_rate' in df.columns and df['funding_rate'].notna().any():
        funding = df['funding_rate'].values.astype(np.float64)
        df['funding_change'] = np.concatenate([[0], np.diff(funding)]) * 1000
        df['funding_change_3'] = np.concatenate([[0]*3, np.diff(funding, n=3)]) * 1000
        
        if NUMBA_AVAILABLE:
            df['funding_sma'] = rolling_mean_numba(funding, 8) * 1000
        else:
            df['funding_sma'] = df['funding_rate'].rolling(8).mean() * 1000
        
        df['funding_vs_sma'] = (funding - df['funding_sma'].values / 1000) * 1000
    else:
        df['funding_change'] = 0
        df['funding_change_3'] = 0
        df['funding_sma'] = 0
        df['funding_vs_sma'] = 0
    
    # Add target
    close = df['close'].values.astype(np.float64)
    future_return = np.concatenate([
        (close[horizon:] / close[:-horizon]) - 1,
        [np.nan] * horizon
    ])
    
    target = np.ones(len(df), dtype=np.int32)  # default flat
    target[future_return > threshold] = 2   # up
    target[future_return < -threshold] = 0  # down
    df['target'] = target
    
    # Define feature names (explicit list - no auto-detection)
    feature_names = [
        # Trend
        'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50',
        'sma_cross_10_20', 'sma_cross_10_50', 'sma_cross_20_50',
        'trend_slope_10', 'trend_slope_20', 'ema_trend', 'ema_trend_slope',
        # Volatility
        'volatility_10', 'volatility_20', 'volatility_50', 'vol_ratio_10_50',
        'atr_14', 'atr_norm', 'bb_width', 'bb_position',
        # Momentum
        'rsi_14', 'macd_norm', 'macd_signal', 'macd_hist_slope', 'adx', 'di_diff',
        # Volume
        'volume_ratio', 'volume_trend', 'obv_slope', 'mfi',
        'vwap_dev_10', 'vwap_dev_20', 'vwap_position',
        # Funding
        'funding_change', 'funding_change_3', 'funding_sma', 'funding_vs_sma'
    ]
    
    # Filter to existing columns
    feature_names = [f for f in feature_names if f in df.columns]
    
    # Handle NaN/Inf (vectorized)
    for col in feature_names:
        arr = df[col].values
        arr = np.where(np.isinf(arr), np.nan, arr)
        df[col] = pd.Series(arr).ffill().bfill().fillna(0).values
    
    # Drop rows where target is NaN
    valid_mask = df['target'].notna()
    df = df[valid_mask]
    
    final_len = len(df)
    print(f"Features built: {len(feature_names)}, samples: {final_len}/{initial_len}")
    
    return df, feature_names


# ============== PARALLEL FEATURE BUILDING ==============

def _build_features_for_symbol(args: Tuple[str, pd.DataFrame, int, float]) -> Optional[Tuple[str, pd.DataFrame, List[str]]]:
    """Build features for a single symbol - designed for parallel execution"""
    symbol, df, horizon, threshold = args
    
    try:
        df, feature_names = build_all_features_fast(df, horizon, threshold)
        return (symbol, df, feature_names)
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None


def build_features_parallel(
    data: Dict[str, pd.DataFrame],
    horizon: int = 6,
    threshold: float = 0.01,
    n_workers: int = None
) -> Dict[str, Tuple[pd.DataFrame, List[str]]]:
    """
    Build features for all symbols in parallel.
    
    Args:
        data: Dict[symbol] = DataFrame
        horizon: Prediction horizon
        threshold: Price move threshold
        n_workers: Number of parallel workers
        
    Returns:
        Dict[symbol] = (DataFrame with features, feature_names)
    """
    from multiprocessing import cpu_count
    n_workers = n_workers or max(1, cpu_count() - 1)
    
    # Prepare tasks
    tasks = [(symbol, df.copy(), horizon, threshold) for symbol, df in data.items()]
    
    print(f"Building features for {len(tasks)} symbols using {n_workers} workers...")
    
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_build_features_for_symbol, task): task[0] for task in tasks}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                symbol, df, feature_names = result
                results[symbol] = (df, feature_names)
    
    print(f"Completed feature building for {len(results)} symbols")
    return results


# ============== BENCHMARK ==============

def benchmark_feature_building(df: pd.DataFrame, iterations: int = 10) -> Dict[str, float]:
    """Benchmark feature building performance"""
    import time
    
    results = {}
    
    # Test vectorized version
    times = []
    for _ in range(iterations):
        start = time.time()
        _, _ = build_all_features_fast(df.copy())
        times.append(time.time() - start)
    
    results['vectorized_mean'] = np.mean(times)
    results['vectorized_std'] = np.std(times)
    
    print(f"Vectorized: {results['vectorized_mean']:.3f}s ± {results['vectorized_std']:.3f}s")
    
    return results


if __name__ == "__main__":
    # Test with sample data
    import os
    import sys
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.parallel_loader import ParallelDataLoader
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print(f"Data directory: {data_dir}")
    
    # Load sample data
    loader = ParallelDataLoader(data_dir)
    data = loader.load_all_klines_parallel(timeframes=['1h'], max_symbols=5)
    
    if data:
        # Benchmark single symbol
        symbol = list(data.keys())[0]
        df = data[symbol]['1h']
        print(f"\nBenchmarking with {symbol} ({len(df)} rows)...")
        benchmark_feature_building(df, iterations=5)
