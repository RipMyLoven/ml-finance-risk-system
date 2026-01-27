"""
Multi-Timeframe Feature Engineering

Универсальные фичи для любой монеты:
- Все значения нормализованы (returns, ratios, z-scores)
- Нет абсолютных цен!
- Multi-timeframe анализ (1m, 5m, 15m, 1h, 4h, 1d)
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def add_normalized_price_features(df: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
    """
    Ценовые фичи (нормализованные!)
    """
    s = suffix  # для multi-timeframe
    
    # Returns (логарифмические)
    df[f'return_1{s}'] = np.log(df['close'] / df['close'].shift(1))
    df[f'return_3{s}'] = np.log(df['close'] / df['close'].shift(3))
    df[f'return_5{s}'] = np.log(df['close'] / df['close'].shift(5))
    df[f'return_10{s}'] = np.log(df['close'] / df['close'].shift(10))
    df[f'return_20{s}'] = np.log(df['close'] / df['close'].shift(20))
    
    # Cumulative returns
    df[f'cum_return_5{s}'] = df[f'return_1{s}'].rolling(5).sum()
    df[f'cum_return_10{s}'] = df[f'return_1{s}'].rolling(10).sum()
    
    # Bar structure (normalized)
    bar_range = df['high'] - df['low']
    df[f'bar_position{s}'] = (df['close'] - df['low']) / (bar_range + 1e-10)
    df[f'bar_range_pct{s}'] = bar_range / df['close']
    df[f'bar_body_pct{s}'] = abs(df['close'] - df['open']) / df['close']
    df[f'upper_shadow{s}'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (bar_range + 1e-10)
    df[f'lower_shadow{s}'] = (df[['open', 'close']].min(axis=1) - df['low']) / (bar_range + 1e-10)
    
    # Gap (normalized)
    df[f'gap{s}'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    return df


def add_normalized_volume_features(df: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
    """
    Объёмные фичи (нормализованные!)
    """
    s = suffix
    
    # Volume ratios
    vol_sma_10 = df['volume_btc'].rolling(10).mean()
    vol_sma_20 = df['volume_btc'].rolling(20).mean()
    
    df[f'volume_ratio{s}'] = df['volume_btc'] / (vol_sma_10 + 1e-10)
    df[f'volume_trend{s}'] = vol_sma_10 / (vol_sma_20 + 1e-10) - 1
    
    # Volume momentum
    df[f'volume_change{s}'] = df['volume_btc'].pct_change()
    df[f'volume_accel{s}'] = df[f'volume_change{s}'].diff()
    
    # Trade intensity
    if 'num_trades' in df.columns:
        trades_sma = df['num_trades'].rolling(10).mean()
        df[f'trades_ratio{s}'] = df['num_trades'] / (trades_sma + 1e-10)
    
    # Buy pressure (очень важно!)
    if 'buy_ratio' in df.columns:
        df[f'buy_pressure{s}'] = df['buy_ratio'] - 0.5  # центрируем
        df[f'buy_pressure_sma{s}'] = df[f'buy_pressure{s}'].rolling(5).mean()
        df[f'buy_pressure_change{s}'] = df[f'buy_pressure{s}'].diff()
    
    # Price-volume relationship
    df[f'pv_corr{s}'] = df['close'].pct_change().rolling(10).corr(df['volume_btc'].pct_change())
    
    return df


def add_normalized_momentum_features(df: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
    """
    Моментум индикаторы (нормализованные!)
    """
    s = suffix
    
    # Price vs SMA (normalized distance)
    for period in [5, 10, 20, 50]:
        sma = df['close'].rolling(period).mean()
        df[f'price_vs_sma{period}{s}'] = (df['close'] - sma) / sma
    
    # SMA slopes (momentum of averages)
    sma_10 = df['close'].rolling(10).mean()
    sma_20 = df['close'].rolling(20).mean()
    df[f'sma10_slope{s}'] = sma_10.pct_change(3)
    df[f'sma20_slope{s}'] = sma_20.pct_change(5)
    
    # EMA crossovers
    ema_fast = df['close'].ewm(span=8).mean()
    ema_slow = df['close'].ewm(span=21).mean()
    df[f'ema_cross{s}'] = (ema_fast - ema_slow) / ema_slow
    df[f'ema_cross_slope{s}'] = df[f'ema_cross{s}'].diff()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df[f'rsi{s}'] = (100 - (100 / (1 + rs))) / 100 - 0.5  # normalized to [-0.5, 0.5]
    
    # RSI divergence
    df[f'rsi_slope{s}'] = df[f'rsi{s}'].diff(3)
    
    # Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df[f'stoch_k{s}'] = (df['close'] - low_14) / (high_14 - low_14 + 1e-10) - 0.5
    df[f'stoch_d{s}'] = df[f'stoch_k{s}'].rolling(3).mean()
    
    # ROC (Rate of Change)
    df[f'roc_5{s}'] = df['close'].pct_change(5)
    df[f'roc_10{s}'] = df['close'].pct_change(10)
    
    # MACD (normalized)
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df[f'macd_norm{s}'] = macd / df['close']
    df[f'macd_signal{s}'] = (macd - signal) / df['close']
    
    return df


def add_normalized_volatility_features(df: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
    """
    Волатильность (нормализованная!)
    """
    s = suffix
    
    # Return volatility
    df[f'volatility_5{s}'] = df['return_1' + s if s else 'return_1'].rolling(5).std()
    df[f'volatility_10{s}'] = df['return_1' + s if s else 'return_1'].rolling(10).std()
    df[f'volatility_20{s}'] = df['return_1' + s if s else 'return_1'].rolling(20).std()
    
    # Volatility ratio (current vs historical)
    df[f'vol_ratio{s}'] = df[f'volatility_5{s}'] / (df[f'volatility_20{s}'] + 1e-10)
    
    # ATR (normalized)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df[f'atr_pct{s}'] = atr / df['close']
    
    # ATR change
    df[f'atr_change{s}'] = atr.pct_change(5)
    
    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df[f'bb_position{s}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10) - 0.5
    df[f'bb_width{s}'] = (bb_upper - bb_lower) / sma20
    df[f'bb_squeeze{s}'] = df[f'bb_width{s}'].rolling(20).rank(pct=True)  # percentile
    
    # High-Low range percentile
    range_pct = (df['high'] - df['low']) / df['close']
    df[f'range_percentile{s}'] = range_pct.rolling(50).rank(pct=True)
    
    return df


def add_pattern_features(df: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
    """
    Паттерны и структура рынка
    """
    s = suffix
    
    # Higher highs / Lower lows
    df[f'higher_high{s}'] = (df['high'] > df['high'].shift(1)).astype(float)
    df[f'lower_low{s}'] = (df['low'] < df['low'].shift(1)).astype(float)
    df[f'hh_count{s}'] = df[f'higher_high{s}'].rolling(5).sum() / 5
    df[f'll_count{s}'] = df[f'lower_low{s}'].rolling(5).sum() / 5
    
    # Trend strength
    df[f'trend_strength{s}'] = df[f'hh_count{s}'] - df[f'll_count{s}']
    
    # Consecutive moves
    up_move = (df['close'] > df['close'].shift(1)).astype(int)
    df[f'consecutive_up{s}'] = up_move.groupby((up_move != up_move.shift()).cumsum()).cumcount() * up_move
    df[f'consecutive_down{s}'] = (1 - up_move).groupby(((1-up_move) != (1-up_move).shift()).cumsum()).cumcount() * (1 - up_move)
    
    # Mean reversion signal
    z_score = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-10)
    df[f'z_score{s}'] = z_score
    df[f'mean_revert_signal{s}'] = -np.tanh(z_score)  # сильное отклонение → ожидаем возврат
    
    return df


def resample_to_timeframe(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Агрегация в OHLCV бары
    """
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    df_indexed = df.set_index('datetime')
    
    agg_dict = {
        'price': ['first', 'max', 'min', 'last'],
        'qty_btc': 'sum',
    }
    
    # Добавляем опциональные колонки
    if 'qty_usd' in df.columns:
        agg_dict['qty_usd'] = 'sum'
    if 'trade_id' in df.columns:
        agg_dict['trade_id'] = 'count'
    if 'is_buyer_maker' in df.columns:
        agg_dict['is_buyer_maker'] = ['sum', 'count']
    
    bars = df_indexed.groupby(pd.Grouper(freq=freq)).agg(agg_dict)
    
    # Плоские названия
    bars.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in bars.columns]
    
    # Переименовываем
    rename_map = {
        'price_first': 'open',
        'price_max': 'high', 
        'price_min': 'low',
        'price_last': 'close',
        'qty_btc_sum': 'volume_btc',
        'qty_usd_sum': 'volume_usd',
        'trade_id_count': 'num_trades',
        'is_buyer_maker_sum': 'sell_trades',
        'is_buyer_maker_count': 'total_trades'
    }
    bars = bars.rename(columns=rename_map)
    
    # Buy ratio
    if 'total_trades' in bars.columns and 'sell_trades' in bars.columns:
        bars['buy_trades'] = bars['total_trades'] - bars['sell_trades']
        bars['buy_ratio'] = bars['buy_trades'] / (bars['total_trades'] + 1e-10)
    
    # Убираем пустые
    bars = bars.dropna(subset=['close'])
    bars = bars[bars['volume_btc'] > 0]
    
    return bars.reset_index()


def build_single_timeframe_features(bars: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
    """
    Строим все фичи для одного таймфрейма
    """
    df = bars.copy()
    
    df = add_normalized_price_features(df, suffix)
    df = add_normalized_volume_features(df, suffix)
    df = add_normalized_momentum_features(df, suffix)
    df = add_normalized_volatility_features(df, suffix)
    df = add_pattern_features(df, suffix)
    
    return df


def build_multi_timeframe_features(
    trades: pd.DataFrame,
    base_freq: str = '1min',
    timeframes: List[str] = ['5min', '15min', '1h', '4h', '1d'],
    horizon: int = 5
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Строим фичи на нескольких таймфреймах
    
    Args:
        trades: DataFrame с trades
        base_freq: базовый таймфрейм (для target и выходных баров)
        timeframes: дополнительные таймфреймы для фичей
        horizon: горизонт предсказания (в барах base_freq)
    
    Returns:
        df: DataFrame с фичами
        feature_names: список названий фичей
    """
    print(f"Building multi-timeframe features...")
    print(f"  Base: {base_freq}")
    print(f"  Additional: {timeframes}")
    
    # Базовый таймфрейм
    print(f"\n  Resampling to {base_freq}...")
    bars_base = resample_to_timeframe(trades, base_freq)
    df = build_single_timeframe_features(bars_base, suffix='')
    
    # Дополнительные таймфреймы
    for tf in timeframes:
        print(f"  Resampling to {tf}...")
        bars_tf = resample_to_timeframe(trades, tf)
        bars_tf = build_single_timeframe_features(bars_tf, suffix=f'_{tf}')
        
        # Merge по времени (берём последнее известное значение)
        # Для каждого бара base берём фичи из соответствующего бара tf
        tf_features = [col for col in bars_tf.columns if col.endswith(f'_{tf}')]
        bars_tf_subset = bars_tf[['datetime'] + tf_features].copy()
        bars_tf_subset = bars_tf_subset.set_index('datetime')
        
        # Reindex к базовому таймфрейму (forward fill)
        df = df.set_index('datetime')
        for col in tf_features:
            df[col] = bars_tf_subset[col].reindex(df.index, method='ffill')
        df = df.reset_index()
    
    # Target
    print(f"\n  Creating target (horizon={horizon} bars)...")
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
    df['target'] = (df['future_return'] > 0).astype(int)
    
    # Убираем NaN
    df = df.dropna()
    
    # Собираем список фичей
    exclude_cols = [
        'datetime', 'open', 'high', 'low', 'close',
        'volume_usd', 'volume_btc', 'num_trades',
        'sell_trades', 'total_trades', 'buy_trades', 'buy_ratio',
        'future_return', 'target', 'source_file', 'timestamp_ms',
        'trade_id', 'qty_usd', 'qty_btc', 'is_buyer_maker', 'price'
    ]
    
    feature_names = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\n  Total features: {len(feature_names)}")
    print(f"  Samples: {len(df):,}")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df, feature_names


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Возвращает группы фичей для анализа
    """
    return {
        'price': ['return_', 'cum_return_', 'bar_position', 'bar_range', 'bar_body', 'shadow', 'gap'],
        'volume': ['volume_ratio', 'volume_trend', 'volume_change', 'trades_ratio', 'buy_pressure', 'pv_corr'],
        'momentum': ['price_vs_sma', 'sma_slope', 'ema_cross', 'rsi', 'stoch', 'roc', 'macd'],
        'volatility': ['volatility_', 'vol_ratio', 'atr', 'bb_', 'range_percentile'],
        'pattern': ['higher_high', 'lower_low', 'trend_strength', 'consecutive', 'z_score', 'mean_revert']
    }
