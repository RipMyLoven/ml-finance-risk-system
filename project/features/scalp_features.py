"""
Scalp Features - фичи для 5m и 15m таймфреймов

Scalp features:
- log returns
- RSI 5m / 15m
- micro volatility
- volume spikes
- orderbook imbalance
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Логарифмические returns для scalp"""
    df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_3'] = np.log(df['close'] / df['close'].shift(3))
    df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
    df['log_return_10'] = np.log(df['close'] / df['close'].shift(10))
    
    # Cumulative returns
    df['cum_return_5'] = df['log_return_1'].rolling(5).sum()
    df['cum_return_10'] = df['log_return_1'].rolling(10).sum()
    
    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """RSI для scalp (короткие периоды)"""
    for period in [5, 7, 14]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        # Нормализуем в диапазон [-1, 1]
        df[f'rsi_{period}'] = (rsi - 50) / 50
    
    # RSI slope
    df['rsi_slope'] = df['rsi_14'].diff(3)
    
    return df


def add_micro_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Микро-волатильность для scalp"""
    # Return volatility (короткие окна)
    df['micro_vol_3'] = df['log_return_1'].rolling(3).std()
    df['micro_vol_5'] = df['log_return_1'].rolling(5).std()
    df['micro_vol_10'] = df['log_return_1'].rolling(10).std()
    
    # Volatility ratio
    df['vol_ratio_3_10'] = df['micro_vol_3'] / (df['micro_vol_10'] + 1e-10)
    
    # ATR (short)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_5'] = tr.rolling(5).mean()
    df['atr_10'] = tr.rolling(10).mean()
    
    # ATR normalized
    df['atr_norm_5'] = df['atr_5'] / df['close']
    df['atr_norm_10'] = df['atr_10'] / df['close']
    
    # ATR expansion
    df['atr_expansion'] = df['atr_5'] / (df['atr_10'] + 1e-10)
    
    return df


def add_volume_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """Volume spikes для scalp"""
    # Volume ratios
    vol_sma_5 = df['volume'].rolling(5).mean()
    vol_sma_10 = df['volume'].rolling(10).mean()
    vol_sma_20 = df['volume'].rolling(20).mean()
    
    df['volume_ratio_5'] = df['volume'] / (vol_sma_5 + 1e-10)
    df['volume_ratio_10'] = df['volume'] / (vol_sma_10 + 1e-10)
    df['volume_ratio_20'] = df['volume'] / (vol_sma_20 + 1e-10)
    
    # Volume spike detection
    df['volume_spike'] = (df['volume_ratio_10'] > 2.0).astype(int)
    
    # Volume trend
    df['volume_trend'] = vol_sma_5 / (vol_sma_20 + 1e-10) - 1
    
    # Volume momentum
    df['volume_momentum'] = df['volume'].pct_change(3)
    
    return df


def add_orderbook_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orderbook imbalance proxy (из taker buy/sell данных)
    
    Если в данных есть taker_buy_base/taker_buy_quote
    """
    if 'taker_buy_base' in df.columns and 'volume' in df.columns:
        # Buy pressure
        df['buy_ratio'] = df['taker_buy_base'] / (df['volume'] + 1e-10)
        df['buy_pressure'] = df['buy_ratio'] - 0.5
        df['buy_pressure_sma'] = df['buy_pressure'].rolling(5).mean()
        
        # Imbalance momentum
        df['imbalance_momentum'] = df['buy_pressure'].diff(3)
        
        # Cumulative imbalance
        df['cum_imbalance_5'] = df['buy_pressure'].rolling(5).sum()
        df['cum_imbalance_10'] = df['buy_pressure'].rolling(10).sum()
    elif 'buy_ratio' in df.columns:
        df['buy_pressure'] = df['buy_ratio'] - 0.5
        df['buy_pressure_sma'] = df['buy_pressure'].rolling(5).mean()
        df['imbalance_momentum'] = df['buy_pressure'].diff(3)
        df['cum_imbalance_5'] = df['buy_pressure'].rolling(5).sum()
        df['cum_imbalance_10'] = df['buy_pressure'].rolling(10).sum()
    elif 'buy_sell_ratio' in df.columns:
        # Если есть taker volume данные
        df['buy_pressure'] = (df['buy_sell_ratio'] - 1) / 2  # Нормализуем
        df['buy_pressure'] = df['buy_pressure'].clip(-1, 1)
        df['buy_pressure_sma'] = df['buy_pressure'].rolling(5).mean()
        df['imbalance_momentum'] = df['buy_pressure'].diff(3)
        df['cum_imbalance_5'] = df['buy_pressure'].rolling(5).sum()
        df['cum_imbalance_10'] = df['buy_pressure'].rolling(10).sum()
    
    return df


def add_derivatives_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить фичи из derivatives данных (funding, OI, LS ratio)
    """
    # Funding Rate фичи
    if 'funding_rate' in df.columns:
        df['funding_rate_norm'] = df['funding_rate'] * 1000  # Нормализуем
        df['funding_rate_sma'] = df['funding_rate'].rolling(8).mean() * 1000
        df['funding_rate_change'] = df['funding_rate'].diff() * 1000
        df['funding_rate_zscore'] = (
            (df['funding_rate'] - df['funding_rate'].rolling(24).mean()) / 
            (df['funding_rate'].rolling(24).std() + 1e-10)
        )
    
    # Open Interest фичи
    if 'sum_open_interest' in df.columns:
        df['oi_change'] = df['sum_open_interest'].pct_change()
        df['oi_change_sma'] = df['oi_change'].rolling(5).mean()
        df['oi_zscore'] = (
            (df['sum_open_interest'] - df['sum_open_interest'].rolling(24).mean()) / 
            (df['sum_open_interest'].rolling(24).std() + 1e-10)
        )
        
        # OI vs Price divergence
        price_change = df['close'].pct_change()
        oi_change = df['sum_open_interest'].pct_change()
        df['oi_price_divergence'] = oi_change - price_change
    
    # Long/Short Ratio фичи
    if 'long_short_ratio' in df.columns:
        df['ls_ratio_norm'] = (df['long_short_ratio'] - 1)  # Нормализуем вокруг 1
        df['ls_ratio_sma'] = df['ls_ratio_norm'].rolling(5).mean()
        df['ls_ratio_change'] = df['long_short_ratio'].pct_change()
        df['ls_ratio_zscore'] = (
            (df['long_short_ratio'] - df['long_short_ratio'].rolling(24).mean()) / 
            (df['long_short_ratio'].rolling(24).std() + 1e-10)
        )
    
    if 'long_account' in df.columns:
        df['long_pct'] = df['long_account']
        df['short_pct'] = df['short_account']
        df['long_short_diff'] = df['long_account'] - df['short_account']
    
    # Taker Volume фичи
    if 'buy_vol' in df.columns and 'sell_vol' in df.columns:
        df['taker_buy_sell_ratio'] = df['buy_vol'] / (df['sell_vol'] + 1e-10)
        df['taker_buy_sell_diff'] = (df['buy_vol'] - df['sell_vol']) / (df['buy_vol'] + df['sell_vol'] + 1e-10)
        df['taker_vol_total'] = df['buy_vol'] + df['sell_vol']
        df['taker_vol_change'] = df['taker_vol_total'].pct_change()
    
    # Premium Index фичи (basis)
    if 'premium_close' in df.columns:
        df['premium_norm'] = df['premium_close'] * 100  # В процентах
        df['premium_sma'] = df['premium_norm'].rolling(8).mean()
        df['premium_zscore'] = (
            (df['premium_close'] - df['premium_close'].rolling(24).mean()) / 
            (df['premium_close'].rolling(24).std() + 1e-10)
        )
    
    # Basis фичи (если есть)
    if 'basis' in df.columns:
        df['basis_sma'] = df['basis'].rolling(8).mean()
        df['basis_zscore'] = (
            (df['basis'] - df['basis'].rolling(24).mean()) / 
            (df['basis'].rolling(24).std() + 1e-10)
        )
    
    return df


def add_price_action(df: pd.DataFrame) -> pd.DataFrame:
    """Price action features для scalp"""
    # Bar structure
    bar_range = df['high'] - df['low']
    df['bar_position'] = (df['close'] - df['low']) / (bar_range + 1e-10)
    df['bar_range_pct'] = bar_range / df['close']
    df['bar_body_pct'] = abs(df['close'] - df['open']) / df['close']
    
    # Shadows
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (bar_range + 1e-10)
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (bar_range + 1e-10)
    
    # Gap
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # High/Low break
    df['high_break'] = ((df['high'] > df['high'].shift(1).rolling(5).max())).astype(int)
    df['low_break'] = ((df['low'] < df['low'].shift(1).rolling(5).min())).astype(int)
    
    return df


def add_momentum_scalp(df: pd.DataFrame) -> pd.DataFrame:
    """Short-term momentum для scalp"""
    # EMA crossovers (short)
    ema_3 = df['close'].ewm(span=3).mean()
    ema_8 = df['close'].ewm(span=8).mean()
    ema_13 = df['close'].ewm(span=13).mean()
    
    df['ema_cross_3_8'] = (ema_3 - ema_8) / ema_8
    df['ema_cross_3_13'] = (ema_3 - ema_13) / ema_13
    df['ema_cross_8_13'] = (ema_8 - ema_13) / ema_13
    
    # Price vs EMA
    df['price_vs_ema3'] = (df['close'] - ema_3) / ema_3
    df['price_vs_ema8'] = (df['close'] - ema_8) / ema_8
    
    # Stochastic (short)
    low_5 = df['low'].rolling(5).min()
    high_5 = df['high'].rolling(5).max()
    df['stoch_k_5'] = (df['close'] - low_5) / (high_5 - low_5 + 1e-10) - 0.5
    df['stoch_d_5'] = df['stoch_k_5'].rolling(3).mean()
    
    # ROC (short)
    df['roc_3'] = df['close'].pct_change(3)
    df['roc_5'] = df['close'].pct_change(5)
    
    return df


def add_target_scalp(df: pd.DataFrame, horizon: int = 12, threshold: float = 0.003) -> pd.DataFrame:
    """
    Target для scalp модели
    
    Classes:
        1  = price_up (> threshold)
        0  = flat
        -1 = price_down (< -threshold)
    
    Args:
        horizon: сколько баров вперёд смотрим (12 баров = 1 час на 5m)
        threshold: минимальный порог движения (0.3%)
    """
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    
    df['target'] = 0  # flat
    df.loc[future_return > threshold, 'target'] = 1   # up
    df.loc[future_return < -threshold, 'target'] = -1  # down (будет 2 для LightGBM)
    
    # Для LightGBM: переводим в 0, 1, 2
    df['target'] = df['target'].map({-1: 0, 0: 1, 1: 2})
    
    # Сохраняем future return для расчёта expected return
    df['future_return'] = future_return
    
    return df


def build_scalp_features(
    df: pd.DataFrame,
    horizon: int = 12,
    threshold: float = 0.003,
    normalize: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Построить все Scalp фичи
    
    Args:
        df: DataFrame с OHLCV данными
        horizon: горизонт предсказания
        threshold: порог для target
        normalize: нормализовать ли фичи
    
    Returns:
        df: DataFrame с фичами
        feature_names: список названий фичей
    """
    initial_len = len(df)
    
    # Добавляем все фичи
    df = add_log_returns(df)
    df = add_rsi(df)
    df = add_micro_volatility(df)
    df = add_volume_spikes(df)
    df = add_orderbook_imbalance(df)
    df = add_price_action(df)
    df = add_momentum_scalp(df)
    df = add_derivatives_features(df)  # НОВЫЕ фичи из derivatives
    df = add_target_scalp(df, horizon, threshold)
    
    # Определяем фичи (исключаем служебные колонки)
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
        'taker_buy_volume', 'taker_buy_quote_volume',
        'buy_ratio', 'target', 'future_return',
        'open_time', 'close_time', 'timestamp', 'datetime', 'date', 'time',
        'symbol', 'buy_volume', 'sell_volume', 'trades_count',
        # Derivatives raw columns
        'funding_rate', 'sum_open_interest', 'sum_open_interest_value',
        'long_short_ratio', 'long_account', 'short_account',
        'buy_sell_ratio', 'buy_vol', 'sell_vol',
        'mark_open', 'mark_high', 'mark_low', 'mark_close',
        'premium_open', 'premium_high', 'premium_low', 'premium_close',
        'basis'
    ]
    
    feature_names = [col for col in df.columns if col not in exclude_cols 
                     and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    # ВАЖНО: Заменяем inf и NaN ПЕРЕД нормализацией, чтобы не терять данные
    # Стратегия: forward-fill -> backward-fill -> 0 для оставшихся NaN
    for col in feature_names:
        # Сначала заменяем inf на NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # Forward fill - заполняем предыдущим значением
        df[col] = df[col].ffill()
        # Backward fill - для начала ряда
        df[col] = df[col].bfill()
        # Оставшиеся NaN заменяем на 0
        df[col] = df[col].fillna(0)
    
    # Убираем только строки где target=NaN (конец ряда из-за shift)
    # Это минимально необходимое удаление
    if 'target' in df.columns:
        df = df[df['target'].notna()]
    
    # Проверка на оставшиеся inf/nan в фичах (safety check)
    for col in feature_names:
        if df[col].isna().any() or np.isinf(df[col]).any():
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Нормализация
    if normalize and len(df) > 0:
        scaler = StandardScaler()
        # Клипаем экстремальные значения перед нормализацией
        for col in feature_names:
            q01 = df[col].quantile(0.001)
            q99 = df[col].quantile(0.999)
            df[col] = df[col].clip(q01, q99)
        
        df[feature_names] = scaler.fit_transform(df[feature_names])
        # Финальная проверка после нормализации
        df[feature_names] = df[feature_names].replace([np.inf, -np.inf], 0).fillna(0)
    
    final_len = len(df)
    
    return df, feature_names


# Список всех scalp фичей для экспорта
SCALP_FEATURE_NAMES = [
    'log_return_1', 'log_return_3', 'log_return_5', 'log_return_10',
    'cum_return_5', 'cum_return_10',
    'rsi_5', 'rsi_7', 'rsi_14', 'rsi_slope',
    'micro_vol_3', 'micro_vol_5', 'micro_vol_10', 'vol_ratio_3_10',
    'atr_5', 'atr_10', 'atr_norm_5', 'atr_norm_10', 'atr_expansion',
    'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
    'volume_spike', 'volume_trend', 'volume_momentum',
    'buy_pressure', 'buy_pressure_sma', 'imbalance_momentum',
    'cum_imbalance_5', 'cum_imbalance_10',
    'bar_position', 'bar_range_pct', 'bar_body_pct',
    'upper_shadow', 'lower_shadow', 'gap', 'high_break', 'low_break',
    'ema_cross_3_8', 'ema_cross_3_13', 'ema_cross_8_13',
    'price_vs_ema3', 'price_vs_ema8',
    'stoch_k_5', 'stoch_d_5', 'roc_3', 'roc_5'
]
