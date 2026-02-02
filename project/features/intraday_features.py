"""
Intraday Features - фичи для 1h, 4h, 12h таймфреймов

Intraday features:
- trend slope
- VWAP deviation
- funding rate change
- OI delta
- BTC correlation
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trend features для intraday"""
    # SMA
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # Price vs SMA (normalized)
    df['price_vs_sma10'] = (df['close'] - df['sma_10']) / df['sma_10']
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
    # SMA crossovers
    df['sma_cross_10_20'] = df['sma_10'] / df['sma_20'] - 1
    df['sma_cross_10_50'] = df['sma_10'] / df['sma_50'] - 1
    df['sma_cross_20_50'] = df['sma_20'] / df['sma_50'] - 1
    
    # Trend slope (линейная регрессия)
    for window in [10, 20]:
        slopes = []
        for i in range(len(df)):
            if i < window:
                slopes.append(0)
            else:
                y = df['close'].iloc[i-window:i].values
                x = np.arange(window)
                slope = np.polyfit(x, y, 1)[0]
                # Нормализуем наклон
                slopes.append(slope / df['close'].iloc[i])
        df[f'trend_slope_{window}'] = slopes
    
    # EMA trend
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['ema_trend'] = (ema_12 - ema_26) / ema_26
    df['ema_trend_slope'] = df['ema_trend'].diff(5)
    
    return df


def add_vwap_features(df: pd.DataFrame) -> pd.DataFrame:
    """VWAP deviation"""
    # Расчёт VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap_cum_vol'] = df['volume'].cumsum()
    df['vwap_cum_tp_vol'] = (df['typical_price'] * df['volume']).cumsum()
    df['vwap'] = df['vwap_cum_tp_vol'] / df['vwap_cum_vol']
    
    # Rolling VWAP (более практичный)
    for window in [10, 20]:
        tp_vol = df['typical_price'] * df['volume']
        df[f'vwap_{window}'] = tp_vol.rolling(window).sum() / df['volume'].rolling(window).sum()
        df[f'vwap_dev_{window}'] = (df['close'] - df[f'vwap_{window}']) / df[f'vwap_{window}']
    
    # VWAP bands
    std_20 = df['close'].rolling(20).std()
    df['vwap_upper'] = df['vwap_20'] + 2 * std_20
    df['vwap_lower'] = df['vwap_20'] - 2 * std_20
    df['vwap_position'] = (df['close'] - df['vwap_lower']) / (df['vwap_upper'] - df['vwap_lower'] + 1e-10)
    
    return df


def add_funding_features(df: pd.DataFrame, funding_rate: pd.Series = None) -> pd.DataFrame:
    """
    Funding rate features
    
    Если funding_rate в df или передан отдельно
    """
    # Проверяем есть ли funding_rate в данных
    if 'funding_rate' in df.columns and df['funding_rate'].notna().any():
        # Funding rate change
        df['funding_change'] = df['funding_rate'].diff() * 1000  # Нормализуем
        df['funding_change_3'] = df['funding_rate'].diff(3) * 1000
        
        # Funding momentum
        df['funding_sma'] = df['funding_rate'].rolling(8).mean() * 1000
        df['funding_vs_sma'] = (df['funding_rate'] - df['funding_rate'].rolling(8).mean()) * 1000
        
        # Extreme funding
        df['funding_extreme_pos'] = (df['funding_rate'] > 0.001).astype(int)
        df['funding_extreme_neg'] = (df['funding_rate'] < -0.001).astype(int)
        
        # Funding zscore
        df['funding_zscore'] = (
            (df['funding_rate'] - df['funding_rate'].rolling(24).mean()) / 
            (df['funding_rate'].rolling(24).std() + 1e-10)
        )
    elif funding_rate is not None and len(funding_rate) > 0:
        # Merge funding rate
        df = df.merge(
            funding_rate.to_frame('funding_rate'),
            left_index=True,
            right_index=True,
            how='left'
        )
        df['funding_rate'] = df['funding_rate'].fillna(method='ffill')
        
        # Funding rate change
        df['funding_change'] = df['funding_rate'].diff() * 1000
        df['funding_change_3'] = df['funding_rate'].diff(3) * 1000
        
        # Funding momentum
        df['funding_sma'] = df['funding_rate'].rolling(8).mean() * 1000
        df['funding_vs_sma'] = (df['funding_rate'] - df['funding_rate'].rolling(8).mean()) * 1000
        
        # Extreme funding
        df['funding_extreme_pos'] = (df['funding_rate'] > 0.001).astype(int)
        df['funding_extreme_neg'] = (df['funding_rate'] < -0.001).astype(int)
        df['funding_zscore'] = 0
    else:
        # Placeholder columns
        df['funding_change'] = 0
        df['funding_change_3'] = 0
        df['funding_sma'] = 0
        df['funding_vs_sma'] = 0
        df['funding_extreme_pos'] = 0
        df['funding_extreme_neg'] = 0
        df['funding_zscore'] = 0
    
    return df


def add_oi_features(df: pd.DataFrame, oi_data: pd.Series = None) -> pd.DataFrame:
    """
    Open Interest features
    
    OI delta показывает приток/отток денег
    """
    # Проверяем есть ли OI в данных
    if 'sum_open_interest' in df.columns and df['sum_open_interest'].notna().any():
        # OI change
        df['oi_change'] = df['sum_open_interest'].pct_change()
        df['oi_change_3'] = df['sum_open_interest'].pct_change(3)
        df['oi_change_10'] = df['sum_open_interest'].pct_change(10)
        
        # OI trend
        oi_sma = df['sum_open_interest'].rolling(10).mean()
        df['oi_vs_sma'] = df['sum_open_interest'] / (oi_sma + 1e-10) - 1
        
        # OI momentum
        df['oi_momentum'] = df['oi_change'].diff()
        
        # Price-OI divergence
        price_change = df['close'].pct_change()
        df['price_oi_corr'] = price_change.rolling(10).corr(df['oi_change'])
        
        # OI zscore
        df['oi_zscore'] = (
            (df['sum_open_interest'] - df['sum_open_interest'].rolling(24).mean()) / 
            (df['sum_open_interest'].rolling(24).std() + 1e-10)
        )
    elif oi_data is not None and len(oi_data) > 0:
        df = df.merge(
            oi_data.to_frame('open_interest'),
            left_index=True,
            right_index=True,
            how='left'
        )
        df['open_interest'] = df['open_interest'].fillna(method='ffill')
        
        # OI change
        df['oi_change'] = df['open_interest'].pct_change()
        df['oi_change_3'] = df['open_interest'].pct_change(3)
        df['oi_change_10'] = df['open_interest'].pct_change(10)
        
        # OI trend
        oi_sma = df['open_interest'].rolling(10).mean()
        df['oi_vs_sma'] = df['open_interest'] / (oi_sma + 1e-10) - 1
        
        # OI momentum
        df['oi_momentum'] = df['oi_change'].diff()
        
        # Price-OI divergence
        price_change = df['close'].pct_change()
        df['price_oi_corr'] = price_change.rolling(10).corr(df['oi_change'])
        df['oi_zscore'] = 0
    else:
        # Placeholder columns
        df['oi_change'] = 0
        df['oi_change_3'] = 0
        df['oi_change_10'] = 0
        df['oi_vs_sma'] = 0
        df['oi_momentum'] = 0
        df['price_oi_corr'] = 0
        df['oi_zscore'] = 0
    
    return df


def add_ls_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Long/Short Ratio features"""
    if 'long_short_ratio' in df.columns and df['long_short_ratio'].notna().any():
        df['ls_ratio_norm'] = df['long_short_ratio'] - 1  # Центрируем вокруг 1
        df['ls_ratio_sma'] = df['ls_ratio_norm'].rolling(8).mean()
        df['ls_ratio_change'] = df['long_short_ratio'].pct_change()
        df['ls_ratio_zscore'] = (
            (df['long_short_ratio'] - df['long_short_ratio'].rolling(24).mean()) / 
            (df['long_short_ratio'].rolling(24).std() + 1e-10)
        )
        
        if 'long_account' in df.columns:
            df['long_short_diff'] = df['long_account'] - df['short_account']
    else:
        df['ls_ratio_norm'] = 0
        df['ls_ratio_sma'] = 0
        df['ls_ratio_change'] = 0
        df['ls_ratio_zscore'] = 0
        df['long_short_diff'] = 0
    
    return df


def add_taker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Taker Volume features"""
    if 'buy_sell_ratio' in df.columns and df['buy_sell_ratio'].notna().any():
        df['taker_ratio_norm'] = df['buy_sell_ratio'] - 1  # Центрируем
        df['taker_ratio_sma'] = df['taker_ratio_norm'].rolling(8).mean()
        
        if 'buy_vol' in df.columns:
            df['taker_imbalance'] = (df['buy_vol'] - df['sell_vol']) / (df['buy_vol'] + df['sell_vol'] + 1e-10)
            df['taker_vol_change'] = (df['buy_vol'] + df['sell_vol']).pct_change()
    else:
        df['taker_ratio_norm'] = 0
        df['taker_ratio_sma'] = 0
        df['taker_imbalance'] = 0
        df['taker_vol_change'] = 0
    
    return df


def add_premium_features(df: pd.DataFrame) -> pd.DataFrame:
    """Premium Index / Basis features"""
    if 'premium_close' in df.columns and df['premium_close'].notna().any():
        df['premium_norm'] = df['premium_close'] * 100
        df['premium_sma'] = df['premium_norm'].rolling(8).mean()
        df['premium_zscore'] = (
            (df['premium_close'] - df['premium_close'].rolling(24).mean()) / 
            (df['premium_close'].rolling(24).std() + 1e-10)
        )
    else:
        df['premium_norm'] = 0
        df['premium_sma'] = 0
        df['premium_zscore'] = 0
    
    if 'basis' in df.columns and df['basis'].notna().any():
        df['basis_sma'] = df['basis'].rolling(8).mean()
        df['basis_zscore'] = (
            (df['basis'] - df['basis'].rolling(24).mean()) / 
            (df['basis'].rolling(24).std() + 1e-10)
        )
    else:
        df['basis_sma'] = 0
        df['basis_zscore'] = 0
    
    return df


def add_btc_correlation(df: pd.DataFrame, btc_returns: pd.Series = None) -> pd.DataFrame:
    """
    Корреляция с BTC
    """
    if btc_returns is not None and len(btc_returns) > 0:
        df = df.merge(
            btc_returns.to_frame('btc_return'),
            left_index=True,
            right_index=True,
            how='left'
        )
        df['btc_return'] = df['btc_return'].fillna(0)
        
        # Rolling correlation
        symbol_return = df['close'].pct_change()
        df['btc_corr_10'] = symbol_return.rolling(10).corr(df['btc_return'])
        df['btc_corr_20'] = symbol_return.rolling(20).corr(df['btc_return'])
        
        # Relative strength vs BTC
        df['relative_strength'] = symbol_return.rolling(10).mean() - df['btc_return'].rolling(10).mean()
        
        # Beta to BTC
        cov = symbol_return.rolling(20).cov(df['btc_return'])
        var = df['btc_return'].rolling(20).var()
        df['btc_beta'] = cov / (var + 1e-10)
    else:
        df['btc_return'] = 0
        df['btc_corr_10'] = 0
        df['btc_corr_20'] = 0
        df['relative_strength'] = 0
        df['btc_beta'] = 1
    
    return df


def add_volatility_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility для intraday"""
    # Returns
    returns = df['close'].pct_change()
    
    # Historical volatility
    df['volatility_10'] = returns.rolling(10).std()
    df['volatility_20'] = returns.rolling(20).std()
    df['volatility_50'] = returns.rolling(50).std()
    
    # Volatility ratio
    df['vol_ratio_10_50'] = df['volatility_10'] / (df['volatility_50'] + 1e-10)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_norm'] = df['atr_14'] / df['close']
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma_20 + 2 * std_20
    df['bb_lower'] = sma_20 - 2 * std_20
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    return df


def add_momentum_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum для intraday"""
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = (100 - (100 / (1 + rs))) / 100 - 0.5
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    df['macd_norm'] = macd / df['close']
    df['macd_signal'] = (macd - signal) / df['close']
    df['macd_hist_slope'] = df['macd_signal'].diff(3)
    
    # ADX (Average Directional Index)
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean() / 100  # normalize to 0-1
    df['di_diff'] = (plus_di - minus_di) / 100
    
    return df


def add_volume_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """Volume features для intraday"""
    # Volume ratios
    vol_sma_10 = df['volume'].rolling(10).mean()
    vol_sma_20 = df['volume'].rolling(20).mean()
    
    df['volume_ratio'] = df['volume'] / (vol_sma_20 + 1e-10)
    df['volume_trend'] = vol_sma_10 / (vol_sma_20 + 1e-10) - 1
    
    # On-Balance Volume
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv_slope'] = obv.diff(10) / (obv.rolling(10).mean() + 1e-10)
    
    # Money Flow Index
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    pos_sum = positive_flow.rolling(14).sum()
    neg_sum = negative_flow.rolling(14).sum()
    
    df['mfi'] = (100 - 100 / (1 + pos_sum / (neg_sum + 1e-10))) / 100 - 0.5
    
    return df


def add_target_intraday(df: pd.DataFrame, horizon: int = 6, threshold: float = 0.01) -> pd.DataFrame:
    """
    Target для intraday модели
    
    Args:
        horizon: сколько баров вперёд (6 баров = 6 часов на 1h)
        threshold: минимальный порог (1%)
    """
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    
    df['target'] = 0  # flat
    df.loc[future_return > threshold, 'target'] = 1   # up
    df.loc[future_return < -threshold, 'target'] = -1  # down
    
    # Для LightGBM: 0=down, 1=flat, 2=up
    df['target'] = df['target'].map({-1: 0, 0: 1, 1: 2})
    
    df['future_return'] = future_return
    
    return df


def build_intraday_features(
    df: pd.DataFrame,
    horizon: int = 6,
    threshold: float = 0.01,
    funding_rate: pd.Series = None,
    oi_data: pd.Series = None,
    btc_returns: pd.Series = None,
    normalize: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Построить все Intraday фичи
    """
    initial_len = len(df)
    
    df = add_trend_features(df)
    df = add_vwap_features(df)
    df = add_funding_features(df, funding_rate)
    df = add_oi_features(df, oi_data)
    df = add_ls_ratio_features(df)      # НОВОЕ
    df = add_taker_features(df)          # НОВОЕ
    df = add_premium_features(df)        # НОВОЕ
    df = add_btc_correlation(df, btc_returns)
    df = add_volatility_intraday(df)
    df = add_momentum_intraday(df)
    df = add_volume_intraday(df)
    df = add_target_intraday(df, horizon, threshold)
    
    # Исключаем служебные колонки
    # CRITICAL: Drop future_return to prevent leakage
    if 'future_return' in df.columns:
        df = df.drop(columns=['future_return'])
    
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
        'taker_buy_volume', 'taker_buy_quote_volume',
        'buy_ratio', 'target', 'future_return',  # Already dropped above, but keep in list for safety
        'sma_10', 'sma_20', 'sma_50', 'typical_price',
        'vwap_cum_vol', 'vwap_cum_tp_vol', 'vwap', 'vwap_10', 'vwap_20',
        'vwap_upper', 'vwap_lower', 'bb_upper', 'bb_lower',
        'funding_rate', 'open_interest', 'btc_return',
        # Derivatives raw columns
        'sum_open_interest', 'sum_open_interest_value',
        'long_short_ratio', 'long_account', 'short_account',
        'buy_sell_ratio', 'buy_vol', 'sell_vol',
        'mark_open', 'mark_high', 'mark_low', 'mark_close',
        'premium_open', 'premium_high', 'premium_low', 'premium_close',
        'basis',
        'open_time', 'close_time', 'timestamp', 'datetime', 'date', 'time',
        'symbol', 'buy_volume', 'sell_volume', 'trades_count'
    ]
    
    feature_names = [col for col in df.columns if col not in exclude_cols
                     and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    # ВАЖНО: Заменяем inf и NaN ПЕРЕД нормализацией, чтобы не терять данные
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
    if 'target' in df.columns:
        df = df[df['target'].notna()]
    
    # Проверка на оставшиеся inf/nan в фичах (safety check)
    for col in feature_names:
        if df[col].isna().any() or np.isinf(df[col]).any():
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    if normalize and len(df) > 0:
        scaler = StandardScaler()
        # Клипаем экстремальные значения перед нормализацией
        for col in feature_names:
            q01 = df[col].quantile(0.001)
            q99 = df[col].quantile(0.999)
            df[col] = df[col].clip(q01, q99)
        
        df[feature_names] = scaler.fit_transform(df[feature_names])
        df[feature_names] = df[feature_names].replace([np.inf, -np.inf], 0).fillna(0)
    
    final_len = len(df)
    
    return df, feature_names
    
    return df, feature_names


INTRADAY_FEATURE_NAMES = [
    'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50',
    'sma_cross_10_20', 'sma_cross_10_50', 'sma_cross_20_50',
    'trend_slope_10', 'trend_slope_20', 'ema_trend', 'ema_trend_slope',
    'vwap_dev_10', 'vwap_dev_20', 'vwap_position',
    'funding_change', 'funding_change_3', 'funding_vs_sma',
    'funding_extreme_pos', 'funding_extreme_neg',
    'oi_change', 'oi_change_3', 'oi_change_10', 'oi_vs_sma', 'oi_momentum', 'price_oi_corr',
    'btc_corr_10', 'btc_corr_20', 'relative_strength', 'btc_beta',
    'volatility_10', 'volatility_20', 'volatility_50', 'vol_ratio_10_50',
    'atr_14', 'atr_norm', 'bb_width', 'bb_position',
    'rsi_14', 'macd_norm', 'macd_signal', 'macd_hist_slope', 'adx', 'di_diff',
    'volume_ratio', 'volume_trend', 'obv_slope', 'mfi'
]
