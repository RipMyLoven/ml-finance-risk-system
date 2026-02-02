"""
Swing Features - фичи для 1d, 3d, 7d таймфреймов

Swing features:
- MA 50 / 200
- market regime (bull / bear / flat)
- macro volatility
- dominance BTC
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler


def add_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    """MA 50/200 и кроссоверы (адаптировано для меньших датасетов)"""
    # Simple Moving Averages (reduced windows for smaller datasets)
    min_window = min(20, len(df) // 3)
    df['sma_20'] = df['close'].rolling(min(20, min_window * 1)).mean()
    df['sma_50'] = df['close'].rolling(min(50, min_window * 2)).mean()
    df['sma_100'] = df['close'].rolling(min(100, min_window * 4)).mean()
    df['sma_200'] = df['close'].rolling(min(200, min_window * 6)).mean()
    
    # Price vs MA (normalized)
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    df['price_vs_sma100'] = (df['close'] - df['sma_100']) / df['sma_100']
    df['price_vs_sma200'] = (df['close'] - df['sma_200']) / df['sma_200']
    
    # MA crossovers
    df['ma_cross_20_50'] = df['sma_20'] / df['sma_50'] - 1
    df['ma_cross_50_100'] = df['sma_50'] / df['sma_100'] - 1
    df['ma_cross_50_200'] = df['sma_50'] / df['sma_200'] - 1
    df['ma_cross_100_200'] = df['sma_100'] / df['sma_200'] - 1
    
    # Golden/Death cross signals
    df['golden_cross'] = ((df['sma_50'] > df['sma_200']) & 
                          (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(int)
    df['death_cross'] = ((df['sma_50'] < df['sma_200']) & 
                         (df['sma_50'].shift(1) >= df['sma_200'].shift(1))).astype(int)
    
    # MA slope (trend strength)
    df['sma50_slope'] = df['sma_50'].pct_change(5)
    df['sma200_slope'] = df['sma_200'].pct_change(10)
    
    # Distance between MAs
    df['ma_spread'] = (df['sma_50'] - df['sma_200']) / df['sma_200']
    
    return df


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market regime classification
    
    Regimes:
    - Bull: устойчивый рост
    - Bear: устойчивое падение
    - Flat: боковик
    """
    # Long-term returns
    df['return_20d'] = df['close'].pct_change(20)
    df['return_50d'] = df['close'].pct_change(50)
    
    # Trend determination
    # Bull: price > SMA50 > SMA200 и return_20d > 0
    # Bear: price < SMA50 < SMA200 и return_20d < 0
    
    bull_condition = (
        (df['close'] > df['sma_50']) & 
        (df['sma_50'] > df['sma_200']) & 
        (df['return_20d'] > 0.02)
    )
    
    bear_condition = (
        (df['close'] < df['sma_50']) & 
        (df['sma_50'] < df['sma_200']) & 
        (df['return_20d'] < -0.02)
    )
    
    # Regime encoding
    df['regime'] = 0  # flat
    df.loc[bull_condition, 'regime'] = 1  # bull
    df.loc[bear_condition, 'regime'] = -1  # bear
    
    # One-hot encoding для фичей
    df['regime_bull'] = (df['regime'] == 1).astype(int)
    df['regime_bear'] = (df['regime'] == -1).astype(int)
    df['regime_flat'] = (df['regime'] == 0).astype(int)
    
    # Regime duration
    regime_change = df['regime'] != df['regime'].shift(1)
    df['regime_duration'] = regime_change.cumsum()
    df['regime_duration'] = df.groupby('regime_duration').cumcount() + 1
    
    # Normalize duration
    df['regime_duration_norm'] = df['regime_duration'] / 50  # max ~50 days
    
    return df


def add_macro_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Macro volatility features (adapted for smaller datasets)"""
    # Returns
    returns = df['close'].pct_change()
    
    # Adaptive window sizes
    min_window = max(5, len(df) // 10)
    
    # Historical volatility (adaptive windows)
    df['volatility_20'] = returns.rolling(min(20, min_window * 2)).std()
    df['volatility_50'] = returns.rolling(min(50, min_window * 4)).std()
    df['volatility_100'] = returns.rolling(min(100, min_window * 8)).std()
    
    # Annualized volatility
    df['volatility_annual'] = df['volatility_20'] * np.sqrt(365)
    
    # Volatility regime
    vol_median = df['volatility_50'].rolling(min(100, len(df) // 3)).median()
    df['vol_regime'] = df['volatility_20'] / (vol_median + 1e-10)
    
    # High volatility flag
    df['high_vol_flag'] = (df['vol_regime'] > 1.5).astype(int)
    df['low_vol_flag'] = (df['vol_regime'] < 0.7).astype(int)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    df['atr_14'] = tr.rolling(min(14, len(df) // 5)).mean()
    df['atr_50'] = tr.rolling(min(50, len(df) // 3)).mean()
    df['atr_norm'] = df['atr_14'] / df['close']
    
    # ATR expansion/contraction
    df['atr_ratio'] = df['atr_14'] / (df['atr_50'] + 1e-10)
    
    # Bollinger Band Width
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['bb_width'] = (4 * std_20) / sma_20
    
    # Historical drawdown (adaptive window)
    rolling_max = df['close'].rolling(min(50, len(df) // 3)).max()
    df['drawdown'] = (df['close'] - rolling_max) / rolling_max
    
    return df


def add_btc_dominance_features(df: pd.DataFrame, btc_dominance: pd.Series = None) -> pd.DataFrame:
    """
    BTC Dominance features
    
    Важно для altcoins: когда dominance растёт, альты обычно падают
    """
    if btc_dominance is not None and len(btc_dominance) > 0:
        df = df.merge(
            btc_dominance.to_frame('btc_dominance'),
            left_index=True,
            right_index=True,
            how='left'
        )
        df['btc_dominance'] = df['btc_dominance'].ffill()
        
        # Dominance change
        df['dominance_change_5d'] = df['btc_dominance'].pct_change(5)
        df['dominance_change_20d'] = df['btc_dominance'].pct_change(20)
        
        # Dominance trend
        dom_sma = df['btc_dominance'].rolling(20).mean()
        df['dominance_vs_sma'] = df['btc_dominance'] / dom_sma - 1
        
        # Dominance regime
        df['dominance_rising'] = (df['dominance_change_5d'] > 0.01).astype(int)
        df['dominance_falling'] = (df['dominance_change_5d'] < -0.01).astype(int)
    else:
        # Default values
        df['btc_dominance'] = 50
        df['dominance_change_5d'] = 0
        df['dominance_change_20d'] = 0
        df['dominance_vs_sma'] = 0
        df['dominance_rising'] = 0
        df['dominance_falling'] = 0
    
    return df


def add_momentum_swing(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum features для swing"""
    # RSI (longer period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = (100 - (100 / (1 + rs))) / 100 - 0.5
    
    # Weekly RSI
    gain_w = delta.where(delta > 0, 0).rolling(7).mean()
    loss_w = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rs_w = gain_w / (loss_w + 1e-10)
    df['rsi_7'] = (100 - (100 / (1 + rs_w))) / 100 - 0.5
    
    # MACD (weekly settings)
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    df['macd_norm'] = macd / df['close']
    df['macd_signal'] = (macd - signal) / df['close']
    
    # Rate of change
    df['roc_10'] = df['close'].pct_change(10)
    df['roc_20'] = df['close'].pct_change(20)
    df['roc_50'] = df['close'].pct_change(50)
    
    # Momentum divergence
    df['price_momentum'] = df['close'].pct_change(20)
    df['rsi_momentum'] = df['rsi_14'].diff(20)
    
    return df


def add_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """Support/Resistance levels (adaptive windows)"""
    # Adaptive windows
    min_w = max(5, len(df) // 10)
    
    # Recent high/low
    df['high_20d'] = df['high'].rolling(min(20, min_w * 2)).max()
    df['low_20d'] = df['low'].rolling(min(20, min_w * 2)).min()
    df['high_50d'] = df['high'].rolling(min(50, min_w * 4)).max()
    df['low_50d'] = df['low'].rolling(min(50, min_w * 4)).min()
    
    # Position in range
    df['range_position_20d'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)
    df['range_position_50d'] = (df['close'] - df['low_50d']) / (df['high_50d'] - df['low_50d'] + 1e-10)
    
    # Distance to high/low
    df['dist_to_high_20d'] = (df['high_20d'] - df['close']) / df['close']
    df['dist_to_low_20d'] = (df['close'] - df['low_20d']) / df['close']
    
    # Breakout detection
    df['breakout_high'] = (df['close'] > df['high_20d'].shift(1)).astype(int)
    df['breakdown_low'] = (df['close'] < df['low_20d'].shift(1)).astype(int)
    
    return df


def add_volume_swing(df: pd.DataFrame) -> pd.DataFrame:
    """Volume features для swing (adaptive windows)"""
    # Adaptive windows
    min_w = max(5, len(df) // 10)
    
    # Volume trend
    vol_sma_10 = df['volume'].rolling(min(10, min_w)).mean()
    vol_sma_50 = df['volume'].rolling(min(50, min_w * 4)).mean()
    
    df['volume_ratio'] = vol_sma_10 / (vol_sma_50 + 1e-10)
    df['volume_trend'] = vol_sma_10.pct_change(min(10, min_w))
    
    # Accumulation/Distribution
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    df['ad_line'] = (clv * df['volume']).cumsum()
    df['ad_slope'] = df['ad_line'].pct_change(min(20, min_w * 2))
    
    # Chaikin Money Flow
    mf_volume = clv * df['volume']
    df['cmf'] = mf_volume.rolling(min(20, min_w * 2)).sum() / (df['volume'].rolling(min(20, min_w * 2)).sum() + 1e-10)
    
    return df


def add_derivatives_features_swing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derivatives features для Swing модели
    
    Использует данные из data_loader_v2:
    - funding_rate
    - sum_open_interest
    - long_short_ratio, long_account, short_account
    - buy_sell_ratio, buy_vol, sell_vol
    - lastFundingRate (premium index)
    """
    # Adaptive windows for swing
    min_w = max(5, len(df) // 10)
    
    # =========================
    # Funding Rate Features (long-term)
    # =========================
    if 'funding_rate' in df.columns:
        fr = df['funding_rate'].fillna(0)
        
        # Cumulative funding (как индикатор рыночного настроения)
        df['funding_cumsum_7d'] = fr.rolling(min(7 * 3, min_w * 3)).sum()  # ~7 дней (3 funding в день)
        df['funding_cumsum_30d'] = fr.rolling(min(30 * 3, min_w * 6)).sum()  # ~30 дней
        
        # Funding trend
        df['funding_ma_7d'] = fr.rolling(min(21, min_w * 2)).mean()
        df['funding_ma_30d'] = fr.rolling(min(90, min_w * 4)).mean()
        df['funding_trend'] = df['funding_ma_7d'] - df['funding_ma_30d']
        
        # Extreme funding (overheated market)
        funding_std = fr.rolling(min(90, min_w * 4)).std()
        df['funding_zscore_swing'] = (fr - df['funding_ma_30d']) / (funding_std + 1e-10)
        df['funding_extreme_long'] = (df['funding_zscore_swing'] > 2).astype(int)
        df['funding_extreme_short'] = (df['funding_zscore_swing'] < -2).astype(int)
    else:
        df['funding_cumsum_7d'] = 0
        df['funding_cumsum_30d'] = 0
        df['funding_ma_7d'] = 0
        df['funding_ma_30d'] = 0
        df['funding_trend'] = 0
        df['funding_zscore_swing'] = 0
        df['funding_extreme_long'] = 0
        df['funding_extreme_short'] = 0
    
    # =========================
    # Open Interest Features (long-term)
    # =========================
    if 'sum_open_interest' in df.columns:
        oi = df['sum_open_interest'].ffill()
        
        # OI trend
        df['oi_ma_7d'] = oi.rolling(min(7, min_w)).mean()
        df['oi_ma_30d'] = oi.rolling(min(30, min_w * 3)).mean()
        df['oi_trend_swing'] = df['oi_ma_7d'] / (df['oi_ma_30d'] + 1e-10) - 1
        
        # OI change (weekly, monthly)
        df['oi_change_7d'] = oi.pct_change(min(7, min_w))
        df['oi_change_30d'] = oi.pct_change(min(30, min_w * 3))
        
        # OI vs Price divergence (long-term)
        price_change_30d = df['close'].pct_change(min(30, min_w * 3))
        df['oi_price_div_swing'] = df['oi_change_30d'] - price_change_30d
        
        # OI at extremes
        oi_std = oi.rolling(min(90, min_w * 4)).std()
        oi_mean = oi.rolling(min(90, min_w * 4)).mean()
        df['oi_zscore_swing'] = (oi - oi_mean) / (oi_std + 1e-10)
    else:
        df['oi_ma_7d'] = 0
        df['oi_ma_30d'] = 0
        df['oi_trend_swing'] = 0
        df['oi_change_7d'] = 0
        df['oi_change_30d'] = 0
        df['oi_price_div_swing'] = 0
        df['oi_zscore_swing'] = 0
    
    # =========================
    # Long/Short Ratio Features
    # =========================
    if 'long_short_ratio' in df.columns:
        ls = df['long_short_ratio'].fillna(1)
        
        # LS ratio trend
        df['ls_ma_7d'] = ls.rolling(min(7, min_w)).mean()
        df['ls_ma_30d'] = ls.rolling(min(30, min_w * 3)).mean()
        df['ls_trend_swing'] = df['ls_ma_7d'] / (df['ls_ma_30d'] + 1e-10) - 1
        
        # Extreme positioning
        ls_std = ls.rolling(min(90, min_w * 4)).std()
        ls_mean = ls.rolling(min(90, min_w * 4)).mean()
        df['ls_zscore_swing'] = (ls - ls_mean) / (ls_std + 1e-10)
        
        # Crowd positioning extremes
        df['crowd_extreme_long'] = (df['ls_zscore_swing'] > 1.5).astype(int)
        df['crowd_extreme_short'] = (df['ls_zscore_swing'] < -1.5).astype(int)
    else:
        df['ls_ma_7d'] = 1
        df['ls_ma_30d'] = 1
        df['ls_trend_swing'] = 0
        df['ls_zscore_swing'] = 0
        df['crowd_extreme_long'] = 0
        df['crowd_extreme_short'] = 0
    
    # =========================
    # Taker Volume Features  
    # =========================
    if 'buy_sell_ratio' in df.columns:
        bsr = df['buy_sell_ratio'].fillna(1)
        
        # Taker ratio trend
        df['taker_ma_7d'] = bsr.rolling(min(7, min_w)).mean()
        df['taker_ma_30d'] = bsr.rolling(min(30, min_w * 3)).mean()
        df['taker_trend_swing'] = df['taker_ma_7d'] - df['taker_ma_30d']
        
        # Cumulative taker pressure
        df['taker_cumsum_7d'] = (bsr - 1).rolling(min(7, min_w)).sum()
    else:
        df['taker_ma_7d'] = 1
        df['taker_ma_30d'] = 1
        df['taker_trend_swing'] = 0
        df['taker_cumsum_7d'] = 0
    
    # =========================
    # Premium Index / Basis
    # =========================
    if 'lastFundingRate' in df.columns:
        premium = df['lastFundingRate'].fillna(0)
        
        # Premium trend
        df['premium_ma_7d'] = premium.rolling(min(7, min_w)).mean()
        df['premium_ma_30d'] = premium.rolling(min(30, min_w * 3)).mean()
        df['premium_trend_swing'] = df['premium_ma_7d'] - df['premium_ma_30d']
        
        # Basis persistence
        df['basis_persistent_pos'] = (premium > 0).rolling(min(7, min_w)).mean()
        df['basis_persistent_neg'] = (premium < 0).rolling(min(7, min_w)).mean()
    else:
        df['premium_ma_7d'] = 0
        df['premium_ma_30d'] = 0
        df['premium_trend_swing'] = 0
        df['basis_persistent_pos'] = 0.5
        df['basis_persistent_neg'] = 0.5
    
    return df


def add_target_swing(df: pd.DataFrame, horizon: int = 7, threshold: float = 0.03) -> pd.DataFrame:
    """
    Target для swing модели
    
    Args:
        horizon: сколько баров вперёд (7 дней)
        threshold: минимальный порог (3%)
    """
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    
    df['target'] = 0  # flat
    df.loc[future_return > threshold, 'target'] = 1   # up
    df.loc[future_return < -threshold, 'target'] = -1  # down
    
    # Для LightGBM: 0=down, 1=flat, 2=up
    df['target'] = df['target'].map({-1: 0, 0: 1, 1: 2})
    
    df['future_return'] = future_return
    
    return df


def build_swing_features(
    df: pd.DataFrame,
    horizon: int = 7,
    threshold: float = 0.03,
    btc_dominance: pd.Series = None,
    normalize: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Построить все Swing фичи
    """
    initial_len = len(df)
    
    df = add_ma_features(df)
    df = add_market_regime(df)
    df = add_macro_volatility(df)
    df = add_btc_dominance_features(df, btc_dominance)
    df = add_momentum_swing(df)
    df = add_support_resistance(df)
    df = add_volume_swing(df)
    df = add_derivatives_features_swing(df)  # NEW: derivatives features
    df = add_target_swing(df, horizon, threshold)
    
    # Исключаем служебные колонки
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
        'buy_ratio', 'target', 'future_return',
        'sma_20', 'sma_50', 'sma_100', 'sma_200',
        'high_20d', 'low_20d', 'high_50d', 'low_50d',
        'regime', 'regime_duration', 'btc_dominance', 'ad_line',
        'open_time', 'close_time', 'timestamp', 'datetime', 'date', 'time',
        'symbol', 'buy_volume', 'sell_volume', 'trades_count',
        # Raw derivatives columns
        'funding_rate', 'funding_time', 'mark_price',
        'sum_open_interest', 'sum_open_interest_value',
        'long_short_ratio', 'long_account', 'short_account',
        'buy_sell_ratio', 'buy_vol', 'sell_vol',
        'lastFundingRate', 'interestRate', 'indexPrice', 'estimatedSettlePrice',
        'oi_ma_7d', 'oi_ma_30d', 'ls_ma_7d', 'ls_ma_30d',
        'funding_ma_7d', 'funding_ma_30d', 'taker_ma_7d', 'taker_ma_30d',
        'premium_ma_7d', 'premium_ma_30d'
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


SWING_FEATURE_NAMES = [
    'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma100', 'price_vs_sma200',
    'ma_cross_20_50', 'ma_cross_50_100', 'ma_cross_50_200', 'ma_cross_100_200',
    'golden_cross', 'death_cross', 'sma50_slope', 'sma200_slope', 'ma_spread',
    'return_20d', 'return_50d',
    'regime_bull', 'regime_bear', 'regime_flat', 'regime_duration_norm',
    'volatility_20', 'volatility_50', 'volatility_100', 'volatility_annual',
    'vol_regime', 'high_vol_flag', 'low_vol_flag',
    'atr_14', 'atr_50', 'atr_norm', 'atr_ratio', 'bb_width', 'drawdown',
    'dominance_change_5d', 'dominance_change_20d', 'dominance_vs_sma',
    'dominance_rising', 'dominance_falling',
    'rsi_14', 'rsi_7', 'macd_norm', 'macd_signal',
    'roc_10', 'roc_20', 'roc_50', 'price_momentum', 'rsi_momentum',
    'range_position_20d', 'range_position_50d',
    'dist_to_high_20d', 'dist_to_low_20d', 'breakout_high', 'breakdown_low',
    'volume_ratio', 'volume_trend', 'ad_slope', 'cmf',
    # Derivatives features
    'funding_cumsum_7d', 'funding_cumsum_30d', 'funding_trend',
    'funding_zscore_swing', 'funding_extreme_long', 'funding_extreme_short',
    'oi_trend_swing', 'oi_change_7d', 'oi_change_30d', 'oi_price_div_swing', 'oi_zscore_swing',
    'ls_trend_swing', 'ls_zscore_swing', 'crowd_extreme_long', 'crowd_extreme_short',
    'taker_trend_swing', 'taker_cumsum_7d',
    'premium_trend_swing', 'basis_persistent_pos', 'basis_persistent_neg'
]
