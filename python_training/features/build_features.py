"""
Feature Engineering для криптотрейдинга

Создаём фичи, которые реально работают на крипторынке:
- Price action features
- Volume features  
- Momentum indicators
- Volatility features
"""
import pandas as pd
import numpy as np
from typing import List, Tuple


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ценовые фичи
    """
    # Returns (логарифмические - более стабильные)
    df['return_1'] = np.log(df['close'] / df['close'].shift(1))
    df['return_5'] = np.log(df['close'] / df['close'].shift(5))
    df['return_15'] = np.log(df['close'] / df['close'].shift(15))
    df['return_30'] = np.log(df['close'] / df['close'].shift(30))
    
    # Price position in bar
    df['bar_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # Bar size
    df['bar_range'] = (df['high'] - df['low']) / df['close']
    df['bar_body'] = abs(df['close'] - df['open']) / df['close']
    
    # Gap
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Объёмные фичи
    """
    # Volume change
    df['volume_ratio_5'] = df['volume_btc'] / df['volume_btc'].rolling(5).mean()
    df['volume_ratio_15'] = df['volume_btc'] / df['volume_btc'].rolling(15).mean()
    
    # Volume trend
    df['volume_sma_5'] = df['volume_btc'].rolling(5).mean()
    df['volume_sma_15'] = df['volume_btc'].rolling(15).mean()
    df['volume_trend'] = df['volume_sma_5'] / df['volume_sma_15']
    
    # Trade intensity
    df['trades_ratio'] = df['num_trades'] / df['num_trades'].rolling(10).mean()
    
    # Buy pressure (очень важная фича!)
    df['buy_pressure'] = df['buy_ratio'] - 0.5  # центрируем вокруг 0
    df['buy_pressure_sma'] = df['buy_pressure'].rolling(5).mean()
    
    # Volume-weighted price move
    df['vwap_distance'] = (df['close'] - df['volume_btc'].rolling(10).apply(
        lambda x: np.average(df.loc[x.index, 'close'], weights=x)
    )) / df['close']
    
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Моментум индикаторы
    """
    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_15'] = df['close'].rolling(15).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    
    # Price vs SMA
    df['price_vs_sma5'] = (df['close'] - df['sma_5']) / df['sma_5']
    df['price_vs_sma15'] = (df['close'] - df['sma_15']) / df['sma_15']
    df['price_vs_sma30'] = (df['close'] - df['sma_30']) / df['sma_30']
    
    # SMA crossovers
    df['sma_cross_5_15'] = df['sma_5'] / df['sma_15'] - 1
    df['sma_cross_5_30'] = df['sma_5'] / df['sma_30'] - 1
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_normalized'] = (df['rsi_14'] - 50) / 50  # -1 to 1
    
    # Rate of Change
    df['roc_5'] = df['close'].pct_change(5)
    df['roc_15'] = df['close'].pct_change(15)
    
    # Momentum
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_15'] = df['close'] - df['close'].shift(15)
    
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Волатильность
    """
    # Standard deviation of returns
    df['volatility_5'] = df['return_1'].rolling(5).std()
    df['volatility_15'] = df['return_1'].rolling(15).std()
    df['volatility_30'] = df['return_1'].rolling(30).std()
    
    # Volatility ratio (текущая vs историческая)
    df['vol_ratio'] = df['volatility_5'] / df['volatility_30']
    
    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_normalized'] = df['atr_14'] / df['close']
    
    # Bollinger Bands position
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
    
    return df


def add_target(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0) -> pd.DataFrame:
    """
    Создание target для классификации
    
    Args:
        df: DataFrame с фичами
        horizon: через сколько баров смотрим
        threshold: минимальное изменение для класса (0 = любое)
    
    Target:
        1 = цена вырастет
        0 = цена упадёт
    """
    # Future return
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
    
    # Binary target
    if threshold > 0:
        # Трёхклассовая: up / neutral / down
        df['target'] = 0  # neutral
        df.loc[df['future_return'] > threshold, 'target'] = 1   # up
        df.loc[df['future_return'] < -threshold, 'target'] = -1  # down
    else:
        # Бинарная: up / down
        df['target'] = (df['future_return'] > 0).astype(int)
    
    return df


def build_all_features(df: pd.DataFrame, horizon: int = 5) -> Tuple[pd.DataFrame, List[str]]:
    """
    Строим все фичи и возвращаем готовый датасет
    
    Returns:
        df: DataFrame с фичами и target
        feature_names: список названий фичей
    """
    print("Building features...")
    
    df = add_price_features(df)
    df = add_volume_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_target(df, horizon=horizon)
    
    # Убираем NaN
    df = df.dropna()
    
    # Список фичей (исключаем служебные колонки)
    exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 
                    'volume_usd', 'volume_btc', 'num_trades',
                    'sell_trades', 'total_trades', 'buy_trades',
                    'sma_5', 'sma_15', 'sma_30', 'bb_upper', 'bb_lower',
                    'future_return', 'target']
    
    feature_names = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Total features: {len(feature_names)}")
    print(f"Samples: {len(df):,}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df, feature_names


if __name__ == "__main__":
    # Тест
    import sys
    sys.path.append('..')
    from utils import load_trades, resample_to_bars
    
    trades = load_trades('../data/BTCUSD_200925-trades-2020-07.csv')
    bars = resample_to_bars(trades, freq='1min')
    df, features = build_all_features(bars)
    
    print("\nFeatures:")
    for f in features:
        print(f"  - {f}")
