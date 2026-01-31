"""
Data Loader - загрузка и агрегация trade данных из CSV

Конвертирует trade-level данные в OHLCV формат для обучения моделей.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def load_trades_csv(filepath: str) -> pd.DataFrame:
    """
    Загрузить trade данные из CSV
    
    Format:
        id,price,qty,base_qty,time,is_buyer_maker
    """
    df = pd.read_csv(filepath)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    df['price'] = df['price'].astype(float)
    df['qty'] = df['qty'].astype(float)
    df['base_qty'] = df['base_qty'].astype(float)
    
    return df


def aggregate_to_ohlcv(
    trades: pd.DataFrame,
    timeframe: str = '5m'
) -> pd.DataFrame:
    """
    Агрегировать trades в OHLCV
    
    Args:
        trades: DataFrame с колонками [timestamp, price, qty, base_qty, is_buyer_maker]
        timeframe: таймфрейм агрегации ('1m', '5m', '15m', '1h', '4h', '1d')
        
    Returns:
        DataFrame с колонками [timestamp, open, high, low, close, volume, 
                               buy_volume, sell_volume, trades_count]
    """
    # Map timeframe to pandas resample rule
    # New pandas uses lowercase: 'min', 'h', 'D', 'W'
    tf_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '4h': '4h',
        '12h': '12h',
        '1d': '1D',
        '3d': '3D',
        '1w': '1W'
    }
    
    rule = tf_map.get(timeframe, '5T')
    
    # Set timestamp as index
    trades = trades.set_index('timestamp')
    
    # Aggregate OHLCV
    ohlcv = pd.DataFrame()
    ohlcv['open'] = trades['price'].resample(rule).first()
    ohlcv['high'] = trades['price'].resample(rule).max()
    ohlcv['low'] = trades['price'].resample(rule).min()
    ohlcv['close'] = trades['price'].resample(rule).last()
    ohlcv['volume'] = trades['qty'].resample(rule).sum()
    
    # Buy/Sell volume (is_buyer_maker=True means seller initiated)
    trades['buy_volume'] = trades['qty'].where(~trades['is_buyer_maker'], 0)
    trades['sell_volume'] = trades['qty'].where(trades['is_buyer_maker'], 0)
    ohlcv['buy_volume'] = trades['buy_volume'].resample(rule).sum()
    ohlcv['sell_volume'] = trades['sell_volume'].resample(rule).sum()
    
    # Trade count
    ohlcv['trades_count'] = trades['price'].resample(rule).count()
    
    # Forward fill для пропусков в данных вместо удаления
    # Это сохранит непрерывность временного ряда
    ohlcv = ohlcv.ffill().bfill()
    
    # Удаляем только строки где ВСЕ OHLCV = NaN (полностью пустые периоды)
    ohlcv = ohlcv.dropna(subset=['open', 'close'], how='all')
    
    # Reset index
    ohlcv = ohlcv.reset_index()
    ohlcv = ohlcv.rename(columns={'timestamp': 'open_time'})
    
    return ohlcv


def extract_symbol_from_filename(filename: str) -> str:
    """
    Извлечь символ из имени файла
    
    Example: BTCUSD_260327-trades-2025-10.csv -> BTCUSD
    """
    basename = os.path.basename(filename)
    # Pattern: SYMBOL_EXPIRY-trades-YYYY-MM.csv
    parts = basename.split('_')
    if len(parts) >= 1:
        return parts[0]
    return "UNKNOWN"


def load_all_data(
    data_dir: str,
    timeframes: List[str] = ['5m', '1h', '1d'],
    symbols: List[str] = None
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Загрузить все данные из директории
    
    Args:
        data_dir: путь к директории с CSV файлами
        timeframes: список таймфреймов для агрегации
        symbols: фильтр по символам (если None - все)
        
    Returns:
        Dict[symbol][timeframe] = DataFrame
    """
    all_data = {}
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return all_data
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Group files by symbol
    symbol_files = {}
    for filepath in csv_files:
        symbol = extract_symbol_from_filename(filepath)
        if symbols is None or symbol in symbols:
            if symbol not in symbol_files:
                symbol_files[symbol] = []
            symbol_files[symbol].append(filepath)
    
    print(f"Processing {len(symbol_files)} symbols: {list(symbol_files.keys())}")
    
    # Process each symbol
    for symbol, files in symbol_files.items():
        print(f"\n  Loading {symbol} ({len(files)} files)...")
        
        # Load and concatenate all files for this symbol
        all_trades = []
        for filepath in sorted(files):
            try:
                trades = load_trades_csv(filepath)
                all_trades.append(trades)
                print(f"    Loaded {os.path.basename(filepath)}: {len(trades)} trades")
            except Exception as e:
                print(f"    Error loading {filepath}: {e}")
        
        if not all_trades:
            continue
        
        # Concatenate all trades
        combined_trades = pd.concat(all_trades, ignore_index=True)
        combined_trades = combined_trades.sort_values('timestamp')
        combined_trades = combined_trades.drop_duplicates(subset=['id'])
        
        print(f"    Total: {len(combined_trades)} trades from {combined_trades['timestamp'].min()} to {combined_trades['timestamp'].max()}")
        
        # Aggregate to different timeframes
        symbol_data = {}
        for tf in timeframes:
            try:
                ohlcv = aggregate_to_ohlcv(combined_trades.copy(), tf)
                if len(ohlcv) > 0:
                    symbol_data[tf] = ohlcv
                    print(f"    {tf}: {len(ohlcv)} candles")
            except Exception as e:
                print(f"    Error aggregating {tf}: {e}")
        
        if symbol_data:
            all_data[symbol] = symbol_data
    
    return all_data


def prepare_training_data(
    data_dir: str = None,
    output_dir: str = None,
    timeframes: List[str] = ['5m', '15m', '1h', '4h', '1d']
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Подготовить данные для обучения
    
    Args:
        data_dir: путь к директории с CSV (по умолчанию ../data относительно project/)
        output_dir: куда сохранять обработанные данные
        
    Returns:
        Dict[symbol][timeframe] = DataFrame
    """
    if data_dir is None:
        # Default: look in parent directory's data folder
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return {}
    
    print(f"Loading data from: {data_dir}")
    
    # Load all data
    data = load_all_data(data_dir, timeframes)
    
    # Optionally save processed data
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for symbol, tf_data in data.items():
            for tf, df in tf_data.items():
                filepath = os.path.join(output_dir, f"{symbol}_{tf}.csv")
                df.to_csv(filepath, index=False)
                print(f"Saved {filepath}")
    
    return data


def get_combined_data_for_timeframe(
    data: Dict[str, Dict[str, pd.DataFrame]],
    timeframe: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Объединить данные всех символов для одного таймфрейма
    
    Returns:
        (combined_df, list of symbols)
    """
    dfs = []
    symbols = []
    
    for symbol, tf_data in data.items():
        if timeframe in tf_data:
            df = tf_data[timeframe].copy()
            df['symbol'] = symbol
            dfs.append(df)
            symbols.append(symbol)
    
    if not dfs:
        return pd.DataFrame(), []
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('open_time')
    
    return combined, symbols


if __name__ == "__main__":
    # Test loading
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    
    print(f"Data directory: {data_dir}")
    
    data = prepare_training_data(data_dir)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    for symbol, tf_data in data.items():
        print(f"\n{symbol}:")
        for tf, df in tf_data.items():
            print(f"  {tf}: {len(df)} candles, {df['open_time'].min()} to {df['open_time'].max()}")
