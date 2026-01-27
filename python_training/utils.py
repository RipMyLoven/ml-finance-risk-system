"""
Вспомогательные функции для работы с данными
"""
import pandas as pd
import numpy as np
import os
import glob
from typing import Tuple, List, Union, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# Стандартные названия колонок (приводим все файлы к единому формату)
STANDARD_COLUMNS = ['trade_id', 'price', 'qty_usd', 'qty_btc', 'timestamp_ms', 'is_buyer_maker']


def detect_csv_format(filepath: str) -> dict:
    """
    Автоматическое определение формата CSV файла
    """
    # Читаем первые строки
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
    
    # Проверяем есть ли заголовок
    first_fields = first_line.split(',')
    has_header = not first_fields[0].replace('.', '').isdigit()
    
    if has_header:
        # Файл с заголовком
        headers = first_fields
        return {
            'has_header': True,
            'headers': headers,
            'skiprows': 0
        }
    else:
        # Файл без заголовка
        return {
            'has_header': False,
            'headers': None,
            'skiprows': 0
        }


def normalize_dataframe(df: pd.DataFrame, source_file: str = None) -> pd.DataFrame:
    """
    Приведение DataFrame к стандартному формату
    """
    # Возможные названия колонок в разных форматах
    column_mapping = {
        # trade_id
        'id': 'trade_id',
        'trade_id': 'trade_id',
        'tid': 'trade_id',
        
        # price
        'price': 'price',
        'p': 'price',
        
        # quantity USD
        'qty': 'qty_usd',
        'qty_usd': 'qty_usd',
        'quantity': 'qty_usd',
        'amount': 'qty_usd',
        
        # quantity base (BTC, SOL, etc)
        'base_qty': 'qty_btc',
        'qty_btc': 'qty_btc',
        'base_quantity': 'qty_btc',
        'size': 'qty_btc',
        
        # timestamp
        'time': 'timestamp_ms',
        'timestamp': 'timestamp_ms',
        'timestamp_ms': 'timestamp_ms',
        'ts': 'timestamp_ms',
        
        # side
        'is_buyer_maker': 'is_buyer_maker',
        'side': 'is_buyer_maker',
        'm': 'is_buyer_maker',
    }
    
    # Переименовываем колонки
    new_columns = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower in column_mapping:
            new_columns[col] = column_mapping[col_lower]
        elif isinstance(col, int):
            # Колонки по номерам (файл без заголовка)
            if col < len(STANDARD_COLUMNS):
                new_columns[col] = STANDARD_COLUMNS[col]
    
    df = df.rename(columns=new_columns)
    
    # Добавляем source file если нужно
    if source_file:
        df['source_file'] = os.path.basename(source_file)
    
    return df


def load_single_csv(
    filepath: str,
    chunksize: Optional[int] = None
) -> pd.DataFrame:
    """
    Загрузка одного CSV файла с автоопределением формата
    """
    fmt = detect_csv_format(filepath)
    
    if fmt['has_header']:
        # Файл с заголовком
        if chunksize:
            chunks = []
            for chunk in pd.read_csv(filepath, chunksize=chunksize):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(filepath)
    else:
        # Файл без заголовка
        if chunksize:
            chunks = []
            for chunk in pd.read_csv(filepath, header=None, names=STANDARD_COLUMNS, chunksize=chunksize):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(filepath, header=None, names=STANDARD_COLUMNS)
    
    # Нормализуем
    df = normalize_dataframe(df, filepath)
    
    return df


def load_trades(
    path: Union[str, List[str]],
    pattern: str = "*.csv",
    chunksize: int = 500_000,
    max_workers: int = 4,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Загрузка trades из одного или нескольких CSV файлов
    
    Args:
        path: Путь к файлу, папке, или список путей к файлам
        pattern: Glob паттерн для поиска файлов в папке (по умолчанию *.csv)
        chunksize: Размер чанка для чтения больших файлов (None для отключения)
        max_workers: Количество потоков для параллельной загрузки
        verbose: Выводить прогресс
    
    Returns:
        DataFrame со всеми trades
    
    Examples:
        # Один файл
        df = load_trades("data/trades.csv")
        
        # Папка с файлами
        df = load_trades("data/", pattern="*-trades-*.csv")
        
        # Список файлов
        df = load_trades(["data/btc.csv", "data/eth.csv"])
        
        # Glob паттерн
        df = load_trades("data/**/trades*.csv")
    """
    # Собираем список файлов
    if isinstance(path, list):
        files = path
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, pattern))
    elif '*' in path or '?' in path:
        files = glob.glob(path, recursive=True)
    else:
        files = [path]
    
    if not files:
        raise ValueError(f"No CSV files found at: {path}")
    
    files = sorted(files)  # Сортируем для воспроизводимости
    
    if verbose:
        print(f"Found {len(files)} CSV file(s) to load:")
        for f in files[:5]:
            print(f"  - {os.path.basename(f)}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
    
    # Загружаем файлы
    all_dfs = []
    
    if len(files) == 1:
        # Один файл - просто загружаем
        df = load_single_csv(files[0], chunksize=chunksize)
        all_dfs.append(df)
    else:
        # Много файлов - параллельная загрузка
        if verbose:
            pbar = tqdm(total=len(files), desc="Loading CSVs")
        
        def load_with_progress(filepath):
            return load_single_csv(filepath, chunksize=chunksize)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_with_progress, f): f for f in files}
            
            for future in as_completed(futures):
                filepath = futures[future]
                try:
                    df = future.result()
                    all_dfs.append(df)
                    if verbose:
                        pbar.update(1)
                except Exception as e:
                    print(f"\n⚠ Error loading {filepath}: {e}")
        
        if verbose:
            pbar.close()
    
    # Объединяем все DataFrame
    if verbose:
        print("Merging dataframes...")
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Конвертация timestamp в datetime
    if 'timestamp_ms' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    # Сортируем по времени
    df = df.sort_values('timestamp_ms').reset_index(drop=True)
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"✓ Загружено {len(df):,} trades")
        print(f"  Период: {df['datetime'].min()} — {df['datetime'].max()}")
        print(f"  Цена: {df['price'].min():.4f} — {df['price'].max():.4f}")
        if 'source_file' in df.columns:
            print(f"  Файлов: {df['source_file'].nunique()}")
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"  Память: {mem_mb:.1f} MB")
        print(f"{'='*50}")
    
    return df


def load_trades_lazy(
    path: Union[str, List[str]],
    pattern: str = "*.csv",
    chunksize: int = 100_000
):
    """
    Ленивая загрузка trades (генератор чанков)
    
    Используй когда данные не помещаются в память.
    
    Example:
        for chunk in load_trades_lazy("data/huge_file.csv"):
            process(chunk)
    """
    # Собираем файлы
    if isinstance(path, list):
        files = path
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, pattern))
    elif '*' in path:
        files = glob.glob(path, recursive=True)
    else:
        files = [path]
    
    files = sorted(files)
    
    for filepath in files:
        fmt = detect_csv_format(filepath)
        
        if fmt['has_header']:
            reader = pd.read_csv(filepath, chunksize=chunksize)
        else:
            reader = pd.read_csv(filepath, header=None, names=STANDARD_COLUMNS, chunksize=chunksize)
        
        for chunk in reader:
            chunk = normalize_dataframe(chunk, filepath)
            if 'timestamp_ms' in chunk.columns:
                chunk['datetime'] = pd.to_datetime(chunk['timestamp_ms'], unit='ms')
            yield chunk


def resample_to_bars(df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """
    Агрегация trades в OHLCV бары
    
    Args:
        df: DataFrame с trades
        freq: частота баров ('1min', '5min', '1h', etc.)
    
    Returns:
        DataFrame с OHLCV барами
    """
    df = df.set_index('datetime')
    
    bars = df.groupby(pd.Grouper(freq=freq)).agg({
        'price': ['first', 'max', 'min', 'last'],
        'qty_usd': 'sum',
        'qty_btc': 'sum',
        'trade_id': 'count',
        'is_buyer_maker': ['sum', 'count']  # для расчёта buy/sell volume
    })
    
    # Плоские названия колонок
    bars.columns = ['open', 'high', 'low', 'close', 
                    'volume_usd', 'volume_btc', 'num_trades',
                    'sell_trades', 'total_trades']
    
    # Buy volume / Sell volume
    bars['buy_trades'] = bars['total_trades'] - bars['sell_trades']
    bars['buy_ratio'] = bars['buy_trades'] / bars['total_trades'].replace(0, 1)
    
    # Убираем пустые бары
    bars = bars.dropna()
    bars = bars[bars['volume_btc'] > 0]
    
    return bars.reset_index()


def train_test_split_temporal(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Walk-forward split (НЕ случайный!)
    
    Для временных рядов нельзя делать случайный split,
    иначе будет look-ahead bias.
    """
    split_idx = int(len(X) * (1 - test_ratio))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"Train: {len(X_train):,} samples ({X_train.index.min()} — {X_train.index.max()})")
    print(f"Test:  {len(X_test):,} samples ({X_test.index.min()} — {X_test.index.max()})")
    
    return X_train, X_test, y_train, y_test
