"""
Data Loader V2 - загрузка данных Binance Futures из CSV

Поддерживает форматы:
- klines_usdt_m_{SYMBOL}_{TIMEFRAME}.csv - OHLCV свечи
- funding_rate_usdt_m_{SYMBOL}.csv - Funding rates
- open_interest_usdt_m_{SYMBOL}.csv - Open Interest
- long_short_ratio_usdt_m_{SYMBOL}.csv - Long/Short ratio
- taker_volume_usdt_m_{SYMBOL}.csv - Taker buy/sell volume
- mark_price_usdt_m_{SYMBOL}.csv - Mark price OHLC
- premium_index_usdt_m_{SYMBOL}.csv - Premium index
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


# ============== КОНСТАНТЫ ==============
DATA_TYPES = [
    'klines',           # OHLCV свечи
    'funding_rate',     # Funding rate
    'open_interest',    # Open Interest
    'long_short_ratio', # Long/Short ratio
    'taker_volume',     # Taker buy/sell volume
    'mark_price',       # Mark price
    'premium_index'     # Premium index
]

TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']


class BinanceDataLoader:
    """
    Загрузчик данных Binance Futures из CSV файлов
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: путь к директории с CSV файлами
        """
        self.data_dir = data_dir
        self.symbols = []
        self._scan_symbols()
    
    def _scan_symbols(self):
        """Сканировать доступные символы"""
        klines_files = glob.glob(os.path.join(self.data_dir, 'klines_usdt_m_*_1h.csv'))
        symbols = set()
        for f in klines_files:
            basename = os.path.basename(f)
            # klines_usdt_m_BTCUSDT_1h.csv -> BTCUSDT
            parts = basename.replace('.csv', '').split('_')
            if len(parts) >= 4:
                symbol = parts[3]  # klines_usdt_m_SYMBOL_tf
                symbols.add(symbol)
        self.symbols = sorted(list(symbols))
        print(f"Found {len(self.symbols)} symbols: {self.symbols[:10]}...")
    
    # ============== KLINES ==============
    def load_klines(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Загрузить OHLCV данные
        
        Returns:
            DataFrame: timestamp, open, high, low, close, volume, quote_volume, 
                      trades, taker_buy_volume, taker_buy_quote_volume
        """
        filepath = os.path.join(self.data_dir, f'klines_usdt_m_{symbol}_{timeframe}.csv')
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            # Стандартизация колонок
            df = df.rename(columns={
                'timestamp': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'close_time': 'close_time',
                'quote_volume': 'quote_volume',
                'trades': 'trades',
                'taker_buy_volume': 'taker_buy_volume',
                'taker_buy_quote_volume': 'taker_buy_quote_volume'
            })
            
            # Конвертация timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Конвертация типов
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                       'taker_buy_volume', 'taker_buy_quote_volume']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            if 'trades' in df.columns:
                df['trades'] = df['trades'].astype(int)
            
            # Добавляем buy_ratio
            if 'taker_buy_volume' in df.columns and 'volume' in df.columns:
                df['buy_ratio'] = df['taker_buy_volume'] / (df['volume'] + 1e-10)
            
            # Удаляем лишние колонки
            drop_cols = ['close_time', 'symbol', 'interval', 'market_type']
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
            
            return df.sort_index()
            
        except Exception as e:
            print(f"Error loading klines {symbol} {timeframe}: {e}")
            return None
    
    # ============== FUNDING RATE ==============
    def load_funding_rate(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Загрузить Funding Rate данные
        
        Returns:
            DataFrame: funding_rate, mark_price
        """
        filepath = os.path.join(self.data_dir, f'funding_rate_usdt_m_{symbol}.csv')
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            # Стандартизация
            df = df.rename(columns={
                'funding_time': 'timestamp',
                'funding_rate': 'funding_rate',
                'mark_price': 'mark_price'
            })
            
            df['timestamp'] = pd.to_datetime(df['funding_time'] if 'funding_time' in df.columns else df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            df['funding_rate'] = df['funding_rate'].astype(float)
            
            # Удаляем лишние колонки
            drop_cols = ['symbol', 'market_type', 'funding_time']
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
            
            return df.sort_index()
            
        except Exception as e:
            print(f"Error loading funding rate {symbol}: {e}")
            return None
    
    # ============== OPEN INTEREST ==============
    def load_open_interest(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Загрузить Open Interest данные
        
        Returns:
            DataFrame: sum_open_interest, sum_open_interest_value
        """
        filepath = os.path.join(self.data_dir, f'open_interest_usdt_m_{symbol}.csv')
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            for col in ['sum_open_interest', 'sum_open_interest_value']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            # Удаляем лишние колонки
            drop_cols = ['symbol', 'market_type']
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
            
            return df.sort_index()
            
        except Exception as e:
            print(f"Error loading open interest {symbol}: {e}")
            return None
    
    # ============== LONG/SHORT RATIO ==============
    def load_long_short_ratio(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Загрузить Long/Short Ratio данные
        
        Returns:
            DataFrame: long_short_ratio, long_account, short_account
        """
        filepath = os.path.join(self.data_dir, f'long_short_ratio_usdt_m_{symbol}.csv')
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            for col in ['long_short_ratio', 'long_account', 'short_account']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            # Удаляем лишние колонки
            drop_cols = ['symbol', 'period']
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
            
            return df.sort_index()
            
        except Exception as e:
            print(f"Error loading long/short ratio {symbol}: {e}")
            return None
    
    # ============== TAKER VOLUME ==============
    def load_taker_volume(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Загрузить Taker Buy/Sell Volume данные
        
        Returns:
            DataFrame: buy_sell_ratio, buy_vol, sell_vol
        """
        filepath = os.path.join(self.data_dir, f'taker_volume_usdt_m_{symbol}.csv')
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            for col in ['buy_sell_ratio', 'buy_vol', 'sell_vol']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            # Удаляем лишние колонки
            drop_cols = ['symbol', 'period']
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
            
            return df.sort_index()
            
        except Exception as e:
            print(f"Error loading taker volume {symbol}: {e}")
            return None
    
    # ============== MARK PRICE ==============
    def load_mark_price(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Загрузить Mark Price OHLC данные
        
        Returns:
            DataFrame: mark_open, mark_high, mark_low, mark_close
        """
        filepath = os.path.join(self.data_dir, f'mark_price_usdt_m_{symbol}.csv')
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Переименовываем с префиксом mark_
            df = df.rename(columns={
                'open': 'mark_open',
                'high': 'mark_high',
                'low': 'mark_low',
                'close': 'mark_close'
            })
            
            for col in ['mark_open', 'mark_high', 'mark_low', 'mark_close']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            # Удаляем лишние колонки
            drop_cols = ['symbol', 'market_type']
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
            
            return df.sort_index()
            
        except Exception as e:
            print(f"Error loading mark price {symbol}: {e}")
            return None
    
    # ============== PREMIUM INDEX ==============
    def load_premium_index(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Загрузить Premium Index OHLC данные
        
        Returns:
            DataFrame: premium_open, premium_high, premium_low, premium_close
        """
        filepath = os.path.join(self.data_dir, f'premium_index_usdt_m_{symbol}.csv')
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Переименовываем с префиксом premium_
            df = df.rename(columns={
                'open': 'premium_open',
                'high': 'premium_high',
                'low': 'premium_low',
                'close': 'premium_close'
            })
            
            for col in ['premium_open', 'premium_high', 'premium_low', 'premium_close']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            # Удаляем лишние колонки
            drop_cols = ['symbol', 'interval']
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
            
            return df.sort_index()
            
        except Exception as e:
            print(f"Error loading premium index {symbol}: {e}")
            return None
    
    # ============== ПОЛНАЯ ЗАГРУЗКА СИМВОЛА ==============
    def load_symbol_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        include_derivatives: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Загрузить все данные для символа и объединить их
        
        Args:
            symbol: торговый символ (BTCUSDT)
            timeframe: таймфрейм для klines
            include_derivatives: включить funding, OI, LS ratio и т.д.
            
        Returns:
            Объединенный DataFrame с всеми данными
        """
        # Основные данные - klines
        df = self.load_klines(symbol, timeframe)
        if df is None or df.empty:
            return None
        
        if not include_derivatives:
            return df
        
        # Добавляем производные данные (merge по ближайшему timestamp)
        
        # Funding Rate (каждые 8 часов)
        funding = self.load_funding_rate(symbol)
        if funding is not None and not funding.empty:
            funding = funding[['funding_rate']].copy()
            df = pd.merge_asof(
                df.reset_index(),
                funding.reset_index(),
                on='timestamp',
                direction='backward'
            ).set_index('timestamp')
            df['funding_rate'] = df['funding_rate'].ffill().fillna(0)
        
        # Open Interest (5min данные)
        oi = self.load_open_interest(symbol)
        if oi is not None and not oi.empty:
            oi = oi[['sum_open_interest', 'sum_open_interest_value']].copy()
            df = pd.merge_asof(
                df.reset_index(),
                oi.reset_index(),
                on='timestamp',
                direction='backward'
            ).set_index('timestamp')
            for col in ['sum_open_interest', 'sum_open_interest_value']:
                if col in df.columns:
                    df[col] = df[col].ffill().fillna(0)
        
        # Long/Short Ratio (5min данные)
        ls = self.load_long_short_ratio(symbol)
        if ls is not None and not ls.empty:
            ls = ls[['long_short_ratio', 'long_account', 'short_account']].copy()
            df = pd.merge_asof(
                df.reset_index(),
                ls.reset_index(),
                on='timestamp',
                direction='backward'
            ).set_index('timestamp')
            for col in ['long_short_ratio', 'long_account', 'short_account']:
                if col in df.columns:
                    df[col] = df[col].ffill().fillna(1.0)
        
        # Taker Volume (5min данные)
        taker = self.load_taker_volume(symbol)
        if taker is not None and not taker.empty:
            taker = taker[['buy_sell_ratio', 'buy_vol', 'sell_vol']].copy()
            df = pd.merge_asof(
                df.reset_index(),
                taker.reset_index(),
                on='timestamp',
                direction='backward'
            ).set_index('timestamp')
            for col in ['buy_sell_ratio', 'buy_vol', 'sell_vol']:
                if col in df.columns:
                    df[col] = df[col].ffill().fillna(1.0)
        
        # Mark Price (1h данные)
        mark = self.load_mark_price(symbol)
        if mark is not None and not mark.empty:
            mark = mark[['mark_close']].copy()
            df = pd.merge_asof(
                df.reset_index(),
                mark.reset_index(),
                on='timestamp',
                direction='backward'
            ).set_index('timestamp')
            df['mark_close'] = df['mark_close'].ffill()
            # Basis = spot - futures (прокси через mark vs close)
            if 'mark_close' in df.columns:
                df['basis'] = (df['close'] - df['mark_close']) / df['mark_close'] * 100
        
        # Premium Index (1h данные)
        premium = self.load_premium_index(symbol)
        if premium is not None and not premium.empty:
            premium = premium[['premium_close']].copy()
            df = pd.merge_asof(
                df.reset_index(),
                premium.reset_index(),
                on='timestamp',
                direction='backward'
            ).set_index('timestamp')
            df['premium_close'] = df['premium_close'].ffill().fillna(0)
        
        return df
    
    # ============== МУЛЬТИ-ТАЙМФРЕЙМ ЗАГРУЗКА ==============
    def load_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str] = ['5m', '15m', '1h', '4h', '1d'],
        include_derivatives: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Загрузить данные для нескольких таймфреймов
        
        Returns:
            Dict[timeframe] = DataFrame
        """
        data = {}
        
        for tf in timeframes:
            df = self.load_symbol_data(symbol, tf, include_derivatives)
            if df is not None and not df.empty:
                data[tf] = df
        
        return data
    
    # ============== ЗАГРУЗКА ВСЕХ СИМВОЛОВ ==============
    def load_all_symbols(
        self,
        timeframes: List[str] = ['1h'],
        symbols: List[str] = None,
        include_derivatives: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Загрузить данные для всех символов
        
        Returns:
            Dict[symbol][timeframe] = DataFrame
        """
        all_data = {}
        
        symbols = symbols or self.symbols
        
        for i, symbol in enumerate(symbols):
            print(f"Loading {symbol} ({i+1}/{len(symbols)})...")
            
            symbol_data = self.load_multi_timeframe(
                symbol, 
                timeframes, 
                include_derivatives
            )
            
            if symbol_data:
                all_data[symbol] = symbol_data
        
        return all_data


# ============== ФУНКЦИИ ДЛЯ ОБУЧЕНИЯ ==============

def prepare_training_data_v2(
    data_dir: str = None,
    timeframes: List[str] = ['5m', '15m', '1h', '4h', '1d'],
    symbols: List[str] = None,
    include_derivatives: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Подготовить данные для обучения (новая версия)
    
    Args:
        data_dir: путь к директории с CSV
        timeframes: список таймфреймов
        symbols: фильтр символов (None = все)
        include_derivatives: включить funding, OI и т.д.
        
    Returns:
        Dict[symbol][timeframe] = DataFrame
    """
    if data_dir is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return {}
    
    print(f"Loading data from: {data_dir}")
    
    loader = BinanceDataLoader(data_dir)
    
    return loader.load_all_symbols(
        timeframes=timeframes,
        symbols=symbols,
        include_derivatives=include_derivatives
    )


def get_combined_data(
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
    
    combined = pd.concat(dfs)
    combined = combined.sort_index()
    
    return combined, symbols


# ============== ТЕСТИРОВАНИЕ ==============
if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    
    print(f"Data directory: {data_dir}")
    print()
    
    loader = BinanceDataLoader(data_dir)
    
    # Тест загрузки одного символа
    print("\n" + "="*60)
    print("TEST: Load BTCUSDT with all derivatives")
    print("="*60)
    
    df = loader.load_symbol_data('BTCUSDT', '1h', include_derivatives=True)
    
    if df is not None:
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"\nSample data:")
        print(df.tail())
        
        print(f"\nMissing values:")
        print(df.isnull().sum())
    
    # Тест мульти-таймфрейм
    print("\n" + "="*60)
    print("TEST: Load BTCUSDT multi-timeframe")
    print("="*60)
    
    mtf_data = loader.load_multi_timeframe('BTCUSDT', ['5m', '1h', '4h', '1d'])
    
    for tf, df in mtf_data.items():
        print(f"  {tf}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    
    # Тест полной загрузки
    print("\n" + "="*60)
    print("TEST: Load all symbols (first 5)")
    print("="*60)
    
    all_data = loader.load_all_symbols(
        timeframes=['1h'],
        symbols=loader.symbols[:5]
    )
    
    print(f"\nLoaded {len(all_data)} symbols:")
    for symbol, tf_data in all_data.items():
        for tf, df in tf_data.items():
            print(f"  {symbol} {tf}: {len(df)} rows")
