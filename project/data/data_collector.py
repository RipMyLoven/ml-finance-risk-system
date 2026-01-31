"""
Data Collector - сбор данных с Binance API

Собирает:
- OHLCV
- Funding rate
- Open Interest
- Volume data
- BTC correlation
- BTC dominance

Поддерживает работу:
- С API ключами (приватные данные)
- Без ключей (только публичные данные через REST API)

Использует все ядра CPU для параллельного сбора данных.
"""

import os
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Количество доступных ядер CPU
N_CORES = mp.cpu_count()

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("Warning: python-binance not installed. Using direct API calls.")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET,
    TF_MAPPING, DATA_LOOKBACK_DAYS, MIN_VOLUME_USDT,
    SCALP_TIMEFRAMES, INTRADAY_TIMEFRAMES, SWING_TIMEFRAMES,
    TOP_SYMBOLS
)


class BinanceDataCollector:
    """
    Сбор данных с Binance Futures API
    
    Работает в двух режимах:
    1. С python-binance библиотекой (если установлена)
    2. Через прямые REST API запросы (fallback)
    """
    
    # API endpoints
    SPOT_BASE_URL = "https://api.binance.com"
    FUTURES_BASE_URL = "https://fapi.binance.com"
    
    # Timeframe mappings for API
    TF_MINUTES = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
        "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080
    }
    
    def __init__(self, api_key: str = None, api_secret: str = None, n_workers: int = -1):
        self.api_key = api_key or BINANCE_API_KEY
        self.api_secret = api_secret or BINANCE_API_SECRET
        
        # Parallel processing settings
        self.n_workers = N_CORES if n_workers == -1 else min(n_workers, N_CORES)
        
        # Setup session with retries
        self.session = self._create_session()
        
        # Try to use binance client
        self.client = None
        if BINANCE_AVAILABLE:
            try:
                if self.api_key and self.api_secret:
                    self.client = Client(self.api_key, self.api_secret)
                else:
                    # Public client without keys
                    self.client = Client("", "")
                # Test connection
                self.client.ping()
                print("Binance client connected successfully")
            except Exception as e:
                print(f"Binance client error: {e}")
                print("Falling back to direct API requests")
                self.client = None
        
        self.base_url = self.FUTURES_BASE_URL
    
    def _create_session(self) -> requests.Session:
        """Create session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    def _api_request(self, url: str, params: dict = None) -> Optional[dict]:
        """Make API request with error handling"""
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return None
    
    def get_all_futures_symbols(self) -> List[str]:
        """Получить все торгуемые фьючерсные пары"""
        # Try binance client first
        if self.client:
            try:
                info = self.client.futures_exchange_info()
                symbols = [
                    s['symbol'] for s in info['symbols']
                    if s['status'] == 'TRADING' and s['symbol'].endswith('USDT')
                ]
                return symbols
            except Exception as e:
                print(f"Client error: {e}, trying direct API...")
        
        # Fallback to direct API
        data = self._api_request(f"{self.FUTURES_BASE_URL}/fapi/v1/exchangeInfo")
        if data and 'symbols' in data:
            return [
                s['symbol'] for s in data['symbols']
                if s['status'] == 'TRADING' and s['symbol'].endswith('USDT')
            ]
        
        return TOP_SYMBOLS
    
    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Получить OHLCV данные
        
        Args:
            symbol: Торговая пара (BTCUSDT)
            interval: Таймфрейм (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Начало периода
            end_time: Конец периода
            limit: Максимум свечей
            
        Returns:
            DataFrame: timestamp, open, high, low, close, volume, quote_volume,
                      trades, taker_buy_base, taker_buy_quote
        """
        klines = None
        
        # Try binance client first
        if self.client:
            try:
                start_ms = int(start_time.timestamp() * 1000) if start_time else None
                end_ms = int(end_time.timestamp() * 1000) if end_time else None
                
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=start_ms,
                    endTime=end_ms,
                    limit=limit
                )
            except Exception as e:
                print(f"Client error for {symbol}: {e}, trying direct API...")
        
        # Fallback to direct API
        if klines is None:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)
            
            klines = self._api_request(f"{self.FUTURES_BASE_URL}/fapi/v1/klines", params)
        
        if not klines:
            return pd.DataFrame()
        
        # Parse klines data
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume',
                   'quote_volume', 'taker_buy_base', 'taker_buy_quote']:
            df[col] = df[col].astype(float)
        
        df['trades'] = df['trades'].astype(int)
        df.drop(['close_time', 'ignore'], axis=1, inplace=True)
        
        # Добавляем buy_ratio (важно для фичей!)
        df['buy_ratio'] = df['taker_buy_base'] / (df['volume'] + 1e-10)
        
        return df
    
    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        days: int = None
    ) -> pd.DataFrame:
        """
        Получить исторические данные (обходит лимит 1000 свечей)
        """
        days = days or DATA_LOOKBACK_DAYS
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            df = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000
            )
            
            if df.empty:
                break
            
            all_data.append(df)
            current_start = df.index[-1] + timedelta(minutes=1)
            time.sleep(0.1)  # Rate limit
        
        if all_data:
            result = pd.concat(all_data)
            result = result[~result.index.duplicated(keep='first')]
            return result.sort_index()
        
        return pd.DataFrame()
    
    def get_funding_rate(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Получить историю funding rate"""
        data = None
        
        # Try binance client
        if self.client:
            try:
                data = self.client.futures_funding_rate(symbol=symbol, limit=limit)
            except Exception as e:
                print(f"Client error for funding rate: {e}")
        
        # Fallback to direct API
        if data is None:
            data = self._api_request(
                f"{self.FUTURES_BASE_URL}/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": limit}
            )
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = df['fundingRate'].astype(float)
            df.set_index('fundingTime', inplace=True)
        
        return df
    
    def get_open_interest(self, symbol: str, interval: str = "5m", limit: int = 500) -> pd.DataFrame:
        """Получить историю Open Interest"""
        data = self._api_request(
            f"{self.FUTURES_BASE_URL}/futures/data/openInterestHist",
            params={
                "symbol": symbol,
                "period": interval,
                "limit": limit
            }
        )
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
            df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_long_short_ratio(self, symbol: str, period: str = "5m", limit: int = 100) -> pd.DataFrame:
        """Получить Long/Short ratio"""
        data = self._api_request(
            f"{self.FUTURES_BASE_URL}/futures/data/globalLongShortAccountRatio",
            params={
                "symbol": symbol,
                "period": period,
                "limit": limit
            }
        )
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['longShortRatio'] = df['longShortRatio'].astype(float)
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_btc_dominance(self) -> float:
        """Получить BTC dominance с CoinGecko"""
        try:
            response = self.session.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data['data']['market_cap_percentage']['btc']
        except Exception as e:
            print(f"Error getting BTC dominance: {e}")
            return 50.0  # Default
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Получить текущую цену"""
        data = self._api_request(
            f"{self.FUTURES_BASE_URL}/fapi/v1/ticker/price",
            params={"symbol": symbol}
        )
        if data and 'price' in data:
            return float(data['price'])
        return None
    
    def get_24h_ticker(self, symbol: str) -> Optional[dict]:
        """Получить 24h статистику"""
        data = self._api_request(
            f"{self.FUTURES_BASE_URL}/fapi/v1/ticker/24hr",
            params={"symbol": symbol}
        )
        return data
    
    def collect_multi_timeframe_data(
        self,
        symbol: str,
        days: int = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Собрать данные по всем таймфреймам для символа
        
        Returns:
            Dict[timeframe] = DataFrame
        """
        days = days or DATA_LOOKBACK_DAYS
        all_timeframes = SCALP_TIMEFRAMES + INTRADAY_TIMEFRAMES + SWING_TIMEFRAMES
        
        data = {}
        
        for tf in all_timeframes:
            print(f"  Collecting {symbol} {tf}...")
            df = self.get_historical_klines(symbol, tf, days)
            if not df.empty:
                data[tf] = df
            time.sleep(0.2)
        
        return data
    
    def collect_all_symbols_data(
        self,
        symbols: List[str] = None,
        days: int = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Собрать данные по всем символам ПАРАЛЛЕЛЬНО
        
        Returns:
            Dict[symbol][timeframe] = DataFrame
        """
        symbols = symbols or TOP_SYMBOLS
        days = days or DATA_LOOKBACK_DAYS
        
        all_data = {}
        
        print(f"Collecting data for {len(symbols)} symbols using {self.n_workers} workers...")
        
        if self.n_workers > 1 and len(symbols) > 1:
            # Параллельный сбор данных с ThreadPoolExecutor
            # (ThreadPool лучше для I/O-bound операций как HTTP запросы)
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(self._collect_symbol_data, symbol, days): symbol
                    for symbol in symbols
                }
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        data = future.result()
                        if data:
                            all_data[symbol] = data
                            print(f"  ✓ {symbol} collected ({len(data)} timeframes)")
                    except Exception as e:
                        print(f"  ✗ Error collecting {symbol}: {e}")
        else:
            # Последовательный сбор
            for symbol in symbols:
                print(f"Collecting data for {symbol}...")
                all_data[symbol] = self.collect_multi_timeframe_data(symbol, days)
        
        print(f"Data collection complete: {len(all_data)} symbols")
        return all_data
    
    def _collect_symbol_data(self, symbol: str, days: int) -> Dict[str, pd.DataFrame]:
        """Сбор данных для одного символа (вызывается параллельно)"""
        try:
            return self.collect_multi_timeframe_data(symbol, days)
        except Exception as e:
            print(f"Error collecting {symbol}: {e}")
            return {}
    
    def get_btc_correlation(
        self,
        symbol: str,
        interval: str = "1h",
        window: int = 24
    ) -> pd.Series:
        """Вычислить rolling correlation с BTC"""
        if symbol == "BTCUSDT":
            return pd.Series()
        
        btc_data = self.get_historical_klines("BTCUSDT", interval, days=30)
        symbol_data = self.get_historical_klines(symbol, interval, days=30)
        
        if btc_data.empty or symbol_data.empty:
            return pd.Series()
        
        # Выравниваем по времени
        merged = pd.merge(
            btc_data[['close']].rename(columns={'close': 'btc_close'}),
            symbol_data[['close']].rename(columns={'close': 'symbol_close'}),
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        btc_returns = merged['btc_close'].pct_change()
        symbol_returns = merged['symbol_close'].pct_change()
        
        correlation = btc_returns.rolling(window=window).corr(symbol_returns)
        return correlation
    
    def save_data(self, data: Dict, base_path: str = "data/raw"):
        """Сохранить данные в CSV"""
        os.makedirs(base_path, exist_ok=True)
        
        for symbol, tf_data in data.items():
            symbol_path = os.path.join(base_path, symbol)
            os.makedirs(symbol_path, exist_ok=True)
            
            for tf, df in tf_data.items():
                file_path = os.path.join(symbol_path, f"{tf}.csv")
                df.to_csv(file_path)
                print(f"Saved {file_path}")
    
    def load_data(self, base_path: str = "data/raw") -> Dict[str, Dict[str, pd.DataFrame]]:
        """Загрузить данные из CSV"""
        data = {}
        
        if not os.path.exists(base_path):
            return data
        
        for symbol in os.listdir(base_path):
            symbol_path = os.path.join(base_path, symbol)
            if os.path.isdir(symbol_path):
                data[symbol] = {}
                for file in os.listdir(symbol_path):
                    if file.endswith('.csv'):
                        tf = file.replace('.csv', '')
                        df = pd.read_csv(
                            os.path.join(symbol_path, file),
                            index_col=0,
                            parse_dates=True
                        )
                        data[symbol][tf] = df
        
        return data


def main():
    """Тестовый сбор данных"""
    collector = BinanceDataCollector()
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    data = collector.collect_all_symbols_data(symbols, days=30)
    
    collector.save_data(data)
    print("Data collection complete!")


if __name__ == "__main__":
    main()
