"""
🔴 LIVE Predictor - Получает данные с биржи в реальном времени

Поддерживаемые биржи:
- Binance
- Bybit

Использование:
    python live_predict.py BTCUSDT           # Binance
    python live_predict.py BTCUSDT binance   # Binance явно
    python live_predict.py BTCUSDT bybit     # Bybit
"""
import sys
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import CryptoPredictor


def normalize_symbol(symbol: str) -> str:
    """
    Нормализует символ - добавляет USDT если нужно
    """
    symbol = symbol.upper().strip()
    
    # Популярные базовые монеты (не пары!)
    base_coins = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'LINK', 
                  'LTC', 'AVAX', 'MATIC', 'UNI', 'ATOM', 'APT', 'ARB', 'OP', 'SUI',
                  'TRX', 'NEAR', 'FIL', 'PEPE', 'SHIB', 'WIF', 'BONK']
    
    # Если это просто название монеты — добавляем USDT
    if symbol in base_coins:
        return symbol + 'USDT'
    
    # Если уже содержит quote currency на конце
    quote_currencies = ['USDT', 'USDC', 'BUSD', 'TUSD', 'FDUSD']
    for quote in quote_currencies:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol
    
    # Проверяем пары с BTC/ETH/BNB
    crypto_quotes = ['BTC', 'ETH', 'BNB']
    for quote in crypto_quotes:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol
    
    # По умолчанию добавляем USDT
    return symbol + 'USDT'


def check_symbol_exists(symbol: str, exchange: str = 'binance') -> tuple:
    """
    Проверяет существование символа на бирже.
    Возвращает (exists: bool, suggested: list)
    """
    try:
        if exchange.lower() == 'binance':
            url = "https://api.binance.com/api/v3/exchangeInfo"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            all_symbols = [s['symbol'] for s in data.get('symbols', [])]
            
            if symbol in all_symbols:
                return True, []
            
            # Ищем похожие
            base = symbol.replace('USDT', '').replace('USDC', '').replace('BTC', '')
            suggestions = [s for s in all_symbols if base in s and 'USDT' in s][:5]
            return False, suggestions
            
        elif exchange.lower() == 'bybit':
            url = "https://api.bybit.com/v5/market/instruments-info"
            params = {'category': 'linear'}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            all_symbols = [s['symbol'] for s in data.get('result', {}).get('list', [])]
            
            if symbol in all_symbols:
                return True, []
            
            base = symbol.replace('USDT', '').replace('USDC', '')
            suggestions = [s for s in all_symbols if base in s][:5]
            return False, suggestions
            
    except Exception as e:
        print(f"⚠️ Не удалось проверить символ: {e}")
        return True, []  # Пробуем всё равно
    
    return False, []


def get_binance_klines(symbol: str, interval: str = '1m', limit: int = 100) -> pd.DataFrame:
    """
    Получить свечи с Binance
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol.upper(),
        'interval': interval,
        'limit': limit
    }
    
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    
    if isinstance(data, dict) and 'code' in data:
        raise Exception(f"Binance error: {data['msg']}")
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume_btc'] = df['volume'].astype(float)
    df['volume_usd'] = df['quote_volume'].astype(float)
    df['num_trades'] = df['trades'].astype(int)
    
    # Buy ratio (приблизительно)
    df['buy_ratio'] = df['taker_buy_base'].astype(float) / (df['volume'].astype(float) + 1e-10)
    
    return df[['datetime', 'open', 'high', 'low', 'close', 'volume_btc', 'volume_usd', 'num_trades', 'buy_ratio']]


def get_bybit_klines(symbol: str, interval: str = '1', limit: int = 100) -> pd.DataFrame:
    """
    Получить свечи с Bybit
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        'category': 'linear',
        'symbol': symbol.upper(),
        'interval': interval,
        'limit': limit
    }
    
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    
    if data['retCode'] != 0:
        raise Exception(f"Bybit error: {data['retMsg']}")
    
    rows = data['result']['list']
    
    df = pd.DataFrame(rows, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    
    df['datetime'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume_btc'] = df['volume'].astype(float)
    df['volume_usd'] = df['turnover'].astype(float)
    df['num_trades'] = 100  # Bybit не даёт количество сделок
    df['buy_ratio'] = 0.5   # Неизвестно
    
    # Bybit возвращает в обратном порядке
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return df[['datetime', 'open', 'high', 'low', 'close', 'volume_btc', 'volume_usd', 'num_trades', 'buy_ratio']]


def get_live_data(symbol: str, exchange: str = 'binance') -> dict:
    """
    Получить live данные на разных таймфреймах
    """
    # Нормализуем символ
    symbol = normalize_symbol(symbol)
    print(f"📡 Получаю данные {symbol} с {exchange}...")
    
    # Проверяем существование символа
    exists, suggestions = check_symbol_exists(symbol, exchange)
    if not exists:
        error_msg = f"❌ Символ {symbol} не найден на {exchange}"
        if suggestions:
            error_msg += f"\n💡 Возможно вы имели в виду: {', '.join(suggestions)}"
        raise ValueError(error_msg)
    
    if exchange.lower() == 'binance':
        bars_1m = get_binance_klines(symbol, '1m', 100)
        bars_5m = get_binance_klines(symbol, '5m', 100)
        bars_15m = get_binance_klines(symbol, '15m', 100)
        bars_1h = get_binance_klines(symbol, '1h', 100)
        bars_4h = get_binance_klines(symbol, '4h', 100)
        bars_1d = get_binance_klines(symbol, '1d', 100)
    elif exchange.lower() == 'bybit':
        bars_1m = get_bybit_klines(symbol, '1', 100)
        bars_5m = get_bybit_klines(symbol, '5', 100)
        bars_15m = get_bybit_klines(symbol, '15', 100)
        bars_1h = get_bybit_klines(symbol, '60', 100)
        bars_4h = get_bybit_klines(symbol, '240', 100)
        bars_1d = get_bybit_klines(symbol, 'D', 100)
    else:
        raise ValueError(f"Неизвестная биржа: {exchange}")
    
    return {
        '1m': bars_1m,
        '5m': bars_5m,
        '15m': bars_15m,
        '1h': bars_1h,
        '4h': bars_4h,
        '1d': bars_1d
    }


def predict_live(symbol: str, exchange: str = 'binance', predictor: CryptoPredictor = None):
    """
    Live предсказание
    """
    if predictor is None:
        predictor = CryptoPredictor()
    
    # Получаем данные
    data = get_live_data(symbol, exchange)
    
    bars_1m = data['1m']
    bars_5m = data['5m']
    bars_15m = data['15m']
    bars_1h = data['1h']
    bars_4h = data['4h']
    bars_1d = data['1d']
    
    # Строим фичи
    from features.multi_timeframe import build_single_timeframe_features
    
    df_1m = build_single_timeframe_features(bars_1m.copy(), suffix='')
    df_5m = build_single_timeframe_features(bars_5m.copy(), suffix='_5min')
    df_15m = build_single_timeframe_features(bars_15m.copy(), suffix='_15min')
    df_1h = build_single_timeframe_features(bars_1h.copy(), suffix='_1h')
    df_4h = build_single_timeframe_features(bars_4h.copy(), suffix='_4h')
    df_1d = build_single_timeframe_features(bars_1d.copy(), suffix='_1d')
    
    # Собираем все фичи
    df = df_1m.copy().set_index('datetime')
    
    for tf_df, suffix in [(df_5m, '_5min'), (df_15m, '_15min'), (df_1h, '_1h'), (df_4h, '_4h'), (df_1d, '_1d')]:
        tf_features = [col for col in tf_df.columns if col.endswith(suffix)]
        tf_subset = tf_df[['datetime'] + tf_features].set_index('datetime')
        for col in tf_features:
            df[col] = tf_subset[col].reindex(df.index, method='ffill')
    
    df = df.reset_index().dropna()
    
    if len(df) == 0:
        return {"error": "Недостаточно данных"}
    
    # Последняя строка
    last_row = df.iloc[-1]
    
    # Собираем фичи в нужном порядке
    features = []
    for f in predictor.feature_names:
        if f in last_row:
            features.append(float(last_row[f]))
        else:
            features.append(0.0)
    
    # Предсказание
    features_array = np.array([features])
    probability = predictor._predict_proba(features_array)[0][1]
    
    # Форматируем символ для отображения (BTCUSDT -> BTC/USDT)
    display_symbol = symbol
    for quote in ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH']:
        if symbol.endswith(quote):
            base = symbol[:-len(quote)]
            display_symbol = f"{base}/{quote}"
            break
    
    # Форматируем результат
    result = predictor._format_prediction(
        probability=probability,
        last_price=last_row['close'],
        last_time=last_row['datetime'],
        features=last_row
    )
    result['symbol'] = display_symbol
    return result


def live_monitor(symbol: str, exchange: str = 'binance', interval: int = 60):
    """
    Мониторинг в реальном времени
    """
    print("="*60)
    print(f"  🔴 LIVE MONITOR: {symbol} @ {exchange}")
    print(f"  Обновление каждые {interval} секунд")
    print("  Нажми Ctrl+C для выхода")
    print("="*60)
    
    predictor = CryptoPredictor()
    
    while True:
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📊 {symbol} @ {exchange.upper()}")
            print("-"*60)
            
            result = predict_live(symbol, exchange, predictor)
            
            if 'error' in result:
                print(f"❌ {result['error']}")
            else:
                predictor.print_prediction(result)
            
            print(f"\n⏱️ Следующее обновление через {interval} сек...")
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n👋 Мониторинг остановлен")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            time.sleep(5)


def main():
    if len(sys.argv) < 2:
        print("""
🔴 LIVE PREDICTOR

Использование:
    python live_predict.py BTCUSDT              # Один раз, Binance
    python live_predict.py BTCUSDT bybit        # Один раз, Bybit
    python live_predict.py BTCUSDT binance live # Мониторинг каждые 60 сек
    python live_predict.py ETHUSDT bybit live 30  # Мониторинг каждые 30 сек

Символы:
    Binance: BTCUSDT, ETHUSDT, SOLUSDT, ADAUSDT, XRPUSDT...
    Bybit:   BTCUSDT, ETHUSDT, SOLUSDT...
        """)
        return
    
    symbol = sys.argv[1].upper()
    exchange = sys.argv[2].lower() if len(sys.argv) > 2 and sys.argv[2].lower() in ['binance', 'bybit'] else 'binance'
    
    # Нормализуем символ (добавляем USDT если нужно)
    normalized_symbol = normalize_symbol(symbol)
    if normalized_symbol != symbol:
        print(f"ℹ️ Символ нормализован: {symbol} → {normalized_symbol}")
        symbol = normalized_symbol
    
    # Проверяем режим
    is_live = 'live' in [arg.lower() for arg in sys.argv]
    
    if is_live:
        # Интервал обновления
        interval = 60
        for arg in sys.argv:
            if arg.isdigit():
                interval = int(arg)
        
        live_monitor(symbol, exchange, interval)
    else:
        # Одноразовый запрос
        try:
            predictor = CryptoPredictor()
            result = predict_live(symbol, exchange, predictor)
            
            if 'error' in result:
                print(f"❌ {result['error']}")
            else:
                predictor.print_prediction(result)
        except ValueError as e:
            print(str(e))
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
