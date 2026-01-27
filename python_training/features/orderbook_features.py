"""
📊 Order Book Features

Фичи из стакана заявок — один из самых сильных предикторов!

Требует: real-time данные из API (нельзя получить из исторических CSV)
"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def get_binance_orderbook(symbol: str, limit: int = 100) -> Dict:
    """
    Получить текущий order book с Binance
    
    Args:
        symbol: Торговая пара (BTCUSDT)
        limit: Глубина стакана (5, 10, 20, 50, 100, 500, 1000)
    
    Returns:
        Dict с bids и asks
    """
    url = "https://api.binance.com/api/v3/depth"
    params = {
        'symbol': symbol.upper(),
        'limit': limit
    }
    
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    
    if 'code' in data:
        raise Exception(f"Binance error: {data['msg']}")
    
    return data


def get_bybit_orderbook(symbol: str, limit: int = 100) -> Dict:
    """
    Получить текущий order book с Bybit
    """
    url = "https://api.bybit.com/v5/market/orderbook"
    params = {
        'category': 'linear',
        'symbol': symbol.upper(),
        'limit': limit
    }
    
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    
    if data.get('retCode') != 0:
        raise Exception(f"Bybit error: {data.get('retMsg')}")
    
    result = data.get('result', {})
    
    # Преобразуем в формат Binance
    return {
        'bids': [[b[0], b[1]] for b in result.get('b', [])],
        'asks': [[a[0], a[1]] for a in result.get('a', [])]
    }


def parse_orderbook(data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Парсинг order book в DataFrame
    
    Returns:
        (bids_df, asks_df) с колонками [price, quantity, cumulative]
    """
    bids = pd.DataFrame(data['bids'], columns=['price', 'quantity'])
    asks = pd.DataFrame(data['asks'], columns=['price', 'quantity'])
    
    bids['price'] = bids['price'].astype(float)
    bids['quantity'] = bids['quantity'].astype(float)
    asks['price'] = asks['price'].astype(float)
    asks['quantity'] = asks['quantity'].astype(float)
    
    # Cumulative volume
    bids['cumulative'] = bids['quantity'].cumsum()
    asks['cumulative'] = asks['quantity'].cumsum()
    
    return bids, asks


def calculate_orderbook_features(symbol: str, exchange: str = 'binance') -> Dict[str, float]:
    """
    Вычислить все фичи из order book
    
    🔥 Это одни из самых информативных фичей!
    
    Returns:
        Dict с фичами
    """
    # Получаем данные
    if exchange.lower() == 'binance':
        data = get_binance_orderbook(symbol, limit=100)
    elif exchange.lower() == 'bybit':
        data = get_bybit_orderbook(symbol, limit=100)
    else:
        raise ValueError(f"Unknown exchange: {exchange}")
    
    bids, asks = parse_orderbook(data)
    
    if len(bids) == 0 or len(asks) == 0:
        return {}
    
    features = {}
    
    # ═══════════════════════════════════════════════════════════════
    # 1. SPREAD (спред)
    # ═══════════════════════════════════════════════════════════════
    
    best_bid = bids['price'].iloc[0]
    best_ask = asks['price'].iloc[0]
    mid_price = (best_bid + best_ask) / 2
    
    # Абсолютный и относительный спред
    spread = best_ask - best_bid
    features['spread_pct'] = spread / mid_price  # обычно 0.0001-0.001
    
    # ═══════════════════════════════════════════════════════════════
    # 2. IMBALANCE (дисбаланс)
    # ═══════════════════════════════════════════════════════════════
    
    # Объём на лучших уровнях
    bid_vol_top1 = bids['quantity'].iloc[0]
    ask_vol_top1 = asks['quantity'].iloc[0]
    
    # Imbalance на топ-1 уровне
    features['imbalance_top1'] = (bid_vol_top1 - ask_vol_top1) / (bid_vol_top1 + ask_vol_top1 + 1e-10)
    
    # Imbalance на топ-5 уровнях
    bid_vol_top5 = bids['quantity'].iloc[:5].sum()
    ask_vol_top5 = asks['quantity'].iloc[:5].sum()
    features['imbalance_top5'] = (bid_vol_top5 - ask_vol_top5) / (bid_vol_top5 + ask_vol_top5 + 1e-10)
    
    # Imbalance на топ-10 уровнях
    bid_vol_top10 = bids['quantity'].iloc[:10].sum()
    ask_vol_top10 = asks['quantity'].iloc[:10].sum()
    features['imbalance_top10'] = (bid_vol_top10 - ask_vol_top10) / (bid_vol_top10 + ask_vol_top10 + 1e-10)
    
    # Imbalance на топ-20 уровнях
    bid_vol_top20 = bids['quantity'].iloc[:20].sum()
    ask_vol_top20 = asks['quantity'].iloc[:20].sum()
    features['imbalance_top20'] = (bid_vol_top20 - ask_vol_top20) / (bid_vol_top20 + ask_vol_top20 + 1e-10)
    
    # ═══════════════════════════════════════════════════════════════
    # 3. DEPTH AT LEVELS (глубина на уровнях)
    # ═══════════════════════════════════════════════════════════════
    
    # Суммарный объём в пределах X% от mid_price
    for pct in [0.1, 0.5, 1.0, 2.0]:
        price_range = mid_price * pct / 100
        
        bid_depth = bids[bids['price'] >= mid_price - price_range]['quantity'].sum()
        ask_depth = asks[asks['price'] <= mid_price + price_range]['quantity'].sum()
        
        total_depth = bid_depth + ask_depth
        features[f'depth_{pct}pct'] = total_depth
        features[f'depth_imbalance_{pct}pct'] = (bid_depth - ask_depth) / (total_depth + 1e-10)
    
    # ═══════════════════════════════════════════════════════════════
    # 4. WEIGHTED PRICE (взвешенная цена)
    # ═══════════════════════════════════════════════════════════════
    
    # Volume-weighted average price для bid и ask
    bid_vwap = (bids['price'] * bids['quantity']).sum() / (bids['quantity'].sum() + 1e-10)
    ask_vwap = (asks['price'] * asks['quantity']).sum() / (asks['quantity'].sum() + 1e-10)
    
    # Отклонение VWAP от mid_price
    features['bid_vwap_dist'] = (bid_vwap - mid_price) / mid_price
    features['ask_vwap_dist'] = (ask_vwap - mid_price) / mid_price
    
    # ═══════════════════════════════════════════════════════════════
    # 5. WALL DETECTION (обнаружение стенок)
    # ═══════════════════════════════════════════════════════════════
    
    # Средний объём
    avg_bid_vol = bids['quantity'].mean()
    avg_ask_vol = asks['quantity'].mean()
    
    # Макс объём (потенциальная стенка)
    max_bid = bids['quantity'].max()
    max_ask = asks['quantity'].max()
    
    # Ratio к среднему (если > 5 — вероятно стенка)
    features['bid_wall_ratio'] = max_bid / (avg_bid_vol + 1e-10)
    features['ask_wall_ratio'] = max_ask / (avg_ask_vol + 1e-10)
    
    # Расстояние до стенки
    bid_wall_idx = bids['quantity'].idxmax()
    ask_wall_idx = asks['quantity'].idxmax()
    
    bid_wall_price = bids.loc[bid_wall_idx, 'price']
    ask_wall_price = asks.loc[ask_wall_idx, 'price']
    
    features['bid_wall_dist'] = (mid_price - bid_wall_price) / mid_price
    features['ask_wall_dist'] = (ask_wall_price - mid_price) / mid_price
    
    # ═══════════════════════════════════════════════════════════════
    # 6. SLOPE (наклон стакана)
    # ═══════════════════════════════════════════════════════════════
    
    # Как быстро растёт cumulative volume
    # Крутой наклон = много ликвидности близко к цене
    if len(bids) >= 10:
        bid_slope = np.polyfit(range(10), bids['cumulative'].iloc[:10].values, 1)[0]
        features['bid_slope'] = bid_slope / (bids['cumulative'].iloc[9] + 1e-10)
    
    if len(asks) >= 10:
        ask_slope = np.polyfit(range(10), asks['cumulative'].iloc[:10].values, 1)[0]
        features['ask_slope'] = ask_slope / (asks['cumulative'].iloc[9] + 1e-10)
    
    # ═══════════════════════════════════════════════════════════════
    # 7. MICRO PRESSURE
    # ═══════════════════════════════════════════════════════════════
    
    # Давление на микроуровне (первые 3 уровня)
    micro_bid = bids['quantity'].iloc[:3].sum()
    micro_ask = asks['quantity'].iloc[:3].sum()
    features['micro_pressure'] = (micro_bid - micro_ask) / (micro_bid + micro_ask + 1e-10)
    
    return features


def get_orderbook_features_normalized(symbol: str, exchange: str = 'binance') -> Dict[str, float]:
    """
    Получить нормализованные фичи для модели
    
    Все фичи уже в диапазоне [-1, 1] или около того
    """
    features = calculate_orderbook_features(symbol, exchange)
    
    # Нормализация depth фичей (они в абсолютных значениях)
    # Превращаем в log-scale
    for key in list(features.keys()):
        if key.startswith('depth_') and not 'imbalance' in key:
            features[key] = np.log1p(features[key])
    
    # Ограничиваем wall ratios
    for key in ['bid_wall_ratio', 'ask_wall_ratio']:
        if key in features:
            features[key] = np.clip(features[key], 0, 20) / 20  # normalize to [0, 1]
    
    return features


# ═══════════════════════════════════════════════════════════════════
# STREAMING (для сбора исторических данных)
# ═══════════════════════════════════════════════════════════════════

def collect_orderbook_snapshots(symbol: str, exchange: str = 'binance', 
                                 interval_sec: float = 1.0, duration_min: int = 60,
                                 output_file: str = None) -> pd.DataFrame:
    """
    Собрать снапшоты order book для обучения
    
    Args:
        symbol: Торговая пара
        exchange: Биржа
        interval_sec: Интервал между снапшотами
        duration_min: Длительность сбора в минутах
        output_file: Куда сохранить CSV
    
    Returns:
        DataFrame со всеми снапшотами
    """
    import time
    from datetime import datetime
    
    snapshots = []
    total_iterations = int(duration_min * 60 / interval_sec)
    
    print(f"📊 Сбор order book {symbol} на {exchange}")
    print(f"   Интервал: {interval_sec}s, Длительность: {duration_min}min")
    print(f"   Всего снапшотов: {total_iterations}")
    
    for i in range(total_iterations):
        try:
            features = calculate_orderbook_features(symbol, exchange)
            features['timestamp'] = datetime.now().isoformat()
            features['symbol'] = symbol
            snapshots.append(features)
            
            if (i + 1) % 60 == 0:
                print(f"   Собрано: {i + 1}/{total_iterations} ({(i+1)/total_iterations*100:.1f}%)")
            
        except Exception as e:
            print(f"   ⚠️ Ошибка: {e}")
        
        time.sleep(interval_sec)
    
    df = pd.DataFrame(snapshots)
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"✅ Сохранено в {output_file}")
    
    return df


if __name__ == "__main__":
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    
    print(f"\n📊 Order Book Features для {symbol}")
    print("=" * 50)
    
    features = calculate_orderbook_features(symbol)
    
    print("\n🔥 IMBALANCE (дисбаланс bids vs asks):")
    print(f"   Top-1:  {features.get('imbalance_top1', 0):+.3f}")
    print(f"   Top-5:  {features.get('imbalance_top5', 0):+.3f}")
    print(f"   Top-10: {features.get('imbalance_top10', 0):+.3f}")
    print(f"   Top-20: {features.get('imbalance_top20', 0):+.3f}")
    
    print("\n📊 SPREAD:")
    print(f"   Spread: {features.get('spread_pct', 0)*100:.4f}%")
    
    print("\n🧱 WALLS (стенки):")
    print(f"   Bid wall ratio: {features.get('bid_wall_ratio', 0):.1f}x")
    print(f"   Ask wall ratio: {features.get('ask_wall_ratio', 0):.1f}x")
    print(f"   Bid wall dist: {features.get('bid_wall_dist', 0)*100:.2f}%")
    print(f"   Ask wall dist: {features.get('ask_wall_dist', 0)*100:.2f}%")
    
    print("\n⚡ MICRO PRESSURE:")
    print(f"   Давление: {features.get('micro_pressure', 0):+.3f}")
    
    if features.get('imbalance_top5', 0) > 0.1:
        print("\n🟢 Перевес покупателей в стакане!")
    elif features.get('imbalance_top5', 0) < -0.1:
        print("\n🔴 Перевес продавцов в стакане!")
    else:
        print("\n⚪ Стакан сбалансирован")
