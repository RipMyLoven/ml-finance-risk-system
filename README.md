# 🤖 AI Crypto Trading System

Система машинного обучения для торговли криптовалютными фьючерсами на Binance.

## 📋 Особенности

- **Multi-Timeframe Analysis** - анализ на 3 таймфреймах (Scalp, Intraday, Swing)
- **Derivatives Data** - использует funding rate, OI, long/short ratio, taker flow
- **LightGBM Models** - быстрые и эффективные модели классификации
- **Optuna Optimization** - автоматический подбор гиперпараметров
- **ONNX Export** - экспорт моделей для использования в продакшене
- **100+ Coins** - поддержка всех основных фьючерсов Binance USDT-M

## 🏗️ Архитектура

```
Binance Futures Data (CSV)
├── klines_usdt_m_{SYMBOL}_{TF}.csv    - OHLCV свечи
├── funding_rate_usdt_m_{SYMBOL}.csv   - Funding Rate
├── open_interest_usdt_m_{SYMBOL}.csv  - Open Interest
├── long_short_ratio_usdt_m_{SYMBOL}.csv - LS Ratio
├── taker_volume_usdt_m_{SYMBOL}.csv   - Taker Buy/Sell
├── mark_price_usdt_m_{SYMBOL}.csv     - Mark Price
└── premium_index_usdt_m_{SYMBOL}.csv  - Premium Index
    ↓
Data Loader V2 (объединение всех данных)
    ↓
Feature Engineering (multi-TF + derivatives)
    ↓
┌───────────────────────────────────────┐
│  TF Models (Scalp / Intraday / Swing) │
└───────────────────────────────────────┘
    ↓
Meta Decision Engine
    ↓
Risk AI
    ↓
Signal Generator
```

## 📁 Структура проекта

```
aiTrainCrypto/
├── data/                              # CSV данные Binance Futures
│   ├── klines_usdt_m_BTCUSDT_1h.csv  # OHLCV свечи
│   ├── klines_usdt_m_BTCUSDT_5m.csv
│   ├── funding_rate_usdt_m_*.csv     # Funding Rate
│   ├── open_interest_usdt_m_*.csv    # Open Interest
│   ├── long_short_ratio_usdt_m_*.csv # Long/Short Ratio
│   ├── taker_volume_usdt_m_*.csv     # Taker Buy/Sell Volume
│   ├── mark_price_usdt_m_*.csv       # Mark Price
│   └── premium_index_usdt_m_*.csv    # Premium Index
│
├── project/
│   ├── config.py                     # Конфигурация
│   ├── main.py                       # Главный модуль
│   ├── train.py                      # Обучение с Optuna
│   ├── onnx_utils.py                 # ONNX утилиты
│   ├── requirements.txt              # Зависимости
│   │
│   ├── data/                         # Загрузчики данных
│   │   ├── data_loader_v2.py         # НОВЫЙ загрузчик всех CSV
│   │   ├── data_collector.py         # Binance API collector
│   │   └── data_loader.py            # Старый loader (legacy)
│   │
│   ├── features/                     # Feature Engineering
│   │   ├── scalp_features.py         # 5m/15m + derivatives
│   │   ├── intraday_features.py      # 1h/4h + derivatives
│   │   └── swing_features.py         # 1d/3d/1w
│   │
│   ├── training/                     # Модули обучения
│   │   ├── train_scalp.py
│   │   ├── train_intraday.py
│   │   ├── train_swing.py
│   │   └── train_risk.py
│   │
│   ├── meta/                         # Meta Decision Engine
│   │   ├── meta_engine.py
│   │   └── ranking.py
│   │
│   ├── signals/                      # Генерация сигналов
│   │   └── signal_generator.py
│   │
│   ├── backtest/                     # Бэктестинг
│   │   └── optinus_runner.py
│   │
│   ├── models/                       # Сохранённые модели
│   │   ├── scalp_lgbm.onnx
│   │   ├── intraday_lgbm.onnx
│   │   ├── swing_lgbm.onnx
│   │   └── risk_lgbm.onnx
│   │
│   └── optuna_studies/               # Optuna studies
│       └── best_params.json
```

## 📊 Формат данных

### OHLCV Свечи (klines)
```csv
timestamp,open,high,low,close,volume,close_time,quote_volume,trades,taker_buy_volume,taker_buy_quote_volume,symbol,interval,market_type
1577836800000,7189.43,7190.52,7170.15,7171.55,2449.049,1577840399999,17576424.43970,3688,996.198,7149370.76353,BTCUSDT,1h,USDT-M
```

### Funding Rate
```csv
symbol,funding_time,funding_rate,mark_price,market_type
BTCUSDT,1577836800000,-0.00012359,,USDT-M
```

### Open Interest
```csv
symbol,timestamp,sum_open_interest,sum_open_interest_value,market_type
BTCUSDT,1769309100000,99749.75600000,8888571082.47720000,USDT-M
```

### Long/Short Ratio
```csv
symbol,timestamp,long_short_ratio,long_account,short_account,period
BTCUSDT,1769350800000,2.6928,0.7292,0.2708,5m
```

### Taker Volume
```csv
symbol,timestamp,buy_sell_ratio,buy_vol,sell_vol,period
BTCUSDT,1769350500000,0.9885,62.3790,63.1020,5m
```

## 🚀 Быстрый старт

### 1. Установка

```bash
cd project
pip install -r requirements.txt
```

### 2. Проверка данных

```bash
# Тест загрузчика данных
python data/data_loader_v2.py
```

### 3. Обучение моделей

```bash
# Быстрое обучение (без Optuna)
python train.py --fast

# Обучение с Optuna оптимизацией (рекомендуется)
python train.py --trials 100

# Обучить только конкретную модель
python train.py --model scalp --trials 50

# С GPU
python train.py --gpu --trials 100

# Продолжить предыдущую оптимизацию
python train.py --continue-study --trials 50
```

## 📈 Модели и Фичи

### Scalp Model (5m таймфрейм)

| Категория | Фичи |
|-----------|------|
| Returns | log_return_1/3/5/10, cum_return |
| Momentum | RSI 5/7/14, Stochastic, ROC, EMA cross |
| Volatility | micro_vol_3/5/10, ATR, vol_ratio |
| Volume | volume_ratio, volume_spike, OBV |
| Derivatives | funding_rate, OI_change, LS_ratio |
| Taker Flow | buy_sell_ratio, taker_imbalance |

### Intraday Model (1h таймфрейм)

| Категория | Фичи |
|-----------|------|
| Trend | SMA 10/20/50, trend_slope, EMA trend |
| VWAP | vwap_deviation, vwap_position |
| Momentum | MACD, ADX, RSI 14, MFI |
| Volatility | BB_width, ATR, volatility_ratio |
| Derivatives | funding_zscore, OI_zscore, LS_ratio |
| Market | premium_index, basis |

### Target Classes

- **0** - Price Down (цена упадёт > threshold)
- **1** - Flat (цена в диапазоне ±threshold)
- **2** - Price Up (цена вырастет > threshold)

| Модель | Таймфрейм | Горизонт | Порог |
|--------|-----------|----------|-------|
| Scalp | 5m | 12 баров (~1 час) | 0.3% |
| Intraday | 1h | 6 баров (~6 часов) | 1.0% |
| Swing | 1d | 7 баров (~7 дней) | 3.0% |

## 🛠️ Команды train.py

```bash
python train.py --help

# Опции:
#   --fast              Быстрое обучение без Optuna
#   --best              Использовать лучшие сохранённые параметры
#   --trials N          Количество Optuna trials (default: 100)
#   --timeout N         Таймаут в секундах
#   --continue-study    Продолжить предыдущую study
#   --study-name NAME   Имя для Optuna study
#   --model TYPE        Модель: all, scalp, intraday, swing, risk
#   --gpu               Использовать GPU
#   --data-path PATH    Путь к данным
```

## 📦 Форматы моделей

| Формат | Файл | Описание |
|--------|------|----------|
| LightGBM | `*.txt` | Нативный формат |
| Pickle | `*.pkl` | Python + метаданные |
| ONNX | `*.onnx` | Кроссплатформенный |

### Использование ONNX

```python
from onnx_utils import ONNXPredictor

predictor = ONNXPredictor("models/scalp_lgbm.onnx")
result = predictor.predict(features)
```

## 🔧 Зависимости

```
numpy>=1.24.0
pandas>=2.0.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
optuna>=3.0.0
onnx>=1.14.0
onnxmltools>=1.11.0
python-binance>=1.0.19
ta>=0.10.2
```

## 📊 Доступные символы

```
1INCHUSDT, 1000SHIBUSDT, AAVEUSDT, ADAUSDT, ALGOUSDT, ALICEUSDT, 
ANKRUSDT, APEUSDT, API3USDT, ARPAUSDT, ARUSDT, ATAUSDT, ATOMUSDT, 
AVAXUSDT, AXSUSDT, BANDUSDT, BATUSDT, BCHUSDT, BELUSDT, BNBUSDT, 
BTCDOMUSDT, BTCUSDT, C98USDT, CELOUSDT, CELRUSDT, CHRUSDT, CHZUSDT,
COMPUSDT, COTIUSDT, CRVUSDT, CTSIUSDT, DASHUSDT, DENTUSDT, DOGEUSDT,
DOTUSDT, DUSKUSDT, DYDXUSDT, EGLDUSDT, ENJUSDT, ENSUSDT, ETCUSDT,
ETHUSDT, FILUSDT, FLOWUSDT, GALAUSDT, GMTUSDT, GRTUSDT, GTCUSDT,
HBARUSDT, HOTUSDT, ICXUSDT, IMXUSDT, IOSTUSDT, IOTAUSDT, IOTXUSDT,
JASMYUSDT, KAVAUSDT, KNCUSDT, KSMUSDT, LINKUSDT, LPTUSDT, LRCUSDT,
LTCUSDT, MANAUSDT, MASKUSDT, MTLUSDT, NEARUSDT, NEOUSDT, NKNUSDT,
OGNUSDT, ONEUSDT, ONTUSDT, PEOPLEUSDT, QTUMUSDT, RLCUSDT, ROSEUSDT,
RSRUSDT, RUNEUSDT, RVNUSDT, SANDUSDT, SFPUSDT, SKLUSDT, SNXUSDT,
SOLUSDT, STORJUSDT, SUSHIUSDT, THETAUSDT, TRBUSDT, TRXUSDT, UNIUSDT,
VETUSDT, WOOUSDT, XLMUSDT, XMRUSDT, XRPUSDT, XTZUSDT, YFIUSDT,
ZECUSDT, ZENUSDT, ZILUSDT, ZRXUSDT
```

## 📝 Лицензия

MIT License

## 🤝 Contributing

1. Fork репозитория
2. Создай feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменений (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Открой Pull Request
