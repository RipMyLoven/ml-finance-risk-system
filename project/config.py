"""
Конфигурация системы AI Trading
"""

# ============== API ==============
BINANCE_API_KEY = ""
BINANCE_API_SECRET = ""

# ============== TIMEFRAMES ==============
SCALP_TIMEFRAMES = ["5m", "15m"]
INTRADAY_TIMEFRAMES = ["1h", "4h", "12h"]
SWING_TIMEFRAMES = ["1d", "3d", "1w"]

# Mapping для Binance API
TF_MAPPING = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
}

# ============== МОДЕЛИ ==============
MODEL_PATHS = {
    "scalp": "models/scalp_lgbm.pkl",
    "intraday": "models/intraday_lgbm.pkl",
    "swing": "models/swing_lgbm.pkl",
    "risk": "models/risk_lgbm.pkl",
}

# ============== TARGETS ==============
# Для классификации: порог движения цены (в %)
PRICE_MOVE_THRESHOLD = {
    "scalp": 0.3,      # 0.3% для scalp
    "intraday": 1.0,   # 1% для intraday
    "swing": 3.0,      # 3% для swing
}

# Горизонт предсказания (в барах)
PREDICTION_HORIZON = {
    "scalp": 12,       # 12 баров (1 час на 5m)
    "intraday": 6,     # 6 баров (6 часов на 1h)
    "swing": 7,        # 7 баров (7 дней на 1d)
}

# ============== META ENGINE ==============
META_WEIGHTS = {
    "scalp": 0.5,
    "intraday": 0.3,
    "swing": 0.2,
}

ENTRY_THRESHOLD = 0.35  # Порог уверенности для входа (средневзвешенный score)
MAX_RISK_SCORE = 0.7

# ============== RISK MANAGEMENT ==============
MAX_RISK_PER_TRADE = 0.02      # 2%
MAX_TOTAL_RISK = 0.10          # 10%
MAX_CONCURRENT_POSITIONS = 5
MAX_CORRELATION = 0.7          # Максимальная корреляция между позициями
HIGH_VOLATILITY_THRESHOLD = 2.5  # ATR multiplier

# ============== RANKING ==============
TOP_COINS_COUNT = 10

# ============== LEVERAGE ==============
LEVERAGE_MAP = {
    "low_risk": 5,      # risk_score < 0.3
    "medium_risk": 3,   # risk_score 0.3-0.5
    "high_risk": 1,     # risk_score > 0.5
}

# ============== TRAINING ==============
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

# LightGBM для Market Models (3-class classification)
LGBM_MARKET_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
}

# LightGBM для Risk Model (regression)
LGBM_RISK_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
}

# ============== BACKTEST ==============
COMMISSION = 0.001  # 0.1%
SLIPPAGE = 0.0005   # 0.05%
INITIAL_CAPITAL = 10000

# ============== DATA ==============
DATA_LOOKBACK_DAYS = 365  # Сколько дней данных загружать
MIN_VOLUME_USDT = 1000000  # Минимальный объем для фильтрации монет

# ============== SYMBOLS ==============
TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT",
    "XLMUSDT", "APTUSDT", "NEARUSDT", "OPUSDT", "ARBUSDT",
]

# ============== MARKET REGIME ==============
REGIME_THRESHOLDS = {
    "bull": 0.02,    # Return > 2% = bull
    "bear": -0.02,   # Return < -2% = bear
    # Между ними = flat
}
