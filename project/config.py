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

# ============== ADVANCED RISK MODEL ==============
# CVaR (Conditional Value at Risk) Configuration
CVAR_CONFIG = {
    "confidence_level": 0.95,       # 95% confidence for CVaR
    "default_window": 252,          # ~1 year rolling window
    "min_samples": 30,              # Minimum samples for calculation
    "stress_multiplier": 1.5,       # Multiplier for stress CVaR
    "block_threshold": 0.08,        # Block trades when CVaR > 8%
    "model_limits": {
        "scalp": 0.05,              # 5% CVaR limit for scalp
        "intraday": 0.07,           # 7% for intraday
        "swing": 0.10,              # 10% for swing
        "portfolio": 0.08           # 8% for portfolio
    }
}

# Kelly Position Sizing Configuration
KELLY_CONFIG = {
    "mode": "quarter",              # quarter, half, adaptive
    "max_kelly_fraction": 0.25,     # Never use more than 25% of Kelly
    "max_leverage": 5.0,            # Maximum leverage globally
    "max_position_pct": 0.10,       # Max 10% in single position
    "min_win_rate": 0.35,           # Minimum required win rate
    "base_volatility": 0.02,        # Base volatility for scaling
    "model_leverage_limits": {
        "scalp": 10.0,              # Scalp can use higher leverage
        "intraday": 5.0,
        "swing": 3.0
    }
}

# Drawdown Controller Configuration
DRAWDOWN_CONFIG = {
    # Position sizing reduction thresholds
    "level_1": 0.05,                # 5% DD → 75% position size
    "level_2": 0.10,                # 10% DD → 50% position size
    "level_3": 0.15,                # 15% DD → 25% position size
    "level_4": 0.20,                # 20% DD → risk-off only
    "emergency": 0.25,              # 25% DD → emergency stop
    
    # Model disable thresholds
    "scalp_disable": 0.08,          # Disable scalp at 8% DD
    "intraday_disable": 0.12,       # Disable intraday at 12% DD
    "swing_disable": 0.18,          # Disable swing at 18% DD
    
    # Recovery thresholds
    "recovery_to_reduced": 0.5,     # 50% recovery → reduced risk
    "recovery_to_full": 0.8,        # 80% recovery → full risk
}

# Risk Model Paths
RISK_MODEL_PATHS = {
    "pkl": "models/risk_lgbm.pkl",
    "onnx": "models/risk_lgbm.onnx",
    "txt": "models/risk_lgbm.txt",
    "optimized_pkl": "optuna_studies/risk_model_optimized.pkl",
    "optimized_onnx": "optuna_studies/risk_model_optimized.onnx"
}

# Inference Pipeline Configuration
INFERENCE_CONFIG = {
    "max_latency_ms": 100.0,        # Maximum acceptable latency
    "enable_kill_switch": True,     # Enable emergency kill switch
    "enable_monitoring": True,      # Enable real-time monitoring
    "max_risk_score": 0.9,          # Auto-block above this
    "min_confidence": 0.3,          # Warn below this
    "regime_shift_window": 50,      # Periods for regime detection
}
