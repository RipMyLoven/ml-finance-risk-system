# ML Finance Risk System — Полное описание проекта

> Этот файл создан как база знаний для AI-агентов и для ответов на вопросы по проекту.  
> Все модули, конфигурации, алгоритмы и архитектурные решения описаны здесь.

---

## Оглавление

1. [Общая концепция](#1-общая-концепция)
2. [Структура директорий](#2-структура-директорий)
3. [Архитектура системы](#3-архитектура-системы)
4. [Конфигурация](#4-конфигурация)
5. [Модели данных и обучение](#5-модели-данных-и-обучение)
6. [Feature Engineering](#6-feature-engineering)
7. [Риск-менеджмент](#7-риск-менеджмент)
8. [Meta Engine](#8-meta-engine)
9. [Signal Generator](#9-signal-generator)
10. [TUI — Terminal User Interface](#10-tui--terminal-user-interface)
11. [Бэктест](#11-бэктест)
12. [Данные и источники](#12-данные-и-источники)
13. [ONNX инфраструктура](#13-onnx-инфраструктура)
14. [Зависимости и требования](#14-зависимости-и-требования)
15. [Быстрый старт](#15-быстрый-старт)

---

## 1. Общая концепция

**ML Finance Risk System** — институциональная система алготрейдинга на крипто-фьючерсах.  
Три торговых ML-модели (scalp / intraday / swing) агрегируются правилами (meta engine) в одно торговое решение,  
которое затем проходит через многоуровневую систему риск-менеджмента перед генерацией сигнала.

**Ключевые принципы:**
- **ONNX-first**: все модели экспортированы в ONNX для детерминированного CPU-инференса
- **Defense-in-depth**: CVaR + Kelly + Drawdown Controller работают независимо и совместно
- **Meta engine — НЕ ML**: агрегация сигналов — взвешенная формула, не нейросеть
- **Режимы**: обучение (train_with_risk.py / train.py) + реалтайм TUI (tui/) + бэктест (backtest/)

---

## 2. Структура директорий

```
ml-finance-risk-system/
├── README.md
├── AGENTS.md                        ← этот файл
├── docs/
│   └── hindamis-standart.md         ← стандарт оценки качества
│
├── project/                         ← основная торговая система
│   ├── main.py                      ← точка входа (batch-режим)
│   ├── config.py                    ← жёстко заданные константы
│   ├── config.yaml                  ← настраиваемая конфигурация (CPU/RAM/LGBM/Optuna)
│   ├── train.py                     ← унифицированный тренер (1500 строк, 4 режима)
│   ├── train_with_risk.py           ← полный 6-фазный пайплайн (2500 строк)
│   ├── onnx_utils.py                ← утилиты ONNX экспорта/валидации
│   │
│   ├── data/
│   │   ├── data_collector.py        ← BinanceDataCollector (OHLCV, funding, OI, BTC corr)
│   │   ├── data_loader.py           ← конвертация trade CSV → OHLCV
│   │   ├── data_loader_v2.py        ← улучшенная версия лоадера
│   │   └── parallel_loader.py       ← параллельная загрузка (joblib ProcessPool)
│   │
│   ├── features/
│   │   ├── fast_features.py         ← Numba JIT-ускоренные вычисления
│   │   ├── intraday_features.py     ← 45+ features для 1h/4h/12h
│   │   ├── scalp_features.py        ← 50+ features для 5m/15m
│   │   └── swing_features.py        ← 70+ features для 1d/3d/1w
│   │
│   ├── models/                      ← обученные артефакты
│   │   ├── scalp_lgbm.onnx          ← scalp модель (ONNX)
│   │   ├── intraday_lgbm.onnx       ← intraday модель (ONNX)
│   │   ├── swing_lgbm.onnx          ← swing модель (ONNX)
│   │   ├── risk_lgbm.onnx           ← risk модель (ONNX)
│   │   ├── *_lgbm_features.json     ← имена признаков для ONNX инференса
│   │   ├── *_lgbm.txt               ← LightGBM текстовый формат
│   │   ├── risk_engine_config_*.json← версионированные конфиги риск-движка
│   │   ├── training_results_*.json  ← результаты обучения
│   │   └── training_assessment.json ← оценка качества тренировки
│   │
│   ├── risk/
│   │   ├── cvar_engine.py           ← CVaR расчёт (исторический, хвостовой риск)
│   │   ├── drawdown_controller.py   ← нелинейный контроль просадки
│   │   ├── inference_pipeline.py    ← продакшн пайплайн с kill-switch
│   │   ├── kelly_sizing.py          ← Kelly Criterion для размера позиции
│   │   ├── risk_features.py         ← 32+ риск-признаков для risk модели
│   │   ├── risk_model.py            ← центральный риск-контроллер (финальный вето)
│   │   └── train_risk_optuna.py     ← обучение risk модели с Optuna
│   │
│   ├── meta/
│   │   ├── meta_engine.py           ← MetaDecisionEngine (правило-основанный)
│   │   └── ranking.py               ← CoinRanker (рейтинг монет)
│   │
│   ├── signals/
│   │   ├── signal_generator.py      ← SignalGenerator → TradingSignal
│   │   └── signals_*.json           ← сохранённые сигналы
│   │
│   ├── training/
│   │   ├── system_config.py         ← автодетект CPU/RAM, оптимальные параметры
│   │   ├── train_fixed.py           ← исправленная версия тренера
│   │   ├── train_intraday.py        ← тренер только для intraday
│   │   ├── train_optimized.py       ← оптимизированный тренер
│   │   ├── train_risk.py            ← тренер risk модели
│   │   ├── train_scalp.py           ← тренер scalp модели
│   │   ├── train_swing.py           ← тренер swing модели
│   │   ├── diagnose_leakage.py      ← диагностика утечки данных
│   │   └── QUICK_FIXES.py           ← быстрые исправления известных проблем
│   │
│   ├── validation/
│   │   ├── baseline_models.py       ← baseline сравнения
│   │   ├── data_sanity.py           ← проверки данных
│   │   ├── ensemble_validation.py   ← валидация ансамбля
│   │   ├── feature_validation.py    ← валидация признаков
│   │   ├── final_review.py          ← финальная проверка системы
│   │   ├── labeling_validation.py   ← валидация разметки
│   │   ├── logging_observability.py ← observability логирование
│   │   └── onnx_validation_report.json
│   │
│   ├── backtest/
│   │   └── optinus_runner.py        ← walk-forward бэктест
│   │
│   ├── optuna_studies/
│   │   └── best_params.json         ← лучшие найденные Optuna параметры
│   │
│   └── tests/
│       └── test_risk_integration.py ← интеграционный тест риск-системы
│
└── tui/                             ← Terminal User Interface (реалтайм)
    ├── __main__.py                  ← python -m tui
    ├── app.py                       ← TradingTUI (Textual App)
    ├── orchestrator.py              ← Orchestrator (async polling loop)
    ├── state.py                     ← AppState (реактивное состояние)
    ├── config.py                    ← AppConfig (конфиг TUI)
    ├── history.py                   ← PredictionHistory
    ├── exchange/
    │   ├── base.py                  ← ExchangeAdapter (абстрактный класс)
    │   └── mexc.py                  ← MexcAdapter (aiohttp, публичный API)
    ├── features/
    │   └── builders.py              ← 4 feature builder класса для инференса
    ├── model_registry/
    │   └── registry.py              ← ONNXModel, ModelEnsemble, TradingPrediction
    ├── panels/
    │   ├── header.py                ← HeaderBar (топ-бар)
    │   ├── status_bar.py            ← StatusBar (нижний бар)
    │   ├── chart_panel.py           ← ASCII ценовой график
    │   ├── model_panel.py           ← таблица предсказаний трёх моделей
    │   ├── meta_panel.py            ← агрегированный мета-сигнал
    │   ├── risk_panel.py            ← визуальный риск-датчик
    │   ├── signal_panel.py          ← детали торгового сигнала (entry/SL/TP)
    │   └── history_panel.py         ← последние 20 предсказаний
    └── overlays/
        ├── help.py                  ← HelpOverlay (горячие клавиши)
        ├── menu.py                  ← MenuOverlay (7 тем, настройки, выход)
        ├── settings.py              ← SettingsOverlay (интервал, панели)
        └── symbol_search.py         ← SymbolSearchOverlay (поиск символа MEXC)
```

---

## 3. Архитектура системы

### Поток данных (production / TUI)

```
MEXC Public API
      ↓  (aiohttp, tui/exchange/mexc.py)
OHLCV DataFrames [5m×350, 60m×350, 1d×450]
      ↓  (tui/features/builders.py в ProcessPoolExecutor)
Feature Vectors [scalp×42, intraday×47, swing×70+, risk×32+]
      ↓  (tui/model_registry/registry.py ONNXModel.predict_single)
Predictions: P_up / P_flat / P_down × 3 models + risk_score
      ↓  (ModelEnsemble._compute_meta)
MetaSignal: LONG / SHORT / NEUTRAL  +  meta_score  +  meta_confidence
      ↓  (AppState → TUI panels)
Отображение: HeaderBar + MetaPanel + ModelPanel + RiskPanel + SignalPanel + ChartPanel
```

### Поток данных (batch / train)

```
Binance Futures API (fapi.binance.com)
      ↓  (project/data/data_collector.py)
Trade-level CSV файлы
      ↓  (project/data/data_loader.py)
OHLCV + derivatives данные (Polars, lazy evaluation)
      ↓  (project/features/*.py, Numba JIT)
150+ признаков на 3 таймфрейма
      ↓  (train_with_risk.py: фазы 1-3)
LightGBM обучение (TimeSeriesSplit, early stopping)
      ↓  (фаза 4)
Risk model (multi-objective: AUC + CVaR + Sharpe)
      ↓  (фаза 5: onnx_utils.py)
ONNX экспорт + валидация (parity check vs LightGBM)
      ↓  (фаза 6)
Risk assessment JSON + версионированные конфиги
```

### Meta Decision Formula

$$\text{score\_long} = 0.5 \cdot P_{\uparrow}^{scalp} + 0.3 \cdot P_{\uparrow}^{intraday} + 0.2 \cdot P_{\uparrow}^{swing}$$

$$\text{score\_short} = 0.5 \cdot P_{\downarrow}^{scalp} + 0.3 \cdot P_{\downarrow}^{intraday} + 0.2 \cdot P_{\downarrow}^{swing}$$

**Условие входа:**
1. `score > entry_threshold` (TUI: 0.35 / project batch: 0.65)
2. Swing модель не противоположна направлению
3. `risk_score < max_risk_score` (0.7)

---

## 4. Конфигурация

### `project/config.py` — жёсткие константы

| Константа | Значение | Описание |
|---|---|---|
| `SCALP_TIMEFRAMES` | `["5m","15m"]` | Таймфреймы scalp |
| `INTRADAY_TIMEFRAMES` | `["1h","4h","12h"]` | Таймфреймы intraday |
| `SWING_TIMEFRAMES` | `["1d","3d","1w"]` | Таймфреймы swing |
| `PRICE_MOVE_THRESHOLD` | scalp=0.3%, intraday=1%, swing=3% | Порог движения для лейбла |
| `PREDICTION_HORIZON` | scalp=12, intraday=6, swing=7 баров | Горизонт предсказания |
| `META_WEIGHTS` | scalp=0.5, intraday=0.3, swing=0.2 | Веса в мета-агрегации |
| `ENTRY_THRESHOLD` | 0.35 (batch: 0.65) | Порог уверенности для входа |
| `MAX_RISK_SCORE` | 0.7 | Максимальный риск-скор |
| `MAX_RISK_PER_TRADE` | 2% | Максимальный риск на сделку |
| `MAX_CONCURRENT_POSITIONS` | 5 | Максимум одновременных позиций |
| `MAX_CORRELATION` | 0.7 | Максимальная корреляция между позициями |
| `LEVERAGE_MAP` | low=5×, medium=3×, high=1× | Плечо по уровню риска |
| `TOP_SYMBOLS` | 20 монет (BTC, ETH, BNB, XRP, SOL, ADA...) | Торгуемые инструменты |
| `RANDOM_STATE` | 42 | Seed |

**LightGBM параметры (market models):**
- `objective="multiclass"`, `num_class=3` (LONG / FLAT / SHORT)
- `num_leaves=31`, `learning_rate=0.05`, `feature_fraction=0.8`, `bagging_fraction=0.8`

**LightGBM параметры (risk model):**
- `objective="regression"` — предсказывает непрерывный risk_score

### `project/config.yaml` — настраиваемая конфигурация

| Секция | Ключевые параметры |
|---|---|
| `cpu` | `usage_percent: 1.0` (все ядра), `exact_cores` опционально |
| `ram` | `usage_percent: 0.92`, `max_gb: 300` |
| `lightgbm` | `max_bin=255`, `num_leaves=127`, `max_depth=12`, `learning_rate=0.05`, `early_stopping_rounds=50`, `num_boost_round=1000` |
| `optuna` | `n_trials=20`, `timeout_per_trial=null` |
| `features` | `extended=true`, `sma_periods=[3,5,7,10,14,20,30,50,100,200]`, `rsi_periods=[5,7,9,14,21,28]` |
| `risk` | `max_leverage=10`, `drawdown_warning=5%`, `cvar_confidence=0.95`, `kelly_fraction=0.25` |
| `paths` | `data`, `models`, `logs`, `cache` (Linux пути по умолчанию) |

**Важно:** `paths` в config.yaml настроены под Linux (`/home/ai/...`). При работе на Windows необходимо обновить пути.

### `tui/config.py` — конфигурация TUI

| Параметр | Значение |
|---|---|
| `mexc_base_url` | `https://api.mexc.com/api/v3` |
| `mexc_timeout` | 15 сек |
| `models_dir` | `../project/models` |
| `timeframes.scalp` | 5m × 350 баров |
| `timeframes.intraday` | 60m × 350 баров |
| `timeframes.swing` | 1d × 450 баров |
| `cache_ttls` | 5m→30с, 60m→120с, 1d→600с |
| `refresh_intervals` | 5m→60с, 60m→300с, 1d→900с |
| `entry_threshold` | 0.35 (TUI использует 0.35, в отличие от batch 0.65) |
| `AppConfig.load(path)` | опциональный JSON-оверрайд для mexc и models_dir |

---

## 5. Модели данных и обучение

### Три торговые модели

| Модель | Таймфреймы | Тип | Файлы |
|---|---|---|---|
| **Scalp** | 5m, 15m | 3-class classification (LONG/FLAT/SHORT) | `scalp_lgbm.onnx`, `scalp_lgbm_features.json` |
| **Intraday** | 1h, 4h, 12h | 3-class classification | `intraday_lgbm.onnx`, `intraday_lgbm_features.json` |
| **Swing** | 1d, 3d, 1w | 3-class classification | `swing_lgbm.onnx`, `swing_lgbm_features.json` |

**Выход каждой модели:** `P_down[0]`, `P_flat[1]`, `P_up[2]` + `expected_return = P_up - P_down`

### Risk модель

- **Тип:** Regression (предсказывает непрерывный риск-скор 0.0–1.0)
- **Файлы:** `risk_lgbm.onnx`, `risk_lgbm_features.json`
- **Multi-objective обучение:** AUC vs CVaR penalty vs Sharpe ratio

### Параметры последней тренировки (swing, 2026-02-02)

| Метрика | Значение |
|---|---|
| CV Score | 95.07% ± 3.39% |
| CV Accuracy | 55.0% ± 3.1% |
| Best iteration | 444 |
| n_features | 81 (risk: 32) |
| n_samples | 187,331 |
| Train time | 44.9 сек |
| Throughput | 4,173 samples/sec |

### `train.py` — унифицированный тренер (1,500 строк)

**4 режима:**
- `--fast`: пропустить Optuna, использовать дефолтные параметры
- `--quality`: расширенная настройка, более глубокие деревья
- `--best`: использовать `optuna_studies/best_params.json`
- `--trials N`: переопределить количество trials Optuna

### `train_with_risk.py` — полный пайплайн (2,500 строк)

**6 фаз:**
1. Загрузка данных (Polars lazy evaluation)
2. Построение 150+ признаков (векторизовано, Numba JIT)
3. Обучение торговых моделей (TimeSeriesSplit, early stopping)
4. Обучение risk модели (multi-objective)
5. ONNX экспорт + валидация
6. Генерация risk assessment

### `training/system_config.py` — автонастройка железа

Функция `detect_system()` определяет CPU/RAM/GPU и возвращает `SystemInfo`.  
Функция `get_optimal_config(system)` возвращает оптимальные параметры:
- `n_data_workers = max(1, physical_cores - 1)`
- `n_feature_workers = max(1, physical_cores)`
- `n_cv_workers = min(5, max(1, physical_cores // 2))`
- `max_memory_gb = available_ram * 0.8`

---

## 6. Feature Engineering

### Scalp Features (`features/scalp_features.py`) — 50+ признаков, 5m/15m

| Группа | Признаки |
|---|---|
| Returns | Log returns (1, 3, 5, 10 периодов) |
| RSI | RSI (5, 7, 14) + slope RSI |
| Volatility | Micro-vol (3, 5, 10), ATR (5, 10) |
| Volume | Volume ratios (5, 10, 20), тейкерский объём |
| Price Action | Bar position, body, shadows, gap |
| Momentum | EMA crosses (3/8/13), Stochastic K/D, ROC |
| Derivatives | Funding rate, OI, LS ratio, premium/basis |

### Intraday Features (`features/intraday_features.py`) — 45+ признаков, 1h/4h/12h

| Группа | Признаки |
|---|---|
| Trend | SMA (10, 20, 50), VWAP, slope SMAs |
| Volatility | ATR, Bollinger Bands |
| Momentum | RSI (14), MACD (12/26/9) |
| Correlation | BTC correlation (для не-BTC символов) |
| Volume | Volume ratio |
| Derivatives | Funding, OI, LS ratio, taker volume, premium |

### Swing Features (`features/swing_features.py`) — 70+ признаков, 1d/3d/1w

| Группа | Признаки |
|---|---|
| Trend | MA (20, 50, 100, 200) |
| Regime | Market regime (bull/bear/flat), breakout |
| Macro | Macro volatility |
| Structure | Support/resistance levels |
| Volume | Volume analysis |
| Derivatives | Funding 7d/30d, OI change, LS ratio, taker, premium/basis |

**Примечание:** `swing_features.py` содержит дублирующийся `return` в конце (мёртвый код, не влияет на функциональность).

### Risk Features (`risk/risk_features.py`) — 32+ признаков для risk модели

| Группа | Кол-во | Описание |
|---|---|---|
| Volatility | 15 | vol_ratio, ATR, Bollinger, clustering |
| Tail Risk | 8 | kurtosis, skewness, VaR, CVaR, extreme counts |
| Liquidity | 10 | volume ratio, illiquidity, spread |
| Market Regime | 10 | trend strength, bull/bear/flat, breakout |
| Correlation | 5 | autocorr, vol-return, momentum consistency |
| Signal Quality | 8 | rolling Sharpe, IR stability, degradation |
| Drawdown | 5 | depth, duration, recovery speed |
| Derivatives | 8 | funding, OI, LS ratio extremes |
| Model Predictions | 4 | confidence, P_up/down, uncertainty (intraday) |
| **Target** | 1 | max_adverse_excursion + stop_loss + vol_spike |

### TUI Feature Builders (`tui/features/builders.py`)

Точная репликация training-time feature computation для live-инференса.

| Класс | Признаков | Таймфрейм |
|---|---|---|
| `ScalpFeatureBuilder` | 42 | 5m |
| `IntradayFeatureBuilder` | 47 | 60m |
| `SwingFeatureBuilder` | 70+ | 1d |
| `RiskFeatureBuilder` | 32+ | 1h (принимает `model_predictions` dict) |

**Нормализация:** `StandardScaler` подгоняется на скачанном окне, затем берётся последняя строка как вектор для инференса.

---

## 7. Риск-менеджмент

### Архитектура (три независимых модуля + пайплайн)

```
TradeProposal (предложение сделки)
        ↓
CVaREngine.check()          ← хвостовой риск портфеля
        ↓
KellySizer.calculate()      ← оптимальный размер позиции
        ↓
DrawdownController.check()  ← динамическая реакция на просадку
        ↓
RiskModel.evaluate()        ← ФИНАЛЬНОЕ ВЕТО
        ↓
RiskDecision: APPROVED / SIZE_REDUCED / REJECTED
```

### CVaR Engine (`risk/cvar_engine.py`) — хвостовой риск

- **Метод:** Исторический CVaR = E[Loss | Loss > VaR₉₅]
- **Лимиты по модели:**

| Модель | CVaR лимит |
|---|---|
| Scalp | 5% |
| Intraday | 7% |
| Swing | 10% |
| Portfolio | 8% |

- **Стресс-сценарии:** множитель 1.5×
- **Буфер PnL:** последние 1000 записей
- **Действие:** блокирует если CVaR > threshold

### Kelly Sizer (`risk/kelly_sizing.py`) — размер позиции

| Параметр | Значение |
|---|---|
| `MAX_KELLY_FRACTION` | 0.25 (никогда не используется сырой Kelly) |
| Коэффициент уверенности | 0.15–1.0 |
| Коэффициент волатильности | 0.3–1.5 |
| Лимит плеча (scalp) | 10× |
| Лимит плеча (intraday) | 5× |
| Лимит плеча (swing) | 3× |
| `min_win_rate` | 35% |
| `max_position` | 15% от портфеля |

**Формула:** `kelly = (p * b - (1-p)) / b` × dampening factor × volatility adjustment

### Drawdown Controller (`risk/drawdown_controller.py`) — динамический отклик

**Нелинейный множитель позиции:**

| Просадка | Множитель |
|---|---|
| 0–5% | 1.0 (полный размер) |
| 5–10% | 0.75 |
| 10–15% | 0.5 |
| 15–20% | 0.25 |
| 20%+ | 0.0 (стоп торговли) |

**Отключение моделей:**
- Scalp отключается при просадке ≥ 8%
- Intraday отключается при просадке ≥ 12%
- Swing отключается при просадке ≥ 18%

**Восстановление:**
- При восстановлении до 50% → режим REDUCED
- При восстановлении до 80% → режим FULL

**5 уровней серьёзности + emergency mode**

### Inference Pipeline (`risk/inference_pipeline.py`) — продакшн безопасность

| Функция | Описание |
|---|---|
| Kill-switch | Немедленная остановка всей торговли |
| Latency enforcement | Блок если инференс > 100ms |
| Regime shift detection | KS-тест на распределение признаков |
| Signal degradation | Мониторинг Sharpe drift |
| Thread safety | `threading.Lock` |
| Modes | `PRODUCTION / BACKTEST / DEBUG` |

**Метрики мониторинга (`MonitoringMetrics`):**
`total_inferences`, `blocked_trades`, `avg_latency_ms`, `max_latency_ms`, `regime_shifts_detected`, `signal_degradation_alerts`, `kill_switch_triggers`, `uptime_hours`

### Risk Model (`risk/risk_model.py`) — центральный контроллер

**Входные данные `TradeProposal`:**
`model_type, symbol, direction, probability, confidence, expected_return, volatility_regime, proposed_size, stop_loss, take_profit`

**Выходные данные `RiskDecision`:**
`decision, approved_size, adjusted_leverage, reason, risk_state, cvar_result, kelly_result, drawdown_metrics, recommendations, timestamp`

**Решения:** `APPROVED` / `SIZE_REDUCED` / `REJECTED`

---

## 8. Meta Engine

**Файл:** `project/meta/meta_engine.py`

**`MetaDecisionEngine`** — правило-основанная (НЕ ML) агрегация:
- Веса нормализованы так, чтобы сумма = 1.0
- Дефолты из `config.META_WEIGHTS` (scalp=0.5, intraday=0.3, swing=0.2)
- `entry_threshold = 0.65` (batch) / `0.35` (TUI)
- `max_risk_score = 0.7`

**`MetaDecision` dataclass:**
`symbol, direction (LONG/SHORT/NEUTRAL), confidence, score_long, score_short, timeframe_used, expected_return, risk_score, should_trade, reason`

**`CoinRanker`** (`meta/ranking.py`) — рейтинг монет:
- Ранжирует `TOP_COINS_COUNT = 10` монет для торговли
- Использует volume, volatility, trend consistency

---

## 9. Signal Generator

**Файл:** `project/signals/signal_generator.py`

**`TradingSignal` dataclass:**

| Группа | Поля |
|---|---|
| Ядро | `symbol, direction (LONG/SHORT), confidence` |
| Вход | `entry_price, entry_range_low, entry_range_high` |
| Риск | `stop_loss, take_profit_1, take_profit_2` |
| Размер | `leverage: int, risk_percent, position_size_pct` |
| Мета | `timeframe, risk_score, expected_return` |
| Время | `generated_at, valid_until` |

**Правила:**
- Max риск на сделку: 1-2%
- Max одновременных позиций: 3-5
- Проверка корреляции между активами
- Нет входа в режиме высокой волатильности

**TUI Signal Panel** (`tui/panels/signal_panel.py`) вычисляет:
- `SL = entry ± 1.5 × ATR`
- `TP1 = entry ± 2.25 × ATR`
- `TP2 = entry ± 3.75 × ATR`
- Плечо: LOW=5×, MEDIUM=3×, иначе=2×

---

## 10. TUI — Terminal User Interface

**Запуск:** `python -m tui` из корня проекта

**Технология:** [Textual](https://github.com/Textualize/textual) (Python TUI framework), btop-inspired дизайн

### Горячие клавиши

| Клавиша | Действие |
|---|---|
| `S` | Поиск символа (MEXC live search) |
| `P` | Пауза/продолжение обновлений |
| `C` | Очистить историю |
| `R` | Принудительное обновление |
| `1/2/3` | Изолировать одну модель в ModelPanel |
| `0` | Показать все модели |
| `Esc` | Меню (темы/настройки/помощь/выход) |
| `Ctrl+Q` | Выход |

### Компоненты TUI

**`TradingTUI`** (app.py) — главное Textual приложение:
- `TITLE = "ML Finance Risk System"`
- 7 тем: dark, dracula, monokai, nord, gruvbox, tokyo-night, textual-light
- Layout: HeaderBar → (MetaPanel + ModelPanel + RiskPanel) → ChartPanel → (SignalPanel + HistoryPanel) → StatusBar

**`Orchestrator`** (orchestrator.py) — асинхронный координатор:
- Polling loop с default `_refresh_interval = 60 сек`
- `_run_inference(symbol)`: загружает 3 OHLCV таймфрейма, строит 4 векторов признаков (в `ProcessPoolExecutor`), запускает `ModelEnsemble.predict()`
- Кэширование: Memory cache (50 entries) + Disk cache, TTL-based

**`AppState`** (state.py) — реактивное хранилище состояния:
- `status`: `LIVE / STALE / ERROR / STARTING / COMPUTING`
- `result: PredictionResult` — последний результат инференса
- `history: List[HistoryEntry]` — последние 100 предсказаний (FIFO)
- `chart_closes/opens` — данные для ChartPanel
- `last_price, last_atr` — для SignalPanel
- `refresh_interval: float = 60.0`
- `paused: bool`, `model_view: int`

### Панели (panels/)

| Панель | Что показывает |
|---|---|
| `HeaderBar` | Символ, статус, UTC часы, latency ms, горячие клавиши |
| `MetaPanel` | ▲ LONG / ▼ SHORT / ◆ NEUTRAL + score bar (-1 до +1) + причина |
| `ModelPanel` | Таблица: Model / Signal / Conf bar / P↑ / P─ / P↓ / E[R] |
| `RiskPanel` | Вертикальный датчик 10 рядов (зелёный→жёлтый→оранжевый→красный) |
| `SignalPanel` | Entry range, SL, TP1, TP2, leverage (если не NEUTRAL) |
| `ChartPanel` | ASCII блок-символы ▁▂▃▄▅▆▇█, до 55 свечей, высота 7 строк |
| `HistoryPanel` | Последние 20 предсказаний: время / символ / сигнал / conf / risk |
| `StatusBar` | Data age, cache hit/miss, countdown, error count, PAUSED |

### Биржевой адаптер (exchange/)

**`MexcAdapter`** (mexc.py) — публичный API без ключей:
- `aiohttp` + retry (3 попытки, backoff)
- Парсинг Kline: 8-column и 12-column форматы MEXC
- `validate_symbol()` → `/ticker/price` endpoint
- `search_symbols(query)` → кэширует все тикеры, prefix-фильтр

### ONNX Registry (model_registry/registry.py)

**`ONNXModel`:**
- ORT сессия: `intra/inter_op_num_threads=2`, CPU-only
- Feature names из компаньонского JSON
- model_type из ONNX metadata (fallback: имя файла)

**`ModelEnsemble`:**
- Загружает 4 ONNX файла из `models_dir`
- `predict()` → `PredictionResult` с метрасигналом
- `get_intraday_scalars()` → dict для RiskFeatureBuilder

---

## 11. Бэктест

**Файл:** `project/backtest/optinus_runner.py`

**Ключевые характеристики:**
- Walk-forward валидация с out-of-sample тестом
- Реалистичные торговые издержки: комиссия + slippage
- Анализ по рыночным режимам
- Noise testing

**`Trade` dataclass:**
`symbol, direction, entry_time/price, exit_time/price, stop_loss, take_profit, size, pnl, pnl_pct, commission, slippage, exit_reason (tp/sl/signal/timeout)`

**`BacktestResult` dataclass:**
`start_date, end_date, initial_capital, final_capital, total_return, total_return_pct, annualized_return, max_drawdown, max_drawdown_pct, sharpe_ratio, sortino_ratio, calmar_ratio, ...`

**Зависимость:** опциональный `import optinus` (third-party), fallback на встроенный бэктестер.

---

## 12. Данные и источники

### `data/data_collector.py` — `BinanceDataCollector`

**Источники данных:**
| Тип | API |
|---|---|
| OHLCV spot | `https://api.binance.com` |
| OHLCV futures | `https://fapi.binance.com` |
| Funding rate | Binance FAPI |
| Open Interest | Binance FAPI |
| Taker volume | Binance FAPI |
| BTC correlation | Вычисляется из OHLCV |
| BTC dominance | Внешний источник |

**Режимы:** dual-mode — `python-binance` если установлен, иначе прямые REST запросы.  
HTTP: `requests.adapters.HTTPAdapter` + `urllib3.util.retry.Retry`.

**TUI обменник:** использует **MEXC** (не Binance), публичный API, без API ключа.

### Форматы данных

- **Обучение:** CSV файлы с trade-level данными → конвертация в OHLCV через `data_loader.py`
- **TUI инференс:** прямой MEXC `/klines` endpoint → DataFrame с колонками `open_time, open, high, low, close, volume`

---

## 13. ONNX инфраструктура

**Файл:** `project/onnx_utils.py`

**Принципы:**
- CPU-only (нет GPU зависимости в проде)
- Детерминированный инференс (`ORT_SEQUENTIAL`, `ORT_ENABLE_ALL`)
- Версионированное сохранение (timestamp в имени файла)
- Parity validation: сравнение с LightGBM выходом

**Model metadata (в ONNX файле):**
- `model_type`: `scalp / intraday / swing / risk`
- Feature names в companion JSON файле

**Артефакты в `project/models/`:**
```
scalp_lgbm.onnx           + scalp_lgbm_features.json
intraday_lgbm.onnx        + intraday_lgbm_features.json
swing_lgbm.onnx           + swing_lgbm_features.json
risk_lgbm.onnx            + risk_lgbm_features.json
scalp_lgbm.txt            ← LightGBM текстовый формат (для импорта)
intraday_lgbm.txt
swing_lgbm.txt
risk_lgbm.txt
risk_engine_config_*.json ← версионированные конфиги риск-движка
training_results_*.json   ← JSON с метриками тренировки
training_assessment.json  ← итоговая оценка
```

---

## 14. Зависимости и требования

### `project/requirements.txt` — основные зависимости

| Пакет | Назначение |
|---|---|
| `lightgbm` | Градиентный бустинг |
| `onnxruntime` | Быстрый ONNX инференс |
| `onnx`, `onnxmltools` | ONNX экспорт |
| `optuna` | Hyperparameter optimization |
| `pandas`, `numpy` | Обработка данных |
| `scikit-learn` | Preprocessing, evaluation |
| `scipy` | Статистика (CVaR, KS-тест) |
| `polars` | Lazy evaluation загрузки данных |
| `numba` | JIT-компиляция (опционально, 10-100× speedup) |
| `python-binance` | Binance API (опционально) |
| `requests` | HTTP REST |
| `matplotlib`, `seaborn` | Визуализация (не для проде) |
| `ta` | Технический анализ |
| `joblib` | Параллельная обработка |

### `tui/requirements.txt` — TUI зависимости

| Пакет | Назначение |
|---|---|
| `textual` | TUI framework |
| `aiohttp` | Async HTTP для MEXC |
| `onnxruntime` | ONNX инференс |

**Требования Python:** 3.9+

---

## 15. Быстрый старт

```bash
# 1. Клонировать и войти
cd ml-finance-risk-system/project

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Подготовить данные (CSV торгов в data/)
mkdir data
# cp /path/to/*.csv data/

# 4. Обновить пути в config.yaml (если Windows)
# paths.data, paths.models, paths.logs, paths.cache

# 5. Обучить все модели
python train_with_risk.py

# 6. Запустить TUI (реалтайм)
cd ..
python -m tui
```

### Только TUI (без тренировки)

Модели уже есть в `project/models/`. Достаточно:
```bash
cd ml-finance-risk-system
pip install -r tui/requirements.txt
python -m tui
```

### Тренировка с флагами

```bash
python train.py --fast          # быстро, без Optuna
python train.py --best          # использовать best_params.json
python train.py --trials 50     # 50 Optuna trials
python train.py --quality       # максимальное качество
```

---

## Примечания для AI-агентов

- **Русский + английский:** комментарии в коде смешанные. `config.yaml` на русском.
- **ENTRY_THRESHOLD расхождение:** `config.py` имеет 0.35, `meta/meta_engine.py` документирует 0.65. TUI использует 0.35.
- **swing_features.py:** дублирующийся `return` в конце (мёртвый код, не функциональный баг).
- **Linux пути в config.yaml:** при работе на Windows обновить `paths` секцию.
- **Numba кэш:** первый запуск медленнее из-за JIT компиляции.
- **ProcessPoolExecutor в TUI:** feature building запускается в отдельном процессе для CPU-bound вычислений.
- **Validation отчёты:** все JSON отчёты в `project/validation/` содержат исторические результаты проверок.
