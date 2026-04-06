# Technical Specification — ML Finance Risk System

**Version:** 2.0-draft  
**Date:** 2026-04-06  
**Status:** Architecture Review

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Problems in Current System](#2-problems-in-current-system)
3. [Target Architecture](#3-target-architecture)
4. [TUI Design Specification](#4-tui-design-specification)
5. [Real-Time Data Pipeline](#5-real-time-data-pipeline)
6. [Model System Design](#6-model-system-design)
7. [Final Project Structure](#7-final-project-structure)
8. [Run Requirements](#8-run-requirements)

---

## 1. System Overview

### 1.1 What the System Currently Does

The system is a multi-model, multi-timeframe machine learning trading signal generator for cryptocurrency futures. As of v1, it operates as follows:

**Data Layer**  
Raw trade-level CSV files are collected from the Binance Futures API via `BinanceDataCollector`. A secondary loader (`data_loader_v2.py`) converts raw trades into OHLCV candles across three timeframe groups: scalp (5 m / 15 m), intraday (1 h / 4 h / 12 h), and swing (1 d / 3 d / 1 w).

**Feature Engineering**  
Three separate feature modules compute 42–56 technical indicators each. The scalp module produces 42 features centred on micro-volatility, log-returns, and short RSI. The intraday module produces 47 features including VWAP deviation, BTC correlation, funding-rate placeholders, and ADX. The swing module produces 56 features including MA-crossovers, market-regime encoding, BTC dominance proxies, and support/resistance levels. The risk module produces 32 features derived from volatility, drawdown, liquidity, price action, and live model outputs.

**Models**  
Four LightGBM models are trained and exported to ONNX:

| Model | Task | Features | Output |
|-------|------|----------|--------|
| `scalp_lgbm` | 3-class classification | 42 | P\_down / P\_flat / P\_up |
| `intraday_lgbm` | 3-class classification | 47 | P\_down / P\_flat / P\_up |
| `swing_lgbm` | 3-class classification | 56 | P\_down / P\_flat / P\_up |
| `risk_lgbm` | Regression | 32 | risk\_score ∈ [0, 1] |

**Meta Decision Engine**  
A deterministic rule-based engine (not ML) aggregates the three trading models using fixed weights: scalp = 0.5, intraday = 0.3, swing = 0.2. A trade is permitted only when the weighted confidence exceeds `ENTRY_THRESHOLD = 0.35`, swing direction does not oppose the signal, and `risk_score ≤ 0.7`.

**Signal Generator**  
Produces a `TradingSignal` per approved coin: ATR-based entry range, stop-loss, two take-profit levels (1 : 1.5 R:R and 1 : 2.5 R:R), leverage (1×/3×/5×), and position-size percentage capped at 20 % of capital. Maximum 5 concurrent positions.

**Backtesting**  
`OptinusBacktester` provides event-driven backtesting with commission (0.1 %) and slippage (0.05 %), walk-forward validation, out-of-sample testing, Gaussian noise robustness tests, and per-regime analysis (bull / flat / bear).

**Inference Path (mexc\_predictor)**  
A separate sub-project using the MEXC public REST API was partially implemented to fetch live OHLCV data, build features, and run inference via `onnxruntime`. It was functional at one point but the files are no longer present in the workspace.

### 1.2 What the System Should Do in Final Version

The final system must be a production-grade, real-time, interactive terminal application that:

- Connects to the MEXC exchange via its public REST API (no API key required for read-only data).
- Fetches, caches, and continuously refreshes OHLCV data for any user-specified trading pair.
- Computes all required features for all four ONNX models without re-training.
- Runs full-ensemble inference in under 200 ms per symbol.
- Presents results in a structured, keyboard-navigable Terminal User Interface (TUI) with multiple panels updating in real time.
- Allows the user to select which model(s) to observe, switch symbols, change timeframes, and inspect feature values — all without restarting the process.
- Displays risk assessment and ensemble meta-signal with clear directional indicators (LONG / NEUTRAL / SHORT).
- Persists prediction history to disk for post-session review.
- Runs from a single command with zero manual configuration required for the default use case.

---

## 2. Problems in Current System

### 2.1 Architecture Issues

**Split codebase with no shared interface**  
The `project/` directory and the `py_training/` directory are two independent codebases that duplicate responsibilities. `py_training/` contains its own model (`universal_model.onnx`), its own feature builder (`build_features.py`, `multi_timeframe.py`), and its own predictor — all disconnected from `project/`. There is no shared abstraction that both can consume.

**Exchange-to-exchange confusion**  
`config.py` hardcodes Binance API endpoints and credentials. The live inference path requires MEXC. These two exchange integrations were never reconciled into a single exchangeable adapter — there is no interface definition, no abstraction layer, and no configuration switch.

**Scalers not persisted**  
The feature engineering modules apply `StandardScaler` at training time over the full dataset. When running live inference, a fresh `StandardScaler` is fitted on the downloaded window (typically 300–450 candles). This means the live z-score normalisation is computed against a different statistical distribution than the training set. Because LightGBM splits are monotone transformations of the input and the model is tree-based, this is tolerable but introduces a latent consistency risk that should be formalised and documented as a deliberate design choice.

**No async anywhere**  
All data fetching, feature building, and inference is synchronous. On a 20-symbol scan, sequential HTTP calls to MEXC at 3–5 calls per symbol create a 60–100 second blocking delay. There is no event loop, no worker pool, and no separation of I/O-bound from CPU-bound work.

**Binance API credentials hardcoded as empty strings**  
`BINANCE_API_KEY = ""` and `BINANCE_API_SECRET = ""` are commit-level secrets placeholders in `config.py`. If real credentials are ever placed here they are at risk of accidental exposure. There is no `.env` file, no secrets management layer, and no documentation of how credentials should be injected.

**Two data loaders with unclear versioning**  
`data_loader.py` and `data_loader_v2.py` coexist with no documented distinction. Code in `main.py` imports from `data_loader.py`; the v2 loader supports derivatives data and appears more capable. The older loader is never explicitly deprecated.

### 2.2 Performance Issues

**Trend-slope calculation is O(n²)**  
In `intraday_features.py`, `add_trend_features()` computes per-bar `np.polyfit` inside a py loop, making it O(n × window) per column. For 300-bar windows at 20 symbols this is the dominant CPU cost and blocks the main thread for several seconds.

**No batching of ONNX inference**  
Each symbol is inferred one sample at a time (`predict_single`), discarding the batching efficiency that `onnxruntime` provides. For a 20-symbol scan, 4 models × 20 symbols = 80 sequential ORT `run()` calls when 4 batch calls of size 20 would suffice.

**Cache not shared across processes**  
The in-memory cache in `MexcClient` is per-instance and per-process. If the system is ever extended to run workers or a scheduler alongside the TUI, the cache provides no cross-process benefit.

**Rolling autocorrelation via `pd.Series.autocorr` inside `apply`**  
`RiskFeatureBuilder._add_autocorr()` calls a py lambda per rolling window, making it O(n × window) without any vectorisation benefit. This is the single most expensive individual feature computation.

### 2.3 Code Structure Issues

**No `__init__.py` in `mexc_predictor`**  
The inference sub-package was structured as a flat directory of scripts rather than a proper py package, making imports fragile and preventing use as a library.

**Feature names duplicated across JSON and py constants**  
Each model has a `*_features.json` sidecar file AND a `FEATURE_NAMES` constant in the corresponding training script. These can silently diverge if one is updated without the other.

**`optuna_studies/best_params.json` is not version-locked to model artifacts**  
There is no mechanism to verify that the stored Optuna hyperparameters correspond to the currently saved `.onnx` and `.txt` model files. A re-optimisation run could overwrite `best_params.json` while the models remain from a prior run.

**Dead code in `swing_features.py`**  
Duplicate `return` statements exist at the end of at least one function (noted in repository memory). This indicates that the file was last edited hastily without a test run.

**`TradingSystem.predict_all()` silently returns uniform priors on model-load failure**  
If any model fails to load, its outputs default to `{P_up: 0.33, P_flat: 0.33, P_down: 0.33}` without any warning to the user or operator. This means a broken model produces the same output as an uncertain model — a critical observability failure.

### 2.4 Missing Features

| Missing Feature | Impact |
|----------------|--------|
| No live TUI | User must parse raw console output or JSON files |
| No real-time data loop | Single-shot prediction only; no continuous monitoring |
| No symbol watchlist management | Cannot track multiple symbols simultaneously |
| No alert / notification system | No way to be notified of LONG/SHORT signals without polling |
| No model hot-reload | New model files require process restart |
| No prediction history viewer | Past signals only accessible by reading JSON files manually |
| No MEXC futures data (funding, OI) | Intraday features that depend on funding/OI are permanently zero-filled |
| No model confidence calibration | Raw probabilities from LightGBM are not temperature-scaled or Platt-calibrated |
| No position tracker | The system generates signals but has no state for tracking open positions |
| No paper trading mode | No way to simulate signal execution without real capital |

---

## 3. Target Architecture

### 3.1 Module Decomposition

The final system is organised into five independent layers communicating through well-defined interfaces:

```
┌─────────────────────────────────────────────┐
│                  TUI Layer                  │  ← Textual / Rich
│  (panels, keyboard nav, async rendering)    │
└──────────────────────┬──────────────────────┘
                       │ read-only subscriptions
┌──────────────────────▼──────────────────────┐
│             Orchestrator Layer              │  ← async event loop
│  (state machine, polling scheduler,         │
│   signal aggregation, history persistence)  │
└─────┬────────────────┬────────────────┬─────┘
      │                │                │
┌─────▼─────┐  ┌───────▼──────┐  ┌─────▼──────┐
│  Exchange │  │   Feature    │  │   Model    │
│  Adapter  │  │   Pipeline   │  │  Registry  │
│  Layer    │  │   Layer      │  │  Layer     │
└─────┬─────┘  └───────┬──────┘  └─────┬──────┘
      │                │               │
┌─────▼────────────────▼───────────────▼──────┐
│              Infrastructure Layer            │
│  (cache, config, logging, disk persistence) │
└─────────────────────────────────────────────┘
```

### 3.2 Data Flow (End-to-End)

```
MEXC REST API
     │
     ▼  (HTTP GET /api/v3/klines)
ExchangeAdapter.fetch_ohlcv(symbol, interval, limit)
     │  returns: pd.DataFrame [open, high, low, close, volume]
     │
     ▼  (TTL cache check: in-memory → disk fallback)
CacheStore.get_or_fetch(cache_key)
     │  returns: pd.DataFrame (cached or freshly fetched)
     │
     ▼
FeaturePipeline.build(df, model_type)
     │  ScalpFeatureBuilder   → np.ndarray shape (42,)
     │  IntradayFeatureBuilder → np.ndarray shape (47,)
     │  SwingFeatureBuilder   → np.ndarray shape (56,)
     │  RiskFeatureBuilder    → np.ndarray shape (32,)
     │
     ▼
ModelRegistry.run_ensemble(feature_vectors)
     │  scalp   → TradingPrediction  (P_up, P_flat, P_down, confidence)
     │  intraday → TradingPrediction
     │  swing   → TradingPrediction
     │  risk    → RiskPrediction     (risk_score, risk_level, max_leverage)
     │
     ▼
MetaEngine.aggregate(trading_preds, risk_pred)
     │  returns: MetaSignal  (direction, score, reason)
     │
     ▼
SignalAssembler.build(meta_signal, price, atr, risk)
     │  returns: TradingSignal (entry, sl, tp1, tp2, leverage, size)
     │
     ▼
Orchestrator.publish(symbol, signal)
     │  → PredictionHistory (disk append)
     │  → TUI StateStore (reactive update)
     │  → AlertEngine (threshold check)
```

### 3.3 Model Pipeline

The model pipeline is strictly read-only at inference time. No scaler is fitted during inference. Instead:

- Feature builders produce raw (un-normalised) feature vectors.
- A `FeatureNormaliser` component (separate from builders) is responsible for applying normalisation. It supports two modes:
  - **Window Z-Score Mode (default):** Fits a `StandardScaler` on the trailing `N` bars of computed features before extracting the last row. This is the current approach and is acceptable for live inference with tree models.
  - **Saved Scaler Mode (preferred):** Loads a `joblib`-serialised `StandardScaler` produced at training time, guaranteeing the exact distribution match. This mode is activated when a `*_scaler.pkl` file is present alongside the ONNX model.
- The `FeatureNormaliser` logs which mode is active on startup.

### 3.4 Exchange Adapter Design

All exchange interactions are routed through a single abstract interface `ExchangeAdapter`. The system ships with one concrete implementation: `MexcAdapter`. The adapter is responsible for:

- Symbol validation (single network call using the ticker endpoint).
- OHLCV fetching with configurable interval, limit, and retry policy.
- Response parsing that is tolerant of column-count changes in the MEXC API response (currently 8 columns; previously 12; the adapter detects and handles both).
- Rate-limit awareness (passive: insert delays between requests; not active throttling).

### 3.5 Meta Decision Engine (Unchanged Logic)

The meta engine logic is preserved from v1. Weights remain scalp = 0.5 / intraday = 0.3 / swing = 0.2. Thresholds remain configurable via `config.json`. The engine is deterministic — given identical model outputs it always produces the same meta signal. No change to this layer is required.

---

## 4. TUI Design Specification

### 4.1 Overview

The TUI is a full-screen terminal application built on the **Textual** framework. It renders entirely in the terminal using ANSI escape sequences and requires no browser, no GUI toolkit, and no display server. The application is keyboard-driven; mouse support is optional.

**Performance contract:** All UI renders must complete in under 16 ms (60 fps cap). Inference and data fetching are strictly off the main thread. The TUI thread never performs blocking I/O.

---

### 4.2 Layout — Panel Map

```
╔══════════════════════════════════════════════════════════════════════════╗
║  HEADER BAR: symbol selector │ timeframe │ status │ clock │ latency      ║
╠═════════════════════╦════════════════════════╦═══════════════════════════╣
║                     ║                        ║                           ║
║   META PANEL        ║   MODEL DETAIL PANEL   ║   RISK PANEL              ║
║   (main signal)     ║   (3 model rows)       ║   (score bar + advice)    ║
║                     ║                        ║                           ║
╠═════════════════════╩══════════╦═════════════╩═══════════════════════════╣
║                                ║                                         ║
║   SIGNAL PANEL                 ║   PREDICTION HISTORY PANEL              ║
║   (entry / SL / TP / leverage) ║   (scrollable, last 20 signals)         ║
║                                ║                                         ║
╠════════════════════════════════╩═════════════════════════════════════════╣
║  STATUS BAR: data age │ cache hit │ next refresh │ errors │ keybindings   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**Minimum terminal size:** 120 × 36 characters. The layout degrades gracefully to 80 × 24 by hiding the Prediction History Panel and compressing the Risk Panel.

---

### 4.3 Panel Descriptions

#### Header Bar
- **Symbol input field:** The user can type a symbol (e.g. `ETHUSDT`) and press Enter to switch. Invalid symbols produce an inline error without disrupting other panels.
- **Timeframe selector:** Cycle through `5m | 1h | 4h | 1d` using `[` and `]` keys. The selected timeframe applies to data display; models always run on their canonical timeframes.
- **Status indicator:** Three states — `LIVE` (green, data fresh), `STALE` (yellow, data older than 2× refresh interval), `ERROR` (red, last fetch failed).
- **Clock / Latency:** UTC clock updated every second. Last inference latency shown in milliseconds.

#### Meta Panel (left, tall)
- Dominant element: a large directional indicator.
  - `▲ LONG` rendered in bold green.
  - `▼ SHORT` rendered in bold red.
  - `◆ NEUTRAL` rendered in dim yellow.
- Below the indicator: a horizontal score bar representing the normalised meta score on a scale of –1.0 to +1.0. The bar is colour-coded: left half red (short), centre grey (neutral zone), right half green (long).
- Below the score bar: a short reason line (e.g. `"swing UP + intraday FLAT, risk LOW"`).
- The panel border flashes briefly (100 ms) on any signal change.

#### Model Detail Panel (centre)
Three rows, one per trading model:

| Column | Content |
|--------|---------|
| Name | `SCALP` / `INTRADAY` / `SWING` |
| Signal | `UP` / `FLAT` / `DOWN` with colour |
| Confidence bar | ASCII bar, 15 chars wide, filled proportionally |
| P↑ / P─ / P↓ | Numeric probabilities to 3 decimal places |
| E[R] | Expected return (P\_up − P\_down), signed |
| Age | Seconds since last update for this model |

The user can press `1`, `2`, `3` to isolate a single model (expand it to the full panel height, showing all 42–56 feature values in a scrollable sub-panel). Press `0` to return to three-row view.

#### Risk Panel (right)
- Risk score displayed as a vertical gauge (0 at bottom, 1.0 at top) with colour gradient: green → yellow → orange → red.
- Risk level text: `LOW` / `MEDIUM` / `HIGH` / `CRITICAL`.
- Maximum suggested leverage shown below the gauge.
- CVaR estimate (if backtest history is available): tail-risk percentage.
- Drawdown state: current portfolio drawdown as a percentage with a severity indicator.

#### Signal Panel (bottom-left)
Shown only when `meta_signal != NEUTRAL` and the signal passes risk checks.

| Field | Example |
|-------|---------|
| Direction | `LONG ▲` |
| Entry range | `$68,980 – $69,120` |
| Stop Loss | `$67,650  (−1.95 %)` |
| Take Profit 1 | `$70,810  (+2.65 %)  R:R 1:1.5` |
| Take Profit 2 | `$72,100  (+4.45 %)  R:R 1:2.5` |
| Leverage | `5×` |
| Position size | `8.4 % of capital` |
| Valid until | `07:45:00 UTC` |

Fields are colour-coded: SL in red, TP values in green.

#### Prediction History Panel (bottom-right)
Scrollable list, newest at top. Each row shows:
```
07:32:14  BTCUSDT  ▲LONG  conf=0.511  risk=LOW  score=+0.228
07:14:02  ETHUSDT  ◆NEUT  conf=0.421  risk=LOW  score=+0.031
06:55:49  BTCUSDT  ▲LONG  conf=0.498  risk=LOW  score=+0.195
```
Pressing `Enter` on a history row expands it to show full signal detail.

#### Status Bar
- `DATA: 3s ago` — age of the most-recently fetched candle set.
- `CACHE: HIT` / `MISS` — last data request cache outcome.
- `NEXT: 57s` — seconds until the next scheduled refresh.
- `ERR: 0` — count of errors since session start. Turns red if > 0.
- Keybinding hints: `[Q]uit  [R]efresh  [S]ymbols  [?]Help`

---

### 4.4 Interactions

| Key | Action |
|-----|--------|
| `Q` | Quit the application gracefully (flush history to disk first). |
| `R` | Force immediate data refresh bypassing cache. |
| `S` | Open symbol search overlay (type to filter MEXC symbols, up/down to select). |
| `[` / `]` | Decrease / increase display timeframe. |
| `1` / `2` / `3` | Isolate scalp / intraday / swing model panel. |
| `0` | Return to three-row model summary. |
| `F` | Toggle feature inspection overlay for the selected model. |
| `H` | Toggle prediction history panel full-screen. |
| `P` | Toggle pause (freeze TUI, data still updates in background). |
| `?` | Show help overlay with all keybindings. |
| `Ctrl+C` | Emergency exit (same as `Q`). |

All overlays (symbol search, help, feature inspection) are modal and close on `Escape`.

---

### 4.5 Real-Time Behaviour

- The TUI re-renders only the panels that have changed state (dirty-flag pattern). A global tick every 1 second updates the clock and age counters. A full panel refresh occurs only when new inference results arrive.
- Inference results are delivered to the TUI via an `asyncio.Queue`. The TUI coroutine dequeues results and marks affected panels dirty.
- If inference takes longer than the refresh interval, a `COMPUTING` spinner appears in the status bar and the previous results remain displayed until new ones arrive (no blank state).
- The history panel is pre-rendered to a string buffer and only scrolled, never re-computed, until a new row is appended.

---

### 4.6 Performance Requirements

| Metric | Target |
|--------|--------|
| TUI render cycle | < 16 ms |
| Symbol switch latency | < 500 ms (fetch + compute + render) |
| Inference latency (all 4 models) | < 200 ms |
| Data fetch latency (MEXC, 350 bars) | < 1 000 ms |
| Memory footprint (idle, 1 symbol) | < 150 MB |
| CPU usage (idle, no active fetch) | < 2 % |
| Startup time (first render) | < 3 s |

---

## 5. Real-Time Data Pipeline

### 5.1 Data Flow from MEXC API

```
User action / scheduler tick
          │
          ▼
  DataScheduler.should_refresh(symbol, interval)
  (checks: time elapsed since last fetch > refresh_interval)
          │  YES
          ▼
  ExchangeAdapter.fetch_ohlcv(symbol, interval, limit=350)
          │
          ├─ HTTP GET https://api.mexc.com/api/v3/klines
          │    params: symbol, interval, limit
          │    timeout: 15 s
          │    retries: 3 (exponential back-off: 2 s, 4 s, 8 s)
          │
          ▼
  ResponseParser.parse(raw_json)
          │  Detects 8-column or 12-column format
          │  Casts all OHLCV to float64
          │  Sets UTC DatetimeIndex
          │  Drops rows with NaN in any OHLCV column
          │
          ▼
  CacheStore.put(cache_key, df, ttl)
          │
          ▼
  FeaturePipeline.build_all(df, symbol)
          │  Returns: Dict[model_type → feature_vector]
          │
          ▼
  ModelRegistry.run_ensemble(feature_vectors)
          │  Returns: PredictionResult
          │
          ▼
  Orchestrator.on_new_result(symbol, result)
          ├─ PredictionHistory.append(symbol, result)
          └─ TUI.notify(symbol, result)
```

### 5.2 Caching Strategy

**Two-tier cache:**

**Tier 1 — In-process memory cache (`MemoryCache`)**  
- Storage: `dict[cache_key → (DataFrame, expiry_timestamp)]`
- TTL: configurable per interval.
  - `5m` candles → 30 s TTL (half a bar)
  - `60m` candles → 120 s TTL
  - `1d` candles → 600 s TTL
- Eviction: LRU with a maximum of 50 entries. Entries are evicted by access time when the limit is reached.
- Thread-safety: protected by `asyncio.Lock` (not `threading.Lock` — the system is single-threaded async).

**Tier 2 — Disk cache (`DiskCache`)**  
- Storage: Apache Parquet files in `~/.ml_finance/cache/`. File naming: `{symbol}_{interval}_{YYYYMMDD}.parquet`.
- Used as fallback when the memory cache is cold (application restart) and the MEXC API is unreachable.
- TTL: 1 calendar day for daily candles; not used for sub-hourly data (too stale).
- Written asynchronously after a successful API fetch.

**Cache key format:** `{symbol}:{interval}:{limit}`  
Example: `BTCUSDT:60m:350`

**Cache invalidation:** The cache is never manually invalidated. It expires naturally by TTL. The `R` key in the TUI triggers a forced bypass (fetch with `use_cache=False` then writes the result back to cache).

### 5.3 Update Frequency

| Interval | Refresh Period | Rationale |
|----------|---------------|-----------|
| `5m` candles | 60 s | One full bar every 5 m; refresh once per minute is sufficient |
| `60m` candles | 300 s | Bar closes every hour; 5-minute check is adequate |
| `1d` candles | 900 s | Bar closes once per day; 15-minute check is generous |

The scheduler uses a priority queue ordered by `next_refresh_due`. A single background coroutine polls the queue every second and dispatches fetches for any overdue entries.

### 5.4 Async Design

The entire pipeline is async-first. The application runs a single `asyncio` event loop. All external I/O operates through async primitives.

**Coroutine hierarchy:**

```
asyncio.run(app.run())
  │
  ├─ TUI.mount_and_render()              ← Textual app coroutine
  │
  ├─ DataScheduler.run_loop()            ← polling scheduler
  │    └─ ExchangeAdapter.fetch_ohlcv()  ← aiohttp session
  │
  ├─ InferenceWorker.run_loop()          ← consumes DataQueue
  │    ├─ FeaturePipeline.build_all()    ← CPU-bound (run in executor)
  │    └─ ModelRegistry.run_ensemble()   ← CPU-bound (run in executor)
  │
  └─ PersistenceWorker.run_loop()        ← consumes ResultQueue
       └─ PredictionHistory.append()     ← disk write
```

**CPU-bound work is offloaded to a `ProcessPoolExecutor`** with a maximum of 2 workers (one for feature building, one for inference). This prevents the async event loop from blocking during numpy operations.

**Error handling in the async pipeline:**
- A fetch failure increments an error counter visible in the status bar and retries after the back-off interval.
- An inference failure logs the exception and returns the last known `PredictionResult` for that symbol (stale result is marked with a `STALE` badge in the TUI).
- Unhandled exceptions in worker coroutines are caught at the top level, logged, and the worker is restarted automatically without crashing the TUI.

---

## 6. Model System Design

### 6.1 ONNX Model Loading

Each ONNX model is loaded by in a `ModelRecord` container at application startup. Loading proceeds in the following order:

1. **Discover** all `*.onnx` files in the `models/` directory.
2. **Match** each ONNX file to a companion `*_features.json` sidecar file. If no sidecar is found, the model is not loaded and a warning is emitted.
3. **Validate** the sidecar feature count against the ONNX model's declared input shape. A mismatch is a hard error that prevents that model from loading.
4. **Read metadata** from the ONNX model's `metadata_props` (specifically the `model_type` key). If the key is absent, the type is inferred from the filename.
5. **Create** an `onnxruntime.InferenceSession` with `CPUExecutionProvider` only. GPU is not used. `intra_op_num_threads` is set to 2.
6. **Register** the session in the `ModelRegistry` keyed by `model_type`.
7. **Optionally load** a companion `*_scaler.pkl` for the Saved Scaler normalisation mode. If absent, window Z-Score mode is used.

Model loading is performed once at startup and the sessions are reused for the lifetime of the process. There is no per-prediction session creation.

**Startup log output per model:**
```
[MODEL] Loaded scalp    | features=42 | normaliser=window_zscore | type=classification
[MODEL] Loaded intraday | features=47 | normaliser=window_zscore | type=classification
[MODEL] Loaded swing    | features=56 | normaliser=window_zscore | type=classification
[MODEL] Loaded risk     | features=32 | normaliser=window_zscore | type=regression
```

### 6.2 Multiple Model Interaction

The four models operate in a defined dependency order:

```
Step 1:  scalp model     → runs independently
Step 2:  intraday model  → runs independently (fetches BTC correlation data if symbol ≠ BTCUSDT)
Step 3:  swing model     → runs independently
         [Steps 1-3 can run concurrently]
Step 4:  risk model      → depends on step 2 output (intraday P_up/P_down used as features)
Step 5:  MetaEngine      → depends on steps 1–4
Step 6:  SignalAssembler → depends on step 5
```

**Concurrency:** Steps 1, 2, and 3 are scheduled as parallel `asyncio` tasks dispatched to the `ProcessPoolExecutor`. Step 4 waits for step 2 to complete (it requires the intraday probabilities as risk features). Steps 5 and 6 run synchronously in the event loop after all four model results are collected.

**Degraded mode:** If any of steps 1–3 fail (model not loaded, feature error), the meta engine still runs with the surviving model outputs. The missing model's contribution is replaced by a neutral prior `{P_up: 0.33, P_flat: 0.33, P_down: 0.33}` and a warning is shown in the TUI status bar. This is an **explicit, visible** fallback, not a silent one (unlike the current v1 behaviour).

If step 4 (risk model) fails, inference continues and the risk score defaults to `0.5` (medium risk). The TUI risk panel shows `N/A` with a warning icon.

### 6.3 Plugin System for Models

The `ModelRegistry` supports runtime plugin loading to allow adding new models without code changes.

**Plugin contract:** A model plugin is a directory placed inside `models/plugins/` containing:
- `model.onnx` — the ONNX model file.
- `features.json` — feature name list matching the model's input shape.
- `plugin.json` — metadata file with the following required keys:
  - `"name"` — unique identifier string (e.g. `"btc_daily_v2"`).
  - `"model_type"` — one of `"classification"` or `"regression"`.
  - `"weight"` — float in [0, 1], used by the meta engine if the plugin replaces a built-in model or contributes to a custom ensemble.
  - `"description"` — human-readable string shown in the TUI model info overlay.
  - `"input_timeframe"` — the candle interval this model was trained on (e.g. `"60m"`).
  - `"target_horizon_bars"` — integer, the prediction horizon during training.

**Plugin discovery:** At startup, the `ModelRegistry` scans `models/plugins/`. Each valid plugin directory is loaded with the same validation pipeline as built-in models (sidecar check, shape check, metadata check).

**Plugin activation in the TUI:** The `S` → `P` (Symbols → Plugins) menu lists all discovered plugins. The user can enable or disable individual plugins per session. Enabled plugins appear as additional rows in the Model Detail Panel and contribute to the meta score if they are typed as `"classification"` and assigned a non-zero weight.

**Hot-reload:** When the user presses `Ctrl+L` (Reload Models), the registry re-scans the `models/` and `models/plugins/` directories and loads any new or changed ONNX files without restarting the process. Existing sessions are not replaced unless the file modification time has changed.

---

## 7. Final Project Structure

```
ml-finance-risk-system/
│
├── README.md
├── docs/
│   └── SPEC.md                         ← This document
│
├── project/                            ← Main production package
│   │
│   ├── __init__.py
│   ├── config.py                       ← Central constants (thresholds, weights, limits)
│   │
│   ├── app.py                          ← Application entry point; wires all layers together
│   ├── orchestrator.py                 ← Async state machine; coordinates scheduler + inference + TUI
│   │
│   ├── exchange/
│   │   ├── __init__.py
│   │   ├── base.py                     ← Abstract ExchangeAdapter interface
│   │   ├── mexc.py                     ← MexcAdapter (currently authoritative)
│   │   └── binance.py                  ← BinanceAdapter (legacy, for training data collection)
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── memory.py                   ← In-process LRU cache
│   │   └── disk.py                     ← Parquet-backed persistent cache
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py                     ← Abstract FeatureBuilder interface
│   │   ├── normaliser.py               ← FeatureNormaliser (window Z-score or saved scaler)
│   │   ├── scalp_features.py           ← 42 features, 5 m / 15 m
│   │   ├── intraday_features.py        ← 47 features, 1 h / 4 h / 12 h
│   │   ├── swing_features.py           ← 56 features, 1 d / 3 d / 1 w
│   │   └── risk_features.py            ← 32 features, any timeframe
│   │
│   ├── models/
│   │   ├── scalp_lgbm.onnx
│   │   ├── scalp_lgbm_features.json
│   │   ├── intraday_lgbm.onnx
│   │   ├── intraday_lgbm_features.json
│   │   ├── swing_lgbm.onnx
│   │   ├── swing_lgbm_features.json
│   │   ├── risk_lgbm.onnx
│   │   ├── risk_lgbm_features.json
│   │   └── plugins/                    ← User-contributed ONNX plugins
│   │       └── (plugin directories as described in §6.3)
│   │
│   ├── model_registry/
│   │   ├── __init__.py
│   │   ├── registry.py                 ← ModelRegistry; loads, validates, serves models
│   │   ├── onnx_session.py             ← ONNXModel wrapper (single session lifecycle)
│   │   └── ensemble.py                 ← ModelEnsemble; coordinates multi-model inference
│   │
│   ├── meta/
│   │   ├── __init__.py
│   │   ├── meta_engine.py              ← Rule-based signal aggregation (weights, thresholds)
│   │   └── ranking.py                  ← CoinRanker; scores and filters multi-symbol candidates
│   │
│   ├── signals/
│   │   ├── __init__.py
│   │   └── signal_generator.py         ← TradingSignal construction (SL, TP, leverage, size)
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── cvar.py                     ← CVaR engine (historical tail risk)
│   │   ├── kelly.py                    ← Kelly criterion position sizer
│   │   └── drawdown.py                 ← Drawdown controller (dynamic leverage reduction)
│   │
│   ├── scheduler.py                    ← DataScheduler; priority queue of pending fetches
│   │
│   ├── history.py                      ← PredictionHistory; append-only JSON-L file + in-memory buffer
│   │
│   ├── tui/
│   │   ├── __init__.py
│   │   ├── app.py                      ← Textual Application subclass
│   │   ├── state.py                    ← Reactive state store (TUI reads from this)
│   │   ├── panels/
│   │   │   ├── header.py               ← Header bar widget
│   │   │   ├── meta_panel.py           ← Meta signal display
│   │   │   ├── model_panel.py          ← Three-row model detail view
│   │   │   ├── risk_panel.py           ← Risk gauge and advice
│   │   │   ├── signal_panel.py         ← Entry / SL / TP display
│   │   │   ├── history_panel.py        ← Scrollable history list
│   │   │   └── status_bar.py           ← Bottom status bar widget
│   │   └── overlays/
│   │       ├── symbol_search.py        ← Modal symbol search overlay
│   │       ├── feature_inspector.py    ← Feature value inspection overlay
│   │       └── help.py                 ← Keybinding help overlay
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── backtester.py               ← CustomBacktester + OptinusBacktester (unchanged from v1)
│   │
│   └── training/
│       ├── __init__.py
│       ├── train_scalp.py
│       ├── train_intraday.py
│       ├── train_swing.py
│       └── train_risk.py
│
├── py_training/                    ← Legacy research notebook (not used in production)
│   └── (existing files, read-only)
│
├── optuna_studies/
│   └── best_params.json
│
└── requirements.txt
```

**Module responsibilities:**

| Module | Responsibility |
|--------|---------------|
| `app.py` | Single entry point; reads config, instantiates all layers, starts the event loop |
| `orchestrator.py` | Central async coroutine; wires data scheduler → inference pipeline → TUI state store |
| `exchange/mexc.py` | All MEXC HTTP interactions; retry logic; response parsing |
| `cache/memory.py` | TTL-based in-process LRU cache; asyncio-safe |
| `cache/disk.py` | Parquet-backed stale-data fallback; async writes |
| `features/*.py` | Pure-function feature builders; accept `pd.DataFrame`, return `np.ndarray` |
| `features/normaliser.py` | Wraps feature builders; applies scaling before returning inference vector |
| `model_registry/registry.py` | Discovers, validates, and hot-reloads ONNX models and plugins |
| `model_registry/ensemble.py` | Runs all active models in parallel; assembles `PredictionResult` |
| `meta/meta_engine.py` | Rule-based signal aggregation; never modifies model weights |
| `signals/signal_generator.py` | ATR-based SL/TP/leverage/position-size computation |
| `risk/*.py` | CVaR, Kelly, drawdown — independent risk sub-systems |
| `scheduler.py` | Tracks per-symbol refresh deadlines; dispatches fetch tasks |
| `history.py` | Append-only prediction log; serves the TUI history panel |
| `tui/app.py` | Textual Application; mounts all panels; handles keyboard events |
| `tui/state.py` | Reactive data store that panels observe; updated by the orchestrator |
| `tui/panels/*.py` | Individual TUI widget implementations |
| `tui/overlays/*.py` | Modal overlay widgets |
| `backtest/backtester.py` | Event-driven backtesting; unchanged from v1 |
| `training/*.py` | Offline training scripts; not loaded at inference time |

---

## 8. Run Requirements

### 8.1 Single Command Start

The system starts with one command from the project root:

```
py -m project BTCUSDT
```

Alternative entry points:

```bash
# Start TUI with a specific symbol
py -m project ETHUSDT

# Start with multiple symbols in watchlist
py -m project BTCUSDT ETHUSDT SOLUSDT

# Start with a specific initial model filter
py -m project BTCUSDT --model intraday

# Headless mode (no TUI, prints JSON to stdout, exits after one prediction)
py -m project BTCUSDT --headless --json

# Run backtester on existing signal history
py -m project backtest --input signals/signals_*.json

# Train models (offline, separate from inference)
py -m project train --data data/ --output models/
```

No configuration file editing is required for the default use case. All defaults are valid for live inference against MEXC using BTCUSDT.

### 8.2 Environment Setup

**py version:** 3.11 or 3.12. py 3.13 is supported but has not been benchmarked.

**Recommended setup procedure:**

```
Step 1: Create a virtual environment
        py -m venv .venv

Step 2: Activate the environment
        Windows:  .venv\Scripts\activate
        macOS/Linux: source .venv/bin/activate

Step 3: Install all dependencies
        pip install -r requirements.txt

Step 4: Verify models are present
        py -m project --check-models

Step 5: Run
        py -m project BTCUSDT
```

**`requirements.txt` — production dependencies:**

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `onnxruntime` | 1.16.0 | ONNX model inference (CPU) |
| `onnx` | 1.14.0 | ONNX metadata reading |
| `numpy` | 1.24.0 | Numerical feature computation |
| `pandas` | 2.0.0 | OHLCV DataFrame handling |
| `scikit-learn` | 1.3.0 | StandardScaler for normalisation |
| `aiohttp` | 3.9.0 | Async HTTP client for MEXC API |
| `textual` | 0.47.0 | TUI framework |
| `rich` | 13.7.0 | Rich text formatting (used by Textual) |
| `pyarrow` | 14.0.0 | Parquet disk cache |

**Training-only dependencies** (not required for inference or TUI):

| Package | Purpose |
|---------|---------|
| `lightgbm >= 4.0.0` | Model training |
| `optuna >= 3.3.0` | Hyperparameter optimisation |
| `onnxmltools >= 1.11.0` | LightGBM → ONNX export |
| `skl2onnx >= 1.15.0` | Sklearn pipeline → ONNX export |

### 8.3 Environment Variables

No environment variables are required for read-only MEXC access. If Binance data collection is needed for retraining:

| Variable | Purpose | Default |
|----------|---------|---------|
| `BINANCE_API_KEY` | Binance Futures read access | empty (not used in inference) |
| `BINANCE_API_SECRET` | Binance Futures authentication | empty (not used in inference) |

These must be injected via a `.env` file (not committed to version control) or as shell environment variables. The application reads them using `os.environ.get()` with empty-string defaults; it never requires them to be set for the inference-only workflow.

### 8.4 Models Directory Resolution

The application resolves the models directory in the following priority order:

1. `--models-dir` CLI argument (absolute or relative path).
2. `MODELS_DIR` environment variable.
3. `models_dir` key in `config.json` adjacent to `app.py`.
4. Fallback: `project/models/` relative to `__file__`.

On startup, the application prints the resolved models directory and the count of successfully loaded models. If zero models are loaded, the application exits with a descriptive error rather than proceeding in a degraded state.

### 8.5 Health Check

Running `py -m project --check-models` performs a pre-flight check:

- Verifies each expected ONNX file exists.
- Verifies each sidecar JSON file exists and feature count matches model input shape.
- Runs a single inference pass with random input on each model.
- Reports latency per model.
- Prints a summary: `OK (4/4 models)` or a list of failures.

This command exits with code `0` on success, `1` on partial failure, `2` on total failure.

---

*End of SPEC.md*
