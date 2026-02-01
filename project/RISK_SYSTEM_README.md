# Advanced Risk-Controlled Trading System

## System Overview

This is a **production-ready multi-model trading system** with institutional-grade risk management. The system consists of:

### Trading Models
- **Scalp Model** (5m/15m timeframes) - High-frequency, small profits
- **Intraday Model** (1h/4h timeframes) - Day trading
- **Swing Model** (1d+ timeframes) - Position trading

### Risk Model (CORE CONTROLLER)
The **Advanced Risk Model** is the central control unit with **FINAL VETO POWER** over all trading decisions.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADE FLOW                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                   │
│  │  Scalp  │   │Intraday │   │  Swing  │                   │
│  │  Model  │   │  Model  │   │  Model  │                   │
│  └────┬────┘   └────┬────┘   └────┬────┘                   │
│       │             │             │                         │
│       └─────────────┼─────────────┘                         │
│                     │                                       │
│                     ▼                                       │
│       ┌─────────────────────────────┐                      │
│       │     ADVANCED RISK MODEL     │                      │
│       │      (CORE CONTROLLER)      │                      │
│       │                             │                      │
│       │  ┌───────────────────────┐ │                      │
│       │  │    CVaR Engine        │ │                      │
│       │  │  - Tail Risk Est.     │ │                      │
│       │  │  - Trade Admission    │ │                      │
│       │  └───────────────────────┘ │                      │
│       │                             │                      │
│       │  ┌───────────────────────┐ │                      │
│       │  │   Kelly Sizer         │ │                      │
│       │  │  - Position Sizing    │ │                      │
│       │  │  - Leverage Control   │ │                      │
│       │  └───────────────────────┘ │                      │
│       │                             │                      │
│       │  ┌───────────────────────┐ │                      │
│       │  │ Drawdown Controller   │ │                      │
│       │  │  - Dynamic Sizing     │ │                      │
│       │  │  - Model Control      │ │                      │
│       │  └───────────────────────┘ │                      │
│       └──────────────┬──────────────┘                      │
│                      │                                      │
│            ┌─────────┴─────────┐                           │
│            │                   │                           │
│            ▼                   ▼                           │
│    ┌───────────────┐  ┌───────────────┐                   │
│    │   APPROVED    │  │   REJECTED    │                   │
│    │  (with size)  │  │  (with reason)│                   │
│    └───────────────┘  └───────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. CVaR Engine (`risk/cvar_engine.py`)

**Conditional Value at Risk** estimation for tail-risk management:

- Historical CVaR calculation
- Rolling window CVaR
- Portfolio-level CVaR with correlation adjustment
- Stress scenario CVaR
- Trade admission filtering

**Justification**: CVaR (Expected Shortfall) captures tail risk better than VaR, essential for crypto markets with fat-tailed distributions.

### 2. Kelly Sizer (`risk/kelly_sizing.py`)

**Fractional Kelly Criterion** for optimal position sizing:

- **Never uses raw Kelly** - always capped and dampened
- Fractional Kelly modes (quarter/half/adaptive)
- Maximum leverage constraints
- Volatility-adjusted scaling
- Confidence-based dampening

**Justification**: Raw Kelly is too aggressive. Fractional Kelly reduces variance by 75% while maintaining most of the expected growth.

### 3. Drawdown Controller (`risk/drawdown_controller.py`)

**Non-linear drawdown response**:

| Drawdown | Position Size | Risk State |
|----------|---------------|------------|
| 0-5%     | 100%          | FULL_RISK  |
| 5-10%    | 75%           | REDUCED    |
| 10-15%   | 50%           | MINIMAL    |
| 15-20%   | 25%           | RISK_OFF   |
| 20%+     | 0%            | EMERGENCY  |

Model-specific disable thresholds:
- Scalp: Disabled at 8% DD
- Intraday: Disabled at 12% DD
- Swing: Disabled at 18% DD

### 4. Advanced Risk Model (`risk/risk_model.py`)

**Central controller** that integrates all components:

- Trade proposal evaluation
- Position sizing decisions
- Trade approval/rejection
- Global risk state management
- Kill-switch control

## Installation

```bash
# Required packages
pip install numpy pandas lightgbm scikit-learn optuna
pip install onnxmltools onnx onnxruntime

# Optional for GPU
pip install lightgbm --install-option=--gpu
pip install onnxruntime-gpu
```

## Usage

### Training

```bash
# Full Optuna optimization
python train_with_risk.py --trials 100

# Risk model only
python train_with_risk.py --risk-only --trials 50

# Fast training (no Optuna)
python train_with_risk.py --fast

# With GPU
python train_with_risk.py --gpu --trials 100
```

### Integration Tests

```bash
python tests/test_risk_integration.py
```

### Using the Risk Model

```python
from risk import AdvancedRiskModel, TradeProposal

# Initialize risk model
risk_model = AdvancedRiskModel(
    initial_equity=10000.0,
    cvar_confidence=0.95,
    kelly_mode=KellyMode.QUARTER,
    max_leverage=5.0
)

# Create trade proposal
proposal = TradeProposal(
    model_type='intraday',
    symbol='BTCUSDT',
    direction='long',
    probability=0.60,
    confidence=0.75,
    expected_return=0.02,
    volatility_regime='normal',
    proposed_size=1000.0,
    stop_loss=0.02,
    take_profit=0.04
)

# Get decision
decision = risk_model.evaluate_trade(proposal)

if decision.decision != TradeDecision.REJECTED:
    # Execute with approved size
    execute_trade(symbol='BTCUSDT', size=decision.approved_size)
else:
    # Trade rejected
    print(f"Rejected: {decision.reason}")
```

### Production Inference

```python
from risk.inference_pipeline import RiskInferencePipeline, InferenceMode

# Create production pipeline
pipeline = RiskInferencePipeline(
    model_path='models/risk_lgbm.onnx',
    mode=InferenceMode.PRODUCTION,
    enable_kill_switch=True
)

# Run inference
result = pipeline.infer(features)

if result.is_blocked:
    print(f"BLOCKED: {result.block_reason}")
else:
    position_size = base_size * result.position_size_factor
```

## Configuration

All parameters are configurable in `config.py`:

### CVaR Configuration
```python
CVAR_CONFIG = {
    "confidence_level": 0.95,
    "default_window": 252,
    "block_threshold": 0.08,
    "model_limits": {
        "scalp": 0.05,
        "intraday": 0.07,
        "swing": 0.10
    }
}
```

### Kelly Configuration
```python
KELLY_CONFIG = {
    "mode": "quarter",
    "max_kelly_fraction": 0.25,
    "max_leverage": 5.0,
    "max_position_pct": 0.10
}
```

### Drawdown Configuration
```python
DRAWDOWN_CONFIG = {
    "level_1": 0.05,
    "level_2": 0.10,
    "level_3": 0.15,
    "level_4": 0.20,
    "emergency": 0.25
}
```

## Optuna Optimization

The Risk Model is trained with **multi-objective optimization**:

### Objectives
1. **RMSE** ↓ - Prediction accuracy
2. **CVaR Penalty** ↓ - Tail risk reduction
3. **Tail Underestimation** ↓ - Avoid underestimating high risk

### Hyperparameters Tuned
- LightGBM parameters (num_leaves, learning_rate, etc.)
- CVaR window size
- Kelly fraction
- Drawdown thresholds
- Volatility scaling

## ONNX Export

Models are exported to ONNX for production inference:

- Sub-millisecond latency
- Deterministic execution
- Cross-platform compatibility
- Validated against LightGBM predictions

## Monitoring

The system provides comprehensive monitoring:

- Real-time risk state
- Trade approval/rejection rates
- Drawdown tracking
- CVaR monitoring
- Latency tracking
- Signal degradation detection
- Regime shift alerts

## Safety Features

### Kill Switch
- Manual activation
- Automatic trigger at emergency drawdown
- All trading halted until manual reset

### Fail-Safe Design
- Conservative estimates when data insufficient
- Block trades on high latency
- Block on low confidence predictions

## File Structure

```
project/
├── risk/
│   ├── __init__.py              # Module exports
│   ├── cvar_engine.py           # CVaR calculations
│   ├── kelly_sizing.py          # Position sizing
│   ├── drawdown_controller.py   # Drawdown management
│   ├── risk_model.py            # Core controller
│   ├── risk_features.py         # Feature builder
│   ├── train_risk_optuna.py     # Optuna optimization
│   └── inference_pipeline.py    # Production inference
├── tests/
│   └── test_risk_integration.py # Integration tests
├── config.py                    # All configurations
├── train_with_risk.py           # Training script
└── RISK_SYSTEM_README.md        # This file
```

## Design Principles

1. **Every decision justified** - No unexplained heuristics
2. **Fail-safe** - Always err on the side of caution
3. **Deterministic** - Same input = same output
4. **Observable** - Full logging and monitoring
5. **Capital preservation** - Risk control takes priority

## Contact

This system is designed for **live trading**. Treat every component as production-critical.

---

*Think like a fund risk officer. This is real money.*
