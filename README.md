# ML Finance Risk System

A Python framework for analyzing and trading cryptocurrency futures using machine learning models, automated signal generation, and risk-aware position sizing.

---

## Features

- Trains LightGBM models for intraday, scalp, and swing trading strategies
- Generates trade signals with configurable thresholds and filters
- Risk management via CVaR estimation, Kelly criterion sizing, and drawdown control
- ONNX export for fast, portable model inference
- Hyperparameter tuning with Optuna

---

## Project Structure

```
project/
├── data/           # Data collection and loading
├── features/       # Feature engineering per strategy
├── models/         # Trained model files (.onnx, .txt)
├── risk/           # Risk engine: CVaR, Kelly sizing, drawdown
├── signals/        # Signal generation
├── training/       # Training scripts per strategy
├── validation/     # Sanity checks and ensemble validation
└── main.py         # Entry point
```

---

## Quick Start

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/your-username/ml-finance-risk-system.git
cd ml-finance-risk-system/project

# Create the data directory and place your .csv files inside it
mkdir data
# cp /your/path/*.csv data/

# Edit config.yaml to match your local paths and settings

# Install dependencies
pip install -r requirements.txt

# Train all models with risk
python train_with_risk.py

# Run TUI
python -m tui

```

---

## Main Dependencies

| Package | Purpose |
|---|---|
| `lightgbm` | Gradient boosting models |
| `onnxruntime` | Fast model inference |
| `optuna` | Hyperparameter optimization |
| `pandas` / `numpy` | Data processing |
| `scikit-learn` | Preprocessing and evaluation |

Install all dependencies via `pip install -r project/requirements.txt`.

---
