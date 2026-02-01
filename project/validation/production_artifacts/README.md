# Crypto Trading System - Production Artifacts

Generated: 2026-02-01 23:55:45

## Contents

```
production_artifacts/
├── onnx_models/          # ONNX model files
├── scalers/              # Feature scalers
├── encoders/             # Label encoders
├── configs/              # Configuration files
├── optuna_studies/       # Hyperparameter studies
├── inference_pipeline.py # Main inference code
├── kill_switch_handler.py
├── kill_switch_config.json
├── deployment_manifest.json
└── seed_config.json
```

## Quick Start

```python
from inference_pipeline import CryptoInferencePipeline

pipeline = CryptoInferencePipeline('./production_artifacts')
result = pipeline.predict({'scalp': features})
```

## Kill Switch

Activate:
```bash
python kill_switch_handler.py activate "Reason"
```

Deactivate:
```bash
python kill_switch_handler.py deactivate
```

## Important Notes

1. All models trained with seed: 42
2. Kill switch auto-triggers on >20% drawdown
3. Monitor logs for risk model warnings
