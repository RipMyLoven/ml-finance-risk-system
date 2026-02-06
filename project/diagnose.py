#!/usr/bin/env python3
"""Quick diagnostic test for train_with_risk imports."""

import sys

def log(msg):
    with open('diag.log', 'a') as f:
        f.write(f"{msg}\n")
    print(msg, flush=True)

# Clear log
open('diag.log', 'w').close()

try:
    log("1. Basic imports...")
    import os
    import yaml
    import platform
    from pathlib import Path
    
    log("2. Multiprocessing...")
    import multiprocessing as mp
    
    log("3. NumPy...")
    import numpy as np
    
    log("4. Polars...")
    import polars as pl
    
    log("5. SciPy...")
    from scipy import stats
    
    log("6. Sklearn...")
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import roc_auc_score
    
    log("7. LightGBM...")
    import lightgbm as lgb
    
    log("8. Optuna...")
    try:
        import optuna
        log("   Optuna OK")
    except ImportError:
        log("   Optuna not available")
    
    log("9. psutil...")
    import psutil
    
    log("10. All imports successful!")
    log("")
    log("11. Now importing train_with_risk module...")
    
    import train_with_risk
    
    log("12. Module imported OK!")
    log("")
    log("13. Testing main() call...")
    log(f"Training mode: {train_with_risk.TRAINING_MODE}")
    
    log("")
    log("=== DIAGNOSTICS COMPLETE ===")
    
except Exception as e:
    log(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    log(traceback.format_exc())
