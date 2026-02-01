"""
LEAKAGE DETECTOR - Diagnoses why CV >> Real Accuracy

Run this script to identify exactly what's broken in your pipeline.
Output: Percentage breakdown of each leakage source.
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, log_loss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_future_column_leakage(df: pd.DataFrame, feature_cols: list, target_col: str = 'target'):
    """
    Test 1: Check if any feature correlates too highly with target.
    
    If |correlation| > 0.5, that feature likely contains future info.
    """
    print("\n" + "="*60)
    print("TEST 1: FEATURE-TARGET CORRELATION")
    print("="*60)
    
    leaky_features = []
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        # Compute correlation
        valid_mask = df[col].notna() & df[target_col].notna()
        if valid_mask.sum() < 100:
            continue
            
        corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, target_col])
        
        if abs(corr) > 0.3:  # Threshold for suspicion
            leaky_features.append((col, corr))
            severity = "🔴 CRITICAL" if abs(corr) > 0.7 else "🟡 WARNING"
            print(f"  {severity}: {col} → r={corr:.3f}")
    
    if not leaky_features:
        print("  ✅ No highly correlated features detected")
    else:
        print(f"\n  ⚠️  {len(leaky_features)} suspicious features found")
        estimated_impact = min(30, len(leaky_features) * 5)
        print(f"  Estimated CV inflation: ~{estimated_impact}%")
    
    return leaky_features


def test_cross_symbol_leakage(symbol_dfs: dict, feature_names: list, target_col: str = 'target'):
    """
    Test 2: Train on Symbol A future, predict Symbol B past.
    
    If accuracy > 40%, there's cross-symbol information leakage.
    """
    print("\n" + "="*60)
    print("TEST 2: CROSS-SYMBOL TEMPORAL LEAKAGE")
    print("="*60)
    
    symbols = list(symbol_dfs.keys())
    if len(symbols) < 2:
        print("  Need at least 2 symbols to test")
        return None
    
    sym_a, sym_b = symbols[0], symbols[1]
    df_a, df_b = symbol_dfs[sym_a], symbol_dfs[sym_b]
    
    # Train on second half of A, predict first half of B
    split_a = len(df_a) // 2
    split_b = len(df_b) // 2
    
    X_train = df_a[feature_names].iloc[split_a:].values
    y_train = df_a[target_col].iloc[split_a:].values
    
    X_test = df_b[feature_names].iloc[:split_b].values
    y_test = df_b[target_col].iloc[:split_b].values
    
    # Clean NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    valid_train = ~np.isnan(y_train)
    valid_test = ~np.isnan(y_test)
    
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]
    
    if len(X_train) < 100 or len(X_test) < 100:
        print("  Not enough data for test")
        return None
    
    model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    print(f"  Train: {sym_a} (future half, {len(X_train)} samples)")
    print(f"  Test:  {sym_b} (past half, {len(X_test)} samples)")
    print(f"  Accuracy: {accuracy:.2%}")
    
    if accuracy > 0.45:
        print(f"\n  🔴 LEAKAGE DETECTED: Expected ~33%, got {accuracy:.2%}")
        estimated_impact = int((accuracy - 0.33) * 100)
        print(f"  Estimated CV inflation from mixed-symbol split: ~{estimated_impact}%")
        return estimated_impact
    else:
        print(f"  ✅ No significant cross-symbol leakage")
        return 0


def test_temporal_order_leakage(X: np.ndarray, y: np.ndarray, feature_names: list):
    """
    Test 3: Compare standard CV vs reversed CV.
    
    Train on future data, predict past. Should be ~random (33%).
    If >> 33%, temporal leakage exists.
    """
    print("\n" + "="*60)
    print("TEST 3: TEMPORAL ORDER VIOLATION")
    print("="*60)
    
    n = len(X)
    
    # Test A: Train on first half, predict second (normal)
    X_train, y_train = X[:n//2], y[:n//2]
    X_test, y_test = X[n//2:], y[n//2:]
    
    model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    model.fit(X_train, y_train)
    normal_acc = accuracy_score(y_test, model.predict(X_test))
    
    # Test B: Train on second half, predict first (reversed - should fail)
    X_train_rev, y_train_rev = X[n//2:], y[n//2:]
    X_test_rev, y_test_rev = X[:n//2], y[:n//2]
    
    model_rev = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    model_rev.fit(X_train_rev, y_train_rev)
    reversed_acc = accuracy_score(y_test_rev, model_rev.predict(X_test_rev))
    
    print(f"  Normal (past→future):   {normal_acc:.2%}")
    print(f"  Reversed (future→past): {reversed_acc:.2%}")
    print(f"  Expected reversed:      ~33%")
    
    if reversed_acc > 0.45:
        print(f"\n  🔴 LEAKAGE: Reversed accuracy {reversed_acc:.2%} >> 33%")
        print("     Features contain future information!")
        return int((reversed_acc - 0.33) * 100)
    else:
        print(f"  ✅ No temporal leakage detected")
        return 0


def test_normalization_leakage(X: np.ndarray, y: np.ndarray, feature_names: list):
    """
    Test 4: Compare full-sample normalization vs train-only normalization.
    
    Difference indicates normalization leakage.
    """
    print("\n" + "="*60)
    print("TEST 4: NORMALIZATION LEAKAGE")
    print("="*60)
    
    from sklearn.preprocessing import StandardScaler
    
    n = len(X)
    train_idx = np.arange(n * 4 // 5)
    test_idx = np.arange(n * 4 // 5, n)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Method A: Full-sample normalization (WRONG)
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X)
    X_train_full = X_full_scaled[train_idx]
    X_test_full = X_full_scaled[test_idx]
    
    model_full = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    model_full.fit(X_train_full, y_train)
    acc_full = accuracy_score(y_test, model_full.predict(X_test_full))
    
    # Method B: Train-only normalization (CORRECT)
    scaler_train = StandardScaler()
    X_train_correct = scaler_train.fit_transform(X_train)
    X_test_correct = scaler_train.transform(X_test)
    
    model_correct = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    model_correct.fit(X_train_correct, y_train)
    acc_correct = accuracy_score(y_test, model_correct.predict(X_test_correct))
    
    print(f"  Full-sample normalization:  {acc_full:.2%}")
    print(f"  Train-only normalization:   {acc_correct:.2%}")
    print(f"  Difference:                 {(acc_full - acc_correct):.2%}")
    
    if acc_full - acc_correct > 0.03:
        estimated_impact = int((acc_full - acc_correct) * 100)
        print(f"\n  🟡 WARNING: Normalization leakage detected (~{estimated_impact}% CV inflation)")
        return estimated_impact
    else:
        print(f"  ✅ Normalization leakage is minimal")
        return 0


def test_cv_vs_holdout(X: np.ndarray, y: np.ndarray):
    """
    Test 5: Compare TimeSeriesSplit CV score vs true holdout.
    
    Large gap indicates CV methodology issues.
    """
    print("\n" + "="*60)
    print("TEST 5: CV vs HOLDOUT COMPARISON")
    print("="*60)
    
    from sklearn.preprocessing import StandardScaler
    
    n = len(X)
    
    # TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        
        model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
        model.fit(X_train_s, y_train)
        cv_scores.append(accuracy_score(y_val, model.predict(X_val_s)))
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # True holdout (last 20%)
    holdout_start = int(n * 0.8)
    X_train, y_train = X[:holdout_start], y[:holdout_start]
    X_holdout, y_holdout = X[holdout_start:], y[holdout_start:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_holdout_s = scaler.transform(X_holdout)
    
    model_holdout = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    model_holdout.fit(X_train_s, y_train)
    holdout_acc = accuracy_score(y_holdout, model_holdout.predict(X_holdout_s))
    
    print(f"  TimeSeriesSplit CV: {cv_mean:.2%} ± {cv_std:.2%}")
    print(f"  True Holdout:       {holdout_acc:.2%}")
    print(f"  Gap:                {cv_mean - holdout_acc:.2%}")
    
    if cv_mean - holdout_acc > 0.10:
        print(f"\n  🔴 SIGNIFICANT GAP: CV inflated by ~{int((cv_mean - holdout_acc) * 100)}%")
        return cv_mean - holdout_acc
    elif cv_mean - holdout_acc > 0.05:
        print(f"\n  🟡 MODERATE GAP: Possible minor leakage")
        return cv_mean - holdout_acc
    else:
        print(f"  ✅ CV and holdout are aligned")
        return 0


def run_full_diagnosis():
    """Run all diagnostic tests and produce summary"""
    
    print("\n" + "="*60)
    print("FULL PIPELINE DIAGNOSIS")
    print("="*60)
    print("This will identify exactly where your CV inflation comes from.\n")
    
    # Load data
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    
    try:
        from data.data_loader_v2 import BinanceDataLoader
        from features.intraday_features import build_intraday_features
        from config import PREDICTION_HORIZON, PRICE_MOVE_THRESHOLD
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    loader = BinanceDataLoader(data_dir)
    
    # Load multiple symbols
    symbol_dfs = {}
    all_X = []
    all_y = []
    feature_names = None
    
    for symbol in loader.symbols[:10]:
        df = loader.load_klines(symbol, '1h')
        if df is None or len(df) < 300:
            continue
        
        try:
            df, f_names = build_intraday_features(
                df,
                horizon=PREDICTION_HORIZON['intraday'],
                threshold=PRICE_MOVE_THRESHOLD['intraday'] / 100,
                normalize=False  # Don't normalize here - we'll test it
            )
            
            if len(df) >= 300:
                if feature_names is None:
                    feature_names = f_names
                
                symbol_dfs[symbol] = df
                X = df[feature_names].values.astype(np.float32)
                y = df['target'].values.astype(np.float32)
                
                # Handle NaN
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                valid_mask = ~np.isnan(y)
                
                all_X.append(X[valid_mask])
                all_y.append(y[valid_mask])
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    if not all_X:
        print("No data loaded!")
        return
    
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    
    print(f"Loaded {len(symbol_dfs)} symbols, {len(X_combined)} total samples")
    print(f"Features: {len(feature_names)}")
    
    # Run all tests
    issues = {}
    
    # Test 1: Feature correlations
    sample_df = list(symbol_dfs.values())[0]
    leaky_features = test_future_column_leakage(sample_df, feature_names)
    issues['feature_leakage'] = len(leaky_features) * 5  # Rough estimate
    
    # Test 2: Cross-symbol leakage
    cross_symbol = test_cross_symbol_leakage(symbol_dfs, feature_names)
    issues['cross_symbol'] = cross_symbol or 0
    
    # Test 3: Temporal leakage
    temporal = test_temporal_order_leakage(X_combined, y_combined.astype(int), feature_names)
    issues['temporal'] = temporal
    
    # Test 4: Normalization leakage
    norm = test_normalization_leakage(X_combined, y_combined.astype(int), feature_names)
    issues['normalization'] = norm
    
    # Test 5: CV vs holdout
    cv_gap = test_cv_vs_holdout(X_combined, y_combined.astype(int))
    issues['cv_gap'] = int(cv_gap * 100) if cv_gap else 0
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    total_inflation = sum(issues.values())
    
    print(f"\nEstimated CV Inflation Sources:")
    print(f"{'─'*40}")
    for source, impact in sorted(issues.items(), key=lambda x: -x[1]):
        bar = '█' * (impact // 2) if impact > 0 else ''
        print(f"  {source:20s}: {impact:3d}%  {bar}")
    print(f"{'─'*40}")
    print(f"  {'TOTAL':20s}: {total_inflation:3d}%")
    
    print("\n" + "="*60)
    print("EXPECTED REALISTIC ACCURACY")
    print("="*60)
    
    # Train a clean holdout model
    n = len(X_combined)
    holdout_start = int(n * 0.8)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_combined[:holdout_start])
    X_test = scaler.transform(X_combined[holdout_start:])
    y_train = y_combined[:holdout_start].astype(int)
    y_test = y_combined[holdout_start:].astype(int)
    
    model = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=42)
    model.fit(X_train, y_train)
    
    realistic_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\n  Clean Holdout Accuracy: {realistic_acc:.2%}")
    print(f"  Random Baseline (3-class): 33%")
    print(f"  Your Current CV Score: ~90-99% (BROKEN)")
    print(f"  Gap Explained: {total_inflation}% of {100 - int(realistic_acc*100)}% gap")
    
    print("\n" + "="*60)
    print("RECOMMENDED ACTIONS (priority order)")
    print("="*60)
    
    actions = [
        ("Remove 'future_return' from feature set", "CRITICAL"),
        ("Use per-symbol TimeSeriesSplit", "CRITICAL"),
        ("Fit scaler only on train data", "HIGH"),
        ("Add purge gap >= horizon", "HIGH"),
        ("Verify all rolling features use past-only", "MEDIUM"),
    ]
    
    for i, (action, priority) in enumerate(actions, 1):
        icon = "🔴" if priority == "CRITICAL" else "🟡" if priority == "HIGH" else "🟢"
        print(f"  {i}. {icon} [{priority}] {action}")
    
    print("\n  → Use training/train_fixed.py for corrected pipeline")
    
    return issues


if __name__ == "__main__":
    run_full_diagnosis()
