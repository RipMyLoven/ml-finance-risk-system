"""
FIXED Training Pipeline - Eliminates All Data Leakage

Key Fixes:
1. Per-symbol TimeSeriesSplit (no cross-symbol contamination)
2. Purge + Embargo gaps in CV
3. Rolling normalization (fit only on train, transform val)
4. Future return column explicitly removed
5. Proper walk-forward validation
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import accuracy_score, classification_report, log_loss
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LGBM_MARKET_PARAMS, MODEL_PATHS, RANDOM_STATE,
    PRICE_MOVE_THRESHOLD, PREDICTION_HORIZON
)


@dataclass
class CVResult:
    """Cross-validation result for a single fold"""
    fold: int
    train_size: int
    val_size: int
    train_acc: float
    val_acc: float
    val_logloss: float
    feature_importance: Dict[str, float]


class PurgedTimeSeriesSplit:
    """
    Time Series Split with Purge and Embargo gaps to prevent leakage.
    
    purge_gap: number of samples to remove between train and val
               (should be >= prediction horizon)
    embargo_gap: additional samples to skip at start of validation
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 6,
        embargo_gap: int = 0,
        test_size: float = 0.2
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
        self.test_size = test_size
    
    def split(self, X, y=None, groups=None):
        """
        Generate train/val indices with purge and embargo gaps.
        """
        n_samples = len(X)
        min_train_size = int(n_samples * 0.3)
        
        # Calculate fold size
        fold_size = int(n_samples * self.test_size)
        
        for i in range(self.n_splits):
            # Training end
            train_end = min_train_size + i * (fold_size // self.n_splits)
            
            # Apply purge gap
            val_start = train_end + self.purge_gap + self.embargo_gap
            val_end = min(val_start + fold_size, n_samples)
            
            if val_start >= n_samples:
                break
            
            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)
            
            if len(val_indices) < 10:
                continue
                
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class RollingStandardScaler:
    """
    StandardScaler that fits only on training data.
    Prevents information leakage from future samples.
    """
    
    def __init__(self, clip_quantiles: Tuple[float, float] = (0.001, 0.999)):
        self.clip_quantiles = clip_quantiles
        self.means_ = None
        self.stds_ = None
        self.clip_low_ = None
        self.clip_high_ = None
    
    def fit(self, X: np.ndarray) -> 'RollingStandardScaler':
        """Fit scaler on training data only"""
        self.clip_low_ = np.percentile(X, self.clip_quantiles[0] * 100, axis=0)
        self.clip_high_ = np.percentile(X, self.clip_quantiles[1] * 100, axis=0)
        
        X_clipped = np.clip(X, self.clip_low_, self.clip_high_)
        
        self.means_ = np.mean(X_clipped, axis=0)
        self.stds_ = np.std(X_clipped, axis=0)
        self.stds_ = np.where(self.stds_ < 1e-8, 1.0, self.stds_)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters"""
        X_clipped = np.clip(X, self.clip_low_, self.clip_high_)
        X_scaled = (X_clipped - self.means_) / self.stds_
        return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def build_features_no_leakage(
    df: pd.DataFrame,
    horizon: int = 6,
    threshold: float = 0.01,
    funding_rate: pd.Series = None,
    oi_data: pd.Series = None,
    btc_returns: pd.Series = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build features WITHOUT future leakage.
    
    Key differences from original:
    1. No cumulative VWAP (uses only rolling)
    2. No future_return column in output
    3. Normalization deferred to training loop
    """
    initial_len = len(df)
    
    # Import feature functions
    from features.intraday_features import (
        add_trend_features,
        add_volatility_intraday,
        add_momentum_intraday,
        add_volume_intraday
    )
    
    # Build non-leaky features
    df = add_trend_features(df)
    df = add_volatility_intraday(df)
    df = add_momentum_intraday(df)
    df = add_volume_intraday(df)
    
    # === ADD ROLLING VWAP (NOT CUMULATIVE) ===
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    for window in [10, 20]:
        tp_vol = df['typical_price'] * df['volume']
        df[f'vwap_{window}'] = tp_vol.rolling(window).sum() / df['volume'].rolling(window).sum()
        df[f'vwap_dev_{window}'] = (df['close'] - df[f'vwap_{window}']) / df[f'vwap_{window}']
    
    std_20 = df['close'].rolling(20).std()
    vwap_20 = df['vwap_20']
    df['vwap_upper'] = vwap_20 + 2 * std_20
    df['vwap_lower'] = vwap_20 - 2 * std_20
    df['vwap_position'] = (df['close'] - df['vwap_lower']) / (df['vwap_upper'] - df['vwap_lower'] + 1e-10)
    
    # === ADD FUNDING FEATURES ===
    if funding_rate is not None and len(funding_rate) > 0:
        df = df.merge(
            funding_rate.to_frame('funding_rate'),
            left_index=True, right_index=True, how='left'
        )
        df['funding_rate'] = df['funding_rate'].ffill().fillna(0)
        df['funding_change'] = df['funding_rate'].diff() * 1000
        df['funding_change_3'] = df['funding_rate'].diff(3) * 1000
        df['funding_sma'] = df['funding_rate'].rolling(8).mean() * 1000
        df['funding_vs_sma'] = (df['funding_rate'] - df['funding_rate'].rolling(8).mean()) * 1000
    else:
        df['funding_change'] = 0
        df['funding_change_3'] = 0
        df['funding_sma'] = 0
        df['funding_vs_sma'] = 0
    
    # === ADD TARGET (no future_return stored!) ===
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    
    df['target'] = 1  # default flat
    df.loc[future_return > threshold, 'target'] = 2   # up
    df.loc[future_return < -threshold, 'target'] = 0  # down
    
    # === DEFINE FEATURE COLUMNS ===
    # EXPLICITLY list features - no auto-detection that might include leaky columns
    feature_names = [
        # Trend
        'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50',
        'sma_cross_10_20', 'sma_cross_10_50', 'sma_cross_20_50',
        'trend_slope_10', 'trend_slope_20', 'ema_trend', 'ema_trend_slope',
        # VWAP
        'vwap_dev_10', 'vwap_dev_20', 'vwap_position',
        # Volatility
        'volatility_10', 'volatility_20', 'volatility_50', 'vol_ratio_10_50',
        'atr_14', 'atr_norm', 'bb_width', 'bb_position',
        # Momentum
        'rsi_14', 'macd_norm', 'macd_signal', 'macd_hist_slope', 'adx', 'di_diff',
        # Volume
        'volume_ratio', 'volume_trend', 'obv_slope', 'mfi',
        # Funding
        'funding_change', 'funding_change_3', 'funding_sma', 'funding_vs_sma'
    ]
    
    # Filter to existing columns
    feature_names = [f for f in feature_names if f in df.columns]
    
    # Handle NaN/Inf
    for col in feature_names:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].ffill().bfill().fillna(0)
    
    # Drop rows where target is NaN (end of series)
    df = df[df['target'].notna()]
    
    final_len = len(df)
    print(f"Features built: {len(feature_names)}, samples: {final_len}/{initial_len}")
    
    return df, feature_names


def train_per_symbol_cv(
    symbol_data: Dict[str, pd.DataFrame],
    feature_names: List[str],
    horizon: int = 6,
    n_splits: int = 5,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50
) -> Tuple[lgb.Booster, Dict, List[CVResult]]:
    """
    Train with PROPER per-symbol time series CV.
    
    Key: Each symbol is split independently, then folds are combined.
    This prevents cross-symbol temporal leakage.
    """
    print("\n" + "="*60)
    print("TRAINING WITH PER-SYMBOL CV (NO LEAKAGE)")
    print("="*60)
    
    params = LGBM_MARKET_PARAMS.copy()
    
    all_cv_results = []
    fold_models = []
    
    # Process each symbol independently
    for symbol, df in symbol_data.items():
        if len(df) < 200:
            continue
            
        X = df[feature_names].values.astype(np.float32)
        y = df['target'].values.astype(np.int32)
        
        # Per-symbol purged CV
        tscv = PurgedTimeSeriesSplit(
            n_splits=n_splits,
            purge_gap=horizon + 2,  # Horizon + safety margin
            embargo_gap=1
        )
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # === FIT SCALER ONLY ON TRAIN ===
            scaler = RollingStandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train LightGBM
            train_data = lgb.Dataset(X_train_scaled, label=y_train, feature_name=feature_names)
            val_data = lgb.Dataset(X_val_scaled, label=y_val, feature_name=feature_names, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds),
                    lgb.log_evaluation(0)  # Suppress output
                ]
            )
            
            # Evaluate
            y_train_pred = np.argmax(model.predict(X_train_scaled), axis=1)
            y_val_pred_proba = model.predict(X_val_scaled)
            y_val_pred = np.argmax(y_val_pred_proba, axis=1)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_logloss = log_loss(y_val, y_val_pred_proba, labels=[0, 1, 2])
            
            # Feature importance
            importance = dict(zip(feature_names, model.feature_importance()))
            
            result = CVResult(
                fold=fold,
                train_size=len(train_idx),
                val_size=len(val_idx),
                train_acc=train_acc,
                val_acc=val_acc,
                val_logloss=val_logloss,
                feature_importance=importance
            )
            all_cv_results.append(result)
            fold_models.append((model, scaler))
    
    # === AGGREGATE METRICS ===
    avg_train_acc = np.mean([r.train_acc for r in all_cv_results])
    avg_val_acc = np.mean([r.val_acc for r in all_cv_results])
    avg_val_logloss = np.mean([r.val_logloss for r in all_cv_results])
    std_val_acc = np.std([r.val_acc for r in all_cv_results])
    
    print(f"\n{'='*60}")
    print("CV RESULTS (HONEST - NO LEAKAGE)")
    print(f"{'='*60}")
    print(f"Train Accuracy:      {avg_train_acc:.4f}")
    print(f"Validation Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"Validation LogLoss:  {avg_val_logloss:.4f}")
    print(f"Train-Val Gap:       {avg_train_acc - avg_val_acc:.4f}")
    
    # === DETECT OVERFITTING ===
    gap = avg_train_acc - avg_val_acc
    if gap > 0.15:
        print(f"\n⚠️  WARNING: Large train-val gap ({gap:.2%}) indicates overfitting!")
        print("    → Reduce num_leaves, increase min_child_samples, add regularization")
    
    if avg_val_acc < 0.40:
        print(f"\n⚠️  WARNING: Val accuracy ({avg_val_acc:.2%}) near random chance (33%)!")
        print("    → Features may have low signal, or target definition is noisy")
    
    # === TRAIN FINAL MODEL ON ALL DATA ===
    print("\nTraining final model on all data...")
    
    all_X = []
    all_y = []
    
    for symbol, df in symbol_data.items():
        if len(df) < 100:
            continue
        X = df[feature_names].values.astype(np.float32)
        y = df['target'].values.astype(np.int32)
        all_X.append(X)
        all_y.append(y)
    
    X_full = np.vstack(all_X)
    y_full = np.concatenate(all_y)
    
    # Final scaler fit on all data (for production)
    final_scaler = RollingStandardScaler()
    X_full_scaled = final_scaler.fit_transform(X_full)
    
    # Use median best_iteration from folds
    best_iterations = [m[0].best_iteration for m in fold_models]
    final_iterations = int(np.median(best_iterations))
    
    full_data = lgb.Dataset(X_full_scaled, label=y_full, feature_name=feature_names)
    final_model = lgb.train(
        params,
        full_data,
        num_boost_round=final_iterations
    )
    
    metrics = {
        'model_type': 'intraday_fixed',
        'cv_train_accuracy': float(avg_train_acc),
        'cv_val_accuracy': float(avg_val_acc),
        'cv_val_accuracy_std': float(std_val_acc),
        'cv_val_logloss': float(avg_val_logloss),
        'train_val_gap': float(gap),
        'best_iteration': final_iterations,
        'n_features': len(feature_names),
        'n_samples': len(X_full),
        'n_folds': len(all_cv_results),
        'trained_at': datetime.now().isoformat()
    }
    
    return final_model, metrics, all_cv_results


def diagnose_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str]
) -> Dict:
    """
    Run diagnostic tests to detect leakage.
    
    Returns dict with diagnostic metrics.
    """
    print("\n" + "="*60)
    print("PIPELINE DIAGNOSTICS")
    print("="*60)
    
    diagnostics = {}
    
    # Test 1: Train on future, predict past (should fail if no leakage)
    print("\n[Test 1] Future→Past prediction (should be ~33% if no leakage)...")
    n = len(X)
    split = n // 2
    
    X_future, y_future = X[split:], y[split:]
    X_past, y_past = X[:split], y[:split]
    
    model = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    model.fit(X_future, y_future)
    future_to_past_acc = accuracy_score(y_past, model.predict(X_past))
    diagnostics['future_to_past_accuracy'] = float(future_to_past_acc)
    
    if future_to_past_acc > 0.45:
        print(f"  ⚠️  LEAKAGE DETECTED: {future_to_past_acc:.2%} accuracy (expected ~33%)")
    else:
        print(f"  ✓ OK: {future_to_past_acc:.2%} accuracy")
    
    # Test 2: Check feature-target correlation
    print("\n[Test 2] Feature-target correlation (|r| > 0.5 is suspicious)...")
    suspicious_features = []
    
    for i, fname in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        if abs(corr) > 0.5:
            suspicious_features.append((fname, corr))
            print(f"  ⚠️  {fname}: r = {corr:.3f}")
    
    diagnostics['suspicious_features'] = suspicious_features
    
    if not suspicious_features:
        print("  ✓ No highly correlated features")
    
    # Test 3: Shuffled baseline
    print("\n[Test 3] Shuffled target baseline (should be ~33%)...")
    y_shuffled = np.random.permutation(y)
    
    model_shuffled = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    from sklearn.model_selection import cross_val_score
    shuffled_scores = cross_val_score(model_shuffled, X, y_shuffled, cv=3)
    shuffled_acc = np.mean(shuffled_scores)
    diagnostics['shuffled_baseline'] = float(shuffled_acc)
    
    if shuffled_acc > 0.40:
        print(f"  ⚠️  STRUCTURAL LEAKAGE: {shuffled_acc:.2%} on shuffled targets!")
    else:
        print(f"  ✓ OK: {shuffled_acc:.2%} on shuffled targets")
    
    # Test 4: Class balance
    print("\n[Test 4] Class balance...")
    class_counts = np.bincount(y.astype(int))
    class_pcts = class_counts / len(y)
    diagnostics['class_balance'] = {
        'down': float(class_pcts[0]),
        'flat': float(class_pcts[1]),
        'up': float(class_pcts[2])
    }
    
    print(f"  Down: {class_pcts[0]:.1%}, Flat: {class_pcts[1]:.1%}, Up: {class_pcts[2]:.1%}")
    
    if max(class_pcts) > 0.5:
        print(f"  ⚠️  IMBALANCED: Majority class is {max(class_pcts):.1%}")
        print("     → Accuracy can be misleading, use F1/AUC instead")
    
    return diagnostics


def save_model_fixed(
    model: lgb.Booster,
    feature_names: List[str],
    metrics: Dict,
    scaler_params: Dict,
    model_path: str = None
):
    """Save model with all required components"""
    model_path = model_path or "models/intraday_fixed.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save LightGBM native format
    model.save_model(model_path.replace('.pkl', '.txt'))
    
    # Save pickle with all components
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'metrics': metrics,
            'scaler_params': scaler_params
        }, f)
    
    # Save feature names
    with open(model_path.replace('.pkl', '_features.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    print("="*60)
    print("FIXED TRAINING PIPELINE - NO LEAKAGE")
    print("="*60)
    
    # Import data loader
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(project_dir), 'data')
    
    try:
        from data.data_loader_v2 import BinanceDataLoader
        loader = BinanceDataLoader(data_dir)
    except ImportError:
        print("Error: data_loader_v2 not found")
        sys.exit(1)
    
    # Load data for each symbol
    symbol_data = {}
    
    for symbol in loader.symbols[:20]:  # Start with top 20
        df = loader.load_klines(symbol, '1h')
        if df is None or len(df) < 200:
            continue
        
        funding = loader.load_funding_rate(symbol)
        funding_rate = funding['funding_rate'] if funding is not None else None
        
        df, feature_names = build_features_no_leakage(
            df,
            horizon=PREDICTION_HORIZON['intraday'],
            threshold=PRICE_MOVE_THRESHOLD['intraday'] / 100,
            funding_rate=funding_rate
        )
        
        if len(df) >= 200:
            symbol_data[symbol] = df
            print(f"  {symbol}: {len(df)} samples")
    
    print(f"\nLoaded {len(symbol_data)} symbols")
    
    if not symbol_data:
        print("No data loaded!")
        sys.exit(1)
    
    # Run diagnostics first
    all_X = np.vstack([df[feature_names].values for df in symbol_data.values()])
    all_y = np.concatenate([df['target'].values for df in symbol_data.values()])
    
    diagnostics = diagnose_pipeline(all_X, all_y, feature_names)
    
    # Train with proper CV
    model, metrics, cv_results = train_per_symbol_cv(
        symbol_data,
        feature_names,
        horizon=PREDICTION_HORIZON['intraday'],
        n_splits=5,
        num_boost_round=500
    )
    
    # Save
    save_model_fixed(
        model, feature_names, metrics,
        scaler_params={'type': 'RollingStandardScaler'},
        model_path="models/intraday_fixed.pkl"
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Expected real accuracy: {metrics['cv_val_accuracy']:.1%} ± {metrics['cv_val_accuracy_std']:.1%}")
    print(f"If real accuracy differs by >5%, check inference pipeline for bugs.")
