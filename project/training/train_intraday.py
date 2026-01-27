"""
Train Intraday Model - обучение модели для 1h/4h/12h таймфреймов

Target: классификация направления
    0 = price_down
    1 = flat  
    2 = price_up

Возвращает:
    P_up, P_down, expected_return
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, log_loss
from typing import Tuple, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LGBM_MARKET_PARAMS, MODEL_PATHS, RANDOM_STATE,
    PRICE_MOVE_THRESHOLD, PREDICTION_HORIZON,
    TRAIN_TEST_SPLIT, VALIDATION_SPLIT
)
from features.intraday_features import build_intraday_features


def load_and_prepare_data(
    data_path: str,
    funding_path: str = None,
    oi_path: str = None,
    btc_path: str = None
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Загрузка и подготовка данных для intraday модели
    
    Args:
        data_path: путь к основным OHLCV данным
        funding_path: путь к funding rate данным
        oi_path: путь к Open Interest данным
        btc_path: путь к BTC данным для корреляции
        
    Returns:
        X: features
        y: targets
        feature_names: список названий фичей
    """
    print(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Load additional data if available
    funding_rate = None
    if funding_path and os.path.exists(funding_path):
        funding_df = pd.read_csv(funding_path, index_col=0, parse_dates=True)
        if 'fundingRate' in funding_df.columns:
            funding_rate = funding_df['fundingRate']
    
    oi_data = None
    if oi_path and os.path.exists(oi_path):
        oi_df = pd.read_csv(oi_path, index_col=0, parse_dates=True)
        if 'sumOpenInterest' in oi_df.columns:
            oi_data = oi_df['sumOpenInterest']
    
    btc_returns = None
    if btc_path and os.path.exists(btc_path):
        btc_df = pd.read_csv(btc_path, index_col=0, parse_dates=True)
        if 'close' in btc_df.columns:
            btc_returns = btc_df['close'].pct_change()
    
    # Build features
    df, feature_names = build_intraday_features(
        df,
        horizon=PREDICTION_HORIZON['intraday'],
        threshold=PRICE_MOVE_THRESHOLD['intraday'] / 100,
        funding_rate=funding_rate,
        oi_data=oi_data,
        btc_returns=btc_returns,
        normalize=True
    )
    
    X = df[feature_names].values.astype(np.float32)
    y = df['target'].values.astype(np.int32)
    
    # Clean NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {np.bincount(y)}")
    
    return X, y, feature_names


def train_intraday_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_splits: int = 5,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50
) -> Tuple[lgb.Booster, Dict]:
    """
    Обучение Intraday модели с TimeSeriesSplit валидацией
    """
    print("\n" + "="*50)
    print("Training Intraday Model (LightGBM)")
    print("="*50)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    params = LGBM_MARKET_PARAMS.copy()
    params['num_boost_round'] = num_boost_round
    
    all_metrics = []
    best_model = None
    best_score = 0
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(100)
            ]
        )
        
        y_pred_proba = model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        acc = accuracy_score(y_val, y_pred)
        logloss = log_loss(y_val, y_pred_proba)
        
        print(f"Fold {fold + 1} - Accuracy: {acc:.4f}, LogLoss: {logloss:.4f}")
        
        all_metrics.append({
            'fold': fold + 1,
            'accuracy': acc,
            'logloss': logloss,
            'best_iteration': model.best_iteration
        })
        
        if acc > best_score:
            best_score = acc
            best_model = model
    
    avg_acc = np.mean([m['accuracy'] for m in all_metrics])
    avg_logloss = np.mean([m['logloss'] for m in all_metrics])
    
    print("\n" + "="*50)
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average LogLoss: {avg_logloss:.4f}")
    print("="*50)
    
    # Retrain on full data
    print("\nRetraining on full data...")
    
    full_train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
    final_model = lgb.train(
        params,
        full_train_data,
        num_boost_round=best_model.best_iteration
    )
    
    metrics = {
        'model_type': 'intraday',
        'avg_accuracy': avg_acc,
        'avg_logloss': avg_logloss,
        'folds': all_metrics,
        'best_iteration': best_model.best_iteration,
        'n_features': len(feature_names),
        'n_samples': len(X),
        'trained_at': datetime.now().isoformat()
    }
    
    return final_model, metrics


def save_model(model: lgb.Booster, feature_names: list, metrics: Dict, model_path: str = None):
    """Сохранение модели"""
    model_path = model_path or MODEL_PATHS['intraday']
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model.save_model(model_path.replace('.pkl', '.txt'))
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'metrics': metrics
        }, f)
    
    print(f"Model saved to {model_path}")
    
    feature_path = model_path.replace('.pkl', '_features.json')
    with open(feature_path, 'w') as f:
        json.dump(feature_names, f, indent=2)


def load_model(model_path: str = None) -> Tuple[lgb.Booster, list, Dict]:
    """Загрузка модели"""
    model_path = model_path or MODEL_PATHS['intraday']
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['model'], data['feature_names'], data['metrics']


def predict(model: lgb.Booster, X: np.ndarray) -> Dict:
    """
    Предсказание модели
    
    Returns:
        Dict с P_up, P_down, P_flat, expected_return
    """
    proba = model.predict(X)
    
    return {
        'P_down': proba[:, 0],
        'P_flat': proba[:, 1],
        'P_up': proba[:, 2],
        'expected_return': proba[:, 2] - proba[:, 0]
    }


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list = None,
    n_splits: int = 5,
    n_iterations: int = 500
) -> Tuple[lgb.Booster, Dict]:
    """
    Wrapper для обучения модели напрямую из данных
    """
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    return train_intraday_model(
        X, y, feature_names,
        n_splits=n_splits,
        num_boost_round=n_iterations
    )


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Intraday Model')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--funding', type=str, default=None, help='Path to funding rate CSV')
    parser.add_argument('--oi', type=str, default=None, help='Path to Open Interest CSV')
    parser.add_argument('--btc', type=str, default=None, help='Path to BTC data CSV')
    parser.add_argument('--output', type=str, default=None, help='Output model path')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--num_boost_round', type=int, default=500, help='Max boosting rounds')
    
    args = parser.parse_args()
    
    # Load data
    X, y, feature_names = load_and_prepare_data(
        args.data, args.funding, args.oi, args.btc
    )
    
    # Train
    model, metrics = train_intraday_model(
        X, y, feature_names,
        n_splits=args.n_splits,
        num_boost_round=args.num_boost_round
    )
    
    # Save
    save_model(model, feature_names, metrics, args.output)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
