"""
Train Risk Model - отдельный AI для оценки риска сделки

Задача: регрессия risk_score

Target:
- stop_loss_hit (0/1)
- max_adverse_excursion
- volatility_spike

Вход:
- confidence из market models
- volatility
- liquidity
- market regime

Выход:
- risk_score
- recommended_leverage
- risk_per_trade
- SL_distance
- TP_distance
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LGBM_RISK_PARAMS, MODEL_PATHS, RANDOM_STATE,
    LEVERAGE_MAP, MAX_RISK_PER_TRADE,
    HIGH_VOLATILITY_THRESHOLD
)


def calculate_risk_targets(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """
    Рассчитать targets для Risk модели
    
    Args:
        df: DataFrame с OHLCV и market model predictions
        horizon: горизонт в барах
        
    Returns:
        DataFrame с risk targets
    """
    # Max Adverse Excursion (MAE) - максимальное движение против позиции
    future_lows = df['low'].rolling(window=horizon).min().shift(-horizon)
    future_highs = df['high'].rolling(window=horizon).max().shift(-horizon)
    
    # MAE для long позиции
    df['mae_long'] = (df['close'] - future_lows) / df['close']
    
    # MAE для short позиции  
    df['mae_short'] = (future_highs - df['close']) / df['close']
    
    # Combined MAE (worst case)
    df['max_adverse_excursion'] = df[['mae_long', 'mae_short']].max(axis=1)
    
    # Stop Loss Hit (при SL = 2%)
    SL_THRESHOLD = 0.02
    df['sl_hit_long'] = (df['mae_long'] > SL_THRESHOLD).astype(int)
    df['sl_hit_short'] = (df['mae_short'] > SL_THRESHOLD).astype(int)
    df['stop_loss_hit'] = df[['sl_hit_long', 'sl_hit_short']].max(axis=1)
    
    # Volatility spike
    returns = df['close'].pct_change()
    rolling_vol = returns.rolling(20).std()
    future_vol = returns.shift(-horizon).rolling(horizon).std()
    df['volatility_spike'] = (future_vol / (rolling_vol + 1e-10) > 1.5).astype(int)
    
    # Combined risk score (0-1)
    # Чем выше MAE и чем чаще SL hit, тем выше риск
    df['risk_score'] = (
        0.4 * df['max_adverse_excursion'].clip(0, 0.1) / 0.1 +  # MAE component
        0.3 * df['stop_loss_hit'] +                               # SL hit component
        0.3 * df['volatility_spike']                              # Vol spike component
    ).clip(0, 1)
    
    return df


def build_risk_features(df: pd.DataFrame, market_predictions: Dict = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Построить фичи для Risk модели
    
    Args:
        df: DataFrame с OHLCV данными
        market_predictions: Dict с предсказаниями market models (P_up, P_down, confidence)
    
    Returns:
        df: DataFrame с фичами
        feature_names: список фичей
    """
    print("Building Risk features...")
    
    # === Volatility features ===
    returns = df['close'].pct_change()
    
    df['volatility_5'] = returns.rolling(5).std()
    df['volatility_10'] = returns.rolling(10).std()
    df['volatility_20'] = returns.rolling(20).std()
    df['volatility_50'] = returns.rolling(50).std()
    
    # Volatility ratio
    df['vol_ratio_5_20'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
    df['vol_ratio_10_50'] = df['volatility_10'] / (df['volatility_50'] + 1e-10)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_50'] = tr.rolling(50).mean()
    df['atr_norm'] = df['atr_14'] / df['close']
    df['atr_ratio'] = df['atr_14'] / (df['atr_50'] + 1e-10)
    
    # === Liquidity features ===
    if 'volume' in df.columns:
        vol_sma_20 = df['volume'].rolling(20).mean()
        vol_sma_50 = df['volume'].rolling(50).mean()
        
        df['volume_ratio'] = df['volume'] / (vol_sma_20 + 1e-10)
        df['volume_trend'] = vol_sma_20 / (vol_sma_50 + 1e-10) - 1
        df['low_liquidity'] = (df['volume_ratio'] < 0.5).astype(int)
    
    # === Market Regime features ===
    # Trend
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    
    df['trend_strength'] = (sma_20 - sma_50) / sma_50
    df['price_vs_sma20'] = (df['close'] - sma_20) / sma_20
    df['price_vs_sma50'] = (df['close'] - sma_50) / sma_50
    
    # Regime detection
    return_20 = df['close'].pct_change(20)
    df['regime_bull'] = (return_20 > 0.05).astype(int)
    df['regime_bear'] = (return_20 < -0.05).astype(int)
    df['regime_flat'] = ((return_20 >= -0.05) & (return_20 <= 0.05)).astype(int)
    
    # === Drawdown features ===
    rolling_max = df['close'].rolling(50).max()
    df['drawdown'] = (df['close'] - rolling_max) / rolling_max
    df['drawdown_depth'] = abs(df['drawdown'])
    
    # === Price action risk ===
    df['bar_range'] = (df['high'] - df['low']) / df['close']
    df['gap'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_risk'] = (df['gap'] > 0.01).astype(int)
    
    # === RSI extremes (risk indicator) ===
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    df['rsi_14'] = (rsi - 50) / 50
    df['rsi_extreme_high'] = (rsi > 80).astype(int)
    df['rsi_extreme_low'] = (rsi < 20).astype(int)
    
    # === Market model confidence (если доступно) ===
    if market_predictions is not None:
        if 'confidence' in market_predictions:
            df['model_confidence'] = market_predictions['confidence']
        if 'P_up' in market_predictions:
            df['model_P_up'] = market_predictions['P_up']
        if 'P_down' in market_predictions:
            df['model_P_down'] = market_predictions['P_down']
        # Uncertainty = 1 - max(P_up, P_down)
        if 'P_up' in market_predictions and 'P_down' in market_predictions:
            df['model_uncertainty'] = 1 - np.maximum(
                market_predictions['P_up'],
                market_predictions['P_down']
            )
    else:
        # Placeholder
        df['model_confidence'] = 0.5
        df['model_P_up'] = 0.33
        df['model_P_down'] = 0.33
        df['model_uncertainty'] = 0.5
    
    # === Correlation risk ===
    df['return_autocorr'] = returns.rolling(20).apply(
        lambda x: x.autocorr() if len(x) > 1 else 0
    )
    
    # === Derivatives Risk Features ===
    # Funding Rate Risk
    if 'funding_rate' in df.columns:
        fr = df['funding_rate'].fillna(0)
        
        # Extreme funding = higher liquidation risk
        fr_std = fr.rolling(100).std()
        fr_mean = fr.rolling(100).mean()
        df['funding_zscore'] = (fr - fr_mean) / (fr_std + 1e-10)
        df['funding_extreme'] = (abs(df['funding_zscore']) > 2).astype(int)
        
        # Funding direction risk (going against funding)
        df['funding_risk'] = abs(fr) * 100  # Scale to percentage
        
        # Cumulative funding pressure
        df['funding_cumsum_risk'] = abs(fr.rolling(24).sum())  # ~1 day
    else:
        df['funding_zscore'] = 0
        df['funding_extreme'] = 0
        df['funding_risk'] = 0
        df['funding_cumsum_risk'] = 0
    
    # Open Interest Risk
    if 'sum_open_interest' in df.columns:
        oi = df['sum_open_interest'].fillna(method='ffill')
        
        # OI spike = potential squeeze risk
        oi_change = oi.pct_change()
        df['oi_change_risk'] = abs(oi_change).clip(0, 0.5)
        
        # High OI = more liquidation cascade risk
        oi_std = oi.rolling(100).std()
        oi_mean = oi.rolling(100).mean()
        df['oi_zscore'] = (oi - oi_mean) / (oi_std + 1e-10)
        df['oi_extreme_high'] = (df['oi_zscore'] > 2).astype(int)
        
        # OI dropping fast = liquidations happening
        df['oi_dropping_fast'] = (oi_change < -0.05).astype(int)
    else:
        df['oi_change_risk'] = 0
        df['oi_zscore'] = 0
        df['oi_extreme_high'] = 0
        df['oi_dropping_fast'] = 0
    
    # Long/Short Ratio Risk
    if 'long_short_ratio' in df.columns:
        ls = df['long_short_ratio'].fillna(1)
        
        # Crowded trade risk
        ls_std = ls.rolling(100).std()
        ls_mean = ls.rolling(100).mean()
        df['ls_zscore'] = (ls - ls_mean) / (ls_std + 1e-10)
        
        # Extreme positioning = squeeze risk
        df['crowd_long_risk'] = (ls > 2.5).astype(int)  # >70% long
        df['crowd_short_risk'] = (ls < 0.67).astype(int)  # >60% short
        df['crowded_trade'] = df['crowd_long_risk'] | df['crowd_short_risk']
    else:
        df['ls_zscore'] = 0
        df['crowd_long_risk'] = 0
        df['crowd_short_risk'] = 0
        df['crowded_trade'] = 0
    
    # Taker Volume Risk
    if 'buy_sell_ratio' in df.columns:
        bsr = df['buy_sell_ratio'].fillna(1)
        
        # Aggressive selling/buying = momentum risk
        df['taker_imbalance'] = abs(bsr - 1)
        df['aggressive_selling'] = (bsr < 0.8).astype(int)
        df['aggressive_buying'] = (bsr > 1.2).astype(int)
    else:
        df['taker_imbalance'] = 0
        df['aggressive_selling'] = 0
        df['aggressive_buying'] = 0
    
    # Premium/Basis Risk
    if 'lastFundingRate' in df.columns:
        premium = df['lastFundingRate'].fillna(0)
        
        # High premium = potential correction risk
        df['basis_risk'] = abs(premium) * 1000  # Scale
        df['high_premium'] = (abs(premium) > 0.001).astype(int)
    else:
        df['basis_risk'] = 0
        df['high_premium'] = 0
    
    # Combined Derivatives Risk Score
    df['derivatives_risk'] = (
        0.2 * df['funding_extreme'] +
        0.2 * df['oi_extreme_high'] +
        0.2 * df['crowded_trade'] +
        0.2 * df['taker_imbalance'].clip(0, 1) +
        0.2 * df['high_premium']
    ).clip(0, 1)
    
    # Feature list
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
        'buy_ratio', 'mae_long', 'mae_short', 'sl_hit_long', 'sl_hit_short',
        'max_adverse_excursion', 'stop_loss_hit', 'volatility_spike', 'risk_score',
        'open_time', 'close_time', 'timestamp', 'datetime', 'date', 'time',
        'symbol', 'buy_volume', 'sell_volume', 'trades_count',
        # Raw derivatives columns
        'funding_rate', 'funding_time', 'mark_price',
        'sum_open_interest', 'sum_open_interest_value',
        'long_short_ratio', 'long_account', 'short_account',
        'buy_sell_ratio', 'buy_vol', 'sell_vol',
        'lastFundingRate', 'interestRate', 'indexPrice', 'estimatedSettlePrice'
    ]
    
    feature_names = [col for col in df.columns if col not in exclude_cols
                     and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    # ВАЖНО: Заменяем inf и NaN ПЕРЕД дальнейшей обработкой
    for col in feature_names:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].ffill().bfill().fillna(0)
    
    print(f"Risk features: {len(feature_names)}")
    
    return df, feature_names
    
    return df, feature_names


def train_risk_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_splits: int = 5,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50
) -> Tuple[lgb.Booster, Dict]:
    """
    Обучение Risk модели (регрессия)
    """
    print("\n" + "="*50)
    print("Training Risk Model (LightGBM Regression)")
    print("="*50)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    params = LGBM_RISK_PARAMS.copy()
    
    all_metrics = []
    best_model = None
    best_score = float('inf')
    
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
        
        y_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"Fold {fold + 1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        all_metrics.append({
            'fold': fold + 1,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'best_iteration': model.best_iteration
        })
        
        if rmse < best_score:
            best_score = rmse
            best_model = model
    
    avg_rmse = np.mean([m['rmse'] for m in all_metrics])
    avg_mae = np.mean([m['mae'] for m in all_metrics])
    avg_r2 = np.mean([m['r2'] for m in all_metrics])
    
    print("\n" + "="*50)
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average R2: {avg_r2:.4f}")
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
        'model_type': 'risk',
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'folds': all_metrics,
        'best_iteration': best_model.best_iteration,
        'n_features': len(feature_names),
        'n_samples': len(X),
        'trained_at': datetime.now().isoformat()
    }
    
    return final_model, metrics


def save_model(model: lgb.Booster, feature_names: list, metrics: Dict, model_path: str = None):
    """Сохранение модели"""
    model_path = model_path or MODEL_PATHS['risk']
    
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
    model_path = model_path or MODEL_PATHS['risk']
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['model'], data['feature_names'], data['metrics']


def predict_risk(
    model: lgb.Booster,
    X: np.ndarray,
    current_price: float,
    volatility: float
) -> Dict:
    """
    Предсказание риска и расчёт параметров
    
    Returns:
        Dict с risk_score, recommended_leverage, risk_per_trade, SL_distance, TP_distance
    """
    risk_score = model.predict(X).clip(0, 1)
    
    # Recommended leverage based on risk
    leverage = np.where(
        risk_score < 0.3, LEVERAGE_MAP['low_risk'],
        np.where(risk_score < 0.5, LEVERAGE_MAP['medium_risk'], LEVERAGE_MAP['high_risk'])
    )
    
    # Risk per trade (снижаем при высоком risk_score)
    risk_per_trade = MAX_RISK_PER_TRADE * (1 - risk_score * 0.5)
    
    # SL distance based on ATR/volatility and risk
    # Базовый SL = 1.5 * ATR, корректируем на риск
    base_sl_multiplier = 1.5
    sl_multiplier = base_sl_multiplier * (1 + risk_score)  # Больше риск = шире SL
    sl_distance = volatility * sl_multiplier
    
    # TP distance (risk:reward ratio)
    # При низком риске R:R = 1:2, при высоком = 1:1.5
    rr_ratio = np.where(risk_score < 0.3, 2.5, np.where(risk_score < 0.5, 2.0, 1.5))
    tp_distance = sl_distance * rr_ratio
    
    return {
        'risk_score': risk_score,
        'recommended_leverage': leverage,
        'risk_per_trade': risk_per_trade,
        'SL_distance': sl_distance,
        'SL_price_long': current_price * (1 - sl_distance),
        'SL_price_short': current_price * (1 + sl_distance),
        'TP_distance': tp_distance,
        'TP_price_long': current_price * (1 + tp_distance),
        'TP_price_short': current_price * (1 - tp_distance)
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
    
    return train_risk_model(
        X, y, feature_names,
        n_splits=n_splits,
        num_boost_round=n_iterations
    )


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Risk Model')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--output', type=str, default=None, help='Output model path')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--num_boost_round', type=int, default=500, help='Max boosting rounds')
    parser.add_argument('--horizon', type=int, default=12, help='Prediction horizon in bars')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    
    # Calculate risk targets
    df = calculate_risk_targets(df, horizon=args.horizon)
    
    # Build features
    df, feature_names = build_risk_features(df)
    
    X = df[feature_names].values.astype(np.float32)
    y = df['risk_score'].values.astype(np.float32)
    
    # Clean
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train
    model, metrics = train_risk_model(
        X, y, feature_names,
        n_splits=args.n_splits,
        num_boost_round=args.num_boost_round
    )
    
    # Save
    save_model(model, feature_names, metrics, args.output)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
