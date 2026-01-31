#!/usr/bin/env python
"""
Advanced Training Script с поддержкой Optuna

Использование:
    python train.py --trials 100          # Optuna оптимизация (100 trials)
    python train.py --fast                # Быстрое обучение без Optuna
    python train.py --gpu --trials 50     # С GPU (если доступно)
    python train.py --model scalp         # Обучить только scalp модель
    python train.py --continue-study      # Продолжить предыдущую оптимизацию
    python train.py --best                # Обучить с лучшими найденными параметрами
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, root_mean_squared_error

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    LGBM_MARKET_PARAMS, LGBM_RISK_PARAMS, MODEL_PATHS, RANDOM_STATE,
    PRICE_MOVE_THRESHOLD, PREDICTION_HORIZON,
    TRAIN_TEST_SPLIT, VALIDATION_SPLIT
)
# Используем новый data_loader_v2 для полных данных
try:
    from data.data_loader_v2 import prepare_training_data_v2 as prepare_training_data, BinanceDataLoader
    DATA_LOADER_V2 = True
except ImportError:
    from data.data_loader import prepare_training_data
    DATA_LOADER_V2 = False
    
from features.scalp_features import build_scalp_features
from features.intraday_features import build_intraday_features
from features.swing_features import build_swing_features
from training.train_risk import build_risk_features

# Try importing Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Run: pip install optuna")

# Try importing ONNX
try:
    import onnx
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxmltools not installed. ONNX export disabled. Run: pip install onnxmltools onnx")


# ============== CONSTANTS ==============
STUDY_DIR = "optuna_studies"
BEST_PARAMS_FILE = "best_params.json"


class OptunaTrainer:
    """
    Advanced trainer with Optuna hyperparameter optimization
    """
    
    def __init__(
        self,
        model_type: str = "all",
        use_gpu: bool = False,
        n_trials: int = 100,
        n_jobs: int = 1,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        continue_study: bool = False
    ):
        """
        Args:
            model_type: 'scalp', 'intraday', 'swing', 'risk', или 'all'
            use_gpu: использовать GPU
            n_trials: количество Optuna trials
            n_jobs: количество параллельных trials
            timeout: таймаут в секундах
            study_name: имя study для сохранения
            continue_study: продолжить предыдущую study
        """
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.study_name = study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.continue_study = continue_study
        
        # Device settings
        self.device = "gpu" if use_gpu else "cpu"
        
        # Data storage
        self.data = {}
        self.best_params = {}
        
        # Create directories
        os.makedirs(STUDY_DIR, exist_ok=True)
        os.makedirs("models", exist_ok=True)
    
    def load_data(self, data_path: Optional[str] = None) -> Dict:
        """Load and prepare all training data"""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        if data_path is None:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(os.path.dirname(project_dir), 'data')
        
        print(f"Data path: {data_path}")
        print(f"Using data_loader_v2: {DATA_LOADER_V2}")
        
        if DATA_LOADER_V2:
            # Новый загрузчик с полными данными (funding, OI, LS ratio и т.д.)
            raw_data = prepare_training_data(
                data_dir=data_path,
                timeframes=['5m', '15m', '1h', '4h', '1d'],
                include_derivatives=True  # Включаем funding, OI, LS ratio
            )
        else:
            # Старый загрузчик (только OHLCV из trades)
            raw_data = prepare_training_data(
                data_dir=data_path,
                timeframes=['5m', '15m', '1h', '4h', '1d', '3d']
            )
        
        if not raw_data:
            raise ValueError("No data loaded!")
        
        print(f"Loaded data for {len(raw_data)} symbols")
        
        # Process data for each model type
        self._prepare_scalp_data(raw_data)
        self._prepare_intraday_data(raw_data)
        self._prepare_swing_data(raw_data)
        self._prepare_risk_data(raw_data)
        
        return self.data
    
    def _prepare_scalp_data(self, raw_data: Dict):
        """Prepare scalp model training data"""
        print("\nPreparing SCALP data...")
        
        all_features = []
        all_targets = []
        feature_names = None
        expected_features = None
        
        for symbol, tf_data in raw_data.items():
            if '5m' not in tf_data:
                continue
            
            try:
                df = tf_data['5m'].copy()
                df, f_names = build_scalp_features(
                    df,
                    horizon=PREDICTION_HORIZON['scalp'],
                    threshold=PRICE_MOVE_THRESHOLD['scalp'] / 100
                )
                
                if len(df) > 100 and 'target' in df.columns:
                    if feature_names is None:
                        feature_names = f_names
                        expected_features = len(f_names)
                    
                    if len(f_names) != expected_features:
                        print(f"  Skipping {symbol}: {len(f_names)} features (expected {expected_features})")
                        continue
                    
                    X = df[feature_names].values
                    y = df['target'].values
                    valid_mask = ~np.isnan(y)
                    all_features.append(X[valid_mask].astype(np.float32))
                    all_targets.append(y[valid_mask].astype(np.int32))
                    
            except Exception as e:
                print(f"  Error processing {symbol}: {e}")
        
        if all_features:
            X = np.vstack(all_features)
            y = np.concatenate(all_targets)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.data['scalp'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names
            }
            print(f"  SCALP: {X.shape[0]:,} samples, {X.shape[1]} features")
        else:
            print("  No scalp data available")
    
    def _prepare_intraday_data(self, raw_data: Dict):
        """Prepare intraday model training data"""
        print("\nPreparing INTRADAY data...")
        
        all_features = []
        all_targets = []
        feature_names = None
        expected_features = None
        
        for symbol, tf_data in raw_data.items():
            if '1h' not in tf_data:
                continue
            
            try:
                df = tf_data['1h'].copy()
                df, f_names = build_intraday_features(
                    df,
                    horizon=PREDICTION_HORIZON['intraday'],
                    threshold=PRICE_MOVE_THRESHOLD['intraday'] / 100
                )
                
                if len(df) > 50 and 'target' in df.columns:
                    if feature_names is None:
                        feature_names = f_names
                        expected_features = len(f_names)
                    
                    if len(f_names) != expected_features:
                        print(f"  Skipping {symbol}: {len(f_names)} features (expected {expected_features})")
                        continue
                    
                    X = df[feature_names].values
                    y = df['target'].values
                    valid_mask = ~np.isnan(y)
                    all_features.append(X[valid_mask].astype(np.float32))
                    all_targets.append(y[valid_mask].astype(np.int32))
                    
            except Exception as e:
                print(f"  Error processing {symbol}: {e}")
        
        if all_features:
            X = np.vstack(all_features)
            y = np.concatenate(all_targets)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.data['intraday'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names
            }
            print(f"  INTRADAY: {X.shape[0]:,} samples, {X.shape[1]} features")
        else:
            print("  No intraday data available")
    
    def _prepare_swing_data(self, raw_data: Dict):
        """Prepare swing model training data"""
        print("\nPreparing SWING data...")
        
        all_features = []
        all_targets = []
        feature_names = None
        expected_features = None
        
        for symbol, tf_data in raw_data.items():
            if '1d' not in tf_data:
                continue
            
            try:
                df = tf_data['1d'].copy()
                
                # Adaptive horizon based on data size
                adaptive_horizon = min(PREDICTION_HORIZON['swing'], max(1, len(df) // 10))
                
                df, f_names = build_swing_features(
                    df,
                    horizon=adaptive_horizon,
                    threshold=PRICE_MOVE_THRESHOLD['swing'] / 100
                )
                
                if len(df) > 30 and 'target' in df.columns:
                    if feature_names is None:
                        feature_names = f_names
                        expected_features = len(f_names)
                    
                    if len(f_names) != expected_features:
                        print(f"  Skipping {symbol}: {len(f_names)} features (expected {expected_features})")
                        continue
                    
                    X = df[feature_names].values
                    y = df['target'].values
                    valid_mask = ~np.isnan(y)
                    all_features.append(X[valid_mask].astype(np.float32))
                    all_targets.append(y[valid_mask].astype(np.int32))
                    
            except Exception as e:
                print(f"  Error processing {symbol}: {e}")
        
        if all_features:
            X = np.vstack(all_features)
            y = np.concatenate(all_targets)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.data['swing'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names
            }
            print(f"  SWING: {X.shape[0]:,} samples, {X.shape[1]} features")
        else:
            print("  No swing data available")
    
    def _prepare_risk_data(self, raw_data: Dict):
        """Prepare risk model training data"""
        print("\nPreparing RISK data...")
        
        all_features = []
        all_targets = []
        feature_names = None
        expected_features = None
        
        for symbol, tf_data in raw_data.items():
            if '5m' not in tf_data:
                continue
            
            try:
                df = tf_data['5m'].copy()
                df, f_names = build_risk_features(df)
                
                if len(df) > 100:
                    if feature_names is None:
                        feature_names = f_names
                        expected_features = len(f_names)
                    
                    if len(f_names) != expected_features:
                        print(f"  Skipping {symbol}: {len(f_names)} features (expected {expected_features})")
                        continue
                    
                    X = df[feature_names].values
                    df['future_vol'] = df['close'].pct_change().rolling(20).std().shift(-20)
                    y = df['future_vol'].values
                    valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
                    all_features.append(X[valid_mask].astype(np.float32))
                    all_targets.append(y[valid_mask].astype(np.float32))
                    
            except Exception as e:
                print(f"  Error processing {symbol}: {e}")
        
        if all_features:
            X = np.vstack(all_features)
            y = np.concatenate(all_targets)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.data['risk'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names
            }
            print(f"  RISK: {X.shape[0]:,} samples, {X.shape[1]} features")
        else:
            print("  No risk data available")
    
    def _create_objective(self, model_type: str) -> callable:
        """Create Optuna objective function for given model type"""
        
        is_classification = model_type != 'risk'
        data = self.data[model_type]
        X, y = data['X'], data['y']
        feature_names = data['feature_names']
        
        def objective(trial: optuna.Trial) -> float:
            # Hyperparameter search space
            params = {
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'num_leaves': trial.suggest_int('num_leaves', 16, 256),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'verbose': -1,
                'n_jobs': -1,
                'random_state': RANDOM_STATE,
            }
            
            if is_classification:
                params['objective'] = 'multiclass'
                params['num_class'] = 3
                params['metric'] = 'multi_logloss'
            else:
                params['objective'] = 'regression'
                params['metric'] = 'rmse'
            
            if self.use_gpu:
                params['device'] = 'gpu'
                params['gpu_platform_id'] = 0
                params['gpu_device_id'] = 0
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    valid_names=['valid'],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(period=0)  # Suppress output
                    ]
                )
                
                if is_classification:
                    y_pred = model.predict(X_val)
                    # Handle case when not all classes are present in validation set
                    try:
                        score = log_loss(y_val, y_pred, labels=[0, 1, 2])
                    except ValueError:
                        score = 1.0  # Worst score as fallback
                else:
                    y_pred = model.predict(X_val)
                    score = root_mean_squared_error(y_val, y_pred)
                
                scores.append(score)
                
                # Pruning
                trial.report(np.mean(scores), fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        return objective
    
    def optimize_with_optuna(self, model_type: str) -> Dict:
        """Run Optuna optimization for a model"""
        
        if not OPTUNA_AVAILABLE:
            print("Optuna not available! Running fast training instead.")
            return self.train_fast(model_type)
        
        print(f"\n{'='*60}")
        print(f"OPTUNA OPTIMIZATION: {model_type.upper()}")
        print(f"{'='*60}")
        
        if model_type not in self.data:
            print(f"No data available for {model_type}")
            return {}
        
        # Study storage
        study_path = os.path.join(STUDY_DIR, f"{model_type}_{self.study_name}.db")
        storage = f"sqlite:///{study_path}"
        
        # Create or load study
        sampler = TPESampler(seed=RANDOM_STATE)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        
        if self.continue_study and os.path.exists(study_path):
            print(f"Continuing study from: {study_path}")
            study = optuna.load_study(
                study_name=f"{model_type}_study",
                storage=storage,
                sampler=sampler,
                pruner=pruner
            )
        else:
            study = optuna.create_study(
                study_name=f"{model_type}_study",
                storage=storage,
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=self.continue_study
            )
        
        # Run optimization
        objective = self._create_objective(model_type)
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            show_progress_bar=True,
            gc_after_trial=True
        )
        
        print(f"\nBest trial:")
        print(f"  Value: {study.best_trial.value:.6f}")
        print(f"  Params: {study.best_trial.params}")
        
        # Save best params
        self.best_params[model_type] = study.best_trial.params
        self._save_best_params()
        
        # Train final model with best params
        return self._train_with_params(model_type, study.best_trial.params)
    
    def train_fast(self, model_type: str) -> Dict:
        """Fast training without Optuna (default params)"""
        
        print(f"\n{'='*60}")
        print(f"FAST TRAINING: {model_type.upper()}")
        print(f"{'='*60}")
        
        if model_type not in self.data:
            print(f"No data available for {model_type}")
            return {}
        
        is_classification = model_type != 'risk'
        
        # Default parameters
        if is_classification:
            params = LGBM_MARKET_PARAMS.copy()
        else:
            params = LGBM_RISK_PARAMS.copy()
        
        if self.use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
        
        return self._train_with_params(model_type, params)
    
    def train_with_best(self, model_type: str) -> Dict:
        """Train with previously found best parameters"""
        
        params = self._load_best_params(model_type)
        
        if not params:
            print(f"No saved best params for {model_type}. Running optimization...")
            return self.optimize_with_optuna(model_type)
        
        print(f"\n{'='*60}")
        print(f"TRAINING WITH BEST PARAMS: {model_type.upper()}")
        print(f"{'='*60}")
        print(f"Loaded params: {params}")
        
        return self._train_with_params(model_type, params)
    
    def _train_with_params(self, model_type: str, params: Dict) -> Dict:
        """Train model with given parameters"""
        
        data = self.data[model_type]
        X, y = data['X'], data['y']
        feature_names = data['feature_names']
        
        is_classification = model_type != 'risk'
        
        # Build full params
        full_params = params.copy()
        full_params['verbose'] = -1
        full_params['n_jobs'] = -1
        full_params['random_state'] = RANDOM_STATE
        
        if is_classification:
            full_params['objective'] = 'multiclass'
            full_params['num_class'] = 3
            full_params['metric'] = 'multi_logloss'
        else:
            full_params['objective'] = 'regression'
            full_params['metric'] = 'rmse'
        
        # Extract n_estimators
        n_estimators = full_params.pop('n_estimators', 500)
        
        # Time series CV for metrics
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        cv_acc = []
        best_iterations = []
        
        print(f"\nCross-validation training...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                full_params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(100)
                ]
            )
            
            if is_classification:
                y_pred_proba = model.predict(X_val)
                y_pred = np.argmax(y_pred_proba, axis=1)
                # Handle case when not all classes are present in validation set
                try:
                    score = log_loss(y_val, y_pred_proba, labels=[0, 1, 2])
                except ValueError:
                    # Fallback: use accuracy as score (inverted)
                    score = 1 - accuracy_score(y_val, y_pred)
                acc = accuracy_score(y_val, y_pred)
                cv_acc.append(acc)
            else:
                y_pred = model.predict(X_val)
                score = root_mean_squared_error(y_val, y_pred)
            
            cv_scores.append(score)
            best_iterations.append(model.best_iteration)
            
            print(f"  Fold {fold+1}: score={score:.4f}" + 
                  (f", acc={acc:.4f}" if is_classification else ""))
        
        # Train final model on all data
        print(f"\nTraining final model on all data...")
        
        final_n_estimators = int(np.mean(best_iterations))
        
        full_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        final_model = lgb.train(
            full_params,
            full_data,
            num_boost_round=final_n_estimators
        )
        
        # Metrics
        metrics = {
            'model_type': model_type,
            'cv_score': float(np.mean(cv_scores)),
            'cv_score_std': float(np.std(cv_scores)),
            'best_iteration': final_n_estimators,
            'n_features': len(feature_names),
            'n_samples': len(X),
            'params': params,
            'trained_at': datetime.now().isoformat()
        }
        
        if is_classification:
            metrics['cv_accuracy'] = float(np.mean(cv_acc))
            metrics['cv_accuracy_std'] = float(np.std(cv_acc))
        
        # Save model
        self._save_model(model_type, final_model, feature_names, metrics)
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE: {model_type.upper()}")
        print(f"{'='*60}")
        print(f"  CV Score: {metrics['cv_score']:.4f} ± {metrics['cv_score_std']:.4f}")
        if is_classification:
            print(f"  CV Accuracy: {metrics['cv_accuracy']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
        print(f"  Best Iteration: {final_n_estimators}")
        
        return metrics
    
    def _save_model(self, model_type: str, model: lgb.Booster, feature_names: List, metrics: Dict):
        """Save trained model in multiple formats"""
        
        model_path = MODEL_PATHS.get(model_type, f"models/{model_type}_lgbm.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save as .txt (LightGBM native)
        txt_path = model_path.replace('.pkl', '.txt')
        model.save_model(txt_path)
        print(f"  Model saved: {txt_path}")
        
        # Save with pickle
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_names': feature_names,
                'metrics': metrics
            }, f)
        print(f"  Model saved: {model_path}")
        
        # Save feature names separately
        feature_path = model_path.replace('.pkl', '_features.json')
        with open(feature_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        # Save as ONNX
        if ONNX_AVAILABLE:
            try:
                onnx_path = model_path.replace('.pkl', '.onnx')
                self._save_onnx(model, feature_names, onnx_path, model_type)
                print(f"  Model saved: {onnx_path}")
            except Exception as e:
                print(f"  Warning: ONNX export failed: {e}")
    
    def _save_onnx(self, model: lgb.Booster, feature_names: List, onnx_path: str, model_type: str):
        """Convert and save model to ONNX format"""
        
        n_features = len(feature_names)
        is_classification = model_type != 'risk'
        
        # Define input type
        initial_type = [('input', FloatTensorType([None, n_features]))]
        
        # Convert to ONNX
        if is_classification:
            # For classification, we need to specify zipmap=False for raw probabilities
            onnx_model = convert_lightgbm(
                model,
                initial_types=initial_type,
                target_opset=15,
                zipmap=False
            )
        else:
            # For regression
            onnx_model = convert_lightgbm(
                model,
                initial_types=initial_type,
                target_opset=15
            )
        
        # Add metadata
        onnx_model.doc_string = f"LightGBM {model_type} model converted to ONNX"
        
        # Add feature names as metadata
        meta = onnx_model.metadata_props.add()
        meta.key = "feature_names"
        meta.value = json.dumps(feature_names)
        
        meta = onnx_model.metadata_props.add()
        meta.key = "model_type"
        meta.value = model_type
        
        meta = onnx_model.metadata_props.add()
        meta.key = "created_at"
        meta.value = datetime.now().isoformat()
        
        # Save ONNX model
        onnx.save(onnx_model, onnx_path)
    
    def _save_best_params(self):
        """Save best parameters to file"""
        params_path = os.path.join(STUDY_DIR, BEST_PARAMS_FILE)
        
        # Load existing params
        all_params = {}
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                all_params = json.load(f)
        
        # Update with new params
        all_params.update(self.best_params)
        
        with open(params_path, 'w') as f:
            json.dump(all_params, f, indent=2)
        
        print(f"  Best params saved: {params_path}")
    
    def _load_best_params(self, model_type: str) -> Optional[Dict]:
        """Load best parameters from file"""
        params_path = os.path.join(STUDY_DIR, BEST_PARAMS_FILE)
        
        if not os.path.exists(params_path):
            return None
        
        with open(params_path, 'r') as f:
            all_params = json.load(f)
        
        return all_params.get(model_type)
    
    def train_all(self, mode: str = 'optuna') -> Dict:
        """Train all models"""
        
        results = {}
        
        model_types = ['scalp', 'intraday', 'swing', 'risk']
        
        if self.model_type != 'all':
            model_types = [self.model_type]
        
        for model_type in model_types:
            if model_type not in self.data:
                print(f"\nSkipping {model_type} - no data available")
                continue
            
            if mode == 'optuna':
                results[model_type] = self.optimize_with_optuna(model_type)
            elif mode == 'fast':
                results[model_type] = self.train_fast(model_type)
            elif mode == 'best':
                results[model_type] = self.train_with_best(model_type)
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Advanced ML Training with Optuna',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
    python train.py --trials 100              # Optuna оптимизация (100 trials)
    python train.py --fast                    # Быстрое обучение без Optuna
    python train.py --gpu --trials 50         # С GPU
    python train.py --model scalp --trials 200  # Только scalp, 200 trials
    python train.py --continue-study          # Продолжить оптимизацию
    python train.py --best                    # Обучить с лучшими параметрами
    python train.py --trials 100 --timeout 3600  # 1 час максимум
        """
    )
    
    # Mode selection
    parser.add_argument('--fast', action='store_true',
                        help='Fast training without Optuna')
    parser.add_argument('--best', action='store_true',
                        help='Train with best saved parameters')
    
    # Optuna settings
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of Optuna trials (default: 100)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds for optimization')
    parser.add_argument('--continue-study', action='store_true',
                        help='Continue previous Optuna study')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Name for Optuna study')
    
    # Training settings
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'scalp', 'intraday', 'swing', 'risk'],
                        help='Which model to train (default: all)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Parallel jobs for Optuna (default: 1)')
    
    # Data settings
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("AI CRYPTO TRAINER")
    print("="*60)
    print(f"Time: {datetime.now()}")
    print(f"Model: {args.model}")
    print(f"GPU: {args.gpu}")
    
    if args.fast:
        print("Mode: FAST (no Optuna)")
    elif args.best:
        print("Mode: BEST (saved params)")
    else:
        print(f"Mode: OPTUNA ({args.trials} trials)")
        if args.timeout:
            print(f"Timeout: {args.timeout}s")
    
    # Initialize trainer
    trainer = OptunaTrainer(
        model_type=args.model,
        use_gpu=args.gpu,
        n_trials=args.trials,
        n_jobs=args.jobs,
        timeout=args.timeout,
        study_name=args.study_name,
        continue_study=args.continue_study
    )
    
    # Load data
    trainer.load_data(args.data_path)
    
    # Determine training mode
    if args.fast:
        mode = 'fast'
    elif args.best:
        mode = 'best'
    else:
        mode = 'optuna'
    
    # Train
    results = trainer.train_all(mode=mode)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for model_type, metrics in results.items():
        if metrics:
            print(f"\n{model_type.upper()}:")
            print(f"  CV Score: {metrics.get('cv_score', 'N/A'):.4f}")
            if 'cv_accuracy' in metrics:
                print(f"  CV Accuracy: {metrics['cv_accuracy']:.4f}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
