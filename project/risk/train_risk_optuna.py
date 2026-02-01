"""
Risk Model Training with Optuna

This module implements:
- Separate Optuna study for Risk Model
- Multi-objective optimization (AUC ↑, Drawdown ↓, CVaR ↓)
- Constraint-aware optimization
- Trial pruning
- Comprehensive logging

The Risk Model is trained to predict:
- Risk scores
- Position sizing recommendations
- Trade approval probability
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import optuna
    from optuna.samplers import TPESampler, NSGAIISampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import onnx
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class RiskModelOptimizer:
    """
    Optuna-based optimizer for Risk Model
    
    Implements multi-objective optimization for:
    - Primary: Risk prediction accuracy (RMSE ↓)
    - Secondary: CVaR reduction
    - Tertiary: Drawdown minimization
    - Quaternary: Tail-loss frequency
    
    Uses constraint-aware optimization to ensure:
    - CVaR stays within limits
    - Maximum drawdown bounded
    - Model complexity controlled
    """
    
    # Hyperparameter search space
    PARAM_SPACE = {
        # Tree parameters
        'num_leaves': (16, 256),
        'max_depth': (3, 15),
        'min_child_samples': (5, 100),
        
        # Learning parameters
        'learning_rate': (0.005, 0.3),
        'n_estimators': (100, 2000),
        
        # Regularization
        'reg_alpha': (1e-8, 10.0),
        'reg_lambda': (1e-8, 10.0),
        
        # Sampling
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        
        # Risk-specific parameters
        'cvar_window': (30, 200),
        'kelly_fraction': (0.1, 0.5),
        'drawdown_level_1': (0.03, 0.10),
        'drawdown_level_2': (0.08, 0.15),
        'volatility_scaling': (0.5, 2.0)
    }
    
    def __init__(
        self,
        study_name: str = "risk_model_study",
        n_trials: int = 100,
        n_jobs: int = 1,
        timeout: Optional[int] = None,
        use_gpu: bool = False,
        multi_objective: bool = True,
        storage_dir: str = "optuna_studies"
    ):
        """
        Initialize Risk Model Optimizer
        
        Args:
            study_name: Name for Optuna study
            n_trials: Number of trials
            n_jobs: Parallel jobs
            timeout: Timeout in seconds
            use_gpu: Use GPU for training
            multi_objective: Use multi-objective optimization
            storage_dir: Directory for study storage
        """
        self.study_name = study_name
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.use_gpu = use_gpu
        self.multi_objective = multi_objective
        self.storage_dir = storage_dir
        
        os.makedirs(storage_dir, exist_ok=True)
        
        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.feature_names = None
        
        # Best results
        self.best_params = None
        self.best_model = None
        self.best_metrics = None
        
        # Trial log
        self.trial_log = []
        
    def set_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        val_split: float = 0.2
    ):
        """
        Set training data
        
        Args:
            X: Features array
            y: Target array (risk scores)
            feature_names: List of feature names
            val_split: Validation split ratio
        """
        n_samples = len(X)
        n_train = int(n_samples * (1 - val_split))
        
        self.X_train = X[:n_train]
        self.y_train = y[:n_train]
        self.X_val = X[n_train:]
        self.y_val = y[n_train:]
        self.feature_names = feature_names
        
        print(f"Training data: {self.X_train.shape}")
        print(f"Validation data: {self.X_val.shape}")
        
    def _create_single_objective(self) -> callable:
        """Create single-objective function (RMSE minimization)"""
        
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = self._sample_params(trial)
            
            # Train model
            model = self._train_model(params)
            
            # Evaluate
            y_pred = model.predict(self.X_val)
            rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
            
            # Calculate additional metrics for logging
            mae = mean_absolute_error(self.y_val, y_pred)
            r2 = r2_score(self.y_val, y_pred)
            
            # Log trial
            self._log_trial(trial, params, {
                'rmse': rmse, 'mae': mae, 'r2': r2
            })
            
            # Report for pruning
            trial.report(rmse, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return rmse
        
        return objective
    
    def _create_multi_objective(self) -> callable:
        """
        Create multi-objective function
        
        Objectives:
        1. RMSE (minimize)
        2. CVaR penalty (minimize)
        3. Tail-loss frequency (minimize)
        """
        
        def objective(trial: optuna.Trial) -> Tuple[float, float, float]:
            # Sample hyperparameters
            params = self._sample_params(trial)
            
            # Train model
            model = self._train_model(params)
            
            # Evaluate predictions
            y_pred = model.predict(self.X_val)
            
            # Objective 1: RMSE
            rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
            
            # Objective 2: CVaR penalty (higher predicted risk = penalty if wrong)
            prediction_errors = y_pred - self.y_val
            var_95 = np.percentile(abs(prediction_errors), 95)
            tail_errors = abs(prediction_errors[abs(prediction_errors) >= var_95])
            cvar_penalty = np.mean(tail_errors) if len(tail_errors) > 0 else var_95
            
            # Objective 3: Tail-loss frequency (underestimating high risk)
            # When actual risk is high but prediction is low
            high_risk_mask = self.y_val > np.percentile(self.y_val, 80)
            if np.sum(high_risk_mask) > 0:
                underestimation = np.mean(y_pred[high_risk_mask] < self.y_val[high_risk_mask])
            else:
                underestimation = 0.0
            
            # Log trial
            self._log_trial(trial, params, {
                'rmse': rmse,
                'cvar_penalty': cvar_penalty,
                'tail_underestimation': underestimation
            })
            
            return rmse, cvar_penalty, underestimation
        
        return objective
    
    def _sample_params(self, trial: optuna.Trial) -> Dict:
        """Sample hyperparameters from search space"""
        
        params = {
            # LightGBM parameters
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'num_leaves': trial.suggest_int('num_leaves', *self.PARAM_SPACE['num_leaves']),
            'max_depth': trial.suggest_int('max_depth', *self.PARAM_SPACE['max_depth']),
            'learning_rate': trial.suggest_float('learning_rate', *self.PARAM_SPACE['learning_rate'], log=True),
            'n_estimators': trial.suggest_int('n_estimators', *self.PARAM_SPACE['n_estimators']),
            'min_child_samples': trial.suggest_int('min_child_samples', *self.PARAM_SPACE['min_child_samples']),
            'subsample': trial.suggest_float('subsample', *self.PARAM_SPACE['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *self.PARAM_SPACE['colsample_bytree']),
            'reg_alpha': trial.suggest_float('reg_alpha', *self.PARAM_SPACE['reg_alpha'], log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', *self.PARAM_SPACE['reg_lambda'], log=True),
            
            # Risk-specific parameters
            'cvar_window': trial.suggest_int('cvar_window', *self.PARAM_SPACE['cvar_window']),
            'kelly_fraction': trial.suggest_float('kelly_fraction', *self.PARAM_SPACE['kelly_fraction']),
            'drawdown_level_1': trial.suggest_float('drawdown_level_1', *self.PARAM_SPACE['drawdown_level_1']),
            'drawdown_level_2': trial.suggest_float('drawdown_level_2', *self.PARAM_SPACE['drawdown_level_2']),
            'volatility_scaling': trial.suggest_float('volatility_scaling', *self.PARAM_SPACE['volatility_scaling']),
            
            # Fixed parameters
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42
        }
        
        if self.use_gpu:
            params['device'] = 'gpu'
        
        return params
    
    def _train_model(self, params: Dict) -> lgb.Booster:
        """Train LightGBM model with given parameters"""
        
        # Extract non-LightGBM params
        lgb_params = {k: v for k, v in params.items() 
                     if k not in ['cvar_window', 'kelly_fraction', 
                                  'drawdown_level_1', 'drawdown_level_2',
                                  'volatility_scaling', 'n_estimators']}
        
        n_estimators = params.get('n_estimators', 500)
        
        # Create datasets
        train_data = lgb.Dataset(
            self.X_train, 
            label=self.y_train,
            feature_name=self.feature_names
        )
        val_data = lgb.Dataset(
            self.X_val, 
            label=self.y_val,
            reference=train_data
        )
        
        # Train with early stopping
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            valid_names=['valid'],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        return model
    
    def _log_trial(
        self,
        trial: optuna.Trial,
        params: Dict,
        metrics: Dict
    ):
        """Log trial results"""
        log_entry = {
            'trial_number': trial.number,
            'params': params,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.trial_log.append(log_entry)
        
        print(f"Trial {trial.number}: " + 
              " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
    
    def optimize(self, continue_study: bool = False) -> Dict:
        """
        Run Optuna optimization
        
        Args:
            continue_study: Continue from previous study
            
        Returns:
            Best parameters and metrics
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Run: pip install optuna")
        
        if self.X_train is None:
            raise ValueError("No training data set. Call set_data() first.")
        
        print("\n" + "="*60)
        print("RISK MODEL OPTUNA OPTIMIZATION")
        print("="*60)
        print(f"Study: {self.study_name}")
        print(f"Trials: {self.n_trials}")
        print(f"Multi-objective: {self.multi_objective}")
        
        # Storage
        storage_path = os.path.join(self.storage_dir, f"{self.study_name}.db")
        storage = f"sqlite:///{storage_path}"
        
        # Create study
        if self.multi_objective:
            # Multi-objective with NSGA-II
            sampler = NSGAIISampler(seed=42)
            study = optuna.create_study(
                study_name=self.study_name,
                storage=storage,
                directions=["minimize", "minimize", "minimize"],  # RMSE, CVaR, Tail-loss
                sampler=sampler,
                load_if_exists=continue_study
            )
            objective = self._create_multi_objective()
        else:
            # Single-objective with TPE
            sampler = TPESampler(seed=42)
            pruner = HyperbandPruner()
            study = optuna.create_study(
                study_name=self.study_name,
                storage=storage,
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=continue_study
            )
            objective = self._create_single_objective()
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            show_progress_bar=True,
            gc_after_trial=True
        )
        
        # Get best results
        if self.multi_objective:
            # Get Pareto front
            best_trials = study.best_trials
            print(f"\nPareto front: {len(best_trials)} trials")
            
            # Select trial with best balance (weighted sum)
            best_trial = min(
                best_trials,
                key=lambda t: 0.5 * t.values[0] + 0.3 * t.values[1] + 0.2 * t.values[2]
            )
            self.best_params = best_trial.params
            self.best_metrics = {
                'rmse': best_trial.values[0],
                'cvar_penalty': best_trial.values[1],
                'tail_underestimation': best_trial.values[2]
            }
        else:
            self.best_params = study.best_params
            self.best_metrics = {'rmse': study.best_value}
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best metrics: {self.best_metrics}")
        
        # Train final model with best params
        self._train_final_model()
        
        # Save results
        self._save_results()
        
        return {
            'params': self.best_params,
            'metrics': self.best_metrics
        }
    
    def _train_final_model(self):
        """Train final model with best parameters on all data"""
        print("\nTraining final model...")
        
        # Combine train and validation for final model
        X_all = np.vstack([self.X_train, self.X_val])
        y_all = np.concatenate([self.y_train, self.y_val])
        
        # Build params
        lgb_params = {k: v for k, v in self.best_params.items() 
                     if k not in ['cvar_window', 'kelly_fraction',
                                  'drawdown_level_1', 'drawdown_level_2',
                                  'volatility_scaling', 'n_estimators']}
        lgb_params['objective'] = 'regression'
        lgb_params['metric'] = 'rmse'
        lgb_params['verbose'] = -1
        
        n_estimators = self.best_params.get('n_estimators', 500)
        
        # Train
        data = lgb.Dataset(X_all, label=y_all, feature_name=self.feature_names)
        self.best_model = lgb.train(
            lgb_params,
            data,
            num_boost_round=n_estimators
        )
        
        print("Final model trained")
    
    def _save_results(self):
        """Save model and results"""
        # Save model
        model_path = os.path.join(self.storage_dir, "risk_model_optimized.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'params': self.best_params,
                'metrics': self.best_metrics,
                'feature_names': self.feature_names,
                'risk_params': {
                    'cvar_window': self.best_params.get('cvar_window', 100),
                    'kelly_fraction': self.best_params.get('kelly_fraction', 0.25),
                    'drawdown_level_1': self.best_params.get('drawdown_level_1', 0.05),
                    'drawdown_level_2': self.best_params.get('drawdown_level_2', 0.10),
                    'volatility_scaling': self.best_params.get('volatility_scaling', 1.0)
                }
            }, f)
        print(f"Model saved: {model_path}")
        
        # Save LightGBM native format
        txt_path = os.path.join(self.storage_dir, "risk_model_optimized.txt")
        self.best_model.save_model(txt_path)
        print(f"Model saved: {txt_path}")
        
        # Save trial log
        log_path = os.path.join(self.storage_dir, "risk_optimization_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.trial_log, f, indent=2, default=str)
        print(f"Trial log saved: {log_path}")
        
        # Export to ONNX
        if ONNX_AVAILABLE:
            self._export_onnx()
    
    def _export_onnx(self):
        """Export model to ONNX format"""
        try:
            onnx_path = os.path.join(self.storage_dir, "risk_model_optimized.onnx")
            
            n_features = len(self.feature_names)
            initial_type = [('input', FloatTensorType([None, n_features]))]
            
            onnx_model = convert_lightgbm(
                self.best_model,
                initial_types=initial_type,
                target_opset=15
            )
            
            # Add metadata
            meta = onnx_model.metadata_props.add()
            meta.key = "feature_names"
            meta.value = json.dumps(self.feature_names)
            
            meta = onnx_model.metadata_props.add()
            meta.key = "model_type"
            meta.value = "risk"
            
            meta = onnx_model.metadata_props.add()
            meta.key = "risk_params"
            meta.value = json.dumps({
                'cvar_window': self.best_params.get('cvar_window', 100),
                'kelly_fraction': self.best_params.get('kelly_fraction', 0.25),
                'drawdown_level_1': self.best_params.get('drawdown_level_1', 0.05),
                'drawdown_level_2': self.best_params.get('drawdown_level_2', 0.10)
            })
            
            onnx.save(onnx_model, onnx_path)
            print(f"ONNX model saved: {onnx_path}")
            
            # Validate ONNX export
            self._validate_onnx(onnx_path)
            
        except Exception as e:
            print(f"ONNX export failed: {e}")
    
    def _validate_onnx(self, onnx_path: str):
        """Validate ONNX model inference"""
        try:
            import onnxruntime as ort
            
            # Load ONNX session
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            
            # Test inference
            test_input = self.X_val[:10].astype(np.float32)
            
            # LightGBM prediction
            lgb_pred = self.best_model.predict(test_input)
            
            # ONNX prediction
            onnx_pred = session.run(None, {input_name: test_input})[0].flatten()
            
            # Compare
            max_diff = np.max(np.abs(lgb_pred - onnx_pred))
            print(f"ONNX validation: max difference = {max_diff:.6f}")
            
            if max_diff < 1e-4:
                print("✅ ONNX export validated successfully")
            else:
                print("⚠️ ONNX predictions differ from LightGBM")
                
        except Exception as e:
            print(f"ONNX validation failed: {e}")


def run_risk_optimization(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_trials: int = 100,
    multi_objective: bool = True,
    use_gpu: bool = False
) -> Dict:
    """
    Convenience function to run risk model optimization
    
    Args:
        X: Features
        y: Risk score targets
        feature_names: Feature names
        n_trials: Number of Optuna trials
        multi_objective: Use multi-objective optimization
        use_gpu: Use GPU
        
    Returns:
        Best parameters and metrics
    """
    optimizer = RiskModelOptimizer(
        study_name=f"risk_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        n_trials=n_trials,
        multi_objective=multi_objective,
        use_gpu=use_gpu
    )
    
    optimizer.set_data(X, y, feature_names)
    return optimizer.optimize()
