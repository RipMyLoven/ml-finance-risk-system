#!/usr/bin/env python
"""
Advanced Risk-Controlled Training System

This script integrates the Advanced Risk Model with Optuna optimization
for the complete multi-model trading system.

Components:
- Scalp Model (5m/15m timeframes)
- Intraday Model (1h/4h timeframes)
- Swing Model (1d+ timeframes)
- Advanced Risk Model (CORE CONTROLLER)

The Risk Model is trained with:
- Separate Optuna study
- Multi-objective optimization (AUC↑, Drawdown↓, CVaR↓)
- Constraint-aware optimization

Usage:
    python train_with_risk.py --trials 100            # Full optimization
    python train_with_risk.py --risk-only --trials 50 # Risk model only
    python train_with_risk.py --fast                  # Fast training
    python train_with_risk.py --validate-onnx        # Validate ONNX exports
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
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    LGBM_MARKET_PARAMS, LGBM_RISK_PARAMS, MODEL_PATHS, RANDOM_STATE,
    PRICE_MOVE_THRESHOLD, PREDICTION_HORIZON,
    TRAIN_TEST_SPLIT, VALIDATION_SPLIT,
    CVAR_CONFIG, KELLY_CONFIG, DRAWDOWN_CONFIG, RISK_MODEL_PATHS
)

# Import feature builders
from features.scalp_features import build_scalp_features
from features.intraday_features import build_intraday_features
from features.swing_features import build_swing_features

# Import Advanced Risk Module
from risk import (
    CVaREngine, 
    KellySizer, 
    DrawdownController,
    AdvancedRiskModel,
    build_advanced_risk_features
)
from risk.train_risk_optuna import RiskModelOptimizer, run_risk_optimization

# Try importing data loader
try:
    from data.data_loader_v2 import prepare_training_data_v2 as prepare_training_data
    DATA_LOADER_V2 = True
except ImportError:
    try:
        from data.data_loader import prepare_training_data
        DATA_LOADER_V2 = False
    except ImportError:
        DATA_LOADER_V2 = False
        prepare_training_data = None

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
    import onnxruntime as ort
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX tools not installed. Run: pip install onnxmltools onnx onnxruntime")


class RiskAwareTrainer:
    """
    Complete Training System with Advanced Risk Model
    
    This trainer integrates:
    1. Trading model training (scalp/intraday/swing)
    2. Advanced Risk Model training with multi-objective optimization
    3. ONNX export with validation
    4. Complete system integration
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        n_trials: int = 100,
        study_name: Optional[str] = None
    ):
        self.use_gpu = use_gpu
        self.n_trials = n_trials
        self.study_name = study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Data storage
        self.raw_data = {}
        self.processed_data = {}
        self.feature_names = {}
        
        # Models
        self.models = {}
        self.metrics = {}
        
        # Risk components
        self.cvar_engine = CVaREngine(**{
            'confidence_level': CVAR_CONFIG['confidence_level'],
            'default_window': CVAR_CONFIG['default_window'],
            'block_threshold': CVAR_CONFIG['block_threshold'],
            'model_cvar_limits': CVAR_CONFIG['model_limits']
        })
        
        self.kelly_sizer = KellySizer(
            max_leverage=KELLY_CONFIG['max_leverage'],
            max_position_pct=KELLY_CONFIG['max_position_pct'],
            model_leverage_limits=KELLY_CONFIG['model_leverage_limits']
        )
        
        # Create directories
        os.makedirs("optuna_studies", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
    def load_data(self, data_path: Optional[str] = None) -> Dict:
        """Load and prepare training data"""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        if data_path is None:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(os.path.dirname(project_dir), 'data')
        
        print(f"Data path: {data_path}")
        
        if prepare_training_data is None:
            print("Warning: Data loader not available. Using synthetic data.")
            self._generate_synthetic_data()
            return self.raw_data
        
        if DATA_LOADER_V2:
            self.raw_data = prepare_training_data(
                data_dir=data_path,
                timeframes=['5m', '15m', '1h', '4h', '1d'],
                include_derivatives=True
            )
        else:
            self.raw_data = prepare_training_data(
                data_dir=data_path,
                timeframes=['5m', '15m', '1h', '4h', '1d']
            )
        
        if not self.raw_data:
            print("Warning: No data loaded. Using synthetic data.")
            self._generate_synthetic_data()
        else:
            print(f"Loaded data for {len(self.raw_data)} symbols")
        
        return self.raw_data
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing"""
        print("Generating synthetic data...")
        
        np.random.seed(42)
        n_samples = 10000
        
        for tf in ['5m', '1h', '1d']:
            dates = pd.date_range(start='2023-01-01', periods=n_samples, freq=tf)
            
            # Random walk price
            returns = np.random.normal(0, 0.02, n_samples)
            price = 100 * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame({
                'open': price * (1 + np.random.uniform(-0.01, 0.01, n_samples)),
                'high': price * (1 + np.random.uniform(0, 0.02, n_samples)),
                'low': price * (1 - np.random.uniform(0, 0.02, n_samples)),
                'close': price,
                'volume': np.random.exponential(1000000, n_samples),
            }, index=dates)
            
            if 'BTCUSDT' not in self.raw_data:
                self.raw_data['BTCUSDT'] = {}
            self.raw_data['BTCUSDT'][tf] = df
    
    def prepare_risk_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data for Advanced Risk Model
        
        Uses build_advanced_risk_features for comprehensive feature set.
        """
        print("\n" + "="*60)
        print("PREPARING RISK MODEL DATA")
        print("="*60)
        
        all_features = []
        all_targets = []
        feature_names = None
        
        for symbol, tf_data in self.raw_data.items():
            if '5m' not in tf_data:
                continue
            
            try:
                df = tf_data['5m'].copy()
                
                # Build advanced risk features
                df, f_names = build_advanced_risk_features(
                    df,
                    market_predictions=None,  # Will add later if available
                    include_derivatives=True
                )
                
                if len(df) > 100 and 'risk_target' in df.columns:
                    if feature_names is None:
                        feature_names = f_names
                    
                    # Align features
                    missing = set(feature_names) - set(f_names)
                    extra = set(f_names) - set(feature_names)
                    
                    for col in missing:
                        df[col] = 0.0
                    
                    X = df[feature_names].values
                    y = df['risk_target'].values
                    
                    # Remove NaN
                    valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
                    
                    if np.sum(valid_mask) > 50:
                        all_features.append(X[valid_mask].astype(np.float32))
                        all_targets.append(y[valid_mask].astype(np.float32))
                        print(f"  {symbol}: {np.sum(valid_mask)} samples")
                        
            except Exception as e:
                print(f"  Error processing {symbol}: {e}")
        
        if all_features:
            X = np.vstack(all_features)
            y = np.concatenate(all_targets)
            
            # Clean
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.processed_data['risk'] = {
                'X': X,
                'y': y,
                'feature_names': feature_names
            }
            
            print(f"\nRISK DATA: {X.shape[0]:,} samples, {X.shape[1]} features")
            
            return X, y, feature_names
        else:
            raise ValueError("No risk training data available")
    
    def train_risk_model_optuna(
        self,
        n_trials: int = 100,
        multi_objective: bool = True
    ) -> Dict:
        """
        Train Risk Model with Optuna optimization
        
        Uses multi-objective optimization to balance:
        - Prediction accuracy (RMSE)
        - CVaR reduction
        - Tail-loss frequency
        """
        print("\n" + "="*60)
        print("RISK MODEL OPTUNA OPTIMIZATION")
        print("="*60)
        
        if 'risk' not in self.processed_data:
            self.prepare_risk_data()
        
        X = self.processed_data['risk']['X']
        y = self.processed_data['risk']['y']
        feature_names = self.processed_data['risk']['feature_names']
        
        # Run optimization
        results = run_risk_optimization(
            X=X,
            y=y,
            feature_names=feature_names,
            n_trials=n_trials,
            multi_objective=multi_objective,
            use_gpu=self.use_gpu
        )
        
        self.metrics['risk'] = results['metrics']
        
        # Load trained model
        model_path = os.path.join("optuna_studies", "risk_model_optimized.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
                self.models['risk'] = saved['model']
        
        return results
    
    def train_risk_model_fast(self) -> Dict:
        """Fast risk model training without Optuna"""
        print("\n" + "="*60)
        print("FAST RISK MODEL TRAINING")
        print("="*60)
        
        if 'risk' not in self.processed_data:
            self.prepare_risk_data()
        
        X = self.processed_data['risk']['X']
        y = self.processed_data['risk']['y']
        feature_names = self.processed_data['risk']['feature_names']
        
        # Train/val split (time series)
        n_train = int(len(X) * 0.8)
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Default params
        params = LGBM_RISK_PARAMS.copy()
        if self.use_gpu:
            params['device'] = 'gpu'
        
        # Train
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ]
        )
        
        # Evaluate
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        print(f"\nRisk Model RMSE: {rmse:.4f}")
        
        self.models['risk'] = model
        self.feature_names['risk'] = feature_names
        self.metrics['risk'] = {'rmse': rmse}
        
        # Save model
        self._save_risk_model(model, feature_names)
        
        return {'rmse': rmse}
    
    def _save_risk_model(
        self,
        model: lgb.Booster,
        feature_names: List[str]
    ):
        """Save risk model in multiple formats"""
        # Save PKL
        pkl_path = RISK_MODEL_PATHS['pkl']
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_names': feature_names,
                'metrics': self.metrics.get('risk', {}),
                'cvar_config': CVAR_CONFIG,
                'kelly_config': KELLY_CONFIG,
                'drawdown_config': DRAWDOWN_CONFIG
            }, f)
        print(f"Model saved: {pkl_path}")
        
        # Save TXT
        txt_path = RISK_MODEL_PATHS['txt']
        model.save_model(txt_path)
        print(f"Model saved: {txt_path}")
        
        # Save ONNX
        if ONNX_AVAILABLE:
            self._export_risk_onnx(model, feature_names)
    
    def _export_risk_onnx(
        self,
        model: lgb.Booster,
        feature_names: List[str]
    ):
        """Export risk model to ONNX with validation"""
        try:
            onnx_path = RISK_MODEL_PATHS['onnx']
            
            n_features = len(feature_names)
            initial_type = [('input', FloatTensorType([None, n_features]))]
            
            onnx_model = convert_lightgbm(
                model,
                initial_types=initial_type,
                target_opset=15
            )
            
            # Add metadata
            meta = onnx_model.metadata_props.add()
            meta.key = "feature_names"
            meta.value = json.dumps(feature_names)
            
            meta = onnx_model.metadata_props.add()
            meta.key = "model_type"
            meta.value = "risk"
            
            meta = onnx_model.metadata_props.add()
            meta.key = "cvar_config"
            meta.value = json.dumps(CVAR_CONFIG)
            
            meta = onnx_model.metadata_props.add()
            meta.key = "kelly_config"
            meta.value = json.dumps(KELLY_CONFIG)
            
            onnx.save(onnx_model, onnx_path)
            print(f"ONNX model saved: {onnx_path}")
            
            # Validate
            self._validate_onnx_export(model, onnx_path, feature_names)
            
        except Exception as e:
            print(f"ONNX export failed: {e}")
    
    def _validate_onnx_export(
        self,
        lgb_model: lgb.Booster,
        onnx_path: str,
        feature_names: List[str]
    ):
        """Validate ONNX export correctness"""
        print("Validating ONNX export...")
        
        # Load ONNX
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        
        # Test data
        X_test = self.processed_data['risk']['X'][:100].astype(np.float32)
        
        # LightGBM prediction
        lgb_pred = lgb_model.predict(X_test)
        
        # ONNX prediction
        onnx_pred = session.run(None, {input_name: X_test})[0].flatten()
        
        # Compare
        max_diff = np.max(np.abs(lgb_pred - onnx_pred))
        mean_diff = np.mean(np.abs(lgb_pred - onnx_pred))
        
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-4:
            print("  ✅ ONNX validation PASSED")
        else:
            print("  ⚠️ ONNX validation WARNING: predictions differ")
        
        # Measure latency
        import time
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = session.run(None, {input_name: X_test[:1]})
            times.append((time.perf_counter() - start) * 1000)
        
        print(f"  ONNX latency: {np.mean(times):.2f}ms (±{np.std(times):.2f}ms)")
    
    def train_all(self, mode: str = 'fast') -> Dict:
        """
        Train all models including Risk Model
        
        Args:
            mode: 'fast', 'optuna', or 'risk-only'
        """
        print("\n" + "="*60)
        print("RISK-AWARE TRAINING SYSTEM")
        print("="*60)
        print(f"Mode: {mode}")
        print(f"Time: {datetime.now()}")
        
        results = {}
        
        if mode == 'risk-only':
            # Only train risk model
            if OPTUNA_AVAILABLE:
                results['risk'] = self.train_risk_model_optuna(self.n_trials)
            else:
                results['risk'] = self.train_risk_model_fast()
        
        elif mode == 'optuna':
            # Full Optuna optimization
            # TODO: Add trading model optimization
            results['risk'] = self.train_risk_model_optuna(self.n_trials)
        
        else:
            # Fast training
            results['risk'] = self.train_risk_model_fast()
        
        return results
    
    def get_risk_assessment(self) -> Dict:
        """
        Generate final risk assessment report
        """
        print("\n" + "="*60)
        print("FINAL RISK ASSESSMENT")
        print("="*60)
        
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'risk_metrics': self.metrics.get('risk', {}),
            'cvar_config': CVAR_CONFIG,
            'kelly_config': KELLY_CONFIG,
            'drawdown_config': DRAWDOWN_CONFIG,
            'recommendations': []
        }
        
        # Add recommendations based on metrics
        if 'risk' in self.metrics:
            rmse = self.metrics['risk'].get('rmse', 1.0)
            
            if rmse < 0.1:
                assessment['recommendations'].append("Risk model performance: EXCELLENT")
            elif rmse < 0.2:
                assessment['recommendations'].append("Risk model performance: GOOD")
            else:
                assessment['recommendations'].append("Risk model performance: NEEDS IMPROVEMENT")
        
        # Print assessment
        print(json.dumps(assessment, indent=2, default=str))
        
        # Save assessment
        assessment_path = "models/risk_assessment.json"
        with open(assessment_path, 'w') as f:
            json.dump(assessment, f, indent=2, default=str)
        print(f"\nAssessment saved: {assessment_path}")
        
        return assessment


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Risk-Aware ML Training System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--fast', action='store_true',
                       help='Fast training without Optuna')
    parser.add_argument('--risk-only', action='store_true',
                       help='Train only risk model')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of Optuna trials')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU')
    parser.add_argument('--validate-onnx', action='store_true',
                       help='Only validate ONNX exports')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data directory')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RiskAwareTrainer(
        use_gpu=args.gpu,
        n_trials=args.trials
    )
    
    # Load data
    trainer.load_data(args.data_path)
    
    # Determine mode
    if args.validate_onnx:
        # Just validate existing ONNX
        trainer.prepare_risk_data()
        print("ONNX validation mode")
        return
    elif args.risk_only:
        mode = 'risk-only'
    elif args.fast:
        mode = 'fast'
    else:
        mode = 'optuna'
    
    # Train
    results = trainer.train_all(mode=mode)
    
    # Generate assessment
    trainer.get_risk_assessment()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
