"""
═══════════════════════════════════════════════════════════════════════════════
 AI CRYPTO TRADING SYSTEM - VALIDATION PIPELINE
═══════════════════════════════════════════════════════════════════════════════

Complete 12-checkpoint validation system for production readiness.

Checkpoints:
    1. Data & Sanity Checks
    2. Feature Engineering Validation
    3. Labeling & Targets
    4. Baseline Models
    5. Optuna Optimization
    6. Advanced Risk Model Validation
    7. Model Interaction & Ensemble
    8. Metrics & Overfitting Control
    9. ONNX Export & Validation
    10. Production Readiness
    11. Logging & Observability
    12. Final Review & Sign-off

Usage:
    # Run all checkpoints
    from validation import run_full_validation
    result = run_full_validation()
    
    # Run individual checkpoint
    from validation import DataSanityChecker
    checker = DataSanityChecker()
    result = checker.run_all_checks()
"""

# Checkpoint 1: Data & Sanity
from .data_sanity import DataSanityChecker, run_data_sanity_check

# Checkpoint 2: Feature Validation
from .feature_validation import FeatureValidator, run_feature_validation

# Checkpoint 3: Labeling & Targets
from .labeling_validation import LabelingValidator, run_labeling_validation

# Checkpoint 4: Baseline Models
from .baseline_models import BaselineTrainer, run_baseline_training

# Checkpoint 5: Optuna Optimization
from .optuna_optimization import OptunaOptimizer, run_optuna_optimization

# Checkpoint 6: Risk Validation
from .risk_validation import AdvancedRiskValidator, run_risk_validation

# Checkpoint 7: Ensemble Validation
from .ensemble_validation import EnsembleValidator, run_ensemble_validation

# Checkpoint 8: Overfitting Control
from .overfitting_control import OverfittingController, run_overfitting_check

# Checkpoint 9: ONNX Validation
from .onnx_validation import ONNXValidator, run_onnx_validation

# Checkpoint 10: Production Readiness
from .production_readiness import ProductionReadinessChecker, run_production_readiness_check

# Checkpoint 11: Logging & Observability
from .logging_observability import LoggingValidator, StructuredLogger, run_logging_validation

# Checkpoint 12: Final Review
from .final_review import FinalReviewer, run_final_review


__all__ = [
    # Checkpoint classes
    'DataSanityChecker',
    'FeatureValidator',
    'LabelingValidator',
    'BaselineTrainer',
    'OptunaOptimizer',
    'AdvancedRiskValidator',
    'EnsembleValidator',
    'OverfittingController',
    'ONNXValidator',
    'ProductionReadinessChecker',
    'LoggingValidator',
    'StructuredLogger',
    'FinalReviewer',
    
    # Runner functions
    'run_data_sanity_check',
    'run_feature_validation',
    'run_labeling_validation',
    'run_baseline_training',
    'run_optuna_optimization',
    'run_risk_validation',
    'run_ensemble_validation',
    'run_overfitting_check',
    'run_onnx_validation',
    'run_production_readiness_check',
    'run_logging_validation',
    'run_final_review',
]


def run_full_validation():
    """Run full 12-checkpoint validation pipeline"""
    from .run_validation import main
    return main()
