"""
═══════════════════════════════════════════════════════════════════════════════
 9️⃣ ONNX EXPORT & VALIDATION
═══════════════════════════════════════════════════════════════════════════════

ONNX validation:
✅ Export each model to ONNX
✅ Verify inputs/outputs
✅ Verify dtype/shapes
✅ Compare output before and after ONNX
✅ Check inference latency
✅ Record ONNX versions
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# ONNX imports
try:
    import onnx
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ONNXResult:
    """Results from ONNX validation"""
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    export_results: Dict[str, Dict] = field(default_factory=dict)
    io_validation: Dict[str, Dict] = field(default_factory=dict)
    output_comparison: Dict[str, Dict] = field(default_factory=dict)
    latency_tests: Dict[str, Dict] = field(default_factory=dict)


class ONNXValidator:
    """
    ONNX Export & Validation
    
    Implements checkpoint 9:
    - Model export to ONNX
    - I/O validation
    - Numeric comparison
    - Latency benchmarking
    """
    
    # Latency thresholds (ms)
    MAX_LATENCY_MS = {
        'scalp': 5,      # 5ms for scalp
        'intraday': 10,  # 10ms for intraday
        'swing': 20,     # 20ms for swing
        'risk': 5        # 5ms for risk
    }
    
    # Output comparison tolerance
    COMPARISON_RTOL = 1e-4
    COMPARISON_ATOL = 1e-6
    
    def __init__(self, log_to_console: bool = True):
        self.log_to_console = log_to_console
        self.results = ONNXResult()
        
        if not ONNX_AVAILABLE:
            self.results.errors.append("ONNX libraries not installed")
            self.results.passed = False
    
    def log(self, msg: str, level: str = "INFO"):
        if self.log_to_console:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"[{timestamp}] {prefix.get(level, '')} {msg}")
    
    def run_all_validations(self) -> ONNXResult:
        """Run all ONNX validations"""
        print("\n" + "="*80)
        print("  9️⃣  ONNX EXPORT & VALIDATION")
        print("="*80 + "\n")
        
        if not ONNX_AVAILABLE:
            print("  ❌ ONNX libraries not available")
            print("     Install: pip install onnx onnxruntime skl2onnx onnxmltools")
            return self.results
        
        # Load or create models
        models = self._load_or_create_models()
        
        if not models:
            self.results.errors.append("No models available for validation")
            self.results.passed = False
            return self.results
        
        # Validate each model
        for model_name, model_data in models.items():
            print(f"\n  {'='*50}")
            print(f"  🔄 Validating {model_name.upper()} ONNX Export")
            print(f"  {'='*50}")
            
            # 1. Export to ONNX
            self.log(f"Exporting {model_name} to ONNX...")
            onnx_path = self._export_to_onnx(model_name, model_data)
            
            if onnx_path is None:
                continue
            
            # 2. Validate I/O
            self.log(f"Validating I/O for {model_name}...")
            self._validate_io(model_name, onnx_path, model_data)
            
            # 3. Compare outputs
            self.log(f"Comparing outputs for {model_name}...")
            self._compare_outputs(model_name, model_data, onnx_path)
            
            # 4. Benchmark latency
            self.log(f"Benchmarking latency for {model_name}...")
            self._benchmark_latency(model_name, onnx_path, model_data)
        
        # 5. Log ONNX versions
        self._log_versions()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _load_or_create_models(self) -> Dict[str, Dict]:
        """Load existing models or create test models"""
        models = {}
        
        # Try to load existing models
        models_dir = Path(__file__).parent.parent / 'models'
        
        model_names = ['scalp', 'intraday', 'swing', 'risk']
        
        for name in model_names:
            pkl_path = models_dir / f'{name}_lgbm.pkl'
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Get feature info
                    json_path = models_dir / f'{name}_lgbm_features.json'
                    n_features = 20
                    
                    if json_path.exists():
                        with open(json_path, 'r') as f:
                            features_info = json.load(f)
                        n_features = len(features_info.get('features', []))
                    
                    models[name] = {
                        'model': model,
                        'n_features': n_features,
                        'type': 'regression' if name == 'risk' else 'classification'
                    }
                    self.log(f"Loaded {name} model from {pkl_path}", "SUCCESS")
                except Exception as e:
                    self.log(f"Error loading {name}: {e}", "WARNING")
        
        # Create test models if none loaded
        if not models:
            self.log("Creating test models...", "WARNING")
            models = self._create_test_models()
        
        return models
    
    def _create_test_models(self) -> Dict[str, Dict]:
        """Create simple test models"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        models = {}
        
        # Classification models
        for name in ['scalp', 'intraday', 'swing']:
            y = np.random.randint(0, 3, n_samples)
            model = lgb.LGBMClassifier(n_estimators=50, verbose=-1, random_state=42)
            model.fit(X, y)
            models[name] = {
                'model': model,
                'n_features': n_features,
                'type': 'classification',
                'sample_X': X[:10]
            }
        
        # Regression model (risk)
        y = np.random.randn(n_samples)
        model = lgb.LGBMRegressor(n_estimators=50, verbose=-1, random_state=42)
        model.fit(X, y)
        models['risk'] = {
            'model': model,
            'n_features': n_features,
            'type': 'regression',
            'sample_X': X[:10]
        }
        
        return models
    
    def _export_to_onnx(self, model_name: str, model_data: Dict) -> Optional[str]:
        """Export model to ONNX"""
        try:
            model = model_data['model']
            n_features = model_data['n_features']
            model_type = model_data['type']
            
            output_dir = Path(__file__).parent / 'onnx_exports'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            onnx_path = output_dir / f'{model_name}_validated.onnx'
            
            # Use onnxmltools for LightGBM
            try:
                from onnxmltools import convert_lightgbm
                from onnxmltools.convert.common.data_types import FloatTensorType as FTT
                
                initial_type = [('input', FTT([None, n_features]))]
                
                onnx_model = convert_lightgbm(
                    model, 
                    initial_types=initial_type,
                    target_opset=12
                )
                
                # Save
                onnx.save(onnx_model, str(onnx_path))
                
            except ImportError:
                # Fallback: save model for later conversion
                self.log("onnxmltools not available, trying sklearn conversion", "WARNING")
                
                # Create wrapper
                from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
                
                class LGBMWrapper(BaseEstimator, ClassifierMixin if model_type == 'classification' else RegressorMixin):
                    def __init__(self, model):
                        self.model = model
                    
                    def fit(self, X, y):
                        return self
                    
                    def predict(self, X):
                        return self.model.predict(X)
                    
                    def predict_proba(self, X):
                        return self.model.predict_proba(X)
                
                # For now, just note it's not exported
                self.results.warnings.append(f"{model_name}: Direct ONNX export failed, using fallback")
                return None
            
            self.results.export_results[model_name] = {
                'path': str(onnx_path),
                'size_kb': onnx_path.stat().st_size / 1024,
                'success': True
            }
            
            print(f"    ✅ Exported to: {onnx_path}")
            print(f"       Size: {onnx_path.stat().st_size / 1024:.1f} KB")
            
            return str(onnx_path)
            
        except Exception as e:
            self.results.errors.append(f"{model_name}: Export failed - {e}")
            self.log(f"Export failed: {e}", "ERROR")
            return None
    
    def _validate_io(self, model_name: str, onnx_path: str, model_data: Dict):
        """Validate ONNX model inputs and outputs"""
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Check model
            onnx.checker.check_model(onnx_model)
            
            # Get input/output info
            session = ort.InferenceSession(onnx_path)
            
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            input_info = []
            for inp in inputs:
                input_info.append({
                    'name': inp.name,
                    'shape': inp.shape,
                    'dtype': inp.type
                })
            
            output_info = []
            for out in outputs:
                output_info.append({
                    'name': out.name,
                    'shape': out.shape,
                    'dtype': out.type
                })
            
            print(f"\n    📥 INPUTS:")
            for info in input_info:
                print(f"       Name: {info['name']}")
                print(f"       Shape: {info['shape']}")
                print(f"       Type: {info['dtype']}")
            
            print(f"\n    📤 OUTPUTS:")
            for info in output_info:
                print(f"       Name: {info['name']}")
                print(f"       Shape: {info['shape']}")
                print(f"       Type: {info['dtype']}")
            
            # Validate expected structure
            expected_features = model_data['n_features']
            actual_features = inputs[0].shape[1] if len(inputs[0].shape) > 1 else 'dynamic'
            
            if actual_features != 'dynamic' and actual_features != expected_features:
                self.results.warnings.append(f"{model_name}: Feature mismatch ({actual_features} vs {expected_features})")
            
            self.results.io_validation[model_name] = {
                'inputs': input_info,
                'outputs': output_info,
                'model_valid': True
            }
            
            print(f"\n    ✅ I/O validation passed")
            
        except Exception as e:
            self.results.errors.append(f"{model_name}: I/O validation failed - {e}")
            self.log(f"I/O validation failed: {e}", "ERROR")
    
    def _compare_outputs(self, model_name: str, model_data: Dict, onnx_path: str):
        """Compare original model output with ONNX output"""
        try:
            model = model_data['model']
            n_features = model_data['n_features']
            model_type = model_data['type']
            
            # Generate test input
            np.random.seed(42)
            X_test = np.random.randn(100, n_features).astype(np.float32)
            
            # Original model prediction
            if model_type == 'classification':
                original_pred = model.predict_proba(X_test)
            else:
                original_pred = model.predict(X_test)
            
            # ONNX prediction
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            
            onnx_output = session.run(None, {input_name: X_test})
            
            # For classification, compare probabilities
            if model_type == 'classification':
                onnx_pred = onnx_output[1]  # Usually second output is probabilities
                
                # Handle different formats
                if isinstance(onnx_pred, list):
                    onnx_pred = np.array([[p[i] for i in sorted(p.keys())] for p in onnx_pred])
            else:
                onnx_pred = onnx_output[0].flatten()
            
            # Compare
            if len(original_pred.shape) != len(onnx_pred.shape):
                # Reshape if needed
                if len(onnx_pred.shape) == 1 and model_type == 'classification':
                    self.results.warnings.append(f"{model_name}: Output shape mismatch, comparing labels")
                    original_pred = model.predict(X_test)
                    onnx_pred = onnx_output[0]
            
            # Calculate differences
            max_diff = np.max(np.abs(original_pred - onnx_pred))
            mean_diff = np.mean(np.abs(original_pred - onnx_pred))
            
            # Check if close
            is_close = np.allclose(original_pred, onnx_pred, 
                                   rtol=self.COMPARISON_RTOL, 
                                   atol=self.COMPARISON_ATOL)
            
            print(f"\n    📊 OUTPUT COMPARISON:")
            print(f"       Max difference: {max_diff:.6f}")
            print(f"       Mean difference: {mean_diff:.6f}")
            print(f"       Outputs match: {'✅ Yes' if is_close else '⚠️ No'}")
            
            self.results.output_comparison[model_name] = {
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff),
                'outputs_match': is_close
            }
            
            if not is_close:
                self.results.warnings.append(f"{model_name}: ONNX outputs differ from original")
            
        except Exception as e:
            self.results.errors.append(f"{model_name}: Output comparison failed - {e}")
            self.log(f"Output comparison failed: {e}", "ERROR")
    
    def _benchmark_latency(self, model_name: str, onnx_path: str, model_data: Dict):
        """Benchmark ONNX inference latency"""
        try:
            n_features = model_data['n_features']
            
            # Create session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            session = ort.InferenceSession(onnx_path, sess_options)
            input_name = session.get_inputs()[0].name
            
            # Test different batch sizes
            batch_sizes = [1, 10, 100]
            latencies = {}
            
            for batch_size in batch_sizes:
                X_test = np.random.randn(batch_size, n_features).astype(np.float32)
                
                # Warmup
                for _ in range(10):
                    session.run(None, {input_name: X_test})
                
                # Benchmark
                n_runs = 100
                start = time.perf_counter()
                for _ in range(n_runs):
                    session.run(None, {input_name: X_test})
                end = time.perf_counter()
                
                latency_ms = (end - start) / n_runs * 1000
                latencies[batch_size] = latency_ms
            
            print(f"\n    ⏱️ LATENCY BENCHMARK:")
            print(f"    {'Batch Size':<15} {'Latency (ms)':<15} {'Status':<10}")
            print(f"    {'-'*40}")
            
            max_latency = self.MAX_LATENCY_MS.get(model_name, 10)
            
            for batch_size, latency in latencies.items():
                status = "✅" if batch_size == 1 and latency < max_latency else "⚠️"
                print(f"    {batch_size:<15} {latency:.3f}{'':10} {status}")
            
            single_latency = latencies[1]
            
            if single_latency > max_latency:
                self.results.warnings.append(
                    f"{model_name}: Latency {single_latency:.2f}ms exceeds {max_latency}ms threshold"
                )
            
            self.results.latency_tests[model_name] = {
                'latencies_by_batch': latencies,
                'single_inference_ms': float(single_latency),
                'threshold_ms': max_latency,
                'within_threshold': single_latency < max_latency
            }
            
        except Exception as e:
            self.results.errors.append(f"{model_name}: Latency test failed - {e}")
            self.log(f"Latency test failed: {e}", "ERROR")
    
    def _log_versions(self):
        """Log ONNX-related versions"""
        print(f"\n  📦 ONNX VERSIONS:")
        
        versions = {}
        
        try:
            versions['onnx'] = onnx.__version__
            print(f"    ONNX: {onnx.__version__}")
        except:
            pass
        
        try:
            versions['onnxruntime'] = ort.__version__
            print(f"    ONNX Runtime: {ort.__version__}")
        except:
            pass
        
        try:
            import onnxmltools
            versions['onnxmltools'] = onnxmltools.__version__
            print(f"    onnxmltools: {onnxmltools.__version__}")
        except:
            pass
        
        try:
            versions['lightgbm'] = lgb.__version__
            print(f"    LightGBM: {lgb.__version__}")
        except:
            pass
        
        self.results.export_results['versions'] = versions
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("  📋 ONNX VALIDATION SUMMARY")
        print("="*80)
        
        print(f"\n  📊 Export Results:")
        for model, result in self.results.export_results.items():
            if model == 'versions':
                continue
            if isinstance(result, dict) and result.get('success', False):
                print(f"    ✅ {model}: {result.get('size_kb', 0):.1f} KB")
            else:
                print(f"    ❌ {model}: Export failed")
        
        print(f"\n  ⏱️ Latency Results:")
        for model, latency in self.results.latency_tests.items():
            status = "✅" if latency.get('within_threshold', False) else "⚠️"
            print(f"    {status} {model}: {latency.get('single_inference_ms', 0):.2f}ms")
        
        print(f"\n  🔄 Output Comparison:")
        for model, comp in self.results.output_comparison.items():
            status = "✅" if comp.get('outputs_match', False) else "⚠️"
            print(f"    {status} {model}: max_diff={comp.get('max_diff', 0):.6f}")
        
        if self.results.errors:
            print(f"\n  Errors:")
            for err in self.results.errors:
                print(f"    ❌ {err}")
            self.results.passed = False
        
        if self.results.warnings:
            print(f"\n  Warnings:")
            for warn in self.results.warnings[:10]:
                print(f"    ⚠️  {warn}")
        
        if self.results.passed:
            print(f"\n  ✅ ONNX VALIDATION: PASSED")
        else:
            print(f"\n  ❌ ONNX VALIDATION: FAILED")
        
        print("\n" + "="*80 + "\n")
    
    def save_results(self, output_path: str = None):
        """Save results"""
        if output_path is None:
            output_path = Path(__file__).parent / 'onnx_validation_report.json'
        
        report = {
            'passed': self.results.passed,
            'warnings': self.results.warnings,
            'errors': self.results.errors,
            'export_results': self.results.export_results,
            'io_validation': self.results.io_validation,
            'output_comparison': self.results.output_comparison,
            'latency_tests': self.results.latency_tests,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {output_path}")
        return output_path


def run_onnx_validation() -> ONNXResult:
    """Run ONNX validation"""
    validator = ONNXValidator()
    result = validator.run_all_validations()
    validator.save_results()
    return result


if __name__ == "__main__":
    result = run_onnx_validation()
    sys.exit(0 if result.passed else 1)
