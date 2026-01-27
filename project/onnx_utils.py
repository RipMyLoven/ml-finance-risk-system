"""
ONNX Model Utilities - загрузка и инференс ONNX моделей

Использование:
    from onnx_utils import ONNXPredictor
    
    predictor = ONNXPredictor("models/scalp_lgbm.onnx")
    predictions = predictor.predict(features)
"""

import json
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

# Try importing ONNX Runtime
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

# Try importing ONNX for metadata
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXPredictor:
    """
    ONNX Model Predictor для инференса
    
    Поддерживает:
    - CPU и GPU inference
    - Batch predictions
    - Classification и Regression модели
    """
    
    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,
        num_threads: int = None
    ):
        """
        Args:
            model_path: путь к .onnx файлу
            use_gpu: использовать GPU (требуется onnxruntime-gpu)
            num_threads: количество потоков для CPU
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime\n"
                "For GPU: pip install onnxruntime-gpu"
            )
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Session options
        sess_options = ort.SessionOptions()
        
        if num_threads:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
        
        # Execution providers
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Create session
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.feature_names = self.metadata.get('feature_names', [])
        self.model_type = self.metadata.get('model_type', 'unknown')
        self.is_classification = self.model_type != 'risk'
    
    def _load_metadata(self) -> Dict:
        """Load metadata from ONNX model"""
        metadata = {}
        
        if ONNX_AVAILABLE:
            model = onnx.load(str(self.model_path))
            for prop in model.metadata_props:
                if prop.key == 'feature_names':
                    metadata[prop.key] = json.loads(prop.value)
                else:
                    metadata[prop.key] = prop.value
        
        return metadata
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict using ONNX model
        
        Args:
            X: features array (n_samples, n_features)
            
        Returns:
            Dict with predictions:
            - Classification: P_down, P_flat, P_up, predicted_class
            - Regression: prediction
        """
        # Ensure correct dtype and shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X = X.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: X})
        
        if self.is_classification:
            # outputs[0] = labels, outputs[1] = probabilities
            if len(outputs) >= 2:
                labels = outputs[0]
                proba = outputs[1]
                
                # Handle different probability formats
                if isinstance(proba, list):
                    # List of dicts format
                    proba_array = np.array([[p.get(0, 0), p.get(1, 0), p.get(2, 0)] for p in proba])
                else:
                    proba_array = proba
                
                return {
                    'P_down': proba_array[:, 0] if proba_array.ndim > 1 else proba_array[0],
                    'P_flat': proba_array[:, 1] if proba_array.ndim > 1 else proba_array[1],
                    'P_up': proba_array[:, 2] if proba_array.ndim > 1 else proba_array[2],
                    'predicted_class': labels,
                    'expected_return': (proba_array[:, 2] - proba_array[:, 0]) if proba_array.ndim > 1 else (proba_array[2] - proba_array[0])
                }
            else:
                # Only probabilities
                proba = outputs[0]
                return {
                    'P_down': proba[:, 0],
                    'P_flat': proba[:, 1],
                    'P_up': proba[:, 2],
                    'predicted_class': np.argmax(proba, axis=1),
                    'expected_return': proba[:, 2] - proba[:, 0]
                }
        else:
            # Regression
            prediction = outputs[0]
            return {
                'prediction': prediction.flatten(),
                'risk_score': np.clip(prediction.flatten() * 100, 0, 1)  # Normalize to 0-1
            }
    
    def predict_single(self, X: np.ndarray) -> Dict[str, float]:
        """Predict single sample"""
        result = self.predict(X.reshape(1, -1))
        return {k: float(v[0]) if isinstance(v, np.ndarray) else float(v) for k, v in result.items()}
    
    def get_info(self) -> Dict:
        """Get model information"""
        return {
            'model_path': str(self.model_path),
            'model_type': self.model_type,
            'is_classification': self.is_classification,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'input_name': self.input_name,
            'input_shape': self.input_shape,
            'output_names': self.output_names,
            'metadata': self.metadata
        }


def load_all_models(models_dir: str = "models", use_gpu: bool = False) -> Dict[str, ONNXPredictor]:
    """
    Load all ONNX models from directory
    
    Returns:
        Dict[model_type] = ONNXPredictor
    """
    models = {}
    models_path = Path(models_dir)
    
    for onnx_file in models_path.glob("*.onnx"):
        model_type = onnx_file.stem.replace("_lgbm", "")
        try:
            models[model_type] = ONNXPredictor(str(onnx_file), use_gpu=use_gpu)
            print(f"Loaded: {model_type} from {onnx_file}")
        except Exception as e:
            print(f"Failed to load {onnx_file}: {e}")
    
    return models


def benchmark_model(predictor: ONNXPredictor, n_samples: int = 1000, n_runs: int = 10) -> Dict:
    """
    Benchmark ONNX model inference speed
    
    Returns:
        Dict with timing statistics
    """
    import time
    
    n_features = len(predictor.feature_names) or 42  # Default
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Warmup
    for _ in range(3):
        predictor.predict(X)
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predictor.predict(X)
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times)
    
    return {
        'n_samples': n_samples,
        'n_runs': n_runs,
        'mean_time_ms': float(times.mean() * 1000),
        'std_time_ms': float(times.std() * 1000),
        'min_time_ms': float(times.min() * 1000),
        'max_time_ms': float(times.max() * 1000),
        'throughput_samples_per_sec': float(n_samples / times.mean())
    }


if __name__ == "__main__":
    # Test ONNX loading
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/scalp_lgbm.onnx"
    
    try:
        print(f"Loading model: {model_path}")
        predictor = ONNXPredictor(model_path)
        
        print("\nModel Info:")
        info = predictor.get_info()
        for k, v in info.items():
            if k != 'feature_names':
                print(f"  {k}: {v}")
        print(f"  n_features: {len(info['feature_names'])}")
        
        # Test prediction
        n_features = len(predictor.feature_names) or 42
        X_test = np.random.randn(5, n_features).astype(np.float32)
        
        print("\nTest Predictions:")
        result = predictor.predict(X_test)
        for k, v in result.items():
            print(f"  {k}: {v[:3]}...")
        
        # Benchmark
        print("\nBenchmark:")
        bench = benchmark_model(predictor)
        print(f"  Mean time: {bench['mean_time_ms']:.2f} ms")
        print(f"  Throughput: {bench['throughput_samples_per_sec']:.0f} samples/sec")
        
    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        print("Train models first: python train.py --fast")
    except ImportError as e:
        print(f"Import error: {e}")
