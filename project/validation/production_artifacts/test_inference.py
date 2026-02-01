#!/usr/bin/env python
"""Quick test for inference pipeline"""

import numpy as np
from inference_pipeline import CryptoInferencePipeline

def main():
    pipeline = CryptoInferencePipeline('./production_artifacts')
    
    # Test prediction
    features = {
        'scalp': np.random.randn(20).astype(np.float32),
        'intraday': np.random.randn(20).astype(np.float32),
        'swing': np.random.randn(20).astype(np.float32)
    }
    
    result = pipeline.predict(features)
    print(f"Status: {result['status']}")
    print(f"Predictions: {result['predictions']}")

if __name__ == "__main__":
    main()
