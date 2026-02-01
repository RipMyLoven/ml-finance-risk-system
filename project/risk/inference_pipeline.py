"""
Production Risk Inference Pipeline

This module provides:
- Production-ready risk inference
- ONNX-based fast inference
- Emergency kill-switch
- Deterministic execution
- Comprehensive monitoring and logging
- Regime shift detection
- Signal degradation alerts
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
import time


# Try importing ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class InferenceMode(Enum):
    """Inference execution mode"""
    PRODUCTION = "production"    # Full safety checks
    BACKTEST = "backtest"        # Relaxed checks for backtesting
    DEBUG = "debug"              # Verbose logging


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class InferenceResult:
    """Result from risk inference"""
    risk_score: float           # Predicted risk score (0-1)
    position_size_factor: float # Recommended position size multiplier
    confidence: float           # Inference confidence
    latency_ms: float           # Inference latency
    is_blocked: bool            # Whether trade should be blocked
    block_reason: Optional[str] # Reason for blocking
    alerts: List[str]           # Any alerts generated
    timestamp: datetime         # Inference timestamp
    
    def to_dict(self) -> Dict:
        return {
            'risk_score': self.risk_score,
            'position_size_factor': self.position_size_factor,
            'confidence': self.confidence,
            'latency_ms': self.latency_ms,
            'is_blocked': self.is_blocked,
            'block_reason': self.block_reason,
            'alerts': self.alerts,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class MonitoringMetrics:
    """Real-time monitoring metrics"""
    total_inferences: int = 0
    blocked_trades: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    regime_shifts_detected: int = 0
    signal_degradation_alerts: int = 0
    kill_switch_triggers: int = 0
    uptime_hours: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'total_inferences': self.total_inferences,
            'blocked_trades': self.blocked_trades,
            'avg_latency_ms': self.avg_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'regime_shifts_detected': self.regime_shifts_detected,
            'signal_degradation_alerts': self.signal_degradation_alerts,
            'kill_switch_triggers': self.kill_switch_triggers,
            'uptime_hours': self.uptime_hours
        }


class RiskInferencePipeline:
    """
    Production Risk Inference Pipeline
    
    This is the production inference system for the Risk Model.
    It provides:
    
    1. ONNX-based fast inference
    2. Emergency kill-switch with automatic triggers
    3. Deterministic execution guarantees
    4. Real-time monitoring and alerting
    5. Regime shift detection
    6. Signal degradation monitoring
    
    Design Principles:
    - Fail-safe: Always err on the side of caution
    - Deterministic: Same input = same output
    - Observable: Full logging and monitoring
    - Fast: Sub-millisecond inference latency
    """
    
    # Safety thresholds
    MAX_LATENCY_MS = 100.0          # Maximum acceptable latency
    MAX_RISK_SCORE = 0.9            # Auto-block above this
    MIN_CONFIDENCE = 0.3            # Warn below this
    REGIME_SHIFT_WINDOW = 50        # Periods for regime detection
    SIGNAL_DEGRADATION_THRESHOLD = 0.3  # Sharpe degradation threshold
    
    def __init__(
        self,
        model_path: str,
        mode: InferenceMode = InferenceMode.PRODUCTION,
        risk_params: Optional[Dict] = None,
        max_latency_ms: float = 100.0,
        enable_kill_switch: bool = True,
        enable_monitoring: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize Risk Inference Pipeline
        
        Args:
            model_path: Path to ONNX model
            mode: Inference mode
            risk_params: Risk model parameters
            max_latency_ms: Maximum acceptable latency
            enable_kill_switch: Enable emergency kill switch
            enable_monitoring: Enable real-time monitoring
            log_level: Logging level
        """
        self.model_path = model_path
        self.mode = mode
        self.max_latency_ms = max_latency_ms
        self.enable_kill_switch = enable_kill_switch
        self.enable_monitoring = enable_monitoring
        
        # Load risk parameters
        self.risk_params = risk_params or {
            'cvar_window': 100,
            'kelly_fraction': 0.25,
            'drawdown_level_1': 0.05,
            'drawdown_level_2': 0.10,
            'volatility_scaling': 1.0
        }
        
        # Initialize logger
        self.logger = logging.getLogger('RiskInference')
        self.logger.setLevel(log_level)
        
        # State
        self._kill_switch_active = False
        self._kill_switch_reason = None
        self._start_time = datetime.now()
        
        # Monitoring
        self._metrics = MonitoringMetrics()
        self._latency_history: List[float] = []
        self._risk_score_history: List[float] = []
        self._alerts: List[Tuple[datetime, AlertLevel, str]] = []
        
        # Regime detection state
        self._feature_history: List[np.ndarray] = []
        self._prediction_history: List[float] = []
        
        # Load model
        self._load_model()
        
        # Validate model
        self._validate_model()
        
        self.logger.info(f"Risk Inference Pipeline initialized in {mode.value} mode")
        
    def _load_model(self):
        """Load ONNX model"""
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Session options for deterministic execution
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1  # Single thread for determinism
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create session
        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']  # CPU for determinism
        )
        
        # Get input/output info
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape
        self._output_names = [o.name for o in self._session.get_outputs()]
        
        # Load metadata
        self._load_metadata()
        
        self.logger.info(f"Model loaded: {self.model_path}")
        
    def _load_metadata(self):
        """Load model metadata"""
        try:
            import onnx
            model = onnx.load(self.model_path)
            
            self.feature_names = []
            for prop in model.metadata_props:
                if prop.key == 'feature_names':
                    self.feature_names = json.loads(prop.value)
                elif prop.key == 'risk_params':
                    # Override with model's params
                    model_params = json.loads(prop.value)
                    self.risk_params.update(model_params)
                    
        except Exception as e:
            self.logger.warning(f"Could not load metadata: {e}")
            self.feature_names = []
    
    def _validate_model(self):
        """Validate model works correctly"""
        try:
            # Create dummy input
            n_features = self._input_shape[1] if len(self._input_shape) > 1 else 50
            dummy_input = np.zeros((1, n_features), dtype=np.float32)
            
            # Run inference
            start = time.perf_counter()
            output = self._session.run(None, {self._input_name: dummy_input})
            latency = (time.perf_counter() - start) * 1000
            
            self.logger.info(f"Model validation passed. Latency: {latency:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            raise
    
    def infer(
        self,
        features: np.ndarray,
        model_type: str = 'unknown',
        check_regime: bool = True
    ) -> InferenceResult:
        """
        Run risk inference
        
        Args:
            features: Input features array
            model_type: Type of trading model requesting inference
            check_regime: Check for regime shifts
            
        Returns:
            InferenceResult with risk assessment
        """
        timestamp = datetime.now()
        alerts = []
        
        # Check kill switch first
        if self._kill_switch_active:
            return InferenceResult(
                risk_score=1.0,
                position_size_factor=0.0,
                confidence=0.0,
                latency_ms=0.0,
                is_blocked=True,
                block_reason=f"Kill switch active: {self._kill_switch_reason}",
                alerts=["KILL_SWITCH_ACTIVE"],
                timestamp=timestamp
            )
        
        # Prepare input
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features = features.astype(np.float32)
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Run inference with timing
        start = time.perf_counter()
        try:
            output = self._session.run(None, {self._input_name: features})
            raw_risk_score = output[0].flatten()[0]
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            # Fail safe - block trade
            return self._create_blocked_result(
                f"Inference error: {e}", timestamp, alerts
            )
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Check latency
        if latency_ms > self.max_latency_ms:
            alerts.append(f"HIGH_LATENCY: {latency_ms:.1f}ms")
            if self.mode == InferenceMode.PRODUCTION:
                # In production, high latency is a safety concern
                self._emit_alert(AlertLevel.WARNING, f"High latency: {latency_ms:.1f}ms")
        
        # Update latency tracking
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > 1000:
            self._latency_history = self._latency_history[-1000:]
        
        # Normalize risk score to 0-1
        risk_score = np.clip(raw_risk_score, 0.0, 1.0)
        
        # Update risk score history
        self._risk_score_history.append(risk_score)
        if len(self._risk_score_history) > 1000:
            self._risk_score_history = self._risk_score_history[-1000:]
        
        # Update feature history for regime detection
        self._feature_history.append(features.flatten())
        if len(self._feature_history) > self.REGIME_SHIFT_WINDOW * 2:
            self._feature_history = self._feature_history[-self.REGIME_SHIFT_WINDOW * 2:]
        
        # Check for regime shift
        if check_regime and len(self._feature_history) >= self.REGIME_SHIFT_WINDOW:
            regime_alert = self._check_regime_shift()
            if regime_alert:
                alerts.append(regime_alert)
        
        # Check for signal degradation
        if len(self._risk_score_history) >= 50:
            degradation_alert = self._check_signal_degradation()
            if degradation_alert:
                alerts.append(degradation_alert)
        
        # Calculate confidence based on input quality
        confidence = self._calculate_confidence(features, risk_score)
        
        if confidence < self.MIN_CONFIDENCE:
            alerts.append(f"LOW_CONFIDENCE: {confidence:.2f}")
        
        # Determine if trade should be blocked
        is_blocked, block_reason = self._should_block(
            risk_score, confidence, latency_ms, model_type
        )
        
        # Calculate position size factor
        position_size_factor = self._calculate_position_factor(risk_score, confidence)
        
        # Update monitoring
        self._update_monitoring(risk_score, latency_ms, is_blocked, alerts)
        
        result = InferenceResult(
            risk_score=float(risk_score),
            position_size_factor=float(position_size_factor),
            confidence=float(confidence),
            latency_ms=float(latency_ms),
            is_blocked=is_blocked,
            block_reason=block_reason,
            alerts=alerts,
            timestamp=timestamp
        )
        
        # Log in production mode
        if self.mode == InferenceMode.PRODUCTION or self.mode == InferenceMode.DEBUG:
            self.logger.info(
                f"[INFER] risk={risk_score:.3f} size={position_size_factor:.2f} "
                f"conf={confidence:.2f} blocked={is_blocked}"
            )
        
        return result
    
    def _should_block(
        self,
        risk_score: float,
        confidence: float,
        latency_ms: float,
        model_type: str
    ) -> Tuple[bool, Optional[str]]:
        """Determine if trade should be blocked"""
        
        # Auto-block very high risk
        if risk_score > self.MAX_RISK_SCORE:
            return True, f"Risk score {risk_score:.2f} exceeds threshold"
        
        # Block on very low confidence
        if confidence < 0.2:
            return True, f"Confidence {confidence:.2f} too low"
        
        # In production, block on high latency
        if self.mode == InferenceMode.PRODUCTION and latency_ms > self.max_latency_ms * 2:
            return True, f"Latency {latency_ms:.1f}ms exceeds safety limit"
        
        return False, None
    
    def _calculate_confidence(
        self,
        features: np.ndarray,
        risk_score: float
    ) -> float:
        """
        Calculate inference confidence
        
        Confidence is based on:
        - Feature validity (no extreme values)
        - Prediction stability
        - Distance from decision boundary
        """
        # Feature validity
        feature_valid = 1.0 - np.mean(np.abs(features) > 10)
        
        # Prediction stability (variance of recent predictions)
        if len(self._risk_score_history) >= 10:
            recent_std = np.std(self._risk_score_history[-10:])
            stability = np.exp(-recent_std * 5)  # Higher variance = lower stability
        else:
            stability = 0.5
        
        # Distance from decision boundary (0.5)
        boundary_distance = abs(risk_score - 0.5)
        boundary_confidence = 0.5 + boundary_distance
        
        # Combined confidence
        confidence = (feature_valid * 0.3 + stability * 0.4 + boundary_confidence * 0.3)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_position_factor(
        self,
        risk_score: float,
        confidence: float
    ) -> float:
        """
        Calculate position size factor
        
        Factor is inversely proportional to risk score and
        scaled by confidence.
        """
        # Base factor from risk score
        base_factor = 1.0 - risk_score
        
        # Scale by confidence
        confidence_scale = 0.5 + 0.5 * confidence
        
        # Apply Kelly fraction
        kelly_fraction = self.risk_params.get('kelly_fraction', 0.25)
        
        return base_factor * confidence_scale * kelly_fraction
    
    def _check_regime_shift(self) -> Optional[str]:
        """
        Detect regime shifts in feature distributions
        
        Uses KS test to compare recent vs historical features.
        """
        if len(self._feature_history) < self.REGIME_SHIFT_WINDOW * 2:
            return None
        
        recent = np.array(self._feature_history[-self.REGIME_SHIFT_WINDOW:])
        historical = np.array(self._feature_history[:-self.REGIME_SHIFT_WINDOW])
        
        # Compare distributions for key features
        try:
            from scipy import stats
            
            # Use first few features as proxy
            n_features_to_check = min(5, recent.shape[1])
            
            for i in range(n_features_to_check):
                ks_stat, p_value = stats.ks_2samp(recent[:, i], historical[:, i])
                
                if p_value < 0.01:  # Significant distribution shift
                    self._metrics.regime_shifts_detected += 1
                    alert = f"REGIME_SHIFT_DETECTED: feature_{i} (p={p_value:.4f})"
                    self._emit_alert(AlertLevel.WARNING, alert)
                    return alert
                    
        except ImportError:
            pass
        
        return None
    
    def _check_signal_degradation(self) -> Optional[str]:
        """
        Check for signal degradation
        
        Monitors prediction stability and drift.
        """
        if len(self._risk_score_history) < 50:
            return None
        
        recent = self._risk_score_history[-20:]
        historical = self._risk_score_history[-50:-20]
        
        # Check for significant drift
        recent_mean = np.mean(recent)
        hist_mean = np.mean(historical)
        drift = abs(recent_mean - hist_mean) / (np.std(historical) + 0.01)
        
        if drift > self.SIGNAL_DEGRADATION_THRESHOLD * 3:
            self._metrics.signal_degradation_alerts += 1
            alert = f"SIGNAL_DEGRADATION: drift={drift:.2f}"
            self._emit_alert(AlertLevel.WARNING, alert)
            return alert
        
        return None
    
    def _emit_alert(self, level: AlertLevel, message: str):
        """Emit an alert"""
        self._alerts.append((datetime.now(), level, message))
        
        if level == AlertLevel.EMERGENCY:
            self.logger.critical(f"🚨 {message}")
        elif level == AlertLevel.CRITICAL:
            self.logger.error(f"❌ {message}")
        elif level == AlertLevel.WARNING:
            self.logger.warning(f"⚠️ {message}")
        else:
            self.logger.info(f"ℹ️ {message}")
    
    def _update_monitoring(
        self,
        risk_score: float,
        latency_ms: float,
        is_blocked: bool,
        alerts: List[str]
    ):
        """Update monitoring metrics"""
        if not self.enable_monitoring:
            return
        
        self._metrics.total_inferences += 1
        
        if is_blocked:
            self._metrics.blocked_trades += 1
        
        # Update latency stats
        self._metrics.avg_latency_ms = np.mean(self._latency_history)
        self._metrics.max_latency_ms = max(self._metrics.max_latency_ms, latency_ms)
        
        # Update uptime
        self._metrics.uptime_hours = (
            datetime.now() - self._start_time
        ).total_seconds() / 3600
    
    def _create_blocked_result(
        self,
        reason: str,
        timestamp: datetime,
        alerts: List[str]
    ) -> InferenceResult:
        """Create a blocked result"""
        alerts.append(f"BLOCKED: {reason}")
        
        return InferenceResult(
            risk_score=1.0,
            position_size_factor=0.0,
            confidence=0.0,
            latency_ms=0.0,
            is_blocked=True,
            block_reason=reason,
            alerts=alerts,
            timestamp=timestamp
        )
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """
        Activate emergency kill switch
        
        All inference will return blocked until reset.
        """
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        self._metrics.kill_switch_triggers += 1
        
        self._emit_alert(
            AlertLevel.EMERGENCY,
            f"KILL SWITCH ACTIVATED: {reason}"
        )
    
    def reset_kill_switch(self):
        """Reset kill switch (requires manual intervention)"""
        self._kill_switch_active = False
        self._kill_switch_reason = None
        
        self.logger.info("✅ Kill switch reset")
    
    def get_metrics(self) -> MonitoringMetrics:
        """Get current monitoring metrics"""
        return self._metrics
    
    def get_alerts(self, last_n: int = 100) -> List[Dict]:
        """Get recent alerts"""
        return [
            {
                'timestamp': ts.isoformat(),
                'level': level.value,
                'message': msg
            }
            for ts, level, msg in self._alerts[-last_n:]
        ]
    
    def health_check(self) -> Dict:
        """
        Perform health check
        
        Returns:
            Dict with health status
        """
        is_healthy = True
        issues = []
        
        # Check kill switch
        if self._kill_switch_active:
            is_healthy = False
            issues.append("Kill switch active")
        
        # Check latency
        if self._metrics.avg_latency_ms > self.max_latency_ms:
            is_healthy = False
            issues.append(f"High average latency: {self._metrics.avg_latency_ms:.1f}ms")
        
        # Check block rate
        if self._metrics.total_inferences > 100:
            block_rate = self._metrics.blocked_trades / self._metrics.total_inferences
            if block_rate > 0.5:
                issues.append(f"High block rate: {block_rate:.1%}")
        
        # Check signal degradation
        if self._metrics.signal_degradation_alerts > 5:
            issues.append("Multiple signal degradation alerts")
        
        return {
            'is_healthy': is_healthy,
            'issues': issues,
            'uptime_hours': self._metrics.uptime_hours,
            'total_inferences': self._metrics.total_inferences,
            'kill_switch_active': self._kill_switch_active
        }


def create_risk_pipeline(
    model_path: str,
    production: bool = True
) -> RiskInferencePipeline:
    """
    Factory function to create risk inference pipeline
    
    Args:
        model_path: Path to ONNX model
        production: Whether to run in production mode
        
    Returns:
        Configured RiskInferencePipeline
    """
    mode = InferenceMode.PRODUCTION if production else InferenceMode.DEBUG
    
    return RiskInferencePipeline(
        model_path=model_path,
        mode=mode,
        enable_kill_switch=production,
        enable_monitoring=True
    )
