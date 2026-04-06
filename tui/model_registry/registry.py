"""
model_registry/registry.py — ONNX model loading, validation, and inference.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

try:
    import onnx as _onnx_lib
    _ONNX_LIB_AVAILABLE = True
except ImportError:
    _ONNX_LIB_AVAILABLE = False

_MODEL_FILES: Dict[str, Tuple[str, str]] = {
    "scalp":    ("scalp_lgbm.onnx",    "scalp_lgbm_features.json"),
    "intraday": ("intraday_lgbm.onnx", "intraday_lgbm_features.json"),
    "swing":    ("swing_lgbm.onnx",    "swing_lgbm_features.json"),
    "risk":     ("risk_lgbm.onnx",     "risk_lgbm_features.json"),
}

_CLASS_LABELS: Dict[int, str] = {0: "DOWN", 1: "FLAT", 2: "UP"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradingPrediction:
    model_name: str
    predicted_class: str
    P_up: float
    P_flat: float
    P_down: float
    confidence: float
    expected_return: float


@dataclass
class RiskPrediction:
    risk_score: float
    risk_level: str
    max_leverage_suggested: float


@dataclass
class PredictionResult:
    symbol: str
    scalp: Optional[TradingPrediction] = None
    intraday: Optional[TradingPrediction] = None
    swing: Optional[TradingPrediction] = None
    risk: Optional[RiskPrediction] = None
    meta_signal: str = "NEUTRAL"
    meta_score: float = 0.0
    errors: List[str] = field(default_factory=list)
    elapsed: float = 0.0

    def to_dict(self) -> dict:
        def _t(p: Optional[TradingPrediction]) -> Optional[dict]:
            if p is None:
                return None
            return {
                "signal": p.predicted_class,
                "confidence": round(p.confidence, 4),
                "P_up": round(p.P_up, 4),
                "P_flat": round(p.P_flat, 4),
                "P_down": round(p.P_down, 4),
                "expected_return": round(p.expected_return, 4),
            }

        def _r(r: Optional[RiskPrediction]) -> Optional[dict]:
            if r is None:
                return None
            return {
                "risk_score": round(r.risk_score, 4),
                "risk_level": r.risk_level,
                "max_leverage": r.max_leverage_suggested,
            }

        return {
            "symbol": self.symbol,
            "meta_signal": self.meta_signal,
            "meta_score": round(self.meta_score, 4),
            "scalp": _t(self.scalp),
            "intraday": _t(self.intraday),
            "swing": _t(self.swing),
            "risk": _r(self.risk),
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Single ONNX model wrapper
# ---------------------------------------------------------------------------

class ONNXModel:
    def __init__(self, onnx_path: Path, features_json: Path, num_threads: int = 2) -> None:
        if not _ORT_AVAILABLE:
            raise ImportError("onnxruntime is not installed.")

        self.onnx_path = Path(onnx_path)
        self.features_json = Path(features_json)

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not self.features_json.exists():
            raise FileNotFoundError(f"Feature list not found: {self.features_json}")

        with open(self.features_json, "r", encoding="utf-8") as fh:
            self.feature_names: List[str] = json.load(fh)
        self.n_features: int = len(self.feature_names)
        self.model_type: str = self._read_model_type()
        self.is_risk: bool = (self.model_type == "risk")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._output_names: List[str] = [o.name for o in self._session.get_outputs()]

    def predict_single(self, feature_vector: np.ndarray) -> dict:
        X = np.asarray(feature_vector, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model '{self.model_type}' expects {self.n_features} features, got {X.shape[1]}"
            )
        outputs = self._session.run(None, {self._input_name: X})
        return self._parse_outputs(outputs)

    def _parse_outputs(self, outputs: list) -> dict:
        if self.is_risk:
            pred = np.array(outputs[0]).flatten()
            return {"risk_score": pred.tolist()}
        return self._parse_trading(outputs)

    @staticmethod
    def _parse_trading(outputs: list) -> dict:
        if len(outputs) >= 2:
            labels_raw = outputs[0]
            proba_raw = outputs[1]
            if isinstance(proba_raw, list):
                proba = np.array([
                    [float(row.get(0, 0.0)), float(row.get(1, 0.0)), float(row.get(2, 0.0))]
                    for row in proba_raw
                ])
            else:
                proba = np.array(proba_raw)
        else:
            proba = np.array(outputs[0])
            labels_raw = np.argmax(proba, axis=1)

        labels = np.array(labels_raw).flatten().astype(int)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        if proba.shape[1] < 3:
            pad = np.zeros((proba.shape[0], 3 - proba.shape[1]))
            proba = np.hstack([proba, pad])

        return {
            "predicted_class": labels,
            "P_down": proba[:, 0].tolist(),
            "P_flat": proba[:, 1].tolist(),
            "P_up": proba[:, 2].tolist(),
            "expected_return": (proba[:, 2] - proba[:, 0]).tolist(),
        }

    def _read_model_type(self) -> str:
        if _ONNX_LIB_AVAILABLE:
            try:
                model = _onnx_lib.load(str(self.onnx_path))
                for prop in model.metadata_props:
                    if prop.key == "model_type":
                        return prop.value
            except Exception:
                pass
        stem = self.onnx_path.stem.lower()
        for t in ("scalp", "intraday", "swing", "risk"):
            if t in stem:
                return t
        return "unknown"


# ---------------------------------------------------------------------------
# Model Ensemble
# ---------------------------------------------------------------------------

class ModelEnsemble:
    _WEIGHTS: Dict[str, float] = {"scalp": 0.5, "intraday": 0.3, "swing": 0.2}

    def __init__(self, models_dir: str | Path, num_threads: int = 2) -> None:
        self._dir = Path(models_dir)
        self._num_threads = num_threads
        self._models: Dict[str, ONNXModel] = {}
        self._loaded = False

    def load(self) -> "ModelEnsemble":
        errors = []
        for name, (onnx_file, json_file) in _MODEL_FILES.items():
            onnx_path = self._dir / onnx_file
            json_path = self._dir / json_file
            try:
                self._models[name] = ONNXModel(onnx_path, json_path, self._num_threads)
                logger.info(
                    "[MODEL] Loaded %-8s | features=%d | type=%s",
                    name, self._models[name].n_features, self._models[name].model_type,
                )
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                logger.error("Failed to load model '%s': %s", name, exc)

        if not self._models:
            raise RuntimeError(f"No models loaded from {self._dir}. Errors: {errors}")

        self._loaded = True
        return self

    @property
    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def predict(
        self,
        symbol: str,
        scalp_vec: Optional[np.ndarray] = None,
        intraday_vec: Optional[np.ndarray] = None,
        swing_vec: Optional[np.ndarray] = None,
        risk_vec: Optional[np.ndarray] = None,
    ) -> PredictionResult:
        if not self._loaded:
            self.load()

        result = PredictionResult(symbol=symbol)

        vectors = {"scalp": scalp_vec, "intraday": intraday_vec, "swing": swing_vec}
        for name, vec in vectors.items():
            if vec is None:
                continue
            model = self._models.get(name)
            if model is None:
                result.errors.append(f"Model '{name}' not loaded")
                continue
            try:
                out = model.predict_single(vec)
                result.__dict__[name] = self._build_trading_prediction(name, out)
            except Exception as exc:
                result.errors.append(f"Inference error for '{name}': {exc}")

        # Risk model
        if risk_vec is not None:
            risk_model = self._models.get("risk")
            if risk_model is not None:
                try:
                    risk_out = risk_model.predict_single(risk_vec)
                    result.risk = self._build_risk_prediction(risk_out)
                except Exception as exc:
                    result.errors.append(f"Inference error for 'risk': {exc}")

        result.meta_signal, result.meta_score = self._compute_meta(
            result.scalp, result.intraday, result.swing,
        )
        return result

    def get_intraday_scalars(self, intraday_vec: np.ndarray) -> Optional[dict]:
        model = self._models.get("intraday")
        if model is None:
            return None
        try:
            out = model.predict_single(intraday_vec)
            return {
                "P_up": float(out["P_up"][0]),
                "P_down": float(out["P_down"][0]),
                "confidence": max(
                    float(out["P_up"][0]),
                    float(out["P_flat"][0]),
                    float(out["P_down"][0]),
                ),
            }
        except Exception:
            return None

    @staticmethod
    def _build_trading_prediction(name: str, out: dict) -> TradingPrediction:
        p_up = float(out["P_up"][0])
        p_flat = float(out["P_flat"][0])
        p_down = float(out["P_down"][0])
        label = int(out["predicted_class"][0])
        return TradingPrediction(
            model_name=name,
            predicted_class=_CLASS_LABELS.get(label, "FLAT"),
            P_up=p_up, P_flat=p_flat, P_down=p_down,
            confidence=max(p_up, p_flat, p_down),
            expected_return=float(out["expected_return"][0]),
        )

    @staticmethod
    def _build_risk_prediction(out: dict) -> RiskPrediction:
        raw = float(out["risk_score"][0])
        score = float(np.clip(raw, 0.0, 1.0))
        if score < 0.25:
            level, max_lev = "LOW", 10.0
        elif score < 0.50:
            level, max_lev = "MEDIUM", 5.0
        elif score < 0.75:
            level, max_lev = "HIGH", 2.0
        else:
            level, max_lev = "CRITICAL", 1.0
        return RiskPrediction(risk_score=score, risk_level=level, max_leverage_suggested=max_lev)

    def _compute_meta(
        self,
        scalp: Optional[TradingPrediction],
        intraday: Optional[TradingPrediction],
        swing: Optional[TradingPrediction],
    ) -> Tuple[str, float]:
        score = 0.0
        total_w = 0.0
        for pred, weight in (
            (scalp, self._WEIGHTS["scalp"]),
            (intraday, self._WEIGHTS["intraday"]),
            (swing, self._WEIGHTS["swing"]),
        ):
            if pred is not None:
                score += weight * (pred.P_up - pred.P_down)
                total_w += weight
        if total_w > 0:
            score /= total_w
        if score > 0.15:
            signal = "LONG"
        elif score < -0.15:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"
        return signal, score
