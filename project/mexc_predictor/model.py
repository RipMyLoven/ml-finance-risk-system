"""
model.py — ONNX model loading and inference.

Wraps onnxruntime sessions for the four trading/risk models and provides a
:class:`ModelEnsemble` that runs all of them together and returns a single
structured :class:`PredictionResult`.

Design notes
------------
* CPU-only execution (deterministic, reproducible).
* Lazy loading: each model is loaded on first use.
* Session is created once and reused across predictions.
* Input is validated against the expected feature count from the JSON
  metadata file stored alongside each ``.onnx`` file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try-import optional libraries
# ---------------------------------------------------------------------------

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ORT_AVAILABLE = False

try:
    import onnx as _onnx_lib

    _ONNX_LIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ONNX_LIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_FILES: Dict[str, Tuple[str, str]] = {
    "scalp":    ("scalp_lgbm.onnx",    "scalp_lgbm_features.json"),
    "intraday": ("intraday_lgbm.onnx", "intraday_lgbm_features.json"),
    "swing":    ("swing_lgbm.onnx",    "swing_lgbm_features.json"),
    "risk":     ("risk_lgbm.onnx",     "risk_lgbm_features.json"),
}

# Class labels used by trading models (LightGBM multiclass).
_CLASS_LABELS: Dict[int, str] = {0: "DOWN", 1: "FLAT", 2: "UP"}


# ---------------------------------------------------------------------------
# Typed output
# ---------------------------------------------------------------------------

@dataclass
class TradingPrediction:
    """Output of a single trading model (scalp / intraday / swing)."""
    model_name: str
    predicted_class: str          # "UP" | "FLAT" | "DOWN"
    P_up: float
    P_flat: float
    P_down: float
    confidence: float             # max(P_up, P_flat, P_down)
    expected_return: float        # P_up - P_down

    def is_bullish(self, threshold: float = 0.45) -> bool:
        return self.P_up >= threshold

    def is_bearish(self, threshold: float = 0.45) -> bool:
        return self.P_down >= threshold


@dataclass
class RiskPrediction:
    """Output of the risk model."""
    risk_score: float             # Raw regression output (higher = riskier).
    risk_level: str               # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    max_leverage_suggested: float


@dataclass
class PredictionResult:
    """Aggregated output returned to the caller."""
    symbol: str
    scalp: Optional[TradingPrediction] = None
    intraday: Optional[TradingPrediction] = None
    swing: Optional[TradingPrediction] = None
    risk: Optional[RiskPrediction] = None
    meta_signal: str = "NEUTRAL"       # Weighted ensemble signal.
    meta_score: float = 0.0            # Directional score [-1, 1].
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "meta_signal": self.meta_signal,
            "meta_score": round(self.meta_score, 4),
            "scalp": self._trading_to_dict(self.scalp),
            "intraday": self._trading_to_dict(self.intraday),
            "swing": self._trading_to_dict(self.swing),
            "risk": self._risk_to_dict(self.risk),
            "errors": self.errors,
        }

    @staticmethod
    def _trading_to_dict(p: Optional[TradingPrediction]) -> Optional[dict]:
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

    @staticmethod
    def _risk_to_dict(r: Optional[RiskPrediction]) -> Optional[dict]:
        if r is None:
            return None
        return {
            "risk_score": round(r.risk_score, 4),
            "risk_level": r.risk_level,
            "max_leverage": r.max_leverage_suggested,
        }


# ---------------------------------------------------------------------------
# Single ONNX model wrapper
# ---------------------------------------------------------------------------

class ONNXModel:
    """
    Thin wrapper around a single ``onnxruntime.InferenceSession``.

    Args:
        onnx_path:    Path to the ``.onnx`` file.
        features_json: Path to the accompanying ``*_features.json`` file.
        num_threads:   CPU thread count for the ORT session (default = 2).
    """

    def __init__(
        self,
        onnx_path: Path,
        features_json: Path,
        num_threads: int = 2,
    ) -> None:
        if not _ORT_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. "
                "Run: pip install onnxruntime"
            )

        self.onnx_path = Path(onnx_path)
        self.features_json = Path(features_json)

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not self.features_json.exists():
            raise FileNotFoundError(
                f"Feature list not found: {self.features_json}"
            )

        # Load expected feature names from the JSON sidecar.
        with open(self.features_json, "r", encoding="utf-8") as fh:
            self.feature_names: List[str] = json.load(fh)
        self.n_features: int = len(self.feature_names)

        # Read model_type from ONNX metadata (used to pick output parsing).
        self.model_type: str = self._read_model_type()
        self.is_risk: bool = (self.model_type == "risk")

        # Build ORT session (CPU only, deterministic).
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self._session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self._input_name: str = self._session.get_inputs()[0].name
        self._output_names: List[str] = [
            o.name for o in self._session.get_outputs()
        ]

        logger.debug(
            "Loaded %s | type=%s | features=%d | outputs=%s",
            self.onnx_path.name,
            self.model_type,
            self.n_features,
            self._output_names,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_single(self, feature_vector: np.ndarray) -> dict:
        """
        Run inference on a single feature vector.

        Args:
            feature_vector: 1-D numpy array of shape ``(n_features,)``
                            or 2-D of shape ``(1, n_features)``.

        Returns:
            Dict with keys depending on model type:
            - Trading: ``P_up``, ``P_flat``, ``P_down``, ``predicted_class``,
                       ``expected_return``.
            - Risk:    ``risk_score``.
        """
        X = self._prepare_input(feature_vector)
        outputs = self._session.run(None, {self._input_name: X})
        return self._parse_outputs(outputs)

    def predict_batch(self, feature_matrix: np.ndarray) -> dict:
        """
        Run inference on a batch of feature vectors.

        Args:
            feature_matrix: 2-D array of shape ``(n_samples, n_features)``.

        Returns:
            Dict with array values (one element per sample).
        """
        X = feature_matrix.astype(np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._validate_shape(X)
        outputs = self._session.run(None, {self._input_name: X})
        return self._parse_outputs(outputs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_input(self, vec: np.ndarray) -> np.ndarray:
        X = np.asarray(vec, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._validate_shape(X)
        return X

    def _validate_shape(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Model '{self.model_type}' expects {self.n_features} features, "
                f"got {X.shape[1]}."
            )

    def _parse_outputs(self, outputs: list) -> dict:
        """Dispatch to trading or risk parser."""
        if self.is_risk:
            return self._parse_risk(outputs)
        return self._parse_trading(outputs)

    @staticmethod
    def _parse_trading(outputs: list) -> dict:
        """
        LightGBM multiclass ONNX output format:
            outputs[0] = label array
            outputs[1] = list[dict] or ndarray of probabilities
        """
        if len(outputs) >= 2:
            labels_raw = outputs[0]
            proba_raw = outputs[1]

            # onnxmltools may return a list of dicts {class_id: prob}.
            if isinstance(proba_raw, list):
                proba = np.array(
                    [
                        [
                            float(row.get(0, 0.0)),
                            float(row.get(1, 0.0)),
                            float(row.get(2, 0.0)),
                        ]
                        for row in proba_raw
                    ]
                )
            else:
                proba = np.array(proba_raw)
        else:
            proba = np.array(outputs[0])
            labels_raw = np.argmax(proba, axis=1)

        labels = np.array(labels_raw).flatten().astype(int)

        # Guard: ensure 3 probability columns.
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        if proba.shape[1] < 3:
            pad = np.zeros((proba.shape[0], 3 - proba.shape[1]))
            proba = np.hstack([proba, pad])

        return {
            "predicted_class": labels,
            "P_down":          proba[:, 0].tolist(),
            "P_flat":          proba[:, 1].tolist(),
            "P_up":            proba[:, 2].tolist(),
            "expected_return": (proba[:, 2] - proba[:, 0]).tolist(),
        }

    @staticmethod
    def _parse_risk(outputs: list) -> dict:
        """Risk model returns a single regression value."""
        pred = np.array(outputs[0]).flatten()
        return {"risk_score": pred.tolist()}

    def _read_model_type(self) -> str:
        if not _ONNX_LIB_AVAILABLE:
            # Fall back to filename heuristic.
            stem = self.onnx_path.stem.lower()
            for t in ("scalp", "intraday", "swing", "risk"):
                if t in stem:
                    return t
            return "unknown"
        model = _onnx_lib.load(str(self.onnx_path))
        for prop in model.metadata_props:
            if prop.key == "model_type":
                return prop.value
        # Fallback.
        stem = self.onnx_path.stem.lower()
        for t in ("scalp", "intraday", "swing", "risk"):
            if t in stem:
                return t
        return "unknown"

    def get_info(self) -> dict:
        """Return a summary dict (useful for debugging)."""
        return {
            "path": str(self.onnx_path),
            "model_type": self.model_type,
            "is_risk": self.is_risk,
            "n_features": self.n_features,
            "input_name": self._input_name,
            "output_names": self._output_names,
        }


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class ModelEnsemble:
    """
    Loads all four ONNX models and runs them as an ensemble.

    The risk model is run last because it requires the trading model outputs
    as input features.

    Meta-signal weights (matching the Meta Engine from architecture/training):
      scalp = 0.5, intraday = 0.3, swing = 0.2

    Args:
        models_dir:   Directory containing ``.onnx`` and ``*_features.json``.
        num_threads:  ORT thread count per model.
    """

    _WEIGHTS: Dict[str, float] = {
        "scalp": 0.5,
        "intraday": 0.3,
        "swing": 0.2,
    }

    def __init__(
        self,
        models_dir: str | Path,
        num_threads: int = 2,
    ) -> None:
        self._dir = Path(models_dir)
        self._num_threads = num_threads
        self._models: Dict[str, ONNXModel] = {}
        self._loaded = False

    def load(self) -> "ModelEnsemble":
        """
        Load all models from disk.  Call this once before predict().

        Returns self for method chaining.
        """
        errors = []
        for name, (onnx_file, json_file) in _MODEL_FILES.items():
            onnx_path = self._dir / onnx_file
            json_path = self._dir / json_file
            try:
                self._models[name] = ONNXModel(
                    onnx_path, json_path, num_threads=self._num_threads
                )
                logger.info(
                    "Loaded model '%s' (%d features)",
                    name,
                    self._models[name].n_features,
                )
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                logger.error("Failed to load model '%s': %s", name, exc)

        if errors:
            logger.warning(
                "%d model(s) failed to load: %s", len(errors), errors
            )

        self._loaded = True
        return self

    def predict(
        self,
        symbol: str,
        scalp_vec: Optional[np.ndarray] = None,
        intraday_vec: Optional[np.ndarray] = None,
        swing_vec: Optional[np.ndarray] = None,
        risk_vec: Optional[np.ndarray] = None,
        intraday_preds: Optional[Dict[str, float]] = None,
    ) -> PredictionResult:
        """
        Run inference for all available models and return a
        :class:`PredictionResult`.

        Args:
            symbol:         Trading pair string (for labelling only).
            scalp_vec:      Feature vector for the scalp model.
            intraday_vec:   Feature vector for the intraday model.
            swing_vec:      Feature vector for the swing model.
            risk_vec:       Feature vector for the risk model.
            intraday_preds: Trading model scalar outputs (P_up, P_down,
                            confidence) to be embedded in risk features.
                            Normally provided by the caller after the
                            intraday model runs.

        Returns:
            :class:`PredictionResult` aggregating all outputs.
        """
        if not self._loaded:
            self.load()

        result = PredictionResult(symbol=symbol)

        # --- Trading models ---
        vectors = {
            "scalp": scalp_vec,
            "intraday": intraday_vec,
            "swing": swing_vec,
        }

        trading_outputs: Dict[str, dict] = {}
        for name, vec in vectors.items():
            if vec is None:
                continue
            model = self._models.get(name)
            if model is None:
                result.errors.append(f"Model '{name}' not loaded.")
                continue
            try:
                out = model.predict_single(vec)
                trading_outputs[name] = out
                pred = self._build_trading_prediction(name, out)
                setattr(result, name, pred)
            except Exception as exc:
                msg = f"Inference error for '{name}': {exc}"
                result.errors.append(msg)
                logger.error(msg)

        # --- Risk model ---
        if risk_vec is not None:
            risk_model = self._models.get("risk")
            if risk_model is not None:
                try:
                    risk_out = risk_model.predict_single(risk_vec)
                    result.risk = self._build_risk_prediction(risk_out)
                except Exception as exc:
                    msg = f"Inference error for 'risk': {exc}"
                    result.errors.append(msg)
                    logger.error(msg)
            else:
                result.errors.append("Model 'risk' not loaded.")

        # --- Meta signal ---
        result.meta_signal, result.meta_score = self._compute_meta(
            result.scalp, result.intraday, result.swing
        )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trading_prediction(
        name: str, out: dict
    ) -> TradingPrediction:
        p_up = float(out["P_up"][0])
        p_flat = float(out["P_flat"][0])
        p_down = float(out["P_down"][0])
        label = int(out["predicted_class"][0])
        return TradingPrediction(
            model_name=name,
            predicted_class=_CLASS_LABELS.get(label, "FLAT"),
            P_up=p_up,
            P_flat=p_flat,
            P_down=p_down,
            confidence=max(p_up, p_flat, p_down),
            expected_return=float(out["expected_return"][0]),
        )

    @staticmethod
    def _build_risk_prediction(out: dict) -> RiskPrediction:
        raw = float(out["risk_score"][0])
        # Clamp to [0, 1] — model was trained with 0-1 targets.
        score = float(np.clip(raw, 0.0, 1.0))

        if score < 0.25:
            level, max_lev = "LOW", 10.0
        elif score < 0.50:
            level, max_lev = "MEDIUM", 5.0
        elif score < 0.75:
            level, max_lev = "HIGH", 2.0
        else:
            level, max_lev = "CRITICAL", 1.0

        return RiskPrediction(
            risk_score=score,
            risk_level=level,
            max_leverage_suggested=max_lev,
        )

    def _compute_meta(
        self,
        scalp: Optional[TradingPrediction],
        intraday: Optional[TradingPrediction],
        swing: Optional[TradingPrediction],
    ) -> Tuple[str, float]:
        """
        Weighted directional score using project architecture weights:
          score = Σ weight_i * (P_up_i - P_down_i).

        Thresholds:
          score > +0.15  → LONG
          score < -0.15  → SHORT
          otherwise      → NEUTRAL
        """
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
            score /= total_w  # Normalise so score stays in [-1, 1].

        if score > 0.15:
            signal = "LONG"
        elif score < -0.15:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"

        return signal, score

    def get_info(self) -> dict:
        """Return a summary of all loaded models."""
        return {name: m.get_info() for name, m in self._models.items()}
