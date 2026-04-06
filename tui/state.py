"""
state.py — Reactive state store for the TUI.

The orchestrator writes here; the TUI panels read from here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .model_registry.registry import PredictionResult


@dataclass
class HistoryEntry:
    timestamp: str
    symbol: str
    meta_signal: str
    meta_score: float
    confidence: float
    risk_level: str
    result: PredictionResult


@dataclass
class AppState:
    """Mutable state store — all TUI panels observe this."""

    # Current symbol
    symbol: str = "BTCUSDT"
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])

    # Display timeframe (for the header)
    display_timeframe: str = "1h"
    timeframe_options: List[str] = field(
        default_factory=lambda: ["5m", "1h", "4h", "1d"]
    )

    # Status
    status: str = "STARTING"     # LIVE | STALE | ERROR | STARTING | COMPUTING
    error_count: int = 0
    last_error: str = ""

    # Latest prediction result
    result: Optional[PredictionResult] = None
    last_update_time: float = 0.0
    last_inference_ms: float = 0.0

    # Cache info
    last_cache_hit: bool = False

    # History
    history: List[HistoryEntry] = field(default_factory=list)

    # Model detail view: 0 = all, 1/2/3 = isolate
    model_view: int = 0

    # Pause mode
    paused: bool = False

    # Feature values for inspection
    feature_values: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Next refresh countdown
    next_refresh_seconds: float = 0.0

    def add_history(self, result: PredictionResult) -> None:
        conf = 0.0
        if result.intraday:
            conf = result.intraday.confidence
        elif result.scalp:
            conf = result.scalp.confidence

        risk_level = result.risk.risk_level if result.risk else "N/A"

        entry = HistoryEntry(
            timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            symbol=result.symbol,
            meta_signal=result.meta_signal,
            meta_score=result.meta_score,
            confidence=conf,
            risk_level=risk_level,
            result=result,
        )
        self.history.insert(0, entry)
        # Keep last 100
        if len(self.history) > 100:
            self.history = self.history[:100]

    @property
    def data_age_seconds(self) -> float:
        if self.last_update_time == 0:
            return 0.0
        return time.monotonic() - self.last_update_time
