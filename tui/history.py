"""
history.py — Append-only prediction history with JSON-L persistence.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .model_registry.registry import PredictionResult

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".ml_finance" / "history"


class PredictionHistory:
    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self._dir = output_dir or _DEFAULT_PATH
        self._dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        self._file = self._dir / f"predictions_{today}.jsonl"

    def append(self, result: PredictionResult) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result.to_dict(),
        }
        try:
            with open(self._file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("Failed to write history: %s", exc)
