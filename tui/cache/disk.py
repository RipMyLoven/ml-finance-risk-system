"""
cache/disk.py — Parquet-backed persistent cache for stale-data fallback.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path.home() / ".ml_finance" / "cache"


class DiskCache:
    """Parquet file cache — used as cold-start fallback."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._dir = cache_dir or _DEFAULT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self._dir / f"{symbol}_{interval}_{today}.parquet"

    def get(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        p = self._path(symbol, interval)
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception as exc:
                logger.warning("Disk cache read failed for %s: %s", p, exc)
        return None

    def put(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        p = self._path(symbol, interval)
        try:
            df.to_parquet(p, engine="pyarrow")
        except Exception as exc:
            logger.warning("Disk cache write failed for %s: %s", p, exc)
