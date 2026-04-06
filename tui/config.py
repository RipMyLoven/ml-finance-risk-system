"""
config.py — Central configuration for the TUI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

_HERE = Path(__file__).resolve().parent


@dataclass
class TimeframeConfig:
    interval: str
    limit: int


@dataclass
class AppConfig:
    mexc_base_url: str = "https://api.mexc.com/api/v3"
    mexc_timeout: int = 15
    mexc_max_retries: int = 3
    mexc_retry_delay: float = 2.0

    models_dir: Path = field(default_factory=lambda: _HERE.parent / "project" / "models")

    timeframes: Dict[str, TimeframeConfig] = field(default_factory=lambda: {
        "scalp": TimeframeConfig(interval="5m", limit=350),
        "intraday": TimeframeConfig(interval="60m", limit=350),
        "swing": TimeframeConfig(interval="1d", limit=450),
    })

    # Cache TTLs (seconds) per interval
    cache_ttls: Dict[str, float] = field(default_factory=lambda: {
        "5m": 30.0,
        "60m": 120.0,
        "1d": 600.0,
    })

    # Refresh intervals (seconds) per model type
    refresh_intervals: Dict[str, float] = field(default_factory=lambda: {
        "5m": 60.0,
        "60m": 300.0,
        "1d": 900.0,
    })

    meta_weights: Dict[str, float] = field(default_factory=lambda: {
        "scalp": 0.5,
        "intraday": 0.3,
        "swing": 0.2,
    })

    entry_threshold: float = 0.35
    max_risk_score: float = 0.7

    @classmethod
    def load(cls, config_path: Path | None = None) -> "AppConfig":
        if config_path and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            cfg = cls()
            if "mexc" in data:
                m = data["mexc"]
                cfg.mexc_base_url = m.get("base_url", cfg.mexc_base_url)
                cfg.mexc_timeout = m.get("timeout_seconds", cfg.mexc_timeout)
            if "models_dir" in data:
                candidate = Path(data["models_dir"])
                if not candidate.is_absolute():
                    candidate = (_HERE / candidate).resolve()
                cfg.models_dir = candidate
            return cfg
        return cls()
