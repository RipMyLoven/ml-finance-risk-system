"""
data.py — MEXC Public API client.

Fetches OHLCV candlestick data from the MEXC spot exchange using the
public REST endpoint (no API key required).

Usage:
    from data import MexcClient
    client = MexcClient()
    df = client.fetch_ohlcv("BTCUSDT", interval="60m", limit=300)
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.mexc.com/api/v3"

# MEXC spot klines column order.
# As of 2025 the endpoint returns 8 fields per candle
# (taker columns and trade count were removed from the public API).
_KLINE_COLS_8 = [
    "open_time", "open", "high", "low", "close",
    "volume", "close_time", "quote_volume",
]
# Older / alternative MEXC versions may still return 12 fields.
_KLINE_COLS_12 = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "_ignore",
]

# MEXC valid interval strings.
VALID_INTERVALS: Tuple[str, ...] = (
    "1m", "5m", "15m", "30m", "60m", "4h", "1d", "1W", "1M",
)

# Human-readable aliases → MEXC format.
INTERVAL_ALIASES: Dict[str, str] = {
    "1h": "60m",
    "4h": "4h",
    "1d": "1d",
    "1D": "1d",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1m": "1m",
    "60m": "60m",
}

# Max candles per single request (MEXC hard limit).
_MAX_LIMIT: int = 1000


# ---------------------------------------------------------------------------
# In-memory cache entry
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    df: pd.DataFrame
    timestamp: float
    ttl: float

    def is_valid(self) -> bool:
        return (time.monotonic() - self.timestamp) < self.ttl


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class MexcClient:
    """
    Thin wrapper around the MEXC public REST API.

    Thread-safe: uses a per-instance lock for the in-memory cache.

    Args:
        base_url:         MEXC API base URL.
        timeout:          HTTP request timeout in seconds.
        max_retries:      Number of retry attempts on transient failures.
        retry_delay:      Seconds to wait between retries.
        cache_ttl:        Cache lifetime in seconds (0 = disabled).
    """

    def __init__(
        self,
        base_url: str = _BASE_URL,
        timeout: int = 15,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        cache_ttl: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._cache_ttl = cache_ttl

        self._cache: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "60m",
        limit: int = 300,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data for *symbol*.

        Args:
            symbol:     Trading pair, e.g. ``"BTCUSDT"``.
            interval:   Candlestick interval. Accepts MEXC native values
                        (``"60m"``, ``"1d"`` …) and common aliases (``"1h"``).
            limit:      Number of candles to fetch (max 1000).
            use_cache:  Return cached data if still valid.

        Returns:
            DataFrame with columns:
            open_time (datetime, UTC index), open, high, low, close,
            volume, quote_volume, trades, taker_buy_base, taker_buy_quote.

        Raises:
            ValueError:  Unknown symbol or invalid interval.
            RuntimeError: API unreachable after all retries.
        """
        symbol = self._normalise_symbol(symbol)
        interval = self._normalise_interval(interval)
        limit = min(limit, _MAX_LIMIT)

        cache_key = f"{symbol}:{interval}:{limit}"

        if use_cache and self._cache_ttl > 0:
            with self._lock:
                entry = self._cache.get(cache_key)
            if entry is not None and entry.is_valid():
                logger.debug("Cache hit for %s", cache_key)
                return entry.df.copy()

        df = self._fetch_with_retry(symbol, interval, limit)

        if use_cache and self._cache_ttl > 0:
            with self._lock:
                self._cache[cache_key] = _CacheEntry(
                    df=df.copy(),
                    timestamp=time.monotonic(),
                    ttl=self._cache_ttl,
                )

        return df

    def validate_symbol(self, symbol: str) -> bool:
        """
        Return True if *symbol* is a valid MEXC trading pair.

        Makes a lightweight ticker call; does not count as a data request.
        """
        symbol = symbol.upper().strip()
        try:
            resp = self._session.get(
                f"{self._base_url}/ticker/price",
                params={"symbol": symbol},
                timeout=self._timeout,
            )
            if resp.status_code == 400:
                return False
            resp.raise_for_status()
            data = resp.json()
            return bool(data.get("symbol") == symbol and data.get("price"))
        except Exception as exc:
            logger.warning("Symbol validation failed for %s: %s", symbol, exc)
            return False

    def clear_cache(self) -> None:
        """Evict all cached entries."""
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_with_retry(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> pd.DataFrame:
        """Call the MEXC klines endpoint with retry logic."""
        url = f"{self._base_url}/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    "Fetching %s %s %d bars (attempt %d/%d)",
                    symbol, interval, limit, attempt, self._max_retries,
                )
                resp = self._session.get(url, params=params, timeout=self._timeout)

                if resp.status_code == 400:
                    body = resp.json() if resp.content else {}
                    raise ValueError(
                        f"MEXC API 400 for {symbol}/{interval}: "
                        f"{body.get('msg', resp.text)}"
                    )

                resp.raise_for_status()
                raw = resp.json()

                if not isinstance(raw, list) or len(raw) == 0:
                    raise ValueError(
                        f"Empty klines response for {symbol}/{interval}"
                    )

                return self._parse_klines(raw)

            except ValueError:
                raise  # Don't retry on invalid symbol / bad params.
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    logger.warning(
                        "Attempt %d failed (%s). Retrying in %.1fs …",
                        attempt, exc, self._retry_delay,
                    )
                    time.sleep(self._retry_delay)

        raise RuntimeError(
            f"MEXC API unreachable after {self._max_retries} attempts: {last_exc}"
        )

    @staticmethod
    def _parse_klines(raw: list) -> pd.DataFrame:
        """
        Convert the raw MEXC klines list to a clean, typed DataFrame.

        Handles both the legacy 12-field format and the current 8-field
        format (MEXC removed taker/trade columns from the public API).

        The index is a UTC-aware DatetimeIndex derived from open_time (ms).
        All OHLCV columns are cast to float64.
        """
        n_cols = len(raw[0]) if raw else 8
        if n_cols >= 12:
            cols = _KLINE_COLS_12
        else:
            cols = _KLINE_COLS_8

        df = pd.DataFrame(raw, columns=cols[:n_cols])

        # Timestamps: MEXC returns milliseconds.
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        # Cast numeric columns.
        base_numeric = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in base_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Optional columns present only in the 12-field format.
        for col in ("taker_buy_base", "taker_buy_quote"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "trades" in df.columns:
            df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
        if "_ignore" in df.columns:
            df = df.drop(columns=["_ignore"])

        df = df.set_index("open_time").sort_index()

        # Sanity: drop rows with any OHLCV NaN (corrupted candles).
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        return df

    @staticmethod
    def _normalise_symbol(symbol: str) -> str:
        """Strip whitespace and upper-case the symbol."""
        clean = symbol.strip().upper()
        if not clean:
            raise ValueError("Symbol must not be empty.")
        return clean

    @staticmethod
    def _normalise_interval(interval: str) -> str:
        """
        Resolve human-readable aliases to MEXC native interval strings.

        Raises ValueError on unrecognised intervals.
        """
        resolved = INTERVAL_ALIASES.get(interval, interval)
        if resolved not in VALID_INTERVALS:
            raise ValueError(
                f"Unknown interval '{interval}'. "
                f"Valid values: {', '.join(VALID_INTERVALS)} "
                f"(aliases: 1h=60m)."
            )
        return resolved


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_client: Optional[MexcClient] = None


def fetch_ohlcv(
    symbol: str,
    interval: str = "60m",
    limit: int = 300,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Module-level wrapper using a shared default :class:`MexcClient` instance.

    Suitable for scripts and notebooks; avoids creating a new HTTP session
    on every call.
    """
    global _default_client
    if _default_client is None:
        _default_client = MexcClient()
    return _default_client.fetch_ohlcv(symbol, interval, limit, use_cache)
