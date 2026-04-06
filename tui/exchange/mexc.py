"""
exchange/mexc.py — Async MEXC public API adapter.

No API key required. Uses aiohttp for non-blocking I/O.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import aiohttp
import pandas as pd

from .base import ExchangeAdapter

logger = logging.getLogger(__name__)

# MEXC kline column layouts.
_KLINE_COLS_8 = [
    "open_time", "open", "high", "low", "close",
    "volume", "close_time", "quote_volume",
]
_KLINE_COLS_12 = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "_ignore",
]

VALID_INTERVALS: Tuple[str, ...] = (
    "1m", "5m", "15m", "30m", "60m", "4h", "1d", "1W", "1M",
)

INTERVAL_ALIASES: Dict[str, str] = {
    "1h": "60m", "4h": "4h", "1d": "1d", "1D": "1d",
    "5m": "5m", "15m": "15m", "30m": "30m", "1m": "1m", "60m": "60m",
}


class MexcAdapter(ExchangeAdapter):
    """Async MEXC REST API adapter."""

    def __init__(
        self,
        base_url: str = "https://api.mexc.com/api/v3",
        timeout: int = 15,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None
        self._symbols_cache: Optional[list[str]] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers={"Accept": "application/json"},
            )
        return self._session

    async def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "60m",
        limit: int = 350,
    ) -> pd.DataFrame:
        symbol = symbol.strip().upper()
        interval = INTERVAL_ALIASES.get(interval, interval)
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Unknown interval '{interval}'")
        limit = min(limit, 1000)

        session = await self._get_session()
        url = f"{self._base_url}/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 400:
                        body = await resp.json()
                        raise ValueError(
                            f"MEXC API 400 for {symbol}/{interval}: "
                            f"{body.get('msg', await resp.text())}"
                        )
                    resp.raise_for_status()
                    raw = await resp.json()

                if not isinstance(raw, list) or len(raw) == 0:
                    raise ValueError(f"Empty klines for {symbol}/{interval}")

                return self._parse_klines(raw)

            except ValueError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    import asyncio
                    logger.warning(
                        "Attempt %d failed (%s). Retrying…", attempt, exc
                    )
                    await asyncio.sleep(self._retry_delay)

        raise RuntimeError(
            f"MEXC unreachable after {self._max_retries} attempts: {last_exc}"
        )

    async def validate_symbol(self, symbol: str) -> bool:
        symbol = symbol.strip().upper()
        try:
            session = await self._get_session()
            async with session.get(
                f"{self._base_url}/ticker/price",
                params={"symbol": symbol},
            ) as resp:
                if resp.status == 400:
                    return False
                resp.raise_for_status()
                data = await resp.json()
                return bool(data.get("symbol") == symbol and data.get("price"))
        except Exception:
            return False

    async def search_symbols(self, query: str) -> list[str]:
        if self._symbols_cache is None:
            try:
                session = await self._get_session()
                async with session.get(f"{self._base_url}/ticker/price") as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    self._symbols_cache = sorted(
                        [item["symbol"] for item in data if item.get("symbol")]
                    )
            except Exception:
                return []

        q = query.strip().upper()
        return [s for s in (self._symbols_cache or []) if q in s][:50]

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    @staticmethod
    def _parse_klines(raw: list) -> pd.DataFrame:
        n_cols = len(raw[0]) if raw else 8
        cols = _KLINE_COLS_12 if n_cols >= 12 else _KLINE_COLS_8
        df = pd.DataFrame(raw, columns=cols[:n_cols])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ("taker_buy_base", "taker_buy_quote"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "trades" in df.columns:
            df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
        if "_ignore" in df.columns:
            df = df.drop(columns=["_ignore"])

        df = df.set_index("open_time").sort_index()
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        return df
