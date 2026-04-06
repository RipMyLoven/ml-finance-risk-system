"""
exchange/base.py — Abstract exchange adapter interface.
"""
from __future__ import annotations

import abc
from typing import Optional

import pandas as pd


class ExchangeAdapter(abc.ABC):
    """All exchange interactions are routed through this interface."""

    @abc.abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        limit: int = 350,
    ) -> pd.DataFrame:
        """Fetch OHLCV candlestick data. Returns DataFrame with OHLCV columns."""
        ...

    @abc.abstractmethod
    async def validate_symbol(self, symbol: str) -> bool:
        """Return True if *symbol* is a valid trading pair on this exchange."""
        ...

    @abc.abstractmethod
    async def search_symbols(self, query: str) -> list[str]:
        """Return symbols matching *query* (prefix search)."""
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """Clean up resources (close HTTP sessions etc.)."""
        ...
