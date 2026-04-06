"""
cache/memory.py — In-process async-safe LRU cache with TTL.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class _Entry:
    value: Any
    expires_at: float


class MemoryCache:
    """TTL-based in-process LRU cache, protected by asyncio.Lock."""

    def __init__(self, max_entries: int = 50) -> None:
        self._max = max_entries
        self._store: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return entry.value

    async def put(self, key: str, value: Any, ttl: float) -> None:
        async with self._lock:
            self._store[key] = _Entry(
                value=value,
                expires_at=time.monotonic() + ttl,
            )
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    async def invalidate(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    def size(self) -> int:
        return len(self._store)
