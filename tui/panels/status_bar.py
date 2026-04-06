"""
panels/status_bar.py — Bottom status bar.
"""
from __future__ import annotations

from textual.widgets import Static


class StatusBar(Static):
    """Bottom bar with data age, cache, refresh timer, errors, keybindings."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data_age = 0.0
        self._cache_hit = False
        self._next_refresh = 0.0
        self._error_count = 0
        self._paused = False

    def update_status(
        self,
        data_age: float = 0.0,
        cache_hit: bool = False,
        next_refresh: float = 0.0,
        error_count: int = 0,
        paused: bool = False,
    ) -> None:
        self._data_age = data_age
        self._cache_hit = cache_hit
        self._next_refresh = next_refresh
        self._error_count = error_count
        self._paused = paused
        self.refresh()

    def render(self) -> str:
        age = f"DATA: {self._data_age:.0f}s ago"
        cache = "[green]CACHE: HIT[/]" if self._cache_hit else "[yellow]CACHE: MISS[/]"
        nxt = f"NEXT: {self._next_refresh:.0f}s"

        if self._error_count > 0:
            err = f"[red]ERR: {self._error_count}[/]"
        else:
            err = "[dim]ERR: 0[/]"

        pause = "  [bold yellow]⏸ PAUSED[/]" if self._paused else ""

        keys = "[dim][Q]uit  [R]efresh  [S]ymbols  [1-3]Models  [?]Help[/]"

        return f" {age}  │  {cache}  │  {nxt}  │  {err}{pause}  │  {keys}"
