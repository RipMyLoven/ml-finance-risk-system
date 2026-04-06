"""
panels/header.py — Header bar: symbol, timeframe, status, clock, latency.
"""
from __future__ import annotations

from datetime import datetime, timezone

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static


class HeaderBar(Static):
    """Top bar showing symbol, timeframe, status, clock, latency."""

    symbol = reactive("BTCUSDT")
    timeframe = reactive("1h")
    status = reactive("STARTING")
    latency_ms = reactive(0.0)

    def render(self) -> str:
        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        status_map = {
            "LIVE": "[bold green]● LIVE[/]",
            "STALE": "[bold yellow]● STALE[/]",
            "ERROR": "[bold red]● ERROR[/]",
            "STARTING": "[dim]● STARTING[/]",
            "COMPUTING": "[bold cyan]◌ COMPUTING[/]",
        }
        st = status_map.get(self.status, f"● {self.status}")

        lat = f"{self.latency_ms:.0f}ms" if self.latency_ms > 0 else "—"

        return (
            f" [bold]{self.symbol}[/]  │  "
            f"TF: [cyan]{self.timeframe}[/]  │  "
            f"{st}  │  "
            f"[dim]{now}[/]  │  "
            f"Latency: [cyan]{lat}[/]"
        )
