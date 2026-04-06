"""
panels/header.py — btop-style top bar with inline key hints.
"""
from __future__ import annotations

from datetime import datetime, timezone

from textual.reactive import reactive
from textual.widgets import Static


class HeaderBar(Static):
    """Top bar: key hints + symbol + status + clock + latency."""

    symbol = reactive("BTCUSDT")
    timeframe = reactive("1h")
    status = reactive("STARTING")
    latency_ms = reactive(0.0)
    preset_label = reactive("60s")

    def render(self) -> str:
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")

        status_map = {
            "LIVE": "[green]LIVE[/]",
            "STALE": "[yellow]STALE[/]",
            "ERROR": "[red]ERROR[/]",
            "STARTING": "[dim]WAIT[/]",
            "COMPUTING": "[cyan]...[/]",
        }
        st = status_map.get(self.status, self.status)

        lat = f"{self.latency_ms:.0f}ms" if self.latency_ms > 0 else "--"

        return (
            f" [red]esc[/][dim]Menu[/]  "
            f"[red]s[/] [bold]{self.symbol}[/]  "
            f"[red]p[/][dim]reset:[/]{self.preset_label}  "
            f"[red]c[/][dim]hart[/]  "
            f"[red]r[/][dim]efresh[/]  "
            f"[red]1[/][red]2[/][red]3[/][dim]model[/]"
            f"  [dim]│[/] {st}"
            f"  [dim]│[/] [dim]{now}[/]"
            f"  [dim]│[/] [cyan]{lat}[/]"
        )
