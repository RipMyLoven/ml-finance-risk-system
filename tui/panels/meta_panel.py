"""
panels/meta_panel.py — Meta signal display (large directional indicator + score bar).
"""
from __future__ import annotations

from textual.widgets import Static


class MetaPanel(Static):
    """Displays the aggregated meta signal: LONG / SHORT / NEUTRAL."""

    DEFAULT_CSS = """
    MetaPanel {
        height: 100%;
        padding: 1 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._signal = "NEUTRAL"
        self._score = 0.0
        self._reason = ""

    def update_signal(self, signal: str, score: float, reason: str = "") -> None:
        self._signal = signal
        self._score = score
        self._reason = reason
        self.refresh()

    def render(self) -> str:
        # Direction indicator
        if self._signal == "LONG":
            arrow = "[bold green]  ▲ LONG[/]"
        elif self._signal == "SHORT":
            arrow = "[bold red]  ▼ SHORT[/]"
        else:
            arrow = "[bold yellow]  ◆ NEUTRAL[/]"

        # Score bar: -1.0 to +1.0 mapped to 0..20
        bar_width = 20
        norm = (self._score + 1.0) / 2.0
        pos = int(max(0, min(bar_width, round(norm * bar_width))))

        bar_chars = []
        for i in range(bar_width):
            if i < bar_width // 2:
                if i < pos:
                    bar_chars.append("[red]█[/]")
                else:
                    bar_chars.append("[dim]░[/]")
            elif i == bar_width // 2:
                bar_chars.append("[dim]│[/]")
            else:
                if i < pos:
                    bar_chars.append("[green]█[/]")
                else:
                    bar_chars.append("[dim]░[/]")

        bar = "".join(bar_chars)

        lines = [
            "[bold]META SIGNAL[/]",
            "",
            arrow,
            "",
            f"  Score: {self._score:+.4f}",
            f"  {bar}",
        ]
        if self._reason:
            lines.extend(["", f"  [dim]{self._reason}[/]"])

        return "\n".join(lines)
