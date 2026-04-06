"""
panels/history_panel.py — Scrollable prediction history list.
"""
from __future__ import annotations

from typing import List

from textual.widgets import Static

from ..state import HistoryEntry


class HistoryPanel(Static):
    """Scrollable list of recent predictions, newest first."""

    DEFAULT_CSS = """
    HistoryPanel {
        height: 100%;
        padding: 1;
        overflow-y: auto;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries: List[HistoryEntry] = []

    def update_history(self, entries: List[HistoryEntry]) -> None:
        self._entries = entries[:20]  # Show last 20
        self.refresh()

    def render(self) -> str:
        lines = ["[bold]PREDICTION HISTORY[/]", ""]

        if not self._entries:
            lines.append("  [dim]No predictions yet[/]")
            return "\n".join(lines)

        for entry in self._entries:
            if entry.meta_signal == "LONG":
                sig = "[green]▲LONG [/]"
            elif entry.meta_signal == "SHORT":
                sig = "[red]▼SHORT[/]"
            else:
                sig = "[yellow]◆NEUT [/]"

            lines.append(
                f"  {entry.timestamp}  {entry.symbol:<10} {sig} "
                f"conf={entry.confidence:.3f}  "
                f"risk={entry.risk_level:<8} "
                f"score={entry.meta_score:+.3f}"
            )

        return "\n".join(lines)
