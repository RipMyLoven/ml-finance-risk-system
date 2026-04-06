"""
overlays/help.py — btop-style help overlay.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


class HelpOverlay(ModalScreen[None]):
    CSS = """
    HelpOverlay {
        align: center middle;
        background: rgba(0, 0, 0, 0.65);
    }
    #help-box {
        width: 52;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    #help-title {
        text-align: center;
        margin-bottom: 1;
    }
    .help-row {
        height: 1;
        padding: 0 1;
    }
    #help-hint {
        margin-top: 1;
        text-align: center;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        keys = [
            ("Esc", "Open menu (settings, themes, quit)"),
            ("S", "Open symbol search"),
            ("R", "Force refresh (bypass cache)"),
            ("C", "Toggle price chart"),
            ("P", "Cycle refresh preset (30s/60s/2m/5m)"),
            ("1 / 2 / 3", "Isolate scalp / intraday / swing"),
            ("0", "Show all models"),
            ("Ctrl+Q", "Quit"),
        ]
        with Vertical(id="help-box"):
            yield Static("[bold]Keyboard Shortcuts[/]", id="help-title")
            for key, desc in keys:
                yield Static(
                    f"  [bold cyan]{key:<12}[/] {desc}",
                    classes="help-row",
                )
            yield Static("[dim]Press Esc to close[/]", id="help-hint")

    def key_escape(self) -> None:
        self.dismiss(None)
