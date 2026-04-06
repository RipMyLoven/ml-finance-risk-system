"""
overlays/help.py — Keybinding help overlay.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static, Button


_HELP_TEXT = """[bold]Keyboard Shortcuts[/]

  [bold cyan]Q[/]         Quit (flush history first)
  [bold cyan]R[/]         Force refresh (bypass cache)
  [bold cyan]S[/]         Open symbol search
  [bold cyan][ / ][/]     Decrease / increase timeframe
  [bold cyan]1 / 2 / 3[/] Isolate scalp / intraday / swing model
  [bold cyan]0[/]         Return to 3-model summary
  [bold cyan]H[/]         Toggle history full-screen
  [bold cyan]P[/]         Toggle pause mode
  [bold cyan]?[/]         Show this help
  [bold cyan]Escape[/]    Close overlay
  [bold cyan]Ctrl+C[/]    Emergency exit
"""


class HelpOverlay(ModalScreen[None]):
    CSS = """
    HelpOverlay {
        align: center middle;
    }
    #help-container {
        width: 60;
        height: auto;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-container"):
            yield Static(_HELP_TEXT)
            with Center():
                yield Button("Close", variant="primary", id="close-help")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-help":
            self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)

    def key_question_mark(self) -> None:
        self.dismiss(None)
