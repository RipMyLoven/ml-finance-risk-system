"""
overlays/settings.py — btop-style settings with arrow-key navigation.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

_INTERVALS = [
    ("30s", 30.0),
    ("60s", 60.0),
    ("2m", 120.0),
    ("5m", 300.0),
]

_ITEMS = ["refresh", "chart", "signal", "history"]


class SettingsOverlay(ModalScreen[dict | None]):
    """btop-style settings: up/down to navigate, left/right to change, Esc to apply & close."""

    BINDINGS = [
        Binding("escape", "apply_close", "", show=False),
        Binding("up", "move_up", "", show=False),
        Binding("down", "move_down", "", show=False),
        Binding("left", "dec", "", show=False),
        Binding("right", "inc", "", show=False),
        Binding("enter", "toggle", "", show=False),
    ]

    CSS = """
    SettingsOverlay {
        align: center middle;
        background: rgba(0, 0, 0, 0.65);
    }
    #settings-box {
        width: 48;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    #settings-title {
        text-align: center;
        margin-bottom: 1;
    }
    .s-row {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        refresh_interval: float = 60.0,
        show_chart: bool = True,
        show_signal: bool = True,
        show_history: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._interval_idx = 1
        for i, (_, v) in enumerate(_INTERVALS):
            if abs(v - refresh_interval) < 1:
                self._interval_idx = i
                break
        self._chart = show_chart
        self._signal = show_signal
        self._history = show_history
        self._cursor = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-box"):
            yield Static("[bold]Settings[/]", id="settings-title")
            yield Static("", id="row-refresh", classes="s-row")
            yield Static("", id="row-chart", classes="s-row")
            yield Static("", id="row-signal", classes="s-row")
            yield Static("", id="row-history", classes="s-row")
            yield Static(
                "\n[dim]Up/Down navigate  Left/Right change  Enter toggle  Esc apply & close[/]",
                id="settings-hint",
            )

    def on_mount(self) -> None:
        self._refresh_rows()

    def _refresh_rows(self) -> None:
        c = self._cursor
        ilabel, _ = _INTERVALS[self._interval_idx]
        total = len(_INTERVALS)

        def row(idx: int, text: str) -> str:
            if idx == c:
                return f"  [reverse] {text} [/]"
            return f"    {text}"

        self.query_one("#row-refresh").update(
            row(0, f"Refresh interval      [bold]<  {ilabel}  >[/]  ({self._interval_idx + 1}/{total})")
        )
        self.query_one("#row-chart").update(
            row(1, f"Price chart           {self._on_off(self._chart)}")
        )
        self.query_one("#row-signal").update(
            row(2, f"Signal panel          {self._on_off(self._signal)}")
        )
        self.query_one("#row-history").update(
            row(3, f"History panel         {self._on_off(self._history)}")
        )

    @staticmethod
    def _on_off(val: bool) -> str:
        return "[green]ON[/]" if val else "[red]OFF[/]"

    def action_move_up(self) -> None:
        self._cursor = (self._cursor - 1) % len(_ITEMS)
        self._refresh_rows()

    def action_move_down(self) -> None:
        self._cursor = (self._cursor + 1) % len(_ITEMS)
        self._refresh_rows()

    def action_inc(self) -> None:
        item = _ITEMS[self._cursor]
        if item == "refresh":
            self._interval_idx = min(len(_INTERVALS) - 1, self._interval_idx + 1)
        elif item == "chart":
            self._chart = not self._chart
        elif item == "signal":
            self._signal = not self._signal
        elif item == "history":
            self._history = not self._history
        self._refresh_rows()
        self._apply_live()

    def action_dec(self) -> None:
        item = _ITEMS[self._cursor]
        if item == "refresh":
            self._interval_idx = max(0, self._interval_idx - 1)
        elif item == "chart":
            self._chart = not self._chart
        elif item == "signal":
            self._signal = not self._signal
        elif item == "history":
            self._history = not self._history
        self._refresh_rows()
        self._apply_live()

    def action_toggle(self) -> None:
        item = _ITEMS[self._cursor]
        if item == "chart":
            self._chart = not self._chart
        elif item == "signal":
            self._signal = not self._signal
        elif item == "history":
            self._history = not self._history
        self._refresh_rows()
        self._apply_live()

    def _apply_live(self) -> None:
        """Push visibility changes to the app immediately."""
        app = self.app
        try:
            app.query_one("#chart-row").display = self._chart
            app.query_one("#signal-panel").display = self._signal
            app.query_one("#history-panel").display = self._history
        except Exception:
            pass

    def action_apply_close(self) -> None:
        _, interval = _INTERVALS[self._interval_idx]
        self.dismiss({
            "refresh_interval": interval,
            "show_chart": self._chart,
            "show_signal": self._signal,
            "show_history": self._history,
        })
