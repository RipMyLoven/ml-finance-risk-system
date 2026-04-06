"""
overlays/menu.py — btop-style ESC menu with inline theme cycling.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Static

_THEMES = [
    ("Dark", "dark"),
    ("Dracula", "dracula"),
    ("Monokai", "monokai"),
    ("Nord", "nord"),
    ("Gruvbox", "gruvbox"),
    ("Tokyo Night", "tokyo-night"),
    ("Light", "textual-light"),
]

_ITEMS = ["theme", "settings", "help", "quit"]


class MenuOverlay(ModalScreen[str | None]):
    """btop-style menu: up/down to move, left/right to cycle theme, Enter/Esc."""

    BINDINGS = [
        Binding("escape", "close_menu", "", show=False),
        Binding("up", "move_up", "", show=False),
        Binding("down", "move_down", "", show=False),
        Binding("left", "prev_option", "", show=False),
        Binding("right", "next_option", "", show=False),
        Binding("enter", "select_item", "", show=False),
    ]

    CSS = """
    MenuOverlay {
        align: center middle;
        background: rgba(0, 0, 0, 0.65);
    }
    #menu-box {
        width: 48;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    #menu-title {
        text-align: center;
        margin-bottom: 1;
    }
    .menu-row {
        height: 1;
        padding: 0 1;
    }
    .menu-desc {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
        color: $text-muted;
    }
    """

    def __init__(self, current_theme: str = "dark", **kwargs) -> None:
        super().__init__(**kwargs)
        self._theme_idx = 0
        for i, (_, val) in enumerate(_THEMES):
            if val == current_theme:
                self._theme_idx = i
                break
        self._cursor = 0  # index into _ITEMS

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-box"):
            yield Static("[bold]ML Finance Risk System[/]", id="menu-title")
            yield Static("", id="row-theme", classes="menu-row")
            yield Static("", id="desc-theme", classes="menu-desc")
            yield Static("", id="row-settings", classes="menu-row")
            yield Static("", id="row-help", classes="menu-row")
            yield Static("", id="row-quit", classes="menu-row")
            yield Static(
                "\n[dim]Up/Down navigate  Left/Right cycle theme  Enter select  Esc close[/]",
                id="menu-hint",
            )

    def on_mount(self) -> None:
        self._refresh_rows()

    def _refresh_rows(self) -> None:
        c = self._cursor
        ti = self._theme_idx
        total = len(_THEMES)
        label, _ = _THEMES[ti]

        def row(idx: int, text: str) -> str:
            if idx == c:
                return f"  [reverse] {text} [/]"
            return f"    {text}"

        theme_text = f"Color theme {ti + 1}/{total}    [bold]<  {label}  >[/]"
        self.query_one("#row-theme").update(row(0, theme_text))

        desc_map = {
            "dark": "Default dark theme",
            "dracula": "Dracula purple palette",
            "monokai": "Monokai warm tones",
            "nord": "Nord arctic blue",
            "gruvbox": "Gruvbox retro warm",
            "tokyo-night": "Tokyo Night cool blue",
            "textual-light": "Light background theme",
        }
        _, val = _THEMES[ti]
        self.query_one("#desc-theme").update(f"    [dim]{desc_map.get(val, '')}[/]")

        self.query_one("#row-settings").update(row(1, "Settings"))
        self.query_one("#row-help").update(row(2, "Help"))
        self.query_one("#row-quit").update(row(3, "[red]Quit[/]"))

    def action_move_up(self) -> None:
        self._cursor = (self._cursor - 1) % len(_ITEMS)
        self._refresh_rows()

    def action_move_down(self) -> None:
        self._cursor = (self._cursor + 1) % len(_ITEMS)
        self._refresh_rows()

    def action_prev_option(self) -> None:
        if _ITEMS[self._cursor] == "theme":
            self._theme_idx = (self._theme_idx - 1) % len(_THEMES)
            self._refresh_rows()
            self._apply_theme()

    def action_next_option(self) -> None:
        if _ITEMS[self._cursor] == "theme":
            self._theme_idx = (self._theme_idx + 1) % len(_THEMES)
            self._refresh_rows()
            self._apply_theme()

    def _apply_theme(self) -> None:
        _, val = _THEMES[self._theme_idx]
        _THEME_MAP = {
            "dark": "textual-dark",
            "dracula": "dracula",
            "monokai": "monokai",
            "nord": "nord",
            "gruvbox": "gruvbox",
            "tokyo-night": "tokyo-night",
            "textual-light": "textual-light",
        }
        self.app.theme = _THEME_MAP.get(val, val)
        if hasattr(self.app, "_theme_name"):
            self.app._theme_name = val

    def action_select_item(self) -> None:
        item = _ITEMS[self._cursor]
        if item == "theme":
            return  # use left/right to cycle, no enter action
        if item == "settings":
            self.dismiss("btn-settings")
        elif item == "help":
            self.dismiss("btn-help")
        elif item == "quit":
            self.dismiss("btn-quit")

    def action_close_menu(self) -> None:
        self.dismiss(None)
