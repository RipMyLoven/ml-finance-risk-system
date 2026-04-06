"""
overlays/symbol_search.py — Modal symbol search overlay.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, ListView, ListItem, Label, Static


class SymbolSearchOverlay(ModalScreen[str]):
    """Type to search MEXC symbols, up/down to select, Enter to confirm."""

    CSS = """
    SymbolSearchOverlay {
        align: center middle;
    }
    #search-container {
        width: 50;
        height: 24;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }
    #search-input {
        margin-bottom: 1;
    }
    #search-results {
        height: 1fr;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="search-container"):
            yield Static("[bold]Symbol Search[/]  [dim](type to filter, Enter to select, Esc to close)[/]")
            yield Input(placeholder="Type symbol (e.g. ETH)…", id="search-input")
            yield ListView(id="search-results")

    def on_mount(self) -> None:
        self.query_one("#search-input", Input).focus()

    async def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.strip()
        if len(query) < 2:
            return

        # Ask orchestrator for symbols
        app = self.app
        if hasattr(app, "orchestrator"):
            results = await app.orchestrator.search_symbols(query)
            self._results = results

            lv = self.query_one("#search-results", ListView)
            await lv.clear()
            for sym in results[:15]:
                await lv.append(ListItem(Label(sym)))

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip().upper()
        if query:
            self.dismiss(query)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        label = event.item.query_one(Label)
        self.dismiss(str(label.renderable))

    def key_escape(self) -> None:
        self.dismiss("")
