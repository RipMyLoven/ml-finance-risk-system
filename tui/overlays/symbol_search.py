"""
overlays/symbol_search.py — Modal symbol search overlay.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, ListView, ListItem, Label, Static

_POPULAR = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "LINKUSDT", "LTCUSDT", "SUIUSDT", "APTUSDT", "ARBUSDT",
]


class SymbolSearchOverlay(ModalScreen[str | None]):
    """Type to search MEXC symbols, up/down to select, Enter to confirm."""

    CSS = """
    SymbolSearchOverlay {
        align: center middle;
        background: rgba(0, 0, 0, 0.65);
    }
    #search-container {
        width: 55;
        height: 28;
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
    #search-hint {
        height: 2;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="search-container"):
            yield Static("[bold]Symbol Search[/]")
            yield Static(
                "[dim]Type to filter · Enter to confirm · Esc to cancel[/]",
                id="search-hint",
            )
            yield Input(placeholder="Type symbol (e.g. ETH, SOL, BTC)", id="search-input")
            yield ListView(id="search-results")

    async def on_mount(self) -> None:
        lv = self.query_one("#search-results", ListView)
        for sym in _POPULAR:
            await lv.append(ListItem(Label(sym)))
        self.query_one("#search-input", Input).focus()

    async def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.strip()
        lv = self.query_one("#search-results", ListView)
        await lv.clear()

        if len(query) < 1:
            # Show popular symbols when input is empty
            for sym in _POPULAR:
                await lv.append(ListItem(Label(sym)))
            return

        if len(query) < 2:
            # Filter popular by partial match
            filtered = [s for s in _POPULAR if query.upper() in s]
            for sym in filtered:
                await lv.append(ListItem(Label(sym)))
            return

        # Ask orchestrator for symbols from MEXC
        app = self.app
        if hasattr(app, "orchestrator"):
            results = await app.orchestrator.search_symbols(query)
            self._results = results
            for sym in results[:20]:
                await lv.append(ListItem(Label(sym)))

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip().upper()
        if query:
            self.dismiss(query)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        label = event.item.query_one(Label)
        text = label.render()
        self.dismiss(str(text).strip())

    def key_escape(self) -> None:
        self.dismiss(None)
