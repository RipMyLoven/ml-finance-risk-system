"""
app.py — Main Textual TUI application.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header

from .config import AppConfig
from .state import AppState
from .orchestrator import Orchestrator
from .history import PredictionHistory
from .panels.header import HeaderBar
from .panels.meta_panel import MetaPanel
from .panels.model_panel import ModelPanel
from .panels.risk_panel import RiskPanel
from .panels.signal_panel import SignalPanel
from .panels.history_panel import HistoryPanel
from .panels.status_bar import StatusBar

logger = logging.getLogger(__name__)


class TradingTUI(App):
    """ML Finance Risk System — Real-Time TUI."""

    TITLE = "ML Finance Risk System"
    CSS = """
    Screen {
        layout: vertical;
    }
    #header-bar {
        height: 3;
        dock: top;
        background: $primary-background;
        border-bottom: solid $primary;
        content-align: left middle;
        padding: 0 1;
    }
    #main-area {
        height: 1fr;
    }
    #top-row {
        height: 2fr;
    }
    #bottom-row {
        height: 1fr;
    }
    #meta-panel {
        width: 1fr;
        border: solid $primary;
    }
    #model-panel {
        width: 2fr;
        border: solid $primary;
    }
    #risk-panel {
        width: 1fr;
        border: solid $primary;
    }
    #signal-panel {
        width: 1fr;
        border: solid $primary;
    }
    #history-panel {
        width: 1fr;
        border: solid $primary;
    }
    #status-bar {
        height: 3;
        dock: bottom;
        background: $primary-background;
        border-top: solid $primary;
        content-align: left middle;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit_app", "Quit", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("s", "search_symbol", "Symbols", show=True),
        Binding("question_mark", "help", "Help", show=True),
        Binding("1", "isolate_1", "Scalp", show=False),
        Binding("2", "isolate_2", "Intraday", show=False),
        Binding("3", "isolate_3", "Swing", show=False),
        Binding("0", "isolate_0", "All Models", show=False),
        Binding("p", "toggle_pause", "Pause", show=False),
        Binding("h", "toggle_history", "History", show=False),
        Binding("left_square_bracket", "prev_timeframe", "TF-", show=False),
        Binding("right_square_bracket", "next_timeframe", "TF+", show=False),
    ]

    def __init__(
        self,
        config: AppConfig,
        initial_symbols: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.state = AppState()
        if initial_symbols:
            self.state.symbol = initial_symbols[0]
            self.state.symbols = initial_symbols
        self.orchestrator = Orchestrator(config, self.state)
        self.history_writer = PredictionHistory()
        self._inference_task: asyncio.Task | None = None
        self._tick_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header-bar")
        with Vertical(id="main-area"):
            with Horizontal(id="top-row"):
                yield MetaPanel(id="meta-panel")
                yield ModelPanel(id="model-panel")
                yield RiskPanel(id="risk-panel")
            with Horizontal(id="bottom-row"):
                yield SignalPanel(id="signal-panel")
                yield HistoryPanel(id="history-panel")
        yield StatusBar(id="status-bar")

    async def on_mount(self) -> None:
        # Load models and start orchestrator
        try:
            await self.orchestrator.start()
        except Exception as exc:
            self.state.status = "ERROR"
            self.state.last_error = str(exc)
            logger.error("Failed to start orchestrator: %s", exc)

        self.orchestrator.set_notify(self._on_new_result)

        # Start background tasks
        self._inference_task = asyncio.create_task(self.orchestrator.run_loop())
        self._tick_task = asyncio.create_task(self._tick_loop())

    def _on_new_result(self) -> None:
        """Called by orchestrator when new prediction arrives."""
        self.call_from_thread(self._update_panels)

    def _update_panels(self) -> None:
        """Push state to all panels."""
        s = self.state
        r = s.result

        # Header
        header = self.query_one("#header-bar", HeaderBar)
        header.symbol = s.symbol
        header.timeframe = s.display_timeframe
        header.status = s.status
        header.latency_ms = s.last_inference_ms

        if r is not None:
            # Meta panel
            meta = self.query_one("#meta-panel", MetaPanel)
            reason_parts = []
            if r.scalp:
                reason_parts.append(f"scalp {r.scalp.predicted_class}")
            if r.intraday:
                reason_parts.append(f"intraday {r.intraday.predicted_class}")
            if r.swing:
                reason_parts.append(f"swing {r.swing.predicted_class}")
            if r.risk:
                reason_parts.append(f"risk {r.risk.risk_level}")
            meta.update_signal(r.meta_signal, r.meta_score, " + ".join(reason_parts))

            # Model panel
            model = self.query_one("#model-panel", ModelPanel)
            model.update_predictions(r.scalp, r.intraday, r.swing)

            # Risk panel
            risk = self.query_one("#risk-panel", RiskPanel)
            risk.update_risk(r.risk)

            # Signal panel
            signal = self.query_one("#signal-panel", SignalPanel)
            signal.update_signal(r, price=0.0, atr=0.0)

            # History
            hist = self.query_one("#history-panel", HistoryPanel)
            hist.update_history(s.history)

            # Write to disk
            self.history_writer.append(r)

        # Status bar
        status = self.query_one("#status-bar", StatusBar)
        status.update_status(
            data_age=s.data_age_seconds,
            cache_hit=s.last_cache_hit,
            next_refresh=s.next_refresh_seconds,
            error_count=s.error_count,
            paused=s.paused,
        )

    async def _tick_loop(self) -> None:
        """Update clock and age counters every second."""
        while True:
            try:
                # Refresh header (clock) and status bar (age)
                self._update_age()
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break

    def _update_age(self) -> None:
        header = self.query_one("#header-bar", HeaderBar)
        header.refresh()

        status = self.query_one("#status-bar", StatusBar)
        status.update_status(
            data_age=self.state.data_age_seconds,
            cache_hit=self.state.last_cache_hit,
            next_refresh=max(0, 60 - self.state.data_age_seconds),
            error_count=self.state.error_count,
            paused=self.state.paused,
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def action_quit_app(self) -> None:
        await self.orchestrator.stop()
        if self._inference_task:
            self._inference_task.cancel()
        if self._tick_task:
            self._tick_task.cancel()
        self.exit()

    def action_refresh(self) -> None:
        self.orchestrator.force_refresh()

    async def action_search_symbol(self) -> None:
        from .overlays.symbol_search import SymbolSearchOverlay

        result = await self.push_screen_wait(SymbolSearchOverlay())
        if result:
            await self.orchestrator.switch_symbol(result)

    def action_help(self) -> None:
        from .overlays.help import HelpOverlay
        self.push_screen(HelpOverlay())

    def action_isolate_0(self) -> None:
        self.state.model_view = 0
        self.query_one("#model-panel", ModelPanel).set_isolated(0)

    def action_isolate_1(self) -> None:
        self.state.model_view = 1
        self.query_one("#model-panel", ModelPanel).set_isolated(1)

    def action_isolate_2(self) -> None:
        self.state.model_view = 2
        self.query_one("#model-panel", ModelPanel).set_isolated(2)

    def action_isolate_3(self) -> None:
        self.state.model_view = 3
        self.query_one("#model-panel", ModelPanel).set_isolated(3)

    def action_toggle_pause(self) -> None:
        self.state.paused = not self.state.paused
        self._update_panels()

    def action_toggle_history(self) -> None:
        # Toggle history panel visibility by expanding it
        pass  # Simple version: history is always visible

    def action_prev_timeframe(self) -> None:
        opts = self.state.timeframe_options
        idx = opts.index(self.state.display_timeframe) if self.state.display_timeframe in opts else 0
        idx = max(0, idx - 1)
        self.state.display_timeframe = opts[idx]
        self._update_panels()

    def action_next_timeframe(self) -> None:
        opts = self.state.timeframe_options
        idx = opts.index(self.state.display_timeframe) if self.state.display_timeframe in opts else 0
        idx = min(len(opts) - 1, idx + 1)
        self.state.display_timeframe = opts[idx]
        self._update_panels()
