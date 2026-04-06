"""
app.py — Main Textual TUI application (btop-inspired design).
"""
from __future__ import annotations

import asyncio
import logging

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical

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
from .panels.chart_panel import ChartPanel
from .panels.status_bar import StatusBar

logger = logging.getLogger(__name__)

_PRESETS = [
    ("30s", 30.0),
    ("60s", 60.0),
    ("2m", 120.0),
    ("5m", 300.0),
]


class TradingTUI(App):
    """ML Finance Risk System — Real-Time TUI."""

    TITLE = "ML Finance Risk System"

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }
    #header-bar {
        height: 1;
        dock: top;
        background: $primary-background;
        padding: 0 1;
    }
    #top-row {
        height: 2fr;
    }
    #chart-row {
        height: 1fr;
        min-height: 12;
        border: solid $secondary;
    }
    #bottom-row {
        height: 1fr;
    }
    #meta-panel {
        width: 1fr;
        border: solid $secondary;
    }
    #model-panel {
        width: 2fr;
        border: solid $secondary;
    }
    #risk-panel {
        width: 1fr;
        border: solid $secondary;
    }
    #signal-panel {
        width: 1fr;
        border: solid $secondary;
    }
    #history-panel {
        width: 1fr;
        border: solid $secondary;
    }
    #status-bar {
        height: 1;
        dock: bottom;
        background: $primary-background;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "open_menu", "Menu", show=False),
        Binding("ctrl+q", "quit_app", "Quit", show=False),
        Binding("s", "search_symbol", "Symbol", show=False),
        Binding("r", "refresh", "Refresh", show=False),
        Binding("c", "toggle_chart", "Chart", show=False),
        Binding("p", "cycle_preset", "Preset", show=False),
        Binding("1", "isolate_1", "Scalp", show=False),
        Binding("2", "isolate_2", "Intraday", show=False),
        Binding("3", "isolate_3", "Swing", show=False),
        Binding("0", "isolate_0", "All", show=False),
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
        self.theme = "textual-dark"
        self._theme_name = "dark"
        if initial_symbols:
            self.state.symbol = initial_symbols[0]
            self.state.symbols = initial_symbols
        self._preset_idx = 1  # 60s default
        self.orchestrator = Orchestrator(config, self.state)
        self.history_writer = PredictionHistory()
        self._inference_task: asyncio.Task | None = None
        self._tick_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header-bar")
        with Horizontal(id="top-row"):
            yield MetaPanel(id="meta-panel")
            yield ModelPanel(id="model-panel")
            yield RiskPanel(id="risk-panel")
        yield ChartPanel(id="chart-row")
        with Horizontal(id="bottom-row"):
            yield SignalPanel(id="signal-panel")
            yield HistoryPanel(id="history-panel")
        yield StatusBar(id="status-bar")

    async def on_mount(self) -> None:
        try:
            await self.orchestrator.start()
        except Exception as exc:
            self.state.status = "ERROR"
            self.state.last_error = str(exc)
            logger.error("Failed to start orchestrator: %s", exc)

        self.orchestrator.set_notify(self._on_new_result)
        self._update_panels()
        self._inference_task = asyncio.create_task(self.orchestrator.run_loop())
        self._tick_task = asyncio.create_task(self._tick_loop())

    def _on_new_result(self) -> None:
        self._update_panels()

    # ------------------------------------------------------------------
    # Panel push
    # ------------------------------------------------------------------

    def _update_panels(self) -> None:
        s = self.state
        r = s.result

        header = self.query_one("#header-bar", HeaderBar)
        header.symbol = s.symbol
        header.status = s.status
        header.latency_ms = s.last_inference_ms
        header.preset_label = _PRESETS[self._preset_idx][0]

        if r is not None:
            meta = self.query_one("#meta-panel", MetaPanel)
            parts = []
            if r.scalp:
                parts.append(f"scalp {r.scalp.predicted_class}")
            if r.intraday:
                parts.append(f"intraday {r.intraday.predicted_class}")
            if r.swing:
                parts.append(f"swing {r.swing.predicted_class}")
            if r.risk:
                parts.append(f"risk {r.risk.risk_level}")
            meta.update_signal(r.meta_signal, r.meta_score, " + ".join(parts))

            self.query_one("#model-panel", ModelPanel).update_predictions(
                r.scalp, r.intraday, r.swing,
            )
            self.query_one("#risk-panel", RiskPanel).update_risk(r.risk)
            self.query_one("#signal-panel", SignalPanel).update_signal(
                r, price=s.last_price, atr=s.last_atr,
            )
            self.query_one("#history-panel", HistoryPanel).update_history(s.history)
            self.history_writer.append(r)

        chart = self.query_one("#chart-row", ChartPanel)
        if s.chart_closes:
            chart.update_chart(
                closes=s.chart_closes,
                opens=s.chart_opens,
                symbol=s.chart_symbol,
                interval=s.chart_interval,
            )

        self.query_one("#status-bar", StatusBar).update_status(
            data_age=s.data_age_seconds,
            cache_hit=s.last_cache_hit,
            next_refresh=max(0, s.refresh_interval - s.data_age_seconds),
            error_count=s.error_count,
            paused=s.paused,
        )

    async def _tick_loop(self) -> None:
        while True:
            try:
                self._update_age()
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break

    def _update_age(self) -> None:
        s = self.state
        header = self.query_one("#header-bar", HeaderBar)
        header.symbol = s.symbol
        header.status = s.status
        header.latency_ms = s.last_inference_ms
        header.preset_label = _PRESETS[self._preset_idx][0]

        self.query_one("#status-bar", StatusBar).update_status(
            data_age=s.data_age_seconds,
            cache_hit=s.last_cache_hit,
            next_refresh=max(0, s.refresh_interval - s.data_age_seconds),
            error_count=s.error_count,
            paused=s.paused,
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_quit_app(self) -> None:
        asyncio.create_task(self._do_quit())

    def action_open_menu(self) -> None:
        from .overlays.menu import MenuOverlay
        self.push_screen(
            MenuOverlay(current_theme=self._theme_name),
            callback=self._on_menu_result,
        )

    def _on_menu_result(self, result: str | None) -> None:
        if not result:
            return
        if result == "btn-quit":
            asyncio.create_task(self._do_quit())
        elif result == "btn-help":
            from .overlays.help import HelpOverlay
            self.push_screen(HelpOverlay())
        elif result == "btn-settings":
            from .overlays.settings import SettingsOverlay
            self.push_screen(
                SettingsOverlay(
                    refresh_interval=self.state.refresh_interval,
                    show_chart=self.state.show_chart,
                    show_signal=self.state.show_signal,
                    show_history=self.state.show_history,
                ),
                callback=self._on_settings_applied,
            )
        elif result.startswith("theme:"):
            theme_name = result.split(":", 1)[1]
            self._theme_name = theme_name
            _THEME_MAP = {
                "dark": "textual-dark",
                "dracula": "dracula",
                "monokai": "monokai",
                "nord": "nord",
                "gruvbox": "gruvbox",
                "tokyo-night": "tokyo-night",
                "textual-light": "textual-light",
            }
            self.theme = _THEME_MAP.get(theme_name, theme_name)

    def _on_settings_applied(self, result: dict | None) -> None:
        if not result:
            return
        s = self.state
        s.refresh_interval = result.get("refresh_interval", s.refresh_interval)
        s.show_chart = result.get("show_chart", s.show_chart)
        s.show_signal = result.get("show_signal", s.show_signal)
        s.show_history = result.get("show_history", s.show_history)
        self.query_one("#chart-row").display = s.show_chart
        self.query_one("#signal-panel").display = s.show_signal
        self.query_one("#history-panel").display = s.show_history
        self.orchestrator.set_refresh_interval(s.refresh_interval)

    async def _do_quit(self) -> None:
        await self.orchestrator.stop()
        if self._inference_task:
            self._inference_task.cancel()
        if self._tick_task:
            self._tick_task.cancel()
        self.exit()

    def action_refresh(self) -> None:
        self.orchestrator.force_refresh()

    def action_search_symbol(self) -> None:
        from .overlays.symbol_search import SymbolSearchOverlay
        self.push_screen(SymbolSearchOverlay(), callback=self._on_symbol_selected)

    def _on_symbol_selected(self, result: str | None) -> None:
        if result:
            asyncio.create_task(self.orchestrator.switch_symbol(result))

    def action_toggle_chart(self) -> None:
        self.state.show_chart = not self.state.show_chart
        self.query_one("#chart-row").display = self.state.show_chart

    def action_cycle_preset(self) -> None:
        self._preset_idx = (self._preset_idx + 1) % len(_PRESETS)
        label, val = _PRESETS[self._preset_idx]
        self.state.refresh_interval = val
        self.orchestrator.set_refresh_interval(val)
        self._update_panels()

    def action_interval_up(self) -> None:
        s = self.state
        s.refresh_interval = min(600.0, s.refresh_interval + 10.0)
        self.orchestrator.set_refresh_interval(s.refresh_interval)
        self._sync_preset_label()
        self._update_panels()

    def action_interval_down(self) -> None:
        s = self.state
        s.refresh_interval = max(10.0, s.refresh_interval - 10.0)
        self.orchestrator.set_refresh_interval(s.refresh_interval)
        self._sync_preset_label()
        self._update_panels()

    def _sync_preset_label(self) -> None:
        val = self.state.refresh_interval
        for i, (lbl, pval) in enumerate(_PRESETS):
            if abs(pval - val) < 1:
                self._preset_idx = i
                return

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
