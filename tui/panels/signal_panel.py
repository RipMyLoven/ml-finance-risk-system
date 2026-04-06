"""
panels/signal_panel.py — Entry / SL / TP display.
"""
from __future__ import annotations

from typing import Optional

from textual.widgets import Static

from ..model_registry.registry import PredictionResult


class SignalPanel(Static):
    """Shows the trading signal details when a non-neutral signal is active."""

    DEFAULT_CSS = """
    SignalPanel {
        height: 100%;
        padding: 1 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._result: Optional[PredictionResult] = None
        self._price: float = 0.0
        self._atr: float = 0.0

    def update_signal(
        self, result: Optional[PredictionResult], price: float = 0.0, atr: float = 0.0
    ) -> None:
        self._result = result
        self._price = price
        self._atr = atr
        self.refresh()

    def render(self) -> str:
        lines = ["[bold]SIGNAL[/]", ""]

        if self._result is None or self._result.meta_signal == "NEUTRAL":
            lines.append("  [dim]No active signal[/]")
            return "\n".join(lines)

        r = self._result
        is_long = r.meta_signal == "LONG"

        # Direction
        if is_long:
            dir_str = "[bold green]LONG ▲[/]"
        else:
            dir_str = "[bold red]SHORT ▼[/]"

        lines.append(f"  Direction: {dir_str}")
        lines.append(f"  Score:     {r.meta_score:+.4f}")

        if self._price > 0 and self._atr > 0:
            atr = self._atr

            if is_long:
                entry_lo = self._price - atr * 0.1
                entry_hi = self._price + atr * 0.1
                sl = self._price - atr * 1.5
                tp1 = self._price + atr * 1.5 * 1.5
                tp2 = self._price + atr * 1.5 * 2.5
            else:
                entry_lo = self._price - atr * 0.1
                entry_hi = self._price + atr * 0.1
                sl = self._price + atr * 1.5
                tp1 = self._price - atr * 1.5 * 1.5
                tp2 = self._price - atr * 1.5 * 2.5

            sl_pct = abs(sl - self._price) / self._price * 100
            tp1_pct = abs(tp1 - self._price) / self._price * 100
            tp2_pct = abs(tp2 - self._price) / self._price * 100

            # Leverage from risk
            lev = 5
            if r.risk and r.risk.risk_level == "LOW":
                lev = 5
            elif r.risk and r.risk.risk_level == "MEDIUM":
                lev = 3
            else:
                lev = 1

            lines.extend([
                "",
                f"  Entry:    ${entry_lo:,.2f} – ${entry_hi:,.2f}",
                f"  [red]SL:       ${sl:,.2f}  (−{sl_pct:.2f}%)[/]",
                f"  [green]TP1:      ${tp1:,.2f}  (+{tp1_pct:.2f}%)  R:R 1:1.5[/]",
                f"  [green]TP2:      ${tp2:,.2f}  (+{tp2_pct:.2f}%)  R:R 1:2.5[/]",
                f"  Leverage: [bold]{lev}×[/]",
            ])
        else:
            lines.extend([
                "",
                "  [dim]Price data not available for SL/TP calculation[/]",
            ])

        # Warnings
        if r.errors:
            lines.extend(["", "  [yellow]Warnings:[/]"])
            for err in r.errors[:3]:
                lines.append(f"    • {err}")

        return "\n".join(lines)
