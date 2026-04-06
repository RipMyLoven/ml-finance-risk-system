"""
panels/model_panel.py — Three-row model detail view.
"""
from __future__ import annotations

from typing import Optional, List

from textual.widgets import Static

from ..model_registry.registry import TradingPrediction


def _conf_bar(value: float, width: int = 15) -> str:
    filled = max(0, min(width, round(value * width)))
    return "[cyan]" + "█" * filled + "[/][dim]" + "░" * (width - filled) + "[/]"


def _signal_color(signal: str, width: int = 8) -> str:
    padded = signal.ljust(width)
    if signal in ("UP", "LONG"):
        return f"[bold green]{padded}[/]"
    if signal in ("DOWN", "SHORT"):
        return f"[bold red]{padded}[/]"
    return f"[yellow]{padded}[/]"


class ModelPanel(Static):
    """Displays the 3 trading model predictions in a table format."""

    DEFAULT_CSS = """
    ModelPanel {
        height: 100%;
        padding: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._models: dict = {}  # name -> TradingPrediction
        self._isolated: int = 0  # 0 = all, 1/2/3 = individual

    def update_predictions(
        self,
        scalp: Optional[TradingPrediction] = None,
        intraday: Optional[TradingPrediction] = None,
        swing: Optional[TradingPrediction] = None,
    ) -> None:
        self._models = {
            "SCALP": scalp,
            "INTRADAY": intraday,
            "SWING": swing,
        }
        self.refresh()

    def set_isolated(self, model_idx: int) -> None:
        self._isolated = model_idx
        self.refresh()

    def render(self) -> str:
        lines = ["[bold]MODEL DETAIL[/]", ""]

        header = f"  {'Model':<10} {'Signal':<8} {'Conf':<17} {'P↑':>6} {'P─':>6} {'P↓':>6} {'E[R]':>8}"
        lines.append(header)
        lines.append("  " + "─" * 66)

        names = ["SCALP", "INTRADAY", "SWING"]
        filter_map = {1: "SCALP", 2: "INTRADAY", 3: "SWING"}

        for name in names:
            if self._isolated > 0 and filter_map.get(self._isolated) != name:
                continue

            pred = self._models.get(name)
            if pred is None:
                lines.append(f"  {name:<10} [dim]N/A[/]")
                continue

            sig = _signal_color(pred.predicted_class)
            conf = _conf_bar(pred.confidence)
            er = f"{pred.expected_return:+.4f}"

            lines.append(
                f"  {name:<10} {sig} {conf} "
                f"{pred.P_up:6.3f} {pred.P_flat:6.3f} {pred.P_down:6.3f} {er:>8}"
            )

            # If isolated, show extra detail
            if self._isolated > 0:
                lines.extend([
                    "",
                    f"    Confidence: {pred.confidence:.4f}",
                    f"    Expected Return: {pred.expected_return:+.6f}",
                    f"    P(UP):   {pred.P_up:.6f}",
                    f"    P(FLAT): {pred.P_flat:.6f}",
                    f"    P(DOWN): {pred.P_down:.6f}",
                ])

        return "\n".join(lines)
