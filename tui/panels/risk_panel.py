"""
panels/risk_panel.py — Risk gauge and advice display.
"""
from __future__ import annotations

from typing import Optional

from textual.widgets import Static

from ..model_registry.registry import RiskPrediction


class RiskPanel(Static):
    """Vertical risk gauge with score, level, and leverage advice."""

    DEFAULT_CSS = """
    RiskPanel {
        height: 100%;
        padding: 1 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._risk: Optional[RiskPrediction] = None

    def update_risk(self, risk: Optional[RiskPrediction]) -> None:
        self._risk = risk
        self.refresh()

    def render(self) -> str:
        lines = ["[bold]RISK ASSESSMENT[/]", ""]

        if self._risk is None:
            lines.append("  [dim]N/A — waiting for data[/]")
            return "\n".join(lines)

        r = self._risk

        # Risk level with color
        level_colors = {
            "LOW": "[bold green]LOW[/]",
            "MEDIUM": "[bold yellow]MEDIUM[/]",
            "HIGH": "[bold #ff6b35]HIGH[/]",
            "CRITICAL": "[bold red]CRITICAL[/]",
        }
        level = level_colors.get(r.risk_level, r.risk_level)

        # Vertical gauge (10 rows, bottom=0, top=1.0)
        gauge_height = 10
        fill_level = int(r.risk_score * gauge_height)

        lines.append(f"  Level: {level}")
        lines.append(f"  Score: {r.risk_score:.4f}")
        lines.append("")

        for row in range(gauge_height, 0, -1):
            val = row / gauge_height
            if row <= fill_level:
                if val > 0.75:
                    char = "[red]██[/]"
                elif val > 0.50:
                    char = "[#ff6b35]██[/]"
                elif val > 0.25:
                    char = "[yellow]██[/]"
                else:
                    char = "[green]██[/]"
            else:
                char = "[dim]░░[/]"

            label = f"{val:.1f}" if row % 2 == 0 else "   "
            lines.append(f"  {label} {char}")

        lines.append(f"  0.0 [dim]░░[/]")
        lines.append("")
        lines.append(f"  Max Leverage: [bold]{r.max_leverage_suggested:.0f}×[/]")

        return "\n".join(lines)
