"""
panels/chart_panel.py — ASCII price chart panel.
"""
from __future__ import annotations

from typing import List, Optional

from textual.widgets import Static

_BLOCKS = " ▁▂▃▄▅▆▇█"


def _fmt_price(p: float) -> str:
    """Smart price formatting: show enough decimals for small coins."""
    if abs(p) >= 1000:
        return f"{p:>10,.0f}"
    if abs(p) >= 1:
        return f"{p:>10,.2f}"
    if abs(p) >= 0.01:
        return f"{p:>10,.4f}"
    return f"{p:>10,.6f}"


def _render_chart(
    closes: List[float],
    opens: Optional[List[float]] = None,
    width: int = 60,
    height: int = 8,
) -> str:
    """Render an ASCII bar chart from price data."""
    if not closes or len(closes) < 2:
        return "  [dim]Waiting for price data...[/]"

    data = closes[-width:]
    op = opens[-width:] if opens and len(opens) >= len(closes) else None

    mn = min(data)
    mx = max(data)
    rng = mx - mn
    if rng == 0:
        rng = max(abs(mn) * 0.01, 0.000001)

    lines: list[str] = []

    for row in range(height, 0, -1):
        row_chars: list[str] = []
        for i, val in enumerate(data):
            norm = (val - mn) / rng * height
            if op and i < len(op):
                color = "green" if val >= op[i] else "red"
            else:
                color = "green" if i == 0 or val >= data[i - 1] else "red"

            if norm >= row:
                row_chars.append(f"[{color}]█[/]")
            elif norm > row - 1:
                frac = norm - (row - 1)
                idx = max(1, min(len(_BLOCKS) - 1, int(frac * 8)))
                row_chars.append(f"[{color}]{_BLOCKS[idx]}[/]")
            else:
                row_chars.append(" ")

        price_at_row = mn + (row / height) * rng
        if row == height or row == 1 or row % 3 == 0:
            label = f" {_fmt_price(price_at_row)} ┤"
        else:
            label = "            │"
        lines.append(label + "".join(row_chars))

    lines.append("            └" + "─" * len(data))

    last = data[-1]
    first = data[0]
    change = last - first
    pct = (change / first) * 100 if first != 0 else 0
    arrow = "▲" if change >= 0 else "▼"
    color = "green" if change >= 0 else "red"
    price_str = _fmt_price(last).strip()
    lines.append(
        f"  [bold]${price_str}[/]  "
        f"[{color}]{arrow} {change:+.4f} ({pct:+.2f}%)[/]  "
        f"[dim]{len(data)} candles[/]"
    )

    return "\n".join(lines)


class ChartPanel(Static):
    """Real-time ASCII price chart."""

    DEFAULT_CSS = """
    ChartPanel {
        height: 100%;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._closes: List[float] = []
        self._opens: List[float] = []
        self._symbol: str = ""
        self._interval: str = ""

    def update_chart(
        self,
        closes: List[float],
        opens: Optional[List[float]] = None,
        symbol: str = "",
        interval: str = "",
    ) -> None:
        self._closes = closes
        self._opens = opens or []
        self._symbol = symbol
        self._interval = interval
        self.refresh()

    def render(self) -> str:
        header = "[bold]PRICE CHART[/]"
        if self._symbol:
            header += f"  [dim]{self._symbol} • {self._interval}[/]"

        if not self._closes:
            return f"{header}\n\n  [dim]Waiting for data...[/]"

        chart = _render_chart(self._closes, self._opens, width=55, height=7)
        return f"{header}\n\n{chart}"
