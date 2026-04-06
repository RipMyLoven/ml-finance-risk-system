"""
__main__.py — Entry point for the TUI.

Usage:
    py -m tui BTCUSDT
    py -m tui BTCUSDT ETHUSDT SOLUSDT
    py -m tui BTCUSDT --headless --json
    py -m tui --check-models
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import AppConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tui",
        description="ML Finance Risk System — Real-Time TUI",
    )
    p.add_argument(
        "symbols",
        nargs="*",
        default=["BTCUSDT"],
        help="Trading pair(s) to track (e.g. BTCUSDT ETHUSDT)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="No TUI — print JSON to stdout and exit",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON (used with --headless)",
    )
    p.add_argument(
        "--check-models",
        action="store_true",
        help="Verify all ONNX models and exit",
    )
    p.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Path to ONNX models directory",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _setup_logging(level: str) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format=fmt,
        stream=sys.stderr,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def _check_models(config: AppConfig) -> int:
    """Pre-flight model check. Returns exit code."""
    import numpy as np
    from .model_registry.registry import ModelEnsemble

    print(f"Models directory: {config.models_dir}")
    print()

    try:
        ensemble = ModelEnsemble(config.models_dir, num_threads=2).load()
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 2

    ok = 0
    total = 4
    for name in ["scalp", "intraday", "swing", "risk"]:
        model = ensemble._models.get(name)
        if model is None:
            print(f"  [FAIL] {name}: not loaded")
            continue

        # Try a random inference
        import time
        dummy = np.random.randn(model.n_features).astype(np.float32)
        t0 = time.perf_counter()
        try:
            model.predict_single(dummy)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [OK]   {name:<12} features={model.n_features:<4} latency={elapsed:.1f}ms")
            ok += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: inference error — {exc}")

    print()
    if ok == total:
        print(f"OK ({ok}/{total} models)")
        return 0
    elif ok > 0:
        print(f"PARTIAL ({ok}/{total} models)")
        return 1
    else:
        print(f"FAIL (0/{total} models)")
        return 2


def _run_headless(config: AppConfig, symbols: list[str], as_json: bool) -> None:
    """Single-shot prediction without TUI."""
    import asyncio
    import time
    from .exchange.mexc import MexcAdapter
    from .features.builders import (
        ScalpFeatureBuilder, IntradayFeatureBuilder,
        SwingFeatureBuilder, RiskFeatureBuilder,
    )
    from .model_registry.registry import ModelEnsemble

    ensemble = ModelEnsemble(config.models_dir, num_threads=2).load()

    async def _predict(symbol: str) -> dict:
        adapter = MexcAdapter(
            base_url=config.mexc_base_url,
            timeout=config.mexc_timeout,
        )
        try:
            tf = config.timeframes
            is_btc = symbol == "BTCUSDT"

            scalp_df = await adapter.fetch_ohlcv(symbol, tf["scalp"].interval, tf["scalp"].limit)
            intraday_df = await adapter.fetch_ohlcv(symbol, tf["intraday"].interval, tf["intraday"].limit)
            swing_df = await adapter.fetch_ohlcv(symbol, tf["swing"].interval, tf["swing"].limit)

            btc_df = None
            if not is_btc:
                btc_df = await adapter.fetch_ohlcv("BTCUSDT", tf["intraday"].interval, tf["intraday"].limit)

            scalp_vec, _ = ScalpFeatureBuilder(normalize=True).build(scalp_df)
            intraday_vec, _ = IntradayFeatureBuilder(normalize=True).build(
                intraday_df, btc_df=btc_df, is_btc=is_btc
            )
            swing_vec, _ = SwingFeatureBuilder(normalize=True).build(swing_df)

            intraday_preds = ensemble.get_intraday_scalars(intraday_vec)

            risk_vec, _ = RiskFeatureBuilder(normalize=True).build(
                intraday_df, model_predictions=intraday_preds
            )

            result = ensemble.predict(
                symbol=symbol,
                scalp_vec=scalp_vec,
                intraday_vec=intraday_vec,
                swing_vec=swing_vec,
                risk_vec=risk_vec,
            )
            return result.to_dict()
        finally:
            await adapter.close()

    for symbol in symbols:
        symbol = symbol.strip().upper()
        t0 = time.perf_counter()
        result = asyncio.run(_predict(symbol))
        elapsed = time.perf_counter() - t0

        if as_json:
            result["elapsed_seconds"] = round(elapsed, 3)
            print(json.dumps(result, indent=2))
        else:
            _print_human(result, elapsed)


def _print_human(result: dict, elapsed: float) -> None:
    """Pretty-print a prediction result."""
    R = "\033[0m"
    B = "\033[1m"
    G = "\033[32m"
    RED = "\033[31m"
    Y = "\033[33m"
    C = "\033[36m"
    DIM = "\033[90m"

    w = 72
    print()
    print("─" * w)
    print(f"  {B}MEXC ONNX Prediction{R}  │  Symbol: {B}{result['symbol']}{R}  │  {elapsed:.2f}s")
    print("─" * w)

    sig = result["meta_signal"]
    clr = G if sig == "LONG" else (RED if sig == "SHORT" else Y)
    print(f"  META: {clr}{B}{sig:<8}{R}  score={result['meta_score']:+.4f}")
    print()

    print(f"  {'Model':<10} {'Signal':<8} {'Conf':>6} {'P↑':>6} {'P─':>6} {'P↓':>6} {'E[R]':>8}")
    print("  " + "·" * (w - 2))

    for name in ("scalp", "intraday", "swing"):
        pred = result.get(name)
        if pred is None:
            print(f"  {name.upper():<10} {DIM}N/A{R}")
            continue
        sig = pred["signal"]
        clr = G if sig == "UP" else (RED if sig == "DOWN" else Y)
        print(
            f"  {name.upper():<10} {clr}{B}{sig:<8}{R} "
            f"{pred['confidence']:6.3f} {pred['P_up']:6.3f} "
            f"{pred['P_flat']:6.3f} {pred['P_down']:6.3f} "
            f"{pred['expected_return']:+8.4f}"
        )

    risk = result.get("risk")
    print()
    if risk:
        lvl = risk["risk_level"]
        clr = G if lvl == "LOW" else (Y if lvl == "MEDIUM" else RED)
        print(f"  RISK: {clr}{B}{lvl}{R}  score={risk['risk_score']:.4f}  max_lev={risk['max_leverage']}×")
    else:
        print(f"  RISK: {DIM}N/A{R}")

    if result.get("errors"):
        print(f"\n  {Y}Warnings:{R}")
        for err in result["errors"]:
            print(f"    • {err}")
    print()


def main() -> None:
    args = _build_parser().parse_args()
    _setup_logging(args.log_level)

    config = AppConfig.load()

    # Override models dir
    if args.models_dir:
        config.models_dir = Path(args.models_dir).resolve()

    # Check models mode
    if args.check_models:
        sys.exit(_check_models(config))

    # Headless mode
    if args.headless:
        _run_headless(config, args.symbols, args.json)
        return

    # TUI mode
    from .app import TradingTUI
    app = TradingTUI(
        config=config,
        initial_symbols=[s.upper() for s in args.symbols],
    )
    app.run()


if __name__ == "__main__":
    main()
