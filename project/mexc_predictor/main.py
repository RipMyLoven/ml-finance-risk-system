"""
main.py — CLI entry point for the MEXC ONNX prediction pipeline.

Usage
-----
    python main.py BTCUSDT
    python main.py ETHUSDT --timeframe 1h
    python main.py SOLUSDT --model intraday
    python main.py BNBUSDT --all --json

Arguments
---------
    symbol          Trading pair (e.g. BTCUSDT).  Required.

    --timeframe     Override timeframe for the requested model(s).
                    Values: 1m 5m 15m 30m 1h 4h 1d
                    (Default: auto-selected per model)

    --model         Run a single specific model.
                    Values: scalp | intraday | swing | all
                    (Default: all)

    --json          Output raw JSON instead of the human-readable table.

    --no-cache      Disable data caching (always fetch fresh).

    --models-dir    Path to the directory containing .onnx model files.
                    (Default: ../models relative to this script's directory)

    --no-validate   Skip symbol validation against MEXC (saves one API call).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure the package directory is on the path when run directly.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data import MexcClient
from features import (
    IntradayFeatureBuilder,
    RiskFeatureBuilder,
    ScalpFeatureBuilder,
    SwingFeatureBuilder,
)
from model import ModelEnsemble, PredictionResult


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, stream=sys.stderr)
    # Silence noisy third-party loggers.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    """Load config.json or return defaults if file is missing."""
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    logger.warning("config.json not found at %s — using defaults.", config_path)
    return {
        "mexc": {"base_url": "https://api.mexc.com/api/v3", "timeout_seconds": 15,
                 "max_retries": 3, "retry_delay_seconds": 2},
        "timeframes": {
            "scalp":    {"interval": "5m",  "limit": 350},
            "intraday": {"interval": "60m", "limit": 350},
            "swing":    {"interval": "1d",  "limit": 450},
        },
        "models_dir": "../models",
        "cache": {"enabled": True, "ttl_seconds": 60},
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="MEXC ONNX crypto prediction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("symbol", type=str, help="Trading pair, e.g. BTCUSDT.")
    p.add_argument(
        "--timeframe",
        type=str,
        default=None,
        metavar="TF",
        help="Override candle timeframe (e.g. 1h, 5m, 1d).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["scalp", "intraday", "swing", "all"],
        help="Which model(s) to run (default: all).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print raw JSON output.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable in-memory data cache.",
    )
    p.add_argument(
        "--models-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Path to the ONNX models directory.",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip MEXC symbol validation.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    return p


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PredictionPipeline:
    """
    Orchestrates data fetching, feature engineering, and model inference.

    Args:
        config:      Loaded config dict.
        models_dir:  Override for the models directory.
        use_cache:   Whether to use the data cache.
    """

    _TF_INTERVAL_ALIAS = {
        "1h": "60m", "60m": "60m",
        "5m": "5m", "15m": "15m", "30m": "30m",
        "4h": "4h", "1d": "1d", "1D": "1d",
    }

    def __init__(
        self,
        config: dict,
        models_dir: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        self._cfg = config
        self._use_cache = use_cache

        # Resolve models directory.
        raw_dir = models_dir or config.get("models_dir", "../models")
        candidate = Path(raw_dir)
        if not candidate.is_absolute():
            candidate = (_HERE / raw_dir).resolve()
        self._models_dir = candidate

        mexc_cfg = config.get("mexc", {})
        cache_cfg = config.get("cache", {})
        ttl = float(cache_cfg.get("ttl_seconds", 60)) if use_cache else 0.0

        self._client = MexcClient(
            base_url=mexc_cfg.get("base_url", "https://api.mexc.com/api/v3"),
            timeout=int(mexc_cfg.get("timeout_seconds", 15)),
            max_retries=int(mexc_cfg.get("max_retries", 3)),
            retry_delay=float(mexc_cfg.get("retry_delay_seconds", 2.0)),
            cache_ttl=ttl,
        )

        self._ensemble = ModelEnsemble(
            models_dir=self._models_dir,
            num_threads=2,
        )
        self._ensemble.load()

    # ------------------------------------------------------------------

    def run(
        self,
        symbol: str,
        model_filter: str = "all",
        tf_override: Optional[str] = None,
        validate_symbol: bool = True,
    ) -> PredictionResult:
        """
        Run the full pipeline for *symbol*.

        Args:
            symbol:          Trading pair string.
            model_filter:    "scalp" | "intraday" | "swing" | "all".
            tf_override:     If provided, use this interval for all models.
            validate_symbol: Call MEXC ticker to validate symbol first.

        Returns:
            :class:`PredictionResult`.
        """
        symbol = symbol.strip().upper()
        t0 = time.perf_counter()

        # Symbol validation.
        if validate_symbol:
            logger.info("Validating symbol %s …", symbol)
            if not self._client.validate_symbol(symbol):
                raise ValueError(
                    f"'{symbol}' does not appear to be a valid MEXC spot "
                    f"trading pair.  Check the symbol and try again."
                )

        is_btc = symbol == "BTCUSDT"
        run_scalp = model_filter in ("scalp", "all")
        run_intraday = model_filter in ("intraday", "all")
        run_swing = model_filter in ("swing", "all")

        tf_cfg: dict = self._cfg.get("timeframes", {})

        # ---- Scalp ----
        scalp_vec = None
        if run_scalp:
            scalp_vec = self._build_scalp(symbol, tf_override or
                                          tf_cfg.get("scalp", {}).get("interval", "5m"),
                                          tf_cfg.get("scalp", {}).get("limit", 350))

        # ---- Intraday ----
        intraday_vec = None
        intraday_scalar: Optional[dict] = None
        if run_intraday:
            intraday_vec, intraday_scalar = self._build_intraday(
                symbol, is_btc,
                tf_override or tf_cfg.get("intraday", {}).get("interval", "60m"),
                tf_cfg.get("intraday", {}).get("limit", 350),
            )

        # ---- Swing ----
        swing_vec = None
        if run_swing:
            swing_vec = self._build_swing(
                symbol,
                tf_override or tf_cfg.get("swing", {}).get("interval", "1d"),
                tf_cfg.get("swing", {}).get("limit", 450),
            )

        # ---- Risk (based on intraday OHLCV + intraday model output) ----
        risk_vec = None
        if intraday_vec is not None:
            risk_vec = self._build_risk(symbol,
                                        tf_override or "60m",
                                        tf_cfg.get("intraday", {}).get("limit", 350),
                                        intraday_scalar)

        result = self._ensemble.predict(
            symbol=symbol,
            scalp_vec=scalp_vec,
            intraday_vec=intraday_vec,
            swing_vec=swing_vec,
            risk_vec=risk_vec,
        )

        elapsed = time.perf_counter() - t0
        logger.info("Pipeline completed for %s in %.2f s.", symbol, elapsed)
        result._elapsed = elapsed  # type: ignore[attr-defined]
        return result

    # ------------------------------------------------------------------
    # Per-model feature builders
    # ------------------------------------------------------------------

    def _build_scalp(
        self, symbol: str, interval: str, limit: int
    ):
        try:
            logger.info("[scalp] Fetching %s %s …", symbol, interval)
            df = self._client.fetch_ohlcv(symbol, interval, limit,
                                          use_cache=self._use_cache)
            vec, _ = ScalpFeatureBuilder(normalize=True).build(df)
            return vec
        except Exception as exc:
            logger.error("[scalp] Error: %s", exc)
            return None

    def _build_intraday(
        self, symbol: str, is_btc: bool, interval: str, limit: int
    ):
        try:
            logger.info("[intraday] Fetching %s %s …", symbol, interval)
            df = self._client.fetch_ohlcv(symbol, interval, limit,
                                          use_cache=self._use_cache)
            btc_df = None
            if not is_btc:
                try:
                    logger.info("[intraday] Fetching BTCUSDT for correlation …")
                    btc_df = self._client.fetch_ohlcv(
                        "BTCUSDT", interval, limit, use_cache=self._use_cache
                    )
                except Exception as exc:
                    logger.warning(
                        "[intraday] BTC fetch failed (%s) — "
                        "correlation features set to 0.", exc
                    )

            vec, _ = IntradayFeatureBuilder(normalize=True).build(
                df, btc_df=btc_df, is_btc=is_btc
            )

            # Extract scalar outputs to feed the risk model.
            # Run a quick un-normalised pass to get the probabilities on this bar.
            scalar: Optional[dict] = None
            try:
                model = self._ensemble._models.get("intraday")
                if model is not None:
                    out = model.predict_single(vec)
                    scalar = {
                        "P_up": float(out["P_up"][0]),
                        "P_down": float(out["P_down"][0]),
                        "confidence": max(
                            float(out["P_up"][0]),
                            float(out["P_flat"][0]),
                            float(out["P_down"][0]),
                        ),
                    }
            except Exception:
                pass

            return vec, scalar
        except Exception as exc:
            logger.error("[intraday] Error: %s", exc)
            return None, None

    def _build_swing(
        self, symbol: str, interval: str, limit: int
    ):
        try:
            logger.info("[swing] Fetching %s %s …", symbol, interval)
            df = self._client.fetch_ohlcv(symbol, interval, limit,
                                          use_cache=self._use_cache)
            vec, _ = SwingFeatureBuilder(normalize=True).build(df)
            return vec
        except Exception as exc:
            logger.error("[swing] Error: %s", exc)
            return None

    def _build_risk(
        self,
        symbol: str,
        interval: str,
        limit: int,
        intraday_preds: Optional[dict],
    ):
        try:
            df = self._client.fetch_ohlcv(symbol, interval, limit,
                                          use_cache=self._use_cache)
            vec, _ = RiskFeatureBuilder(normalize=True).build(
                df, model_predictions=intraday_preds
            )
            return vec
        except Exception as exc:
            logger.error("[risk] Error: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Formatting — human-readable console output
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_GREY   = "\033[90m"


def _colour_signal(signal: str) -> str:
    if signal in ("UP", "LONG"):
        return f"{_GREEN}{_BOLD}{signal}{_RESET}"
    if signal in ("DOWN", "SHORT"):
        return f"{_RED}{_BOLD}{signal}{_RESET}"
    return f"{_YELLOW}{signal}{_RESET}"


def _colour_risk(level: str) -> str:
    colours = {
        "LOW": _GREEN,
        "MEDIUM": _YELLOW,
        "HIGH": f"\033[91m",        # bright red
        "CRITICAL": f"\033[31m{_BOLD}",
    }
    return f"{colours.get(level, '')}{level}{_RESET}"


def _bar(value: float, width: int = 20, char: str = "█") -> str:
    """Draw a simple ASCII progress bar for probabilities."""
    filled = max(0, min(width, round(value * width)))
    return char * filled + "░" * (width - filled)


def _print_trading_row(
    name: str,
    pred,
    label_width: int = 10,
) -> None:
    if pred is None:
        print(f"  {name:<{label_width}} {_GREY}N/A{_RESET}")
        return

    sig_str = _colour_signal(pred.predicted_class)
    conf_bar = _bar(pred.confidence, width=15)

    print(
        f"  {name.upper():<{label_width}}"
        f"  {sig_str:<24}"
        f"  conf={pred.confidence:.3f} {_CYAN}{conf_bar}{_RESET}"
        f"  P↑={pred.P_up:.3f}  P─={pred.P_flat:.3f}  P↓={pred.P_down:.3f}"
        f"  E[R]={pred.expected_return:+.4f}"
    )


def print_result(result: PredictionResult, elapsed: float = 0.0) -> None:
    """Pretty-print a PredictionResult to stdout."""
    width = 72
    print()
    print("─" * width)
    print(f"  {_BOLD}MEXC ONNX Prediction{_RESET}  │  Symbol: {_BOLD}{result.symbol}{_RESET}")
    print("─" * width)

    # Meta signal.
    meta_color = (
        _GREEN if result.meta_signal == "LONG"
        else (_RED if result.meta_signal == "SHORT" else _YELLOW)
    )
    meta_bar_val = (result.meta_score + 1.0) / 2.0  # map [-1,1] → [0,1]
    meta_bar = _bar(meta_bar_val, width=20)
    print(
        f"  META SIGNAL : {meta_color}{_BOLD}{result.meta_signal:<8}{_RESET}"
        f"  score={result.meta_score:+.4f}  {_CYAN}{meta_bar}{_RESET}"
    )
    print()

    # Individual model rows.
    print(f"  {'Model':<10}  {'Signal':<20}  {'Confidence + Probabilities'}")
    print("  " + "·" * (width - 2))
    _print_trading_row("Scalp",    result.scalp)
    _print_trading_row("Intraday", result.intraday)
    _print_trading_row("Swing",    result.swing)

    # Risk model.
    print()
    if result.risk is not None:
        r = result.risk
        risk_score_bar = _bar(r.risk_score, width=20)
        print(
            f"  {'RISK':<10}  level={_colour_risk(r.risk_level):<22}"
            f"  score={r.risk_score:.4f} {risk_score_bar}"
            f"  max_lev={r.max_leverage_suggested:.0f}×"
        )
    else:
        print(f"  {'RISK':<10}  {_GREY}N/A{_RESET}")

    # Errors / warnings.
    if result.errors:
        print()
        print(f"  {_YELLOW}Warnings:{_RESET}")
        for err in result.errors:
            print(f"    • {err}")

    print()
    print(f"  {_GREY}Elapsed: {elapsed:.2f} s{_RESET}")
    print("─" * width)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)

    # Load config.
    config_path = _HERE / "config.json"
    cfg = _load_config(config_path)

    # Resolve models directory.
    models_dir = args.models_dir
    if models_dir is None:
        raw = cfg.get("models_dir", "../models")
        models_dir_path = Path(raw)
        if not models_dir_path.is_absolute():
            models_dir_path = (_HERE / raw).resolve()
        models_dir = str(models_dir_path)

    try:
        pipeline = PredictionPipeline(
            config=cfg,
            models_dir=models_dir,
            use_cache=not args.no_cache,
        )
    except Exception as exc:
        print(f"\n[ERROR] Failed to initialise pipeline: {exc}", file=sys.stderr)
        logger.exception("Pipeline initialisation failed.")
        return 1

    try:
        result = pipeline.run(
            symbol=args.symbol,
            model_filter=args.model,
            tf_override=args.timeframe,
            validate_symbol=not args.no_validate,
        )
    except ValueError as exc:
        # User-facing errors (bad symbol, not enough data, …).
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}", file=sys.stderr)
        logger.exception("Pipeline run failed.")
        return 1

    elapsed = getattr(result, "_elapsed", 0.0)

    if args.json:
        output = result.to_dict()
        output["elapsed_seconds"] = round(elapsed, 3)
        print(json.dumps(output, indent=2))
    else:
        print_result(result, elapsed=elapsed)

    # Exit codes: 0 = NEUTRAL, 1 = LONG, 2 = SHORT.
    code_map = {"LONG": 1, "SHORT": 2, "NEUTRAL": 0}
    return code_map.get(result.meta_signal, 0)


if __name__ == "__main__":
    sys.exit(main())
