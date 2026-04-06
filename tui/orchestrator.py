"""
orchestrator.py — Async coordinator: data scheduler → inference → TUI state.
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import AppConfig
from .exchange.mexc import MexcAdapter
from .cache.memory import MemoryCache
from .cache.disk import DiskCache
from .state import AppState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPU-bound work (runs in ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _build_features_sync(
    model_type: str,
    df_bytes: bytes,
    btc_df_bytes: Optional[bytes],
    is_btc: bool,
    intraday_preds: Optional[dict],
) -> Optional[bytes]:
    """Build features for one model type. Runs in a child process."""
    import pickle
    from .features.builders import (
        ScalpFeatureBuilder,
        IntradayFeatureBuilder,
        SwingFeatureBuilder,
        RiskFeatureBuilder,
    )

    df = pickle.loads(df_bytes)
    btc_df = pickle.loads(btc_df_bytes) if btc_df_bytes else None

    try:
        if model_type == "scalp":
            vec, _ = ScalpFeatureBuilder(normalize=True).build(df)
        elif model_type == "intraday":
            vec, _ = IntradayFeatureBuilder(normalize=True).build(
                df, btc_df=btc_df, is_btc=is_btc
            )
        elif model_type == "swing":
            vec, _ = SwingFeatureBuilder(normalize=True).build(df)
        elif model_type == "risk":
            vec, _ = RiskFeatureBuilder(normalize=True).build(
                df, model_predictions=intraday_preds
            )
        else:
            return None
        return pickle.dumps(vec)
    except Exception as exc:
        logger.error("Feature build error for %s: %s", model_type, exc)
        return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, config: AppConfig, state: AppState) -> None:
        self._config = config
        self._state = state
        self._exchange = MexcAdapter(
            base_url=config.mexc_base_url,
            timeout=config.mexc_timeout,
            max_retries=config.mexc_max_retries,
            retry_delay=config.mexc_retry_delay,
        )
        self._mem_cache = MemoryCache(max_entries=50)
        self._disk_cache = DiskCache()
        self._ensemble = None  # Lazy loaded
        self._running = False
        self._force_refresh = asyncio.Event()
        self._notify_callback = None  # set by TUI

    def set_notify(self, callback) -> None:
        self._notify_callback = callback

    def _load_models(self) -> None:
        from .model_registry.registry import ModelEnsemble
        self._ensemble = ModelEnsemble(
            models_dir=self._config.models_dir,
            num_threads=2,
        ).load()
        logger.info(
            "Models loaded from %s: %s",
            self._config.models_dir,
            self._ensemble.loaded_models,
        )

    async def start(self) -> None:
        self._load_models()
        self._running = True
        self._state.status = "LIVE"

    async def stop(self) -> None:
        self._running = False
        await self._exchange.close()

    def force_refresh(self) -> None:
        self._force_refresh.set()

    async def run_loop(self) -> None:
        """Main polling loop — runs as background task in the TUI event loop."""
        refresh_interval = 60.0  # default for 5m candles
        while self._running:
            try:
                if not self._state.paused:
                    t0 = time.perf_counter()
                    self._state.status = "COMPUTING"
                    await self._run_inference(self._state.symbol)
                    elapsed = (time.perf_counter() - t0) * 1000
                    self._state.last_inference_ms = elapsed
                    self._state.last_update_time = time.monotonic()
                    self._state.status = "LIVE"

                    if self._notify_callback:
                        self._notify_callback()

                # Wait for next refresh or forced refresh
                try:
                    await asyncio.wait_for(
                        self._force_refresh.wait(),
                        timeout=refresh_interval,
                    )
                    self._force_refresh.clear()
                except asyncio.TimeoutError:
                    pass

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._state.error_count += 1
                self._state.last_error = str(exc)
                self._state.status = "ERROR"
                logger.error("Orchestrator error: %s", exc)
                await asyncio.sleep(5)

    async def switch_symbol(self, symbol: str) -> None:
        symbol = symbol.strip().upper()
        valid = await self._exchange.validate_symbol(symbol)
        if not valid:
            self._state.last_error = f"Invalid symbol: {symbol}"
            self._state.error_count += 1
            return
        self._state.symbol = symbol
        if symbol not in self._state.symbols:
            self._state.symbols.append(symbol)
        self.force_refresh()

    async def search_symbols(self, query: str) -> list[str]:
        return await self._exchange.search_symbols(query)

    # ------------------------------------------------------------------
    # Inference pipeline
    # ------------------------------------------------------------------

    async def _run_inference(self, symbol: str) -> None:
        is_btc = symbol == "BTCUSDT"
        tf = self._config.timeframes

        # Fetch data (with cache)
        scalp_df = await self._fetch_cached(symbol, tf["scalp"].interval, tf["scalp"].limit)
        intraday_df = await self._fetch_cached(symbol, tf["intraday"].interval, tf["intraday"].limit)
        swing_df = await self._fetch_cached(symbol, tf["swing"].interval, tf["swing"].limit)

        # BTC correlation data for non-BTC symbols
        btc_df = None
        if not is_btc:
            try:
                btc_df = await self._fetch_cached("BTCUSDT", tf["intraday"].interval, tf["intraday"].limit)
            except Exception:
                pass

        # Build features (CPU-bound, run synchronously in event loop
        # since ProcessPoolExecutor with pickle has overhead issues with pandas)
        from .features.builders import (
            ScalpFeatureBuilder, IntradayFeatureBuilder,
            SwingFeatureBuilder, RiskFeatureBuilder,
        )

        loop = asyncio.get_event_loop()

        scalp_vec = None
        intraday_vec = None
        swing_vec = None
        risk_vec = None
        intraday_preds = None

        # Build features in executor to not block the event loop
        try:
            scalp_vec, _ = await loop.run_in_executor(
                None, lambda: ScalpFeatureBuilder(normalize=True).build(scalp_df)
            )
        except Exception as exc:
            logger.error("[scalp] Feature error: %s", exc)

        try:
            intraday_vec, _ = await loop.run_in_executor(
                None, lambda: IntradayFeatureBuilder(normalize=True).build(
                    intraday_df, btc_df=btc_df, is_btc=is_btc
                )
            )
        except Exception as exc:
            logger.error("[intraday] Feature error: %s", exc)

        try:
            swing_vec, _ = await loop.run_in_executor(
                None, lambda: SwingFeatureBuilder(normalize=True).build(swing_df)
            )
        except Exception as exc:
            logger.error("[swing] Feature error: %s", exc)

        # Get intraday predictions for risk model input
        if intraday_vec is not None and self._ensemble:
            intraday_preds = self._ensemble.get_intraday_scalars(intraday_vec)

        # Risk features
        try:
            risk_vec, _ = await loop.run_in_executor(
                None, lambda: RiskFeatureBuilder(normalize=True).build(
                    intraday_df, model_predictions=intraday_preds
                )
            )
        except Exception as exc:
            logger.error("[risk] Feature error: %s", exc)

        # Run ensemble inference
        if self._ensemble:
            result = self._ensemble.predict(
                symbol=symbol,
                scalp_vec=scalp_vec,
                intraday_vec=intraday_vec,
                swing_vec=swing_vec,
                risk_vec=risk_vec,
            )
            self._state.result = result
            self._state.add_history(result)

    async def _fetch_cached(
        self, symbol: str, interval: str, limit: int,
    ) -> pd.DataFrame:
        cache_key = f"{symbol}:{interval}:{limit}"

        # Try memory cache
        cached = await self._mem_cache.get(cache_key)
        if cached is not None:
            self._state.last_cache_hit = True
            return cached

        self._state.last_cache_hit = False

        # Try disk cache as fallback
        disk_df = self._disk_cache.get(symbol, interval)

        try:
            df = await self._exchange.fetch_ohlcv(symbol, interval, limit)
        except Exception as exc:
            if disk_df is not None:
                logger.warning("Using disk cache fallback for %s", cache_key)
                return disk_df
            raise

        # Store in caches
        ttl = self._config.cache_ttls.get(interval, 60.0)
        await self._mem_cache.put(cache_key, df, ttl)
        self._disk_cache.put(symbol, interval, df)

        return df
