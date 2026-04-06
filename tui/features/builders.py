"""
features/builders.py — Feature engineering for all four ONNX models.

Replicates the exact feature computation used during training so that
live inference features are in the same distribution as the training features.

Normalization: StandardScaler fitted on the downloaded window, last row
extracted as the inference vector. Standard approach for tree models.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + _EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def _true_range(df: pd.DataFrame) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    n = float(window)
    x = np.arange(window, dtype=np.float64)
    x_sum = x.sum()
    x2_sum = (x ** 2).sum()
    denom = n * x2_sum - x_sum ** 2
    if denom == 0:
        return pd.Series(0.0, index=series.index)

    def _slope_fn(y: np.ndarray) -> float:
        num = n * float(np.dot(x, y)) - x_sum * float(y.sum())
        last = y[-1] if y[-1] != 0 else _EPS
        return (num / denom) / last

    return series.rolling(window).apply(_slope_fn, raw=True)


def _normalise(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    df[feature_names] = df[feature_names].replace([np.inf, -np.inf], np.nan)
    df[feature_names] = df[feature_names].fillna(0.0)
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names].values)
    df[feature_names] = df[feature_names].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Scalp Feature Builder (42 features, 5m bars)
# ---------------------------------------------------------------------------

class ScalpFeatureBuilder:
    FEATURE_NAMES: List[str] = [
        "log_return_1", "log_return_3", "log_return_5", "log_return_10",
        "cum_return_5", "cum_return_10",
        "rsi_5", "rsi_7", "rsi_14", "rsi_slope",
        "micro_vol_3", "micro_vol_5", "micro_vol_10", "vol_ratio_3_10",
        "atr_5", "atr_10", "atr_norm_5", "atr_norm_10", "atr_expansion",
        "volume_ratio_5", "volume_ratio_10", "volume_ratio_20",
        "volume_spike", "volume_trend", "volume_momentum",
        "bar_position", "bar_range_pct", "bar_body_pct",
        "upper_shadow", "lower_shadow",
        "gap", "high_break", "low_break",
        "ema_cross_3_8", "ema_cross_3_13", "ema_cross_8_13",
        "price_vs_ema3", "price_vs_ema8",
        "stoch_k_5", "stoch_d_5",
        "roc_3", "roc_5",
    ]

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def build(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        d = df.copy()
        d = self._add_log_returns(d)
        d = self._add_rsi(d)
        d = self._add_micro_volatility(d)
        d = self._add_volume(d)
        d = self._add_price_action(d)
        d = self._add_momentum(d)
        d = d.dropna(subset=self.FEATURE_NAMES)
        if len(d) == 0:
            raise ValueError("Not enough data for scalp features (need ≥50 candles)")
        if self.normalize:
            d = _normalise(d, self.FEATURE_NAMES)
        last_row = d[self.FEATURE_NAMES].iloc[-1].values.astype(np.float32)
        return last_row, d

    @staticmethod
    def _add_log_returns(d: pd.DataFrame) -> pd.DataFrame:
        lr = np.log(d["close"] / d["close"].shift(1))
        d["log_return_1"] = lr
        d["log_return_3"] = np.log(d["close"] / d["close"].shift(3))
        d["log_return_5"] = np.log(d["close"] / d["close"].shift(5))
        d["log_return_10"] = np.log(d["close"] / d["close"].shift(10))
        d["cum_return_5"] = lr.rolling(5).sum()
        d["cum_return_10"] = lr.rolling(10).sum()
        return d

    @staticmethod
    def _add_rsi(d: pd.DataFrame) -> pd.DataFrame:
        for p in (5, 7, 14):
            d[f"rsi_{p}"] = (_rsi(d["close"], p) - 50.0) / 50.0
        d["rsi_slope"] = d["rsi_14"].diff(3)
        return d

    @staticmethod
    def _add_micro_volatility(d: pd.DataFrame) -> pd.DataFrame:
        lr = d["log_return_1"]
        d["micro_vol_3"] = lr.rolling(3).std()
        d["micro_vol_5"] = lr.rolling(5).std()
        d["micro_vol_10"] = lr.rolling(10).std()
        d["vol_ratio_3_10"] = d["micro_vol_3"] / (d["micro_vol_10"] + _EPS)
        tr = _true_range(d)
        d["atr_5"] = tr.rolling(5).mean()
        d["atr_10"] = tr.rolling(10).mean()
        d["atr_norm_5"] = d["atr_5"] / d["close"]
        d["atr_norm_10"] = d["atr_10"] / d["close"]
        d["atr_expansion"] = d["atr_5"] / (d["atr_10"] + _EPS)
        return d

    @staticmethod
    def _add_volume(d: pd.DataFrame) -> pd.DataFrame:
        v = d["volume"]
        sma5 = v.rolling(5).mean()
        sma10 = v.rolling(10).mean()
        sma20 = v.rolling(20).mean()
        d["volume_ratio_5"] = v / (sma5 + _EPS)
        d["volume_ratio_10"] = v / (sma10 + _EPS)
        d["volume_ratio_20"] = v / (sma20 + _EPS)
        d["volume_spike"] = (d["volume_ratio_10"] > 2.0).astype(float)
        d["volume_trend"] = sma5 / (sma20 + _EPS) - 1.0
        d["volume_momentum"] = v.pct_change(3)
        return d

    @staticmethod
    def _add_price_action(d: pd.DataFrame) -> pd.DataFrame:
        bar_range = d["high"] - d["low"]
        body = (d["close"] - d["open"]).abs()
        d["bar_position"] = (d["close"] - d["low"]) / (bar_range + _EPS)
        d["bar_range_pct"] = bar_range / d["close"]
        d["bar_body_pct"] = body / d["close"]
        candle_top = d[["open", "close"]].max(axis=1)
        candle_bot = d[["open", "close"]].min(axis=1)
        d["upper_shadow"] = (d["high"] - candle_top) / (bar_range + _EPS)
        d["lower_shadow"] = (candle_bot - d["low"]) / (bar_range + _EPS)
        d["gap"] = (d["open"] - d["close"].shift(1)) / (d["close"].shift(1) + _EPS)
        d["high_break"] = (d["high"] > d["high"].shift(1).rolling(5).max()).astype(float)
        d["low_break"] = (d["low"] < d["low"].shift(1).rolling(5).min()).astype(float)
        return d

    @staticmethod
    def _add_momentum(d: pd.DataFrame) -> pd.DataFrame:
        ema3 = d["close"].ewm(span=3, adjust=False).mean()
        ema8 = d["close"].ewm(span=8, adjust=False).mean()
        ema13 = d["close"].ewm(span=13, adjust=False).mean()
        d["ema_cross_3_8"] = (ema3 - ema8) / (ema8 + _EPS)
        d["ema_cross_3_13"] = (ema3 - ema13) / (ema13 + _EPS)
        d["ema_cross_8_13"] = (ema8 - ema13) / (ema13 + _EPS)
        d["price_vs_ema3"] = (d["close"] - ema3) / (ema3 + _EPS)
        d["price_vs_ema8"] = (d["close"] - ema8) / (ema8 + _EPS)
        low5 = d["low"].rolling(5).min()
        high5 = d["high"].rolling(5).max()
        d["stoch_k_5"] = (d["close"] - low5) / (high5 - low5 + _EPS) - 0.5
        d["stoch_d_5"] = d["stoch_k_5"].rolling(3).mean()
        d["roc_3"] = d["close"].pct_change(3)
        d["roc_5"] = d["close"].pct_change(5)
        return d


# ---------------------------------------------------------------------------
# Intraday Feature Builder (47 features, 1h bars)
# ---------------------------------------------------------------------------

class IntradayFeatureBuilder:
    FEATURE_NAMES: List[str] = [
        "price_vs_sma10", "price_vs_sma20", "price_vs_sma50",
        "sma_cross_10_20", "sma_cross_10_50", "sma_cross_20_50",
        "trend_slope_10", "trend_slope_20",
        "ema_trend", "ema_trend_slope",
        "vwap_dev_10", "vwap_dev_20", "vwap_position",
        "funding_change", "funding_change_3",
        "funding_sma", "funding_vs_sma",
        "funding_extreme_pos", "funding_extreme_neg",
        "oi_change", "oi_change_3", "oi_change_10",
        "oi_vs_sma", "oi_momentum", "price_oi_corr",
        "btc_corr_10", "btc_corr_20", "relative_strength", "btc_beta",
        "volatility_10", "volatility_20", "volatility_50", "vol_ratio_10_50",
        "atr_14", "atr_norm",
        "bb_width", "bb_position",
        "rsi_14", "macd_norm", "macd_signal", "macd_hist_slope",
        "adx", "di_diff",
        "volume_ratio", "volume_trend", "obv_slope", "mfi",
    ]

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def build(
        self,
        df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None,
        is_btc: bool = False,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        d = df.copy()
        d = self._add_trend(d)
        d = self._add_vwap(d)
        d = self._add_funding_placeholder(d)
        d = self._add_oi_placeholder(d)
        d = self._add_btc_correlation(d, btc_df, is_btc)
        d = self._add_volatility(d)
        d = self._add_momentum(d)
        d = self._add_volume(d)
        d = d.dropna(subset=self.FEATURE_NAMES)
        if len(d) == 0:
            raise ValueError("Not enough data for intraday features (need ≥60 candles)")
        if self.normalize:
            d = _normalise(d, self.FEATURE_NAMES)
        last_row = d[self.FEATURE_NAMES].iloc[-1].values.astype(np.float32)
        return last_row, d

    @staticmethod
    def _add_trend(d: pd.DataFrame) -> pd.DataFrame:
        sma10 = d["close"].rolling(10).mean()
        sma20 = d["close"].rolling(20).mean()
        sma50 = d["close"].rolling(50).mean()
        d["price_vs_sma10"] = (d["close"] - sma10) / (sma10 + _EPS)
        d["price_vs_sma20"] = (d["close"] - sma20) / (sma20 + _EPS)
        d["price_vs_sma50"] = (d["close"] - sma50) / (sma50 + _EPS)
        d["sma_cross_10_20"] = sma10 / (sma20 + _EPS) - 1.0
        d["sma_cross_10_50"] = sma10 / (sma50 + _EPS) - 1.0
        d["sma_cross_20_50"] = sma20 / (sma50 + _EPS) - 1.0
        d["trend_slope_10"] = _rolling_slope(d["close"], 10)
        d["trend_slope_20"] = _rolling_slope(d["close"], 20)
        ema12 = d["close"].ewm(span=12, adjust=False).mean()
        ema26 = d["close"].ewm(span=26, adjust=False).mean()
        d["ema_trend"] = (ema12 - ema26) / (ema26 + _EPS)
        d["ema_trend_slope"] = d["ema_trend"].diff(5)
        return d

    @staticmethod
    def _add_vwap(d: pd.DataFrame) -> pd.DataFrame:
        tp = (d["high"] + d["low"] + d["close"]) / 3.0
        for w in (10, 20):
            tp_vol = tp * d["volume"]
            vwap_w = tp_vol.rolling(w).sum() / (d["volume"].rolling(w).sum() + _EPS)
            d[f"vwap_dev_{w}"] = (d["close"] - vwap_w) / (vwap_w + _EPS)
        sma20 = d["close"].rolling(20).mean()
        std20 = d["close"].rolling(20).std()
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20
        d["vwap_position"] = (d["close"] - bb_lower) / (bb_upper - bb_lower + _EPS)
        return d

    @staticmethod
    def _add_funding_placeholder(d: pd.DataFrame) -> pd.DataFrame:
        for col in ("funding_change", "funding_change_3", "funding_sma",
                     "funding_vs_sma", "funding_extreme_pos", "funding_extreme_neg"):
            d[col] = 0.0
        return d

    @staticmethod
    def _add_oi_placeholder(d: pd.DataFrame) -> pd.DataFrame:
        for col in ("oi_change", "oi_change_3", "oi_change_10",
                     "oi_vs_sma", "oi_momentum", "price_oi_corr"):
            d[col] = 0.0
        return d

    @staticmethod
    def _add_btc_correlation(
        d: pd.DataFrame, btc_df: Optional[pd.DataFrame], is_btc: bool,
    ) -> pd.DataFrame:
        if is_btc:
            d["btc_corr_10"] = 1.0
            d["btc_corr_20"] = 1.0
            d["relative_strength"] = 0.0
            d["btc_beta"] = 1.0
            return d
        sym_ret = d["close"].pct_change()
        if btc_df is not None and len(btc_df) > 0:
            btc_ret = btc_df["close"].pct_change().reindex(d.index, method="ffill")
        else:
            d["btc_corr_10"] = 0.0
            d["btc_corr_20"] = 0.0
            d["relative_strength"] = 0.0
            d["btc_beta"] = 1.0
            return d
        d["btc_corr_10"] = sym_ret.rolling(10).corr(btc_ret)
        d["btc_corr_20"] = sym_ret.rolling(20).corr(btc_ret)
        d["relative_strength"] = sym_ret.rolling(10).mean() - btc_ret.rolling(10).mean()
        cov = sym_ret.rolling(20).cov(btc_ret)
        var = btc_ret.rolling(20).var()
        d["btc_beta"] = cov / (var + _EPS)
        return d

    @staticmethod
    def _add_volatility(d: pd.DataFrame) -> pd.DataFrame:
        ret = d["close"].pct_change()
        for w in (10, 20, 50):
            d[f"volatility_{w}"] = ret.rolling(w).std()
        d["vol_ratio_10_50"] = d["volatility_10"] / (d["volatility_50"] + _EPS)
        tr = _true_range(d)
        d["atr_14"] = tr.rolling(14).mean()
        d["atr_norm"] = d["atr_14"] / d["close"]
        sma20 = d["close"].rolling(20).mean()
        std20 = d["close"].rolling(20).std()
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20
        d["bb_width"] = (bb_upper - bb_lower) / (sma20 + _EPS)
        d["bb_position"] = (d["close"] - bb_lower) / (bb_upper - bb_lower + _EPS)
        return d

    @staticmethod
    def _add_momentum(d: pd.DataFrame) -> pd.DataFrame:
        d["rsi_14"] = _rsi(d["close"], 14) / 100.0 - 0.5
        ema12 = d["close"].ewm(span=12, adjust=False).mean()
        ema26 = d["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        d["macd_norm"] = macd / (d["close"] + _EPS)
        d["macd_signal"] = (macd - signal) / (d["close"] + _EPS)
        d["macd_hist_slope"] = d["macd_signal"].diff(3)
        high_diff = d["high"].diff()
        low_diff = -d["low"].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
        tr = _true_range(d)
        atr14 = tr.rolling(14).mean()
        plus_di = 100.0 * plus_dm.rolling(14).mean() / (atr14 + _EPS)
        minus_di = 100.0 * minus_dm.rolling(14).mean() / (atr14 + _EPS)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + _EPS)
        d["adx"] = dx.rolling(14).mean() / 100.0
        d["di_diff"] = (plus_di - minus_di) / 100.0
        return d

    @staticmethod
    def _add_volume(d: pd.DataFrame) -> pd.DataFrame:
        vol_sma10 = d["volume"].rolling(10).mean()
        vol_sma20 = d["volume"].rolling(20).mean()
        d["volume_ratio"] = d["volume"] / (vol_sma20 + _EPS)
        d["volume_trend"] = vol_sma10 / (vol_sma20 + _EPS) - 1.0
        obv = (np.sign(d["close"].diff()) * d["volume"]).cumsum()
        obv_mean = obv.rolling(10).mean()
        d["obv_slope"] = obv.diff(10) / (obv_mean.abs() + _EPS)
        tp = (d["high"] + d["low"] + d["close"]) / 3.0
        mf = tp * d["volume"]
        pos_mf = mf.where(tp > tp.shift(1), 0.0)
        neg_mf = mf.where(tp < tp.shift(1), 0.0)
        mfr = pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum() + _EPS)
        d["mfi"] = (100.0 - 100.0 / (1.0 + mfr)) / 100.0 - 0.5
        return d


# ---------------------------------------------------------------------------
# Swing Feature Builder (56 features, 1d bars)
# ---------------------------------------------------------------------------

class SwingFeatureBuilder:
    FEATURE_NAMES: List[str] = [
        # Extra futures-only features (zero-filled on MEXC spot)
        "taker_buy_volume", "taker_buy_quote_volume",
        "mark_close", "basis", "premium_close",
        # Core swing features (56)
        "price_vs_sma20", "price_vs_sma50", "price_vs_sma100", "price_vs_sma200",
        "ma_cross_20_50", "ma_cross_50_100", "ma_cross_50_200", "ma_cross_100_200",
        "golden_cross", "death_cross",
        "sma50_slope", "sma200_slope", "ma_spread",
        "return_20d", "return_50d",
        "regime_bull", "regime_bear", "regime_flat", "regime_duration_norm",
        "volatility_20", "volatility_50", "volatility_100", "volatility_annual",
        "vol_regime", "high_vol_flag", "low_vol_flag",
        "atr_14", "atr_50", "atr_norm", "atr_ratio",
        "bb_width", "drawdown",
        "dominance_change_5d", "dominance_change_20d",
        "dominance_vs_sma", "dominance_rising", "dominance_falling",
        "rsi_14", "rsi_7",
        "macd_norm", "macd_signal",
        "roc_10", "roc_20", "roc_50",
        "price_momentum", "rsi_momentum",
        "range_position_20d", "range_position_50d",
        "dist_to_high_20d", "dist_to_low_20d",
        "breakout_high", "breakdown_low",
        "volume_ratio", "volume_trend",
        "ad_slope", "cmf",
        # Funding/OI swing features (zero-filled on MEXC spot)
        "funding_cumsum_7d", "funding_cumsum_30d",
        "funding_trend", "funding_zscore_swing",
        "funding_extreme_long", "funding_extreme_short",
        "oi_trend_swing", "oi_change_7d", "oi_change_30d",
        "oi_price_div_swing", "oi_zscore_swing",
        "ls_trend_swing", "ls_zscore_swing",
        "crowd_extreme_long", "crowd_extreme_short",
        "taker_trend_swing", "taker_cumsum_7d",
        "premium_trend_swing",
        "basis_persistent_pos", "basis_persistent_neg",
    ]

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    # Features only available from Binance Futures — zero-filled on MEXC spot
    _FUTURES_ONLY = {
        "taker_buy_volume", "taker_buy_quote_volume",
        "mark_close", "basis", "premium_close",
        "funding_cumsum_7d", "funding_cumsum_30d",
        "funding_trend", "funding_zscore_swing",
        "funding_extreme_long", "funding_extreme_short",
        "oi_trend_swing", "oi_change_7d", "oi_change_30d",
        "oi_price_div_swing", "oi_zscore_swing",
        "ls_trend_swing", "ls_zscore_swing",
        "crowd_extreme_long", "crowd_extreme_short",
        "taker_trend_swing", "taker_cumsum_7d",
        "premium_trend_swing",
        "basis_persistent_pos", "basis_persistent_neg",
    }

    def build(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        d = df.copy()
        d = self._add_ma(d)
        d = self._add_regime(d)
        d = self._add_volatility(d)
        d = self._add_btc_dominance_placeholder(d)
        d = self._add_momentum(d)
        d = self._add_support_resistance(d)
        d = self._add_volume(d)
        # Zero-fill futures-only features not available from MEXC spot
        for col in self._FUTURES_ONLY:
            d[col] = 0.0
        core_features = [f for f in self.FEATURE_NAMES if f not in self._FUTURES_ONLY]
        d = d.dropna(subset=core_features)
        if len(d) == 0:
            raise ValueError("Not enough data for swing features (need ≥250 daily candles)")
        if self.normalize:
            d = _normalise(d, self.FEATURE_NAMES)
        last_row = d[self.FEATURE_NAMES].iloc[-1].values.astype(np.float32)
        return last_row, d

    @staticmethod
    def _add_ma(d: pd.DataFrame) -> pd.DataFrame:
        n = len(d)
        mw = max(10, n // 5)

        def _sma(w: int) -> pd.Series:
            return d["close"].rolling(min(w, mw * (w // 20 + 1))).mean()

        sma20 = _sma(20)
        sma50 = _sma(50)
        sma100 = _sma(100)
        sma200 = _sma(200)
        d["price_vs_sma20"] = (d["close"] - sma20) / (sma20 + _EPS)
        d["price_vs_sma50"] = (d["close"] - sma50) / (sma50 + _EPS)
        d["price_vs_sma100"] = (d["close"] - sma100) / (sma100 + _EPS)
        d["price_vs_sma200"] = (d["close"] - sma200) / (sma200 + _EPS)
        d["ma_cross_20_50"] = sma20 / (sma50 + _EPS) - 1.0
        d["ma_cross_50_100"] = sma50 / (sma100 + _EPS) - 1.0
        d["ma_cross_50_200"] = sma50 / (sma200 + _EPS) - 1.0
        d["ma_cross_100_200"] = sma100 / (sma200 + _EPS) - 1.0
        d["golden_cross"] = ((sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))).astype(float)
        d["death_cross"] = ((sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))).astype(float)
        d["sma50_slope"] = sma50.pct_change(5)
        d["sma200_slope"] = sma200.pct_change(10)
        d["ma_spread"] = (sma50 - sma200) / (sma200 + _EPS)
        return d

    @staticmethod
    def _add_regime(d: pd.DataFrame) -> pd.DataFrame:
        n = len(d)
        mw = max(5, n // 10)
        sma50 = d["close"].rolling(min(50, mw * 4)).mean()
        sma200 = d["close"].rolling(min(200, mw * 8)).mean()
        ret20 = d["close"].pct_change(min(20, mw * 2))
        ret50 = d["close"].pct_change(min(50, mw * 4))
        d["return_20d"] = ret20
        d["return_50d"] = ret50
        bull = (d["close"] > sma50) & (sma50 > sma200) & (ret20 > 0.02)
        bear = (d["close"] < sma50) & (sma50 < sma200) & (ret20 < -0.02)
        d["regime_bull"] = bull.astype(float)
        d["regime_bear"] = bear.astype(float)
        d["regime_flat"] = (~bull & ~bear).astype(float)
        regime = pd.Series(0, index=d.index)
        regime[bull] = 1
        regime[bear] = -1
        regime_change = (regime != regime.shift(1)).astype(int)
        grp = regime_change.cumsum()
        d["regime_duration_norm"] = (grp.groupby(grp).cumcount() + 1) / 50.0
        return d

    @staticmethod
    def _add_volatility(d: pd.DataFrame) -> pd.DataFrame:
        n = len(d)
        mw = max(5, n // 10)
        ret = d["close"].pct_change()
        for w in (20, 50, 100):
            d[f"volatility_{w}"] = ret.rolling(min(w, mw * max(1, w // 20))).std()
        d["volatility_annual"] = d["volatility_20"] * np.sqrt(365.0)
        vol_med = d["volatility_50"].rolling(min(100, mw * 8)).median()
        d["vol_regime"] = d["volatility_20"] / (vol_med + _EPS)
        d["high_vol_flag"] = (d["vol_regime"] > 1.5).astype(float)
        d["low_vol_flag"] = (d["vol_regime"] < 0.7).astype(float)
        tr = _true_range(d)
        d["atr_14"] = tr.rolling(min(14, mw)).mean()
        d["atr_50"] = tr.rolling(min(50, mw * 4)).mean()
        d["atr_norm"] = d["atr_14"] / (d["close"] + _EPS)
        d["atr_ratio"] = d["atr_14"] / (d["atr_50"] + _EPS)
        sma20 = d["close"].rolling(20).mean()
        std20 = d["close"].rolling(20).std()
        d["bb_width"] = (4.0 * std20) / (sma20 + _EPS)
        roll_max = d["close"].rolling(min(50, mw * 4)).max()
        d["drawdown"] = (d["close"] - roll_max) / (roll_max + _EPS)
        return d

    @staticmethod
    def _add_btc_dominance_placeholder(d: pd.DataFrame) -> pd.DataFrame:
        for col in ("dominance_change_5d", "dominance_change_20d",
                     "dominance_vs_sma", "dominance_rising", "dominance_falling"):
            d[col] = 0.0
        return d

    @staticmethod
    def _add_momentum(d: pd.DataFrame) -> pd.DataFrame:
        d["rsi_14"] = _rsi(d["close"], 14) / 100.0 - 0.5
        d["rsi_7"] = _rsi(d["close"], 7) / 100.0 - 0.5
        ema12 = d["close"].ewm(span=12, adjust=False).mean()
        ema26 = d["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        d["macd_norm"] = macd / (d["close"] + _EPS)
        d["macd_signal"] = (macd - signal) / (d["close"] + _EPS)
        d["roc_10"] = d["close"].pct_change(10)
        d["roc_20"] = d["close"].pct_change(20)
        d["roc_50"] = d["close"].pct_change(50)
        d["price_momentum"] = d["close"].pct_change(20)
        d["rsi_momentum"] = d["rsi_14"].diff(20)
        return d

    @staticmethod
    def _add_support_resistance(d: pd.DataFrame) -> pd.DataFrame:
        n = len(d)
        mw = max(5, n // 10)
        high20 = d["high"].rolling(min(20, mw * 2)).max()
        low20 = d["low"].rolling(min(20, mw * 2)).min()
        high50 = d["high"].rolling(min(50, mw * 4)).max()
        low50 = d["low"].rolling(min(50, mw * 4)).min()
        d["range_position_20d"] = (d["close"] - low20) / (high20 - low20 + _EPS)
        d["range_position_50d"] = (d["close"] - low50) / (high50 - low50 + _EPS)
        d["dist_to_high_20d"] = (high20 - d["close"]) / (d["close"] + _EPS)
        d["dist_to_low_20d"] = (d["close"] - low20) / (d["close"] + _EPS)
        d["breakout_high"] = (d["close"] > high20.shift(1)).astype(float)
        d["breakdown_low"] = (d["close"] < low20.shift(1)).astype(float)
        return d

    @staticmethod
    def _add_volume(d: pd.DataFrame) -> pd.DataFrame:
        n = len(d)
        mw = max(5, n // 10)
        vol_sma10 = d["volume"].rolling(min(10, mw)).mean()
        vol_sma50 = d["volume"].rolling(min(50, mw * 4)).mean()
        d["volume_ratio"] = vol_sma10 / (vol_sma50 + _EPS)
        d["volume_trend"] = vol_sma10.pct_change(min(10, mw))
        clv = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / (d["high"] - d["low"] + _EPS)
        ad_line = (clv * d["volume"]).cumsum()
        d["ad_slope"] = ad_line.pct_change(min(20, mw * 2))
        mf_vol = clv * d["volume"]
        w = min(20, mw * 2)
        d["cmf"] = mf_vol.rolling(w).sum() / (d["volume"].rolling(w).sum() + _EPS)
        return d


# ---------------------------------------------------------------------------
# Risk Feature Builder (32 features)
# ---------------------------------------------------------------------------

class RiskFeatureBuilder:
    FEATURE_NAMES: List[str] = [
        "volatility_5", "volatility_10", "volatility_20", "volatility_50",
        "vol_ratio_5_20", "vol_ratio_10_50",
        "atr_14", "atr_50", "atr_norm", "atr_ratio",
        "volume_ratio", "volume_trend", "low_liquidity",
        "trend_strength",
        "price_vs_sma20", "price_vs_sma50",
        "regime_bull", "regime_bear", "regime_flat",
        "drawdown", "drawdown_depth",
        "bar_range", "gap", "gap_risk",
        "rsi_14", "rsi_extreme_high", "rsi_extreme_low",
        "model_confidence", "model_P_up", "model_P_down", "model_uncertainty",
        "return_autocorr",
    ]

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def build(
        self,
        df: pd.DataFrame,
        model_predictions: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        d = df.copy()
        d = self._add_volatility(d)
        d = self._add_liquidity(d)
        d = self._add_regime(d)
        d = self._add_price_action(d)
        d = self._add_rsi_extremes(d)
        d = self._add_model_features(d, model_predictions)
        d = self._add_autocorr(d)
        d = d.dropna(subset=self.FEATURE_NAMES)
        if len(d) == 0:
            raise ValueError("Not enough data for risk features (need ≥60 candles)")
        if self.normalize:
            d = _normalise(d, self.FEATURE_NAMES)
        last_row = d[self.FEATURE_NAMES].iloc[-1].values.astype(np.float32)
        return last_row, d

    @staticmethod
    def _add_volatility(d: pd.DataFrame) -> pd.DataFrame:
        ret = d["close"].pct_change()
        for w in (5, 10, 20, 50):
            d[f"volatility_{w}"] = ret.rolling(w).std()
        d["vol_ratio_5_20"] = d["volatility_5"] / (d["volatility_20"] + _EPS)
        d["vol_ratio_10_50"] = d["volatility_10"] / (d["volatility_50"] + _EPS)
        tr = _true_range(d)
        d["atr_14"] = tr.rolling(14).mean()
        d["atr_50"] = tr.rolling(50).mean()
        d["atr_norm"] = d["atr_14"] / (d["close"] + _EPS)
        d["atr_ratio"] = d["atr_14"] / (d["atr_50"] + _EPS)
        return d

    @staticmethod
    def _add_liquidity(d: pd.DataFrame) -> pd.DataFrame:
        vol_sma20 = d["volume"].rolling(20).mean()
        vol_sma50 = d["volume"].rolling(50).mean()
        d["volume_ratio"] = d["volume"] / (vol_sma20 + _EPS)
        d["volume_trend"] = vol_sma20 / (vol_sma50 + _EPS) - 1.0
        d["low_liquidity"] = (d["volume_ratio"] < 0.5).astype(float)
        return d

    @staticmethod
    def _add_regime(d: pd.DataFrame) -> pd.DataFrame:
        sma20 = d["close"].rolling(20).mean()
        sma50 = d["close"].rolling(50).mean()
        d["trend_strength"] = (sma20 - sma50) / (sma50 + _EPS)
        d["price_vs_sma20"] = (d["close"] - sma20) / (sma20 + _EPS)
        d["price_vs_sma50"] = (d["close"] - sma50) / (sma50 + _EPS)
        ret20 = d["close"].pct_change(20)
        d["regime_bull"] = (ret20 > 0.05).astype(float)
        d["regime_bear"] = (ret20 < -0.05).astype(float)
        d["regime_flat"] = ((ret20 >= -0.05) & (ret20 <= 0.05)).astype(float)
        roll_max = d["close"].rolling(50).max()
        d["drawdown"] = (d["close"] - roll_max) / (roll_max + _EPS)
        d["drawdown_depth"] = d["drawdown"].abs().rolling(10).max()
        return d

    @staticmethod
    def _add_price_action(d: pd.DataFrame) -> pd.DataFrame:
        d["bar_range"] = (d["high"] - d["low"]) / (d["close"] + _EPS)
        d["gap"] = (d["open"] - d["close"].shift(1)).abs() / (d["close"].shift(1) + _EPS)
        d["gap_risk"] = (d["gap"] > 0.01).astype(float)
        return d

    @staticmethod
    def _add_rsi_extremes(d: pd.DataFrame) -> pd.DataFrame:
        rsi_raw = _rsi(d["close"], 14)
        d["rsi_14"] = rsi_raw / 100.0 - 0.5
        d["rsi_extreme_high"] = (rsi_raw > 80).astype(float)
        d["rsi_extreme_low"] = (rsi_raw < 20).astype(float)
        return d

    @staticmethod
    def _add_model_features(
        d: pd.DataFrame, preds: Optional[Dict[str, float]],
    ) -> pd.DataFrame:
        if preds is None:
            d["model_confidence"] = 0.5
            d["model_P_up"] = 0.33
            d["model_P_down"] = 0.33
            d["model_uncertainty"] = 0.5
        else:
            p_up = float(preds.get("P_up", 0.33))
            p_down = float(preds.get("P_down", 0.33))
            conf = float(preds.get("confidence", max(p_up, p_down)))
            d["model_confidence"] = conf
            d["model_P_up"] = p_up
            d["model_P_down"] = p_down
            d["model_uncertainty"] = 1.0 - conf
        return d

    @staticmethod
    def _add_autocorr(d: pd.DataFrame) -> pd.DataFrame:
        ret = d["close"].pct_change()
        d["return_autocorr"] = ret.rolling(20).apply(
            lambda x: float(pd.Series(x).autocorr()) if len(x) > 2 else 0.0,
            raw=False,
        )
        return d
