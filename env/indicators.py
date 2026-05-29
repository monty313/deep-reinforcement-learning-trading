"""
env/indicators.py
Compute all technical indicators required by the spec for a single OHLCV DataFrame.
Covers STRAT-001 through STRAT-011, extra indicators, and time/calendar features.
All indicators are computed on whatever TF the input DataFrame represents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib


# ── helpers ───────────────────────────────────────────────────────────────────

def _sma(series: pd.Series, period: int, shift: int = 0) -> pd.Series:
    """Simple moving average with optional forward shift (positive = look further back)."""
    if period <= 1:
        s = series.astype(float)
    else:
        s = series.astype(float).rolling(window=period, min_periods=1).mean()
    if shift:
        s = s.shift(shift)
    return s


def _bb_bands(series: pd.Series, period: int, nbdev: float = 2.0):
    """Returns (upper, middle, lower) as pd.Series."""
    upper, middle, lower = talib.BBANDS(
        series.values.astype(float), timeperiod=period,
        nbdevup=nbdev, nbdevdn=nbdev, matype=0
    )
    idx = series.index
    return (
        pd.Series(upper, index=idx),
        pd.Series(middle, index=idx),
        pd.Series(lower, index=idx),
    )


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns [open, high, low, close, volume],
    return a new DataFrame with all indicator columns appended.
    No rows are dropped — NaN values in early rows are kept.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    out    = pd.DataFrame(index=df.index)

    # ── 3.1 Price & volatility ────────────────────────────────────────────────
    out["open"]   = open_
    out["high"]   = high
    out["low"]    = low
    out["close"]  = close
    out["volume"] = df["volume"]

    atr14 = pd.Series(talib.ATR(high.values.astype(float),
                                low.values.astype(float),
                                close.values.astype(float), timeperiod=14),
                      index=df.index)
    out["atr14"]            = atr14
    out["atr14_sma1_sh2"]   = _sma(atr14, 1, shift=2)

    # ── STRAT-001 (Regime Pulse Tracker) ──────────────────────────────────────
    bb200_u, bb200_m, bb200_l = _bb_bands(close, 200)
    bb20_u,  bb20_m,  bb20_l  = _bb_bands(close, 20)

    out["bb200_upper"] = bb200_u
    out["bb200_mid"]   = bb200_m
    out["bb200_lower"] = bb200_l
    out["bb20_upper"]  = bb20_u
    out["bb20_mid"]    = bb20_m
    out["bb20_lower"]  = bb20_l
    out["rsi7"]        = pd.Series(talib.RSI(close.values.astype(float), timeperiod=7), index=df.index)

    # ── STRAT-002 (CCI Surge Sentinel) ────────────────────────────────────────
    out["cci30"]  = pd.Series(talib.CCI(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=30),  index=df.index)
    out["cci100"] = pd.Series(talib.CCI(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=100), index=df.index)
    # rsi7 already added

    # ── STRAT-003 (CCI Trinity Vanguard) ──────────────────────────────────────
    out["cci14"]  = pd.Series(talib.CCI(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=14),  index=df.index)
    out["cci900"] = pd.Series(talib.CCI(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=900), index=df.index)
    out["cci14_sma20"]  = _sma(out["cci14"],  20)
    out["cci100_sma20"] = _sma(out["cci100"], 20)
    out["cci900_sma20"] = _sma(out["cci900"], 20)

    # ── STRAT-004 (SMA Stack Prophet) ─────────────────────────────────────────
    out["sma50"]        = _sma(close, 50)
    out["sma4"]         = _sma(close, 4)
    out["sma4_sh4"]     = _sma(close, 4, shift=4)

    # ── STRAT-005 (SMA Reversion Rally) ──────────────────────────────────────
    out["sma30"]  = _sma(close, 30)
    out["rsi5"]   = pd.Series(talib.RSI(close.values.astype(float), timeperiod=5), index=df.index)
    # sma50 already added

    # ── STRAT-006 (Dual Momentum-Volatility Filter) ───────────────────────────
    adx14 = pd.Series(talib.ADX(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=14), index=df.index)
    out["adx14"]          = adx14
    out["adx14_sma1_sh5"] = _sma(adx14, 1, shift=5)
    out["atr14_sma1_sh5"] = _sma(atr14, 1, shift=5)

    # ── STRAT-007 (SMA Fan Accord) ────────────────────────────────────────────
    # sma50 & sma4 already added
    out["sma4_sh1"] = _sma(close, 4, shift=1)
    out["sma4_sh2"] = _sma(close, 4, shift=2)
    out["sma4_sh3"] = _sma(close, 4, shift=3)
    # sma4_sh4 already added

    # ── STRAT-008 (CCI BB Outbreak Hunter) ────────────────────────────────────
    out["cci300"] = pd.Series(talib.CCI(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=300), index=df.index)
    for cci_col in ["cci30", "cci100", "cci300"]:
        bb_u, bb_m, bb_l = _bb_bands(out[cci_col], period=14, nbdev=1.0)
        out[f"{cci_col}_bb14_upper"] = bb_u
        out[f"{cci_col}_bb14_mid"]   = bb_m
        out[f"{cci_col}_bb14_lower"] = bb_l

    # ── STRAT-009 (Index Open Breakout) ──────────────────────────────────────
    out["sma200"] = _sma(close, 200)  # used for both STRAT-009 and STRAT-010

    # ── STRAT-010 (Red News + COT Bias) ──────────────────────────────────────
    # sma200 already added

    # ── STRAT-011 (Shifted CCI Momentum Aligner) ──────────────────────────────
    out["cci140"] = pd.Series(talib.CCI(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=140), index=df.index)
    out["cci140_sma1_sh4"] = _sma(out["cci140"], 1, shift=4)
    # cci14 already added

    # ── 3.3 Extra indicators ──────────────────────────────────────────────────
    out["atr14_sma1_sh2"]   = out["atr14_sma1_sh2"]   # already computed above
    out["cci30_sma1_sh2"]   = _sma(out["cci30"],  1, shift=2)
    out["cci100_sma1_sh2"]  = _sma(out["cci100"], 1, shift=2)

    # SMA(4) shift+8 on BB(20) upper and lower bands
    out["bb20_upper_sma4_sh8"]  = _sma(bb20_u, 4, shift=8)
    out["bb20_lower_sma4_sh8"]  = _sma(bb20_l, 4, shift=8)
    # SMA(4) shift+8 on BB(200) upper and lower bands
    out["bb200_upper_sma4_sh8"] = _sma(bb200_u, 4, shift=8)
    out["bb200_lower_sma4_sh8"] = _sma(bb200_l, 4, shift=8)

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append CET/CEST cyclical time features.
    Assumes df.index is already in CET/CEST (Europe/Berlin).
    """
    out = df.copy()
    hour = df.index.hour + df.index.minute / 60.0
    dow  = df.index.dayofweek  # 0=Monday
    out["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    out["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    out["sin_dow"]  = np.sin(2 * np.pi * dow / 7)
    out["cos_dow"]  = np.cos(2 * np.pi * dow / 7)
    return out


def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: indicators + time features for one TF DataFrame."""
    ind = compute_indicators(df)
    return add_time_features(ind)


# ── phase mask helpers ────────────────────────────────────────────────────────

def _aligned(val, sma_val) -> int:
    """Return +1 if above, -1 if below, 0 if NaN/equal."""
    if pd.isna(val) or pd.isna(sma_val):
        return 0
    if val > sma_val:
        return 1
    if val < sma_val:
        return -1
    return 0


def can_open_trade_phase1(row_1m: pd.Series, row_15m: pd.Series) -> bool:
    """
    Phase 1 – CCI alignment on 1m and 15m.
    New trade allowed only when CCI30 and CCI100 are both above their
    SMA(1,+2) on both TFs, or both below on both TFs.
    """
    dir_1m = (
        _aligned(row_1m.get("cci30"), row_1m.get("cci30_sma1_sh2")) ==
        _aligned(row_1m.get("cci100"), row_1m.get("cci100_sma1_sh2"))
        and _aligned(row_1m.get("cci30"), row_1m.get("cci30_sma1_sh2")) != 0
    )
    dir_15m = (
        _aligned(row_15m.get("cci30"), row_15m.get("cci30_sma1_sh2")) ==
        _aligned(row_15m.get("cci100"), row_15m.get("cci100_sma1_sh2"))
        and _aligned(row_15m.get("cci30"), row_15m.get("cci30_sma1_sh2")) != 0
    )
    if not (dir_1m and dir_15m):
        return False
    # same direction on both TFs
    d1 = _aligned(row_1m.get("cci30"), row_1m.get("cci30_sma1_sh2"))
    d15 = _aligned(row_15m.get("cci30"), row_15m.get("cci30_sma1_sh2"))
    return d1 == d15


def must_be_in_trade_phase2(row_1m: pd.Series, row_1h: pd.Series) -> bool:
    """
    Phase 2 – SMA on BB upper band: both TFs agree in direction.
    Returns True when agent MUST hold an active trade.
    """
    def _dir(row):
        price = row.get("close")
        sma   = row.get("bb20_upper_sma4_sh8")
        return _aligned(price, sma)

    d1 = _dir(row_1m)
    d1h = _dir(row_1h)
    return d1 != 0 and d1 == d1h


def must_be_in_trade_phase3(row_1m: pd.Series, row_15m: pd.Series) -> bool:
    """
    Phase 3 – BB midlines: price above/below BOTH BB200_mid and BB20_mid
    on both 1m and 15m in the same direction.
    """
    def _dir(row):
        price    = row.get("close")
        bb200_m  = row.get("bb200_mid")
        bb20_m   = row.get("bb20_mid")
        if any(pd.isna(v) for v in [price, bb200_m, bb20_m]):
            return 0
        if price > bb200_m and price > bb20_m:
            return 1
        if price < bb200_m and price < bb20_m:
            return -1
        return 0

    d1  = _dir(row_1m)
    d15 = _dir(row_15m)
    return d1 != 0 and d1 == d15
