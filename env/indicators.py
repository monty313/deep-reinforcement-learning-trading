"""
env/indicators.py
Compute all technical indicators required by the spec for a single OHLCV DataFrame.
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


def _bb_bands(series: pd.Series, period: int, nbdev: float = 1.0):
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


def compute_indicators(df: pd.DataFrame, compute_heavy: bool = False) -> pd.DataFrame:
    """
    Given a DataFrame with columns [open, high, low, close, volume],
    return a new DataFrame with all indicator columns appended.
    No rows are dropped — NaN values in early rows are kept.
    compute_heavy: if False, skips cci900 (slow on 1m data).
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    out    = pd.DataFrame(index=df.index)
    # Cache float64 arrays once (talib requires float64)
    c = close.values.astype(np.float64)
    h = high.values.astype(np.float64)
    l = low.values.astype(np.float64)

    # ── Price & volatility ────────────────────────────────────────────────────
    out["open"]   = open_
    out["high"]   = high
    out["low"]    = low
    out["close"]  = close
    out["volume"] = df["volume"]

    atr14 = pd.Series(talib.ATR(h, l, c, timeperiod=14), index=df.index)
    atr45 = pd.Series(talib.ATR(h, l, c, timeperiod=45), index=df.index)
    out["atr14"]            = atr14
    out["atr45"]            = atr45
    out["atr14_sma1_sh8"]   = _sma(atr14, 1, shift=8)
    out["atr45_sma1_sh8"]   = _sma(atr45, 1, shift=8)

    # ── Bollinger Bands (nbdev=1.0 everywhere) ────────────────────────────────
    bb200_u, bb200_m, bb200_l = _bb_bands(close, 200, nbdev=1.0)
    bb20_u,  bb20_m,  bb20_l  = _bb_bands(close, 20,  nbdev=1.0)

    out["bb200_upper"] = bb200_u
    out["bb200_mid"]   = bb200_m
    out["bb200_lower"] = bb200_l
    out["bb20_upper"]  = bb20_u
    out["bb20_mid"]    = bb20_m
    out["bb20_lower"]  = bb20_l

    # ── RSI ───────────────────────────────────────────────────────────────────
    out["rsi7"]  = pd.Series(talib.RSI(c, timeperiod=7), index=df.index)
    out["rsi5"]  = pd.Series(talib.RSI(close.values.astype(float), timeperiod=5), index=df.index)

    # ── CCI indicators ────────────────────────────────────────────────────────
    out["cci14"]  = pd.Series(talib.CCI(h, l, c, timeperiod=14),  index=df.index)
    out["cci30"]  = pd.Series(talib.CCI(h, l, c, timeperiod=30),  index=df.index)
    out["cci100"] = pd.Series(talib.CCI(h, l, c, timeperiod=100), index=df.index)
    out["cci140"] = pd.Series(talib.CCI(h, l, c, timeperiod=140), index=df.index)
    out["cci300"] = pd.Series(talib.CCI(h, l, c, timeperiod=300), index=df.index)

    if compute_heavy:
        out["cci900"]      = pd.Series(talib.CCI(h, l, c, timeperiod=900), index=df.index)
        out["cci900_sma20"] = _sma(out["cci900"], 20)
    else:
        out["cci900"]      = np.nan
        out["cci900_sma20"] = np.nan

    # CCI shifted SMAs — shift=8 used by Phase 1
    out["cci30_sma1_sh8"]   = _sma(out["cci30"],  1, shift=8)
    out["cci100_sma1_sh8"]  = _sma(out["cci100"], 1, shift=8)
    out["cci14_sma20"]      = _sma(out["cci14"],  20)
    out["cci100_sma20"]     = _sma(out["cci100"], 20)
    out["cci140_sma1_sh4"]  = _sma(out["cci140"], 1, shift=4)

    # CCI BB bands (for STRAT-008)
    for cci_col in ["cci30", "cci100", "cci300"]:
        bb_u, bb_m, bb_l = _bb_bands(out[cci_col], period=14, nbdev=1.0)
        out[f"{cci_col}_bb14_upper"] = bb_u
        out[f"{cci_col}_bb14_mid"]   = bb_m
        out[f"{cci_col}_bb14_lower"] = bb_l

    # ── SMA indicators ────────────────────────────────────────────────────────
    out["sma4"]      = _sma(close, 4)
    out["sma4_sh1"]  = _sma(close, 4, shift=1)
    out["sma4_sh2"]  = _sma(close, 4, shift=2)
    out["sma4_sh3"]  = _sma(close, 4, shift=3)
    out["sma4_sh4"]  = _sma(close, 4, shift=4)
    out["sma30"]     = _sma(close, 30)
    out["sma50"]     = _sma(close, 50)
    out["sma200"]    = _sma(close, 200)

    # SMA(2) stack for Phase 5
    out["sma2_sh0"]  = _sma(close, 2, shift=0)
    out["sma2_sh1"]  = _sma(close, 2, shift=1)
    out["sma2_sh2"]  = _sma(close, 2, shift=2)
    out["sma2_sh3"]  = _sma(close, 2, shift=3)
    out["sma2_sh4"]  = _sma(close, 2, shift=4)

    # ── High/Low SMA bands for Phase 2 & 3 ───────────────────────────────────
    out["high_sma4_sh8"] = _sma(high, 4, shift=8)
    out["low_sma4_sh8"]  = _sma(low,  4, shift=8)

    # ── ADX ───────────────────────────────────────────────────────────────────
    adx14 = pd.Series(talib.ADX(h, l, c, timeperiod=14), index=df.index)
    out["adx14"] = adx14

    # ── BB SMA shift references (kept for backward compat) ───────────────────
    out["bb20_upper_sma4_sh8"]  = _sma(bb20_u,  4, shift=8)
    out["bb20_lower_sma4_sh8"]  = _sma(bb20_l,  4, shift=8)
    out["bb200_upper_sma4_sh8"] = _sma(bb200_u, 4, shift=8)
    out["bb200_lower_sma4_sh8"] = _sma(bb200_l, 4, shift=8)

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append CET/CEST cyclical time features in-place (no full copy).
    Assumes df.index is already in CET/CEST (Europe/Berlin).
    """
    hour = df.index.hour + df.index.minute / 60.0
    dow  = df.index.dayofweek  # 0=Monday
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    df["sin_dow"]  = np.sin(2 * np.pi * dow / 7)
    df["cos_dow"]  = np.cos(2 * np.pi * dow / 7)
    return df


def build_feature_df(df: pd.DataFrame, compute_heavy: bool = False) -> pd.DataFrame:
    """Full pipeline: indicators + time features for one TF DataFrame."""
    ind = compute_indicators(df, compute_heavy=compute_heavy)
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


# ── Phase 0: CCI Extreme Gate ─────────────────────────────────────────────────
def phase0_cci_extreme(row_1m: pd.Series, row_15m: pd.Series) -> bool:
    """
    force_in_and_gate: CCI30 AND CCI100 both > +100 OR both < -100
    on BOTH 1m and 15m in the same direction.
    """
    def _extreme(row):
        c30  = row.get("cci30")
        c100 = row.get("cci100")
        if pd.isna(c30) or pd.isna(c100):
            return (False, 0)
        if c30 > 100 and c100 > 100:
            return (True,  1)
        if c30 < -100 and c100 < -100:
            return (True, -1)
        return (False, 0)

    a1,  d1  = _extreme(row_1m)
    a15, d15 = _extreme(row_15m)
    return a1 and a15 and d1 == d15


# ── Phase 1: CCI Directional Alignment ───────────────────────────────────────
def phase1_cci_align(row_1m: pd.Series, row_15m: pd.Series) -> bool:
    """
    open_gate: CCI30 and CCI100 each above/below their SMA(1, shift=8)
    on BOTH 1m and 15m, all four signals agreeing.
    """
    def _dir(row):
        d30  = _aligned(row.get("cci30"),  row.get("cci30_sma1_sh8"))
        d100 = _aligned(row.get("cci100"), row.get("cci100_sma1_sh8"))
        if d30 == 0 or d100 == 0:
            return 0
        if d30 == d100:
            return d30
        return 0

    d1  = _dir(row_1m)
    d15 = _dir(row_15m)
    if d1 == 0 or d15 == 0:
        return False
    return d1 == d15


# ── Phase 2: Lagged High/Low SMA Band — Trend ─────────────────────────────────
def phase2_hilo_trend(row_1m: pd.Series, row_30m: pd.Series) -> bool:
    """
    force_in_and_gate: close above BOTH high_sma4_sh8 and low_sma4_sh8
    OR below BOTH, on both 1m and 30m in the same direction.
    """
    def _dir(row):
        price    = row.get("close")
        sma_high = row.get("high_sma4_sh8")
        sma_low  = row.get("low_sma4_sh8")
        if any(pd.isna(v) for v in [price, sma_high, sma_low]):
            return 0
        if price > sma_high and price > sma_low:
            return  1
        if price < sma_high and price < sma_low:
            return -1
        return 0

    d1  = _dir(row_1m)
    d30 = _dir(row_30m)
    return d1 != 0 and d1 == d30


# ── Phase 3: Lagged High/Low SMA Band — Counter-TF Reversion ─────────────────
def phase3_hilo_counter(row_1m: pd.Series, row_15m: pd.Series) -> bool:
    """
    force_in_and_gate: 1m and 15m are on OPPOSITE sides of the
    high_sma4_sh8 / low_sma4_sh8 band.
    """
    def _dir(row):
        price    = row.get("close")
        sma_high = row.get("high_sma4_sh8")
        sma_low  = row.get("low_sma4_sh8")
        if any(pd.isna(v) for v in [price, sma_high, sma_low]):
            return 0
        if price > sma_high and price > sma_low:
            return  1
        if price < sma_high and price < sma_low:
            return -1
        return 0

    d1  = _dir(row_1m)
    d15 = _dir(row_15m)
    return d1 != 0 and d15 != 0 and d1 != d15


# ── Phase 4: Bollinger Band Position ─────────────────────────────────────────
def phase4_bb_position(row_1m: pd.Series, row_15m: pd.Series) -> bool:
    """
    force_in_and_gate: 1m: close > bb200_mid AND > bb20_upper (bullish)
    or < bb200_mid AND < bb20_lower (bearish).
    15m: close > bb200_mid AND > bb20_mid or vice versa.
    Both TFs agree direction.
    """
    def _dir_1m(row):
        p    = row.get("close")
        b200m = row.get("bb200_mid")
        b20u  = row.get("bb20_upper")
        b20l  = row.get("bb20_lower")
        if any(pd.isna(v) for v in [p, b200m, b20u, b20l]):
            return 0
        if p > b200m and p > b20u:
            return  1
        if p < b200m and p < b20l:
            return -1
        return 0

    def _dir_15m(row):
        p    = row.get("close")
        b200m = row.get("bb200_mid")
        b20m  = row.get("bb20_mid")
        if any(pd.isna(v) for v in [p, b200m, b20m]):
            return 0
        if p > b200m and p > b20m:
            return  1
        if p < b200m and p < b20m:
            return -1
        return 0

    d1  = _dir_1m(row_1m)
    d15 = _dir_15m(row_15m)
    return d1 != 0 and d1 == d15


# ── Phase 5: SMA(2) Stack Alignment ──────────────────────────────────────────
def phase5_sma_stack(row_1m: pd.Series, row_1h: pd.Series) -> bool:
    """
    force_in_and_gate: sma2_sh0 > sma2_sh1 > ... > sma2_sh4 (bullish)
    or the reverse (bearish) on BOTH 1m and 1H.
    """
    def _dir(row):
        vals = [row.get(f"sma2_sh{i}") for i in range(5)]
        if any(pd.isna(v) for v in vals):
            return 0
        if all(vals[i] > vals[i + 1] for i in range(4)):
            return  1
        if all(vals[i] < vals[i + 1] for i in range(4)):
            return -1
        return 0

    d1  = _dir(row_1m)
    d1h = _dir(row_1h)
    return d1 != 0 and d1 == d1h


# ── Phase 6: ATR Expansion Force-In ──────────────────────────────────────────
def phase6_atr_expansion(row_1m: pd.Series, row_1h: pd.Series) -> bool:
    """
    force_in_and_gate: ATR14 > atr14_sma1_sh8 AND ATR45 > atr45_sma1_sh8
    on BOTH 1m and 1H simultaneously.
    """
    def _expanding(row):
        a14     = row.get("atr14")
        a14_ref = row.get("atr14_sma1_sh8")
        a45     = row.get("atr45")
        a45_ref = row.get("atr45_sma1_sh8")
        if any(pd.isna(v) for v in [a14, a14_ref, a45, a45_ref]):
            return False
        return a14 > a14_ref and a45 > a45_ref

    return _expanding(row_1m) and _expanding(row_1h)


# ── Backward-compatible aliases (Phase 7 = free, no mask needed) ─────────────
def can_open_trade_phase1(row_1m: pd.Series, row_15m: pd.Series) -> bool:
    """Alias kept for any legacy callers."""
    return phase1_cci_align(row_1m, row_15m)


def must_be_in_trade_phase2(row_1m: pd.Series, row_1h: pd.Series) -> bool:
    """Alias kept for any legacy callers — maps to phase5_sma_stack logic."""
    return phase5_sma_stack(row_1m, row_1h)


def must_be_in_trade_phase3(row_1m: pd.Series, row_15m: pd.Series) -> bool:
    """Alias kept for any legacy callers — maps to phase4_bb_position logic."""
    return phase4_bb_position(row_1m, row_15m)
