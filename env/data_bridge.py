"""
env/data_bridge.py
Bridge between data/loader.py output and env/ftmo_game.py.

Takes raw OHLCV data_dict[symbol][tf_minutes] = DataFrame
and returns feature_dict[symbol][tf_minutes] = feature-enriched DataFrame
ready for FTMOGame.

Also computes a safe init_idx (warm-up offset into the 1m master clock)
so indicators have enough history before training starts.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from env.indicators import build_feature_df

# Largest indicator lookback in the spec (CCI 900, SMA 200, BB 200).
# We need at least this many 1m bars before we can start trading.
MIN_WARMUP_BARS_1M = 1200   # ~20 hours of 1m data


def build_feature_data(
    data_dict:    Dict[str, Dict[int, pd.DataFrame]],
    symbols:      List[str] = None,
    add_news:     bool      = True,
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Apply build_feature_df (all indicators) to every symbol x timeframe.
    Optionally adds high_impact_news_soon column to 1m frames.
    Returns a new nested dict with enriched DataFrames.
    """
    if symbols is None:
        symbols = list(data_dict.keys())

    # Lazy import to avoid circular dependency at module load time
    if add_news:
        try:
            from env.news_detector import is_high_impact_news_window
            _news_fn = is_high_impact_news_window
        except Exception:
            _news_fn = None
    else:
        _news_fn = None

    feature_dict: Dict[str, Dict[int, pd.DataFrame]] = {}
    for sym in symbols:
        feature_dict[sym] = {}
        for tf, df in data_dict[sym].items():
            print(f"  Building features {sym} {tf}m ({len(df):,} rows) ...", flush=True)
            enriched = build_feature_df(df)
            # Add news feature to 1m frame only (higher TFs inherit via lookup)
            if tf == 1 and _news_fn is not None:
                enriched["high_impact_news_soon"] = enriched.index.map(
                    lambda t: _news_fn(t, minutes_forward=60)
                ).astype(float)
            feature_dict[sym][tf] = enriched
    return feature_dict


def find_common_start(
    feature_dict: Dict[str, Dict[int, pd.DataFrame]],
    symbols:      List[str] = None,
) -> pd.Timestamp:
    """
    Return the latest 'first valid index' across all symbols on the 1m frame,
    so every symbol has data from that point onwards.
    """
    if symbols is None:
        symbols = list(feature_dict.keys())
    starts = []
    for sym in symbols:
        df_1m = feature_dict[sym][1]
        # First row where close is not NaN (indicators need warmup rows)
        valid = df_1m["close"].dropna()
        if len(valid):
            starts.append(valid.index[0])
    return max(starts) if starts else None


def compute_init_idx(
    feature_dict: Dict[str, Dict[int, pd.DataFrame]],
    symbols:      List[str] = None,
    warmup:       int       = MIN_WARMUP_BARS_1M,
) -> int:
    """
    Return an integer index into the master 1m clock (first symbol's 1m frame)
    that is >= warmup bars AND after all symbols have data.
    """
    if symbols is None:
        symbols = list(feature_dict.keys())

    master = feature_dict[symbols[0]][1]
    common_start = find_common_start(feature_dict, symbols)

    if common_start is not None:
        idx = master.index.searchsorted(common_start, side="left")
    else:
        idx = 0

    return max(idx, warmup)


def slice_feature_data(
    feature_dict: Dict[str, Dict[int, pd.DataFrame]],
    date_from:    str = None,
    date_to:      str = None,
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Slice all feature DataFrames to [date_from, date_to] (CET strings YYYY-MM-DD).
    """
    import pytz
    CET = pytz.timezone("Europe/Berlin")

    out: Dict[str, Dict[int, pd.DataFrame]] = {}
    for sym, tfs in feature_dict.items():
        out[sym] = {}
        for tf, df in tfs.items():
            sliced = df
            if date_from:
                sliced = sliced[sliced.index >= pd.Timestamp(date_from, tz=CET)]
            if date_to:
                sliced = sliced[sliced.index <= pd.Timestamp(date_to, tz=CET)]
            out[sym][tf] = sliced
    return out
