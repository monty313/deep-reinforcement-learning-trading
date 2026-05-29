"""
env/news_detector.py
Economic calendar integration for news-aware trading.
Uses a locally cached CSV of high-impact events (updated daily from MQL5).

Feature:  is_high_impact_news_window(timestamp, minutes_forward=60) -> float [0,1]
Reward:   news_reward_penalty(timestamp, open_positions) -> float (<= 0)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz

CET = pytz.timezone("Europe/Berlin")

# Local cache file — update daily from MQL5 economic calendar
CACHE_PATH = Path("data/economic_calendar.csv")

# Columns expected in cache: datetime (CET), currency, impact, event
IMPACT_HIGH = "high"


def _load_calendar() -> Optional[pd.DataFrame]:
    """Load cached economic calendar CSV. Returns None if unavailable."""
    if not CACHE_PATH.exists():
        return None
    df = pd.read_csv(CACHE_PATH)
    if "datetime" not in df.columns:
        return None
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False, errors="coerce")
    # Localise to CET if tz-naive
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(CET, ambiguous="infer",
                                                        nonexistent="shift_forward")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(CET)
    return df


_CALENDAR: Optional[pd.DataFrame] = _load_calendar()


def reload_calendar():
    """Force reload from disk (call after updating the cache file)."""
    global _CALENDAR
    _CALENDAR = _load_calendar()


def is_high_impact_news_window(
    timestamp,
    minutes_forward: int = 60,
    minutes_before:  int = 15,
) -> float:
    """
    Returns 1.0 if there is a high-impact news event within the window
    [timestamp - minutes_before, timestamp + minutes_forward] in CET,
    else 0.0.

    When no calendar is loaded, always returns 0.0 (safe default: no penalty).
    """
    if _CALENDAR is None or _CALENDAR.empty:
        return 0.0

    try:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize(CET)
        else:
            ts = ts.tz_convert(CET)
    except Exception:
        return 0.0

    window_start = ts - pd.Timedelta(minutes=minutes_before)
    window_end   = ts + pd.Timedelta(minutes=minutes_forward)

    high_events = _CALENDAR[
        (_CALENDAR["impact"].str.lower() == IMPACT_HIGH) &
        (_CALENDAR["datetime"] >= window_start) &
        (_CALENDAR["datetime"] <= window_end)
    ]
    return 1.0 if not high_events.empty else 0.0


def news_reward_penalty(
    timestamp,
    num_open_positions: int,
    penalty_per_position: float = 0.5,
) -> float:
    """
    Return a negative reward adjustment if trading during high-impact news.
    Agent is NOT forced flat — it learns whether to trade or avoid.
    """
    if num_open_positions == 0:
        return 0.0
    if is_high_impact_news_window(timestamp) > 0:
        return -penalty_per_position * num_open_positions
    return 0.0


def create_sample_calendar(out_path: str = "data/economic_calendar.csv"):
    """
    Create a minimal sample calendar CSV so the detector works out-of-the-box
    without a live MQL5 connection.  Replace with real data for live trading.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sample = pd.DataFrame([
        {"datetime": "2024-01-26 14:30:00", "currency": "USD", "impact": "high", "event": "US GDP"},
        {"datetime": "2024-02-01 14:15:00", "currency": "EUR", "impact": "high", "event": "ECB Rate Decision"},
        {"datetime": "2024-03-07 09:30:00", "currency": "GBP", "impact": "high", "event": "BoE Rate Decision"},
    ])
    sample.to_csv(out_path, index=False)
    return sample
