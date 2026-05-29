"""
monitoring/trade_duration_analyzer.py
Analyse trade holding-time distribution and flag overtrading / bag-holding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def analyze_trade_durations(trades: List[dict]) -> dict:
    """
    trades : list of dicts with at least 'entry_time' and 'exit_time'
             (or 'time' as entry and a computed duration_bars field).

    Falls back to trades that have 'duration_bars' or 'duration_minutes' directly.
    """
    if not trades:
        return {"error": "no trades provided"}

    durations_min: List[float] = []

    for t in trades:
        if "entry_time" in t and "exit_time" in t:
            try:
                entry = pd.Timestamp(t["entry_time"])
                exit_ = pd.Timestamp(t["exit_time"])
                dur   = (exit_ - entry).total_seconds() / 60.0
                durations_min.append(dur)
            except Exception:
                pass
        elif "duration_minutes" in t:
            durations_min.append(float(t["duration_minutes"]))
        elif "duration_bars" in t:
            durations_min.append(float(t["duration_bars"]))   # treat bars as minutes (1m bars)

    if not durations_min:
        return {"error": "could not compute durations — missing time fields"}

    arr = np.array(durations_min)
    avg = float(arr.mean())
    result = {
        "count":      len(arr),
        "avg_min":    round(avg, 2),
        "median_min": round(float(np.median(arr)), 2),
        "min_min":    round(float(arr.min()), 2),
        "max_min":    round(float(arr.max()), 2),
        "std_min":    round(float(arr.std()), 2),
        "pct_under_5min":  round(float((arr < 5).mean()) * 100, 2),
        "pct_over_4hr":   round(float((arr > 240).mean()) * 100, 2),
        "overtrading_flag": avg < 5,
        "bag_holding_flag": avg > 240,
    }

    if result["overtrading_flag"]:
        result["recommendation"] = "Average holding time < 5 min — possible overtrading; consider adding hold-time penalty to reward."
    elif result["bag_holding_flag"]:
        result["recommendation"] = "Average holding time > 4 hr — agent may be holding losers; consider max-hold-time rule."
    else:
        result["recommendation"] = "Holding time looks healthy."

    return result


def holding_time_reward_penalty(avg_hold_minutes: float, penalty: float = 0.5) -> float:
    """Return a reward penalty if avg holding time < 5 minutes (overtrading)."""
    return -penalty if avg_hold_minutes < 5.0 else 0.0


def save_duration_report(trades: List[dict], out_path: str = "trade_duration_report.json"):
    stats = analyze_trade_durations(trades)
    Path(out_path).write_text(json.dumps(stats, indent=2, default=str))
    return stats
