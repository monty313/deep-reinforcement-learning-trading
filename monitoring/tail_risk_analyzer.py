"""
monitoring/tail_risk_analyzer.py
Worst-days distribution and tail risk statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def analyze_worst_days(daily_returns: pd.Series, percentile: float = 5) -> dict:
    """
    Extract tail statistics from a daily-returns series.

    Parameters
    ----------
    daily_returns : Series of daily return fractions (e.g. 0.025 = 2.5 %)
    percentile    : worst-N-percent threshold (default 5 = bottom 5 % of days)

    Returns
    -------
    dict with tail statistics and the worst-day dates/values
    """
    r = daily_returns.dropna()
    if len(r) == 0:
        return {}

    cutoff      = np.percentile(r, percentile)
    tail        = r[r <= cutoff]
    worst_5     = r.nsmallest(5)

    result = {
        "percentile_threshold":  percentile,
        "cutoff_return":         round(float(cutoff), 6),
        "tail_day_count":        int(len(tail)),
        "tail_mean_loss":        round(float(tail.mean()), 6),
        "tail_std":              round(float(tail.std()), 6),
        "worst_single_day":      round(float(r.min()), 6),
        "best_single_day":       round(float(r.max()), 6),
        "return_skewness":       round(float(r.skew()), 4),
        "return_kurtosis":       round(float(r.kurt()), 4),
        "worst_5_days": {
            str(idx): round(float(val), 6)
            for idx, val in worst_5.items()
        },
        "alert_worst_day_pct":   round(float(r.min()) * 100, 4),
        "flag_review":           bool(r.min() < -0.02),   # flag if any day > 2 % loss
    }
    return result


def tail_risk_report(
    daily_df:  pd.DataFrame,
    out_path:  str = "tail_risk_report.json",
) -> dict:
    """
    Build and save a tail risk report from a daily metrics DataFrame.
    Expects column 'daily_return_pct' (in percent, e.g. 2.5).
    """
    returns_frac = daily_df["daily_return_pct"] / 100.0
    stats = analyze_worst_days(returns_frac)
    Path(out_path).write_text(json.dumps(stats, indent=2, default=str))
    return stats
