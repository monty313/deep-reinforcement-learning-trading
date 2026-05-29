"""
monitoring/regression_detector.py
Detect when live/recent performance has degraded vs the training baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from monitoring.metrics_calculator import calculate_sharpe, rolling_sharpe


def detect_regression(
    recent_returns:   pd.Series,
    baseline_sharpe:  float,
    threshold_pct:    float = 15.0,
    window:           int   = 20,
    out_path:         str   = "regression_report.json",
) -> dict:
    """
    Compare rolling Sharpe of recent_returns against baseline_sharpe.

    threshold_pct : flag regression if current Sharpe < baseline * (1 - threshold/100)
    """
    current_sharpe = calculate_sharpe(recent_returns)
    rs             = rolling_sharpe(recent_returns, window=window)
    rolling_last   = float(rs.dropna().iloc[-1]) if not rs.dropna().empty else current_sharpe

    degradation_pct = ((baseline_sharpe - current_sharpe) / abs(baseline_sharpe) * 100
                       if baseline_sharpe != 0 else 0.0)
    regressed = current_sharpe < baseline_sharpe * (1.0 - threshold_pct / 100.0)

    report = {
        "baseline_sharpe":     round(baseline_sharpe, 4),
        "current_sharpe":      round(current_sharpe, 4),
        "rolling_sharpe_last": round(rolling_last, 4),
        "degradation_pct":     round(degradation_pct, 2),
        "regression_detected": regressed,
        "threshold_pct":       threshold_pct,
        "status":              "DEGRADED" if regressed else "OK",
        "alert_message": (
            f"Performance degraded {degradation_pct:.1f}% below baseline — investigate"
            if regressed else "Performance within acceptable range"
        ),
    }
    Path(out_path).write_text(json.dumps(report, indent=2))
    return report


def reward_sharpe_correlation(
    episode_rewards:  list,
    test_sharpes:     list,
    out_path:         str = "reward_correlation_report.json",
) -> dict:
    """
    Compute correlation between training reward per episode and test Sharpe.
    Both lists must have the same length (one entry per training episode/run).
    """
    import numpy as np
    if len(episode_rewards) < 3 or len(test_sharpes) < 3:
        return {"error": "Need at least 3 data points"}

    corr = float(np.corrcoef(episode_rewards, test_sharpes)[0, 1])
    status = "PASS" if corr >= 0.7 else ("WARN" if corr >= 0.5 else "FAIL")

    report = {
        "correlation":      round(corr, 4),
        "status":           status,
        "recommendation":   (
            "Keep reward function"         if status == "PASS" else
            "Review reward weights"        if status == "WARN" else
            "Reward function misaligned — redesign reward"
        ),
        "n_episodes":       len(episode_rewards),
    }
    Path(out_path).write_text(json.dumps(report, indent=2))
    return report
