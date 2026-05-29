"""
monitoring/regime_detector.py
Classify market regime (calm / normal / volatile) from equity/volatility series
and separate performance metrics by regime.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from monitoring.metrics_calculator import calculate_sharpe, calculate_profit_factor, calculate_hit_rate


REGIMES = ("calm", "normal", "volatile")


def classify_regime(
    returns_series:    pd.Series,
    volatility_series: pd.Series = None,
    low_pct:  float = 20.0,
    high_pct: float = 80.0,
) -> pd.Series:
    """
    Label each bar as 'calm', 'normal', or 'volatile'.

    If volatility_series is None, rolling 20-bar std of returns is used.
    Thresholds are percentile-based on the full series.
    """
    if volatility_series is None:
        vol = returns_series.rolling(20).std().fillna(method="bfill")
    else:
        vol = volatility_series.reindex(returns_series.index).fillna(method="ffill")

    low_thresh  = np.percentile(vol.dropna(), low_pct)
    high_thresh = np.percentile(vol.dropna(), high_pct)

    labels = pd.Series("normal", index=returns_series.index)
    labels[vol <= low_thresh]  = "calm"
    labels[vol >= high_thresh] = "volatile"
    return labels


def separate_metrics_by_regime(
    daily_df:      pd.DataFrame,
    trades:        List[dict] = None,
    regime_col:    str = "regime",
) -> Dict[str, dict]:
    """
    Given a daily metrics DataFrame with a 'regime' column,
    compute Sharpe and pass_rate for each regime.

    daily_df must have: daily_return_pct, result, and the regime_col.
    """
    results = {}
    for regime in REGIMES:
        subset = daily_df[daily_df[regime_col] == regime]
        if subset.empty:
            results[regime] = {"sharpe": None, "pass_rate": None, "days": 0}
            continue

        ret_frac = subset["daily_return_pct"] / 100.0
        sharpe   = calculate_sharpe(ret_frac)
        pass_rate = float((subset["result"] == "pass").mean()) if "result" in subset.columns else None

        regime_trades = []
        if trades:
            dates = set(subset["date"].astype(str))
            regime_trades = [t for t in trades if str(t.get("time", ""))[:10] in dates]

        results[regime] = {
            "days":        len(subset),
            "sharpe":      round(sharpe, 4),
            "pass_rate":   round(pass_rate, 4) if pass_rate is not None else None,
            "profit_factor": calculate_profit_factor(regime_trades) if regime_trades else None,
            "hit_rate":    calculate_hit_rate(regime_trades) if regime_trades else None,
        }
    return results


def annotate_regimes(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'regime' column to a daily metrics DataFrame.
    Uses daily_return_pct as proxy for volatility signal.
    """
    returns = daily_df["daily_return_pct"] / 100.0
    labels  = classify_regime(returns)
    daily_df = daily_df.copy()
    daily_df["regime"] = labels.values
    return daily_df
