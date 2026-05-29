"""
monitoring/metrics_calculator.py
Core risk/performance metrics: Sharpe, Sortino, max DD, profit factor,
hit rate, recovery time.  All functions are pure (no side effects).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Sharpe ────────────────────────────────────────────────────────────────────

def calculate_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (risk-free = 0)."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(periods_per_year))


# ── Sortino ───────────────────────────────────────────────────────────────────

def calculate_sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    r = returns.dropna()
    downside = r[r < 0]
    if len(r) < 2 or len(downside) == 0:
        return 0.0
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return 0.0
    return float((r.mean() / downside_std) * np.sqrt(periods_per_year))


# ── Max Drawdown ──────────────────────────────────────────────────────────────

def calculate_max_dd_from_peak(equity: pd.Series) -> Tuple[float, int]:
    """
    Returns (max_drawdown_fraction, duration_in_bars).
    duration = length of the longest drawdown period.
    """
    eq = equity.dropna()
    if len(eq) < 2:
        return 0.0, 0
    peak    = eq.cummax()
    dd      = (eq - peak) / peak
    max_dd  = float(dd.min())

    # Duration of the worst drawdown
    in_dd   = (dd < 0).astype(int)
    groups  = (in_dd != in_dd.shift()).cumsum()
    lengths = in_dd.groupby(groups).sum()
    duration = int(lengths.max()) if len(lengths) > 0 else 0

    return abs(max_dd), duration


# ── Profit Factor ─────────────────────────────────────────────────────────────

def calculate_profit_factor(trades: List[dict]) -> float:
    """Gross profit / gross loss from a list of trade dicts with 'pnl_abs'."""
    if not trades:
        return 0.0
    pnls   = [t.get("pnl_abs", t.get("pnl_pct", 0)) for t in trades]
    gross_win  = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return round(gross_win / gross_loss, 4)


# ── Hit Rate ──────────────────────────────────────────────────────────────────

def calculate_hit_rate(trades: List[dict]) -> float:
    """Fraction of trades that were profitable."""
    if not trades:
        return 0.0
    pnls = [t.get("pnl_abs", t.get("pnl_pct", 0)) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    return round(wins / len(pnls), 4)


# ── Win/Loss Ratio ────────────────────────────────────────────────────────────

def calculate_avg_win_loss_ratio(trades: List[dict]) -> float:
    """Average win size / average loss size."""
    pnls  = [t.get("pnl_abs", t.get("pnl_pct", 0)) for t in trades]
    wins  = [p for p in pnls if p > 0]
    losses= [abs(p) for p in pnls if p < 0]
    if not wins or not losses:
        return 0.0
    return round(np.mean(wins) / np.mean(losses), 4)


# ── Recovery Time ─────────────────────────────────────────────────────────────

def calculate_recovery_time(equity: pd.Series) -> int:
    """
    Number of bars to recover from the deepest drawdown trough back to
    the previous peak.  Returns 0 if equity never reached a new high after
    the trough, or if there was no drawdown.
    """
    eq   = equity.dropna().reset_index(drop=True)
    peak = eq.cummax()
    dd   = (eq - peak) / peak
    if dd.min() >= 0:
        return 0
    trough_idx = int(dd.idxmin())
    trough_val = peak.iloc[trough_idx]   # previous peak we need to recover to
    after = eq.iloc[trough_idx:]
    recovered = after[after >= trough_val]
    if recovered.empty:
        return 0
    return int(recovered.index[0]) - trough_idx


# ── Rolling Sharpe ────────────────────────────────────────────────────────────

def rolling_sharpe(returns: pd.Series, window: int = 20,
                   periods_per_year: int = 252) -> pd.Series:
    """Rolling annualised Sharpe over `window` periods."""
    roll_mean = returns.rolling(window).mean()
    roll_std  = returns.rolling(window).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = (roll_mean / roll_std) * np.sqrt(periods_per_year)
    return rs.fillna(0.0)


# ── Full Summary Builder ──────────────────────────────────────────────────────

def build_metrics_summary(
    equity_series:  pd.Series,
    daily_returns:  pd.Series,
    trades:         List[dict],
    extra:          Optional[dict] = None,
) -> dict:
    """
    Compute all standard metrics and return a dict ready for JSON output.
    """
    sharpe   = calculate_sharpe(daily_returns)
    sortino  = calculate_sortino(daily_returns)
    max_dd, dd_dur = calculate_max_dd_from_peak(equity_series)
    pf       = calculate_profit_factor(trades)
    hit      = calculate_hit_rate(trades)
    wl       = calculate_avg_win_loss_ratio(trades)
    rec      = calculate_recovery_time(equity_series)

    summary = {
        "sharpe":            round(sharpe, 4),
        "sortino":           round(sortino, 4),
        "max_dd":            round(max_dd, 4),
        "max_dd_duration_bars": dd_dur,
        "profit_factor":     pf,
        "hit_rate":          hit,
        "avg_win_loss_ratio": wl,
        "recovery_bars":     rec,
        "total_trades":      len(trades),
    }
    if extra:
        summary.update(extra)
    return summary


def save_metrics_summary(summary: dict, path: str = "metrics_summary.json"):
    Path(path).write_text(json.dumps(summary, indent=2))
