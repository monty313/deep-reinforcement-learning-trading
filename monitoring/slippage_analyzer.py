"""
monitoring/slippage_analyzer.py
Compare backtest metrics vs live metrics to quantify slippage/commission impact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def compare_backtest_vs_live_metrics(
    backtest_csv: str,
    live_csv:     Optional[str] = None,
    out_path:     str = "slippage_impact_report.json",
) -> dict:
    """
    Load backtest and (optionally) live trade CSVs, compute metrics for each,
    and report the discrepancy.

    Both CSVs are expected to have columns:
        date, daily_return_pct, daily_max_dd_pct, result

    If live_csv is None, a mock 'live' is simulated by applying a 0.3 %
    slippage haircut to each daily return so the report is always generated.
    """
    bt = pd.read_csv(backtest_csv)
    bt["daily_return_frac"] = bt["daily_return_pct"] / 100.0

    if live_csv and Path(live_csv).exists():
        lv = pd.read_csv(live_csv)
        lv["daily_return_frac"] = lv["daily_return_pct"] / 100.0
        source = "live"
    else:
        # Simulate: assume 0.3 % per-day slippage + commission drag
        lv = bt.copy()
        lv["daily_return_frac"] = bt["daily_return_frac"] - 0.003
        lv["daily_return_pct"]  = lv["daily_return_frac"] * 100.0
        source = "simulated (no live CSV provided)"

    def _metrics(df: pd.DataFrame) -> dict:
        r = df["daily_return_frac"].dropna()
        std = r.std()
        sharpe = float((r.mean() / std) * np.sqrt(252)) if std > 0 else 0.0
        pass_rate = float((df["result"] == "pass").mean()) if "result" in df.columns else float("nan")
        return {
            "sharpe":      round(sharpe, 4),
            "mean_return": round(float(r.mean()), 6),
            "std_return":  round(float(std), 6),
            "pass_rate":   round(pass_rate, 4),
            "max_dd":      round(float(df["daily_max_dd_pct"].max() / 100.0), 4)
                           if "daily_max_dd_pct" in df.columns else None,
        }

    bt_m = _metrics(bt)
    lv_m = _metrics(lv)

    sharpe_delta = bt_m["sharpe"] - lv_m["sharpe"]
    pct_degradation = (sharpe_delta / bt_m["sharpe"] * 100) if bt_m["sharpe"] != 0 else 0.0

    report = {
        "live_data_source":    source,
        "backtest":            bt_m,
        "live":                lv_m,
        "sharpe_delta":        round(sharpe_delta, 4),
        "pct_degradation":     round(pct_degradation, 2),
        "alert":               pct_degradation > 5.0,
        "alert_message":       (
            f"ALERT: Live underperforming backtest by {pct_degradation:.1f}% — check slippage"
            if pct_degradation > 5.0 else "OK"
        ),
        "assumed_daily_slippage_pct": 0.3 if source.startswith("simulated") else None,
    }

    Path(out_path).write_text(json.dumps(report, indent=2))
    return report
