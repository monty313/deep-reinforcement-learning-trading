"""
monitoring/daily_report.py
Generate end-of-day performance report and flag if retraining is needed.
Designed to run at market close (~17:00 CET).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from monitoring.metrics_calculator import calculate_sharpe, rolling_sharpe


REPORT_DIR = Path("daily_reports")


def generate_daily_report(
    metrics_csv:     str,
    baseline_sharpe: float = 1.0,
    out_dir:         str   = "daily_reports",
) -> dict:
    """
    Load metrics.csv and compute today's + rolling statistics.
    Saves a JSON report to out_dir/YYYY-MM-DD.json.

    Triggers retraining flags when:
      - Rolling 20-day Sharpe < 0.8 for 3 consecutive days
      - Rolling 5-day pass_rate < 50 % for 2 consecutive days
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    today_str   = str(date.today())
    today_row   = df[df["date"].dt.date == date.today()]

    daily_return = float(today_row["daily_return_pct"].iloc[0] / 100.0) if not today_row.empty else 0.0
    result_today = str(today_row["result"].iloc[0]) if not today_row.empty else "unknown"

    returns_frac     = df["daily_return_pct"] / 100.0
    rolling_20       = rolling_sharpe(returns_frac, window=20)
    rolling_sharpe_v = float(rolling_20.iloc[-1]) if not rolling_20.dropna().empty else 0.0

    pass_mask        = (df["result"] == "pass")
    consecutive_pass = int(pass_mask[::-1].cumprod().sum())
    pass_rate_5d     = float(pass_mask.tail(5).mean())
    pass_rate_20d    = float(pass_mask.tail(20).mean())

    # Retraining triggers
    sharpe_low_3d = (rolling_20.tail(3) < 0.8).all() if len(rolling_20.dropna()) >= 3 else False
    passrate_low  = pass_rate_5d < 0.5

    recommendation = "CONTINUE"
    if sharpe_low_3d and passrate_low:
        recommendation = "RETRAIN — both Sharpe and pass rate degraded"
    elif sharpe_low_3d:
        recommendation = "REVIEW — rolling Sharpe < 0.8 for 3 days"
    elif passrate_low:
        recommendation = "REVIEW — pass rate < 50 % over last 5 days"

    report = {
        "date":                  today_str,
        "daily_return":          round(daily_return, 6),
        "result_today":          result_today,
        "rolling_sharpe_20d":    round(rolling_sharpe_v, 4),
        "baseline_sharpe":       round(baseline_sharpe, 4),
        "consecutive_pass_days": consecutive_pass,
        "pass_rate_5d":          round(pass_rate_5d, 4),
        "pass_rate_20d":         round(pass_rate_20d, 4),
        "retrain_flag":          sharpe_low_3d or passrate_low,
        "recommendation":        recommendation,
        "generated_at":          datetime.utcnow().isoformat(),
    }

    out_path = Path(out_dir) / f"{today_str}.json"
    out_path.write_text(json.dumps(report, indent=2))
    return report


def load_report_history(out_dir: str = "daily_reports") -> List[dict]:
    reports = []
    for p in sorted(Path(out_dir).glob("*.json")):
        try:
            reports.append(json.loads(p.read_text()))
        except Exception:
            pass
    return reports
