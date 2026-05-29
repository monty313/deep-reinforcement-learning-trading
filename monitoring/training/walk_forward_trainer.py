"""
training/walk_forward_trainer.py
Walk-forward evaluation: train on rolling windows, test on held-out periods.

Usage:
    python -m training.walk_forward_trainer
    python -m training.walk_forward_trainer --config config/training_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import load_all
from env.data_bridge import build_feature_data, compute_init_idx, slice_feature_data
from env.ftmo_game import FTMOGame
from agents.dqn_agent import DQNAgent
from training.curriculum_trainer import run_phase, forward_test
from monitoring.metrics_calculator import calculate_sharpe, calculate_profit_factor, calculate_hit_rate


# Default walk-forward windows (train_start, train_end, test_start, test_end)
DEFAULT_WINDOWS = [
    ("2021-02-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2023-01-01", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2024-01-01", "2024-12-31", "2025-01-01", "2025-03-31"),
]


def run_window(window_id: int, train_start: str, train_end: str,
               test_start: str, test_end: str,
               cfg: dict, all_feature_data: dict) -> dict:
    """Train a fresh agent on one window and evaluate on the test period."""
    symbols = cfg["symbols"]
    rl_cfg  = cfg["RL"]
    ftmo_cfg = {
        "profit_target_pct": cfg["FTMO"]["daily_profit_target_pct"],
        "max_dd_pct":        cfg["FTMO"]["daily_max_drawdown_pct"],
    }

    train_data = slice_feature_data(all_feature_data, train_start, train_end)
    test_data  = slice_feature_data(all_feature_data, test_start,  test_end)
    init_idx   = compute_init_idx(train_data, symbols)

    dummy_env = FTMOGame(
        data_dict        = train_data,
        symbols          = symbols,
        reward_cfg       = cfg["REWARD"],
        ftmo_cfg         = ftmo_cfg,
        trading_mode     = cfg["TRADING_MODE"],
        curriculum_phase = 1,
        risk_fractions   = cfg["ACTIONS"]["risk_fractions"],
        lkbk             = rl_cfg["LKBK"],
        init_idx         = init_idx,
    )
    state_dim = dummy_env.get_state().shape[1]

    agent = DQNAgent(
        symbols        = symbols,
        state_dim      = state_dim,
        rl_config      = rl_cfg,
        risk_fractions = cfg["ACTIONS"]["risk_fractions"],
    )

    advance = cfg["CURRICULUM"]["advance_consecutive_pass_days"]
    for phase_cfg in cfg["CURRICULUM"]["phases"]:
        agent = run_phase(
            phase_cfg    = phase_cfg,
            data_dict    = train_data,
            cfg          = cfg,
            agent        = agent,
            logger       = None,
            run_id       = f"wf_w{window_id}",
            advance_days = advance,
        )

    test_df = forward_test(
        config_path = str(ROOT / "config" / "training_config.yaml"),
        data_dict   = test_data,
        agent       = agent,
        run_id      = f"wf_w{window_id}_test",
    )

    if test_df.empty:
        return {
            "window": window_id, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "sharpe": 0.0, "max_dd": 0.0, "pass_rate": 0.0,
            "status": "no_data",
        }

    returns = test_df["daily_return_pct"] / 100.0
    sharpe  = calculate_sharpe(returns)
    pass_rt = float((test_df["result"] == "pass").mean())
    max_dd  = float(test_df["daily_max_dd_pct"].max() / 100.0)

    return {
        "window":      window_id,
        "train_start": train_start,
        "train_end":   train_end,
        "test_start":  test_start,
        "test_end":    test_end,
        "sharpe":      round(sharpe, 4),
        "max_dd":      round(max_dd, 4),
        "pass_rate":   round(pass_rt, 4),
        "status":      "PASS" if sharpe >= 0.9 else "FAIL",
    }


def run_walk_forward(
    config_path: str = "config/training_config.yaml",
    windows:     list = None,
    out_path:    str  = "walk_forward_report.csv",
):
    windows = windows or DEFAULT_WINDOWS

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dates   = cfg["dates"]
    symbols = cfg["symbols"]

    # Determine full date range across all windows
    all_starts = [w[0] for w in windows]
    all_ends   = [w[3] for w in windows]
    global_start = min(all_starts)
    global_end   = max(all_ends)

    print(f"Walk-forward: loading data {global_start} -> {global_end} …")
    raw = load_all(
        symbols   = symbols,
        csv_map   = cfg.get("csv_map"),
        date_from = global_start,
        date_to   = global_end,
    )
    all_features = build_feature_data(raw, symbols)

    rows = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows, start=1):
        print(f"\nWindow {i}: train [{tr_s},{tr_e}]  test [{te_s},{te_e}]")
        result = run_window(i, tr_s, tr_e, te_s, te_e, cfg, all_features)
        rows.append(result)
        print(f"  -> Sharpe={result['sharpe']:.3f}  PassRate={result['pass_rate']:.2%}  {result['status']}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    all_pass = bool((df["sharpe"] >= 0.9).all())
    any_fail = bool((df["sharpe"] < 0.8).any())

    summary = {
        "windows":         len(windows),
        "all_pass_sharpe": all_pass,
        "regime_shift_flag": any_fail,
        "mean_sharpe":     round(float(df["sharpe"].mean()), 4),
        "min_sharpe":      round(float(df["sharpe"].min()),  4),
        "verdict":         "PASS" if all_pass else "FAIL",
    }
    Path("walk_forward_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*50}")
    print(f"Walk-forward result: {summary['verdict']}")
    print(f"  Mean Sharpe: {summary['mean_sharpe']:.3f}   Min: {summary['min_sharpe']:.3f}")
    if any_fail:
        print("  WARNING: At least one window Sharpe < 0.8 — possible regime shift.")
    print(f"  Report: {out_path}")
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--out",    default="walk_forward_report.csv")
    args = parser.parse_args()
    run_walk_forward(args.config, out_path=args.out)
