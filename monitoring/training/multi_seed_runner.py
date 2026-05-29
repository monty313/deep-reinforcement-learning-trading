"""
training/multi_seed_runner.py
Train the curriculum N times with different random seeds, collect metrics,
and output seed_robustness_report.csv.

Usage:
    python -m training.multi_seed_runner
    python -m training.multi_seed_runner --seeds 0 1 2 3 4 --config config/training_config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import load_all
from env.data_bridge import build_feature_data, compute_init_idx, slice_feature_data
from env.ftmo_game import FTMOGame
from agents.dqn_agent import DQNAgent
from training.curriculum_trainer import run_phase, forward_test
from monitoring.metrics_calculator import build_metrics_summary, save_metrics_summary


def set_seed(seed: int):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_one_seed(seed: int, cfg: dict, train_data: dict, fwd_data: dict,
                 init_idx: int, run_id: str) -> dict:
    """Train curriculum for one seed; return forward-test metrics dict."""
    set_seed(seed)

    ftmo_cfg = {
        "profit_target_pct": cfg["FTMO"]["daily_profit_target_pct"],
        "max_dd_pct":        cfg["FTMO"]["daily_max_drawdown_pct"],
    }
    rl_cfg  = cfg["RL"]
    symbols = cfg["symbols"]

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
            run_id       = f"{run_id}_seed{seed}",
            advance_days = advance,
        )

    fwd_df = forward_test(
        config_path = str(ROOT / "config" / "training_config.yaml"),
        data_dict   = fwd_data,
        agent       = agent,
        run_id      = f"{run_id}_seed{seed}",
    )

    if fwd_df.empty:
        return {"seed": seed, "sharpe": 0.0, "max_dd": 0.0, "pass_rate": 0.0,
                "profit_factor": 0.0, "status": "no_data"}

    returns_frac = fwd_df["daily_return_pct"] / 100.0
    trades       = []   # forward_test doesn't return trade list; leave empty for now

    summary = build_metrics_summary(
        equity_series  = pd.Series(fwd_df["end_equity"].values),
        daily_returns  = returns_frac,
        trades         = trades,
        extra          = {
            "seed":      seed,
            "pass_rate": float((fwd_df["result"] == "pass").mean()),
        },
    )
    return summary


def run_multi_seed(
    seeds:      list = None,
    config_path: str = "config/training_config.yaml",
    out_path:    str = "seed_robustness_report.csv",
):
    seeds = seeds or [0, 1, 2, 3, 4]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dates   = cfg["dates"]
    symbols = cfg["symbols"]

    print(f"Loading data for multi-seed run ({len(seeds)} seeds) …")
    raw = load_all(
        symbols   = symbols,
        csv_map   = cfg.get("csv_map"),
        date_from = dates["train_start"],
        date_to   = dates.get("fwd_end"),
    )
    feature_data = build_feature_data(raw, symbols)
    train_data   = slice_feature_data(feature_data, dates["train_start"], dates["train_end"])
    fwd_data     = slice_feature_data(feature_data, dates["fwd_start"],   dates.get("fwd_end"))
    init_idx     = compute_init_idx(train_data, symbols)

    rows = []
    for seed in seeds:
        print(f"\n{'='*50}\n  Seed {seed}\n{'='*50}")
        result = run_one_seed(seed, cfg, train_data, fwd_data, init_idx,
                              run_id=f"multiseed")
        rows.append(result)
        print(f"  Seed {seed} -> Sharpe={result.get('sharpe','?'):.3f}  "
              f"MaxDD={result.get('max_dd','?'):.3f}  "
              f"PassRate={result.get('pass_rate','?'):.2%}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    # Summary statistics
    sharpe_vals = df["sharpe"].dropna()
    summary = {
        "seeds":         seeds,
        "sharpe_mean":   round(float(sharpe_vals.mean()), 4),
        "sharpe_std":    round(float(sharpe_vals.std()),  4),
        "sharpe_min":    round(float(sharpe_vals.min()),  4),
        "sharpe_max":    round(float(sharpe_vals.max()),  4),
        "pass_all":      bool((sharpe_vals >= 0.8).all()),
        "low_variance":  bool(sharpe_vals.std() < 0.4),
        "verdict":       "PASS" if (sharpe_vals >= 0.8).all() and sharpe_vals.std() < 0.4 else "FAIL",
    }
    Path("seed_robustness_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*50}")
    print(f"Multi-seed result: {summary['verdict']}")
    print(f"  Sharpe: mean={summary['sharpe_mean']:.3f}  std={summary['sharpe_std']:.3f}")
    print(f"  Report saved to {out_path}")
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",  nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--out",    default="seed_robustness_report.csv")
    args = parser.parse_args()
    run_multi_seed(args.seeds, args.config, args.out)
