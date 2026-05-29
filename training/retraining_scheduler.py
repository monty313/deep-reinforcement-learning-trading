"""
training/retraining_scheduler.py
Quarterly retraining with A/B comparison against deployed model.

Usage:
    python -m training.retraining_scheduler
    python -m training.retraining_scheduler --config config/training_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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
from monitoring.metrics_calculator import calculate_sharpe

VERSION_LOG = Path("model_version_log.json")


def load_version_log() -> list:
    if VERSION_LOG.exists():
        return json.loads(VERSION_LOG.read_text())
    return []


def save_version_log(log: list):
    VERSION_LOG.write_text(json.dumps(log, indent=2))


def train_new_model(cfg: dict, train_data: dict, init_idx: int,
                    run_id: str) -> DQNAgent:
    """Full curriculum train; return trained agent."""
    symbols = cfg["symbols"]
    rl_cfg  = cfg["RL"]
    ftmo_cfg = {
        "profit_target_pct": cfg["FTMO"]["daily_profit_target_pct"],
        "max_dd_pct":        cfg["FTMO"]["daily_max_drawdown_pct"],
    }

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
            run_id       = run_id,
            advance_days = advance,
        )
    return agent


def run_quarterly_retrain(
    config_path:       str   = "config/training_config.yaml",
    ab_window_days:    int   = 7,
    improvement_pct:   float = 10.0,
    force_deploy:      bool  = False,
):
    """
    Train on the latest 12 months of data, A/B test against any previously
    deployed model on the same recent window, deploy only if Sharpe improves > 10%.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from dateutil.relativedelta import relativedelta
    today       = datetime.today()
    train_start = (today - relativedelta(months=12)).strftime("%Y-%m-%d")
    train_end   = today.strftime("%Y-%m-%d")
    ab_start    = (today - pd.Timedelta(days=ab_window_days)).strftime("%Y-%m-%d")

    symbols = cfg["symbols"]
    run_id  = today.strftime("quarterly_%Y%m%d")

    print(f"Quarterly retrain: {train_start} -> {train_end}")
    raw = load_all(
        symbols   = symbols,
        csv_map   = cfg.get("csv_map"),
        date_from = train_start,
        date_to   = train_end,
    )
    all_features = build_feature_data(raw, symbols)
    train_data   = slice_feature_data(all_features, train_start, train_end)
    ab_data      = slice_feature_data(all_features, ab_start, train_end)
    init_idx     = compute_init_idx(train_data, symbols)

    print("Training new model …")
    new_agent = train_new_model(cfg, train_data, init_idx, run_id)

    # Evaluate new model on A/B window
    new_fwd = forward_test(config_path, ab_data, new_agent, run_id=f"{run_id}_ab")
    new_sharpe = calculate_sharpe(new_fwd["daily_return_pct"] / 100.0) if not new_fwd.empty else 0.0

    # Compare against last deployed model (if logged)
    version_log = load_version_log()
    old_sharpe  = version_log[-1].get("sharpe", 0.0) if version_log else 0.0

    improvement = ((new_sharpe - old_sharpe) / abs(old_sharpe) * 100
                   if old_sharpe != 0 else 100.0)
    deploy = force_deploy or (new_sharpe > old_sharpe * (1.0 + improvement_pct / 100.0))

    entry = {
        "run_id":      run_id,
        "trained_on":  f"{train_start} to {train_end}",
        "deployed_at": today.isoformat() if deploy else None,
        "sharpe":      round(new_sharpe, 4),
        "old_sharpe":  round(old_sharpe, 4),
        "improvement_pct": round(improvement, 2),
        "deployed":    deploy,
    }

    if deploy:
        weights_dir = Path(cfg["PATHS"]["weights_dir"]) / run_id
        weights_dir.mkdir(parents=True, exist_ok=True)
        new_agent.save(
            str(weights_dir / "weights.h5"),
            str(weights_dir / "replay.pkl"),
            str(weights_dir / "risk.json"),
        )
        print(f"Deployed new model (Sharpe {new_sharpe:.3f} vs {old_sharpe:.3f}, +{improvement:.1f}%)")
    else:
        print(f"Kept existing model (new Sharpe {new_sharpe:.3f} did not improve "
              f"{improvement_pct}% over {old_sharpe:.3f})")

    version_log.append(entry)
    save_version_log(version_log)
    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="config/training_config.yaml")
    parser.add_argument("--ab-days",       type=int,   default=7)
    parser.add_argument("--improve-pct",   type=float, default=10.0)
    parser.add_argument("--force-deploy",  action="store_true")
    args = parser.parse_args()
    run_quarterly_retrain(args.config, args.ab_days, args.improve_pct, args.force_deploy)
