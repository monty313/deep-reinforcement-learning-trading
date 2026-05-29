"""
train.py
Main training entry point for the Self-Healing FTMO RL system.

Usage:
    python train.py                          # uses config/training_config.yaml
    python train.py --config path/to/cfg.yaml
    python train.py --phase 2               # resume from phase 2
    python train.py --symbols EURUSD GBPUSD # override symbols on command line

Change the symbol list, date ranges, or FTMO params in config/training_config.yaml
without touching this file.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.loader import load_all, split_data
from env.data_bridge import build_feature_data, compute_init_idx, slice_feature_data
from agents.dqn_agent import DQNAgent
from env.ftmo_game import FTMOGame
from training.curriculum_trainer import run_phase, forward_test, _paths

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="FTMO RL curriculum trainer")
    p.add_argument("--config",  default="config/training_config.yaml",
                   help="Path to YAML config")
    p.add_argument("--phase",   type=int, default=1,
                   help="Curriculum phase to start from (1-4)")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Override symbols list, e.g. --symbols EURUSD GBPUSD")
    p.add_argument("--run-id",  default=None,
                   help="Unique run ID for checkpoint dir (auto-generated if omitted)")
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="Freeze N early layers when loading weights for transfer learning")
    p.add_argument("--test-mode", action="store_true",
                   help="Quick smoke-test: stop after 7 episodes per phase")
    p.add_argument("--daily-target", type=float, default=None,
                   help="Override daily profit target % (e.g., 2.5 for 2.5%)")
    p.add_argument("--daily-risk", type=float, default=None,
                   help="Override max daily drawdown % (e.g., 1.0 for 1.0%)")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[train] Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.symbols:
        cfg["symbols"] = args.symbols
    if args.test_mode:
        cfg["RL"]["TEST_MODE"] = True
    if args.daily_target is not None:
        cfg["FTMO"]["daily_profit_target_pct"] = args.daily_target / 100.0  # convert % to decimal
    if args.daily_risk is not None:
        cfg["FTMO"]["daily_max_drawdown_pct"] = args.daily_risk / 100.0    # convert % to decimal

    symbols   = cfg["symbols"]
    dates     = cfg["dates"]
    run_id    = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"  FTMO RL Training  |  run_id: {run_id}")
    print(f"  Symbols : {symbols}")
    print(f"  Train   : {dates['train_start']} -> {dates['train_end']}")
    print(f"  Val     : {dates['val_start']}   -> {dates['val_end']}")
    print(f"  Fwd     : {dates['fwd_start']}   -> present")
    print(f"{'='*60}\n")

    # ── 1. Load raw CSVs ──────────────────────────────────────────────────────
    print("Loading CSVs ...")
    raw_data = load_all(
        symbols    = symbols,
        csv_map    = cfg.get("csv_map"),
        data_dir   = cfg.get("data_dir"),
        date_from  = dates["train_start"],
        date_to    = dates.get("fwd_end"),   # load full range; slice later
    )

    # ── 2. Build indicator features ───────────────────────────────────────────
    print("\nBuilding features ...")
    feature_data = build_feature_data(raw_data, symbols)

    # ── 3. Split into train / val / fwd ──────────────────────────────────────
    train_data = slice_feature_data(feature_data, dates["train_start"], dates["train_end"])
    val_data   = slice_feature_data(feature_data, dates["val_start"],   dates["val_end"])
    fwd_data   = slice_feature_data(feature_data, dates["fwd_start"],   dates.get("fwd_end"))

    # ── 4. Compute warm-up offset ─────────────────────────────────────────────
    init_idx = compute_init_idx(train_data, symbols)
    print(f"\nWarm-up offset (init_idx): {init_idx}")

    # ── 5. Build agent ────────────────────────────────────────────────────────
    ftmo_cfg = {
        "profit_target_pct": cfg["FTMO"]["daily_profit_target_pct"],
        "max_dd_pct":        cfg["FTMO"]["daily_max_drawdown_pct"],
    }
    training_mode_rl = cfg["FTMO"].get("training_mode", True)  # True for training, False for live MT5
    rl_cfg = cfg["RL"]

    # Probe state_dim from a dummy env
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
        training_mode    = training_mode_rl,  # Pass training mode
    )
    state_dim = dummy_env.get_state().shape[1]
    print(f"State dimension: {state_dim}")

    agent = DQNAgent(
        symbols        = symbols,
        state_dim      = state_dim,
        rl_config      = rl_cfg,
        risk_fractions = cfg["ACTIONS"]["risk_fractions"],
    )

    # Optionally preload from a previous phase
    if rl_cfg.get("PRELOAD") and args.phase > 1:
        prev = _paths(cfg, args.phase - 1, run_id)
        agent.load(prev["weights"], prev["replay"], prev["risk"],
                   freeze_layers=args.freeze_layers)

    # ── 6. Run curriculum phases ──────────────────────────────────────────────
    advance = cfg["CURRICULUM"]["advance_consecutive_pass_days"]

    for phase_cfg in cfg["CURRICULUM"]["phases"]:
        if phase_cfg["id"] < args.phase:
            continue

        agent = run_phase(
            phase_cfg    = phase_cfg,
            data_dict    = train_data,
            cfg          = cfg,
            agent        = agent,
            logger       = None,             # set to WandbLogger(cfg, run_id) to enable W&B
            run_id       = run_id,
            advance_days = advance,
        )

        # Transfer weights to next phase
        paths = _paths(cfg, phase_cfg["id"], run_id)
        if phase_cfg["id"] < 4:
            agent.load(paths["weights"], paths["replay"], paths["risk"],
                       freeze_layers=args.freeze_layers)

    # ── 7. Forward test ───────────────────────────────────────────────────────
    print("\nRunning forward test ...")
    fwd_df = forward_test(str(cfg_path), fwd_data, agent, run_id=run_id)
    print(f"Forward test rows: {len(fwd_df)}")

    # ── 8. Validation run (inference only) ───────────────────────────────────
    print("\nRunning validation ...")
    val_df = forward_test(str(cfg_path), val_data, agent, run_id=f"{run_id}_val")
    print(f"Validation rows: {len(val_df)}")

    print(f"\nOK Training complete — run_id: {run_id}")


if __name__ == "__main__":
    main()
