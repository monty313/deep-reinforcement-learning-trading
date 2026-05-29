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
import time
from datetime import datetime
from pathlib import Path

import yaml
from tqdm.auto import tqdm

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.loader import load_all, split_data
from env.data_bridge import build_feature_data, compute_init_idx
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

    RUN_START = time.perf_counter()

    def _stage(label: str) -> float:
        """Print a stage banner and return the start time."""
        t = time.perf_counter()
        print(f"\n{'─'*60}", flush=True)
        print(f"[START] {label}", flush=True)
        return t

    def _done(label: str, t0: float):
        elapsed = time.perf_counter() - t0
        print(f"[DONE]  {label}  ({elapsed:.1f}s / {elapsed/60:.1f} min)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  FTMO RL Training  |  run_id: {run_id}", flush=True)
    print(f"  Symbols : {symbols}", flush=True)
    print(f"  Train   : {dates['train_start']} -> {dates['train_end']}", flush=True)
    print(f"  Val     : {dates['val_start']}   -> {dates['val_end']}", flush=True)
    print(f"  Fwd     : {dates['fwd_start']}   -> present", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ── 1. Load raw CSVs (one window at a time to avoid OOM on 2M-row files) ───
    # Build features per window so indicators are never computed on the full
    # 2M-row dataset.  Each window gets its own load+feature pass.
    t0 = _stage("Stage 1/8 — Load raw CSVs + build features (per window)")

    def _load_and_build(label: str, from_dt: str, to_dt: str):
        print(f"  [{label}] loading {from_dt} -> {to_dt}", flush=True)
        raw = load_all(
            symbols    = symbols,
            csv_map    = cfg.get("csv_map"),
            data_dir   = cfg.get("data_dir"),
            date_from  = from_dt,
            date_to    = to_dt,
        )
        print(f"  [{label}] building features ...", flush=True)
        return build_feature_data(raw, symbols)

    train_data = _load_and_build("train", dates["train_start"], dates["train_end"])
    val_data   = _load_and_build("val",   dates["val_start"],   dates["val_end"])
    fwd_data   = _load_and_build("fwd",   dates["fwd_start"],   dates.get("fwd_end"))

    _done("Stage 1/8 — Load raw CSVs + build features", t0)

    # ── 2. (merged into stage 1) ──────────────────────────────────────────────
    # Feature building now happens per-window above; no full-dataset pass needed.

    # ── 3. (merged into stage 1) ─────────────────────────────────────────────
    # Windows are already sliced by date at load time; no further slicing needed.

    # ── 4. Compute warm-up offset ─────────────────────────────────────────────
    t0 = _stage("Stage 4/8 — Compute warm-up offset")
    init_idx = compute_init_idx(train_data, symbols)
    print(f"  init_idx = {init_idx}", flush=True)
    _done("Stage 4/8 — Warm-up offset", t0)

    # ── 5. Build agent ────────────────────────────────────────────────────────
    t0 = _stage("Stage 5/8 — Build DQN agent")
    ftmo_cfg = {
        "profit_target_pct": cfg["FTMO"]["daily_profit_target_pct"],
        "max_dd_pct":        cfg["FTMO"]["daily_max_drawdown_pct"],
    }
    training_mode_rl = cfg["FTMO"].get("training_mode", True)
    rl_cfg = cfg["RL"]

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
        training_mode    = training_mode_rl,
    )
    state_dim = dummy_env.get_state().shape[1]
    print(f"  State dimension: {state_dim}", flush=True)

    agent = DQNAgent(
        symbols        = symbols,
        state_dim      = state_dim,
        rl_config      = rl_cfg,
        risk_fractions = cfg["ACTIONS"]["risk_fractions"],
    )
    _done("Stage 5/8 — Build DQN agent", t0)

    # Optionally preload from a previous phase
    if rl_cfg.get("PRELOAD") and args.phase > 1:
        prev = _paths(cfg, args.phase - 1, run_id)
        agent.load(prev["weights"], prev["replay"], prev["risk"],
                   freeze_layers=args.freeze_layers)

    # ── 6. Run curriculum phases ──────────────────────────────────────────────
    advance      = cfg["CURRICULUM"]["advance_consecutive_pass_days"]
    phases_to_run = [p for p in cfg["CURRICULUM"]["phases"] if p["id"] >= args.phase]

    t0 = _stage(f"Stage 6/8 — Curriculum training ({len(phases_to_run)} phases)")
    for phase_cfg in tqdm(phases_to_run, desc="Curriculum phases", unit="phase"):
        agent = run_phase(
            phase_cfg    = phase_cfg,
            data_dict    = train_data,
            cfg          = cfg,
            agent        = agent,
            logger       = None,
            run_id       = run_id,
            advance_days = advance,
        )
        paths = _paths(cfg, phase_cfg["id"], run_id)
        if phase_cfg["id"] < 4:
            agent.load(paths["weights"], paths["replay"], paths["risk"],
                       freeze_layers=args.freeze_layers)
    _done("Stage 6/8 — Curriculum training", t0)

    # ── 7. Forward test ───────────────────────────────────────────────────────
    t0 = _stage("Stage 7/8 — Forward test (inference, no weight updates)")
    fwd_df = forward_test(str(cfg_path), fwd_data, agent, run_id=run_id)
    print(f"  Forward test rows: {len(fwd_df)}", flush=True)
    _done("Stage 7/8 — Forward test", t0)

    # ── 8. Validation run ─────────────────────────────────────────────────────
    t0 = _stage("Stage 8/8 — Validation run")
    val_df = forward_test(str(cfg_path), val_data, agent, run_id=f"{run_id}_val")
    print(f"  Validation rows: {len(val_df)}", flush=True)
    _done("Stage 8/8 — Validation run", t0)

    total_elapsed = time.perf_counter() - RUN_START
    print(f"\n{'='*60}", flush=True)
    print(f"  TRAINING COMPLETE — run_id: {run_id}", flush=True)
    print(f"  Total time: {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
