"""
gpu_rl_trading/training/train.py
Main GPU training loop for EURUSD.

Usage:
    python -m gpu_rl_trading.training.train --csv /path/to/EURUSD_M1.csv

Or import run_training() from a notebook.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gpu_rl_trading.config.settings import CFG
from gpu_rl_trading.env.indicators import build_feature_matrix
from gpu_rl_trading.env.environment import BatchedFTMOEnv, NUM_ACTIONS
from gpu_rl_trading.agent.dqn import DQNAgent


def load_eurusd_csv(csv_path: str, date_from: str = None, date_to: str = None) -> np.ndarray:
    """
    Load MT5 M1 CSV (tab-separated, MT5 History Center format) and return
    (T, 5) numpy array [open, high, low, close, volume].
    """
    print(f"[data] Loading {csv_path} ...", flush=True)
    df = pd.read_csv(csv_path, sep="\t", dtype=str)
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(
        df["date"].str.replace(".", "-", regex=False) + " " + df["time"],
        format="%Y-%m-%d %H:%M:%S", errors="coerce",
    )
    df = df.dropna(subset=["datetime"])
    vol_col = "tickvol" if "tickvol" in df.columns else "vol"
    df = df[["datetime", "open", "high", "low", "close", vol_col]].rename(columns={vol_col: "volume"})
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().set_index("datetime").sort_index()
    if date_from:
        df = df[df.index >= pd.Timestamp(date_from)]
    if date_to:
        df = df[df.index <= pd.Timestamp(date_to)]
    print(f"[data] Loaded {len(df):,} rows  ({df.index[0].date()} – {df.index[-1].date()})", flush=True)
    return df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    files = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def run_training(cfg: dict = None, resume: bool = False) -> DQNAgent:
    """
    Main entry point. cfg overrides defaults in CFG.
    Returns the trained agent.
    """
    if cfg is None:
        cfg = CFG.copy()
    else:
        merged = CFG.copy()
        merged.update(cfg)
        cfg = merged

    assert cfg["DATA_CSV_EURUSD"], "Set cfg['DATA_CSV_EURUSD'] to your CSV path."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    # ── 1. Load data + build features ────────────────────────────────────────
    ohlcv = load_eurusd_csv(cfg["DATA_CSV_EURUSD"], cfg["DATE_FROM"], cfg["DATE_TO"])
    print("[features] Building indicator matrix ...", flush=True)
    t0 = time.perf_counter()
    features = build_feature_matrix(ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4])
    print(f"[features] Done  shape={features.shape}  {time.perf_counter()-t0:.1f}s", flush=True)

    # ── 2. Build env + agent ──────────────────────────────────────────────────
    env = BatchedFTMOEnv(features, cfg, device)
    cfg["STATE_DIM"] = env.state_dim
    print(f"[env] state_dim={env.state_dim}  batch={cfg['BATCH_SIZE_ENV']}", flush=True)

    agent = DQNAgent(env.state_dim, NUM_ACTIONS, cfg, device)

    ckpt_dir = Path(cfg["CHECKPOINT_DIR"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(cfg["METRICS_DIR"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. Optional resume ────────────────────────────────────────────────────
    start_ep = 0
    if resume:
        ck = latest_checkpoint(ckpt_dir)
        if ck:
            agent.load(str(ck))
            try:
                start_ep = int(ck.stem.split("_ep")[-1])
            except Exception:
                pass
        else:
            print("[resume] No checkpoint found — starting fresh.", flush=True)

    # ── 4. Training loop ──────────────────────────────────────────────────────
    all_daily:   list = []
    all_ep_rows: list = []
    ckpt_every  = cfg.get("CHECKPOINT_EVERY", 10)

    print(f"\n[train] Starting episode {start_ep} — {cfg['NUM_EPISODES']} total", flush=True)
    for ep in range(start_ep, start_ep + cfg["NUM_EPISODES"]):
        t_ep   = time.perf_counter()
        state  = env.reset()
        ep_reward = torch.zeros(cfg["BATCH_SIZE_ENV"], device=device)
        dones  = torch.zeros(cfg["BATCH_SIZE_ENV"], dtype=torch.bool, device=device)
        step   = 0

        while not dones.all():
            actions    = agent.select_actions(state)
            next_state, rewards, dones = env.step(actions)
            ep_reward += rewards

            agent.store(state, actions, rewards, next_state, dones)
            loss = agent.train_step()
            state = next_state
            step += 1

        agent.decay_epsilon(ep)
        ep_elapsed = time.perf_counter() - t_ep
        mean_reward = ep_reward.mean().item()

        print(
            f"Ep {ep:04d} | reward={mean_reward:+.3f} | "
            f"eps={agent.epsilon:.3f} | steps={step} | {ep_elapsed:.1f}s",
            flush=True,
        )

        # collect metrics
        all_daily.extend(env.daily_metrics_log)
        all_ep_rows.append({
            "episode":      ep,
            "total_reward": round(mean_reward, 6),
            "steps":        step,
        })

        # checkpoint
        if (ep + 1) % ckpt_every == 0:
            ck_path = ckpt_dir / f"eurusd_gpu_ep{ep+1:04d}.pt"
            agent.save(str(ck_path))

        # flush CSVs every 20 episodes
        if (ep + 1) % 20 == 0:
            if all_daily:
                pd.DataFrame(all_daily).to_csv(
                    metrics_dir / "metrics_EURUSD_ftmo_gpu.csv", index=False)
            pd.DataFrame(all_ep_rows).to_csv(
                metrics_dir / "episode_rewards_EURUSD_gpu.csv", index=False)

    # final flush
    if all_daily:
        pd.DataFrame(all_daily).to_csv(
            metrics_dir / "metrics_EURUSD_ftmo_gpu.csv", index=False)
        print(f"[metrics] FTMO log -> {metrics_dir / 'metrics_EURUSD_ftmo_gpu.csv'}", flush=True)
    pd.DataFrame(all_ep_rows).to_csv(
        metrics_dir / "episode_rewards_EURUSD_gpu.csv", index=False)
    print(f"[metrics] Episode rewards -> {metrics_dir / 'episode_rewards_EURUSD_gpu.csv'}", flush=True)

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     required=True, help="Path to EURUSD M1 CSV")
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--batch",    type=int, default=None)
    args = parser.parse_args()

    overrides = {"DATA_CSV_EURUSD": args.csv}
    if args.episodes:
        overrides["NUM_EPISODES"] = args.episodes
    if args.batch:
        overrides["BATCH_SIZE_ENV"] = args.batch

    run_training(cfg=overrides, resume=args.resume)
