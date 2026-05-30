"""
gpu_rl_trading/training/train.py
Main GPU training loop for EURUSD — full 8-phase curriculum.

Phase advancement: 10 consecutive PASS days (across any batch item) OR
                   max_episodes_per_phase cap.

Usage:
    python -m gpu_rl_trading.training.train --csv /path/to/EURUSD_M1.csv
    python -m gpu_rl_trading.training.train --csv ... --resume --start-phase 2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from gpu_rl_trading.config.settings import CFG
from gpu_rl_trading.env.indicators import build_feature_matrix
from gpu_rl_trading.env.environment import BatchedFTMOEnv, NUM_ACTIONS
from gpu_rl_trading.agent.dqn import DQNAgent


# ── helpers ───────────────────────────────────────────────────────────────────

def load_eurusd_csv(csv_path: str, date_from: str = None, date_to: str = None) -> np.ndarray:
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


def latest_checkpoint(ckpt_dir: Path, phase: int = None) -> Path | None:
    pattern = f"*ph{phase}*.pt" if phase is not None else "*.pt"
    files   = sorted(ckpt_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


# ── per-episode reward shaping tracker ───────────────────────────────────────

class EpisodeRewardShaper:
    """
    Tracks cross-episode statistics and computes bonus rewards at episode end.

    Bonuses (all normalised to ~equity scale):
      - Consecutive PASS days streak (grows with streak length)
      - Lower dd than previous episode avg
      - Higher return than previous episode avg
      - PASS rate all-time high
      - 5-episode consistency window (PASS rate ≥ threshold → growing bonus)
      - Each additional consecutive consistent episode multiplies the bonus
    """

    def __init__(self):
        self.ep_pass_rates:    List[float] = []
        self.ep_avg_rets:      List[float] = []
        self.ep_avg_dds:       List[float] = []
        self.ath_pass_rate:    float       = 0.0
        self.consistency_streak: int       = 0   # episodes where pass_rate >= 0.5

    def compute_bonus(
        self,
        daily_log:         list,   # env.daily_metrics_log for this episode
        consec_pass_days:  int,    # longest consecutive PASS streak this episode
    ) -> float:
        """Return scalar bonus reward for this episode."""
        if not daily_log:
            return 0.0

        flags    = [r["ftmo_flag"]          for r in daily_log]
        rets     = [r["daily_return_pct"]   for r in daily_log]
        dds      = [r["daily_max_drawdown_pct"] for r in daily_log]
        n        = len(flags)
        passes   = flags.count("PASS")
        pass_rate = passes / n if n > 0 else 0.0
        avg_ret  = sum(rets) / n
        avg_dd   = sum(dds)  / n

        bonus = 0.0

        # ── 1. Consecutive PASS day streak ────────────────────────────────────
        # grows non-linearly: streak^1.5 * 0.01
        if consec_pass_days >= 2:
            bonus += (consec_pass_days ** 1.5) * 0.01

        # ── 2. Better return than last episode ────────────────────────────────
        if self.ep_avg_rets and avg_ret > self.ep_avg_rets[-1]:
            improvement = avg_ret - self.ep_avg_rets[-1]
            bonus += min(improvement * 0.05, 0.05)   # cap at 0.05

        # ── 3. Lower dd than last episode ─────────────────────────────────────
        if self.ep_avg_dds and avg_dd < self.ep_avg_dds[-1]:
            improvement = self.ep_avg_dds[-1] - avg_dd
            bonus += min(improvement * 0.05, 0.05)

        # ── 4. PASS rate all-time high ────────────────────────────────────────
        if pass_rate > self.ath_pass_rate:
            bonus += 0.10
            self.ath_pass_rate = pass_rate
            print(f"  [★ ATH pass rate] {pass_rate:.1%}", flush=True)

        # ── 5. Win-rate improvement ───────────────────────────────────────────
        if len(self.ep_pass_rates) >= 3:
            prev_avg = sum(self.ep_pass_rates[-3:]) / 3
            if pass_rate > prev_avg:
                bonus += (pass_rate - prev_avg) * 0.20

        # ── 6. 5-episode consistency window ──────────────────────────────────
        # If pass_rate >= 0.50 this episode, increment streak; else reset.
        # Bonus grows with streak: 0.05 * streak (so ep5=0.25, ep10=0.50, etc.)
        if pass_rate >= 0.50:
            self.consistency_streak += 1
            consistency_bonus = 0.05 * self.consistency_streak
            bonus += consistency_bonus
            if self.consistency_streak >= 5:
                print(f"  [★ consistency] {self.consistency_streak} episodes "
                      f"≥50% pass rate  bonus={consistency_bonus:.3f}", flush=True)
        else:
            self.consistency_streak = 0

        # ── record for next episode ───────────────────────────────────────────
        self.ep_pass_rates.append(pass_rate)
        self.ep_avg_rets.append(avg_ret)
        self.ep_avg_dds.append(avg_dd)

        return bonus


# ── single phase training loop ────────────────────────────────────────────────

def run_phase(
    phase:       int,
    env:         BatchedFTMOEnv,
    agent:       DQNAgent,
    cfg:         dict,
    ckpt_dir:    Path,
    metrics_dir: Path,
    shaper:      EpisodeRewardShaper,
    all_daily:   list,
    all_ep_rows: list,
    global_ep:   int,
    resume_ep:   int = 0,
) -> tuple:
    """
    Run one curriculum phase.
    Returns (global_ep, advanced, consecutive_pass_days_best).

    Advances when ANY batch item hits 10 consecutive PASS days,
    or when max_episodes_per_phase is reached.
    """
    ADVANCE_DAYS   = cfg.get("ADVANCE_DAYS",             10)
    MAX_EP_PHASE   = cfg.get("MAX_EPISODES_PER_PHASE",  500)
    CKPT_EVERY     = cfg.get("CHECKPOINT_EVERY",          10)
    B              = cfg["BATCH_SIZE_ENV"]
    device         = agent.device

    # per-batch consecutive PASS day counter
    consec_pass    = [0] * B
    best_consec    = 0
    ep_in_phase    = 0

    print(f"\n{'='*60}", flush=True)
    print(f"  [PHASE {phase}] Starting  (advance at {ADVANCE_DAYS} consec PASS days)", flush=True)
    print(f"{'='*60}", flush=True)

    # update env phase
    env.phase = phase

    while ep_in_phase < MAX_EP_PHASE:
        t_ep      = time.perf_counter()
        state     = env.reset()
        ep_reward = torch.zeros(B, device=device)
        dones     = torch.zeros(B, dtype=torch.bool, device=device)
        step      = 0

        while not dones.all():
            actions              = agent.select_actions(state)
            next_state, rewards, dones = env.step(actions)
            ep_reward           += rewards
            agent.store(state, actions, rewards, next_state, dones)
            agent.train_step()
            state = next_state
            step += 1

        ep_elapsed   = time.perf_counter() - t_ep
        mean_reward  = ep_reward.mean().item()
        ftmo_summary = env.episode_summary()

        # ── per-batch consecutive PASS tracking ───────────────────────────────
        # group daily_metrics_log by batch, find each batch's longest consec streak
        ep_best_consec = 0
        batch_logs = {b: [] for b in range(B)}
        for row in env.daily_metrics_log:
            batch_logs[row["batch"]].append(row["ftmo_flag"])

        for b in range(B):
            flags = batch_logs[b]
            streak = 0
            for f in flags:
                if f == "PASS":
                    streak += 1
                    consec_pass[b] = max(consec_pass[b], streak)
                else:
                    streak = 0
            ep_best_consec = max(ep_best_consec, consec_pass[b])

        best_consec = max(best_consec, ep_best_consec)

        # ── episode-level reward shaping bonus ────────────────────────────────
        ep_bonus = shaper.compute_bonus(env.daily_metrics_log, ep_best_consec)
        if ep_bonus > 0:
            # inject bonus into replay buffer as a terminal reward signal
            dummy = torch.zeros(1, env.state_dim, device=device)
            agent.memory.push(
                dummy,
                torch.zeros(1, dtype=torch.long, device=device),
                torch.tensor([ep_bonus], device=device),
                dummy,
                torch.ones(1, dtype=torch.bool, device=device),
            )

        agent.decay_epsilon(global_ep)
        ep_in_phase += 1
        global_ep   += 1

        print(
            f"  Ep {global_ep:04d} [ph{phase}] | reward={mean_reward:+.3f} "
            f"| bonus={ep_bonus:+.3f} | eps={agent.epsilon:.3f} "
            f"| best_consec={best_consec} | {ep_elapsed:.1f}s\n"
            f"         FTMO: {ftmo_summary}",
            flush=True,
        )

        # collect metrics
        all_daily.extend(env.daily_metrics_log)
        all_ep_rows.append({
            "episode":      global_ep,
            "phase":        phase,
            "total_reward": round(mean_reward, 6),
            "ep_bonus":     round(ep_bonus, 6),
            "best_consec":  best_consec,
            "steps":        step,
        })

        # checkpoint
        if ep_in_phase % CKPT_EVERY == 0:
            ck_path = ckpt_dir / f"eurusd_gpu_ph{phase}_ep{global_ep:04d}.pt"
            agent.save(str(ck_path))

        # flush CSVs every 20 episodes
        if ep_in_phase % 20 == 0:
            _flush_csvs(all_daily, all_ep_rows, metrics_dir)

        # ── advancement check ─────────────────────────────────────────────────
        if best_consec >= ADVANCE_DAYS:
            print(f"\n[PHASE {phase}] ✓ {ADVANCE_DAYS} consecutive PASS days — advancing!",
                  flush=True)
            agent.save(str(ckpt_dir / f"eurusd_gpu_ph{phase}_final.pt"))
            return global_ep, True, best_consec

    print(f"\n[PHASE {phase}] Max episodes ({MAX_EP_PHASE}) reached — advancing.", flush=True)
    agent.save(str(ckpt_dir / f"eurusd_gpu_ph{phase}_final.pt"))
    return global_ep, False, best_consec


def _flush_csvs(all_daily: list, all_ep_rows: list, metrics_dir: Path):
    if all_daily:
        pd.DataFrame(all_daily).to_csv(
            metrics_dir / "metrics_EURUSD_ftmo_gpu.csv", index=False)
    if all_ep_rows:
        pd.DataFrame(all_ep_rows).to_csv(
            metrics_dir / "episode_rewards_EURUSD_gpu.csv", index=False)


# ── main entry point ──────────────────────────────────────────────────────────

def run_training(
    cfg:         dict = None,
    resume:      bool = False,
    start_phase: int  = 0,
) -> DQNAgent:
    """
    Full 8-phase curriculum training loop.
    Automatically advances through phases 0→7.

    Args:
        cfg         : config overrides
        resume      : load latest checkpoint before starting
        start_phase : which phase to begin from (0-7)
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
    t0       = time.perf_counter()
    features = build_feature_matrix(
        ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4])
    print(f"[features] Done  shape={features.shape}  {time.perf_counter()-t0:.1f}s", flush=True)

    # ── 2. Build env + agent ──────────────────────────────────────────────────
    cfg["PHASE"] = start_phase
    env          = BatchedFTMOEnv(features, cfg, device)
    cfg["STATE_DIM"] = env.state_dim
    print(f"[env] state_dim={env.state_dim}  batch={cfg['BATCH_SIZE_ENV']}", flush=True)

    agent = DQNAgent(env.state_dim, NUM_ACTIONS, cfg, device)

    ckpt_dir    = Path(cfg["CHECKPOINT_DIR"])
    metrics_dir = Path(cfg["METRICS_DIR"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. Optional resume ────────────────────────────────────────────────────
    resume_ep = 0
    if resume:
        ck = latest_checkpoint(ckpt_dir)
        if ck:
            ckpt_meta      = torch.load(str(ck), map_location="cpu")
            ckpt_state_dim = ckpt_meta.get("state_dim", env.state_dim)
            use_partial    = (ckpt_state_dim != env.state_dim)
            if use_partial:
                print(f"[resume] state_dim {ckpt_state_dim}→{env.state_dim} "
                      f"— transfer learning", flush=True)
            agent.load(str(ck), partial=use_partial)
            try:
                resume_ep = int(ck.stem.split("_ep")[-1])
            except Exception:
                pass
        else:
            print("[resume] No checkpoint found — starting fresh.", flush=True)

    # ── 4. Full curriculum loop ───────────────────────────────────────────────
    all_daily:   list = []
    all_ep_rows: list = []
    shaper            = EpisodeRewardShaper()
    global_ep         = resume_ep

    phases = list(range(start_phase, 8))
    print(f"\n[train] Curriculum: phases {phases}", flush=True)
    print(f"[train] Advance after {cfg.get('ADVANCE_DAYS', 10)} consecutive PASS days "
          f"or {cfg.get('MAX_EPISODES_PER_PHASE', 500)} episodes per phase\n", flush=True)

    for phase in phases:
        global_ep, advanced, best_consec = run_phase(
            phase       = phase,
            env         = env,
            agent       = agent,
            cfg         = cfg,
            ckpt_dir    = ckpt_dir,
            metrics_dir = metrics_dir,
            shaper      = shaper,
            all_daily   = all_daily,
            all_ep_rows = all_ep_rows,
            global_ep   = global_ep,
            resume_ep   = resume_ep,
        )
        _flush_csvs(all_daily, all_ep_rows, metrics_dir)
        reason = "consecutive PASS days" if advanced else "episode cap"
        print(f"\n[train] Phase {phase} complete ({reason})  "
              f"best_consec={best_consec}  total_ep={global_ep}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  [DONE] All phases complete  total_episodes={global_ep}", flush=True)
    print(f"{'='*60}", flush=True)

    _flush_csvs(all_daily, all_ep_rows, metrics_dir)
    print(f"[metrics] FTMO    -> {metrics_dir / 'metrics_EURUSD_ftmo_gpu.csv'}", flush=True)
    print(f"[metrics] Rewards -> {metrics_dir / 'episode_rewards_EURUSD_gpu.csv'}", flush=True)

    return agent


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         required=True)
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--start-phase", type=int, default=0)
    parser.add_argument("--batch",       type=int, default=None)
    args = parser.parse_args()

    overrides = {"DATA_CSV_EURUSD": args.csv}
    if args.batch:
        overrides["BATCH_SIZE_ENV"] = args.batch

    run_training(cfg=overrides, resume=args.resume, start_phase=args.start_phase)
