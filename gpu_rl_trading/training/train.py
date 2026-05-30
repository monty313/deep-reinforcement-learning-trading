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


def _ckpt_episode(p: Path) -> int:
    """Extract episode number from checkpoint filename for sorting. Returns -1 if unparseable."""
    try:
        return int(p.stem.split("_ep")[-1]) if "_ep" in p.stem else -1
    except ValueError:
        return -1


def latest_checkpoint(ckpt_dir: Path, phase: int = None) -> Path | None:
    """Return the checkpoint with the highest episode number (not mtime)."""
    pattern = f"*ph{phase}*.pt" if phase is not None else "*.pt"
    files   = sorted(ckpt_dir.glob(pattern), key=_ckpt_episode)
    return files[-1] if files else None


# ── Episode-level potential-based reward shaper ───────────────────────────────

class EpisodeRewardShaper:
    """
    Computes a single episode-end bonus using the same Φ potential function
    used inside the environment for day-level shaping.

    Φ_episode = (pass_rate × avg_ret_normalised) / (1 + λ × avg_dd_normalised)

    The bonus is the change in Φ relative to a smoothed rolling average of
    recent episodes, normalised by the running std-dev.  This is the
    potential-based form:  bonus = clip(α × (Φ_ep - Φ_smooth) / σ_Φ, -c, +c)

    Because Φ is normalised to configured targets (not raw percentages), the
    same α and clip values work for any daily_target_pct / max_dd_pct combination
    — fully supporting parameterizable targets without retuning.
    """

    def __init__(self, cfg: dict):
        self.target_pct  = float(cfg["DAILY_TARGET_PCT"])
        self.max_dd_pct  = float(cfg["DAILY_MAX_DD_PCT"])
        self.alpha       = float(cfg.get("SHAPE_ALPHA",  0.01))
        self.clip_val    = float(cfg.get("SHAPE_CLIP",   0.03))
        self.lam         = float(cfg.get("SHAPE_LAMBDA", 5.0))
        self.warmup      = int(  cfg.get("SHAPE_WARMUP", 50))
        self.window      = 20   # smoothing window for Φ history

        self._phi_history:  List[float] = []   # recent Φ values for smoothing
        self._ath_phi:      float       = 0.0  # all-time high Φ
        self.global_ep:     int         = 0    # set by trainer each episode

    def _phi(self, pass_rate: float, avg_ret: float, avg_dd: float) -> float:
        """
        Compute Φ normalised to configured target/risk.
        Φ = 1.0 means hitting the target exactly every day with no dd.
        Φ scales correctly for any target_pct / max_dd_pct combination.
        """
        ret_norm = avg_ret / (self.target_pct * 100.0 + 1e-8)
        dd_norm  = avg_dd  / (self.max_dd_pct  * 100.0 + 1e-8)
        return (pass_rate * max(ret_norm, 0.0)) / (1.0 + self.lam * dd_norm)

    def compute_bonus(self, daily_log: list) -> float:
        """
        Return scalar episode-end bonus reward (added to replay as terminal signal).
        Returns 0.0 during warm-up.
        """
        if not daily_log or self.global_ep < self.warmup:
            # still in warm-up — record history but no bonus yet
            if daily_log:
                flags  = [r["ftmo_flag"]              for r in daily_log]
                rets   = [r["daily_return_pct"]       for r in daily_log]
                dds    = [r["daily_max_drawdown_pct"] for r in daily_log]
                n      = len(flags)
                phi_ep = self._phi(flags.count("PASS") / n,
                                   sum(rets) / n, sum(dds) / n)
                self._phi_history.append(phi_ep)
            return 0.0

        flags     = [r["ftmo_flag"]              for r in daily_log]
        rets      = [r["daily_return_pct"]       for r in daily_log]
        dds       = [r["daily_max_drawdown_pct"] for r in daily_log]
        n         = len(flags)
        passes    = flags.count("PASS")
        pass_rate = passes / n
        avg_ret   = sum(rets) / n
        avg_dd    = sum(dds)  / n

        phi_ep = self._phi(pass_rate, avg_ret, avg_dd)

        # smoothed baseline = mean of recent Φ history
        phi_smooth = float(np.mean(self._phi_history[-self.window:])) \
                     if self._phi_history else 0.0
        sigma      = float(np.std( self._phi_history[-self.window:])) \
                     if len(self._phi_history) > 1 else 1.0
        sigma      = max(sigma, 1e-4)

        # potential-based shaping: reward progress, penalise regression
        delta   = phi_ep - phi_smooth
        bonus   = float(np.clip(self.alpha * delta / sigma,
                                -self.clip_val, self.clip_val))

        # ATH Φ milestone — one-time bonus for reaching a new best
        if phi_ep > self._ath_phi:
            ath_bonus      = min((phi_ep - self._ath_phi) * 0.10, 0.05)
            bonus         += ath_bonus
            self._ath_phi  = phi_ep
            print(f"  [★ ATH Φ={phi_ep:.4f}]  pass={pass_rate:.1%}  "
                  f"ret={avg_ret:+.2f}%  dd={avg_dd:.2f}%  bonus={ath_bonus:.4f}",
                  flush=True)

        self._phi_history.append(phi_ep)
        if len(self._phi_history) > 100:
            self._phi_history.pop(0)

        if bonus != 0.0:
            print(f"  [Φ shaping] ep={self.global_ep}  "
                  f"Φ={phi_ep:.4f}  smooth={phi_smooth:.4f}  "
                  f"bonus={bonus:+.4f}", flush=True)

        return bonus


# ── single phase training loop ────────────────────────────────────────────────

def run_phase(
    phase:             int,
    env:               BatchedFTMOEnv,
    agent:             DQNAgent,
    cfg:               dict,
    ckpt_dir:          Path,
    metrics_dir:       Path,
    shaper:            EpisodeRewardShaper,
    all_daily:         list,
    all_ep_rows:       list,
    global_ep:         int,
    resume_consec:     list = None,  # restored consec_pass from checkpoint
    resume_ep_in_phase: int = 0,     # episodes already completed in this phase
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
    # restore from checkpoint if available so phase advancement progress is preserved
    if resume_consec and len(resume_consec) != B:
        print(f"  [resume] WARNING: BATCH_SIZE_ENV changed "
              f"({len(resume_consec)} → {B}) — consec_pass reset to 0", flush=True)
    consec_pass = resume_consec if resume_consec and len(resume_consec) == B else [0] * B
    best_consec = max(consec_pass)
    ep_in_phase = resume_ep_in_phase

    if resume_consec or resume_ep_in_phase > 0:
        print(f"  [resume] consec_pass={consec_pass}  best={best_consec}"
              f"  ep_in_phase={ep_in_phase}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  [PHASE {phase}] Starting  (advance at {ADVANCE_DAYS} consec PASS days)", flush=True)
    print(f"{'='*60}", flush=True)

    # update env phase and reset Φ history so new phase has clean normalization baseline
    env.phase = phase
    env.reset_phi_history()

    while ep_in_phase < MAX_EP_PHASE:
        t_ep      = time.perf_counter()
        env.start_episode(global_ep)
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

        # ── episode-level Φ shaping bonus ─────────────────────────────────────
        shaper.global_ep = global_ep
        ep_bonus = shaper.compute_bonus(env.daily_metrics_log)
        if ep_bonus != 0.0:
            # Add to the last real terminal transition in the replay buffer
            # by storing a proper (s, a, r=bonus, s', done=True) tuple using
            # the final state the episode ended on — not a dummy zero state.
            final_state = state   # state after last env.step()
            agent.memory.push(
                final_state,
                torch.zeros(B, dtype=torch.long, device=device),
                torch.full((B,), ep_bonus, device=device),
                final_state,
                torch.ones(B, dtype=torch.bool, device=device),
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

        # checkpoint — save all resume state so resume is exact
        if ep_in_phase % CKPT_EVERY == 0:
            ck_path = ckpt_dir / f"eurusd_gpu_ph{phase}_ep{global_ep:04d}.pt"
            agent.save(str(ck_path), extra={
                "phase":          phase,
                "consec_pass":    consec_pass,
                "ep_in_phase":    ep_in_phase,
                "global_ep":      global_ep,   # authoritative episode counter
            })

        # flush CSVs every 20 episodes
        if ep_in_phase % 20 == 0:
            _flush_csvs(all_daily, all_ep_rows, metrics_dir)

        # ── advancement check ─────────────────────────────────────────────────
        if best_consec >= ADVANCE_DAYS:
            print(f"\n[PHASE {phase}] ✓ {ADVANCE_DAYS} consecutive PASS days — advancing!",
                  flush=True)
            agent.save(str(ckpt_dir / f"eurusd_gpu_ph{phase}_final.pt"),
                       extra={"phase": phase, "consec_pass": consec_pass,
                              "ep_in_phase": ep_in_phase, "global_ep": global_ep})
            return global_ep, True, best_consec

    print(f"\n[PHASE {phase}] Max episodes ({MAX_EP_PHASE}) reached — advancing.", flush=True)
    agent.save(str(ckpt_dir / f"eurusd_gpu_ph{phase}_final.pt"),
               extra={"phase": phase, "consec_pass": consec_pass,
                      "ep_in_phase": ep_in_phase, "global_ep": global_ep})
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
    resume_ep          = 0
    resume_phase       = start_phase
    resume_consec      = None   # restored per-batch consec_pass list
    resume_ep_in_phase = 0      # episodes already done in the resumed phase

    if resume:
        # Find latest checkpoint matching start_phase first, then fall back to any
        ck = latest_checkpoint(ckpt_dir, phase=start_phase)
        if ck is None:
            ck = latest_checkpoint(ckpt_dir)   # fallback: any phase

        if ck:
            ckpt_meta      = torch.load(str(ck), map_location="cpu")
            ckpt_state_dim = ckpt_meta.get("state_dim", env.state_dim)
            ckpt_phase     = ckpt_meta.get("phase", start_phase)
            use_partial    = (ckpt_state_dim != env.state_dim)

            if ckpt_phase != start_phase:
                print(f"[resume] WARNING: checkpoint is phase {ckpt_phase} "
                      f"but start_phase={start_phase} — loading anyway", flush=True)

            if use_partial:
                print(f"[resume] state_dim {ckpt_state_dim}→{env.state_dim} "
                      f"— transfer learning", flush=True)

            ckpt_returned = agent.load(str(ck), partial=use_partial)

            # extract episode number — prefer metadata over filename
            meta_ep   = ckpt_returned.get("global_ep", None)
            fname_ep  = _ckpt_episode(ck)

            if meta_ep is not None:
                resume_ep = meta_ep
                if fname_ep >= 0 and fname_ep != meta_ep:
                    print(f"[resume] NOTE: filename says ep {fname_ep} "
                          f"but metadata says ep {meta_ep} — using metadata", flush=True)
            elif fname_ep >= 0:
                resume_ep = fname_ep
                print(f"[resume] episode from filename: {resume_ep}", flush=True)
            else:
                print(f"[resume] WARNING: cannot determine episode from '{ck.stem}' "
                      f"— starting from ep 0", flush=True)

            # validate phase match
            if ckpt_phase != start_phase:
                print(f"[resume] WARNING: checkpoint phase={ckpt_phase} "
                      f"but start_phase={start_phase} — proceeding", flush=True)

            # restore consec_pass, ep_in_phase if saved
            resume_consec      = ckpt_returned.get("consec_pass", None)
            resume_ep_in_phase = ckpt_returned.get("ep_in_phase", 0)
            if resume_consec:
                print(f"[resume] consec_pass: {resume_consec}", flush=True)
            if resume_ep_in_phase:
                print(f"[resume] ep_in_phase: {resume_ep_in_phase}", flush=True)

            print(f"[resume] ✓ ep={resume_ep}  phase={ckpt_phase}"
                  f"  ep_in_phase={resume_ep_in_phase}  "
                  f"replay={agent.memory.size}  epsilon={agent.epsilon:.3f}",
                  flush=True)
        else:
            print("[resume] No checkpoint found — starting fresh.", flush=True)

    # ── 4. Full curriculum loop ───────────────────────────────────────────────
    all_daily:   list = []
    all_ep_rows: list = []
    shaper            = EpisodeRewardShaper(cfg)
    global_ep         = resume_ep

    phases = list(range(start_phase, 8))
    print(f"\n[train] Curriculum: phases {phases}", flush=True)
    print(f"[train] Advance after {cfg.get('ADVANCE_DAYS', 10)} consecutive PASS days "
          f"or {cfg.get('MAX_EPISODES_PER_PHASE', 500)} episodes per phase\n", flush=True)

    for phase in phases:
        # only restore resume state on the first phase — subsequent phases start fresh
        phase_consec      = resume_consec      if phase == resume_phase else None
        phase_ep_in_phase = resume_ep_in_phase if phase == resume_phase else 0
        global_ep, advanced, best_consec = run_phase(
            phase              = phase,
            env                = env,
            agent              = agent,
            cfg                = cfg,
            ckpt_dir           = ckpt_dir,
            metrics_dir        = metrics_dir,
            shaper             = shaper,
            all_daily          = all_daily,
            all_ep_rows        = all_ep_rows,
            global_ep          = global_ep,
            resume_consec      = phase_consec,
            resume_ep_in_phase = phase_ep_in_phase,
        )
        resume_consec      = None   # only used on first phase
        resume_ep_in_phase = 0
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
