"""
training/curriculum_trainer.py
8-phase curriculum trainer with transfer learning and optional Ray parallel runs.
Advance criteria: 5 consecutive FTMO pass-days in current phase (or 500 ep cap).
"""

from __future__ import annotations

import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from agents.dqn_agent import DQNAgent
from env.ftmo_game import FTMOGame, NUM_ACTIONS
from monitoring.wandb_logger import WandbLogger

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _paths(cfg: dict, phase: int, run_id: str) -> dict:
    base = Path(cfg["PATHS"]["weights_dir"]) / run_id / f"phase{phase}"
    base.mkdir(parents=True, exist_ok=True)
    return {
        "weights": str(base / "weights.h5"),
        "replay":  str(base / "replay.pkl"),
        "risk":    str(base / "risk.pkl"),
        "trades":  str(base / "trades.pkl"),
    }


# ── single-phase training loop ────────────────────────────────────────────────

def _latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Return the most recently modified .h5 file in ckpt_dir, or None."""
    files = sorted(ckpt_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def run_phase(
    phase_cfg:              dict,
    data_dict:              dict,
    cfg:                    dict,
    agent:                  DQNAgent,
    logger:                 Optional[WandbLogger],
    run_id:                 str,
    advance_days:           int  = 10,
    checkpoint_every:       int  = 10,
    resume_from_checkpoint: bool = False,
) -> DQNAgent:
    """
    Run one curriculum phase until 10 consecutive pass-days or data exhaustion.
    Returns the agent (with updated weights) ready for the next phase.
    """
    phase_id   = phase_cfg["id"]
    symbols    = cfg["symbols"]
    rl_cfg     = cfg["RL"]
    ftmo_cfg   = {"profit_target_pct": cfg["FTMO"]["daily_profit_target_pct"],
                  "max_dd_pct":        cfg["FTMO"]["daily_max_drawdown_pct"]}
    reward_cfg = cfg["REWARD"]
    paths      = _paths(cfg, phase_id, run_id)

    # ── metrics output paths ──────────────────────────────────────────────────
    sym_slug   = "_".join(symbols)
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    ftmo_csv_path   = metrics_dir / f"metrics_{sym_slug}_ftmo.csv"
    reward_csv_path = metrics_dir / f"episode_rewards_{sym_slug}.csv"

    # ── checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir = Path("checkpoints") / run_id / f"phase{phase_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── resume from latest checkpoint ────────────────────────────────────────
    start_episode = 0
    if resume_from_checkpoint:
        latest = _latest_checkpoint(ckpt_dir)
        if latest:
            agent.q_net.load_weights(str(latest))
            agent.r_net.load_weights(str(latest))
            # parse episode number from filename e.g. eurusd_model_ep042.h5
            try:
                start_episode = int(latest.stem.split("_ep")[-1])
            except Exception:
                start_episode = 0
            print(f"[resume] Loaded checkpoint {latest.name} (ep {start_episode})", flush=True)
        else:
            print("[resume] No checkpoint found — starting fresh.", flush=True)

    # accumulators across the whole phase
    all_daily_rows:   List[dict] = []
    all_episode_rows: List[dict] = []

    # Build environment for this phase
    training_mode = cfg["FTMO"].get("training_mode", True)
    max_trades_per_day = cfg.get("CURRICULUM", {}).get("max_trades_per_day", 800)
    env = FTMOGame(
        data_dict          = data_dict,
        symbols            = symbols,
        reward_cfg         = reward_cfg,
        ftmo_cfg           = ftmo_cfg,
        trading_mode       = cfg["TRADING_MODE"],
        curriculum_phase   = phase_id,
        risk_fractions     = agent.risk_fractions,
        lkbk               = rl_cfg["LKBK"],
        training_mode      = training_mode,
        phase_cfg          = phase_cfg,
        max_trades_per_day = max_trades_per_day,
    )

    consecutive_pass    = 0
    episode             = start_episode
    pnls                = []
    trade_logs          = pd.DataFrame()
    phase_start         = time.perf_counter()
    max_episodes        = cfg.get("CURRICULUM", {}).get("max_episodes_per_phase", 500)
    max_episode_steps   = cfg.get("CURRICULUM", {}).get("max_episode_steps", 5000)

    print(f"\n{'='*60}", flush=True)
    print(f"  [START] Phase {phase_id}: {phase_cfg['name']}", flush=True)
    print(f"{'='*60}", flush=True)

    ep_bar = tqdm(desc=f"Phase {phase_id} episodes", unit="ep", dynamic_ncols=True)

    while True:
        if env.curr_idx >= env.max_idx:
            print(f"\n[Phase {phase_id}] Data exhausted after {episode} episodes.", flush=True)
            break
        if episode >= max_episodes:
            print(f"\n[Phase {phase_id}] Max episodes ({max_episodes}) reached — advancing.", flush=True)
            break

        episode += 1
        ep_start  = time.perf_counter()
        # Save position before reset so each episode picks up where the last ended.
        # reset() rewinds to init_idx; we advance past it to continue sequentially.
        saved_idx = env.curr_idx
        env.reset()
        if saved_idx > env.init_idx:
            env.curr_idx = saved_idx
        ep_start_date = str(env._curr_time().date())
        state_tp1 = env.get_state()
        eps       = DQNAgent.epsilon(episode, rl_cfg["EPSILON"], rl_cfg["EPS_MIN"])
        game_over      = False
        step           = 0
        episode_reward = 0.0

        train_every = rl_cfg.get("TRAIN_EVERY", 4)
        while not game_over:
            if env.curr_idx >= env.max_idx or step >= max_episode_steps:
                game_over = True
                break

            step     += 1
            state_t   = state_tp1
            actions   = agent.select_actions(state_t, eps)
            reward, game_over = env.act(actions)
            episode_reward += reward
            env.step()
            state_tp1 = env.get_state()

            flat_action = agent.actions_to_flat(actions)
            agent.exp_replay.remember(
                [state_t, flat_action, reward, state_tp1], game_over
            )
            if step % train_every == 0:
                agent.train_step(rl_cfg["BATCH_SIZE"])

            if game_over and rl_cfg["UPDATE_QR"]:
                agent.sync_r_net()

        ep_elapsed    = time.perf_counter() - ep_start
        ep_end_date   = str(env._curr_time().date())
        pnls.append(env.equity - env.initial_equity)

        # collect daily FTMO rows emitted this episode
        all_daily_rows.extend(env.daily_metrics_log)

        # record episode reward
        ep_row = {
            "episode":      episode,
            "total_reward": round(episode_reward, 6),
            "start_date":   ep_start_date,
            "end_date":     ep_end_date,
        }
        all_episode_rows.append(ep_row)
        print(
            f"  Ep {episode:04d} | reward={episode_reward:+.3f} | "
            f"equity {env.equity:,.0f} | eps {eps:.3f} | {ep_elapsed:.1f}s",
            flush=True,
        )
        last_result = env.ftmo_day.classify() if env._day_results else "ok"
        if last_result == "pass":
            consecutive_pass += 1
        else:
            consecutive_pass = 0

        ep_bar.update(1)
        ep_bar.set_postfix(
            equity=f"{env.equity:,.0f}",
            streak=env.days_in_streak,
            consec=consecutive_pass,
            eps=f"{eps:.3f}",
            secs=f"{ep_elapsed:.1f}s",
        )
        if logger:
            logger.log({
                "phase":     phase_id,
                "episode":   episode,
                "equity":    env.equity,
                "streak":    env.days_in_streak,
                "eps":       eps,
                "last_day":  last_result,
            })

        # ── checkpoint every N episodes ───────────────────────────────────────
        if episode % checkpoint_every == 0:
            ckpt_file = ckpt_dir / f"{sym_slug}_model_ep{episode:04d}.h5"
            agent.q_net.save_weights(str(ckpt_file))
            print(f"  [ckpt] saved {ckpt_file.name}", flush=True)

        # ── flush metrics CSVs every 50 episodes ─────────────────────────────
        if episode % 50 == 0:
            agent.save(paths["weights"], paths["replay"], paths["risk"])
            tl = pd.DataFrame(env.trade_log)
            tl.to_pickle(paths["trades"])
            if all_daily_rows:
                pd.DataFrame(all_daily_rows).to_csv(ftmo_csv_path, index=False)
            if all_episode_rows:
                pd.DataFrame(all_episode_rows).to_csv(reward_csv_path, index=False)

        # Advance phase when consecutive pass-days target reached
        if consecutive_pass >= advance_days:
            print(f"\n[Phase {phase_id}] Achieved {advance_days} consecutive pass-days -> advancing!",
                  flush=True)
            break

    ep_bar.close()
    phase_elapsed = time.perf_counter() - phase_start
    print(f"[DONE]  Phase {phase_id} — {episode} episodes in "
          f"{phase_elapsed:.1f}s ({phase_elapsed/60:.1f} min)", flush=True)
    agent.save(paths["weights"], paths["replay"], paths["risk"])

    # final CSV flush
    if all_daily_rows:
        pd.DataFrame(all_daily_rows).to_csv(ftmo_csv_path, index=False)
        print(f"[metrics] FTMO daily log  -> {ftmo_csv_path}", flush=True)
    if all_episode_rows:
        pd.DataFrame(all_episode_rows).to_csv(reward_csv_path, index=False)
        print(f"[metrics] Episode rewards -> {reward_csv_path}", flush=True)

    return agent


# ── full curriculum run ───────────────────────────────────────────────────────

def run_curriculum(
    config_path: str,
    data_dict:   dict,
    run_id:      str          = None,
    start_phase: int          = 1,
    freeze_layers: int        = 0,
) -> DQNAgent:
    """
    Run all 4 phases sequentially, transferring weights between phases.
    """
    cfg       = load_config(config_path)
    run_id    = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    symbols   = cfg["symbols"]
    rl_cfg    = cfg["RL"]
    logger    = WandbLogger(cfg, run_id=run_id) if not rl_cfg.get("TEST_MODE") else None

    # Build a dummy env to get state_dim
    training_mode = cfg["FTMO"].get("training_mode", True)
    ftmo_cfg = {"profit_target_pct": cfg["FTMO"]["daily_profit_target_pct"],
                "max_dd_pct":        cfg["FTMO"]["daily_max_drawdown_pct"]}
    env0 = FTMOGame(
        data_dict=data_dict, symbols=symbols,
        reward_cfg=cfg["REWARD"], ftmo_cfg=ftmo_cfg,
        trading_mode=cfg["TRADING_MODE"], curriculum_phase=0,
        lkbk=rl_cfg["LKBK"],
        training_mode=training_mode,
    )
    state_dim = env0.get_state().shape[1]

    agent = DQNAgent(
        symbols        = symbols,
        state_dim      = state_dim,
        rl_config      = rl_cfg,
        risk_fractions = cfg["ACTIONS"]["risk_fractions"],
    )

    # Optionally preload weights
    if rl_cfg.get("PRELOAD") and start_phase > 1:
        prev_paths = _paths(cfg, start_phase - 1, run_id)
        agent.load(prev_paths["weights"], prev_paths["replay"],
                   prev_paths["risk"], freeze_layers=freeze_layers)

    advance = cfg["CURRICULUM"]["advance_consecutive_pass_days"]
    for phase_cfg in cfg["CURRICULUM"]["phases"]:
        if phase_cfg["id"] < start_phase:
            continue
        agent = run_phase(phase_cfg, data_dict, cfg, agent, logger, run_id, advance)
        # Transfer: reload with freeze for next phase
        paths = _paths(cfg, phase_cfg["id"], run_id)
        if phase_cfg["id"] < 7:
            agent.load(paths["weights"], paths["replay"], paths["risk"],
                       freeze_layers=freeze_layers)

    if logger:
        logger.finish()

    return agent


# ── forward test ──────────────────────────────────────────────────────────────

def forward_test(
    config_path: str,
    data_dict:   dict,
    agent:       DQNAgent,
    run_id:      str = "fwd_test",
) -> pd.DataFrame:
    """
    Run agent in inference mode on forward-test data.  No weight updates.
    Uses training_mode=False (live MT5 mode: hard stop on daily target).
    """
    cfg      = load_config(config_path)
    symbols  = cfg["symbols"]
    ftmo_cfg = {"profit_target_pct": cfg["FTMO"]["daily_profit_target_pct"],
                "max_dd_pct":        cfg["FTMO"]["daily_max_drawdown_pct"]}

    env = FTMOGame(
        data_dict=data_dict, symbols=symbols,
        reward_cfg=cfg["REWARD"], ftmo_cfg=ftmo_cfg,
        trading_mode=cfg["TRADING_MODE"], curriculum_phase=7,
        lkbk=cfg["RL"]["LKBK"],
        training_mode=False,
    )

    env.reset()
    total_steps = env.max_idx - env.curr_idx
    results     = []
    fwd_start   = time.perf_counter()
    print(f"[START] forward_test — {total_steps:,} steps", flush=True)

    with tqdm(total=total_steps, desc="Forward test", unit="bar",
              dynamic_ncols=True) as pbar:
        while env.curr_idx < env.max_idx:
            state   = env.get_state()
            actions = agent.select_actions(state, epsilon=0.0)   # greedy
            env.act(actions)
            env.step()
            results.append({
                "time":   env._curr_time(),
                "equity": env.equity,
                "streak": env.days_in_streak,
            })
            pbar.update(1)
            if len(results) % 10_000 == 0:
                pbar.set_postfix(equity=f"{env.equity:,.0f}",
                                 streak=env.days_in_streak)

    elapsed = time.perf_counter() - fwd_start
    df = pd.DataFrame(results)

    # Compute daily metrics for the forward test output
    if not df.empty:
        df["date"] = pd.to_datetime(df["time"]).dt.date
        daily = (
            df.groupby("date")
            .agg(start_equity=("equity", "first"),
                 end_equity=("equity", "last"),
                 peak_equity=("equity", "max"))
            .reset_index()
        )
        daily["daily_return_pct"] = (
            (daily["end_equity"] - daily["start_equity"]) / daily["start_equity"] * 100
        )
        daily["daily_max_dd_pct"] = (
            (daily["peak_equity"] - daily["end_equity"]) / daily["peak_equity"] * 100
        )
        daily["result"] = daily.apply(
            lambda r: ("pass" if r["daily_return_pct"] / 100 >= cfg["FTMO"]["daily_profit_target_pct"]
                       and r["daily_max_dd_pct"] / 100 <= cfg["FTMO"]["daily_max_drawdown_pct"]
                       else ("ok" if r["daily_return_pct"] >= 0 else "fail")),
            axis=1,
        )
        out_path = Path(cfg["PATHS"]["trade_logs_dir"]) / run_id / "forward_test.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(out_path, index=False)
        print(f"[DONE]  forward_test — {len(daily)} days  {elapsed:.1f}s  "
              f"saved to {out_path}", flush=True)
        return daily

    out_path = Path(cfg["PATHS"]["trade_logs_dir"]) / run_id / "forward_test.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[DONE]  forward_test — {elapsed:.1f}s  saved to {out_path}", flush=True)
    return df


# ── Ray parallel launcher ─────────────────────────────────────────────────────

def spark_train(
    config_path:  str,
    data_dict:    dict,
    num_workers:  int = 4,
    num_episodes: int = 1000,
):
    """
    Run multiple independent curriculum training jobs in parallel via Ray.
    Each worker gets a unique run_id and writes to its own directory.
    """
    if not _RAY_AVAILABLE:
        print("[spark_train] Ray not installed. Running single job.")
        return run_curriculum(config_path, data_dict)

    ray.init(ignore_reinit_error=True)

    @ray.remote
    def _worker(worker_id):
        run_id = f"worker_{worker_id}_{datetime.now().strftime('%H%M%S')}"
        np.random.seed(worker_id)
        return run_curriculum(config_path, data_dict, run_id=run_id)

    futures = [_worker.remote(i) for i in range(num_workers)]
    agents  = ray.get(futures)
    ray.shutdown()
    return agents
