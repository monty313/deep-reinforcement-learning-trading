"""
training/curriculum_trainer.py
4-phase curriculum trainer with transfer learning and optional Ray parallel runs.
Advance criteria: 10 consecutive FTMO pass-days in current phase.
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

def run_phase(
    phase_cfg:     dict,
    data_dict:     dict,
    cfg:           dict,
    agent:         DQNAgent,
    logger:        Optional[WandbLogger],
    run_id:        str,
    advance_days:  int = 10,
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

    # Build environment for this phase
    training_mode = cfg["FTMO"].get("training_mode", True)
    env = FTMOGame(
        data_dict        = data_dict,
        symbols          = symbols,
        reward_cfg       = reward_cfg,
        ftmo_cfg         = ftmo_cfg,
        trading_mode     = cfg["TRADING_MODE"],
        curriculum_phase = phase_id,
        risk_fractions   = agent.risk_fractions,
        lkbk             = rl_cfg["LKBK"],
        training_mode    = training_mode,  # Pass training mode for curriculum
    )

    consecutive_pass    = 0
    episode             = 0
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
        state_tp1 = env.get_state()
        eps       = DQNAgent.epsilon(episode, rl_cfg["EPSILON"], rl_cfg["EPS_MIN"])
        game_over = False
        step      = 0

        train_every = rl_cfg.get("TRAIN_EVERY", 4)
        while not game_over:
            if env.curr_idx >= env.max_idx or step >= max_episode_steps:
                game_over = True
                break

            step     += 1
            state_t   = state_tp1
            actions   = agent.select_actions(state_t, eps)
            reward, game_over = env.act(actions)
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

        ep_elapsed = time.perf_counter() - ep_start
        pnls.append(env.equity - env.initial_equity)
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
        if episode % 50 == 0 or consecutive_pass >= advance_days - 1:
            print(f"  Ep {episode:04d} | ph{phase_id} | "
                  f"equity {env.equity:,.0f} | streak {env.days_in_streak} | "
                  f"consec_pass {consecutive_pass} | eps {eps:.4f} | "
                  f"{ep_elapsed:.1f}s", flush=True)

        if logger:
            logger.log({
                "phase":     phase_id,
                "episode":   episode,
                "equity":    env.equity,
                "streak":    env.days_in_streak,
                "eps":       eps,
                "last_day":  last_result,
            })

        # Save every 50 episodes
        if not episode % 50:
            agent.save(paths["weights"], paths["replay"], paths["risk"])
            tl = pd.DataFrame(env.trade_log)
            tl.to_pickle(paths["trades"])

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
        trading_mode=cfg["TRADING_MODE"], curriculum_phase=1,
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
        if phase_cfg["id"] < 4:
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
        trading_mode=cfg["TRADING_MODE"], curriculum_phase=4,
        lkbk=cfg["RL"]["LKBK"],
        training_mode=False,  # Live mode: hard stop on daily target
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
