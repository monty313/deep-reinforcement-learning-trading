"""
test_run.py
Quick smoke-test: loads 2 months of data for all 4 symbols,
runs a handful of RL episodes, then writes metrics.csv with
daily returns and drawdowns per symbol and for the combined account.

Run:
    python test_run.py

Output:
    metrics.csv  (daily metrics for each symbol + combined account)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.loader import load_all
from env.data_bridge import build_feature_data, compute_init_idx
from env.ftmo_game import FTMOGame, NUM_ACTIONS
from agents.dqn_agent import DQNAgent

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = "config/training_config.yaml"
TEST_FROM   = "2023-01-01"   # 2-month window
TEST_TO     = "2023-02-28"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# Override for speed
cfg["RL"]["TEST_MODE"]  = True
cfg["RL"]["MAX_MEM"]    = 200
cfg["RL"]["BATCH_SIZE"] = 16
cfg["RL"]["HIDDEN_MULT"] = 1   # smaller network
cfg["RL"]["LKBK"]       = 5   # short lookback -> smaller state

# Step every N 1m bars so the loop finishes quickly (skips bars, doesn't skip act)
STEP_EVERY = 60   # act once per hour of 1m bars

symbols  = cfg["symbols"]
rl_cfg   = cfg["RL"]
ftmo_cfg = {
    "profit_target_pct": cfg["FTMO"]["daily_profit_target_pct"],
    "max_dd_pct":        cfg["FTMO"]["daily_max_drawdown_pct"],
}

print(f"Smoke test: {TEST_FROM} to {TEST_TO}  |  symbols: {symbols}")

# ── 1. Load & build features ──────────────────────────────────────────────────
print("\nLoading data ...")
raw = load_all(
    symbols   = symbols,
    csv_map   = cfg.get("csv_map"),
    date_from = TEST_FROM,
    date_to   = TEST_TO,
)

print("Building features ...")
feat = build_feature_data(raw, symbols)
init_idx = compute_init_idx(feat, symbols)
print(f"init_idx = {init_idx}")

# ── 2. Build agent ────────────────────────────────────────────────────────────
env0 = FTMOGame(
    data_dict        = feat,
    symbols          = symbols,
    reward_cfg       = cfg["REWARD"],
    ftmo_cfg         = ftmo_cfg,
    trading_mode     = cfg["TRADING_MODE"],
    curriculum_phase = 7,          # no masking for smoke test
    risk_fractions   = cfg["ACTIONS"]["risk_fractions"],
    lkbk             = rl_cfg["LKBK"],
    init_idx         = init_idx,
)
state_dim = env0.get_state().shape[1]
print(f"State dim: {state_dim}")

agent = DQNAgent(
    symbols        = symbols,
    state_dim      = state_dim,
    rl_config      = rl_cfg,
    risk_fractions = cfg["ACTIONS"]["risk_fractions"],
)

# ── 3. Run episodes ───────────────────────────────────────────────────────────
env   = env0
env.reset()
episode    = 0
max_ep     = 7      # TEST_MODE keeps it short
equity_log = []     # [(timestamp, equity)]

print("\nRunning episodes ...")
while env.curr_idx < env.max_idx and episode < max_ep:
    episode += 1
    eps     = DQNAgent.epsilon(episode, rl_cfg["EPSILON"], rl_cfg["EPS_MIN"])
    game_over = False

    while not game_over and env.curr_idx < env.max_idx:
        state_t  = env.get_state()
        actions  = agent.select_actions(state_t, eps)
        reward, game_over = env.act(actions)

        equity_log.append({
            "time":   env._curr_time(),
            "equity": env.equity,
        })

        # Advance STEP_EVERY bars between decisions (smoke test only)
        for _ in range(STEP_EVERY):
            env.step()
            if env.curr_idx >= env.max_idx:
                game_over = True
                break

        state_tp1 = env.get_state()

        flat_action = agent.actions_to_flat(actions)
        agent.exp_replay.remember(
            [state_t, flat_action, reward, state_tp1], game_over
        )
        agent.train_step(rl_cfg["BATCH_SIZE"])
        if game_over and rl_cfg["UPDATE_QR"]:
            agent.sync_r_net()

    print(f"  Episode {episode} | equity {env.equity:,.2f} | streak {env.days_in_streak}")

# ── 4. Build metrics.csv ──────────────────────────────────────────────────────
eq_df = pd.DataFrame(equity_log)

if eq_df.empty:
    print("No equity data captured — check init_idx vs data length.")
    sys.exit(0)

eq_df["date"] = pd.to_datetime(eq_df["time"]).dt.date

daily = (
    eq_df.groupby("date")
    .agg(start_equity=("equity", "first"),
         end_equity  =("equity", "last"),
         peak_equity =("equity", "max"))
    .reset_index()
)
daily["daily_return_pct"] = (
    (daily["end_equity"] - daily["start_equity"]) / daily["start_equity"] * 100
)
daily["daily_max_dd_pct"] = (
    (daily["peak_equity"] - daily["end_equity"]) / daily["peak_equity"] * 100
)
daily["target_hit"] = daily["daily_return_pct"] >= 2.5
daily["dd_breach"]  = daily["daily_max_dd_pct"] > 1.0
daily["result"]     = daily.apply(
    lambda r: "pass" if r["target_hit"] and not r["dd_breach"]
    else ("ok" if r["daily_return_pct"] >= 0 and not r["dd_breach"]
          else "fail"),
    axis=1,
)
daily["symbols"]    = str(symbols)

out_path = Path(cfg["PATHS"]["metrics_file"])
out_path.parent.mkdir(parents=True, exist_ok=True)
daily.to_csv(out_path, index=False)
print(f"\nOK metrics.csv written to {out_path.resolve()}")
print(daily[["date", "daily_return_pct", "daily_max_dd_pct", "result"]].to_string(index=False))

# Per-symbol breakdown (approximated from trade log)
if env.trade_log:
    trades_df = pd.DataFrame(env.trade_log)
    per_sym = (
        trades_df.groupby("symbol")
        .agg(trades=("pnl_pct", "count"),
             avg_pnl =("pnl_pct", "mean"),
             total_pnl=("pnl_pct", "sum"))
        .reset_index()
    )
    sym_path = out_path.parent / "metrics_per_symbol.csv"
    per_sym.to_csv(sym_path, index=False)
    print(f"\nPer-symbol summary saved to {sym_path.resolve()}")
    print(per_sym.to_string(index=False))

# ── 5. Extended metrics summary (Sharpe, Sortino, profit factor, tail risk) ───
from monitoring.metrics_calculator import build_metrics_summary, save_metrics_summary
from monitoring.tail_risk_analyzer  import tail_risk_report
from monitoring.slippage_analyzer   import compare_backtest_vs_live_metrics
from monitoring.trade_duration_analyzer import save_duration_report

equity_series  = pd.Series(daily["end_equity"].values)
daily_ret_frac = daily["daily_return_pct"] / 100.0
trades_list    = env.trade_log

summary = build_metrics_summary(
    equity_series = equity_series,
    daily_returns = daily_ret_frac,
    trades        = trades_list,
    extra         = {
        "pass_rate":    float((daily["result"] == "pass").mean()),
        "total_fees":   getattr(env, "total_fees", 0.0),
        "date_from":    TEST_FROM,
        "date_to":      TEST_TO,
        "symbols":      symbols,
    },
)
save_metrics_summary(summary, "metrics_summary.json")
print(f"\nmetrics_summary.json: Sharpe={summary['sharpe']:.3f}  "
      f"Sortino={summary['sortino']:.3f}  MaxDD={summary['max_dd']:.4f}  "
      f"PF={summary['profit_factor']:.3f}  HitRate={summary['hit_rate']:.2%}")

# Tail risk report
tail = tail_risk_report(daily, "tail_risk_report.json")
print(f"tail_risk_report.json: worst day={tail.get('worst_single_day',0)*100:.3f}%  "
      f"flag_review={tail.get('flag_review', False)}")

# Slippage report (simulated — no live CSV yet)
slippage = compare_backtest_vs_live_metrics(str(out_path), out_path="slippage_impact_report.json")
print(f"slippage_impact_report.json: {slippage['alert_message']}")

# Trade duration report
if trades_list:
    dur = save_duration_report(trades_list, "trade_duration_report.json")
    print(f"trade_duration_report.json: avg={dur.get('avg_min','?')} min  "
          f"{dur.get('recommendation','')}")
