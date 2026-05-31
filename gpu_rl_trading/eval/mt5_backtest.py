"""
gpu_rl_trading/eval/mt5_backtest.py

Backtest the trained DQN agent using historical bars pulled directly from
a running MT5 terminal — no CSV file needed.

REQUIREMENTS:
  - Windows PC with MetaTrader 5 installed and logged into a demo/live account
  - pip install MetaTrader5
  - The MT5 terminal must be OPEN and logged in before running this script

HOW IT WORKS:
  1. Connects to your running MT5 terminal
  2. Downloads 1m OHLCV bars for the symbol/date range you specify
  3. Builds the same feature matrix used during training
  4. Runs the trained agent bar-by-bar in greedy mode (epsilon=0)
  5. Simulates FTMO-style P&L and logs every day as PASS / OK / FAIL
  6. Prints a full summary and saves results to CSV

USAGE (Python script):
    python -m gpu_rl_trading.eval.mt5_backtest ^
        --checkpoint "C:/path/to/eurusd_gpu_ph7_final.pt" ^
        --symbol     EURUSD ^
        --date-from  2024-01-01 ^
        --date-to    2024-12-31 ^
        --out        backtest_mt5_2024.csv

USAGE (from another Python file):
    from gpu_rl_trading.eval.mt5_backtest import run_mt5_backtest
    results = run_mt5_backtest(
        checkpoint_path = "C:/path/to/eurusd_gpu_ph7_final.pt",
        symbol          = "EURUSD",
        date_from       = "2024-01-01",
        date_to         = "2024-12-31",
        out_csv         = "backtest_mt5_2024.csv",
    )

NOTE: This runs locally on Windows — NOT in Colab. MT5 only runs on Windows.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gpu_rl_trading.config.settings import CFG
from gpu_rl_trading.env.indicators import build_feature_matrix
from gpu_rl_trading.env.environment import NUM_ACTIONS
from gpu_rl_trading.agent.dqn import DQNAgent
from gpu_rl_trading.eval.backtest import SequentialBacktestEnv, compute_metrics


# ── MT5 connection helpers ────────────────────────────────────────────────────

def _require_mt5():
    """Import MetaTrader5 or give a clear install message."""
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError:
        print(
            "\n[ERROR] MetaTrader5 package not found.\n"
            "Install it with:  pip install MetaTrader5\n"
            "Then make sure your MT5 terminal is open and logged in.\n",
            flush=True,
        )
        sys.exit(1)


def connect_mt5(mt5, login: int = None, password: str = None,
                server: str = None) -> bool:
    """
    Initialize connection to the running MT5 terminal.

    If login/password/server are None, MT5 connects to whichever account
    is already logged in on the open terminal — the most common case.
    """
    kwargs = {}
    if login:    kwargs["login"]    = login
    if password: kwargs["password"] = password
    if server:   kwargs["server"]   = server

    if not mt5.initialize(**kwargs):
        err = mt5.last_error()
        print(f"[ERROR] MT5 initialize() failed: {err}", flush=True)
        print("Make sure MetaTrader 5 is open and logged into an account.", flush=True)
        return False

    info = mt5.account_info()
    if info is None:
        print("[ERROR] Could not retrieve account info — are you logged in?", flush=True)
        return False

    print(f"[MT5] Connected: account={info.login}  broker={info.company}"
          f"  server={info.server}  balance={info.balance:.2f} {info.currency}",
          flush=True)
    return True


def fetch_bars(mt5, symbol: str, date_from: str, date_to: str) -> np.ndarray:
    """
    Download 1m OHLCV bars from MT5 for the given symbol and date range.
    Returns a (T, 5) float32 array: [open, high, low, close, volume].
    """
    from_dt = datetime.strptime(date_from, "%Y-%m-%d")
    to_dt   = datetime.strptime(date_to,   "%Y-%m-%d").replace(
                  hour=23, minute=59, second=59)

    # ensure symbol is in Market Watch
    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] Symbol '{symbol}' not found in MT5. "
              f"Add it to Market Watch and try again.", flush=True)
        sys.exit(1)

    print(f"[MT5] Downloading {symbol} M1 bars  {date_from} → {date_to} ...",
          flush=True)

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, from_dt, to_dt)

    if rates is None or len(rates) == 0:
        print(f"[ERROR] No bars returned for {symbol} in that date range.\n"
              f"  - Check that your broker has M1 history for this period\n"
              f"  - In MT5: Tools → History Center → download the data first",
              flush=True)
        sys.exit(1)

    print(f"[MT5] Downloaded {len(rates):,} bars  "
          f"({datetime.utcfromtimestamp(rates[0]['time']).date()} – "
          f"{datetime.utcfromtimestamp(rates[-1]['time']).date()})",
          flush=True)

    ohlcv = np.column_stack([
        rates["open"].astype(np.float32),
        rates["high"].astype(np.float32),
        rates["low"].astype(np.float32),
        rates["close"].astype(np.float32),
        rates["tick_volume"].astype(np.float32),
    ])
    return ohlcv


# ── main backtest function ────────────────────────────────────────────────────

def run_mt5_backtest(
    checkpoint_path: str,
    symbol:          str  = "EURUSD",
    date_from:       str  = None,
    date_to:         str  = None,
    out_csv:         str  = None,
    cfg_overrides:   dict = None,
    mt5_login:       Optional[int] = None,
    mt5_password:    Optional[str] = None,
    mt5_server:      Optional[str] = None,
) -> dict:
    """
    Connect to MT5, download bars, run the trained agent in inference mode,
    and return full backtest metrics.

    Args:
        checkpoint_path : path to .pt checkpoint (eurusd_gpu_ph7_final.pt)
        symbol          : MT5 symbol name, e.g. "EURUSD" or "EURUSD.r"
        date_from       : start date "YYYY-MM-DD" (use data AFTER training period)
        date_to         : end date "YYYY-MM-DD"
        out_csv         : optional path to save daily results CSV
        cfg_overrides   : optional dict to override CFG values
        mt5_login       : MT5 account number (optional — uses open terminal login)
        mt5_password    : MT5 password (optional)
        mt5_server      : MT5 server name (optional)

    Returns:
        dict with keys: "metrics", "daily_df", "equity_curve", "action_log"
    """
    mt5 = _require_mt5()

    # ── 1. Connect to MT5 ─────────────────────────────────────────────────────
    if not connect_mt5(mt5, mt5_login, mt5_password, mt5_server):
        sys.exit(1)

    # ── 2. Download bars ──────────────────────────────────────────────────────
    ohlcv = fetch_bars(mt5, symbol, date_from, date_to)
    mt5.shutdown()
    print("[MT5] Disconnected.", flush=True)

    # ── 3. Build features ─────────────────────────────────────────────────────
    cfg = CFG.copy()
    if cfg_overrides:
        cfg.update(cfg_overrides)

    print("[backtest] Building feature matrix ...", flush=True)
    features = build_feature_matrix(
        ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4])
    print(f"[backtest] Features shape={features.shape}", flush=True)

    # ── 4. Build env ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[backtest] device={device}", flush=True)

    cfg["PHASE"] = 7
    env = SequentialBacktestEnv(features, cfg, device)
    print(f"[backtest] state_dim={env.state_dim}  bars={env.T}  warmup={env._warmup}",
          flush=True)

    # ── 5. Load agent ─────────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_state_dim = ckpt.get("state_dim", env.state_dim)

    cfg["STATE_DIM"]      = env.state_dim
    cfg["BATCH_SIZE_ENV"] = 1
    cfg["MEMORY_SIZE"]    = 1   # no replay buffer needed for inference
    agent = DQNAgent(env.state_dim, NUM_ACTIONS, cfg, device)

    if ckpt_state_dim != env.state_dim:
        print(f"[backtest] WARNING: checkpoint state_dim={ckpt_state_dim} "
              f"vs env state_dim={env.state_dim} — partial load", flush=True)
        agent.q_net.load_partial(ckpt["q_net"], ckpt_state_dim)
    else:
        agent.q_net.load_state_dict(ckpt["q_net"])

    agent.q_net.eval()
    agent.epsilon = 0.0
    print(f"[backtest] Checkpoint loaded: {checkpoint_path}", flush=True)
    print(f"[backtest] epsilon=0.0  (greedy, no exploration)", flush=True)

    # ── 6. Run inference ──────────────────────────────────────────────────────
    state = env.reset()
    done  = False
    bar   = 0

    print(f"[backtest] Running {symbol} inference over "
          f"{env.T - env._warmup:,} bars ...", flush=True)

    with torch.no_grad():
        while not done:
            q_vals = agent.q_net(state)
            action = int(q_vals.argmax(dim=1).item())
            state, done = env.step(action)
            bar += 1
            if bar % 50_000 == 0:
                print(f"  ... {bar:,} bars  equity=${env._equity:,.2f}",
                      flush=True)

    print(f"[backtest] Done. {bar:,} bars  {len(env.daily_log)} days logged.",
          flush=True)

    # ── 7. Metrics + print ────────────────────────────────────────────────────
    metrics = compute_metrics(env.daily_log, env.equity_curve, cfg["INITIAL_EQUITY"])

    print(f"\n{'='*55}", flush=True)
    print(f"  MT5 BACKTEST: {symbol}  ({date_from} to {date_to})", flush=True)
    print(f"{'='*55}", flush=True)
    print(f"  Total days      : {metrics.get('total_days', 0)}", flush=True)
    print(f"  PASS days       : {metrics.get('pass_days', 0)}"
          f"  ({metrics.get('pass_rate_pct', 0)}%)", flush=True)
    print(f"  OK days         : {metrics.get('ok_days', 0)}", flush=True)
    print(f"  FAIL days       : {metrics.get('fail_days', 0)}", flush=True)
    print(f"  Avg daily ret   : {metrics.get('avg_daily_return_pct', 0):+.4f}%",
          flush=True)
    print(f"  Avg daily DD    : {metrics.get('avg_daily_dd_pct', 0):.4f}%",
          flush=True)
    print(f"  Max DD (overall): {metrics.get('max_dd_overall_pct', 0):.4f}%",
          flush=True)
    print(f"  Total return    : {metrics.get('total_return_pct', 0):+.4f}%",
          flush=True)
    print(f"  Final equity    : ${metrics.get('final_equity', 0):,.2f}", flush=True)
    print(f"  Sharpe ratio    : {metrics.get('sharpe_ratio', 0):.4f}", flush=True)
    print(f"{'='*55}\n", flush=True)

    # ── 8. Save CSV ───────────────────────────────────────────────────────────
    import pandas as pd
    daily_df = pd.DataFrame(env.daily_log)

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(out_path, index=False)
        print(f"[backtest] Daily results saved -> {out_path}", flush=True)

    return {
        "metrics":      metrics,
        "daily_df":     daily_df,
        "equity_curve": env.equity_curve,
        "action_log":   env.action_log,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest a trained DQN agent using MT5 bar data")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--symbol",     default="EURUSD",
                        help="MT5 symbol name (default: EURUSD)")
    parser.add_argument("--date-from",  required=True,
                        help="Start date YYYY-MM-DD (after training period)")
    parser.add_argument("--date-to",    required=True,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--out",        default=None,
                        help="Path to save daily results CSV")
    parser.add_argument("--login",      type=int, default=None,
                        help="MT5 account number (optional)")
    parser.add_argument("--password",   default=None,
                        help="MT5 password (optional)")
    parser.add_argument("--server",     default=None,
                        help="MT5 server name (optional)")
    args = parser.parse_args()

    run_mt5_backtest(
        checkpoint_path = args.checkpoint,
        symbol          = args.symbol,
        date_from       = args.date_from,
        date_to         = args.date_to,
        out_csv         = args.out,
        mt5_login       = args.login,
        mt5_password    = args.password,
        mt5_server      = args.server,
    )
