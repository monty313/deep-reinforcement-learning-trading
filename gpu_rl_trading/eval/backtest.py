"""
gpu_rl_trading/eval/backtest.py

Run a trained DQN agent in pure inference mode (epsilon=0, no training)
over an out-of-sample date range and report full FTMO metrics.

Usage (from Python / Colab):
    from gpu_rl_trading.eval.backtest import run_backtest
    results = run_backtest(
        checkpoint_path = "/content/drive/MyDrive/gpu_rl_trading/checkpoints/eurusd_gpu_ph7_final.pt",
        csv_path        = "/content/drive/MyDrive/EURUSD_M1.csv",
        date_from       = "2024-01-01",
        date_to         = "2024-12-31",
    )

Usage (CLI):
    python -m gpu_rl_trading.eval.backtest \\
        --checkpoint /path/to/eurusd_gpu_ph7_final.pt \\
        --csv        /path/to/EURUSD_M1.csv \\
        --date-from  2024-01-01 \\
        --date-to    2024-12-31 \\
        --out        backtest_results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch

from gpu_rl_trading.config.settings import CFG
from gpu_rl_trading.env.indicators import build_feature_matrix
from gpu_rl_trading.env.environment import BatchedFTMOEnv, NUM_ACTIONS, COL_CLOSE
from gpu_rl_trading.agent.dqn import DQNAgent
from gpu_rl_trading.training.train import load_eurusd_csv


# ── single-episode sequential backtest env ────────────────────────────────────

class SequentialBacktestEnv:
    """
    Walks through the ENTIRE out-of-sample period bar by bar (no random
    episode windows). B=1, phase=7 (no mask). Designed for inference only.

    Unlike BatchedFTMOEnv (which samples random episode windows), this env
    starts at bar 0 and steps through every bar exactly once so results
    represent the agent's performance over the full evaluation period.
    """

    def __init__(self, features: np.ndarray, cfg: dict, device: torch.device):
        self.cfg            = cfg
        self.device         = device
        self.lkbk           = cfg["LOOKBACK"]
        self.initial_equity = float(cfg["INITIAL_EQUITY"])
        self.target_pct     = float(cfg["DAILY_TARGET_PCT"])
        self.max_dd_pct     = float(cfg["DAILY_MAX_DD_PCT"])
        self.tf_factors     = cfg["TF_FACTORS"]

        T, F = features.shape
        self.T = T
        self.F = F

        self._feat_1m = torch.tensor(features, dtype=torch.float32, device=device)

        # precompute resampled TF tensors (same as training env)
        self._resampled: Dict[int, torch.Tensor] = {}
        for tf in self.tf_factors:
            if tf == 1:
                self._resampled[tf] = self._feat_1m
            else:
                idx = torch.arange(tf - 1, T, tf, device=device)
                self._resampled[tf] = self._feat_1m[idx]

        from gpu_rl_trading.env.environment import _SIGN_T, _SIZE_T
        self._sign = _SIGN_T.to(device)
        self._size = _SIZE_T.to(device)

        # episode state (reset in reset())
        self._curr_step    = 0
        self._equity       = self.initial_equity
        self._realised_pnl = 0.0
        self._position     = 0.0   # +1 long, -1 short, 0 flat
        self._entry_px     = 0.0
        self._lots         = 0.0
        self._day_start_eq = self.initial_equity
        self._day_high_eq  = self.initial_equity
        self._prev_day     = -1

        # output logs
        self.daily_log:  List[dict] = []
        self.equity_curve: List[float] = []
        self.action_log:   List[int]   = []

        # warmup: skip first N bars so all indicators have valid values
        self._warmup = max(self.tf_factors) * 2 + self.lkbk + 200
        self.state_dim = self._compute_state_dim()

    def _compute_state_dim(self) -> int:
        return self.lkbk * self.F * len(self.tf_factors) + 6

    def reset(self) -> torch.Tensor:
        self._curr_step    = self._warmup
        self._equity       = self.initial_equity
        self._realised_pnl = 0.0
        self._position     = 0.0
        self._entry_px     = 0.0
        self._lots         = 0.0
        self._day_start_eq = self.initial_equity
        self._day_high_eq  = self.initial_equity
        self._prev_day     = -1
        self.daily_log     = []
        self.equity_curve  = []
        self.action_log    = []
        return self._get_state()

    def _abs_idx(self) -> int:
        return min(self._curr_step, self.T - 1)

    def _get_state(self) -> torch.Tensor:
        abs_idx = self._abs_idx()
        abs_t   = torch.tensor(abs_idx, device=self.device)
        parts   = []

        for tf in self.tf_factors:
            feat   = self._resampled[tf]
            tf_idx = (abs_t // tf).clamp(0, feat.shape[0] - 1)
            offsets = torch.arange(self.lkbk - 1, -1, -1, device=self.device)
            win_idx = (tf_idx - offsets).clamp(0, feat.shape[0] - 1)
            window  = feat[win_idx].unsqueeze(0)                     # (1, lkbk, F)
            mu  = window.mean(dim=(1, 2), keepdim=True)
            std = window.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            parts.append(((window - mu) / std).reshape(1, -1))

        close_px   = self._feat_1m[abs_idx, COL_CLOSE].item()
        unrealised = 0.0
        if self._position != 0 and self._entry_px != 0:
            unrealised = (close_px - self._entry_px) / (self._entry_px + 1e-8) * self._position

        eq_chg      = (self._equity - self.initial_equity) / self.initial_equity
        target_eq   = self._day_start_eq * (1.0 + self.target_pct)
        gap_to_tgt  = (target_eq - self._equity) / (self.initial_equity + 1e-8)
        dd_used     = max(0.0, (self._day_high_eq - self._equity) / (self._day_high_eq + 1e-8))
        dd_headroom = max(0.0, self.max_dd_pct - dd_used)
        daily_ret   = (self._equity - self._day_start_eq) / (self._day_start_eq + 1e-8)

        pos_feat = torch.tensor(
            [[self._position, unrealised, eq_chg, gap_to_tgt, dd_headroom, daily_ret]],
            dtype=torch.float32, device=self.device,
        )
        parts.append(pos_feat)
        return torch.cat(parts, dim=1)   # (1, state_dim)

    def step(self, action: int) -> tuple:
        """
        Step one bar. Returns (next_state, done).
        Reward is not used during inference — equity tracked directly.
        """
        abs_idx   = self._abs_idx()
        close_px  = self._feat_1m[abs_idx, COL_CLOSE].item()
        sign      = self._sign[action].item()

        # close on reversal
        if self._position != 0 and sign != 0 and sign != self._position:
            price_diff      = (close_px - self._entry_px) * self._position
            pnl             = price_diff * self._lots * 100_000.0
            self._realised_pnl += pnl
            self._position  = 0.0
            self._entry_px  = 0.0
            self._lots      = 0.0

        # open new position
        if self._position == 0 and sign != 0:
            size_mult   = self._size[action].item()
            target_eq   = self._day_start_eq * (1.0 + self.target_pct)
            gap_dollars = max(0.0, target_eq - self._equity)
            atr_price   = self._feat_1m[abs_idx, 5].item()   # col 5 = atr14
            atr_price   = max(atr_price, 1e-6)
            atr_dollars = atr_price * 100_000.0
            target_lots = gap_dollars / (atr_dollars + 1e-8)
            headroom_dollars = max(0.0,
                self.max_dd_pct * self._equity
                - max(0.0, self._day_high_eq - self._equity))
            conservative_lots = (headroom_dollars / (atr_dollars + 1e-8)) * 0.5
            base_lots   = target_lots if gap_dollars > 0 else conservative_lots
            max_lev     = max(0.01, self._equity * 100.0 / 100_000.0)
            lots        = min(max(base_lots * size_mult, 0.01), min(max_lev, 100.0))
            self._position = sign
            self._entry_px = close_px
            self._lots     = lots

        # unrealised PnL
        if self._position != 0:
            unreal = (close_px - self._entry_px) * self._position * self._lots * 100_000.0
        else:
            unreal = 0.0
        self._equity = self.initial_equity + self._realised_pnl + unreal
        self._equity = max(self._equity, 0.0)   # floor at 0

        # day boundary
        day_idx = abs_idx // 1440
        if self._prev_day >= 0 and day_idx != self._prev_day:
            ret_pct = (self._equity - self._day_start_eq) / (self._day_start_eq + 1e-8) * 100
            hi_eq   = self._day_high_eq
            dd_pct  = max(0.0, (hi_eq - self._equity) / (hi_eq + 1e-8) * 100)
            flag    = ("PASS" if ret_pct >= self.target_pct * 100 and dd_pct <= self.max_dd_pct * 100
                       else "OK" if ret_pct >= 0.0 and dd_pct <= self.max_dd_pct * 100
                       else "FAIL")
            self.daily_log.append({
                "day_idx":                int(self._prev_day),
                "daily_return_pct":       round(ret_pct, 4),
                "daily_max_drawdown_pct": round(dd_pct, 4),
                "ftmo_flag":              flag,
                "equity_eod":             round(self._equity, 2),
            })
            self._day_start_eq = self._equity
            self._day_high_eq  = self._equity

        self._day_high_eq = max(self._day_high_eq, self._equity)
        self._prev_day    = day_idx
        self.equity_curve.append(self._equity)
        self.action_log.append(action)

        self._curr_step += 1
        done = self._curr_step >= self.T
        return self._get_state(), done


# ── metrics summary ────────────────────────────────────────────────────────────

def compute_metrics(daily_log: List[dict], equity_curve: List[float],
                    initial_equity: float) -> dict:
    if not daily_log:
        return {}

    flags   = [r["ftmo_flag"] for r in daily_log]
    rets    = [r["daily_return_pct"] for r in daily_log]
    dds     = [r["daily_max_drawdown_pct"] for r in daily_log]
    n       = len(flags)

    passes  = flags.count("PASS")
    oks     = flags.count("OK")
    fails   = flags.count("FAIL")

    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    drawdowns = (peak - eq) / (peak + 1e-8) * 100
    max_dd_overall = float(drawdowns.max())

    total_return_pct = (eq[-1] - initial_equity) / initial_equity * 100

    # Sharpe: daily returns, annualised (252 trading days)
    daily_rets_pct = np.array(rets)
    sharpe = 0.0
    if daily_rets_pct.std() > 0:
        sharpe = float((daily_rets_pct.mean() / daily_rets_pct.std()) * np.sqrt(252))

    return {
        "total_days":          n,
        "pass_days":           passes,
        "ok_days":             oks,
        "fail_days":           fails,
        "pass_rate_pct":       round(passes / n * 100, 2),
        "avg_daily_return_pct": round(float(np.mean(rets)), 4),
        "avg_daily_dd_pct":    round(float(np.mean(dds)), 4),
        "max_dd_overall_pct":  round(max_dd_overall, 4),
        "total_return_pct":    round(total_return_pct, 4),
        "final_equity":        round(float(eq[-1]), 2),
        "sharpe_ratio":        round(sharpe, 4),
    }


# ── main backtest function ─────────────────────────────────────────────────────

def run_backtest(
    checkpoint_path: str,
    csv_path:        str,
    date_from:       str  = None,
    date_to:         str  = None,
    out_csv:         str  = None,
    cfg_overrides:   dict = None,
) -> dict:
    """
    Run the trained agent in inference mode over the specified date range.

    Args:
        checkpoint_path : path to .pt checkpoint (e.g. eurusd_gpu_ph7_final.pt)
        csv_path        : path to EURUSD 1m CSV (same format as training)
        date_from       : start date string "YYYY-MM-DD" (use data AFTER training period)
        date_to         : end date string "YYYY-MM-DD"
        out_csv         : optional path to save daily results CSV
        cfg_overrides   : optional dict to override any CFG values

    Returns:
        dict with keys:
            "metrics"    : summary dict (pass_rate, sharpe, total_return, etc.)
            "daily_df"   : pd.DataFrame of per-day results
            "equity_curve": list of per-bar equity values
            "action_log" : list of per-bar actions taken (0-6)
    """
    cfg = CFG.copy()
    if cfg_overrides:
        cfg.update(cfg_overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[backtest] device={device}", flush=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    ohlcv = load_eurusd_csv(csv_path, date_from, date_to)
    print("[backtest] Building feature matrix ...", flush=True)
    features = build_feature_matrix(
        ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4])
    print(f"[backtest] Features shape={features.shape}", flush=True)

    # ── 2. Build sequential env ───────────────────────────────────────────────
    cfg["PHASE"] = 7   # no mask during inference
    env = SequentialBacktestEnv(features, cfg, device)
    print(f"[backtest] state_dim={env.state_dim}  bars={env.T}  warmup={env._warmup}",
          flush=True)

    # ── 3. Load agent ─────────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_state_dim = ckpt.get("state_dim", env.state_dim)

    cfg["STATE_DIM"]      = env.state_dim
    cfg["BATCH_SIZE_ENV"] = 1   # sequential env uses B=1
    cfg["MEMORY_SIZE"]    = 1   # no replay buffer needed for inference
    agent = DQNAgent(env.state_dim, NUM_ACTIONS, cfg, device)

    if ckpt_state_dim != env.state_dim:
        print(f"[backtest] WARNING: checkpoint state_dim={ckpt_state_dim} "
              f"vs env state_dim={env.state_dim} — using partial load", flush=True)
        agent.q_net.load_partial(ckpt["q_net"], ckpt_state_dim)
    else:
        agent.q_net.load_state_dict(ckpt["q_net"])

    agent.q_net.eval()
    agent.epsilon = 0.0   # pure greedy — no exploration
    print(f"[backtest] Loaded checkpoint: {checkpoint_path}", flush=True)
    print(f"[backtest] epsilon=0.0 (greedy inference, no exploration)", flush=True)

    # ── 4. Run inference ──────────────────────────────────────────────────────
    state = env.reset()
    done  = False
    bar   = 0

    print(f"[backtest] Running inference over {env.T - env._warmup:,} bars ...",
          flush=True)

    with torch.no_grad():
        while not done:
            q_vals = agent.q_net(state)              # (1, NUM_ACTIONS)
            action = int(q_vals.argmax(dim=1).item())
            state, done = env.step(action)
            bar += 1
            if bar % 50_000 == 0:
                print(f"  ... {bar:,} bars processed  equity=${env._equity:,.2f}",
                      flush=True)

    print(f"[backtest] Done. {bar:,} bars  {len(env.daily_log)} days logged.",
          flush=True)

    # ── 5. Compute + print metrics ────────────────────────────────────────────
    metrics = compute_metrics(env.daily_log, env.equity_curve, cfg["INITIAL_EQUITY"])

    print(f"\n{'='*55}", flush=True)
    print(f"  BACKTEST RESULTS  ({date_from} to {date_to})", flush=True)
    print(f"{'='*55}", flush=True)
    print(f"  Total days     : {metrics['total_days']}", flush=True)
    print(f"  PASS days      : {metrics['pass_days']}  ({metrics['pass_rate_pct']}%)", flush=True)
    print(f"  OK days        : {metrics['ok_days']}", flush=True)
    print(f"  FAIL days      : {metrics['fail_days']}", flush=True)
    print(f"  Avg daily ret  : {metrics['avg_daily_return_pct']:+.4f}%", flush=True)
    print(f"  Avg daily DD   : {metrics['avg_daily_dd_pct']:.4f}%", flush=True)
    print(f"  Max DD (overall): {metrics['max_dd_overall_pct']:.4f}%", flush=True)
    print(f"  Total return   : {metrics['total_return_pct']:+.4f}%", flush=True)
    print(f"  Final equity   : ${metrics['final_equity']:,.2f}", flush=True)
    print(f"  Sharpe ratio   : {metrics['sharpe_ratio']:.4f}", flush=True)
    print(f"{'='*55}\n", flush=True)

    # ── 6. Save results ───────────────────────────────────────────────────────
    daily_df = pd.DataFrame(env.daily_log)

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(out_path, index=False)
        print(f"[backtest] Daily results saved -> {out_path}", flush=True)

    return {
        "metrics":     metrics,
        "daily_df":    daily_df,
        "equity_curve": env.equity_curve,
        "action_log":  env.action_log,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest a trained DQN agent")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--csv",        required=True,
                        help="Path to EURUSD 1m CSV file")
    parser.add_argument("--date-from",  default=None,
                        help="Start date YYYY-MM-DD (should be after training period)")
    parser.add_argument("--date-to",    default=None,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--out",        default=None,
                        help="Optional path to save daily results CSV")
    args = parser.parse_args()

    run_backtest(
        checkpoint_path = args.checkpoint,
        csv_path        = args.csv,
        date_from       = args.date_from,
        date_to         = args.date_to,
        out_csv         = args.out,
    )
