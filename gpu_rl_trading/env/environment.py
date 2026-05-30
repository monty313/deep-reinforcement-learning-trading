"""
gpu_rl_trading/env/environment.py

BatchedFTMOEnv — runs B parallel episodes on GPU tensors.
Each episode = a window of EPISODE_BARS 1m steps starting at a random offset.
State = concatenated lookback windows across TF resamplings.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Tuple

from gpu_rl_trading.env.indicators import build_feature_matrix

FLAT        = 0
BUY_SMALL   = 1
BUY_MED     = 2
BUY_LARGE   = 3
SELL_SMALL  = 4
SELL_MED    = 5
SELL_LARGE  = 6
NUM_ACTIONS = 7

_SIGN = torch.tensor([0, 1, 1, 1, -1, -1, -1], dtype=torch.float32)
_RISK = torch.tensor([0.0, 0.005, 0.010, 0.020, 0.005, 0.010, 0.020], dtype=torch.float32)


class BatchedFTMOEnv:
    """
    B parallel episodes stepped in lockstep on a single GPU/CPU device.

    Args:
        features_1m : (T, F) float32 numpy array (full history, all indicators)
        cfg         : config dict from settings.CFG
        device      : torch device
    """

    NUM_FEATURES = 27   # must match build_feature_matrix output width

    def __init__(self, features_1m: np.ndarray, cfg: dict, device: torch.device):
        self.cfg     = cfg
        self.device  = device
        self.B       = cfg["BATCH_SIZE_ENV"]
        self.lkbk    = cfg["LOOKBACK"]
        self.ep_bars = cfg["EPISODE_BARS"]
        self.initial_equity = cfg["INITIAL_EQUITY"]
        self.target_pct     = cfg["DAILY_TARGET_PCT"]
        self.max_dd_pct     = cfg["DAILY_MAX_DD_PCT"]

        # TF resample factors (from M1)
        self.tf_factors = cfg["TF_FACTORS"]   # e.g. [1, 15, 60, 1440]

        # Store full history as GPU tensor
        T, F = features_1m.shape
        self.T = T
        self.F = F
        self._feat_1m = torch.tensor(features_1m, dtype=torch.float32, device=device)

        # Precompute resampled tensors (simple last-bar-in-window resampling)
        self._resampled: Dict[int, torch.Tensor] = {}
        for tf in self.tf_factors:
            if tf == 1:
                self._resampled[tf] = self._feat_1m
            else:
                # keep every tf-th bar (last bar of each window)
                idx = torch.arange(tf - 1, T, tf, device=device)
                self._resampled[tf] = self._feat_1m[idx]

        # Episode state tensors  (B,)
        self._start_idx  = torch.zeros(self.B, dtype=torch.long, device=device)
        self._curr_step  = torch.zeros(self.B, dtype=torch.long, device=device)
        self._equity     = torch.full((self.B,), self.initial_equity, device=device)
        self._day_start_eq = torch.full((self.B,), self.initial_equity, device=device)
        self._day_high_eq  = torch.full((self.B,), self.initial_equity, device=device)
        self._position   = torch.zeros(self.B, dtype=torch.float32, device=device)  # +1/-1/0
        self._entry_px   = torch.zeros(self.B, dtype=torch.float32, device=device)
        self._lots       = torch.zeros(self.B, dtype=torch.float32, device=device)
        self._prev_day   = torch.full((self.B,), -1, dtype=torch.long, device=device)

        # FTMO daily metrics log (Python lists — filled once per day boundary)
        self.daily_metrics_log = []
        self.episode_reward_log = []

        # action encoding helpers on device
        self._sign = _SIGN.to(device)
        self._risk = _RISK.to(device)

        # state dimension
        self.state_dim = self._compute_state_dim()

    def _compute_state_dim(self) -> int:
        """Compute state vector length from lookback × features × TFs + position features."""
        tf_feats = self.lkbk * self.F * len(self.tf_factors)
        pos_feats = 3   # position side, unrealised pct, equity_chg
        return tf_feats + pos_feats

    # ── episode reset ─────────────────────────────────────────────────────────
    def reset(self) -> torch.Tensor:
        """
        Start B new episodes at random offsets in the history.
        Returns initial state tensor (B, state_dim).
        """
        warmup = max(self.tf_factors) + self.lkbk + 200
        max_start = self.T - self.ep_bars - warmup
        starts = torch.randint(warmup, max(warmup + 1, max_start), (self.B,), device=self.device)
        self._start_idx   = starts
        self._curr_step   = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self._equity      = torch.full((self.B,), self.initial_equity, device=self.device)
        self._day_start_eq = self._equity.clone()
        self._day_high_eq  = self._equity.clone()
        self._position    = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self._entry_px    = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self._lots        = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self._prev_day    = torch.full((self.B,), -1, dtype=torch.long, device=self.device)
        self.daily_metrics_log  = []
        self.episode_reward_log = []
        return self._get_state()

    # ── state assembly ────────────────────────────────────────────────────────
    def _get_state(self) -> torch.Tensor:
        """Build (B, state_dim) state tensor from current position."""
        abs_idx = self._start_idx + self._curr_step   # (B,)
        parts = []
        for tf in self.tf_factors:
            feat = self._resampled[tf]   # (T', F)
            # map abs_idx in 1m space to tf index
            tf_idx = (abs_idx // tf).clamp(self.lkbk, feat.shape[0] - 1)
            # gather lookback window: (B, lkbk, F)
            window_idx = tf_idx.unsqueeze(1) - torch.arange(self.lkbk - 1, -1, -1, device=self.device).unsqueeze(0)
            window_idx = window_idx.clamp(0, feat.shape[0] - 1)
            window = feat[window_idx]   # (B, lkbk, F)
            # z-score per sample
            mu  = window.mean(dim=(1, 2), keepdim=True)
            std = window.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            window = (window - mu) / std
            parts.append(window.reshape(self.B, -1))

        # position features
        curr_close = self._feat_1m[abs_idx.clamp(0, self.T - 1), 3]   # close price col=3
        unrealised = torch.where(
            self._position != 0,
            (curr_close - self._entry_px) / (self._entry_px + 1e-8) * self._position,
            torch.zeros_like(curr_close),
        )
        eq_chg = (self._equity - self.initial_equity) / self.initial_equity
        pos_feat = torch.stack([self._position, unrealised, eq_chg], dim=1)
        parts.append(pos_feat)

        return torch.cat(parts, dim=1)   # (B, state_dim)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        actions: (B,) long tensor of action indices
        Returns (next_state, rewards, dones)  all shape (B,).
        """
        abs_idx   = (self._start_idx + self._curr_step).clamp(0, self.T - 1)
        curr_close = self._feat_1m[abs_idx, 3]
        sign       = self._sign[actions]   # (B,)
        lots_frac  = self._risk[actions]   # (B,)
        lots       = (self._equity * lots_frac / 100_000).clamp(min=0.001)

        rewards = torch.zeros(self.B, device=self.device)

        # ── close on reversal ────────────────────────────────────────────────
        close_mask = (self._position != 0) & (sign != 0) & (sign != self._position)
        pnl_pct    = torch.where(
            close_mask,
            (curr_close - self._entry_px) / (self._entry_px + 1e-8) * self._position,
            torch.zeros_like(curr_close),
        )
        pnl_abs = pnl_pct * self._lots * 1000
        self._equity = torch.where(close_mask, self._equity + pnl_abs, self._equity)
        rewards      = torch.where(close_mask, rewards + pnl_pct, rewards)
        self._position = torch.where(close_mask, torch.zeros_like(self._position), self._position)
        self._entry_px = torch.where(close_mask, torch.zeros_like(self._entry_px), self._entry_px)
        self._lots     = torch.where(close_mask, torch.zeros_like(self._lots), self._lots)

        # ── open new position ─────────────────────────────────────────────────
        open_mask = (self._position == 0) & (sign != 0)
        self._position = torch.where(open_mask, sign, self._position)
        self._entry_px = torch.where(open_mask, curr_close, self._entry_px)
        self._lots     = torch.where(open_mask, lots, self._lots)

        # unrealised equity update
        unreal_pnl = ((curr_close - self._entry_px) / (self._entry_px + 1e-8)
                      * self._position * self._lots * 1000)
        self._equity = self.initial_equity + unreal_pnl  # simplified

        # ── day boundary & FTMO ───────────────────────────────────────────────
        # day index = bar_idx // 1440
        day_idx = abs_idx // 1440
        new_day = (day_idx != self._prev_day) & (self._prev_day >= 0)
        if new_day.any():
            for b in range(self.B):
                if new_day[b].item():
                    ret_pct = float((self._equity[b] - self._day_start_eq[b]) / self._day_start_eq[b] * 100)
                    hi_eq   = float(self._day_high_eq[b])
                    eq_now  = float(self._equity[b])
                    dd_pct  = max(0.0, (hi_eq - eq_now) / (hi_eq + 1e-8) * 100)
                    flag    = ("PASS" if ret_pct >= 2.5 and dd_pct <= 1.0 else
                               "OK"   if ret_pct >= 0.0 and dd_pct <= 1.0 else "FAIL")
                    self.daily_metrics_log.append({
                        "batch": b,
                        "day_idx": int(self._prev_day[b]),
                        "daily_return_pct": round(ret_pct, 4),
                        "daily_max_drawdown_pct": round(dd_pct, 4),
                        "ftmo_flag": flag,
                    })
                    print(f"  [b{b}] day {int(self._prev_day[b])} | ret={ret_pct:+.2f}% | dd={dd_pct:.2f}% | {flag}", flush=True)

        # reset day trackers on new day
        self._day_start_eq = torch.where(new_day, self._equity, self._day_start_eq)
        self._day_high_eq  = torch.where(new_day, self._equity, self._day_high_eq)
        self._day_high_eq  = torch.maximum(self._day_high_eq, self._equity)
        self._prev_day     = day_idx

        # FTMO drawdown penalty
        dd_now = (self._day_high_eq - self._equity) / (self._day_high_eq + 1e-8)
        fail_mask = dd_now > self.max_dd_pct
        rewards = torch.where(fail_mask, rewards - 2.0, rewards)

        # advance
        self._curr_step += 1
        dones = (self._curr_step >= self.ep_bars)

        next_state = self._get_state()
        return next_state, rewards, dones
