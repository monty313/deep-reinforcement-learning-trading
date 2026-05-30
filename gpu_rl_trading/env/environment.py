"""
gpu_rl_trading/env/environment.py

BatchedFTMOEnv — runs B parallel episodes on GPU tensors.
Each episode = a window of EPISODE_BARS 1m steps starting at a random offset.

Phase curriculum:
  phase 0-6 : indicator-gated (mask_type = force_in_and_gate or open_gate)
  phase 7   : free — no mask, agent decides everything
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Tuple

FLAT        = 0
BUY_SMALL   = 1
BUY_MED     = 2
BUY_LARGE   = 3
SELL_SMALL  = 4
SELL_MED    = 5
SELL_LARGE  = 6
NUM_ACTIONS = 7

# sign: +1 buy, -1 sell, 0 flat
_SIGN_T = torch.tensor([0, 1, 1, 1, -1, -1, -1], dtype=torch.float32)
# risk fraction of equity per lot
_RISK_T = torch.tensor([0.0, 0.005, 0.010, 0.020, 0.005, 0.010, 0.020], dtype=torch.float32)

# ── feature column indices in build_feature_matrix output ────────────────────
# 0=open 1=high 2=low 3=close 4=volume
# 5=atr14 6=atr45 7=rsi7 8=rsi14 9=cci14 10=cci30 11=cci100
# 12=sma20 13=sma50 14=sma200
# 15=bb20_upper 16=bb20_mid 17=bb20_lower
# 18=bb200_upper 19=bb200_mid 20=bb200_lower
# 21=high_sma4_sh8 22=low_sma4_sh8
# 23=sma2_sh0 24=sma2_sh1 25=sma2_sh2
# 26=atr14_sma1_sh8
COL_CLOSE       = 3
COL_CCI30       = 10
COL_CCI100      = 11
COL_BB20_UPPER  = 15
COL_BB20_MID    = 16
COL_BB20_LOWER  = 17
COL_BB200_MID   = 19
COL_HIGH_SMA4   = 21
COL_LOW_SMA4    = 22
COL_SMA2_SH0    = 23
COL_SMA2_SH1    = 24
COL_SMA2_SH2    = 25
COL_ATR14       = 5
COL_ATR14_SH8   = 26
COL_ATR45       = 6


class BatchedFTMOEnv:
    """
    B parallel episodes stepped in lockstep on GPU tensors.

    Curriculum phases (set via cfg["PHASE"]):
      0 — CCI Extreme Gate       force_in_and_gate  [1m, 15m]
      1 — CCI Directional Align  open_gate          [1m, 15m]
      2 — Hi/Lo SMA Trend        force_in_and_gate  [1m, 30m]
      3 — Hi/Lo SMA Counter-TF   force_in_and_gate  [1m, 15m]
      4 — BB Position            force_in_and_gate  [1m, 15m]
      5 — SMA Stack              force_in_and_gate  [1m, 60m]
      6 — ATR Expansion          force_in_and_gate  [1m, 60m]
      7 — Full FTMO, no mask     free
    """

    NUM_FEATURES = 27

    def __init__(self, features_1m: np.ndarray, cfg: dict, device: torch.device):
        self.cfg            = cfg
        self.device         = device
        self.B              = cfg["BATCH_SIZE_ENV"]
        self.lkbk           = cfg["LOOKBACK"]
        self.ep_bars        = cfg["EPISODE_BARS"]
        self.initial_equity = float(cfg["INITIAL_EQUITY"])
        self.target_pct     = float(cfg["DAILY_TARGET_PCT"])
        self.max_dd_pct     = float(cfg["DAILY_MAX_DD_PCT"])
        self.phase          = int(cfg.get("PHASE", 7))
        self.tf_factors     = cfg["TF_FACTORS"]

        T, F = features_1m.shape
        self.T = T
        self.F = F

        # Full 1m history on device
        self._feat_1m = torch.tensor(features_1m, dtype=torch.float32, device=device)

        # Precomputed per-TF tensors (last bar of each TF window)
        self._resampled: Dict[int, torch.Tensor] = {}
        for tf in self.tf_factors:
            if tf == 1:
                self._resampled[tf] = self._feat_1m
            else:
                idx = torch.arange(tf - 1, T, tf, device=device)
                self._resampled[tf] = self._feat_1m[idx]

        # Action lookup tables on device
        self._sign = _SIGN_T.to(device)
        self._risk = _RISK_T.to(device)

        # Allocate episode tensors (filled in reset())
        self._start_idx    = torch.zeros(self.B, dtype=torch.long, device=device)
        self._curr_step    = torch.zeros(self.B, dtype=torch.long, device=device)
        self._equity       = torch.full((self.B,), self.initial_equity, device=device)
        self._realised_pnl = torch.zeros(self.B, device=device)
        self._day_start_eq = torch.full((self.B,), self.initial_equity, device=device)
        self._day_high_eq  = torch.full((self.B,), self.initial_equity, device=device)
        self._position     = torch.zeros(self.B, device=device)
        self._entry_px     = torch.zeros(self.B, device=device)
        self._lots         = torch.zeros(self.B, device=device)
        self._prev_day     = torch.full((self.B,), -1, dtype=torch.long, device=device)
        self._active       = torch.ones(self.B, dtype=torch.bool, device=device)

        self.daily_metrics_log: List[dict] = []
        self.state_dim = self._compute_state_dim()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _compute_state_dim(self) -> int:
        return self.lkbk * self.F * len(self.tf_factors) + 3

    def _abs_idx(self) -> torch.Tensor:
        return (self._start_idx + self._curr_step).clamp(0, self.T - 1)

    def _row(self, abs_idx: torch.Tensor, tf: int) -> torch.Tensor:
        """Return (B, F) feature row for each batch item at current time, for given TF."""
        tf_idx = (abs_idx // tf).clamp(0, self._resampled[tf].shape[0] - 1)
        return self._resampled[tf][tf_idx]   # (B, F)

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> torch.Tensor:
        warmup    = max(self.tf_factors) * 2 + self.lkbk + 200
        max_start = max(warmup + 1, self.T - self.ep_bars - warmup)
        starts = torch.randint(warmup, max_start, (self.B,), device=self.device)

        self._start_idx    = starts
        self._curr_step    = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self._equity       = torch.full((self.B,), self.initial_equity, device=self.device)
        self._realised_pnl = torch.zeros(self.B, device=self.device)
        self._day_start_eq = self._equity.clone()
        self._day_high_eq  = self._equity.clone()
        self._position     = torch.zeros(self.B, device=self.device)
        self._entry_px     = torch.zeros(self.B, device=self.device)
        self._lots         = torch.zeros(self.B, device=self.device)
        self._prev_day     = torch.full((self.B,), -1, dtype=torch.long, device=self.device)
        self._active       = torch.ones(self.B, dtype=torch.bool, device=self.device)
        self.daily_metrics_log = []
        return self._get_state()

    # ── state ─────────────────────────────────────────────────────────────────
    def _get_state(self) -> torch.Tensor:
        abs_idx = self._abs_idx()
        parts = []
        for tf in self.tf_factors:
            feat   = self._resampled[tf]
            tf_idx = (abs_idx // tf).clamp(self.lkbk, feat.shape[0] - 1)
            offsets = torch.arange(self.lkbk - 1, -1, -1, device=self.device)
            win_idx = (tf_idx.unsqueeze(1) - offsets.unsqueeze(0)).clamp(0, feat.shape[0] - 1)
            window  = feat[win_idx]                                      # (B, lkbk, F)
            mu  = window.mean(dim=(1, 2), keepdim=True)
            std = window.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            parts.append(((window - mu) / std).reshape(self.B, -1))

        curr_close = self._feat_1m[abs_idx, COL_CLOSE]
        unrealised = torch.where(
            self._position != 0,
            (curr_close - self._entry_px) / (self._entry_px + 1e-8) * self._position,
            torch.zeros_like(curr_close),
        )
        eq_chg = (self._equity - self.initial_equity) / self.initial_equity
        parts.append(torch.stack([self._position, unrealised, eq_chg], dim=1))
        return torch.cat(parts, dim=1)

    # ── phase mask ────────────────────────────────────────────────────────────
    def _apply_phase_mask(
        self,
        actions: torch.Tensor,
        abs_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns modified actions tensor respecting the current phase mask.
        force_in_and_gate: if condition met and flat → force BUY_SMALL
                           if condition NOT met → block opens (set to FLAT for flat agents)
        open_gate:         if condition NOT met → block opens only
        free:              pass through unchanged
        """
        phase = self.phase
        if phase == 7:
            return actions

        sign = self._sign[actions]

        # get indicator rows for the two relevant TFs
        if phase in (0, 1, 3, 4):
            row1m  = self._row(abs_idx, 1)
            row2   = self._row(abs_idx, 15)
            tf2    = 15
        elif phase == 2:
            row1m  = self._row(abs_idx, 1)
            row2   = self._row(abs_idx, 30) if 30 in self._resampled else self._row(abs_idx, 15)
            tf2    = 30
        else:  # phase 5, 6
            row1m  = self._row(abs_idx, 1)
            row2   = self._row(abs_idx, 60)
            tf2    = 60

        # ── compute condition per batch item ──────────────────────────────────
        if phase == 0:
            # CCI Extreme Gate: cci30>100 AND cci100>100 on both TFs (or both <-100)
            c30_1  = row1m[:, COL_CCI30];  c100_1  = row1m[:, COL_CCI100]
            c30_2  = row2[:, COL_CCI30];   c100_2  = row2[:, COL_CCI100]
            bull   = (c30_1 > 100) & (c100_1 > 100) & (c30_2 > 100) & (c100_2 > 100)
            bear   = (c30_1 < -100) & (c100_1 < -100) & (c30_2 < -100) & (c100_2 < -100)
            cond   = bull | bear
            mask_type = "force"

        elif phase == 1:
            # CCI Directional Alignment: cci30 and cci100 both above/below their
            # shifted SMA on both TFs in the same direction
            # Use cci30 vs bb20_mid as proxy for direction alignment
            c30_1 = row1m[:, COL_CCI30]; c100_1 = row1m[:, COL_CCI100]
            c30_2 = row2[:, COL_CCI30];  c100_2 = row2[:, COL_CCI100]
            # positive when both positive, negative when both negative
            d1    = torch.sign(c30_1) * (torch.sign(c30_1) == torch.sign(c100_1)).float()
            d2    = torch.sign(c30_2) * (torch.sign(c30_2) == torch.sign(c100_2)).float()
            cond  = (d1 != 0) & (d2 != 0) & (d1 == d2)
            mask_type = "gate"

        elif phase == 2:
            # Hi/Lo SMA Trend: close above BOTH high_sma4_sh8 and low_sma4_sh8 on both TFs
            px1   = row1m[:, COL_CLOSE]; hi1 = row1m[:, COL_HIGH_SMA4]; lo1 = row1m[:, COL_LOW_SMA4]
            px2   = row2[:, COL_CLOSE];  hi2 = row2[:, COL_HIGH_SMA4];  lo2 = row2[:, COL_LOW_SMA4]
            bull  = (px1 > hi1) & (px1 > lo1) & (px2 > hi2) & (px2 > lo2)
            bear  = (px1 < hi1) & (px1 < lo1) & (px2 < hi2) & (px2 < lo2)
            cond  = bull | bear
            mask_type = "force"

        elif phase == 3:
            # Hi/Lo SMA Counter-TF: 1m and 15m on OPPOSITE sides of band
            px1   = row1m[:, COL_CLOSE]; hi1 = row1m[:, COL_HIGH_SMA4]; lo1 = row1m[:, COL_LOW_SMA4]
            px2   = row2[:, COL_CLOSE];  hi2 = row2[:, COL_HIGH_SMA4];  lo2 = row2[:, COL_LOW_SMA4]
            above1 = (px1 > hi1) & (px1 > lo1)
            below1 = (px1 < hi1) & (px1 < lo1)
            above2 = (px2 > hi2) & (px2 > lo2)
            below2 = (px2 < hi2) & (px2 < lo2)
            cond   = (above1 & below2) | (below1 & above2)
            mask_type = "force"

        elif phase == 4:
            # BB Position: 1m close > bb200_mid AND > bb20_upper (bull)
            #              15m close > bb200_mid AND > bb20_mid (bull)
            px1   = row1m[:, COL_CLOSE]
            b200m1 = row1m[:, COL_BB200_MID]; b20u1 = row1m[:, COL_BB20_UPPER]; b20l1 = row1m[:, COL_BB20_LOWER]
            px2    = row2[:, COL_CLOSE]
            b200m2 = row2[:, COL_BB200_MID];  b20m2 = row2[:, COL_BB20_MID]
            bull   = (px1 > b200m1) & (px1 > b20u1) & (px2 > b200m2) & (px2 > b20m2)
            bear   = (px1 < b200m1) & (px1 < b20l1) & (px2 < b200m2) & (px2 < b20m2)
            cond   = bull | bear
            mask_type = "force"

        elif phase == 5:
            # SMA Stack: sma2_sh0 > sh1 > sh2 on both TFs
            s0_1 = row1m[:, COL_SMA2_SH0]; s1_1 = row1m[:, COL_SMA2_SH1]; s2_1 = row1m[:, COL_SMA2_SH2]
            s0_2 = row2[:, COL_SMA2_SH0];  s1_2 = row2[:, COL_SMA2_SH1];  s2_2 = row2[:, COL_SMA2_SH2]
            bull  = (s0_1 > s1_1) & (s1_1 > s2_1) & (s0_2 > s1_2) & (s1_2 > s2_2)
            bear  = (s0_1 < s1_1) & (s1_1 < s2_1) & (s0_2 < s1_2) & (s1_2 < s2_2)
            cond  = bull | bear
            mask_type = "force"

        elif phase == 6:
            # ATR Expansion: atr14 > atr14_sma1_sh8 AND atr45 > atr45_sma1_sh8 on both TFs
            # atr45_sma1_sh8 not precomputed separately — use atr45 vs atr14_sh8 as proxy
            a14_1 = row1m[:, COL_ATR14]; ref14_1 = row1m[:, COL_ATR14_SH8]
            a45_1 = row1m[:, COL_ATR45]; a14_2 = row2[:, COL_ATR14]; ref14_2 = row2[:, COL_ATR14_SH8]
            a45_2 = row2[:, COL_ATR45]
            cond  = (a14_1 > ref14_1) & (a45_1 > a14_1) & (a14_2 > ref14_2) & (a45_2 > a14_2)
            mask_type = "force"

        else:
            return actions

        flat     = torch.tensor(FLAT, device=self.device)
        buy_s    = torch.tensor(BUY_SMALL, device=self.device)
        new_actions = actions.clone()

        if mask_type == "force":
            # condition TRUE and flat → force into a trade
            force_open = cond & (self._position == 0)
            new_actions = torch.where(force_open, buy_s.expand_as(new_actions), new_actions)
            # condition FALSE → block opens (keep flat if already flat)
            block_open  = ~cond & (self._position == 0) & (sign != 0)
            new_actions = torch.where(block_open, flat.expand_as(new_actions), new_actions)
        else:  # open_gate
            # condition FALSE → block opens
            block_open  = ~cond & (self._position == 0) & (sign != 0)
            new_actions = torch.where(block_open, flat.expand_as(new_actions), new_actions)

        return new_actions

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        actions : (B,) long tensor
        Returns (next_state, rewards, dones) all shape (B,).
        Only active (not-done) episodes are updated; done episodes return zeros.
        """
        abs_idx    = self._abs_idx()
        curr_close = self._feat_1m[abs_idx, COL_CLOSE]
        rewards    = torch.zeros(self.B, device=self.device)

        # apply phase mask before executing actions
        actions = self._apply_phase_mask(actions, abs_idx)
        sign     = self._sign[actions]
        lots_frac = self._risk[actions]

        # ── close on direction reversal ───────────────────────────────────────
        close_mask = self._active & (self._position != 0) & (sign != 0) & (sign != self._position)
        if close_mask.any():
            # pnl in account currency: price_diff * lots * 100_000 (1 std lot = 100k units)
            price_diff = (curr_close - self._entry_px) * self._position
            pnl_abs    = price_diff * self._lots * 100_000.0
            # reward = pnl as fraction of initial equity (so agent sees meaningful signal)
            reward_sig = pnl_abs / self.initial_equity
            self._realised_pnl = torch.where(close_mask, self._realised_pnl + pnl_abs, self._realised_pnl)
            rewards            = torch.where(close_mask, rewards + reward_sig, rewards)
            self._position     = torch.where(close_mask, torch.zeros_like(self._position), self._position)
            self._entry_px     = torch.where(close_mask, torch.zeros_like(self._entry_px), self._entry_px)
            self._lots         = torch.where(close_mask, torch.zeros_like(self._lots), self._lots)

        # ── open new position ─────────────────────────────────────────────────
        open_mask = self._active & (self._position == 0) & (sign != 0)
        if open_mask.any():
            # lots = risk_frac * equity / (pip_value * 100_000)
            # simplified: target ~risk_frac% of equity per 10-pip move
            # lots = (equity * risk_frac) / (0.001 * 100_000)  →  equity * risk_frac / 100
            lots = (self._equity * lots_frac / 100.0).clamp(min=0.001, max=100.0)
            self._position = torch.where(open_mask, sign,       self._position)
            self._entry_px = torch.where(open_mask, curr_close, self._entry_px)
            self._lots     = torch.where(open_mask, lots,       self._lots)

        # ── equity = realised + unrealised ────────────────────────────────────
        unreal_pnl = torch.where(
            self._position != 0,
            (curr_close - self._entry_px) * self._position * self._lots * 100_000.0,
            torch.zeros_like(self._position),
        )
        self._equity = self.initial_equity + self._realised_pnl + unreal_pnl

        # ── day boundary ──────────────────────────────────────────────────────
        day_idx = abs_idx // 1440
        new_day = (day_idx != self._prev_day) & (self._prev_day >= 0) & self._active

        if new_day.any():
            # log metrics (silent — summary printed per episode in training loop)
            for b in range(self.B):
                if new_day[b].item():
                    ret_pct = float((self._equity[b] - self._day_start_eq[b])
                                    / (self._day_start_eq[b] + 1e-8) * 100)
                    hi_eq   = float(self._day_high_eq[b])
                    eq_now  = float(self._equity[b])
                    dd_pct  = max(0.0, (hi_eq - eq_now) / (hi_eq + 1e-8) * 100)
                    flag    = ("PASS" if ret_pct >= 2.5 and dd_pct <= 1.0 else
                               "OK"   if ret_pct >= 0.0 and dd_pct <= 1.0 else "FAIL")
                    self.daily_metrics_log.append({
                        "batch":                  b,
                        "day_idx":                int(self._prev_day[b]),
                        "daily_return_pct":       round(ret_pct, 4),
                        "daily_max_drawdown_pct": round(dd_pct, 4),
                        "ftmo_flag":              flag,
                    })
                    print(f"  [b{b}] day {int(self._prev_day[b])} | ret={ret_pct:+.2f}% | dd={dd_pct:.2f}% | {flag}", flush=True)

            # reset day trackers
            self._day_start_eq = torch.where(new_day, self._equity.detach(), self._day_start_eq)
            self._day_high_eq  = torch.where(new_day, self._equity.detach(), self._day_high_eq)

        # update intraday high
        self._day_high_eq = torch.maximum(self._day_high_eq, self._equity.detach())
        self._prev_day    = torch.where(self._active, day_idx, self._prev_day)

        # FTMO drawdown breach penalty
        dd_now    = (self._day_high_eq - self._equity) / (self._day_high_eq + 1e-8)
        fail_mask = self._active & (dd_now > self.max_dd_pct)
        rewards   = torch.where(fail_mask, rewards - 0.02, rewards)   # -2% of equity scale

        # daily PASS/OK bonus
        if new_day.any():
            for b in range(self.B):
                if new_day[b].item() and self._active[b].item():
                    log = self.daily_metrics_log
                    if log and log[-1]["batch"] == b:
                        if log[-1]["ftmo_flag"] == "PASS":
                            rewards[b] = rewards[b] + 0.025   # +2.5% of equity scale
                        elif log[-1]["ftmo_flag"] == "OK":
                            rewards[b] = rewards[b] + 0.005

        # ── zero out inactive episodes ────────────────────────────────────────
        rewards = torch.where(self._active, rewards, torch.zeros_like(rewards))

        # ── advance step ──────────────────────────────────────────────────────
        self._curr_step = torch.where(self._active,
                                      self._curr_step + 1,
                                      self._curr_step)
        dones = (self._curr_step >= self.ep_bars)
        self._active = self._active & ~dones

        next_state = self._get_state()
        return next_state, rewards, dones

    # ── summary helper ────────────────────────────────────────────────────────
    def episode_summary(self) -> str:
        """Return a one-line FTMO summary for the just-completed episode."""
        if not self.daily_metrics_log:
            return "no days logged"
        flags = [r["ftmo_flag"] for r in self.daily_metrics_log]
        total = len(flags)
        passes = flags.count("PASS")
        oks    = flags.count("OK")
        fails  = flags.count("FAIL")
        avg_ret = sum(r["daily_return_pct"] for r in self.daily_metrics_log) / total
        return (f"days={total}  PASS={passes}  OK={oks}  FAIL={fails}  "
                f"avg_ret={avg_ret:+.3f}%")
