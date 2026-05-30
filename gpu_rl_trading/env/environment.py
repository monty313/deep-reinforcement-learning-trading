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

# Size multipliers: fraction of the "target-gap lot" to trade.
# small = 25% of what's needed to close the gap
# med   = 50%
# large = 100%
# Agent learns which size is appropriate given current FTMO state.
_SIZE_T = torch.tensor([0.0, 0.25, 0.50, 1.00, 0.25, 0.50, 1.00], dtype=torch.float32)

# Hard floor/ceiling on lots regardless of dynamic sizing (FTMO 1:100 leverage)
_MIN_LOTS = 0.01
_MAX_LOTS = 100.0   # 100 lots = full 1:100 leverage on $100k account

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
        self._size = _SIZE_T.to(device)

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

        # ── Potential-based reward shaping (Φ) ───────────────────────────────
        # Φ = (pass_rate × avg_return) / (1 + λ × avg_drawdown)
        # Generalizes to any target/risk: pass_rate and avg_return are expressed
        # as multiples of the configured target, so Φ=1.0 means "exactly on target".
        # This makes the shaping signal identical in scale regardless of whether
        # the daily target is 2.5% or 5% or any other value.
        self._shape_alpha   = float(cfg.get("SHAPE_ALPHA",   0.01))  # gain
        self._shape_clip    = float(cfg.get("SHAPE_CLIP",    0.03))  # max magnitude
        self._shape_lambda  = float(cfg.get("SHAPE_LAMBDA",  5.0))   # dd penalty weight
        self._shape_warmup  = int(  cfg.get("SHAPE_WARMUP",  50))    # episodes before shaping on

        # per-batch Φ trackers
        # _phi_prev / _days_seen / _ep_* reset each episode (per-episode accumulators)
        # _phi_history intentionally persists across episodes — it is the rolling
        # window used to compute σ_Φ for normalization.  Resetting it would destroy
        # the baseline needed to measure progress. It self-manages at max 20 entries.
        self._phi_prev      = [0.0] * self.B
        self._phi_history   = [[] for _ in range(self.B)]  # cross-episode rolling window
        self._days_seen     = [0]  * self.B
        self._ep_pass_count = [0]  * self.B
        self._ep_ret_sum    = [0.0]* self.B
        self._ep_dd_sum     = [0.0]* self.B
        self._episode_count = 0    # set by trainer via env.start_episode()

        self.daily_metrics_log: List[dict] = []
        self.state_dim = self._compute_state_dim()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _compute_state_dim(self) -> int:
        # indicator lookback window + 6 FTMO/position features
        return self.lkbk * self.F * len(self.tf_factors) + 6

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
        self._active        = torch.ones(self.B, dtype=torch.bool, device=self.device)
        # initialise _phi_prev to the current rolling average so the first day
        # of each episode doesn't get an artificial positive bonus from comparing
        # against 0. If no history yet, 0.0 is the correct neutral baseline.
        self._phi_prev      = [
            float(np.mean(self._phi_history[b][-20:])) if self._phi_history[b] else 0.0
            for b in range(self.B)
        ]
        self._days_seen     = [0]   * self.B
        self._ep_pass_count = [0]   * self.B
        self._ep_ret_sum    = [0.0] * self.B
        self._ep_dd_sum     = [0.0] * self.B
        self.daily_metrics_log = []
        return self._get_state()

    def start_episode(self, global_episode: int):
        """Call from trainer at start of each episode to track warm-up."""
        self._episode_count = global_episode

    def reset_phi_history(self):
        """
        Clear the cross-episode Φ rolling window.
        Call this when transitioning to a new curriculum phase so the
        normalization baseline isn't biased by the previous phase's Φ distribution.
        """
        self._phi_history = [[] for _ in range(self.B)]
        self._phi_prev    = [0.0] * self.B

    # ── state ─────────────────────────────────────────────────────────────────
    def _get_state(self) -> torch.Tensor:
        abs_idx = self._abs_idx()
        parts = []
        for tf in self.tf_factors:
            feat   = self._resampled[tf]
            tf_idx = (abs_idx // tf).clamp(0, feat.shape[0] - 1)
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
        eq_chg      = (self._equity - self.initial_equity) / self.initial_equity
        # FTMO state features: gap to target, remaining dd headroom, daily return so far
        target_eq   = self._day_start_eq * (1.0 + self.target_pct)
        gap_to_tgt  = (target_eq - self._equity) / (self.initial_equity + 1e-8)   # normalised
        dd_used     = (self._day_high_eq - self._equity) / (self._day_high_eq + 1e-8)
        dd_headroom = (self.max_dd_pct - dd_used).clamp(min=0.0)
        daily_ret   = (self._equity - self._day_start_eq) / (self._day_start_eq + 1e-8)
        pos_feat    = torch.stack(
            [self._position, unrealised, eq_chg, gap_to_tgt, dd_headroom, daily_ret],
            dim=1,
        )
        parts.append(pos_feat)
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

    # ── potential function Φ ─────────────────────────────────────────────────
    def _compute_phi(self, b: int) -> float:
        """
        Φ(b) = (pass_rate × avg_return_normalised) / (1 + λ × avg_dd_normalised)

        Normalisation makes Φ target/risk agnostic:
          - avg_return is expressed as a multiple of daily_target_pct
            so Φ = 1.0 means "hitting the target exactly every day"
          - avg_dd is expressed as a multiple of max_dd_pct
            so the λ penalty is proportional to how far over the limit we are

        This means the same α and clip values work whether the target is 2.5% or 5%,
        supporting parameterizable targets without retuning the shaping hyperparams.
        """
        n = self._days_seen[b]
        if n == 0:
            return 0.0
        pass_rate  = self._ep_pass_count[b] / n
        avg_ret    = self._ep_ret_sum[b]    / n
        avg_dd     = self._ep_dd_sum[b]     / n

        # normalise to configured target and risk — makes Φ scale-invariant
        ret_norm = avg_ret / (self.target_pct * 100.0 + 1e-8)   # 1.0 = on target
        dd_norm  = avg_dd  / (self.max_dd_pct  * 100.0 + 1e-8)  # 1.0 = at limit

        phi = (pass_rate * max(ret_norm, 0.0)) / (1.0 + self._shape_lambda * dd_norm)
        return float(phi)

    def _shape_reward(self, b: int, phi_now: float) -> float:
        """
        Compute shaping term: clip(α × (Φ_now - Φ_prev) / σ_Φ, -clip, +clip).
        Returns 0.0 during warm-up.
        """
        if self._episode_count < self._shape_warmup:
            self._phi_prev[b] = phi_now
            return 0.0

        delta = phi_now - self._phi_prev[b]
        history = self._phi_history[b]
        history.append(phi_now)
        if len(history) > 20:
            history.pop(0)

        sigma = float(np.std(history)) if len(history) > 1 else 1.0
        sigma = max(sigma, 1e-4)

        shaping = self._shape_alpha * delta / sigma
        shaping = float(np.clip(shaping, -self._shape_clip, self._shape_clip))
        self._phi_prev[b] = phi_now
        return shaping

    # ── dynamic lot sizing ────────────────────────────────────────────────────
    def _dynamic_lots(self, actions: torch.Tensor, curr_close: torch.Tensor) -> torch.Tensor:
        """
        Compute lot size dynamically based on current FTMO state.

        Core idea: how many lots do I need to close the gap to the daily target
        in a single average move (ATR)?  Then scale by the size multiplier the
        agent chose (small=25%, med=50%, large=100%).

        Formula:
          gap_$      = max(target_equity - current_equity, 0)
          atr_$      = atr_pips * pip_value_per_lot   (≈ ATR in price * 100_000)
          target_lots = gap_$ / atr_$          → lots to close gap in one ATR move
          lots        = target_lots * size_mult  → agent's chosen fraction

        If already past target (gap=0), size off remaining drawdown headroom instead:
          headroom_$ = max_dd_pct * current_equity - current_dd_$
          lots       = (headroom_$ / atr_$) * size_mult * 0.5  (half headroom)

        Clamped between _MIN_LOTS and _MAX_LOTS (1:100 leverage ceiling).
        """
        size_mult   = self._size[actions]                          # (B,) 0/0.25/0.5/1.0

        # daily target gap in dollars
        target_eq   = self._day_start_eq * (1.0 + self.target_pct)
        gap_dollars = (target_eq - self._equity).clamp(min=0.0)   # (B,)

        # ATR in price units from the 1m feature (col 5 = atr14)
        abs_idx     = self._abs_idx()
        atr_price   = self._feat_1m[abs_idx, COL_ATR14].clamp(min=1e-6)  # (B,)

        # pip value per lot: for EURUSD 1 lot = 100,000 units,
        # so $-value of 1 atr move = atr_price * 100_000
        atr_dollars = atr_price * 100_000.0                        # (B,)

        # lots needed to close the gap in one ATR move
        target_lots = gap_dollars / (atr_dollars + 1e-8)           # (B,)

        # when gap is already closed, size conservatively off drawdown headroom
        headroom_dollars = (
            self.max_dd_pct * self._equity
            - (self._day_high_eq - self._equity).clamp(min=0.0)
        ).clamp(min=0.0)
        conservative_lots = (headroom_dollars / (atr_dollars + 1e-8)) * 0.5

        base_lots = torch.where(gap_dollars > 0, target_lots, conservative_lots)

        # apply agent's size choice and clamp to leverage limits
        max_leverage_lots = (self._equity * 100.0 / 100_000.0).clamp(min=_MIN_LOTS)
        lots = (base_lots * size_mult).clamp(min=_MIN_LOTS)
        lots = torch.minimum(lots, max_leverage_lots)
        lots = torch.minimum(lots, torch.tensor(_MAX_LOTS, device=self.device))
        return lots

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        actions : (B,) long tensor
        Returns (next_state, rewards, dones, executed_actions) all shape (B,).
        executed_actions are the phase-masked actions actually applied — store
        these in replay (not the raw selected actions) so Q-values are correct.
        Only active (not-done) episodes are updated; done episodes return zeros.
        """
        abs_idx    = self._abs_idx()
        curr_close = self._feat_1m[abs_idx, COL_CLOSE]
        rewards    = torch.zeros(self.B, device=self.device)

        # apply phase mask before executing actions
        actions = self._apply_phase_mask(actions, abs_idx)
        sign    = self._sign[actions]

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
            lots = self._dynamic_lots(actions, curr_close)
            self._position = torch.where(open_mask, sign,       self._position)
            self._entry_px = torch.where(open_mask, curr_close, self._entry_px)
            self._lots     = torch.where(open_mask, lots,       self._lots)

        # ── equity = realised + unrealised ────────────────────────────────────
        unreal_pnl = torch.where(
            self._position != 0,
            (curr_close - self._entry_px) * self._position * self._lots * 100_000.0,
            torch.zeros_like(self._position),
        )
        # Guard NaN: if feature data has NaN, prices can become NaN and
        # propagate silently. Replace NaN with 0 so equity stays finite.
        unreal_pnl = torch.nan_to_num(unreal_pnl, nan=0.0, posinf=0.0, neginf=0.0)
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
                    target_threshold = self.target_pct * 100.0
                    dd_threshold     = self.max_dd_pct  * 100.0
                    flag    = ("PASS" if ret_pct >= target_threshold and dd_pct <= dd_threshold else
                               "OK"   if ret_pct >= 0.0              and dd_pct <= dd_threshold else "FAIL")
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

        # ── FTMO intraday drawdown breach penalty ────────────────────────────
        dd_now    = (self._day_high_eq - self._equity) / (self._day_high_eq + 1e-8)
        fail_mask = self._active & (dd_now > self.max_dd_pct)
        rewards   = torch.where(fail_mask, rewards - 0.02, rewards)

        # ── day-boundary: base FTMO rewards + Φ-based progress shaping ───────
        if new_day.any():
            batch_latest = {}
            for row in reversed(self.daily_metrics_log):
                b = row["batch"]
                if b not in batch_latest:
                    batch_latest[b] = row

            for b in range(self.B):
                if not (new_day[b].item() and self._active[b].item()):
                    continue
                row = batch_latest.get(b)
                if row is None:
                    continue

                flag    = row["ftmo_flag"]
                ret_pct = row["daily_return_pct"]
                dd_pct  = row["daily_max_drawdown_pct"]

                # ── Primary FTMO reward (dominant signal, never touched) ──────
                if flag == "PASS":
                    rewards[b] = rewards[b] + 0.025
                elif flag == "OK":
                    rewards[b] = rewards[b] + 0.005
                else:
                    rewards[b] = rewards[b] - 0.010

                # ── Update episode accumulators for Φ ─────────────────────────
                self._days_seen[b]     += 1
                self._ep_ret_sum[b]    += ret_pct
                self._ep_dd_sum[b]     += dd_pct
                if flag == "PASS":
                    self._ep_pass_count[b] += 1

                # ── Potential-based progress shaping ──────────────────────────
                # Φ captures all progressive goals in one normalised score:
                #   pass_rate, avg_return, avg_dd — all relative to configured
                #   targets so the signal scales correctly with any target/risk.
                phi_now = self._compute_phi(b)
                shaping = self._shape_reward(b, phi_now)
                if shaping != 0.0:
                    rewards[b] = rewards[b] + shaping

        # ── zero out inactive episodes ────────────────────────────────────────
        rewards = torch.where(self._active, rewards, torch.zeros_like(rewards))

        # ── advance step ──────────────────────────────────────────────────────
        self._curr_step = torch.where(self._active,
                                      self._curr_step + 1,
                                      self._curr_step)
        dones = (self._curr_step >= self.ep_bars)
        self._active = self._active & ~dones

        next_state = self._get_state()
        return next_state, rewards, dones, actions

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
