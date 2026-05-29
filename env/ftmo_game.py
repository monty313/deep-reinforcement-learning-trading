"""
env/ftmo_game.py
FTMO-aware multi-asset Game environment extending the Quantra template.
Supports 4-phase curriculum masking and full FTMO daily profit/drawdown rules.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from env.indicators import (
    build_feature_df,
    can_open_trade_phase1,
    must_be_in_trade_phase2,
    must_be_in_trade_phase3,
)
from env.ftmo_fees import calculate_pnl_after_fees
try:
    from env.news_detector import news_reward_penalty as _news_penalty
except Exception:
    def _news_penalty(*a, **kw): return 0.0

# ── Action constants ──────────────────────────────────────────────────────────
FLAT       = 0
BUY_SMALL  = 1
BUY_MED    = 2
BUY_LARGE  = 3
SELL_SMALL = 4
SELL_MED   = 5
SELL_LARGE = 6
NUM_ACTIONS = 7

# Sign: +1 for buy, -1 for sell, 0 for flat
_ACTION_SIGN = {
    FLAT: 0, BUY_SMALL: 1, BUY_MED: 1, BUY_LARGE: 1,
    SELL_SMALL: -1, SELL_MED: -1, SELL_LARGE: -1,
}
# Size label
_ACTION_SIZE = {
    FLAT: None, BUY_SMALL: "small", BUY_MED: "med", BUY_LARGE: "large",
    SELL_SMALL: "small", SELL_MED: "med", SELL_LARGE: "large",
}


# ── FTMO daily tracker ────────────────────────────────────────────────────────
class FTMODay:
    """Tracks intraday equity for FTMO pass/fail logic (per CET day)."""

    def __init__(self, start_equity: float,
                 profit_target_pct: float = 0.025,
                 max_dd_pct: float        = 0.010,
                 training_mode: bool      = True):
        self.start_equity      = start_equity
        self.profit_target_pct = profit_target_pct
        self.max_dd_pct        = max_dd_pct
        self.training_mode     = training_mode  # True: can trade past target; False: hard stop
        self.intraday_high     = start_equity
        self.current_equity    = start_equity
        self.target_hit        = False
        self.failed            = False

    def update(self, equity: float):
        self.current_equity = equity
        self.intraday_high  = max(self.intraday_high, equity)
        # Check drawdown
        if equity <= self.intraday_high * (1.0 - self.max_dd_pct):
            self.failed = True
        # Check target
        if equity >= self.start_equity * (1.0 + self.profit_target_pct):
            self.target_hit = True

    @property
    def can_open_new_trade(self) -> bool:
        if self.failed:
            return False
        # In live mode (training_mode=False), hard stop once target hit
        if not self.training_mode and self.target_hit:
            return False
        return True

    @property
    def daily_return(self) -> float:
        return (self.current_equity - self.start_equity) / self.start_equity

    @property
    def daily_drawdown(self) -> float:
        return (self.intraday_high - self.current_equity) / self.intraday_high

    def classify(self) -> str:
        if self.failed:
            return "fail"
        if self.target_hit and not self.failed:
            return "pass"
        if self.daily_return >= 0:
            return "ok"
        return "fail"


# ── Per-asset position tracker ────────────────────────────────────────────────
class AssetPosition:
    def __init__(self):
        self.side: int        = 0    # +1, -1, 0
        self.entry: float     = 0.0
        self.lots: float      = 0.0
        self.unrealised: float = 0.0

    def open(self, side: int, price: float, lots: float):
        self.side  = side
        self.entry = price
        self.lots  = lots

    def close(self):
        self.side  = 0
        self.entry = 0.0
        self.lots  = 0.0
        self.unrealised = 0.0

    def update_unrealised(self, curr_price: float):
        if self.side != 0 and self.entry != 0:
            self.unrealised = (curr_price - self.entry) / self.entry * self.side
        else:
            self.unrealised = 0.0

    @property
    def size_pct_equity(self) -> float:
        return abs(self.lots) if self.side != 0 else 0.0


# ── Multi-asset FTMO Game ─────────────────────────────────────────────────────
class FTMOGame:
    """
    Multi-asset, multi-timeframe RL environment with FTMO constraints
    and 4-phase curriculum masking.

    data_dict : {symbol: {tf_minutes: feature_DataFrame}}
    Each feature_DataFrame is the output of env.indicators.build_feature_df().
    The 1m DataFrame is the "clock"; all higher TFs are looked up by timestamp.
    """

    def __init__(
        self,
        data_dict:         Dict[str, Dict[int, pd.DataFrame]],
        symbols:           List[str],
        reward_cfg:        dict,
        ftmo_cfg:          dict,
        trading_mode:      str   = "ftmo",
        curriculum_phase:  int   = 1,
        initial_equity:    float = 100_000.0,
        risk_fractions:    dict  = None,
        lkbk:              int   = 20,
        init_idx:          int   = None,
        training_mode:     bool  = True,  # True: training (can trade past target); False: live
    ):
        self.data_dict        = data_dict
        self.symbols          = symbols
        self.reward_cfg       = reward_cfg
        self.ftmo_cfg         = ftmo_cfg
        self.trading_mode     = trading_mode
        self.rl_training_mode = training_mode  # True: training, False: live
        self.curriculum_phase = curriculum_phase
        self.initial_equity   = initial_equity
        self.lkbk             = lkbk

        # Learnable risk fractions (small/med/large as % of equity)
        self.risk_fractions = risk_fractions or {
            "small": 0.005, "med": 0.010, "large": 0.020
        }

        # Use 1m bars of first symbol as the master clock
        self._clock = data_dict[symbols[0]][1]
        n = len(self._clock)
        # init_idx: allow caller to pass a pre-computed warm-up offset;
        # fall back to lkbk+200 so indicators have enough history.
        self.init_idx   = int(init_idx) if init_idx is not None else max(lkbk + 200, 1200)
        self.init_idx   = min(self.init_idx, n - 2)   # safety clamp
        self.curr_idx   = self.init_idx
        self.max_idx    = n - 1

        # State
        self.positions: Dict[str, AssetPosition] = {s: AssetPosition() for s in symbols}
        self.equity          = initial_equity
        self.ftmo_day        = FTMODay(
            initial_equity,
            training_mode=self.rl_training_mode,
            **ftmo_cfg
        )
        self._curr_day: date = None
        self.days_in_streak  = 0
        self._day_results: List[str] = []

        self.is_over   = False
        self.reward    = 0.0
        self.trade_log = []
        self.total_fees = 0.0  # Track cumulative fees

        # Rolling daily returns for Sharpe bonus (last 20 results)
        self._rolling_returns: List[float] = []
        self._ath_sharpe: float = 0.0

        # Cache for _lookup_row: {(symbol, tf): last_iloc_index}
        # Avoids repeated O(log N) binary searches during sequential playback.
        self._lookup_cache: Dict[Tuple[str, int], int] = {}

        # Performance monitoring
        self.step_count = 0
        self._realised_pnl = 0.0

        self.reset()

    # ── index helpers ─────────────────────────────────────────────────────────
    def _curr_time(self):
        return self._clock.index[self.curr_idx]

    def _lookup_row(self, symbol: str, tf: int) -> Optional[pd.Series]:
        """Get the most recent row of (symbol, tf) at or before curr_time."""
        df  = self.data_dict[symbol][tf]
        t   = self._curr_time()
        # Use cached position when available to avoid repeated binary searches
        key = (symbol, tf)
        cached_idx = self._lookup_cache.get(key, 0)
        idx_arr = df.index
        n = len(idx_arr)

        # A8: Validate dataframe is sorted
        assert df.index.is_monotonic_increasing, f"[A8] df[{symbol}, {tf}] index is not monotonic!"
        # A1: Validate cache bounds
        assert 0 <= cached_idx < n, f"[A1] Cache out of bounds: {cached_idx} not in [0, {n})"

        # Walk forward from cache — O(1) amortised during sequential playback
        while cached_idx + 1 < n and idx_arr[cached_idx + 1] <= t:
            cached_idx += 1

        # Clamp: if cache is ahead of t, fall back to binary search
        if cached_idx >= n or idx_arr[cached_idx] > t:
            cached_idx = df.index.searchsorted(t, side="right") - 1
            # A7: Trace cache fallback for debugging
            if self.step_count % 10000 == 0:
                print(f"[A7] _lookup_row({symbol}, {tf}): binary search fallback at curr_idx={self.curr_idx}, cached_idx was {self._lookup_cache.get(key, 0)}")

        # A4: Validate result is at correct boundary
        if cached_idx >= 0:
            assert idx_arr[cached_idx] <= t, f"[A4] Cache result {idx_arr[cached_idx]} > {t}"
            if cached_idx < n - 1:
                assert idx_arr[cached_idx + 1] > t, f"[A4] Cache skipped row at {idx_arr[cached_idx + 1]}"

        self._lookup_cache[key] = cached_idx
        if cached_idx < 0:
            return None
        return df.iloc[cached_idx]

    # ── position sizing ───────────────────────────────────────────────────────
    def _lots_for_action(self, action: int) -> float:
        size_label = _ACTION_SIZE[action]
        if size_label is None:
            return 0.0
        frac = self.risk_fractions[size_label]
        return max(0.001, self.equity * frac / 100_000)  # rough lot scaling

    # ── FTMO enforcement ──────────────────────────────────────────────────────
    def _check_day_boundary(self):
        curr_day = self._curr_time().date()
        if self._curr_day is None:
            self._curr_day = curr_day
            return
        if curr_day != self._curr_day:
            result = self.ftmo_day.classify()
            self._day_results.append(result)
            self._update_streak(result)
            self.ftmo_day = FTMODay(
                self.equity,
                training_mode=self.rl_training_mode,
                **self.ftmo_cfg
            )
            self._curr_day = curr_day

    def _update_streak(self, result: str):
        if result == "pass":
            self.days_in_streak += 1
        else:
            self.days_in_streak = 0

    def _is_weekend_close_window(self) -> bool:
        """True if current time is >= 22:50 CET on Friday."""
        t = self._curr_time()
        return t.weekday() == 4 and t.hour * 60 + t.minute >= 22 * 60 + 50

    def _rolling_sharpe(self) -> float:
        """Annualised Sharpe over the last 20 daily returns."""
        r = self._rolling_returns[-20:]
        if len(r) < 5:
            return 0.0
        arr = np.array(r, dtype=float)
        std = arr.std()
        if std == 0:
            return 0.0
        return float((arr.mean() / std) * np.sqrt(252))

    def _daily_reward(self, result: str) -> float:
        cfg = self.reward_cfg
        if result == "pass":
            base = cfg.get("pass_day_bonus", 2.0)
        elif result == "ok":
            base = cfg.get("ok_day_bonus", 0.5)
        else:
            base = cfg.get("fail_day_penalty", -2.0)
        streak_bonus = cfg.get("streak_scale", 0.1) * self.days_in_streak
        low_dd_bonus = 0.0
        if self.ftmo_day.daily_drawdown < cfg.get("low_dd_threshold", 0.005):
            low_dd_bonus = cfg.get("low_dd_bonus", 0.3)

        # Sharpe bonus (only on passing days)
        sharpe_bonus = 0.0
        if result == "pass" and cfg.get("sharpe_bonus_enabled", False):
            rs = self._rolling_sharpe()
            if rs > 1.0:
                sharpe_bonus += cfg.get("sharpe_bonus_scale", 0.1) * (rs - 1.0)
            if rs > self._ath_sharpe:
                sharpe_bonus += cfg.get("sharpe_ath_bonus", 0.2)
                self._ath_sharpe = rs

        # Track daily return for rolling Sharpe
        self._rolling_returns.append(self.ftmo_day.daily_return)

        return base + streak_bonus + low_dd_bonus + sharpe_bonus

    # ── state assembly ────────────────────────────────────────────────────────
    def _assemble_state(self) -> np.ndarray:
        # A9: Validate cache size doesn't grow unbounded
        assert len(self._lookup_cache) <= len(self.symbols) * len(self.data_dict[self.symbols[0]]) * 1.5, \
            f"[A9] Cache unbounded: {len(self._lookup_cache)} entries, expected <= {len(self.symbols) * 4}"

        parts = []
        for sym in self.symbols:
            for tf in [1440, 60, 15, 1]:
                row = self._lookup_row(sym, tf)
                if row is None:
                    parts.append(np.zeros(50))
                    continue
                # Reuse the cache entry set by _lookup_row (avoids second searchsorted)
                end_idx   = self._lookup_cache.get((sym, tf), 0) + 1
                start_idx = max(0, end_idx - self.lkbk)
                df        = self.data_dict[sym][tf]
                window    = df.iloc[start_idx:end_idx]

                # A6: Validate cache result matches expectation
                if len(window) > 0:
                    assert df.index[end_idx - 1] == row.name, \
                        f"[A6] Cache mismatch: row {row.name} != cached {df.index[end_idx - 1]}"

                arr       = window.select_dtypes(include=[np.number]).values
                arr       = arr[-min(10, len(arr)):]   # last 10 rows
                arr       = arr.flatten()
                # z-score
                std = arr.std()
                if std > 0:
                    arr = (arr - arr.mean()) / std
                parts.append(arr)

            # per-asset position features
            pos = self.positions[sym]
            parts.append(np.array([
                pos.side,
                pos.size_pct_equity,
                pos.unrealised,
            ]))

        # account-level features
        eq_chg  = (self.equity - self.ftmo_day.start_equity) / self.ftmo_day.start_equity
        dd      = self.ftmo_day.daily_drawdown
        streak  = self.days_in_streak
        parts.append(np.array([eq_chg, dd, streak]))

        # pad / trim each part to fixed length then concatenate
        fixed_parts = []
        for p in parts:
            if len(p) < 200:
                p = np.concatenate([p, np.zeros(200 - len(p))])
            else:
                p = p[:200]
            fixed_parts.append(p)

        return np.concatenate(fixed_parts)

    # ── curriculum mask ───────────────────────────────────────────────────────
    def _can_open(self, symbol: str) -> bool:
        """Check phase mask for opening new trades on this symbol."""
        if self.trading_mode == "free" or self.curriculum_phase == 4:
            return True
        if not self.ftmo_day.can_open_new_trade:
            return False
        row_1m  = self._lookup_row(symbol, 1)
        if self.curriculum_phase == 1:
            row_15m = self._lookup_row(symbol, 15)
            if row_1m is None or row_15m is None:
                return False
            return can_open_trade_phase1(row_1m, row_15m)
        if self.curriculum_phase == 2:
            row_1h = self._lookup_row(symbol, 60)
            if row_1m is None or row_1h is None:
                return False
            # Phase 2 allows entry when condition holds; agent decides
            return must_be_in_trade_phase2(row_1m, row_1h)
        if self.curriculum_phase == 3:
            row_15m = self._lookup_row(symbol, 15)
            if row_1m is None or row_15m is None:
                return False
            return must_be_in_trade_phase3(row_1m, row_15m)
        return True

    def _must_enter(self, symbol: str) -> bool:
        """Phase 2 & 3 force entry when condition holds."""
        if self.curriculum_phase == 2:
            row_1m = self._lookup_row(symbol, 1)
            row_1h = self._lookup_row(symbol, 60)
            if row_1m is not None and row_1h is not None:
                return must_be_in_trade_phase2(row_1m, row_1h)
        if self.curriculum_phase == 3:
            row_1m  = self._lookup_row(symbol, 1)
            row_15m = self._lookup_row(symbol, 15)
            if row_1m is not None and row_15m is not None:
                return must_be_in_trade_phase3(row_1m, row_15m)
        return False

    # ── act ───────────────────────────────────────────────────────────────────
    def act(self, actions: Dict[str, int]) -> Tuple[float, bool]:
        """
        actions: {symbol: action_int}
        Returns (total_reward, is_over).
        """
        self._check_day_boundary()
        t = self._curr_time()
        step_reward = 0.0

        # Force flat after 22:50 CET Friday — no positions over the weekend
        if self._is_weekend_close_window():
            for sym in self.symbols:
                pos = self.positions[sym]
                if pos.side != 0:
                    price_row = self._lookup_row(sym, 1)
                    if price_row is not None:
                        pnl = self._close_position(sym, float(price_row["close"]), t)
                        step_reward += pnl
                        pos.close()

        for sym in self.symbols:
            action     = actions.get(sym, FLAT)
            pos        = self.positions[sym]

            # Block new openings in weekend window
            if self._is_weekend_close_window():
                actions[sym] = FLAT
                action = FLAT
            price_row  = self._lookup_row(sym, 1)
            if price_row is None:
                continue
            curr_price = float(price_row["close"])

            sign = _ACTION_SIGN[action]

            # Force entry in phases 2 & 3 when must_enter
            if pos.side == 0 and self._must_enter(sym) and sign == 0:
                sign = 1   # default to long; agent will learn better direction

            # Close trade if direction reverses
            if pos.side != 0 and sign != 0 and sign != pos.side:
                pnl = self._close_position(sym, curr_price, t)
                step_reward += pnl
                pos.close()

            # Open new trade
            if pos.side == 0 and sign != 0:
                if self._can_open(sym):
                    lots = self._lots_for_action(action)
                    pos.open(sign, curr_price, lots)

            pos.update_unrealised(curr_price)

        # News awareness penalty — agent learns to adapt, not hard-blocked
        num_open = sum(1 for p in self.positions.values() if p.side != 0)
        step_reward += _news_penalty(t, num_open,
                                     self.reward_cfg.get("news_penalty", 0.5))

        # Update equity (simplified: sum unrealised across all assets)
        total_unrealised = sum(p.unrealised * p.lots * 1000 for p in self.positions.values())
        self.equity = self.initial_equity + total_unrealised + self._realised_pnl
        self.ftmo_day.update(self.equity)

        # Terminal condition
        if self.curr_idx >= self.max_idx:
            self.is_over = True
            result = self.ftmo_day.classify()
            step_reward += self._daily_reward(result)

        return step_reward, self.is_over

    def _close_position(self, symbol: str, curr_price: float, t) -> float:
        pos = self.positions[symbol]
        if pos.entry == 0:
            return 0.0
        pnl_pct = (curr_price - pos.entry) / pos.entry * pos.side
        pnl_abs = pnl_pct * pos.lots * 1000

        # Calculate fee and PnL after fees
        pnl_after_fees, fee = calculate_pnl_after_fees(
            symbol=symbol,
            entry_price=pos.entry,
            exit_price=curr_price,
            lots=pos.lots,
            pnl_abs=pnl_abs
        )

        self._realised_pnl += pnl_after_fees
        self.total_fees += fee

        self.trade_log.append({
            "time": t, "symbol": symbol,
            "side": pos.side, "entry": pos.entry,
            "exit": curr_price, "lots": pos.lots,
            "pnl_abs_before_fees": pnl_abs,
            "fee": fee,
            "pnl_abs": pnl_after_fees,
            "pnl_pct": pnl_pct,
        })
        return pnl_pct  # return PnL % as reward component

    def get_state(self) -> np.ndarray:
        state = self._assemble_state()
        return state.reshape(1, -1)

    def step(self):
        """Advance the clock by one 1m bar."""
        # B4: Guard against out-of-bounds
        base_len = len(self.data_dict[self.symbols[0]][1])
        if self.curr_idx + 1 >= base_len:
            self.is_over = True
            return

        self.curr_idx += 1
        self.step_count += 1

        # D5: Verify cache state is sensible
        assert len(self._lookup_cache) <= len(self.symbols) * 4 * 2, \
            f"[D5] Cache grew unexpectedly: {len(self._lookup_cache)} entries"

    def reset(self):
        self.curr_idx       = self.init_idx
        self.equity         = self.initial_equity
        self._realised_pnl  = 0.0
        self.total_fees     = 0.0  # Reset fees
        self._curr_day      = None
        self.days_in_streak = 0
        self._day_results   = []
        self.positions      = {s: AssetPosition() for s in self.symbols}
        self.ftmo_day       = FTMODay(
            self.initial_equity,
            training_mode=self.rl_training_mode,
            **self.ftmo_cfg
        )
        self.is_over        = False
        self.reward         = 0.0
        self.trade_log      = []

        # D3: Comprehensive cache reset with validation
        cache_size_before = len(self._lookup_cache)
        self._lookup_cache  = {}   # invalidate cache on reset

        # D5: Validate reset state
        assert len(self._lookup_cache) == 0, "[D5] Cache not cleared on reset"
        assert all(p.side == 0 for p in self.positions.values()), "[D5] Positions not cleared on reset"
        assert self._realised_pnl == 0.0, "[D5] PnL not reset"
        assert self.curr_idx == self.init_idx, "[D5] curr_idx not reset to init_idx"

        # B1: Log episode boundary for traceability
        print(f"[B1] reset() at step={self.step_count}: cache_size_before={cache_size_before}, curr_idx={self.curr_idx}")

        # advance to init_idx
        self._check_day_boundary()
