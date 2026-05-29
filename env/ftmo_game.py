"""
env/ftmo_game.py
FTMO-aware multi-asset Game environment with 8-phase curriculum masking.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from env.indicators import (
    build_feature_df,
    phase0_cci_extreme,
    phase1_cci_align,
    phase2_hilo_trend,
    phase3_hilo_counter,
    phase4_bb_position,
    phase5_sma_stack,
    phase6_atr_expansion,
    # backward-compat aliases still importable
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

_ACTION_SIGN = {
    FLAT: 0, BUY_SMALL: 1, BUY_MED: 1, BUY_LARGE: 1,
    SELL_SMALL: -1, SELL_MED: -1, SELL_LARGE: -1,
}
_ACTION_SIZE = {
    FLAT: None, BUY_SMALL: "small", BUY_MED: "med", BUY_LARGE: "large",
    SELL_SMALL: "small", SELL_MED: "med", SELL_LARGE: "large",
}

# ── Phase mask dispatch ───────────────────────────────────────────────────────
# Maps mask name from config -> (function, mask_type, [tf1, tf2])
# mask_type: "force_in_and_gate" | "open_gate" | "free"
_MASK_REGISTRY: Dict[str, tuple] = {
    "phase0_cci_extreme":  (phase0_cci_extreme,  "force_in_and_gate", [1, 15]),
    "phase1_cci_align":    (phase1_cci_align,     "open_gate",         [1, 15]),
    "phase2_hilo_trend":   (phase2_hilo_trend,    "force_in_and_gate", [1, 30]),
    "phase3_hilo_counter": (phase3_hilo_counter,  "force_in_and_gate", [1, 15]),
    "phase4_bb_position":  (phase4_bb_position,   "force_in_and_gate", [1, 15]),
    "phase5_sma_stack":    (phase5_sma_stack,      "force_in_and_gate", [1, 60]),
    "phase6_atr_expansion":(phase6_atr_expansion,  "force_in_and_gate", [1, 60]),
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
        self.training_mode     = training_mode
        self.intraday_high     = start_equity
        self.current_equity    = start_equity
        self.target_hit        = False
        self.failed            = False

    def update(self, equity: float):
        self.current_equity = equity
        self.intraday_high  = max(self.intraday_high, equity)
        if equity <= self.intraday_high * (1.0 - self.max_dd_pct):
            self.failed = True
        if equity >= self.start_equity * (1.0 + self.profit_target_pct):
            self.target_hit = True

    @property
    def can_open_new_trade(self) -> bool:
        if self.failed:
            return False
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
        self.side: int         = 0
        self.entry: float      = 0.0
        self.lots: float       = 0.0
        self.unrealised: float = 0.0

    def open(self, side: int, price: float, lots: float):
        self.side  = side
        self.entry = price
        self.lots  = lots

    def close(self):
        self.side      = 0
        self.entry     = 0.0
        self.lots      = 0.0
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
    and 8-phase curriculum masking.

    data_dict : {symbol: {tf_minutes: feature_DataFrame}}
    The 1m DataFrame is the "clock"; all higher TFs are looked up by timestamp.
    """

    def __init__(
        self,
        data_dict:         Dict[str, Dict[int, pd.DataFrame]],
        symbols:           List[str],
        reward_cfg:        dict,
        ftmo_cfg:          dict,
        trading_mode:      str   = "ftmo",
        curriculum_phase:  int   = 0,
        initial_equity:    float = 100_000.0,
        risk_fractions:    dict  = None,
        lkbk:              int   = 20,
        init_idx:          int   = None,
        training_mode:     bool  = True,
        phase_cfg:         dict  = None,   # full phase config entry from YAML
        max_trades_per_day: int  = 800,
    ):
        self.data_dict         = data_dict
        self.symbols           = symbols
        self.reward_cfg        = reward_cfg
        self.ftmo_cfg          = ftmo_cfg
        self.trading_mode      = trading_mode
        self.rl_training_mode  = training_mode
        self.curriculum_phase  = curriculum_phase
        self.initial_equity    = initial_equity
        self.lkbk              = lkbk
        self.max_trades_per_day = max_trades_per_day

        self.risk_fractions = risk_fractions or {
            "small": 0.005, "med": 0.010, "large": 0.020
        }

        # Resolve mask function and type from phase_cfg or fall back to legacy int
        self._mask_fn   = None
        self._mask_type = "free"
        self._mask_tfs  = []

        if phase_cfg is not None:
            mask_name = phase_cfg.get("mask")
            if mask_name and mask_name in _MASK_REGISTRY:
                self._mask_fn, self._mask_type, self._mask_tfs = _MASK_REGISTRY[mask_name]
        else:
            # Legacy int-based phase dispatch (backward compat)
            _legacy = {
                0: ("phase0_cci_extreme",  _MASK_REGISTRY.get("phase0_cci_extreme",  (None,"free",[]))),
                1: ("phase1_cci_align",    _MASK_REGISTRY.get("phase1_cci_align",    (None,"free",[]))),
                2: ("phase2_hilo_trend",   _MASK_REGISTRY.get("phase2_hilo_trend",   (None,"free",[]))),
                3: ("phase3_hilo_counter", _MASK_REGISTRY.get("phase3_hilo_counter", (None,"free",[]))),
                4: ("phase4_bb_position",  _MASK_REGISTRY.get("phase4_bb_position",  (None,"free",[]))),
                5: ("phase5_sma_stack",    _MASK_REGISTRY.get("phase5_sma_stack",    (None,"free",[]))),
                6: ("phase6_atr_expansion",_MASK_REGISTRY.get("phase6_atr_expansion",(None,"free",[]))),
                7: (None, (None, "free", [])),
            }
            if curriculum_phase in _legacy:
                _, entry = _legacy[curriculum_phase]
                if isinstance(entry, tuple) and len(entry) == 3:
                    self._mask_fn, self._mask_type, self._mask_tfs = entry

        # Master clock
        self._clock   = data_dict[symbols[0]][1]
        n             = len(self._clock)
        self.init_idx = int(init_idx) if init_idx is not None else max(lkbk + 200, 1200)
        self.init_idx = min(self.init_idx, n - 2)
        self.curr_idx = self.init_idx
        self.max_idx  = n - 1

        # State
        self.positions: Dict[str, AssetPosition] = {s: AssetPosition() for s in symbols}
        self.equity         = initial_equity
        self.ftmo_day       = FTMODay(initial_equity, training_mode=self.rl_training_mode, **ftmo_cfg)
        self._curr_day: date = None
        self.days_in_streak  = 0
        self._day_results: List[str] = []
        self._daily_trade_count: int = 0   # reset each new CET day

        self.is_over    = False
        self.reward     = 0.0
        self.trade_log  = []
        self.total_fees = 0.0

        self._rolling_returns: List[float] = []
        self._ath_sharpe: float = 0.0

        self._lookup_cache: Dict[Tuple[str, int], int] = {}
        self._base_len = len(self._clock)
        self.step_count = 0
        self._realised_pnl = 0.0

        self.reset()

    # ── index helpers ─────────────────────────────────────────────────────────
    def _curr_time(self):
        return self._clock.index[self.curr_idx]

    def _lookup_row(self, symbol: str, tf: int) -> Optional[pd.Series]:
        """Get the most recent row of (symbol, tf) at or before curr_time."""
        if tf not in self.data_dict[symbol]:
            return None
        df  = self.data_dict[symbol][tf]
        t   = self._curr_time()
        key = (symbol, tf)
        cached_idx = self._lookup_cache.get(key, 0)
        idx_arr    = df.index
        n          = len(idx_arr)

        while cached_idx + 1 < n and idx_arr[cached_idx + 1] <= t:
            cached_idx += 1

        if cached_idx >= n or idx_arr[cached_idx] > t:
            cached_idx = df.index.searchsorted(t, side="right") - 1

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
        return max(0.001, self.equity * frac / 100_000)

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
            self._daily_trade_count = 0   # reset daily trade counter

    def _update_streak(self, result: str):
        if result == "pass":
            self.days_in_streak += 1
        else:
            self.days_in_streak = 0

    def _is_weekend_close_window(self) -> bool:
        t = self._curr_time()
        return t.weekday() == 4 and t.hour * 60 + t.minute >= 22 * 60 + 50

    def _rolling_sharpe(self) -> float:
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

        sharpe_bonus = 0.0
        if result == "pass" and cfg.get("sharpe_bonus_enabled", False):
            rs = self._rolling_sharpe()
            if rs > 1.0:
                sharpe_bonus += cfg.get("sharpe_bonus_scale", 0.1) * (rs - 1.0)
            if rs > self._ath_sharpe:
                sharpe_bonus += cfg.get("sharpe_ath_bonus", 0.2)
                self._ath_sharpe = rs

        self._rolling_returns.append(self.ftmo_day.daily_return)
        return base + streak_bonus + low_dd_bonus + sharpe_bonus

    # ── state assembly ────────────────────────────────────────────────────────
    def _assemble_state(self) -> np.ndarray:
        parts = []
        for sym in self.symbols:
            for tf in [1440, 60, 15, 1]:
                row = self._lookup_row(sym, tf)
                if row is None:
                    parts.append(np.zeros(50))
                    continue
                end_idx   = self._lookup_cache.get((sym, tf), 0) + 1
                start_idx = max(0, end_idx - self.lkbk)
                df        = self.data_dict[sym][tf]
                window    = df.iloc[start_idx:end_idx]
                arr       = window.select_dtypes(include=[np.number]).values
                arr       = arr[-min(10, len(arr)):]
                arr       = arr.flatten()
                std = arr.std()
                if std > 0:
                    arr = (arr - arr.mean()) / std
                parts.append(arr)

            pos = self.positions[sym]
            parts.append(np.array([pos.side, pos.size_pct_equity, pos.unrealised]))

        eq_chg = (self.equity - self.ftmo_day.start_equity) / self.ftmo_day.start_equity
        dd     = self.ftmo_day.daily_drawdown
        parts.append(np.array([eq_chg, dd, self.days_in_streak]))

        fixed_parts = []
        for p in parts:
            if len(p) < 200:
                p = np.concatenate([p, np.zeros(200 - len(p))])
            else:
                p = p[:200]
            fixed_parts.append(p)

        return np.concatenate(fixed_parts)

    # ── phase mask evaluation ─────────────────────────────────────────────────
    def _eval_mask(self, symbol: str) -> Tuple[bool, bool]:
        """
        Returns (condition_met, must_force_open).
        condition_met: whether the mask condition is active.
        must_force_open: True only for force_in_and_gate when agent is flat.
        """
        if self._mask_fn is None or self._mask_type == "free":
            return True, False

        tfs = self._mask_tfs
        if len(tfs) < 2:
            return True, False

        row1  = self._lookup_row(symbol, tfs[0])
        row2  = self._lookup_row(symbol, tfs[1])
        if row1 is None or row2 is None:
            # Missing data — allow trading, no force
            return True, False

        condition = self._mask_fn(row1, row2)

        if self._mask_type == "force_in_and_gate":
            return condition, condition  # if condition: must be in trade
        if self._mask_type == "open_gate":
            return condition, False      # condition gates open, never forces
        return True, False

    def _can_open(self, symbol: str) -> bool:
        """Returns True if a new trade may be opened for this symbol."""
        if not self.ftmo_day.can_open_new_trade:
            return False
        if self._daily_trade_count >= self.max_trades_per_day:
            return False
        condition, _ = self._eval_mask(symbol)
        return condition

    def _must_enter(self, symbol: str) -> bool:
        """Returns True when force_in_and_gate condition requires the agent to open."""
        if self._mask_type != "force_in_and_gate":
            return False
        _, must_force = self._eval_mask(symbol)
        return must_force

    # ── act ───────────────────────────────────────────────────────────────────
    def act(self, actions: Dict[str, int]) -> Tuple[float, bool]:
        """
        actions: {symbol: action_int}
        Returns (total_reward, is_over).
        """
        self._check_day_boundary()
        t = self._curr_time()
        step_reward = 0.0

        # Force flat after 22:50 CET Friday
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
            action = actions.get(sym, FLAT)
            pos    = self.positions[sym]

            if self._is_weekend_close_window():
                action = FLAT

            price_row = self._lookup_row(sym, 1)
            if price_row is None:
                continue
            curr_price = float(price_row["close"])
            sign       = _ACTION_SIGN[action]

            # Force entry when mask demands it and agent is flat
            if pos.side == 0 and sign == 0 and self._must_enter(sym):
                if self.ftmo_day.can_open_new_trade and self._daily_trade_count < self.max_trades_per_day:
                    sign = 1   # default long; agent learns direction over time

            # Close on direction reversal
            if pos.side != 0 and sign != 0 and sign != pos.side:
                pnl = self._close_position(sym, curr_price, t)
                step_reward += pnl
                pos.close()

            # Open new trade
            if pos.side == 0 and sign != 0:
                if self._can_open(sym):
                    lots = self._lots_for_action(action)
                    pos.open(sign, curr_price, lots)
                    self._daily_trade_count += 1

            pos.update_unrealised(curr_price)

        num_open = sum(1 for p in self.positions.values() if p.side != 0)
        step_reward += _news_penalty(t, num_open, self.reward_cfg.get("news_penalty", 0.5))

        total_unrealised = sum(p.unrealised * p.lots * 1000 for p in self.positions.values())
        self.equity = self.initial_equity + total_unrealised + self._realised_pnl
        self.ftmo_day.update(self.equity)

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

        pnl_after_fees, fee = calculate_pnl_after_fees(
            symbol=symbol,
            entry_price=pos.entry,
            exit_price=curr_price,
            lots=pos.lots,
            pnl_abs=pnl_abs
        )

        self._realised_pnl += pnl_after_fees
        self.total_fees    += fee

        self.trade_log.append({
            "time":                t,
            "symbol":              symbol,
            "side":                pos.side,
            "entry":               pos.entry,
            "exit":                curr_price,
            "lots":                pos.lots,
            "pnl_abs_before_fees": pnl_abs,
            "fee":                 fee,
            "pnl_abs":             pnl_after_fees,
            "pnl_pct":             pnl_pct,
        })
        return pnl_pct

    def get_state(self) -> np.ndarray:
        return self._assemble_state().reshape(1, -1)

    def step(self):
        """Advance the clock by one 1m bar."""
        if self.curr_idx + 1 >= self._base_len:
            self.is_over = True
            return
        self.curr_idx  += 1
        self.step_count += 1

    def reset(self):
        self.curr_idx          = self.init_idx
        self.equity            = self.initial_equity
        self._realised_pnl     = 0.0
        self.total_fees        = 0.0
        self._curr_day         = None
        self.days_in_streak    = 0
        self._day_results      = []
        self.positions         = {s: AssetPosition() for s in self.symbols}
        self.ftmo_day          = FTMODay(
            self.initial_equity,
            training_mode=self.rl_training_mode,
            **self.ftmo_cfg
        )
        self.is_over           = False
        self.reward            = 0.0
        self.trade_log         = []
        self._daily_trade_count = 0
        self._lookup_cache     = {}
        self._check_day_boundary()
