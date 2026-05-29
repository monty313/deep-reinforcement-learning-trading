"""
monitoring/concentration_analyzer.py
Track notional exposure and enforce per-symbol / total leverage limits.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Default FTMO-style limits (override via config)
DEFAULT_MAX_TOTAL_NOTIONAL_PCT  = 0.05   # 5 % of equity
DEFAULT_MAX_SYMBOL_NOTIONAL_PCT = 0.03   # 3 % per symbol


class ConcentrationAnalyzer:
    """
    Tracks per-symbol notional exposure and enforces concentration limits.
    Used by MT5Bridge before sending orders live.
    """

    def __init__(
        self,
        max_total_pct:  float = DEFAULT_MAX_TOTAL_NOTIONAL_PCT,
        max_symbol_pct: float = DEFAULT_MAX_SYMBOL_NOTIONAL_PCT,
    ):
        self.max_total_pct  = max_total_pct
        self.max_symbol_pct = max_symbol_pct
        self._exposure: Dict[str, float] = {}   # symbol -> notional as fraction of equity
        self.log: List[dict] = []

    def update(self, symbol: str, lots: float, price: float, equity: float):
        """Update tracked exposure for a symbol."""
        notional = lots * price / equity if equity > 0 else 0.0
        self._exposure[symbol] = notional

    def clear(self, symbol: str):
        self._exposure.pop(symbol, None)

    def total_exposure(self) -> float:
        return sum(self._exposure.values())

    def symbol_exposure(self, symbol: str) -> float:
        return self._exposure.get(symbol, 0.0)

    def check_action(self, symbol: str, lots: float, price: float,
                     equity: float) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Call before executing a new position.
        """
        new_sym_exp   = lots * price / equity if equity > 0 else 0.0
        total_after   = self.total_exposure() - self._exposure.get(symbol, 0.0) + new_sym_exp

        if new_sym_exp > self.max_symbol_pct:
            msg = (f"Rejected action — {symbol} exposure {new_sym_exp:.2%} "
                   f"exceeds per-symbol limit {self.max_symbol_pct:.2%}")
            self.log.append({"symbol": symbol, "lots": lots, "reason": msg})
            return False, msg

        if total_after > self.max_total_pct:
            msg = (f"Rejected action — total exposure {total_after:.2%} "
                   f"exceeds limit {self.max_total_pct:.2%}")
            self.log.append({"symbol": symbol, "lots": lots, "reason": msg})
            return False, msg

        return True, "ok"

    def daily_report(self, equity: float) -> dict:
        return {
            "date":            str(date.today()),
            "equity":          equity,
            "total_notional_pct": round(self.total_exposure() * 100, 4),
            "per_symbol":      {s: round(v * 100, 4) for s, v in self._exposure.items()},
            "limit_total_pct": self.max_total_pct * 100,
            "limit_symbol_pct": self.max_symbol_pct * 100,
            "breach":          self.total_exposure() > self.max_total_pct,
        }

    def save_report(self, equity: float, path: str = "concentration_report.json"):
        report = self.daily_report(equity)
        Path(path).write_text(json.dumps(report, indent=2))
        return report
