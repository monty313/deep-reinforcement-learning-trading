"""
agents/meta_learner.py
Placeholder meta-learner: reads per-agent returns & drawdowns,
computes rolling Sharpe/DD/correlation, and reallocates capital across agents.
Replace the allocation logic with a proper meta-RL or bandit algorithm later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class MetaLearner:
    """
    Monitors performance of multiple trained agents and allocates capital
    proportional to rolling risk-adjusted returns.

    Usage:
        meta = MetaLearner(agent_ids=["worker_0", "worker_1", "worker_2"])
        meta.update("worker_0", daily_return=0.03, daily_dd=0.005)
        alloc = meta.get_allocations()
    """

    def __init__(
        self,
        agent_ids:    List[str],
        window:       int   = 30,
        risk_free:    float = 0.0,
        min_weight:   float = 0.05,
    ):
        self.agent_ids  = agent_ids
        self.window     = window
        self.risk_free  = risk_free
        self.min_weight = min_weight
        self._history: Dict[str, List[dict]] = {a: [] for a in agent_ids}

    # ── data ingestion ────────────────────────────────────────────────────────
    def update(self, agent_id: str, daily_return: float, daily_dd: float):
        """Record one day's performance for an agent."""
        if agent_id not in self._history:
            self._history[agent_id] = []
        self._history[agent_id].append({
            "daily_return": daily_return,
            "daily_dd":     daily_dd,
        })

    def load_from_csv(self, agent_id: str, csv_path: str):
        """Load historical performance from a forward-test CSV."""
        df = pd.read_csv(csv_path, parse_dates=["time"])
        df["date"]         = df["time"].dt.date
        daily              = df.groupby("date").agg(start=("equity", "first"), end=("equity", "last"), peak=("equity", "max")).reset_index()
        daily["ret"]       = (daily["end"] - daily["start"]) / daily["start"]
        daily["dd"]        = (daily["peak"] - daily["end"]) / daily["peak"]
        for _, row in daily.iterrows():
            self.update(agent_id, row["ret"], row["dd"])

    # ── metrics ───────────────────────────────────────────────────────────────
    def _rolling_sharpe(self, agent_id: str) -> float:
        rets = [r["daily_return"] for r in self._history[agent_id][-self.window:]]
        if len(rets) < 2:
            return 0.0
        arr = np.array(rets)
        std = arr.std()
        return float((arr.mean() - self.risk_free) / std) if std > 0 else 0.0

    def _rolling_max_dd(self, agent_id: str) -> float:
        dds = [r["daily_dd"] for r in self._history[agent_id][-self.window:]]
        return max(dds) if dds else 0.0

    def summary(self) -> pd.DataFrame:
        rows = []
        for aid in self.agent_ids:
            rows.append({
                "agent_id":      aid,
                "rolling_sharpe": self._rolling_sharpe(aid),
                "rolling_max_dd": self._rolling_max_dd(aid),
                "n_days":         len(self._history[aid]),
            })
        return pd.DataFrame(rows)

    # ── capital allocation ────────────────────────────────────────────────────
    def get_allocations(self) -> Dict[str, float]:
        """
        Allocate capital proportional to rolling Sharpe (clipped at 0).
        Each agent gets at least min_weight.
        Returns {agent_id: weight} where weights sum to 1.0.
        """
        sharpes = {a: max(0.0, self._rolling_sharpe(a)) for a in self.agent_ids}
        total   = sum(sharpes.values())

        if total == 0:
            # Equal weight fallback
            w = 1.0 / len(self.agent_ids)
            return {a: w for a in self.agent_ids}

        raw = {a: sharpes[a] / total for a in self.agent_ids}
        # Apply minimum weight floor
        floored = {a: max(self.min_weight, raw[a]) for a in self.agent_ids}
        s       = sum(floored.values())
        return {a: floored[a] / s for a in self.agent_ids}

    # ── hooks for future meta-RL / bandit upgrade ─────────────────────────────
    def suggest_hyperparams(self, agent_id: str) -> dict:
        """Placeholder: returns default hyperparams. Replace with bandit/meta-RL later."""
        return {
            "LEARNING_RATE": 0.001,
            "EPSILON":       0.9,
            "BATCH_SIZE":    32,
        }

    def suggest_phase_advance(self, agent_id: str, consecutive_pass: int,
                               threshold: int = 10) -> bool:
        """Placeholder: advance phase when consecutive_pass >= threshold."""
        return consecutive_pass >= threshold
