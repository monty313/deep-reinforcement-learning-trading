"""
monitoring/shap_explainer.py
SHAP-based action explanation for the DQN agent.
Provides explain_action(state) and chatbot-style query API.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

from env.ftmo_game import NUM_ACTIONS, FLAT, BUY_SMALL, BUY_MED, BUY_LARGE, SELL_SMALL, SELL_MED, SELL_LARGE

_ACTION_NAMES = {
    FLAT: "flat", BUY_SMALL: "buy_small", BUY_MED: "buy_med", BUY_LARGE: "buy_large",
    SELL_SMALL: "sell_small", SELL_MED: "sell_med", SELL_LARGE: "sell_large",
}


class SHAPExplainer:
    """
    Wraps a trained DQN q_network with SHAP KernelExplainer.
    Call explain_action(state) to get top-N feature contributions.
    """

    def __init__(
        self,
        q_net,
        feature_names:  List[str],
        background_data: np.ndarray,
        shap_log_dir:    str = "shap_logs",
        top_n:           int = 10,
    ):
        self.q_net         = q_net
        self.feature_names = feature_names
        self.top_n         = top_n
        self.log_dir       = Path(shap_log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._shap_history: List[dict] = []

        if not _SHAP_AVAILABLE:
            print("[SHAPExplainer] shap not installed — explanations disabled.")
            self.explainer = None
            return

        def _predict(x):
            return self.q_net.predict(x, verbose=0)

        # Use a small background sample (100 rows) for speed
        bg = background_data[:100] if len(background_data) > 100 else background_data
        self.explainer = shap.KernelExplainer(_predict, bg)

    # ── core explain ──────────────────────────────────────────────────────────
    def explain_action(
        self,
        state: np.ndarray,
        metadata: dict = None,
    ) -> dict:
        """
        Given a state vector (1 × state_dim), return:
        - chosen_action: int
        - action_name: str
        - top_features: [{name, shap_value}] sorted by |shap_value| desc
        """
        if self.explainer is None:
            return {"error": "shap not available"}

        q_vals  = self.q_net.predict(state, verbose=0)[0]
        action  = int(np.argmax(q_vals))

        shap_values = self.explainer.shap_values(state, nsamples=100)
        # shap_values shape: [num_actions, 1, state_dim]
        sv_for_action = np.array(shap_values[action]).flatten()

        top_idx = np.argsort(np.abs(sv_for_action))[::-1][:self.top_n]
        top_features = [
            {"name": self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}",
             "shap_value": float(sv_for_action[i])}
            for i in top_idx
        ]

        result = {
            "timestamp":     datetime.utcnow().isoformat(),
            "chosen_action": action,
            "action_name":   _ACTION_NAMES.get(action, str(action)),
            "q_values":      q_vals.tolist(),
            "top_features":  top_features,
            "metadata":      metadata or {},
        }
        self._shap_history.append(result)
        return result

    # ── chatbot API ───────────────────────────────────────────────────────────
    def why_did_agent_trade(self, state: np.ndarray, metadata: dict = None) -> str:
        """Human-readable explanation of the last action."""
        exp = self.explain_action(state, metadata)
        if "error" in exp:
            return exp["error"]
        lines = [f"Action: {exp['action_name']}"]
        for f in exp["top_features"][:5]:
            direction = "↑" if f["shap_value"] > 0 else "↓"
            lines.append(f"  {direction} {f['name']}: {f['shap_value']:+.4f}")
        return "\n".join(lines)

    def top_features_this_week(self, n: int = 5) -> str:
        """Aggregate SHAP values over the last 7 days of history."""
        if not self._shap_history:
            return "No SHAP history yet."
        recent = self._shap_history[-500:]
        agg: Dict[str, float] = {}
        for rec in recent:
            for f in rec.get("top_features", []):
                agg[f["name"]] = agg.get(f["name"], 0.0) + abs(f["shap_value"])
        ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:n]
        lines = ["Top features this week:"]
        for name, total in ranked:
            lines.append(f"  {name}: {total:.4f} (cumulative |SHAP|)")
        return "\n".join(lines)

    # ── periodic logging ──────────────────────────────────────────────────────
    def save_summary(self, tag: str = ""):
        path = self.log_dir / f"shap_{tag}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "w") as f:
            json.dump(self._shap_history[-200:], f, default=str)
        print(f"[SHAPExplainer] Saved summary to {path}")
