"""
monitoring/wandb_logger.py
Weights & Biases logging hooks for FTMO RL training.
"""

from __future__ import annotations

from typing import Optional

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class WandbLogger:
    """
    Thin wrapper around wandb for FTMO RL metrics.
    Falls back to no-op if wandb is not installed.
    """

    def __init__(self, cfg: dict, run_id: str = None, project: str = "ftmo-rl"):
        self.enabled = _WANDB_AVAILABLE
        if not self.enabled:
            print("[WandbLogger] wandb not installed — logging disabled.")
            return

        rl_cfg = cfg.get("RL", {})
        wandb.init(
            project = project,
            name    = run_id,
            config  = {
                "TRADING_MODE":            cfg.get("TRADING_MODE"),
                "daily_profit_target_pct": cfg["FTMO"]["daily_profit_target_pct"],
                "daily_max_drawdown_pct":  cfg["FTMO"]["daily_max_drawdown_pct"],
                "symbols":                 cfg["MT5"]["symbols"],
                "LEARNING_RATE":           rl_cfg.get("LEARNING_RATE"),
                "DISCOUNT_RATE":           rl_cfg.get("DISCOUNT_RATE"),
                "BATCH_SIZE":              rl_cfg.get("BATCH_SIZE"),
                "MAX_MEM":                 rl_cfg.get("MAX_MEM"),
                "LKBK":                    rl_cfg.get("LKBK"),
            },
            reinit=True,
        )

    def log(self, metrics: dict, step: Optional[int] = None):
        if not self.enabled:
            return
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def log_model(self, path: str, name: str = "model"):
        if not self.enabled:
            return
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def log_shap_summary(self, shap_dict: dict, step: Optional[int] = None):
        """Log top-N SHAP feature importances."""
        if not self.enabled:
            return
        self.log({f"shap/{k}": v for k, v in shap_dict.items()}, step=step)

    def finish(self):
        if self.enabled:
            wandb.finish()
