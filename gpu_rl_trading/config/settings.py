"""Central config — edit this file or override in notebook."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # gpu_rl_trading/

CFG = {
    # Data
    "DATA_CSV_EURUSD": None,   # set in notebook: e.g. "/content/drive/MyDrive/EURUSD_M1.csv"
    "SYMBOL":          "EURUSD",
    "DATE_FROM":       "2021-01-01",
    "DATE_TO":         "2024-06-30",

    # Episode
    "EPISODE_BARS":    43_200,    # ~30 trading days of 1m bars
    "BATCH_SIZE_ENV":  8,         # parallel episodes on GPU
    "LOOKBACK":        20,        # bars of history in state

    # Timeframes (resample factors from 1m)
    "TF_FACTORS": [1, 15, 60, 1440],

    # FTMO
    "DAILY_TARGET_PCT":   0.025,
    "DAILY_MAX_DD_PCT":   0.010,
    "INITIAL_EQUITY":     100_000.0,

    # Agent / training
    "STATE_DIM":       None,      # filled at runtime
    "NUM_ACTIONS":     7,
    "HIDDEN":          256,
    "LR":              5e-4,
    "GAMMA":           0.95,
    "EPSILON_START":   0.9,
    "EPSILON_MIN":     0.05,
    "BATCH_SIZE_RL":   256,
    "MEMORY_SIZE":     100_000,
    "TRAIN_EVERY":     4,
    "SYNC_EVERY":      200,       # steps between target-net sync
    "PHASE":                  0,      # starting phase (0-7)
    "ADVANCE_DAYS":           10,     # consecutive PASS days to advance phase
    "MAX_EPISODES_PER_PHASE": 500,    # hard cap per phase
    "NUM_EPISODES":           200,    # legacy fallback (ignored in full curriculum)
    "CHECKPOINT_EVERY": 10,

    # Paths (relative to gpu_rl_trading/)
    "CHECKPOINT_DIR":  str(ROOT / "checkpoints"),
    "METRICS_DIR":     str(ROOT / "metrics"),
}
