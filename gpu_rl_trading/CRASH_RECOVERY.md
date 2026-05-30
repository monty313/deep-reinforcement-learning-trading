# GPU Training — Crash Recovery Guide

## The Problem

Colab's `/content/` directory is wiped every time the runtime disconnects or crashes.
Default checkpoints save to `/content/.../checkpoints/` — **they will be lost.**

**Always save checkpoints to Google Drive.**

---

## Before You Start Training (Do This Once)

Add `CHECKPOINT_DIR` and `METRICS_DIR` to your training call so everything persists:

```python
from google.colab import drive
drive.mount("/content/drive")

import os, sys
REPO = "/content/deep-reinforcement-learning-trading"
if not os.path.exists(REPO):
    !git clone https://github.com/monty313/deep-reinforcement-learning-trading.git {REPO}
else:
    !cd {REPO} && git pull
sys.path.insert(0, REPO)

DATA_CSV = "/content/drive/MyDrive/RL-Trading-Data/EURUSD_M1_202101131130_202605270000_2020_2026.csv"

from gpu_rl_trading.training.train import run_training

agent = run_training(cfg={
    "DATA_CSV_EURUSD":  DATA_CSV,
    "PHASE":            0,
    "NUM_EPISODES":     200,
    "BATCH_SIZE_ENV":   8,
    "BATCH_SIZE_RL":    256,
    "CHECKPOINT_EVERY": 10,
    "CHECKPOINT_DIR":   "/content/drive/MyDrive/RL-Trading-Checkpoints/gpu",    # ← Drive
    "METRICS_DIR":      "/content/drive/MyDrive/RL-Trading-Checkpoints/metrics", # ← Drive
}, resume=False)
```

---

## After a Crash — Step by Step

### 1. Hard-reset the runtime
**Runtime → Disconnect and delete runtime**
Always do this even if the notebook looks connected. A crashed runtime leaves stale state.

### 2. Re-run the setup cell
```python
from google.colab import drive
drive.mount("/content/drive")

import os, sys
REPO = "/content/deep-reinforcement-learning-trading"
if not os.path.exists(REPO):
    !git clone https://github.com/monty313/deep-reinforcement-learning-trading.git {REPO}
else:
    !cd {REPO} && git pull
sys.path.insert(0, REPO)

DATA_CSV = "/content/drive/MyDrive/RL-Trading-Data/EURUSD_M1_202101131130_202605270000_2020_2026.csv"
```

### 3. Resume training — only change is `resume=True`
```python
from gpu_rl_trading.training.train import run_training

agent = run_training(cfg={
    "DATA_CSV_EURUSD":  DATA_CSV,
    "PHASE":            0,
    "NUM_EPISODES":     200,
    "BATCH_SIZE_ENV":   8,
    "BATCH_SIZE_RL":    256,
    "CHECKPOINT_EVERY": 10,
    "CHECKPOINT_DIR":   "/content/drive/MyDrive/RL-Trading-Checkpoints/gpu",
    "METRICS_DIR":      "/content/drive/MyDrive/RL-Trading-Checkpoints/metrics",
}, resume=True)   # ← the only change
```

### 4. Confirm it resumed (not restarted)

**Success — resumed from checkpoint:**
```
[ckpt] loaded <- /content/drive/MyDrive/RL-Trading-Checkpoints/gpu/eurusd_gpu_ep0050.pt
[train] Starting episode 50 — 200 total
```

**Problem — started fresh (no checkpoint found):**
```
[resume] No checkpoint found — starting fresh.
[train] Starting episode 0 — 200 total
```
If you see this, check that `CHECKPOINT_DIR` points to the correct Drive folder.

---

## Check What Checkpoints Exist

Run this before resuming to see what's available:

```python
from pathlib import Path

ckpt_dir = Path("/content/drive/MyDrive/RL-Trading-Checkpoints/gpu")
files = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
if files:
    for f in files:
        print(f.name)
    print(f"\nWill resume from: {files[-1].name}")
else:
    print("No checkpoints found.")
```

---

## Quick Reference

| Situation | Action |
|---|---|
| Starting fresh | `resume=False`, checkpoints → Drive |
| Colab crashed | Hard-reset → re-run setup → `resume=True` |
| Not sure if resumed | Look for `[ckpt] loaded <-` in output |
| Checkpoints missing | They were in `/content/` — must start over, use Drive next time |
| Want to start a new run | Change `resume=False` and optionally change `CHECKPOINT_DIR` to a new folder |
