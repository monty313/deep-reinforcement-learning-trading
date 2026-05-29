# Checkpoint Log

Each checkpoint is a git tag. To rewind to any checkpoint:

```
git checkout <tag-name>          # inspect (detached HEAD)
git checkout -B main <tag-name>  # HARD RESET main to that checkpoint
```

---

## checkpoint/initial-clean
**Commit:** 9ca6fa7  
**When:** First clean push — full FTMO RL system, no large CSV files tracked.  
**What's in it:** All core modules (env, agents, training, monitoring, data), config, test_run.py, train.py. No progress logging yet.  
**Rewind to this if:** Any change after this point breaks the import chain or the smoke test.

---

## checkpoint/after-tqdm-progress-logging
**Commit:** f163fb1  
**When:** After adding tqdm + timing logs. No model logic changed.  
**What changed:**
- `requirements.txt` — added `tqdm>=4.66`
- `data/loader.py` — chunked reader shows tqdm bar per symbol; per-TF resample timing
- `env/data_bridge.py` — `[START]/[DONE]` banners + tqdm bar across all symbol×timeframe frames
- `training/curriculum_trainer.py` — episode tqdm bar with live postfix; phase start/done banners; forward_test step bar + daily CSV output fixed
- `train.py` — 8-stage `[START]/[DONE]` banners; curriculum phase tqdm bar; total wall-clock timer  
**Rewind to this if:** A future change breaks training logic and you want to keep the progress logging but undo everything after.

---

## Rules for future checkpoints
1. Create a tag **before** starting each edit batch: `git tag -a "checkpoint/before-<what>" -m "<note>"`
2. Commit the edit batch, then tag it: `git tag -a "checkpoint/after-<what>" -m "<note>"`
3. Tag names follow the pattern: `checkpoint/before-<stage>` and `checkpoint/after-<stage>`
4. Stages map to observable pipeline steps:
   - `csv-loading`
   - `feature-building`
   - `phase-1-training`
   - `phase-2-training`
   - `phase-3-training`
   - `phase-4-training`
   - `forward-test`
   - `validation`
   - `metrics-reporting`
5. Never tag a broken state — only tag after verifying the file parses cleanly.
6. If a change breaks the run, report: **"Rewind to checkpoint/after-tqdm-progress-logging"** (or whichever is the last known-good tag).
