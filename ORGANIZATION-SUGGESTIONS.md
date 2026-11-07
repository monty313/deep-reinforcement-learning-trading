# ORGANIZATION-SUGGESTIONS

Repository: `monty313/deep-reinforcement-learning-trading`
Date: `2026-11-07`

## Purpose-aligned opinion
Trading intelligence and execution research platform, emphasizing strategy quality and operational safety.

## Current snapshot (for orientation)
Detected primary languages: **Python, Notebook, MQL5**

Top-level entries seen:
- `.github` (dir)
- `.gitignore` (file)
- `agents` (dir)
- `config` (dir)
- `data` (dir)
- `docs` (dir)
- `env` (dir)
- `execution` (dir)
- `gpu_rl_trading` (dir)
- `HANDOFF-2026-11-07.md` (file)
- `HANDOFF_IRAC.md` (file)
- `monitoring` (dir)
- `outputs` (dir)
- `README.md` (file)
- `requirements.txt` (file)
- `smoke_test.py` (file)
- `temp_test_runner.ipynb` (file)
- `test_run.py` (file)
- `test_synthetic_eurusd.tsv` (file)
- `tests` (dir)

## Recommended target structure
```text
deep-reinforcement-learning-trading/
  experts/             # Expert Advisors (.mq5)
  indicators/          # custom indicators (.mq5)
  include/             # shared MQL5 headers/helpers
  configs/             # symbol/timeframe/parameter presets
  tests/               # strategy tester scenarios + expected metrics
  docs/                # deployment + broker setup instructions
  scripts/             # export/import and utility tooling
```

## Why this structure fits this repo
- Separates **core logic** from **operations/configuration**, reducing rework when context is lost.
- Improves onboarding for both humans and LLMs by making intent obvious from file location.
- Makes testing and deployment more deterministic by standardizing run and validation paths.

## Priority reorganization backlog
- [ ] Lock one canonical run path (single command for local start and single command for validation).
- [ ] Move mixed experimental files into clearly named modules by responsibility.
- [ ] Centralize configuration and remove hardcoded credentials/paths from code.
- [ ] Add/refresh tests for the highest-risk logic first (signals, execution, scheduling, or model outputs).
- [ ] Document architecture decisions in `docs/` and keep them current with code changes.

## Safe migration sequence
1. Create the target folders that do not exist yet (no behavioral change).
1. Relocate files in small batches while preserving imports and entrypoints.
1. Add thin compatibility wrappers for old paths if needed, then remove after validation.
1. Run current validation after each batch and record results in docs.
1. Tag the repo state once the new structure is stable.

## LLM handoff note
When revisiting this repo later, do this first:
1. Read `README.md`, `HANDOFF-2026-11-07.md`, and this file.
2. Confirm current run command and test command still work.
3. Execute one backlog item at a time and update docs in the same commit.
