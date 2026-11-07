# deep-reinforcement-learning-trading

## Project mission
Develop and operate a trading/market intelligence system focused on deep reinforcement learning trading.

## What this repository currently contains
Top-level structure:
- `.github` (dir)
- `.gitignore` (file)
- `agents` (dir)
- `config` (dir)
- `data` (dir)
- `docs` (dir)
- `env` (dir)
- `execution` (dir)
- `gpu_rl_trading` (dir)
- `HANDOFF_IRAC.md` (file)
- `monitoring` (dir)
- `outputs` (dir)
- `requirements.txt` (file)
- `smoke_test.py` (file)
- `temp_test_runner.ipynb` (file)
- `test_run.py` (file)
- `test_synthetic_eurusd.tsv` (file)
- `tests` (dir)
- `train.py` (file)
- `training` (dir)

Primary languages/tooling detected: Python (59), Markdown (7), YAML (3), Jupyter Notebook (2), MQL5 (1)

Likely key implementation files:
- (no key source files auto-detected)

## Quickstart (operator + LLM friendly)
Use this sequence first, then adapt to repo-specific details.

1. Clone and enter repo:
   ```bash
   git clone https://github.com/monty313/deep-reinforcement-learning-trading.git
   cd deep-reinforcement-learning-trading
   ```
2. Setup environment/dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run locally:
   ```bash
   python main.py  # adjust to project entrypoint
   ```
4. Validate changes:
   ```bash
   pytest -q
   ```

## Operating intent (CEO-mode guide for future LLMs)
When continuing this project, optimize for business outcome first:

- Preserve or improve strategy quality, reliability, and reproducibility.
- Prefer small, testable, reversible changes over broad rewrites.
- Keep docs synchronized with code behavior.
- Add explicit assumptions, constraints, and success criteria in every major change.

## Delivery checklist
- [ ] Environment setup is deterministic.
- [ ] Primary run path is documented and works.
- [ ] Validation/test path is documented and works.
- [ ] Key risks and TODOs are tracked in the handoff file.

## Branch and release notes
- Default branch: `main`
- Keep PRs focused and attach proof-of-run or proof-of-test for behavior changes.
