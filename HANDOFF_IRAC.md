# deep-reinforcement-learning-trading — IRAC handoff (2026-10-17)

## Issue
GPU DQN forex trading system with MT5 backtest and live EA (predecessor of rl-trading-live).
Central question: Is the DQN agent profitable live? Unknown; work moved to rl-trading-live.

## Rule
docs/SPEC.md, docs/plan.md, CHECKPOINTS.md. Doctrine: scalping-only; 5m CCI 30/100 burst entry; 1R=0.25% equity; max 4 concurrent; daily floor -3R; pyramid winners OK, add-to-losers banned.

## Application
45 commits, last 4f1da02 2026-06-01 (EA per-symbol bars, lot margin check). 45b5260 live EA + Python agent. 33bc2af MT5 backtest. rl-trading-live/AUDIT.md (2026-06-01) names this repo's gpu_rl_trading/ as authoritative core that was ported. No README. No live results in repo.

## Conclusion
Status: abandoned (superseded). Abandoned 2026-06-01; ported to rl-trading-live.
Next: 1) Archive repo or mark as superseded.
