# GPU RL Trading — TODO

## Pending

- [ ] **Parameterizable targets** — feed `daily_target_pct` and `max_dd_pct` as explicit state features, randomize them each episode during training (e.g. target 1–5%, dd 0.5–2%) so the policy generalizes to any target/risk combination set at inference time

---

## Completed

- [x] Dynamic lot sizing based on ATR and FTMO gap
- [x] Phase masks matching CPU curriculum (phases 0–7)
- [x] Correct PnL formula (price_diff * lots * 100,000)
- [x] FTMO 1:100 leverage lot ceiling
- [x] Equity accumulation fix (realised + unrealised)
- [x] FTMO state features in observation (gap_to_tgt, dd_headroom, daily_ret)
- [x] Checkpoint save/resume to Google Drive
- [x] Crash recovery guide (CRASH_RECOVERY.md)
