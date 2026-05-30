# GPU RL Trading — TODO

## Pending

- [x] **Phase advancement logic** — GPU training loop currently runs all `NUM_EPISODES` on one phase regardless of performance. Add consecutive PASS-day tracking per batch item: when any batch item achieves 5 consecutive PASS days, advance to the next phase (same logic as CPU `curriculum_trainer.py`). Also add multi-phase loop so training automatically progresses from phase 0 → 7 without manual intervention.

- [ ] **Learned lot sizing (adaptive position sizing network)** — replace the current ATR-based `_dynamic_lots` formula with a small neural net that takes equity, gap-to-target, dd-headroom, and ATR as inputs and outputs lot size directly. Trained alongside the DQN. Directly fixes the catastrophic `-80%` days caused by oversized positions.

- [ ] **Smarter exploration (replace epsilon-greedy)** — replace fixed epsilon decay with a learned exploration strategy that prioritizes uncertain or high-value states. Agent explores directions most likely to find PASS days instead of pure random noise. Faster learning, fewer wasted episodes.

- [ ] **Trade prediction auxiliary model** — small separate model that predicts "will this trade be profitable in the next N bars?" using indicator features. Feed prediction as an extra state feature so the agent has a directional head start rather than learning everything from reward alone.

- [ ] **Market regime detection** — classifier that labels each bar as trending, ranging, or volatile based on indicators (ATR, CCI, BB width). Feed regime label as a state feature so the agent learns different strategies per market condition instead of one policy for everything.

- [ ] **LLM-assisted hyperparameter tuning** — after each phase completes, send the FTMO metrics CSV to Claude/GPT API and receive suggested hyperparameter adjustments (LR, batch size, episode length, lot sizing params). Removes guesswork from between-phase tuning.

- [ ] **Parameterizable targets** — feed `daily_target_pct` and `max_dd_pct` as explicit state features, randomize them each episode during training (e.g. target 1–5%, dd 0.5–2%) so the policy generalizes to any target/risk combination set at inference time

---

## Completed

- [x] **Potential-based reward shaping (Φ)** — replaced 10 ad-hoc reward signals with single Φ = (pass_rate × avg_ret_norm) / (1 + λ × avg_dd_norm), normalised to configured targets so shaping generalizes across any target/risk combination without retuning. Warm-up gate (50 eps), normalized by running σ, clipped to ±0.03. Primary PASS/OK/FAIL rewards unchanged and dominant.

- [x] **Transfer learning compatibility** — checkpoints store `state_dim`; `load(partial=True)` preserves old feature weights and zero-inits new columns; `train.py` auto-detects when partial load is needed on resume

- [x] Dynamic lot sizing based on ATR and FTMO gap
- [x] Phase masks matching CPU curriculum (phases 0–7)
- [x] Correct PnL formula (price_diff * lots * 100,000)
- [x] FTMO 1:100 leverage lot ceiling
- [x] Equity accumulation fix (realised + unrealised)
- [x] FTMO state features in observation (gap_to_tgt, dd_headroom, daily_ret)
- [x] Checkpoint save/resume to Google Drive
- [x] Crash recovery guide (CRASH_RECOVERY.md)
