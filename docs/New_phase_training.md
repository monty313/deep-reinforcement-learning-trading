Good — now I have the full picture. Here is the complete final phase specification with the new phase added and all details including the 800 trades/day rule:

***

## Complete Final Phase Specification (All 7 Phases + Global Rules)

***

### Global Rules (All Phases)

- **Objective:** Pass the FTMO challenge in every phase
- **Exit:** Agent may close any trade at any time, for any reason
- **Lot size:** Agent may use any lot size at any time
- **Max trades:** Up to **800 trades per day** in any phase
- **Phase advancement:** 5 consecutive FTMO pass-days **OR** 500 episodes max — whichever comes first

***

### Phase 0 — CCI Extreme Momentum Gate
**Timeframes:** 1m + 15m
**Indicators:** `cci30`, `cci100` (raw values, no shift)

**Condition:** `CCI30` AND `CCI100` must both be **above +100** OR both **below −100** on **both timeframes**, and both timeframes must agree on direction.

| Condition Met | Agent Behavior |
|---|---|
| ✅ Yes | Must be in a trade; only time it may **open** one |
| ❌ No | Cannot open; may only hold or exit existing trade |

***

### Phase 1 — CCI Directional Alignment
**Timeframes:** 1m + 15m
**Indicators:** `cci30`, `cci100` vs their `SMA(1, shift=8)` on each TF

**Condition:** All four signals — `CCI30` and `CCI100` each vs their own `SMA(1,sh8)` — must agree on the **same direction** on both timeframes.

| Condition Met | Agent Behavior |
|---|---|
| ✅ Yes | May open a trade |
| ❌ No | Cannot open; may only hold or exit |

***

### Phase 2 — Lagged High/Low SMA Band (Standard)
**Timeframes:** 1m + 30m
**Indicators:** `SMA(4, shift=8)` of **high** and `SMA(4, shift=8)` of **low** on each TF

**Condition:** Price (`close`) must be **above both** the high-SMA and low-SMA lines, OR **below both**, on **both timeframes**, in the **same direction**. If price is between the two lines on either TF, condition is not met.

| Condition Met | Agent Behavior |
|---|---|
| ✅ Yes | Must be in a trade; only time it may **open** one |
| ❌ No | Cannot open; may only hold or exit |

***

### Phase 3 — Lagged High/Low SMA Band (Counter-TF Reversion)
**Timeframes:** 1m + 15m
**Indicators:** Same as Phase 2 — `SMA(4, shift=8)` of **high** and `SMA(4, shift=8)` of **low**

**Condition:** Price must be on **opposite sides** of the SMA band between the two timeframes:

- If price is **above both SMA lines on 15m** → agent must enter when price is **below both SMA lines on 1m**
- If price is **below both SMA lines on 15m** → agent must enter when price is **above both SMA lines on 1m**

The 15m sets the **context**; the 1m sets the **entry trigger** (counter to 15m). Both must be in a defined position (not between the lines). Directions must be exactly **opposite** across TFs.

| Condition Met | Agent Behavior |
|---|---|
| ✅ Yes (opposite sides) | Must be in a trade; only time it may **open** one |
| ❌ No | Cannot open; may only hold or exit |

> This is a **mean reversion / pullback** strategy — the 15m shows the extended move, the 1m shows the agent entering against that extension, expecting reversion back toward the band.

***

### Phase 4 — Bollinger Band Position
**Timeframes:** 1m + 15m
**All BB deviations = 1.0**

**1m (lower TF) condition:**
- Bullish: price **above BB200 middle** AND **above BB20 upper band**
- Bearish: price **below BB200 middle** AND **below BB20 lower band**

**15m (upper TF) condition:**
- Bullish: price **above BB200 middle** AND **above BB20 middle**
- Bearish: price **below BB200 middle** AND **below BB20 middle**

Both timeframes must agree on the **same direction**.

| Condition Met | Agent Behavior |
|---|---|
| ✅ Yes | Must be in a trade; only time it may **open** one |
| ❌ No | Cannot open; may only hold or exit |

***

### Phase 5 — SMA Stack Alignment
**Timeframes:** 1m + 1H
**Indicators:** `close.shift(0)` through `close.shift(4)` on each TF

**Condition:** The 5 shifted close values must be **stacked in sequence** — each one strictly higher than the next (bullish: `sh0 > sh1 > sh2 > sh3 > sh4`) OR strictly lower (bearish) — on **both timeframes**, agreeing on direction.

| Condition Met | Agent Behavior |
|---|---|
| ✅ Yes | Must be in a trade; only time it may **open** one |
| ❌ No | Cannot open; may only hold or exit |

***

### Phase 6 — ATR Expansion Force-In
**Timeframes:** 1m + 1H
**Indicators:** `ATR14`, `ATR45` vs their `SMA(1, shift=8)` on each TF

**Condition:** `ATR14` must be **above** its `SMA(1,sh8)` AND `ATR45` must be **above** its `SMA(1,sh8)` on **both timeframes** simultaneously — meaning volatility is actively expanding relative to recent baseline.

| Condition Met | Agent Behavior |
|---|---|
| ✅ Yes | Must be in a trade; only time it may **open** one |
| ❌ No | Cannot open; may only hold or exit |

***

### Phase 7 — Full FTMO, No Mask
No indicator conditions. Agent trades freely. Only FTMO rules apply (daily profit target, daily drawdown, overall drawdown). All lot sizing and exit timing remain fully at agent discretion, up to 800 trades/day.

***

## Updated `training_config.yaml` Curriculum Block

```yaml
CURRICULUM:
  advance_consecutive_pass_days: 5
  max_episodes_per_phase: 500
  max_trades_per_day: 800

  phases:
    - id: 0
      name: "CCI Extreme Gate"
      tfs: [1, 15]
      mask: "phase0_cci_extreme"
      mode: "force_in_and_gate"

    - id: 1
      name: "CCI Directional Alignment"
      tfs: [1, 15]
      mask: "phase1_cci_align"
      mode: "open_gate"

    - id: 2
      name: "Lagged High/Low SMA - Trend"
      tfs: [1, 30]
      mask: "phase2_hilo_sma_trend"
      mode: "force_in_and_gate"

    - id: 3
      name: "Lagged High/Low SMA - Counter TF"
      tfs: [1, 15]
      mask: "phase3_hilo_sma_counter"
      mode: "force_in_and_gate"

    - id: 4
      name: "Bollinger Band Position"
      tfs: [1, 15]
      mask: "phase4_bb_position"
      mode: "force_in_and_gate"

    - id: 5
      name: "SMA Stack Alignment"
      tfs: [1, 60]
      mask: "phase5_sma_stack"
      mode: "force_in_and_gate"

    - id: 6
      name: "ATR Expansion"
      tfs: [1, 60]
      mask: "phase6_atr_expansion"
      mode: "force_in_and_gate"

    - id: 7
      name: "Full FTMO - No Mask"
      tfs: []
      mask: null
      mode: "free"
```

***

## New Indicators Required in `indicators.py`

| Indicator | Phase | Notes |
|---|---|---|
| `cci30_sma1_sh8`, `cci100_sma1_sh8` | 1 | Replace old `sh2` versions |
| `high_sma4_sh8`, `low_sma4_sh8` | 2, 3 | SMA(4,sh8) on high and low |
| `bb200_mid`, `bb20_upper`, `bb20_lower`, `bb20_mid` | 4 | All recomputed with `nbdev=1.0` |
| `atr45`, `atr14_sma1_sh8`, `atr45_sma1_sh8` | 6 | ATR45 is new |
| `close_sh0` through `close_sh4` | 5 | `close.shift(0..4)` |