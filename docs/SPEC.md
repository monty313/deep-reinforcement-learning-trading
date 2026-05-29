# SPEC: Self-Healing FTMO RL Trading System

This document is the authoritative source of truth for implementation.
See `copilots instructions for rl model.pdf` for the full design brief.

## Quick-start

```bash
# Install dependencies
pip install -r requirements.txt

# Edit MT5 credentials
# config/ftmo_config.yaml → MT5.login / password / server

# Run full 4-phase curriculum
python -c "
from execution.mt5_bridge import MT5Bridge
from training.curriculum_trainer import run_curriculum
from datetime import datetime

bridge = MT5Bridge(login=YOUR_LOGIN, password='YOUR_PASS', server='YOUR_SERVER')
bridge.connect()
data = bridge.load_all(date_from=datetime(2018,1,1), date_to=datetime(2022,12,31))
bridge.disconnect()

agent = run_curriculum('config/ftmo_config.yaml', data)
"

# Run dashboard
streamlit run monitoring/dashboard_app.py
```

## Package structure

```
config/
    ftmo_config.yaml          ← master config (FTMO profile, MT5 symbols, curriculum)
env/
    indicators.py             ← STRAT-001..011 + extra indicators + phase masks
    ftmo_game.py              ← Multi-asset FTMO Game (4-phase curriculum)
agents/
    dqn_agent.py              ← DQN + ExperienceReplay (7-action multi-asset)
    meta_learner.py           ← Placeholder meta-learner + capital allocator
training/
    curriculum_trainer.py     ← 4-phase runner + transfer learning + Ray parallel
monitoring/
    wandb_logger.py           ← W&B logging hooks
    shap_explainer.py         ← SHAP explain_action() + chatbot API
    dashboard_app.py          ← Streamlit FTMO dashboard
execution/
    mt5_bridge.py             ← MT5 connect, load candles (CET), send orders
```

## FTMO profile (default)
- Daily profit target: **+2.5%** of start-of-day equity
- Trailing daily max drawdown: **−1%** from intraday equity high (including open trades)
- Day classified as pass / ok / fail at CET midnight

## Curriculum phases
| Phase | Condition | Advance trigger |
|-------|-----------|-----------------|
| 1 | CCI alignment 1m + 15m | 10 consecutive pass-days |
| 2 | SMA on BB upper band 1m + 1H | 10 consecutive pass-days |
| 3 | BB midlines 1m + 15m | 10 consecutive pass-days |
| 4 | No mask — full FTMO | Production |

## Action space (per asset per 1m bar)
`flat | buy_small | buy_med | buy_large | sell_small | sell_med | sell_large`
Risk fractions (small/med/large) are learnable parameters, not hard-coded.

## Data ranges
- Training:     2018-01-01 → 2022-12-31
- Validation:   2023-01-01 → 2023-12-31
- Forward test: 2024-01-01 → present
