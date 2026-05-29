import json
from pathlib import Path

root = Path('.')
old = 'reward_exponential_pnl'
for p in sorted(root.rglob('*.ipynb')):
    if '.ipynb_checkpoints' in p.parts:
        continue
    txt = p.read_text(encoding='utf-8')
    count_old = txt.count(old)
    if count_old == 0:
        continue
    has_ftmo = 'FTMO_COST' in txt
    has_new = 'reward_ftmo_pnl' in txt
    print(f'{p} | old={count_old} ftmo={has_ftmo} new={has_new}')
