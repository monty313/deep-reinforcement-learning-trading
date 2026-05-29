import json
from pathlib import Path

root = Path('.')
ignore_parts = {'.ipynb_checkpoints'}
patterns = [
    'from data_modules.quantra_reinforcement_learning import reward_exponential_pnl',
    "'RF': reward_exponential_pnl",
    'env = Game(bars5m, bars1d, bars1h, reward_exponential_pnl',
]
replacement_map = {
    'from data_modules.quantra_reinforcement_learning import reward_exponential_pnl':
        'from data_modules.quantra_reinforcement_learning import reward_ftmo_pnl',
    "'RF': reward_exponential_pnl":
        "'RF': reward_ftmo_pnl",
}

ftmo_insert = [
    "    'FTMO_COST': 0.001,\n",
    "    'FTMO_LOT': 1.0,\n",
    "    'MAX_DRAWNDOWN': 0.01,\n",
    "    'MAX_LOSS_PER_TRADE': 0.01,\n",
    "    'MAX_TRADE_LEN': 60,\n",
]

summary = []
for p in sorted(root.rglob('*.ipynb')):
    if any(part in ignore_parts for part in p.parts):
        continue
    nb = json.loads(p.read_text(encoding='utf-8'))
    modified = False
    found = {pat: 0 for pat in patterns}
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell['source'])
        for pat in patterns:
            found[pat] += src.count(pat)
        new = src
        for old, new_text in replacement_map.items():
            new = new.replace(old, new_text)
        if 'env = Game(bars5m, bars1d, bars1h, reward_exponential_pnl' in new:
            new = new.replace(
                'env = Game(bars5m, bars1d, bars1h, reward_exponential_pnl,\n'
                '           lkbk=LKBK, init_idx=START_IDX)',
                'env = Game(bars5m, bars1d, bars1h, reward_ftmo_pnl,\n'
                '           lkbk=LKBK, init_idx=START_IDX,\n'
                '           ftmo_cost=rl_config.get(\'FTMO_COST\', 0.001),\n'
                '           ftmo_lot=rl_config.get(\'FTMO_LOT\', 1.0),\n'
                '           max_drawdown=rl_config.get(\'MAX_DRAWNDOWN\', None),\n'
                '           max_loss_per_trade=rl_config.get(\'MAX_LOSS_PER_TRADE\', None),\n'
                '           max_trade_len=rl_config.get(\'MAX_TRADE_LEN\', None))'
            )
        if "env = Game(bars5m, bars1d, bars1h, rl_config['RF'],\n               lkbk=rl_config['LKBK'], init_idx=rl_config['START_IDX'])" in new:
            new = new.replace(
                "env = Game(bars5m, bars1d, bars1h, rl_config['RF'],\n               lkbk=rl_config['LKBK'], init_idx=rl_config['START_IDX'])",
                "env = Game(bars5m, bars1d, bars1h, rl_config['RF'],\n               lkbk=rl_config['LKBK'], init_idx=rl_config['START_IDX'],\n               ftmo_cost=rl_config.get('FTMO_COST', 0.001),\n               ftmo_lot=rl_config.get('FTMO_LOT', 1.0),\n               max_drawdown=rl_config.get('MAX_DRAWNDOWN', None),\n               max_loss_per_trade=rl_config.get('MAX_LOSS_PER_TRADE', None),\n               max_trade_len=rl_config.get('MAX_TRADE_LEN', None))"
            )
        if new != src:
            cell['source'] = new.splitlines(True)
            modified = True
    full = json.dumps(nb)
    if 'FTMO_COST' not in full:
        for cell in nb['cells']:
            if cell.get('cell_type') != 'code':
                continue
            src = ''.join(cell['source'])
            if 'rl_config' in src and "'START_IDX'" in src and '{' in src and ':' in src:
                lines = src.splitlines(True)
                for i, line in enumerate(lines):
                    if "'START_IDX'" in line and 'FTMO_COST' not in src:
                        lines[i+1:i+1] = ftmo_insert
                        cell['source'] = lines
                        modified = True
                        break
                if modified:
                    break
    summary.append((p, found, modified))
    if modified:
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')

print('Audit and patch completed')
for p, found, modified in summary:
    totals = sum(found.values())
    if totals or modified:
        print(f'{p}: old_count={totals} modified={modified} ' + ' '.join(f"{k}={v}" for k, v in found.items()))
