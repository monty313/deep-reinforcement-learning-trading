import json
from pathlib import Path

root = Path('.')
old_name = 'reward_exponential_pnl'
new_name = 'reward_ftmo_pnl'
old_import = 'from data_modules.quantra_reinforcement_learning import reward_exponential_pnl'
new_import = 'from data_modules.quantra_reinforcement_learning import reward_ftmo_pnl'
old_rf = "'RF': reward_exponential_pnl"
new_rf = "'RF': reward_ftmo_pnl"

ftmo_insert = [
    "    'FTMO_COST': 0.001,\n",
    "    'FTMO_LOT': 1.0,\n",
    "    'MAX_DRAWDOWN': 0.01,\n",
    "    'MAX_LOSS_PER_TRADE': 0.01,\n",
    "    'MAX_TRADE_LEN': 60,\n",
]

patched_files = []
for p in sorted(root.rglob('*.ipynb')):
    if '.ipynb_checkpoints' in p.parts:
        continue
    nb = json.loads(p.read_text(encoding='utf-8'))
    modified = False
    full_text = json.dumps(nb)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        new = src
        new = new.replace(old_import, new_import)
        new = new.replace(old_rf, new_rf)
        new = new.replace(old_name, new_name)

        if "env = Game(bars5m, bars1d, bars1h, reward_ftmo_pnl," in new:
            pass
        elif "env = Game(bars5m, bars1d, bars1h, reward_exponential_pnl," in new:
            new = new.replace(
                "env = Game(bars5m, bars1d, bars1h, reward_exponential_pnl,\n",
                "env = Game(bars5m, bars1d, bars1h, reward_ftmo_pnl,\n"
            )
        if "env = Game(bars5m, bars1d, bars1h, rl_config['RF']," in new and "ftmo_cost=" not in new:
            new = new.replace(
                "env = Game(bars5m, bars1d, bars1h, rl_config['RF'],\n               lkbk=rl_config['LKBK'], init_idx=rl_config['START_IDX'])",
                "env = Game(bars5m, bars1d, bars1h, rl_config['RF'],\n               lkbk=rl_config['LKBK'], init_idx=rl_config['START_IDX'],\n               ftmo_cost=rl_config.get('FTMO_COST', 0.001),\n               ftmo_lot=rl_config.get('FTMO_LOT', 1.0),\n               max_drawdown=rl_config.get('MAX_DRAWDOWN', None),\n               max_loss_per_trade=rl_config.get('MAX_LOSS_PER_TRADE', None),\n               max_trade_len=rl_config.get('MAX_TRADE_LEN', None))"
            )

        if new != src:
            cell['source'] = new.splitlines(True)
            modified = True

    full_text = json.dumps(nb)
    if 'FTMO_COST' not in full_text:
        for cell in nb.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            src = ''.join(cell.get('source', []))
            if 'rl_config' in src and "'START_IDX'" in src and 'FTMO_COST' not in src:
                lines = src.splitlines(True)
                for i, line in enumerate(lines):
                    if "'START_IDX'" in line and 'FTMO_COST' not in src:
                        lines[i+1:i+1] = ftmo_insert
                        cell['source'] = lines
                        modified = True
                        break
                if modified:
                    break

    if modified:
        p.write_text(json.dumps(nb, indent=1), encoding='utf-8')
        patched_files.append(str(p))
        print('patched', p)

print('done', len(patched_files), 'files patched')
