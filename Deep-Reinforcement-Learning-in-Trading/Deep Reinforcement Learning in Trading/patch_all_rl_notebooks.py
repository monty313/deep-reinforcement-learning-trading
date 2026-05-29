import json
from pathlib import Path

root = Path('.')
notebooks = list(root.rglob('*.ipynb'))
print('scanning', len(notebooks), 'notebooks')
ftmo_insert = [
    "    'FTMO_COST': 0.001,\n",
    "    'FTMO_LOT': 1.0,\n",
    "    'MAX_DRAWNDOWN': 0.01,\n",
    "    'MAX_LOSS_PER_TRADE': 0.01,\n",
    "    'MAX_TRADE_LEN': 60,\n",
]
for p in notebooks:
    if '.ipynb_checkpoints' in str(p):
        continue
    nb = json.loads(p.read_text(encoding='utf-8'))
    modified = False
    full_text = json.dumps(nb)
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell['source'])
        new = src
        replaced = False
        if 'from data_modules.quantra_reinforcement_learning import reward_exponential_pnl' in new:
            new = new.replace('from data_modules.quantra_reinforcement_learning import reward_exponential_pnl',
                              'from data_modules.quantra_reinforcement_learning import reward_ftmo_pnl')
            replaced = True
        if "'RF': reward_exponential_pnl" in new:
            new = new.replace("'RF': reward_exponential_pnl", "'RF': reward_ftmo_pnl")
            replaced = True
        if 'reward_exponential_pnl' in new and 'from data_modules.quantra_reinforcement_learning import reward_exponential_pnl' in new:
            replaced = True
        if "env = Game(bars5m, bars1d, bars1h, reward_exponential_pnl," in new:
            replacement = (
                "env = Game(bars5m, bars1d, bars1h, reward_ftmo_pnl,\n"
                "           lkbk=LKBK, init_idx=START_IDX,\n"
                "           ftmo_cost=rl_config.get('FTMO_COST', 0.001),\n"
                "           ftmo_lot=rl_config.get('FTMO_LOT', 1.0),\n"
                "           max_drawdown=rl_config.get('MAX_DRAWNDOWN', None),\n"
                "           max_loss_per_trade=rl_config.get('MAX_LOSS_PER_TRADE', None),\n"
                "           max_trade_len=rl_config.get('MAX_TRADE_LEN', None))"
            )
            new = new.replace("env = Game(bars5m, bars1d, bars1h, reward_exponential_pnl,\n           lkbk=LKBK, init_idx=START_IDX)", replacement)
            replaced = True
        if "env = Game(bars5m, bars1d, bars1h, rl_config['RF'],\n               lkbk=rl_config['LKBK'], init_idx=rl_config['START_IDX'])" in new:
            replacement = (
                "env = Game(bars5m, bars1d, bars1h, rl_config['RF'],\n"
                "               lkbk=rl_config['LKBK'], init_idx=rl_config['START_IDX'],\n"
                "               ftmo_cost=rl_config.get('FTMO_COST', 0.001),\n"
                "               ftmo_lot=rl_config.get('FTMO_LOT', 1.0),\n"
                "               max_drawdown=rl_config.get('MAX_DRAWNDOWN', None),\n"
                "               max_loss_per_trade=rl_config.get('MAX_LOSS_PER_TRADE', None),\n"
                "               max_trade_len=rl_config.get('MAX_TRADE_LEN', None))"
            )
            new = new.replace("env = Game(bars5m, bars1d, bars1h, rl_config['RF'],\n               lkbk=rl_config['LKBK'], init_idx=rl_config['START_IDX'])", replacement)
            replaced = True
        if replaced and new != src:
            cell['source'] = new.splitlines(True)
            modified = True
    if 'FTMO_COST' not in full_text:
        for cell in nb['cells']:
            if cell.get('cell_type') != 'code':
                continue
            src = ''.join(cell['source'])
            if 'rl_config' in src and "'START_IDX'" in src and '"RF"' not in src:
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
        print('patched', p)
print('done')
