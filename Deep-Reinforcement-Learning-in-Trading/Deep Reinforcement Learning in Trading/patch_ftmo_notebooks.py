import json
from pathlib import Path

base = Path('.')
files = [
    Path('Performance Analysis_ Real World Price Data/RL Model on Real World Price Data.ipynb'),
    Path('Performance Analysis_ Synthetic Data/Apply RL on Synthetic Mixed Wave Pattern.ipynb'),
    Path('Backtesting Implementation/Backtesting Implementation.ipynb'),
    Path('Artificial Neural Network Implementation/ANN in Keras.ipynb'),
]

ftmo_insert = [
    "    'FTMO_COST': 0.001,\n",
    "    'FTMO_LOT': 1.0,\n",
    "    'MAX_DRAWDOWN': 0.01,\n",
    "    'MAX_LOSS_PER_TRADE': 0.01,\n",
    "    'MAX_TRADE_LEN': 60,\n",
]

for p in files:
    path = base / p
    if not path.exists():
        print('missing', path)
        continue
    nb = json.loads(path.read_text(encoding='utf-8'))
    modified = False
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell['source'])
        new = src
        new = new.replace(
            'from data_modules.quantra_reinforcement_learning import reward_exponential_pnl',
            'from data_modules.quantra_reinforcement_learning import reward_ftmo_pnl')
        new = new.replace("'RF': reward_exponential_pnl", "'RF': reward_ftmo_pnl")
        if p.name == 'ANN in Keras.ipynb':
            old = ('env = Game(bars5m, bars1d, bars1h, reward_exponential_pnl,\n'
                   '           lkbk=LKBK, init_idx=START_IDX)')
            new = new.replace(old,
                'env = Game(bars5m, bars1d, bars1h, reward_ftmo_pnl,\n'
                '           lkbk=LKBK, init_idx=START_IDX,\n'
                '           ftmo_cost=rl_config.get(\'FTMO_COST\', 0.001),\n'
                '           ftmo_lot=rl_config.get(\'FTMO_LOT\', 1.0),\n'
                '           max_drawdown=rl_config.get(\'MAX_DRAWDOWN\', None),\n'
                '           max_loss_per_trade=rl_config.get(\'MAX_LOSS_PER_TRADE\', None),\n'
                '           max_trade_len=rl_config.get(\'MAX_TRADE_LEN\', None))')
        if p.name == 'Backtesting Implementation.ipynb':
            old = ('env = Game(bars5m, bars1d, bars1h, rl_config[\'RF\'],\n'
                   '               lkbk=rl_config[\'LKBK\'], init_idx=rl_config[\'START_IDX\'])')
            new = new.replace(old,
                'env = Game(bars5m, bars1d, bars1h, rl_config[\'RF\'],\n'
                '               lkbk=rl_config[\'LKBK\'], init_idx=rl_config[\'START_IDX\'],\n'
                '               ftmo_cost=rl_config.get(\'FTMO_COST\', 0.001),\n'
                '               ftmo_lot=rl_config.get(\'FTMO_LOT\', 1.0),\n'
                '               max_drawdown=rl_config.get(\'MAX_DRAWDOWN\', None),\n'
                '               max_loss_per_trade=rl_config.get(\'MAX_LOSS_PER_TRADE\', None),\n'
                '               max_trade_len=rl_config.get(\'MAX_TRADE_LEN\', None))')
        if src != new:
            modified = True
            cell['source'] = new.splitlines(True)
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
    if modified:
        path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
        print('patched', p)
    else:
        print('no change', p)
