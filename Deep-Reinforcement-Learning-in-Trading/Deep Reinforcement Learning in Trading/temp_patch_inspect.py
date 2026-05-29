import json
from pathlib import Path

root = Path('.')
for name in [Path('Game Class/Game Class.ipynb'), Path('Positions and Rewards/Reward System.ipynb')]:
    p = root / name
    nb = json.loads(p.read_text(encoding='utf-8'))
    print('FILE', name)
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell['source'])
        if 'reward_exponential_pnl' in src or 'FTMO_COST' in src:
            print(' CELL', i)
            for line in src.splitlines():
                if any(tag in line for tag in ['reward_exponential_pnl', 'FTMO_COST', 'FTMO_LOT', 'MAX_DRAWNDOWN', 'MAX_LOSS_PER_TRADE', 'MAX_TRADE_LEN']):
                    print('   ', line)
    print()