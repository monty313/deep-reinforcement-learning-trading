"""
env/ftmo_fees.py
FTMO transaction fee calculator based on asset class.
Reference: https://ftmo.com/en/blog/whats-consuming-all-your-profits/
"""

from typing import Dict

# FTMO fee structure (as of 2025-05-28)
FTMO_FEES = {
    # FX pairs: $5 per lot
    "EURUSD": {"type": "fx", "fee_per_lot": 5.0},
    "GBPUSD": {"type": "fx", "fee_per_lot": 5.0},
    "USDJPY": {"type": "fx", "fee_per_lot": 5.0},
    "AUDUSD": {"type": "fx", "fee_per_lot": 5.0},
    "USDCAD": {"type": "fx", "fee_per_lot": 5.0},
    "NZDUSD": {"type": "fx", "fee_per_lot": 5.0},

    # Metals: 0.001% of volume
    "XAUUSD": {"type": "metals", "fee_pct": 0.00001},  # 0.001% as decimal
    "XAGUSD": {"type": "metals", "fee_pct": 0.00001},

    # Indices: $0
    "US30": {"type": "indices", "fee_per_lot": 0.0},
    "GER40": {"type": "indices", "fee_per_lot": 0.0},
    "SPX500": {"type": "indices", "fee_per_lot": 0.0},

    # Energy: $0
    "CRUDE": {"type": "energy", "fee_per_lot": 0.0},
    "NATGAS": {"type": "energy", "fee_per_lot": 0.0},

    # Crypto: $0
    "BTCUSD": {"type": "crypto", "fee_per_lot": 0.0},
    "ETHUSD": {"type": "crypto", "fee_per_lot": 0.0},

    # Stocks: 0.004% of volume
    "AAPL": {"type": "stocks", "fee_pct": 0.00004},  # 0.004% as decimal
    "MSFT": {"type": "stocks", "fee_pct": 0.00004},
    "TSLA": {"type": "stocks", "fee_pct": 0.00004},
}


def get_fee_config(symbol: str) -> Dict:
    """
    Get fee configuration for a symbol.

    Args:
        symbol: Trading symbol (e.g., "EURUSD", "XAUUSD")

    Returns:
        Dict with "type" and fee details. Defaults to FX if symbol not found.
    """
    return FTMO_FEES.get(symbol, {"type": "fx", "fee_per_lot": 5.0})


def calculate_fee(
    symbol: str,
    entry_price: float,
    exit_price: float,
    lots: float,
    volume: float = None,
) -> float:
    """
    Calculate transaction fee for a trade.

    Args:
        symbol: Trading symbol (e.g., "EURUSD", "XAUUSD")
        entry_price: Entry price of the trade
        exit_price: Exit price of the trade
        lots: Number of lots traded
        volume: Total volume in units (optional, for metals/stocks)

    Returns:
        Fee amount in account currency (USD)
    """
    config = get_fee_config(symbol)
    fee_type = config.get("type")

    if fee_type == "fx":
        # FX: flat $5 per lot per round trip (entry + exit)
        fee_per_lot = config.get("fee_per_lot", 5.0)
        return fee_per_lot * abs(lots) * 2  # 2x: entry + exit

    elif fee_type == "metals":
        # Metals: 0.001% of volume
        if volume is None:
            volume = abs(lots) * 1000  # Rough estimate if not provided
        fee_pct = config.get("fee_pct", 0.00001)  # 0.001% as decimal
        return volume * entry_price * fee_pct * 2  # 2x: entry + exit

    elif fee_type == "stocks":
        # Stocks: 0.004% of volume
        if volume is None:
            volume = abs(lots) * 100  # Rough estimate if not provided
        fee_pct = config.get("fee_pct", 0.00004)  # 0.004% as decimal
        return volume * entry_price * fee_pct * 2  # 2x: entry + exit

    else:
        # Indices, Energy, Crypto: $0
        return 0.0


def calculate_pnl_after_fees(
    symbol: str,
    entry_price: float,
    exit_price: float,
    lots: float,
    pnl_abs: float,
    volume: float = None,
) -> tuple:
    """
    Calculate PnL after deducting transaction fees.

    Args:
        symbol: Trading symbol
        entry_price: Entry price
        exit_price: Exit price
        lots: Number of lots
        pnl_abs: Absolute PnL before fees (in account currency)
        volume: Total volume (optional)

    Returns:
        Tuple of (pnl_after_fees, fee_amount)
    """
    fee = calculate_fee(symbol, entry_price, exit_price, lots, volume)
    pnl_after_fees = pnl_abs - fee
    return pnl_after_fees, fee


def calculate_batch_fees(
    trades: list,
) -> Dict:
    """
    Calculate total fees for a batch of trades.

    Args:
        trades: List of trade dicts with keys:
                {symbol, entry_price, exit_price, lots, pnl_abs, volume (optional)}

    Returns:
        Dict with:
        {
            "total_fees": float,
            "fees_by_symbol": {symbol: fee},
            "fees_by_type": {type: fee},
            "total_pnl_before_fees": float,
            "total_pnl_after_fees": float,
        }
    """
    total_fees = 0.0
    total_pnl_before = 0.0
    total_pnl_after = 0.0
    fees_by_symbol = {}
    fees_by_type = {}

    for trade in trades:
        symbol = trade.get("symbol", "EURUSD")
        entry = trade.get("entry_price", 0.0)
        exit_p = trade.get("exit_price", 0.0)
        lots = trade.get("lots", 0.0)
        pnl_abs = trade.get("pnl_abs", 0.0)
        volume = trade.get("volume")

        fee = calculate_fee(symbol, entry, exit_p, lots, volume)
        pnl_after = pnl_abs - fee

        total_fees += fee
        total_pnl_before += pnl_abs
        total_pnl_after += pnl_after

        fees_by_symbol[symbol] = fees_by_symbol.get(symbol, 0.0) + fee

        config = get_fee_config(symbol)
        fee_type = config.get("type")
        fees_by_type[fee_type] = fees_by_type.get(fee_type, 0.0) + fee

    return {
        "total_fees": total_fees,
        "fees_by_symbol": fees_by_symbol,
        "fees_by_type": fees_by_type,
        "total_pnl_before_fees": total_pnl_before,
        "total_pnl_after_fees": total_pnl_after,
    }
