"""
execution/mt5_bridge.py
MT5 data bridge: connect, load historical candles, convert to CET/CEST,
stream live bars and send orders.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytz

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False

# ── timezone ──────────────────────────────────────────────────────────────────
CET = pytz.timezone("Europe/Berlin")

# ── timeframe map ─────────────────────────────────────────────────────────────
_TF_MAP = {
    1:    "mt5.TIMEFRAME_M1",
    5:    "mt5.TIMEFRAME_M5",
    15:   "mt5.TIMEFRAME_M15",
    60:   "mt5.TIMEFRAME_H1",
    1440: "mt5.TIMEFRAME_D1",
}

_TF_RESAMPLE = {
    1:    "1min",
    5:    "5min",
    15:   "15min",
    60:   "1H",
    1440: "1D",
}

SYMBOLS   = ["EURUSD", "GBPUSD", "XAUUSD", "US30", "NAS100"]
TIMEFRAMES = [1440, 60, 15, 1]   # minutes

OHLCV_AGG = {
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum",
}


# ─────────────────────────────────────────────────────────────────────────────
class MT5Bridge:
    """
    Connects to MT5, loads historical candles and converts timestamps to CET.
    Also provides order routing for live/forward testing.
    """

    def __init__(self, login: int = 0, password: str = "", server: str = "",
                 cache_dir: str = "data_cache"):
        self.login    = login
        self.password = password
        self.server   = server
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._connected = False

    # ── connection ────────────────────────────────────────────────────────────
    def connect(self) -> bool:
        if not _MT5_AVAILABLE:
            print("[MT5Bridge] MetaTrader5 package not installed – using cached/CSV data only.")
            return False
        if not mt5.initialize():
            print("[MT5Bridge] mt5.initialize() failed:", mt5.last_error())
            return False
        if self.login:
            ok = mt5.login(self.login, password=self.password, server=self.server)
            if not ok:
                print("[MT5Bridge] Login failed:", mt5.last_error())
                return False
        self._connected = True
        print("[MT5Bridge] Connected to MT5.")
        return True

    def disconnect(self):
        if _MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False

    # ── historical candles ────────────────────────────────────────────────────
    def load_candles(
        self,
        symbol: str,
        tf_minutes: int,
        date_from: datetime,
        date_to:   datetime,
    ) -> pd.DataFrame:
        """
        Load OHLCV candles from MT5 (or cache).
        Timestamps are returned in CET/CEST (Europe/Berlin).
        """
        cache_key = f"{symbol}_{tf_minutes}m_{date_from.date()}_{date_to.date()}.pkl"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            return pd.read_pickle(cache_path)

        if not self._connected:
            raise RuntimeError("[MT5Bridge] Not connected and no cache found.")

        tf_const = getattr(mt5, f"TIMEFRAME_M{tf_minutes}" if tf_minutes < 60
                           else f"TIMEFRAME_H{tf_minutes//60}" if tf_minutes < 1440
                           else "TIMEFRAME_D1")

        utc_from = date_from.replace(tzinfo=timezone.utc)
        utc_to   = date_to.replace(tzinfo=timezone.utc)

        rates = mt5.copy_rates_range(symbol, tf_const, utc_from, utc_to)
        if rates is None or len(rates) == 0:
            raise ValueError(f"[MT5Bridge] No data for {symbol} tf={tf_minutes}m")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["time"] = df["time"].dt.tz_convert(CET)
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"})[["open", "high", "low", "close", "volume"]]
        df.to_pickle(cache_path)
        return df

    # ── load all symbols × timeframes ─────────────────────────────────────────
    def load_all(
        self,
        symbols:    List[str]  = SYMBOLS,
        timeframes: List[int]  = TIMEFRAMES,
        date_from:  datetime   = datetime(2018, 1, 1),
        date_to:    datetime   = datetime(2024, 12, 31),
    ) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Returns nested dict: data[symbol][tf_minutes] = DataFrame (CET index).
        """
        data: Dict[str, Dict[int, pd.DataFrame]] = {}
        for sym in symbols:
            data[sym] = {}
            for tf in timeframes:
                print(f"  Loading {sym} {tf}m ...")
                data[sym][tf] = self.load_candles(sym, tf, date_from, date_to)
        return data

    # ── resample from 1m to higher TFs ───────────────────────────────────────
    @staticmethod
    def resample(df_1m: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
        """Resample a 1m DataFrame to a higher timeframe."""
        rule = _TF_RESAMPLE[tf_minutes]
        return df_1m.resample(rule, label="left", closed="right").agg(OHLCV_AGG).dropna()

    # ── order routing (live / forward test) ──────────────────────────────────
    def send_order(
        self,
        symbol:    str,
        action:    str,          # "buy" | "sell" | "flat"
        lots:      float = 0.01,
        sl_points: int   = 200,
        tp_points: int   = 400,
        comment:   str   = "FTMO-RL",
    ) -> Optional[dict]:
        """Send a market order via MT5."""
        if not self._connected or not _MT5_AVAILABLE:
            print(f"[MT5Bridge] send_order skipped (not connected): {action} {symbol}")
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"[MT5Bridge] Symbol {symbol} not found.")
            return None

        point = info.point
        price = mt5.symbol_info_tick(symbol).ask if action == "buy" else mt5.symbol_info_tick(symbol).bid

        if action == "flat":
            positions = mt5.positions_get(symbol=symbol)
            results = []
            for pos in (positions or []):
                close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                close_price = mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
                req = {
                    "action":    mt5.TRADE_ACTION_DEAL,
                    "symbol":    symbol,
                    "volume":    pos.volume,
                    "type":      close_type,
                    "price":     close_price,
                    "position":  pos.ticket,
                    "comment":   comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                results.append(mt5.order_send(req))
            return {"closed": results}

        order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
        sl = (price - sl_points * point) if action == "buy" else (price + sl_points * point)
        tp = (price + tp_points * point) if action == "buy" else (price - tp_points * point)

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    symbol,
            "volume":    lots,
            "type":      order_type,
            "price":     price,
            "sl":        sl,
            "tp":        tp,
            "comment":   comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result._asdict() if result else None


# ── convenience loader for backtesting without live MT5 ──────────────────────
def load_csv_data(path: str, tz: str = "Europe/Berlin") -> pd.DataFrame:
    """Load a CSV with a datetime index and convert to CET."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(tz)
    df.columns = [c.lower() for c in df.columns]
    return df
