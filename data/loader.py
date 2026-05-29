"""
data/loader.py
Load MT5-exported M1 CSVs, convert to CET/CEST, resample to 15m / H1 / D1.
Returns:  data[symbol][tf_minutes] = pd.DataFrame  (index = CET-aware DatetimeIndex)

CSV format expected (MT5 History Center export, tab-separated):
  <DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <VOL>  <SPREAD>
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytz
from tqdm.auto import tqdm

# ── constants ─────────────────────────────────────────────────────────────────
CET = pytz.timezone("Europe/Berlin")

OHLCV_AGG = {
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum",
}

RESAMPLE_RULES = {
    1:    "1min",
    15:   "15min",
    60:   "1h",
    1440: "1d",
}

# ── actual file names in the project root ────────────────────────────────────
# Keys are the canonical symbol names used throughout the system.
DEFAULT_CSV_MAP: Dict[str, str] = {
    "EURUSD": "EURUSD_M1_202101131130_202605270000_2020_2026.csv",
    "GBPUSD": "GBPUSD_M1_202101131952_202605270000.csv",
    "XAUUSD": "XAUUSD_M1_202009230753_202605262259.csv",
    "US30":   "US30_M1_202007231046_202605262359.csv",
}

# ── project root (two levels up from this file: data/loader.py) ───────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_csv(symbol: str, csv_map: Dict[str, str], data_dir: Optional[Path] = None) -> Path:
    """Locate the CSV for a symbol — checks data_dir first, then project root."""
    filename = csv_map.get(symbol)
    if filename is None:
        raise FileNotFoundError(f"No CSV mapping for symbol '{symbol}'.")
    candidates = []
    if data_dir:
        candidates.append(Path(data_dir) / filename)
    candidates.append(_PROJECT_ROOT / filename)
    candidates.append(_PROJECT_ROOT / "data" / filename)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"CSV for '{symbol}' not found. Tried: {[str(c) for c in candidates]}"
    )


def load_m1(
    symbol:   str,
    csv_path: Path,
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a single MT5 M1 CSV and return a CET-indexed OHLCV DataFrame.
    date_from / date_to are optional ISO strings "YYYY-MM-DD".
    """
    # Read in chunks so large CSVs (2M+ rows) don't exhaust RAM.
    # MT5 date column is "YYYY.MM.DD" — compare as string against ISO dates.
    t0 = time.perf_counter()
    chunks = []
    reader = pd.read_csv(
        csv_path,
        sep="\t",
        header=0,
        dtype=str,
        chunksize=100_000,
    )
    done = False
    file_size_mb = csv_path.stat().st_size / 1_048_576
    pbar = tqdm(desc=f"  Reading {symbol} ({file_size_mb:.0f} MB)",
                unit="chunk", leave=False)
    for chunk in reader:
        pbar.update(1)
        if done:
            break
        # Normalise column names once (same for every chunk)
        chunk.columns = [c.strip().strip("<>").lower() for c in chunk.columns]
        chunk = chunk.reset_index(drop=True)
        date_str = chunk["date"].str.replace(".", "-", regex=False)
        if date_from:
            chunk = chunk[date_str >= date_from].reset_index(drop=True)
            date_str = chunk["date"].str.replace(".", "-", regex=False)
        if date_to:
            past_end = (date_str > date_to)
            if past_end.any():
                done = True   # all subsequent chunks are past date_to
            chunk = chunk[~past_end].reset_index(drop=True)
        if not chunk.empty:
            chunks.append(chunk)
    pbar.close()

    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.concat(chunks, ignore_index=True)

    # Re-normalise (concat resets index; columns already normalised above)
    # Build a datetime column from date + time
    df["datetime"] = pd.to_datetime(
        df["date"].str.replace(".", "-", regex=False) + " " + df["time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"])

    # Use tickvol as volume (MT5 exports tick volume)
    vol_col = "tickvol" if "tickvol" in df.columns else "vol"

    df = df[["datetime", "open", "high", "low", "close", vol_col]].copy()
    df = df.rename(columns={vol_col: "volume"})
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    # MT5 exports in exchange/server time — treat as UTC then convert to CET
    df["datetime"] = df["datetime"].dt.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward")
    df["datetime"] = df["datetime"].dt.tz_convert(CET)
    df = df.set_index("datetime").sort_index()

    return df


def resample_m1(df_1m: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """Resample a 1m DataFrame to the requested timeframe (15, 60, or 1440 minutes)."""
    rule = RESAMPLE_RULES[tf_minutes]
    return (
        df_1m.resample(rule, label="left", closed="right")
        .agg(OHLCV_AGG)
        .dropna()
    )


def load_all(
    symbols:   List[str]         = None,
    csv_map:   Dict[str, str]    = None,
    data_dir:  Optional[str]     = None,
    date_from: Optional[str]     = None,
    date_to:   Optional[str]     = None,
    timeframes: List[int]        = None,
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Load all symbols and return nested dict:
        data[symbol][tf_minutes] = DataFrame

    Parameters
    ----------
    symbols   : list of symbol strings; defaults to all in DEFAULT_CSV_MAP
    csv_map   : override file name mapping {symbol: filename}
    data_dir  : extra directory to search for CSV files
    date_from : "YYYY-MM-DD" start filter (applied to 1m data before resampling)
    date_to   : "YYYY-MM-DD" end filter
    timeframes: list of minute values; defaults to [1, 15, 60, 1440]
    """
    if symbols is None:
        symbols = list(DEFAULT_CSV_MAP.keys())
    if csv_map is None:
        csv_map = DEFAULT_CSV_MAP
    if timeframes is None:
        timeframes = [1, 15, 60, 1440]

    data: Dict[str, Dict[int, pd.DataFrame]] = {}
    total_start = time.perf_counter()

    for sym in tqdm(symbols, desc="Loading symbols", unit="sym"):
        csv_path = _find_csv(sym, csv_map, data_dir)
        t0 = time.perf_counter()
        print(f"\n[START] Loading {sym}  ({csv_path.name})", flush=True)

        df_1m = load_m1(sym, csv_path, date_from=date_from, date_to=date_to)
        print(f"[DONE]  Loading {sym} 1m — {len(df_1m):,} rows  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

        data[sym] = {1: df_1m}
        resample_tfs = [tf for tf in timeframes if tf != 1]
        for tf in tqdm(resample_tfs, desc=f"  Resampling {sym}", unit="tf", leave=False):
            tr = time.perf_counter()
            data[sym][tf] = resample_m1(df_1m, tf)
            print(f"         {sym} {tf}m: {len(data[sym][tf]):,} rows  "
                  f"({time.perf_counter()-tr:.2f}s)", flush=True)

    print(f"\n[DONE]  load_all — {len(symbols)} symbols in "
          f"{time.perf_counter()-total_start:.1f}s", flush=True)
    return data


def split_data(
    data:       Dict[str, Dict[int, pd.DataFrame]],
    train_end:  str,
    val_end:    str,
) -> tuple:
    """
    Split data into train / validation / forward-test dicts.
    train_end / val_end are "YYYY-MM-DD" strings (CET boundary).
    Returns (train_data, val_data, fwd_data) each with same structure as data.
    """
    def _slice(d, start, end):
        out = {}
        for sym, tfs in d.items():
            out[sym] = {}
            for tf, df in tfs.items():
                mask = pd.Series(True, index=df.index)
                if start:
                    mask &= df.index >= pd.Timestamp(start, tz=CET)
                if end:
                    mask &= df.index <= pd.Timestamp(end, tz=CET)
                out[sym][tf] = df[mask]
        return out

    train = _slice(data, None,       train_end)
    val   = _slice(data, train_end,  val_end)
    fwd   = _slice(data, val_end,    None)
    return train, val, fwd
