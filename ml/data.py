"""
Parquet loader for data/cache.

Files come in two naming conventions:
  SYMBOL_TF.parquet                            (one big file)
  SYMBOL_TF_YYYY-MM-DD_YYYY-MM-DD.parquet      (date-ranged shards)

We read all matching files, dedupe by timestamp, sort, slice [start, end).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

_FNAME_RE = re.compile(
    r"^(?P<sym>[A-Z]{6}|XAUUSD)(?:_(?P<tag>mt5|polygon))?_(?P<tf>\d+[mh])"
    r"(?:_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2})?\.parquet$"
)

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _list_files(symbol: str, tf: str, source: str | None = None) -> list[Path]:
    """source: None = any; "mt5" = MT5-tagged only; "polygon" = polygon only."""
    out = []
    for p in CACHE_DIR.iterdir():
        m = _FNAME_RE.match(p.name)
        if not m: continue
        if m.group("sym") != symbol or m.group("tf") != tf: continue
        tag = m.group("tag")
        if source is None:
            out.append(p)
        elif source == "mt5" and tag == "mt5":
            out.append(p)
        elif source == "polygon" and tag != "mt5":
            out.append(p)
    return out


def load_ohlcv(
    symbol: str,
    tf: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame indexed by UTC datetime with lowercase OHLCV columns,
    sliced to [start, end) if given. Empty DataFrame if no files found.
    """
    files = _list_files(symbol, tf, source=source)
    if not files:
        raise FileNotFoundError(f"No parquet files for {symbol} {tf} (source={source}) in {CACHE_DIR}")

    parts = [pd.read_parquet(p) for p in files]
    df = pd.concat(parts, axis=0)
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in _OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol} {tf} missing columns: {missing}")
    # Keep spread_pips if present (MT5 source) for cost modeling
    keep = list(_OHLCV_COLS)
    if "spread_pips" in df.columns:
        keep.append("spread_pips")
    df = df[keep]
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if start is not None:
        df = df.loc[pd.Timestamp(start, tz="UTC"):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end, tz="UTC") - pd.Timedelta("1ns")]

    return df


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample OHLCV to a higher TF. Used for HTF features."""
    rule = {"1h": "1H", "4h": "4H", "1d": "1D"}.get(target_tf, target_tf)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule, label="right", closed="right").agg(agg).dropna()
    return out
