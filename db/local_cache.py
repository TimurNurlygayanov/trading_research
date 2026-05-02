"""
Filesystem-only metadata for cached OHLCV parquets in data/cache/.

Used by the dashboard's practice page when Supabase is not configured.
Mirrors a subset of db.supabase_client.get_data_cache().
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"

# Filename forms produced by backtest.data_fetcher:
#   <sym>_<tf>_<YYYY-MM-DD>_<YYYY-MM-DD>.parquet  (date-range chunk)
#   <sym>_<tf>.parquet                            (full-history snapshot)
_FN_DATE  = re.compile(r"^([A-Z]{6})_([0-9]+[mhdw])_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
_FN_FULL  = re.compile(r"^([A-Z]{6})_([0-9]+[mhdw])\.parquet$")

_CACHE_TTL_SEC = 60.0
_scan_cache: tuple[float, dict[tuple[str, str], dict]] = (0.0, {})


def _read_index_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp, int] | None:
    """Read just the parquet index to get first/last timestamp + row count."""
    try:
        # columns=[] gives a DataFrame with the index but no columns;
        # use len(df.index) since df.empty would be True for zero-column frames.
        df = pd.read_parquet(path, columns=[])
    except Exception:
        return None
    if len(df.index) == 0:
        return None
    return df.index.min(), df.index.max(), len(df.index)


def _scan() -> dict[tuple[str, str], dict[str, Any]]:
    """Walk data/cache and aggregate per (symbol, tf). Cached for _CACHE_TTL_SEC."""
    global _scan_cache
    now = time.time()
    if now - _scan_cache[0] < _CACHE_TTL_SEC and _scan_cache[1]:
        return _scan_cache[1]

    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not CACHE_DIR.exists():
        _scan_cache = (now, out)
        return out

    for p in sorted(CACHE_DIR.glob("*.parquet")):
        m = _FN_DATE.match(p.name) or _FN_FULL.match(p.name)
        if not m:
            continue
        sym = m.group(1)
        tf  = m.group(2)
        rng = _read_index_range(p)
        if rng is None:
            continue
        first, last, rows = rng
        rec = out.setdefault((sym, tf), {
            "symbol":     sym,
            "timeframe":  tf,
            "first_date": first,
            "last_date":  last,
            "bar_count":  0,
            "files":      0,
        })
        if first < rec["first_date"]:
            rec["first_date"] = first
        if last  > rec["last_date"]:
            rec["last_date"]  = last
        rec["bar_count"] += rows
        rec["files"]     += 1

    _scan_cache = (now, out)
    return out


def list_cached_datasets(symbol: str | None = None) -> list[dict[str, Any]]:
    """Return [{symbol, timeframe, first_date, last_date, bar_count, files}, ...]."""
    out = []
    for (sym, tf), rec in sorted(_scan().items()):
        if symbol and sym != symbol:
            continue
        out.append({
            "symbol":     sym,
            "timeframe":  tf,
            "first_date": rec["first_date"].isoformat(),
            "last_date":  rec["last_date"].isoformat(),
            "bar_count":  rec["bar_count"],
            "files":      rec["files"],
        })
    return out


def get_dataset_range(symbol: str, timeframe: str) -> dict[str, Any] | None:
    """Return {first_date, last_date, bar_count} for one (symbol, tf), or None."""
    rec = _scan().get((symbol, timeframe))
    if rec is None:
        return None
    return {
        "first_date": rec["first_date"].isoformat(),
        "last_date":  rec["last_date"].isoformat(),
        "bar_count":  rec["bar_count"],
    }


def load_ohlcv(symbol: str, timeframe: str,
               start: str, end: str) -> pd.DataFrame:
    """Concat all cache parquets for (symbol, tf) and slice [start, end].

    The standard fetch_ohlcv() cache requires an exact-date-range match.
    This helper covers cross-chunk requests entirely from local files —
    needed when the API is unreachable (e.g. dashboard offline mode).
    """
    if not CACHE_DIR.exists():
        return pd.DataFrame()

    parts = []
    for p in sorted(CACHE_DIR.glob(f"{symbol}_{timeframe}_*.parquet")):
        if not _FN_DATE.match(p.name):
            continue
        try:
            parts.append(pd.read_parquet(p))
        except Exception:
            continue
    full = CACHE_DIR / f"{symbol}_{timeframe}.parquet"
    if full.exists():
        try:
            parts.append(pd.read_parquet(full))
        except Exception:
            pass

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)]
