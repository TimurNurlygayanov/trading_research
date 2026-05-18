"""
Fetch EURUSD bars from a running MT5 terminal (FTMO Demo by default) and
save them in the same parquet format used by data/cache/.

Includes the per-bar `spread` column from MT5 (in points; 1 point = 0.00001 on
EURUSD 5-digit pricing → 0.1 pip). The mean/median of this column gives us the
realistic FTMO average spread for cost modeling.

Run:
    python scripts/fetch_mt5.py --symbol EURUSD --timeframe 5m --start 2025-01-01 --end 2026-05-19
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import MetaTrader5 as mt5

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

_TF_MAP = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
}


def fetch(symbol: str, tf: str, start: str, end: str) -> pd.DataFrame:
    if not mt5.initialize():
        sys.exit(f"MT5 initialize() failed: {mt5.last_error()}")

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    tf_const = _TF_MAP[tf]

    # MT5 limits a single call; chunk by month to be safe.
    chunks = []
    cur = start_dt
    while cur < end_dt:
        next_cur = min(cur + pd.Timedelta(days=31), end_dt)
        bars = mt5.copy_rates_range(symbol, tf_const, cur, next_cur)
        if bars is None or len(bars) == 0:
            print(f"  no data {cur.date()}..{next_cur.date()}")
        else:
            chunks.append(pd.DataFrame(bars))
            print(f"  fetched {len(bars):,} bars  {cur.date()}..{next_cur.date()}")
        cur = next_cur

    info = mt5.symbol_info(symbol)
    point = info.point if info else 0.00001
    mt5.shutdown()

    if not chunks:
        sys.exit("No data fetched.")

    df = pd.concat(chunks, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    # Rename to our convention
    df = df.rename(columns={"tick_volume": "volume"})
    # Keep: open high low close volume + spread (points)
    cols = ["open", "high", "low", "close", "volume", "spread"]
    df = df[cols]
    df["spread_pips"] = df["spread"] * point / 0.0001
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out-name", default=None, help="Override output filename")
    args = p.parse_args()

    print(f">>> fetching {args.symbol} {args.timeframe}  {args.start}..{args.end}")
    df = fetch(args.symbol, args.timeframe, args.start, args.end)

    print(f"\n>>> {len(df):,} total bars, range {df.index.min()} .. {df.index.max()}")
    print(f"  spread (pips): mean={df['spread_pips'].mean():.3f}  median={df['spread_pips'].median():.3f}  max={df['spread_pips'].max():.2f}")

    name = args.out_name or f"{args.symbol}_mt5_{args.timeframe}_{args.start}_{args.end}.parquet"
    out_path = CACHE_DIR / name
    df.to_parquet(out_path)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
