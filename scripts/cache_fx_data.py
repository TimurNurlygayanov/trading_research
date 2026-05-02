"""
Bulk-cache FX OHLCV data for reuse across strategies and backtests.

Fetches each (pair, timeframe, year) into the standard parquet cache
(data/cache/<symbol>_<tf>_<start>_<end>.parquet) so subsequent calls to
backtest.data_fetcher.fetch_ohlcv() hit the local cache.

Default
  Pairs:      EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCHF, USDCAD, USDJPY
  Timeframes: 1m, 5m, 1h
  Years:      2024, 2025, 2026 (2026 capped at today)

Already-cached chunks are skipped (the fetcher checks cache first), so re-runs
are idempotent and cheap.

Usage
  python -m scripts.cache_fx_data
  python -m scripts.cache_fx_data --pairs EURUSD GBPUSD --tfs 1m 5m
  python -m scripts.cache_fx_data --all-fx               # all 20 FX pairs
  python -m scripts.cache_fx_data --years 2025 2026
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.data_fetcher import fetch_ohlcv, _FX_PAIRS, _CACHE_DIR  # noqa: E402

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
                 "USDCHF", "USDCAD", "USDJPY"]
DEFAULT_TFS   = ["1m", "5m", "1h"]
DEFAULT_YEARS = [2024, 2025, 2026]


def _year_range(yr: int) -> tuple[str, str]:
    """Return (start, end) for a calendar year, capping 2026 at today."""
    start = f"{yr}-01-01"
    today = date.today()
    nominal_end = date(yr + 1, 1, 1)
    end = min(nominal_end, today + (date.today() - date.today()))
    # Use today's date if we'd otherwise reach into the future
    end_actual = min(nominal_end, today)
    return start, end_actual.isoformat()


def _cache_size_mb() -> float:
    if not _CACHE_DIR.exists():
        return 0.0
    return sum(p.stat().st_size for p in _CACHE_DIR.glob("*.parquet")) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs",  nargs="+", default=None,
                    help="Pair symbols (default: 7 majors)")
    ap.add_argument("--tfs",    nargs="+", default=DEFAULT_TFS,
                    help="Timeframes (default: 1m 5m 1h)")
    ap.add_argument("--years",  nargs="+", type=int, default=DEFAULT_YEARS,
                    help="Calendar years (default: 2024 2025 2026)")
    ap.add_argument("--all-fx", action="store_true",
                    help="Use the full 20-pair FX universe instead of majors")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan and skip fetches")
    args = ap.parse_args()

    if args.all_fx:
        pairs = sorted(_FX_PAIRS)
    else:
        pairs = args.pairs or DEFAULT_PAIRS

    tfs   = args.tfs
    years = args.years

    total_jobs = len(pairs) * len(tfs) * len(years)
    print(f"\nPairs ({len(pairs)}): {', '.join(pairs)}")
    print(f"TFs   ({len(tfs)}): {', '.join(tfs)}")
    print(f"Years ({len(years)}): {', '.join(map(str, years))}")
    print(f"Total chunks: {total_jobs}")
    print(f"Cache dir : {_CACHE_DIR}")
    print(f"Cache size: {_cache_size_mb():.1f} MB before run")

    if args.dry_run:
        for sym in pairs:
            for tf in tfs:
                for yr in years:
                    s, e = _year_range(yr)
                    print(f"  PLAN  {sym}  {tf}  {s} → {e}")
        return

    ok = fail = skipped = 0
    t0 = time.time()
    for i, sym in enumerate(pairs, 1):
        for tf in tfs:
            for yr in years:
                start, end = _year_range(yr)
                if start >= end:
                    print(f"  [{sym} {tf} {yr}]  skip — start≥end ({start} → {end})")
                    skipped += 1
                    continue
                tag = f"[{i}/{len(pairs)}] {sym} {tf} {yr}"
                try:
                    t1 = time.time()
                    df = fetch_ohlcv(sym, tf, start, end)
                    dt = time.time() - t1
                    print(f"  {tag:<28}  {len(df):>9,} bars   {dt:>5.1f}s")
                    ok += 1
                except Exception as exc:
                    print(f"  {tag:<28}  FAIL: {exc}")
                    fail += 1

    dt_total = time.time() - t0
    print(f"\nDone in {dt_total:.1f}s.  ok={ok}  fail={fail}  skipped={skipped}")
    print(f"Cache size: {_cache_size_mb():.1f} MB after run")


if __name__ == "__main__":
    main()
