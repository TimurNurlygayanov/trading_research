"""
Modal job: pre-loads and caches OHLCV data to the Modal Volume.

Downloads EURUSD data at 1m, 5m, and 1h timeframes (configurable).
Saves per-dataset stats + recent bars to Supabase data_cache table
so the /data dashboard page can show analysis without re-reading the Volume.

Timeframe-specific date ranges (1m data from 2015 would be ~4M bars):
  1m  → 2024-01-01 onwards   (~500k bars, ~50 MB parquet)
  5m  → 2022-01-01 onwards   (~500k bars, ~50 MB parquet)
  1h  → 2015-01-01 onwards   (full history)

Run manually:  modal run modal_jobs/preload_data_job.py
Auto-trigger:  POST /api/data/preload on the orchestrator
"""
import os as _os
import modal

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

app = modal.App("trading-research-preload")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "pandas>=2.3.2",
        "numpy",
        "requests==2.32.3",
        "supabase==2.9.1",
        "python-dotenv==1.0.1",
        "pyarrow",
        "massive",
    )
    .add_local_dir(_os.path.join(_ROOT, "db"),       remote_path="/root/db")
    .add_local_dir(_os.path.join(_ROOT, "backtest"), remote_path="/root/backtest")
)

ohlcv_cache = modal.Volume.from_name("trading-research-ohlcv-cache", create_if_missing=True)
CACHE_DIR = "/ohlcv_cache"

# Timeframe → (start_date, bars_per_trading_day)  — weekdays only for FX
_TF_CONFIG = {
    "1m":  {"start": "2024-01-01", "bars_per_day": 1440},
    "5m":  {"start": "2022-01-01", "bars_per_day": 288},
    "1h":  {"start": "2015-01-01", "bars_per_day": 24},
    "4h":  {"start": "2015-01-01", "bars_per_day": 6},
    "1d":  {"start": "2015-01-01", "bars_per_day": 1},
}

DEFAULT_SYMBOLS    = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD"]
DEFAULT_TIMEFRAMES = ["4h", "1h", "5m", "1m"]


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=3600,
    secrets=[modal.Secret.from_name("trading-research-secrets")],
    volumes={CACHE_DIR: ohlcv_cache},
)
def preload_ohlcv_data(
    symbols: list | None = None,
    timeframes: list | None = None,
) -> dict:
    """
    Download and cache OHLCV data for each symbol/timeframe combination.
    Updates Supabase data_cache table with metadata + recent bars for chart.
    Returns a summary dict keyed by "SYMBOL_timeframe".
    """
    import os
    import traceback

    import pandas as pd

    from backtest.data_fetcher import fetch_ohlcv
    from db import supabase_client as db_client

    symbols    = symbols    or DEFAULT_SYMBOLS
    timeframes = timeframes or DEFAULT_TIMEFRAMES

    results: dict = {}

    for symbol in symbols:
        for timeframe in timeframes:
            key = f"{symbol}_{timeframe}"
            cfg = _TF_CONFIG.get(timeframe, {"start": "2015-01-01", "bars_per_day": None})
            start_date = cfg["start"]

            try:
                print(f"[preload] Fetching {key} from {start_date}…")
                df = fetch_ohlcv(symbol, timeframe, start=start_date, end="2026-12-31")

                # Persist to Modal Volume
                os.makedirs(CACHE_DIR, exist_ok=True)
                cache_file = f"{CACHE_DIR}/{key}.parquet"
                df.to_parquet(cache_file)
                ohlcv_cache.commit()
                file_size_mb = round(os.path.getsize(cache_file) / 1024 / 1024, 2)

                bar_count  = len(df)
                first_date = df.index[0].isoformat()
                last_date  = df.index[-1].isoformat()

                # Completeness: actual bars vs maximum possible (24h × calendar days).
                # FX data can include weekends and late-Sunday opens, so calendar days
                # (not business days) gives the correct denominator.
                # A healthy EURUSD dataset is typically 85–95% (gaps = weekends/holidays).
                bpd = cfg.get("bars_per_day")
                completeness_pct: float | None = None
                if bpd:
                    calendar_days = (df.index[-1].date() - df.index[0].date()).days + 1
                    expected = calendar_days * bpd
                    if expected:
                        completeness_pct = min(round(bar_count / expected * 100, 1), 100.0)

                # Bars-per-year breakdown — shows which years have good data coverage
                bars_by_year: dict = {}
                bars_by_hour: dict = {}
                for ts in df.index:
                    y = str(ts.year)
                    h = ts.hour
                    bars_by_year[y] = bars_by_year.get(y, 0) + 1
                    bars_by_hour[h] = bars_by_hour.get(h, 0) + 1
                print(f"[preload] {key} coverage by year: {bars_by_year}")
                print(f"[preload] {key} coverage by hour (UTC): { {h: bars_by_hour[h] for h in sorted(bars_by_hour)} }")

                # Recent 500 bars for chart display
                recent = df.tail(500)
                recent_bars = [
                    {
                        "t": str(idx),
                        "o": round(float(row["Open"]),   5),
                        "h": round(float(row["High"]),   5),
                        "l": round(float(row["Low"]),    5),
                        "c": round(float(row["Close"]),  5),
                        "v": round(float(row["Volume"]), 2),
                    }
                    for idx, row in recent.iterrows()
                ]

                db_client.upsert_data_cache(symbol, timeframe, {
                    "symbol":           symbol,
                    "timeframe":        timeframe,
                    "bar_count":        bar_count,
                    "first_date":       first_date,
                    "last_date":        last_date,
                    "expected_start":   start_date,
                    "file_size_mb":     file_size_mb,
                    "price_min":        round(float(df["Close"].min()), 5),
                    "price_max":        round(float(df["Close"].max()), 5),
                    "price_mean":       round(float(df["Close"].mean()), 5),
                    "price_std":        round(float(df["Close"].std()),  5),
                    "avg_volume":       round(float(df["Volume"].mean()), 2),
                    "completeness_pct": completeness_pct,
                    "bars_by_year":     bars_by_year,
                    "recent_bars":      recent_bars,
                })

                results[key] = {
                    "ok":         True,
                    "bar_count":  bar_count,
                    "first_date": first_date,
                    "last_date":  last_date,
                    "size_mb":    file_size_mb,
                }
                print(f"[preload] {key}: {bar_count:,} bars, {file_size_mb:.1f} MB")

            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[preload] ERROR {key}: {exc}\n{tb[:500]}")
                results[key] = {"ok": False, "error": str(exc)}

    return results


@app.local_entrypoint()
def main():
    """Allow `modal run modal_jobs/preload_data_job.py` for manual pre-loading."""
    result = preload_ohlcv_data.remote()
    for key, val in result.items():
        if val.get("ok"):
            print(f"  ✓ {key}: {val['bar_count']:,} bars  ({val['size_mb']} MB)")
        else:
            print(f"  ✗ {key}: {val['error']}")
