"""
Modal job: runs statistical probability research across symbols and timeframes.

Tests 42 market conditions (candle patterns, EMA levels, session times, etc.)
for forward-return bias. Results stored in prob_research_results table.

Run manually: modal run modal_jobs/prob_research_job.py
"""
import os as _os
import modal

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

app = modal.App("trading-research-prob")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "pandas>=2.3.2",
        "numpy",
        "scipy==1.14.1",
        "requests==2.32.3",
        "supabase==2.9.1",
        "python-dotenv==1.0.1",
        "pyarrow",
        "massive",
    )
    .add_local_dir(_os.path.join(_ROOT, "db"),      remote_path="/root/db")
    .add_local_dir(_os.path.join(_ROOT, "agents"),  remote_path="/root/agents")
    .add_local_dir(_os.path.join(_ROOT, "backtest"), remote_path="/root/backtest")
)

ohlcv_cache = modal.Volume.from_name("trading-research-ohlcv-cache", create_if_missing=True)
CACHE_DIR = "/ohlcv_cache"

DEFAULT_SYMBOLS    = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD"]
DEFAULT_TIMEFRAMES = ["4h", "1h", "5m", "1m", "1d"]
DEFAULT_FORWARD_BARS = [1, 4, 12, 24]


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    secrets=[modal.Secret.from_name("trading-research-secrets")],
    volumes={CACHE_DIR: ohlcv_cache},
)
def run_prob_research(
    symbols: list | None = None,
    timeframes: list | None = None,
    forward_bars: list | None = None,
) -> dict:
    """
    Run statistical probability research for all condition × symbol × timeframe combinations.

    For each combination, tests 42 market conditions for forward-return bias and
    upserts results into the prob_research_results table.
    Uses cached OHLCV data from the Modal Volume when available.
    """
    import os
    import traceback

    from backtest.data_fetcher import fetch_ohlcv, split_train_oos
    from agents.prob_researcher import get_all_specs, run_analysis
    from db import supabase_client as db

    symbols      = symbols      or DEFAULT_SYMBOLS
    timeframes   = timeframes   or DEFAULT_TIMEFRAMES
    forward_bars = forward_bars or DEFAULT_FORWARD_BARS

    specs = get_all_specs(forward_bars)
    total_results = 0
    cache_new_files = False

    for symbol in symbols:
        for tf in timeframes:
            cache_file = f"{CACHE_DIR}/{symbol}_{tf}.parquet"
            try:
                if os.path.exists(cache_file):
                    print(f"[prob_research] Loading cached {symbol}_{tf}")
                    df = __import__("pandas").read_parquet(cache_file)
                else:
                    print(f"[prob_research] Fetching {symbol}_{tf} from API")
                    df = fetch_ohlcv(symbol, tf, start="2015-01-01", end="2026-12-31")
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    df.to_parquet(cache_file)
                    cache_new_files = True

                train_df, _ = split_train_oos(df)
                print(f"[prob_research] {symbol}_{tf}: {len(train_df):,} train bars, {len(specs)} conditions")

                for spec in specs:
                    try:
                        analysis_rows = run_analysis(
                            train_df,
                            spec.condition_id,
                            tf,
                            forward_bars,
                        )
                        for row in analysis_rows:
                            row["symbol"]         = symbol
                            row["timeframe"]      = tf
                            row["condition_desc"] = spec.description
                            row["category"]       = spec.category
                            row["params"]         = spec.params
                            db.upsert_prob_result(row)
                            total_results += 1
                    except Exception as exc:
                        tb = traceback.format_exc()
                        print(f"[prob_research] ERROR {symbol}_{tf} {spec.condition_id}: {exc}\n{tb[:300]}")

            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[prob_research] ERROR loading {symbol}_{tf}: {exc}\n{tb[:300]}")

    if cache_new_files:
        ohlcv_cache.commit()

    return {
        "ok":           True,
        "total_results": total_results,
        "symbols":      symbols,
        "timeframes":   timeframes,
    }


@app.local_entrypoint()
def main():
    """Allow `modal run modal_jobs/prob_research_job.py` for manual execution."""
    result = run_prob_research.remote()
    print(f"Prob research complete:")
    print(f"  symbols:       {result['symbols']}")
    print(f"  timeframes:    {result['timeframes']}")
    print(f"  total_results: {result['total_results']:,}")
