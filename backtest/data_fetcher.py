"""
Historical OHLCV data fetcher using the Massive (formerly Polygon.io) Python client.

pip install -U massive

Ticker conventions:
  FX:     "C:EURUSD"  (prefix C:)
  Crypto: "X:BTCUSD"  (prefix X:)
  Stocks: "SPY"       (no prefix)

Convenience aliases (no prefix needed in callers):
  fetch_ohlcv("EURUSD", "1h", ...)   → internally uses "C:EURUSD"
  fetch_ohlcv("BTCUSD", "1h", ...)   → internally uses "X:BTCUSD"

Returns a pandas DataFrame with columns:
  Open, High, Low, Close, Volume
  Index: DatetimeIndex, UTC-aware, sorted ascending.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Ticker prefix mapping ────────────────────────────────────────────────────
# FX pairs: prefix C:
_FX_PAIRS = {
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF",
    "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "EURAUD", "GBPAUD", "AUDCAD", "AUDCHF", "AUDJPY",
    "CADJPY", "CHFJPY", "EURCHF", "EURCAD", "GBPCAD",
}
# Crypto: prefix X:
_CRYPTO = {
    "BTCUSD", "ETHUSD", "SOLUSD", "ADAUSD", "XRPUSD",
    "DOTUSD", "LINKUSD", "MATICUSD",
}


def _to_massive_ticker(symbol: str) -> str:
    """Add the correct prefix for the Massive/Polygon API."""
    sym = symbol.upper().replace("/", "").replace("-", "")
    if sym in _FX_PAIRS:
        return f"C:{sym}"
    if sym in _CRYPTO:
        return f"X:{sym}"
    return sym  # stocks — no prefix


# ── Timeframe parsing ────────────────────────────────────────────────────────
def _parse_timeframe(timeframe: str) -> tuple[int, str]:
    """
    Convert shorthand like "1h", "4h", "15m", "1d" to (multiplier, timespan).
    Timespan values accepted by Massive: minute, hour, day, week, month.
    """
    tf = timeframe.lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1]), "minute"
    if tf.endswith("h"):
        return int(tf[:-1]), "hour"
    if tf.endswith("d"):
        return int(tf[:-1]), "day"
    if tf.endswith("w"):
        return int(tf[:-1]), "week"
    raise ValueError(f"Unrecognised timeframe: {timeframe!r}. Use e.g. '1m','15m','1h','4h','1d'.")


# ── Main fetch function ──────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    adjusted: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles via the Massive Python client.

    Parameters
    ----------
    symbol    : e.g. "EURUSD", "GBPUSD", "BTCUSD", "SPY"
    timeframe : e.g. "1m", "15m", "1h", "4h", "1d"
    start     : "YYYY-MM-DD"
    end       : "YYYY-MM-DD"
    adjusted  : whether to return split/dividend-adjusted data (relevant for stocks)

    Returns
    -------
    DataFrame columns [Open, High, Low, Close, Volume], DatetimeIndex UTC, sorted asc.
    """
    api_key = os.environ.get("MARKET_DATA_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "MARKET_DATA_API_KEY is not set. "
            "Add it to your .env file. Get a key at massive.com."
        )

    try:
        from massive import RESTClient  # pip install -U massive
    except ImportError:
        raise ImportError(
            "massive package not installed. Run: pip install -U massive"
        )

    ticker = _to_massive_ticker(symbol)
    multiplier, timespan = _parse_timeframe(timeframe)

    client = RESTClient(api_key=api_key)

    # Massive/Polygon returns at most 50,000 bars per API call.
    # We chunk the date range so each chunk stays well under that limit.
    # Safe chunk sizes (targeting ~30k bars max per call):
    #   1m  → 20 days  (1m × 1440/day × 20 = 28,800)
    #   5m  → 100 days (5m ×  288/day × 100 = 28,800)
    #   1h  → 365 days (1h ×   24/day × 365 =  8,760)
    #   4h+ → 365 days
    _BARS_PER_DAY = {
        "minute": multiplier * 1440,  # e.g. 1m=1440, 5m=288
        "hour":   multiplier * 24,
        "day":    multiplier,
    }
    bars_per_day = _BARS_PER_DAY.get(timespan, 24)
    # Keep each chunk under 30k bars; minimum 1 day, maximum 365 days
    chunk_days = max(1, min(365, 30_000 // max(bars_per_day, 1)))

    all_bars: list[dict] = []

    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)

    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(
            chunk_start + timedelta(days=chunk_days - 1),
            end_dt,
        )

        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=chunk_start.isoformat(),
                to=chunk_end.isoformat(),
                adjusted=adjusted,
                sort="asc",
                limit=50000,
            )
        except Exception as e:
            raise RuntimeError(
                f"Massive API error fetching {ticker} "
                f"{chunk_start}→{chunk_end}: {e}"
            ) from e

        for bar in aggs:
            all_bars.append({
                "datetime": pd.Timestamp(bar.timestamp, unit="ms", tz="UTC"),
                "Open":   float(bar.open),
                "High":   float(bar.high),
                "Low":    float(bar.low),
                "Close":  float(bar.close),
                "Volume": float(bar.volume) if bar.volume is not None else 0.0,
            })

        chunk_start = chunk_end + timedelta(days=1)

    if not all_bars:
        raise ValueError(
            f"No data returned for {ticker} ({timeframe}) "
            f"between {start} and {end}. "
            "Check your API key, symbol, and date range."
        )

    df = pd.DataFrame(all_bars).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]  # drop any duplicate timestamps

    # Basic integrity checks
    if (df["High"] < df["Low"]).any():
        n = (df["High"] < df["Low"]).sum()
        raise ValueError(f"Data integrity error: {n} bars where High < Low for {ticker}")
    if (df["Close"] <= 0).any():
        raise ValueError(f"Data integrity error: non-positive Close prices for {ticker}")

    return df


# ── Train / OOS split ────────────────────────────────────────────────────────

def split_train_oos(
    df: pd.DataFrame,
    oos_start: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into training (all data before OOS_START_DATE) and OOS (2026+).

    Returns (train_df, oos_df).
    train_df is used for Optuna tuning + walk-forward.
    oos_df  is the untouched hold-out — never used for parameter fitting.
    """
    if oos_start is None:
        oos_start = os.environ.get("OOS_START_DATE", "2026-01-01")

    cutoff = pd.Timestamp(oos_start, tz="UTC")
    train = df[df.index < cutoff].copy()
    oos   = df[df.index >= cutoff].copy()

    if train.empty:
        raise ValueError(f"No training data before {oos_start}")
    if oos.empty:
        raise ValueError(
            f"No OOS data from {oos_start} onwards. "
            "Fetch data up to today and ensure OOS_START_DATE is in the past."
        )

    return train, oos
