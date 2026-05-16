"""
Strategy 002 — Local-extremum reversal with rejection-wick filter, 1:2 RR.

SHORT signal at bar i:
  High[i] is the local max over the last N bars (including bar i)
  AND Close[i] < Open[i]                 (bearish bar)
  AND (High[i] − Open[i]) > (Open[i] − Close[i])
                                          (upper wick longer than body)
  Entry = Close[i]
  SL    = High[i]
  TP    = Entry − 2·(High[i] − Entry)

LONG signal at bar i:
  Low[i] is the local min over the last N bars (including bar i)
  AND Close[i] > Open[i]                 (bullish bar)
  AND (Open[i] − Low[i]) > (Close[i] − Open[i])
                                          (lower wick longer than body)
  Entry = Close[i]
  SL    = Low[i]
  TP    = Entry + 2·(Entry − Low[i])

Exit walk starts at bar i+1. First-touch wins; if SL and TP are both
touched on the same bar, SL is assumed first (worst-case).
"""
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


_MT5_TF_MAP = {
    "1m":  1,       # TIMEFRAME_M1
    "5m":  5,       # TIMEFRAME_M5
    "15m": 15,      # TIMEFRAME_M15
    "30m": 30,      # TIMEFRAME_M30
    "1h":  16385,   # TIMEFRAME_H1
    "4h":  16388,   # TIMEFRAME_H4
    "1d":  16408,   # TIMEFRAME_D1
}


def get_data(ticker="EURUSD", timeframe="5m", start="2026-01-01", end="2026-05-01"):
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end   = pd.Timestamp(end,   tz="UTC")
    mt5.initialize()

    tf_const = _MT5_TF_MAP.get(timeframe, 1)
    mt5.symbol_select(ticker, True)
    CHUNK = 50_000

    frames, pos = [], 0
    while True:
        chunk = mt5.copy_rates_from_pos(ticker, tf_const, pos, CHUNK)
        if chunk is None or len(chunk) < CHUNK:
            if chunk is not None and len(chunk):
                frames.append(pd.DataFrame(chunk))
            break
        frames.append(pd.DataFrame(chunk))
        oldest = pd.Timestamp(int(frames[-1]["time"].min()), unit="s", tz="UTC")
        if oldest <= ts_start:
            break
        pos += CHUNK

    df = pd.concat(frames[::-1])
    df = df.drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(columns={"open": "Open", "high": "High",
                            "low": "Low", "close": "Close", "tick_volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


PIP_SIZE    = 0.0001
SPREAD_PIPS = 1.0
RR          = 2.0


def compute_signals(df: pd.DataFrame, n_lookback: int,
                    wick_filter: bool = True):
    """Return (long_entry, short_entry) bool arrays.

    SHORT = local-max High + bearish bar; optionally requires upper wick > body.
    LONG  = local-min Low  + bullish bar; optionally requires lower wick > body.
    """
    high  = df["High"].values
    low   = df["Low"].values
    open_ = df["Open"].values
    close = df["Close"].values

    roll_max_hi = pd.Series(high).rolling(n_lookback, min_periods=n_lookback).max().values
    roll_min_lo = pd.Series(low ).rolling(n_lookback, min_periods=n_lookback).min().values

    is_local_max = np.isfinite(roll_max_hi) & (high == roll_max_hi)
    is_local_min = np.isfinite(roll_min_lo) & (low  == roll_min_lo)

    bearish = close < open_
    bullish = close > open_
    body       = np.abs(close - open_)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    short_sig = is_local_max & bearish
    long_sig  = is_local_min & bullish
    if wick_filter:
        short_sig &= (upper_wick > body)
        long_sig  &= (lower_wick > body)
    return long_sig, short_sig


def backtest(df: pd.DataFrame, signal: np.ndarray, direction: int,
             rr: float = RR) -> dict:
    """direction = +1 (long) or -1 (short)."""
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(df)
    trades: list[dict] = []

    pos = False
    entry_px = sl_px = tp_px = 0.0
    entry_i  = 0

    for i in range(n):
        if pos:
            if direction == 1:
                hit_sl = low[i]  <= sl_px
                hit_tp = high[i] >= tp_px
            else:
                hit_sl = high[i] >= sl_px
                hit_tp = low[i]  <= tp_px

            if hit_sl:                 # SL wins ties (worst-case)
                exit_px, kind = sl_px, "sl"
            elif hit_tp:
                exit_px, kind = tp_px, "tp"
            else:
                continue

            pips = (exit_px - entry_px) * direction / PIP_SIZE
            trades.append({"pips": pips, "hold": i - entry_i, "exit": kind})
            pos = False
            continue

        if signal[i]:
            pos = True
            entry_px = close[i]
            if direction == 1:
                sl_px = low[i]
                tp_px = entry_px + rr * (entry_px - sl_px)
            else:
                sl_px = high[i]
                tp_px = entry_px - rr * (sl_px - entry_px)
            entry_i = i

    if not trades:
        return {"n": 0}

    pips_gross = np.array([t["pips"] for t in trades])
    pips_net   = pips_gross - SPREAD_PIPS
    wins       = pips_net > 0
    days       = max(1, (df.index[-1] - df.index[0]).days)
    sharpe     = (pips_net.mean() / pips_net.std()
                  * np.sqrt(len(trades) / (days / 365.25))
                  if len(trades) > 1 and pips_net.std() > 0 else 0.0)
    n_tp       = sum(1 for t in trades if t["exit"] == "tp")
    return {
        "n":         len(trades),
        "pips_net":  float(pips_net.sum()),
        "win_pct":   float(wins.mean() * 100),
        "tp_pct":    float(n_tp / len(trades) * 100),
        "avg_win":   float(pips_net[wins].mean())  if wins.any()    else 0.0,
        "avg_loss":  float(pips_net[~wins].mean()) if (~wins).any() else 0.0,
        "avg_hold":  float(np.mean([t["hold"] for t in trades])),
        "sharpe":    float(sharpe),
    }


# ── Sweep ──
TICKER = "EURUSD"
TFS    = ["5m", "1h", "4h"]
NS     = [5, 10, 20, 50]

print(f"Loading {TICKER} on {len(TFS)} timeframes…")
data = {}
for tf in TFS:
    df = get_data(ticker=TICKER, timeframe=tf)
    data[tf] = df
    print(f"  {tf}: {len(df)} bars")

hdr = (f"{'TF':<4} {'N':>4} {'Dir':<5} {'Trades':>7} {'Net p':>9} {'Sharpe':>7} "
       f"{'Win%':>6} {'TP%':>6} {'AvgW':>7} {'AvgL':>7} {'Hold':>6}")
print(f"\n{hdr}")
print("-" * len(hdr))

for tf in TFS:
    df = data[tf]
    for n_lb in NS:
        long_sig, short_sig = compute_signals(df, n_lookback=n_lb)
        for label, sig, direction in [("LONG", long_sig, +1),
                                       ("SHORT", short_sig, -1)]:
            m = backtest(df, sig, direction)
            if m.get("n", 0) == 0:
                print(f"{tf:<4} {n_lb:>4} {label:<5} {'-':>7}")
                continue
            print(f"{tf:<4} {n_lb:>4} {label:<5} {m['n']:>7} {m['pips_net']:>+9.1f} "
                  f"{m['sharpe']:>+7.2f} {m['win_pct']:>5.1f}% {m['tp_pct']:>5.1f}% "
                  f"{m['avg_win']:>+7.1f} {m['avg_loss']:>+7.1f} {m['avg_hold']:>6.1f}")
