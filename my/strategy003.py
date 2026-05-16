"""
Strategy 003 — 50-bar breakout, fixed +5 pip take-profit, fixed-pip stop.

SHORT signal at bar i:
  Low[i] < min(Low[i-50 : i])    (strict new 50-bar low)
  Entry = Close[i]
  SL    = Entry + SL_PIPS·pip
  TP    = Entry − 5·pip

LONG signal at bar i (mirror):
  High[i] > max(High[i-50 : i])  (strict new 50-bar high)
  Entry = Close[i]
  SL    = Entry − SL_PIPS·pip
  TP    = Entry + 5·pip

Exit walk starts at bar i+1. First-touch wins; SL wins ties (worst-case).
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
TP_PIPS     = 5.0
N_LOOKBACK  = 50


def compute_signals(df: pd.DataFrame, n_lookback: int = N_LOOKBACK):
    """Return (long_entry, short_entry) bool arrays.

    LONG  = High[i] strictly > max High over the n_lookback bars BEFORE i.
    SHORT = Low[i]  strictly < min Low  over the n_lookback bars BEFORE i.
    """
    high = df["High"].values
    low  = df["Low"].values

    prev_max_hi = pd.Series(high).rolling(n_lookback, min_periods=n_lookback).max().shift(1).values
    prev_min_lo = pd.Series(low ).rolling(n_lookback, min_periods=n_lookback).min().shift(1).values

    long_sig  = np.isfinite(prev_max_hi) & (high > prev_max_hi)
    short_sig = np.isfinite(prev_min_lo) & (low  < prev_min_lo)
    return long_sig, short_sig


def backtest(df: pd.DataFrame, signal: np.ndarray, direction: int,
             sl_pips: float, tp_pips: float = TP_PIPS) -> dict:
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
            sl_px = entry_px - direction * sl_pips * PIP_SIZE
            tp_px = entry_px + direction * tp_pips * PIP_SIZE
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
TICKER    = "EURUSD"
TFS       = ["5m", "1h", "4h"]
SL_VALUES = [5, 10, 20, 50]

print(f"Loading {TICKER} on {len(TFS)} timeframes…")
data = {}
for tf in TFS:
    df = get_data(ticker=TICKER, timeframe=tf)
    data[tf] = df
    print(f"  {tf}: {len(df)} bars")

hdr = (f"{'TF':<4} {'SL':>3} {'Dir':<5} {'Trades':>7} {'Net p':>9} {'Sharpe':>7} "
       f"{'Win%':>6} {'TP%':>6} {'AvgW':>7} {'AvgL':>7} {'Hold':>6}")
print(f"\n{hdr}")
print("-" * len(hdr))

for tf in TFS:
    df = data[tf]
    long_sig, short_sig = compute_signals(df)
    for sl_pips in SL_VALUES:
        for label, sig, direction in [("LONG", long_sig, +1),
                                       ("SHORT", short_sig, -1)]:
            m = backtest(df, sig, direction, sl_pips=sl_pips)
            if m.get("n", 0) == 0:
                print(f"{tf:<4} {sl_pips:>3} {label:<5} {'-':>7}")
                continue
            print(f"{tf:<4} {sl_pips:>3} {label:<5} {m['n']:>7} {m['pips_net']:>+9.1f} "
                  f"{m['sharpe']:>+7.2f} {m['win_pct']:>5.1f}% {m['tp_pct']:>5.1f}% "
                  f"{m['avg_win']:>+7.1f} {m['avg_loss']:>+7.1f} {m['avg_hold']:>6.1f}")
