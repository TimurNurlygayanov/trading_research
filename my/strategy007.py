"""
Strategy 007 — EURUSD 5m, EMA9(HL/2) vs daily-anchored VWAP cross.

Indicators:
  EMA9 = EMA(length=9) of (High + Low)/2
  VWAP = cumulative(typical_price * tick_volume) / cumulative(tick_volume),
         RESET at the start of each UTC date (daily anchor — standard intraday default).
         typical_price = (H + L + C)/3.

Signals (evaluated at each bar's close):
  cross_up   : ema9 was <= vwap on the previous bar and > vwap on this bar  → LONG
  cross_down : ema9 was >= vwap on the previous bar and < vwap on this bar  → SHORT

Trade management:
  Entry  : at this bar's Close on the cross.
  No SL.
  TP     : fixed N pips. Hit detected intra-bar via High/Low.
  Exit   : whichever comes first —
             (a) intra-bar TP touch  → "tp" exit at TP price,
             (b) opposite cross at a later bar's close → "flip" exit at that close
                 AND immediate re-entry in the opposite direction (always-in market
                 after the first signal, until a TP).
  After a TP exit we are flat until the next fresh cross.

TP sweep: 5, 7, 10, 12, 15, 20, 25, 30 pips.
Spread:   1 pip (charged once per trade, on net pips).
Period:   last 3 months → 2026-02-15 .. 2026-05-15.
"""
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


_MT5_TF_MAP = {"5m": mt5.TIMEFRAME_M5, "15m": mt5.TIMEFRAME_M15, "1h": mt5.TIMEFRAME_H1}


def get_data(ticker, timeframe, start, end):
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end   = pd.Timestamp(end,   tz="UTC")
    mt5.initialize()
    mt5.symbol_select(ticker, True)

    tf_const = _MT5_TF_MAP[timeframe]
    CHUNK    = 50_000

    frames, pos = [], 0
    while True:
        chunk = mt5.copy_rates_from_pos(ticker, tf_const, pos, CHUNK)
        if chunk is None or len(chunk) == 0:
            break
        frames.append(pd.DataFrame(chunk))
        oldest = pd.Timestamp(int(frames[-1]["time"].min()), unit="s", tz="UTC")
        if oldest <= ts_start or len(chunk) < CHUNK:
            break
        pos += CHUNK

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames[::-1])
    df = df.drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(columns={"open": "Open", "high": "High",
                            "low": "Low", "close": "Close", "tick_volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


PIP_SIZE    = 0.0001
SPREAD_PIPS = 1.0
EMA_LEN     = 9
TP_SWEEP    = [5, 7, 10, 12, 15, 20, 25, 30]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    hl2     = (df["High"] + df["Low"]) / 2
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol     = df["Volume"].astype(float)

    df["ema9"] = hl2.ewm(span=EMA_LEN, adjust=False).mean()

    # Daily-anchored VWAP — reset at the start of every UTC date.
    day = df.index.normalize()
    pv  = (typical * vol).groupby(day).cumsum()
    cv  = vol.groupby(day).cumsum().replace(0, np.nan)
    df["vwap"] = pv / cv
    return df


def detect_crosses(df: pd.DataFrame):
    diff      = df["ema9"] - df["vwap"]
    prev_diff = diff.shift(1)
    cross_up   = ((prev_diff <= 0) & (diff > 0)).to_numpy()
    cross_down = ((prev_diff >= 0) & (diff < 0)).to_numpy()
    # Drop the very first bar of each day from cross signals — VWAP just reset
    # so any "cross" there is an artifact of the anchor, not a real crossover.
    first_of_day = (df.index.normalize() != pd.Series(df.index.normalize())
                    .shift(1).fillna(pd.Timestamp("1970-01-01", tz="UTC"))
                    .to_numpy())
    cross_up   = cross_up   & ~first_of_day
    cross_down = cross_down & ~first_of_day
    return cross_up, cross_down


def backtest(df: pd.DataFrame, cross_up: np.ndarray, cross_down: np.ndarray,
             tp_pips: float):
    """Always-in-market after first signal, until a TP fires (then flat until
    next fresh cross). Returns trade list."""
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(df)
    tp_d  = tp_pips * PIP_SIZE

    trades   = []
    position = 0          # 0 / +1 / -1
    entry_px = np.nan
    entry_i  = -1

    for i in range(n):
        # 1) Exit logic for the current bar if we hold a position from an earlier bar.
        if position != 0 and i > entry_i:
            tp_hit = False
            if position == 1:
                if high[i] >= entry_px + tp_d:
                    exit_px, kind = entry_px + tp_d, "tp"
                    tp_hit = True
            else:
                if low[i] <= entry_px - tp_d:
                    exit_px, kind = entry_px - tp_d, "tp"
                    tp_hit = True

            if tp_hit:
                trades.append({
                    "entry_i": entry_i, "exit_i": i, "dir": position,
                    "entry_px": entry_px, "exit_px": exit_px,
                    "pips": (exit_px - entry_px) * position / PIP_SIZE,
                    "hold": i - entry_i, "exit": kind,
                })
                position = 0
                # Fall through — still allow a fresh entry on this bar below.

        # 2) Cross signal at this bar's close.
        if cross_up[i]:
            if position == -1:
                exit_px = close[i]
                trades.append({
                    "entry_i": entry_i, "exit_i": i, "dir": position,
                    "entry_px": entry_px, "exit_px": exit_px,
                    "pips": (exit_px - entry_px) * position / PIP_SIZE,
                    "hold": i - entry_i, "exit": "flip",
                })
                position, entry_px, entry_i = 1, close[i], i
            elif position == 0:
                position, entry_px, entry_i = 1, close[i], i
        elif cross_down[i]:
            if position == 1:
                exit_px = close[i]
                trades.append({
                    "entry_i": entry_i, "exit_i": i, "dir": position,
                    "entry_px": entry_px, "exit_px": exit_px,
                    "pips": (exit_px - entry_px) * position / PIP_SIZE,
                    "hold": i - entry_i, "exit": "flip",
                })
                position, entry_px, entry_i = -1, close[i], i
            elif position == 0:
                position, entry_px, entry_i = -1, close[i], i

    # Close any dangling position at the final bar's close (mark-to-market, not a TP).
    if position != 0:
        exit_px = close[-1]
        trades.append({
            "entry_i": entry_i, "exit_i": n - 1, "dir": position,
            "entry_px": entry_px, "exit_px": exit_px,
            "pips": (exit_px - entry_px) * position / PIP_SIZE,
            "hold": n - 1 - entry_i, "exit": "eod",
        })
    return trades


def summarize(trades, df, tp_pips):
    if not trades:
        return {"tp": tp_pips, "n": 0, "pips_net": 0.0, "win_pct": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "avg_hold": 0.0,
                "sharpe": 0.0, "n_tp": 0, "n_flip": 0, "n_eod": 0,
                "n_long": 0, "n_short": 0, "best_trade": 0.0, "worst_trade": 0.0}
    pips_gross = np.array([t["pips"] for t in trades])
    pips_net   = pips_gross - SPREAD_PIPS
    wins       = pips_net > 0
    days       = max(1, (df.index[-1] - df.index[0]).days)
    sharpe     = (pips_net.mean() / pips_net.std()
                  * np.sqrt(len(trades) / (days / 365.25))
                  if len(trades) > 1 and pips_net.std() > 0 else 0.0)
    return {
        "tp":          tp_pips,
        "n":           len(trades),
        "pips_net":    float(pips_net.sum()),
        "win_pct":     float(wins.mean() * 100),
        "avg_win":     float(pips_net[wins].mean())  if wins.any()  else 0.0,
        "avg_loss":    float(pips_net[~wins].mean()) if (~wins).any() else 0.0,
        "avg_hold":    float(np.mean([t["hold"] for t in trades])),
        "sharpe":      float(sharpe),
        "n_tp":        sum(t["exit"] == "tp"   for t in trades),
        "n_flip":      sum(t["exit"] == "flip" for t in trades),
        "n_eod":       sum(t["exit"] == "eod"  for t in trades),
        "n_long":      sum(t["dir"] == 1  for t in trades),
        "n_short":     sum(t["dir"] == -1 for t in trades),
        "best_trade":  float(pips_net.max()),
        "worst_trade": float(pips_net.min()),
    }


def per_month_breakdown(trades, df):
    if not trades:
        return []
    idx = df.index
    rows = []
    months = sorted({(idx[t["entry_i"]].year, idx[t["entry_i"]].month) for t in trades})
    for ym in months:
        sub = [t for t in trades if (idx[t["entry_i"]].year, idx[t["entry_i"]].month) == ym]
        pn  = np.array([t["pips"] for t in sub]) - SPREAD_PIPS
        rows.append({
            "ym":       f"{ym[0]:04d}-{ym[1]:02d}",
            "n":        len(sub),
            "pips_net": float(pn.sum()),
            "win_pct":  float((pn > 0).mean() * 100),
            "n_tp":     sum(t["exit"] == "tp"   for t in sub),
            "n_flip":   sum(t["exit"] == "flip" for t in sub),
        })
    return rows


# ----- run -----
TICKER = "EURUSD"
START  = "2026-02-15"
END    = "2026-05-15"

print(f"Strategy 007 — EMA9(HL/2) vs daily-anchored VWAP cross on {TICKER} 5m")
print(f"Period: {START} .. {END}    Spread: {SPREAD_PIPS} pip\n")

print(f"Loading {TICKER} 5m ...")
df = get_data(TICKER, "5m", START, END)
if df.empty:
    raise SystemExit(f"No data for {TICKER}")
print(f"  {len(df)} bars  ({df.index.min()} .. {df.index.max()})")

df = add_indicators(df)
cross_up, cross_down = detect_crosses(df)
print(f"  cross-ups: {int(cross_up.sum())}   cross-downs: {int(cross_down.sum())}")

print(f"\nTP sweep (pips):")
print(f"{'TP':>4} {'N':>5} {'L/S':>9} {'TP':>5} {'Flip':>5} {'EOD':>4} "
      f"{'Win%':>6} {'NetPips':>9} {'AvgW':>6} {'AvgL':>7} {'Hold':>6} "
      f"{'Sharpe':>7} {'Best':>6} {'Worst':>7}")
print("-" * 100)

all_summaries = []
all_trades    = {}
for tp_pips in TP_SWEEP:
    trades = backtest(df, cross_up, cross_down, tp_pips)
    s = summarize(trades, df, tp_pips)
    all_summaries.append(s)
    all_trades[tp_pips] = trades
    ls = f"{s['n_long']}/{s['n_short']}"
    print(f"{tp_pips:>4} {s['n']:>5} {ls:>9} {s['n_tp']:>5} {s['n_flip']:>5} "
          f"{s['n_eod']:>4} {s['win_pct']:>5.1f}% {s['pips_net']:>+9.1f} "
          f"{s['avg_win']:>+6.1f} {s['avg_loss']:>+7.1f} {s['avg_hold']:>6.1f} "
          f"{s['sharpe']:>+7.2f} {s['best_trade']:>+6.1f} {s['worst_trade']:>+7.1f}")

best = max(all_summaries, key=lambda s: s["pips_net"])
print(f"\nBest TP by net pips: {best['tp']} pips  →  "
      f"net {best['pips_net']:+.1f}  sharpe {best['sharpe']:+.2f}  "
      f"win% {best['win_pct']:.1f}  n={best['n']}")

print(f"\nPer-month breakdown @ TP={best['tp']} pips:")
print(f"{'Month':>8} {'N':>5} {'TP':>4} {'Flip':>5} {'Win%':>6} {'NetPips':>9}")
print("-" * 50)
for r in per_month_breakdown(all_trades[best["tp"]], df):
    print(f"{r['ym']:>8} {r['n']:>5} {r['n_tp']:>4} {r['n_flip']:>5} "
          f"{r['win_pct']:>5.1f}% {r['pips_net']:>+9.1f}")
