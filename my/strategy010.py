"""
Strategy 010 — Robustness check for the EMA=15 / TP=50 winner from 009.

Two cheap tests in one script:
  1. Per-month breakdown of the original in-sample window
        IS: 2026-02-15 .. 2026-05-15  (the 3 months from 007–009).
  2. Out-of-sample re-run on the previous 3 months
       OOS: 2025-11-15 .. 2026-02-15  (data never seen during EMA/TP picking).

Same logic as 009 (the winner cell): EMA(15) of (H+L)/2, daily-anchored VWAP,
enter on cross at bar close, intra-bar TP at +50 pips, flip on opposite cross
at bar close. Spread 1 pip per trade.

The point: if OOS is also profitable AND the months in IS aren't carried by
one outlier, the EMA=15/TP=50 pick is at least minimally credible.
"""
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


_MT5_TF_MAP = {"5m": mt5.TIMEFRAME_M5}


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
EMA_LEN     = 15
TP_PIPS     = 50.0


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    hl2     = (df["High"] + df["Low"]) / 2
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol     = df["Volume"].astype(float)

    df["ema"] = hl2.ewm(span=EMA_LEN, adjust=False).mean()
    day = df.index.normalize()
    pv  = (typical * vol).groupby(day).cumsum()
    cv  = vol.groupby(day).cumsum().replace(0, np.nan)
    df["vwap"] = pv / cv
    return df


def detect_crosses(df: pd.DataFrame):
    diff = df["ema"] - df["vwap"]
    prev = diff.shift(1)
    cross_up   = ((prev <= 0) & (diff > 0)).to_numpy()
    cross_down = ((prev >= 0) & (diff < 0)).to_numpy()
    first_of_day = (df.index.normalize() != pd.Series(df.index.normalize())
                    .shift(1).fillna(pd.Timestamp("1970-01-01", tz="UTC"))
                    .to_numpy())
    cross_up   = cross_up   & ~first_of_day
    cross_down = cross_down & ~first_of_day
    return cross_up, cross_down


def backtest(df, cross_up, cross_down, tp_pips):
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(df)
    tp_d  = tp_pips * PIP_SIZE

    trades   = []
    position = 0
    entry_px = np.nan
    entry_i  = -1

    def book(exit_i, exit_px, kind):
        trades.append({
            "entry_i": entry_i, "exit_i": exit_i, "dir": position,
            "entry_px": entry_px, "exit_px": exit_px,
            "pips": (exit_px - entry_px) * position / PIP_SIZE,
            "hold": exit_i - entry_i, "exit": kind,
        })

    for i in range(n):
        if position != 0 and i > entry_i:
            tp_hit = False
            if position == 1 and high[i] >= entry_px + tp_d:
                exit_px, tp_hit = entry_px + tp_d, True
            elif position == -1 and low[i] <= entry_px - tp_d:
                exit_px, tp_hit = entry_px - tp_d, True
            if tp_hit:
                book(i, exit_px, "tp")
                position = 0

        if cross_up[i]:
            if position == -1:
                book(i, close[i], "flip")
                position, entry_px, entry_i = 1, close[i], i
            elif position == 0:
                position, entry_px, entry_i = 1, close[i], i
        elif cross_down[i]:
            if position == 1:
                book(i, close[i], "flip")
                position, entry_px, entry_i = -1, close[i], i
            elif position == 0:
                position, entry_px, entry_i = -1, close[i], i

    if position != 0:
        book(n - 1, close[-1], "eod")
    return trades


def summarize(trades, df):
    if not trades:
        return None
    pips_gross = np.array([t["pips"] for t in trades])
    pips_net   = pips_gross - SPREAD_PIPS
    wins       = pips_net > 0
    days       = max(1, (df.index[-1] - df.index[0]).days)
    sharpe     = (pips_net.mean() / pips_net.std()
                  * np.sqrt(len(trades) / (days / 365.25))
                  if len(trades) > 1 and pips_net.std() > 0 else 0.0)
    return {
        "n":         len(trades),
        "pips_net":  float(pips_net.sum()),
        "win_pct":   float(wins.mean() * 100),
        "avg_win":   float(pips_net[wins].mean())  if wins.any()  else 0.0,
        "avg_loss":  float(pips_net[~wins].mean()) if (~wins).any() else 0.0,
        "avg_hold":  float(np.mean([t["hold"] for t in trades])),
        "sharpe":    float(sharpe),
        "n_tp":      sum(t["exit"] == "tp"   for t in trades),
        "n_flip":    sum(t["exit"] == "flip" for t in trades),
        "n_eod":     sum(t["exit"] == "eod"  for t in trades),
        "n_long":    sum(t["dir"] == 1  for t in trades),
        "n_short":   sum(t["dir"] == -1 for t in trades),
        "best":      float(pips_net.max()),
        "worst":     float(pips_net.min()),
    }


def per_month(trades, df):
    if not trades:
        return []
    idx  = df.index
    rows = []
    months = sorted({(idx[t["entry_i"]].year, idx[t["entry_i"]].month) for t in trades})
    for ym in months:
        sub = [t for t in trades if (idx[t["entry_i"]].year, idx[t["entry_i"]].month) == ym]
        pn  = np.array([t["pips"] for t in sub]) - SPREAD_PIPS
        wins = pn > 0
        rows.append({
            "ym":       f"{ym[0]:04d}-{ym[1]:02d}",
            "n":        len(sub),
            "pips_net": float(pn.sum()),
            "win_pct":  float(wins.mean() * 100),
            "n_tp":     sum(t["exit"] == "tp"   for t in sub),
            "n_flip":   sum(t["exit"] == "flip" for t in sub),
            "best":     float(pn.max()),
            "worst":    float(pn.min()),
        })
    return rows


def run_window(label, start, end):
    print(f"\n{'='*78}")
    print(f"{label}:  {start}  ..  {end}")
    print(f"{'='*78}")
    df = get_data("EURUSD", "5m", start, end)
    if df.empty:
        print("  (no data)")
        return None
    df = add_indicators(df)
    cu, cd = detect_crosses(df)
    print(f"  bars: {len(df)}    cross-ups: {int(cu.sum())}    cross-downs: {int(cd.sum())}")
    trades = backtest(df, cu, cd, TP_PIPS)
    s = summarize(trades, df)
    if s is None:
        print("  (no trades)")
        return None
    print(f"\n  Overall:  n={s['n']}  L/S={s['n_long']}/{s['n_short']}  "
          f"TP={s['n_tp']}  flip={s['n_flip']}  eod={s['n_eod']}")
    print(f"            net {s['pips_net']:+.1f}  win% {s['win_pct']:.1f}  "
          f"sharpe {s['sharpe']:+.2f}")
    print(f"            avg_win {s['avg_win']:+.2f}  avg_loss {s['avg_loss']:+.2f}  "
          f"avg_hold {s['avg_hold']:.1f}  best {s['best']:+.1f}  worst {s['worst']:+.1f}")

    print(f"\n  Per-month:")
    print(f"  {'Month':>8} {'N':>4} {'TP':>4} {'Flip':>5} {'Win%':>6} "
          f"{'NetPips':>9} {'Best':>6} {'Worst':>7}")
    print("  " + "-" * 56)
    for r in per_month(trades, df):
        print(f"  {r['ym']:>8} {r['n']:>4} {r['n_tp']:>4} {r['n_flip']:>5} "
              f"{r['win_pct']:>5.1f}% {r['pips_net']:>+9.1f} "
              f"{r['best']:>+6.1f} {r['worst']:>+7.1f}")
    return s


print(f"Strategy 010 — robustness check for EMA={EMA_LEN}, TP={TP_PIPS:.0f} pips on EURUSD 5m")
print(f"Spread: {SPREAD_PIPS} pip per trade")

oos = run_window("OUT-OF-SAMPLE (never tuned on)",
                 "2025-11-15", "2026-02-15")
is_ = run_window("IN-SAMPLE (the original 3 months)",
                 "2026-02-15", "2026-05-15")

print(f"\n{'='*78}")
print("VERDICT")
print(f"{'='*78}")
if oos and is_:
    print(f"  IS  net: {is_['pips_net']:+8.1f}  sharpe {is_['sharpe']:+.2f}  "
          f"({is_['n']} trades)")
    print(f"  OOS net: {oos['pips_net']:+8.1f}  sharpe {oos['sharpe']:+.2f}  "
          f"({oos['n']} trades)")
    sign_is  = "POS" if is_['pips_net']  > 0 else "NEG"
    sign_oos = "POS" if oos['pips_net'] > 0 else "NEG"
    print(f"  → IS {sign_is}, OOS {sign_oos}")
