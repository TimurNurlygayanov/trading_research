"""
Strategy 011 — Long-only VWAP+ATR breakout, hold until profitable.

Entry (long only):
  green candle (Close > Open)
  AND Close > VWAP + 1 * ATR(14)
  AND no position currently open
  Entry at the close of the signal bar.

Exit:
  Limit order placed at entry + N pips immediately after entry.
  Filled on first intra-bar High >= that level.
  N is swept: 1, 2, 3, 5 pips.
    N=1 → 0 net pip after spread (true breakeven scratch)
    N=2 → +1 net pip ("we are profitable")
    N=3 → +2 net pip
    N=5 → +4 net pip
  No SL, no time stop. If never profitable, the trade stays OPEN through end of window.

Trades that never reach the exit level are marked "open" — reported separately
with current unrealized PnL and max adverse excursion (MAE).

Daily-anchored VWAP (typical * tick_volume). ATR is Wilder's, length 14.
Spread: 1 pip, charged once per CLOSED trade.

Two windows so we don't fool ourselves:
  OOS: 2025-11-15 .. 2026-02-15
  IS : 2026-02-15 .. 2026-05-15
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
ATR_LEN     = 14
ATR_MULT    = 1.0
N_SWEEP     = [1, 2, 3, 5]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c, o = df["High"], df["Low"], df["Close"], df["Open"]

    # Wilder ATR
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / ATR_LEN, adjust=False).mean()

    # Daily-anchored VWAP
    typical = (h + l + c) / 3
    vol     = df["Volume"].astype(float)
    day     = df.index.normalize()
    pv      = (typical * vol).groupby(day).cumsum()
    cv      = vol.groupby(day).cumsum().replace(0, np.nan)
    df["vwap"] = pv / cv
    return df


def entry_signal(df: pd.DataFrame) -> np.ndarray:
    green        = (df["Close"] > df["Open"]).to_numpy()
    above_thresh = (df["Close"] > df["vwap"] + ATR_MULT * df["atr"]).to_numpy()
    have_atr     = np.isfinite(df["atr"].to_numpy())
    return green & above_thresh & have_atr


def backtest(df: pd.DataFrame, sig: np.ndarray, n_pips: float):
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(df)
    tp_d  = n_pips * PIP_SIZE

    trades = []
    position = 0
    entry_px = np.nan
    entry_i  = -1
    mae_px   = np.nan        # min low seen since entry

    for i in range(n):
        if position == 1 and i > entry_i:
            mae_px = min(mae_px, low[i])
            if high[i] >= entry_px + tp_d:
                exit_px = entry_px + tp_d
                trades.append({
                    "entry_i": entry_i, "exit_i": i,
                    "entry_px": entry_px, "exit_px": exit_px,
                    "pips":   (exit_px - entry_px) / PIP_SIZE,
                    "mae_pips": (mae_px - entry_px) / PIP_SIZE,
                    "hold": i - entry_i, "status": "closed",
                })
                position = 0
                continue   # don't re-enter on same bar

        if position == 0 and sig[i]:
            position = 1
            entry_px = close[i]
            entry_i  = i
            mae_px   = close[i]    # MAE seeded at entry

    # Any still-open position: record unrealized
    if position == 1:
        mae_px = min(mae_px, low[entry_i + 1: ].min() if entry_i + 1 < n else mae_px)
        trades.append({
            "entry_i": entry_i, "exit_i": n - 1,
            "entry_px": entry_px, "exit_px": close[-1],
            "pips":   (close[-1] - entry_px) / PIP_SIZE,
            "mae_pips": (mae_px - entry_px) / PIP_SIZE,
            "hold": n - 1 - entry_i, "status": "open",
        })
    return trades


def summarize(trades, df, n_pips):
    closed = [t for t in trades if t["status"] == "closed"]
    open_  = [t for t in trades if t["status"] == "open"]

    pips_closed_net = np.array([t["pips"] for t in closed]) - SPREAD_PIPS if closed else np.array([])
    pips_open_unr   = np.array([t["pips"] for t in open_])  - SPREAD_PIPS if open_  else np.array([])
    mae_closed      = np.array([t["mae_pips"] for t in closed]) if closed else np.array([])
    mae_open        = np.array([t["mae_pips"] for t in open_])  if open_  else np.array([])
    hold_closed     = np.array([t["hold"] for t in closed]) if closed else np.array([])

    return {
        "n_pips":          n_pips,
        "n_closed":        len(closed),
        "n_open":          len(open_),
        "pips_realized":   float(pips_closed_net.sum()) if closed else 0.0,
        "pips_unrealized": float(pips_open_unr.sum())   if open_  else 0.0,
        "avg_hold":        float(hold_closed.mean())    if closed else 0.0,
        "max_hold":        int(hold_closed.max())       if closed else 0,
        "max_open_hold":   int(max(t["hold"] for t in open_)) if open_ else 0,
        "worst_mae_closed":float(mae_closed.min())      if closed else 0.0,
        "median_mae_closed": float(np.median(mae_closed)) if closed else 0.0,
        "worst_mae_open":  float(mae_open.min())        if open_  else 0.0,
        "worst_open_pl":   float(pips_open_unr.min())   if open_  else 0.0,
    }


def per_month(trades, df):
    if not trades:
        return []
    idx = df.index
    rows = []
    months = sorted({(idx[t["entry_i"]].year, idx[t["entry_i"]].month) for t in trades})
    for ym in months:
        sub = [t for t in trades if (idx[t["entry_i"]].year, idx[t["entry_i"]].month) == ym]
        cl  = [t for t in sub if t["status"] == "closed"]
        op  = [t for t in sub if t["status"] == "open"]
        rows.append({
            "ym":         f"{ym[0]:04d}-{ym[1]:02d}",
            "n_closed":   len(cl),
            "n_open":     len(op),
            "pips_real":  float(sum(t["pips"] for t in cl) - SPREAD_PIPS * len(cl)),
            "pips_unr":   float(sum(t["pips"] for t in op) - SPREAD_PIPS * len(op)),
            "worst_mae":  float(min((t["mae_pips"] for t in sub), default=0.0)),
        })
    return rows


def run_window(label, start, end):
    print(f"\n{'='*78}\n{label}:  {start}  ..  {end}\n{'='*78}")
    df = get_data("EURUSD", "5m", start, end)
    if df.empty:
        print("  (no data)")
        return
    df  = add_indicators(df)
    sig = entry_signal(df)
    print(f"  bars: {len(df)}   entry signals: {int(sig.sum())}")

    print(f"\n  {'N':>3} {'Closed':>6} {'Open':>5} {'Realized':>9} {'Unreal':>9} "
          f"{'Total':>9} {'AvgHold':>7} {'MaxHold':>7} {'OpenHold':>8} "
          f"{'MedMAE':>7} {'WorstMAE':>8} {'WorstOpenP/L':>13}")
    print("  " + "-" * 110)
    primary_trades = None
    for n_pips in N_SWEEP:
        trades = backtest(df, sig, n_pips)
        s = summarize(trades, df, n_pips)
        total = s["pips_realized"] + s["pips_unrealized"]
        print(f"  {n_pips:>3} {s['n_closed']:>6} {s['n_open']:>5} "
              f"{s['pips_realized']:>+9.1f} {s['pips_unrealized']:>+9.1f} "
              f"{total:>+9.1f} {s['avg_hold']:>7.1f} {s['max_hold']:>7d} "
              f"{s['max_open_hold']:>8d} {s['median_mae_closed']:>+7.1f} "
              f"{s['worst_mae_closed']:>+8.1f} {s['worst_open_pl']:>+13.1f}")
        if n_pips == 2:
            primary_trades = trades

    if primary_trades is None:
        return
    print(f"\n  Per-month @ N=2 (entry+2 pips = +1 net pip exit):")
    print(f"  {'Month':>8} {'Cl':>4} {'Op':>3} {'Real':>8} {'Unrl':>8} "
          f"{'WorstMAE':>8}")
    print("  " + "-" * 50)
    for r in per_month(primary_trades, df):
        print(f"  {r['ym']:>8} {r['n_closed']:>4} {r['n_open']:>3} "
              f"{r['pips_real']:>+8.1f} {r['pips_unr']:>+8.1f} "
              f"{r['worst_mae']:>+8.1f}")


print(f"Strategy 011 — long-only VWAP+ATR breakout, hold until profitable")
print(f"  entry: green & Close > VWAP + {ATR_MULT}*ATR({ATR_LEN})    "
      f"exit: limit at entry + N pips    no SL    spread {SPREAD_PIPS} pip")

run_window("OUT-OF-SAMPLE", "2025-11-15", "2026-02-15")
run_window("IN-SAMPLE",     "2026-02-15", "2026-05-15")
