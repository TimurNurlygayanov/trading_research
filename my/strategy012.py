"""
Strategy 012 — Long-only VWAP+ATR breakout. TP = "hold until profitable",
                SL = current VWAP (dynamic, recomputed every bar).

Entry (long only):
  green candle (Close > Open)
  AND Close > VWAP + 1 * ATR(14)
  AND no position currently open
  Entry at the signal bar's Close.

Exits (first to fire on a later bar — SL has priority within the same bar):
  SL : Low[i] <= VWAP[i]   → fill at VWAP[i].
       Initial SL is ~1 ATR below entry. If VWAP rises above entry, the
       "SL" becomes a profit-locking trail; if VWAP falls, SL falls too.
  TP : High[i] >= entry + N pips → fill at entry + N pips.

N is swept over {1, 2, 3, 5} pips.  No time stop.  Spread 1 pip per trade.
Daily-anchored VWAP, Wilder ATR(14).

Windows:
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
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / ATR_LEN, adjust=False).mean()

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
    vwap  = df["vwap"].values
    n     = len(df)
    tp_d  = n_pips * PIP_SIZE

    trades = []
    position = 0
    entry_px = np.nan
    entry_i  = -1
    mae_px   = np.nan

    for i in range(n):
        if position == 1 and i > entry_i:
            mae_px = min(mae_px, low[i])

            stop_hit = np.isfinite(vwap[i]) and (low[i] <= vwap[i])
            tp_hit   = high[i] >= entry_px + tp_d

            if stop_hit:
                exit_px, kind = vwap[i], "stop"
            elif tp_hit:
                exit_px, kind = entry_px + tp_d, "tp"
            else:
                exit_px, kind = None, None

            if kind is not None:
                trades.append({
                    "entry_i": entry_i, "exit_i": i,
                    "entry_px": entry_px, "exit_px": exit_px,
                    "pips": (exit_px - entry_px) / PIP_SIZE,
                    "mae_pips": (mae_px - entry_px) / PIP_SIZE,
                    "hold": i - entry_i, "kind": kind, "status": "closed",
                })
                position = 0
                continue

        if position == 0 and sig[i]:
            position = 1
            entry_px = close[i]
            entry_i  = i
            mae_px   = close[i]

    if position == 1:
        tail_low = low[entry_i + 1:].min() if entry_i + 1 < n else entry_px
        mae_px   = min(mae_px, tail_low)
        trades.append({
            "entry_i": entry_i, "exit_i": n - 1,
            "entry_px": entry_px, "exit_px": close[-1],
            "pips": (close[-1] - entry_px) / PIP_SIZE,
            "mae_pips": (mae_px - entry_px) / PIP_SIZE,
            "hold": n - 1 - entry_i, "kind": "open", "status": "open",
        })
    return trades


def summarize(trades, df, n_pips):
    closed = [t for t in trades if t["status"] == "closed"]
    open_  = [t for t in trades if t["status"] == "open"]
    if not closed and not open_:
        return None

    tp_t   = [t for t in closed if t["kind"] == "tp"]
    stop_t = [t for t in closed if t["kind"] == "stop"]
    stop_wins   = [t for t in stop_t if t["pips"] > 0]
    stop_losses = [t for t in stop_t if t["pips"] <= 0]

    pn_closed = np.array([t["pips"] for t in closed]) - SPREAD_PIPS if closed else np.array([])
    pn_open   = np.array([t["pips"] for t in open_])  - SPREAD_PIPS if open_  else np.array([])
    wins      = pn_closed > 0
    days      = max(1, (df.index[-1] - df.index[0]).days)
    sharpe    = (pn_closed.mean() / pn_closed.std()
                 * np.sqrt(len(pn_closed) / (days / 365.25))
                 if len(pn_closed) > 1 and pn_closed.std() > 0 else 0.0)

    return {
        "n_pips":          n_pips,
        "n_closed":        len(closed),
        "n_open":          len(open_),
        "n_tp":            len(tp_t),
        "n_stop":          len(stop_t),
        "n_stop_wins":     len(stop_wins),
        "n_stop_losses":   len(stop_losses),
        "pips_realized":   float(pn_closed.sum()) if closed else 0.0,
        "pips_unrealized": float(pn_open.sum())   if open_  else 0.0,
        "win_pct":         float(wins.mean() * 100) if closed else 0.0,
        "avg_win":         float(pn_closed[wins].mean())  if wins.any()  else 0.0,
        "avg_loss":        float(pn_closed[~wins].mean()) if (~wins).any() else 0.0,
        "avg_stop_loss":   float(np.mean([t["pips"] - SPREAD_PIPS for t in stop_losses])) if stop_losses else 0.0,
        "avg_stop_win":    float(np.mean([t["pips"] - SPREAD_PIPS for t in stop_wins]))   if stop_wins   else 0.0,
        "worst_trade":     float(pn_closed.min()) if closed else 0.0,
        "best_trade":      float(pn_closed.max()) if closed else 0.0,
        "avg_hold":        float(np.mean([t["hold"] for t in closed])) if closed else 0.0,
        "max_hold":        int(max((t["hold"] for t in closed), default=0)),
        "sharpe":          float(sharpe),
        "median_mae":      float(np.median([t["mae_pips"] for t in closed])) if closed else 0.0,
        "worst_mae":       float(min((t["mae_pips"] for t in closed), default=0.0)),
    }


def per_month(trades, df):
    if not trades:
        return []
    idx  = df.index
    rows = []
    months = sorted({(idx[t["entry_i"]].year, idx[t["entry_i"]].month) for t in trades})
    for ym in months:
        sub = [t for t in trades if (idx[t["entry_i"]].year, idx[t["entry_i"]].month) == ym]
        cl  = [t for t in sub if t["status"] == "closed"]
        op  = [t for t in sub if t["status"] == "open"]
        rows.append({
            "ym":        f"{ym[0]:04d}-{ym[1]:02d}",
            "n_closed":  len(cl),
            "n_tp":      sum(t["kind"] == "tp"   for t in cl),
            "n_stop":    sum(t["kind"] == "stop" for t in cl),
            "n_open":    len(op),
            "pips_real": float(sum(t["pips"] for t in cl) - SPREAD_PIPS * len(cl)),
            "pips_unr":  float(sum(t["pips"] for t in op) - SPREAD_PIPS * len(op)),
            "worst_mae": float(min((t["mae_pips"] for t in sub), default=0.0)),
        })
    return rows


def run_window(label, start, end):
    print(f"\n{'='*100}\n{label}:  {start}  ..  {end}\n{'='*100}")
    df = get_data("EURUSD", "5m", start, end)
    if df.empty:
        print("  (no data)")
        return
    df  = add_indicators(df)
    sig = entry_signal(df)
    print(f"  bars: {len(df)}   entry signals: {int(sig.sum())}")

    print(f"\n  {'N':>3} {'Cl':>4} {'Op':>3} {'TP':>4} {'Stp':>4} {'StpW':>5} {'StpL':>5} "
          f"{'Real':>8} {'Unrl':>7} {'Total':>8} {'Win%':>6} "
          f"{'AvgW':>6} {'AvgL':>7} {'StpL':>7} {'Hold':>6} {'MaxH':>6} "
          f"{'MedMAE':>7} {'WstMAE':>7} {'Shp':>6}")
    print("  " + "-" * 140)
    primary_trades = None
    for n_pips in N_SWEEP:
        trades = backtest(df, sig, n_pips)
        s = summarize(trades, df, n_pips)
        if s is None:
            print(f"  {n_pips:>3}  (no data)")
            continue
        total = s["pips_realized"] + s["pips_unrealized"]
        print(f"  {n_pips:>3} {s['n_closed']:>4} {s['n_open']:>3} "
              f"{s['n_tp']:>4} {s['n_stop']:>4} {s['n_stop_wins']:>5} "
              f"{s['n_stop_losses']:>5} {s['pips_realized']:>+8.1f} "
              f"{s['pips_unrealized']:>+7.1f} {total:>+8.1f} "
              f"{s['win_pct']:>5.1f}% {s['avg_win']:>+6.1f} "
              f"{s['avg_loss']:>+7.1f} {s['avg_stop_loss']:>+7.1f} "
              f"{s['avg_hold']:>6.1f} {s['max_hold']:>6d} "
              f"{s['median_mae']:>+7.1f} {s['worst_mae']:>+7.1f} "
              f"{s['sharpe']:>+6.2f}")
        if n_pips == 2:
            primary_trades = trades

    if primary_trades is None:
        return
    print(f"\n  Per-month @ N=2:")
    print(f"  {'Month':>8} {'Cl':>4} {'TP':>4} {'Stp':>4} {'Op':>3} "
          f"{'Real':>8} {'Unrl':>8} {'WstMAE':>7}")
    print("  " + "-" * 60)
    for r in per_month(primary_trades, df):
        print(f"  {r['ym']:>8} {r['n_closed']:>4} {r['n_tp']:>4} "
              f"{r['n_stop']:>4} {r['n_open']:>3} {r['pips_real']:>+8.1f} "
              f"{r['pips_unr']:>+8.1f} {r['worst_mae']:>+7.1f}")


print(f"Strategy 012 — long VWAP+ATR breakout, TP=entry+N, SL=current VWAP")
print(f"  entry: green & Close > VWAP + {ATR_MULT}*ATR({ATR_LEN})    spread {SPREAD_PIPS} pip")

run_window("OUT-OF-SAMPLE", "2025-11-15", "2026-02-15")
run_window("IN-SAMPLE",     "2026-02-15", "2026-05-15")
