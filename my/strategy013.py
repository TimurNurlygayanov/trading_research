"""
Strategy 013 — Mean-reversion flip: fade green-bar VWAP overextensions.

Entry (SHORT only):
  green candle (Close > Open)
  AND Close > VWAP + K * ATR(14)
  AND no position currently open
  Entry at the signal bar's Close.

  K (entry multiplier) is swept over {1.0, 1.2, 2.0}.

Exits (first to fire — SL has priority within same bar):
  TP : Low[i] <= VWAP[i]            → fill at VWAP[i].   (mean-reversion target.)
  SL : High[i] >= entry + 1*ATR_at_entry → fill at that fixed price.

So distances are:
  TP distance ~ K * ATR (entry was K*ATR above VWAP).
  SL distance =  1 * ATR (fixed at entry time).
  Theoretical R:R = K : 1.

Daily-anchored VWAP, Wilder ATR(14). Spread: 1 pip per closed trade.

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


PIP_SIZE       = 0.0001
SPREAD_PIPS    = 1.0
ATR_LEN        = 14
SL_ATR_MULT    = 1.0
ENTRY_K_SWEEP  = [1.0, 1.2, 2.0]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c = df["High"], df["Low"], df["Close"]
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


def entry_signal(df: pd.DataFrame, k: float) -> np.ndarray:
    green        = (df["Close"] > df["Open"]).to_numpy()
    above_thresh = (df["Close"] > df["vwap"] + k * df["atr"]).to_numpy()
    have_atr     = np.isfinite(df["atr"].to_numpy())
    return green & above_thresh & have_atr


def backtest(df: pd.DataFrame, sig: np.ndarray):
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    vwap  = df["vwap"].values
    atr   = df["atr"].values
    n     = len(df)

    trades = []
    position = 0
    entry_px = sl_px = np.nan
    entry_i  = -1
    mae_px   = np.nan

    for i in range(n):
        if position == -1 and i > entry_i:
            mae_px = max(mae_px, high[i])

            stop_hit = high[i] >= sl_px
            tp_hit   = np.isfinite(vwap[i]) and (low[i] <= vwap[i])

            if stop_hit:
                exit_px, kind = sl_px, "stop"
            elif tp_hit:
                exit_px, kind = vwap[i], "tp"
            else:
                exit_px, kind = None, None

            if kind is not None:
                trades.append({
                    "entry_i": entry_i, "exit_i": i,
                    "entry_px": entry_px, "exit_px": exit_px,
                    "pips": (entry_px - exit_px) / PIP_SIZE,   # short pnl
                    "mae_pips": (entry_px - mae_px) / PIP_SIZE,
                    "hold": i - entry_i, "kind": kind, "status": "closed",
                })
                position = 0
                continue

        if position == 0 and sig[i] and np.isfinite(atr[i]):
            position  = -1
            entry_px  = close[i]
            entry_i   = i
            sl_px     = entry_px + SL_ATR_MULT * atr[i]
            mae_px    = close[i]

    if position == -1:
        tail_high = high[entry_i + 1:].max() if entry_i + 1 < n else entry_px
        mae_px    = max(mae_px, tail_high)
        trades.append({
            "entry_i": entry_i, "exit_i": n - 1,
            "entry_px": entry_px, "exit_px": close[-1],
            "pips": (entry_px - close[-1]) / PIP_SIZE,
            "mae_pips": (entry_px - mae_px) / PIP_SIZE,
            "hold": n - 1 - entry_i, "kind": "open", "status": "open",
        })
    return trades


def summarize(trades, df):
    closed = [t for t in trades if t["status"] == "closed"]
    open_  = [t for t in trades if t["status"] == "open"]
    if not closed and not open_:
        return None

    tp_t   = [t for t in closed if t["kind"] == "tp"]
    stop_t = [t for t in closed if t["kind"] == "stop"]

    pn_closed = np.array([t["pips"] for t in closed]) - SPREAD_PIPS if closed else np.array([])
    pn_open   = np.array([t["pips"] for t in open_])  - SPREAD_PIPS if open_  else np.array([])
    wins      = pn_closed > 0
    days      = max(1, (df.index[-1] - df.index[0]).days)
    sharpe    = (pn_closed.mean() / pn_closed.std()
                 * np.sqrt(len(pn_closed) / (days / 365.25))
                 if len(pn_closed) > 1 and pn_closed.std() > 0 else 0.0)

    return {
        "n_closed":        len(closed),
        "n_open":          len(open_),
        "n_tp":            len(tp_t),
        "n_stop":          len(stop_t),
        "pips_realized":   float(pn_closed.sum()) if closed else 0.0,
        "pips_unrealized": float(pn_open.sum())   if open_  else 0.0,
        "win_pct":         float(wins.mean() * 100) if closed else 0.0,
        "avg_win":         float(pn_closed[wins].mean())  if wins.any()  else 0.0,
        "avg_loss":        float(pn_closed[~wins].mean()) if (~wins).any() else 0.0,
        "avg_tp_win":      float(np.mean([t["pips"] - SPREAD_PIPS for t in tp_t]))   if tp_t   else 0.0,
        "avg_stop_loss":   float(np.mean([t["pips"] - SPREAD_PIPS for t in stop_t])) if stop_t else 0.0,
        "best_trade":      float(pn_closed.max()) if closed else 0.0,
        "worst_trade":     float(pn_closed.min()) if closed else 0.0,
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
    print(f"\n{'='*110}\n{label}:  {start}  ..  {end}\n{'='*110}")
    df = get_data("EURUSD", "5m", start, end)
    if df.empty:
        print("  (no data)")
        return
    df = add_indicators(df)
    print(f"  bars: {len(df)}")

    print(f"\n  {'K':>4} {'Sig':>5} {'Cl':>5} {'Op':>3} {'TP':>5} {'Stp':>5} "
          f"{'Real':>8} {'Unrl':>7} {'Total':>8} {'Win%':>6} "
          f"{'AvgTP':>6} {'AvgSL':>7} {'Hold':>5} {'MaxH':>5} "
          f"{'MedMAE':>7} {'WstMAE':>7} {'Shp':>6}")
    print("  " + "-" * 120)

    best_by_total = None
    primary_K = ENTRY_K_SWEEP[0]
    primary_trades = None

    for k in ENTRY_K_SWEEP:
        sig = entry_signal(df, k)
        trades = backtest(df, sig)
        s = summarize(trades, df)
        if s is None:
            print(f"  {k:>4.1f}  (no data)")
            continue
        total = s["pips_realized"] + s["pips_unrealized"]
        print(f"  {k:>4.1f} {int(sig.sum()):>5d} {s['n_closed']:>5} {s['n_open']:>3} "
              f"{s['n_tp']:>5} {s['n_stop']:>5} "
              f"{s['pips_realized']:>+8.1f} {s['pips_unrealized']:>+7.1f} "
              f"{total:>+8.1f} {s['win_pct']:>5.1f}% "
              f"{s['avg_tp_win']:>+6.1f} {s['avg_stop_loss']:>+7.1f} "
              f"{s['avg_hold']:>5.1f} {s['max_hold']:>5d} "
              f"{s['median_mae']:>+7.1f} {s['worst_mae']:>+7.1f} "
              f"{s['sharpe']:>+6.2f}")
        if best_by_total is None or total > best_by_total[1]:
            best_by_total = (k, total, trades)
        if k == primary_K:
            primary_trades = trades

    if best_by_total is None:
        return
    bk, btot, btrades = best_by_total
    print(f"\n  Per-month @ best K={bk}  (total {btot:+.1f} pips):")
    print(f"  {'Month':>8} {'Cl':>4} {'TP':>4} {'Stp':>4} {'Op':>3} "
          f"{'Real':>8} {'Unrl':>8} {'WstMAE':>7}")
    print("  " + "-" * 60)
    for r in per_month(btrades, df):
        print(f"  {r['ym']:>8} {r['n_closed']:>4} {r['n_tp']:>4} "
              f"{r['n_stop']:>4} {r['n_open']:>3} {r['pips_real']:>+8.1f} "
              f"{r['pips_unr']:>+8.1f} {r['worst_mae']:>+7.1f}")


print(f"Strategy 013 — mean-reversion SHORT fade of VWAP+K*ATR breakouts")
print(f"  entry: green & Close > VWAP + K*ATR    TP at VWAP    SL at entry + {SL_ATR_MULT}*ATR")
print(f"  K sweep: {ENTRY_K_SWEEP}   spread {SPREAD_PIPS} pip")

run_window("OUT-OF-SAMPLE", "2025-11-15", "2026-02-15")
run_window("IN-SAMPLE",     "2026-02-15", "2026-05-15")
