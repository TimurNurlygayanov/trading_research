"""
Strategy 009 — EURUSD 5m, slower-EMA(HL/2) vs daily-anchored VWAP cross.
                EMA-length × TP grid sweep on top of strategy 007's machinery.

Entry / exit logic is identical to strategy 007:
  - EMA(N) of (High+Low)/2 vs daily-anchored VWAP (typical * tick_volume).
  - Enter on cross at bar close. No SL.
  - Exit on first of: intra-bar TP touch, or opposite cross at bar close (flip).
  - First bar of each UTC day is excluded from cross signals (VWAP-reset artifact).

Sweep:
  EMA lengths : 15, 20, 30, 50, 80, 100, 150, 200
  TPs (pips)  : 10, 20, 30, 50, and "no TP" (only flip exits)

Period: 2026-02-15 .. 2026-05-15.  Spread: 1 pip per trade.
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
EMA_SWEEP   = [15, 20, 30, 50, 80, 100, 150, 200]
TP_SWEEP    = [10, 20, 30, 50, None]   # None = no TP, flip-only


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Daily-anchored VWAP added once; EMA is recomputed per sweep."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol     = df["Volume"].astype(float)
    day     = df.index.normalize()
    pv      = (typical * vol).groupby(day).cumsum()
    cv      = vol.groupby(day).cumsum().replace(0, np.nan)
    df["vwap"] = pv / cv
    return df


def crosses_for_ema(df: pd.DataFrame, ema_len: int):
    hl2  = (df["High"] + df["Low"]) / 2
    ema  = hl2.ewm(span=ema_len, adjust=False).mean()
    diff = ema - df["vwap"]
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
    """tp_pips: float or None (no TP — exits only on flip / EOD)."""
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(df)
    tp_d  = (tp_pips * PIP_SIZE) if tp_pips is not None else None

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
        if position != 0 and i > entry_i and tp_d is not None:
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
        return {"n": 0, "pips_net": 0.0, "win_pct": 0.0, "sharpe": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "n_tp": 0, "n_flip": 0,
                "n_eod": 0, "avg_hold": 0.0,
                "best_trade": 0.0, "worst_trade": 0.0}
    pips_gross = np.array([t["pips"] for t in trades])
    pips_net   = pips_gross - SPREAD_PIPS
    wins       = pips_net > 0
    days       = max(1, (df.index[-1] - df.index[0]).days)
    sharpe     = (pips_net.mean() / pips_net.std()
                  * np.sqrt(len(trades) / (days / 365.25))
                  if len(trades) > 1 and pips_net.std() > 0 else 0.0)
    return {
        "n":           len(trades),
        "pips_net":    float(pips_net.sum()),
        "win_pct":     float(wins.mean() * 100),
        "avg_win":     float(pips_net[wins].mean())  if wins.any()  else 0.0,
        "avg_loss":    float(pips_net[~wins].mean()) if (~wins).any() else 0.0,
        "sharpe":      float(sharpe),
        "n_tp":        sum(t["exit"] == "tp"   for t in trades),
        "n_flip":      sum(t["exit"] == "flip" for t in trades),
        "n_eod":       sum(t["exit"] == "eod"  for t in trades),
        "avg_hold":    float(np.mean([t["hold"] for t in trades])),
        "best_trade":  float(pips_net.max()),
        "worst_trade": float(pips_net.min()),
    }


# ----- run -----
TICKER = "EURUSD"
START  = "2026-02-15"
END    = "2026-05-15"

print(f"Strategy 009 — EMA-length × TP grid on {TICKER} 5m  "
      f"({START} .. {END})  spread {SPREAD_PIPS} pip\n")

print(f"Loading {TICKER} 5m ...")
df = get_data(TICKER, "5m", START, END)
if df.empty:
    raise SystemExit(f"No data for {TICKER}")
print(f"  {len(df)} bars  ({df.index.min()} .. {df.index.max()})\n")

df = add_vwap(df)

# Grid: rows = EMA, cols = TP. Print net pips, then trades, then Sharpe.
results = {}      # (ema, tp) -> summary
n_crosses = {}    # ema -> (up, down)

for ema_len in EMA_SWEEP:
    cu, cd = crosses_for_ema(df, ema_len)
    n_crosses[ema_len] = (int(cu.sum()), int(cd.sum()))
    for tp in TP_SWEEP:
        trades = backtest(df, cu, cd, tp)
        results[(ema_len, tp)] = summarize(trades, df)

tp_labels = [f"TP={tp}" if tp is not None else "NoTP" for tp in TP_SWEEP]

def print_grid(metric_key, fmt, title):
    print(f"\n{title}")
    print(f"{'EMA':>5} {'Up/Dn':>9}  " + "  ".join(f"{lbl:>9}" for lbl in tp_labels))
    print("-" * (16 + 11 * len(TP_SWEEP)))
    for ema_len in EMA_SWEEP:
        nu, nd = n_crosses[ema_len]
        cells = []
        for tp in TP_SWEEP:
            v = results[(ema_len, tp)][metric_key]
            cells.append(fmt.format(v))
        print(f"{ema_len:>5} {nu:>4}/{nd:<4}  " + "  ".join(f"{c:>9}" for c in cells))

print_grid("pips_net", "{:+.1f}", "Grid: NET PIPS (after 1-pip spread per trade)")
print_grid("sharpe",   "{:+.2f}", "Grid: SHARPE (trade-level, annualized)")
print_grid("n",        "{:d}",    "Grid: TRADE COUNT")
print_grid("win_pct",  "{:.1f}",  "Grid: WIN %")

# Find best combo by net pips and show full detail.
best_key = max(results.keys(), key=lambda k: results[k]["pips_net"])
s = results[best_key]
ema_b, tp_b = best_key
tp_label = f"TP={tp_b}" if tp_b is not None else "no TP"
print(f"\nBest combo: EMA={ema_b}  {tp_label}")
print(f"  trades:   {s['n']}  (TP={s['n_tp']}  flip={s['n_flip']}  eod={s['n_eod']})")
print(f"  net pips: {s['pips_net']:+.1f}    win%: {s['win_pct']:.1f}    "
      f"sharpe: {s['sharpe']:+.2f}")
print(f"  avg win:  {s['avg_win']:+.2f}    avg loss: {s['avg_loss']:+.2f}    "
      f"avg hold: {s['avg_hold']:.1f} bars")
print(f"  best:     {s['best_trade']:+.1f}    worst:    {s['worst_trade']:+.1f}")
