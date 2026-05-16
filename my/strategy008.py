"""
Strategy 008 — EURUSD 5m, EMA9(HL/2) vs daily-anchored VWAP cross.
                Exit-on-first-profit variant of strategy 007.

Same entry as 007: enter on EMA9-vs-VWAP cross at bar close.
No SL.

Two exit modes are tested side-by-side:

  Mode A — "tiny TP" (intra-bar):
    Place a limit at entry + N pips immediately after entry.
    Exit on first intra-bar touch (H/L) at that level.
    Equivalent to a very small fixed TP. Hit-rate is high because brief
    green excursions count.

  Mode B — "first green close" (close-only):
    Wait for the FIRST BAR whose CLOSE is at least N pips above entry
    (gross, before spread). Exit at that close.
    Filters out single-bar fakes that would trigger Mode A.

Both modes still flip on opposite cross at bar close (no SL otherwise).
Sweep N (gross pips above entry) over {1, 2, 3, 4, 5}.
Spread 1 pip charged once per trade at summary time.

Period: 2026-02-15 .. 2026-05-15.
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
N_SWEEP     = [1, 2, 3, 4, 5]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    hl2     = (df["High"] + df["Low"]) / 2
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol     = df["Volume"].astype(float)

    df["ema9"] = hl2.ewm(span=EMA_LEN, adjust=False).mean()

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
    first_of_day = (df.index.normalize() != pd.Series(df.index.normalize())
                    .shift(1).fillna(pd.Timestamp("1970-01-01", tz="UTC"))
                    .to_numpy())
    cross_up   = cross_up   & ~first_of_day
    cross_down = cross_down & ~first_of_day
    return cross_up, cross_down


def backtest(df, cross_up, cross_down, threshold_pips, mode):
    """mode: 'tp' for intra-bar tiny-TP, 'close' for first-green-close."""
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(df)
    thr   = threshold_pips * PIP_SIZE

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
        # --- exit current position if it triggers on this bar ---
        if position != 0 and i > entry_i:
            hit = False
            if mode == "tp":
                if position == 1 and high[i] >= entry_px + thr:
                    exit_px, hit = entry_px + thr, True
                elif position == -1 and low[i] <= entry_px - thr:
                    exit_px, hit = entry_px - thr, True
            else:  # mode == "close"
                if position == 1 and close[i] >= entry_px + thr:
                    exit_px, hit = close[i], True
                elif position == -1 and close[i] <= entry_px - thr:
                    exit_px, hit = close[i], True
            if hit:
                book(i, exit_px, "scratch")
                position = 0
                # fall through to allow same-bar re-entry on a cross

        # --- cross signal at bar close ---
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


def summarize(trades, df, threshold, mode):
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
        "mode":        mode,
        "threshold":   threshold,
        "n":           len(trades),
        "pips_net":    float(pips_net.sum()),
        "win_pct":     float(wins.mean() * 100),
        "avg_win":     float(pips_net[wins].mean())  if wins.any()  else 0.0,
        "avg_loss":    float(pips_net[~wins].mean()) if (~wins).any() else 0.0,
        "avg_hold":    float(np.mean([t["hold"] for t in trades])),
        "sharpe":      float(sharpe),
        "n_scratch":   sum(t["exit"] == "scratch" for t in trades),
        "n_flip":      sum(t["exit"] == "flip"    for t in trades),
        "n_eod":       sum(t["exit"] == "eod"     for t in trades),
        "n_long":      sum(t["dir"] == 1  for t in trades),
        "n_short":     sum(t["dir"] == -1 for t in trades),
        "best_trade":  float(pips_net.max()),
        "worst_trade": float(pips_net.min()),
    }


# ----- run -----
TICKER = "EURUSD"
START  = "2026-02-15"
END    = "2026-05-15"

print(f"Strategy 008 — EMA9/VWAP cross + exit-on-first-profit on {TICKER} 5m")
print(f"Period: {START} .. {END}    Spread: {SPREAD_PIPS} pip\n")

print(f"Loading {TICKER} 5m ...")
df = get_data(TICKER, "5m", START, END)
if df.empty:
    raise SystemExit(f"No data for {TICKER}")
print(f"  {len(df)} bars  ({df.index.min()} .. {df.index.max()})")

df = add_indicators(df)
cross_up, cross_down = detect_crosses(df)
print(f"  cross-ups: {int(cross_up.sum())}   cross-downs: {int(cross_down.sum())}")

header = (f"{'N':>3} {'n':>5} {'L/S':>9} {'Scrt':>5} {'Flip':>5} {'EOD':>4} "
          f"{'Win%':>6} {'NetPips':>9} {'AvgW':>6} {'AvgL':>7} {'Hold':>6} "
          f"{'Sharpe':>7} {'Best':>6} {'Worst':>7}")

for mode, label in [("tp",    "Mode A — intra-bar tiny TP (limit at entry + N pips, H/L fill)"),
                    ("close", "Mode B — first bar CLOSE >= entry + N pips")]:
    print(f"\n{label}")
    print(header)
    print("-" * 100)
    for n_pips in N_SWEEP:
        trades = backtest(df, cross_up, cross_down, n_pips, mode)
        s = summarize(trades, df, n_pips, mode)
        if s is None:
            print(f"{n_pips:>3}  (no trades)")
            continue
        ls = f"{s['n_long']}/{s['n_short']}"
        print(f"{n_pips:>3} {s['n']:>5} {ls:>9} {s['n_scratch']:>5} "
              f"{s['n_flip']:>5} {s['n_eod']:>4} {s['win_pct']:>5.1f}% "
              f"{s['pips_net']:>+9.1f} {s['avg_win']:>+6.1f} "
              f"{s['avg_loss']:>+7.1f} {s['avg_hold']:>6.1f} "
              f"{s['sharpe']:>+7.2f} {s['best_trade']:>+6.1f} "
              f"{s['worst_trade']:>+7.1f}")
