"""
Strategy 020 — EMA(50) trend + daily-candle confirmation on EURUSD.
Tested independently on 5m and 1h.

Entry rules (at the close of bar t on the trade timeframe):
  LONG  when EMA(50)[t] > EMA(50)[t-5]
             AND the previous completed daily candle is green
             AND the current (forming) daily candle is green so far
  SHORT when EMA(50)[t] < EMA(50)[t-5]
             AND the previous completed daily candle is red
             AND the current (forming) daily candle is red so far

A daily candle is "green" if d_close > d_open, "red" if d_close < d_open.
For the current (forming) day, d_close is the Close of bar t and d_open
is the Open of the first intraday bar of the same UTC date.

Exit: SL = entry -/+ 2 * ATR(14), TP = entry +/- 6 * ATR(14).
ATR(14) is computed on the trade timeframe and fixed at entry. A 1-pip
spread is deducted at exit. A generous MAX_HOLD prevents stale trades.
"""
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PIP_SIZE    = 0.0001
SPREAD_PIPS = 1.0
SL_ATR      = 2.0
TP_ATR      = 6.0
EMA_LEN     = 50
EMA_LOOKBK  = 5     # slope is EMA[t] vs EMA[t-EMA_LOOKBK]
ATR_LEN     = 14

# MAX_HOLD per timeframe (cap to avoid stale trades; 6*ATR can take a while).
MAX_HOLD = {
    "5m": 5000,    # ~17 trading days
    "1h": 500,     # ~3 trading weeks
}

_TF = {
    "5m": mt5.TIMEFRAME_M5,
    "1h": mt5.TIMEFRAME_H1,
}


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def get_data(ticker: str, tf: str, start: str, end: str) -> pd.DataFrame:
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end   = pd.Timestamp(end,   tz="UTC")
    mt5.initialize()
    mt5.symbol_select(ticker, True)

    CHUNK = 50_000
    frames, pos = [], 0
    while True:
        chunk = mt5.copy_rates_from_pos(ticker, _TF[tf], pos, CHUNK)
        if chunk is None or len(chunk) == 0:
            break
        frames.append(pd.DataFrame(chunk))
        oldest = pd.Timestamp(int(frames[-1]["time"].min()), unit="s", tz="UTC")
        if oldest <= ts_start or len(chunk) < CHUNK:
            break
        pos += CHUNK

    df = pd.concat(frames[::-1]).drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(columns={
        "open": "Open", "high": "High",
        "low": "Low",   "close": "Close", "tick_volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]]


# -----------------------------------------------------------------------------
# Indicators
# -----------------------------------------------------------------------------
def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    tr = np.empty(len(close))
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr[1:] = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prev_close),
        np.abs(low[1:]  - prev_close),
    ])
    atr = np.empty(len(close))
    atr[:n] = np.nan
    atr[n - 1] = tr[:n].mean()
    for i in range(n, len(close)):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1.0)
    out = np.empty(len(x)); out[:n - 1] = np.nan
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


# -----------------------------------------------------------------------------
# Signals
# -----------------------------------------------------------------------------
def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].values
    high  = out["High"].values
    low   = out["Low"].values

    # ATR(14) on the trade timeframe.
    out["atr"] = _wilder_atr(high, low, close, n=ATR_LEN)

    # EMA(50) slope: EMA[t] > EMA[t - EMA_LOOKBK] → up, else down.
    ema = _ema(close, n=EMA_LEN)
    ema_s = pd.Series(ema, index=out.index)
    out["ema_up"]   = (ema_s > ema_s.shift(EMA_LOOKBK)).astype("Int8")
    out["ema_down"] = (ema_s < ema_s.shift(EMA_LOOKBK)).astype("Int8")

    # Daily candle filter.
    # Group by UTC date; today's daily open = first intraday Open of the day.
    date = out.index.normalize()
    daily_open_today = out.groupby(date)["Open"].transform("first")
    out["today_green"] = (out["Close"] > daily_open_today).astype("Int8")
    out["today_red"]   = (out["Close"] < daily_open_today).astype("Int8")

    # Previous completed daily candle color.
    daily = out.groupby(date).agg(d_open=("Open", "first"),
                                  d_close=("Close", "last"))
    daily["prev_green"] = (daily["d_close"].shift(1) > daily["d_open"].shift(1)).astype("Int8")
    daily["prev_red"]   = (daily["d_close"].shift(1) < daily["d_open"].shift(1)).astype("Int8")

    # Map per-day flags onto intraday bars by date (preserves DatetimeIndex).
    out["prev_green"] = pd.Series(date, index=out.index).map(daily["prev_green"]).astype("Int8")
    out["prev_red"]   = pd.Series(date, index=out.index).map(daily["prev_red"]).astype("Int8")

    out["enter_long"]  = ((out["ema_up"]   == 1) &
                         (out["today_green"] == 1) &
                         (out["prev_green"]  == 1)).astype(bool)
    out["enter_short"] = ((out["ema_down"] == 1) &
                         (out["today_red"]   == 1) &
                         (out["prev_red"]    == 1)).astype(bool)
    return out


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------
def simulate(df: pd.DataFrame, max_hold: int) -> dict:
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    atr   = df["atr"].values
    el    = df["enter_long"].values
    es    = df["enter_short"].values
    n     = len(df)

    pos      = 0     # 0 flat, +1 long, -1 short
    entry_px = 0.0
    sl_px    = 0.0
    tp_px    = 0.0
    entry_i  = 0
    trades: list[dict] = []

    for i in range(n):
        if pos == 1:
            held = i - entry_i
            if low[i] <= sl_px:
                pips = (sl_px - entry_px) / PIP_SIZE - SPREAD_PIPS
                trades.append({"pips": pips, "hold": held, "exit": "sl", "dir": "L"})
                pos = 0
                continue
            if high[i] >= tp_px:
                pips = (tp_px - entry_px) / PIP_SIZE - SPREAD_PIPS
                trades.append({"pips": pips, "hold": held, "exit": "tp", "dir": "L"})
                pos = 0
                continue
            if held >= max_hold:
                pips = (close[i] - entry_px) / PIP_SIZE - SPREAD_PIPS
                trades.append({"pips": pips, "hold": held, "exit": "time", "dir": "L"})
                pos = 0
                continue
        elif pos == -1:
            held = i - entry_i
            if high[i] >= sl_px:
                pips = (entry_px - sl_px) / PIP_SIZE - SPREAD_PIPS
                trades.append({"pips": pips, "hold": held, "exit": "sl", "dir": "S"})
                pos = 0
                continue
            if low[i] <= tp_px:
                pips = (entry_px - tp_px) / PIP_SIZE - SPREAD_PIPS
                trades.append({"pips": pips, "hold": held, "exit": "tp", "dir": "S"})
                pos = 0
                continue
            if held >= max_hold:
                pips = (entry_px - close[i]) / PIP_SIZE - SPREAD_PIPS
                trades.append({"pips": pips, "hold": held, "exit": "time", "dir": "S"})
                pos = 0
                continue

        if pos == 0:
            a = atr[i]
            if not np.isfinite(a) or a <= 0:
                continue
            if el[i]:
                pos      = 1
                entry_px = close[i]
                entry_i  = i
                sl_px    = entry_px - SL_ATR * a
                tp_px    = entry_px + TP_ATR * a
            elif es[i]:
                pos      = -1
                entry_px = close[i]
                entry_i  = i
                sl_px    = entry_px + SL_ATR * a
                tp_px    = entry_px - TP_ATR * a

    return _summarize(trades, df.index)


def _summarize(trades: list[dict], index: pd.DatetimeIndex) -> dict:
    if not trades:
        return {"n": 0}
    pips = np.array([t["pips"] for t in trades])
    cum  = np.cumsum(pips)
    days = max(1, (index[-1] - index[0]).days)
    wins = pips > 0
    sharpe = (pips.mean() / pips.std() * np.sqrt(len(trades) / (days / 365.25))
              if len(trades) > 1 and pips.std() > 0 else 0.0)
    return {
        "n":        len(trades),
        "pips":     float(pips.sum()),
        "win_pct":  float(wins.mean() * 100),
        "avg_win":  float(pips[wins].mean())  if wins.any()  else 0.0,
        "avg_loss": float(pips[~wins].mean()) if (~wins).any() else 0.0,
        "max_dd":   float((cum - np.maximum.accumulate(cum)).min()),
        "avg_hold": float(np.mean([t["hold"] for t in trades])),
        "sharpe":   float(sharpe),
        "tp_pct":   float(np.mean([t["exit"] == "tp"   for t in trades]) * 100),
        "sl_pct":   float(np.mean([t["exit"] == "sl"   for t in trades]) * 100),
        "time_pct": float(np.mean([t["exit"] == "time" for t in trades]) * 100),
        "long_pct": float(np.mean([t["dir"]  == "L"    for t in trades]) * 100),
        "trades":   trades,
    }


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------
def _print_row(label: str, m: dict) -> None:
    if m.get("n", 0) == 0:
        print(f"{label:>6} {'-':>5}")
        return
    print(f"{label:>6} {m['n']:>5} {m['pips']:>+9.1f} "
          f"{m['sharpe']:>+7.2f} {m['win_pct']:>5.1f}% "
          f"{m['avg_win']:>+6.1f} {m['avg_loss']:>+7.1f} "
          f"{m['max_dd']:>+8.1f} {m['avg_hold']:>6.1f} "
          f"{m['tp_pct']:>4.1f}% {m['sl_pct']:>4.1f}% {m['time_pct']:>4.1f}% "
          f"{m['long_pct']:>4.1f}%")


def report(tf: str, m_all: dict, m_long: dict, m_short: dict) -> None:
    hdr = (f"{'side':>6} {'n':>5} {'pips':>9} {'sharpe':>7} {'win%':>6} "
           f"{'avgW':>6} {'avgL':>7} {'maxDD':>8} {'hold':>6} "
           f"{'tp%':>5} {'sl%':>5} {'tim%':>5} {'L%':>5}")
    print(f"\n[{tf}] results")
    print(hdr)
    print("-" * len(hdr))
    _print_row("all",   m_all)
    _print_row("long",  m_long)
    _print_row("short", m_short)


def split_by_dir(m: dict, direction: str) -> dict:
    if m.get("n", 0) == 0:
        return {"n": 0}
    sub = [t for t in m["trades"] if t["dir"] == direction]
    if not sub:
        return {"n": 0}
    # Reuse summarize on the subset; need a synthetic index span = original.
    pips = np.array([t["pips"] for t in sub])
    cum  = np.cumsum(pips)
    # Annualize using the same wall-clock span as the all-trades run.
    wins = pips > 0
    # We don't have the bar index here, so we approximate Sharpe using the
    # subset itself; it's a diagnostic split, not the headline number.
    sharpe = (pips.mean() / pips.std() * np.sqrt(len(sub))
              if len(sub) > 1 and pips.std() > 0 else 0.0)
    return {
        "n":        len(sub),
        "pips":     float(pips.sum()),
        "win_pct":  float(wins.mean() * 100),
        "avg_win":  float(pips[wins].mean())  if wins.any()  else 0.0,
        "avg_loss": float(pips[~wins].mean()) if (~wins).any() else 0.0,
        "max_dd":   float((cum - np.maximum.accumulate(cum)).min()),
        "avg_hold": float(np.mean([t["hold"] for t in sub])),
        "sharpe":   float(sharpe),
        "tp_pct":   float(np.mean([t["exit"] == "tp"   for t in sub]) * 100),
        "sl_pct":   float(np.mean([t["exit"] == "sl"   for t in sub]) * 100),
        "time_pct": float(np.mean([t["exit"] == "time" for t in sub]) * 100),
        "long_pct": 100.0 if direction == "L" else 0.0,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_tf(ticker: str, tf: str, start: str, end: str) -> None:
    print(f"\nLoading {ticker} {tf} ({start} .. {end})...")
    df = get_data(ticker, tf, start, end)
    print(f"  bars: {len(df)}  ({df.index[0]} .. {df.index[-1]})")

    sig = build_signals(df)
    # Drop rows where any required signal input is missing.
    sig = sig.dropna(subset=["atr", "prev_green", "prev_red"]).copy()
    n_long  = int(sig["enter_long"].sum())
    n_short = int(sig["enter_short"].sum())
    print(f"  signal bars: long={n_long}  short={n_short} (pre-position-filter)")

    m_all = simulate(sig, MAX_HOLD[tf])
    m_long  = split_by_dir(m_all, "L")
    m_short = split_by_dir(m_all, "S")
    report(tf, m_all, m_long, m_short)


if __name__ == "__main__":
    TICKER = "EURUSD"
    START  = "2025-01-01"
    END    = "2026-05-01"

    for tf in ("1h", "5m"):
        run_tf(TICKER, tf, START, END)
