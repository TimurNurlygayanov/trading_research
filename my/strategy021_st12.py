"""
Strategy 021a — SuperTrend(ATR=1.2) on EURUSD 1h.
Entry on green SuperTrend + daily confirmation.

Entry rules (at the close of bar t on 1h):
  LONG when SuperTrend is green
         AND the previous completed daily candle is green
         AND the current (forming) daily candle is green so far

Exit: SL = SuperTrend level, TP = entry + 3*(entry - SL).
"""
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Constants
PIP_SIZE    = 0.0001
SPREAD_PIPS = 1.0
ST_PERIOD   = 10
ST_MULT     = 1.2
ATR_LEN     = 14
MAX_HOLD    = 500  # ~3 trading weeks on 1h

_TF = {"1h": mt5.TIMEFRAME_H1}


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


def _supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int, mult: float) -> tuple[np.ndarray, np.ndarray]:
    hl2 = (high + low) / 2.0
    atr = _wilder_atr(high, low, close, n=period)

    basic_ub = hl2 + mult * atr
    basic_lb = hl2 - mult * atr

    n = len(close)
    ub = np.full(n, np.nan)
    lb = np.full(n, np.nan)
    st = np.full(n, np.nan)
    trend = np.zeros(n, dtype=np.int8)

    # First valid ATR index
    start = period - 1
    while start < n and not np.isfinite(atr[start]):
        start += 1
    if start >= n:
        return st, np.zeros(n, dtype=np.int8)

    ub[start] = basic_ub[start]
    lb[start] = basic_lb[start]
    if close[start] > ub[start]:
        trend[start] = 1
        st[start] = lb[start]
    else:
        trend[start] = -1
        st[start] = ub[start]

    for i in range(start + 1, n):
        ub[i] = basic_ub[i] if basic_ub[i] < ub[i-1] or high[i-1] > ub[i-1] else ub[i-1]
        lb[i] = basic_lb[i] if basic_lb[i] > lb[i-1] or low[i-1]  < lb[i-1] else lb[i-1]

        if trend[i-1] == 1:
            st[i] = lb[i]
            if close[i] <= st[i]:
                trend[i] = -1
                st[i] = ub[i]
            else:
                trend[i] = 1
        else:
            st[i] = ub[i]
            if close[i] >= st[i]:
                trend[i] = 1
                st[i] = lb[i]
            else:
                trend[i] = -1

    is_green = (trend == 1).astype(np.int8)
    return st, is_green


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].values
    high  = out["High"].values
    low   = out["Low"].values

    st, st_green = _supertrend(high, low, close, ST_PERIOD, ST_MULT)
    out["st"]      = st
    out["st_green"] = st_green

    date = out.index.normalize()
    daily_open_today = out.groupby(date)["Open"].transform("first")
    out["today_green"] = (out["Close"] > daily_open_today).astype("Int8")

    daily = out.groupby(date).agg(d_open=("Open", "first"),
                                  d_close=("Close", "last"))
    daily["prev_green"] = (daily["d_close"].shift(1) > daily["d_open"].shift(1)).astype("Int8")

    out["prev_green"] = pd.Series(date, index=out.index).map(daily["prev_green"]).astype("Int8")

    out["enter_long"] = ((out["st_green"] == 1) &
                         (out["today_green"] == 1) &
                         (out["prev_green"]  == 1)).astype(bool)
    return out


def simulate(df: pd.DataFrame, max_hold: int) -> dict:
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    st    = df["st"].values
    el    = df["enter_long"].values
    n     = len(df)

    pos      = 0
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

        if pos == 0 and el[i]:
            if np.isfinite(st[i]) and st[i] > 0:
                pos      = 1
                entry_px = close[i]
                entry_i  = i
                sl_px    = st[i]
                risk     = entry_px - sl_px
                tp_px    = entry_px + 3.0 * risk

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


def _print_row(label: str, m: dict) -> None:
    if m.get("n", 0) == 0:
        print(f"{label:>6} {'-':>5}")
        return
    print(f"{label:>6} {m['n']:>5} {m['pips']:>+9.1f} "
          f"{m['sharpe']:>+7.2f} {m['win_pct']:>5.1f}% "
          f"{m['avg_win']:>+6.1f} {m['avg_loss']:>+7.1f} "
          f"{m['max_dd']:>+8.1f} {m['avg_hold']:>6.1f} "
          f"{m['tp_pct']:>4.1f}% {m['sl_pct']:>4.1f}% {m['time_pct']:>4.1f}%")


def report(m_all: dict) -> None:
    hdr = (f"{'side':>6} {'n':>5} {'pips':>9} {'sharpe':>7} {'win%':>6} "
           f"{'avgW':>6} {'avgL':>7} {'maxDD':>8} {'hold':>6} "
           f"{'tp%':>5} {'sl%':>5} {'tim%':>5}")
    print(f"\n[1h] SuperTrend(ST={ST_MULT}) results")
    print(hdr)
    print("-" * len(hdr))
    _print_row("all", m_all)


if __name__ == "__main__":
    TICKER = "EURUSD"
    START  = "2025-01-01"
    END    = "2026-05-01"

    print(f"\nLoading {TICKER} 1h ({START} .. {END})...")
    df = get_data(TICKER, "1h", START, END)
    print(f"  bars: {len(df)}  ({df.index[0]} .. {df.index[-1]})")

    sig = build_signals(df)
    sig = sig.dropna(subset=["st", "prev_green"]).copy()
    n_long = int(sig["enter_long"].sum())
    print(f"  signal bars: long={n_long} (pre-position-filter)")

    m_all = simulate(sig, MAX_HOLD)
    report(m_all)
