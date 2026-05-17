"""
Strategy ST33 Backtest Runner — 2026 Forex Data (backtesting.py engine)

ST33 Logic (Long only):
  - SuperTrend (ATR-based) flips bullish AND current daily candle is green
  - SL:  SuperTrend level (trailing — never moves against the trade)
  - TP:  3:1 R:R from entry
  - Time window: 12:30 UTC to 21:00 UTC (1h before US session + US session)

Lookahead protections (audit at the bottom of this file):
  1. trade_on_close=False         → signal fires at bar i, fills at bar i+1 OPEN
  2. SuperTrend via pandas_ta     → vectorized but indexed view in next() never exposes future bars
  3. Daily-green = close[t] > day_open[t]   (NOT day_close, which would be lookahead)
  4. SuperTrend flip uses st_dir[-1] vs st_dir[-2] only
  5. Trailing SL updates AFTER entry (next bar onward) and only raises, never lowers
"""
import sys
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================================
# CONSTANTS
# ============================================================================

_MT5_TF_MAP = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 16385, "4h": 16388, "1d": 16408,
}

# Time window in UTC minutes-since-midnight
SESS_START_MIN = 12 * 60 + 30   # 12:30 UTC
SESS_END_MIN = 21 * 60          # 21:00 UTC


# ============================================================================
# DATA LOADING (MT5)
# ============================================================================


def get_mt5_data(
    ticker: str,
    timeframe: str = "5m",
    start: str = "2026-01-01",
    end: str = "2026-12-31",
) -> pd.DataFrame:
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")
    if not mt5.initialize():
        print(f"  ERROR: MT5 init failed for {ticker}")
        return pd.DataFrame()
    mt5.symbol_select(ticker, True)
    tf_const = _MT5_TF_MAP.get(timeframe, 5)
    CHUNK = 50_000
    frames = []
    pos = 0
    print(f"  Fetching {ticker} {timeframe}...")
    while True:
        chunk = mt5.copy_rates_from_pos(ticker, tf_const, pos, CHUNK)
        if chunk is None or len(chunk) < CHUNK:
            if chunk is not None and len(chunk) > 0:
                frames.append(pd.DataFrame(chunk))
            break
        frames.append(pd.DataFrame(chunk))
        oldest = pd.Timestamp(int(frames[-1]["time"].min()), unit="s", tz="UTC")
        if oldest <= ts_start:
            break
        pos += CHUNK
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames[::-1]).drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume",
    })
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    if not out.empty:
        print(f"  Loaded {len(out)} bars ({out.index[0]} → {out.index[-1]})")
    return out


# ============================================================================
# STRATEGY
# ============================================================================


def make_st33_strategy(st_period: int, st_mult: float, rr_ratio: float):
    """Build ST33 Strategy subclass with params closed over."""

    class _ST33(Strategy):
        def init(self):
            high = pd.Series(np.asarray(self.data.High, dtype=float))
            low = pd.Series(np.asarray(self.data.Low, dtype=float))
            close = pd.Series(np.asarray(self.data.Close, dtype=float))

            # SuperTrend via pandas_ta — fully vectorized, uses only past+current bars at each index
            st_df = ta.supertrend(high, low, close, length=st_period, multiplier=st_mult)
            st_col = f"SUPERT_{st_period}_{float(st_mult)}"
            d_col = f"SUPERTd_{st_period}_{float(st_mult)}"
            if st_df is None or st_col not in st_df.columns:
                # pandas_ta versions vary on float formatting (CLAUDE.md §5f)
                st_col = f"SUPERT_{st_period}_{st_mult}"
                d_col = f"SUPERTd_{st_period}_{st_mult}"
            st_arr = st_df[st_col].values.astype(float)
            dir_arr = st_df[d_col].values.astype(float)

            # Day-open: first Open of each calendar day, broadcast to every bar in that day.
            # NO LEAK: day_open at bar t is the OPEN of bar t's first-of-day, which is known
            # at the start of the day (long before bar t). We do NOT use day-close.
            idx = pd.DatetimeIndex(self.data.index)
            opens_series = pd.Series(np.asarray(self.data.Open, dtype=float), index=idx)
            day_open_arr = opens_series.groupby(idx.normalize()).transform("first").values

            self.st = self.I(lambda: st_arr, name="ST", overlay=True)
            self.st_dir = self.I(lambda: dir_arr, name="STd", overlay=False)
            self.day_open = self.I(lambda: day_open_arr, name="DayOpen", overlay=False)

        def next(self):
            i = len(self.data) - 1
            if i < 2:
                return

            # ── Trail SL on any open long position ───────────────────────
            # Runs every bar; only raises SL, never lowers (no leak — uses current ST value only)
            if self.position and self.position.is_long:
                st_now = float(self.st[-1])
                if not np.isnan(st_now) and self.st_dir[-1] == 1:
                    for tr in self.trades:
                        if tr.is_long and (tr.sl is None or st_now > tr.sl):
                            try:
                                tr.sl = st_now
                            except Exception:
                                pass

            # ── Time filter (UTC) ────────────────────────────────────────
            ts = self.data.index[-1]
            bar_min = ts.hour * 60 + ts.minute
            if not (SESS_START_MIN <= bar_min < SESS_END_MIN):
                return

            # Don't pyramid
            if self.position:
                return

            # ── Entry trigger: SuperTrend flip to uptrend on THIS bar ────
            # dir[-2] != 1 (was not up) AND dir[-1] == 1 (now up)
            if not (self.st_dir[-2] != 1 and self.st_dir[-1] == 1):
                return

            # ── Daily green: current close > day-open (no leak) ──────────
            close_now = float(self.data.Close[-1])
            day_open_now = float(self.day_open[-1])
            if close_now <= day_open_now:
                return

            # ── Compute SL/TP ────────────────────────────────────────────
            sl = float(self.st[-1])
            if np.isnan(sl) or sl >= close_now:
                return
            risk = close_now - sl
            tp = close_now + rr_ratio * risk

            # ── Order: market buy at NEXT bar open (trade_on_close=False) ─
            try:
                self.buy(sl=sl, tp=tp)
            except Exception:
                pass

    _ST33.__name__ = f"ST33_p{st_period}_m{st_mult}_rr{rr_ratio}"
    return _ST33


# ============================================================================
# RUNNER
# ============================================================================


def run_one(
    df: pd.DataFrame,
    st_period: int = 10,
    st_mult: float = 3.0,
    rr_ratio: float = 3.0,
    commission: float = 0.0002,
) -> Optional[dict]:
    if df.empty:
        return None
    Strat = make_st33_strategy(st_period, st_mult, rr_ratio)
    bt = Backtest(
        df, Strat,
        cash=10_000,
        commission=commission,
        trade_on_close=False,   # ← LOOKAHEAD PROTECTION: fills at next bar open
        exclusive_orders=True,
    )
    try:
        stats = bt.run()
    except Exception as e:
        print(f"  Backtest error: {e}")
        return None

    trades = stats._trades
    n = len(trades)
    if n == 0:
        return {"trades": 0, "return_pct": 0.0, "max_dd_pct": 0.0,
                "win_pct": 0.0, "pf": 0.0, "avg_pnl": 0.0, "exposure_pct": 0.0,
                "avg_bars": 0.0, "tp_hits": 0, "sl_hits": 0}

    win = float((trades["PnL"] > 0).mean()) * 100
    avg_bars = float((trades["ExitBar"] - trades["EntryBar"]).mean())
    pf = stats.get("Profit Factor", 0.0)
    pf = 0.0 if pd.isna(pf) else float(pf)

    # Classify exits: PnL > 0 → likely TP/trailing-profit; PnL ≤ 0 → SL
    tp_hits = int((trades["PnL"] > 0).sum())
    sl_hits = int((trades["PnL"] <= 0).sum())

    return {
        "trades": n,
        "return_pct": float(stats["Return [%]"]),
        "max_dd_pct": float(stats["Max. Drawdown [%]"]),
        "exposure_pct": float(stats["Exposure Time [%]"]),
        "win_pct": win,
        "pf": pf,
        "avg_pnl": float(trades["PnL"].mean()),
        "avg_bars": avg_bars,
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
    }


def main():
    print("=" * 110)
    print("ST33 BACKTEST — 2026 Forex Data (backtesting.py engine, trade_on_close=False)")
    print("=" * 110)

    USD_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD"]
    TIMEFRAMES = ["5m", "15m", "1h"]
    START = "2026-01-01"
    END = "2026-12-31"

    for tf in TIMEFRAMES:
        print(f"\n[Timeframe = {tf}]")
        print("-" * 110)
        print(f"{'Pair':<10} {'Trades':>7} {'Return%':>9} {'MaxDD%':>9} {'Win%':>7} "
              f"{'PF':>6} {'AvgPnL':>10} {'AvgBars':>9} {'Wins':>5} {'Losses':>7}")
        print("-" * 110)
        tot_n = 0
        tot_ret = 0.0
        for pair in USD_PAIRS:
            df = get_mt5_data(pair, tf, START, END)
            if df.empty:
                print(f"{pair:<10} no data")
                continue
            r = run_one(df)
            if r is None:
                print(f"{pair:<10} error")
                continue
            if r["trades"] == 0:
                print(f"{pair:<10} {0:>7}   no trades")
                continue
            print(
                f"{pair:<10} {r['trades']:>7} {r['return_pct']:>+8.2f}% "
                f"{r['max_dd_pct']:>+8.2f}% {r['win_pct']:>6.1f}% {r['pf']:>6.2f} "
                f"${r['avg_pnl']:>+9.2f} {r['avg_bars']:>9.1f} {r['tp_hits']:>5} {r['sl_hits']:>7}"
            )
            tot_n += r["trades"]
            tot_ret += r["return_pct"]
        print("-" * 110)
        print(f"{'SUM':<10} {tot_n:>7} {tot_ret:>+8.2f}% (sum of pair returns — not compounded)")

    print("\n" + "=" * 110)
    print("ST33 Backtest Complete — backtesting.py engine, no future leakage")
    print("=" * 110)


if __name__ == "__main__":
    main()


# ============================================================================
# LEAKAGE AUDIT (read me before trusting results)
# ============================================================================
#
# 1. ENTRY EXECUTION TIMING
#    backtesting.py is constructed with `trade_on_close=False`. This means a
#    signal generated in next() on bar i is filled at the OPEN of bar i+1.
#    The strategy never gets to trade at the same bar's close → no leak.
#
# 2. SUPERTREND
#    pandas_ta.supertrend is vectorized at init() over the full series, but
#    the result is wrapped by self.I() and accessed via [-1] / [-2] in next().
#    backtesting.py masks indicator arrays to current-bar-and-earlier in next().
#    Each SuperTrend value at index t is also computed only from H/L/C ≤ t.
#
# 3. DAILY GREEN FILTER
#    We use today's calendar-day OPEN (the first bar's Open of that day)
#    compared to the CURRENT bar's CLOSE. Both values are known at decision
#    time. We deliberately do NOT use the day's eventual close
#    (which would be lookahead in a backtest).
#
# 4. SUPERTREND FLIP DETECTION
#    st_dir[-1] != 1 on the previous bar AND st_dir[-1] == 1 on the current bar.
#    Uses only data ≤ current bar.
#
# 5. SL / TP PRICES
#    sl = SuperTrend value at the signal bar (known).
#    tp = close + rr * (close - sl). Uses signal-bar close as the entry proxy.
#    Actual fill is at next bar open, so the absolute sl/tp prices may be
#    slightly mis-aligned with entry, but this is REALISTIC slippage — not leak.
#
# 6. TRAILING STOP
#    On each bar after entry, if SuperTrend is still in uptrend mode AND its
#    value exceeds the current SL, raise tr.sl to that value. Never lowers.
#    Uses only st[-1] (current bar) — no future data.
#
# 7. TIME FILTER
#    Uses self.data.index[-1].hour/.minute — current bar timestamp only.
#
# 8. COMMISSION
#    0.0002 round-trip (≈ 2 pips on EURUSD price ~1.0) — project convention.
