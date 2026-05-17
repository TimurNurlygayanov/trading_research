"""
Strategy ST33 Daily — Backtest on 1d timeframe with N-day green filter (backtesting.py engine)

Logic:
  - SuperTrend (default 10, 3.0) flips bullish on the daily timeframe
  - AND the LAST N daily candles (including current) are ALL green (close > open)
  - SL: SuperTrend level (trailing — only raises)
  - TP: rr_ratio:1 R:R from entry close
  - NO time-of-day filter (each bar IS a full UTC day)

Date range extended to 2020-2026 to get enough daily bars per pair.

Lookahead protections (same as st033_backtest_2026.py):
  1. trade_on_close=False           → signal fires at bar i, fills at bar i+1 OPEN
  2. SuperTrend via pandas_ta       → masked to ≤ current bar in next()
  3. Last-N-green filter uses bars [-N..-1] only (closed bars known at signal time)
  4. Trailing SL raises only, uses st[-1]
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


_MT5_TF_MAP = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
               "1h": 16385, "4h": 16388, "1d": 16408}


def get_mt5_data(ticker, timeframe="1d", start="2020-01-01", end="2026-12-31"):
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")
    if not mt5.initialize():
        print(f"  ERROR: MT5 init failed for {ticker}")
        return pd.DataFrame()
    mt5.symbol_select(ticker, True)
    tf_const = _MT5_TF_MAP.get(timeframe, 16408)
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
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "tick_volume": "Volume"})
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    if not out.empty:
        print(f"  Loaded {len(out)} bars ({out.index[0].date()} → {out.index[-1].date()})")
    return out


def make_st33_daily_strategy(st_period: int, st_mult: float,
                             rr_ratio: float, green_streak: int):
    """ST33 daily — SuperTrend flip + last `green_streak` daily candles all green."""

    class _ST33Daily(Strategy):
        def init(self):
            high = pd.Series(np.asarray(self.data.High, dtype=float))
            low = pd.Series(np.asarray(self.data.Low, dtype=float))
            close = pd.Series(np.asarray(self.data.Close, dtype=float))
            opens = pd.Series(np.asarray(self.data.Open, dtype=float))

            st_df = ta.supertrend(high, low, close, length=st_period, multiplier=st_mult)
            st_col = f"SUPERT_{st_period}_{float(st_mult)}"
            d_col = f"SUPERTd_{st_period}_{float(st_mult)}"
            if st_df is None or st_col not in st_df.columns:
                st_col = f"SUPERT_{st_period}_{st_mult}"
                d_col = f"SUPERTd_{st_period}_{st_mult}"
            st_arr = st_df[st_col].values.astype(float)
            dir_arr = st_df[d_col].values.astype(float)

            # Rolling count of last `green_streak` candles that are green.
            # At each bar t, this uses bars (t - green_streak + 1) .. t — all known at t.
            is_green = (close.values > opens.values).astype(int)
            green_count = pd.Series(is_green).rolling(green_streak).sum().values

            self.st = self.I(lambda: st_arr, name="ST", overlay=True)
            self.st_dir = self.I(lambda: dir_arr, name="STd", overlay=False)
            self.green_count = self.I(lambda: green_count, name="GreenN", overlay=False)

        def next(self):
            i = len(self.data) - 1
            if i < max(green_streak, 2):
                return

            # Trail SL on open long position (only raises)
            if self.position and self.position.is_long:
                st_now = float(self.st[-1])
                if not np.isnan(st_now) and self.st_dir[-1] == 1:
                    for tr in self.trades:
                        if tr.is_long and (tr.sl is None or st_now > tr.sl):
                            try:
                                tr.sl = st_now
                            except Exception:
                                pass

            if self.position:
                return

            # SuperTrend flip to uptrend on this bar
            if not (self.st_dir[-2] != 1 and self.st_dir[-1] == 1):
                return

            # Last N daily candles all green
            if int(self.green_count[-1]) < green_streak:
                return

            close_now = float(self.data.Close[-1])
            sl = float(self.st[-1])
            if np.isnan(sl) or sl >= close_now:
                return
            risk = close_now - sl
            tp = close_now + rr_ratio * risk

            try:
                self.buy(sl=sl, tp=tp)
            except Exception:
                pass

    _ST33Daily.__name__ = f"ST33D_p{st_period}_m{st_mult}_rr{rr_ratio}_g{green_streak}"
    return _ST33Daily


def run_one(df, st_period=10, st_mult=3.0, rr_ratio=3.0,
            green_streak=5, commission=0.0002) -> Optional[dict]:
    if df.empty:
        return None
    Strat = make_st33_daily_strategy(st_period, st_mult, rr_ratio, green_streak)
    bt = Backtest(df, Strat, cash=10_000, commission=commission,
                  trade_on_close=False, exclusive_orders=True)
    try:
        stats = bt.run()
    except Exception as e:
        print(f"  Error: {e}")
        return None

    trades = stats._trades
    n = len(trades)
    if n == 0:
        return {"trades": 0, "return_pct": 0.0, "max_dd_pct": 0.0,
                "win_pct": 0.0, "pf": 0.0, "avg_pnl": 0.0, "avg_bars": 0.0}

    win = float((trades["PnL"] > 0).mean()) * 100
    avg_bars = float((trades["ExitBar"] - trades["EntryBar"]).mean())
    pf = stats.get("Profit Factor", 0.0)
    pf = 0.0 if pd.isna(pf) else float(pf)
    return {
        "trades": n,
        "return_pct": float(stats["Return [%]"]),
        "max_dd_pct": float(stats["Max. Drawdown [%]"]),
        "win_pct": win,
        "pf": pf,
        "avg_pnl": float(trades["PnL"].mean()),
        "avg_bars": avg_bars,
    }


def main():
    print("=" * 110)
    print("ST33 DAILY BACKTEST — 2020-2026 (SuperTrend + 5 consecutive green daily candles)")
    print("=" * 110)

    PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
             "USDCHF", "USDCAD", "XAUUSD"]
    START = "2020-01-01"
    END = "2026-12-31"
    GREEN_STREAK = 5

    print(f"\nParams: SuperTrend(10, 3.0)   R:R = 1:3   GreenStreak = {GREEN_STREAK} days")
    print(f"Range:  {START} → {END}")
    print("-" * 110)
    print(f"{'Pair':<10} {'Trades':>7} {'Return%':>9} {'MaxDD%':>9} {'Win%':>7} "
          f"{'PF':>6} {'AvgPnL':>10} {'AvgBars':>9}")
    print("-" * 110)

    tot_n = 0
    tot_ret = 0.0
    for pair in PAIRS:
        df = get_mt5_data(pair, "1d", START, END)
        if df.empty:
            print(f"{pair:<10} no data")
            continue
        r = run_one(df, green_streak=GREEN_STREAK)
        if r is None:
            print(f"{pair:<10} error")
            continue
        if r["trades"] == 0:
            print(f"{pair:<10} {0:>7}   no trades")
            continue
        print(f"{pair:<10} {r['trades']:>7} {r['return_pct']:>+8.2f}% "
              f"{r['max_dd_pct']:>+8.2f}% {r['win_pct']:>6.1f}% {r['pf']:>6.2f} "
              f"${r['avg_pnl']:>+9.2f} {r['avg_bars']:>9.1f}")
        tot_n += r["trades"]
        tot_ret += r["return_pct"]
    print("-" * 110)
    print(f"{'SUM':<10} {tot_n:>7} {tot_ret:>+8.2f}% (sum of pair returns — not compounded)")

    # Also try a sensitivity sweep on the green-streak parameter
    print(f"\n[Sensitivity: green_streak parameter — EURUSD only]")
    print("-" * 70)
    print(f"{'Streak':<8} {'Trades':>7} {'Return%':>9} {'MaxDD%':>9} {'Win%':>7} {'PF':>6}")
    print("-" * 70)
    df_eur = get_mt5_data("EURUSD", "1d", START, END)
    if not df_eur.empty:
        for streak in [1, 2, 3, 5, 7]:
            r = run_one(df_eur, green_streak=streak)
            if r is None or r["trades"] == 0:
                print(f"{streak:<8} no trades")
                continue
            print(f"{streak:<8} {r['trades']:>7} {r['return_pct']:>+8.2f}% "
                  f"{r['max_dd_pct']:>+8.2f}% {r['win_pct']:>6.1f}% {r['pf']:>6.2f}")

    print("\n" + "=" * 110)
    print("ST33 Daily Backtest Complete")
    print("=" * 110)


if __name__ == "__main__":
    main()
