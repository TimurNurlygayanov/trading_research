"""
Strategy ST34 Backtest Runner — FTMO $100k Swing config

Production variant (after loser-analysis on 4 USD majors, 2026 Jan–May):
  - S2-only (S1 disabled — negative expectancy at every RR/SL we tested).
  - OLD SL: level + 1 × ATR  (wider stop → less commission impact in R-multiples).
  - RR default = 2.0.
  - Filters: EMA21 > EMA50  AND  hour NOT in {8,9,10,11,14,15} UTC.
  - FTMO conditions: $100k account, ~1 pip RT commission, fixed-lot sizing.

  SIGNAL 2 — Wick cluster rejection:
    - level = max(max(open, close)) over the PRIOR 10 bars (excludes current bar)
    - Clean-level filter: in the PRIOR 20 bars, NO body crosses level
    - >= 3 of those 10 bars had high > level
    - curr bar is red AND high < level (no new high)
    - EMA21[i] > EMA50[i]                                  (trend regime filter)
    - hour(curr) NOT in {8,9,10,11,14,15} UTC               (session filter)
    - Enter short at close of confirm bar
    - SL = level + 1 × ATR(14)
    - TP = entry − RR × (SL − entry)

FTMO drawdown reporting:
  - Max total drawdown (vs 10% FTMO limit)
  - Worst single-day P&L  (vs 5% FTMO daily-loss limit)
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

# Session filter: skip signals whose confirm-bar hour (UTC) is in this set.
BAD_HOURS_UTC = {8, 9, 10, 11, 14, 15}


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


def make_st34_strategy(
    rr_ratio: float = 2.0,
    atr_len: int = 14,
    s2_lookback: int = 10,
    s2_body_clear: int = 20,
    s2_min_wicks: int = 3,
    s2_sl_atr_mult: float = 1.0,
    ema21_len: int = 21,
    ema50_len: int = 50,
    bad_hours: frozenset = frozenset(BAD_HOURS_UTC),
    lot_size: int = 100_000,    # 1 standard lot = 100k base units
):
    """Build ST34 Strategy subclass with params closed over (per CLAUDE.md §5b)."""

    class _ST34(Strategy):
        def init(self):
            o = np.asarray(self.data.Open,  dtype=float)
            h = np.asarray(self.data.High,  dtype=float)
            _l = np.asarray(self.data.Low,  dtype=float)
            c = np.asarray(self.data.Close, dtype=float)

            atr_arr = ta.atr(
                pd.Series(h), pd.Series(_l), pd.Series(c), length=atr_len
            ).values.astype(float)

            ema21_arr = ta.ema(pd.Series(c), length=ema21_len).values.astype(float)
            ema50_arr = ta.ema(pd.Series(c), length=ema50_len).values.astype(float)

            body_top = np.maximum(o, c)
            level_arr = (
                pd.Series(body_top)
                .rolling(s2_lookback)
                .max()
                .shift(1)
                .values.astype(float)
            )
            body_max_wide = (
                pd.Series(body_top)
                .rolling(s2_body_clear)
                .max()
                .shift(1)
                .values.astype(float)
            )

            # wick_count[i] = sum over j in [i-lookback .. i-1] of (high[j] > level[i])
            n = len(c)
            wick_arr = np.zeros(n, dtype=float)
            for i in range(s2_lookback + 1, n):
                lvl = level_arr[i]
                if np.isnan(lvl):
                    continue
                wick_arr[i] = np.sum(h[i - s2_lookback : i] > lvl)

            self.atr = self.I(lambda: atr_arr, name="ATR", overlay=False)
            self.level = self.I(lambda: level_arr, name="S2_level", overlay=True)
            self.body_max_wide = self.I(lambda: body_max_wide, name="S2_bodymax", overlay=False)
            self.wick_count = self.I(lambda: wick_arr, name="S2_wicks", overlay=False)
            self.ema21 = self.I(lambda: ema21_arr, name="EMA21", overlay=True)
            self.ema50 = self.I(lambda: ema50_arr, name="EMA50", overlay=True)

        def next(self):
            i = len(self.data) - 1
            if i < max(atr_len, s2_lookback, s2_body_clear, ema50_len) + 1:
                return
            if self.position:
                return

            o_curr = float(self.data.Open[-1])
            h_curr = float(self.data.High[-1])
            c_curr = float(self.data.Close[-1])
            atr_now = float(self.atr[-1])
            if np.isnan(atr_now):
                return

            # ── Session filter ────────────────────────────────────────────
            hour = self.data.index[-1].hour
            if hour in bad_hours:
                return

            # ── EMA21 > EMA50 trend filter ────────────────────────────────
            ema21_now = float(self.ema21[-1])
            ema50_now = float(self.ema50[-1])
            if np.isnan(ema21_now) or np.isnan(ema50_now):
                return
            if not (ema21_now > ema50_now):
                return

            # ── S2 detection ─────────────────────────────────────────────
            lvl = float(self.level[-1])
            wc = float(self.wick_count[-1])
            body_max_w = float(self.body_max_wide[-1])
            if np.isnan(lvl) or wc < s2_min_wicks:
                return
            if np.isnan(body_max_w) or body_max_w > lvl:
                return

            red_curr = c_curr < o_curr
            no_new_hi = h_curr < lvl
            if not (red_curr and no_new_hi):
                return

            entry = c_curr
            sl = lvl + s2_sl_atr_mult * atr_now
            if sl <= entry:
                return
            risk = sl - entry
            tp = entry - rr_ratio * risk
            if tp <= 0:
                return

            try:
                self.sell(size=lot_size, sl=sl, tp=tp)
            except Exception as e:
                # surface, don't swallow silently
                print(f"  Sell rejected at bar {i}: {e}")

    _ST34.__name__ = (
        f"ST34_rr{rr_ratio}_lots{lot_size//100_000}"
    )
    return _ST34


# ============================================================================
# RUNNER
# ============================================================================


def run_one(
    df: pd.DataFrame,
    pair: str,
    rr_ratio: float = 2.0,
    commission: float = 0.00005,   # ~1 pip RT on EURUSD (FTMO-like)
    cash: float = 100_000,
    lots: int = 1,
    leverage: float = 30.0,
    **strat_kwargs,
) -> Optional[dict]:
    if df.empty:
        return None
    Strat = make_st34_strategy(
        rr_ratio=rr_ratio,
        lot_size=lots * 100_000,
        **strat_kwargs,
    )
    bt = Backtest(
        df, Strat,
        cash=cash,
        commission=commission,
        margin=1.0 / leverage,          # FTMO-style 30:1 max leverage on FX
        trade_on_close=False,
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
        return {"pair": pair, "trades": 0, "return_pct": 0.0, "max_dd_pct": 0.0,
                "win_pct": 0.0, "pf": 0.0, "avg_pnl": 0.0, "exposure_pct": 0.0,
                "avg_bars": 0.0, "worst_day_pct": 0.0,
                "ftmo_dd_ok": True, "ftmo_day_ok": True}

    win = float((trades["PnL"] > 0).mean()) * 100
    avg_bars = float((trades["ExitBar"] - trades["EntryBar"]).mean())
    pf = stats.get("Profit Factor", 0.0)
    pf = 0.0 if pd.isna(pf) else float(pf)

    # Worst single-day P&L  (FTMO 5% daily-loss check)
    if "ExitTime" in trades.columns:
        t = trades.copy()
        t["day"] = pd.to_datetime(t["ExitTime"]).dt.date
        day_pnl = t.groupby("day")["PnL"].sum()
        worst_day = float(day_pnl.min()) if len(day_pnl) else 0.0
    else:
        worst_day = 0.0
    worst_day_pct = 100 * worst_day / cash

    max_dd_pct = float(stats["Max. Drawdown [%]"])
    return_pct = float(stats["Return [%]"])

    return {
        "pair": pair,
        "trades": n,
        "return_pct": return_pct,
        "max_dd_pct": max_dd_pct,
        "exposure_pct": float(stats["Exposure Time [%]"]),
        "win_pct": win,
        "pf": pf,
        "avg_pnl": float(trades["PnL"].mean()),
        "avg_bars": avg_bars,
        "worst_day_pct": worst_day_pct,
        "ftmo_dd_ok":  max_dd_pct  > -10.0,   # FTMO max-loss limit
        "ftmo_day_ok": worst_day_pct > -5.0,  # FTMO daily-loss limit
    }


def main():
    USD_PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]
    TF = "5m"
    START = "2026-01-01"
    END = "2026-05-15"
    CASH = 100_000
    COMMISSION = 0.00005    # ~1 pip RT on EURUSD

    print("=" * 130)
    print("ST34 — FTMO $100k Swing Backtest  (S2-only, OLD SL, RR=2, EMA21>EMA50, avoid UTC 8-11 + 14-15)")
    print(f"  Account: ${CASH:,}  |  Commission: {COMMISSION} (~1 pip RT)  |  Period: {START} → {END}")
    print("=" * 130)

    # Load each pair once
    data = {p: get_mt5_data(p, TF, START, END) for p in USD_PAIRS}

    for lots in (1, 2, 5):
        print(f"\n=== Lots per trade = {lots}  (notional = ${lots * 100_000:,}/trade on majors) ===")
        print(f"{'Pair':<10}{'Trades':>7}{'Return%':>9}{'MaxDD%':>9}{'WorstDay%':>11}"
              f"{'Win%':>7}{'PF':>6}{'AvgPnL$':>10}{'AvgBars':>9}"
              f"{'DD-OK':>7}{'Day-OK':>8}")
        print("-" * 100)

        sum_ret = 0.0
        sum_trades = 0
        ftmo_violations = []
        for pair in USD_PAIRS:
            df = data[pair]
            if df.empty:
                print(f"{pair:<10}  no data"); continue
            r = run_one(df, pair, lots=lots, cash=CASH, commission=COMMISSION)
            if r is None or r["trades"] == 0:
                print(f"{pair:<10}  no trades"); continue
            print(
                f"{pair:<10}{r['trades']:>7}{r['return_pct']:>+8.2f}%"
                f"{r['max_dd_pct']:>+8.2f}%{r['worst_day_pct']:>+10.2f}%"
                f"{r['win_pct']:>6.1f}%{r['pf']:>6.2f}${r['avg_pnl']:>+9.2f}"
                f"{r['avg_bars']:>9.1f}{'✓' if r['ftmo_dd_ok'] else '✗':>7}"
                f"{'✓' if r['ftmo_day_ok'] else '✗':>8}"
            )
            sum_ret += r["return_pct"]
            sum_trades += r["trades"]
            if not r["ftmo_dd_ok"]:
                ftmo_violations.append(f"{pair} max-DD breached ({r['max_dd_pct']:.2f}%)")
            if not r["ftmo_day_ok"]:
                ftmo_violations.append(f"{pair} daily-loss breached ({r['worst_day_pct']:.2f}%)")
        print("-" * 100)
        print(f"{'SUM':<10}{sum_trades:>7}{sum_ret:>+8.2f}% (sum of pair returns — pair sleeves run independently on $100k each)")
        if ftmo_violations:
            print("FTMO VIOLATIONS:")
            for v in ftmo_violations:
                print(f"  ✗ {v}")
        else:
            print("FTMO checks: all pairs within 5% daily / 10% total loss limits")


if __name__ == "__main__":
    main()
