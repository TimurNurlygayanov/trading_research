"""
SuperTrend false-breakout reversal — 5M EURUSD US-session strategy.

Setup
  Trend filter:  SuperTrend(length, multiplier)  → direction == -1 means downtrend
  Session:       US session only (default 13:00–21:00 UTC)

Trigger (each in-session bar in confirmed downtrend)
  bar.High  > SuperTrend     # price probed above the resistance line
  bar.Close < SuperTrend     # but was rejected back below
  → place STOP-BUY order at the SuperTrend level
    (waits for price to rally back to that level before entering long)

Risk
  SL = entry − sl_atr × ATR(atr_length)
  TP = entry + tp_atr × ATR(atr_length)        # default 1:3 R:R
  Order TTL = order_ttl bars; cancelled if not filled

Cost notes
  Default commission = 0.00005 (0.5 pip / side, realistic ECN spread on EURUSD).
  Default 1 pip = 0.0001 of price.

Usage
  python -m scripts.supertrend_falsebreak_backtest                   # baseline
  python -m scripts.supertrend_falsebreak_backtest --sweep            # also sweep params
  python -m scripts.supertrend_falsebreak_backtest --sl-atr 1.5 --tp-atr 4.0
  python -m scripts.supertrend_falsebreak_backtest --st-length 14 --st-mult 2

All metrics are in-sample. If a config looks tradeable, push it through the
regular pipeline (/ideas) for walk-forward validation.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv


# ── Strategy factory ─────────────────────────────────────────────────────────
def make_strategy(st_length: int, st_mult: float,
                  atr_length: int, sl_atr: float, tp_atr: float,
                  session_start: int, session_end: int,
                  order_ttl: int):
    """Build a backtesting.py Strategy subclass with all params closed over."""

    class _STFalseBreak(Strategy):
        def init(self):
            high  = pd.Series(np.asarray(self.data.High,  dtype=float))
            low   = pd.Series(np.asarray(self.data.Low,   dtype=float))
            close = pd.Series(np.asarray(self.data.Close, dtype=float))

            # SuperTrend — pandas_ta returns a DataFrame.
            # Column names use the actual params (CLAUDE.md §5f).
            st_df  = ta.supertrend(high, low, close, length=st_length, multiplier=st_mult)
            st_col = f"SUPERT_{st_length}_{float(st_mult)}"
            d_col  = f"SUPERTd_{st_length}_{float(st_mult)}"
            if st_df is None or st_col not in st_df.columns:
                # Some pandas_ta versions format multiplier differently — try int form
                st_col = f"SUPERT_{st_length}_{st_mult}"
                d_col  = f"SUPERTd_{st_length}_{st_mult}"
            st_arr  = st_df[st_col].values.astype(float)
            dir_arr = st_df[d_col].values.astype(float)

            atr_arr = ta.atr(high, low, close, length=atr_length).values.astype(float)

            self.st     = self.I(lambda: st_arr,  name="ST",  overlay=True)
            self.st_dir = self.I(lambda: dir_arr, name="STd", overlay=False)
            self.atr    = self.I(lambda: atr_arr, name="ATR", overlay=False)

            self.pending_order = None   # type: ignore[assignment]
            self.pending_bar   = None

        def _cancel_pending_entry(self):
            """Cancel ONLY the pending entry order. Never touch SL/TP for an open position."""
            if self.pending_order is not None and self.pending_order in list(self.orders):
                try:
                    self.pending_order.cancel()
                except Exception:
                    pass
            self.pending_order = None
            self.pending_bar   = None

        def next(self):
            cur_idx = len(self.data) - 1

            # ── session gate ─────────────────────────────────────────────
            ts = self.data.index[-1]
            h  = ts.hour if hasattr(ts, "hour") else 12
            in_session = session_start <= h < session_end

            if not in_session:
                # Cancel waiting entry; SL/TP for open positions stay active.
                self._cancel_pending_entry()
                return

            # ── expire stale pending entry ───────────────────────────────
            if (self.pending_bar is not None and
                    cur_idx - self.pending_bar >= order_ttl):
                self._cancel_pending_entry()

            # ── one position at a time ───────────────────────────────────
            if self.position:
                return

            # ── trend gate: must be in confirmed downtrend ───────────────
            if self.st_dir[-1] != -1:
                return

            st_level = self.st[-1]
            atr_now  = self.atr[-1]
            if np.isnan(st_level) or np.isnan(atr_now) or atr_now <= 0:
                return

            bar_high  = self.data.High[-1]
            bar_close = self.data.Close[-1]

            # ── trigger pattern: probed above ST, rejected below ─────────
            if bar_high > st_level and bar_close < st_level:
                self._cancel_pending_entry()  # supersede prior pending entry
                stop_price = float(st_level)
                sl_price   = stop_price - sl_atr * atr_now
                tp_price   = stop_price + tp_atr * atr_now
                # Sanity: SL must be below stop, TP above
                if sl_price < stop_price < tp_price:
                    try:
                        self.pending_order = self.buy(
                            stop=stop_price, sl=sl_price, tp=tp_price)
                        self.pending_bar = cur_idx
                    except Exception:
                        pass

    _STFalseBreak.__name__ = (f"STFB_l{st_length}_m{st_mult}"
                               f"_sl{sl_atr}_tp{tp_atr}_ttl{order_ttl}")
    return _STFalseBreak


# ── Backtest harness ─────────────────────────────────────────────────────────
def bars_per_year_for_tf(tf: str) -> float:
    tf = tf.lower().strip()
    n, unit = int(tf[:-1]), tf[-1]
    minutes = {"m": n, "h": n * 60, "d": n * 1440}[unit]
    return 365 * 24 * 60 / minutes


def run_one(df: pd.DataFrame, params: dict, commission: float, tf: str) -> dict:
    Strat = make_strategy(**params)
    bt = Backtest(df, Strat, cash=10_000, commission=commission)
    stats = bt.run()
    trades = stats._trades

    n = len(trades)
    win = float((trades["PnL"] > 0).mean()) if n else 0.0
    avg_dur = float((trades["ExitBar"] - trades["EntryBar"]).mean()) if n else 0.0

    pf = stats.get("Profit Factor", float("nan"))
    pf = 0.0 if pd.isna(pf) else float(pf)

    tsr = 0.0
    if n > 1:
        r = trades["ReturnPct"]
        std = r.std()
        if std > 0:
            bpy = bars_per_year_for_tf(tf)
            tpy = n * bpy / max(len(df), 1)
            tsr = float(r.mean() / std * np.sqrt(max(tpy, 1)))

    return {
        "st_len":        params["st_length"],
        "st_mult":       params["st_mult"],
        "sl_atr":        params["sl_atr"],
        "tp_atr":        params["tp_atr"],
        "ttl":           params["order_ttl"],
        "n_trades":      n,
        "avg_bars":      round(avg_dur, 1),
        "return_pct":    round(float(stats["Return [%]"]),         2),
        "win_pct":       round(win * 100,                          1),
        "max_dd_pct":    round(float(stats["Max. Drawdown [%]"]),  2),
        "exposure_pct": round(float(stats["Exposure Time [%]"]),   1),
        "profit_factor": round(pf,                                 2),
        "trade_sharpe":  round(tsr,                                3),
    }


def print_result(r: dict):
    print(f"  Trades        : {r['n_trades']}")
    print(f"  Avg bars held : {r['avg_bars']}")
    print(f"  Return        : {r['return_pct']:+.2f}%")
    print(f"  Win rate      : {r['win_pct']:.1f}%")
    print(f"  Max drawdown  : {r['max_dd_pct']:+.2f}%")
    print(f"  Exposure      : {r['exposure_pct']:.1f}%")
    print(f"  Profit factor : {r['profit_factor']}")
    print(f"  Trade Sharpe  : {r['trade_sharpe']:+.3f}")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbol",        default="EURUSD")
    ap.add_argument("--tf",            default="5m")
    ap.add_argument("--start",         default="2022-01-01")
    ap.add_argument("--end",           default="2025-12-31")
    ap.add_argument("--st-length",     type=int,   default=10)
    ap.add_argument("--st-mult",       type=float, default=3.0)
    ap.add_argument("--atr-length",    type=int,   default=14)
    ap.add_argument("--sl-atr",        type=float, default=1.0)
    ap.add_argument("--tp-atr",        type=float, default=3.0)
    ap.add_argument("--session-start", type=int,   default=13, help="UTC hour, inclusive")
    ap.add_argument("--session-end",   type=int,   default=21, help="UTC hour, exclusive")
    ap.add_argument("--order-ttl",     type=int,   default=12, help="bars to keep pending order")
    ap.add_argument("--commission",    type=float, default=0.00005,
                    help="per-side commission (0.00005 = 0.5 pip on EURUSD)")
    ap.add_argument("--sweep",         action="store_true",
                    help="run a small parameter sweep around the baseline")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent

    # ── fetch ────────────────────────────────────────────────────────────
    print(f"Fetching {args.symbol} {args.tf}  {args.start} → {args.end}")
    df = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  fetched {len(df):,} bars\n")
    if df.empty:
        print("No data — aborting.")
        return

    base = dict(
        st_length     = args.st_length,
        st_mult       = args.st_mult,
        atr_length    = args.atr_length,
        sl_atr        = args.sl_atr,
        tp_atr        = args.tp_atr,
        session_start = args.session_start,
        session_end   = args.session_end,
        order_ttl     = args.order_ttl,
    )

    # ── baseline ─────────────────────────────────────────────────────────
    print("=" * 80)
    print("BASELINE — your spec")
    print(f"  SuperTrend({args.st_length}, {args.st_mult})")
    print(f"  SL = {args.sl_atr} × ATR({args.atr_length})  |  "
          f"TP = {args.tp_atr} × ATR({args.atr_length})")
    print(f"  Session: {args.session_start:02d}:00–{args.session_end:02d}:00 UTC")
    print(f"  Order TTL: {args.order_ttl} bars  |  commission: {args.commission}/side")
    print("=" * 80)
    res = run_one(df, base, args.commission, args.tf)
    print_result(res)

    # ── sweep ────────────────────────────────────────────────────────────
    if not args.sweep:
        return

    print("\n" + "=" * 80)
    print("PARAMETER SWEEP")
    print("=" * 80)
    grid = list(product(
        (7, 10, 14, 20),       # st_length
        (2.0, 3.0, 4.0),        # st_mult
        (0.5, 1.0, 1.5),        # sl_atr
        (2.0, 3.0, 4.0, 5.0),   # tp_atr
        (12, 24),               # order_ttl
    ))
    print(f"  {len(grid)} configs to test\n")

    rows = []
    for (st_l, st_m, sl, tp, ttl) in grid:
        params = dict(base, st_length=st_l, st_mult=st_m,
                       sl_atr=sl, tp_atr=tp, order_ttl=ttl)
        try:
            rows.append(run_one(df, params, args.commission, args.tf))
        except Exception as e:
            print(f"  FAILED st={st_l}/{st_m} sl={sl} tp={tp} ttl={ttl}: {e}")

    full = pd.DataFrame(rows)
    out_path = out_dir / f"st_falsebreak_sweep_{args.symbol}_{args.tf}.csv"
    full.to_csv(out_path, index=False)

    if full.empty:
        print("No results.")
        return

    header = (f"  {'#':>3}  {'stL':>3} {'stM':>4} {'slA':>4} {'tpA':>4} {'ttl':>3}   "
              f"{'trades':>6} {'avgBar':>6}  {'ret%':>7} {'win%':>5} "
              f"{'dd%':>7} {'PF':>5} {'tSR':>5}")

    print("Top 15 by trade Sharpe:")
    print(header)
    top = full.sort_values("trade_sharpe", ascending=False).head(15)
    for i, r in enumerate(top.itertuples(), 1):
        print(f"  {i:>3}  {r.st_len:>3} {r.st_mult:>4} {r.sl_atr:>4} "
              f"{r.tp_atr:>4} {r.ttl:>3}   "
              f"{r.n_trades:>6} {r.avg_bars:>6.1f}  "
              f"{r.return_pct:>+7.2f} {r.win_pct:>5.1f} "
              f"{r.max_dd_pct:>+7.2f} {r.profit_factor:>5.2f} {r.trade_sharpe:>+5.2f}")

    print("\nTop 15 by total return:")
    print(header)
    topr = full.sort_values("return_pct", ascending=False).head(15)
    for i, r in enumerate(topr.itertuples(), 1):
        print(f"  {i:>3}  {r.st_len:>3} {r.st_mult:>4} {r.sl_atr:>4} "
              f"{r.tp_atr:>4} {r.ttl:>3}   "
              f"{r.n_trades:>6} {r.avg_bars:>6.1f}  "
              f"{r.return_pct:>+7.2f} {r.win_pct:>5.1f} "
              f"{r.max_dd_pct:>+7.2f} {r.profit_factor:>5.2f} {r.trade_sharpe:>+5.2f}")

    print(f"\n→ saved {out_path}")


if __name__ == "__main__":
    main()
