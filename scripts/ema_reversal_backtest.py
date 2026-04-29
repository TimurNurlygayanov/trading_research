"""
EMA-reversal trade-flip strategy.

Logic
  source = log((High + Low) / 2)
  ema    = EMA(source, period)

  At each bar t with rolling lookback L:
    up-reversal   = ema[t] > min(ema[t-L:t])  AND  min(ema[t-L:t]) < ema[t-L]
    down-reversal = ema[t] < max(ema[t-L:t])  AND  max(ema[t-L:t]) > ema[t-L]

  Both conditions stay true for many bars after the turn — we only trade on
  the FIRST bar each condition flips from false to true (the transition).

Trades
  long  on up-reversal event
  exit  on next down-reversal event, OR after `max_hold` bars (max_hold = 0
        disables the time cap and exits only on the opposite reversal)
  --long-short: additionally short on down-reversal events (long-flat-short)

Sweep
  period × lookback × max_hold  →  rank by per-trade Sharpe and total return.

Caveat
  All metrics are in-sample. If a combo looks worth trading, push it through
  the regular pipeline (/ideas) for walk-forward validation.

Usage
  python -m scripts.ema_reversal_backtest
  python -m scripts.ema_reversal_backtest --tf 5m --start 2022-01-01 --end 2025-12-31
  python -m scripts.ema_reversal_backtest --long-short
  python -m scripts.ema_reversal_backtest --top 30 --commission 0.00005
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv

# ── grid ────────────────────────────────────────────────────────────────────
PERIODS    = (21, 34, 50, 89, 144, 200)
LOOKBACKS  = (5, 10, 20, 30)
MAX_HOLDS  = (0, 30, 60, 120, 240)   # 0 = exit only on opposite reversal


# ── reversal events ─────────────────────────────────────────────────────────
def reversal_events(ema: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series]:
    """
    Return (up_event, down_event): boolean series, True only on the first bar
    each reversal condition transitions from False → True.
    """
    rolling_min = ema.rolling(lookback).min()
    rolling_max = ema.rolling(lookback).max()
    ema_lag     = ema.shift(lookback)

    up_state   = (ema > rolling_min) & (rolling_min < ema_lag)
    down_state = (ema < rolling_max) & (rolling_max > ema_lag)

    up_event   = up_state.fillna(False)   & ~up_state.shift(1).fillna(False)
    down_event = down_state.fillna(False) & ~down_state.shift(1).fillna(False)
    return up_event, down_event


# ── strategy ────────────────────────────────────────────────────────────────
def make_strategy(period: int, lookback: int, max_hold: int, long_short: bool):
    class _EmaReversal(Strategy):
        def init(self):
            high = np.asarray(self.data.High, dtype=float)
            low  = np.asarray(self.data.Low,  dtype=float)
            src  = np.log((high + low) / 2)
            ema  = ta.ema(pd.Series(src), length=period)
            up, dn = reversal_events(ema, lookback)
            self.up_evt = self.I(lambda: up.astype(float).values, name="up", overlay=False)
            self.dn_evt = self.I(lambda: dn.astype(float).values, name="dn", overlay=False)

        def next(self):
            up_now = self.up_evt[-1] > 0.5
            dn_now = self.dn_evt[-1] > 0.5

            # ── manage open position ───────────────────────────────────
            if self.position and self.trades:
                trade = self.trades[0]
                bars_held = len(self.data) - 1 - trade.entry_bar

                # Time cap
                if max_hold > 0 and bars_held >= max_hold:
                    self.position.close()
                    return

                # Opposite reversal: close long on dn, close short on up
                if trade.is_long and dn_now:
                    self.position.close()
                    if long_short:
                        self.sell()
                    return
                if trade.is_short and up_now:
                    self.position.close()
                    if long_short:
                        self.buy()
                    return
                return

            # ── flat: take new entries ─────────────────────────────────
            if up_now:
                self.buy()
            elif long_short and dn_now:
                self.sell()

    _EmaReversal.__name__ = (f"EmaRev_p{period}_lb{lookback}_h{max_hold}"
                              f"{'_LS' if long_short else ''}")
    return _EmaReversal


# ── backtest harness ────────────────────────────────────────────────────────
def bars_per_year_for_tf(tf: str) -> float:
    tf = tf.lower().strip()
    n, unit = int(tf[:-1]), tf[-1]
    minutes = {"m": n, "h": n * 60, "d": n * 1440}[unit]
    return 365 * 24 * 60 / minutes


def run_one(df: pd.DataFrame, period: int, lookback: int, max_hold: int,
            long_short: bool, commission: float, tf: str) -> dict:
    Strat = make_strategy(period, lookback, max_hold, long_short)
    bt = Backtest(df, Strat, cash=10_000, commission=commission)
    stats = bt.run()
    trades = stats._trades

    n_trades = len(trades)
    win_rate = float((trades["PnL"] > 0).mean()) if n_trades else 0.0
    avg_dur_bars = 0.0
    if n_trades > 0:
        avg_dur_bars = float((trades["ExitBar"] - trades["EntryBar"]).mean())

    trade_sharpe = 0.0
    if n_trades > 1:
        r = trades["ReturnPct"]
        std = r.std()
        if std > 0:
            bpy = bars_per_year_for_tf(tf)
            trades_per_year = n_trades * bpy / max(len(df), 1)
            trade_sharpe = float(r.mean() / std * np.sqrt(max(trades_per_year, 1)))

    return {
        "period":       period,
        "lookback":     lookback,
        "max_hold":     max_hold,
        "n_trades":     n_trades,
        "avg_bars":     round(avg_dur_bars, 1),
        "return_pct":   round(float(stats["Return [%]"]),         2),
        "win_pct":      round(win_rate * 100,                     1),
        "max_dd_pct":   round(float(stats["Max. Drawdown [%]"]),  2),
        "exposure_pct": round(float(stats["Exposure Time [%]"]),  1),
        "trade_sharpe": round(trade_sharpe,                       3),
    }


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbol",     default="EURUSD")
    ap.add_argument("--tf",         default="1h")
    ap.add_argument("--start",      default="2018-01-01")
    ap.add_argument("--end",        default="2025-12-31")
    ap.add_argument("--commission", type=float, default=0.0001,
                    help="per-trade commission as fraction (0.0001 = 1 pip on EURUSD)")
    ap.add_argument("--top",        type=int, default=20,
                    help="how many best combos to display in the leaderboard")
    ap.add_argument("--long-short", action="store_true",
                    help="also short on down-reversal (default = long-only)")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent

    print(f"Fetching {args.symbol} {args.tf}  {args.start} → {args.end}")
    df = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  fetched {len(df):,} bars\n")
    if df.empty:
        print("No data — aborting.")
        return

    n_combos = len(PERIODS) * len(LOOKBACKS) * len(MAX_HOLDS)
    mode = "long-short" if args.long_short else "long-only"
    print("=" * 80)
    print(f"EMA-reversal sweep  ({n_combos} combos, mode={mode}, "
          f"commission={args.commission}/side)")
    print(f"  periods:    {PERIODS}")
    print(f"  lookbacks:  {LOOKBACKS}")
    print(f"  max_holds:  {MAX_HOLDS}  (0 = exit only on opposite reversal)")
    print("=" * 80)
    header = (f"  {'#':>3}  {'p':>4} {'lb':>3} {'maxH':>5}   "
              f"{'trades':>6} {'avgBar':>6}  {'ret%':>7} {'win%':>5} "
              f"{'dd%':>7} {'expo%':>5} {'tSR':>5}")

    rows = []
    for period in PERIODS:
        for lb in LOOKBACKS:
            for mh in MAX_HOLDS:
                try:
                    res = run_one(df, period, lb, mh, args.long_short,
                                  commission=args.commission, tf=args.tf)
                    rows.append(res)
                except Exception as e:
                    print(f"  FAILED p={period} lb={lb} mh={mh}: {e}")

    full = pd.DataFrame(rows)
    suffix = "_LS" if args.long_short else ""
    full_path = out_dir / f"ema_reversal_{args.symbol}_{args.tf}{suffix}.csv"
    full.to_csv(full_path, index=False)

    if full.empty:
        print("No successful runs.")
        return

    # ── leaderboard by per-trade Sharpe ──────────────────────────────────
    top = full.sort_values("trade_sharpe", ascending=False).head(args.top)
    print("\nTop {} by per-trade Sharpe:".format(args.top))
    print(header)
    for i, r in enumerate(top.itertuples(), 1):
        print(f"  {i:>3}  {r.period:>4} {r.lookback:>3} {r.max_hold:>5}   "
              f"{r.n_trades:>6} {r.avg_bars:>6.1f}  "
              f"{r.return_pct:>+7.2f} {r.win_pct:>5.1f} "
              f"{r.max_dd_pct:>+7.2f} {r.exposure_pct:>5.1f} {r.trade_sharpe:>+5.2f}")

    # ── leaderboard by total return (different signal of merit) ──────────
    top_r = full.sort_values("return_pct", ascending=False).head(args.top)
    print(f"\nTop {args.top} by return %:")
    print(header)
    for i, r in enumerate(top_r.itertuples(), 1):
        print(f"  {i:>3}  {r.period:>4} {r.lookback:>3} {r.max_hold:>5}   "
              f"{r.n_trades:>6} {r.avg_bars:>6.1f}  "
              f"{r.return_pct:>+7.2f} {r.win_pct:>5.1f} "
              f"{r.max_dd_pct:>+7.2f} {r.exposure_pct:>5.1f} {r.trade_sharpe:>+5.2f}")

    # ── headline summary ────────────────────────────────────────────────
    best_sr = full.loc[full["trade_sharpe"].idxmax()]
    best_rt = full.loc[full["return_pct"].idxmax()]
    print(f"\n→ saved {full_path}")
    print(f"\nBest by trade Sharpe : period={best_sr['period']}  "
          f"lookback={best_sr['lookback']}  max_hold={best_sr['max_hold']}  "
          f"→ tSR={best_sr['trade_sharpe']}  ret={best_sr['return_pct']}%")
    print(f"Best by total return : period={best_rt['period']}  "
          f"lookback={best_rt['lookback']}  max_hold={best_rt['max_hold']}  "
          f"→ tSR={best_rt['trade_sharpe']}  ret={best_rt['return_pct']}%")


if __name__ == "__main__":
    main()
