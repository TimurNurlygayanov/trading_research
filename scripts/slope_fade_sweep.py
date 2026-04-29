"""
Slope-fade strategy sweep + backtest.

Pipeline
  source       = log((High + Low) / 2)
  smoothed     = smoother(source, period)             # SMA / EMA / WMA / HMA / DEMA / TEMA / RMA / ZLMA
  slope        = (smoothed[t] - smoothed[t-N]) * 1e4  # in bps (log-diff × 1e4)

Signal hypothesis (mean reversion, from prior correlation study)
  long  when slope <= q20% threshold
  short when slope >= q80% threshold
  exit  after `horizon` bars

Phase 1 — signal sweep
  scan grid (smoother × period × lookback × horizon)
  rank by combined Q1/Q5 hit-rate lift (= q1_lift - q5_lift)

Phase 2 — backtest top-K
  use the q20/q80 cutoffs from Phase 1 as static thresholds
  realistic commission, fixed-bar exits
  reports per-trade Sharpe (project convention — see CLAUDE.md §5c)

Caveat
  The q20/q80 thresholds are computed in-sample. Treat the Phase-2 metrics as
  an upper bound on edge, not as walk-forward validation. If something looks
  promising, submit it to the pipeline for proper train/OOS testing.

Usage
  python -m scripts.slope_fade_sweep                       # default EURUSD 1h, 2018→2025
  python -m scripts.slope_fade_sweep --symbol GBPUSD --tf 1h
  python -m scripts.slope_fade_sweep --top 15 --commission 0.00015
  python -m scripts.slope_fade_sweep --no-backtest         # signal scan only
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

warnings.filterwarnings("ignore")  # silence pandas_ta + future warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv

# ── grid ─────────────────────────────────────────────────────────────────────
SMOOTHERS = {
    "sma":  ta.sma,
    "ema":  ta.ema,
    "wma":  ta.wma,
    "hma":  ta.hma,
    "dema": ta.dema,
    "tema": ta.tema,
    "rma":  ta.rma,
    "zlma": ta.zlma,
}
PERIODS    = (14, 21, 34, 50, 89, 144, 200)
LOOKBACKS  = (1, 5, 10, 20)
HORIZONS   = (3, 10, 20)
Q_LO_PCT   = 0.20
Q_HI_PCT   = 0.80


# ── feature construction ─────────────────────────────────────────────────────
def build_source(df: pd.DataFrame) -> pd.Series:
    """source = log((H + L) / 2)"""
    return np.log((df["High"] + df["Low"]) / 2)


def smooth_safe(fn, src: pd.Series, period: int) -> pd.Series | None:
    """Some pandas_ta smoothers fail on very short periods or specific data shapes — be defensive."""
    try:
        out = fn(src, length=period)
        return out if isinstance(out, pd.Series) and out.notna().sum() > 0 else None
    except Exception:
        return None


def slope_of(smoothed: pd.Series, lookback: int) -> pd.Series:
    return (smoothed - smoothed.shift(lookback)) * 10000  # bps (log-diff × 1e4)


# ── Phase 1: signal sweep ───────────────────────────────────────────────────
def sweep_signal(df: pd.DataFrame) -> pd.DataFrame:
    src = build_source(df)
    close = df["Close"]
    targets   = {h: (close.shift(-h) > close).astype(int) for h in HORIZONS}
    baselines = {h: targets[h].mean() for h in HORIZONS}

    rows = []
    for sm_name, fn in SMOOTHERS.items():
        for period in PERIODS:
            sm = smooth_safe(fn, src, period)
            if sm is None:
                continue
            for lb in LOOKBACKS:
                slope = slope_of(sm, lb)
                for h in HORIZONS:
                    sub = pd.concat([slope.rename("slope"), targets[h].rename("tgt")],
                                    axis=1).dropna()
                    if len(sub) < 1000:
                        continue
                    q_lo = sub["slope"].quantile(Q_LO_PCT)
                    q_hi = sub["slope"].quantile(Q_HI_PCT)
                    q1_hit = sub.loc[sub["slope"] <= q_lo, "tgt"].mean()
                    q5_hit = sub.loc[sub["slope"] >= q_hi, "tgt"].mean()
                    base = baselines[h]
                    rows.append({
                        "smoother":  sm_name,
                        "period":    period,
                        "lookback":  lb,
                        "horizon":   h,
                        "q1_hit":    q1_hit,
                        "q5_hit":    q5_hit,
                        "q1_lift":   q1_hit - base,
                        "q5_lift":   q5_hit - base,
                        "score":     (q1_hit - base) - (q5_hit - base),
                        "q_lo":      q_lo,
                        "q_hi":      q_hi,
                        "n":         len(sub),
                    })
    return pd.DataFrame(rows)


# ── Phase 2: backtest ────────────────────────────────────────────────────────
def make_strategy(sm_name: str, period: int, lookback: int,
                  hold_bars: int, q_lo: float, q_hi: float,
                  exit_first_profit: bool = False):
    """
    Return a backtesting.py Strategy subclass with params closed over.

    exit_first_profit=False (default): exit after `hold_bars` regardless of P&L.
    exit_first_profit=True            : exit on the first bar where unrealized
                                        gross P&L > 0, or time-out at `hold_bars`
                                        (whichever comes first).
                                        NB: "profitable" here ignores exit
                                        commission — a tick of profit may net
                                        slightly negative after costs.
    """
    fn = SMOOTHERS[sm_name]

    class _FadeSlope(Strategy):
        def init(self):
            high = np.asarray(self.data.High, dtype=float)
            low  = np.asarray(self.data.Low,  dtype=float)
            src  = np.log((high + low) / 2)
            sm_s = fn(pd.Series(src), length=period)
            sm_arr = sm_s.values if sm_s is not None else np.full_like(src, np.nan)
            lag = np.concatenate([np.full(lookback, np.nan), sm_arr[:-lookback]])
            slope = (sm_arr - lag) * 10000
            self.slope = self.I(lambda: slope, name="slope", overlay=False)

        def next(self):
            s = self.slope[-1]
            if np.isnan(s):
                return

            if self.position and self.trades:
                trade = self.trades[0]

                # First-profit exit (when enabled)
                if exit_first_profit:
                    cur = self.data.Close[-1]
                    is_profit = ((trade.is_long  and cur > trade.entry_price) or
                                 (trade.is_short and cur < trade.entry_price))
                    if is_profit:
                        self.position.close()
                        return

                # Time-based exit (always the backstop)
                bars_held = len(self.data) - 1 - trade.entry_bar
                if bars_held >= hold_bars:
                    self.position.close()
                return

            # Entries
            if s <= q_lo:
                self.buy()
            elif s >= q_hi:
                self.sell()

    suffix = "_fp" if exit_first_profit else ""
    _FadeSlope.__name__ = f"FadeSlope_{sm_name}_p{period}_lb{lookback}_h{hold_bars}{suffix}"
    return _FadeSlope


def bars_per_year_for_tf(tf: str) -> float:
    """FX trades ~24/7 in our model. Convert timeframe → bars/year for Sharpe annualization."""
    tf = tf.lower().strip()
    n, unit = int(tf[:-1]), tf[-1]
    minutes = {"m": n, "h": n * 60, "d": n * 1440}[unit]
    return 365 * 24 * 60 / minutes


def run_one_backtest(df: pd.DataFrame, params: dict, commission: float, tf: str,
                     exit_first_profit: bool = False) -> dict:
    Strat = make_strategy(
        sm_name           = params["smoother"],
        period            = int(params["period"]),
        lookback          = int(params["lookback"]),
        hold_bars         = int(params["horizon"]),
        q_lo              = float(params["q_lo"]),
        q_hi              = float(params["q_hi"]),
        exit_first_profit = exit_first_profit,
    )
    bt = Backtest(df, Strat, cash=10_000, commission=commission)
    stats = bt.run()
    trades = stats._trades

    n_trades = len(trades)
    win_rate = float((trades["PnL"] > 0).mean()) if n_trades else 0.0

    # Per-trade annualized Sharpe (project convention — equity-curve Sharpe is
    # near-zero for intraday strategies; see CLAUDE.md §5c).
    trade_sharpe = 0.0
    if n_trades > 1:
        r = trades["ReturnPct"]
        std = r.std()
        if std > 0:
            bpy = bars_per_year_for_tf(tf)
            trades_per_year = n_trades * bpy / max(len(df), 1)
            trade_sharpe = float(r.mean() / std * np.sqrt(max(trades_per_year, 1)))

    return {
        "smoother":     params["smoother"],
        "period":       params["period"],
        "lookback":     params["lookback"],
        "horizon":      params["horizon"],
        "n_trades":     n_trades,
        "return_pct":   round(float(stats["Return [%]"]),         2),
        "win_pct":      round(win_rate * 100,                     1),
        "max_dd_pct":   round(float(stats["Max. Drawdown [%]"]),  2),
        "exposure_pct": round(float(stats["Exposure Time [%]"]),  1),
        "trade_sharpe": round(trade_sharpe,                       3),
        "score":        round(float(params.get("score", 0)),      4),
    }


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbol",     default="EURUSD")
    ap.add_argument("--tf",         default="1h")
    ap.add_argument("--start",      default="2018-01-01")
    ap.add_argument("--end",        default="2025-12-31")
    ap.add_argument("--top",        type=int, default=10,
                    help="how many top combos to backtest")
    ap.add_argument("--commission", type=float, default=0.0001,
                    help="per-trade commission as fraction (0.0001 = 1 pip on EURUSD)")
    ap.add_argument("--no-backtest", action="store_true",
                    help="run signal sweep only, skip Phase 2")
    ap.add_argument("--exit-first-profit", action="store_true",
                    help="exit on first profitable bar (gross P&L) instead of fixed-bar hold")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent

    # ── fetch ────────────────────────────────────────────────────────────
    print(f"Fetching {args.symbol} {args.tf}  {args.start} → {args.end}")
    df = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  fetched {len(df):,} bars\n")
    if df.empty:
        print("No data — aborting.")
        return

    # ── Phase 1 ──────────────────────────────────────────────────────────
    n_combos = len(SMOOTHERS) * len(PERIODS) * len(LOOKBACKS) * len(HORIZONS)
    print("=" * 80)
    print(f"PHASE 1: signal sweep  ({n_combos} combos)")
    print(f"  smoothers: {list(SMOOTHERS)}")
    print(f"  periods:   {PERIODS}")
    print(f"  lookbacks: {LOOKBACKS}")
    print(f"  horizons:  {HORIZONS}")
    print(f"  source:    log((H + L) / 2)")
    print("=" * 80)

    sweep = sweep_signal(df)
    sweep_path = out_dir / f"slope_sweep_{args.symbol}_{args.tf}.csv"
    sweep.to_csv(sweep_path, index=False)

    if sweep.empty:
        print("No combos produced data. Aborting.")
        return

    print("\nTop 15 by score (= q1_lift − q5_lift, larger = stronger two-sided fade):")
    cols = ["smoother", "period", "lookback", "horizon",
            "q1_hit", "q5_hit", "q1_lift", "q5_lift", "score", "n"]
    top = sweep.sort_values("score", ascending=False).head(15)
    fmt = lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v)
    print(top[cols].to_string(index=False,
          formatters={c: fmt for c in cols if top[c].dtype.kind == "f"}))
    print(f"\n→ saved {sweep_path}")

    if args.no_backtest:
        return

    # ── Phase 2 ──────────────────────────────────────────────────────────
    top_k = sweep.sort_values("score", ascending=False).head(args.top)
    exit_mode = "first-profit" if args.exit_first_profit else f"time-out at h bars"
    print("\n" + "=" * 80)
    print(f"PHASE 2: backtesting top-{args.top}  (commission={args.commission}/side, exit={exit_mode})")
    print("=" * 80)
    print(f"{'#':>2}  {'smoother':>5} {'p':>3} {'lb':>2} {'h':>2}   "
          f"{'trades':>6}  {'ret%':>7}  {'win%':>5}  {'dd%':>7}  {'expo%':>5}  {'tSR':>5}")

    bt_rows = []
    for i, (_, row) in enumerate(top_k.iterrows(), 1):
        params = row.to_dict()
        try:
            res = run_one_backtest(df, params, commission=args.commission, tf=args.tf,
                                   exit_first_profit=args.exit_first_profit)
            bt_rows.append(res)
            print(f"{i:>2}  {res['smoother']:>5} {res['period']:>3} "
                  f"{res['lookback']:>2} {res['horizon']:>2}   "
                  f"{res['n_trades']:>6}  "
                  f"{res['return_pct']:>+7.2f}  {res['win_pct']:>5.1f}  "
                  f"{res['max_dd_pct']:>+7.2f}  {res['exposure_pct']:>5.1f}  "
                  f"{res['trade_sharpe']:>+5.2f}")
        except Exception as e:
            print(f"{i:>2}  FAILED: {e}")

    bt_df = pd.DataFrame(bt_rows)
    suffix = "_fp" if args.exit_first_profit else ""
    bt_path = out_dir / f"slope_backtest_{args.symbol}_{args.tf}{suffix}.csv"
    bt_df.to_csv(bt_path, index=False)
    print(f"\n→ saved {bt_path}")

    if not bt_df.empty:
        print("\nSorted by per-trade Sharpe:")
        print(bt_df.sort_values("trade_sharpe", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
