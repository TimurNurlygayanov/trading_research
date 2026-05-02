"""
Prop-firm sizing analyzer for Strategy 1.

Loads a trades.csv produced by strategy1_backtest_curves.py (gap-aware), then
projects max daily loss, max drawdown, and time-to-target across a grid of
lot-size factors. Scaling is linear in PnL — half lot size → half PnL/loss.

Usage
  python -m scripts.strategy1_propfirm_sizing
  python -m scripts.strategy1_propfirm_sizing --trades data/backtest_curves/20260502_135152/trades.csv
  python -m scripts.strategy1_propfirm_sizing --account 100000 --target 15000 \\
      --max-daily-loss 5000 --max-drawdown 10000
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def find_latest_trades_csv() -> Path:
    """Return the most recent gap-aware trades.csv from data/backtest_curves/."""
    base = ROOT / "data" / "backtest_curves"
    candidates = sorted(base.glob("*/trades.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No trades.csv found in data/backtest_curves/")
    return candidates[-1]


def daily_pnl(trades: pd.DataFrame) -> pd.Series:
    """Aggregate trade PnL by UTC exit-date."""
    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    t["date"] = t["exit_time"].dt.date
    return t.groupby("date")["pnl"].sum().sort_index()


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((equity - peak).min())


def max_rolling_loss(daily: pd.Series, window: int) -> float:
    """Largest cumulative loss over any consecutive `window` days."""
    if daily.empty or window < 1:
        return 0.0
    return float(daily.rolling(window).sum().min())


def project_at_size(daily: pd.Series, factor: float) -> dict:
    d = daily * factor
    cum = d.cumsum()
    return {
        "factor":           factor,
        "annual_pnl":       float(d.sum()) / max(1, (d.index[-1] - d.index[0]).days / 365.25),
        "avg_day":          float(d.mean()),
        "median_day":       float(d.median()),
        "p25_day":          float(d.quantile(0.25)),
        "p75_day":          float(d.quantile(0.75)),
        "worst_day":        float(d.min()),
        "worst_3day":       max_rolling_loss(d, 3),
        "worst_5day":       max_rolling_loss(d, 5),
        "max_dd":           max_drawdown(cum),
        "n_days":           int(d.shape[0]),
        "n_loss_days":      int((d < 0).sum()),
        "n_win_days":       int((d > 0).sum()),
    }


def time_to_target(daily: pd.Series, factor: float, target: float,
                   pessimism: str = "p25") -> dict:
    """
    Estimate days to reach `target` profit under different scenarios:
      best     — trailing 30-day window with the most favorable mean
      avg      — historical mean
      median   — historical median day
      p25      — 25th-percentile daily PnL (pessimistic)
      worst    — assume the worst 30-day stretch is your starting window
    """
    d = (daily * factor).values
    mean = float(np.mean(d))
    median = float(np.median(d))
    p25 = float(np.percentile(d, 25))

    # Best 30-day stretch (mean per day)
    if len(d) >= 30:
        rolling_mean = pd.Series(d).rolling(30).mean()
        best_30 = float(rolling_mean.max())
        worst_30 = float(rolling_mean.min())
    else:
        best_30 = mean
        worst_30 = mean

    def days_to(rate: float) -> float:
        if rate <= 0:
            return float("inf")
        return target / rate

    return {
        "best":   days_to(best_30),
        "avg":    days_to(mean),
        "median": days_to(median),
        "p25":    days_to(p25),
        "worst":  days_to(worst_30),
    }


def report(args, trades_csv: Path) -> None:
    trades = pd.read_csv(trades_csv)
    if "scenario" in trades.columns:
        scenarios = trades["scenario"].unique()
        if len(scenarios) > 1:
            chosen = scenarios[0]
            print(f"  trades.csv has {len(scenarios)} scenarios. Using '{chosen}'.")
            trades = trades[trades["scenario"] == chosen]

    daily = daily_pnl(trades)
    n_days = len(daily)
    span_days = (daily.index[-1] - daily.index[0]).days if n_days > 1 else 1
    print(f"\nLoaded: {trades_csv}")
    print(f"Trades: {len(trades):,}   Trading days: {n_days:,}   Span: {span_days} days "
          f"({span_days/365.25:.2f} years)")

    # Reference (full-size) stats
    ref = project_at_size(daily, 1.0)
    print(f"\n=== FULL-SIZE STATS (as in your last backtest) ===")
    print(f"  Net PnL/yr      : ${ref['annual_pnl']:>+12,.0f}")
    print(f"  Avg day         : ${ref['avg_day']:>+12,.0f}")
    print(f"  Median day      : ${ref['median_day']:>+12,.0f}")
    print(f"  Worst single day: ${ref['worst_day']:>+12,.0f}")
    print(f"  Worst 3-day     : ${ref['worst_3day']:>+12,.0f}")
    print(f"  Worst 5-day     : ${ref['worst_5day']:>+12,.0f}")
    print(f"  Max drawdown    : ${ref['max_dd']:>+12,.0f}")

    # Constraint check across factors
    factors = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05]
    real_world_dd_mult = args.real_world_dd_mult

    print(f"\n=== CONSTRAINT CHECK across lot-size factors ===")
    print(f"  Account ${args.account:,.0f}   "
          f"Daily limit ${args.max_daily_loss:,.0f}   "
          f"Max DD ${args.max_drawdown:,.0f}   "
          f"Target ${args.target:,.0f}")
    print(f"  Real-world DD multiplier: {real_world_dd_mult}× backtest")
    print()
    print(f"  {'Factor':>7}  {'Worst day':>12}  {'WorstDay×1.5':>14}  "
          f"{'BacktestDD':>12}  {'Real DD':>10}  {'DailyOK':>8}  {'DDOK':>5}")
    print(f"  {'-' * 88}")
    safe_factors = []
    for f in factors:
        s = project_at_size(daily, f)
        worst_inflated = s["worst_day"] * 1.5  # real-world worst-day cushion
        real_dd = s["max_dd"] * real_world_dd_mult
        daily_ok = abs(worst_inflated) < args.max_daily_loss
        dd_ok    = abs(real_dd)        < args.max_drawdown
        flag = "✓" if (daily_ok and dd_ok) else "✗"
        print(f"  {f:>7.2f}  ${s['worst_day']:>+11,.0f}  ${worst_inflated:>+13,.0f}  "
              f"${s['max_dd']:>+11,.0f}  ${real_dd:>+9,.0f}  "
              f"{'✓' if daily_ok else '✗':>8}  {'✓' if dd_ok else '✗':>5}  {flag}")
        if daily_ok and dd_ok:
            safe_factors.append(f)

    if not safe_factors:
        print(f"\n  No size in the grid satisfies both limits — strategy is too "
              f"risky for this prop firm config. Lower the lot size further or "
              f"trim pairs.")
        return

    # Pick the largest safe factor (most aggressive that still passes)
    best_factor = max(safe_factors)
    rec = project_at_size(daily, best_factor)
    print(f"\n=== RECOMMENDED SIZING ===")
    print(f"  Largest factor that survives both rules: {best_factor:g}× full size")
    print(f"  → If 'full size' is 1 lot per pair across 5 pairs concurrently,")
    print(f"    use {best_factor:g} lots per pair (or equivalent fraction).")
    print(f"  Avg day P&L           : ${rec['avg_day']:>+10,.0f}")
    print(f"  Median day P&L        : ${rec['median_day']:>+10,.0f}")
    print(f"  Worst day (backtest)  : ${rec['worst_day']:>+10,.0f}")
    print(f"  Backtest MaxDD        : ${rec['max_dd']:>+10,.0f}")
    print(f"  Real-world MaxDD est. : ${rec['max_dd']*real_world_dd_mult:>+10,.0f}")

    # Risk-per-trade equivalent
    pnl_per_trade = (trades["pnl"] * best_factor).abs().mean()
    sl_pnl_per_trade = trades.loc[trades["exit_kind"] == "sl", "pnl"] * best_factor
    avg_loss = float(sl_pnl_per_trade.mean()) if len(sl_pnl_per_trade) else 0.0
    p95_loss = float(sl_pnl_per_trade.quantile(0.05)) if len(sl_pnl_per_trade) else 0.0
    print(f"\n  Avg per-trade  PnL    : ±${pnl_per_trade:>10,.2f}")
    print(f"  Avg loss (SL)         : ${avg_loss:>+10,.2f}")
    print(f"  95th-percentile loss  : ${p95_loss:>+10,.2f}  (5% of SL trades worse than this)")

    # Time-to-target projections
    print(f"\n=== TIME TO ${args.target:,.0f} PROFIT ===")
    tt = time_to_target(daily, best_factor, args.target)
    for label, days in [
        ("Best 30-day stretch (lucky run)", tt["best"]),
        ("Historical average",              tt["avg"]),
        ("Historical median day",           tt["median"]),
        ("25th-percentile day (pessimistic)", tt["p25"]),
        ("Worst 30-day stretch (unlucky)",  tt["worst"]),
    ]:
        if not np.isfinite(days):
            print(f"  {label:<38}: never (negative rate)")
        else:
            print(f"  {label:<38}: {days:>5.1f} trading days  "
                  f"({days*7/5:>5.1f} calendar days)")

    print(f"\n=== PESSIMISTIC FORECAST ===")
    pess_days = tt["p25"] if np.isfinite(tt["p25"]) else tt["avg"]
    pess_cal = pess_days * 7 / 5
    print(f"  Hitting $15K target in a typical-bad run: ~{pess_days:.0f} trading days "
          f"(≈ {pess_cal:.0f} calendar days, {pess_cal/7:.1f} weeks).")
    print(f"  Hitting $15K with a normal run         : ~{tt['avg']:.0f} trading days "
          f"(≈ {tt['avg']*7/5:.0f} calendar days, {tt['avg']*7/5/7:.1f} weeks).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trades", default=None,
                    help="Path to trades.csv (default: latest in data/backtest_curves/)")
    ap.add_argument("--account",         type=float, default=100_000)
    ap.add_argument("--target",          type=float, default=15_000)
    ap.add_argument("--max-daily-loss",  type=float, default=5_000)
    ap.add_argument("--max-drawdown",    type=float, default=10_000)
    ap.add_argument("--real-world-dd-mult", type=float, default=1.5,
                    help="Multiplier on backtest DD to estimate real-world DD "
                         "(default 1.5: backtest DDs systematically understate "
                         "real-world DDs).")
    args = ap.parse_args()

    trades_csv = Path(args.trades) if args.trades else find_latest_trades_csv()
    report(args, trades_csv)


if __name__ == "__main__":
    main()
