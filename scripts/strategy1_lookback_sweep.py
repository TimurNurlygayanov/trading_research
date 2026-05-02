"""
Lookback sensitivity sweep for Strategy 1 (range fade).

Runs simulate_with_trades across multiple lookback values and timeframes on
the same window, with IB-realistic costs.  Prints portfolio + per-pair tables.

Tightness filter and hour/day filters are disabled by default so the only
variable is `lookback`.

Usage
  python -m scripts.strategy1_lookback_sweep
  python -m scripts.strategy1_lookback_sweep --start 2026-01-01 --end 2026-05-02
  python -m scripts.strategy1_lookback_sweep --lookbacks 10 20 30 40 50 --tfs 1m 5m
  python -m scripts.strategy1_lookback_sweep --lot 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from strategy1_backtest_curves import (   # noqa: E402
    simulate_with_trades, compute_metrics, equity_curve, drawdown_series,
    DEFAULT_SPREADS_PIPS,
)
from backtest.data_fetcher import fetch_ohlcv     # noqa: E402

CONFIG_PATH = Path(__file__).parent / "strategy1_config.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def run_sweep(args, cfg) -> None:
    pairs = list(cfg["pairs"].keys())
    p_base = dict(cfg["params"])
    use_filters = args.use_filters
    apply_costs = not args.no_costs
    spread_mult = float(args.spread_mult)
    lot = args.lot

    print(f"\nWindow: {args.start} → {args.end}   lot={lot}   "
          f"filters={use_filters}   costs={'IB' if apply_costs else 'none'}   "
          f"tight={'1.5' if args.use_tight else 'off'}")
    print(f"TPx={p_base['tp_atr']}  SLx={p_base['sl_atr']}  ATRp={p_base['atr_period']}")

    # Fetch once per (pair, tf)
    data: dict[tuple[str, str], object] = {}
    for tf in args.tfs:
        for sym in pairs:
            print(f"  Fetching {sym} {tf} ...", end=" ", flush=True)
            df = fetch_ohlcv(sym, tf, args.start, args.end)
            print(f"{len(df):,} bars" if not df.empty else "no data")
            if not df.empty:
                data[(sym, tf)] = df

    # Run sweep
    rows: list[dict] = []
    for tf in args.tfs:
        for lb in args.lookbacks:
            p = dict(p_base)
            p["lookback"]  = lb
            p["tight_atr"] = p_base["tight_atr"] if args.use_tight else None

            per_pair: dict[str, dict] = {}
            for sym in pairs:
                df = data.get((sym, tf))
                if df is None:
                    continue
                spread = DEFAULT_SPREADS_PIPS.get(sym, 1.0) * spread_mult
                sim = simulate_with_trades(
                    df, p, cfg["pairs"][sym], lot, sym,
                    use_filters=use_filters,
                    spread_pips=spread,
                    apply_costs=apply_costs,
                )
                per_pair[sym] = {
                    "trades": sim.trades,
                    "metrics": compute_metrics(sim.trades, df),
                }

            # Portfolio aggregate from concatenated trade pnl events
            all_eq = []
            for sym, payload in per_pair.items():
                eq = equity_curve(payload["trades"])
                if not eq.empty:
                    all_eq.append(eq.diff().fillna(eq.iloc[0]))
            if all_eq:
                import pandas as pd
                events = pd.concat(all_eq).sort_index()
                port_eq = events.cumsum()
                port_pnl = float(port_eq.iloc[-1])
                port_dd  = float(drawdown_series(port_eq).min())
            else:
                port_pnl = 0.0
                port_dd  = 0.0

            n_trades = sum(m["metrics"]["n_trades"] for m in per_pair.values())
            n_tp     = sum(m["metrics"]["n_tp"]     for m in per_pair.values())
            n_sl     = sum(m["metrics"]["n_sl"]     for m in per_pair.values())
            wr       = n_tp / (n_tp + n_sl) if (n_tp + n_sl) else 0.0
            avg_sharpe = float(np.mean([m["metrics"]["sharpe"]
                                        for m in per_pair.values()
                                        if m["metrics"]["n_trades"] > 0])) \
                         if per_pair else 0.0

            rows.append({
                "tf":         tf,
                "lookback":   lb,
                "trades":     n_trades,
                "win_rate":   wr,
                "net_pnl":    port_pnl,
                "max_dd":     port_dd,
                "avg_sharpe": avg_sharpe,
                "per_pair":   {s: m["metrics"]["net_pnl"] for s, m in per_pair.items()},
            })

    # Portfolio summary table
    print("\n  === PORTFOLIO SUMMARY ===")
    print(f"  {'TF':<4}  {'LB':>4}  {'#Trades':>8}  {'Win%':>6}  "
          f"{'Net PnL':>10}  {'MaxDD':>10}  {'AvgSharpe':>10}")
    print(f"  {'-'*60}")
    for r in rows:
        print(f"  {r['tf']:<4}  {r['lookback']:>4}  {r['trades']:>8,}  "
              f"{r['win_rate']*100:>5.1f}%  ${r['net_pnl']:>+9,.0f}  "
              f"${r['max_dd']:>+9,.0f}  {r['avg_sharpe']:>10.2f}")

    # Per-pair tables (one per timeframe)
    for tf in args.tfs:
        print(f"\n  === Per-pair Net PnL  ({tf}) ===")
        header = f"  {'LB':>4}  " + "  ".join(f"{s:>10}" for s in pairs) + f"  {'TOTAL':>10}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for r in rows:
            if r["tf"] != tf:
                continue
            cells = [f"${r['per_pair'].get(s, 0):>+9,.0f}" for s in pairs]
            total = sum(r['per_pair'].values())
            print(f"  {r['lookback']:>4}  " + "  ".join(cells) + f"  ${total:>+9,.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start",       default="2026-01-01")
    ap.add_argument("--end",         default="2026-05-02")
    ap.add_argument("--lookbacks",   nargs="+", type=int,
                    default=[10, 20, 30, 40, 50])
    ap.add_argument("--tfs",         nargs="+", default=["1m", "5m"])
    ap.add_argument("--lot",         type=float, default=1.0,
                    help="Lot size (default 1.0 — 0.1 is unprofitable due to "
                         "IB commission floor)")
    ap.add_argument("--use-filters", action="store_true",
                    help="Apply hour/day filters from config (1m-tuned)")
    ap.add_argument("--use-tight",   action="store_true",
                    help="Apply tightness filter from config (1.5×ATR)")
    ap.add_argument("--no-costs",    action="store_true")
    ap.add_argument("--spread-mult", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config()
    run_sweep(args, cfg)


if __name__ == "__main__":
    main()
