"""
Portfolio range-fade — all USD pairs run simultaneously.

Loads per-pair hour/day filters from scripts/strategy1_config.json and runs
the range-fade strategy on each pair independently.  Reports per-pair PnL and
portfolio totals.

Usage
  python -m scripts.strategy1_portfolio
  python -m scripts.strategy1_portfolio --start 2023-01-01 --end 2025-01-01
  python -m scripts.strategy1_portfolio --no-filters
  python -m scripts.strategy1_portfolio --years 2022 2023 2024
  python -m scripts.strategy1_portfolio --years 2022 2023 2024 2025
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from strategy1 import simulate, _prepare          # noqa: E402
from backtest.data_fetcher import fetch_ohlcv     # noqa: E402

CONFIG_PATH = Path(__file__).parent / "strategy1_config.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _year_range(yr: int) -> tuple[str, str]:
    start = f"{yr}-01-01"
    end   = min(date(yr + 1, 1, 1), date.today()).isoformat()
    return start, end


def run_portfolio(start: str, end: str, cfg: dict,
                  use_filters: bool) -> dict[str, dict]:
    """Run simulate() on every pair in cfg.  Returns {symbol: result}."""
    p      = cfg["params"]
    tf     = p["timeframe"]
    results: dict[str, dict] = {}

    for sym, pcfg in cfg["pairs"].items():
        print(f"  Fetching {sym} {tf}  {start} ...", end=" ", flush=True)
        df = fetch_ohlcv(sym, tf, start, end)
        if df.empty:
            print("no data")
            results[sym] = {"total": 0}
            continue
        print(f"{len(df):,} bars")

        close, low, high, atr_vals, rh, rl, warmup, bh, bd = _prepare(
            df, p["atr_period"], p["lookback"])

        if use_filters:
            hour_filter = set(pcfg["hours"])
            dow_filter  = set(pcfg["days"])
        else:
            hour_filter = dow_filter = None

        results[sym] = simulate(
            close, low, high, atr_vals, rh, rl,
            p["tp_atr"], p["sl_atr"], p["commission"],
            p["tight_atr"], warmup, fade=True,
            usd_quote=pcfg["usd_quote"],
            bar_hours=bh, bar_dows=bd,
            hour_filter=hour_filter, dow_filter=dow_filter)

    return results


def _print_results(results: dict[str, dict], label: str) -> None:
    symbols  = list(results.keys())
    col_w    = 12
    sep      = "-" * (10 + col_w * len(symbols) + 2)

    def _row(lbl: str, vals: list[str]) -> str:
        return f"  {lbl:<8}" + "".join(f"{v:>{col_w}}" for v in vals)

    print(f"\n  {label}")
    print(f"  {sep}")

    for metric, title, fmt in [
        ("net_pnl",  "Net PnL ($)",  lambda v: f"${v:+.0f}"),
        ("win_rate", "Win rate (%)", lambda v: f"{v*100:.1f}%"),
        ("total",    "# Trades",     lambda v: str(int(v))),
    ]:
        print(f"\n  [{title}]")
        print(_row("", symbols))
        print(f"  {'-' * (8 + col_w * len(symbols))}")
        vals = []
        for sym in symbols:
            r = results[sym]
            vals.append(fmt(r[metric]) if r and r["total"] > 0 else "n/a")
        print(_row("", vals))

    # Portfolio totals
    total_pnl    = sum(r.get("net_pnl", 0) for r in results.values())
    total_trades = sum(r.get("total",   0) for r in results.values())
    total_tp     = sum(r.get("n_tp",    0) for r in results.values())
    total_sl     = sum(r.get("n_sl",    0) for r in results.values())
    port_wr      = total_tp / (total_tp + total_sl) if (total_tp + total_sl) > 0 else 0.0

    print(f"\n  {'─'*40}")
    print(f"  Portfolio  Net PnL : ${total_pnl:+,.0f}")
    print(f"  Portfolio  Trades  : {total_trades:,}")
    print(f"  Portfolio  Win rate: {port_wr*100:.1f}%")


def _print_yearly(yearly: dict[int, dict[str, dict]], symbols: list[str],
                  cfg: dict) -> None:
    p      = cfg["params"]
    col_w  = 12

    def _row(lbl: str, vals: list[str]) -> str:
        return f"  {lbl:<8}" + "".join(f"{v:>{col_w}}" for v in vals)

    for metric, title, fmt in [
        ("net_pnl",  "Net PnL ($)",  lambda v: f"${v:+.0f}"),
        ("win_rate", "Win rate (%)", lambda v: f"{v*100:.1f}%"),
        ("total",    "# Trades",     lambda v: str(int(v))),
    ]:
        print(f"\n  [{title}]  TP {p['tp_atr']}x  SL {p['sl_atr']}x  "
              f"tight {p['tight_atr']}x  {p['timeframe']}")
        print(_row("", symbols + ["TOTAL"]))
        print(f"  {'-' * (8 + col_w * (len(symbols) + 1))}")
        for yr, res in sorted(yearly.items()):
            row_vals = []
            for sym in symbols:
                r = res.get(sym, {})
                row_vals.append(fmt(r[metric]) if r and r.get("total", 0) > 0 else "n/a")
            # total column
            if metric == "net_pnl":
                tot = sum(r.get("net_pnl", 0) for r in res.values())
                row_vals.append(f"${tot:+.0f}")
            elif metric == "win_rate":
                tp = sum(r.get("n_tp", 0) for r in res.values())
                sl = sum(r.get("n_sl", 0) for r in res.values())
                wr = tp / (tp + sl) if (tp + sl) > 0 else 0.0
                row_vals.append(f"{wr*100:.1f}%")
            else:
                row_vals.append(str(sum(r.get("total", 0) for r in res.values())))
            print(_row(str(yr), row_vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start",      default=None,
                    help="Start date (overrides --years)")
    ap.add_argument("--end",        default=None,
                    help="End date (overrides --years)")
    ap.add_argument("--years",      nargs="+", type=int, default=None,
                    help="Calendar years to run (default: 2022 2023 2024)")
    ap.add_argument("--no-filters", action="store_true",
                    help="Ignore hour/day filters — trade all bars")
    args = ap.parse_args()

    cfg        = load_config()
    symbols    = list(cfg["pairs"].keys())
    use_filters = not args.no_filters
    filter_str  = "with filters" if use_filters else "NO filters"

    if args.start and args.end:
        # Single range mode
        print(f"\nFetching data  {args.start} → {args.end}  ({filter_str})")
        results = run_portfolio(args.start, args.end, cfg, use_filters)
        _print_results(results, f"{args.start} → {args.end}  [{filter_str}]")
        return

    # Multi-year mode
    years = args.years or [2022, 2023, 2024]
    yearly: dict[int, dict[str, dict]] = {}
    for yr in years:
        start, end = _year_range(yr)
        print(f"\n── {yr}  ({filter_str}) ──")
        yearly[yr] = run_portfolio(start, end, cfg, use_filters)

    _print_yearly(yearly, symbols, cfg)


if __name__ == "__main__":
    main()
