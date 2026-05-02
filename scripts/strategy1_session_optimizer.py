"""
Find optimal trading sessions (UTC hours / days-of-week) per pair, for two
configurations:
  - 1m timeframe, lookback 10
  - 5m timeframe, lookback 30

Method
  1. TRAIN: baseline sim on the train window (no filters)
  2. Bucket each trade by entry-hour and entry-DOW; pick positive-PnL buckets
     meeting min-trade-count and min-expectancy thresholds
  3. TRAIN-VALIDATE: re-run with chosen filters → measure in-sample fit
  4. TEST (out-of-sample, optional): apply the chosen filters to the held-out
     window — this is the metric you should trust
  5. Save JSON config snippet usable in strategy1_config.json structure

5m bars are resampled from cached 1m to avoid API calls.

Usage
  # Single-window in-sample (legacy)
  python -m scripts.strategy1_session_optimizer --start 2024-01-01 --end 2026-05-02

  # Train on 2024-2025, test on 2026, drop GBPUSD
  python -m scripts.strategy1_session_optimizer \\
    --start 2024-01-01 --train-end 2026-01-01 \\
    --test-start 2026-01-01 --end 2026-05-02 \\
    --exclude GBPUSD
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ── Data loading w/ 1m → 5m resampling ────────────────────────────────────────

_RESAMPLE_RULE = {"1m": None, "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}


def get_bars(symbol: str, tf: str, start: str, end: str) -> pd.DataFrame:
    """1m straight from cache; everything else resampled from 1m."""
    df_1m = fetch_ohlcv(symbol, "1m", start, end)
    rule = _RESAMPLE_RULE.get(tf)
    if rule is None:
        return df_1m
    if df_1m.empty:
        return df_1m
    out = df_1m.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Open", "High", "Low", "Close"])
    return out


# ── Bucketing ─────────────────────────────────────────────────────────────────

def bucket_by_hour(trades) -> dict[int, list[float]]:
    out: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        out[int(t.entry_time.hour)].append(t.pnl)
    return out


def bucket_by_dow(trades) -> dict[int, list[float]]:
    out: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        out[int(t.entry_time.dayofweek)].append(t.pnl)
    return out


def select_good_buckets(buckets: dict[int, list[float]],
                        min_trades: int,
                        min_expectancy: float,
                        min_total_pnl: float) -> list[int]:
    good = []
    for k, pnls in buckets.items():
        if len(pnls) < min_trades:
            continue
        total = float(np.sum(pnls))
        expect = float(np.mean(pnls))
        if total >= min_total_pnl and expect >= min_expectancy:
            good.append(k)
    return sorted(good)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_hour_table(symbol: str, buckets: dict[int, list[float]]) -> None:
    print(f"\n  [{symbol}] Net PnL by entry-hour (UTC)")
    print(f"  {'H':>3}  {'Trades':>7}  {'Net PnL':>11}  {'Expect.':>9}  {'Win%':>6}")
    for h in range(24):
        pnls = buckets.get(h, [])
        if not pnls:
            print(f"  {h:>3}  {'-':>7}  {'-':>11}  {'-':>9}  {'-':>6}")
            continue
        wins = sum(1 for p in pnls if p > 0)
        total = float(np.sum(pnls))
        exp_  = float(np.mean(pnls))
        wr    = wins / len(pnls)
        print(f"  {h:>3}  {len(pnls):>7,}  ${total:>+10,.0f}  ${exp_:>+8.2f}  {wr*100:>5.1f}%")


def print_dow_table(symbol: str, buckets: dict[int, list[float]]) -> None:
    print(f"\n  [{symbol}] Net PnL by day-of-week (UTC)")
    print(f"  {'DOW':>3}  {'Trades':>7}  {'Net PnL':>11}  {'Expect.':>9}  {'Win%':>6}")
    for d in range(7):
        pnls = buckets.get(d, [])
        if not pnls:
            print(f"  {DOW_NAMES[d]:>3}  {'-':>7}  {'-':>11}  {'-':>9}  {'-':>6}")
            continue
        wins = sum(1 for p in pnls if p > 0)
        total = float(np.sum(pnls))
        exp_  = float(np.mean(pnls))
        wr    = wins / len(pnls)
        print(f"  {DOW_NAMES[d]:>3}  {len(pnls):>7,}  ${total:>+10,.0f}  ${exp_:>+8.2f}  {wr*100:>5.1f}%")


# ── Driver ────────────────────────────────────────────────────────────────────

def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC")
    return df[(df.index >= s) & (df.index < e)]


def _run_sim(df, p, pair_cfg, sym, args, hours=None, days=None):
    """Run a single sim. If hours/days are None, no filter is applied."""
    spread = DEFAULT_SPREADS_PIPS.get(sym, 1.0) * args.spread_mult
    use_filters = hours is not None or days is not None
    pair_cfg_eff = dict(pair_cfg)
    pair_cfg_eff["hours"] = hours if hours is not None else list(range(24))
    pair_cfg_eff["days"]  = days  if days  is not None else list(range(7))
    sim = simulate_with_trades(
        df, p, pair_cfg_eff, args.lot, sym,
        use_filters=use_filters, spread_pips=spread, apply_costs=True)
    return sim, compute_metrics(sim.trades, df)


def optimize(args, cfg) -> dict:
    pairs = [p for p in cfg["pairs"].keys() if p not in (args.exclude or [])]
    if args.exclude:
        print(f"Excluding pairs: {args.exclude}")
    p_base = dict(cfg["params"])
    p_base["tight_atr"] = None

    configs = [
        {"tf": "1m", "lookback": args.lookback_1m},
        {"tf": "5m", "lookback": args.lookback_5m},
    ]

    has_test = bool(args.test_start)
    train_start = args.start
    train_end   = args.train_end or args.end
    test_start  = args.test_start
    test_end    = args.end

    if has_test:
        print(f"TRAIN: {train_start} → {train_end}    "
              f"TEST: {test_start} → {test_end}    "
              f"lot={args.lot}   costs=IB (spread×{args.spread_mult:g})")
    else:
        print(f"Window: {train_start} → {train_end}   lot={args.lot}   "
              f"costs=IB (spread×{args.spread_mult:g})")
    print(f"Min trades/bucket: {args.min_trades}   "
          f"min expectancy: ${args.min_expectancy:.2f}   "
          f"min total PnL: ${args.min_total_pnl:.0f}")

    out: dict = {
        "train_window": {"start": train_start, "end": train_end},
        "test_window":  ({"start": test_start, "end": test_end} if has_test else None),
        "excluded":     args.exclude or [],
        "results":      {},
    }

    for cfg_run in configs:
        tf, lb = cfg_run["tf"], cfg_run["lookback"]
        p = dict(p_base); p["lookback"] = lb
        key = f"{tf}_lb{lb}"
        out["results"][key] = {"timeframe": tf, "lookback": lb, "pairs": {}}

        print(f"\n{'='*78}\n  CONFIG: {tf}  lookback={lb}\n{'='*78}")

        # Fetch full range once, slice for train/test
        full_end = test_end if has_test else train_end
        bars_per_pair: dict[str, pd.DataFrame] = {}
        for sym in pairs:
            df = get_bars(sym, tf, train_start, full_end)
            print(f"  {sym} {tf}: {len(df):,} bars" if not df.empty
                  else f"  {sym} {tf}: no data")
            if not df.empty:
                bars_per_pair[sym] = df

        rows = []
        for sym, df_full in bars_per_pair.items():
            df_train = _slice(df_full, train_start, train_end)
            df_test  = _slice(df_full, test_start, test_end) if has_test else None

            pair_cfg = cfg["pairs"][sym]

            # ─ TRAIN: baseline (no filter) → bucket → pick filters
            sim_train_base, mt_train_base = _run_sim(df_train, p, pair_cfg, sym, args)
            hour_buckets = bucket_by_hour(sim_train_base.trades)
            dow_buckets  = bucket_by_dow(sim_train_base.trades)
            good_hours = select_good_buckets(
                hour_buckets, args.min_trades, args.min_expectancy, args.min_total_pnl)
            good_days  = select_good_buckets(
                dow_buckets,  args.min_trades, args.min_expectancy, args.min_total_pnl)

            if args.show_buckets:
                print_hour_table(sym, hour_buckets)
                print_dow_table(sym, dow_buckets)

            # ─ TRAIN: confirmation (in-sample fit)
            _, mt_train_opt = _run_sim(df_train, p, pair_cfg, sym, args,
                                       hours=good_hours, days=good_days)

            # ─ TEST: baseline + optimized (out-of-sample)
            if has_test:
                _, mt_test_base = _run_sim(df_test, p, pair_cfg, sym, args)
                _, mt_test_opt  = _run_sim(df_test, p, pair_cfg, sym, args,
                                            hours=good_hours, days=good_days)
            else:
                mt_test_base = mt_test_opt = None

            print(f"\n  → {sym} chosen hours: {good_hours}")
            print(f"  → {sym} chosen days : {[DOW_NAMES[d] for d in good_days]}  ({good_days})")
            print(f"     TRAIN base : {mt_train_base['n_trades']:>6,} trades  "
                  f"${mt_train_base['net_pnl']:>+10,.0f}  "
                  f"Shp={mt_train_base['sharpe']:.2f}  "
                  f"DD=${mt_train_base['max_dd']:>+8,.0f}")
            print(f"     TRAIN opt  : {mt_train_opt['n_trades']:>6,} trades  "
                  f"${mt_train_opt['net_pnl']:>+10,.0f}  "
                  f"Shp={mt_train_opt['sharpe']:.2f}  "
                  f"DD=${mt_train_opt['max_dd']:>+8,.0f}")
            if has_test:
                print(f"     TEST  base : {mt_test_base['n_trades']:>6,} trades  "
                      f"${mt_test_base['net_pnl']:>+10,.0f}  "
                      f"Shp={mt_test_base['sharpe']:.2f}  "
                      f"DD=${mt_test_base['max_dd']:>+8,.0f}")
                print(f"     TEST  opt  : {mt_test_opt['n_trades']:>6,} trades  "
                      f"${mt_test_opt['net_pnl']:>+10,.0f}  "
                      f"Shp={mt_test_opt['sharpe']:.2f}  "
                      f"DD=${mt_test_opt['max_dd']:>+8,.0f}")

            rows.append({
                "symbol":      sym,
                "good_hours":  good_hours,
                "good_days":   good_days,
                "train_base":  mt_train_base,
                "train_opt":   mt_train_opt,
                "test_base":   mt_test_base,
                "test_opt":    mt_test_opt,
            })

            out["results"][key]["pairs"][sym] = {
                "usd_quote": pair_cfg.get("usd_quote", True),
                "hours":     good_hours,
                "days":      good_days,
                "train_base_pnl":    mt_train_base["net_pnl"],
                "train_opt_pnl":     mt_train_opt["net_pnl"],
                "train_base_sharpe": mt_train_base["sharpe"],
                "train_opt_sharpe":  mt_train_opt["sharpe"],
                "test_base_pnl":     mt_test_base["net_pnl"]    if has_test else None,
                "test_opt_pnl":      mt_test_opt["net_pnl"]     if has_test else None,
                "test_base_sharpe":  mt_test_base["sharpe"]     if has_test else None,
                "test_opt_sharpe":   mt_test_opt["sharpe"]      if has_test else None,
            }

        # Portfolio summary table
        print(f"\n  === {tf} lb={lb} portfolio summary ===")
        if has_test:
            print(f"  {'Pair':<8}  {'TR base':>10}  {'TR opt':>10}  "
                  f"{'TE base':>10}  {'TE opt':>10}  "
                  f"{'TE Δ':>9}  {'TE base Shp':>11}  {'TE opt Shp':>11}")
            print(f"  {'-'*97}")
            tr_b = tr_o = te_b = te_o = 0.0
            for r in rows:
                tb, to_ = r["train_base"]["net_pnl"], r["train_opt"]["net_pnl"]
                eb, eo  = r["test_base"]["net_pnl"],  r["test_opt"]["net_pnl"]
                tr_b += tb; tr_o += to_; te_b += eb; te_o += eo
                print(f"  {r['symbol']:<8}  ${tb:>+9,.0f}  ${to_:>+9,.0f}  "
                      f"${eb:>+9,.0f}  ${eo:>+9,.0f}  ${eo - eb:>+8,.0f}  "
                      f"{r['test_base']['sharpe']:>11.2f}  "
                      f"{r['test_opt']['sharpe']:>11.2f}")
            print(f"  {'TOTAL':<8}  ${tr_b:>+9,.0f}  ${tr_o:>+9,.0f}  "
                  f"${te_b:>+9,.0f}  ${te_o:>+9,.0f}  ${te_o - te_b:>+8,.0f}")
        else:
            print(f"  {'Pair':<8}  {'Base Net':>11}  {'Opt Net':>11}  {'Δ':>10}  "
                  f"{'Base Shp':>9}  {'Opt Shp':>9}")
            tr_b = tr_o = 0.0
            for r in rows:
                tb, to_ = r["train_base"]["net_pnl"], r["train_opt"]["net_pnl"]
                tr_b += tb; tr_o += to_
                print(f"  {r['symbol']:<8}  ${tb:>+10,.0f}  ${to_:>+10,.0f}  "
                      f"${to_ - tb:>+9,.0f}  "
                      f"{r['train_base']['sharpe']:>9.2f}  "
                      f"{r['train_opt']['sharpe']:>9.2f}")
            print(f"  {'TOTAL':<8}  ${tr_b:>+10,.0f}  ${tr_o:>+10,.0f}  ${tr_o - tr_b:>+9,.0f}")

    return out


def write_config_snippets(out: dict, args) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "data" / "session_optimizer" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full optimizer dump
    (out_dir / "optimizer_run.json").write_text(json.dumps(out, indent=2, default=str))

    # Per-config drop-in snippet for strategy1_config.json
    for key, cfg_block in out["results"].items():
        snippet = {
            "_comment": (f"Auto-generated by strategy1_session_optimizer.py — "
                         f"window {out['window']['start']} → {out['window']['end']}"),
            "strategy": "range_fade",
            "params": {
                "lookback":    cfg_block["lookback"],
                "atr_period":  14,
                "tp_atr":      1.0,
                "sl_atr":      2.0,
                "tight_atr":   None,
                "commission":  7.0,  # legacy field; real cost model in backtest
                "timeframe":   cfg_block["timeframe"],
            },
            "pairs": {
                sym: {
                    "usd_quote": data["usd_quote"],
                    "hours":     data["hours"],
                    "days":      data["days"],
                }
                for sym, data in cfg_block["pairs"].items()
            },
        }
        path = out_dir / f"strategy1_config_{key}.json"
        path.write_text(json.dumps(snippet, indent=2))
        print(f"  Wrote {path}")

    print(f"\n  All artifacts in {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start",          default="2024-01-01",
                    help="Train window start (default 2024-01-01)")
    ap.add_argument("--end",            default="2026-05-02",
                    help="End of full window. If --test-start is given, "
                         "this is the test-window end.")
    ap.add_argument("--train-end",      default=None,
                    help="Train-window end. Required for OOS mode.")
    ap.add_argument("--test-start",     default=None,
                    help="Test-window start. Triggers OOS mode.")
    ap.add_argument("--exclude",        nargs="+", default=None,
                    help="Pair symbols to drop entirely (e.g. --exclude GBPUSD)")
    ap.add_argument("--lookback-1m",    type=int, default=10)
    ap.add_argument("--lookback-5m",    type=int, default=30)
    ap.add_argument("--lot",            type=float, default=1.0,
                    help="Lot size (default 1.0)")
    ap.add_argument("--min-trades",     type=int, default=50,
                    help="Minimum trades in a bucket to consider it (default 50)")
    ap.add_argument("--min-expectancy", type=float, default=0.0,
                    help="Minimum per-trade expectancy ($) for a bucket "
                         "(default 0)")
    ap.add_argument("--min-total-pnl",  type=float, default=0.0,
                    help="Minimum cumulative PnL ($) for a bucket (default 0)")
    ap.add_argument("--spread-mult",    type=float, default=1.0)
    ap.add_argument("--show-buckets",   action="store_true",
                    help="Print per-hour and per-DOW tables (verbose)")
    args = ap.parse_args()

    cfg = load_config()
    out = optimize(args, cfg)
    write_config_snippets(out, args)


if __name__ == "__main__":
    main()
