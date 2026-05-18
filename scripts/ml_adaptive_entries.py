"""
CLI: walk-forward CatBoost entry classifier with FTMO-compliant simulation.

Examples:
  # Fast smoke test on EURUSD 5m, ~5 minutes:
  python scripts/ml_adaptive_entries.py --symbol EURUSD --timeframe 5m \
      --start 2025-01-01 --end 2026-05-01 --fast --no-1m

  # Heavy run, with 1m label/sim resolution:
  python scripts/ml_adaptive_entries.py --symbol EURUSD --timeframe 5m \
      --start 2025-01-01 --end 2026-05-01 --heavy
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data import load_ohlcv, resample_ohlcv
from ml.leakage import static_scan
from ml.report import build_report
from ml.walkforward import run_walkforward


_DEFAULT_SPREAD_PIPS = {
    "EURUSD": 0.8, "GBPUSD": 1.0, "USDJPY": 1.0,
    "AUDUSD": 1.0, "NZDUSD": 1.2, "USDCAD": 1.2, "USDCHF": 1.2,
}

_PIP_SIZE = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01,
    "AUDUSD": 0.0001, "NZDUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001,
}


def _tf_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(tf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2026-05-01")
    p.add_argument("--train-months", type=int, default=3)
    p.add_argument("--trade-months", type=int, default=1)
    p.add_argument("--rr", type=float, default=3.0)
    p.add_argument("--sl-atr", type=float, default=1.5)
    p.add_argument("--max-hold-bars", type=int, default=48)
    p.add_argument("--risk-pct", type=float, default=0.005)
    p.add_argument("--spread-pips", type=float, default=None)
    p.add_argument("--commission", type=float, default=0.0002)
    p.add_argument("--start-equity", type=float, default=100_000.0)
    p.add_argument("--daily-loss-pct", type=float, default=0.04)
    p.add_argument("--max-loss-pct", type=float, default=0.08)
    p.add_argument("--min-precision", type=float, default=0.45)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--heavy", action="store_true")
    p.add_argument("--depth", type=int, default=None, help="Override CatBoost depth")
    p.add_argument("--iterations", type=int, default=None, help="Override CatBoost iterations")
    p.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    p.add_argument("--max-windows", type=int, default=None, help="Stop after N windows (diagnostic)")
    p.add_argument("--side-only", choices=["long", "short"], default=None,
                   help="Restrict trading to one side only (model still trained on both)")
    p.add_argument("--proba-threshold", type=float, default=None,
                   help="Override auto-tuned decision threshold (e.g. 0.7 = only high-confidence)")
    p.add_argument("--ema-filter", action="store_true",
                   help="Hard filter: skip longs when EMA50 slope < 0, skip shorts when EMA50 slope > 0")
    p.add_argument("--slippage-pips-sl", type=float, default=0.0,
                   help="Adverse pip slippage on SL fills (real FTMO: 0.3-1.0)")
    p.add_argument("--slippage-pips-tp", type=float, default=0.0,
                   help="Adverse pip slippage on TP fills (real FTMO: ~0)")
    p.add_argument("--ftmo-realistic", action="store_true",
                   help="Preset: spread 1.2, commission 0.00004, SL slippage 0.7 pips, TP slippage 0.2 pips")
    p.add_argument("--no-1m", action="store_true", help="Skip 1m resolution (faster, less accurate)")
    p.add_argument("--no-htf", action="store_true", help="Skip higher-TF features")
    p.add_argument("--skip-every-other-window", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--scan-only", action="store_true", help="Run leakage static scan and exit")
    p.add_argument("--source", choices=["mt5", "polygon"], default=None,
                   help="Filter data files by source tag (default: any)")
    args = p.parse_args()

    if args.fast and args.heavy:
        sys.exit("Pass exactly one of --fast / --heavy")
    if not (args.fast or args.heavy):
        args.fast = True

    if args.ftmo_realistic:
        if args.spread_pips is None:
            args.spread_pips = 1.2
        args.commission = 0.00004
        if args.slippage_pips_sl == 0.0:
            args.slippage_pips_sl = 0.7
        if args.slippage_pips_tp == 0.0:
            args.slippage_pips_tp = 0.2

    # Static leakage scan on features.py
    print(">>> static leakage scan")
    root = Path(__file__).resolve().parent.parent
    scan = static_scan(root / "ml" / "features.py")
    if scan.issues:
        print("LEAKAGE ISSUES FOUND:")
        for i in scan.issues:
            print(f"  - {i}")
        sys.exit("Aborting. Fix leakage before training.")
    print("  clean.")
    if args.scan_only:
        return

    # Spread / pip
    spread_pips = args.spread_pips if args.spread_pips is not None else _DEFAULT_SPREAD_PIPS.get(args.symbol, 1.0)
    pip = _PIP_SIZE.get(args.symbol, 0.0001)
    spread = spread_pips * pip
    pip_value_quote = 100.0 if args.symbol.endswith("JPY") else 1.0  # for USDJPY etc.

    # Load data
    print(f">>> loading {args.symbol} {args.timeframe}")
    df = load_ohlcv(args.symbol, args.timeframe, args.start, args.end, source=args.source)
    print(f"  {len(df):,} bars, {df.index.min()} - {df.index.max()}")
    if df.empty:
        sys.exit("No data.")

    # Higher TF (1h for 5m)
    df_htf = None
    if not args.no_htf:
        try:
            df_htf = load_ohlcv(args.symbol, "1h", args.start, args.end, source=args.source)
            print(f"  HTF 1h: {len(df_htf):,} bars")
        except FileNotFoundError:
            df_htf = resample_ohlcv(df, "1h")
            print(f"  HTF 1h (resampled): {len(df_htf):,} bars")

    # 1m data for label/sim resolution
    df_1m = None
    if not args.no_1m:
        try:
            df_1m = load_ohlcv(args.symbol, "1m", args.start, args.end)
            print(f"  1m: {len(df_1m):,} bars")
        except FileNotFoundError:
            print("  no 1m data — falling back to TF resolution")

    # Run
    t0 = time.time()
    wf = run_walkforward(
        df_full=df,
        df_htf=df_htf,
        df_1m=df_1m,
        start=pd.Timestamp(args.start, tz="UTC"),
        end=pd.Timestamp(args.end, tz="UTC"),
        train_months=args.train_months,
        trade_months=args.trade_months,
        sl_atr=args.sl_atr,
        rr=args.rr,
        max_hold_bars=args.max_hold_bars,
        spread=spread,
        commission=args.commission,
        risk_pct=args.risk_pct,
        tf_minutes=_tf_minutes(args.timeframe),
        fast=args.fast,
        min_precision=args.min_precision,
        daily_loss_pct=args.daily_loss_pct,
        max_loss_pct=args.max_loss_pct,
        start_equity=args.start_equity,
        skip_every_other=args.skip_every_other_window,
        model_overrides={"depth": args.depth, "iterations": args.iterations, "learning_rate": args.lr},
        max_windows=args.max_windows,
        side_only=args.side_only,
        proba_threshold_override=args.proba_threshold,
        ema_filter=args.ema_filter,
        slippage_pips_sl=args.slippage_pips_sl,
        slippage_pips_tp=args.slippage_pips_tp,
        pip_size=pip,
    )
    elapsed = time.time() - t0
    print(f"\n>>> walk-forward done in {elapsed/60:.1f} min")

    # Report
    out_dir = Path(args.out) if args.out else (root / "runs" / f"{args.symbol}_{args.timeframe}_{uuid.uuid4().hex[:6]}")
    params = vars(args) | {"spread": spread, "pip": pip, "pip_value_quote": pip_value_quote}
    build_report(wf, out_dir, params)


if __name__ == "__main__":
    main()
