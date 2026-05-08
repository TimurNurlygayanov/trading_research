"""
MFE (Max Favorable Excursion) analysis for the double-EMA extrema strategy.

For each trade, measures the maximum intrabar move in the trade's direction
from entry until the opposite signal fires.

Reports:
  - Percentile distribution of MFE in pips
  - Hit-rate at user-specified TP targets
  - Optimal TP levels for given hit-rate thresholds

Usage
-----
  python -m scripts.analyze_mfe \\
      --login 1513313327 --server FTMO-Demo \\
      --timeframe 5m --start 2026-01-01 --end 2026-05-07 \\
      --ema-period 9 --window 20 --min-swing 0
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

from scripts.strategy1_regime_backtest import (
    _load_data_mt5, _ema, _atr,
    PAIRS, HALF_SPREAD_PIPS, PIP_VALUE, PIP_SIZE, COMMISSION_PER_LOT,
)


def simulate_with_mfe(
    pair: str,
    df: pd.DataFrame,
    ema_period: int,
    window: int,
    min_swing_pips: float,
) -> list[dict]:
    """Simulate double-EMA signal, record MFE for each trade."""
    opens  = df["Open"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values

    hl2   = (highs + lows) / 2.0
    ema_v = _ema(_ema(hl2, ema_period), ema_period)

    min_swing = min_swing_pips * PIP_SIZE
    warmup = ema_period * 2 + window + 5

    def _sig(i: int) -> int:
        dw      = ema_v[i - window + 1 : i + 1]
        pos_max = int(np.argmax(dw))
        pos_min = int(np.argmin(dw))
        if pos_max == pos_min:
            return 0
        if (dw[pos_max] - dw[pos_min]) < min_swing:
            return 0
        if pos_max > pos_min and pos_max < window - 1:
            return -1
        if pos_min > pos_max and pos_min < window - 1:
            return 1
        return 0

    position = 0
    trade: dict | None = None
    trades: list[dict] = []

    for i in range(warmup, len(df)):
        bar_high  = highs[i]
        bar_low   = lows[i]
        bar_close = closes[i]

        if trade is not None:
            # Update intrabar MFE (check from first bar after entry)
            if i > trade["entry_idx"]:
                if position == 1:
                    trade["mfe"] = max(trade["mfe"], bar_high - trade["entry"])
                else:
                    trade["mfe"] = max(trade["mfe"], trade["entry"] - bar_low)

            sig = _sig(i)
            if sig != 0 and sig != position:
                trades.append({
                    "direction": position,
                    "entry":     trade["entry"],
                    "exit":      bar_close,
                    "entry_idx": trade["entry_idx"],
                    "exit_idx":  i,
                    "hold_bars": i - trade["entry_idx"],
                    "mfe_pips":  trade["mfe"] / PIP_SIZE,
                    "net_pips":  (bar_close - trade["entry"]) * position / PIP_SIZE,
                })
                position = sig
                trade = {"entry_idx": i, "entry": bar_close, "mfe": 0.0}
            continue

        sig = _sig(i)
        if sig != 0:
            position = sig
            trade = {"entry_idx": i, "entry": bar_close, "mfe": 0.0}

    return trades


def mfe_report(trades: list[dict], tf: str,
               tp_pips: float, tp_lots: float, entry_lots: float,
               target_usd: float, pair: str = "EURUSD") -> None:
    if not trades:
        print("  No trades."); return

    mfe = np.array([t["mfe_pips"] for t in trades])
    pip_val = PIP_VALUE[pair]
    hs      = HALF_SPREAD_PIPS[pair]
    comm    = COMMISSION_PER_LOT

    # Cost of partial close (tp_lots out of entry_lots)
    cost = (hs * pip_val * tp_lots        # entry spread share
            + hs * pip_val * tp_lots      # exit spread
            + comm * tp_lots)             # commission share

    print(f"\n{'='*70}")
    print(f"  {tf.upper()}  —  {len(trades)} trades  ({pair})")
    print(f"  Entry: {entry_lots} lots  |  TP close: {tp_lots} lots  |  Target: ${target_usd:.0f} net")
    print(f"  Required TP: {tp_pips:.1f} pips  (covers ${cost:.2f} costs)")
    print(f"{'='*70}")

    # Hit-rate at the required TP
    hit_pct = (mfe >= tp_pips).mean() * 100
    print(f"\n  Hit-rate at {tp_pips:.1f}-pip TP:  {hit_pct:.1f}%  ({(mfe >= tp_pips).sum()} / {len(mfe)} trades)")

    # Percentile table
    print(f"\n  MFE percentile distribution:")
    print(f"  {'Pctile':>8}  {'MFE (pips)':>12}  {'Net $ (partial)':>16}  {'Hit-rate':>10}")
    print(f"  {'-'*50}")
    for pct in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99]:
        p_pips  = float(np.percentile(mfe, pct))
        net_usd = p_pips * pip_val * tp_lots - cost
        hr      = (mfe >= p_pips).mean() * 100
        marker  = " ◄ target" if abs(p_pips - tp_pips) < 0.3 else ""
        print(f"  {pct:>7}%  {p_pips:>12.1f}  {net_usd:>+15.2f}  {hr:>9.1f}%{marker}")

    # Optimal TP for common hit-rate thresholds
    print(f"\n  TP to hit in X% of trades:")
    for hr_target in [50, 60, 70, 75, 80, 85, 90, 95]:
        p_pips  = float(np.percentile(mfe, 100 - hr_target))
        net_usd = p_pips * pip_val * tp_lots - cost
        print(f"    {hr_target}% hit-rate:  {p_pips:.1f} pips  →  ${net_usd:+.2f} net per partial close")

    # MFE vs net_pips correlation — check if big MFE trades tend to win overall
    net = np.array([t["net_pips"] for t in trades])
    print(f"\n  Avg MFE:  {mfe.mean():.1f} pips  |  Median: {np.median(mfe):.1f} pips")
    print(f"  Avg net pips at exit: {net.mean():.1f}  |  % with net>0: {(net>0).mean()*100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair",       default="EURUSD")
    ap.add_argument("--start",      default="2026-01-01")
    ap.add_argument("--end",        default="2026-05-07")
    ap.add_argument("--ema-period", type=int,   default=9)
    ap.add_argument("--window",     type=int,   default=20)
    ap.add_argument("--min-swing",  type=float, default=0.0)
    ap.add_argument("--login",      type=int,   default=None)
    ap.add_argument("--password",   default=None)
    ap.add_argument("--server",     default=None)
    ap.add_argument("--mt5-path",   default=None)
    args = ap.parse_args()

    pair = args.pair

    for tf, entry_lots, tp_lots, target_usd in [
        ("1m",  2.0, 1.0, 20.0),
        ("5m",  2.0, 1.0, 20.0),
        ("1h",  1.0, 0.5, 50.0),
    ]:
        pip_val = PIP_VALUE[pair]
        hs      = HALF_SPREAD_PIPS[pair]
        comm    = COMMISSION_PER_LOT
        cost    = hs * pip_val * tp_lots * 2 + comm * tp_lots
        tp_pips = (target_usd + cost) / (pip_val * tp_lots)

        print(f"\nLoading {tf} data…")
        try:
            data = _load_data_mt5(
                [pair], args.start, args.end, timeframe=tf,
                login=args.login, password=args.password,
                server=args.server, mt5_path=args.mt5_path,
            )
        except Exception as e:
            print(f"  MT5 failed: {e}"); continue

        df = data.get(pair)
        if df is None or df.empty:
            print("  No data."); continue

        trades = simulate_with_mfe(
            pair, df,
            ema_period=args.ema_period,
            window=args.window,
            min_swing_pips=args.min_swing,
        )

        mfe_report(trades, tf, tp_pips, tp_lots, entry_lots, target_usd, pair)


if __name__ == "__main__":
    main()
