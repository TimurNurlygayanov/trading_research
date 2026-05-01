"""
Big red candle reversal — EUR/USD 1m.

Entry  : red candle whose body (open - close) > entry_atr_mult * ATR(14)
         optional: EMA200 must be rising (trend filter)
Exit   : TP when price reaches tp_atr_mult * ATR above entry
         SL when price drops sl_atr_mult * ATR below entry
Size   : 1 lot (100,000 units); 1 pip = $10

One trade at a time.  New signals while in a trade are ignored.
Entry is taken at the close of the signal candle.

Usage
  python -m scripts.ema_pullback_patient
  python -m scripts.ema_pullback_patient --trend-ema 200 --sweep
  python -m scripts.ema_pullback_patient --sl-atr 2 --tp-atr 3 --trend-ema 200
  python -m scripts.ema_pullback_patient --start 2023-01-01 --end 2025-01-01
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv

LOT_SIZE = 100_000  # EUR/USD 1 lot


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(close: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(close).ewm(span=span, adjust=False).mean().values


def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n     = len(close)
    tr    = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    atr    = np.empty(n)
    atr[0] = tr[0]
    alpha  = 1.0 / period
    for i in range(1, n):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate(close: np.ndarray,
             open_: np.ndarray,
             low: np.ndarray,
             high: np.ndarray,
             atr_vals: np.ndarray,
             trend_ema: np.ndarray | None,
             entry_atr_mult: float,
             tp_atr_mult: float,
             sl_atr_mult: float,
             commission_usd: float,
             trend_slope_bars: int,
             warmup: int) -> dict:

    n = len(close)

    n_tp = n_sl = n_open = 0
    tp_pnl_total = sl_pnl_total = open_pnl_total = 0.0
    bars_tp: list[int] = []
    worst_dd_list: list[float] = []

    in_trade    = False
    entry_price = 0.0
    tp_price    = 0.0
    sl_price    = 0.0
    entry_bar   = 0
    worst_low   = 0.0

    for i in range(warmup, n):
        if in_trade:
            worst_low = min(worst_low,
                            (close[i] - entry_price) * LOT_SIZE - commission_usd)

            sl_hit = low[i]  <= sl_price
            tp_hit = high[i] >= tp_price
            if sl_hit and tp_hit:
                sl_hit, tp_hit = True, False  # conservative: SL wins

            if sl_hit:
                n_sl += 1
                sl_pnl_total += (sl_price - entry_price) * LOT_SIZE - commission_usd
                worst_dd_list.append(worst_low)
                in_trade = False
            elif tp_hit:
                n_tp += 1
                tp_pnl_total += (tp_price - entry_price) * LOT_SIZE - commission_usd
                bars_tp.append(i - entry_bar)
                worst_dd_list.append(worst_low)
                in_trade = False

        else:
            if trend_ema is not None and trend_ema[i] <= trend_ema[i - trend_slope_bars]:
                continue

            # Entry: red candle with body > entry_atr_mult * ATR
            body = open_[i] - close[i]
            if body > entry_atr_mult * atr_vals[i]:
                in_trade    = True
                entry_price = close[i]
                entry_bar   = i
                worst_low   = 0.0
                atr_e       = atr_vals[i]
                tp_price    = close[i] + tp_atr_mult * atr_e
                sl_price    = close[i] - sl_atr_mult * atr_e

    if in_trade:
        pnl = (close[n - 1] - entry_price) * LOT_SIZE - commission_usd
        n_open += 1
        open_pnl_total += pnl
        worst_dd_list.append(min(worst_low, pnl))

    total = n_tp + n_sl + n_open
    if total == 0:
        return {"total": 0}

    win_rate    = n_tp / (n_tp + n_sl) if (n_tp + n_sl) > 0 else 0.0
    net_pnl     = tp_pnl_total + sl_pnl_total + open_pnl_total
    avg_tp_bars = float(np.mean(bars_tp)) if bars_tp else 0.0
    avg_dd      = float(np.mean(worst_dd_list)) if worst_dd_list else 0.0
    worst_dd    = float(np.min(worst_dd_list)) if worst_dd_list else 0.0

    return {
        "total": total, "n_tp": n_tp, "n_sl": n_sl, "n_open": n_open,
        "win_rate": win_rate, "net_pnl": net_pnl,
        "tp_pnl": tp_pnl_total, "sl_pnl": sl_pnl_total,
        "avg_tp_pnl": tp_pnl_total / n_tp if n_tp else 0.0,
        "avg_sl_pnl": sl_pnl_total / n_sl if n_sl else 0.0,
        "avg_tp_bars": avg_tp_bars,
        "avg_dd": avg_dd, "worst_dd": worst_dd,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prepare(df: pd.DataFrame,
             atr_period: int,
             trend_ema_period: int | None,
             trend_slope_bars: int) -> tuple:
    close     = df["Close"].values.astype(float)
    open_     = df["Open"].values.astype(float)
    low       = df["Low"].values.astype(float)
    high      = df["High"].values.astype(float)
    atr_vals  = _atr(df, atr_period)
    trend_ema = _ema(close, trend_ema_period) if trend_ema_period else None
    warmup    = max(atr_period,
                    trend_ema_period if trend_ema_period else 0) + trend_slope_bars + 1
    return close, open_, low, high, atr_vals, trend_ema, warmup


def _print_single(r: dict, args: argparse.Namespace, df: pd.DataFrame) -> None:
    if r["total"] == 0:
        print("No trades triggered."); return

    trend_str = (f"EMA{args.trend_ema} rising/{args.trend_slope_bars}b"
                 if args.trend_ema else "none")
    print(f"\n{'='*64}")
    print(f"  Big red candle reversal  |  EUR/USD 1m")
    print(f"  Entry : body (O-C) > {args.atr_mult}×ATR{args.atr_period}  [red candle]")
    print(f"  TP    : entry + {args.tp_atr}×ATR  |  SL: entry - {args.sl_atr}×ATR")
    print(f"  Comm  : ${args.commission:.1f}  |  Trend: {trend_str}")
    print(f"  Data  : {len(df):,} bars  ({df.index[0]} → {df.index[-1]})")
    print(f"{'='*64}")
    print(f"  Total trades  : {r['total']}")
    print(f"  TP hits       : {r['n_tp']}  ({r['win_rate']*100:.1f}%)")
    print(f"  SL hits       : {r['n_sl']}  ({r['n_sl']/r['total']*100:.1f}%)")
    print(f"  Still open    : {r['n_open']}")
    if r["n_tp"]:
        print(f"\n  TP avg bars   : {r['avg_tp_bars']:.1f}"
              f"   TP avg PnL : ${r['avg_tp_pnl']:+.2f}")
    if r["n_sl"]:
        print(f"  SL avg PnL    : ${r['avg_sl_pnl']:+.2f}")
    print(f"\n  Worst unrealized loss (single trade) : ${r['worst_dd']:+.2f}")
    print(f"  Avg worst unrealized loss            : ${r['avg_dd']:+.2f}")
    print(f"\n  TP PnL  : ${r['tp_pnl']:+.2f}")
    if r["n_sl"]:
        print(f"  SL PnL  : ${r['sl_pnl']:+.2f}")
    print(f"  Net PnL : ${r['net_pnl']:+.2f}")


def _print_grid(results: dict, mults: list[float], args: argparse.Namespace) -> None:
    trend_str = (f"EMA{args.trend_ema} rising/{args.trend_slope_bars}b"
                 if args.trend_ema else "no trend filter")
    header = (f"\n  Entry: body > {args.atr_mult}×ATR  |  "
              f"Trend: {trend_str}")
    col_w = 10
    sl_labels = [f"SL {m}×" for m in mults]

    def _row(label: str, vals: list[str]) -> str:
        return f"  {label:<12}" + "".join(f"{v:>{col_w}}" for v in vals)

    for metric, title, fmt in [
        ("net_pnl",     "Net PnL ($)",    lambda v: f"${v:+.0f}"),
        ("win_rate",    "Win rate (%)",   lambda v: f"{v*100:.1f}%"),
        ("total",       "# Trades",       lambda v: str(v)),
        ("avg_tp_bars", "Avg TP bars",    lambda v: f"{v:.1f}"),
    ]:
        print(header)
        print(f"  [{title}]")
        print(_row("", sl_labels))
        print(f"  {'-' * (12 + col_w * len(mults))}")
        for tp in mults:
            row_vals = []
            for sl in mults:
                r = results.get((tp, sl), {})
                row_vals.append(fmt(r[metric]) if r and r["total"] > 0 else "n/a")
            print(_row(f"TP {tp}×", row_vals))


# ── Entry points ──────────────────────────────────────────────────────────────

def run_single(df: pd.DataFrame, args: argparse.Namespace) -> None:
    close, open_, low, high, atr_vals, trend_ema, warmup = _prepare(
        df, args.atr_period, args.trend_ema, args.trend_slope_bars)
    r = simulate(close, open_, low, high, atr_vals, trend_ema,
                 args.atr_mult, args.tp_atr, args.sl_atr,
                 args.commission, args.trend_slope_bars, warmup)
    _print_single(r, args, df)


def run_sweep(df: pd.DataFrame, args: argparse.Namespace) -> None:
    mults = [1.0, 2.0, 3.0, 5.0, 10.0]
    close, open_, low, high, atr_vals, trend_ema, warmup = _prepare(
        df, args.atr_period, args.trend_ema, args.trend_slope_bars)
    results: dict[tuple, dict] = {}
    for tp in mults:
        for sl in mults:
            results[(tp, sl)] = simulate(
                close, open_, low, high, atr_vals, trend_ema,
                args.atr_mult, tp, sl,
                args.commission, args.trend_slope_bars, warmup)
    _print_grid(results, mults, args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start",            default="2024-01-01")
    ap.add_argument("--end",              default="2025-01-01")
    ap.add_argument("--atr-period",       type=int,   default=14)
    ap.add_argument("--atr-mult",         type=float, default=2.0,
                    help="Body must exceed this many ATRs to trigger entry (default 2)")
    ap.add_argument("--commission",       type=float, default=7.0,
                    help="Round-trip commission in USD (default 7 = 0.7 pip)")
    ap.add_argument("--tp-atr",           type=float, default=3.0,
                    help="TP in ATR multiples above entry (default 3)")
    ap.add_argument("--sl-atr",           type=float, default=2.0,
                    help="SL in ATR multiples below entry (default 2)")
    ap.add_argument("--trend-ema",        type=int,   default=None,
                    help="Trend filter EMA period (e.g. 200); only enter when rising")
    ap.add_argument("--trend-slope-bars", type=int,   default=5,
                    help="Bars back for trend EMA slope check (default 5)")
    ap.add_argument("--sweep",            action="store_true",
                    help="Sweep SL/TP grid [1,2,3,5,10]×ATR and print tables")
    args = ap.parse_args()

    print(f"Fetching EURUSD 1m  {args.start} → {args.end} ...")
    df = fetch_ohlcv("EURUSD", "1m", args.start, args.end)
    if df.empty:
        print("No data returned."); return
    print(f"  {len(df):,} bars")

    if args.sweep:
        run_sweep(df, args)
    else:
        run_single(df, args)


if __name__ == "__main__":
    main()
