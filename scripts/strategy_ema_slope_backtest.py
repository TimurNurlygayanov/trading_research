"""
EMA-slope trend-following strategy backtest.

Entry logic
-----------
  BUY  : EMA slope > +min_slope × ATR  AND  close > EMA
  SELL : EMA slope < -min_slope × ATR  AND  close < EMA

  EMA source: "close" (standard) or "hl2" ((H+L)/2 — smoother, less whipsaw)

SL / TP
-------
  SL  = EMA value at entry time (price must return to EMA to stop us out)
  TP  = entry ± rr × |entry − EMA|

One trade per pair at a time; no new entry while in a trade.

Usage
-----
  # MT5 5m, Jan-May 2026, IS=Jan-Mar, OOS=Apr-May
  python -m scripts.strategy_ema_slope_backtest \\
      --source mt5 --timeframe 5m \\
      --start 2026-01-01 --end 2026-05-06 \\
      --oos-start 2026-04-01 \\
      --login 1513313327 --server FTMO-Demo

  # Quick single-config check
  python -m scripts.strategy_ema_slope_backtest \\
      --source mt5 --timeframe 5m \\
      --start 2026-01-01 --end 2026-05-06 --oos-start 2026-04-01 \\
      --login 1513313327 --server FTMO-Demo \\
      --ema-period 9 --rr 2.0 --min-slope 0.05
"""
from __future__ import annotations

import argparse
import logging
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

# Reuse MT5 loader and cost model from regime backtest
from scripts.strategy1_regime_backtest import (
    _load_data_mt5, _MT5_TF_MAP, _TF_MINUTES,
    _atr, _ema,
    PAIRS, HALF_SPREAD_PIPS, PIP_VALUE, PIP_SIZE, COMMISSION_PER_LOT,
)
from backtest.data_fetcher import fetch_ohlcv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ema_slope")


# ── Strategy simulation ───────────────────────────────────────────────────────

def simulate_ema_slope(
    pair: str,
    df: pd.DataFrame,
    ema_period: int,
    atr_period: int,
    rr: float,
    min_slope_atr: float,   # EMA slope must exceed this × ATR to enter
    min_dist_atr: float,    # |close − EMA| must exceed this × ATR to enter
    lots: float,
    ema_source: str,        # "close" or "hl2"
    sl_atr_mult: float | None = None,  # when set: SL = sl_atr_mult×ATR, TP = rr×sl_atr_mult×ATR
) -> list[dict]:
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    times  = df.index

    src    = (highs + lows) / 2 if ema_source == "hl2" else closes
    ema_v  = _ema(src, ema_period)
    atr_v  = _atr(df, atr_period)

    half_sp_val = HALF_SPREAD_PIPS[pair] * PIP_SIZE
    entry_cost  = HALF_SPREAD_PIPS[pair] * PIP_VALUE[pair] * lots   # spread at entry
    sl_cost     = HALF_SPREAD_PIPS[pair] * PIP_VALUE[pair] * lots   # spread at SL exit
    commission  = COMMISSION_PER_LOT * lots

    warmup   = max(ema_period, atr_period) + 2
    in_trade = None
    trades: list[dict] = []

    for i in range(warmup, len(df)):
        bar_open  = df["Open"].values[i]
        bar_high  = highs[i]
        bar_low   = lows[i]
        bar_close = closes[i]
        bar_time  = times[i]
        ema_cur   = ema_v[i]
        atr_cur   = atr_v[i]

        # ── Exit ──────────────────────────────────────────────────────────────
        if in_trade is not None:
            tp  = in_trade["tp"]
            sl  = in_trade["sl"]
            dir = in_trade["direction"]

            exit_price = None
            exit_type  = None

            if dir == 1:    # BUY
                if bar_open <= sl:
                    exit_price, exit_type = bar_open, "sl_gap"
                elif bar_low <= sl:
                    exit_price, exit_type = sl, "sl"
                elif bar_high >= tp:
                    exit_price, exit_type = tp, "tp"
            else:           # SELL
                if bar_open >= sl:
                    exit_price, exit_type = bar_open, "sl_gap"
                elif bar_high >= sl:
                    exit_price, exit_type = sl, "sl"
                elif bar_low <= tp:
                    exit_price, exit_type = tp, "tp"

            if exit_price is not None:
                pip_move = (exit_price - in_trade["entry"]) * dir / PIP_SIZE
                gross    = pip_move * PIP_VALUE[pair] * lots
                exit_sl_cost = sl_cost if "sl" in exit_type else 0.0
                net  = gross - entry_cost - exit_sl_cost - commission
                won  = net > 0
                trades.append({
                    "time":       in_trade["time"],
                    "close_time": bar_time,
                    "pair":       pair,
                    "direction":  dir,
                    "entry":      in_trade["entry"],
                    "tp":         tp,
                    "sl":         sl,
                    "exit":       exit_price,
                    "exit_type":  exit_type,
                    "gross":      gross,
                    "net":        net,
                    "won":        won,
                    "sl_dist_pips": abs(in_trade["entry"] - sl) / PIP_SIZE,
                })
                in_trade = None

        # ── Entry ─────────────────────────────────────────────────────────────
        if in_trade is not None or atr_cur <= 0:
            continue

        slope    = ema_v[i] - ema_v[i - 1]
        dist     = bar_close - ema_cur          # +ve → close above EMA

        # Slope filter
        if abs(slope) < min_slope_atr * atr_cur:
            continue

        # Direction: buy if EMA rising + close above EMA; sell if falling + below
        if slope > 0 and dist > min_dist_atr * atr_cur:
            direction = 1
        elif slope < 0 and dist < -min_dist_atr * atr_cur:
            direction = -1
        else:
            continue

        entry = bar_close + half_sp_val * direction

        if sl_atr_mult is not None:
            # ATR-based fixed SL/TP
            sl = entry - direction * sl_atr_mult * atr_cur
            tp = entry + direction * rr * sl_atr_mult * atr_cur
        else:
            # EMA-distance SL/TP
            sl_dist = abs(dist)
            if sl_dist < 0.1 * PIP_SIZE:        # skip if entry == EMA (zero SL)
                continue
            sl = ema_cur
            tp = entry + direction * rr * sl_dist

        in_trade = {
            "time":      bar_time,
            "entry":     entry,
            "sl":        sl,
            "tp":        tp,
            "direction": direction,
        }

    return trades


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(trades: list[dict], label: str, period: str) -> dict:
    if not trades:
        return {"label": label, "period": period, "n": 0, "pnl": 0.0,
                "win_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "sharpe": 0.0, "max_dd": 0.0, "ev": 0.0,
                "avg_sl_pips": 0.0}
    nets = np.array([t["net"] for t in trades])
    won  = np.array([t["won"] for t in trades], dtype=bool)
    n    = len(trades)
    pnl  = nets.sum()
    wr   = won.mean()
    aw   = nets[won].mean()  if won.any()   else 0.0
    al   = nets[~won].mean() if (~won).any() else 0.0
    if n > 1 and nets.std() > 0:
        days = max(1, (trades[-1]["time"] - trades[0]["time"]).days)
        sharpe = nets.mean() / nets.std() * np.sqrt(n / (days / 365.25))
    else:
        sharpe = 0.0
    cum    = np.cumsum(nets)
    max_dd = (cum - np.maximum.accumulate(cum)).min()
    avg_sl = np.mean([t["sl_dist_pips"] for t in trades])
    return {
        "label": label, "period": period, "n": n,
        "pnl": round(pnl, 1), "win_pct": round(wr * 100, 1),
        "avg_win": round(aw, 2), "avg_loss": round(al, 2),
        "sharpe": round(sharpe, 2), "max_dd": round(max_dd, 1),
        "ev": round(pnl / n, 2), "avg_sl_pips": round(avg_sl, 1),
    }


def run_config(
    label: str,
    pairs_data: dict,
    oos_start: pd.Timestamp,
    ema_period: int,
    atr_period: int,
    rr: float,
    min_slope_atr: float,
    min_dist_atr: float,
    lots: float,
    ema_source: str,
    sl_atr_mult: float | None = None,
) -> tuple[dict, dict]:
    all_is, all_oos = [], []
    for pair, df in pairs_data.items():
        trades = simulate_ema_slope(
            pair, df, ema_period, atr_period, rr,
            min_slope_atr, min_dist_atr, lots, ema_source, sl_atr_mult,
        )
        for t in trades:
            (all_is if t["time"] < oos_start else all_oos).append(t)
    return metrics(all_is, label, "IS"), metrics(all_oos, label, "OOS")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--start",        default="2026-01-01")
    ap.add_argument("--end",          default="2026-05-06")
    ap.add_argument("--oos-start",    default="2026-04-01")
    ap.add_argument("--timeframe",    default="5m", choices=list(_MT5_TF_MAP.keys()))
    ap.add_argument("--lots",         type=float, default=1.0)
    ap.add_argument("--atr-period",   type=int,   default=14)
    # Single-config overrides (skip sweep if all three provided)
    ap.add_argument("--ema-period",   type=int,   default=None)
    ap.add_argument("--rr",           type=float, default=None)
    ap.add_argument("--min-slope",    type=float, default=None,
                    help="Min EMA slope as fraction of ATR (0 = any slope)")
    ap.add_argument("--sl-atr",       type=float, default=None,
                    help="SL = sl-atr × ATR; TP = rr × sl-atr × ATR. "
                         "When omitted, SL is placed at EMA level.")
    ap.add_argument("--ema-source",   default="close", choices=["close", "hl2"])
    ap.add_argument("--pairs",        nargs="+", default=None)
    # Data source
    ap.add_argument("--source",       default="mt5", choices=["api", "mt5"])
    ap.add_argument("--mt5-path",     default=None)
    ap.add_argument("--login",        type=int, default=None)
    ap.add_argument("--password",     default=None)
    ap.add_argument("--server",       default=None)
    ap.add_argument("--symbol-suffix", default="")
    args = ap.parse_args()

    pairs = args.pairs or PAIRS
    oos_start = pd.Timestamp(args.oos_start, tz="UTC")

    # ── Load data ────────────────────────────────────────────────────────────
    log.info("Loading %s %s data for %s  (%s → %s)",
             args.timeframe, args.source, pairs, args.start, args.end)
    pairs_data: dict[str, pd.DataFrame] = {}

    if args.source == "mt5":
        try:
            pairs_data = _load_data_mt5(
                pairs, args.start, args.end,
                timeframe=args.timeframe,
                mt5_path=args.mt5_path, login=args.login,
                password=args.password, server=args.server,
                symbol_suffix=args.symbol_suffix,
            )
        except Exception as exc:
            log.error("MT5 load failed: %s", exc)
    else:
        for pair in pairs:
            try:
                df = fetch_ohlcv(pair, args.timeframe, args.start, args.end)
                pairs_data[pair] = df
                log.info("  %s: %d bars", pair, len(df))
            except Exception as exc:
                log.error("  %s failed: %s", pair, exc)

    if not pairs_data:
        log.error("No data loaded.")
        return

    # ── Parameter sweep ───────────────────────────────────────────────────────
    sl_atr = args.sl_atr                          # None → EMA-distance mode

    # Any arg provided → pin that axis; otherwise use defaults
    ema_periods  = [args.ema_period] if args.ema_period is not None else [9, 20]
    rr_vals      = [args.rr]        if args.rr          is not None else (
                       [1.0] if sl_atr is not None else [2.0, 3.0])
    slope_vals   = [args.min_slope] if args.min_slope   is not None else [0.0, 0.05, 0.15]
    src_vals     = [args.ema_source]

    configs = [
        (ema_p, rr, slope, src)
        for ema_p  in ema_periods
        for rr     in rr_vals
        for slope  in slope_vals
        for src    in src_vals
    ]

    sl_tag = f"atr{sl_atr}" if sl_atr is not None else "ema"
    results_is, results_oos = [], []
    for ema_p, rr, slope, src in configs:
        lbl = f"ema{ema_p}_{src}_rr{rr}_{sl_tag}_slope{slope}"
        log.info("Running: %s", lbl)
        m_is, m_oos = run_config(
            lbl, pairs_data, oos_start,
            ema_period=ema_p, atr_period=args.atr_period,
            rr=rr, min_slope_atr=slope, min_dist_atr=0.0,
            lots=args.lots, ema_source=src, sl_atr_mult=sl_atr,
        )
        results_is.append(m_is)
        results_oos.append(m_oos)

    # ── Output ────────────────────────────────────────────────────────────────
    cols = ["label", "n", "pnl", "win_pct", "avg_win", "avg_loss",
            "ev", "max_dd", "sharpe", "avg_sl_pips"]

    df_is  = pd.DataFrame(results_is)[cols].sort_values("pnl", ascending=False)
    df_oos = pd.DataFrame(results_oos)[cols].sort_values("pnl", ascending=False)

    def _fmt(df, title):
        print(f"\n{'='*100}")
        print(f"  {title}  (timeframe={args.timeframe}, lots={args.lots})")
        print(f"{'='*100}")
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    _fmt(df_is,  f"IN-SAMPLE  ({args.start} → {args.oos_start})")
    _fmt(df_oos, f"OUT-OF-SAMPLE  ({args.oos_start} → {args.end})")

    if len(df_oos) >= 3:
        print(f"\n{'='*100}")
        print("  TOP-3 OOS — IS vs OOS")
        print(f"{'='*100}")
        for lbl in df_oos.head(3)["label"]:
            ir = df_is [df_is ["label"] == lbl].iloc[0]
            or_ = df_oos[df_oos["label"] == lbl].iloc[0]
            print(f"  {lbl}")
            print(f"    IS : n={ir['n']:5d}  pnl=${ir['pnl']:8,.1f}  "
                  f"win={ir['win_pct']:.1f}%  sharpe={ir['sharpe']:.2f}  "
                  f"maxDD=${ir['max_dd']:,.1f}")
            print(f"    OOS: n={or_['n']:5d}  pnl=${or_['pnl']:8,.1f}  "
                  f"win={or_['win_pct']:.1f}%  sharpe={or_['sharpe']:.2f}  "
                  f"maxDD=${or_['max_dd']:,.1f}")


if __name__ == "__main__":
    main()
