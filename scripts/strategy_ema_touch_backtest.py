"""
EMA-touch / EMA-cross strategy backtest.

BUY signal  (EMA bounce):
  - Green candle (close > open)
  - Low < EMA  (candle dipped below EMA — tested it as support)
  - Close > EMA and High > EMA  (bounced back above)
  → Entry: bar close (+half spread)
  → SL:    bar low
  → TP:    entry + rr × (entry − SL)

SELL signal  (EMA break):
  - Red candle (close < open)
  - Open > EMA  (started above EMA)
  - Close < EMA  (broke below — resistance held)
  → Entry: bar close (−half spread)
  → SL:    bar high
  → TP:    entry − rr × (SL − entry)

Sweep: EMA periods [9, 20, 50], rr [1.5], min_sl_pips [0.5, 1.0, 2.0]

Usage
-----
  python -m scripts.strategy_ema_touch_backtest \\
      --source mt5 --timeframe 5m \\
      --start 2026-01-01 --end 2026-05-07 \\
      --oos-start 2026-04-01 \\
      --login 1513313327 --server FTMO-Demo
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

from scripts.strategy1_regime_backtest import (
    _load_data_mt5, _MT5_TF_MAP,
    _atr, _ema,
    PAIRS, HALF_SPREAD_PIPS, PIP_VALUE, PIP_SIZE, COMMISSION_PER_LOT,
)
from backtest.data_fetcher import fetch_ohlcv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ema_touch")


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_ema_touch(
    pair:        str,
    df:          pd.DataFrame,
    ema_period:  int,
    atr_period:  int,
    rr:          float,
    min_sl_pips: float,      # minimum SL distance (candle_range / pip_size)
    lots:        float,
) -> list[dict]:

    opens  = df["Open"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    times  = df.index

    ema_v = _ema(closes, ema_period)
    atr_v = _atr(df, atr_period)

    half_sp   = HALF_SPREAD_PIPS[pair] * PIP_SIZE
    pip_val   = PIP_VALUE[pair]
    comm      = COMMISSION_PER_LOT * lots
    spread_cost = HALF_SPREAD_PIPS[pair] * pip_val * lots   # one-way
    min_sl_price = min_sl_pips * PIP_SIZE

    warmup   = max(ema_period, atr_period) + 1
    in_trade = None
    trades: list[dict] = []

    for i in range(warmup, len(df)):
        bar_open  = opens[i]
        bar_high  = highs[i]
        bar_low   = lows[i]
        bar_close = closes[i]
        bar_time  = times[i]
        ema_cur   = ema_v[i]

        # ── Exit ──────────────────────────────────────────────────────────────
        if in_trade is not None:
            tp  = in_trade["tp"]
            sl  = in_trade["sl"]
            dir = in_trade["direction"]

            exit_price = exit_type = None

            if dir == 1:       # LONG
                if bar_open <= sl:
                    exit_price, exit_type = bar_open, "sl_gap"
                elif bar_low <= sl:
                    exit_price, exit_type = sl, "sl"
                elif bar_high >= tp:
                    exit_price, exit_type = tp, "tp"
            else:              # SHORT
                if bar_open >= sl:
                    exit_price, exit_type = bar_open, "sl_gap"
                elif bar_high >= sl:
                    exit_price, exit_type = sl, "sl"
                elif bar_low <= tp:
                    exit_price, exit_type = tp, "tp"

            if exit_price is not None:
                pip_move = (exit_price - in_trade["entry"]) * dir / PIP_SIZE
                gross    = pip_move * pip_val * lots
                sl_cost  = spread_cost if "sl" in exit_type else 0.0
                net      = gross - in_trade["entry_cost"] - sl_cost - comm
                trades.append({
                    "time":       in_trade["time"],
                    "close_time": bar_time,
                    "pair":       pair,
                    "direction":  dir,
                    "entry":      in_trade["entry"],
                    "sl":         sl,
                    "tp":         tp,
                    "exit":       exit_price,
                    "exit_type":  exit_type,
                    "net":        net,
                    "won":        net > 0,
                    "sl_pips":    abs(in_trade["entry"] - sl) / PIP_SIZE,
                    "atr":        atr_v[i],
                    "ema":        ema_cur,
                })
                in_trade = None

        # ── Entry ─────────────────────────────────────────────────────────────
        if in_trade is not None:
            continue

        green = bar_close > bar_open
        red   = bar_close < bar_open

        if green and bar_low < ema_cur and bar_close > ema_cur:
            # BUY: green candle that dipped below EMA and closed above
            direction = 1
            entry     = bar_close + half_sp          # buy at ask
            sl        = bar_low
            sl_dist   = entry - sl
            if sl_dist < min_sl_price:
                continue
            tp = entry + rr * sl_dist
            in_trade = {
                "time": bar_time, "entry": entry,
                "sl": sl, "tp": tp, "direction": 1,
                "entry_cost": spread_cost,
            }

        elif red and bar_open > ema_cur and bar_close < ema_cur:
            # SELL: red candle that opened above EMA and closed below
            direction = -1
            entry     = bar_close - half_sp          # sell at bid
            sl        = bar_high
            sl_dist   = sl - entry
            if sl_dist < min_sl_price:
                continue
            tp = entry - rr * sl_dist
            in_trade = {
                "time": bar_time, "entry": entry,
                "sl": sl, "tp": tp, "direction": -1,
                "entry_cost": spread_cost,
            }

    return trades


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(trades: list[dict], label: str, period: str) -> dict:
    base = {"label": label, "period": period, "n": 0, "pnl": 0.0,
            "win_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "sharpe": 0.0, "max_dd": 0.0, "ev": 0.0, "avg_sl_pips": 0.0}
    if not trades:
        return base
    nets = np.array([t["net"] for t in trades])
    won  = np.array([t["won"] for t in trades], dtype=bool)
    n    = len(trades)
    pnl  = nets.sum()
    aw   = nets[won].mean()  if won.any()   else 0.0
    al   = nets[~won].mean() if (~won).any() else 0.0
    if n > 1 and nets.std() > 0:
        days   = max(1, (trades[-1]["time"] - trades[0]["time"]).days)
        sharpe = nets.mean() / nets.std() * np.sqrt(n / (days / 365.25))
    else:
        sharpe = 0.0
    cum    = np.cumsum(nets)
    max_dd = (cum - np.maximum.accumulate(cum)).min()
    avg_sl = np.mean([t["sl_pips"] for t in trades])
    return {
        "label": label, "period": period, "n": n,
        "pnl": round(pnl, 1), "win_pct": round(won.mean() * 100, 1),
        "avg_win": round(aw, 2), "avg_loss": round(al, 2),
        "sharpe": round(sharpe, 2), "max_dd": round(max_dd, 1),
        "ev": round(pnl / n, 2), "avg_sl_pips": round(avg_sl, 2),
    }


def run_config(
    label: str, pairs_data: dict, oos_start: pd.Timestamp,
    ema_period: int, atr_period: int, rr: float,
    min_sl_pips: float, lots: float,
) -> tuple[dict, dict]:
    all_is, all_oos = [], []
    for pair, df in pairs_data.items():
        for t in simulate_ema_touch(pair, df, ema_period, atr_period,
                                    rr, min_sl_pips, lots):
            (all_is if t["time"] < oos_start else all_oos).append(t)
    return metrics(all_is, label, "IS"), metrics(all_oos, label, "OOS")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--start",         default="2026-01-01")
    ap.add_argument("--end",           default="2026-05-07")
    ap.add_argument("--oos-start",     default="2026-04-01")
    ap.add_argument("--timeframe",     default="5m", choices=list(_MT5_TF_MAP.keys()))
    ap.add_argument("--atr-period",    type=int,   default=14)
    ap.add_argument("--rr",            type=float, default=1.5)
    ap.add_argument("--lots",          type=float, default=1.0)
    ap.add_argument("--pairs",         nargs="+",  default=None)
    ap.add_argument("--source",        default="mt5", choices=["api", "mt5"])
    ap.add_argument("--mt5-path",      default=None)
    ap.add_argument("--login",         type=int,   default=None)
    ap.add_argument("--password",      default=None)
    ap.add_argument("--server",        default=None)
    ap.add_argument("--symbol-suffix", default="")
    args = ap.parse_args()

    pairs     = args.pairs or PAIRS
    oos_start = pd.Timestamp(args.oos_start, tz="UTC")

    log.info("Loading %s %s data  (%s → %s)", args.timeframe, args.source,
             args.start, args.end)
    pairs_data: dict[str, pd.DataFrame] = {}
    if args.source == "mt5":
        try:
            pairs_data = _load_data_mt5(
                pairs, args.start, args.end, timeframe=args.timeframe,
                mt5_path=args.mt5_path, login=args.login,
                password=args.password, server=args.server,
                symbol_suffix=args.symbol_suffix,
            )
        except Exception as exc:
            log.error("MT5 load failed: %s", exc); return
    else:
        for pair in pairs:
            try:
                df = fetch_ohlcv(pair, args.timeframe, args.start, args.end)
                pairs_data[pair] = df
                log.info("  %s: %d bars", pair, len(df))
            except Exception as exc:
                log.error("  %s: %s", pair, exc)

    if not pairs_data:
        log.error("No data."); return

    results_is, results_oos = [], []
    for ema_p in [9, 20, 50]:
        for min_sl in [0.5, 1.0, 2.0]:
            lbl = f"ema{ema_p}_rr{args.rr}_minsl{min_sl}"
            log.info("Running: %s", lbl)
            m_is, m_oos = run_config(
                lbl, pairs_data, oos_start,
                ema_period=ema_p, atr_period=args.atr_period,
                rr=args.rr, min_sl_pips=min_sl, lots=args.lots,
            )
            results_is.append(m_is)
            results_oos.append(m_oos)

    cols = ["label", "n", "pnl", "win_pct", "avg_win", "avg_loss",
            "ev", "max_dd", "sharpe", "avg_sl_pips"]

    df_is  = pd.DataFrame(results_is)[cols].sort_values("pnl", ascending=False)
    df_oos = pd.DataFrame(results_oos)[cols].sort_values("pnl", ascending=False)

    def _fmt(df, title):
        print(f"\n{'='*95}")
        print(f"  {title}  (tf={args.timeframe}, rr={args.rr}, lots={args.lots})")
        print(f"{'='*95}")
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    _fmt(df_is,  f"IN-SAMPLE  ({args.start} → {args.oos_start})")
    _fmt(df_oos, f"OUT-OF-SAMPLE  ({args.oos_start} → {args.end})")

    print(f"\n{'='*95}")
    print("  TOP-3 OOS vs IS")
    print(f"{'='*95}")
    for lbl in df_oos.head(3)["label"]:
        ir  = df_is [df_is ["label"] == lbl].iloc[0]
        or_ = df_oos[df_oos["label"] == lbl].iloc[0]
        print(f"\n  {lbl}")
        print(f"    IS  n={ir['n']:4d}  pnl=${ir['pnl']:8,.1f}  win={ir['win_pct']:.1f}%  "
              f"avg_sl={ir['avg_sl_pips']:.2f}p  sharpe={ir['sharpe']:.2f}  maxDD=${ir['max_dd']:,.1f}")
        print(f"    OOS n={or_['n']:4d}  pnl=${or_['pnl']:8,.1f}  win={or_['win_pct']:.1f}%  "
              f"avg_sl={or_['avg_sl_pips']:.2f}p  sharpe={or_['sharpe']:.2f}  maxDD=${or_['max_dd']:,.1f}")


if __name__ == "__main__":
    main()
