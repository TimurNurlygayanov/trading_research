"""
SuperTrend breakout + trailing-stop strategy  (LONG only).

Signal:
  - SuperTrend is RED (bearish, upper band above price)
  - High of current bar OR previous bar >= ST upper band  (price tested resistance)
  - Close still below ST upper band                       (closed back below)
  → State: PENDING

Entry:
  When ST flips GREEN (close > upper band) on any subsequent bar:
    fill at the upper band level that was just crossed.

Trailing stop:
  SL = ST lower band, updated each bar (moves up only, never down).
  Exit when bar_low ≤ SL or bar_open < SL (gap through stop).

Notes:
  - No fixed TP — ride the trend until stop is hit.
  - Pending cancelled after max_pending bars if ST hasn't flipped.

Sweep: ST periods [7, 10, 14] × multipliers [2.0, 3.0, 4.0]

Usage
-----
  python -m scripts.strategy_st_trail_backtest \\
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
    _atr,
    PAIRS, HALF_SPREAD_PIPS, PIP_VALUE, PIP_SIZE, COMMISSION_PER_LOT,
)
from backtest.data_fetcher import fetch_ohlcv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("st_trail")


# ── SuperTrend ────────────────────────────────────────────────────────────────

def _supertrend(df: pd.DataFrame, period: int, mult: float):
    """
    Returns (st_val, st_dir, final_upper, final_lower).
    st_dir: 1=bullish (green), -1=bearish (red).
    st_val: the active band (lower when green, upper when red).
    """
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    n = len(closes)

    atr_v = _atr(df, period)
    hl2   = (highs + lows) / 2.0

    raw_upper = hl2 + mult * atr_v
    raw_lower = hl2 - mult * atr_v

    final_upper = raw_upper.copy()
    final_lower = raw_lower.copy()
    st_val = np.zeros(n)
    st_dir = np.zeros(n, dtype=int)

    st_dir[0] = -1
    st_val[0] = raw_upper[0]

    for i in range(1, n):
        # Upper band only steps down, or resets when price crossed above it
        if raw_upper[i] < final_upper[i - 1] or closes[i - 1] > final_upper[i - 1]:
            final_upper[i] = raw_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Lower band only steps up, or resets when price crossed below it
        if raw_lower[i] > final_lower[i - 1] or closes[i - 1] < final_lower[i - 1]:
            final_lower[i] = raw_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction: flips when close crosses the relevant final band
        if st_dir[i - 1] == -1:
            st_dir[i] = 1 if closes[i] > final_upper[i] else -1
        else:
            st_dir[i] = -1 if closes[i] < final_lower[i] else 1

        st_val[i] = final_lower[i] if st_dir[i] == 1 else final_upper[i]

    return st_val, st_dir, final_upper, final_lower


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_st_trail(
    pair:        str,
    df:          pd.DataFrame,
    st_period:   int,
    st_mult:     float,
    lots:        float,
    max_pending: int = 20,
) -> list[dict]:

    opens  = df["Open"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    times  = df.index

    st_val, st_dir, final_upper, final_lower = _supertrend(df, st_period, st_mult)
    atr_v = _atr(df, 14)

    half_sp     = HALF_SPREAD_PIPS[pair] * PIP_SIZE
    pip_val     = PIP_VALUE[pair]
    comm        = COMMISSION_PER_LOT * lots
    spread_cost = HALF_SPREAD_PIPS[pair] * pip_val * lots

    warmup = st_period * 3 + 1

    state       = "watching"
    pending_bar = 0
    trade: dict | None = None
    trades: list[dict] = []

    for i in range(warmup, len(df)):
        bar_open  = opens[i]
        bar_high  = highs[i]
        bar_low   = lows[i]
        bar_close = closes[i]
        bar_time  = times[i]
        dir_cur   = st_dir[i]

        # ── In trade ──────────────────────────────────────────────────────
        if state == "in_trade":
            sl = trade["sl"]

            exit_price = exit_type = None
            if bar_open < sl:
                exit_price, exit_type = bar_open, "sl_gap"
            elif bar_low <= sl:
                exit_price, exit_type = sl, "sl"

            if exit_price is not None:
                pip_move = (exit_price - trade["entry"]) / PIP_SIZE
                gross    = pip_move * pip_val * lots
                net      = gross - trade["entry_cost"] - spread_cost - comm
                trades.append({
                    "time":       trade["time"],
                    "close_time": bar_time,
                    "pair":       pair,
                    "entry":      trade["entry"],
                    "exit":       exit_price,
                    "exit_type":  exit_type,
                    "net":        net,
                    "won":        net > 0,
                    "sl0_pips":   abs(trade["entry"] - trade["sl0"]) / PIP_SIZE,
                    "hold_bars":  i - trade["bar_idx"],
                    "atr":        atr_v[i],
                })
                state = "watching"
                trade = None
            elif dir_cur == 1:
                # Trail SL upward with ST lower band
                trade["sl"] = max(trade["sl"], final_lower[i])
            continue

        # ── Pending buy-stop ───────────────────────────────────────────────
        if state == "pending":
            if dir_cur == 1:
                # ST flipped green — fill at the upper band just crossed
                entry_lvl = final_upper[i]
                entry     = entry_lvl + half_sp
                state     = "in_trade"
                trade = {
                    "time":       bar_time,
                    "bar_idx":    i,
                    "entry":      entry,
                    "sl":         final_lower[i],
                    "sl0":        final_lower[i],   # initial SL level for risk reporting
                    "entry_cost": spread_cost,
                }
                pending_bar = 0
            elif i - pending_bar >= max_pending:
                state = "watching"   # timeout — cancel pending
            continue

        # ── Signal detection ──────────────────────────────────────────────
        if dir_cur == -1:
            prev_h  = highs[i - 1]   if i > warmup else 0.0
            prev_st = st_val[i - 1]  if i > warmup else 1e9
            touched = bar_high >= st_val[i] or prev_h >= prev_st
            if touched and bar_close < st_val[i]:
                state       = "pending"
                pending_bar = i

    return trades


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(trades: list[dict], label: str, period: str) -> dict:
    base = {"label": label, "period": period, "n": 0, "pnl": 0.0,
            "win_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "sharpe": 0.0, "max_dd": 0.0, "ev": 0.0,
            "avg_sl0_pips": 0.0, "avg_hold": 0.0}
    if not trades:
        return base
    nets  = np.array([t["net"]       for t in trades])
    won   = np.array([t["won"]       for t in trades], dtype=bool)
    holds = np.array([t["hold_bars"] for t in trades])
    n     = len(trades)
    pnl   = nets.sum()
    aw    = nets[won].mean()   if won.any()    else 0.0
    al    = nets[~won].mean()  if (~won).any() else 0.0
    if n > 1 and nets.std() > 0:
        days   = max(1, (trades[-1]["time"] - trades[0]["time"]).days)
        sharpe = nets.mean() / nets.std() * np.sqrt(n / (days / 365.25))
    else:
        sharpe = 0.0
    cum    = np.cumsum(nets)
    max_dd = (cum - np.maximum.accumulate(cum)).min()
    avg_sl = np.mean([t["sl0_pips"] for t in trades])
    return {
        "label": label, "period": period, "n": n,
        "pnl": round(pnl, 1), "win_pct": round(won.mean() * 100, 1),
        "avg_win": round(aw, 2), "avg_loss": round(al, 2),
        "sharpe": round(sharpe, 2), "max_dd": round(max_dd, 1),
        "ev": round(pnl / n, 2), "avg_sl0_pips": round(avg_sl, 2),
        "avg_hold": round(holds.mean(), 1),
    }


def run_config(
    label: str, pairs_data: dict, oos_start: pd.Timestamp,
    st_period: int, st_mult: float, lots: float, max_pending: int,
) -> tuple[dict, dict]:
    all_is, all_oos = [], []
    for pair, df in pairs_data.items():
        for t in simulate_st_trail(pair, df, st_period, st_mult, lots, max_pending):
            (all_is if t["time"] < oos_start else all_oos).append(t)
    return metrics(all_is, label, "IS"), metrics(all_oos, label, "OOS")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--start",         default="2026-01-01")
    ap.add_argument("--end",           default="2026-05-07")
    ap.add_argument("--oos-start",     default="2026-04-01")
    ap.add_argument("--timeframe",     default="5m", choices=list(_MT5_TF_MAP.keys()))
    ap.add_argument("--lots",          type=float, default=1.0)
    ap.add_argument("--max-pending",   type=int,   default=20)
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
    for st_p in [7, 10, 14]:
        for st_m in [2.0, 3.0, 4.0]:
            lbl = f"st{st_p}_m{st_m}"
            log.info("Running: %s", lbl)
            m_is, m_oos = run_config(
                lbl, pairs_data, oos_start,
                st_period=st_p, st_mult=st_m,
                lots=args.lots, max_pending=args.max_pending,
            )
            results_is.append(m_is)
            results_oos.append(m_oos)

    cols = ["label", "n", "pnl", "win_pct", "avg_win", "avg_loss",
            "ev", "max_dd", "sharpe", "avg_sl0_pips", "avg_hold"]

    df_is  = pd.DataFrame(results_is)[cols].sort_values("pnl", ascending=False)
    df_oos = pd.DataFrame(results_oos)[cols].sort_values("pnl", ascending=False)

    def _fmt(df, title):
        print(f"\n{'='*110}")
        print(f"  {title}  (tf={args.timeframe}, lots={args.lots})")
        print(f"{'='*110}")
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    _fmt(df_is,  f"IN-SAMPLE  ({args.start} → {args.oos_start})")
    _fmt(df_oos, f"OUT-OF-SAMPLE  ({args.oos_start} → {args.end})")

    print(f"\n{'='*110}")
    print("  TOP-3 OOS vs IS")
    print(f"{'='*110}")
    for lbl in df_oos.head(3)["label"]:
        ir  = df_is [df_is ["label"] == lbl].iloc[0]
        or_ = df_oos[df_oos["label"] == lbl].iloc[0]
        print(f"\n  {lbl}")
        print(f"    IS  n={ir['n']:4d}  pnl=${ir['pnl']:8,.1f}  win={ir['win_pct']:.1f}%  "
              f"avg_win=${ir['avg_win']:,.2f}  avg_loss=${ir['avg_loss']:,.2f}  "
              f"hold={ir['avg_hold']:.0f}b  sharpe={ir['sharpe']:.2f}  maxDD=${ir['max_dd']:,.1f}")
        print(f"    OOS n={or_['n']:4d}  pnl=${or_['pnl']:8,.1f}  win={or_['win_pct']:.1f}%  "
              f"avg_win=${or_['avg_win']:,.2f}  avg_loss=${or_['avg_loss']:,.2f}  "
              f"hold={or_['avg_hold']:.0f}b  sharpe={or_['sharpe']:.2f}  maxDD=${or_['max_dd']:,.1f}")


if __name__ == "__main__":
    main()
