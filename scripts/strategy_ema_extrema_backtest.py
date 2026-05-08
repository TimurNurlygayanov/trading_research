"""
EMA local-extrema flip strategy — multi-variant, multi-timeframe backtest.

Signal (no look-ahead):
  Take last `window` bars of EMA.  Find where peak (argmax) and valley (argmin) sit.
  - Peak more recent → EMA declining → SHORT
  - Valley more recent → EMA rising  → LONG

Exit variants:
  no_sl         — hold until opposite signal flips; no SL
  close_profit  — every bar: if closing now would be net-positive, exit (go flat, await next signal)
  tp50_no_sl    — close 50%% at 1 ATR, hold remainder until flip
  tp50_closeP   — close 50%% at 1 ATR, close remainder as soon as net-positive

Cost model (proportional to lot fraction closed):
  2 × half-spread + COMMISSION_PER_LOT  (entry spread + exit spread + round-trip commission)

Sweep: EMA [9, 20] × window [10, 20] × 4 variants × N timeframes

Usage
-----
  python -m scripts.strategy_ema_extrema_backtest \\
      --source mt5 --timeframes 1m 5m 1h \\
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
    _ema, _atr,
    PAIRS, HALF_SPREAD_PIPS, PIP_VALUE, PIP_SIZE, COMMISSION_PER_LOT,
)
from backtest.data_fetcher import fetch_ohlcv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ema_extrema")


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_ema_extrema(
    pair:             str,
    df:               pd.DataFrame,
    ema_period:       int,
    window:           int,
    lots:             float,
    close_profit:     bool  = False,   # exit as soon as intrabar net P&L > 0
    tp_partial:       float = 0.0,     # 0 = disabled, 0.5 = close half at 1 ATR
    tp_atr_mult:      float = 1.0,
    min_swing_pips:   float = 0.0,     # min EMA swing from prior opposite extremum to qualify signal
) -> list[dict]:

    opens  = df["Open"].values
    lows   = df["Low"].values
    highs  = df["High"].values
    closes = df["Close"].values
    times  = df.index

    hl2   = (highs + lows) / 2.0
    ema_v = _ema(_ema(hl2, ema_period), ema_period)   # double EMA of HL/2
    atr_v = _atr(df, 14)

    pip_val  = PIP_VALUE[pair]
    sp_cost  = HALF_SPREAD_PIPS[pair] * pip_val * lots
    comm     = COMMISSION_PER_LOT * lots

    warmup = ema_period * 2 + window + 5

    min_swing = min_swing_pips * PIP_SIZE

    def _sig(i: int) -> int:
        """Local max/min of double-EMA within the last `window` bars.
        Double EMA is smooth enough that no confirmation bar is needed.
        Peak not at latest bar → SHORT. Valley not at latest bar → LONG.
        """
        dw      = ema_v[i - window + 1 : i + 1]
        pos_max = int(np.argmax(dw))
        pos_min = int(np.argmin(dw))
        if pos_max == pos_min:
            return 0
        if (dw[pos_max] - dw[pos_min]) < min_swing:
            return 0
        # Peak more recent than valley, and not at the current bar
        if pos_max > pos_min and pos_max < window - 1:
            return -1
        # Valley more recent than peak, and not at the current bar
        if pos_min > pos_max and pos_min < window - 1:
            return 1
        return 0

    def _net(gross: float, frac: float) -> float:
        """Net after costs proportional to `frac` of full position."""
        return gross - 2.0 * sp_cost * frac - comm * frac

    position = 0
    trade: dict | None = None
    trades: list[dict] = []

    for i in range(warmup, len(df)):
        bar_high  = highs[i]
        bar_low   = lows[i]
        bar_close = closes[i]
        bar_time  = times[i]

        # ── In trade ──────────────────────────────────────────────────────
        if trade is not None:
            lots_rem = trade["lots_rem"]

            # 1. Partial TP (if enabled, not yet hit)
            if tp_partial > 0 and not trade["tp_hit"]:
                tp_price = trade["entry"] + position * tp_atr_mult * trade["atr"]
                if (position == 1 and bar_high >= tp_price) or \
                   (position == -1 and bar_low <= tp_price):
                    frac     = tp_partial
                    pip_move = (tp_price - trade["entry"]) * position / PIP_SIZE
                    gross    = pip_move * pip_val * lots * frac
                    trades.append(_rec(trade, bar_time, i,
                                       tp_price, _net(gross, frac), "tp_partial"))
                    trade["tp_hit"]   = True
                    trade["lots_rem"] = 1.0 - tp_partial

            # 2. Close-when-profitable check (intrabar: high for LONG, low for SHORT)
            if close_profit:
                lots_rem    = trade["lots_rem"]
                check_price = bar_high if position == 1 else bar_low
                pip_move    = (check_price - trade["entry"]) * position / PIP_SIZE
                gross       = pip_move * pip_val * lots * lots_rem
                if _net(gross, lots_rem) > 0:
                    trades.append(_rec(trade, bar_time, i,
                                       check_price, _net(gross, lots_rem), "profit_close"))
                    position = 0
                    trade    = None
                    continue   # flat — wait for next signal

            # 3. Signal flip: close remaining + open opposite
            if trade is not None:
                sig = _sig(i)
                if sig != 0 and sig != position:
                    lots_rem = trade["lots_rem"]
                    pip_move = (bar_close - trade["entry"]) * position / PIP_SIZE
                    gross    = pip_move * pip_val * lots * lots_rem
                    trades.append(_rec(trade, bar_time, i,
                                       bar_close, _net(gross, lots_rem), "flip"))
                    position = sig
                    trade    = {"time": bar_time, "bar_idx": i, "entry": bar_close,
                                "atr": atr_v[i], "tp_hit": False, "lots_rem": 1.0}
            continue

        # ── Flat: enter on signal ──────────────────────────────────────────
        sig = _sig(i)
        if sig != 0:
            position = sig
            trade    = {"time": bar_time, "bar_idx": i, "entry": bar_close,
                        "atr": atr_v[i], "tp_hit": False, "lots_rem": 1.0}

    return trades


def _rec(trade: dict, close_time, close_idx: int,
         exit_price: float, net: float, exit_type: str) -> dict:
    return {
        "time":       trade["time"],
        "close_time": close_time,
        "entry":      trade["entry"],
        "exit":       exit_price,
        "exit_type":  exit_type,
        "net":        net,
        "won":        net > 0,
        "hold_bars":  close_idx - trade["bar_idx"],
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(trades: list[dict], label: str, period: str) -> dict:
    base = {"label": label, "period": period, "n": 0, "pnl": 0.0,
            "win_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "sharpe": 0.0, "max_dd": 0.0, "ev": 0.0, "avg_hold": 0.0}
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
    return {
        "label":    label,  "period": period, "n": n,
        "pnl":      round(pnl, 1),
        "win_pct":  round(won.mean() * 100, 1),
        "avg_win":  round(aw, 2),
        "avg_loss": round(al, 2),
        "sharpe":   round(sharpe, 2),
        "max_dd":   round(max_dd, 1),
        "ev":       round(pnl / n, 2),
        "avg_hold": round(holds.mean(), 1),
    }


def run_tf(
    tf: str, pairs_data: dict, oos_start: pd.Timestamp, lots: float,
) -> None:
    VARIANTS = [
        # (suffix,        close_profit, tp_partial, tp_atr)
        ("no_sl",         False,        0.0,        1.0),
        ("close_profit",  True,         0.0,        1.0),
        ("tp50_no_sl",    False,        0.5,        1.0),
        ("tp50_closeP",   True,         0.5,        1.0),
    ]

    results_is, results_oos = [], []
    for ema_p in [9, 20, 50]:
        for win in [5, 10, 20]:
            for min_sw in [0.0, 3.0, 5.0, 10.0]:
                for vsuffix, cp, tp_frac, tp_atr in VARIANTS:
                    lbl = f"ema{ema_p}_w{win}_sw{min_sw}_{vsuffix}"
                    all_is, all_oos = [], []
                    for pair, df in pairs_data.items():
                        for t in simulate_ema_extrema(
                            pair, df, ema_p, win, lots,
                            close_profit=cp,
                            tp_partial=tp_frac, tp_atr_mult=tp_atr,
                            min_swing_pips=min_sw,
                        ):
                            (all_is if t["time"] < oos_start else all_oos).append(t)
                    results_is.append(metrics(all_is,  lbl, "IS"))
                    results_oos.append(metrics(all_oos, lbl, "OOS"))

    cols = ["label", "n", "pnl", "win_pct", "avg_win", "avg_loss",
            "ev", "max_dd", "sharpe", "avg_hold"]
    df_is  = pd.DataFrame(results_is)[cols].sort_values("pnl", ascending=False)
    df_oos = pd.DataFrame(results_oos)[cols].sort_values("pnl", ascending=False)

    W = 110
    def _fmt(df, title):
        print(f"\n{'='*W}")
        print(f"  {title}  (tf={tf}, lots={lots})")
        print(f"{'='*W}")
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    _fmt(df_is,  f"IN-SAMPLE")
    _fmt(df_oos, f"OUT-OF-SAMPLE")

    # Variant summary
    print(f"\n{'='*W}")
    print(f"  VARIANT SUMMARY — OOS  tf={tf}")
    print(f"{'='*W}")
    variant_names = [v[0] for v in VARIANTS]
    rows_summary = []
    for vsuffix in variant_names:
        rows = df_oos[df_oos["label"].str.endswith(vsuffix)]
        if rows.empty:
            continue
        best = rows.loc[rows["pnl"].idxmax()]
        rows_summary.append({
            "variant":     vsuffix,
            "avg_pnl":     rows["pnl"].mean(),
            "avg_ev":      rows["ev"].mean(),
            "avg_win%":    rows["win_pct"].mean(),
            "avg_sharpe":  rows["sharpe"].mean(),
            "best_label":  best["label"],
            "best_pnl":    best["pnl"],
        })
    sm = pd.DataFrame(rows_summary)
    for _, r in sm.iterrows():
        print(f"  {r['variant']:<16}  avg_pnl=${r['avg_pnl']:8,.0f}  avg_ev=${r['avg_ev']:6.2f}  "
              f"avg_win={r['avg_win%']:.1f}%  avg_sharpe={r['avg_sharpe']:6.2f}"
              f"  best={r['best_label']} (${r['best_pnl']:,.0f})")

    # Top-3 OOS configs
    print(f"\n{'='*W}")
    print(f"  TOP-3 OOS  tf={tf}")
    print(f"{'='*W}")
    for lbl in df_oos.head(3)["label"]:
        ir  = df_is [df_is ["label"] == lbl].iloc[0]
        or_ = df_oos[df_oos["label"] == lbl].iloc[0]
        print(f"\n  {lbl}")
        print(f"    IS  n={ir['n']:5d}  pnl=${ir['pnl']:9,.1f}  win={ir['win_pct']:.1f}%  "
              f"ev=${ir['ev']:6.2f}  hold={ir['avg_hold']:.0f}b  sharpe={ir['sharpe']:.2f}")
        print(f"    OOS n={or_['n']:5d}  pnl=${or_['pnl']:9,.1f}  win={or_['win_pct']:.1f}%  "
              f"ev=${or_['ev']:6.2f}  hold={or_['avg_hold']:.0f}b  sharpe={or_['sharpe']:.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--start",         default="2026-01-01")
    ap.add_argument("--end",           default="2026-05-07")
    ap.add_argument("--oos-start",     default="2026-04-01")
    ap.add_argument("--timeframes",    nargs="+", default=["5m"],
                    choices=list(_MT5_TF_MAP.keys()),
                    help="One or more timeframes to run, e.g. --timeframes 1m 5m 1h")
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

    for tf in args.timeframes:
        log.info("━━━ Timeframe: %s ━━━  loading %s data (%s → %s)",
                 tf, args.source, args.start, args.end)
        pairs_data: dict[str, pd.DataFrame] = {}

        if args.source == "mt5":
            try:
                pairs_data = _load_data_mt5(
                    pairs, args.start, args.end, timeframe=tf,
                    mt5_path=args.mt5_path, login=args.login,
                    password=args.password, server=args.server,
                    symbol_suffix=args.symbol_suffix,
                )
            except Exception as exc:
                log.error("MT5 load failed for %s: %s", tf, exc)
                continue
        else:
            for pair in pairs:
                try:
                    df = fetch_ohlcv(pair, tf, args.start, args.end)
                    pairs_data[pair] = df
                    log.info("  %s %s: %d bars", tf, pair, len(df))
                except Exception as exc:
                    log.error("  %s %s: %s", tf, pair, exc)

        if not pairs_data:
            log.error("No data for %s.", tf)
            continue

        run_tf(tf, pairs_data, oos_start, args.lots)


if __name__ == "__main__":
    main()
