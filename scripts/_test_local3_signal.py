"""
One-off: compare argmax/argmin signal vs a 3-point local-extremum signal.

local3 logic (faster than argmax over a 20-bar window):
  LONG  if dema[-1] > dema[-3] AND dema[-3] < dema[-5]   (V-shape, dema[-3] is the bottom)
  SHORT if dema[-1] < dema[-3] AND dema[-3] > dema[-5]   (inverted-V, dema[-3] is the top)

Runs the 1h backtest on all 8 default pairs (per-pair, no FTMO filter — this
script is purely a signal-shape comparison) and prints IS/OOS metrics
side-by-side.
"""
from __future__ import annotations
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

import scripts.strategy_combined_backtest as cb
from scripts.strategy1_regime_backtest import (
    _load_data_mt5, _ema, PAIRS, PIP_SIZE,
)

_original_compute_signals = cb._compute_signals


def _compute_signals_local3(df, ema_period, window, min_swing_pips,
                            early_entry=False, pip_size=PIP_SIZE):
    """3-point V/inverted-V detection on DEMA samples at [-1, -3, -5]."""
    highs = df["High"].values
    lows  = df["Low"].values
    hl2   = (highs + lows) / 2.0
    ema_v = _ema(_ema(hl2, ema_period), ema_period)
    n     = len(ema_v)
    sigs  = np.zeros(n, dtype=int)
    warmup = ema_period * 2 + 5
    for i in range(warmup, n):
        d0 = ema_v[i]      # dema[-1] — current bar
        d1 = ema_v[i - 2]  # dema[-3] — 2 bars ago
        d2 = ema_v[i - 4]  # dema[-5] — 4 bars ago
        if d0 > d1 and d1 < d2:
            sigs[i] = 1     # local min → LONG
        elif d0 < d1 and d1 > d2:
            sigs[i] = -1    # local max → SHORT
    return sigs


def run_backtest(data_1h, oos_start):
    is_trades, oos_trades = [], []
    common_kw = dict(
        ema_1h=9, window=20, min_swing=0.0,
        lots_1h=2.0,
        close_profit=True, daily_stop_usd=0.0,
        min_bars_1h=2,
        partial_usd=0.0,
        sl_pips_1h=40.0,
        max_dist_pips=0.0,
        early_1h=True,
    )
    for pair in PAIRS:
        df1 = data_1h.get(pair)
        if df1 is None or df1.empty:
            continue
        for t in cb.simulate_pair(pair, df1, n_exit=0, **common_kw):
            (is_trades if t["time"] < oos_start else oos_trades).append(t)
    return is_trades, oos_trades


def metrics_line(trades, period):
    if not trades:
        return f"  {period:>4s}  (no trades)"
    nets = np.array([t["net"] for t in trades])
    won  = np.array([t["won"] for t in trades])
    n = len(trades)
    pnl = nets.sum()
    win_pct = won.mean() * 100
    if n > 1 and nets.std() > 0:
        days = max(1, (trades[-1]["close_time"] - trades[0]["time"]).days)
        sharpe = nets.mean() / nets.std() * np.sqrt(n / (days / 365.25))
    else:
        sharpe = 0.0
    cum = np.cumsum(nets)
    max_dd = (cum - np.maximum.accumulate(cum)).min()
    daily = {}
    for t in trades:
        d = t["close_time"].date()
        daily[d] = daily.get(d, 0.0) + t["net"]
    pnls = list(daily.values())
    worst = min(pnls) if pnls else 0.0
    best = max(pnls) if pnls else 0.0
    return (f"  {period:>4s}  n={n:>5d}  pnl=${pnl:>+10,.0f}  "
            f"win={win_pct:5.1f}%  sharpe={sharpe:5.2f}  "
            f"max_dd=${max_dd:>+9,.0f}  worst_day=${worst:>+8,.0f}  "
            f"best_day=${best:>+8,.0f}")


def main():
    print("Loading 1h data…")
    data_1h = _load_data_mt5(PAIRS, "2026-01-01", "2026-05-07", timeframe="1h")
    oos_start = pd.Timestamp("2026-04-01", tz="UTC")

    print("\n" + "=" * 130)
    print("  ARGMAX/ARGMIN over 20-bar window  (current default)")
    print("=" * 130)
    cb._compute_signals = _original_compute_signals
    is_t, oos_t = run_backtest(data_1h, oos_start)
    print(metrics_line(is_t, "IS"))
    print(metrics_line(oos_t, "OOS"))

    print("\n" + "=" * 130)
    print("  LOCAL3  (V-shape on dema[-1], dema[-3], dema[-5])")
    print("=" * 130)
    cb._compute_signals = _compute_signals_local3
    is_t, oos_t = run_backtest(data_1h, oos_start)
    print(metrics_line(is_t, "IS"))
    print(metrics_line(oos_t, "OOS"))


if __name__ == "__main__":
    main()
