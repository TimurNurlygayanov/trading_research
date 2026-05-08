"""
Replay pure-fade strategy on today's 1m bars (2026-05-06).
Uses full history from 2024-01-01 for proper ATR/EMA warmup.
"""
import sys, warnings
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.strategy1_regime_backtest import (
    simulate_pair, regime_pure_fade,
    PAIRS, HALF_SPREAD_PIPS, PIP_VALUE, PIP_SIZE,
)
from backtest.data_fetcher import fetch_ohlcv

TODAY = pd.Timestamp("2026-05-06", tz="UTC")
LOOKBACK   = 10
ATR_PERIOD = 14
EMA_PERIOD = 20
LOTS       = 1.0
TP_FADE    = 1.0
SL_FADE    = 2.0

print(f"Loading data (uses cache)...")
all_trades = []
for pair in PAIRS:
    df = fetch_ohlcv(pair, "1m", "2024-01-01", "2026-05-07")
    trades = simulate_pair(
        pair, df, LOOKBACK, ATR_PERIOD, EMA_PERIOD,
        TP_FADE, SL_FADE, 2.0, 1.0,
        LOTS, None, regime_pure_fade, None,
    )
    today_trades = [t for t in trades if t["time"].date() == TODAY.date()]
    all_trades.extend(today_trades)
    wins  = sum(1 for t in today_trades if t["won"])
    total = len(today_trades)
    pnl   = sum(t["net"] for t in today_trades)
    wr    = wins / total * 100 if total else 0
    print(f"  {pair}: {total:3d} trades  win={wr:.0f}%  pnl=${pnl:,.1f}")

if not all_trades:
    print("No trades found for today.")
    sys.exit()

nets  = np.array([t["net"] for t in all_trades])
won   = np.array([t["won"] for t in all_trades])
total = len(all_trades)

print(f"\n=== TODAY ({TODAY.date()}) — pure fade, all pairs ===")
print(f"  Trades:    {total}")
print(f"  Win rate:  {won.mean()*100:.1f}%  ({won.sum()} wins / {(~won).sum()} losses)")
print(f"  Avg win:   ${nets[won].mean():.2f}")
print(f"  Avg loss:  ${nets[~won].mean():.2f}")
print(f"  Net P&L:   ${nets.sum():,.2f}")
print(f"  EV/trade:  ${nets.mean():.2f}")

# Hourly breakdown
print(f"\n=== HOURLY BREAKDOWN ===")
print(f"  Hour  | Trades | Win%  | P&L")
print(f"  ------|--------|-------|--------")
by_hour = {}
for t in all_trades:
    h = t["time"].hour
    by_hour.setdefault(h, []).append(t)
for h in sorted(by_hour):
    g = by_hour[h]
    wr_h = sum(1 for t in g if t["won"]) / len(g) * 100
    pnl_h = sum(t["net"] for t in g)
    flag = " <<< LOSING" if wr_h < 50 else ""
    print(f"  {h:02d}h   | {len(g):6d} | {wr_h:4.0f}% | ${pnl_h:8.1f}{flag}")

# Mode breakdown (fade vs breakout — always fade here, but shows mode field)
# What would circuit breaker have done?
print(f"\n=== CIRCUIT BREAKER SIMULATION (6/10 losses -> pause) ===")
for break_hours in [1, 2, 3]:
    from scripts.strategy1_regime_backtest import simulate_pair as _sp
    pair_trades = []
    for pair in PAIRS:
        df = fetch_ohlcv(pair, "1m", "2024-01-01", "2026-05-07")
        trades = _sp(
            pair, df, LOOKBACK, ATR_PERIOD, EMA_PERIOD,
            TP_FADE, SL_FADE, 2.0, 1.0,
            LOTS, None, regime_pure_fade, None,
            cb_window=10, cb_min_losses=6, cb_break_bars=break_hours * 60,
        )
        pair_trades.extend(t for t in trades if t["time"].date() == TODAY.date())
    if pair_trades:
        n_cb = len(pair_trades)
        wr_cb = sum(1 for t in pair_trades if t["won"]) / n_cb * 100
        pnl_cb = sum(t["net"] for t in pair_trades)
        skipped = total - n_cb
        print(f"  {break_hours}h break: {n_cb} trades ({skipped} skipped)  "
              f"win={wr_cb:.0f}%  pnl=${pnl_cb:,.1f}")
