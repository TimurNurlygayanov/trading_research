"""
Small green candle fade -- short entry.

Signal: green candle (close > open) whose range (H-L) < range_atr_mult x ATR(14)
Entry:  short at the next bar's open
SL:     entry + sl_atr x ATR
TP:     entry - tp_atr x ATR

Usage:
  python -m scripts.small_green_fade_backtest
  python -m scripts.small_green_fade_backtest --symbol EURUSD --tf 5m
  python -m scripts.small_green_fade_backtest --range-mult 0.5 --sl-atr 2 --tp-atr 2
  python -m scripts.small_green_fade_backtest --session-start 0 --session-end 23
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv

PIP = 0.0001


def _cy_hours(df: pd.DataFrame) -> np.ndarray:
    idx = df.index
    try:
        idx_utc = idx.tz_localize("UTC")
    except TypeError:
        idx_utc = idx.tz_convert("UTC")
    return idx_utc.tz_convert("Asia/Nicosia").hour.values


def add_features(df: pd.DataFrame, atr_len: int = 14) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = ta.atr(out["High"], out["Low"], out["Close"], length=atr_len)
    return out


def detect_signals(df: pd.DataFrame, cy_hours_arr: np.ndarray,
                   session_start: int, session_end: int,
                   range_mult: float) -> pd.DataFrame:
    open_  = df["Open"].values
    high   = df["High"].values
    low    = df["Low"].values
    close  = df["Close"].values
    atr_v  = df["atr"].values
    n      = len(df)

    rows: list[dict] = []
    for i in range(n - 1):
        if np.isnan(atr_v[i]) or atr_v[i] <= 0:
            continue
        if not (session_start <= cy_hours_arr[i] < session_end):
            continue
        atr   = atr_v[i]
        green = close[i] > open_[i]
        small = (high[i] - low[i]) < range_mult * atr
        if green and small:
            rows.append({
                "bar_idx":    i,
                "signal_time": df.index[i],
                "atr":        atr,
                "candle_range": high[i] - low[i],
                "cy_hour":    int(cy_hours_arr[i]),
                "dow":        int(df.index.dayofweek[i]),
            })

    return pd.DataFrame(rows)


def _sim_one(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
             close: np.ndarray, n: int,
             sig_idx: int, atr: float, sl_atr: float, tp_atr: float,
             max_bars: int) -> dict:
    if sig_idx + 1 >= n:
        return dict(outcome="no_entry", entry_price=np.nan, exit_price=np.nan,
                    bars_held=0, pnl_pips=0.0, exit_idx=-1, tp_hit=np.nan)

    entry    = open_[sig_idx + 1]
    sl_price = entry + sl_atr * atr
    tp_price = entry - tp_atr * atr

    end = min(sig_idx + 1 + max_bars, n)
    for j in range(sig_idx + 1, end):
        if high[j] >= sl_price:   # SL wins on ambiguous bar
            pnl = (entry - sl_price) / PIP
            return dict(outcome="sl", entry_price=entry, exit_price=sl_price,
                        bars_held=j - sig_idx, pnl_pips=pnl, exit_idx=j, tp_hit=0)
        if low[j] <= tp_price:
            pnl = (entry - tp_price) / PIP
            return dict(outcome="tp", entry_price=entry, exit_price=tp_price,
                        bars_held=j - sig_idx, pnl_pips=pnl, exit_idx=j, tp_hit=1)

    pnl = (entry - close[end - 1]) / PIP
    return dict(outcome="timeout", entry_price=entry, exit_price=close[end - 1],
                bars_held=end - 1 - sig_idx, pnl_pips=pnl,
                exit_idx=end - 1, tp_hit=np.nan)


def simulate_all(df: pd.DataFrame, signals: pd.DataFrame,
                 sl_atr: float, tp_atr: float,
                 commission_pips: float, max_bars: int) -> pd.DataFrame:
    if signals.empty:
        return signals
    open_  = df["Open"].values
    high   = df["High"].values
    low    = df["Low"].values
    close  = df["Close"].values
    n      = len(df)
    out = []
    for r in signals.itertuples(index=False):
        sim = _sim_one(open_, high, low, close, n,
                       r.bar_idx, r.atr, sl_atr, tp_atr, max_bars)
        sim["pnl_pips_net"] = sim["pnl_pips"] - 2 * commission_pips
        out.append({**r._asdict(), **sim})
    return pd.DataFrame(out)


def sequential_take(simulated: pd.DataFrame) -> pd.DataFrame:
    if simulated.empty:
        return simulated
    s = simulated.sort_values("bar_idx").reset_index(drop=True)
    last_exit = -1
    keep = []
    for r in s.itertuples():
        if r.bar_idx <= last_exit:
            continue
        keep.append(r.Index)
        last_exit = max(last_exit, r.exit_idx)
    return s.loc[keep].reset_index(drop=True)


def _bars_per_year(tf: str) -> float:
    n, unit = int(tf[:-1]), tf[-1]
    mins = {"m": n, "h": n * 60, "d": n * 1440}[unit]
    return 365 * 24 * 60 / mins


def report(name: str, trades: pd.DataFrame, df_len: int, tf: str) -> None:
    if trades.empty:
        print(f"\n-- {name}: 0 trades --")
        return
    nt  = len(trades)
    pnl = trades["pnl_pips_net"].sum()
    avg = trades["pnl_pips_net"].mean()
    win = (trades["tp_hit"] == 1).sum()
    los = (trades["tp_hit"] == 0).sum()
    out = trades["tp_hit"].isna().sum()
    wr  = win / max(nt, 1) * 100
    eq  = trades["pnl_pips_net"].cumsum()
    dd  = (eq - eq.cummax()).min()
    bpy = _bars_per_year(tf)
    tpy = nt * bpy / max(df_len, 1)
    r   = trades["pnl_pips_net"]
    tsr = float(r.mean() / r.std() * np.sqrt(max(tpy, 1))) if r.std() > 0 else 0.0

    print(f"\n-- {name} --")
    print(f"  Trades         : {nt}")
    print(f"  TP hit         : {win}  ({wr:.1f}%)")
    print(f"  SL hit         : {los}")
    print(f"  Timeout        : {out}")
    print(f"  Total P&L      : {pnl:+.1f} pips")
    print(f"  Avg per trade  : {avg:+.2f} pips")
    print(f"  Max DD (pips)  : {dd:+.1f}")
    print(f"  Trade Sharpe   : {tsr:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Small green candle fade -- short")
    ap.add_argument("--symbol",         default="EURUSD")
    ap.add_argument("--tf",             default="5m")
    ap.add_argument("--start",          default="2021-01-01")
    ap.add_argument("--end",            default="2025-12-31")
    ap.add_argument("--atr-len",        type=int,   default=14)
    ap.add_argument("--range-mult",     type=float, default=0.5,
                    help="signal fires when H-L < range_mult x ATR (default 0.5)")
    ap.add_argument("--sl-atr",         type=float, default=2.0,
                    help="SL = entry + sl_atr x ATR")
    ap.add_argument("--tp-atr",         type=float, default=2.0,
                    help="TP = entry - tp_atr x ATR")
    ap.add_argument("--max-bars",       type=int,   default=500)
    ap.add_argument("--commission",     type=float, default=0.00005)
    ap.add_argument("--session-start",  type=int,   default=7,
                    help="session open, Cyprus local hour (default 7)")
    ap.add_argument("--session-end",    type=int,   default=16,
                    help="session close, Cyprus local hour (default 16)")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent

    print(f"Fetching {args.symbol} {args.tf}  {args.start} -> {args.end}")
    raw = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  fetched {len(raw):,} bars\n")
    if raw.empty:
        return

    print("Computing ATR...")
    df = add_features(raw, atr_len=args.atr_len)

    print("Converting to Cyprus time...")
    cy_hours_arr = _cy_hours(df)

    print("Detecting signals...")
    sigs = detect_signals(df, cy_hours_arr,
                          args.session_start, args.session_end,
                          args.range_mult)
    print(f"  signals found: {len(sigs)}")

    if sigs.empty:
        print("No signals.")
        return

    print("Simulating...")
    commission_pips = args.commission / PIP
    sim = simulate_all(df, sigs, args.sl_atr, args.tp_atr,
                       commission_pips, args.max_bars)
    seq = sequential_take(sim)

    print("\n" + "=" * 70)
    print(f"RESULTS -- {args.symbol} {args.tf}  "
          f"session={args.session_start:02d}:00-{args.session_end:02d}:00 CY")
    print(f"  signal: green candle, H-L < {args.range_mult}xATR  |  "
          f"SL={args.sl_atr}xATR  TP={args.tp_atr}xATR")
    print("=" * 70)
    report("ALL signals (independent)", sim, len(df), args.tf)
    report("SEQUENTIAL (one position at a time)", seq, len(df), args.tf)

    sig_path = out_dir / f"small_green_fade_all_{args.symbol}_{args.tf}.csv"
    seq_path = out_dir / f"small_green_fade_seq_{args.symbol}_{args.tf}.csv"
    sim.to_csv(sig_path, index=False)
    seq.to_csv(seq_path, index=False)
    print(f"\n-> {sig_path}")
    print(f"-> {seq_path}")


if __name__ == "__main__":
    main()
