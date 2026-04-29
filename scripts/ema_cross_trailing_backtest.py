"""
EMA crossover trailing stop -- long only.

Entry:  price closes above EMA (was below on previous bar)
        enter at next bar's open
Trailing stop: EMA value updated each bar
               exit when bar's LOW crosses below current EMA value
               (conservative: exit at EMA price, not bar close)
TP:     optional fixed TP in ATR multiples (0 = no TP, pure trailing stop)

Tests multiple EMA periods: 9, 13, 21, 50

Usage:
  python -m scripts.ema_cross_trailing_backtest
  python -m scripts.ema_cross_trailing_backtest --symbol EURUSD --tf 1h
  python -m scripts.ema_cross_trailing_backtest --ema 21 --tf 15m
  python -m scripts.ema_cross_trailing_backtest --tp-atr 3.0
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


def add_features(df: pd.DataFrame, ema_periods: list[int], atr_len: int = 14) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = ta.atr(out["High"], out["Low"], out["Close"], length=atr_len)
    for p in ema_periods:
        out[f"ema{p}"] = ta.ema(out["Close"], length=p)
    return out


def detect_signals(df: pd.DataFrame, cy_hours_arr: np.ndarray,
                   ema_col: str, session_start: int, session_end: int) -> pd.DataFrame:
    close  = df["Close"].values
    ema_v  = df[ema_col].values
    atr_v  = df["atr"].values
    n      = len(df)

    rows: list[dict] = []
    for i in range(1, n - 1):
        if np.isnan(ema_v[i]) or np.isnan(ema_v[i - 1]) or np.isnan(atr_v[i]):
            continue
        if not (session_start <= cy_hours_arr[i] < session_end):
            continue
        # crossover: prev close below EMA, current close above EMA
        if close[i - 1] < ema_v[i - 1] and close[i] > ema_v[i]:
            rows.append({
                "bar_idx":    i,
                "signal_time": df.index[i],
                "atr":        atr_v[i],
                "ema_at_signal": ema_v[i],
                "cy_hour":    int(cy_hours_arr[i]),
                "dow":        int(df.index.dayofweek[i]),
            })

    return pd.DataFrame(rows)


def _sim_one(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
             close: np.ndarray, ema_v: np.ndarray, atr_v: np.ndarray,
             n: int, sig_idx: int, atr: float,
             tp_atr: float, max_bars: int) -> dict:
    if sig_idx + 1 >= n:
        return dict(outcome="no_entry", entry_price=np.nan, exit_price=np.nan,
                    bars_held=0, pnl_pips=0.0, exit_idx=-1, tp_hit=np.nan,
                    max_fav_atr=0.0, max_adv_atr=0.0)

    entry     = open_[sig_idx + 1]
    tp_price  = entry + tp_atr * atr if tp_atr > 0 else np.inf

    max_fav   = 0.0
    max_adv   = 0.0

    end = min(sig_idx + 1 + max_bars, n)
    for j in range(sig_idx + 1, end):
        ema_now = ema_v[j]
        cur_atr = atr_v[j] if not np.isnan(atr_v[j]) and atr_v[j] > 0 else atr

        # track excursions
        fav = (high[j] - entry) / cur_atr
        adv = (entry - low[j]) / cur_atr
        if fav > max_fav:
            max_fav = fav
        if adv > max_adv:
            max_adv = adv

        # TP check first (favorable)
        if tp_price < np.inf and high[j] >= tp_price:
            pnl = (tp_price - entry) / PIP
            return dict(outcome="tp", entry_price=entry, exit_price=tp_price,
                        bars_held=j - sig_idx, pnl_pips=pnl,
                        exit_idx=j, tp_hit=1,
                        max_fav_atr=max_fav, max_adv_atr=max_adv)

        # Trailing stop: close below current EMA (exit at next bar open)
        if not np.isnan(ema_now) and close[j] < ema_now:
            # exit at open of next bar (signal triggers on close)
            if j + 1 < n:
                exit_price = open_[j + 1]
                exit_idx   = j + 1
            else:
                exit_price = close[j]
                exit_idx   = j
            pnl = (exit_price - entry) / PIP
            outcome = "trail_stop"
            tp_hit  = 1 if pnl > 0 else 0
            return dict(outcome=outcome, entry_price=entry, exit_price=exit_price,
                        bars_held=exit_idx - sig_idx, pnl_pips=pnl,
                        exit_idx=exit_idx, tp_hit=tp_hit,
                        max_fav_atr=max_fav, max_adv_atr=max_adv)

    # timeout
    pnl = (close[end - 1] - entry) / PIP
    return dict(outcome="timeout", entry_price=entry, exit_price=close[end - 1],
                bars_held=end - 1 - sig_idx, pnl_pips=pnl,
                exit_idx=end - 1, tp_hit=np.nan,
                max_fav_atr=max_fav, max_adv_atr=max_adv)


def simulate_all(df: pd.DataFrame, signals: pd.DataFrame,
                 ema_col: str, tp_atr: float,
                 commission_pips: float, max_bars: int) -> pd.DataFrame:
    if signals.empty:
        return signals
    open_  = df["Open"].values
    high   = df["High"].values
    low    = df["Low"].values
    close  = df["Close"].values
    ema_v  = df[ema_col].values
    atr_v  = df["atr"].values
    n      = len(df)
    out = []
    for r in signals.itertuples(index=False):
        sim = _sim_one(open_, high, low, close, ema_v, atr_v, n,
                       r.bar_idx, r.atr, tp_atr, max_bars)
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

    # outcome breakdown
    trail_wins  = ((trades["outcome"] == "trail_stop") & (trades["pnl_pips"] > 0)).sum()
    trail_loss  = ((trades["outcome"] == "trail_stop") & (trades["pnl_pips"] <= 0)).sum()
    avg_hold    = trades["bars_held"].mean()
    avg_fav     = trades["max_fav_atr"].mean()
    avg_adv     = trades["max_adv_atr"].mean()

    print(f"\n-- {name} --")
    print(f"  Trades          : {nt}")
    print(f"  Win/Loss/Timeout: {win}/{los}/{out}  ({wr:.1f}% win)")
    print(f"  Trail stop wins : {trail_wins}  losses: {trail_loss}")
    print(f"  Total P&L       : {pnl:+.1f} pips")
    print(f"  Avg per trade   : {avg:+.2f} pips")
    print(f"  Max DD (pips)   : {dd:+.1f}")
    print(f"  Trade Sharpe    : {tsr:+.3f}")
    print(f"  Avg bars held   : {avg_hold:.1f}")
    print(f"  Avg max fav ATR : {avg_fav:.2f}  avg max adv ATR: {avg_adv:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EMA crossover with EMA trailing stop -- long only")
    ap.add_argument("--symbol",        default="EURUSD")
    ap.add_argument("--tf",            default="1h")
    ap.add_argument("--start",         default="2022-01-01")
    ap.add_argument("--end",           default="2025-12-31")
    ap.add_argument("--ema",           type=int, default=0,
                    help="single EMA period to test (0 = test all: 9,13,21,50)")
    ap.add_argument("--atr-len",       type=int, default=14)
    ap.add_argument("--tp-atr",        type=float, default=0.0,
                    help="fixed TP in ATR multiples (0 = no TP, pure trailing)")
    ap.add_argument("--max-bars",      type=int, default=500)
    ap.add_argument("--commission",    type=float, default=0.00005)
    ap.add_argument("--session-start", type=int, default=7)
    ap.add_argument("--session-end",   type=int, default=16)
    args = ap.parse_args()

    ema_periods = [args.ema] if args.ema > 0 else [9, 13, 21, 50]

    print(f"Fetching {args.symbol} {args.tf}  {args.start} -> {args.end}")
    raw = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  fetched {len(raw):,} bars")
    if raw.empty:
        return

    print("Computing EMAs + ATR...")
    df = add_features(raw, ema_periods, atr_len=args.atr_len)

    print("Converting to Cyprus time...")
    cy_hours_arr = _cy_hours(df)

    commission_pips = args.commission / PIP
    out_dir = Path(__file__).resolve().parent

    print("\n" + "=" * 70)
    print(f"RESULTS -- {args.symbol} {args.tf}  "
          f"session={args.session_start:02d}:00-{args.session_end:02d}:00 CY")
    tp_str = f"TP={args.tp_atr}xATR" if args.tp_atr > 0 else "no TP"
    print(f"  entry: close crosses above EMA  |  trailing stop: EMA  |  {tp_str}")
    print("=" * 70)

    for p in ema_periods:
        ema_col = f"ema{p}"
        sigs = detect_signals(df, cy_hours_arr, ema_col,
                              args.session_start, args.session_end)
        if sigs.empty:
            print(f"\nEMA {p}: no signals")
            continue

        sim = simulate_all(df, sigs, ema_col, args.tp_atr,
                           commission_pips, args.max_bars)
        seq = sequential_take(sim)

        report(f"EMA {p} -- ALL signals (independent)", sim, len(df), args.tf)
        report(f"EMA {p} -- SEQUENTIAL (one at a time)", seq, len(df), args.tf)

        # Hour breakdown for sequential
        if len(seq) >= 20:
            print(f"\n  EMA {p} sequential -- by CY hour:")
            for h in sorted(seq["cy_hour"].unique()):
                sub = seq[seq["cy_hour"] == h]
                if len(sub) < 5:
                    continue
                wr_h = (sub["tp_hit"] == 1).sum() / len(sub) * 100
                pnl_h = sub["pnl_pips_net"].sum()
                print(f"    hour {h:02d}: n={len(sub):3d}  win={wr_h:.0f}%  pnl={pnl_h:+.1f}")

        sim.to_csv(out_dir / f"ema_cross_trail_all_{args.symbol}_{args.tf}_ema{p}.csv", index=False)
        seq.to_csv(out_dir / f"ema_cross_trail_seq_{args.symbol}_{args.tf}_ema{p}.csv", index=False)

    print()


if __name__ == "__main__":
    main()
