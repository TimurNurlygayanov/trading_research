"""
Higher-Low long strategy — 1M EURUSD, London + US sessions only.

Pattern (causal, with confirmation lag)
  swing low at bar t     := Low[t] is the min of Low[t-n .. t+n]   (n = swing_n)
  higher low             := the next swing low after a previous swing low,
                            with Low[new] > Low[previous]
  confirmation bar       := t + n  (the earliest bar at which we know t was a swing)

Trade
  Entry  : market BUY at confirm_bar + 1 open
  SL     : Low of the higher-low bar
  TP     : entry + tp_rr × (entry − SL)        (default 1:10)

Session
  Trade only when confirm_bar's UTC hour ∈ [session_start, session_end)
  Default 08:00–22:00 UTC = London open through NY close.

What this script does
  1. Detect every higher-low signal causally.
  2. Phase-1 analysis: counts, swing-height distribution, stop/TP distances,
     raw forward returns at multiple horizons.
  3. Phase-2 backtest: forward-simulate each signal independently
     (TP / SL / timeout), plus sequential one-trade-at-a-time.
  4. Save labelled CSV with features (atr, rsi, ema_slope, swing_height_pips, …)
     for later RandomForest filter training.

Usage
  python -m scripts.higher_low_backtest
  python -m scripts.higher_low_backtest --swing-n 15 --tp-rr 8
  python -m scripts.higher_low_backtest --session-start 13 --session-end 21   # NY only
  python -m scripts.higher_low_backtest --start 2024-01-01 --end 2025-12-31
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


# ── feature add (for RF training later) ─────────────────────────────────────
def add_features(df: pd.DataFrame, atr_len: int, rsi_len: int,
                 ema_len: int, slope_n: int) -> pd.DataFrame:
    out = df.copy()
    out["atr"]  = ta.atr(out["High"], out["Low"], out["Close"], length=atr_len)
    out["rsi"]  = ta.rsi(out["Close"], length=rsi_len)
    out["ema"]  = ta.ema(out["Close"], length=ema_len)
    out["ema_slope_bps"] = (out["ema"] / out["ema"].shift(slope_n) - 1) * 10_000
    return out


# ── higher-low detection ────────────────────────────────────────────────────
def detect_higher_lows(df: pd.DataFrame, swing_n: int,
                       session_start: int, session_end: int) -> pd.DataFrame:
    """
    Return DataFrame of higher-low signals. Each row corresponds to the
    CONFIRMATION bar (= swing_idx + swing_n).
    """
    low = df["Low"].values
    n   = len(df)

    # Rolling-min over centered 2*swing_n+1 window: NaN at the edges
    win = 2 * swing_n + 1
    rolling_min = (
        pd.Series(low)
        .rolling(win, center=True, min_periods=win)
        .min()
        .values
    )
    is_swing = np.zeros(n, dtype=bool)
    for i in range(swing_n, n - swing_n):
        if not np.isnan(rolling_min[i]) and low[i] == rolling_min[i]:
            is_swing[i] = True

    hours = df.index.hour.values
    rows = []
    prev_swing_low = None
    prev_swing_idx = None

    for i in range(swing_n, n - swing_n):
        if not is_swing[i]:
            continue

        if prev_swing_low is not None and low[i] > prev_swing_low:
            confirm_idx = i + swing_n
            if confirm_idx >= n:
                prev_swing_low, prev_swing_idx = low[i], i
                continue
            h = int(hours[confirm_idx])
            if not (session_start <= h < session_end):
                prev_swing_low, prev_swing_idx = low[i], i
                continue
            rows.append({
                "confirm_idx":      confirm_idx,
                "swing_idx":        i,
                "prev_swing_idx":   prev_swing_idx,
                "swing_low":        float(low[i]),
                "prev_swing_low":   float(prev_swing_low),
                "swing_height_pips": (low[i] - prev_swing_low) / PIP,
                "bars_between":     i - prev_swing_idx,
                "session_hour":     h,
            })

        # Always update prev_swing_* on every confirmed swing
        prev_swing_low, prev_swing_idx = low[i], i

    sig = pd.DataFrame(rows)
    if not sig.empty:
        sig["signal_time"] = df.index[sig["confirm_idx"].values]
        # Pull features at confirmation bar
        for col in ("atr", "rsi", "ema_slope_bps"):
            if col in df.columns:
                sig[col] = df[col].values[sig["confirm_idx"].values]
    return sig


# ── forward simulation ──────────────────────────────────────────────────────
def simulate_signal(df: pd.DataFrame, confirm_idx: int, sl_price: float,
                    tp_rr: float, max_bars: int) -> dict:
    """
    LONG entry at confirm_idx + 1 open. SL = sl_price (below entry).
    TP = entry + tp_rr × (entry − SL). Conservative on bar-overlap (SL wins).
    """
    n = len(df)
    if confirm_idx + 1 >= n:
        return dict(outcome="no_entry", entry_price=np.nan, exit_price=np.nan,
                    sl_price=sl_price, tp_price=np.nan,
                    bars_held=0, pnl_pips=0.0, exit_idx=-1, tp_hit=np.nan,
                    stop_pips=0.0)

    entry = float(df["Open"].iloc[confirm_idx + 1])
    if entry <= sl_price:
        # Already at/below SL on entry bar — invalid setup, instant stop-out
        return dict(outcome="gap_invalid", entry_price=entry,
                    exit_price=sl_price, sl_price=sl_price, tp_price=np.nan,
                    bars_held=0, pnl_pips=(sl_price - entry) / PIP,
                    exit_idx=confirm_idx + 1, tp_hit=0,
                    stop_pips=(entry - sl_price) / PIP)

    stop_dist = entry - sl_price
    tp_price  = entry + tp_rr * stop_dist

    end = min(confirm_idx + 1 + max_bars, n)
    high = df["High"].values
    low  = df["Low"].values

    for j in range(confirm_idx + 1, end):
        h, l = high[j], low[j]
        hit_sl = l <= sl_price
        hit_tp = h >= tp_price

        if hit_sl:  # conservative on overlap
            return dict(outcome="sl", entry_price=entry, exit_price=sl_price,
                        sl_price=sl_price, tp_price=tp_price,
                        bars_held=j - confirm_idx,
                        pnl_pips=(sl_price - entry) / PIP,
                        exit_idx=j, tp_hit=0,
                        stop_pips=stop_dist / PIP)
        if hit_tp:
            return dict(outcome="tp", entry_price=entry, exit_price=tp_price,
                        sl_price=sl_price, tp_price=tp_price,
                        bars_held=j - confirm_idx,
                        pnl_pips=(tp_price - entry) / PIP,
                        exit_idx=j, tp_hit=1,
                        stop_pips=stop_dist / PIP)

    last_close = float(df["Close"].iloc[end - 1])
    return dict(outcome="timeout", entry_price=entry, exit_price=last_close,
                sl_price=sl_price, tp_price=tp_price,
                bars_held=end - 1 - confirm_idx,
                pnl_pips=(last_close - entry) / PIP,
                exit_idx=end - 1, tp_hit=np.nan,
                stop_pips=stop_dist / PIP)


def simulate_all(df: pd.DataFrame, signals: pd.DataFrame,
                 tp_rr: float, max_bars: int, commission_pips: float) -> pd.DataFrame:
    if signals.empty:
        return signals
    rows = []
    for r in signals.itertuples(index=False):
        sim = simulate_signal(df, r.confirm_idx, r.swing_low, tp_rr, max_bars)
        sim["pnl_pips_net"] = sim["pnl_pips"] - 2 * commission_pips
        rows.append({**r._asdict(), **sim})
    return pd.DataFrame(rows)


def sequential_take(simulated: pd.DataFrame) -> pd.DataFrame:
    if simulated.empty:
        return simulated
    s = simulated.sort_values("confirm_idx").reset_index(drop=True)
    last_exit = -1
    keep = []
    for r in s.itertuples():
        if r.confirm_idx <= last_exit:
            continue
        keep.append(r.Index)
        last_exit = max(last_exit, r.exit_idx)
    return s.loc[keep].reset_index(drop=True)


# ── reports ─────────────────────────────────────────────────────────────────
def report(name: str, trades: pd.DataFrame, df_len: int, tf: str):
    if trades.empty:
        print(f"\n── {name}: 0 trades ──")
        return
    n   = len(trades)
    pnl = trades["pnl_pips_net"].sum()
    avg = trades["pnl_pips_net"].mean()
    win = (trades["tp_hit"] == 1).sum()
    los = (trades["tp_hit"] == 0).sum()
    out = trades["tp_hit"].isna().sum()
    win_rate = win / max(n, 1) * 100

    eq = trades["pnl_pips_net"].cumsum()
    dd = (eq - eq.cummax()).min()

    bpy = bars_per_year_for_tf(tf)
    tpy = n * bpy / max(df_len, 1)
    r = trades["pnl_pips_net"]
    tsr = float(r.mean() / r.std() * np.sqrt(max(tpy, 1))) if r.std() > 0 else 0.0

    print(f"\n── {name} ──")
    print(f"  Signals/Trades : {n}")
    print(f"  TP hit         : {win}  ({win_rate:.1f}%)")
    print(f"  SL hit         : {los}")
    print(f"  Timeout        : {out}")
    print(f"  Total P&L      : {pnl:+.1f} pips")
    print(f"  Avg per trade  : {avg:+.2f} pips")
    print(f"  Max DD (pips)  : {dd:+.1f}")
    print(f"  Trade Sharpe   : {tsr:+.3f}")


def bars_per_year_for_tf(tf: str) -> float:
    tf = tf.lower().strip()
    n, unit = int(tf[:-1]), tf[-1]
    minutes = {"m": n, "h": n * 60, "d": n * 1440}[unit]
    return 365 * 24 * 60 / minutes


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbol",        default="EURUSD")
    ap.add_argument("--tf",            default="1m")
    ap.add_argument("--start",         default="2024-01-01")
    ap.add_argument("--end",           default="2025-12-31")
    ap.add_argument("--swing-n",       type=int,   default=10,
                    help="bars on each side for swing detection (centered window=2n+1)")
    ap.add_argument("--tp-rr",         type=float, default=10.0)
    ap.add_argument("--max-bars",      type=int,   default=2000,
                    help="forward-sim horizon (timeout if SL/TP not hit). "
                         "1M default 2000 = ~33 hours.")
    ap.add_argument("--session-start", type=int,   default=8,
                    help="UTC inclusive (default 8 = London open)")
    ap.add_argument("--session-end",   type=int,   default=22,
                    help="UTC exclusive (default 22 = NY close)")
    ap.add_argument("--commission",    type=float, default=0.00005,
                    help="per-side commission (0.5 pip default)")
    ap.add_argument("--atr-len",       type=int,   default=14)
    ap.add_argument("--rsi-len",       type=int,   default=14)
    ap.add_argument("--ema-len",       type=int,   default=50)
    ap.add_argument("--slope-n",       type=int,   default=20)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent

    # ── fetch ──────────────────────────────────────────────────────────────
    print(f"Fetching {args.symbol} {args.tf}  {args.start} → {args.end}")
    raw = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  fetched {len(raw):,} bars")
    if raw.empty:
        return

    print("\nComputing features (ATR / RSI / EMA slope)…")
    df = add_features(raw, args.atr_len, args.rsi_len, args.ema_len, args.slope_n)

    print(f"\nDetecting higher lows  (swing_n={args.swing_n}, "
          f"session={args.session_start:02d}-{args.session_end:02d}UTC)…")
    sigs = detect_higher_lows(df, args.swing_n, args.session_start, args.session_end)
    print(f"  signals: {len(sigs)}  "
          f"({len(sigs) / max(len(df),1) * 100:.3f}% of bars)")
    if sigs.empty:
        print("No signals — try a smaller --swing-n or wider session.")
        return

    # ── Phase 1: descriptive analysis ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE 1 — analysis")
    print("=" * 80)
    sh = sigs["swing_height_pips"]
    print(f"  Swing height (new − prev swing low, pips):")
    print(f"    median = {sh.median():.1f}   p25 = {sh.quantile(.25):.1f}   "
          f"p75 = {sh.quantile(.75):.1f}   max = {sh.max():.1f}")

    bb = sigs["bars_between"]
    print(f"  Bars between consecutive swing lows:")
    print(f"    median = {bb.median():.0f}   p25 = {bb.quantile(.25):.0f}   "
          f"p75 = {bb.quantile(.75):.0f}")

    # Stop distance = entry - SL = next bar open - swing_low. Pre-compute via
    # the simulator dry-run so we account for bar-1 entries that gap.
    print(f"\n  Forward direction (no SL/TP — pure close-vs-close from confirm bar):")
    close = df["Close"]
    for h in (10, 30, 60, 240):  # 10 min, 30 min, 1 hr, 4 hr
        idxs = sigs["confirm_idx"].values
        valid = idxs + h < len(df)
        v = idxs[valid]
        bullish = (close.values[v + h] > close.values[v]).mean()
        # Baseline: random bar
        baseline_full = (close.shift(-h) > close).mean()
        print(f"    H={h:>3} bars   bullish={bullish*100:5.1f}%   "
              f"baseline={baseline_full*100:5.1f}%   "
              f"lift={ (bullish - baseline_full)*100:+5.2f} pp")

    # ── Phase 2: full forward simulation ──────────────────────────────────
    commission_pips = args.commission / PIP
    print("\n" + "=" * 80)
    print(f"PHASE 2 — backtest  "
          f"(LONG, SL=swing_low, TP={args.tp_rr}R, "
          f"cost={2*commission_pips:.2f} pips r/t, max_bars={args.max_bars})")
    print("=" * 80)
    sim = simulate_all(df, sigs, args.tp_rr, args.max_bars, commission_pips)
    seq = sequential_take(sim)

    report("ALL signals (independent — RF training set)", sim, len(df), args.tf)
    report("SEQUENTIAL  (one trade at a time)",            seq, len(df), args.tf)

    if not sim.empty:
        be = 1.0 / (1 + args.tp_rr) * 100
        wr = (sim["tp_hit"] == 1).mean() * 100
        print(f"\nBreak-even win rate at 1:{args.tp_rr} R:R = {be:.1f}%")
        print(f"Observed (independent)                  = {wr:.1f}%   "
              f"edge = {wr-be:+.1f} pp")
        # Distribution of outcomes
        oc = sim["outcome"].value_counts()
        print(f"\nOutcome breakdown (independent):")
        for k, v in oc.items():
            print(f"  {k:12s}  n={v:>4d}   ({v/len(sim)*100:5.1f}%)")

    if not seq.empty:
        seq2 = seq.copy()
        seq2["year"] = pd.to_datetime(seq2["signal_time"]).dt.year
        print("\nSequential P&L by year:")
        for y, grp in seq2.groupby("year"):
            n = len(grp)
            wr = (grp["tp_hit"] == 1).mean() * 100
            pnl = grp["pnl_pips_net"].sum()
            print(f"  {y}   n={n:>4}   win%={wr:5.1f}   pnl={pnl:+9.1f} pips")

    # ── save ───────────────────────────────────────────────────────────────
    sig_path = out_dir / f"higher_lows_{args.symbol}_{args.tf}_tp{args.tp_rr}_n{args.swing_n}.csv"
    seq_path = out_dir / f"higher_lows_seq_{args.symbol}_{args.tf}_tp{args.tp_rr}_n{args.swing_n}.csv"
    keep = ["signal_time", "outcome", "tp_hit", "side" if "side" in sim.columns else None,
            "entry_price", "exit_price", "sl_price", "tp_price",
            "stop_pips", "pnl_pips", "pnl_pips_net", "bars_held",
            "swing_low", "prev_swing_low", "swing_height_pips", "bars_between",
            "session_hour", "atr", "rsi", "ema_slope_bps"]
    keep = [c for c in keep if c is not None and c in sim.columns]
    sim[keep].to_csv(sig_path, index=False)
    seq[[c for c in keep if c in seq.columns]].to_csv(seq_path, index=False)
    print(f"\n→ saved {sig_path}  ({len(sim)} signals — for RF training)")
    print(f"→ saved {seq_path}  ({len(seq)} taken trades — sequential backtest)")


if __name__ == "__main__":
    main()
