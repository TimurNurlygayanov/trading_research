"""
Shooting Star (a.k.a. Falling Star) reversal — 1H EURUSD analysis + backtest.

Pattern definition
  body          = |Close − Open|
  upper_shadow  = High − max(Open, Close)
  lower_shadow  = min(Open, Close) − Low

  Conditions (all must hold):
    upper_shadow ≥ shadow_to_body × body     (default 2.0)
    lower_shadow ≤ wick_to_body × body       (default 0.5)
    body         ≥ min_body × PIP             (default 3 pips — drop dojis)
    Close > EMA(Close, ema_filter)            (uptrend prefilter, default 20;
                                               set ema_filter=0 to disable)

Strategy
  When bar t is a confirmed shooting star:
    SHORT at bar t+1 open
    SL = High of bar t
    TP = entry − tp_rr × (SL − entry)         (default tp_rr = 4)

What this script outputs
  Phase 1 — pattern analysis
    distribution of body / upper-shadow / SL distance
    forward returns at H = 5, 10, 20 bars (no SL/TP, just close-vs-close)
    "high breached in next N bars" rates → tells you raw SL hit probability

  Phase 2 — strategy backtest
    independent forward simulation for each star (RF training set later)
    sequential one-trade-at-a-time backtest (realistic stats)
    saves per-trade CSV with features and outcome label

Usage
  python -m scripts.shooting_star_backtest
  python -m scripts.shooting_star_backtest --tp-rr 3.0 --ema-filter 50
  python -m scripts.shooting_star_backtest --no-uptrend-filter   # disable EMA prefilter
  python -m scripts.shooting_star_backtest --start 2018-01-01
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

PIP = 0.0001  # EURUSD


# ── pattern detection ───────────────────────────────────────────────────────
def detect_stars(df: pd.DataFrame,
                 shadow_to_body: float = 2.0,
                 wick_to_body:   float = 0.5,
                 min_body_pips:  float = 3.0,
                 ema_filter:     int   = 20) -> pd.DataFrame:
    """Return df with extra columns: body, upper, lower, is_star."""
    out = df.copy()
    out["body"]  = (out["Close"] - out["Open"]).abs()
    out["upper"] = out["High"] - out[["Open", "Close"]].max(axis=1)
    out["lower"] = out[["Open", "Close"]].min(axis=1) - out["Low"]

    body_ok  = out["body"]  >= min_body_pips * PIP
    upper_ok = out["upper"] >= shadow_to_body * out["body"]
    lower_ok = out["lower"] <= wick_to_body  * out["body"]

    if ema_filter and ema_filter > 0:
        ema_c = ta.ema(out["Close"], length=ema_filter)
        out["ema_filter"] = ema_c
        trend_ok = out["Close"] > ema_c
    else:
        out["ema_filter"] = np.nan
        trend_ok = pd.Series(True, index=out.index)

    out["is_star"] = body_ok & upper_ok & lower_ok & trend_ok
    return out


# ── Phase 1: descriptive analysis ───────────────────────────────────────────
def phase1_analysis(df: pd.DataFrame, horizons=(5, 10, 20)):
    stars = df[df["is_star"]].copy()
    print("\n" + "=" * 80)
    print("PHASE 1 — pattern analysis")
    print("=" * 80)
    print(f"  bars total : {len(df):,}")
    print(f"  stars      : {len(stars):,}  ({len(stars)/max(len(df),1)*100:.2f}%)")

    if stars.empty:
        return stars

    print(f"\n  Body (pips)        median={stars['body'].median()/PIP:.1f}   "
          f"p25={stars['body'].quantile(.25)/PIP:.1f}   "
          f"p75={stars['body'].quantile(.75)/PIP:.1f}")
    print(f"  Upper shadow (pips) median={stars['upper'].median()/PIP:.1f}   "
          f"p25={stars['upper'].quantile(.25)/PIP:.1f}   "
          f"p75={stars['upper'].quantile(.75)/PIP:.1f}")
    print(f"  Lower shadow (pips) median={stars['lower'].median()/PIP:.1f}   "
          f"p75={stars['lower'].quantile(.75)/PIP:.1f}")
    sl_dist = stars["High"] - stars["Close"]
    print(f"  SL distance (high-close, pips) median={sl_dist.median()/PIP:.1f}   "
          f"p25={sl_dist.quantile(.25)/PIP:.1f}   "
          f"p75={sl_dist.quantile(.75)/PIP:.1f}")

    # Forward close-vs-close direction (no SL/TP)
    print("\n  Forward direction (close < star.Close — bearish hit rate, "
          "ignoring SL/TP):")
    close = df["Close"]
    base_idx = stars.index
    for h in horizons:
        # forward close lookup
        future = close.shift(-h)
        bearish = (future.loc[base_idx] < close.loc[base_idx]).mean()
        baseline = (close.shift(-h) < close).mean()
        print(f"    H={h:>2} bars   bearish={bearish*100:5.1f}%   "
              f"baseline={baseline*100:5.1f}%   "
              f"lift={ (bearish - baseline)*100:+5.2f} pp")

    # Did the High break in next N bars (= SL would have been hit)
    print("\n  High broken (= SL hit if we shorted) within next N bars:")
    high_arr = df["High"].values
    star_pos = np.where(df["is_star"].values)[0]
    star_high = df["High"].values[star_pos]
    n_total = len(star_pos)
    for h in (5, 10, 20, 50, 100):
        breaches = 0
        for k, p in enumerate(star_pos):
            end = min(p + h + 1, len(df))
            if (high_arr[p+1:end] > star_high[k]).any():
                breaches += 1
        print(f"    within {h:>3} bars   high_broken={breaches/n_total*100:5.1f}%")

    return stars


# ── Phase 2: forward simulation ─────────────────────────────────────────────
def simulate_signal(df: pd.DataFrame, sig_idx: int, sl_price: float,
                    tp_rr: float, max_bars: int) -> dict:
    """
    Short at sig_idx+1 open. SL = sl_price (above entry). TP = entry − tp_rr × (SL − entry).
    Walk forward; conservative on overlap (SL wins).
    """
    n = len(df)
    if sig_idx + 1 >= n:
        return dict(outcome="no_entry", entry_price=np.nan, exit_price=np.nan,
                    sl_price=sl_price, tp_price=np.nan,
                    bars_held=0, pnl_pips=0.0, exit_idx=-1, tp_hit=np.nan,
                    stop_pips=0.0)

    entry = float(df["Open"].iloc[sig_idx + 1])
    if entry >= sl_price:
        # Gap up above the star high — invalid setup, would be stopped instantly
        return dict(outcome="gap_invalid", entry_price=entry,
                    exit_price=sl_price, sl_price=sl_price, tp_price=np.nan,
                    bars_held=0, pnl_pips=(entry - sl_price) / PIP,
                    exit_idx=sig_idx + 1, tp_hit=0,
                    stop_pips=(sl_price - entry) / PIP)

    stop_dist = sl_price - entry
    tp_price = entry - tp_rr * stop_dist

    end = min(sig_idx + 1 + max_bars, n)
    high = df["High"].values
    low  = df["Low"].values

    for j in range(sig_idx + 1, end):
        h, l = high[j], low[j]
        hit_sl = h >= sl_price
        hit_tp = l <= tp_price

        if hit_sl:  # conservative on overlap
            return dict(outcome="sl", entry_price=entry, exit_price=sl_price,
                        sl_price=sl_price, tp_price=tp_price,
                        bars_held=j - sig_idx, pnl_pips=(entry - sl_price) / PIP,
                        exit_idx=j, tp_hit=0, stop_pips=stop_dist / PIP)
        if hit_tp:
            return dict(outcome="tp", entry_price=entry, exit_price=tp_price,
                        sl_price=sl_price, tp_price=tp_price,
                        bars_held=j - sig_idx, pnl_pips=(entry - tp_price) / PIP,
                        exit_idx=j, tp_hit=1, stop_pips=stop_dist / PIP)

    last_close = float(df["Close"].iloc[end - 1])
    return dict(outcome="timeout", entry_price=entry, exit_price=last_close,
                sl_price=sl_price, tp_price=tp_price,
                bars_held=end - 1 - sig_idx, pnl_pips=(entry - last_close) / PIP,
                exit_idx=end - 1, tp_hit=np.nan, stop_pips=stop_dist / PIP)


def simulate_all(df: pd.DataFrame, stars: pd.DataFrame,
                 tp_rr: float, max_bars: int, commission_pips: float) -> pd.DataFrame:
    if stars.empty:
        return pd.DataFrame()
    rows = []
    bar_index_map = {ts: i for i, ts in enumerate(df.index)}
    for ts, row in stars.iterrows():
        i = bar_index_map.get(ts)
        if i is None:
            continue
        sl = float(row["High"])
        sim = simulate_signal(df, i, sl, tp_rr, max_bars)
        sim["pnl_pips_net"]   = sim["pnl_pips"] - 2 * commission_pips
        sim["signal_time"]    = ts
        sim["star_high"]      = float(row["High"])
        sim["star_close"]     = float(row["Close"])
        sim["body_pips"]      = float(row["body"]) / PIP
        sim["upper_pips"]     = float(row["upper"]) / PIP
        sim["lower_pips"]     = float(row["lower"]) / PIP
        sim["bar_idx"]        = i
        rows.append(sim)
    return pd.DataFrame(rows)


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


def report(name: str, trades: pd.DataFrame, df_len: int, tf: str):
    if trades.empty:
        print(f"\n── {name}: 0 trades ──")
        return
    n = len(trades)
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
    ap.add_argument("--symbol",         default="EURUSD")
    ap.add_argument("--tf",             default="1h")
    ap.add_argument("--start",          default="2018-01-01")
    ap.add_argument("--end",            default="2025-12-31")
    ap.add_argument("--shadow-to-body", type=float, default=2.0,
                    help="upper shadow must be ≥ this × body")
    ap.add_argument("--wick-to-body",   type=float, default=0.5,
                    help="lower wick must be ≤ this × body")
    ap.add_argument("--min-body-pips",  type=float, default=3.0)
    ap.add_argument("--ema-filter",     type=int,   default=20,
                    help="prior uptrend filter: Close > EMA(Close, this). "
                         "Set 0 to disable.")
    ap.add_argument("--no-uptrend-filter", action="store_true",
                    help="shortcut for --ema-filter 0")
    ap.add_argument("--tp-rr",          type=float, default=4.0,
                    help="reward:risk multiple (default 1:4)")
    ap.add_argument("--max-bars",       type=int,   default=200)
    ap.add_argument("--commission",     type=float, default=0.00005,
                    help="per-side commission as price fraction (0.5 pip default)")
    args = ap.parse_args()

    if args.no_uptrend_filter:
        args.ema_filter = 0

    out_dir = Path(__file__).resolve().parent

    print(f"Fetching {args.symbol} {args.tf}  {args.start} → {args.end}")
    raw = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  fetched {len(raw):,} bars")
    if raw.empty:
        return

    print("\nDetecting shooting stars…")
    print(f"  shadow≥{args.shadow_to_body}×body, wick≤{args.wick_to_body}×body, "
          f"body≥{args.min_body_pips} pips, EMA filter = "
          f"{'OFF' if args.ema_filter == 0 else f'EMA({args.ema_filter})'}")

    df = detect_stars(raw,
                      shadow_to_body=args.shadow_to_body,
                      wick_to_body=args.wick_to_body,
                      min_body_pips=args.min_body_pips,
                      ema_filter=args.ema_filter)
    stars = phase1_analysis(df)
    if stars.empty:
        print("\nNo stars detected — try relaxing the criteria.")
        return

    # ── Phase 2 ──────────────────────────────────────────────────────────
    commission_pips = args.commission / PIP
    print("\n" + "=" * 80)
    print(f"PHASE 2 — strategy backtest  "
          f"(short, SL=high, TP={args.tp_rr}R, "
          f"cost={2*commission_pips:.2f} pips round-trip, max_bars={args.max_bars})")
    print("=" * 80)

    sim = simulate_all(df, stars, args.tp_rr, args.max_bars, commission_pips)
    seq = sequential_take(sim)

    report("ALL signals (independent — RF dataset)", sim, len(df), args.tf)
    report("SEQUENTIAL  (one trade at a time)",      seq, len(df), args.tf)

    # ── breakeven check vs theoretical 1:R ──────────────────────────────
    if not sim.empty:
        # break-even hit rate at 1:R reward-to-risk: 1/(R+1)
        be_winrate = 1.0 / (1 + args.tp_rr) * 100
        win_rate = (sim["tp_hit"] == 1).mean() * 100
        edge = win_rate - be_winrate
        print(f"\nBreak-even win rate at 1:{args.tp_rr} R:R = {be_winrate:.1f}%")
        print(f"Observed win rate                      = {win_rate:.1f}%")
        print(f"Edge over break-even                   = {edge:+.1f} pp")

    # ── per-year breakdown ──────────────────────────────────────────────
    if not seq.empty:
        seq2 = seq.copy()
        seq2["year"] = pd.to_datetime(seq2["signal_time"]).dt.year
        print("\nSequential P&L by year:")
        for y, grp in seq2.groupby("year"):
            n = len(grp)
            wr = (grp["tp_hit"] == 1).mean() * 100
            pnl = grp["pnl_pips_net"].sum()
            print(f"  {y}   n={n:>3}   win%={wr:5.1f}   pnl={pnl:+8.1f} pips")

    # ── save ────────────────────────────────────────────────────────────
    suffix = f"_tp{args.tp_rr}"
    sig_path = out_dir / f"shooting_stars_{args.symbol}_{args.tf}{suffix}.csv"
    seq_path = out_dir / f"shooting_stars_seq_{args.symbol}_{args.tf}{suffix}.csv"
    sim.drop(columns=["bar_idx"], errors="ignore").to_csv(sig_path, index=False)
    seq.drop(columns=["bar_idx"], errors="ignore").to_csv(seq_path, index=False)
    print(f"\n→ saved {sig_path}  ({len(sim)} stars — for later analysis / RF training)")
    print(f"→ saved {seq_path}  ({len(seq)} taken trades — sequential backtest)")


if __name__ == "__main__":
    main()
