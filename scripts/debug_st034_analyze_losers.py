"""
ST34 loser analysis.

For every detected trade, compute a feature vector (wick/body sizes, ATR,
RSI, EMA distance, volume, time-of-day, ...). Split winners vs losers and
report which features statistically separate them, so we can pick filter
candidates.

Usage:
  python scripts/debug_st034_analyze_losers.py --tf 5m --start 2026-01-01 --end 2026-05-15
  python scripts/debug_st034_analyze_losers.py --signal s1
  python scripts/debug_st034_analyze_losers.py --signal s2 --pairs EURUSD GBPUSD
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.st034_backtest_2026 import get_mt5_data
from scripts.debug_st034_visualize import detect_and_simulate


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def add_indicators(df: pd.DataFrame) -> dict:
    """Precompute indicator arrays once per pair."""
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float) if "Volume" in df.columns else np.zeros(len(df))

    # Higher-timeframe candle context (no-lookahead: uses only bar i's bucket).
    idx = pd.DatetimeIndex(df.index)
    o_series = pd.Series(o, index=idx)
    # First open of the calendar DAY containing bar i, broadcast to every bar of that day
    day_open = o_series.groupby(idx.normalize()).transform("first").values
    # First open of the hour containing bar i
    hour_floor = idx.floor("H")
    hour_open = o_series.groupby(hour_floor).transform("first").values
    # Previous completed daily candle's color (yesterday's close vs yesterday's open).
    daily = pd.DataFrame({"O": o, "C": c}, index=idx).resample("D").agg({"O": "first", "C": "last"}).dropna()
    daily["red"] = (daily["C"] < daily["O"]).astype(int)
    prev_day_red_map = daily["red"].shift(1)
    prev_day_red = pd.Series(idx.normalize()).map(prev_day_red_map).values.astype(float)

    atr14 = ta.atr(pd.Series(h), pd.Series(l), pd.Series(c), length=14).values
    rsi14 = ta.rsi(pd.Series(c), length=14).values
    ema21 = ta.ema(pd.Series(c), length=21).values
    ema50 = ta.ema(pd.Series(c), length=50).values
    ema200 = ta.ema(pd.Series(c), length=200).values

    # EMA slopes — normalized by ATR. Lookback = half the EMA's period,
    # so each slope captures meaningful change at that EMA's timescale.
    def _slope(ema: np.ndarray, lookback: int) -> np.ndarray:
        prev = np.concatenate([np.full(lookback, np.nan), ema[:-lookback]])
        return (ema - prev) / lookback
    slope_ema21 = _slope(ema21, 10)
    slope_ema50 = _slope(ema50, 25)
    slope_ema200 = _slope(ema200, 100)
    # Bollinger Bands (20, 2)
    bb = ta.bbands(pd.Series(c), length=20, std=2.0)
    bb_upper = bb["BBU_20_2.0"].values if bb is not None and "BBU_20_2.0" in bb.columns else np.full(len(c), np.nan)
    bb_lower = bb["BBL_20_2.0"].values if bb is not None and "BBL_20_2.0" in bb.columns else np.full(len(c), np.nan)
    # ADX (trend strength)
    adx_df = ta.adx(pd.Series(h), pd.Series(l), pd.Series(c), length=14)
    adx14 = adx_df["ADX_14"].values if adx_df is not None and "ADX_14" in adx_df.columns else np.full(len(c), np.nan)

    # Rolling 20-bar avg volume
    vol_avg20 = pd.Series(v).rolling(20).mean().values

    return dict(o=o, h=h, l=l, c=c, v=v,
                atr=atr14, rsi=rsi14,
                ema21=ema21, ema50=ema50, ema200=ema200,
                slope_ema21=slope_ema21, slope_ema50=slope_ema50,
                slope_ema200=slope_ema200,
                bb_upper=bb_upper, bb_lower=bb_lower,
                adx=adx14, vol_avg20=vol_avg20,
                day_open=day_open, hour_open=hour_open,
                prev_day_red=prev_day_red)


def trade_features(trade: dict, ind: dict, df: pd.DataFrame) -> dict:
    """Compute feature vector for a single trade (evaluated AT signal bar)."""
    i = trade["signal_idx"]
    o = ind["o"]; h = ind["h"]; l = ind["l"]; c = ind["c"]; v = ind["v"]
    atr_now = ind["atr"][i]

    # Pattern-bar geometry
    body_curr = abs(c[i] - o[i])
    wick_curr_top = h[i] - max(o[i], c[i])
    wick_curr_bot = min(o[i], c[i]) - l[i]
    range_curr = h[i] - l[i]

    body_prev = abs(c[i - 1] - o[i - 1])
    wick_prev_top = h[i - 1] - max(o[i - 1], c[i - 1])
    range_prev = h[i - 1] - l[i - 1]

    # Normalize by ATR so features are comparable across regimes/pairs
    def _r(x):
        return float(x) / atr_now if atr_now and not np.isnan(atr_now) else np.nan

    feats = {
        "sig": trade["sig_type"],
        "won": trade["won"],
        "exit_reason": trade["exit_reason"],
        "pnl_pips": (trade["pnl"]) * 10000,
        "hold_bars": trade["exit_idx"] - trade["fill_idx"],
        "atr_pips": atr_now * 10000 if atr_now and not np.isnan(atr_now) else np.nan,

        # --- pattern geometry (red bar = confirm bar i) ---
        "red_body_atr":      _r(body_curr),
        "red_top_wick_atr":  _r(wick_curr_top),
        "red_bot_wick_atr":  _r(wick_curr_bot),
        "red_range_atr":     _r(range_curr),
        "red_body_to_wick":  float(body_curr) / wick_curr_top if wick_curr_top > 0 else np.nan,
        "red_body_to_range": float(body_curr) / range_curr if range_curr > 0 else np.nan,

        # --- prior bar geometry ---
        "prev_body_atr":     _r(body_prev),
        "prev_top_wick_atr": _r(wick_prev_top),
        "prev_range_atr":    _r(range_prev),

        # --- combined wicks ---
        "max_top_wick_atr":  _r(max(wick_curr_top, wick_prev_top)),
        "sum_top_wick_atr":  _r(wick_curr_top + wick_prev_top),

        # --- indicators at signal bar ---
        "rsi": float(ind["rsi"][i]) if not np.isnan(ind["rsi"][i]) else np.nan,
        "adx": float(ind["adx"][i]) if not np.isnan(ind["adx"][i]) else np.nan,
        "dist_ema21_atr":  _r(c[i] - ind["ema21"][i])  if not np.isnan(ind["ema21"][i])  else np.nan,
        "dist_ema50_atr":  _r(c[i] - ind["ema50"][i])  if not np.isnan(ind["ema50"][i])  else np.nan,
        "dist_ema200_atr": _r(c[i] - ind["ema200"][i]) if not np.isnan(ind["ema200"][i]) else np.nan,
        "ema21_gt_ema50":  int(ind["ema21"][i] > ind["ema50"][i])  if not np.isnan(ind["ema21"][i])  else np.nan,
        "ema50_gt_ema200": int(ind["ema50"][i] > ind["ema200"][i]) if not np.isnan(ind["ema50"][i]) else np.nan,

        # --- EMA slopes (per-bar change, normalized by ATR) ---
        "slope_ema21_atr":  _r(ind["slope_ema21"][i])  if not np.isnan(ind["slope_ema21"][i])  else np.nan,
        "slope_ema50_atr":  _r(ind["slope_ema50"][i])  if not np.isnan(ind["slope_ema50"][i])  else np.nan,
        "slope_ema200_atr": _r(ind["slope_ema200"][i]) if not np.isnan(ind["slope_ema200"][i]) else np.nan,
        "ema21_falling":  int(ind["slope_ema21"][i]  < 0) if not np.isnan(ind["slope_ema21"][i])  else np.nan,
        "ema50_falling":  int(ind["slope_ema50"][i]  < 0) if not np.isnan(ind["slope_ema50"][i])  else np.nan,
        "ema200_falling": int(ind["slope_ema200"][i] < 0) if not np.isnan(ind["slope_ema200"][i]) else np.nan,
        "bb_pos": (c[i] - ind["bb_lower"][i]) / (ind["bb_upper"][i] - ind["bb_lower"][i])
                  if not np.isnan(ind["bb_upper"][i]) and ind["bb_upper"][i] > ind["bb_lower"][i] else np.nan,
        "vol_ratio": float(v[i] / ind["vol_avg20"][i]) if ind["vol_avg20"][i] > 0 else np.nan,

        # --- session ---
        "hour": int(df.index[i].hour),
        "dow":  int(df.index[i].dayofweek),

        # --- higher-timeframe trend alignment ---
        "daily_red":    int(c[i] < ind["day_open"][i])  if not np.isnan(ind["day_open"][i])  else np.nan,
        "hourly_red":   int(c[i] < ind["hour_open"][i]) if not np.isnan(ind["hour_open"][i]) else np.nan,
        "prev_day_red": float(ind["prev_day_red"][i]) if not np.isnan(ind["prev_day_red"][i]) else np.nan,
        "daily_move_atr":  float((c[i] - ind["day_open"][i])  / atr_now) if atr_now and not np.isnan(atr_now) else np.nan,
        "hourly_move_atr": float((c[i] - ind["hour_open"][i]) / atr_now) if atr_now and not np.isnan(atr_now) else np.nan,

        # --- S2-specific ---
        "s2_wick_count": float(trade["meta"]["wick_count"]) if trade["sig_type"] == "S2" else np.nan,
    }
    return feats


# ============================================================================
# MAIN
# ============================================================================

NUMERIC_FEATURES = [
    "red_body_atr", "red_top_wick_atr", "red_bot_wick_atr", "red_range_atr",
    "red_body_to_wick", "red_body_to_range",
    "prev_body_atr", "prev_top_wick_atr", "prev_range_atr",
    "max_top_wick_atr", "sum_top_wick_atr",
    "rsi", "adx",
    "dist_ema21_atr", "dist_ema50_atr", "dist_ema200_atr",
    "slope_ema21_atr", "slope_ema50_atr", "slope_ema200_atr",
    "bb_pos", "vol_ratio",
    "s2_wick_count",
    "daily_move_atr", "hourly_move_atr",
    "atr_pips", "hold_bars",
]

BINARY_FEATURES = ["ema21_gt_ema50", "ema50_gt_ema200",
                   "ema21_falling", "ema50_falling", "ema200_falling",
                   "daily_red", "hourly_red", "prev_day_red"]


def _safe_med(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.median()) if len(s) else float("nan")


def _safe_mean(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean()) if len(s) else float("nan")


def compare(df_trades: pd.DataFrame) -> None:
    wn = df_trades[df_trades["won"]]
    ls = df_trades[~df_trades["won"]]

    print(f"\n{'Feature':<22} {'win_med':>10} {'lose_med':>10} {'win-lose':>10}  "
          f"{'win_mean':>10} {'lose_mean':>10}  {'separation':>12}")
    print("-" * 100)

    rows = []
    for feat in NUMERIC_FEATURES:
        w_med = _safe_med(wn[feat])
        l_med = _safe_med(ls[feat])
        w_mean = _safe_mean(wn[feat])
        l_mean = _safe_mean(ls[feat])
        # Crude separation score: |mean_w - mean_l| / pooled_std
        w_vals = wn[feat].dropna().values
        l_vals = ls[feat].dropna().values
        if len(w_vals) > 1 and len(l_vals) > 1:
            pooled = np.sqrt((w_vals.std() ** 2 + l_vals.std() ** 2) / 2.0)
            sep = abs(w_mean - l_mean) / pooled if pooled > 0 else 0.0
        else:
            sep = float("nan")
        rows.append((feat, w_med, l_med, w_med - l_med, w_mean, l_mean, sep))

    # Sort by separation score (most discriminative first)
    rows.sort(key=lambda r: -(r[6] if not np.isnan(r[6]) else -1))
    for r in rows:
        feat, w_med, l_med, diff, w_mean, l_mean, sep = r
        print(f"{feat:<22} {w_med:>10.3f} {l_med:>10.3f} {diff:>+10.3f}  "
              f"{w_mean:>10.3f} {l_mean:>10.3f}  {sep:>12.3f}")

    print(f"\n{'Binary filter':<22} {'value':>6} {'n_kept':>7} {'retain':>7} "
          f"{'win%':>7} {'lift':>7} {'sum_pips':>10}")
    print("-" * 80)
    baseline = df_trades["won"].mean() * 100
    total = len(df_trades)
    for feat in BINARY_FEATURES:
        s = df_trades[[feat, "won", "pnl_pips"]].dropna(subset=[feat])
        for val in (0, 1):
            kept = s[s[feat] == val]
            if len(kept) == 0:
                continue
            wr = kept["won"].mean() * 100
            lift = wr - baseline
            print(f"{feat:<22} {val:>6} {len(kept):>7} "
                  f"{len(kept) / total * 100:>6.1f}% {wr:>6.1f}% "
                  f"{lift:>+6.1f} {kept['pnl_pips'].sum():>+10.1f}")

    # Hour-of-day breakdown
    print(f"\n{'Hour':<6} {'n':>5} {'win%':>7} {'avg_pips':>10}")
    print("-" * 35)
    for hour, sub in df_trades.groupby("hour"):
        n = len(sub)
        wr = 100 * sub["won"].mean()
        ap = sub["pnl_pips"].mean()
        print(f"{hour:<6} {n:>5} {wr:>6.1f}% {ap:>+9.2f}")


def filter_candidates(df_trades: pd.DataFrame) -> None:
    """Sweep single-feature thresholds and find the ones that bump win rate
    while preserving enough trades."""
    print("\n" + "=" * 100)
    print("Single-feature threshold sweep (keep ≥ 40% of trades, bump win%):")
    print("=" * 100)
    baseline = df_trades["won"].mean() * 100
    total = len(df_trades)
    print(f"Baseline: n={total}  win%={baseline:.1f}\n")

    candidates = []
    for feat in NUMERIC_FEATURES:
        s = df_trades[[feat, "won", "pnl_pips"]].dropna(subset=[feat])
        if len(s) < 20:
            continue
        # Try a grid of percentile thresholds, both ≤ and ≥
        for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            thr = np.percentile(s[feat], pct)
            for op_name, mask in (("<=", s[feat] <= thr), (">=", s[feat] >= thr)):
                kept = s[mask]
                if len(kept) / total < 0.40:
                    continue
                wr = kept["won"].mean() * 100
                lift = wr - baseline
                if lift >= 5.0:   # ≥ +5 pp improvement
                    candidates.append((feat, op_name, thr, len(kept),
                                       len(kept) / total, wr, lift,
                                       kept["pnl_pips"].sum()))

    if not candidates:
        print("(none — no single-feature filter delivered ≥ +5 pp win-rate at ≥40% retention)")
        return

    candidates.sort(key=lambda r: -r[6])  # by lift
    print(f"{'feat':<22} {'op':>4} {'thr':>10} {'n_kept':>7} "
          f"{'retain':>7} {'win%':>7} {'lift':>7} {'sum_pips':>10}")
    print("-" * 100)
    for c in candidates[:25]:
        feat, op, thr, n, ret, wr, lift, pips = c
        print(f"{feat:<22} {op:>4} {thr:>10.3f} {n:>7} "
              f"{ret * 100:>6.1f}% {wr:>6.1f}% {lift:>+6.1f} {pips:>+10.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+",
                    default=["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"])
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-05-15")
    ap.add_argument("--signal", choices=["s1", "s2", "both"], default="both")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--s2-only", action="store_true",
                    help="Run detection with S1 disabled (mirrors production)")
    ap.add_argument("--sl-mode", choices=["old", "new"], default="new",
                    help="'old' = wick-top SL (S1) / level+ATR SL (S2); 'new' = tight SLs")
    ap.add_argument("--csv", default="st034_trade_features.csv")
    args = ap.parse_args()

    all_rows: list[dict] = []
    for pair in args.pairs:
        print(f"\n=== {pair} {args.tf} ===")
        df = get_mt5_data(pair, args.tf, args.start, args.end)
        if df.empty:
            print("  no data")
            continue
        trades = detect_and_simulate(df, rr_ratio=args.rr,
                                     s1_enable=not args.s2_only,
                                     sl_mode=args.sl_mode)
        if args.signal != "both":
            trades = [t for t in trades if t["sig_type"].lower() == args.signal]
        if not trades:
            print("  no trades")
            continue
        ind = add_indicators(df)
        for tr in trades:
            row = trade_features(tr, ind, df)
            row["pair"] = pair
            row["entry_t"] = tr["entry_t"]
            all_rows.append(row)
        wins = sum(1 for t in trades if t["won"])
        print(f"  trades={len(trades)}  wins={wins}  losers={len(trades) - wins}  "
              f"win%={100 * wins / len(trades):.1f}")

    if not all_rows:
        print("\nNo trades collected.")
        return

    df_all = pd.DataFrame(all_rows)
    out_csv = Path(args.csv)
    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df_all)} trade rows → {out_csv}")

    print("\n" + "=" * 100)
    print(f"FEATURE COMPARISON — winners vs losers   "
          f"(signal={args.signal}, n_total={len(df_all)})")
    print("=" * 100)

    if args.signal == "both":
        for sig in ("S1", "S2"):
            sub = df_all[df_all["sig"] == sig]
            if len(sub) < 5:
                continue
            print(f"\n--- {sig}  (n={len(sub)},  win%={100 * sub['won'].mean():.1f}) ---")
            compare(sub)
            filter_candidates(sub)
    else:
        compare(df_all)
        filter_candidates(df_all)


if __name__ == "__main__":
    main()
