"""
EMA touch research pipeline.

For each of EMA 9, 13, 21, 50: detect every bar where price enters the EMA
zone, classify as support touch (from above) or resistance touch (from below),
and measure what happens next.

Features per touch:
  ema             -- which EMA (9/13/21/50)
  approach_side   -- support | resistance
  ema_slope_atr   -- EMA slope over slope_n bars, in ATR/bar units
                     (positive = rising, negative = falling)
  slope_aligned   -- 1 if slope direction agrees with approach
                     (rising EMA + support touch, or falling EMA + resistance)
  ema_alignment   -- 1 if EMA9>EMA13>EMA21>EMA50 (full bullish stack)
  next_ema_dist   -- distance to nearest EMA on the bounce side, in ATR
                     (small = EMAs are clustered = stronger zone)
  candle_range_atr, wick_ratio, body_pct, close_inside
  cy_hour, dow

Outcomes (forward-looking, NOT causal):
  max_bounce_atr, max_against_atr
  did_bounce_1atr, did_bounce_2atr, did_break

Usage:
  python -m scripts.ema_touch_research
  python -m scripts.ema_touch_research --tf 1h --start 2022-01-01
  python -m scripts.ema_touch_research --fwd-bars 20 --zone-atr 0.15
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
EMA_PERIODS = [9, 13, 21, 50]


# ── helpers ------------------------------------------------------------------
def _cy_hours(df: pd.DataFrame) -> np.ndarray:
    idx = df.index
    try:
        idx_utc = idx.tz_localize("UTC")
    except TypeError:
        idx_utc = idx.tz_convert("UTC")
    return idx_utc.tz_convert("Asia/Nicosia").hour.values


def add_features(df: pd.DataFrame, atr_len: int = 14,
                 slope_n: int = 5) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = ta.atr(out["High"], out["Low"], out["Close"], length=atr_len)
    for p in EMA_PERIODS:
        out[f"ema{p}"] = ta.ema(out["Close"], length=p)
        # slope in ATR/bar: (ema[t] - ema[t-n]) / (atr * n)
        out[f"ema{p}_slope"] = (
            (out[f"ema{p}"] - out[f"ema{p}"].shift(slope_n))
            / (out["atr"] * slope_n)
        )
    return out


# ── outcome measurement ------------------------------------------------------
def _measure_outcome(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     n: int, i: int, side: str,
                     zone_bot: float, zone_top: float,
                     atr: float, fwd_bars: int) -> dict:
    end = min(i + fwd_bars + 1, n)
    if i + 1 >= n:
        return dict(max_bounce_atr=np.nan, max_against_atr=np.nan,
                    did_bounce_1atr=False, did_bounce_2atr=False,
                    did_break=False)
    fh = high[i + 1:end]
    fl = low[i + 1:end]
    fc = close[i + 1:end]
    ref = close[i]

    if side == "support":
        bounce_moves  = fh - ref
        against_moves = ref - fl
        broke = bool(np.any(fc < zone_bot))
    else:
        bounce_moves  = ref - fl
        against_moves = fh - ref
        broke = bool(np.any(fc > zone_top))

    mb = float(np.max(bounce_moves))  if bounce_moves.size  else 0.0
    ma = float(np.max(against_moves)) if against_moves.size else 0.0

    return dict(
        max_bounce_atr   = mb / atr if atr > 0 else np.nan,
        max_against_atr  = ma / atr if atr > 0 else np.nan,
        did_bounce_1atr  = mb >= 1.0 * atr,
        did_bounce_2atr  = mb >= 2.0 * atr,
        did_break        = broke,
    )


# ── main scan ----------------------------------------------------------------
def run_research(df: pd.DataFrame, cy_hours_arr: np.ndarray,
                 zone_atr: float, fwd_bars: int) -> pd.DataFrame:

    open_  = df["Open"].values
    high   = df["High"].values
    low    = df["Low"].values
    close  = df["Close"].values
    atr_v  = df["atr"].values
    dow    = df.index.dayofweek.values
    ts     = df.index
    n      = len(df)

    ema_arr   = {p: df[f"ema{p}"].values     for p in EMA_PERIODS}
    slope_arr = {p: df[f"ema{p}_slope"].values for p in EMA_PERIODS}

    # prev_inside[p] = set of bar indices where price was inside EMA p zone last bar
    prev_inside = {p: False for p in EMA_PERIODS}

    rows: list[dict] = []

    for i in range(1, n - fwd_bars):
        atr = atr_v[i]
        if np.isnan(atr) or atr <= 0:
            prev_inside = {p: False for p in EMA_PERIODS}
            continue

        # Check all EMA values valid
        if any(np.isnan(ema_arr[p][i]) for p in EMA_PERIODS):
            prev_inside = {p: False for p in EMA_PERIODS}
            continue

        half     = zone_atr * atr
        c        = close[i]
        e9, e13  = ema_arr[9][i],  ema_arr[13][i]
        e21, e50 = ema_arr[21][i], ema_arr[50][i]

        # Bullish alignment: EMA9 > EMA13 > EMA21 > EMA50
        aligned_bull = int(e9 > e13 > e21 > e50)
        aligned_bear = int(e9 < e13 < e21 < e50)

        for p in EMA_PERIODS:
            ema_val  = ema_arr[p][i]
            zone_top = ema_val + half
            zone_bot = ema_val - half

            bar_inside = (low[i] <= zone_top) and (high[i] >= zone_bot)

            if not bar_inside:
                prev_inside[p] = False
                continue

            # Touch = entering from outside
            if prev_inside[p]:
                prev_inside[p] = True
                continue
            prev_inside[p] = True

            prev_c = close[i - 1]
            if prev_c > zone_top:
                side = "support"
            elif prev_c < zone_bot:
                side = "resistance"
            else:
                continue   # was already near EMA

            # ── slope features ──────────────────────────────────────────────
            slope       = slope_arr[p][i]
            # slope_aligned: rising EMA supports a support touch, falling supports a resistance touch
            slope_aligned = int(
                (side == "support"   and slope > 0) or
                (side == "resistance" and slope < 0)
            )

            # ── next EMA distance (bounce side) ──────────────────────────────
            # For support touch (bounce = up): nearest EMA ABOVE ema_val
            # For resistance touch (bounce = down): nearest EMA BELOW ema_val
            other_emas = [ema_arr[q][i] for q in EMA_PERIODS if q != p]
            if side == "support":
                above = [e for e in other_emas if e > ema_val]
                next_ema_dist = float(min(above) - ema_val) / atr if above else np.nan
            else:
                below = [e for e in other_emas if e < ema_val]
                next_ema_dist = float(ema_val - max(below)) / atr if below else np.nan

            # ── candle features ──────────────────────────────────────────────
            candle_range = high[i] - low[i]
            range_atr    = candle_range / atr if atr > 0 else np.nan
            if candle_range > 0:
                body_pct = abs(close[i] - open_[i]) / candle_range
                if side == "support":
                    wick = min(open_[i], close[i]) - low[i]
                else:
                    wick = high[i] - max(open_[i], close[i])
                wick_ratio = max(wick, 0) / candle_range
            else:
                body_pct = wick_ratio = np.nan

            close_inside = int(zone_bot <= close[i] <= zone_top)

            # ── outcome ──────────────────────────────────────────────────────
            outcome = _measure_outcome(
                high, low, close, n, i, side,
                zone_bot, zone_top, atr, fwd_bars)

            rows.append({
                "bar_idx":        i,
                "signal_time":    ts[i],
                "ema":            p,
                "approach_side":  side,
                "ema_value":      round(float(ema_val), 5),
                "ema_slope_atr":  round(float(slope), 4) if not np.isnan(slope) else np.nan,
                "slope_aligned":  slope_aligned,
                "ema_alignment":  aligned_bull if side == "support" else aligned_bear,
                "next_ema_dist":  round(float(next_ema_dist), 3) if not np.isnan(next_ema_dist) else np.nan,
                "candle_range_atr": round(float(range_atr), 3) if not np.isnan(range_atr) else np.nan,
                "wick_ratio":     round(float(wick_ratio), 3) if not np.isnan(wick_ratio) else np.nan,
                "body_pct":       round(float(body_pct), 3)   if not np.isnan(body_pct)   else np.nan,
                "close_inside":   close_inside,
                "cy_hour":        int(cy_hours_arr[i]),
                "dow":            int(dow[i]),
                **outcome,
            })

    return pd.DataFrame(rows)


# ── stats helpers ------------------------------------------------------------
def _pct(x): return f"{x*100:5.1f}%"

def stats_row(df: pd.DataFrame, label: str, min_n: int = 10) -> None:
    n = len(df)
    if n < min_n:
        return
    b1 = df["did_bounce_1atr"].mean()
    b2 = df["did_bounce_2atr"].mean()
    br = df["did_break"].mean()
    mb = df["max_bounce_atr"].mean()
    ma = df["max_against_atr"].mean()
    print(f"  {label:50s}  n={n:4d}  "
          f"bounce1={_pct(b1)}  bounce2={_pct(b2)}  "
          f"break={_pct(br)}  "
          f"avg_move={mb:.2f}vs{ma:.2f}ATR")


def print_stats(ev: pd.DataFrame) -> None:
    if ev.empty:
        print("No touch events.")
        return

    print(f"\nTotal EMA touch events : {len(ev)}")
    print(f"  support    : {(ev['approach_side']=='support').sum()}")
    print(f"  resistance : {(ev['approach_side']=='resistance').sum()}")

    # -- By EMA period --------------------------------------------------------
    print("\n--- By EMA period ---")
    for p in EMA_PERIODS:
        sub = ev[ev["ema"] == p]
        stats_row(sub, f"EMA {p:2d}")
        for side in ("support", "resistance"):
            stats_row(sub[sub["approach_side"] == side], f"  EMA {p:2d} {side}")

    # -- Slope alignment ------------------------------------------------------
    print("\n--- EMA slope alignment (slope agrees with touch direction) ---")
    for p in EMA_PERIODS:
        sub = ev[ev["ema"] == p]
        stats_row(sub[sub["slope_aligned"] == 1], f"  EMA {p:2d} slope aligned")
        stats_row(sub[sub["slope_aligned"] == 0], f"  EMA {p:2d} slope against")

    # -- Full EMA stack alignment ---------------------------------------------
    print("\n--- Full stack alignment (EMA9>13>21>50) ---")
    for p in EMA_PERIODS:
        sub = ev[ev["ema"] == p]
        stats_row(sub[sub["ema_alignment"] == 1], f"  EMA {p:2d} aligned stack")
        stats_row(sub[sub["ema_alignment"] == 0], f"  EMA {p:2d} misaligned")

    # -- Next EMA distance (clustered vs spread) -------------------------------
    print("\n--- Next EMA distance quartiles (small = EMAs clustered = stronger zone) ---")
    for p in EMA_PERIODS:
        sub = ev[ev["ema"] == p].dropna(subset=["next_ema_dist"])
        if len(sub) < 20:
            continue
        sub = sub.copy()
        sub["dist_q"] = pd.qcut(sub["next_ema_dist"], 4,
                                 labels=["Q1 close", "Q2", "Q3", "Q4 far"])
        for q in ["Q1 close", "Q2", "Q3", "Q4 far"]:
            stats_row(sub[sub["dist_q"] == q],
                      f"  EMA {p:2d} next_ema {q}")

    # -- Wick ratio -----------------------------------------------------------
    print("\n--- Wick ratio quartiles ---")
    ev2 = ev.dropna(subset=["wick_ratio"]).copy()
    ev2["wq"] = pd.qcut(ev2["wick_ratio"], 4,
                         labels=["Q1 tiny", "Q2", "Q3", "Q4 large"])
    for q in ["Q1 tiny", "Q2", "Q3", "Q4 large"]:
        stats_row(ev2[ev2["wq"] == q], f"  wick {q}")

    # -- Close inside vs wick-only -------------------------------------------
    print("\n--- Close inside EMA zone vs wick-only ---")
    stats_row(ev[ev["close_inside"] == 1], "close inside zone")
    stats_row(ev[ev["close_inside"] == 0], "wick only")

    # -- Best combos ----------------------------------------------------------
    print("\n--- Key combos ---")
    # Strong bounce signal: large wick, slope aligned, full stack
    for p in EMA_PERIODS:
        sub = ev[
            (ev["ema"] == p) &
            (ev["wick_ratio"] > 0.5) &
            (ev["close_inside"] == 0) &
            (ev["slope_aligned"] == 1) &
            (ev["ema_alignment"] == 1)
        ]
        stats_row(sub, f"  EMA {p:2d}: wick>0.5 + aligned slope + stack")

    print()
    # Breakout signal: tiny wick, close inside, slope against
    for p in EMA_PERIODS:
        sub = ev[
            (ev["ema"] == p) &
            (ev["wick_ratio"] < 0.1) &
            (ev["close_inside"] == 1)
        ]
        stats_row(sub, f"  EMA {p:2d}: tiny wick + close inside (breakout)")

    # -- Session hours --------------------------------------------------------
    print("\n--- By Cyprus hour (all EMAs combined) ---")
    for h in sorted(ev["cy_hour"].unique()):
        sub = ev[ev["cy_hour"] == h]
        if len(sub) >= 20:
            stats_row(sub, f"  hour {h:02d}", min_n=20)

    # -- Slope magnitude ------------------------------------------------------
    print("\n--- EMA slope magnitude quartiles (|slope| in ATR/bar) ---")
    ev3 = ev.dropna(subset=["ema_slope_atr"]).copy()
    ev3["abs_slope"] = ev3["ema_slope_atr"].abs()
    ev3["sq"] = pd.qcut(ev3["abs_slope"], 4,
                         labels=["Q1 flat", "Q2", "Q3", "Q4 steep"])
    for q in ["Q1 flat", "Q2", "Q3", "Q4 steep"]:
        stats_row(ev3[ev3["sq"] == q], f"  slope {q}")


# ── main --------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="EMA touch research pipeline")
    ap.add_argument("--symbol",    default="EURUSD")
    ap.add_argument("--tf",        default="1h")
    ap.add_argument("--start",     default="2022-01-01")
    ap.add_argument("--end",       default="2025-12-31")
    ap.add_argument("--atr-len",   type=int,   default=14)
    ap.add_argument("--slope-n",   type=int,   default=5,
                    help="bars used to compute EMA slope")
    ap.add_argument("--zone-atr",  type=float, default=0.10,
                    help="half-width of EMA zone in ATR units (default 0.10)")
    ap.add_argument("--fwd-bars",  type=int,   default=20,
                    help="bars to look forward for outcome measurement")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent

    print(f"Fetching {args.symbol} {args.tf}  {args.start} -> {args.end}")
    raw = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  {len(raw):,} bars\n")
    if raw.empty:
        return

    df  = add_features(raw, atr_len=args.atr_len, slope_n=args.slope_n)
    cy  = _cy_hours(df)

    print(f"Scanning EMA touch events (zone = +/-{args.zone_atr} ATR)...")
    ev = run_research(df, cy, zone_atr=args.zone_atr, fwd_bars=args.fwd_bars)
    print(f"  touch events found: {len(ev)}")

    print("\n" + "=" * 70)
    print(f"EMA TOUCH ANALYSIS  {args.symbol} {args.tf}  "
          f"zone=+/-{args.zone_atr}ATR  fwd={args.fwd_bars}bars")
    print("columns: bounce1/bounce2 = hit 1/2 ATR in expected direction | "
          "break = closed through zone | avg_move = bounce vs against ATR")
    print("=" * 70)
    print_stats(ev)

    out_path = out_dir / f"ema_touches_{args.symbol}_{args.tf}.csv"
    ev.to_csv(out_path, index=False)
    print(f"\n-> {out_path}  ({len(ev)} touch events)")


if __name__ == "__main__":
    main()
