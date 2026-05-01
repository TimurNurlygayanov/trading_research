"""
ARIMA + Kalman pullback entry strategy.

Logic
  Long  : arima_hl2 AND kalman_hl2 both predict UP at the same bar
          AND one of the pullback conditions is met
  Short : arima_hl2 AND kalman_hl2 both predict DOWN at the same bar

Entry styles
  next_bar  Enter at close of the bar right after the signal (baseline)
  last_red  Scan forward up to 5 bars; enter at close of first red candle
            (long) or first green candle (short). Skip trade if none found.
  big_wick  Scan forward up to 5 bars; enter at close of first candle whose
            lower shadow (long) / upper shadow (short) > 1.5x the candle body.

Exit
  either   exit when arima OR kalman reverses direction
  both     exit when BOTH models reverse direction

Pullback conditions (long side only)
  rsi_lt50   RSI(14) < 50 at the signal bar
  below_ema  close < EMA(20) at the signal bar
  neg3       close[t] < close[t-3]
  atr_dip    close < (max high over last 10 bars) - 0.5*ATR14

Usage
  python -m scripts.arima_kalman_pullback
  python -m scripts.arima_kalman_pullback --tf 1h --horizon 20
  python -m scripts.arima_kalman_pullback --strong-only
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv

SCRIPTS_DIR = Path(__file__).resolve().parent

# ── Technical indicators ──────────────────────────────────────────────────────

def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close.astype(float))
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period
    ag = pd.Series(gain).ewm(alpha=alpha, adjust=False).mean().values
    al = pd.Series(loss).ewm(alpha=alpha, adjust=False).mean().values
    rs  = ag / (al + 1e-10)
    rsi = np.empty(len(close))
    rsi[0] = 50.0
    rsi[1:] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _ema(close: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(close).ewm(span=span, adjust=False).mean().values


def _atr14(df: pd.DataFrame) -> np.ndarray:
    rng = (df["High"] - df["Low"]).values
    return pd.Series(rng).ewm(span=14, adjust=False).mean().values


# ── Candle entry scanner ──────────────────────────────────────────────────────

def find_candle_entry(df: pd.DataFrame, signal_bar: int,
                      entry_style: str, direction: str,
                      max_wait: int = 5) -> float | None:
    """
    Starting from bar index `signal_bar` (0-based into df), scan forward up to
    max_wait bars for the qualifying entry candle. Returns the close price of
    that candle, or None if no qualifying candle is found.

    entry_style 'next_bar' : always returns close[signal_bar] (no scan).
    entry_style 'last_red' : long  → first red candle  (close < open)
                             short → first green candle (close > open)
    entry_style 'big_wick' : long  → first candle where lower shadow > 1.5 * body
                             short → first candle where upper shadow > 1.5 * body
    """
    close  = df["Close"].values
    open_  = df["Open"].values
    high   = df["High"].values
    low    = df["Low"].values
    n      = len(close)

    if entry_style == "next_bar":
        idx = signal_bar
        return float(close[idx]) if idx < n else None

    for idx in range(signal_bar, min(signal_bar + max_wait + 1, n)):
        body         = abs(close[idx] - open_[idx])
        lower_shadow = min(close[idx], open_[idx]) - low[idx]
        upper_shadow = high[idx] - max(close[idx], open_[idx])

        if entry_style == "last_red":
            if direction == "long"  and close[idx] < open_[idx]:
                return float(close[idx])
            if direction == "short" and close[idx] > open_[idx]:
                return float(close[idx])

        elif entry_style == "big_wick":
            min_body = 1e-10
            if direction == "long"  and lower_shadow > max(body, min_body) * 1.5:
                return float(close[idx])
            if direction == "short" and upper_shadow > max(body, min_body) * 1.5:
                return float(close[idx])

    return None  # no qualifying candle found within window


# ── Pullback conditions ───────────────────────────────────────────────────────

def add_pullback_flags(df: pd.DataFrame, bar_indices: np.ndarray) -> pd.DataFrame:
    close  = df["Close"].values
    high   = df["High"].values
    rsi    = _rsi(close)
    ema20  = _ema(close, 20)
    atr    = _atr14(df)

    rows = []
    for i in bar_indices:
        if i < 10:
            continue
        c = close[i - 1]   # signal bar close (i is 1-indexed in records)

        rsi_lt50  = bool(rsi[i - 1] < 50)
        below_ema = bool(c < ema20[i - 1])
        neg3      = bool(i >= 4 and c < close[i - 4])
        local_hi  = float(np.max(high[max(0, i - 10): i]))
        atr_dip   = bool(c < local_hi - 0.5 * atr[i - 1])

        rows.append({
            "bar_i":     i,
            "rsi_lt50":  rsi_lt50,
            "below_ema": below_ema,
            "neg3":      neg3,
            "atr_dip":   atr_dip,
        })

    return pd.DataFrame(rows).set_index("bar_i") if rows else pd.DataFrame()


# ── Signal alignment ──────────────────────────────────────────────────────────

def build_aligned_signals(records: pd.DataFrame, horizon: int,
                          strong_only: bool) -> pd.DataFrame:
    """
    Find bars where arima_hl2 AND kalman_hl2 both agree on direction.
    Adds columns: combined_up (both +1), combined_down (both -1).
    """
    sub = records[records["horizon"] == horizon].copy()
    if strong_only:
        sub = sub[sub["signal_strong"]]

    arima = (sub[(sub["model"] == "arima_hl2")  & (sub["signal_type"] == "endpoint")]
             .set_index("i")[["signal", "actual_dir", "actual_ret"]]
             .rename(columns={"signal": "arima_sig"}))

    kalman = (sub[(sub["model"] == "kalman_hl2") & (sub["signal_type"] == "endpoint")]
              .set_index("i")[["signal"]]
              .rename(columns={"signal": "kalman_sig"}))

    merged = arima.join(kalman, how="inner")
    merged["combined_up"]   = (merged["arima_sig"] == 1)  & (merged["kalman_sig"] == 1)
    merged["combined_down"] = (merged["arima_sig"] == -1) & (merged["kalman_sig"] == -1)
    return merged.reset_index().rename(columns={"index": "i", "level_0": "i"})


# ── Strategy simulation ───────────────────────────────────────────────────────

def simulate(df: pd.DataFrame, aligned: pd.DataFrame, pullback: pd.DataFrame,
             pb_col: str | None, exit_mode: str,
             direction: str = "long",
             entry_style: str = "next_bar") -> dict:
    """
    direction   : 'long' or 'short'
    entry_style : 'next_bar', 'last_red', 'big_wick'
    pb_col      : None = no pullback filter, or column name in pullback df
    exit_mode   : 'either' or 'both'
    """
    signal_col = "combined_up" if direction == "long" else "combined_down"
    exit_dir   = -1 if direction == "long" else 1

    mask = aligned[signal_col].values
    if direction == "long" and pb_col is not None and not pullback.empty:
        pb_flags = aligned["i"].map(
            pullback[pb_col].to_dict() if pb_col in pullback.columns else {}
        ).fillna(False).values.astype(bool)
        mask = mask & pb_flags

    bars       = aligned["i"].values
    arima_sig  = aligned["arima_sig"].values
    kalman_sig = aligned["kalman_sig"].values

    trades      = []
    in_trade    = False
    entry_price = None

    for b, a_sig, k_sig, m in zip(bars, arima_sig, kalman_sig, mask):
        # b is 1-indexed in records; close[b] is the bar after signal (0-indexed into df)
        next_bar_idx = b  # close[b] == bar after signal

        # Exit check (use next-bar price, same as entry convention)
        if in_trade:
            if exit_mode == "either":
                exit_sig = (a_sig == exit_dir) or (k_sig == exit_dir)
            else:
                exit_sig = (a_sig == exit_dir) and (k_sig == exit_dir)
            if exit_sig:
                exit_price = (float(df["Close"].values[next_bar_idx])
                              if next_bar_idx < len(df) else float(df["Close"].values[-1]))
                if direction == "long":
                    trades.append(exit_price / entry_price - 1)
                else:
                    trades.append(entry_price / exit_price - 1)
                in_trade    = False
                entry_price = None

        # Entry check
        if not in_trade and m:
            ep = find_candle_entry(df, next_bar_idx, entry_style, direction)
            if ep is not None:
                in_trade    = True
                entry_price = ep

    if not trades:
        return {"trades": 0}

    arr = np.array(trades)
    n   = len(arr)
    sr  = float(arr.mean() / (arr.std() + 1e-10) * np.sqrt(252)) if n > 1 else 0.0
    return {
        "trades":    n,
        "win_rate":  float((arr > 0).mean()),
        "mean_ret":  float(arr.mean()),
        "total_ret": float((1 + arr).prod() - 1),
        "sharpe":    sr,
    }


# ── Correlation: pullback condition vs actual outcome ─────────────────────────

def pullback_hit_rates(aligned: pd.DataFrame, pullback: pd.DataFrame) -> pd.DataFrame:
    up = aligned[aligned["combined_up"]].copy()
    if pullback.empty or up.empty:
        return pd.DataFrame()

    up = up.join(pullback, on="i", how="left")
    base_hit = (up["actual_dir"] == 1).mean()
    rows = [{"condition": "no_filter", "n": len(up),
             "hit_rate": base_hit, "lift": 0.0,
             "mean_ret": up["actual_ret"].mean()}]

    for col in ("rsi_lt50", "below_ema", "neg3", "atr_dip"):
        if col not in up.columns:
            continue
        sub = up[up[col].fillna(False)]
        if len(sub) < 5:
            continue
        hr = (sub["actual_dir"] == 1).mean()
        rows.append({"condition": col, "n": len(sub), "hit_rate": hr,
                     "lift": hr - base_hit, "mean_ret": sub["actual_ret"].mean()})

        anti = up[~up[col].fillna(False)]
        if len(anti) >= 5:
            hr2 = (anti["actual_dir"] == 1).mean()
            rows.append({"condition": f"NOT_{col}", "n": len(anti),
                         "hit_rate": hr2, "lift": hr2 - base_hit,
                         "mean_ret": anti["actual_ret"].mean()})

    pb_cols = [c for c in ("rsi_lt50", "below_ema", "neg3", "atr_dip") if c in up.columns]
    if len(pb_cols) >= 2:
        up["pb_count"] = up[pb_cols].fillna(False).sum(axis=1)
        sub2 = up[up["pb_count"] >= 2]
        if len(sub2) >= 5:
            hr3 = (sub2["actual_dir"] == 1).mean()
            rows.append({"condition": "any_2_of_4", "n": len(sub2),
                         "hit_rate": hr3, "lift": hr3 - base_hit,
                         "mean_ret": sub2["actual_ret"].mean()})

    return pd.DataFrame(rows)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(v, fmt=".4f"):
    return f"{v:{fmt}}" if isinstance(v, float) and not np.isnan(v) else "  n/a "


def _print_sim_table(label_prefix: str, df: pd.DataFrame, aligned: pd.DataFrame,
                     pb_with_any2: pd.DataFrame, pb_cols_to_test: list,
                     direction: str, entry_styles: list):
    print(f"\n  [STRATEGY SIM -- {label_prefix}]")
    print(f"  {'Entry':<10} {'Filter':<14} {'exit':>6}  "
          f"{'trades':>6}  {'win%':>5}  {'mean_ret':>8}  {'total%':>7}  {'sharpe':>7}")
    print(f"  {'-'*76}")

    for es in entry_styles:
        for pb_col in pb_cols_to_test:
            for exit_mode in ("either", "both"):
                sim = simulate(df, aligned, pb_with_any2, pb_col, exit_mode,
                               direction=direction, entry_style=es)
                if not sim or sim["trades"] == 0:
                    continue
                label = pb_col if pb_col else "no_filter"
                print(
                    f"  {es:<10} {label:<14} {exit_mode:>6}  "
                    f"{sim['trades']:>6}  {sim['win_rate']*100:>4.1f}%  "
                    f"{sim['mean_ret']*100:>+7.3f}%  {sim['total_ret']*100:>+6.1f}%  "
                    f"{sim['sharpe']:>+7.2f}"
                )


# ── Main per-timeframe runner ─────────────────────────────────────────────────

def run_tf(tf: str, horizon: int, strong_only: bool, start: str, end: str):
    rec_path = SCRIPTS_DIR / f"arima_corr_EURUSD_{tf}_records.csv"
    if not rec_path.exists():
        print(f"  [SKIP] no records file: {rec_path.name}")
        print(f"  Run arima_direction_correlation.py --tf {tf} first.")
        return

    records = pd.read_csv(rec_path)
    print(f"\nFetching EURUSD {tf}  {start} to {end} ...")
    df = fetch_ohlcv("EURUSD", tf, start, end)
    if df.empty:
        print("  No data."); return
    print(f"  {len(df):,} bars")

    aligned = build_aligned_signals(records, horizon, strong_only)
    if aligned.empty:
        print("  No aligned signals."); return

    valid = aligned[aligned["i"] < len(df) - 1]   # need at least 1 bar after signal
    if valid.empty:
        print("  No valid bar indices within loaded data."); return
    aligned = valid

    bar_indices = aligned["i"].values
    pullback    = add_pullback_flags(df, bar_indices)

    n_up   = aligned["combined_up"].sum()
    n_down = aligned["combined_down"].sum()

    print(f"\n{'='*80}")
    print(f"  EURUSD {tf}  H={horizon}  {'strong' if strong_only else 'all'} signals")
    print(f"  Signal bars: {len(aligned)}  |  "
          f"ARIMA-UP: {(aligned['arima_sig']==1).sum()}  "
          f"KALMAN-UP: {(aligned['kalman_sig']==1).sum()}  "
          f"BOTH-UP: {n_up}  BOTH-DOWN: {n_down}")
    print(f"{'='*80}")

    # ── Hit-rate by pullback condition (long) ─────────────────────────────────
    hr_table = pullback_hit_rates(aligned, pullback)
    if not hr_table.empty:
        print(f"\n  [HIT RATE -- actual UP after H={horizon} bars, ARIMA+Kalman both UP]")
        print(f"  {'Condition':<16} {'n':>5}  {'hit_rate':>8}  {'lift':>7}  {'mean_ret':>9}")
        print(f"  {'-'*52}")
        for _, r in hr_table.iterrows():
            marker = " <--" if r["lift"] > 0.03 else ""
            print(f"  {r['condition']:<16} {int(r['n']):>5}  "
                  f"{_f(r['hit_rate']):>8}  {_f(r['lift'],'+.4f'):>7}  "
                  f"{_f(r['mean_ret'],'.5f'):>9}{marker}")

    # ── Build any_2_of_4 composite pullback flag ──────────────────────────────
    def _any2_mask(pb_df):
        cols = [c for c in ("rsi_lt50","below_ema","neg3","atr_dip") if c in pb_df.columns]
        if len(cols) < 2:
            return pb_df
        pb_df = pb_df.copy()
        pb_df["any_2_of_4"] = pb_df[cols].fillna(False).sum(axis=1) >= 2
        return pb_df

    pb_with_any2 = _any2_mask(pullback)

    entry_styles   = ["next_bar", "last_red", "big_wick"]
    long_pb_cols   = [None, "rsi_lt50", "below_ema", "neg3", "atr_dip", "any_2_of_4"]
    short_pb_cols  = [None]   # no pullback filters for shorts (opposite direction)

    # ── Long strategy ─────────────────────────────────────────────────────────
    _print_sim_table(
        "LONG: BOTH UP + pullback filter",
        df, aligned, pb_with_any2, long_pb_cols, "long", entry_styles,
    )

    # ── Short strategy ────────────────────────────────────────────────────────
    _print_sim_table(
        "SHORT: BOTH DOWN",
        df, aligned, pb_with_any2, short_pb_cols, "short", entry_styles,
    )

    # ── Condition overlap ─────────────────────────────────────────────────────
    if not pullback.empty:
        pb_cols = [c for c in ("rsi_lt50","below_ema","neg3","atr_dip")
                   if c in pullback.columns]
        counts  = pullback[pb_cols].fillna(False).sum(axis=1)
        up_bars = aligned[aligned["combined_up"]]["i"]
        cnt_up  = counts.reindex(up_bars).fillna(0)

        print(f"\n  [PULLBACK CONDITION OVERLAP -- bars with BOTH models UP]")
        for k in range(5):
            n = int((cnt_up == k).sum())
            if n:
                print(f"    {k} conditions met: {n} bars  "
                      f"({n / len(cnt_up) * 100:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tf",          nargs="+", default=["1h"])
    ap.add_argument("--horizon",     type=int, default=20)
    ap.add_argument("--start",       default="2025-01-01")
    ap.add_argument("--end",         default="2025-12-31")
    ap.add_argument("--strong-only", action="store_true")
    args = ap.parse_args()

    for tf in args.tf:
        run_tf(tf, args.horizon, args.strong_only, args.start, args.end)

    print("\nDone.")


if __name__ == "__main__":
    main()
