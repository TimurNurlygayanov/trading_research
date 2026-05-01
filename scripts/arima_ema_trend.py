"""
EMA trend + ARIMA + Kalman strategy.

Entry
  Long : EMA(N) is rising  (ema[i] > ema[i-slope_bars])
         AND arima_hl2 AND kalman_hl2 both predict UP (endpoint signal)
  Short: EMA(N) is falling (ema[i] < ema[i-slope_bars])
         AND arima_hl2 AND kalman_hl2 both predict DOWN

Exit
  either  : exit when arima OR  kalman reverses direction
  both    : exit when arima AND kalman both reverse direction

Entry style
  next_bar : enter at close of the bar after the signal
  last_red : wait up to 5 bars for first red candle (long) / green (short)

Loads pre-computed records from arima_direction_correlation.py.
Tests EMA periods 20, 50, 100, 200 side by side.

Usage
  python -m scripts.arima_ema_trend
  python -m scripts.arima_ema_trend --tf 1h --horizons 10 20
  python -m scripts.arima_ema_trend --slope-bars 1 3 5 --ema-periods 50 100
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


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(close: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(close).ewm(span=span, adjust=False).mean().values


def _atr14(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n     = len(close)
    tr    = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    atr    = np.empty(n)
    atr[0] = tr[0]
    alpha  = 1.0 / period
    for i in range(1, n):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


# ── Entry candle scanner ──────────────────────────────────────────────────────

def _find_entry(df: pd.DataFrame, start_idx: int,
                entry_style: str, direction: str,
                max_wait: int = 5) -> float | None:
    """
    From start_idx (0-based into df), scan forward up to max_wait bars.
    'next_bar' : return close[start_idx] immediately.
    'last_red' : long -> first red candle; short -> first green candle.
    """
    close = df["Close"].values
    open_ = df["Open"].values
    n     = len(close)

    if entry_style == "next_bar":
        return float(close[start_idx]) if start_idx < n else None

    for idx in range(start_idx, min(start_idx + max_wait + 1, n)):
        if entry_style == "last_red":
            if direction == "long"  and close[idx] < open_[idx]:
                return float(close[idx])
            if direction == "short" and close[idx] > open_[idx]:
                return float(close[idx])
    return None


# ── SL/TP simulation ─────────────────────────────────────────────────────────

def simulate_sltp(df: pd.DataFrame, aligned: pd.DataFrame,
                  entry_mask: np.ndarray,
                  direction: str, entry_style: str,
                  atr: np.ndarray,
                  sl_mult: float, tp_mult: float) -> dict:
    """
    Fixed ATR-based SL and TP. Walks every bar after entry checking
    high/low against SL and TP levels.
    When both SL and TP are hit on the same bar, open[i] is used to
    decide which was closer (conservative: SL wins if open < sl for long).
    """
    close  = df["Close"].values
    high   = df["High"].values
    low    = df["Low"].values
    open_  = df["Open"].values
    n      = len(close)

    bars       = aligned["i"].values
    entry_d    = 1 if direction == "long" else -1

    trades      = []
    in_trade    = False
    entry_price = None
    sl_price    = None
    tp_price    = None

    # Map bar index → mask value for fast lookup
    mask_map = {int(b): bool(m) for b, m in zip(bars, entry_mask)}

    # Walk every bar (signals fire at bars[k], entry at bars[k] which is next_idx)
    # We need the full bar sequence, so iterate all bars
    signal_set = set(int(b) for b in bars)

    for i in range(1, n):
        # Check SL/TP on current bar if in trade
        if in_trade:
            sl_hit = (low[i]  <= sl_price) if direction == "long"  else (high[i] >= sl_price)
            tp_hit = (high[i] >= tp_price) if direction == "long"  else (low[i]  <= tp_price)

            if sl_hit and tp_hit:
                # Both hit same bar: use open to determine order
                if direction == "long":
                    exit_p = sl_price if open_[i] <= sl_price else tp_price
                else:
                    exit_p = sl_price if open_[i] >= sl_price else tp_price
            elif sl_hit:
                exit_p = sl_price
            elif tp_hit:
                exit_p = tp_price
            else:
                exit_p = None

            if exit_p is not None:
                trades.append((exit_p / entry_price - 1) * entry_d)
                in_trade = False
                entry_price = sl_price = tp_price = None

        # Check for new entry signal (i is bars[k], signal was at i-1 in records convention)
        if not in_trade and i in signal_set and mask_map.get(i, False):
            ep = _find_entry(df, i, entry_style, direction)
            if ep is not None:
                atr_val = float(atr[i - 1]) if i > 0 else float(atr[0])
                in_trade    = True
                entry_price = ep
                if direction == "long":
                    sl_price = ep - sl_mult * atr_val
                    tp_price = ep + tp_mult * atr_val
                else:
                    sl_price = ep + sl_mult * atr_val
                    tp_price = ep - tp_mult * atr_val

    if not trades:
        return {"trades": 0}

    arr = np.array(trades)
    n_t = len(arr)
    sr  = float(arr.mean() / (arr.std() + 1e-10) * np.sqrt(252)) if n_t > 1 else 0.0
    return {"trades": n_t, "win_rate": float((arr > 0).mean()),
            "mean_ret": float(arr.mean()),
            "total_ret": float((1 + arr).prod() - 1), "sharpe": sr}


# ── Signal alignment ──────────────────────────────────────────────────────────

def build_aligned(records: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Both arima_hl2 AND kalman_hl2 endpoint signals at the same bar."""
    sub = records[records["horizon"] == horizon].copy()

    arima = (sub[(sub["model"] == "arima_hl2") & (sub["signal_type"] == "endpoint")]
             .set_index("i")[["signal", "actual_dir", "actual_ret"]]
             .rename(columns={"signal": "arima_sig"}))

    kalman = (sub[(sub["model"] == "kalman_hl2") & (sub["signal_type"] == "endpoint")]
              .set_index("i")[["signal"]]
              .rename(columns={"signal": "kalman_sig"}))

    merged = arima.join(kalman, how="inner")
    merged["combined_up"]   = (merged["arima_sig"] == 1)  & (merged["kalman_sig"] == 1)
    merged["combined_down"] = (merged["arima_sig"] == -1) & (merged["kalman_sig"] == -1)
    return merged.reset_index().rename(columns={"index": "i", "level_0": "i"})


# ── EMA trend filter ──────────────────────────────────────────────────────────

def ema_trend_flag(ema_vals: np.ndarray, bar_indices: np.ndarray,
                   slope_bars: int, direction: str) -> np.ndarray:
    """
    Returns bool array aligned to bar_indices.
    True when EMA is trending in the requested direction over slope_bars bars.
    bar_indices are 1-indexed (records convention); ema_vals is 0-indexed.
    """
    flags = np.zeros(len(bar_indices), dtype=bool)
    for k, i in enumerate(bar_indices):
        idx = i - 1   # convert to 0-based
        prev = idx - slope_bars
        if prev < 0:
            continue
        if direction == "long":
            flags[k] = bool(ema_vals[idx] > ema_vals[prev])
        else:
            flags[k] = bool(ema_vals[idx] < ema_vals[prev])
    return flags


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(df: pd.DataFrame, aligned: pd.DataFrame,
             entry_mask: np.ndarray,
             direction: str, entry_style: str, exit_mode: str,
             exit_dir: int | None = None) -> dict:
    """
    exit_dir: signal value that triggers exit. Defaults to -1 for long, +1 for short.
              Pass +1 for contrarian long (exit when model turns bullish again).
    """
    close      = df["Close"].values
    bars       = aligned["i"].values
    arima_sig  = aligned["arima_sig"].values
    kalman_sig = aligned["kalman_sig"].values
    if exit_dir is None:
        exit_dir = -1 if direction == "long" else 1

    trades      = []
    in_trade    = False
    entry_price = None

    for b, a_sig, k_sig, m in zip(bars, arima_sig, kalman_sig, entry_mask):
        next_idx = b   # b is 1-indexed; close[b] is the bar after signal

        # Exit check
        if in_trade:
            if exit_mode == "either":
                exit_sig = (a_sig == exit_dir) or  (k_sig == exit_dir)
            else:
                exit_sig = (a_sig == exit_dir) and (k_sig == exit_dir)
            if exit_sig:
                ep = (float(close[next_idx]) if next_idx < len(close)
                      else float(close[-1]))
                trades.append((ep / entry_price - 1) * (1 if direction == "long" else -1))
                in_trade    = False
                entry_price = None

        # Entry check
        if not in_trade and m:
            ep = _find_entry(df, next_idx, entry_style, direction)
            if ep is not None:
                in_trade    = True
                entry_price = ep

    if not trades:
        return {"trades": 0}

    arr = np.array(trades)
    n   = len(arr)
    sr  = float(arr.mean() / (arr.std() + 1e-10) * np.sqrt(252)) if n > 1 else 0.0
    return {"trades": n, "win_rate": float((arr > 0).mean()),
            "mean_ret": float(arr.mean()),
            "total_ret": float((1 + arr).prod() - 1), "sharpe": sr}


# ── Per-timeframe runner ──────────────────────────────────────────────────────

def _f(v, fmt=".4f"):
    return f"{v:{fmt}}" if isinstance(v, float) and not np.isnan(v) else "  n/a "


def run_tf(tf: str, horizons: list[int], ema_periods: list[int],
           slope_bars_list: list[int],
           entry_styles: list[str],
           combined_ema_periods: list[tuple],
           start: str, end: str) -> None:

    rec_path = SCRIPTS_DIR / f"arima_corr_EURUSD_{tf}_records.csv"
    if not rec_path.exists():
        print(f"  [SKIP] no records: {rec_path.name} -- run arima_direction_correlation.py first")
        return

    records = pd.read_csv(rec_path)
    print(f"\nFetching EURUSD {tf}  {start} to {end} ...")
    df = fetch_ohlcv("EURUSD", tf, start, end)
    if df.empty:
        print("  No data."); return
    print(f"  {len(df):,} bars")

    close = df["Close"].values

    # Pre-compute all EMA series (include periods from combined filters too)
    all_periods = set(ema_periods)
    for (p1, _, p2, _) in combined_ema_periods:
        all_periods.update([p1, p2])
    ema_cache: dict[int, np.ndarray] = {p: _ema(close, p) for p in all_periods}

    for horizon in horizons:
        aligned = build_aligned(records, horizon)
        if aligned.empty:
            print(f"  No signals for H={horizon}"); continue

        # Keep only bars within loaded data (need 1 extra bar for entry)
        aligned = aligned[aligned["i"] < len(df) - 1].copy()
        if aligned.empty:
            continue

        bar_indices = aligned["i"].values
        n_up   = aligned["combined_up"].sum()
        n_down = aligned["combined_down"].sum()

        print(f"\n{'='*76}")
        print(f"  EURUSD {tf}  H={horizon}  |  "
              f"signal bars: {len(aligned)}  BOTH-UP: {n_up}  BOTH-DOWN: {n_down}")
        print(f"{'='*76}")

        # ── Baseline: ARIMA+Kalman, no EMA filter ────────────────────────────
        print(f"\n  {'Config':<30} {'dir':<6} {'exit':>6}  "
              f"{'n':>5}  {'win%':>5}  {'mean%':>7}  {'tot%':>6}  {'sharpe':>7}")
        print(f"  {'-'*72}")

        for exit_mode in ("either", "both"):
            for direction, col in (("long", "combined_up"), ("short", "combined_down")):
                mask = aligned[col].values
                sim  = simulate(df, aligned, mask, direction, "next_bar", exit_mode)
                if sim["trades"]:
                    print(f"  {'no_ema_filter':<30} {direction:<6} {exit_mode:>6}  "
                          f"{sim['trades']:>5}  {sim['win_rate']*100:>4.1f}%  "
                          f"{sim['mean_ret']*100:>+6.3f}%  "
                          f"{sim['total_ret']*100:>+5.1f}%  "
                          f"{sim['sharpe']:>+7.2f}")

        # ── EMA trend filter grid ─────────────────────────────────────────────
        print()
        for ema_p in ema_periods:
            ema_vals = ema_cache[ema_p]
            for slope_b in slope_bars_list:
                for exit_mode in ("either", "both"):
                    for direction, col in (("long","combined_up"),("short","combined_down")):
                        base_mask = aligned[col].values
                        ema_flag  = ema_trend_flag(ema_vals, bar_indices,
                                                   slope_b, direction)
                        mask      = base_mask & ema_flag

                        for es in entry_styles:
                            sim = simulate(df, aligned, mask, direction, es, exit_mode)
                            if not sim or sim["trades"] == 0:
                                continue
                            label = f"EMA{ema_p} slope{slope_b} {es}"
                            print(f"  {label:<30} {direction:<6} {exit_mode:>6}  "
                                  f"{sim['trades']:>5}  {sim['win_rate']*100:>4.1f}%  "
                                  f"{sim['mean_ret']*100:>+6.3f}%  "
                                  f"{sim['total_ret']*100:>+5.1f}%  "
                                  f"{sim['sharpe']:>+7.2f}")

        # ── Contrarian: EMA up + model predicts DOWN → buy (and vice versa) ────
        print(f"\n  [CONTRARIAN -- EMA trend OPPOSITE to model signal]")
        for ema_p in ema_periods:
            ema_vals = ema_cache[ema_p]
            for slope_b in slope_bars_list:
                for exit_mode in ("either", "both"):
                    # Contrarian long: combined_down signal + EMA trending UP
                    base_mask = aligned["combined_down"].values
                    ema_flag  = ema_trend_flag(ema_vals, bar_indices, slope_b, "long")
                    mask      = base_mask & ema_flag
                    for es in entry_styles:
                        # Exit when model turns +1 (reversion confirmed)
                        sim = simulate(df, aligned, mask, "long", es, exit_mode,
                                       exit_dir=1)
                        if not sim or sim["trades"] == 0:
                            continue
                        label = f"CTR EMA{ema_p}s{slope_b} {es}"
                        print(f"  {label:<30} {'long':<6} {exit_mode:>6}  "
                              f"{sim['trades']:>5}  {sim['win_rate']*100:>4.1f}%  "
                              f"{sim['mean_ret']*100:>+6.3f}%  "
                              f"{sim['total_ret']*100:>+5.1f}%  "
                              f"{sim['sharpe']:>+7.2f}")
                    # Contrarian short: combined_up signal + EMA trending DOWN
                    base_mask = aligned["combined_up"].values
                    ema_flag  = ema_trend_flag(ema_vals, bar_indices, slope_b, "short")
                    mask      = base_mask & ema_flag
                    for es in entry_styles:
                        sim = simulate(df, aligned, mask, "short", es, exit_mode,
                                       exit_dir=-1)
                        if not sim or sim["trades"] == 0:
                            continue
                        label = f"CTR EMA{ema_p}s{slope_b} {es}"
                        print(f"  {label:<30} {'short':<6} {exit_mode:>6}  "
                              f"{sim['trades']:>5}  {sim['win_rate']*100:>4.1f}%  "
                              f"{sim['mean_ret']*100:>+6.3f}%  "
                              f"{sim['total_ret']*100:>+5.1f}%  "
                              f"{sim['sharpe']:>+7.2f}")

        # ── Combined EMA filters (both must be trending same direction) ───────
        combos = combined_ema_periods  # list of (p1, slope1, p2, slope2) tuples
        if combos:
            print()
        for (p1, s1, p2, s2) in combos:
            if p1 not in ema_cache or p2 not in ema_cache:
                continue
            for exit_mode in ("either", "both"):
                for direction, col in (("long","combined_up"),("short","combined_down")):
                    base_mask = aligned[col].values
                    flag1 = ema_trend_flag(ema_cache[p1], bar_indices, s1, direction)
                    flag2 = ema_trend_flag(ema_cache[p2], bar_indices, s2, direction)
                    mask  = base_mask & flag1 & flag2
                    for es in entry_styles:
                        sim = simulate(df, aligned, mask, direction, es, exit_mode)
                        if not sim or sim["trades"] == 0:
                            continue
                        label = f"EMA{p1}s{s1}+EMA{p2}s{s2} {es}"
                        print(f"  {label:<30} {direction:<6} {exit_mode:>6}  "
                              f"{sim['trades']:>5}  {sim['win_rate']*100:>4.1f}%  "
                              f"{sim['mean_ret']*100:>+6.3f}%  "
                              f"{sim['total_ret']*100:>+5.1f}%  "
                              f"{sim['sharpe']:>+7.2f}")

        # ── SL/TP grid ────────────────────────────────────────────────────────
        # Best 1h config: EMA200 slope5, long, next_bar
        # Best 5m config: CTR EMA21 slope3, long, last_red (contrarian)
        atr = _atr14(df)
        sl_mults = [0.5, 1.0, 1.5, 2.0]
        tp_mults = [1.0, 1.5, 2.0, 3.0, 4.0]

        sltp_configs = []
        # Trend long — EMA200 slope5, next_bar
        if 200 in ema_cache:
            ema_flag = ema_trend_flag(ema_cache[200], bar_indices, 5, "long")
            mask_trend = aligned["combined_up"].values & ema_flag
            sltp_configs.append(("EMA200s5 long next_bar", mask_trend, "long", "next_bar"))
        # Contrarian long — EMA21 slope3 up + model down, last_red
        if 21 in ema_cache:
            ema_flag_ctr = ema_trend_flag(ema_cache[21], bar_indices, 3, "long")
            mask_ctr = aligned["combined_down"].values & ema_flag_ctr
            sltp_configs.append(("CTR EMA21s3 long last_red", mask_ctr, "long", "last_red"))

        for cfg_label, mask, direction, es in sltp_configs:
            if mask.sum() == 0:
                continue
            print(f"\n  [SL/TP grid: {cfg_label}]  tf={tf}  H={horizon}")
            # Header row
            header = f"  {'SL\\TP':>8}" + "".join(f"  TP×{tp:>3}" for tp in tp_mults)
            print(header)
            print(f"  {'-' * (len(header) - 2)}")
            for sl in sl_mults:
                row = f"  SL×{sl:<4}"
                for tp in tp_mults:
                    sim = simulate_sltp(df, aligned, mask, direction, es, atr, sl, tp)
                    if sim["trades"] == 0:
                        row += f"  {'n/a':>7}"
                    else:
                        row += f"  {sim['sharpe']:>+7.2f}"
                print(row)
            # Also print trade count grid
            print(f"  {'(trades)':>8}")
            for sl in sl_mults:
                row = f"  SL×{sl:<4}"
                for tp in tp_mults:
                    sim = simulate_sltp(df, aligned, mask, direction, es, atr, sl, tp)
                    row += f"  {sim['trades']:>7}"
                print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tf",           nargs="+", default=["1h"])
    ap.add_argument("--horizons",     nargs="+", type=int, default=[20])
    ap.add_argument("--ema-periods",  nargs="+", type=int, default=[21, 50, 200])
    ap.add_argument("--slope-bars",   nargs="+", type=int, default=[3, 5])
    ap.add_argument("--entry-styles", nargs="+", default=["next_bar", "last_red"])
    ap.add_argument("--start",        default="2025-01-01")
    ap.add_argument("--end",          default="2025-12-31")
    args = ap.parse_args()

    # Combined EMA filters: (period1, slope1, period2, slope2)
    combined = [
        (200, 5, 50, 5),
        (200, 5, 50, 1),
        (200, 3, 50, 3),
        (200, 5, 21, 3),
        (200, 3, 21, 3),
    ]

    for tf in args.tf:
        run_tf(tf, args.horizons, args.ema_periods,
               args.slope_bars, args.entry_styles,
               combined, args.start, args.end)
    print("\nDone.")


if __name__ == "__main__":
    main()
