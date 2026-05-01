"""
ARIMA + SuperTrend strategy.

At every SuperTrend(7, 2) flip bar, fit ARIMA(2,1,0) live on the
lookback window and get a fresh H-step forecast. If the forecast
direction agrees with the ST flip direction -> enter trade.
Exit on the opposite ST flip.

Optionally require Kalman filter to also agree.

No pre-computed records needed -- ARIMA is fitted only at ~5k flip
bars per year (not every bar), so it runs in under a minute on 1m.

Usage
  python -m scripts.arima_supertrend
  python -m scripts.arima_supertrend --tf 1m 5m 1h
  python -m scripts.arima_supertrend --st-mult 3 --require-kalman
  python -m scripts.arima_supertrend --st-period 7 --st-mult 2 --horizon 20
"""
from __future__ import annotations

import argparse
import sys
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv

LOOKBACK     = 60
ARIMA_ORDER  = (2, 1, 0)
N_WORKERS    = max(1, min(12, cpu_count() - 2))


# ── SuperTrend ────────────────────────────────────────────────────────────────

def _supertrend(df: pd.DataFrame, period: int, multiplier: float) -> np.ndarray:
    """Returns int8 array: +1 = bullish, -1 = bearish (Wilder ATR)."""
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n     = len(close)

    tr     = np.empty(n)
    tr[0]  = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))

    atr    = np.empty(n)
    atr[0] = tr[0]
    alpha  = 1.0 / period
    for i in range(1, n):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha

    hl2         = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    upper       = upper_basic.copy()
    lower       = lower_basic.copy()
    st          = np.ones(n, dtype=np.int8)

    for i in range(1, n):
        upper[i] = (upper_basic[i]
                    if upper_basic[i] < upper[i - 1] or close[i - 1] > upper[i - 1]
                    else upper[i - 1])
        lower[i] = (lower_basic[i]
                    if lower_basic[i] > lower[i - 1] or close[i - 1] < lower[i - 1]
                    else lower[i - 1])
        if st[i - 1] == -1:
            st[i] = np.int8(1) if close[i] > upper[i] else np.int8(-1)
        else:
            st[i] = np.int8(-1) if close[i] < lower[i] else np.int8(1)

    return st


# ── ARIMA + Kalman forecast at a single bar ───────────────────────────────────

def _kalman_signal(series: np.ndarray, horizon: int) -> int:
    """Return +1/-1 based on Kalman H-step forecast vs current price."""
    var_obs = float(np.var(np.diff(series))) or 1e-10
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * (var_obs * 1e-3)
    R = np.array([[var_obs]])
    x = np.array([series[0], 0.0])
    P = np.eye(2) * var_obs
    for obs in series:
        x = F @ x; P = F @ P @ F.T + Q
        S = float((H @ P @ H.T)[0, 0]) + R[0, 0]
        K = (P @ H.T) / S
        x = x + K.ravel() * (obs - float(H @ x))
        P = (np.eye(2) - K @ H) @ P
    xf = x.copy()
    for _ in range(horizon):
        xf = F @ xf
    return 1 if float(xf[0]) > series[-1] else -1


def _fit_flip(args: tuple) -> tuple[int, int, int]:
    """Worker: fit ARIMA+Kalman at one flip bar. Returns (bar_idx, arima_sig, kalman_sig)."""
    bar_idx, window, horizon = args
    hl2 = (window[:, 0] + window[:, 1]) / 2.0   # (high+low)/2
    cur = hl2[-1]

    arima_sig = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fc = np.asarray(ARIMA(hl2, order=ARIMA_ORDER).fit().forecast(horizon))
            arima_sig = 1 if fc[-1] > cur else -1
        except Exception:
            pass

    kalman_sig = _kalman_signal(hl2, horizon)
    return bar_idx, arima_sig, kalman_sig


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(close: np.ndarray, st: np.ndarray,
             flip_sigs: dict[int, tuple[int, int]],
             direction: str, require_kalman: bool) -> dict:
    """
    flip_sigs : {bar_idx: (arima_sig, kalman_sig)}
    direction : 'long' or 'short'
    """
    entry_d = 1 if direction == "long" else -1
    n       = len(close)

    trades      = []
    in_trade    = False
    entry_price = None

    for i in range(1, n):
        flipped = st[i] != st[i - 1]

        if in_trade and flipped and st[i] == -entry_d:
            trades.append((close[i] / entry_price - 1) * entry_d)
            in_trade    = False
            entry_price = None

        if not in_trade and flipped and st[i] == entry_d and i in flip_sigs:
            a_sig, k_sig = flip_sigs[i]
            if a_sig != entry_d:
                continue
            if require_kalman and k_sig != entry_d:
                continue
            in_trade    = True
            entry_price = float(close[i])

    if not trades:
        return {"trades": 0}

    arr = np.array(trades)
    n_t = len(arr)
    sr  = float(arr.mean() / (arr.std() + 1e-10) * np.sqrt(252)) if n_t > 1 else 0.0
    return {
        "trades":    n_t,
        "win_rate":  float((arr > 0).mean()),
        "mean_ret":  float(arr.mean()),
        "total_ret": float((1 + arr).prod() - 1),
        "sharpe":    sr,
    }


# ── Per-timeframe runner ──────────────────────────────────────────────────────

def _f(v, fmt=".4f"):
    return f"{v:{fmt}}" if isinstance(v, float) and not np.isnan(v) else "  n/a "


def run_tf(tf: str, horizon: int, st_period: int, st_mult: float,
           require_kalman: bool, start: str, end: str):

    print(f"\nFetching EURUSD {tf}  {start} to {end} ...")
    df = fetch_ohlcv("EURUSD", tf, start, end)
    if df.empty:
        print("  No data."); return
    n = len(df)
    print(f"  {n:,} bars")

    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)

    st = _supertrend(df, period=st_period, multiplier=st_mult)

    # Collect all flip bar indices (need at least LOOKBACK bars of history)
    flip_bars = [i for i in range(max(LOOKBACK, 1), n)
                 if st[i] != st[i - 1]]

    flips_up   = sum(1 for i in flip_bars if st[i] == 1)
    flips_down = sum(1 for i in flip_bars if st[i] == -1)

    print(f"  ST({st_period},{st_mult}) flips: GREEN={flips_up}  RED={flips_down}  "
          f"-- fitting ARIMA on {len(flip_bars)} bars ({N_WORKERS} workers) ...")

    # Build work items: (bar_idx, window_array[LOOKBACK x 2], horizon)
    work = [
        (i, np.stack([high[i - LOOKBACK:i], low[i - LOOKBACK:i]], axis=1), horizon)
        for i in flip_bars
    ]

    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(_fit_flip, work)

    flip_sigs: dict[int, tuple[int, int]] = {
        bar_idx: (a_sig, k_sig)
        for bar_idx, a_sig, k_sig in results
        if a_sig != 0
    }
    valid_flips = len(flip_sigs)

    print(f"\n{'='*72}")
    print(f"  EURUSD {tf}  H={horizon}  ST({st_period},{st_mult})"
          + ("  +Kalman" if require_kalman else ""))
    print(f"  Total flips: {len(flip_bars)}  |  ARIMA fitted: {len(flip_bars)}  "
          f"|  valid signals: {valid_flips}")
    print(f"{'='*72}")

    print(f"  {'Direction':<8} {'trades':>6}  {'win%':>5}  "
          f"{'mean_ret':>8}  {'total%':>7}  {'sharpe':>7}")
    print(f"  {'-'*52}")

    for direction in ("long", "short"):
        sim = simulate(close, st, flip_sigs, direction, require_kalman)
        if not sim or sim["trades"] == 0:
            print(f"  {direction:<8}  (no trades)")
            continue
        print(
            f"  {direction:<8} {sim['trades']:>6}  {sim['win_rate']*100:>4.1f}%  "
            f"{sim['mean_ret']*100:>+7.3f}%  {sim['total_ret']*100:>+6.1f}%  "
            f"{sim['sharpe']:>+7.2f}"
        )

    # ARIMA-only vs ARIMA+Kalman comparison
    if not require_kalman:
        print(f"\n  [ARIMA-only vs ARIMA+Kalman]")
        print(f"  {'Config':<20} {'dir':<6} {'trades':>6}  {'win%':>5}  {'sharpe':>7}")
        print(f"  {'-'*48}")
        for rk in (False, True):
            label = "ARIMA+Kalman" if rk else "ARIMA only  "
            for direction in ("long", "short"):
                sim = simulate(close, st, flip_sigs, direction, rk)
                if not sim or sim["trades"] == 0:
                    continue
                print(f"  {label:<20} {direction:<6} {sim['trades']:>6}  "
                      f"{sim['win_rate']*100:>4.1f}%  {sim['sharpe']:>+7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tf",             nargs="+", default=["1m", "5m", "1h"])
    ap.add_argument("--horizon",        type=int,   default=20)
    ap.add_argument("--st-period",      type=int,   default=7)
    ap.add_argument("--st-mult",        type=float, default=2.0)
    ap.add_argument("--start",          default="2025-01-01")
    ap.add_argument("--end",            default="2025-12-31")
    ap.add_argument("--require-kalman", action="store_true")
    args = ap.parse_args()

    if "multiprocessing" in sys.modules:
        from multiprocessing import freeze_support
        freeze_support()

    for tf in args.tf:
        run_tf(tf, args.horizon, args.st_period, args.st_mult,
               args.require_kalman, args.start, args.end)

    print("\nDone.")


if __name__ == "__main__":
    main()
