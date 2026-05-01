"""
Squeeze Momentum + forecasting model strategy.

Entry
  Squeeze Momentum fires (BB breaks outside KC: squeeze ON -> OFF)
  AND the chosen model predicts HL/2 will be higher (long) / lower
  (short) after H candles.

Models
  arima   ARIMA(2,1,0) on HL/2 window        -- baseline
  kalman  Kalman linear-trend filter          -- fast baseline
  rf      RandomForestClassifier (sklearn)    -- nonlinear, lagged features
  lgbm    LightGBMClassifier                  -- gradient boosting
  all     run all four and compare

Features for RF / LightGBM (built from last train_bars history)
  20 lagged HL/2 returns
  momentum over 3, 5, 10, 20, 50 bars
  rolling return volatility over 5, 10, 20 bars
  RSI(14)   ATR(14) / HL2

Exit
  Momentum histogram reverses sign  OR  squeeze turns back ON

Usage
  python -m scripts.arima_squeeze
  python -m scripts.arima_squeeze --model all --horizons 10 20
  python -m scripts.arima_squeeze --model lgbm --horizons 20
  python -m scripts.arima_squeeze --tf 1h --train-bars 600
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

ARIMA_LOOKBACK = 60
ARIMA_ORDER    = (2, 1, 0)
N_WORKERS      = max(1, min(12, cpu_count() - 2))


# ── Squeeze Momentum ──────────────────────────────────────────────────────────

def _squeeze_momentum(df: pd.DataFrame,
                      bb_period: int = 20, bb_mult: float = 2.0,
                      kc_period: int = 20, kc_mult: float = 1.5,
                      mom_period: int = 12) -> tuple[np.ndarray, np.ndarray]:
    close = df["Close"].values.astype(float)
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    n     = len(close)

    bb_sma   = pd.Series(close).rolling(bb_period).mean().values
    bb_std   = pd.Series(close).rolling(bb_period).std(ddof=0).values
    bb_upper = bb_sma + bb_mult * bb_std
    bb_lower = bb_sma - bb_mult * bb_std

    tr    = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    kc_atr   = pd.Series(tr).ewm(alpha=1.0 / kc_period, adjust=False).mean().values
    kc_sma   = pd.Series(close).rolling(kc_period).mean().values
    kc_upper = kc_sma + kc_mult * kc_atr
    kc_lower = kc_sma - kc_mult * kc_atr

    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    hh  = pd.Series(high).rolling(kc_period).max().values
    ll  = pd.Series(low).rolling(kc_period).min().values
    mid = ((hh + ll) / 2.0 + kc_sma) / 2.0
    delta = close - mid

    momentum = np.full(n, np.nan)
    for i in range(mom_period - 1, n):
        y = delta[i - mom_period + 1: i + 1]
        x = np.arange(mom_period, dtype=float)
        s, b = np.polyfit(x, y, 1)
        momentum[i] = s * (mom_period - 1) + b

    return squeeze_on, momentum


# ── Feature engineering ───────────────────────────────────────────────────────

def _make_features(hl2: np.ndarray, atr: np.ndarray, i: int) -> np.ndarray | None:
    """
    Build feature vector at bar i.
    Requires at least 51 bars of history.
    """
    if i < 51:
        return None
    ret  = np.diff(hl2[i - 51: i + 1]) / (hl2[i - 51: i] + 1e-10)   # 51 returns
    feats: list[float] = []

    # 20 lagged returns
    feats.extend(ret[-20:].tolist())

    # momentum over 3, 5, 10, 20, 50 bars
    for k in (3, 5, 10, 20, 50):
        feats.append(float((hl2[i] - hl2[i - k]) / (hl2[i - k] + 1e-10)))

    # rolling volatility of returns over 5, 10, 20 bars
    for w in (5, 10, 20):
        feats.append(float(np.std(ret[-w:])))

    # RSI(14) approximation
    gains = np.where(ret > 0, ret, 0.0)
    losses = np.where(ret < 0, -ret, 0.0)
    avg_g = float(np.mean(gains[-14:]))
    avg_l = float(np.mean(losses[-14:]))
    rsi = 100.0 - 100.0 / (1.0 + avg_g / (avg_l + 1e-10))
    feats.append(rsi)

    # ATR ratio
    feats.append(float(atr[i] / (hl2[i] + 1e-10)))

    return np.array(feats, dtype=np.float32)


# ── Model predictors ──────────────────────────────────────────────────────────

def _arima_signal(hl2_window: np.ndarray, horizon: int) -> int:
    cur = hl2_window[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fc = float(np.asarray(
                ARIMA(hl2_window, order=ARIMA_ORDER).fit().forecast(horizon)
            )[-1])
            return 1 if fc > cur else -1
        except Exception:
            return 0


def _kalman_signal(hl2_window: np.ndarray, horizon: int) -> int:
    var_obs = float(np.var(np.diff(hl2_window))) or 1e-10
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * (var_obs * 1e-3)
    R = np.array([[var_obs]])
    x = np.array([hl2_window[0], 0.0])
    P = np.eye(2) * var_obs
    for obs in hl2_window:
        x = F @ x; P = F @ P @ F.T + Q
        S = float((H @ P @ H.T)[0, 0]) + R[0, 0]
        K = (P @ H.T) / S
        x = x + K.ravel() * (obs - float(H @ x))
        P = (np.eye(2) - K @ H) @ P
    xf = x.copy()
    for _ in range(horizon):
        xf = F @ xf
    return 1 if float(xf[0]) > hl2_window[-1] else -1


def _ml_signal(model_name: str,
               X_train: np.ndarray, y_train: np.ndarray,
               x_pred: np.ndarray) -> int:
    """Train RF or LightGBM on X_train/y_train, predict direction for x_pred."""
    if len(np.unique(y_train)) < 2:
        return 0

    if model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=4,
            min_samples_leaf=5, random_state=42, n_jobs=1)
    else:  # lgbm
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, num_leaves=15,
            min_child_samples=5, learning_rate=0.05,
            random_state=42, n_jobs=1, verbose=-1)

    clf.fit(X_train, y_train)
    prob = clf.predict_proba(x_pred.reshape(1, -1))[0]
    # index 1 = UP class (y=1), index 0 = DOWN class (y=0)
    return 1 if prob[1] > 0.5 else -1


# ── Worker for ARIMA (parallel) ───────────────────────────────────────────────

def _arima_worker(args: tuple) -> tuple[int, int]:
    bar_idx, hl2_window, horizon = args
    sig = _arima_signal(hl2_window, horizon)
    return bar_idx, sig


# ── Build signals at fire bars ────────────────────────────────────────────────

def build_signals(hl2: np.ndarray, atr: np.ndarray,
                  fire_bars: list[int],
                  horizon: int,
                  models: list[str],
                  train_bars: int) -> dict[str, dict[int, int]]:
    """
    Returns {model_name: {bar_idx: signal (+1/-1)}}
    """
    results: dict[str, dict[int, int]] = {m: {} for m in models}

    # ARIMA: parallel
    if "arima" in models:
        work = [(i, hl2[max(0, i - ARIMA_LOOKBACK): i], horizon)
                for i in fire_bars if i >= ARIMA_LOOKBACK]
        with Pool(processes=N_WORKERS) as pool:
            for bar_idx, sig in pool.map(_arima_worker, work):
                if sig != 0:
                    results["arima"][bar_idx] = sig

    # Kalman: fast, sequential
    if "kalman" in models:
        for i in fire_bars:
            if i < ARIMA_LOOKBACK:
                continue
            sig = _kalman_signal(hl2[i - ARIMA_LOOKBACK: i], horizon)
            results["kalman"][i] = sig

    # RF / LightGBM: walk-forward, sequential (sklearn/lgbm already use n_jobs internally)
    for model_name in [m for m in models if m in ("rf", "lgbm")]:
        need = train_bars + horizon + 52   # 52 = feature lookback
        for i in fire_bars:
            if i < need:
                continue
            # Build training set: bars [i-train_bars-horizon .. i-horizon]
            X, y = [], []
            for t in range(i - train_bars, i - horizon):
                feat = _make_features(hl2, atr, t)
                if feat is None:
                    continue
                label = 1 if hl2[t + horizon] > hl2[t] else 0
                X.append(feat); y.append(label)

            if len(X) < 30:
                continue

            x_pred = _make_features(hl2, atr, i)
            if x_pred is None:
                continue

            sig = _ml_signal(model_name,
                              np.array(X), np.array(y), x_pred)
            results[model_name][i] = sig

    return results


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(close: np.ndarray,
             squeeze_on: np.ndarray,
             momentum: np.ndarray,
             signals: dict[int, int],
             direction: str) -> dict:
    entry_d = 1 if direction == "long" else -1
    n = len(close)
    trades = []; in_trade = False; entry_p = None

    for i in range(1, n):
        if np.isnan(momentum[i]):
            continue
        fired = bool(squeeze_on[i - 1]) and not bool(squeeze_on[i])

        if in_trade:
            mom_flip = (entry_d == 1 and momentum[i] < 0) or \
                       (entry_d == -1 and momentum[i] > 0)
            if mom_flip or bool(squeeze_on[i]):
                trades.append((close[i] / entry_p - 1) * entry_d)
                in_trade = False; entry_p = None

        if not in_trade and fired:
            mom_dir = 1 if momentum[i] > 0 else -1
            if mom_dir != entry_d:
                continue
            if signals.get(i, 0) != entry_d:
                continue
            in_trade = True; entry_p = float(close[i])

    return _stats(trades)


def simulate_no_filter(close: np.ndarray,
                       squeeze_on: np.ndarray,
                       momentum: np.ndarray,
                       direction: str) -> dict:
    """Squeeze only, no model filter -- baseline."""
    entry_d = 1 if direction == "long" else -1
    n = len(close)
    trades = []; in_trade = False; entry_p = None

    for i in range(1, n):
        if np.isnan(momentum[i]):
            continue
        fired = bool(squeeze_on[i - 1]) and not bool(squeeze_on[i])
        if in_trade:
            mom_flip = (entry_d == 1 and momentum[i] < 0) or \
                       (entry_d == -1 and momentum[i] > 0)
            if mom_flip or bool(squeeze_on[i]):
                trades.append((close[i] / entry_p - 1) * entry_d)
                in_trade = False; entry_p = None
        if not in_trade and fired:
            if (1 if momentum[i] > 0 else -1) == entry_d:
                in_trade = True; entry_p = float(close[i])

    return _stats(trades)


def _stats(trades: list) -> dict:
    if not trades:
        return {"trades": 0}
    arr = np.array(trades)
    n   = len(arr)
    sr  = float(arr.mean() / (arr.std() + 1e-10) * np.sqrt(252)) if n > 1 else 0.0
    return {"trades": n, "win_rate": float((arr > 0).mean()),
            "mean_ret": float(arr.mean()),
            "total_ret": float((1 + arr).prod() - 1), "sharpe": sr}


# ── ATR helper ────────────────────────────────────────────────────────────────

def _atr14(df: pd.DataFrame) -> np.ndarray:
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
    return pd.Series(tr).ewm(alpha=1.0 / 14, adjust=False).mean().values


# ── Hit rate ──────────────────────────────────────────────────────────────────

def print_hit_rates(hl2: np.ndarray, fire_bars: list[int],
                    all_sigs: dict[str, dict[int, int]],
                    horizon: int) -> None:
    print(f"\n  [HIT RATE at H={horizon} -- squeeze fire bars]")
    print(f"  {'Model':<12} {'n':>5}  {'hit%':>6}  {'agree_w_actual':>14}")
    print(f"  {'-'*40}")
    for model, sigs in all_sigs.items():
        hits = total = 0
        for i, sig in sigs.items():
            nxt = i + horizon
            if nxt >= len(hl2):
                continue
            actual = 1 if hl2[nxt] > hl2[i] else -1
            total += 1; hits += (sig == actual)
        if total:
            print(f"  {model:<12} {total:>5}  {hits/total*100:>5.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def _f(v, fmt=".4f"):
    return f"{v:{fmt}}" if isinstance(v, float) and not np.isnan(v) else "  n/a "


def run_tf(tf: str, horizons: list[int], models: list[str],
           bb_period: int, bb_mult: float,
           kc_period: int, kc_mult: float,
           mom_period: int, train_bars: int,
           start: str, end: str) -> None:

    print(f"\nFetching EURUSD {tf}  {start} to {end} ...")
    df = fetch_ohlcv("EURUSD", tf, start, end)
    if df.empty:
        print("  No data."); return
    print(f"  {len(df):,} bars")

    close = df["Close"].values.astype(float)
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    hl2   = (high + low) / 2.0
    atr   = _atr14(df)

    squeeze_on, momentum = _squeeze_momentum(
        df, bb_period, bb_mult, kc_period, kc_mult, mom_period)

    min_history = max(ARIMA_LOOKBACK, train_bars + max(horizons) + 52)
    fire_bars = [i for i in range(min_history, len(df) - max(horizons))
                 if bool(squeeze_on[i - 1]) and not bool(squeeze_on[i])
                 and not np.isnan(momentum[i])]

    fires_bull = sum(1 for i in fire_bars if momentum[i] > 0)
    fires_bear = len(fire_bars) - fires_bull
    print(f"  Squeeze fires: {len(fire_bars)}  "
          f"(bull={fires_bull}  bear={fires_bear})")

    for horizon in horizons:
        print(f"\n  Building signals H={horizon}  "
              f"(models: {', '.join(models)}) ...")
        all_sigs = build_signals(hl2, atr, fire_bars, horizon, models, train_bars)

        print(f"\n{'='*72}")
        print(f"  EURUSD {tf}  H={horizon}  "
              f"Squeeze(BB={bb_period}/{bb_mult}, KC={kc_period}/{kc_mult})")
        print(f"{'='*72}")

        print_hit_rates(hl2, fire_bars, all_sigs, horizon)

        print(f"\n  {'Model':<14} {'dir':<6} {'trades':>6}  {'win%':>5}  "
              f"{'mean_ret':>8}  {'total%':>7}  {'sharpe':>7}")
        print(f"  {'-'*60}")

        # Squeeze-only baseline first
        for direction in ("long", "short"):
            sim = simulate_no_filter(close, squeeze_on, momentum, direction)
            if sim["trades"]:
                print(f"  {'squeeze_only':<14} {direction:<6} {sim['trades']:>6}  "
                      f"{sim['win_rate']*100:>4.1f}%  "
                      f"{sim['mean_ret']*100:>+7.3f}%  "
                      f"{sim['total_ret']*100:>+6.1f}%  "
                      f"{sim['sharpe']:>+7.2f}")
        print()

        # Each model
        for model in models:
            sigs = all_sigs[model]
            for direction in ("long", "short"):
                sim = simulate(close, squeeze_on, momentum, sigs, direction)
                if sim["trades"]:
                    print(f"  {model:<14} {direction:<6} {sim['trades']:>6}  "
                          f"{sim['win_rate']*100:>4.1f}%  "
                          f"{sim['mean_ret']*100:>+7.3f}%  "
                          f"{sim['total_ret']*100:>+6.1f}%  "
                          f"{sim['sharpe']:>+7.2f}")
            print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tf",          nargs="+", default=["1h"])
    ap.add_argument("--horizons",    nargs="+", type=int, default=[10, 20])
    ap.add_argument("--model",       default="all",
                    choices=["arima", "kalman", "rf", "lgbm", "all"])
    ap.add_argument("--train-bars",  type=int, default=500)
    ap.add_argument("--bb-period",   type=int,   default=20)
    ap.add_argument("--bb-mult",     type=float, default=2.0)
    ap.add_argument("--kc-period",   type=int,   default=20)
    ap.add_argument("--kc-mult",     type=float, default=1.5)
    ap.add_argument("--mom-period",  type=int,   default=12)
    ap.add_argument("--start",       default="2022-01-01")
    ap.add_argument("--end",         default="2025-12-31")
    args = ap.parse_args()

    from multiprocessing import freeze_support
    freeze_support()

    models = (["arima", "kalman", "rf", "lgbm"]
              if args.model == "all" else [args.model])

    for tf in args.tf:
        run_tf(tf, args.horizons, models,
               args.bb_period, args.bb_mult,
               args.kc_period, args.kc_mult,
               args.mom_period, args.train_bars,
               args.start, args.end)

    print("\nDone.")


if __name__ == "__main__":
    main()
