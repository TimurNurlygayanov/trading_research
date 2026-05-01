"""
Time-series direction prediction correlation study -- EURUSD

Tests whether rolling model forecasts have predictive power for the actual
price direction H candles ahead.

Models
  arima_hl2    ARIMA(2,1,0) on (High+Low)/2
  sarima_hl2   SARIMA(1,1,0)(1,0,0,s) on (High+Low)/2, s=TF-adaptive
  arima_ema50  ARIMA(2,1,0) on EMA-50
  linear_hl2   OLS linear trend extrapolation on HL/2   [fast baseline]
  ets_hl2      Holt-Winters ETS (additive damped trend) on HL/2 [fast, trend-aware]
  kalman_hl2   Local linear trend Kalman filter on HL/2 [fast, adaptive]

Signal types (per model)
  endpoint  sign of forecast[H-1] - current_value
  slope     sign of linear trend across full H-step forecast

Quality filters
  ATR-14 threshold: |predicted_delta| > atr_mult*ATR14
  Signal-strength quintiles: hit rate by predicted move size

Usage
  python -m scripts.arima_direction_correlation
  python -m scripts.arima_direction_correlation --tf 1h --start 2022-01-01
  python -m scripts.arima_direction_correlation --tf 1h 5m 1m --atr-mult 0.3
  python -m scripts.arima_direction_correlation --sequential   # warm-start mode
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Pre-import once so workers don't pay import cost per call
from statsmodels.tsa.arima.model import ARIMA                    # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX           # noqa: E402
from statsmodels.tsa.holtwinters import ExponentialSmoothing     # noqa: E402

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv

# ── Global config ─────────────────────────────────────────────────────────────

HORIZONS     = [5, 20]
LOOKBACK     = 60
ARIMA_ORDER  = (2, 1, 0)
SARIMA_ORDER = (1, 1, 0)
ATR_MULT_DEFAULT = 0.1   # signal threshold as fraction of ATR14
N_WORKERS    = max(1, min(12, ((__import__("os")).cpu_count() or 4) - 2))

TF_SEASONAL = {"1m": 12, "5m": 12, "15m": 8, "1h": 24, "4h": 6, "1d": 5}

TF_STEP = {"1m": 400, "5m": 100, "15m": 40, "1h": 20, "4h": 10, "1d": 5}

TF_DATA_YEARS = {
    "1m": 0.5, "5m": 1.5, "15m": 2.0,
    "1h": 4.0, "4h": 5.0, "1d":  8.0,
}

# ── Fast model helpers ────────────────────────────────────────────────────────

def _linear_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    """OLS trend line extrapolated H steps ahead. ~0 ms."""
    n = len(series)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, series, 1)
    return slope * np.arange(n, n + horizon) + intercept


def _ets_forecast(series: np.ndarray, horizon: int) -> np.ndarray | None:
    """Holt-Winters additive damped trend. ~1-5 ms (no MLE, just smoothing)."""
    try:
        m = ExponentialSmoothing(
            series, trend="add", damped_trend=True, initialization_method="estimated"
        ).fit(optimized=True)
        return np.asarray(m.forecast(horizon))
    except Exception:
        return None


def _kalman_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    """
    Local linear trend Kalman filter -- state = [level, trend].
    Analytically exact, ~0 ms. No external dependencies.
    """
    var_obs = float(np.var(np.diff(series))) or 1e-10
    # State transition: level[t+1] = level[t] + trend[t], trend[t+1] = trend[t]
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * (var_obs * 1e-3)   # small process noise = smooth trend
    R = np.array([[var_obs]])

    x = np.array([series[0], 0.0])
    P = np.eye(2) * var_obs

    for obs in series:
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        # update
        S = float((H @ P @ H.T)[0, 0]) + R[0, 0]
        K = (P @ H.T) / S
        x = x + K.ravel() * (obs - float(H @ x))
        P = (np.eye(2) - K @ H) @ P

    out = np.empty(horizon)
    xf = x.copy()
    for k in range(horizon):
        xf = F @ xf
        out[k] = xf[0]
    return out


# ── Worker function (must be top-level for ProcessPoolExecutor pickling) ──────

def _fit_bar(args: tuple) -> dict:
    """
    Fit ARIMA-HL2, SARIMA-HL2, ARIMA-EMA50 for one bar.
    Returns dict with model forecasts as numpy arrays (or None on failure).
    """
    (i, win_hl2, win_ema50, max_h, arima_order, sarima_order, sarima_season) = args

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            fc_arima_hl2 = np.asarray(
                ARIMA(win_hl2, order=arima_order).fit().forecast(max_h)
            )
        except Exception:
            fc_arima_hl2 = None

        try:
            fc_arima_ema50 = np.asarray(
                ARIMA(win_ema50, order=arima_order).fit().forecast(max_h)
            )
        except Exception:
            fc_arima_ema50 = None

        try:
            fc_sarima = np.asarray(
                SARIMAX(
                    win_hl2,
                    order=sarima_order,
                    seasonal_order=sarima_season,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False).forecast(max_h)
            )
        except Exception:
            fc_sarima = None

    # Fast models -- no warnings needed
    fc_linear = _linear_forecast(win_hl2, max_h)
    fc_ets    = _ets_forecast(win_hl2, max_h)
    fc_kalman = _kalman_forecast(win_hl2, max_h)

    return {
        "i":            i,
        "arima_hl2":    fc_arima_hl2,
        "sarima_hl2":   fc_sarima,
        "arima_ema50":  fc_arima_ema50,
        "linear_hl2":   fc_linear,
        "ets_hl2":      fc_ets,
        "kalman_hl2":   fc_kalman,
    }


def _fit_bar_warmstart(args: tuple, prev_params: dict) -> tuple[dict, dict]:
    """Sequential variant: reuse previous fit's params as starting point."""
    (i, win_hl2, win_ema50, max_h, arima_order, sarima_order, sarima_season) = args
    result = {"i": i}
    new_params = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for name, data in [("arima_hl2", win_hl2), ("arima_ema50", win_ema50)]:
            try:
                m = ARIMA(data, order=arima_order)
                sp = prev_params.get(name)
                fit = m.fit(start_params=sp) if sp is not None else m.fit()
                result[name] = np.asarray(fit.forecast(max_h))
                new_params[name] = fit.params
            except Exception:
                result[name] = None

        try:
            sp = prev_params.get("sarima_hl2")
            m = SARIMAX(
                win_hl2,
                order=sarima_order,
                seasonal_order=sarima_season,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = m.fit(start_params=sp, disp=False) if sp is not None else m.fit(disp=False)
            result["sarima_hl2"] = np.asarray(fit.forecast(max_h))
            new_params["sarima_hl2"] = fit.params
        except Exception:
            result["sarima_hl2"] = None

    result["linear_hl2"] = _linear_forecast(win_hl2, max_h)
    result["ets_hl2"]    = _ets_forecast(win_hl2, max_h)
    result["kalman_hl2"] = _kalman_forecast(win_hl2, max_h)

    return result, new_params


# ── Signal extraction from a forecast array ───────────────────────────────────

def _extract_signals(fc: np.ndarray, cur_val: float, h: int, atr: float, atr_mult: float) -> dict:
    """
    Given forecast array (length max_h), extract signal at horizon h.
    Returns dict with 'endpoint' and 'slope' signals (+1/-1) and strengths,
    or None if signal is too weak.
    """
    threshold = atr * atr_mult

    # Endpoint: forecast[h-1] vs current
    delta_ep = float(fc[h - 1]) - cur_val
    ep_signal = 1 if delta_ep > 0 else -1
    ep_strong = abs(delta_ep) >= threshold

    # Slope: linear trend of forecast[0..h-1]
    slope = float(np.polyfit(np.arange(h), fc[:h], 1)[0])
    sl_signal = 1 if slope > 0 else -1
    sl_thresh = threshold / h   # slope threshold relative to horizon length
    sl_strong = abs(slope) >= sl_thresh

    return {
        "endpoint_signal":  ep_signal,
        "endpoint_delta":   delta_ep,
        "endpoint_strong":  ep_strong,
        "slope_signal":     sl_signal,
        "slope_val":        slope,
        "slope_strong":     sl_strong,
    }


# ── Rolling signal computation ────────────────────────────────────────────────

class Record(NamedTuple):
    i:              int
    model:          str
    signal_type:    str     # "endpoint" or "slope"
    horizon:        int
    signal:         int     # +1 / -1
    signal_strong:  bool    # passed ATR threshold
    signal_mag:     float   # |delta| / ATR  (normalised strength)
    actual_dir:     int     # +1 / -1
    actual_ret:     float


def compute_rolling_signals(
    df: pd.DataFrame,
    lookback: int,
    step: int,
    sarima_season: int,
    atr_mult: float,
    sequential: bool,
) -> pd.DataFrame:
    close  = df["Close"].values
    hl2    = ((df["High"] + df["Low"]) / 2).values
    ema50  = pd.Series(close).ewm(span=50, adjust=False).mean().values

    # ATR-14 (simplified: EWM of bar range)
    bar_range = (df["High"] - df["Low"]).values
    atr14  = pd.Series(bar_range).ewm(span=14, adjust=False).mean().values

    max_h  = max(HORIZONS)
    N      = len(df)
    sarima_season_full = (1, 0, 0, sarima_season)

    bar_indices = list(range(lookback, N - max_h, step))
    total       = len(bar_indices)
    print(f"    {total} fit-points x 6 models  ({N_WORKERS} workers) ...", flush=True)

    # Precompute args for each bar
    tasks = [
        (
            i,
            hl2[i - lookback: i].copy(),
            ema50[i - lookback: i].copy(),
            max_h,
            ARIMA_ORDER,
            SARIMA_ORDER,
            sarima_season_full,
        )
        for i in bar_indices
    ]

    if sequential:
        fitted = []
        prev_params: dict = {}
        for cnt, args in enumerate(tasks, 1):
            if cnt % max(1, total // 10) == 0:
                print(f"      {cnt}/{total}", flush=True)
            res, prev_params = _fit_bar_warmstart(args, prev_params)
            fitted.append(res)
    else:
        fitted_map: dict[int, dict] = {}
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_fit_bar, t): t[0] for t in tasks}
            done = 0
            for future in as_completed(futures):
                res = future.result()
                fitted_map[res["i"]] = res
                done += 1
                if done % max(1, total // 10) == 0:
                    print(f"      {done}/{total}", flush=True)
        fitted = [fitted_map[i] for i in bar_indices if i in fitted_map]

    # Assemble records
    rows = []
    for res in fitted:
        i = res["i"]
        if i + max_h >= N:
            continue

        cur_hl2   = float(hl2[i - 1])
        cur_ema50 = float(ema50[i - 1])
        cur_close = float(close[i - 1])
        atr_val   = float(atr14[i - 1]) if not np.isnan(atr14[i - 1]) else 1e-6

        for h in HORIZONS:
            if i + h >= N:
                continue
            fut_close  = float(close[i + h - 1])
            actual_ret = fut_close / cur_close - 1
            actual_dir = 1 if actual_ret > 0 else -1

            for model_name, fc, cur_val in [
                ("arima_hl2",   res.get("arima_hl2"),   cur_hl2),
                ("sarima_hl2",  res.get("sarima_hl2"),  cur_hl2),
                ("arima_ema50", res.get("arima_ema50"), cur_ema50),
                ("linear_hl2",  res.get("linear_hl2"),  cur_hl2),
                ("ets_hl2",     res.get("ets_hl2"),     cur_hl2),
                ("kalman_hl2",  res.get("kalman_hl2"),  cur_hl2),
            ]:
                if fc is None or np.any(np.isnan(fc)):
                    continue

                sigs = _extract_signals(fc, cur_val, h, atr_val, atr_mult)

                for stype in ("endpoint", "slope"):
                    sig    = sigs[f"{stype}_signal"]
                    strong = sigs[f"{stype}_strong"]
                    raw_v  = sigs["endpoint_delta"] if stype == "endpoint" else sigs["slope_val"]
                    mag    = abs(raw_v) / (atr_val or 1e-6)

                    rows.append(Record(
                        i=i, model=model_name, signal_type=stype,
                        horizon=h, signal=sig, signal_strong=strong,
                        signal_mag=mag, actual_dir=actual_dir, actual_ret=actual_ret,
                    ))

    return pd.DataFrame(rows)


# ── Statistics ────────────────────────────────────────────────────────────────

def correlation_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, stype, h), grp in records.groupby(["model", "signal_type", "horizon"]):
        # All signals
        _append_stats(rows, model, stype, h, "all", grp)
        # Strong signals only (ATR-filtered)
        strong = grp[grp["signal_strong"]]
        if len(strong) >= 20:
            _append_stats(rows, model, stype, h, "strong", strong)
    return pd.DataFrame(rows)


def _append_stats(rows, model, stype, h, subset, grp):
    n         = len(grp)
    base_up   = (grp["actual_dir"] == 1).mean()
    up_mask   = grp["signal"] == 1
    dn_mask   = grp["signal"] == -1

    hit_up  = (grp.loc[up_mask,  "actual_dir"] == 1).mean() if up_mask.sum() > 5  else np.nan
    hit_dn  = (grp.loc[dn_mask,  "actual_dir"] == -1).mean() if dn_mask.sum() > 5 else np.nan
    mret_up = grp.loc[up_mask, "actual_ret"].mean() if up_mask.sum() > 5 else np.nan
    edge    = (hit_up + hit_dn) / 2 - 0.5 if not (np.isnan(hit_up) or np.isnan(hit_dn)) else np.nan

    phi, pval = np.nan, np.nan
    if n >= 20:
        try:
            phi, pval = scipy_stats.pearsonr(grp["signal"], grp["actual_dir"])
        except Exception:
            pass

    rows.append({
        "model": model, "signal_type": stype, "horizon": h, "subset": subset,
        "n": n, "n_up": int(up_mask.sum()), "n_dn": int(dn_mask.sum()),
        "base_up": base_up, "hit_up": hit_up, "hit_dn": hit_dn,
        "lift_up": hit_up - base_up if not np.isnan(hit_up) else np.nan,
        "edge": edge, "mean_ret_up": mret_up,
        "phi": phi, "phi_p": pval,
    })


def strength_quintiles(records: pd.DataFrame) -> pd.DataFrame:
    """For each (model, signal_type, horizon), bin by signal_mag and report hit rate."""
    rows = []
    for (model, stype, h), grp in records.groupby(["model", "signal_type", "horizon"]):
        sub = grp[grp["signal"] == 1].copy()
        if len(sub) < 50:
            continue
        sub["q"] = pd.qcut(sub["signal_mag"], 5, labels=False, duplicates="drop")
        for q in sorted(sub["q"].dropna().unique()):
            g = sub[sub["q"] == q]
            rows.append({
                "model": model, "signal_type": stype, "horizon": h,
                "quintile": int(q) + 1,
                "n": len(g),
                "mag_lo": round(g["signal_mag"].min(), 3),
                "mag_hi": round(g["signal_mag"].max(), 3),
                "hit_up": (g["actual_dir"] == 1).mean(),
            })
    return pd.DataFrame(rows)


# ── Strategy simulation ───────────────────────────────────────────────────────

def _build_eod_set(df: pd.DataFrame) -> set[int]:
    """Return set of bar indices that are the last bar of each trading day (UTC)."""
    dates = df.index.normalize()
    eod = set()
    for i in range(len(df) - 1):
        if dates[i] != dates[i + 1]:
            eod.add(i)
    eod.add(len(df) - 1)  # last bar of dataset
    return eod


def simulate_strategy(df: pd.DataFrame, records: pd.DataFrame,
                      model: str, stype: str, horizon: int,
                      strong_only: bool = False,
                      eod_close: bool = False) -> dict:
    """
    Hold long while signal=+1, flip on -1. Enter/exit at close of signal bar.
    If eod_close=True, also force-exit at end of each trading day.
    """
    sub = records[
        (records["model"] == model) &
        (records["signal_type"] == stype) &
        (records["horizon"] == horizon)
    ].copy()
    if strong_only:
        sub = sub[sub["signal_strong"]]
    if sub.empty:
        return {}

    close    = df["Close"].values
    bars     = sub["i"].values
    sigs     = sub["signal"].values
    eod_set  = _build_eod_set(df) if eod_close else set()

    trades = []
    in_trade, entry_close, entry_bar = False, None, None

    # Build a lookup: bar_index -> signal (only at signal bars)
    sig_map = {b: s for b, s in zip(bars, sigs)}
    all_bars = sorted(sig_map)

    for b in all_bars:
        s = sig_map[b]
        # Enter/exit at next bar's close (b is 1-indexed; close[b] is the bar AFTER signal)
        bar_close = float(close[b]) if b < len(close) else float(close[b - 1])

        # EOD force-exit
        if in_trade and eod_close:
            # check if any EOD bar falls between entry_bar and b (exclusive)
            for eod_b in range(entry_bar + 1, b):
                if eod_b in eod_set:
                    eod_close_price = float(close[eod_b])
                    trades.append(eod_close_price / entry_close - 1)
                    in_trade = False
                    entry_close = None
                    break

        if not in_trade and s == 1:
            in_trade = True
            entry_close = bar_close
            entry_bar = b
        elif in_trade and s == -1:
            trades.append(bar_close / entry_close - 1)
            in_trade, entry_close, entry_bar = False, None, None

    if not trades:
        return {"trades": 0}
    arr = np.array(trades)
    return {
        "trades":    len(arr),
        "win_rate":  float((arr > 0).mean()),
        "mean_ret":  float(arr.mean()),
        "total_ret": float((1 + arr).prod() - 1),
    }


# ── Pretty print ──────────────────────────────────────────────────────────────

def _f(v, fmt=".4f"):
    return f"{v:{fmt}}" if isinstance(v, float) and not np.isnan(v) else "  n/a "


def print_results(tf: str, corr: pd.DataFrame, quint: pd.DataFrame,
                  records: pd.DataFrame, df: pd.DataFrame):
    n_pts = len(records) // 12  # 3 models x 2 signal_types x 2 horizons
    print(f"\n{'='*90}")
    print(f"  EURUSD  {tf}  --  ~{n_pts} fit-points  ({len(df):,} bars)")
    print(f"{'='*90}")

    for subset in ("all", "strong"):
        sub = corr[corr["subset"] == subset]
        if sub.empty:
            continue
        label = "ALL SIGNALS" if subset == "all" else "STRONG SIGNALS (ATR-filtered)"
        print(f"\n  [{label}]")
        print(f"  {'Model':<14} {'sig':>5} {'H':>3} {'n':>5}  "
              f"{'base':>6} {'hit_up':>6} {'hit_dn':>6} "
              f"{'lift':>6} {'edge':>6} {'mret_up':>8}  {'phi':>6}  {'p':>8}")
        print(f"  {'-'*82}")
        for _, r in sub.sort_values(["horizon", "model", "signal_type"]).iterrows():
            sig_str = "***" if (not np.isnan(r["phi_p"]) and r["phi_p"] < 0.001) else \
                      "**"  if (not np.isnan(r["phi_p"]) and r["phi_p"] < 0.01)  else \
                      "*"   if (not np.isnan(r["phi_p"]) and r["phi_p"] < 0.05)  else ""
            print(
                f"  {r['model']:<14} {r['signal_type']:>5} {int(r['horizon']):>3} {int(r['n']):>5}  "
                f"{_f(r['base_up']):>6} {_f(r['hit_up']):>6} {_f(r['hit_dn']):>6} "
                f"{_f(r['lift_up']):>6} {_f(r['edge']):>6} {_f(r['mean_ret_up'],'.5f'):>8}  "
                f"{_f(r['phi']):>6}  {_f(r['phi_p'],'.2e'):>8} {sig_str}"
            )

    # Strength quintiles (UP signal hit rate by signal magnitude)
    if not quint.empty:
        print(f"\n  [HIT RATE BY SIGNAL STRENGTH -- UP predictions only]")
        print(f"  {'Model':<14} {'sig':>5} {'H':>3}  "
              f"  Q1(weak)    Q2          Q3          Q4          Q5(strong)")
        print(f"  {'-'*82}")
        for (model, stype, h), g in quint.groupby(["model", "signal_type", "horizon"]):
            qs = g.sort_values("quintile")["hit_up"].tolist()
            qs_str = "  ".join(f"{v:.3f}" if not np.isnan(v) else " n/a " for v in qs)
            print(f"  {model:<14} {stype:>5} {int(h):>3}  {qs_str}")

    # Strategy sim — all models, strong-only, with and without EOD close
    ALL_MODELS = ("arima_hl2", "sarima_hl2", "arima_ema50",
                  "linear_hl2", "ets_hl2", "kalman_hl2")
    print(f"\n  [STRATEGY SIMULATION -- strong signals, hold vs EOD-close comparison]")
    print(f"  {'Model':<14} {'sig':>5} {'H':>3}  "
          f"  ---hold-long---              ---EOD-close---")
    print(f"  {'':14} {'':5} {'':3}  "
          f"  {'tr':>4} {'win%':>5} {'tot%':>7}    "
          f"  {'tr':>4} {'win%':>5} {'tot%':>7}")
    print(f"  {'-'*82}")
    for (model, stype, h) in [
        (m, st, hh)
        for hh in HORIZONS
        for m in ALL_MODELS
        for st in ("endpoint", "slope")
    ]:
        sim_hold = simulate_strategy(df, records, model, stype, h, strong_only=True)
        sim_eod  = simulate_strategy(df, records, model, stype, h, strong_only=True,
                                     eod_close=True)
        if (not sim_hold or sim_hold["trades"] == 0) and \
           (not sim_eod  or sim_eod["trades"] == 0):
            continue
        def _s(sim):
            if not sim or sim["trades"] == 0:
                return f"{'--':>4} {'--':>5} {'--':>7}"
            return (f"{sim['trades']:>4}  {sim['win_rate']*100:>4.1f}%"
                    f"  {sim['total_ret']*100:>+6.1f}%")
        print(f"  {model:<14} {stype:>5} {int(h):>3}    {_s(sim_hold)}      {_s(sim_eod)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbol",     default="EURUSD")
    ap.add_argument("--tf",         nargs="+", default=["1h", "5m", "1m"])
    ap.add_argument("--start",      default=None)
    ap.add_argument("--end",        default="2025-12-31")
    ap.add_argument("--lookback",   type=int, default=LOOKBACK)
    ap.add_argument("--step",       type=int, default=None)
    ap.add_argument("--atr-mult",   type=float, default=ATR_MULT_DEFAULT,
                    help="Signal threshold as fraction of ATR14 (default 0.1)")
    ap.add_argument("--sequential", action="store_true",
                    help="Sequential warm-start mode instead of parallel")
    ap.add_argument("--out",        default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for tf in args.tf:
        years = TF_DATA_YEARS.get(tf, 2.0)
        start = args.start or str(
            pd.Timestamp(args.end) - pd.DateOffset(years=years)
        )[:10]
        step     = args.step or TF_STEP.get(tf, 20)
        seasonal = TF_SEASONAL.get(tf, 12)

        print(f"\nFetching {args.symbol} {tf}  {start} to {args.end}")
        df = fetch_ohlcv(args.symbol, tf, start, args.end)
        if df.empty:
            print("  No data - skipping.")
            continue
        print(f"  {len(df):,} bars  |  step={step}  lookback={args.lookback}"
              f"  sarima_s={seasonal}  atr_mult={args.atr_mult}")

        records = compute_rolling_signals(
            df, args.lookback, step, seasonal, args.atr_mult, args.sequential
        )
        if records.empty:
            print("  No records produced - skipping.")
            continue

        corr  = correlation_summary(records)
        quint = strength_quintiles(records)

        print_results(tf, corr, quint, records, df)

        base = f"arima_corr_{args.symbol}_{tf}"
        corr.to_csv(out_dir / f"{base}_summary.csv", index=False)
        records.to_csv(out_dir / f"{base}_records.csv", index=False)
        quint.to_csv(out_dir / f"{base}_quintiles.csv", index=False)
        print(f"\n  -> saved {base}_summary/records/quintiles.csv")

    print("\nDone.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # required for Windows frozen executables
    main()
