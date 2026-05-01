"""
1H deep-dive: ARIMA+Kalman at every bar + feature correlation study.

Fits ARIMA(2,1,0) + Kalman on HL/2 at every 1H bar.
For each bar where both models agree, computes:
  RSI(14), ADX(14), EMA50 slope, candle patterns, session/hour/weekday.

Correlates each feature against H-bar forward profit (sign-adjusted by signal
direction) and prints group-mean tables so you can see which filters add edge.

Usage
  python -m scripts.arima_1h_study
  python -m scripts.arima_1h_study --start 2022-01-01 --end 2025-12-31
  python -m scripts.arima_1h_study --horizon 10
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

LOOKBACK    = 60
ARIMA_ORDER = (2, 1, 0)
N_WORKERS   = max(1, min(12, cpu_count() - 2))


# ── Indicators ────────────────────────────────────────────────────────────────

def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    n     = len(close)
    out   = np.full(n, np.nan)
    delta = np.diff(close)
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    if period >= n: return out
    avg_g = float(np.mean(gain[:period]))
    avg_l = float(np.mean(loss[:period]))
    alpha = 1.0 / period
    for i in range(period, n - 1):
        avg_g = avg_g * (1 - alpha) + gain[i] * alpha
        avg_l = avg_l * (1 - alpha) + loss[i] * alpha
        out[i + 1] = 100.0 - 100.0 / (1.0 + avg_g / (avg_l + 1e-10))
    return out


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (adx, plus_di, minus_di)."""
    n     = len(close)
    tr    = np.empty(n);  tr[0]  = high[0] - low[0]
    dm_p  = np.empty(n);  dm_p[0] = 0.0
    dm_m  = np.empty(n);  dm_m[0] = 0.0
    for i in range(1, n):
        tr[i]   = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        up      = high[i] - high[i-1]
        dn      = low[i-1] - low[i]
        dm_p[i] = up if (up > dn and up > 0) else 0.0
        dm_m[i] = dn if (dn > up and dn > 0) else 0.0

    alpha  = 1.0 / period
    atr14  = np.empty(n); atr14[0] = tr[0]
    sdmp   = np.empty(n); sdmp[0]  = dm_p[0]
    sdmm   = np.empty(n); sdmm[0]  = dm_m[0]
    for i in range(1, n):
        atr14[i] = atr14[i-1]*(1-alpha) + tr[i]*alpha
        sdmp[i]  = sdmp[i-1]*(1-alpha)  + dm_p[i]*alpha
        sdmm[i]  = sdmm[i-1]*(1-alpha)  + dm_m[i]*alpha

    pdi = 100.0 * sdmp / (atr14 + 1e-10)
    mdi = 100.0 * sdmm / (atr14 + 1e-10)
    dx  = 100.0 * np.abs(pdi - mdi) / (pdi + mdi + 1e-10)

    adx_out = np.full(n, np.nan)
    start   = period * 2
    if start < n:
        adx_out[start] = float(np.mean(dx[period:start+1]))
        for i in range(start+1, n):
            adx_out[i] = adx_out[i-1]*(1-alpha) + dx[i]*alpha
    return adx_out, pdi, mdi


def _ema(close: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(close).ewm(span=span, adjust=False).mean().values


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> np.ndarray:
    n    = len(close)
    tr   = np.empty(n); tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr  = np.empty(n); atr[0] = tr[0]
    a    = 1.0 / period
    for i in range(1, n):
        atr[i] = atr[i-1]*(1-a) + tr[i]*a
    return atr


# ── Kalman forecast ───────────────────────────────────────────────────────────

def _kalman(series: np.ndarray, horizon: int) -> tuple[int, float]:
    """Returns (signal +1/-1, magnitude = |forecast-last|/std)."""
    var_obs = float(np.var(np.diff(series))) or 1e-10
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * var_obs * 1e-3
    R = np.array([[var_obs]])
    x = np.array([series[0], 0.0]); P = np.eye(2) * var_obs
    for obs in series:
        x = F @ x;  P = F @ P @ F.T + Q
        S = float((H @ P @ H.T)[0, 0]) + R[0, 0]
        K = (P @ H.T) / S
        x = x + K.ravel() * (obs - float(H @ x))
        P = (np.eye(2) - K @ H) @ P
    xf = x.copy()
    for _ in range(horizon): xf = F @ xf
    sig = 1 if float(xf[0]) > series[-1] else -1
    mag = abs(float(xf[0]) - series[-1]) / (float(np.std(series)) + 1e-10)
    return sig, mag


# ── Worker ────────────────────────────────────────────────────────────────────

def _fit_bar(args: tuple) -> tuple:
    bar_idx, window, horizon = args
    hl2 = (window[:, 0] + window[:, 1]) / 2.0
    cur = hl2[-1]
    a_sig = 0; a_mag = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fc    = np.asarray(ARIMA(hl2, order=ARIMA_ORDER).fit().forecast(horizon))
            a_sig = 1 if fc[-1] > cur else -1
            a_mag = abs(fc[-1] - cur) / (float(np.std(np.diff(hl2))) + 1e-10)
        except Exception:
            pass
    k_sig, k_mag = _kalman(hl2, horizon)
    return bar_idx, a_sig, k_sig, a_mag, k_mag


# ── Candle features ───────────────────────────────────────────────────────────

def _candle_features(open_: np.ndarray, high: np.ndarray,
                     low: np.ndarray, close: np.ndarray) -> dict[str, np.ndarray]:
    rng    = high - low + 1e-10
    body   = np.abs(close - open_)
    body_r = body / rng
    up_w   = (high - np.maximum(open_, close)) / rng
    dn_w   = (np.minimum(open_, close) - low)  / rng
    n      = len(close)

    bull_eng = np.zeros(n, int)
    bear_eng = np.zeros(n, int)
    for i in range(1, n):
        if close[i] > open_[i-1] and open_[i] < close[i-1] and close[i-1] < open_[i-1]:
            bull_eng[i] = 1
        if close[i] < open_[i-1] and open_[i] > close[i-1] and close[i-1] > open_[i-1]:
            bear_eng[i] = 1

    return {
        "body_ratio":    body_r,
        "upper_wick":    up_w,
        "lower_wick":    dn_w,
        "bullish":       (close > open_).astype(int),
        "doji":          (body_r < 0.1).astype(int),
        "hammer":        ((dn_w > 0.6) & (body_r < 0.3) & (up_w < 0.15)).astype(int),
        "shooting_star": ((up_w > 0.6) & (body_r < 0.3) & (dn_w < 0.15)).astype(int),
        "bull_engulf":   bull_eng,
        "bear_engulf":   bear_eng,
    }


def _session(hour: int) -> str:
    if hour < 7:  return "asian"
    if hour < 13: return "london"
    if hour < 16: return "overlap"
    if hour < 21: return "ny"
    return "quiet"


# ── Correlation helpers ───────────────────────────────────────────────────────

def _pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4: return 0.0, 1.0
    r = float(np.corrcoef(x, y)[0, 1])
    t = r * np.sqrt((n - 2) / (1 - r**2 + 1e-10))
    from scipy.stats import t as t_dist
    p = float(2 * t_dist.sf(abs(t), df=n-2))
    return r, p


# ── Main ──────────────────────────────────────────────────────────────────────

def run(start: str, end: str, horizon: int) -> None:
    print(f"Fetching EURUSD 1h  {start} → {end} ...")
    df = fetch_ohlcv("EURUSD", "1h", start, end)
    if df.empty:
        print("No data."); return
    n = len(df)
    print(f"  {n:,} bars  —  fitting ARIMA+Kalman at every bar  ({N_WORKERS} workers) ...")

    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    open_ = df["Open"].values.astype(float)

    work = [
        (i, np.stack([high[i-LOOKBACK:i], low[i-LOOKBACK:i]], axis=1), horizon)
        for i in range(LOOKBACK, n)
    ]
    with Pool(processes=N_WORKERS) as pool:
        raw = pool.map(_fit_bar, work)

    sig_map = {bar: (a, k, am, km) for bar, a, k, am, km in raw if a != 0}
    print(f"  {len(sig_map):,} bars with valid ARIMA signal")

    # indicators
    rsi_arr        = _rsi(close)
    adx_arr, pdi, mdi = _adx(high, low, close)
    ema50          = _ema(close, 50)
    atr_arr        = _atr(high, low, close)
    cndl           = _candle_features(open_, high, low, close)

    idx = df.index
    try:
        hours = idx.hour
    except AttributeError:
        hours = pd.DatetimeIndex(idx).hour
    try:
        dows = idx.dayofweek
    except AttributeError:
        dows = pd.DatetimeIndex(idx).dayofweek

    # build feature table
    rows = []
    for i in range(LOOKBACK, n - horizon):
        if i not in sig_map: continue
        a_sig, k_sig, a_mag, k_mag = sig_map[i]
        if a_sig != k_sig: continue          # only combined signals

        d   = a_sig                           # direction: +1 or -1
        ret = (close[i + horizon] - close[i]) / close[i]
        profit = ret * d                      # positive = model was right

        h       = int(hours[i])
        slope3  = (ema50[i] - ema50[max(0, i-3)]) / (ema50[max(0, i-3)] + 1e-10) * 100
        slope5  = (ema50[i] - ema50[max(0, i-5)]) / (ema50[max(0, i-5)] + 1e-10) * 100
        rsi_v   = float(rsi_arr[i]) if not np.isnan(rsi_arr[i]) else np.nan
        adx_v   = float(adx_arr[i]) if not np.isnan(adx_arr[i]) else np.nan

        rows.append({
            "i": i, "direction": d, "profit": profit, "correct": int(profit > 0),
            # model strength
            "arima_mag":    a_mag,
            "kalman_mag":   k_mag,
            "both_strong":  int(a_mag > 1.0 and k_mag > 1.0),
            # RSI
            "rsi":          rsi_v,
            "rsi_oversold":   int(rsi_v < 30)  if not np.isnan(rsi_v) else 0,
            "rsi_overbought": int(rsi_v > 70)  if not np.isnan(rsi_v) else 0,
            "rsi_extreme":    int(rsi_v < 30 or rsi_v > 70) if not np.isnan(rsi_v) else 0,
            "rsi_neutral":    int(40 < rsi_v < 60)           if not np.isnan(rsi_v) else 0,
            # ADX
            "adx":          adx_v,
            "adx_strong":   int(adx_v > 25)    if not np.isnan(adx_v) else 0,
            "adx_very_strong": int(adx_v > 40) if not np.isnan(adx_v) else 0,
            "adx_weak":     int(adx_v < 20)    if not np.isnan(adx_v) else 0,
            "di_aligned":   int((d == 1 and pdi[i] > mdi[i]) or
                                (d == -1 and mdi[i] > pdi[i])),
            # EMA50 slope
            "ema_slope3":   slope3,
            "ema_slope5":   slope5,
            "ema_aligned3": int((d == 1 and slope3 > 0) or (d == -1 and slope3 < 0)),
            "ema_aligned5": int((d == 1 and slope5 > 0) or (d == -1 and slope5 < 0)),
            # candle
            "body_ratio":   float(cndl["body_ratio"][i]),
            "upper_wick":   float(cndl["upper_wick"][i]),
            "lower_wick":   float(cndl["lower_wick"][i]),
            "candle_aligned": int((d == 1 and close[i] > open_[i]) or
                                  (d == -1 and close[i] < open_[i])),
            "doji":         int(cndl["doji"][i]),
            "hammer":       int(cndl["hammer"][i]),
            "shooting_star":int(cndl["shooting_star"][i]),
            "bull_engulf":  int(cndl["bull_engulf"][i]),
            "bear_engulf":  int(cndl["bear_engulf"][i]),
            # time
            "hour":         h,
            "session":      _session(h),
            "dow":          int(dows[i]),
        })

    df_feat = pd.DataFrame(rows)
    if df_feat.empty:
        print("No combined signals found."); return

    ns     = len(df_feat)
    n_up   = int((df_feat["direction"] == 1).sum())
    n_dn   = int((df_feat["direction"] == -1).sum())
    base_hr = df_feat["correct"].mean() * 100

    print(f"\n  Combined signals: {ns}   UP={n_up}   DOWN={n_dn}   "
          f"baseline hit-rate={base_hr:.1f}%   "
          f"mean profit={df_feat['profit'].mean()*100:+.4f}%")

    # ── Continuous correlations ───────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  CONTINUOUS FEATURE CORRELATIONS  (Pearson r vs H={horizon} forward profit)")
    print(f"{'='*68}")
    print(f"  {'Feature':<22}  {'r':>7}  {'p-value':>9}  {'sig'}")
    print(f"  {'-'*55}")
    for feat in ("arima_mag", "kalman_mag", "rsi", "adx",
                 "ema_slope3", "ema_slope5", "body_ratio", "upper_wick", "lower_wick"):
        if feat not in df_feat.columns: continue
        col = df_feat[feat].values.astype(float)
        tgt = df_feat["profit"].values.astype(float)
        r, p = _pearsonr(col, tgt)
        star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {feat:<22}  {r:>+7.3f}  {p:>9.4f}  {star}")

    # ── Binary filter table ───────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  BINARY FILTER  (hit-rate when feature=1 vs baseline={base_hr:.1f}%)")
    print(f"{'='*68}")
    print(f"  {'Filter':<35}  {'n':>5}  {'hit%':>6}  {'Δhit%':>7}  {'mean_profit':>12}")
    print(f"  {'-'*68}")
    binary_filters = [
        ("both_strong (ARIMA+K magnitude>1)",      "both_strong"),
        ("rsi_extreme (<30 or >70)",               "rsi_extreme"),
        ("rsi_oversold (<30)",                     "rsi_oversold"),
        ("rsi_overbought (>70)",                   "rsi_overbought"),
        ("rsi_neutral (40-60)",                    "rsi_neutral"),
        ("adx_strong (>25)",                       "adx_strong"),
        ("adx_very_strong (>40)",                  "adx_very_strong"),
        ("adx_weak (<20)",                         "adx_weak"),
        ("di_aligned with signal",                 "di_aligned"),
        ("ema_aligned (slope3)",                   "ema_aligned3"),
        ("ema_aligned (slope5)",                   "ema_aligned5"),
        ("candle aligned with signal",             "candle_aligned"),
        ("doji candle",                            "doji"),
        ("hammer",                                 "hammer"),
        ("shooting star",                          "shooting_star"),
        ("bullish engulfing",                      "bull_engulf"),
        ("bearish engulfing",                      "bear_engulf"),
    ]
    for label, col in binary_filters:
        if col not in df_feat.columns: continue
        sub = df_feat[df_feat[col] == 1]
        if len(sub) < 5: continue
        hr   = sub["correct"].mean() * 100
        diff = hr - base_hr
        mp   = sub["profit"].mean() * 100
        mark = " <--" if abs(diff) > 4 else ""
        print(f"  {label:<35}  {len(sub):>5}  {hr:>5.1f}%  {diff:>+6.1f}%  {mp:>+11.4f}%{mark}")

    # ── Session analysis ──────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  SESSION ANALYSIS  (UTC hours: Asian 0-7, London 7-13, Overlap 13-16, NY 16-21)")
    print(f"{'='*68}")
    print(f"  {'Session':<10}  {'n':>5}  {'hit%':>6}  {'Δhit%':>7}  {'mean_profit':>12}  {'std_profit':>11}")
    print(f"  {'-'*62}")
    for sess in ("asian", "london", "overlap", "ny", "quiet"):
        sub = df_feat[df_feat["session"] == sess]
        if len(sub) < 3: continue
        hr = sub["correct"].mean()*100
        print(f"  {sess:<10}  {len(sub):>5}  {hr:>5.1f}%  {hr-base_hr:>+6.1f}%  "
              f"{sub['profit'].mean()*100:>+11.4f}%  {sub['profit'].std()*100:>10.4f}%")

    # ── Hour-of-day ───────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  HOUR-OF-DAY  (UTC)")
    print(f"{'='*68}")
    print(f"  {'Hour':>5}  {'n':>5}  {'hit%':>6}  {'Δhit%':>7}  {'mean_profit':>12}")
    print(f"  {'-'*45}")
    for h in range(24):
        sub = df_feat[df_feat["hour"] == h]
        if len(sub) < 5: continue
        hr = sub["correct"].mean()*100
        mark = " <--" if abs(hr - base_hr) > 5 else ""
        print(f"  {h:>4}h  {len(sub):>5}  {hr:>5.1f}%  {hr-base_hr:>+6.1f}%  "
              f"{sub['profit'].mean()*100:>+11.4f}%{mark}")

    # ── Day of week ───────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  DAY-OF-WEEK")
    print(f"{'='*68}")
    print(f"  {'Day':<6}  {'n':>5}  {'hit%':>6}  {'Δhit%':>7}  {'mean_profit':>12}")
    print(f"  {'-'*45}")
    for d, name in enumerate(("Mon","Tue","Wed","Thu","Fri")):
        sub = df_feat[df_feat["dow"] == d]
        if len(sub) < 3: continue
        hr = sub["correct"].mean()*100
        print(f"  {name:<6}  {len(sub):>5}  {hr:>5.1f}%  {hr-base_hr:>+6.1f}%  "
              f"{sub['profit'].mean()*100:>+11.4f}%")

    # ── Combination filters ───────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  COMBINATION FILTERS")
    print(f"{'='*68}")
    print(f"  {'Filter':<42}  {'n':>5}  {'hit%':>6}  {'Δhit%':>7}  {'mean_profit':>12}")
    print(f"  {'-'*68}")

    def _show(label, mask):
        sub = df_feat[mask]
        if len(sub) < 5: return
        hr = sub["correct"].mean()*100
        print(f"  {label:<42}  {len(sub):>5}  {hr:>5.1f}%  {hr-base_hr:>+6.1f}%  "
              f"{sub['profit'].mean()*100:>+11.4f}%")

    _show("EMA5 aligned + ADX strong",
          (df_feat["ema_aligned5"]==1) & (df_feat["adx_strong"]==1))
    _show("EMA5 aligned + DI aligned",
          (df_feat["ema_aligned5"]==1) & (df_feat["di_aligned"]==1))
    _show("EMA5 aligned + ADX strong + DI aligned",
          (df_feat["ema_aligned5"]==1) & (df_feat["adx_strong"]==1) & (df_feat["di_aligned"]==1))
    _show("EMA5 aligned + candle aligned",
          (df_feat["ema_aligned5"]==1) & (df_feat["candle_aligned"]==1))
    _show("ADX strong + DI aligned",
          (df_feat["adx_strong"]==1) & (df_feat["di_aligned"]==1))
    _show("ADX strong + candle aligned",
          (df_feat["adx_strong"]==1) & (df_feat["candle_aligned"]==1))
    _show("EMA5 + ADX strong + candle aligned",
          (df_feat["ema_aligned5"]==1) & (df_feat["adx_strong"]==1) & (df_feat["candle_aligned"]==1))
    _show("RSI extreme + EMA5 aligned",
          (df_feat["rsi_extreme"]==1) & (df_feat["ema_aligned5"]==1))
    _show("RSI extreme + ADX strong",
          (df_feat["rsi_extreme"]==1) & (df_feat["adx_strong"]==1))
    _show("London/Overlap session",
          df_feat["session"].isin(["london","overlap"]))
    _show("London + EMA5 aligned",
          (df_feat["session"]=="london") & (df_feat["ema_aligned5"]==1))
    _show("London + ADX strong",
          (df_feat["session"]=="london") & (df_feat["adx_strong"]==1))
    _show("London + EMA5 + ADX strong",
          (df_feat["session"]=="london") & (df_feat["ema_aligned5"]==1) & (df_feat["adx_strong"]==1))
    _show("NY + EMA5 aligned",
          (df_feat["session"]=="ny") & (df_feat["ema_aligned5"]==1))
    _show("Overlap + EMA5 aligned",
          (df_feat["session"]=="overlap") & (df_feat["ema_aligned5"]==1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start",   default="2022-01-01")
    ap.add_argument("--end",     default="2025-12-31")
    ap.add_argument("--horizon", type=int, default=20)
    args = ap.parse_args()
    run(args.start, args.end, args.horizon)
    print("\nDone.")


if __name__ == "__main__":
    main()
