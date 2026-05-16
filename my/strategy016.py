"""
Strategy 016 — Regime-conditioned RF buy classifier on EURUSD 5m.
Variant of 014 with 1:5 risk:reward (TP = 100 pips, SL = 20 pips).

Pipeline:
  1. Load 2025 5m data (train) and 2026 5m data (test) from MT5.
  2. Compute per-bar features:
        RSI(14), EMA(50) slope, ATR(14),
        (Close - rolling_max{10,20,100}) / ATR,
        (Close - rolling_min{10,20,100}) / ATR,
        (Close - HL2)                    / ATR
  3. Compute per-bar k-means features over past 200 bars:
        F1 = (max(Close) - min(Close)) / ATR
        F2 = mean( Close[s+20] > Close[s] ) for s in [t-199, t-20]
  4. Per-bar binary label: 1 if High >= entry + TP before Low <= entry - SL
     within 200 forward bars (SL priority on same-bar tie).
        TP = 100 pips, SL = 20 pips, 1-pip spread deducted at exit.
  5. Fit StandardScaler + KMeans(k=3) on 2025 k-means features.
  6. For each cluster: fit a RandomForestClassifier on per-bar features
     using only 2025 bars in that cluster (label = step 4).
  7. On 2026: assign each bar to a cluster, predict P(profitable) with
     that cluster's RF, sweep probability thresholds, simulate trades,
     report stats.

The exit rule used for both labelling and live simulation is the same:
TP = +100 pips, SL = -20 pips, max hold = 200 bars. Spread (1 pip) is
deducted at exit time.
"""
import sys
import time

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta  # noqa: F401  registers df.ta accessor

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PIP_SIZE     = 0.0001
SPREAD_PIPS  = 1.0
TP_PIPS      = 100.0
SL_PIPS      = 20.0
MAX_HOLD     = 200
KM_WINDOW    = 200
FWD_WIN      = 20      # forward-shift used in km-feature F2 and nowhere else
N_CLUSTERS   = 3
RF_TREES     = 200
RF_MAX_DEPTH = 8
RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_TF_M5 = 5


def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end   = pd.Timestamp(end,   tz="UTC")
    mt5.initialize()
    mt5.symbol_select(ticker, True)

    CHUNK = 50_000
    frames, pos = [], 0
    while True:
        chunk = mt5.copy_rates_from_pos(ticker, _TF_M5, pos, CHUNK)
        if chunk is None or len(chunk) == 0:
            break
        frames.append(pd.DataFrame(chunk))
        oldest = pd.Timestamp(int(frames[-1]["time"].min()), unit="s", tz="UTC")
        if oldest <= ts_start or len(chunk) < CHUNK:
            break
        pos += CHUNK

    df = pd.concat(frames[::-1]).drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(columns={
        "open": "Open", "high": "High",
        "low": "Low",   "close": "Close", "tick_volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]]


# -----------------------------------------------------------------------------
# Indicators
# -----------------------------------------------------------------------------
def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    tr = np.empty(len(close))
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr[1:] = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prev_close),
        np.abs(low[1:]  - prev_close),
    ])
    atr = np.empty(len(close))
    atr[:n] = np.nan
    atr[n - 1] = tr[:n].mean()
    for i in range(n, len(close)):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    diff = np.diff(close, prepend=close[0])
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_g = np.empty(len(close)); avg_g[:n] = np.nan
    avg_l = np.empty(len(close)); avg_l[:n] = np.nan
    avg_g[n - 1] = gain[1:n].mean()
    avg_l[n - 1] = loss[1:n].mean()
    for i in range(n, len(close)):
        avg_g[i] = (avg_g[i - 1] * (n - 1) + gain[i]) / n
        avg_l[i] = (avg_l[i - 1] * (n - 1) + loss[i]) / n
    rs  = np.where(avg_l == 0, np.inf, avg_g / np.where(avg_l == 0, 1, avg_l))
    rsi = 100 - 100 / (1 + rs)
    return rsi


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1.0)
    out = np.empty(len(x)); out[:n - 1] = np.nan
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------
PER_BAR_FEATURES = [
    "rsi14", "ema50_slope", "atr14",
    "dmax10", "dmax20", "dmax100",
    "dmin10", "dmin20", "dmin100",
    "hl2_pos",
]

KM_FEATURES = ["range_atr", "up_frac"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].values
    high  = out["High"].values
    low   = out["Low"].values

    atr = _wilder_atr(high, low, close, n=14)
    rsi = _rsi(close, n=14)
    ema50 = _ema(close, n=50)

    out["atr14"] = atr
    out["rsi14"] = rsi
    ema50_s = pd.Series(ema50, index=out.index)
    atr_safe = pd.Series(atr, index=out.index).replace(0.0, np.nan)
    out["ema50_slope"] = ((ema50_s - ema50_s.shift(5)) / atr_safe).values

    for w in (10, 20, 100):
        rmax = pd.Series(close).rolling(w).max().values
        rmin = pd.Series(close).rolling(w).min().values
        out[f"dmax{w}"] = (close - rmax) / atr
        out[f"dmin{w}"] = (close - rmin) / atr

    hl2 = (high + low) / 2.0
    out["hl2_pos"] = (close - hl2) / atr

    # ------------------- km features (window = past 200 bars) ----------------
    rmax_w = pd.Series(close).rolling(KM_WINDOW).max().values
    rmin_w = pd.Series(close).rolling(KM_WINDOW).min().values
    out["range_atr"] = (rmax_w - rmin_w) / atr

    # F2: fraction of bars s in [t-199, t-20] with Close[s+20] > Close[s]
    fwd_up = (pd.Series(close).shift(-FWD_WIN) > pd.Series(close)).astype(float).values
    # rolling mean over 200 bars, then shift by FWD_WIN so the window is fully
    # in the past (no lookahead at bar t).
    out["up_frac"] = (
        pd.Series(fwd_up).rolling(KM_WINDOW - FWD_WIN).mean().shift(FWD_WIN).values
    )

    return out


# -----------------------------------------------------------------------------
# Labels: TP-before-SL within MAX_HOLD bars
# -----------------------------------------------------------------------------
def compute_labels(df: pd.DataFrame) -> np.ndarray:
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(df)
    tp_d  = TP_PIPS * PIP_SIZE
    sl_d  = SL_PIPS * PIP_SIZE

    labels = np.full(n, np.nan)
    for i in range(n - MAX_HOLD - 1):
        entry = close[i]
        tp_px = entry + tp_d
        sl_px = entry - sl_d
        # Conservative tie-break: if a bar's range covers both TP and SL, count SL.
        hit = 0
        for k in range(1, MAX_HOLD + 1):
            j = i + k
            sl_hit = low[j]  <= sl_px
            tp_hit = high[j] >= tp_px
            if sl_hit:
                hit = 0
                break
            if tp_hit:
                hit = 1
                break
        labels[i] = hit
    return labels


# -----------------------------------------------------------------------------
# Live simulation
# -----------------------------------------------------------------------------
def simulate(
    df: pd.DataFrame,
    enter: np.ndarray,
) -> dict:
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(df)
    tp_d  = TP_PIPS * PIP_SIZE
    sl_d  = SL_PIPS * PIP_SIZE

    pos      = False
    entry_px = 0.0
    entry_i  = 0
    trades: list[dict] = []

    for i in range(n):
        if pos:
            sl_px = entry_px - sl_d
            tp_px = entry_px + tp_d
            sl_hit = low[i]  <= sl_px
            tp_hit = high[i] >= tp_px
            held   = i - entry_i
            if sl_hit:
                pips = (sl_px - entry_px) / PIP_SIZE - SPREAD_PIPS
                trades.append({"pips": pips, "hold": held, "exit": "sl"})
                pos = False
                continue
            if tp_hit:
                pips = (tp_px - entry_px) / PIP_SIZE - SPREAD_PIPS
                trades.append({"pips": pips, "hold": held, "exit": "tp"})
                pos = False
                continue
            if held >= MAX_HOLD:
                pips = (close[i] - entry_px) / PIP_SIZE - SPREAD_PIPS
                trades.append({"pips": pips, "hold": held, "exit": "time"})
                pos = False
                continue

        if not pos and enter[i]:
            pos      = True
            entry_px = close[i]
            entry_i  = i

    if not trades:
        return {"n": 0}
    pips = np.array([t["pips"] for t in trades])
    cum  = np.cumsum(pips)
    days = max(1, (df.index[-1] - df.index[0]).days)
    wins = pips > 0
    sharpe = (pips.mean() / pips.std() * np.sqrt(len(trades) / (days / 365.25))
              if len(trades) > 1 and pips.std() > 0 else 0.0)
    return {
        "n":        len(trades),
        "pips":     float(pips.sum()),
        "win_pct":  float(wins.mean() * 100),
        "avg_win":  float(pips[wins].mean())  if wins.any()  else 0.0,
        "avg_loss": float(pips[~wins].mean()) if (~wins).any() else 0.0,
        "max_dd":   float((cum - np.maximum.accumulate(cum)).min()),
        "avg_hold": float(np.mean([t["hold"] for t in trades])),
        "sharpe":   float(sharpe),
        "tp_pct":   float(np.mean([t["exit"] == "tp"   for t in trades]) * 100),
        "sl_pct":   float(np.mean([t["exit"] == "sl"   for t in trades]) * 100),
        "time_pct": float(np.mean([t["exit"] == "time" for t in trades]) * 100),
    }


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def prepare(df: pd.DataFrame, *, want_labels: bool) -> pd.DataFrame:
    feat = add_features(df)
    if want_labels:
        feat["label"] = compute_labels(feat)
    # Drop rows missing any feature we'll use. Labels (if present) may be NaN
    # in the tail; we handle that when slicing the training set.
    needed = PER_BAR_FEATURES + KM_FEATURES
    feat = feat.dropna(subset=needed).copy()
    return feat


def train(df_train: pd.DataFrame):
    print(f"[train] preparing features on {len(df_train)} bars...")
    t0 = time.time()
    feat = prepare(df_train, want_labels=True)
    feat = feat.dropna(subset=["label"]).copy()
    feat["label"] = feat["label"].astype(int)
    print(f"[train] feature matrix: {len(feat)} rows ({time.time() - t0:.1f}s)")

    X_km = feat[KM_FEATURES].values
    scaler = StandardScaler().fit(X_km)
    X_km_s = scaler.transform(X_km)

    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE).fit(X_km_s)
    feat["cluster"] = km.labels_

    print("\n[train] cluster summary (on 2025):")
    summary = feat.groupby("cluster").agg(
        n=("label", "size"),
        win_rate=("label", "mean"),
        range_atr_mean=("range_atr", "mean"),
        up_frac_mean=("up_frac", "mean"),
    )
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    rfs: dict[int, RandomForestClassifier] = {}
    for c in range(N_CLUSTERS):
        sub = feat[feat["cluster"] == c]
        if len(sub) < 100:
            print(f"[train] cluster {c}: too few rows ({len(sub)}), skipping RF")
            continue
        X = sub[PER_BAR_FEATURES].values
        y = sub["label"].values
        rf = RandomForestClassifier(
            n_estimators=RF_TREES,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=50,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ).fit(X, y)
        oob_msg = ""
        rfs[c] = rf
        print(f"[train] cluster {c}: RF fit on {len(sub)} rows, "
              f"base rate={y.mean():.3f}{oob_msg}")
    return scaler, km, rfs


def test(df_test: pd.DataFrame, scaler, km, rfs, thresholds=(0.55, 0.60, 0.65, 0.70, 0.75)):
    print(f"\n[test] preparing features on {len(df_test)} bars...")
    feat = prepare(df_test, want_labels=False)
    print(f"[test] feature matrix: {len(feat)} rows")

    X_km_s = scaler.transform(feat[KM_FEATURES].values)
    feat["cluster"] = km.predict(X_km_s)

    # Predict P(label=1) using the cluster-specific RF.
    probs = np.full(len(feat), np.nan)
    X_bar = feat[PER_BAR_FEATURES].values
    cl    = feat["cluster"].values
    for c, rf in rfs.items():
        mask = cl == c
        if mask.any():
            probs[mask] = rf.predict_proba(X_bar[mask])[:, 1]
    feat["prob"] = probs

    print("\n[test] 2026 cluster distribution:")
    print(feat["cluster"].value_counts().sort_index().to_string())

    hdr = (f"{'thr':>5} {'n':>5} {'pips':>8} {'sharpe':>7} {'win%':>6} "
           f"{'avgW':>6} {'avgL':>7} {'maxDD':>7} {'hold':>5} "
           f"{'tp%':>5} {'sl%':>5} {'tim%':>5}")
    print(f"\n[test] overall by probability threshold:")
    print(hdr)
    print("-" * len(hdr))
    for thr in thresholds:
        enter = (feat["prob"].values >= thr)
        m = simulate(feat, enter)
        if m.get("n", 0) == 0:
            print(f"{thr:>5.2f} {'-':>5}")
            continue
        print(f"{thr:>5.2f} {m['n']:>5} {m['pips']:>+8.1f} "
              f"{m['sharpe']:>+7.2f} {m['win_pct']:>5.1f}% "
              f"{m['avg_win']:>+6.1f} {m['avg_loss']:>+7.1f} "
              f"{m['max_dd']:>+7.1f} {m['avg_hold']:>5.1f} "
              f"{m['tp_pct']:>4.1f}% {m['sl_pct']:>4.1f}% {m['time_pct']:>4.1f}%")

    print(f"\n[test] per-cluster at thr=0.60:")
    print(hdr.replace("thr", "clu"))
    print("-" * len(hdr))
    for c in sorted(rfs.keys()):
        sub_mask  = (feat["cluster"].values == c)
        enter     = sub_mask & (feat["prob"].values >= 0.60)
        m = simulate(feat, enter)
        if m.get("n", 0) == 0:
            print(f"{c:>5d} {'-':>5}")
            continue
        print(f"{c:>5d} {m['n']:>5} {m['pips']:>+8.1f} "
              f"{m['sharpe']:>+7.2f} {m['win_pct']:>5.1f}% "
              f"{m['avg_win']:>+6.1f} {m['avg_loss']:>+7.1f} "
              f"{m['max_dd']:>+7.1f} {m['avg_hold']:>5.1f} "
              f"{m['tp_pct']:>4.1f}% {m['sl_pct']:>4.1f}% {m['time_pct']:>4.1f}%")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    TICKER = "EURUSD"
    print(f"Loading {TICKER} 5m train (2025)...")
    df_train = get_data(TICKER, "2025-01-01", "2026-01-01")
    print(f"  train bars: {len(df_train)}  ({df_train.index[0]} .. {df_train.index[-1]})")

    print(f"Loading {TICKER} 5m test (2026)...")
    df_test = get_data(TICKER, "2026-01-01", "2026-06-01")
    print(f"  test bars: {len(df_test)}  ({df_test.index[0]} .. {df_test.index[-1]})")

    scaler, km, rfs = train(df_train)
    test(df_test, scaler, km, rfs)
