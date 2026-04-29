"""
Price Level Bounce - EURUSD 5m, 7AM-4PM Cyprus time, long only.

Levels used as both support and resistance:
  • EMA 21, EMA 50, EMA 200  (dynamic)
  • Confirmed swing pivot highs / lows  (pivot_n bars each side for confirmation)
  • Previous day high / low
  • Round numbers (50-pip grid by default)

Entry condition (long only):
  Close is between a support level (below) and resistance level (above).
  SL  = closest support - atr_sl_mult x ATR(14)   [default 0.5]
  TP  = closest resistance
  RR  = (TP - close) / (close - SL) >= min_rr      [default 5.0]
  -> BUY at the next bar's open.

Session gate uses Cyprus local time (EET/EEST via Asia/Nicosia tz).

What the script produces:
  level_bounce_all_*.csv   - every signal independently simulated -> RF training set
  level_bounce_seq_*.csv   - sequential one-position-at-a-time realistic backtest
  level_bounce_rf_*.csv    - RF-filtered signals (if --no-rf not set)

Phase 2 (integrated):
  RandomForestClassifier trained on IS data (default pre-2025),
  evaluated OOS. Signals filtered by predicted win probability.

Usage:
  python -m scripts.level_bounce_rf_backtest
  python -m scripts.level_bounce_rf_backtest --start 2021-01-01 --end 2025-12-31
  python -m scripts.level_bounce_rf_backtest --min-rr 5 --rf-test-start 2025-01-01
  python -m scripts.level_bounce_rf_backtest --no-rf
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


# -- timezone ------------------------------------------------------------------
def _cy_hours(df: pd.DataFrame) -> np.ndarray:
    """Return Cyprus local hour for each bar (handles UTC-naive or tz-aware index)."""
    idx = df.index
    try:
        idx_utc = idx.tz_localize("UTC")
    except TypeError:
        idx_utc = idx.tz_convert("UTC")
    return idx_utc.tz_convert("Asia/Nicosia").hour.values


# -- indicators ----------------------------------------------------------------
def add_features(df: pd.DataFrame, slope_n: int = 5,
                 atr_len: int = 14, rsi_len: int = 14) -> pd.DataFrame:
    out = df.copy()
    out["ema21"]  = ta.ema(out["Close"], length=21)
    out["ema50"]  = ta.ema(out["Close"], length=50)
    out["ema200"] = ta.ema(out["Close"], length=200)
    out["atr"]    = ta.atr(out["High"], out["Low"], out["Close"], length=atr_len)
    out["rsi"]    = ta.rsi(out["Close"], length=rsi_len)
    out["ema21_slope"]  = (out["ema21"]  / out["ema21"].shift(slope_n)  - 1) * 10_000
    out["ema50_slope"]  = (out["ema50"]  / out["ema50"].shift(slope_n)  - 1) * 10_000
    out["ema200_slope"] = (out["ema200"] / out["ema200"].shift(slope_n) - 1) * 10_000
    return out


# -- pivot detection ----------------------------------------------------------─
def precompute_pivots(df: pd.DataFrame, pivot_n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised pivot detection using a centered rolling window of width 2*pivot_n+1.
    pivot_highs[j] / pivot_lows[j] - True iff that bar is a confirmed local extremum.
    Callers must enforce the causal constraint (only access j <= current_bar - pivot_n).
    """
    high_s = pd.Series(df["High"].values)
    low_s  = pd.Series(df["Low"].values)
    w = 2 * pivot_n + 1
    roll_max = high_s.rolling(w, center=True, min_periods=pivot_n + 1).max()
    roll_min = low_s.rolling(w,  center=True, min_periods=pivot_n + 1).min()
    pivot_highs = (high_s == roll_max).values
    pivot_lows  = (low_s  == roll_min).values
    return pivot_highs, pivot_lows


# -- daily H/L ----------------------------------------------------------------─
def precompute_daily_hl(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two arrays (prev_day_high, prev_day_low) indexed by bar, NaN where
    no previous day exists. Uses UTC day boundaries (close enough for levels).
    """
    n     = len(df)
    pdh   = np.full(n, np.nan)
    pdl   = np.full(n, np.nan)
    daily = df.resample("1D").agg({"High": "max", "Low": "min"})
    # Map each bar to its UTC date, then look up prev-day stats
    bar_dates = df.index.normalize()          # truncate to date, tz-aware or naive
    for i in range(1, len(daily)):
        prev_h = float(daily["High"].iloc[i - 1])
        prev_l = float(daily["Low"].iloc[i - 1])
        day    = daily.index[i]
        mask   = bar_dates == day
        pdh[mask] = prev_h
        pdl[mask] = prev_l
    return pdh, pdl


# -- signal detection ----------------------------------------------------------
def detect_signals(df: pd.DataFrame, cy_hours_arr: np.ndarray,
                   session_start: int, session_end: int,
                   pivot_highs: np.ndarray, pivot_lows: np.ndarray,
                   prev_day_h: np.ndarray, prev_day_l: np.ndarray,
                   pivot_n: int, swing_lookback: int,
                   round_step: float, min_rr: float,
                   atr_sl_mult: float, min_level_dist_atr: float) -> pd.DataFrame:
    # Pre-extract everything as numpy - no df.iloc inside the loop
    close      = df["Close"].values
    high       = df["High"].values
    low        = df["Low"].values
    atr_v      = df["atr"].values
    rsi_v      = df["rsi"].values
    e21        = df["ema21"].values
    e50        = df["ema50"].values
    e200       = df["ema200"].values
    e21s       = df["ema21_slope"].values
    e50s       = df["ema50_slope"].values
    e200s      = df["ema200_slope"].values
    dow        = df.index.dayofweek.values
    timestamps = df.index
    n          = len(df)

    # Pivot prices as value arrays (NaN where bar is not a pivot)
    piv_h = np.where(pivot_highs, high, np.nan)
    piv_l = np.where(pivot_lows,  low,  np.nan)

    rows: list[dict] = []
    for i in range(n - 1):   # -1: need next bar for entry
        if (np.isnan(e200[i]) or np.isnan(atr_v[i]) or np.isnan(rsi_v[i])
                or atr_v[i] <= 0):
            continue
        if not (session_start <= cy_hours_arr[i] < session_end):
            continue

        atr      = atr_v[i]
        c        = close[i]
        min_dist = min_level_dist_atr * atr

        # -- collect all level prices into a small list ----------------------
        lvls: list[float] = []

        # EMA levels (already numpy scalars - fast)
        if not np.isnan(e21[i]):  lvls.append(e21[i])
        if not np.isnan(e50[i]):  lvls.append(e50[i])
        if not np.isnan(e200[i]): lvls.append(e200[i])

        # Confirmed swing pivots in [i - swing_lookback, i - pivot_n]
        s = max(0, i - swing_lookback)
        e = max(s, i - pivot_n + 1)          # causal confirmation constraint
        if e > s:
            ph_sl = piv_h[s:e]
            pl_sl = piv_l[s:e]
            valid_h = ph_sl[~np.isnan(ph_sl)]
            valid_l = pl_sl[~np.isnan(pl_sl)]
            if valid_h.size: lvls.extend(valid_h.tolist())
            if valid_l.size: lvls.extend(valid_l.tolist())

        # Previous day H/L (pre-computed arrays - O(1) lookup)
        if not np.isnan(prev_day_h[i]): lvls.append(prev_day_h[i])
        if not np.isnan(prev_day_l[i]): lvls.append(prev_day_l[i])

        # Round numbers (50-pip grid, 3 steps each side)
        base = round(c / round_step) * round_step
        for k in range(-3, 4):
            lvls.append(base + k * round_step)

        if not lvls:
            continue

        lvls_arr = np.unique(np.round(lvls, 5))

        # Find closest support below and resistance above
        below = lvls_arr[lvls_arr < c - min_dist]
        above = lvls_arr[lvls_arr > c + min_dist]
        if not below.size or not above.size:
            continue

        support    = below[-1]   # max below close
        resistance = above[0]    # min above close

        sl_price = support - atr_sl_mult * atr
        risk     = c - sl_price
        if risk <= 0:
            continue
        rr = (resistance - c) / risk
        if rr < min_rr:
            continue

        rows.append({
            "bar_idx":              i,
            "signal_time":          timestamps[i],
            "support":              round(support, 5),
            "resistance":           round(resistance, 5),
            "sl_price":             round(sl_price, 5),
            "tp_price":             round(resistance, 5),
            "rr":                   round(rr, 2),
            "ema21":                e21[i],
            "ema50":                e50[i],
            "ema200":               e200[i],
            "ema21_slope":          e21s[i],
            "ema50_slope":          e50s[i],
            "ema200_slope":         e200s[i],
            "rsi":                  rsi_v[i],
            "atr":                  atr,
            "atr_frac":             atr / c,
            "dist_support_atr":     (c - support) / atr,
            "dist_resistance_atr":  (resistance - c) / atr,
            "ema_aligned":          int(e21[i] > e50[i] > e200[i]),
            "above_ema200":         int(c > e200[i]),
            "cy_hour":              int(cy_hours_arr[i]),
            "dow":                  int(dow[i]),
        })

    return pd.DataFrame(rows)


# -- forward simulation --------------------------------------------------------
def _sim_one(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
             close: np.ndarray, n: int,
             sig_idx: int, sl_price: float, tp_price: float,
             max_bars: int) -> dict:
    """Inner simulation - arrays passed in, no DataFrame access."""
    if sig_idx + 1 >= n:
        return dict(outcome="no_entry", entry_price=np.nan, exit_price=np.nan,
                    bars_held=0, pnl_pips=0.0, exit_idx=-1, tp_hit=np.nan)

    entry = open_[sig_idx + 1]
    if entry >= tp_price or entry <= sl_price:
        return dict(outcome="no_entry", entry_price=entry, exit_price=entry,
                    bars_held=0, pnl_pips=0.0, exit_idx=-1, tp_hit=np.nan)

    end = min(sig_idx + 1 + max_bars, n)
    for j in range(sig_idx + 1, end):
        if low[j] <= sl_price:   # conservative: SL wins on ambiguous bar
            pnl = (sl_price - entry) / PIP
            return dict(outcome="sl", entry_price=entry, exit_price=sl_price,
                        bars_held=j - sig_idx, pnl_pips=pnl, exit_idx=j, tp_hit=0)
        if high[j] >= tp_price:
            pnl = (tp_price - entry) / PIP
            return dict(outcome="tp", entry_price=entry, exit_price=tp_price,
                        bars_held=j - sig_idx, pnl_pips=pnl, exit_idx=j, tp_hit=1)

    pnl = (close[end - 1] - entry) / PIP
    return dict(outcome="timeout", entry_price=entry, exit_price=close[end - 1],
                bars_held=end - 1 - sig_idx, pnl_pips=pnl,
                exit_idx=end - 1, tp_hit=np.nan)


def simulate_all(df: pd.DataFrame, signals: pd.DataFrame,
                 commission_pips: float, max_bars: int) -> pd.DataFrame:
    if signals.empty:
        return signals
    # Extract arrays once for all signals
    open_  = df["Open"].values
    high   = df["High"].values
    low    = df["Low"].values
    close  = df["Close"].values
    n      = len(df)
    out = []
    for r in signals.itertuples(index=False):
        sim = _sim_one(open_, high, low, close, n,
                       r.bar_idx, r.sl_price, r.tp_price, max_bars)
        sim["pnl_pips_net"] = sim["pnl_pips"] - 2 * commission_pips
        out.append({**r._asdict(), **sim})
    return pd.DataFrame(out)


# -- sequential filter --------------------------------------------------------─
def sequential_take(simulated: pd.DataFrame) -> pd.DataFrame:
    """Drop signals that fire while an earlier trade is still open."""
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


# -- reporting ----------------------------------------------------------------─
def _bars_per_year(tf: str) -> float:
    n, unit = int(tf[:-1]), tf[-1]
    mins = {"m": n, "h": n * 60, "d": n * 1440}[unit]
    return 365 * 24 * 60 / mins


def report(name: str, trades: pd.DataFrame, df_len: int, tf: str) -> None:
    if trades.empty:
        print(f"\n-- {name}: 0 trades --")
        return
    nt  = len(trades)
    pnl = trades["pnl_pips_net"].sum()
    avg = trades["pnl_pips_net"].mean()
    win = (trades["tp_hit"] == 1).sum()
    los = (trades["tp_hit"] == 0).sum()
    out = trades["tp_hit"].isna().sum()
    wr  = win / max(nt, 1) * 100
    eq  = trades["pnl_pips_net"].cumsum()
    dd  = (eq - eq.cummax()).min()

    bpy = _bars_per_year(tf)
    tpy = nt * bpy / max(df_len, 1)
    r   = trades["pnl_pips_net"]
    tsr = float(r.mean() / r.std() * np.sqrt(max(tpy, 1))) if r.std() > 0 else 0.0

    print(f"\n-- {name} --")
    print(f"  Trades         : {nt}")
    print(f"  TP hit         : {win}  ({wr:.1f}%)")
    print(f"  SL hit         : {los}")
    print(f"  Timeout        : {out}")
    print(f"  Total P&L      : {pnl:+.1f} pips")
    print(f"  Avg per trade  : {avg:+.2f} pips")
    print(f"  Max DD (pips)  : {dd:+.1f}")
    print(f"  Trade Sharpe   : {tsr:+.3f}")


# -- random forest ------------------------------------------------------------─
RF_FEATURES = [
    "dist_support_atr", "dist_resistance_atr", "rr",
    "ema21_slope", "ema50_slope", "ema200_slope",
    "rsi", "atr_frac", "ema_aligned", "above_ema200",
    "cy_hour", "dow",
]


def train_rf(sim: pd.DataFrame, test_start: str) -> tuple:
    """Train RF on IS data, evaluate OOS. Returns (model, threshold)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("\nRF skipped - sklearn not installed (pip install scikit-learn)")
        return None, 0.5

    labeled = sim[sim["tp_hit"].isin([0, 1])].copy()
    labeled["tp_hit"] = labeled["tp_hit"].astype(int)

    # Normalize signal_time to tz-naive UTC for comparison
    sig_t = pd.to_datetime(labeled["signal_time"]).dt.tz_localize(None)
    ts    = pd.Timestamp(test_start)
    train = labeled[sig_t < ts]
    test  = labeled[sig_t >= ts]

    if len(train) < 50 or len(test) < 20:
        print(f"\nRF skipped - not enough labeled samples "
              f"(train={len(train)}, oos={len(test)})")
        return None, 0.5

    X_tr = train[RF_FEATURES].fillna(0)
    y_tr = train["tp_hit"]
    X_te = test[RF_FEATURES].fillna(0)
    y_te = test["tp_hit"]

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_te, proba)

    # Sweep thresholds on OOS, pick best F1
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.3, 0.75, 46):
        preds = (proba >= t).astype(int)
        if preds.sum() == 0:
            continue
        prec  = float(y_te[preds == 1].mean())
        rec   = float(y_te[y_te == 1].shape[0] and
                      preds[y_te == 1].mean() or 0)
        f1    = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\n-- Random Forest --")
    print(f"  IS  signals    : {len(train)}  (labeled, pre-{test_start})")
    print(f"  OOS signals    : {len(test)}   (labeled, {test_start}+)")
    print(f"  AUC-ROC (OOS)  : {auc:.3f}")
    print(f"  Best threshold : {best_t:.2f}  (F1={best_f1:.3f})")
    print(f"\n  Feature importances (top 8):")
    imp = sorted(zip(RF_FEATURES, clf.feature_importances_), key=lambda x: -x[1])
    for feat, fi in imp[:8]:
        print(f"    {feat:30s}  {fi:.3f}")

    return clf, best_t


def apply_rf_filter(sim_full: pd.DataFrame, clf, threshold: float) -> pd.DataFrame:
    X     = sim_full[RF_FEATURES].fillna(0)
    proba = clf.predict_proba(X)[:, 1]
    out   = sim_full.copy()
    out["rf_prob"] = proba
    return out[proba >= threshold].reset_index(drop=True)


# -- main ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Price Level Bounce - EURUSD 5m, long only")
    ap.add_argument("--symbol",              default="EURUSD")
    ap.add_argument("--tf",                  default="5m")
    ap.add_argument("--start",               default="2021-01-01")
    ap.add_argument("--end",                 default="2025-12-31")
    ap.add_argument("--session-start",       type=int,   default=7,
                    help="session open hour, Cyprus local (default 7 = 7AM)")
    ap.add_argument("--session-end",         type=int,   default=16,
                    help="session close hour, Cyprus local (default 16 = 4PM)")
    ap.add_argument("--pivot-n",             type=int,   default=5,
                    help="bars each side required to confirm a swing pivot")
    ap.add_argument("--swing-lookback",      type=int,   default=100,
                    help="how many bars back to search for swing levels")
    ap.add_argument("--round-step",          type=float, default=0.0050,
                    help="round-number grid size in price (0.005 = 50 pips)")
    ap.add_argument("--atr-len",             type=int,   default=14)
    ap.add_argument("--rsi-len",             type=int,   default=14)
    ap.add_argument("--slope-n",             type=int,   default=5)
    ap.add_argument("--min-rr",              type=float, default=5.0,
                    help="minimum risk:reward ratio to accept a signal")
    ap.add_argument("--atr-sl-mult",         type=float, default=0.5,
                    help="SL distance below support in ATR units")
    ap.add_argument("--min-level-dist-atr",  type=float, default=0.2,
                    help="ignore levels within this x ATR of current close")
    ap.add_argument("--max-bars",            type=int,   default=500,
                    help="forward simulation timeout in bars")
    ap.add_argument("--commission",          type=float, default=0.00005,
                    help="per-side cost as price fraction (0.00005 = 0.5 pip)")
    ap.add_argument("--rf-test-start",       default="2025-01-01",
                    help="IS/OOS split date for RandomForest evaluation")
    ap.add_argument("--no-rf",               action="store_true",
                    help="skip RandomForest phase")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent

    print(f"Fetching {args.symbol} {args.tf}  {args.start} -> {args.end}")
    raw = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  fetched {len(raw):,} bars\n")
    if raw.empty:
        return

    print("Computing indicators...")
    df = add_features(raw, slope_n=args.slope_n,
                      atr_len=args.atr_len, rsi_len=args.rsi_len)

    print("Precomputing swing pivots...")
    pivot_highs, pivot_lows = precompute_pivots(df, pivot_n=args.pivot_n)

    print("Precomputing daily H/L...")
    prev_day_h, prev_day_l = precompute_daily_hl(df)

    print("Converting timestamps to Cyprus time...")
    cy_hours_arr = _cy_hours(df)

    print("Detecting signals...")
    sigs = detect_signals(
        df, cy_hours_arr,
        args.session_start, args.session_end,
        pivot_highs, pivot_lows, prev_day_h, prev_day_l,
        args.pivot_n, args.swing_lookback, args.round_step,
        args.min_rr, args.atr_sl_mult, args.min_level_dist_atr,
    )
    print(f"  signals found: {len(sigs)}")

    if sigs.empty:
        print("No signals - try looser --min-rr or --min-level-dist-atr.")
        return

    print("Simulating signals...")
    commission_pips = args.commission / PIP
    sim = simulate_all(df, sigs, commission_pips, args.max_bars)
    seq = sequential_take(sim)

    print("\n" + "=" * 80)
    print(f"RESULTS - {args.symbol} {args.tf}  "
          f"session={args.session_start:02d}:00-{args.session_end:02d}:00 CY")
    print(f"  min RR={args.min_rr}  SL=support-{args.atr_sl_mult}xATR  TP=resistance")
    print(f"  pivot_n={args.pivot_n}  swing_lookback={args.swing_lookback}  "
          f"round_step={int(args.round_step/0.0001)}pips")
    print("=" * 80)
    report("ALL signals (independent / RF training set)", sim, len(df), args.tf)
    report("SEQUENTIAL (realistic, one position at a time)", seq, len(df), args.tf)

    sym  = args.symbol
    sig_path = out_dir / f"level_bounce_all_{sym}_{args.tf}.csv"
    seq_path = out_dir / f"level_bounce_seq_{sym}_{args.tf}.csv"
    sim.to_csv(sig_path, index=False)
    seq.to_csv(seq_path, index=False)
    print(f"\n-> {sig_path}  ({len(sim)} signals -- RF training set)")
    print(f"-> {seq_path}  ({len(seq)} taken trades -- realistic backtest)")

    if args.no_rf:
        return

    # -- Random Forest --------------------------------------------------------─
    clf, threshold = train_rf(sim, args.rf_test_start)
    if clf is None:
        return

    sim_rf = apply_rf_filter(sim, clf, threshold)
    seq_rf = sequential_take(sim_rf)

    print(f"\nAfter RF filter (threshold >= {threshold:.2f}):")
    print(f"  Signals kept : {len(sim_rf)} / {len(sim)}  "
          f"({100 * len(sim_rf) / max(len(sim), 1):.1f}%)")
    report("ALL signals + RF filter", sim_rf, len(df), args.tf)
    report("SEQUENTIAL + RF filter", seq_rf, len(df), args.tf)

    ts       = pd.Timestamp(args.rf_test_start)
    sig_t    = pd.to_datetime(sim_rf["signal_time"]).dt.tz_localize(None)
    sim_oos  = sim_rf[sig_t >= ts].copy()
    seq_oos  = sequential_take(sim_oos)
    report(f"SEQUENTIAL + RF filter  (OOS {args.rf_test_start}+)", seq_oos, len(df), args.tf)

    rf_path = out_dir / f"level_bounce_rf_{sym}_{args.tf}.csv"
    sim_rf.to_csv(rf_path, index=False)
    print(f"\n-> {rf_path}  (RF-filtered signals)")


if __name__ == "__main__":
    main()
