"""
Random Forest classifier on EMA touch events.

Reads the CSV produced by ema_touch_research.py and trains two models:
  bounce_model  -- predicts did_bounce_1atr  (price bounces >= 1 ATR in expected direction)
  break_model   -- predicts did_break        (price closes through EMA zone)

IS/OOS temporal split: first 70% of rows = training, last 30% = test.

Backtest (no SL):
  Bounce signal: enter in bounce direction at next bar open, TP = 1 ATR, timeout = max_bars.
  Break  signal: enter in break direction at next bar open, TP = 1 ATR, timeout = max_bars.
  No stop loss -- position held until TP or timeout.

Usage:
  python -m scripts.ema_rf_classifier --tf 5m
  python -m scripts.ema_rf_classifier --tf 1m
  python -m scripts.ema_rf_classifier --tf 5m --ema 50 --model bounce
  python -m scripts.ema_rf_classifier --tf 5m --min-proba 0.65
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv

PIP = 0.0001

# ── Feature engineering ──────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # encode approach side
    out["side_num"] = (out["approach_side"] == "resistance").astype(int)

    # hour cyclical encoding
    out["hour_sin"] = np.sin(2 * np.pi * out["cy_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["cy_hour"] / 24)

    # dow cyclical encoding
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 5)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 5)

    # clip extreme values
    out["wick_ratio_c"]    = out["wick_ratio"].clip(-3, 3)
    out["candle_range_c"]  = out["candle_range_atr"].clip(0, 5)
    out["next_ema_dist_c"] = out["next_ema_dist"].clip(0, 10)
    out["ema_slope_c"]     = out["ema_slope_atr"].clip(-2, 2)

    return out


FEATURE_COLS = [
    "ema",
    "side_num",
    "candle_range_c",
    "wick_ratio_c",
    "body_pct",
    "close_inside",
    "slope_aligned",
    "ema_alignment",
    "next_ema_dist_c",
    "ema_slope_c",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


# ── Model training ────────────────────────────────────────────────────────────

def train_rf(X_train: np.ndarray, y_train: np.ndarray,
             n_estimators: int = 300, max_depth: int = 8) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=50,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def threshold_sweep(clf: RandomForestClassifier,
                    X_oos: np.ndarray, y_oos: np.ndarray,
                    label: str) -> float:
    """Print precision/recall at various thresholds; return best threshold by F1."""
    proba = clf.predict_proba(X_oos)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_oos, proba)

    print(f"\n  {label} -- threshold sweep (OOS):")
    print(f"  {'thresh':>6}  {'prec':>6}  {'rec':>6}  {'n_pred':>7}  {'F1':>6}")
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.50, 0.85, 0.05):
        mask = proba >= t
        if mask.sum() == 0:
            continue
        p = y_oos[mask].mean()
        r = y_oos[mask].sum() / max(y_oos.sum(), 1)
        f1 = 2 * p * r / (p + r + 1e-9)
        flag = " <--" if f1 > best_f1 else ""
        print(f"  {t:6.2f}  {p:6.3f}  {r:6.3f}  {mask.sum():7d}  {f1:6.3f}{flag}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh


def feature_importance_report(clf: RandomForestClassifier, label: str) -> None:
    imp = sorted(zip(FEATURE_COLS, clf.feature_importances_),
                 key=lambda x: -x[1])
    print(f"\n  {label} -- feature importances:")
    for name, val in imp[:10]:
        bar = "#" * int(val * 80)
        print(f"    {name:<22} {val:.4f}  {bar}")


# ── No-SL backtest ────────────────────────────────────────────────────────────

def backtest_no_sl(ohlcv: pd.DataFrame, signals: pd.DataFrame,
                   direction: str,        # "bounce" or "break"
                   tp_atr: float, max_bars: int,
                   commission_pips: float) -> pd.DataFrame:
    """
    direction="bounce": enter in expected bounce direction at next bar open.
    direction="break":  enter in break direction (same as approach_side continuation).

    Exit logic (no fixed SL):
      1. TP hit (high/low crosses tp_price)     -> outcome "tp"
      2. EMA re-cross: close goes back through the EMA zone   -> outcome "ema_exit"
         (natural loss-limiter -- the EMA is the soft stop)
      3. Timeout after max_bars                 -> outcome "timeout"
    """
    open_  = ohlcv["Open"].values
    high   = ohlcv["High"].values
    low    = ohlcv["Low"].values
    close  = ohlcv["Close"].values
    atr_v  = ohlcv["atr"].values
    n      = len(ohlcv)

    # precompute EMA columns for each period
    import pandas_ta as ta_mod
    ema_arrays: dict[int, np.ndarray] = {}
    for p in [9, 13, 21, 50]:
        col = f"ema_{p}"
        if col not in ohlcv.columns:
            ohlcv = ohlcv.copy()
            ohlcv[col] = ta_mod.ema(ohlcv["Close"], length=p).values
        ema_arrays[p] = ohlcv[col].values

    # build a bar_idx -> ohlcv row index map via signal_time
    time_to_idx = {t: i for i, t in enumerate(ohlcv.index)}

    results = []
    for row in signals.itertuples(index=False):
        bar_i = time_to_idx.get(row.signal_time)
        if bar_i is None or bar_i + 1 >= n:
            continue

        entry_i = bar_i + 1
        entry   = open_[entry_i]
        atr     = atr_v[bar_i]
        if np.isnan(atr) or atr <= 0:
            continue

        ema_v = ema_arrays.get(int(row.ema))
        if ema_v is None:
            continue

        # determine trade direction
        is_long: bool
        if direction == "bounce":
            # support = long, resistance = short
            is_long = (row.approach_side == "support")
        else:
            # break: if coming from above (support) it broke down = short
            # if coming from below (resistance) it broke up = long
            is_long = (row.approach_side == "resistance")

        tp_price = entry + tp_atr * atr if is_long else entry - tp_atr * atr
        end      = min(entry_i + max_bars, n)

        outcome = "timeout"
        exit_price = close[end - 1]
        exit_i = end - 1

        for j in range(entry_i, end):
            ema_now = ema_v[j] if not np.isnan(ema_v[j]) else entry

            if is_long:
                if high[j] >= tp_price:
                    outcome = "tp"
                    exit_price = tp_price
                    exit_i = j
                    break
                # EMA re-cross: close drops back below EMA (exit at next open)
                if close[j] < ema_now and j + 1 < n:
                    outcome = "ema_exit"
                    exit_price = open_[j + 1]
                    exit_i = j + 1
                    break
            else:
                if low[j] <= tp_price:
                    outcome = "tp"
                    exit_price = tp_price
                    exit_i = j
                    break
                # EMA re-cross: close rises back above EMA
                if close[j] > ema_now and j + 1 < n:
                    outcome = "ema_exit"
                    exit_price = open_[j + 1]
                    exit_i = j + 1
                    break

        pnl_raw = (exit_price - entry) / PIP if is_long else (entry - exit_price) / PIP
        pnl_net = pnl_raw - 2 * commission_pips

        results.append({
            "signal_time":  row.signal_time,
            "approach_side": row.approach_side,
            "is_long":      is_long,
            "entry":        entry,
            "exit":         exit_price,
            "outcome":      outcome,
            "pnl_pips":     pnl_raw,
            "pnl_pips_net": pnl_net,
            "bars_held":    exit_i - entry_i,
            "cy_hour":      row.cy_hour,
            "ema_period":   row.ema,
            "proba":        row.proba,
        })

    return pd.DataFrame(results)


def report(name: str, trades: pd.DataFrame) -> None:
    if trades.empty:
        print(f"\n-- {name}: 0 trades --")
        return
    nt  = len(trades)
    pnl = trades["pnl_pips_net"].sum()
    avg = trades["pnl_pips_net"].mean()
    tp_n      = (trades["outcome"] == "tp").sum()
    ema_exit_n= (trades["outcome"] == "ema_exit").sum()
    timeout_n = (trades["outcome"] == "timeout").sum()
    wr        = tp_n / nt * 100
    eq  = trades["pnl_pips_net"].cumsum()
    dd  = (eq - eq.cummax()).min()
    r   = trades["pnl_pips_net"]
    tpy = nt
    tsr = float(r.mean() / r.std() * np.sqrt(tpy)) if r.std() > 0 else 0.0

    avg_tp  = trades.loc[trades["outcome"] == "tp",       "pnl_pips_net"].mean() if tp_n else 0
    avg_ema = trades.loc[trades["outcome"] == "ema_exit", "pnl_pips_net"].mean() if ema_exit_n else 0
    avg_tmo = trades.loc[trades["outcome"] == "timeout",  "pnl_pips_net"].mean() if timeout_n else 0

    print(f"\n-- {name} --")
    print(f"  Trades       : {nt}")
    print(f"  TP           : {tp_n} ({wr:.1f}%)  avg={avg_tp:+.2f}p")
    print(f"  EMA exit     : {ema_exit_n}  avg={avg_ema:+.2f}p")
    print(f"  Timeout      : {timeout_n}  avg={avg_tmo:+.2f}p")
    print(f"  Total P&L    : {pnl:+.1f} pips")
    print(f"  Avg / trade  : {avg:+.2f} pips")
    print(f"  Max DD       : {dd:+.1f} pips")
    print(f"  Trade Sharpe : {tsr:+.3f}")

    # breakdown by EMA period
    print(f"  By EMA period:")
    for p in sorted(trades["ema_period"].dropna().unique()):
        sub = trades[trades["ema_period"] == p]
        wr_p = (sub["outcome"] == "tp").sum() / len(sub) * 100
        print(f"    EMA {p:2d}: n={len(sub):4d}  win={wr_p:.0f}%  pnl={sub['pnl_pips_net'].sum():+.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="RF classifier on EMA touches")
    ap.add_argument("--tf",          default="5m", choices=["1m", "5m", "1h"])
    ap.add_argument("--symbol",      default="EURUSD")
    ap.add_argument("--ema",         type=int, default=0,
                    help="filter to single EMA period (0 = all)")
    ap.add_argument("--model",       default="both", choices=["bounce", "break", "both"])
    ap.add_argument("--min-proba",   type=float, default=0.60,
                    help="minimum RF probability to enter a trade")
    ap.add_argument("--tp-atr",      type=float, default=1.0)
    ap.add_argument("--max-bars",    type=int,   default=60)
    ap.add_argument("--commission",  type=float, default=0.00005)
    ap.add_argument("--train-frac",  type=float, default=0.70)
    ap.add_argument("--n-trees",     type=int,   default=300)
    ap.add_argument("--max-depth",   type=int,   default=8)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    csv_path   = script_dir / f"ema_touches_{args.symbol}_{args.tf}.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        print(f"Run: python -m scripts.ema_touch_research --tf {args.tf} first")
        return

    print(f"Loading {csv_path.name}...")
    raw = pd.read_csv(csv_path)
    print(f"  {len(raw):,} touch events")

    # Filter EMA period
    if args.ema > 0:
        raw = raw[raw["ema_period"] == args.ema].reset_index(drop=True)
        print(f"  filtered to EMA {args.ema}: {len(raw):,} rows")

    # Drop rows with missing targets or features
    targets = ["did_bounce_1atr", "did_break"]
    raw = raw.dropna(subset=targets + ["wick_ratio", "ema_slope_atr"])
    print(f"  after dropna: {len(raw):,} rows")

    # Parse signal_time back to datetime (needed for OHLCV join in backtest)
    raw["signal_time"] = pd.to_datetime(raw["signal_time"])

    # Feature engineering
    feat = build_features(raw)
    X = feat[FEATURE_COLS].values.astype(np.float32)

    # Temporal split
    split = int(len(feat) * args.train_frac)
    X_tr, X_oos  = X[:split],  X[split:]
    df_tr, df_oos = feat.iloc[:split], feat.iloc[split:]

    print(f"  Train: {len(df_tr):,}  OOS: {len(df_oos):,}")
    print(f"  OOS period: {df_oos['signal_time'].min()} -> {df_oos['signal_time'].max()}")

    commission_pips = args.commission / PIP

    # Fetch OHLCV for the OOS period (needed for no-SL backtest)
    oos_start = df_oos["signal_time"].min().strftime("%Y-%m-%d")
    oos_end   = df_oos["signal_time"].max().strftime("%Y-%m-%d")
    print(f"\nFetching {args.symbol} {args.tf} OHLCV for OOS backtest...")
    ohlcv_raw = fetch_ohlcv(args.symbol, args.tf, oos_start, oos_end)
    import pandas_ta as ta
    ohlcv_raw["atr"] = ta.atr(ohlcv_raw["High"], ohlcv_raw["Low"],
                               ohlcv_raw["Close"], length=14)
    print(f"  {len(ohlcv_raw):,} bars")

    models_to_run = ["bounce", "break"] if args.model == "both" else [args.model]

    for model_type in models_to_run:
        target_col = "did_bounce_1atr" if model_type == "bounce" else "did_break"
        y_tr  = df_tr[target_col].values.astype(int)
        y_oos = df_oos[target_col].values.astype(int)

        pos_rate = y_tr.mean()
        print(f"\n{'='*70}")
        print(f"MODEL: {model_type.upper()}  target={target_col}")
        print(f"  Train positive rate : {pos_rate:.3f}")
        print(f"  OOS   positive rate : {y_oos.mean():.3f}")

        print("  Training Random Forest...")
        clf = train_rf(X_tr, y_tr, args.n_trees, args.max_depth)

        # OOS evaluation
        y_pred = clf.predict(X_oos)
        print("\n  OOS classification report:")
        print(classification_report(y_oos, y_pred, digits=3))

        best_thresh = threshold_sweep(clf, X_oos, y_oos, model_type)
        feature_importance_report(clf, model_type)

        # Use user-supplied min_proba or the best threshold
        use_thresh = max(args.min_proba, best_thresh)
        print(f"\n  Using threshold: {use_thresh:.2f}")

        proba_oos = clf.predict_proba(X_oos)[:, 1]
        high_conf = df_oos.copy()
        high_conf["proba"] = proba_oos
        high_conf = high_conf[proba_oos >= use_thresh].reset_index(drop=True)
        print(f"  High-confidence OOS signals: {len(high_conf)}")

        if high_conf.empty:
            print("  No signals above threshold.")
            continue

        # No-SL backtest
        print(f"  Backtesting (TP={args.tp_atr}xATR, max_bars={args.max_bars}, no SL)...")
        trades = backtest_no_sl(
            ohlcv_raw, high_conf,
            direction=model_type,
            tp_atr=args.tp_atr,
            max_bars=args.max_bars,
            commission_pips=commission_pips,
        )

        report(f"{model_type.upper()} model  thresh={use_thresh:.2f}  OOS", trades)

        # Baseline: all OOS signals (no filter)
        baseline = df_oos.copy()
        baseline["proba"] = 0.0
        bl_trades = backtest_no_sl(
            ohlcv_raw, baseline,
            direction=model_type,
            tp_atr=args.tp_atr,
            max_bars=args.max_bars,
            commission_pips=commission_pips,
        )
        report(f"{model_type.upper()} BASELINE (no filter)  OOS", bl_trades)

        # Save
        out_path = script_dir / f"ema_rf_{model_type}_{args.symbol}_{args.tf}.csv"
        trades.to_csv(out_path, index=False)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
