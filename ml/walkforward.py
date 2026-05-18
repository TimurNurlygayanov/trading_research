"""
Rolling walk-forward driver.

Slides a (train_months, trade_months) window over [start, end], stepping by trade_months.
For each window:
  1. Build features + labels on train_window with embargo at the tail.
  2. Purged train/val split (last 15% of train_window = val, with embargo before val).
  3. Train two CatBoosts (long, short) on train, tune threshold on val.
  4. Simulate bar-by-bar on trade_window.
  5. Carry resulting equity into the next window (compounding).

Returns aggregated trades + equity + per-window stats.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ml.dataset import make_dataset, make_doubled, add_side_for_inference
from ml.features import build_features
from ml.labels import _atr_for_labels
from ml.model import train as train_model, TrainedModel, CAT_FEATURES
from ml.simulator import simulate, SimResult, Trade


@dataclass
class WindowResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    trade_start: pd.Timestamp
    trade_end: pd.Timestamp
    model_stats: dict
    sim: SimResult
    n_train: int
    n_val: int
    n_trade: int


@dataclass
class WalkForwardResult:
    windows: list[WindowResult] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    halted: bool = False
    halt_reason: str | None = None


def _purged_split(X: pd.DataFrame, L: pd.DataFrame, embargo: int, val_frac: float = 0.15):
    """Split chronologically, leaving an embargo gap between train and val."""
    n = len(X)
    n_val = max(50, int(n * val_frac))
    val_start = n - n_val
    train_end = val_start - embargo
    if train_end <= 100:
        raise ValueError(f"Not enough samples for purged split: {n}")
    return (
        X.iloc[:train_end], L.iloc[:train_end],
        X.iloc[val_start:], L.iloc[val_start:],
    )


def run_walkforward(
    df_full: pd.DataFrame,
    df_htf: pd.DataFrame | None,
    df_1m: pd.DataFrame | None,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int,
    trade_months: int,
    sl_atr: float,
    rr: float,
    max_hold_bars: int,
    spread: float,
    commission: float,
    risk_pct: float,
    tf_minutes: int,
    fast: bool,
    min_precision: float,
    daily_loss_pct: float = 0.04,
    max_loss_pct: float = 0.08,
    start_equity: float = 100_000.0,
    skip_every_other: bool = False,
    verbose: bool = True,
    model_overrides: dict | None = None,
    max_windows: int | None = None,
    side_only: str | None = None,
    proba_threshold_override: float | None = None,
    ema_filter: bool = False,
    slippage_pips_sl: float = 0.0,
    slippage_pips_tp: float = 0.0,
    pip_size: float = 0.0001,
) -> WalkForwardResult:
    result = WalkForwardResult()
    equity = start_equity
    all_equity_pts: list[tuple[pd.Timestamp, float]] = []
    def _to_utc(ts):
        ts = pd.Timestamp(ts)
        return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    cur = _to_utc(start)
    end_ts = _to_utc(end)
    step = relativedelta(months=trade_months)
    train_delta = relativedelta(months=train_months)
    window_idx = 0

    while True:
        train_start = cur
        train_end = cur + train_delta
        trade_start = train_end
        trade_end = train_end + step
        if trade_end > end_ts:
            break

        if skip_every_other and window_idx % 2 == 1:
            cur = cur + step
            window_idx += 1
            continue

        if verbose:
            print(f"\n[window {window_idx}] train {train_start.date()}..{train_end.date()} | "
                  f"trade {trade_start.date()}..{trade_end.date()}")

        # We need warm-up history before train_start for indicators (~500 bars).
        warmup_start = train_start - relativedelta(months=1)
        df_block = df_full.loc[warmup_start:trade_end]
        if df_block.empty:
            cur = cur + step
            window_idx += 1
            continue

        # Build dataset over the train window (with warmup tail in features for stability)
        df_train_block = df_full.loc[warmup_start:train_end]
        X_full, L_full = make_dataset(
            df_train_block, df_htf=df_htf, df_1m=df_1m,
            sl_atr=sl_atr, rr=rr, max_hold_bars=max_hold_bars,
            spread=spread, tf_minutes=tf_minutes, fast=fast,
        )
        # Restrict to actual train window (drop warmup rows from the head)
        X_train_all = X_full.loc[X_full.index >= train_start]
        L_train_all = L_full.loc[L_full.index >= train_start]
        if len(X_train_all) < 300:
            if verbose:
                print(f"  skip: only {len(X_train_all)} train samples")
            cur = cur + step
            window_idx += 1
            continue

        embargo = max_hold_bars
        # Split on unique-timestamp matrix first, THEN double each side — avoids cross-leakage
        X_tr, L_tr, X_va, L_va = _purged_split(X_train_all, L_train_all, embargo=embargo)
        X_tr_d, y_tr_d = make_doubled(X_tr, L_tr)
        X_va_d, y_va_d = make_doubled(X_va, L_va)

        # One unified binary model: predict P(TP hit) given features + side
        m = train_model(X_tr_d, y_tr_d, X_va_d, y_va_d, fast=fast,
                        min_precision=min_precision, calibrate=not fast,
                        overrides=model_overrides)

        if verbose:
            lift = m.val_pr_auc / m.val_base_rate if m.val_base_rate > 0 else float("nan")
            print(f"  base={m.train_base_rate:.3f}/{m.val_base_rate:.3f}  "
                  f"PR-AUC train={m.train_pr_auc:.3f} val={m.val_pr_auc:.3f} "
                  f"(lift {lift:.2f}x)  "
                  f"thr={m.threshold:.2f} P={m.val_precision_at_threshold:.2f} R={m.val_recall_at_threshold:.2f}")

        # Build features over the TRADE block (no labels needed)
        df_trade_block = df_full.loc[warmup_start:trade_end]
        X_trade_full = build_features(df_trade_block, df_htf=df_htf, fast=fast)
        X_trade = X_trade_full.loc[(X_trade_full.index >= trade_start)
                                   & (X_trade_full.index < trade_end)]
        X_trade = X_trade.dropna()
        df_trade = df_full.loc[X_trade.index]
        if len(df_trade) < 10:
            if verbose:
                print("  skip: no trade-window samples")
            cur = cur + step
            window_idx += 1
            continue

        # Ensure categorical dtype on trade-block features matches training
        for c in CAT_FEATURES:
            if c in X_trade.columns:
                X_trade[c] = X_trade[c].fillna(-1).astype("int32")

        # Score each bar twice — once per side
        X_long = add_side_for_inference(X_trade, +1)
        X_short = add_side_for_inference(X_trade, -1)
        X_long["side"] = X_long["side"].astype("int32")
        X_short["side"] = X_short["side"].astype("int32")
        p_long = m.predict_proba(X_long)
        p_short = m.predict_proba(X_short)
        if side_only == "long":
            p_short = np.zeros_like(p_short)
        elif side_only == "short":
            p_long = np.zeros_like(p_long)
        if ema_filter and "ema50_slope" in X_trade.columns:
            slope = X_trade["ema50_slope"].values
            p_long[slope < 0] = 0.0
            p_short[slope > 0] = 0.0
            if verbose:
                n_blocked_l = int((slope < 0).sum())
                n_blocked_s = int((slope > 0).sum())
                print(f"  ema-filter: blocked {n_blocked_l} long bars (slope<0), {n_blocked_s} short bars (slope>0)")

        # ATR aligned to df_trade
        atr_series = _atr_for_labels(df_trade_block, 14).reindex(df_trade.index)

        sim = simulate(
            df_trade,
            df_1m=df_1m,
            p_long=p_long,
            p_short=p_short,
            threshold_long=proba_threshold_override if proba_threshold_override is not None else m.threshold,
            threshold_short=proba_threshold_override if proba_threshold_override is not None else m.threshold,
            atr_series=atr_series,
            sl_atr=sl_atr,
            rr=rr,
            max_hold_bars=max_hold_bars,
            spread=spread,
            commission=commission,
            start_equity=equity,
            risk_pct=risk_pct,
            daily_loss_pct=daily_loss_pct,
            max_loss_pct=max_loss_pct,
            tf_minutes=tf_minutes,
            slippage_pips_sl=slippage_pips_sl,
            slippage_pips_tp=slippage_pips_tp,
            pip_size=pip_size,
        )

        if verbose:
            n_tp = sum(1 for t in sim.trades if t.outcome == "tp")
            n_sl = sum(1 for t in sim.trades if t.outcome == "sl")
            print(f"  trades={len(sim.trades)}  tp={n_tp} sl={n_sl}  "
                  f"equity {equity:,.0f} -> {sim.end_equity:,.0f} "
                  f"({(sim.end_equity/equity - 1)*100:+.2f}%)")

        equity = sim.end_equity
        result.windows.append(WindowResult(
            train_start=train_start, train_end=train_end,
            trade_start=trade_start, trade_end=trade_end,
            model_stats={
                "train_pr_auc": m.train_pr_auc,
                "val_pr_auc": m.val_pr_auc,
                "train_base_rate": m.train_base_rate,
                "val_base_rate": m.val_base_rate,
                "threshold": m.threshold,
                "precision": m.val_precision_at_threshold,
                "recall": m.val_recall_at_threshold,
            },
            sim=sim,
            n_train=len(X_tr_d),
            n_val=len(X_va_d),
            n_trade=len(df_trade),
        ))
        result.trades.extend(sim.trades)
        for ts, e in zip(sim.equity_curve.index, sim.equity_curve.values):
            all_equity_pts.append((ts, e))

        if sim.halted:
            result.halted = True
            result.halt_reason = sim.halt_reason
            if verbose:
                print(f"  HALTED: {sim.halt_reason}")
            break

        cur = cur + step
        window_idx += 1
        if max_windows is not None and len(result.windows) >= max_windows:
            if verbose:
                print(f"\nstopped after {max_windows} windows (--max-windows)")
            break

    if all_equity_pts:
        eq = pd.Series([v for _, v in all_equity_pts], index=[t for t, _ in all_equity_pts])
        eq = eq[~eq.index.duplicated(keep="last")].sort_index()
        result.equity_curve = eq

    return result
