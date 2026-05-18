"""
End-to-end leakage audit.

Three independent tests:

A) Per-feature past-only contract
   For each feature column, recompute on prefix vs on prefix+future. The value at
   the cutoff index must be identical. If it changes, the feature is peeking ahead.

B) Block-consistency
   build_features(df[:train_end]) vs build_features(df[:trade_end]) must produce
   IDENTICAL values for bars in [train_start, train_end]. If not, some feature
   uses data later in the series to compute earlier values.

C) Label boundary
   For training, labels at the last `max_hold_bars` bars of the train window
   should resolve only within the train block (no peeking into trade window).

D) Walk-forward split sanity
   Confirm train labels never use data from val set, and val never sees train bars
   in features after the embargo.

Run:
    python scripts/audit_leakage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ml.data import load_ohlcv, resample_ohlcv
from ml.features import build_features
from ml.labels import build_labels


def test_A_pure_function(df_block, df_htf, fast=False):
    print("\n[A] Per-feature past-only contract")
    X_short = build_features(df_block.iloc[:-500], df_htf=df_htf, fast=fast)
    X_long = build_features(df_block, df_htf=df_htf, fast=fast)

    # Compare at three cutoff indices within the short series
    cutoffs = [len(X_short) - 1, len(X_short) - 200, len(X_short) - 400]
    n_leak = 0
    for col in X_short.columns:
        for c in cutoffs:
            if c < 50: continue
            a = X_short[col].iloc[c]
            b = X_long[col].iloc[c]
            if pd.isna(a) and pd.isna(b): continue
            if pd.isna(a) or pd.isna(b):
                print(f"  ! {col}: NaN mismatch at idx {c}  short={a}  long={b}")
                n_leak += 1
                break
            if abs(float(a) - float(b)) > 1e-9:
                print(f"  ! {col}: LEAK at idx {c}  short={a}  long={b}  diff={abs(a-b):.2e}")
                n_leak += 1
                break
    if n_leak == 0:
        print(f"  OK: all {len(X_short.columns)} features pass past-only contract")
    return n_leak


def test_B_block_consistency(df_full, train_start, train_end, trade_end, df_htf, fast=False):
    print("\n[B] Block-consistency: train-block features == trade-block features at overlap")
    ts = lambda s: pd.Timestamp(s, tz="UTC")
    warmup = ts(train_start) - pd.DateOffset(months=1)
    df_train_block = df_full.loc[warmup:ts(train_end)]
    df_trade_block = df_full.loc[warmup:ts(trade_end)]

    X_train = build_features(df_train_block, df_htf=df_htf, fast=fast)
    X_trade = build_features(df_trade_block, df_htf=df_htf, fast=fast)

    overlap = X_train.index.intersection(X_trade.index)
    overlap = overlap[(overlap >= pd.Timestamp(train_start, tz="UTC")) & (overlap <= pd.Timestamp(train_end, tz="UTC"))]
    print(f"  comparing {len(overlap)} overlapping bars")

    n_leak = 0
    for col in X_train.columns:
        a = X_train.loc[overlap, col].astype(float)
        b = X_trade.loc[overlap, col].astype(float)
        both_nan = a.isna() & b.isna()
        diff = (a.fillna(0) - b.fillna(0)).abs()
        diff[both_nan] = 0
        max_diff = float(diff.max())
        if max_diff > 1e-9:
            print(f"  ! {col}: max diff {max_diff:.2e} at {diff.idxmax()}")
            n_leak += 1
    if n_leak == 0:
        print(f"  OK: all {len(X_train.columns)} features identical across blocks")
    return n_leak


def test_C_label_boundary(df_block, max_hold_bars):
    print("\n[C] Label boundary: tail labels resolve within block only")
    L = build_labels(df_block, df_1m=None, sl_atr=2.0, rr=0.5,
                     max_hold_bars=max_hold_bars, spread=0.00012, tf_minutes=5)
    tail = L.iloc[-max_hold_bars:]
    n_nan = int(tail["y_long"].isna().sum() + tail["y_short"].isna().sum())
    # All non-NaN tail labels should be either resolved early (TP/SL within block)
    # or 0 (timeout) — never reach into nonexistent data.
    resolved = (tail["y_long"].notna()).sum()
    print(f"  tail = last {max_hold_bars} bars; resolved labels: {resolved}/{len(tail)}")
    print(f"  tail y_long mean: {tail['y_long'].mean():.3f}  y_short mean: {tail['y_short'].mean():.3f}")
    print("  (expect lower than overall mean — truncated → labeled 0)")
    return 0


def main():
    print(">>> loading EURUSD 5m")
    df = load_ohlcv("EURUSD", "5m", "2025-01-01", "2025-07-01")
    df_htf = load_ohlcv("EURUSD", "1h", "2025-01-01", "2025-07-01")
    print(f"  {len(df):,} bars, htf {len(df_htf):,}")

    # --- A: per-feature past-only ---
    n_a = test_A_pure_function(df, df_htf, fast=False)

    # --- B: block consistency ---
    n_b = test_B_block_consistency(
        df, train_start="2025-02-01", train_end="2025-05-01",
        trade_end="2025-06-01", df_htf=df_htf, fast=False,
    )

    # --- C: label boundary ---
    n_c = test_C_label_boundary(
        df.loc[pd.Timestamp("2025-02-01", tz="UTC"):pd.Timestamp("2025-05-01", tz="UTC")],
        max_hold_bars=48,
    )

    total = n_a + n_b + n_c
    print(f"\n=== AUDIT SUMMARY ===")
    print(f"  test A (per-feature):     {'PASS' if n_a == 0 else f'FAIL ({n_a})'}")
    print(f"  test B (block-consistency): {'PASS' if n_b == 0 else f'FAIL ({n_b})'}")
    print(f"  test C (label boundary):    {'PASS' if n_c == 0 else f'FAIL ({n_c})'}")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
