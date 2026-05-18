"""
Glue: take OHLCV + HTF -> features X, labels y. Aligns and drops warmup NaN rows.
"""
from __future__ import annotations

import pandas as pd

from ml.features import build_features
from ml.labels import build_labels


def make_dataset(
    df: pd.DataFrame,
    df_htf: pd.DataFrame | None,
    df_1m: pd.DataFrame | None,
    sl_atr: float,
    rr: float,
    max_hold_bars: int,
    spread: float,
    tf_minutes: int,
    fast: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (X, labels_df) with the same index. labels_df has columns from build_labels.
    Rows with NaN features or NaN labels are dropped.
    """
    X = build_features(df, df_htf=df_htf, fast=fast)
    L = build_labels(
        df,
        df_1m=df_1m,
        sl_atr=sl_atr,
        rr=rr,
        max_hold_bars=max_hold_bars,
        spread=spread,
        tf_minutes=tf_minutes,
    )

    # Align
    common = X.index.intersection(L.index)
    X = X.loc[common]
    L = L.loc[common]

    # Drop rows where any feature is NaN (warmup) or label is NaN (tail)
    mask = X.notna().all(axis=1) & L["y_long"].notna() & L["y_short"].notna()
    X = X.loc[mask]
    L = L.loc[mask]
    return X, L


def make_doubled(X: pd.DataFrame, L: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Stack long and short examples into a single matrix with a 'side' feature.
    Each input bar produces two output rows: (side=+1, y=y_long), (side=-1, y=y_short).
    Returns (X_doubled, y_doubled), sorted stable by (timestamp, side).
    """
    X_long = X.copy()
    X_long["side"] = 1
    X_long["__y"] = L["y_long"].astype(int).values
    X_short = X.copy()
    X_short["side"] = -1
    X_short["__y"] = L["y_short"].astype(int).values

    full = pd.concat([X_long, X_short], axis=0)
    full = full.sort_index(kind="stable")
    y = full["__y"].astype(int)
    Xd = full.drop(columns=["__y"])
    return Xd, y


def add_side_for_inference(X: pd.DataFrame, side: int) -> pd.DataFrame:
    """At inference time, score the same bars twice — once per side."""
    Xc = X.copy()
    Xc["side"] = side
    return Xc
