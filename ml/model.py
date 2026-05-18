"""
CatBoost wrapper.

Two configurations:
  - fast   : depth=4, iterations=300, lr=0.1
  - heavy  : depth=6, iterations=2000, lr=0.03, early stopping, isotonic calibration
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression


# session feature is integer-categorical
CAT_FEATURES = [
    "session", "weekday", "st_dir", "bull_engulf", "bear_engulf", "inside_bar", "side",
    "d_today_green", "d_prev_green",
]


@dataclass
class TrainedModel:
    cb: CatBoostClassifier
    calibrator: IsotonicRegression | None
    feature_names: list[str]
    cat_idx: list[int]
    threshold: float
    val_pr_auc: float
    val_precision_at_threshold: float
    val_recall_at_threshold: float
    train_pr_auc: float = 0.0
    train_base_rate: float = 0.0
    val_base_rate: float = 0.0

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names]
        pool = Pool(X, cat_features=self.cat_idx)
        p = self.cb.predict_proba(pool)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return p


def _params(fast: bool, overrides: dict | None = None) -> dict:
    base = dict(
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    if fast:
        base.update(iterations=300, depth=4, learning_rate=0.1)
    else:
        base.update(iterations=2000, depth=6, learning_rate=0.03, early_stopping_rounds=50)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                base[k] = v
    return base


def _tune_threshold(p_val: np.ndarray, y_val: np.ndarray, min_precision: float = 0.45) -> tuple[float, float, float]:
    """
    Find smallest threshold whose precision on val >= min_precision.
    If unachievable, fall back to the threshold maximizing F1.
    Returns (threshold, precision, recall).
    """
    thresholds = np.linspace(0.10, 0.95, 86)
    best = (0.5, 0.0, 0.0)
    best_f1 = -1.0
    fallback = (0.5, 0.0, 0.0)
    for thr in thresholds:
        pred = p_val >= thr
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        if tp + fp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if prec >= min_precision and rec > best[2]:
            best = (float(thr), float(prec), float(rec))
        if f1 > best_f1:
            best_f1 = f1
            fallback = (float(thr), float(prec), float(rec))
    if best[2] == 0.0:
        return fallback
    return best


def _pr_auc(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    fast: bool = True,
    min_precision: float = 0.45,
    calibrate: bool = False,
    overrides: dict | None = None,
) -> TrainedModel:
    feats = list(X_train.columns)
    cat_idx = [feats.index(c) for c in CAT_FEATURES if c in feats]

    # CatBoost wants categoricals as int/str; ensure ints
    X_tr = X_train.copy()
    X_va = X_val.copy()
    for c in CAT_FEATURES:
        if c in X_tr.columns:
            X_tr[c] = X_tr[c].fillna(-1).astype("int32")
            X_va[c] = X_va[c].fillna(-1).astype("int32")

    train_pool = Pool(X_tr, y_train.astype(int), cat_features=cat_idx)
    val_pool = Pool(X_va, y_val.astype(int), cat_features=cat_idx)

    cb = CatBoostClassifier(**_params(fast, overrides))
    cb.fit(train_pool, eval_set=val_pool, use_best_model=not fast)

    p_train = cb.predict_proba(train_pool)[:, 1]
    p_val = cb.predict_proba(val_pool)[:, 1]
    train_pr_auc = _pr_auc(p_train, y_train.astype(int).values)
    train_base = float(y_train.astype(int).mean())
    val_base = float(y_val.astype(int).mean())

    calibrator = None
    if calibrate:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(p_val, y_val.astype(int).values)
        p_val = calibrator.predict(p_val)

    pr_auc = _pr_auc(p_val, y_val.astype(int).values)
    thr, prec, rec = _tune_threshold(p_val, y_val.astype(int).values, min_precision)

    return TrainedModel(
        cb=cb,
        calibrator=calibrator,
        feature_names=feats,
        cat_idx=cat_idx,
        threshold=thr,
        val_pr_auc=pr_auc,
        val_precision_at_threshold=prec,
        val_recall_at_threshold=rec,
        train_pr_auc=train_pr_auc,
        train_base_rate=train_base,
        val_base_rate=val_base,
    )
