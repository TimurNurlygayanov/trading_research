"""
Walk-forward optimization for trading strategies.

Splits training data into N folds and evaluates out-of-sample performance
on each fold to detect overfitting and validate robustness.

The walk-forward procedure:
  - Fold 1: train on data[0:1*fold_size], test on data[1*fold_size:2*fold_size]
  - Fold 2: train on data[0:2*fold_size], test on data[2*fold_size:3*fold_size]
  - ...
  - Fold N-1: train on data[0:(N-1)*fold_size], test on data[(N-1)*fold_size:N*fold_size]

This is "anchored" (expanding window) walk-forward, which is more conservative
and appropriate for strategies that benefit from more training data over time.

Usage:
  result = walk_forward(
      strategy_class=MyStrategy,
      df=train_df,
      param_space={"rsi_period": ("int", 5, 30)},
      n_folds=5,
      n_trials=30,
  )
  print(result.oos_sharpes)       # [0.9, 1.1, 0.7, 1.2, 0.8]
  print(result.passed)            # True
  print(result.overfitting_flag)  # False
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Type

import numpy as np
import pandas as pd
from backtesting import Strategy

from backtest.engine import BacktestResult, run_backtest
from backtest.optimizer import ParamSpace, optimize_strategy

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Results from a complete walk-forward analysis."""

    n_folds: int
    oos_sharpes: list[float]          # OOS Sharpe per fold
    is_sharpes: list[float]           # In-sample Sharpe per fold (after optimization)
    best_params_per_fold: list[dict[str, Any]]  # Best params found per fold

    # Summary statistics
    mean_oos_sharpe: float
    std_oos_sharpe: float
    min_oos_sharpe: float

    # Overfitting detection
    overfitting_flag: bool            # True if IS >> OOS consistently
    overfitting_ratio: float          # mean(IS) / mean(OOS); >2.0 is suspicious

    # Overall pass/fail
    passed: bool
    reject_reason: str | None

    # Fold-level detail
    fold_details: list[dict[str, Any]] = field(default_factory=list)


def walk_forward(
    strategy_class: Type[Strategy],
    df: pd.DataFrame,
    param_space: ParamSpace,
    n_folds: int = 5,
    n_trials: int = 30,
    n_jobs: int = -1,
    min_oos_sharpe: float = 0.0,
    min_fold_bars: int = 500,
    seed: int = 42,
    fixed_params: dict | None = None,
) -> WalkForwardResult:
    """
    Run anchored walk-forward optimization and out-of-sample evaluation.

    Parameters
    ----------
    strategy_class : backtesting.py Strategy subclass
    df             : OHLCV DataFrame — TRAINING DATA ONLY (never pass OOS data here)
    param_space    : hyperparameter search space (same format as optimizer.py)
    n_folds        : number of folds (default 5; minimum 3)
    n_trials       : Optuna trials per fold (fewer than full optimization is fine)
    n_jobs         : parallel jobs for Optuna within each fold
    min_oos_sharpe : minimum acceptable OOS Sharpe (default 0.0 = just positive)
    min_fold_bars  : minimum number of bars required per fold to be valid
    seed           : base random seed (each fold gets seed + fold_idx for variety)

    Returns
    -------
    WalkForwardResult with per-fold metrics and overfitting diagnosis.
    """
    if n_folds < 3:
        raise ValueError(f"n_folds must be >= 3, got {n_folds}")

    total_bars = len(df)
    # Need at least (n_folds + 1) segments: n_folds train expansions + n_folds test segments
    # With anchored WF: we need the data split into (n_folds + 1) chunks
    fold_size = total_bars // (n_folds + 1)

    if fold_size < min_fold_bars:
        raise ValueError(
            f"Each fold would have only {fold_size} bars (minimum: {min_fold_bars}). "
            f"Either reduce n_folds or provide more training data. "
            f"Total bars available: {total_bars}"
        )

    logger.info(
        f"Walk-forward: {n_folds} folds, {fold_size} bars/fold, "
        f"{total_bars} total bars, {n_trials} Optuna trials/fold"
    )

    oos_sharpes: list[float] = []
    is_sharpes: list[float] = []
    best_params_per_fold: list[dict[str, Any]] = []
    fold_details: list[dict[str, Any]] = []

    for fold_idx in range(n_folds):
        # Anchored: train grows from start, test is the next segment
        train_end_bar = (fold_idx + 1) * fold_size
        test_start_bar = train_end_bar
        test_end_bar = test_start_bar + fold_size

        train_df = df.iloc[:train_end_bar].copy()
        test_df = df.iloc[test_start_bar:test_end_bar].copy()

        train_start = train_df.index[0].date()
        train_end = train_df.index[-1].date()
        test_start = test_df.index[0].date()
        test_end = test_df.index[-1].date()

        logger.info(
            f"Fold {fold_idx + 1}/{n_folds}: "
            f"train {train_start} → {train_end} ({len(train_df)} bars), "
            f"test  {test_start} → {test_end} ({len(test_df)} bars)"
        )

        # ── Optimize on training data ──────────────────────────────────────
        try:
            best_params, study = optimize_strategy(
                strategy_class=strategy_class,
                train_df=train_df,
                param_space=param_space,
                n_trials=n_trials,
                n_jobs=n_jobs,
                seed=seed + fold_idx,
                direction="maximize",
                metric="sharpe",
                fixed_params=fixed_params,
            )
            is_sharpe = study.best_value if study.best_trial else 0.0
        except Exception as e:
            logger.warning(f"Fold {fold_idx + 1} optimization failed: {e}")
            best_params = {}
            is_sharpe = 0.0

        # ── Evaluate on test (OOS) data ────────────────────────────────────
        try:
            oos_result = run_backtest(
                strategy_class=strategy_class,
                df=test_df,
                params=best_params,
            )
            oos_sharpe = oos_result.sharpe
            oos_trades = oos_result.total_trades
            oos_passed = oos_result.passed
        except Exception as e:
            logger.warning(f"Fold {fold_idx + 1} OOS backtest failed: {e}")
            oos_sharpe = 0.0
            oos_trades = 0
            oos_passed = False

        oos_sharpes.append(oos_sharpe)
        is_sharpes.append(is_sharpe)
        best_params_per_fold.append(best_params)

        fold_details.append({
            "fold": fold_idx + 1,
            "train_start": str(train_start),
            "train_end": str(train_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "is_sharpe": round(is_sharpe, 3),
            "oos_sharpe": round(oos_sharpe, 3),
            "oos_trades": oos_trades,
            "best_params": best_params,
        })

        logger.info(
            f"Fold {fold_idx + 1} complete: IS Sharpe={is_sharpe:.3f}, "
            f"OOS Sharpe={oos_sharpe:.3f}, OOS trades={oos_trades}"
        )

    # ── Aggregate statistics ─────────────────────────────────────────────────
    oos_arr = np.array(oos_sharpes)
    is_arr = np.array(is_sharpes)

    mean_oos = float(np.mean(oos_arr))
    std_oos = float(np.std(oos_arr))
    min_oos = float(np.min(oos_arr))
    mean_is = float(np.mean(is_arr))

    # Overfitting: IS performance is much better than OOS
    if mean_oos > 0:
        overfitting_ratio = mean_is / mean_oos
    elif mean_is > 0:
        overfitting_ratio = 10.0  # IS positive, OOS non-positive = severe overfit
    else:
        overfitting_ratio = 1.0

    overfitting_flag = overfitting_ratio > 2.5

    # Count negative OOS folds (consecutive losses signal instability)
    n_negative_folds = int(np.sum(oos_arr < 0))
    n_folds_below_min = int(np.sum(oos_arr < min_oos_sharpe))

    # ── Pass/fail determination ──────────────────────────────────────────────
    reject_reason = None
    if mean_oos < min_oos_sharpe:
        reject_reason = (
            f"Mean OOS Sharpe {mean_oos:.3f} is below minimum {min_oos_sharpe}. "
            f"Walk-forward scores: {[round(s, 2) for s in oos_sharpes]}"
        )
    elif n_negative_folds > n_folds // 2:
        reject_reason = (
            f"{n_negative_folds}/{n_folds} folds have negative OOS Sharpe — "
            "strategy is not consistently profitable out-of-sample."
        )
    elif overfitting_flag:
        reject_reason = (
            f"Overfitting detected: IS/OOS Sharpe ratio = {overfitting_ratio:.1f}x "
            f"(IS mean={mean_is:.2f}, OOS mean={mean_oos:.2f}). "
            "Strategy is curve-fitted to training data."
        )

    passed = reject_reason is None

    result = WalkForwardResult(
        n_folds=n_folds,
        oos_sharpes=oos_sharpes,
        is_sharpes=is_sharpes,
        best_params_per_fold=best_params_per_fold,
        mean_oos_sharpe=round(mean_oos, 3),
        std_oos_sharpe=round(std_oos, 3),
        min_oos_sharpe=round(min_oos, 3),
        overfitting_flag=overfitting_flag,
        overfitting_ratio=round(overfitting_ratio, 2),
        passed=passed,
        reject_reason=reject_reason,
        fold_details=fold_details,
    )

    logger.info(
        f"Walk-forward complete: mean OOS={mean_oos:.3f}, std={std_oos:.3f}, "
        f"min={min_oos:.3f}, overfit_ratio={overfitting_ratio:.2f}, passed={passed}"
    )

    return result


def select_robust_params(result: WalkForwardResult) -> dict[str, Any]:
    """
    From a completed walk-forward result, select the most robust hyperparameters.

    Strategy: return the params from the fold with the median OOS Sharpe
    (not the best fold, to avoid selecting lucky outlier params).

    Returns the params dict from the median-performing fold.
    """
    if not result.oos_sharpes:
        return {}

    oos_arr = np.array(result.oos_sharpes)
    sorted_indices = np.argsort(oos_arr)
    median_idx = int(sorted_indices[(len(sorted_indices) - 1) // 2])
    return result.best_params_per_fold[median_idx]
