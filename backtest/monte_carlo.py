"""
Monte Carlo permutation test for trading strategy statistical significance.

Tests whether a strategy's Sharpe ratio could be achieved by chance.
The null hypothesis is: the strategy has no predictive power, and the
observed performance is due to lucky ordering of trades.

Method:
  1. Take the actual trade PnL series (from backtesting.py trades DataFrame)
  2. Shuffle the PnL values N times (permutation = random reordering)
  3. Compute Sharpe ratio for each permuted series
  4. p-value = fraction of permutations that beat the actual Sharpe

If p < 0.05: strategy is statistically significant at 95% confidence level.
If p > 0.05: cannot reject the null hypothesis — results may be due to chance.

Note: This tests trade sequence significance, not parameter sensitivity.
For parameter sensitivity, use walk-forward analysis.

Usage:
  result = monte_carlo_test(trades_df, actual_sharpe=1.2, n_permutations=1000)
  print(result.p_value)    # 0.023
  print(result.passed)     # True (p < 0.05)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SQRT_252 = float(np.sqrt(252))  # annualization factor for daily returns


@dataclass
class MonteCarloResult:
    """Result of a Monte Carlo permutation test."""

    p_value: float               # fraction of permutations >= actual_sharpe
    actual_sharpe: float         # the real strategy Sharpe
    permuted_sharpe_mean: float  # mean Sharpe across all permutations
    permuted_sharpe_std: float   # std of permuted Sharpe distribution
    permuted_sharpe_p95: float   # 95th percentile of permuted distribution
    n_permutations: int          # number of permutations run
    n_trades: int                # number of trades in the strategy

    passed: bool                 # True if p_value < significance_level
    significance_level: float    # threshold used (default 0.05)

    # Z-score: how many std devs above the random baseline is the actual Sharpe
    z_score: float


def monte_carlo_test(
    trades_df: pd.DataFrame,
    actual_sharpe: float,
    n_permutations: int = 1000,
    significance_level: float = 0.05,
    seed: int = 42,
    pnl_column: str = "PnL",
) -> MonteCarloResult:
    """
    Run a Monte Carlo permutation test on trade PnL series.

    Parameters
    ----------
    trades_df          : DataFrame from backtesting.py with trade-level results.
                         Must contain a PnL column (or ReturnPct column as fallback).
    actual_sharpe      : The Sharpe ratio computed from the real backtest.
    n_permutations     : Number of random shuffles to run (1000 is standard).
                         Use 5000 for publication-quality results.
    significance_level : p-value threshold for passing (default 0.05 = 95% confidence).
    seed               : Random seed for reproducibility.
    pnl_column         : Column name for trade PnL values.

    Returns
    -------
    MonteCarloResult with p_value, z_score, and pass/fail determination.
    """
    # ── Extract PnL series ───────────────────────────────────────────────────
    if trades_df is None or trades_df.empty:
        logger.warning("monte_carlo_test: empty trades DataFrame, returning default result")
        return MonteCarloResult(
            p_value=1.0,
            actual_sharpe=actual_sharpe,
            permuted_sharpe_mean=0.0,
            permuted_sharpe_std=0.0,
            permuted_sharpe_p95=0.0,
            n_permutations=n_permutations,
            n_trades=0,
            passed=False,
            significance_level=significance_level,
            z_score=0.0,
        )

    # Try PnL column, fall back to ReturnPct
    if pnl_column in trades_df.columns:
        pnl = trades_df[pnl_column].values.astype(np.float64)
    elif "ReturnPct" in trades_df.columns:
        pnl = trades_df["ReturnPct"].values.astype(np.float64)
    else:
        raise ValueError(
            f"trades_df must contain '{pnl_column}' or 'ReturnPct' column. "
            f"Available columns: {list(trades_df.columns)}"
        )

    # Remove NaN/inf
    pnl = pnl[np.isfinite(pnl)]
    n_trades = len(pnl)

    if n_trades < 10:
        logger.warning(f"monte_carlo_test: only {n_trades} valid trades, results unreliable")
        return MonteCarloResult(
            p_value=1.0,
            actual_sharpe=actual_sharpe,
            permuted_sharpe_mean=0.0,
            permuted_sharpe_std=0.0,
            permuted_sharpe_p95=0.0,
            n_permutations=n_permutations,
            n_trades=n_trades,
            passed=False,
            significance_level=significance_level,
            z_score=0.0,
        )

    # ── Vectorized permutation test ──────────────────────────────────────────
    # Shape: (n_permutations, n_trades) — generate all permutations at once
    rng = np.random.default_rng(seed)
    permuted = rng.permuted(
        np.tile(pnl, (n_permutations, 1)),
        axis=1,
    )  # shape: (n_permutations, n_trades)

    # Compute Sharpe for each permutation: mean(pnl) / std(pnl) * sqrt(n_trades)
    # This is a trade-level Sharpe (not annualized), used for relative comparison
    perm_means = permuted.mean(axis=1)        # (n_permutations,)
    perm_stds = permuted.std(axis=1, ddof=1)  # (n_permutations,)

    # Avoid division by zero
    safe_stds = np.where(perm_stds > 1e-10, perm_stds, 1e-10)
    perm_sharpes = (perm_means / safe_stds) * np.sqrt(n_trades)  # (n_permutations,)

    # Actual trade-level Sharpe for comparison (must use same formula)
    actual_mean = float(np.mean(pnl))
    actual_std = float(np.std(pnl, ddof=1))
    if actual_std > 1e-10:
        actual_trade_sharpe = (actual_mean / actual_std) * np.sqrt(n_trades)
    else:
        actual_trade_sharpe = 0.0

    # ── p-value: fraction of permutations >= actual ──────────────────────────
    p_value = float(np.mean(perm_sharpes >= actual_trade_sharpe))

    # Distribution statistics
    perm_mean = float(np.mean(perm_sharpes))
    perm_std = float(np.std(perm_sharpes))
    perm_p95 = float(np.percentile(perm_sharpes, 95))

    # Z-score: how many standard deviations above baseline
    if perm_std > 1e-10:
        z_score = (actual_trade_sharpe - perm_mean) / perm_std
    else:
        z_score = 0.0

    passed = p_value < significance_level

    logger.info(
        f"Monte Carlo ({n_permutations} permutations, {n_trades} trades): "
        f"p={p_value:.4f}, z={z_score:.2f}, "
        f"actual_sharpe={actual_sharpe:.3f} (trade-level={actual_trade_sharpe:.3f}), "
        f"perm_mean={perm_mean:.3f}, passed={passed}"
    )

    return MonteCarloResult(
        p_value=round(p_value, 4),
        actual_sharpe=actual_sharpe,
        permuted_sharpe_mean=round(perm_mean, 3),
        permuted_sharpe_std=round(perm_std, 3),
        permuted_sharpe_p95=round(perm_p95, 3),
        n_permutations=n_permutations,
        n_trades=n_trades,
        passed=passed,
        significance_level=significance_level,
        z_score=round(float(z_score), 3),
    )


def block_bootstrap_test(
    trades_df: pd.DataFrame,
    actual_sharpe: float,
    n_permutations: int = 1000,
    block_size: int = 20,
    significance_level: float = 0.05,
    seed: int = 42,
    pnl_column: str = "PnL",
) -> MonteCarloResult:
    """
    Block bootstrap variant that preserves local autocorrelation structure.

    Trades near each other may share regime characteristics (e.g., trending
    vs. ranging market), so shuffling in blocks preserves this structure
    while still testing the null hypothesis.

    Recommended when n_trades > 500 and you suspect regime clustering.

    Parameters
    ----------
    block_size : number of consecutive trades to keep together when shuffling
                 (default 20; use 10-50 depending on average holding period)
    """
    if trades_df is None or trades_df.empty:
        return monte_carlo_test(trades_df, actual_sharpe, n_permutations, significance_level, seed, pnl_column)

    if pnl_column in trades_df.columns:
        pnl = trades_df[pnl_column].values.astype(np.float64)
    elif "ReturnPct" in trades_df.columns:
        pnl = trades_df["ReturnPct"].values.astype(np.float64)
    else:
        raise ValueError(f"trades_df must contain '{pnl_column}' or 'ReturnPct'")

    pnl = pnl[np.isfinite(pnl)]
    n_trades = len(pnl)

    if n_trades < block_size * 3:
        # Fall back to simple permutation if too few trades for block bootstrap
        logger.info(
            f"block_bootstrap: {n_trades} trades < {block_size * 3} min, "
            "falling back to simple permutation test"
        )
        return monte_carlo_test(trades_df, actual_sharpe, n_permutations, significance_level, seed, pnl_column)

    # Build blocks
    n_blocks = int(np.ceil(n_trades / block_size))
    # Pad pnl to be exactly n_blocks * block_size
    pad_length = n_blocks * block_size - n_trades
    pnl_padded = np.pad(pnl, (0, pad_length), mode="wrap")
    blocks = pnl_padded.reshape(n_blocks, block_size)  # (n_blocks, block_size)

    rng = np.random.default_rng(seed)
    perm_sharpes = np.empty(n_permutations, dtype=np.float64)

    for i in range(n_permutations):
        # Shuffle block order, then flatten back to trade series
        shuffled_blocks = rng.permutation(blocks)          # (n_blocks, block_size)
        shuffled_pnl = shuffled_blocks.ravel()[:n_trades]  # trim back to original length
        mean_v = float(np.mean(shuffled_pnl))
        std_v = float(np.std(shuffled_pnl, ddof=1))
        if std_v > 1e-10:
            perm_sharpes[i] = (mean_v / std_v) * np.sqrt(n_trades)
        else:
            perm_sharpes[i] = 0.0

    # Actual
    actual_mean = float(np.mean(pnl))
    actual_std = float(np.std(pnl, ddof=1))
    actual_trade_sharpe = (actual_mean / actual_std) * np.sqrt(n_trades) if actual_std > 1e-10 else 0.0

    p_value = float(np.mean(perm_sharpes >= actual_trade_sharpe))
    perm_mean = float(np.mean(perm_sharpes))
    perm_std = float(np.std(perm_sharpes))
    perm_p95 = float(np.percentile(perm_sharpes, 95))
    z_score = (actual_trade_sharpe - perm_mean) / perm_std if perm_std > 1e-10 else 0.0

    passed = p_value < significance_level

    logger.info(
        f"Block bootstrap ({n_permutations} permutations, block={block_size}, {n_trades} trades): "
        f"p={p_value:.4f}, z={z_score:.2f}, passed={passed}"
    )

    return MonteCarloResult(
        p_value=round(p_value, 4),
        actual_sharpe=actual_sharpe,
        permuted_sharpe_mean=round(perm_mean, 3),
        permuted_sharpe_std=round(perm_std, 3),
        permuted_sharpe_p95=round(perm_p95, 3),
        n_permutations=n_permutations,
        n_trades=n_trades,
        passed=passed,
        significance_level=significance_level,
        z_score=round(float(z_score), 3),
    )
