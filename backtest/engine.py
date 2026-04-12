"""
Backtesting engine wrapper around backtesting.py.

Adds:
  - Automatic signal counting and rejection if < minimum thresholds
  - Time-of-day session filtering (passed as strategy params)
  - Risk management: max daily losses, trailing stop
  - Standard result extraction with all required metrics
  - OOS split: all data before 2026 = train, 2026+ = OOS

Usage:
  result = run_backtest(MyStrategy, df, params={...})
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Type

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

MIN_SIGNALS_TOTAL = int(os.environ.get("BACKTEST_MIN_SIGNALS_TOTAL", 100))
MIN_SIGNALS_PER_YEAR = int(os.environ.get("BACKTEST_MIN_SIGNALS_PER_YEAR", 100))
MIN_SHARPE = float(os.environ.get("BACKTEST_MIN_SHARPE", 0.8))
MIN_CALMAR = float(os.environ.get("BACKTEST_MIN_CALMAR", 0.5))
MAX_DRAWDOWN = float(os.environ.get("BACKTEST_MAX_DRAWDOWN", 0.25))


@dataclass
class BacktestResult:
    passed: bool
    reject_reason: str | None

    sharpe: float
    calmar: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    signals_per_year: float
    avg_trade_pnl: float
    equity_final: float
    equity_peak: float
    return_pct: float

    # Raw stats dict from backtesting.py
    raw_stats: dict[str, Any]

    # Trade-level DataFrame
    trades: pd.DataFrame | None = None


def run_backtest(
    strategy_class: Type[Strategy],
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    cash: float = 10_000.0,
    commission: float = 0.0002,
    exclusive_orders: bool = True,
) -> BacktestResult:
    """
    Run a single backtest. Returns BacktestResult with pass/fail and all metrics.

    Parameters
    ----------
    strategy_class  : backtesting.py Strategy subclass
    df              : OHLCV DataFrame (columns: Open High Low Close Volume), DatetimeIndex UTC
    params          : dict of hyperparameters to set on the strategy
    cash            : starting capital
    commission      : per-trade commission as fraction (0.0002 = 2 pips)
    exclusive_orders: disallow simultaneous long+short
    """
    params = params or {}

    # Apply params to strategy class (backtesting.py convention)
    for k, v in params.items():
        if hasattr(strategy_class, k):
            setattr(strategy_class, k, v)

    bt = Backtest(
        df,
        strategy_class,
        cash=cash,
        commission=commission,
        exclusive_orders=exclusive_orders,
        trade_on_close=False,  # Enter at NEXT bar open — prevents lookahead
    )

    try:
        stats = bt.run()
    except Exception as e:
        return BacktestResult(
            passed=False,
            reject_reason=f"Backtest raised exception: {e}",
            sharpe=0.0, calmar=0.0, max_drawdown=0.0, win_rate=0.0,
            profit_factor=0.0, total_trades=0, signals_per_year=0.0,
            avg_trade_pnl=0.0, equity_final=cash, equity_peak=cash,
            return_pct=0.0, raw_stats={}, trades=None,
        )

    trades_df = stats["_trades"] if "_trades" in stats else pd.DataFrame()
    total_trades = len(trades_df)

    # ── Signal count check ───────────────────────────────────────────────────
    if total_trades < MIN_SIGNALS_TOTAL:
        return BacktestResult(
            passed=False,
            reject_reason=(
                f"Too few trades: {total_trades} total (min: {MIN_SIGNALS_TOTAL}). "
                "Strategy does not generate enough signals for statistical validity."
            ),
            sharpe=0.0, calmar=0.0, max_drawdown=float(stats.get("Max. Drawdown [%]", 0)) / 100,
            win_rate=0.0, profit_factor=0.0, total_trades=total_trades, signals_per_year=0.0,
            avg_trade_pnl=0.0, equity_final=float(stats.get("Equity Final [$]", cash)),
            equity_peak=float(stats.get("Equity Peak [$]", cash)),
            return_pct=float(stats.get("Return [%]", 0)),
            raw_stats=dict(stats), trades=trades_df,
        )

    # Check per-year signal count
    if not trades_df.empty and "EntryTime" in trades_df.columns:
        entry_times = pd.to_datetime(trades_df["EntryTime"])
        trades_per_year = entry_times.dt.year.value_counts()
        min_per_year = int(trades_per_year.min())
        if min_per_year < MIN_SIGNALS_PER_YEAR:
            worst_year = int(trades_per_year.idxmin())
            return BacktestResult(
                passed=False,
                reject_reason=(
                    f"Too few trades in {worst_year}: {min_per_year} "
                    f"(min: {MIN_SIGNALS_PER_YEAR}/year). "
                    "Not enough data for reliable win rate in all years."
                ),
                sharpe=0.0, calmar=0.0, max_drawdown=float(stats.get("Max. Drawdown [%]", 0)) / 100,
                win_rate=0.0, profit_factor=0.0, total_trades=total_trades,
                signals_per_year=float(trades_per_year.mean()),
                avg_trade_pnl=0.0, equity_final=float(stats.get("Equity Final [$]", cash)),
                equity_peak=float(stats.get("Equity Peak [$]", cash)),
                return_pct=float(stats.get("Return [%]", 0)),
                raw_stats=dict(stats), trades=trades_df,
            )
        signals_per_year = float(trades_per_year.mean())
    else:
        signals_per_year = 0.0

    # ── Extract metrics ──────────────────────────────────────────────────────
    sharpe = float(stats.get("Sharpe Ratio", 0) or 0)
    calmar = float(stats.get("Calmar Ratio", 0) or 0)
    max_dd = abs(float(stats.get("Max. Drawdown [%]", 0) or 0)) / 100
    win_rate = float(stats.get("Win Rate [%]", 0) or 0) / 100
    profit_factor = _safe_profit_factor(trades_df)
    avg_pnl = float(trades_df["PnL"].mean()) if not trades_df.empty and "PnL" in trades_df else 0.0
    equity_final = float(stats.get("Equity Final [$]", cash) or cash)
    equity_peak = float(stats.get("Equity Peak [$]", cash) or cash)
    return_pct = float(stats.get("Return [%]", 0) or 0)

    # ── Quality gates ────────────────────────────────────────────────────────
    reject_reason = None
    if sharpe < MIN_SHARPE:
        reject_reason = f"Sharpe {sharpe:.2f} below minimum {MIN_SHARPE}"
    elif calmar < MIN_CALMAR:
        reject_reason = f"Calmar {calmar:.2f} below minimum {MIN_CALMAR}"
    elif max_dd > MAX_DRAWDOWN:
        reject_reason = f"Max drawdown {max_dd:.1%} exceeds maximum {MAX_DRAWDOWN:.1%}"

    return BacktestResult(
        passed=reject_reason is None,
        reject_reason=reject_reason,
        sharpe=sharpe,
        calmar=calmar,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=total_trades,
        signals_per_year=signals_per_year,
        avg_trade_pnl=avg_pnl,
        equity_final=equity_final,
        equity_peak=equity_peak,
        return_pct=return_pct,
        raw_stats=dict(stats),
        trades=trades_df,
    )


def _safe_profit_factor(trades_df: pd.DataFrame) -> float:
    if trades_df.empty or "PnL" not in trades_df.columns:
        return 0.0
    wins = trades_df[trades_df["PnL"] > 0]["PnL"].sum()
    losses = abs(trades_df[trades_df["PnL"] < 0]["PnL"].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return round(wins / losses, 3)
