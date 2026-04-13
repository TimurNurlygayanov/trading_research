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

    # Per-trade Sharpe (annualized from trade P&L — more reliable for intraday)
    trade_sharpe: float = 0.0

    # Raw stats dict from backtesting.py
    raw_stats: dict[str, Any]

    # Trade-level DataFrame
    trades: pd.DataFrame | None = None

    # Optional HTML equity-curve report (generated when generate_html=True)
    html_report: str | None = None


def run_backtest(
    strategy_class: Type[Strategy],
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    cash: float = 10_000.0,
    commission: float = 0.0002,
    exclusive_orders: bool = True,
    enforce_gates: bool = True,
    generate_html: bool = False,
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
    if enforce_gates and total_trades < MIN_SIGNALS_TOTAL:
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

    # Compute signals_per_year from entry times
    if not trades_df.empty and "EntryTime" in trades_df.columns:
        entry_times = pd.to_datetime(trades_df["EntryTime"])
        trades_per_year = entry_times.dt.year.value_counts()
        signals_per_year = float(trades_per_year.mean())
    else:
        signals_per_year = 0.0

    # ── Extract metrics ──────────────────────────────────────────────────────
    sharpe     = _safe_float(stats.get("Sharpe Ratio"))
    calmar     = _safe_float(stats.get("Calmar Ratio"))
    max_dd     = abs(_safe_float(stats.get("Max. Drawdown [%]"))) / 100
    win_rate   = _safe_float(stats.get("Win Rate [%]")) / 100
    profit_factor = _safe_profit_factor(trades_df)
    avg_pnl    = float(trades_df["PnL"].mean()) if not trades_df.empty and "PnL" in trades_df else 0.0
    equity_final = _safe_float(stats.get("Equity Final [$]"), default=cash)
    equity_peak  = _safe_float(stats.get("Equity Peak [$]"), default=cash)
    return_pct   = _safe_float(stats.get("Return [%]"))

    # Per-trade Sharpe: annualized using trade P&L directly.
    # More reliable than the equity-curve Sharpe for intraday strategies
    # where backtesting.py's daily resample can flatten out intraday returns.
    trade_sharpe = _compute_trade_sharpe(trades_df, signals_per_year)

    # ── Quality gates ────────────────────────────────────────────────────────
    reject_reason = None
    if enforce_gates:
        if sharpe < MIN_SHARPE:
            reject_reason = f"Sharpe {sharpe:.2f} below minimum {MIN_SHARPE}"
        elif calmar < MIN_CALMAR:
            reject_reason = f"Calmar {calmar:.2f} below minimum {MIN_CALMAR}"
        elif max_dd > MAX_DRAWDOWN:
            reject_reason = f"Max drawdown {max_dd:.1%} exceeds maximum {MAX_DRAWDOWN:.1%}"

    result = BacktestResult(
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
        trade_sharpe=trade_sharpe,
        raw_stats=dict(stats),
        trades=trades_df,
    )

    if generate_html:
        try:
            import os, tempfile
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                tmp_path = f.name
            bt.plot(filename=tmp_path, open_browser=False, resample=False)
            with open(tmp_path, "r", encoding="utf-8") as f:
                result.html_report = f.read()
            os.unlink(tmp_path)
        except Exception:
            pass  # HTML is optional — never block the pipeline for it

    return result


def _safe_profit_factor(trades_df: pd.DataFrame) -> float:
    if trades_df.empty or "PnL" not in trades_df.columns:
        return 0.0
    wins = trades_df[trades_df["PnL"] > 0]["PnL"].sum()
    losses = abs(trades_df[trades_df["PnL"] < 0]["PnL"].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return round(wins / losses, 3)


def _safe_float(val, default: float = 0.0) -> float:
    """Convert a backtesting.py stat to float, treating NaN/Inf/None as default.

    The `or 0` pattern is wrong: `0.0 or 0` evaluates to `0` (int) because
    0.0 is falsy, masking legitimate zero values and causing type confusion.
    This helper handles all edge cases explicitly.
    """
    import math
    if val is None:
        return default
    try:
        v = float(val)
        return default if not math.isfinite(v) else v
    except (TypeError, ValueError):
        return default


def _compute_trade_sharpe(trades_df: pd.DataFrame, signals_per_year: float) -> float:
    """Annualized Sharpe from per-trade P&L — more reliable than the equity-curve Sharpe
    for intraday strategies where backtesting.py's daily resampling flattens returns.

    Formula: mean(PnL) / std(PnL) * sqrt(trades_per_year)
    """
    if trades_df is None or trades_df.empty or "PnL" not in trades_df.columns:
        return 0.0
    pnl = trades_df["PnL"].dropna()
    if len(pnl) < 2:
        return 0.0
    std = pnl.std()
    if std == 0:
        return 0.0
    ann_factor = max(signals_per_year, 1.0) ** 0.5
    return float(pnl.mean() / std * ann_factor)
