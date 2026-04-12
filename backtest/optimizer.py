"""
Optuna-based hyperparameter optimizer for trading strategies.
Runs on Modal (multi-CPU) for fast tuning.

Usage:
  best_params, study = optimize_strategy(
      strategy_class=MyStrategy,
      train_df=df_train,
      param_space={
          "rsi_period": ("int", 5, 30),
          "atr_multiplier": ("float", 1.0, 4.0),
          "start_hour": ("int", 0, 23),
          "end_hour": ("int", 0, 23),
          "max_daily_losses": ("int", 1, 5),
      },
      n_trials=50,
      n_jobs=4,
  )
"""
from __future__ import annotations

import logging
from typing import Any, Type

import optuna
import pandas as pd
from backtesting import Strategy

from backtest.engine import run_backtest

optuna.logging.set_verbosity(optuna.logging.WARNING)

ParamSpace = dict[str, tuple]


def optimize_strategy(
    strategy_class: Type[Strategy],
    train_df: pd.DataFrame,
    param_space: ParamSpace,
    n_trials: int = 50,
    n_jobs: int = -1,
    seed: int = 42,
    direction: str = "maximize",
    metric: str = "sharpe",
) -> tuple[dict[str, Any], optuna.Study]:
    """
    Optimize strategy hyperparameters on training data.

    Parameters
    ----------
    strategy_class : backtesting.py Strategy subclass
    train_df       : OHLCV DataFrame (training period only — never include OOS data)
    param_space    : dict of {param: (type, *args)} where type is "int", "float", or "categorical"
                     Examples:
                       "rsi_period": ("int", 5, 30)         -> suggest_int(5, 30)
                       "threshold": ("float", 0.001, 0.1)   -> suggest_float(0.001, 0.1)
                       "mode": ("categorical", ["A", "B"])  -> suggest_categorical(["A", "B"])
    n_trials       : number of Optuna trials (50 is usually enough for 5-10 params)
    n_jobs         : parallel jobs (-1 = all CPUs — use on Modal for speed)
    seed           : random seed for reproducibility
    direction      : "maximize" (default) or "minimize"
    metric         : which metric to optimize: "sharpe" | "calmar" | "profit_factor"

    Returns
    -------
    (best_params, study)
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, param_space)

        # Constraint: start_hour must be < end_hour
        if "start_hour" in params and "end_hour" in params:
            if params["start_hour"] >= params["end_hour"]:
                return -10.0  # Invalid config

        result = run_backtest(strategy_class, train_df, params=params)
        if not result.passed and result.total_trades < 50:
            return -5.0  # Too few signals — prune this direction

        value = {
            "sharpe": result.sharpe,
            "calmar": result.calmar,
            "profit_factor": result.profit_factor,
        }.get(metric, result.sharpe)

        return value if not (value != value) else -10.0  # handle NaN

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)

    best_params = study.best_params if study.best_trial else {}
    logging.info(
        f"Optimization complete: {n_trials} trials, best {metric}={study.best_value:.3f}, "
        f"params={best_params}"
    )
    return best_params, study


def _suggest_params(trial: optuna.Trial, space: ParamSpace) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            log = len(spec) > 3 and spec[3] == "log"
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=log)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown param type: {kind} for param {name}")
    return params
