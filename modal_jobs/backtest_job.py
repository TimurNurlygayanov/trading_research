"""
Modal job: runs the full backtest pipeline for a strategy.
Called by the Render orchestrator via HTTP when a strategy is ready for backtesting.

Pipeline:
1. Load strategy code from DB
2. Fetch OHLCV data (all available history)
3. Split train/OOS
4. Leakage check on code
5. Dynamic code execution: exec() the strategy class into a local namespace
6. Optimize with Optuna on train data (50 trials, 4 CPUs)
7. Walk-forward validation on train data
8. Run final backtest on full train data with best params
9. Run OOS backtest (2026+) with best params
10. Update DB with all results
11. Return results dict

Uses Modal for parallelism: 4 CPUs, 8 GB RAM, 30 min timeout.
"""
import modal

app = modal.App("trading-research-backtest")

# Define the Modal image with all required packages
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "pandas>=2.3.2",
        "pandas_ta==0.4.71b0",
        "numpy",
        "backtesting==0.3.3",
        "optuna==3.6.1",
        "scikit-learn==1.5.2",
        "requests==2.32.3",
        "supabase==2.9.1",
        "python-dotenv==1.0.1",
        "scipy==1.14.1",
    )
)

@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=1800,  # 30 minutes
    secrets=[modal.Secret.from_name("trading-research-secrets")],
)
def run_backtest_pipeline(strategy_id: str) -> dict:
    """
    Full backtest pipeline for one strategy.
    Imports are inside the function because this runs in Modal's remote environment.
    """
    import sys
    import os
    import json
    import traceback
    import types

    # These imports work because we pip_install them in the image
    import pandas as pd
    import numpy as np
    from backtesting import Strategy

    # Add the project root to path so our modules are importable
    # (Modal mounts the local code directory)

    try:
        from db import supabase_client as db
        from agents.utils import add_pipeline_note
        from backtest.data_fetcher import fetch_ohlcv, split_train_oos
        from backtest.engine import run_backtest
        from backtest.leakage_detector import check_leakage
        from backtest.optimizer import optimize_strategy
        from backtest.walk_forward import walk_forward_validation
        from backtest.monte_carlo import run_monte_carlo

        # 1. Load strategy
        strategy = db.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {strategy_id} not found")

        code = strategy.get("backtest_code")
        if not code:
            raise ValueError("No backtest_code found on strategy")

        param_space_raw = strategy.get("hyperparams") or {}
        # hyperparams at this stage stores param_space from implementer
        param_space = param_space_raw if isinstance(param_space_raw, dict) else json.loads(param_space_raw)

        symbols = strategy.get("indicators", {}).get("symbols", ["EURUSD"])
        if isinstance(symbols, str):
            symbols = [symbols]
        symbol = symbols[0] if symbols else "EURUSD"

        timeframes = strategy.get("timeframes") or ["1h"]
        timeframe = timeframes[0]

        db.update_strategy(strategy_id, {"status": "backtesting"})

        add_pipeline_note(strategy_id, f"Backtest job started on Modal — symbol={symbol}, timeframe={timeframe}.")

        # 2. Leakage check
        leakage_result = check_leakage(code)
        if not leakage_result.passed:
            db.update_strategy(strategy_id, {
                "status": "failed",
                "leakage_score": leakage_result.score,
                "leakage_issues": leakage_result.issues,
                "error_log": f"Leakage check failed (score={leakage_result.score}): {leakage_result.issues[:3]}",
            })
            add_pipeline_note(strategy_id, f"Backtest FAILED leakage check — score {leakage_result.score}/10. Issues: {'; '.join(leakage_result.issues[:3])}")
            return {"passed": False, "reason": "leakage_check_failed", "issues": leakage_result.issues}

        # 3. Fetch data
        df = fetch_ohlcv(symbol, timeframe, start="2015-01-01", end="2026-12-31")
        train_df, oos_df = split_train_oos(df)

        # 4. Execute the strategy class from code string
        namespace: dict = {}
        exec(compile(code, "<strategy>", "exec"), namespace)
        strategy_classes = [
            v for v in namespace.values()
            if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy
        ]
        if not strategy_classes:
            raise ValueError("No Strategy subclass found in generated code")
        strategy_class = strategy_classes[0]

        # 5. Optimize on training data
        # Convert param_space from JSON format to optimizer tuple format.
        # LLM may produce either:
        #   ["int", 5, 30]               → ("int", 5, 30)
        #   ["float", 1.0, 4.0]          → ("float", 1.0, 4.0)
        #   ["categorical", 1.2, 1.5, 2] → ("categorical", [1.2, 1.5, 2])
        #   ["categorical", [1.2, 1.5]]  → ("categorical", [1.2, 1.5])
        opt_space = {}
        for k, v in param_space.items():
            if not isinstance(v, list) or len(v) < 2:
                continue
            kind = v[0]
            if kind == "categorical":
                choices = v[1] if isinstance(v[1], list) else v[1:]
                opt_space[k] = ("categorical", list(choices))
            else:
                opt_space[k] = tuple(v)

        best_params, study = optimize_strategy(
            strategy_class, train_df, opt_space, n_trials=50, n_jobs=4
        )

        # 6. Walk-forward validation
        wf_scores = walk_forward_validation(
            strategy_class, train_df, opt_space, n_folds=5
        )

        # 7. Final backtest on full training data
        train_result = run_backtest(strategy_class, train_df, params=best_params)

        add_pipeline_note(
            strategy_id,
            f"Optuna done — best params: {best_params}. "
            f"Walk-forward scores: {[round(s,3) for s in (wf_scores or [])]}."
        )

        if not train_result.passed:
            db.update_strategy(strategy_id, {
                "status": "failed",
                "backtest_sharpe": train_result.sharpe,
                "max_drawdown": train_result.max_drawdown,
                "total_signals": train_result.total_trades,
                "error_log": train_result.reject_reason,
                "hyperparams": best_params,
            })
            add_pipeline_note(strategy_id, f"Backtest FAILED — {train_result.reject_reason}")
            return {"passed": False, "reason": train_result.reject_reason}

        # 8. OOS backtest
        oos_result = run_backtest(strategy_class, oos_df, params=best_params)

        # 9. Monte Carlo test on training trades
        mc_pvalue = None
        if train_result.trades is not None and not train_result.trades.empty:
            mc_result = run_monte_carlo(train_result.trades, train_result.sharpe)
            mc_pvalue = mc_result.p_value

        # 10. Update DB with full results
        db.update_strategy(strategy_id, {
            "status": "validating",
            "backtest_sharpe": train_result.sharpe,
            "backtest_calmar": train_result.calmar,
            "max_drawdown": train_result.max_drawdown,
            "total_signals": train_result.total_trades,
            "signals_per_year": train_result.signals_per_year,
            "win_rate": train_result.win_rate,
            "profit_factor": train_result.profit_factor,
            "avg_trade_pnl": train_result.avg_trade_pnl,
            "oos_sharpe": oos_result.sharpe if oos_result else None,
            "oos_win_rate": oos_result.win_rate if oos_result else None,
            "oos_total_trades": oos_result.total_trades if oos_result else None,
            "leakage_score": leakage_result.score,
            "leakage_issues": leakage_result.issues,
            "monte_carlo_pvalue": mc_pvalue,
            "walk_forward_scores": wf_scores,
            "hyperparams": best_params,
        })

        oos_sharpe_str = f"{oos_result.sharpe:.3f}" if oos_result else "N/A"
        mc_str = f"{mc_pvalue:.4f}" if mc_pvalue is not None else "N/A"
        add_pipeline_note(
            strategy_id,
            f"Backtest passed — train Sharpe {train_result.sharpe:.3f}, "
            f"OOS Sharpe {oos_sharpe_str}, {train_result.total_trades} trades, "
            f"Monte Carlo p={mc_str}. Sending to validator."
        )

        return {
            "passed": True,
            "strategy_id": strategy_id,
            "train_sharpe": train_result.sharpe,
            "oos_sharpe": oos_result.sharpe if oos_result else None,
            "total_trades": train_result.total_trades,
            "mc_pvalue": mc_pvalue,
        }

    except Exception as e:
        tb = traceback.format_exc()
        try:
            from db import supabase_client as db2
            db2.update_strategy(strategy_id, {
                "status": "failed",
                "error_log": f"{type(e).__name__}: {e}\n{tb[:500]}",
                "retry_count": (db2.get_strategy(strategy_id) or {}).get("retry_count", 0) + 1,
            })
        except Exception:
            pass
        raise
