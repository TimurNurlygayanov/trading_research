"""
Modal job: runs the full backtest pipeline for a strategy.

Pipeline:
1. Load strategy code from DB
2. Fetch OHLCV data — served from Modal Volume cache after first download
3. Split train/OOS
4. Leakage check on code
5. Dynamic code execution: exec() the strategy class into a local namespace
6. Optimize with Optuna on train data (25 trials, 8 CPUs)
7. Walk-forward validation: 3 folds × 10 trials
8. Run final backtest on full train data with best params
9. Run OOS backtest (2026+) with best params
10. Update DB with all results

Uses Modal for parallelism: 8 CPUs, 8 GB RAM, 20 min timeout.
OHLCV data cached in Modal Volume — only downloaded once per symbol/timeframe.
"""
import os as _os
import modal

_HERE = _os.path.dirname(_os.path.abspath(__file__))   # .../modal_jobs/
_ROOT = _os.path.dirname(_HERE)                         # .../trading_research/

app = modal.App("trading-research-backtest")

# Bake local source packages into the image so they're always importable.
# Using add_local_dir with absolute paths avoids all CWD / __file__ ambiguity.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "pandas>=2.3.2",
        "pandas_ta==0.4.71b0",
        "numpy",
        "backtesting==0.3.3",
        "bokeh>=3.0",
        "optuna==3.6.1",
        "scikit-learn==1.5.2",
        "requests==2.32.3",
        "supabase==2.9.1",
        "python-dotenv==1.0.1",
        "scipy==1.14.1",
        "pyarrow",
        "massive",             # market data client used by data_fetcher
        "anthropic>=0.25.0",
    )
    .add_local_dir(_os.path.join(_ROOT, "db"),       remote_path="/root/db")
    .add_local_dir(_os.path.join(_ROOT, "agents"),   remote_path="/root/agents")
    .add_local_dir(_os.path.join(_ROOT, "backtest"), remote_path="/root/backtest")
)

# Persistent volume — OHLCV data cached here across runs
ohlcv_cache = modal.Volume.from_name("trading-research-ohlcv-cache", create_if_missing=True)
CACHE_DIR = "/ohlcv_cache"

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=300,  # 5 minutes — just one run, no optimization
    secrets=[modal.Secret.from_name("trading-research-secrets")],
    volumes={CACHE_DIR: ohlcv_cache},
)
def run_quick_backtest(strategy_id: str) -> dict:
    """
    Quick backtest: run strategy code with default class-level params, no optimization.
    Goal: confirm the strategy logic works and get a first read on results in ~2 min.
    Stores quick_test_* metrics and sets status to 'quick_tested'.
    The full optimization pipeline runs separately after this.
    """
    import os
    import traceback
    import pandas as pd
    from backtesting import Strategy

    try:
        from db import supabase_client as db
        from agents.utils import add_pipeline_note
        from backtest.data_fetcher import fetch_ohlcv, split_train_oos
        from backtest.engine import run_backtest
        from backtest.leakage_detector import check_leakage

        strategy = db.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {strategy_id} not found")

        code = strategy.get("backtest_code")
        if not code:
            raise ValueError("No backtest_code on strategy")

        indicators_meta = strategy.get("indicators") or {}
        symbols = indicators_meta.get("symbols", ["EURUSD"])
        if isinstance(symbols, str):
            symbols = [symbols]
        symbol = symbols[0] if symbols else "EURUSD"

        timeframes = (
            indicators_meta.get("timeframes")
            or strategy.get("timeframes")
            or ["1h"]
        )
        primary_tf = timeframes[0] if isinstance(timeframes, list) and timeframes else "1h"

        db.update_strategy(strategy_id, {"status": "quick_testing"})
        add_pipeline_note(strategy_id,
            f"Quick test started — {symbol} {primary_tf}, default params, no optimization.")

        # Leakage check (fast, before any data work)
        leakage_result = check_leakage(code)
        if not leakage_result.passed:
            db.update_strategy(strategy_id, {
                "status": "failed",
                "leakage_score": leakage_result.score,
                "leakage_issues": leakage_result.issues,
                "error_log": (
                    f"Leakage check failed (score={leakage_result.score}): "
                    f"{leakage_result.issues[:3]}"
                ),
            })
            add_pipeline_note(strategy_id,
                f"Quick test FAILED leakage — score {leakage_result.score}/10. "
                f"Issues: {'; '.join(leakage_result.issues[:3])}")
            return {"passed": False, "reason": "leakage_check_failed",
                    "issues": leakage_result.issues}

        # Load data (from cache if available)
        cache_file = f"{CACHE_DIR}/{symbol}_{primary_tf}.parquet"
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            add_pipeline_note(strategy_id,
                f"Data loaded from cache ({len(df)} bars).")
        else:
            df = fetch_ohlcv(symbol, primary_tf, start="2015-01-01", end="2026-12-31")
            os.makedirs(CACHE_DIR, exist_ok=True)
            df.to_parquet(cache_file)
            ohlcv_cache.commit()
            add_pipeline_note(strategy_id,
                f"Data downloaded and cached ({len(df)} bars).")
        train_df, _ = split_train_oos(df)

        # Execute the strategy class
        namespace: dict = {}
        exec(compile(code, "<strategy>", "exec"), namespace)
        strategy_classes = {
            v.__name__: v for v in namespace.values()
            if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy
        }
        if not strategy_classes:
            raise ValueError("No Strategy subclass found in generated code")
        expected_name = indicators_meta.get("strategy_class", "")
        if expected_name and expected_name in strategy_classes:
            strategy_class = strategy_classes[expected_name]
        else:
            strategy_class = next(iter(strategy_classes.values()))

        # Run with default class-level params — no optimization, enforce_gates=False
        # so we see actual metrics even if they don't pass quality gates yet.
        # generate_html=True produces an interactive Bokeh equity-curve report.
        result = run_backtest(strategy_class, train_df, params={},
                              enforce_gates=False, generate_html=True)

        # Save quick test results and advance pipeline regardless of metric quality.
        # The full optimization may find much better params.
        db.update_strategy(strategy_id, {
            "status": "quick_tested",
            "modal_job_id": None,
            "quick_test_sharpe": result.sharpe,
            "quick_test_calmar": result.calmar,
            "quick_test_drawdown": result.max_drawdown,
            "quick_test_trades": result.total_trades,
            "quick_test_win_rate": result.win_rate,
            "quick_test_signals_per_year": result.signals_per_year,
            "leakage_score": leakage_result.score,
            "leakage_issues": leakage_result.issues,
            # Store equity-curve HTML report so user can inspect the strategy visually
            "report_text": result.html_report,
            "error_log": None,
        })

        trade_note = (
            f"Sharpe={result.sharpe:.4f} (trade_sharpe={result.trade_sharpe:.4f}), "
            f"trades={result.total_trades}, "
            f"win={result.win_rate:.0%}, "
            f"pf={result.profit_factor:.3f}, "
            f"drawdown={result.max_drawdown:.1%}"
        )
        if result.total_trades == 0:
            trade_note += " — ⚠️ NO TRADES with default params (optimizer may still find signal)"
        add_pipeline_note(strategy_id,
            f"Quick test done — {trade_note}. Proceeding to full optimization.")

        return {
            "passed": True,
            "strategy_id": strategy_id,
            "quick_sharpe": result.sharpe,
            "quick_trades": result.total_trades,
            "quick_win_rate": result.win_rate,
        }

    except Exception as e:
        tb = traceback.format_exc()
        try:
            from db import supabase_client as db2
            db2.update_strategy(strategy_id, {
                "status": "failed",
                "error_log": f"Quick backtest error: {type(e).__name__}: {e}\n{tb[:500]}",
            })
        except Exception:
            pass
        raise


@app.function(
    image=image,
    cpu=8,
    memory=8192,
    timeout=1200,  # 20 minutes — enough headroom with caching + fewer trials
    secrets=[modal.Secret.from_name("trading-research-secrets")],
    volumes={CACHE_DIR: ohlcv_cache},
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
        from backtest.walk_forward import walk_forward
        from backtest.monte_carlo import monte_carlo_test as run_monte_carlo

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
        # ── REQUIRED: every strategy must be validated on both 1h and 5m ──
        REQUIRED_TIMEFRAMES = ["1h", "5m"]
        for rtf in REQUIRED_TIMEFRAMES:
            if rtf not in timeframes:
                timeframes.append(rtf)
        # Primary timeframe is the first one (used for optimization)
        primary_tf = timeframes[0]

        db.update_strategy(strategy_id, {"status": "backtesting"})

        add_pipeline_note(strategy_id, f"Backtest job started on Modal — symbol={symbol}, timeframes={timeframes} (primary={primary_tf}).")

        # 2. Leakage check (fast — before any data work)
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

        # 3. Fetch data for primary timeframe
        import os as _os
        cache_file = f"{CACHE_DIR}/{symbol}_{primary_tf}.parquet"
        if _os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            add_pipeline_note(strategy_id, f"Data loaded from cache for {primary_tf} ({len(df)} bars).")
        else:
            df = fetch_ohlcv(symbol, primary_tf, start="2015-01-01", end="2026-12-31")
            _os.makedirs(CACHE_DIR, exist_ok=True)
            df.to_parquet(cache_file)
            ohlcv_cache.commit()
            add_pipeline_note(strategy_id, f"Data downloaded and cached for {primary_tf} ({len(df)} bars).")
        train_df, oos_df = split_train_oos(df)

        # 4. Execute the strategy class from code string
        namespace: dict = {}
        exec(compile(code, "<strategy>", "exec"), namespace)
        strategy_classes = {
            v.__name__: v for v in namespace.values()
            if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy
        }
        if not strategy_classes:
            raise ValueError("No Strategy subclass found in generated code")
        # Prefer the class whose name matches what the Implementer recorded
        expected_class_name = (strategy.get("indicators") or {}).get("strategy_class", "")
        if expected_class_name and expected_class_name in strategy_classes:
            strategy_class = strategy_classes[expected_class_name]
        else:
            strategy_class = next(iter(strategy_classes.values()))

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

        # 5.5. Recent-data gate: quick sanity check on last 90 days of OOS data.
        # Strategy must be non-negative (or have too few trades to judge) in recent history.
        # This runs in < 5 seconds and saves the full walk-forward if the strategy is broken recently.
        RECENT_CHECK_DAYS = 90
        if len(oos_df) >= 50:
            try:
                recent_cutoff = oos_df.index[-1] - pd.Timedelta(days=RECENT_CHECK_DAYS)
                recent_df = oos_df[oos_df.index >= recent_cutoff]
                if len(recent_df) >= 30:
                    recent_result = run_backtest(strategy_class, recent_df, params=best_params)
                    add_pipeline_note(
                        strategy_id,
                        f"Recent {RECENT_CHECK_DAYS}-day check: "
                        f"Sharpe={recent_result.sharpe:.3f}, trades={recent_result.total_trades}."
                    )
                    if recent_result.total_trades >= 3 and recent_result.sharpe < -0.3:
                        reason = (
                            f"Strategy unprofitable in recent {RECENT_CHECK_DAYS} days: "
                            f"Sharpe={recent_result.sharpe:.3f}, trades={recent_result.total_trades}. "
                            "Possible regime change or curve-fitting."
                        )
                        db.update_strategy(strategy_id, {
                            "status": "failed",
                            "error_log": reason,
                            "hyperparams": best_params,
                        })
                        add_pipeline_note(strategy_id, f"Recent data gate FAILED — {reason}")
                        return {"passed": False, "reason": reason}
            except Exception as _rce:
                add_pipeline_note(strategy_id, f"Recent data check skipped (error: {_rce}).")

        # 6. Walk-forward validation
        wf_result = walk_forward(strategy_class, train_df, opt_space, n_folds=3, n_trials=10)
        wf_scores = wf_result.oos_sharpes  # list of OOS Sharpe per fold

        add_pipeline_note(
            strategy_id,
            f"Optuna done — best params: {best_params}.\n"
            f"Walk-forward OOS Sharpes: {[round(s,3) for s in wf_scores]}. "
            f"Mean={wf_result.mean_oos_sharpe:.3f}, overfit ratio={wf_result.overfitting_ratio:.1f}x."
        )

        if not wf_result.passed:
            db.update_strategy(strategy_id, {
                "status": "failed",
                "walk_forward_scores": wf_scores,
                "hyperparams": best_params,
                "error_log": wf_result.reject_reason,
            })
            add_pipeline_note(strategy_id, f"Walk-forward FAILED — {wf_result.reject_reason}")
            return {"passed": False, "reason": wf_result.reject_reason}

        # 7. Final backtest on full training data
        train_result = run_backtest(strategy_class, train_df, params=best_params)

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

        # OOS degradation gate: OOS Sharpe must be at least 40% of train Sharpe
        # and must be positive. A strategy with train=2.0 / OOS=0.1 is curve-fitted.
        if oos_result and train_result.sharpe > 0:
            min_oos_sharpe = max(0.6, 0.5 * train_result.sharpe)
            if oos_result.sharpe < min_oos_sharpe:
                reason = (
                    f"OOS degradation too severe: train Sharpe={train_result.sharpe:.3f}, "
                    f"OOS Sharpe={oos_result.sharpe:.3f} "
                    f"(minimum required={min_oos_sharpe:.3f} — floor 0.6 or 50% of train)."
                )
                db.update_strategy(strategy_id, {
                    "status": "failed",
                    "backtest_sharpe": train_result.sharpe,
                    "oos_sharpe": oos_result.sharpe,
                    "error_log": reason,
                    "hyperparams": best_params,
                })
                add_pipeline_note(strategy_id, f"OOS degradation FAILED — {reason}")
                return {"passed": False, "reason": reason}

        # 9. Monte Carlo test (requires enough trades for meaningful p-value)
        mc_pvalue = None
        MIN_TRADES_FOR_MC = 30
        if (
            train_result.trades is not None
            and not train_result.trades.empty
            and len(train_result.trades) >= MIN_TRADES_FOR_MC
        ):
            mc_result = run_monte_carlo(train_result.trades, train_result.sharpe)
            mc_pvalue = mc_result.p_value
        elif train_result.trades is not None:
            add_pipeline_note(
                strategy_id,
                f"Monte Carlo skipped — only {len(train_result.trades)} trades "
                f"(minimum {MIN_TRADES_FOR_MC} required for reliable p-value)."
            )

        # 10. Update DB with full results (clear modal_job_id so validator can dispatch)
        db.update_strategy(strategy_id, {
            "status": "validating",
            "modal_job_id": None,
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
