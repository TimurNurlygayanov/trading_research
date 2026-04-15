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
    memory=8192,
    timeout=900,  # 15 minutes — tests all timeframes sequentially
    secrets=[modal.Secret.from_name("trading-research-secrets")],
    volumes={CACHE_DIR: ohlcv_cache},
)
def run_quick_backtest(strategy_id: str) -> dict:
    """
    Multi-timeframe quick test: run strategy with default params on every standard timeframe
    and pick the best one. Goal: don't discard a good strategy just because it was tested
    on the wrong timeframe. Stores best_timeframe + quick_test_* metrics, sets 'quick_tested'.
    """
    import os
    import traceback
    import pandas as pd
    from backtesting import Strategy

    # Test all standard timeframes — ordered fastest→slowest so partial results
    # are useful if the job times out near the end.
    QUICK_TEST_TIMEFRAMES = ["4h", "1h", "15m", "5m", "1m"]

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

        db.update_strategy(strategy_id, {"status": "quick_testing"})
        add_pipeline_note(strategy_id,
            f"Quick test started — {symbol}, testing all timeframes: "
            f"{QUICK_TEST_TIMEFRAMES}, default params, no optimization.")

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

        # Execute the strategy class once (shared across all timeframe runs)
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

        # ── Run backtest on each timeframe ────────────────────────────────────
        def _clean(v, ndigits=4):
            """Round float; replace NaN/inf with None so JSONB accepts it."""
            import math
            if v is None:
                return None
            try:
                f = float(v)
                return None if (math.isnan(f) or math.isinf(f)) else round(f, ndigits)
            except (TypeError, ValueError):
                return None

        tf_results: dict = {}   # tf -> metrics dict
        tf_dates:   dict = {}   # tf -> (start_str, end_str, n_bars)
        cache_committed = False

        for tf in QUICK_TEST_TIMEFRAMES:
            try:
                cache_file = f"{CACHE_DIR}/{symbol}_{tf}.parquet"
                if os.path.exists(cache_file):
                    df = pd.read_parquet(cache_file)
                else:
                    df = fetch_ohlcv(symbol, tf, start="2015-01-01", end="2026-12-31")
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    df.to_parquet(cache_file)
                    cache_committed = False  # will commit after all downloads

                train_df, _ = split_train_oos(df)
                tf_dates[tf] = (
                    train_df.index[0].strftime("%Y-%m-%d"),
                    train_df.index[-1].strftime("%Y-%m-%d"),
                    len(train_df),
                )
                result = run_backtest(strategy_class, train_df, params={},
                                      enforce_gates=False)

                tf_results[tf] = {
                    "sharpe":           _clean(result.sharpe),
                    "equity_sharpe":    _clean(result.equity_sharpe),
                    "trades":           result.total_trades,
                    "win_rate":         _clean(result.win_rate),
                    "drawdown":         _clean(result.max_drawdown),
                    "profit_factor":    _clean(result.profit_factor, 3),
                    "signals_per_year": _clean(result.signals_per_year, 1),
                    "bars":             len(train_df),
                }
            except Exception as tf_err:
                tf_results[tf] = {"error": str(tf_err)[:200], "sharpe": -999.0}

        if not cache_committed:
            try:
                ohlcv_cache.commit()
            except Exception:
                pass

        # ── Pick the best timeframe ───────────────────────────────────────────
        # Require at least 10 trades to be considered valid; prefer by Sharpe.
        valid = {tf: m for tf, m in tf_results.items()
                 if "error" not in m and m.get("trades", 0) >= 10}
        if valid:
            best_tf = max(valid, key=lambda tf: valid[tf]["sharpe"])
        else:
            # No valid timeframe — pick whichever had the most trades (even if 0)
            best_tf = max(
                (tf for tf in tf_results if "error" not in tf_results[tf]),
                key=lambda tf: tf_results[tf].get("trades", 0),
                default=QUICK_TEST_TIMEFRAMES[1],  # fallback: 1h
            )

        best = tf_results.get(best_tf, {})

        # ── Hard fail: 0 trades on every timeframe ───────────────────────────
        # This means the strategy code has a bug — session filter kills all signals,
        # indicator init fails, or entry condition is never True on any bar.
        # Proceeding to optimization is pointless; flag it now.
        ran_ok = [m for m in tf_results.values() if "error" not in m]
        total_trades_across_all = sum(m.get("trades", 0) for m in ran_ok)
        if ran_ok and total_trades_across_all == 0:
            tf_summary = "\n".join(
                f"  {tf:>4s}: {tf_results[tf].get('error', '0 trades')}"
                for tf in QUICK_TEST_TIMEFRAMES
            )
            reason = (
                "Zero trades on ALL timeframes with default params. "
                "Likely causes: session filter (start_hour/end_hour) blocks all bars, "
                "indicator returns NaN for the entire series, or entry condition is never True. "
                "Fix the strategy code before re-submitting."
            )
            db.update_strategy(strategy_id, {
                "status": "failed",
                "modal_job_id": None,
                "quick_test_all_timeframes": tf_results,
                "error_log": reason,
                "leakage_score": leakage_result.score,
                "leakage_issues": leakage_result.issues,
            })
            add_pipeline_note(strategy_id,
                f"Quick test FAILED — 0 trades on all timeframes.\n{tf_summary}\n\n{reason}")
            return {"passed": False, "reason": "zero_trades_all_timeframes",
                    "tf_results": tf_results}

        # Build a compact one-line summary per timeframe for the pipeline note
        tf_lines = []
        for tf in QUICK_TEST_TIMEFRAMES:
            m = tf_results.get(tf, {})
            if "error" in m:
                tf_lines.append(f"  {tf:>4s}: ERROR — {m['error'][:60]}")
            else:
                marker = " ◀ best" if tf == best_tf else ""
                tf_lines.append(
                    f"  {tf:>4s}: Sharpe={m['sharpe']:+.4f}  "
                    f"trades={m['trades']}  win={m['win_rate']:.0%}  "
                    f"dd={m['drawdown']:.1%}{marker}"
                )

        # Date range info for the best timeframe
        if best_tf in tf_dates:
            t0, t1, n_bars = tf_dates[best_tf]
            years = round((pd.Timestamp(t1) - pd.Timestamp(t0)).days / 365.25, 1)
            date_line = f"\nTrain period: {t0} → {t1} ({years}y, {n_bars:,} bars on {best_tf})"
        else:
            date_line = ""

        add_pipeline_note(strategy_id,
            f"Multi-timeframe quick test complete — best: {best_tf} "
            f"(Sharpe={best.get('sharpe', 0):+.4f}, "
            f"trades={best.get('trades', 0)}, "
            f"win={best.get('win_rate', 0):.0%}, "
            f"pf={best.get('profit_factor', 0):.3f}, "
            f"dd={best.get('drawdown', 0):.1%})\n"
            + "\n".join(tf_lines)
            + date_line)

        # Generate HTML report for best timeframe + capture trades for analyzer
        html_report = None
        trades_for_db = None
        try:
            cache_file = f"{CACHE_DIR}/{symbol}_{best_tf}.parquet"
            best_df = pd.read_parquet(cache_file) if os.path.exists(cache_file) else None
            if best_df is not None:
                train_df, _ = split_train_oos(best_df)
                html_result = run_backtest(strategy_class, train_df, params={},
                                           enforce_gates=False, generate_html=True)
                html_report = html_result.html_report

                # Serialise essential trade columns for strategy_analyzer
                if html_result.trades is not None and not html_result.trades.empty:
                    keep = [c for c in ["EntryTime", "ExitTime", "PnL", "ReturnPct", "Size"]
                            if c in html_result.trades.columns]
                    if keep:
                        t = html_result.trades[keep].copy()
                        for col in ["EntryTime", "ExitTime"]:
                            if col in t.columns:
                                t[col] = t[col].astype(str)
                        trades_for_db = t.to_dict("records")
        except Exception:
            pass

        # Save results — quick_test_* fields reflect the best timeframe
        db.update_strategy(strategy_id, {
            "status":                      "quick_tested",
            "modal_job_id":                None,
            "best_timeframe":              best_tf,
            "quick_test_all_timeframes":   tf_results,
            "quick_test_trade_records":    trades_for_db,   # JSONB trade list for strategy_analyzer
            "analysis_done":               False,            # reset flag for analyzer
            "quick_test_sharpe":           best.get("sharpe"),
            "quick_test_calmar":           None,
            "quick_test_drawdown":         best.get("drawdown"),
            "quick_test_trades":           best.get("trades"),   # int count
            "quick_test_win_rate":         best.get("win_rate"),
            "quick_test_signals_per_year": best.get("signals_per_year"),
            "leakage_score":               leakage_result.score,
            "leakage_issues":              leakage_result.issues,
            "report_text":                 html_report,
            "error_log":                   None,
        })

        return {
            "passed":        True,
            "strategy_id":   strategy_id,
            "best_timeframe": best_tf,
            "best_sharpe":   best.get("sharpe"),
            "best_trades":   best.get("trades"),
            "all_timeframes": tf_results,
        }

    except Exception as e:
        tb = traceback.format_exc()
        try:
            from db import supabase_client as db2
            db2.update_strategy(strategy_id, {
                "status": "failed",
                "modal_job_id": None,   # always clear so UI doesn't show stuck job
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

        # Use the best timeframe discovered by the multi-TF quick test.
        # Falls back to the implementer's suggestion, then 1h.
        best_timeframe = strategy.get("best_timeframe")
        timeframes = strategy.get("timeframes") or ["1h"]
        if isinstance(timeframes, str):
            timeframes = [timeframes]
        primary_tf = best_timeframe or (timeframes[0] if timeframes else "1h")

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

        # 3. Execute the strategy class from code string (no data needed yet)
        import os as _os
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

        # 4. Build param_space for optimizer
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

        # 5. Determine which timeframes are worth optimizing.
        # Use quick-test results: any TF that produced at least 1 trade is viable.
        # Skip TFs with 0 trades — there is nothing to optimize.
        quick_test_all = strategy.get("quick_test_all_timeframes") or {}
        viable_tfs = [
            tf for tf, m in quick_test_all.items()
            if isinstance(m, dict) and m.get("trades", 0) > 0
        ]
        # Sort by trade count descending so higher-signal TFs run first
        viable_tfs.sort(key=lambda tf: quick_test_all[tf].get("trades", 0), reverse=True)
        if not viable_tfs:
            viable_tfs = [primary_tf]  # fallback if quick-test data is missing

        add_pipeline_note(strategy_id,
            f"Viable timeframes for optimization: {viable_tfs} "
            f"(skipping TFs with 0 trades)."
        )

        # 6. Optimize on every viable timeframe; pick the winner by best train Sharpe.
        tf_opt_results: dict = {}   # tf → {best_params, sharpe, trades, train_df, oos_df}

        for opt_tf in viable_tfs:
            try:
                cache_file = f"{CACHE_DIR}/{symbol}_{opt_tf}.parquet"
                if _os.path.exists(cache_file):
                    opt_df = pd.read_parquet(cache_file)
                else:
                    opt_df = fetch_ohlcv(symbol, opt_tf, start="2015-01-01", end="2026-12-31")
                    _os.makedirs(CACHE_DIR, exist_ok=True)
                    opt_df.to_parquet(cache_file)
                    ohlcv_cache.commit()

                opt_train_df, opt_oos_df = split_train_oos(opt_df)
                opt_params, _ = optimize_strategy(
                    strategy_class, opt_train_df, opt_space, n_trials=50, n_jobs=4
                )
                opt_result = run_backtest(
                    strategy_class, opt_train_df, params=opt_params, enforce_gates=False
                )
                sharpe = opt_result.sharpe if opt_result else -999.0
                trades = opt_result.total_trades if opt_result else 0
                tf_opt_results[opt_tf] = {
                    "best_params": opt_params,
                    "sharpe":      sharpe,
                    "trades":      trades,
                    "train_df":    opt_train_df,
                    "oos_df":      opt_oos_df,
                }
                add_pipeline_note(
                    strategy_id,
                    f"  {opt_tf}: optimized Sharpe={sharpe:.3f}, trades={trades}."
                )
            except Exception as tf_err:
                add_pipeline_note(
                    strategy_id,
                    f"  {opt_tf}: optimization failed — {tf_err}"
                )

        if not tf_opt_results:
            raise ValueError("All timeframe optimizations failed")

        # Winner = best optimized train Sharpe
        best_opt_tf = max(tf_opt_results, key=lambda tf: tf_opt_results[tf]["sharpe"])
        best_r      = tf_opt_results[best_opt_tf]
        primary_tf  = best_opt_tf
        best_params = best_r["best_params"]
        train_df    = best_r["train_df"]
        oos_df      = best_r["oos_df"]

        add_pipeline_note(
            strategy_id,
            f"Winner: {primary_tf} (Sharpe={best_r['sharpe']:.3f}, "
            f"trades={best_r['trades']}). Running full pipeline on {primary_tf}."
        )
        db.update_strategy(strategy_id, {"best_timeframe": primary_tf})

        # Early-exit: if optimization found only 0-trade parameter combos, failing here
        # gives a specific "optimization_regression" error the code_fixer can act on —
        # much better than letting walk-forward produce a generic quality-rejection message.
        if best_r["trades"] == 0:
            quick_test_all = strategy.get("quick_test_all_timeframes") or {}
            qt_max_trades = max(
                (m.get("trades", 0) for m in quick_test_all.values() if isinstance(m, dict)),
                default=0,
            )
            reason = (
                f"optimization_regression: optimizer produced 0 trades with best params "
                f"{best_params} on {primary_tf}. "
                f"Quick test had {qt_max_trades} trades with default params. "
                f"Likely causes: param space too wide (hitting 0-trade regions), "
                f"NaN indicator values for some param values, or session filter "
                f"becoming too restrictive with non-default params."
            )
            db.update_strategy(strategy_id, {
                "status":       "failed",
                "error_log":    reason,
                "hyperparams":  best_params,
                "modal_job_id": None,
            })
            add_pipeline_note(strategy_id, f"Optimization regression — {reason}")
            return {"passed": False, "reason": reason}

        add_pipeline_note(strategy_id,
            f"Train period: {train_df.index[0].strftime('%Y-%m-%d')} → "
            f"{train_df.index[-1].strftime('%Y-%m-%d')} "
            f"({round((train_df.index[-1] - train_df.index[0]).days / 365.25, 1)}y, "
            f"{len(train_df):,} bars). "
            f"OOS: {oos_df.index[0].strftime('%Y-%m-%d')} → "
            f"{oos_df.index[-1].strftime('%Y-%m-%d')} "
            f"({len(oos_df):,} bars)."
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
                "modal_job_id": None,   # clear so UI doesn't show stuck job
                "error_log": f"{type(e).__name__}: {e}\n{tb[:500]}",
                "retry_count": (db2.get_strategy(strategy_id) or {}).get("retry_count", 0) + 1,
            })
        except Exception:
            pass
        raise
