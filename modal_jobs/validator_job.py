"""
Modal job: runs the Validator, Summariser, and Learner agents for a strategy.

Pipeline:
1. Run validator agent (LLM + rule checks)
   - If validator provides corrected_code that passes leakage, save it and
     re-dispatch to the backtest job (up to MAX_VALIDATOR_CORRECTIONS=2 times).
2. If validation passes, run summariser
3. Run learner (updates knowledge base)
4. Mark strategy as "live" or "rejected"

Modal config: 2 CPUs, 4 GB RAM, 10 min timeout.
"""

MAX_VALIDATOR_CORRECTIONS = 2  # how many times the validator can fix & re-backtest
import modal

app = modal.App("trading-research-validator")

# Reuse the same image definition as the backtest job so Modal can cache it.
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
        "anthropic>=0.25.0",
        "structlog>=24.0.0",
    )
)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=600,  # 10 minutes
    secrets=[modal.Secret.from_name("trading-research-secrets")],
)
def run_validator_pipeline(strategy_id: str) -> dict:
    """
    Run Validator -> Summariser -> Learner for a single strategy.

    Imports are inside the function because this runs in Modal's remote
    environment where the project root is mounted.

    Returns a dict with keys:
        passed        (bool)   whether the strategy passed validation
        strategy_id   (str)    echo of the input
        reason        (str)    rejection reason if not passed, else None
    """
    import traceback

    try:
        from db import supabase_client as db
        from agents.utils import add_pipeline_note

        # ----------------------------------------------------------------
        # 1. Validator
        # ----------------------------------------------------------------
        from agents.validator import run_validator

        validator_result = run_validator(strategy_id)
        passed: bool = validator_result.get("passed", False)

        if not passed:
            # Check if the validator provided corrected code worth re-backtesting
            strategy = db.get_strategy(strategy_id)
            correction_count = (strategy or {}).get("validator_corrections", 0)
            final_code = validator_result.get("final_code", "")
            corrections_made = validator_result.get("corrections_made", [])

            if (
                final_code
                and corrections_made
                and correction_count < MAX_VALIDATOR_CORRECTIONS
            ):
                # Save corrected code, increment counter, re-dispatch to backtest
                db.update_strategy(strategy_id, {
                    "status": "implemented",
                    "backtest_code": final_code,
                    "validator_corrections": correction_count + 1,
                    "error_log": None,
                })
                corrections_text = "\n".join(f"- {c}" for c in corrections_made[:10])
                add_pipeline_note(
                    strategy_id,
                    f"Validator found issues — applied {len(corrections_made)} correction(s) "
                    f"(attempt {correction_count + 1}/{MAX_VALIDATOR_CORRECTIONS}):\n{corrections_text}\n\n"
                    f"Re-dispatching to backtest with fixed code."
                )
                # Re-dispatch to Modal backtest job
                import modal as _modal
                backtest_fn = _modal.Function.from_name(
                    "trading-research-backtest", "run_backtest_pipeline"
                )
                backtest_fn.spawn(strategy_id)
                return {
                    "passed": False,
                    "strategy_id": strategy_id,
                    "reason": "corrections_applied_rebacktesting",
                }

            # No correctable code or correction limit reached → reject
            issues = validator_result.get("leakage_issues", []) + validator_result.get("logic_bugs", [])
            reason = "; ".join(issues[:3]) if issues else validator_result.get("validator_notes", "Validation failed")
            db.update_strategy(strategy_id, {
                "status": "failed",
                "error_log": reason,
            })
            add_pipeline_note(
                strategy_id,
                f"Validator REJECTED after {correction_count} correction attempt(s).\n{reason}"
            )
            return {
                "passed": False,
                "strategy_id": strategy_id,
                "reason": reason,
            }

        # Validator passed — corrected code already persisted by run_validator()
        # ----------------------------------------------------------------
        # 2. Summariser
        # ----------------------------------------------------------------
        from agents.summariser import run_summariser

        try:
            run_summariser(strategy_id)
        except Exception as exc:
            tb = traceback.format_exc()
            db.update_strategy(strategy_id, {
                "status": "failed",
                "error_log": f"summariser error: {type(exc).__name__}: {exc}\n{tb[:400]}",
            })
            return {
                "passed": False,
                "strategy_id": strategy_id,
                "reason": f"summariser_failed: {exc}",
            }

        # ----------------------------------------------------------------
        # 3. Learner
        # ----------------------------------------------------------------
        from agents.learner import run_learner

        try:
            run_learner(strategy_id)
        except Exception as exc:
            # Learner failure is non-fatal: the strategy is still valid.
            # Log the error but do not block promotion to "live".
            tb = traceback.format_exc()
            db.update_strategy(strategy_id, {
                "error_log": f"learner error (non-fatal): {type(exc).__name__}: {exc}\n{tb[:400]}",
            })

        # ----------------------------------------------------------------
        # 4. Mark strategy as live
        # ----------------------------------------------------------------
        add_pipeline_note(strategy_id, "Validator passed. Summariser and Learner complete. Strategy is LIVE.")
        db.update_strategy(strategy_id, {"status": "live"})

        return {
            "passed": True,
            "strategy_id": strategy_id,
            "reason": None,
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
