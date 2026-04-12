"""
Queue worker: processes the strategy pipeline in priority order.

Priority order:
  1. User ideas (highest) -> pre_filter
  2. Filtered strategies -> implementer -> Modal backtest
  3. Implemented strategies -> check Modal job status
  4. Done strategies -> summariser -> learner
  5. Failed strategies -> classify error -> auto-fix or retry

All LLM-calling agents are guarded by check_budget().
All errors are caught, logged, and written to strategy.error_log.
"""
from __future__ import annotations

import structlog

from db import supabase_client as db
from orchestrator.budget_guard import check_budget, BudgetExceeded

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Modal dispatch
# ---------------------------------------------------------------------------

def _dispatch_backtest_job(strategy_id: str) -> None:
    """
    Dispatch a strategy to Modal for the full backtest pipeline.
    Stores the Modal call ID so the dashboard can check job status.
    """
    try:
        import modal
        fn = modal.Function.from_name("trading-research-backtest", "run_backtest_pipeline")
        call = fn.spawn(strategy_id)
        job_id = getattr(call, "object_id", None)
        db.update_strategy(strategy_id, {"modal_job_id": job_id})
        log.info("modal_backtest_dispatched", strategy_id=strategy_id, job_id=job_id)
    except ImportError:
        log.warning("modal_not_installed", strategy_id=strategy_id)
    except Exception as exc:
        log.error("modal_dispatch_failed", strategy_id=strategy_id, error=str(exc))
        db.update_strategy(strategy_id, {
            "status": "failed",
            "error_log": f"Modal dispatch error: {type(exc).__name__}: {exc}",
        })


def _dispatch_validator_job(strategy_id: str) -> None:
    """
    Dispatch a strategy to Modal for validation / summarise / learn.
    Stores the Modal call ID so the dashboard can check job status.
    """
    try:
        import modal
        fn = modal.Function.from_name("trading-research-validator", "run_validator_pipeline")
        call = fn.spawn(strategy_id)
        job_id = getattr(call, "object_id", None)
        db.update_strategy(strategy_id, {"modal_job_id": job_id})
        log.info("modal_validator_dispatched", strategy_id=strategy_id, job_id=job_id)
    except ImportError:
        log.warning("modal_not_installed", strategy_id=strategy_id)
    except Exception as exc:
        log.error("modal_validator_dispatch_failed", strategy_id=strategy_id, error=str(exc))
        db.update_strategy(strategy_id, {
            "status": "failed",
            "error_log": f"Modal validator dispatch error: {type(exc).__name__}: {exc}",
        })


# ---------------------------------------------------------------------------
# Step processors
# ---------------------------------------------------------------------------

def _process_user_ideas() -> int:
    """
    Pick up pending user ideas, create strategy records, run pre-filter.
    Returns number of ideas processed.
    """
    ideas = db.get_pending_user_ideas()
    if not ideas:
        return 0

    processed = 0
    for idea in ideas:
        idea_id = idea.get("id")
        try:
            # Create strategy record — insert_strategy returns a dict, extract the id
            strategy_record = db.insert_strategy({
                "source": "user",
                "status": "idea",
                "name": idea.get("title", "Untitled Idea"),
                "hypothesis": idea.get("description", ""),
                # entry_logic preserves the original user text forever —
                # hypothesis may be refined by pre-filter / implementer
                "entry_logic": idea.get("description", ""),
            })
            strategy_id = strategy_record["id"]

            # Mark idea as picked_up immediately so it is never re-processed,
            # even if pre_filter fails below
            db.mark_idea_picked_up(idea_id, strategy_id)

            log.info("idea_picked_up", idea_id=idea_id,
                     strategy_id=strategy_id, title=idea.get("title"))

            # Pre-filter: guard budget first
            try:
                check_budget("pre_filter")
            except BudgetExceeded as budget_err:
                log.warning("budget_exceeded_pre_filter",
                            strategy_id=strategy_id, error=str(budget_err))
                db.update_strategy(strategy_id, {
                    "status": "failed",
                    "error_log": str(budget_err),
                })
                processed += 1
                continue

            from agents.pre_filter import run_pre_filter
            try:
                run_pre_filter(strategy_id)
                log.info("pre_filter_complete", strategy_id=strategy_id)
                db.mark_idea_done(idea_id)
            except Exception as exc:
                log.error("pre_filter_failed",
                          strategy_id=strategy_id, error=str(exc))
                db.update_strategy(strategy_id, {
                    "status": "failed",
                    "error_log": f"pre_filter error: {type(exc).__name__}: {exc}",
                })
                db.mark_idea_failed(idea_id, str(exc))

            processed += 1

        except Exception as exc:
            log.error("idea_processing_failed", idea_id=idea_id, error=str(exc))

    return processed


def _process_filtered_strategies() -> int:
    """
    Run the implementer on strategies that passed pre-filter.
    Returns number of strategies processed.
    """
    strategies = db.get_strategies_by_status("filtered", limit=3)
    if not strategies:
        return 0

    processed = 0
    for strategy in strategies:
        strategy_id = strategy.get("id")
        try:
            try:
                check_budget("implementer")
            except BudgetExceeded as budget_err:
                log.warning(
                    "budget_exceeded_implementer",
                    strategy_id=strategy_id,
                    error=str(budget_err),
                )
                # Leave strategy in "filtered" so it retries after budget resets
                continue

            from agents.implementer import run_implementer
            try:
                result = run_implementer(strategy_id)
                log.info(
                    "implementer_complete",
                    strategy_id=strategy_id,
                    has_code=bool(result.get("code")),
                )
                # Dispatch to Modal for heavy compute
                _dispatch_backtest_job(strategy_id)
            except Exception as exc:
                log.error(
                    "implementer_failed",
                    strategy_id=strategy_id,
                    error=str(exc),
                )
                db.update_strategy(strategy_id, {
                    "status": "failed",
                    "error_log": f"implementer error: {type(exc).__name__}: {exc}",
                })

            processed += 1

        except Exception as exc:
            log.error(
                "filtered_processing_failed",
                strategy_id=strategy_id,
                error=str(exc),
            )

    return processed


def _process_implemented_strategies() -> int:
    """
    Two sub-steps:
      1. Dispatch "implemented" strategies to Modal backtest job.
         (Covers both fresh implementations and retries reset to "implemented".)
      2. Dispatch "validating" strategies (backtest done) to Modal validator job.
    Returns total number of strategies dispatched.
    """
    dispatched = 0

    # Step 1: implemented → Modal backtest
    implemented = db.get_strategies_by_status("implemented", limit=5)
    for strategy in implemented:
        strategy_id = strategy.get("id")
        try:
            _dispatch_backtest_job(strategy_id)
            dispatched += 1
        except Exception as exc:
            log.error("backtest_dispatch_failed", strategy_id=strategy_id, error=str(exc))

    # Step 2: validating → Modal validator
    validating = db.get_strategies_by_status("validating", limit=10)
    for strategy in validating:
        strategy_id = strategy.get("id")
        try:
            _dispatch_validator_job(strategy_id)
            dispatched += 1
        except Exception as exc:
            log.error("validator_dispatch_failed", strategy_id=strategy_id, error=str(exc))

    return dispatched


    # NOTE: summariser + learner run inside the Modal validator_job after backtesting.
    # No separate "done" processing needed here — strategies reach "done" already summarised.


def _process_failed_strategies() -> int:
    """
    Inspect recently-failed strategies and attempt automatic recovery.

    Error classes and responses:
      quality_rejection  → leave as failed (legitimate pipeline gate)
      infrastructure     → auto-retry: reset to "implemented", redispatch to Modal
      code_bug           → run code_fixer LLM, update backtest_code, redispatch
      unknown            → retry once without code change if retry_count < 2

    Hard limits:
      - retry_count >= 3  → give up (avoid infinite loops)
      - No backtest_code  → can't fix code, give up
    Returns number of strategies acted on.
    """
    from agents.code_fixer import classify_error, fix_strategy_code
    from agents.utils import add_pipeline_note

    MAX_AUTO_RETRIES = 3

    # Only look at strategies that recently failed and still have code
    sb = db.get_client()
    result = (
        sb.table("strategies")
        .select("id, name, status, error_log, backtest_code, hypothesis, "
                "entry_logic, retry_count, auto_fix_count")
        .eq("status", "failed")
        .order("updated_at", desc=True)
        .limit(20)
        .execute()
    )
    # Filter in Python: only those with backtest_code set
    failed = [s for s in (result.data or []) if s.get("backtest_code")]

    acted = 0
    for strategy in failed:
        strategy_id = strategy["id"]
        error_log   = strategy.get("error_log") or ""
        retry_count = strategy.get("retry_count") or 0
        fix_count   = strategy.get("auto_fix_count") or 0
        code        = strategy.get("backtest_code") or ""

        if retry_count >= MAX_AUTO_RETRIES:
            continue  # already retried enough

        error_class = classify_error(error_log)

        if error_class == "quality_rejection":
            continue  # legitimate failure, leave it

        try:
            if error_class == "infrastructure":
                # Transient infra error — just reset and redispatch, no code change
                db.update_strategy(strategy_id, {
                    "status":      "implemented",
                    "error_log":   None,
                    "retry_count": retry_count + 1,
                })
                add_pipeline_note(
                    strategy_id,
                    f"Auto-retry #{retry_count + 1}: infrastructure error detected "
                    f"({error_log[:120]}). Redispatching to Modal."
                )
                _dispatch_backtest_job(strategy_id)
                log.info("auto_retry_infra", strategy_id=strategy_id,
                         retry_count=retry_count + 1)
                acted += 1

            elif error_class == "code_bug":
                try:
                    check_budget("code_fixer")
                except BudgetExceeded as be:
                    log.warning("budget_exceeded_code_fixer",
                                strategy_id=strategy_id, error=str(be))
                    continue

                description = strategy.get("hypothesis") or strategy.get("entry_logic") or ""
                fixed_code = fix_strategy_code(
                    code=code,
                    error_log=error_log,
                    strategy_description=description,
                    strategy_id=strategy_id,
                )
                if not fixed_code:
                    log.warning("code_fixer_no_fix", strategy_id=strategy_id)
                    db.update_strategy(strategy_id, {
                        "error_log": f"[auto-fix attempted but LLM produced no valid code] {error_log}",
                        "retry_count": retry_count + 1,
                    })
                    continue

                db.update_strategy(strategy_id, {
                    "status":         "implemented",
                    "backtest_code":  fixed_code,
                    "error_log":      None,
                    "retry_count":    retry_count + 1,
                    "auto_fix_count": fix_count + 1,
                })
                add_pipeline_note(
                    strategy_id,
                    f"Auto code-fix #{fix_count + 1}: detected Python error in strategy code. "
                    f"LLM fix applied — redispatching to Modal backtest.\n"
                    f"Original error: {error_log[:200]}"
                )
                _dispatch_backtest_job(strategy_id)
                log.info("auto_fix_dispatched", strategy_id=strategy_id,
                         fix_count=fix_count + 1)
                acted += 1

            else:  # unknown
                if retry_count < 2:
                    db.update_strategy(strategy_id, {
                        "status":      "implemented",
                        "error_log":   None,
                        "retry_count": retry_count + 1,
                    })
                    add_pipeline_note(
                        strategy_id,
                        f"Auto-retry #{retry_count + 1}: unknown error, retrying once. "
                        f"Error was: {error_log[:150]}"
                    )
                    _dispatch_backtest_job(strategy_id)
                    log.info("auto_retry_unknown", strategy_id=strategy_id)
                    acted += 1

        except Exception as exc:
            log.error("auto_fix_failed", strategy_id=strategy_id, error=str(exc))

    return acted


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_queue() -> None:
    """
    Main queue processing loop. Called every 10 minutes by the scheduler.

    Processing order (highest priority first):
      1. User ideas
      2. Filtered strategies (ready for implementation)
      3. Implemented/validating strategies (Modal job status check)
      4. Failed strategies (auto-fix code bugs, retry infra errors)
    """
    log.info("queue_worker_start")

    ideas_processed       = _process_user_ideas()
    filtered_processed    = _process_filtered_strategies()
    validating_dispatched = _process_implemented_strategies()
    auto_fixed            = _process_failed_strategies()

    log.info(
        "queue_worker_done",
        ideas_processed=ideas_processed,
        filtered_processed=filtered_processed,
        validating_dispatched=validating_dispatched,
        auto_fixed=auto_fixed,
    )
