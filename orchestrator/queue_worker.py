"""
Queue worker: processes the strategy pipeline in priority order.

Priority order:
  1. User ideas (highest) -> pre_filter
  2. Filtered strategies -> implementer -> Modal backtest
  3. Implemented strategies -> check Modal job status
  4. Done strategies -> summariser -> learner

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
    Fails gracefully if Modal is not installed or not configured.
    """
    try:
        import modal  # noqa: F401
        from modal_jobs.backtest_job import run_backtest_pipeline

        # Spawn asynchronously so the orchestrator is not blocked.
        run_backtest_pipeline.spawn(strategy_id)
        log.info("modal_backtest_dispatched", strategy_id=strategy_id)
    except ImportError:
        log.warning(
            "modal_not_installed",
            strategy_id=strategy_id,
            msg="Modal package not available; skipping remote dispatch.",
        )
    except Exception as exc:
        log.error(
            "modal_dispatch_failed",
            strategy_id=strategy_id,
            error=str(exc),
        )
        db.update_strategy(strategy_id, {
            "status": "failed",
            "error_log": f"Modal dispatch error: {type(exc).__name__}: {exc}",
        })


def _dispatch_validator_job(strategy_id: str) -> None:
    """
    Dispatch a strategy to Modal for validation / summarise / learn.
    Fails gracefully if Modal is not installed or not configured.
    """
    try:
        import modal  # noqa: F401
        from modal_jobs.validator_job import run_validator_pipeline

        run_validator_pipeline.spawn(strategy_id)
        log.info("modal_validator_dispatched", strategy_id=strategy_id)
    except ImportError:
        log.warning(
            "modal_not_installed",
            strategy_id=strategy_id,
            msg="Modal package not available; skipping validator dispatch.",
        )
    except Exception as exc:
        log.error(
            "modal_validator_dispatch_failed",
            strategy_id=strategy_id,
            error=str(exc),
        )
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
            # Create a new strategy record from the idea
            strategy_data = {
                "source": "user",
                "status": "idea",
                "name": idea.get("title", "Untitled Idea"),
                "hypothesis": idea.get("description", ""),
            }
            strategy_id = db.insert_strategy(strategy_data)
            db.mark_idea_picked_up(idea_id, strategy_id)

            log.info(
                "idea_picked_up",
                idea_id=idea_id,
                strategy_id=strategy_id,
                title=idea.get("title"),
            )

            # Pre-filter is an LLM call -> guard budget first
            try:
                check_budget("pre_filter")
            except BudgetExceeded as budget_err:
                log.warning(
                    "budget_exceeded_pre_filter",
                    strategy_id=strategy_id,
                    error=str(budget_err),
                )
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
            except Exception as exc:
                log.error(
                    "pre_filter_failed",
                    strategy_id=strategy_id,
                    error=str(exc),
                )
                db.update_strategy(strategy_id, {
                    "status": "failed",
                    "error_log": f"pre_filter error: {type(exc).__name__}: {exc}",
                })

            processed += 1

        except Exception as exc:
            log.error(
                "idea_processing_failed",
                idea_id=idea_id,
                error=str(exc),
            )

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
    Check strategies that have been submitted to Modal for backtesting.
    Strategies in "backtesting" status are being processed by Modal.
    Strategies that Modal has moved to "validating" are dispatched to the
    Modal validator job.
    Returns number of strategies acted on.
    """
    validating = db.get_strategies_by_status("validating", limit=10)
    if not validating:
        return 0

    dispatched = 0
    for strategy in validating:
        strategy_id = strategy.get("id")
        try:
            _dispatch_validator_job(strategy_id)
            dispatched += 1
        except Exception as exc:
            log.error(
                "validator_dispatch_failed",
                strategy_id=strategy_id,
                error=str(exc),
            )

    return dispatched


def _process_done_strategies() -> int:
    """
    Run summariser and learner on strategies that have completed validation.
    Returns number of strategies processed.
    """
    done_strategies = db.get_strategies_by_status("done", limit=5)
    if not done_strategies:
        return 0

    processed = 0
    for strategy in done_strategies:
        strategy_id = strategy.get("id")
        try:
            # Summariser is an LLM call
            try:
                check_budget("summariser")
            except BudgetExceeded as budget_err:
                log.warning(
                    "budget_exceeded_summariser",
                    strategy_id=strategy_id,
                    error=str(budget_err),
                )
                continue

            from agents.summariser import run_summariser
            try:
                run_summariser(strategy_id)
                log.info("summariser_complete", strategy_id=strategy_id)
            except Exception as exc:
                log.error(
                    "summariser_failed",
                    strategy_id=strategy_id,
                    error=str(exc),
                )
                db.update_strategy(strategy_id, {
                    "status": "failed",
                    "error_log": f"summariser error: {type(exc).__name__}: {exc}",
                })
                processed += 1
                continue

            # Learner is an LLM call
            try:
                check_budget("learner")
            except BudgetExceeded as budget_err:
                log.warning(
                    "budget_exceeded_learner",
                    strategy_id=strategy_id,
                    error=str(budget_err),
                )
                processed += 1
                continue

            from agents.learner import run_learner
            try:
                run_learner(strategy_id)
                log.info("learner_complete", strategy_id=strategy_id)
            except Exception as exc:
                log.error(
                    "learner_failed",
                    strategy_id=strategy_id,
                    error=str(exc),
                )
                db.update_strategy(strategy_id, {
                    "status": "failed",
                    "error_log": f"learner error: {type(exc).__name__}: {exc}",
                })

            processed += 1

        except Exception as exc:
            log.error(
                "done_processing_failed",
                strategy_id=strategy_id,
                error=str(exc),
            )

    return processed


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
      4. Done strategies (summarise + learn)
    """
    log.info("queue_worker_start")

    ideas_processed = _process_user_ideas()
    filtered_processed = _process_filtered_strategies()
    validating_processed = _process_implemented_strategies()
    done_processed = _process_done_strategies()

    log.info(
        "queue_worker_done",
        ideas_processed=ideas_processed,
        filtered_processed=filtered_processed,
        validating_dispatched=validating_processed,
        done_processed=done_processed,
    )
