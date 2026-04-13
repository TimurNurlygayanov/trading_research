"""
Queue worker: processes the strategy pipeline in priority order.

Pipeline stages (in order):
  idea → filtered → implementing → implemented
       → quick_testing  (Modal: run with defaults, no optimization, ~2 min)
       → quick_tested   (results visible in UI)
       → backtesting    (Modal: Optuna + walk-forward + Monte Carlo, ~15-25 min)
       → validating     (Modal: LLM validator + summariser + learner)
       → live | failed

Research flow (parallel path when implementer needs data first):
  filtered → implementing → awaiting_research
       (research_tasks run on Modal in parallel)
       → implemented (once all research tasks done, implementer re-runs with results)
       → quick_testing → ...

All LLM-calling agents are guarded by check_budget().
All errors are caught, logged, and written to strategy.error_log.
"""
from __future__ import annotations

import structlog

from db import supabase_client as db
from orchestrator.budget_guard import check_budget, BudgetExceeded

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Modal dispatch helpers
# ---------------------------------------------------------------------------

def _dispatch_quick_backtest_job(strategy_id: str) -> None:
    """Dispatch strategy to Modal for the quick (default-params) backtest."""
    try:
        import modal
        fn = modal.Function.from_name("trading-research-backtest", "run_quick_backtest")
        call = fn.spawn(strategy_id)
        db.update_strategy(strategy_id, {
            "status": "quick_testing",
            "modal_job_id": call.object_id,
        })
        log.info("modal_quick_backtest_dispatched",
                 strategy_id=strategy_id, job_id=call.object_id)
    except ImportError:
        log.warning("modal_not_installed", strategy_id=strategy_id)
    except Exception as exc:
        log.error("modal_quick_dispatch_failed", strategy_id=strategy_id, error=str(exc))
        db.update_strategy(strategy_id, {
            "status": "failed",
            "error_log": f"Modal quick backtest dispatch error: {type(exc).__name__}: {exc}",
        })


def _dispatch_backtest_job(strategy_id: str) -> None:
    """Dispatch strategy to Modal for the full backtest pipeline (Optuna + WF + MC)."""
    try:
        import modal
        fn = modal.Function.from_name("trading-research-backtest", "run_backtest_pipeline")
        call = fn.spawn(strategy_id)
        db.update_strategy(strategy_id, {
            "status": "backtesting",
            "modal_job_id": call.object_id,
        })
        log.info("modal_backtest_dispatched",
                 strategy_id=strategy_id, job_id=call.object_id)
    except ImportError:
        log.warning("modal_not_installed", strategy_id=strategy_id)
    except Exception as exc:
        log.error("modal_dispatch_failed", strategy_id=strategy_id, error=str(exc))
        db.update_strategy(strategy_id, {
            "status": "failed",
            "error_log": f"Modal dispatch error: {type(exc).__name__}: {exc}",
        })


def _dispatch_validator_job(strategy_id: str) -> None:
    """Dispatch strategy to Modal for validation / summarise / learn."""
    try:
        import modal
        fn = modal.Function.from_name("trading-research-validator", "run_validator_pipeline")
        call = fn.spawn(strategy_id)
        db.update_strategy(strategy_id, {"modal_job_id": call.object_id})
        log.info("modal_validator_dispatched",
                 strategy_id=strategy_id, job_id=call.object_id)
    except ImportError:
        log.warning("modal_not_installed", strategy_id=strategy_id)
    except Exception as exc:
        log.error("modal_validator_dispatch_failed", strategy_id=strategy_id, error=str(exc))
        db.update_strategy(strategy_id, {
            "status": "failed",
            "error_log": f"Modal validator dispatch error: {type(exc).__name__}: {exc}",
        })


def _dispatch_pending_research_tasks() -> int:
    """Find research tasks stuck in 'pending' and dispatch them to Modal."""
    pending = db.get_research_tasks(status="pending", limit=10)
    dispatched = 0
    for task in pending:
        task_id = task["id"]
        try:
            import modal
            fn = modal.Function.from_name("trading-research-research", "run_research_task")
            call = fn.spawn(task_id)
            db.update_research_task(task_id, {
                "status": "running",
                "modal_job_id": call.object_id,
            })
            dispatched += 1
            log.info("modal_research_dispatched", task_id=task_id, job_id=call.object_id)
        except ImportError:
            log.warning("modal_not_installed_research", task_id=task_id)
            break
        except Exception as exc:
            log.error("modal_research_dispatch_failed", task_id=task_id, error=str(exc))
    return dispatched


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
        strategy_id: str | None = None
        try:
            strategy_record = db.insert_strategy({
                "source": "user",
                "status": "idea",
                "name": idea.get("title", "Untitled Idea"),
                "hypothesis": idea.get("description", ""),
                "entry_logic": idea.get("description", ""),
            })
            strategy_id = strategy_record["id"]

            db.mark_idea_picked_up(idea_id, strategy_id)
            log.info("idea_picked_up", idea_id=idea_id,
                     strategy_id=strategy_id, title=idea.get("title"))

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
            if strategy_id:
                try:
                    db.update_strategy(strategy_id, {
                        "status": "failed",
                        "error_log": f"idea_processing_failed: {type(exc).__name__}: {exc}",
                    })
                except Exception:
                    pass

    return processed


def _recover_stuck_idea_strategies() -> int:
    """
    Safety net: pick up strategies stuck in 'idea' status (pre_filter never ran).
    """
    sb = db.get_client()
    result = (
        sb.table("strategies")
        .select("id, name")
        .eq("status", "idea")
        .order("created_at")
        .limit(5)
        .execute()
    )
    strategies = result.data or []
    if not strategies:
        return 0

    recovered = 0
    for strategy in strategies:
        strategy_id = strategy["id"]
        try:
            try:
                check_budget("pre_filter")
            except BudgetExceeded as budget_err:
                log.warning("budget_exceeded_recover_idea",
                            strategy_id=strategy_id, error=str(budget_err))
                db.update_strategy(strategy_id, {
                    "status": "failed",
                    "error_log": str(budget_err),
                })
                recovered += 1
                continue

            from agents.pre_filter import run_pre_filter
            log.info("recovering_stuck_idea", strategy_id=strategy_id,
                     name=strategy.get("name"))
            run_pre_filter(strategy_id)
            log.info("recovered_idea_pre_filter_complete", strategy_id=strategy_id)
            recovered += 1
        except Exception as exc:
            log.error("recover_idea_failed", strategy_id=strategy_id, error=str(exc))
            db.update_strategy(strategy_id, {
                "status": "failed",
                "error_log": f"recover_idea pre_filter error: {type(exc).__name__}: {exc}",
            })
            recovered += 1

    return recovered


def _process_filtered_strategies() -> int:
    """
    Run the implementer on strategies that passed pre-filter.
    Two possible outcomes:
      - Code produced → strategy becomes 'implemented' → dispatch quick backtest
      - Research requested → strategy becomes 'awaiting_research'
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
                log.warning("budget_exceeded_implementer",
                            strategy_id=strategy_id, error=str(budget_err))
                continue

            from agents.implementer import run_implementer
            try:
                result = run_implementer(strategy_id, allow_research_requests=True)
                if result.get("needs_research"):
                    # Strategy is now in 'awaiting_research' — queue worker will
                    # check back once all research tasks complete
                    log.info("implementer_requested_research",
                             strategy_id=strategy_id, tasks=result.get("task_ids"))
                else:
                    # Code produced — dispatch to quick backtest immediately
                    log.info("implementer_complete",
                             strategy_id=strategy_id,
                             has_code=bool(result.get("code")))
                    _dispatch_quick_backtest_job(strategy_id)
            except Exception as exc:
                log.error("implementer_failed",
                          strategy_id=strategy_id, error=str(exc))
                db.update_strategy(strategy_id, {
                    "status": "failed",
                    "error_log": f"implementer error: {type(exc).__name__}: {exc}",
                })

            processed += 1

        except Exception as exc:
            log.error("filtered_processing_failed",
                      strategy_id=strategy_id, error=str(exc))

    return processed


def _process_awaiting_research_strategies() -> int:
    """
    Check strategies that are waiting for research tasks to complete.
    When all research tasks for a strategy are done, re-run the implementer
    with research results injected as context, then dispatch to quick backtest.
    Returns number of strategies unblocked.
    """
    strategies = db.get_strategies_awaiting_research(limit=5)
    if not strategies:
        return 0

    unblocked = 0
    for strategy in strategies:
        strategy_id = strategy["id"]
        task_ids: list[str] = strategy.get("pending_research_ids") or []

        if not task_ids:
            # No tasks listed — shouldn't happen, but recover gracefully
            log.warning("awaiting_research_no_tasks", strategy_id=strategy_id)
            db.update_strategy(strategy_id, {"status": "filtered"})
            continue

        # Load all research tasks and check completion
        tasks = [db.get_research_task(tid) for tid in task_ids if tid]
        tasks = [t for t in tasks if t]  # filter None

        if not tasks:
            # Tasks disappeared — reset to filtered so implementer can retry
            db.update_strategy(strategy_id, {
                "status": "filtered",
                "pending_research_ids": [],
            })
            continue

        done_statuses = {"done", "failed"}
        all_done = all(t.get("status") in done_statuses for t in tasks)
        if not all_done:
            continue  # still running — check next cycle

        # All tasks finished — collect results from completed ones
        research_results = []
        for task in tasks:
            if task.get("status") == "done":
                research_results.append({
                    "title": task.get("title", ""),
                    "question": task.get("question", ""),
                    "result_summary": task.get("result_summary", ""),
                    "key_findings": task.get("key_findings") or [],
                })

        log.info("research_tasks_complete", strategy_id=strategy_id,
                 tasks_done=len([t for t in tasks if t.get("status") == "done"]),
                 tasks_failed=len([t for t in tasks if t.get("status") == "failed"]))

        try:
            check_budget("implementer")
        except BudgetExceeded as budget_err:
            log.warning("budget_exceeded_after_research",
                        strategy_id=strategy_id, error=str(budget_err))
            continue

        # Re-run implementer with research results — force it to produce code
        db.update_strategy(strategy_id, {
            "status": "filtered",
            "pending_research_ids": [],
        })

        from agents.implementer import run_implementer
        try:
            result = run_implementer(
                strategy_id,
                research_results=research_results,
                allow_research_requests=False,  # must produce code this time
            )
            log.info("implementer_after_research_complete",
                     strategy_id=strategy_id, has_code=bool(result.get("code")))
            _dispatch_quick_backtest_job(strategy_id)
            unblocked += 1
        except Exception as exc:
            log.error("implementer_after_research_failed",
                      strategy_id=strategy_id, error=str(exc))
            db.update_strategy(strategy_id, {
                "status": "failed",
                "error_log": f"implementer (post-research) error: {type(exc).__name__}: {exc}",
            })

    return unblocked


def _process_implemented_strategies() -> int:
    """
    Dispatch 'implemented' strategies to the quick backtest job.
    Also dispatch 'validating' strategies (backtest done) to the validator.
    Returns total dispatched.
    """
    dispatched = 0

    # implemented → quick backtest (NEW: was full backtest before)
    implemented = db.get_strategies_by_status("implemented", limit=5)
    for strategy in implemented:
        strategy_id = strategy.get("id")
        if strategy.get("modal_job_id"):
            continue  # already dispatched
        try:
            _dispatch_quick_backtest_job(strategy_id)
            dispatched += 1
        except Exception as exc:
            log.error("quick_backtest_dispatch_failed",
                      strategy_id=strategy_id, error=str(exc))

    # validating → Modal validator
    validating = db.get_strategies_by_status("validating", limit=10)
    for strategy in validating:
        strategy_id = strategy.get("id")
        if strategy.get("modal_job_id"):
            continue  # already dispatched
        try:
            _dispatch_validator_job(strategy_id)
            dispatched += 1
        except Exception as exc:
            log.error("validator_dispatch_failed",
                      strategy_id=strategy_id, error=str(exc))

    return dispatched


def _process_quick_tested_strategies() -> int:
    """
    Dispatch 'quick_tested' strategies to the full backtest pipeline
    (Optuna optimization + walk-forward + Monte Carlo + OOS).
    Returns number of strategies dispatched.
    """
    quick_tested = db.get_strategies_by_status("quick_tested", limit=5)
    if not quick_tested:
        return 0

    dispatched = 0
    for strategy in quick_tested:
        strategy_id = strategy.get("id")
        if strategy.get("modal_job_id"):
            continue  # already dispatched
        try:
            _dispatch_backtest_job(strategy_id)
            dispatched += 1
            log.info("full_backtest_dispatched_after_quick",
                     strategy_id=strategy_id,
                     quick_sharpe=strategy.get("quick_test_sharpe"),
                     quick_trades=strategy.get("quick_test_trades"))
        except Exception as exc:
            log.error("full_backtest_dispatch_failed",
                      strategy_id=strategy_id, error=str(exc))

    return dispatched


def _process_failed_strategies() -> int:
    """
    Inspect recently-failed strategies and attempt automatic recovery.

    Error classes and responses:
      quality_rejection  → leave as failed (legitimate pipeline gate)
      infrastructure     → auto-retry: reset to 'implemented', redispatch quick backtest
      code_bug           → run code_fixer LLM, update backtest_code, redispatch
      unknown            → retry once without code change if retry_count < 2

    Hard limits:
      - retry_count >= 3  → give up
      - No backtest_code  → can't fix code, give up
    Returns number of strategies acted on.
    """
    from agents.code_fixer import classify_error, fix_strategy_code
    from agents.utils import add_pipeline_note

    MAX_AUTO_RETRIES = 3

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
    failed = [s for s in (result.data or []) if s.get("backtest_code")]

    acted = 0
    for strategy in failed:
        strategy_id = strategy["id"]
        error_log   = strategy.get("error_log") or ""
        retry_count = strategy.get("retry_count") or 0
        fix_count   = strategy.get("auto_fix_count") or 0
        code        = strategy.get("backtest_code") or ""

        if retry_count >= MAX_AUTO_RETRIES:
            continue

        error_class = classify_error(error_log)

        if error_class == "quality_rejection":
            continue  # legitimate failure, leave it

        try:
            if error_class == "infrastructure":
                # Transient infra error — reset to 'implemented' so it goes
                # through quick_test again before full backtest
                db.update_strategy(strategy_id, {
                    "status":      "implemented",
                    "error_log":   None,
                    "modal_job_id": None,
                    "retry_count": retry_count + 1,
                })
                add_pipeline_note(
                    strategy_id,
                    f"Auto-retry #{retry_count + 1}: infrastructure error. "
                    f"Redispatching to quick backtest. Error: {error_log[:120]}"
                )
                _dispatch_quick_backtest_job(strategy_id)
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
                    "modal_job_id":   None,
                    "retry_count":    retry_count + 1,
                    "auto_fix_count": fix_count + 1,
                })
                add_pipeline_note(
                    strategy_id,
                    f"Auto code-fix #{fix_count + 1}: detected Python error. "
                    f"LLM fix applied — redispatching to quick backtest.\n"
                    f"Original error: {error_log[:200]}"
                )
                _dispatch_quick_backtest_job(strategy_id)
                log.info("auto_fix_dispatched", strategy_id=strategy_id,
                         fix_count=fix_count + 1)
                acted += 1

            else:  # unknown
                if retry_count < 2:
                    db.update_strategy(strategy_id, {
                        "status":      "implemented",
                        "error_log":   None,
                        "modal_job_id": None,
                        "retry_count": retry_count + 1,
                    })
                    add_pipeline_note(
                        strategy_id,
                        f"Auto-retry #{retry_count + 1}: unknown error, retrying. "
                        f"Error was: {error_log[:150]}"
                    )
                    _dispatch_quick_backtest_job(strategy_id)
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
      1. User ideas → pre-filter
      2. Filtered strategies → implementer (may request research)
      3. Awaiting-research strategies → check if tasks done → re-run implementer
      4. Research tasks stuck in 'pending' → dispatch to Modal
      5. Implemented strategies → quick backtest
      6. Quick-tested strategies → full optimization backtest
      7. Validating strategies → validator
      8. Failed strategies → auto-fix code bugs / retry infra errors
    """
    log.info("queue_worker_start")

    ideas_processed       = _process_user_ideas()
    recovered_ideas       = _recover_stuck_idea_strategies()
    filtered_processed    = _process_filtered_strategies()
    research_unblocked    = _process_awaiting_research_strategies()
    research_dispatched   = _dispatch_pending_research_tasks()
    # 'implemented' → quick backtest; 'validating' → validator
    modal_dispatched      = _process_implemented_strategies()
    # 'quick_tested' → full backtest
    full_backtest_queued  = _process_quick_tested_strategies()
    auto_fixed            = _process_failed_strategies()

    log.info(
        "queue_worker_done",
        ideas_processed=ideas_processed,
        recovered_ideas=recovered_ideas,
        filtered_processed=filtered_processed,
        research_unblocked=research_unblocked,
        research_dispatched=research_dispatched,
        modal_dispatched=modal_dispatched,
        full_backtest_queued=full_backtest_queued,
        auto_fixed=auto_fixed,
    )
