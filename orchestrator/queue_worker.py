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


_RESEARCH_MAX_RETRIES = 3
_RESEARCH_STUCK_TIMEOUT_MINUTES = 13  # Modal timeout is 10 min; 13 min means it's definitely dead

# Strategy job timeouts — Modal kills containers silently, leaving stale status + modal_job_id.
# Set slightly above the Modal function timeout so we don't reset jobs that are still running.
_BACKTEST_STUCK_TIMEOUT_MINUTES = 52   # Modal timeout is 45 min (2700s)
_QUICK_TEST_STUCK_TIMEOUT_MINUTES = 20  # Modal timeout is 15 min (900s)


def _recover_stuck_research_tasks() -> int:
    """
    Reset research tasks stuck in 'running' back to 'failed' so the retry
    loop can pick them up.  Modal's function timeout is 10 minutes; if a
    task is still 'running' after _RESEARCH_STUCK_TIMEOUT_MINUTES the Modal
    container crashed without updating the DB.
    """
    from datetime import datetime, timezone, timedelta
    from dateutil import parser as _dateparser  # already available via pandas dep

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=_RESEARCH_STUCK_TIMEOUT_MINUTES)

    running = db.get_research_tasks(status="running", limit=20)
    log.info("research_watchdog_check running_count=%s", len(running))
    recovered = 0
    for task in running:
        updated_at_raw = task.get("updated_at") or task.get("created_at") or ""
        try:
            updated_at = _dateparser.parse(updated_at_raw)
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
        except Exception:
            # Can't parse timestamp — treat as stuck
            updated_at = datetime.min.replace(tzinfo=timezone.utc)

        if updated_at < cutoff:
            db.update_research_task(task["id"], {
                "status":       "failed",
                "modal_job_id": None,
                "error_log": (
                    f"Task timed out: still 'running' after "
                    f"{_RESEARCH_STUCK_TIMEOUT_MINUTES} min "
                    f"(Modal job likely crashed without updating DB)."
                ),
            })
            recovered += 1
            log.info("research_task_timeout_recovered task_id=%s updated_at=%s",
                     task["id"], updated_at_raw)
    return recovered


def _recover_stuck_strategy_jobs() -> int:
    """
    Reset strategies stuck in 'backtesting' or 'quick_testing' after Modal timeout.

    Modal kills the container without running the except block, so the DB is left
    with a stale status and modal_job_id. This watchdog detects that and resets
    to 'failed' so the auto-fix handler can retry or notify the user.
    """
    from datetime import datetime, timezone, timedelta
    from dateutil import parser as _dateparser

    recovered = 0
    sb = db.get_client()

    for status, timeout_min in [
        ("backtesting",  _BACKTEST_STUCK_TIMEOUT_MINUTES),
        ("quick_testing", _QUICK_TEST_STUCK_TIMEOUT_MINUTES),
    ]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_min)
        result = (
            sb.table("strategies")
            .select("id, name, modal_job_id, updated_at")
            .eq("status", status)
            .not_.is_("modal_job_id", "null")
            .lt("updated_at", cutoff.isoformat())
            .limit(10)
            .execute()
        )
        for s in (result.data or []):
            sid = s["id"]
            db.update_strategy(sid, {
                "status":       "failed",
                "modal_job_id": None,
                "error_log": (
                    f"Job timed out: stuck in '{status}' for >{timeout_min} min. "
                    "Modal container was killed without updating the DB. "
                    "Auto-retrying via failed strategy handler."
                ),
            })
            recovered += 1
            log.info("strategy_job_timeout_recovered",
                     strategy_id=sid, status=status, timeout_min=timeout_min)

    return recovered


def _retry_failed_research_tasks() -> int:
    """Reset failed research tasks (retry_count < max) back to pending."""
    failed = db.get_research_tasks(status="failed", limit=50)
    reset = 0
    for task in failed:
        retries = task.get("retry_count") or 0
        if retries >= _RESEARCH_MAX_RETRIES:
            continue
        db.update_research_task(task["id"], {
            "status":      "pending",
            "retry_count": retries + 1,
            "modal_job_id": None,
        })
        reset += 1
        log.info("research_task_reset_for_retry task_id=%s retry=%s", task["id"], retries + 1)
    return reset


def _auto_generate_research_tasks() -> int:
    """
    Keep the research queue populated using three modes in priority order:
      1. Static catalogue (free, fast)
      2. Param sweeps of best partial results (LLM, targeted)
      3. LLM invention: novel indicators + combos (LLM, creative)
    Only runs when the pending queue is below the threshold.
    """
    pending_count = len(db.get_research_tasks(status="pending", limit=30))
    if pending_count >= 20:
        return 0  # queue is healthy, nothing to do

    from agents.indicator_researcher import (
        generate_research_tasks,
        generate_param_sweep_tasks,
    )

    # Phase 1: static catalogue
    created = generate_research_tasks()
    if created > 0:
        log.info("auto_generate_research_static created=%s", created)
        return created

    # Phase 2: param sweeps of partial results
    try:
        created = generate_param_sweep_tasks(n_partials=5, variations_per=5)
        if created > 0:
            log.info("auto_generate_research_sweeps created=%s", created)
            return created
    except Exception as exc:
        log.warning("auto_generate_research_sweeps_failed error=%s", exc)

    # Phase 3: agenda-driven research (structured hypotheses, not random combos)
    try:
        from agents.research_agenda import process_all_agendas
        created = process_all_agendas(limit_per_agenda=5)  # drip-feed 5 tasks per agenda per cycle
        if created > 0:
            log.info("auto_generate_research_agenda created=%s", created)
            return created
    except Exception as exc:
        log.warning("auto_generate_research_agenda_failed error=%s", exc)

    return 0


_MAX_CONCURRENT_RESEARCH = 1  # one at a time — 8k output tokens/min is too tight for two


def _dispatch_pending_research_tasks() -> int:
    """Find research tasks in 'pending' and dispatch them to Modal.

    At most _MAX_CONCURRENT_RESEARCH jobs run simultaneously to avoid
    hitting the Anthropic output-token-per-minute rate limit.

    Routes by task type:
      indicator_research → run_indicator_research_task
      everything else    → run_research_task
    """
    running_count = len(db.get_research_tasks(status="running", limit=20))
    slots = _MAX_CONCURRENT_RESEARCH - running_count
    if slots <= 0:
        return 0

    pending = db.get_research_tasks(status="pending", limit=slots)
    dispatched = 0
    for task in pending:
        task_id   = task["id"]
        task_type = task.get("type", "market_analysis")
        try:
            import modal
            # indicator_research requires a research_spec; fall back to the
            # general researcher if one is missing (handles misrouted tasks).
            if task_type == "indicator_research" and task.get("research_spec"):
                fn_name = "run_indicator_research_task"
            else:
                fn_name = "run_research_task"
            fn   = modal.Function.from_name("trading-research-research", fn_name)
            call = fn.spawn(task_id)
            db.update_research_task(task_id, {
                "status": "running",
                "modal_job_id": call.object_id,
            })
            dispatched += 1
            log.info("modal_research_dispatched task_id=%s task_type=%s job_id=%s",
                     task_id, task_type, call.object_id)
        except ImportError:
            log.warning("modal_not_installed_research task_id=%s", task_id)
            break
        except Exception as exc:
            log.error("modal_research_dispatch_failed task_id=%s error=%s", task_id, exc)
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

    For strategies that are not yet part of a campaign (campaign_id=None,
    is_campaign_root=False), first run the Variation Planner to generate
    N structurally diverse child strategies before implementing the root.

    Two implementer outcomes:
      - Code produced → 'implemented' → dispatch quick backtest
      - Research requested → 'awaiting_research'
    Returns number of strategies processed.
    """
    import os as _os
    N_VARIATIONS = int(_os.environ.get("CAMPAIGN_N_VARIATIONS", "8"))

    strategies = db.get_strategies_by_status("filtered", limit=3)
    if not strategies:
        return 0

    processed = 0
    for strategy in strategies:
        strategy_id = strategy.get("id")
        try:
            is_campaign_child = bool(strategy.get("campaign_id"))
            is_campaign_root  = bool(strategy.get("is_campaign_root"))

            # ── Variation planning (only for fresh, standalone strategies) ────
            if not is_campaign_child and not is_campaign_root:
                # Mark as root FIRST so a failure doesn't cause infinite replanning
                db.update_strategy(strategy_id, {"is_campaign_root": True})
                try:
                    check_budget("variation_planner")
                except BudgetExceeded as be:
                    log.warning("budget_exceeded_variation_planner",
                                strategy_id=strategy_id, error=str(be))
                    # Fall through — implement as single strategy
                else:
                    try:
                        from agents.variation_planner import run_variation_planner
                        variations = run_variation_planner(strategy_id, N_VARIATIONS)
                        seed_hyp = strategy.get("hypothesis", "")
                        for v in variations:
                            db.insert_strategy({
                                "campaign_id":      strategy_id,
                                "name":             v.get("name", "Variation"),
                                "hypothesis":       v.get("description", ""),
                                "entry_logic":      seed_hyp,  # seed kept as context
                                "source":           "researcher",
                                "status":           "filtered",
                                "pre_filter_score": strategy.get("pre_filter_score"),
                                "pre_filter_notes": strategy.get("pre_filter_notes"),
                            })
                        log.info("campaign_created", strategy_id=strategy_id,
                                 n_children=len(variations))
                    except Exception as exc:
                        log.error("variation_planner_failed",
                                  strategy_id=strategy_id, error=str(exc))
                        # Fall through — implement as single strategy anyway

            # ── Run implementer ───────────────────────────────────────────────
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
                    log.info("implementer_requested_research",
                             strategy_id=strategy_id, tasks=result.get("task_ids"))
                else:
                    log.info("implementer_complete", strategy_id=strategy_id,
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
    For each 'quick_tested' strategy:

    Campaign strategies (campaign_id set OR is_campaign_root=True):
      1. Run strategy_analyzer once (analysis_done=False).
         If analyzer patches the code → re-run quick test.
      2. Auto-archive if quality gate fails (Sharpe < 0 AND trades < 30).
      3. Otherwise stay in 'quick_tested' — campaign review handles bulk approval.

    Standalone strategies (no campaign):
      1. Run strategy_analyzer (same as above).
      2. Set status='awaiting_review' for individual user review.

    Returns number of strategies acted on.
    """
    from agents.strategy_analyzer import run_strategy_analyzer
    from agents.utils import add_pipeline_note

    quick_tested = db.get_strategies_by_status("quick_tested", limit=10)
    if not quick_tested:
        return 0

    acted = 0
    for strategy in quick_tested:
        strategy_id   = strategy.get("id")
        if strategy.get("modal_job_id"):
            continue  # still running

        is_campaign   = bool(strategy.get("campaign_id")) or bool(strategy.get("is_campaign_root"))
        analysis_done = strategy.get("analysis_done", False)
        has_trades    = bool(strategy.get("quick_test_trades"))  # int count

        # Campaign strategies that are already analyzed just wait for campaign review
        if is_campaign and analysis_done:
            continue

        # ── Run analyzer (first time, any strategy type) ─────────────────────
        if not analysis_done and has_trades:
            try:
                check_budget("strategy_analyzer")
            except BudgetExceeded as be:
                log.warning("budget_exceeded_analyzer", strategy_id=strategy_id,
                            error=str(be))
                db.update_strategy(strategy_id, {"analysis_done": True})
            else:
                try:
                    result = run_strategy_analyzer(strategy_id)
                    db.update_strategy(strategy_id, {"analysis_done": True})
                    if result.get("code_patched"):
                        log.info("analyzer_code_patched_requeue_quick",
                                 strategy_id=strategy_id)
                        _dispatch_quick_backtest_job(strategy_id)
                        acted += 1
                        continue
                    log.info("analyzer_done", strategy_id=strategy_id,
                             improvements=result.get("improvements", []))
                except Exception as exc:
                    log.error("strategy_analyzer_failed", strategy_id=strategy_id,
                              error=str(exc))
                    db.update_strategy(strategy_id, {"analysis_done": True})

        # ── Campaign strategies: auto-archive weak variations ─────────────────
        if is_campaign:
            sharpe = strategy.get("quick_test_sharpe") or 0.0
            trades = strategy.get("quick_test_trades") or 0
            if sharpe < 0 and trades < 30:
                db.update_strategy(strategy_id, {
                    "status":    "archived",
                    "error_log": (
                        f"Campaign quality gate: Sharpe={sharpe:.4f}, trades={trades}. "
                        "Auto-archived — did not meet minimum thresholds."
                    ),
                })
                log.info("campaign_variation_archived", strategy_id=strategy_id,
                         sharpe=sharpe, trades=trades)
                acted += 1
            else:
                log.info("campaign_variation_ready", strategy_id=strategy_id,
                         sharpe=sharpe, trades=trades)
            continue  # campaign review panel handles approval

        # ── Standalone strategies: individual review ─────────────────────────
        try:
            db.update_strategy(strategy_id, {"status": "awaiting_review"})
            add_pipeline_note(
                strategy_id,
                "Quick test + analysis complete. "
                "Waiting for your review — approve to start full optimization, "
                "or request changes to revise the strategy code."
            )
            log.info("strategy_awaiting_review", strategy_id=strategy_id,
                     quick_sharpe=strategy.get("quick_test_sharpe"))
            acted += 1
        except Exception as exc:
            log.error("set_awaiting_review_failed",
                      strategy_id=strategy_id, error=str(exc))

    return acted


def _process_campaign_completion() -> int:
    """
    Detect campaigns where all variations have finished quick testing.
    Adds a summary note to the root so the user knows the campaign is ready to review.
    Safe to call every cycle — uses a marker in comments to avoid duplicate summaries.
    Returns number of campaigns newly summarized.
    """
    from agents.utils import add_pipeline_note

    sb = db.get_client()

    # Find campaign roots that are in a stable quick-test state
    REVIEWABLE = ("quick_tested", "awaiting_review", "backtesting",
                  "validating", "live", "failed", "archived")
    result = (
        sb.table("strategies")
        .select("id, name, status, quick_test_sharpe, quick_test_trades, "
                "best_timeframe, comments")
        .eq("is_campaign_root", True)
        .in_("status", list(REVIEWABLE))
        .execute()
    )
    roots = result.data or []

    summarized = 0
    for root in roots:
        root_id  = root["id"]
        children = db.get_campaign_children(root_id)
        if not children:
            continue

        # All children must be in a terminal quick-test state
        TERMINAL = {"quick_tested", "awaiting_review", "backtesting", "validating",
                    "live", "failed", "archived"}
        if not all(c.get("status") in TERMINAL for c in children):
            continue

        # Check if we already wrote a campaign summary
        import json as _json
        comments = root.get("comments") or []
        if isinstance(comments, str):
            try:
                comments = _json.loads(comments)
            except Exception:
                comments = []
        if any("[campaign_summary]" in (c.get("text", "")) for c in comments):
            continue  # already done

        # Build results table across root + children
        root_full      = db.get_strategy(root_id)
        all_variations = [root_full] + children if root_full else children
        with_results   = [s for s in all_variations if s.get("quick_test_sharpe") is not None]
        with_results.sort(key=lambda s: s.get("quick_test_sharpe") or 0, reverse=True)

        passed = [s for s in with_results
                  if (s.get("quick_test_sharpe") or 0) > 0
                  and (s.get("quick_test_trades") or 0) >= 30]
        archived = [c for c in children if c.get("status") == "archived"]

        lines = [
            f"[campaign_summary] Campaign complete — {len(all_variations)} variations tested.",
            f"Passed quality gate (Sharpe > 0, trades ≥ 30): {len(passed)} / {len(all_variations)}.",
            f"Auto-archived (weak): {len(archived)}.",
            "",
            "All results (sorted by Sharpe):",
        ]
        for i, s in enumerate(with_results[:10], 1):
            sh   = s.get("quick_test_sharpe") or 0
            tr   = s.get("quick_test_trades") or 0
            tf   = s.get("best_timeframe") or "?"
            mark = " ← seed" if s.get("id") == root_id else ""
            stat = s.get("status", "?")
            lines.append(
                f"  {i}. {s.get('name','?')}: "
                f"Sharpe={sh:+.4f}, trades={tr}, TF={tf}, status={stat}{mark}"
            )

        add_pipeline_note(root_id, "\n".join(lines))
        log.info("campaign_summarized", root_id=root_id,
                 total=len(all_variations), passed=len(passed))
        summarized += 1

    return summarized


_MAX_AUTO_RETRIES = 3
# Don't retry a strategy that failed less than this many minutes ago —
# gives Modal time to settle and avoids hammering a broken job immediately.
_MIN_RETRY_WAIT_MINUTES = 15


def _cleanup_old_failed_strategies() -> int:
    """
    Archive failed strategies that have been sitting in 'failed' for > 24 hours.

    Rules:
      - Failed > 24h AND retry_count >= MAX_AUTO_RETRIES  → archive (no more retries)
      - Failed > 24h AND no backtest_code                 → archive (unfixable)
      - Failed > 24h AND error is quality_rejection        → archive (legitimate gate)

    Strategies that still have retries remaining are left alone — they will be
    picked up by _process_failed_strategies on the next loop.
    """
    from datetime import datetime, timezone, timedelta
    from agents.code_fixer import classify_error
    from agents.utils import add_pipeline_note

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

    sb = db.get_client()
    result = (
        sb.table("strategies")
        .select("id, name, retry_count, backtest_code, error_log, updated_at")
        .eq("status", "failed")
        .lt("updated_at", cutoff)
        .limit(100)
        .execute()
    )

    archived = 0
    for s in (result.data or []):
        sid         = s["id"]
        retry_count = s.get("retry_count") or 0
        has_code    = bool(s.get("backtest_code"))
        error_log   = s.get("error_log") or ""
        error_class = classify_error(error_log)

        should_archive = (
            retry_count >= _MAX_AUTO_RETRIES
            or not has_code
            or error_class == "quality_rejection"
        )
        if not should_archive:
            continue

        db.update_strategy(sid, {"status": "archived", "modal_job_id": None})
        reason = (
            "quality rejection"      if error_class == "quality_rejection" else
            "no code to fix"         if not has_code else
            f"retry limit ({retry_count}/{_MAX_AUTO_RETRIES})"
        )
        add_pipeline_note(sid, f"Auto-archived after 24 h in failed state ({reason}).")
        log.info("strategy_auto_archived sid=%s reason=%s", sid, reason)
        archived += 1

    return archived


def _is_promising_quick_test(strategy: dict) -> bool:
    """Return True if quick test showed real trades + meaningful Sharpe on any timeframe.

    Used to distinguish 'optimization problem' from 'genuine bad strategy' when a
    quality-rejection gate fires after the full backtest pipeline.
    """
    tf_data = strategy.get("quick_test_all_timeframes") or {}
    for m in tf_data.values():
        if isinstance(m, dict) and m.get("trades", 0) >= 10 and m.get("sharpe", 0) >= 0.3:
            return True
    return False


def _process_failed_strategies() -> int:
    """
    Inspect recently-failed strategies and attempt automatic recovery.

    Error classes and responses:
      quality_rejection  → leave as failed (legitimate pipeline gate)
      infrastructure     → auto-retry: reset to 'implemented', redispatch quick backtest
      code_bug           → run code_fixer LLM, update backtest_code, redispatch
      unknown            → retry once without code change if retry_count < 2

    Hard limits:
      - retry_count >= _MAX_AUTO_RETRIES → give up
      - No backtest_code                 → can't fix code, give up
      - Failed < _MIN_RETRY_WAIT_MINUTES → too soon, skip this loop
    Returns number of strategies acted on.
    """
    from datetime import datetime, timezone, timedelta
    from agents.code_fixer import classify_error, fix_strategy_code
    from agents.utils import add_pipeline_note

    too_recent_cutoff = (
        datetime.now(timezone.utc) - timedelta(minutes=_MIN_RETRY_WAIT_MINUTES)
    ).isoformat()

    sb = db.get_client()
    result = (
        sb.table("strategies")
        .select("id, name, status, error_log, backtest_code, hypothesis, "
                "entry_logic, retry_count, auto_fix_count, quick_test_all_timeframes, "
                "hyperparams, updated_at")
        .eq("status", "failed")
        .lt("updated_at", too_recent_cutoff)   # only strategies that failed >15 min ago
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
            # If this strategy looked promising in quick test but failed optimization,
            # give the optimizer analyst one shot before giving up permanently.
            # Gate: only on the first failure (retry_count==0) to avoid infinite loops.
            if retry_count == 0 and _is_promising_quick_test(strategy):
                from agents.code_fixer import analyze_optimization_failure
                try:
                    check_budget("code_fixer")
                except BudgetExceeded as be:
                    log.warning("budget_exceeded_opt_analyst",
                                strategy_id=strategy_id, error=str(be))
                    continue
                try:
                    analysis = analyze_optimization_failure(strategy, error_log)
                    if analysis and analysis.get("action") == "simplify":
                        new_hp = analysis.get("hyperparams") or {}
                        if new_hp:
                            db.update_strategy(strategy_id, {
                                "status":         "implemented",
                                "hyperparams":    new_hp,
                                "error_log":      None,
                                "modal_job_id":   None,
                                "retry_count":    retry_count + 1,
                                "auto_fix_count": fix_count + 1,
                            })
                            add_pipeline_note(
                                strategy_id,
                                f"Optimization analyst: param space simplified to "
                                f"{list(new_hp.keys())} — retrying full backtest. "
                                f"Original failure: {error_log[:150]}"
                            )
                            _dispatch_backtest_job(strategy_id)
                            log.info("opt_analyst_simplify", strategy_id=strategy_id)
                            acted += 1
                            continue

                    # Analyst confirmed rejection (or returned nothing useful)
                    reason = (analysis or {}).get(
                        "reason", "Optimization results confirm no generalizable edge."
                    )
                    add_pipeline_note(
                        strategy_id,
                        f"Optimization analyst: confirmed genuine rejection — {reason}"
                    )
                    # Bump retry_count to ceiling so cleanup archives it promptly
                    db.update_strategy(strategy_id, {
                        "retry_count": _MAX_AUTO_RETRIES,
                    })
                    log.info("opt_analyst_confirmed_rejection", strategy_id=strategy_id)
                except Exception as exc:
                    log.error("opt_analyst_error",
                              strategy_id=strategy_id, error=str(exc))
            continue  # quality rejection — leave as failed

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

                # Build extra context: multi-TF test results are very useful for
                # diagnosing zero-trades (shows which timeframes/signals are missing)
                tf_data = strategy.get("quick_test_all_timeframes")
                extra_ctx = None
                if tf_data:
                    import json as _json
                    tf_summary = []
                    for tf, m in (tf_data.items() if isinstance(tf_data, dict) else []):
                        if "error" in m:
                            tf_summary.append(f"  {tf}: ERROR — {m['error'][:80]}")
                        else:
                            tf_summary.append(
                                f"  {tf}: trades={m.get('trades',0)}, "
                                f"sharpe={m.get('sharpe',0):.4f}, "
                                f"win={m.get('win_rate',0):.0%}"
                            )
                    if tf_summary:
                        extra_ctx = "Multi-timeframe test results (default params):\n" + "\n".join(tf_summary)

                # For optimization regressions: include the params that caused 0 trades
                # so the LLM can pinpoint which param value broke the strategy.
                if "optimization_regression" in error_log.lower():
                    hyperparams = strategy.get("hyperparams")
                    if hyperparams:
                        import json as _json
                        opt_ctx = f"Failed optimized params (caused 0 trades): {_json.dumps(hyperparams)}"
                        extra_ctx = f"{extra_ctx}\n{opt_ctx}" if extra_ctx else opt_ctx

                fixed_code = fix_strategy_code(
                    code=code,
                    error_log=error_log,
                    strategy_description=description,
                    strategy_id=strategy_id,
                    extra_context=extra_ctx,
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
    strategy_jobs_recovered = _recover_stuck_strategy_jobs()
    filtered_processed    = _process_filtered_strategies()
    research_unblocked    = _process_awaiting_research_strategies()
    research_recovered    = _recover_stuck_research_tasks()
    research_retried      = _retry_failed_research_tasks()
    research_generated    = _auto_generate_research_tasks()
    research_dispatched   = _dispatch_pending_research_tasks()
    # 'implemented' → quick backtest; 'validating' → validator
    modal_dispatched      = _process_implemented_strategies()
    # 'quick_tested' → analyzer → awaiting_review (standalone) or campaign hold
    full_backtest_queued  = _process_quick_tested_strategies()
    # Summarize completed campaigns so user sees consolidated results
    campaigns_summarized  = _process_campaign_completion()
    auto_fixed            = _process_failed_strategies()
    archived              = _cleanup_old_failed_strategies()

    log.info(
        "queue_worker_done",
        ideas_processed=ideas_processed,
        recovered_ideas=recovered_ideas,
        strategy_jobs_recovered=strategy_jobs_recovered,
        filtered_processed=filtered_processed,
        research_unblocked=research_unblocked,
        research_recovered=research_recovered,
        research_retried=research_retried,
        research_generated=research_generated,
        research_dispatched=research_dispatched,
        modal_dispatched=modal_dispatched,
        full_backtest_queued=full_backtest_queued,
        campaigns_summarized=campaigns_summarized,
        auto_fixed=auto_fixed,
        archived=archived,
    )
