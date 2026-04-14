"""
Main orchestrator entry point. Runs on Render.com as an always-on worker.

Responsibilities:
  - FastAPI /health endpoint for Render health checks
  - APScheduler:
      every 10 min  -> queue_worker.process_queue()
      every 4 hours -> research cycle placeholder
      every 1 hour  -> log budget status
  - Graceful SIGTERM handling (Render sends SIGTERM on shutdown)
"""
from __future__ import annotations

import os
import sys
import traceback  # noqa: F401  (used in scheduled job wrappers)
from contextlib import asynccontextmanager

import structlog
import traceback as _traceback
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import BackgroundTasks, FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from db import supabase_client as db
from orchestrator.budget_guard import get_remaining_budget
from orchestrator.queue_worker import process_queue

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Scheduler (module-level so scheduled job wrappers can reference it)
# ---------------------------------------------------------------------------

scheduler = BackgroundScheduler(timezone="UTC")

# ---------------------------------------------------------------------------
# FastAPI lifespan — replaces manual signal handler
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──────────────────────────────────────────────────────────────
    log.info("orchestrator_starting", python_version=sys.version,
             port=os.environ.get("PORT", 8000))
    try:
        db.get_daily_spend()
        log.info("db_connection_ok")
    except Exception as exc:
        log.error("db_connection_failed", error=str(exc))

    # Run any pending DB migrations before starting the scheduler
    try:
        from db.migrate import run_migrations
        run_migrations()
    except Exception as exc:
        log.error("startup_migrations_failed", error=str(exc))

    _scheduled_budget_log()

    from datetime import datetime, timedelta
    scheduler.add_job(_scheduled_queue_worker, trigger="interval", minutes=10,
                      id="queue_worker", replace_existing=True,
                      next_run_time=datetime.utcnow() + timedelta(seconds=10))
    scheduler.add_job(_scheduled_research_cycle, trigger="interval",
                      hours=int(os.environ.get("RESEARCH_INTERVAL_HOURS", 4)),
                      id="research_cycle", replace_existing=True)
    scheduler.add_job(_scheduled_budget_log, trigger="interval", hours=1,
                      id="budget_log", replace_existing=True)
    scheduler.start()
    log.info("scheduler_started", jobs=[j.id for j in scheduler.get_jobs()])

    yield  # app runs here

    # ── shutdown (uvicorn sends SIGTERM → lifespan exits cleanly) ────────────
    log.info("orchestrator_shutting_down")
    if scheduler.running:
        scheduler.shutdown(wait=False)
    log.info("orchestrator_stopped")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Trading Research Orchestrator", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Health endpoint for Render.com uptime checks and monitoring."""
    try:
        today_spend = db.get_daily_spend()
        remaining = get_remaining_budget()
    except Exception as exc:
        log.warning("health_db_error", error=str(exc))
        today_spend = None
        remaining = None

    return {
        "status": "ok",
        "daily_spend": today_spend,
        "remaining_budget": remaining,
    }


# ---------------------------------------------------------------------------
# Dashboard API
# ---------------------------------------------------------------------------

@app.get("/api/stats")
def api_stats() -> JSONResponse:
    """Summary numbers for the dashboard header cards."""
    try:
        sb = db.get_client()
        rows = sb.table("strategies").select("status").execute().data or []
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1

        total = len(rows)
        done = counts.get("done", 0) + counts.get("live", 0)
        failed = counts.get("failed", 0) + counts.get("rejected", 0)
        in_progress = sum(counts.get(s, 0) for s in (
            "filtered", "implementing", "implemented", "backtesting", "validating"
        ))

        today_spend = db.get_daily_spend()
        remaining = get_remaining_budget()
    except Exception as exc:
        log.error("api_stats_error", error=str(exc), traceback=_traceback.format_exc())
        return JSONResponse({"error": str(exc)}, status_code=500)

    return JSONResponse({
        "total": total,
        "done": done,
        "failed": failed,
        "in_progress": max(0, in_progress),
        "pending_ideas": counts.get("idea", 0) + counts.get("filtered", 0),
        "daily_spend_usd": round(today_spend, 4),
        "remaining_budget_usd": round(remaining, 4),
        "status_breakdown": counts,
    })


@app.post("/api/strategy/{strategy_id}/restart")
def api_restart_strategy(strategy_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Force-redispatch a stuck in-progress strategy to Modal.
    Works for: implemented, backtesting, validating.
      implemented / quick_testing → reset to implemented → redispatch quick backtest
      quick_tested / backtesting  → dispatch full backtest
      awaiting_review             → dispatch full backtest (skip review)
      validating                  → redispatch validator job
    """
    try:
        from orchestrator.queue_worker import (
            _dispatch_quick_backtest_job, _dispatch_backtest_job, _dispatch_validator_job,
        )
        strategy = db.get_strategy(strategy_id)
        if not strategy:
            return JSONResponse({"error": "not found"}, status_code=404)

        status = strategy.get("status")
        if status in ("implemented", "quick_testing"):
            db.update_strategy(strategy_id, {
                "status": "implemented", "error_log": None, "modal_job_id": None,
            })
            background_tasks.add_task(_dispatch_quick_backtest_job, strategy_id)
            log.info("strategy_restarted_quick_backtest", strategy_id=strategy_id)
            return JSONResponse({"ok": True, "dispatched_to": "quick_backtest"})
        elif status in ("quick_tested", "awaiting_review", "backtesting"):
            db.update_strategy(strategy_id, {
                "status": "quick_tested", "error_log": None, "modal_job_id": None,
            })
            background_tasks.add_task(_dispatch_backtest_job, strategy_id)
            log.info("strategy_restarted_full_backtest", strategy_id=strategy_id)
            return JSONResponse({"ok": True, "dispatched_to": "full_backtest"})
        elif status == "validating":
            db.update_strategy(strategy_id, {"modal_job_id": None})
            background_tasks.add_task(_dispatch_validator_job, strategy_id)
            log.info("strategy_restarted_validator", strategy_id=strategy_id)
            return JSONResponse({"ok": True, "dispatched_to": "validator"})
        else:
            return JSONResponse(
                {"error": f"restart not available for status '{status}'"},
                status_code=400,
            )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/strategy/{strategy_id}/approve")
def api_approve_strategy(strategy_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
    """
    User approves the quick-test results and starts full optimization.
    Only valid when status == 'awaiting_review'.
    """
    try:
        from orchestrator.queue_worker import _dispatch_backtest_job
        from agents.utils import add_pipeline_note

        strategy = db.get_strategy(strategy_id)
        if not strategy:
            return JSONResponse({"error": "not found"}, status_code=404)

        status = strategy.get("status")
        # Allow approval from awaiting_review (standalone) or quick_tested (campaign child)
        if status not in ("awaiting_review", "quick_tested"):
            return JSONResponse(
                {"error": f"Cannot approve strategy with status '{status}'"},
                status_code=400,
            )

        db.update_strategy(strategy_id, {
            "status": "quick_tested",
            "modal_job_id": None,
            "error_log": None,
        })
        add_pipeline_note(strategy_id, "User approved — starting full optimization (Optuna + walk-forward).")
        background_tasks.add_task(_dispatch_backtest_job, strategy_id)
        log.info("strategy_approved_for_optimization", strategy_id=strategy_id)
        return JSONResponse({"ok": True})
    except Exception as exc:
        log.error("approve_strategy_error", strategy_id=strategy_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/strategy/{strategy_id}/revise")
async def api_revise_strategy(
    strategy_id: str, request: Request, background_tasks: BackgroundTasks
) -> JSONResponse:
    """
    User requests a code change during the review stage.
    Body: {"message": "add a volume filter and limit to London session"}
    The reviewer agent updates backtest_code and re-dispatches the quick test.
    """
    try:
        from agents.strategy_reviewer import run_strategy_reviewer
        from orchestrator.queue_worker import _dispatch_quick_backtest_job
        from orchestrator.budget_guard import check_budget, BudgetExceeded

        body = await request.json()
        message = (body.get("message") or "").strip()
        if not message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        strategy = db.get_strategy(strategy_id)
        if not strategy:
            return JSONResponse({"error": "not found"}, status_code=404)

        try:
            check_budget("strategy_reviewer")
        except BudgetExceeded as be:
            return JSONResponse({"error": str(be)}, status_code=429)

        def _run_revision():
            result = run_strategy_reviewer(strategy_id, message)
            if "error" not in result:
                _dispatch_quick_backtest_job(strategy_id)

        background_tasks.add_task(_run_revision)
        log.info("strategy_revision_requested", strategy_id=strategy_id,
                 message=message[:80])
        return JSONResponse({"ok": True, "message": "Revision queued — re-running quick test."})
    except Exception as exc:
        log.error("revise_strategy_error", strategy_id=strategy_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/strategy/{strategy_id}/campaign")
def api_campaign_children(strategy_id: str) -> JSONResponse:
    """
    Return all variations in a campaign, including the root as variation #1.
    Sorted by quick_test_sharpe descending.
    """
    try:
        children  = db.get_campaign_children(strategy_id)
        root_full = db.get_strategy(strategy_id)
        if not root_full:
            return JSONResponse({"error": "not found"}, status_code=404)

        root_entry = {
            "id":                        root_full["id"],
            "name":                      root_full.get("name", ""),
            "status":                    root_full.get("status", ""),
            "hypothesis":                root_full.get("hypothesis", ""),
            "quick_test_sharpe":         root_full.get("quick_test_sharpe"),
            "quick_test_trades":         root_full.get("quick_test_trades"),
            "quick_test_win_rate":       root_full.get("quick_test_win_rate"),
            "quick_test_drawdown":       root_full.get("quick_test_drawdown"),
            "quick_test_signals_per_year": root_full.get("quick_test_signals_per_year"),
            "best_timeframe":            root_full.get("best_timeframe"),
            "error_log":                 root_full.get("error_log"),
            "is_root":                   True,
        }
        all_variations = [root_entry] + [dict(c, is_root=False) for c in children]
        all_variations.sort(
            key=lambda x: x.get("quick_test_sharpe") or -999,
            reverse=True,
        )
        passed = sum(
            1 for v in all_variations
            if (v.get("quick_test_sharpe") or 0) > 0
            and (v.get("quick_test_trades") or 0) >= 30
        )
        return JSONResponse({
            "variations": all_variations,
            "total":      len(all_variations),
            "passed":     passed,
        })
    except Exception as exc:
        log.error("api_campaign_children_error", strategy_id=strategy_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/strategy/{strategy_id}/retry")
def api_retry_strategy(strategy_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Reset a failed strategy to the appropriate previous status so the queue
    worker picks it up again on the next cycle.

    Retry logic (most conservative — go back one step):
      - Has backtest_code → reset to "implemented"  (re-dispatch to Modal)
      - Has pre_filter_score but no code → reset to "filtered" (re-run implementer)
      - Nothing yet → reset to "idea" (re-run pre_filter)
    """
    try:
        strategy = db.get_strategy(strategy_id)
        if not strategy:
            return JSONResponse({"error": "not found"}, status_code=404)
        if strategy.get("status") != "failed":
            return JSONResponse({"error": "only failed strategies can be retried"}, status_code=400)

        if strategy.get("backtest_code"):
            retry_status = "implemented"
        elif strategy.get("pre_filter_score") is not None:
            retry_status = "filtered"
        else:
            retry_status = "idea"

        retry_count = (strategy.get("retry_count") or 0) + 1
        db.update_strategy(strategy_id, {
            "status": retry_status,
            "error_log": None,
            "retry_count": retry_count,
        })
        log.info("strategy_retry", strategy_id=strategy_id, retry_status=retry_status)
        background_tasks.add_task(_scheduled_queue_worker)
        return JSONResponse({"ok": True, "retry_status": retry_status})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# Statuses that mean the pipeline has moved on — no job is actively running
_TERMINAL_STATUSES = {"quick_tested", "awaiting_review", "implemented",
                      "backtesting", "validating", "validated", "live",
                      "failed", "archived", "filtered", "idea"}
# Max age before we consider an in-progress strategy stuck (minutes)
_STUCK_MINUTES = 25


@app.get("/api/strategy/{strategy_id}/modal-status")
def api_modal_status(strategy_id: str) -> JSONResponse:
    """Check whether the Modal job for this strategy is still running, done, or stuck."""
    try:
        strategy = db.get_strategy(strategy_id)
        if not strategy:
            return JSONResponse({"error": "not found"}, status_code=404)

        status = strategy.get("status", "")
        job_id = strategy.get("modal_job_id")

        # ── DB-first check: if status already advanced, job is done ──────────
        # Modal's polling API (call.get) can report "running" for completed jobs.
        # The authoritative source is the DB status set by the Modal function itself.
        if status in _TERMINAL_STATUSES or not job_id:
            return JSONResponse({"status": "done",
                                 "message": f"Strategy is now '{status}' — job finished."})

        # ── Stuck detection ──────────────────────────────────────────────────
        # If the strategy has been in an in-progress state for > _STUCK_MINUTES
        # with a job_id still set, the Modal job likely crashed/timed-out before
        # it could update the DB.
        updated_at_str = strategy.get("updated_at") or strategy.get("created_at") or ""
        try:
            import datetime as _dt
            updated_at = _dt.datetime.fromisoformat(
                updated_at_str.replace("Z", "+00:00").replace("+00:00+00:00", "+00:00")
            )
            age_minutes = (_dt.datetime.now(_dt.timezone.utc) - updated_at).total_seconds() / 60
        except Exception:
            age_minutes = 0

        if age_minutes > _STUCK_MINUTES:
            return JSONResponse({
                "status": "stuck",
                "age_minutes": round(age_minutes),
                "job_id": job_id,
                "message": (
                    f"Job has been running for {round(age_minutes)} min "
                    f"(limit: {_STUCK_MINUTES} min). "
                    "It likely timed out or crashed without updating the database. "
                    "Use 'Restart' to try again."
                ),
            })

        # ── Poll Modal (only for genuinely recent jobs) ──────────────────────
        try:
            import modal
            call = modal.FunctionCall.from_id(job_id)
            try:
                result = call.get(timeout=2)
                return JSONResponse({"status": "done", "result": str(result)[:500]})
            except TimeoutError:
                return JSONResponse({"status": "running", "job_id": job_id,
                                     "age_minutes": round(age_minutes)})
            except Exception as poll_exc:
                # Any non-timeout exception from Modal usually means job ended
                # (succeeded, failed, or expired). Trust the DB status.
                return JSONResponse({"status": "done",
                                     "message": f"Modal poll error (job likely finished): {poll_exc}",
                                     "job_id": job_id})
        except Exception as modal_import_err:
            return JSONResponse({"status": "unknown",
                                 "error": f"Could not import modal: {modal_import_err}"})

    except Exception as exc:
        log.error("modal_status_check_error", strategy_id=strategy_id,
                  error=str(exc), traceback=_traceback.format_exc())
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/strategy/{strategy_id}/update")
async def api_update_strategy(strategy_id: str, request: Request,
                               background_tasks: BackgroundTasks) -> JSONResponse:
    """Update hypothesis/name on a failed strategy and reset to 'idea' for full re-run."""
    try:
        body = await request.json()
        strategy = db.get_strategy(strategy_id)
        if not strategy:
            return JSONResponse({"error": "not found"}, status_code=404)

        updates: dict = {"error_log": None, "retry_count": (strategy.get("retry_count") or 0) + 1}
        if "name" in body and body["name"].strip():
            updates["name"] = body["name"].strip()
        if "hypothesis" in body and body["hypothesis"].strip():
            updates["hypothesis"] = body["hypothesis"].strip()
        # Always restart from scratch when content changes
        updates["status"] = "idea"
        updates["pre_filter_score"] = None
        updates["backtest_code"] = None
        updates["hyperparams"] = None

        db.update_strategy(strategy_id, updates)
        log.info("strategy_updated", strategy_id=strategy_id)
        background_tasks.add_task(_scheduled_queue_worker)
        return JSONResponse({"ok": True})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.delete("/api/strategy/{strategy_id}")
def api_delete_strategy(strategy_id: str) -> JSONResponse:
    try:
        sb = db.get_client()
        # Clear all FK references before deleting
        sb.table("user_ideas").update({"strategy_id": None}).eq("strategy_id", strategy_id).execute()
        sb.table("spend_log").delete().eq("strategy_id", strategy_id).execute()
        sb.table("knowledge_base").delete().eq("strategy_id", strategy_id).execute()
        sb.table("strategies").delete().eq("id", strategy_id).execute()
        log.info("strategy_deleted", strategy_id=strategy_id)
        return JSONResponse({"ok": True})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/strategy/{strategy_id}/comment")
async def api_add_comment(strategy_id: str, request: Request) -> JSONResponse:
    """Append a user comment to a strategy without touching the original description."""
    try:
        body = await request.json()
        text = (body.get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "comment text required"}, status_code=400)

        strategy = db.get_strategy(strategy_id)
        if not strategy:
            return JSONResponse({"error": "not found"}, status_code=404)

        import datetime
        existing = strategy.get("comments") or []
        if isinstance(existing, str):
            import json as _json
            existing = _json.loads(existing)
        new_comment = {"text": text, "ts": datetime.datetime.utcnow().isoformat()[:19]}
        existing.append(new_comment)
        db.update_strategy(strategy_id, {"comments": existing})
        return JSONResponse({"ok": True, "comments": existing})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/strategy/{strategy_id}/tags")
async def api_update_tags(strategy_id: str, request: Request) -> JSONResponse:
    try:
        body = await request.json()
        tags = [t.strip() for t in body.get("tags", []) if t.strip()]
        db.update_strategy(strategy_id, {"tags": tags})
        return JSONResponse({"ok": True, "tags": tags})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/strategies")
def api_strategies(
    status: str = Query("all"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> JSONResponse:
    """Paginated strategy list for the dashboard table."""
    try:
        sb = db.get_client()
        q = sb.table("strategies").select(
            "id, name, source, status, pre_filter_score, "
            "backtest_sharpe, backtest_calmar, max_drawdown, win_rate, "
            "signals_per_year, total_signals, leakage_score, profit_factor, "
            "oos_sharpe, oos_win_rate, oos_total_trades, monte_carlo_pvalue, "
            "walk_forward_scores, hypothesis, entry_logic, hyperparams, best_session_hours, "
            "quick_test_sharpe, quick_test_trades, quick_test_win_rate, "
            "quick_test_drawdown, quick_test_signals_per_year, "
            "best_timeframe, quick_test_all_timeframes, "
            "campaign_id, is_campaign_root, "
            "error_log, report_url, tags, comments, modal_job_id, created_at, updated_at"
        )
        if status != "all":
            q = q.eq("status", status)
        result = q.order("updated_at", desc=True).range(offset, offset + limit - 1).execute()
        return JSONResponse({"strategies": result.data or [], "offset": offset, "limit": limit})
    except Exception as exc:
        log.error("api_strategies_error", error=str(exc), traceback=_traceback.format_exc())
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/strategy/{strategy_id}")
def api_strategy_detail(strategy_id: str) -> JSONResponse:
    """Full strategy record for the detail panel."""
    try:
        strategy = db.get_strategy(strategy_id)
        if not strategy:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(strategy)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Generated-ideas API (Research page)
# ---------------------------------------------------------------------------

@app.get("/api/generated-ideas")
def api_generated_ideas(
    status: str = Query("pending"),
    limit: int = Query(50, ge=1, le=200),
) -> JSONResponse:
    try:
        ideas = db.get_generated_ideas(status=status, limit=limit)
        return JSONResponse({"ideas": ideas})
    except Exception as exc:
        log.error("api_generated_ideas_error", error=str(exc), traceback=_traceback.format_exc())
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/generated-ideas/{idea_id}/approve")
def api_approve_generated_idea(
    idea_id: str, background_tasks: BackgroundTasks
) -> JSONResponse:
    """Approve a generated idea: mark as approved and insert into user_ideas queue."""
    try:
        sb = db.get_client()
        result = sb.table("generated_ideas").select("*").eq("id", idea_id).execute()
        if not result.data:
            return JSONResponse({"error": "not found"}, status_code=404)
        idea = result.data[0]

        # Insert into user_ideas pipeline
        inserted = sb.table("user_ideas").insert({
            "title":       idea["title"],
            "description": idea["summary"],
            "notes":       f"Source: {idea.get('source_title', '')} ({idea.get('source_type', '')})\n{idea.get('source_url', '')}",
            "status":      "pending",
        }).execute()
        user_idea_id = inserted.data[0]["id"] if inserted.data else None

        update_data: dict = {"status": "approved"}
        if user_idea_id:
            update_data["user_idea_id"] = user_idea_id
        db.update_generated_idea(idea_id, update_data)
        log.info("generated_idea_approved", idea_id=idea_id, user_idea_id=user_idea_id)

        # Kick the queue immediately
        background_tasks.add_task(_scheduled_queue_worker)
        return JSONResponse({"ok": True, "user_idea_id": user_idea_id})
    except Exception as exc:
        log.error("api_approve_idea_error", error=str(exc), traceback=_traceback.format_exc())
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/generated-ideas/{idea_id}/dismiss")
def api_dismiss_generated_idea(idea_id: str) -> JSONResponse:
    try:
        db.update_generated_idea(idea_id, {"status": "dismissed"})
        return JSONResponse({"ok": True})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/queue/run")
def api_queue_run(background_tasks: BackgroundTasks) -> JSONResponse:
    """Trigger an immediate queue processing cycle (runs in background)."""
    background_tasks.add_task(_scheduled_queue_worker)
    return JSONResponse({"ok": True, "message": "Queue worker started in background."})


@app.post("/api/generated-ideas/refresh")
def api_refresh_generated_ideas(background_tasks: BackgroundTasks) -> JSONResponse:
    """Trigger a manual idea-generation cycle (runs in background)."""
    background_tasks.add_task(_scheduled_research_cycle)
    return JSONResponse({"ok": True, "message": "Research cycle started in background."})


# ---------------------------------------------------------------------------
# Data cache API
# ---------------------------------------------------------------------------

@app.get("/api/data/cache")
def api_data_cache() -> JSONResponse:
    """Return metadata for all cached OHLCV datasets (no chart data)."""
    try:
        datasets = db.get_data_cache()
        return JSONResponse({"datasets": datasets})
    except Exception as exc:
        log.error("api_data_cache_error", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/data/cache/{symbol}/{timeframe}")
def api_data_cache_bars(symbol: str, timeframe: str) -> JSONResponse:
    """Return recent_bars for a single dataset (used by the price chart)."""
    try:
        bars = db.get_data_cache_bars(symbol.upper(), timeframe.lower())
        return JSONResponse({"bars": bars})
    except Exception as exc:
        log.error("api_data_cache_bars_error", symbol=symbol, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/data/preload")
def api_data_preload(background_tasks: BackgroundTasks) -> JSONResponse:
    """Trigger the Modal preload job to fetch/refresh OHLCV data."""
    def _run_preload():
        try:
            import modal as _modal
            fn = _modal.Function.from_name(
                "trading-research-preload", "preload_ohlcv_data"
            )
            fn.spawn()
            log.info("preload_job_spawned")
        except Exception as exc:
            log.error("preload_job_spawn_failed", error=str(exc))

    background_tasks.add_task(_run_preload)
    return JSONResponse({"ok": True, "message": "Preload job started on Modal."})


# ---------------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Research Dashboard</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0a0d14; color: #e2e8f0; min-height: 100vh; }

  /* ── nav ── */
  .nav { display: flex; align-items: center; gap: 0; background: #111827;
         border-bottom: 1px solid #1f2937; padding: 0 24px; }
  .nav-logo { font-weight: 700; font-size: 0.95rem; color: #f8fafc;
              padding: 14px 0; margin-right: 32px; letter-spacing: -.01em; }
  .nav a { display: block; padding: 14px 16px; font-size: 0.85rem; color: #94a3b8;
           text-decoration: none; border-bottom: 2px solid transparent; transition: color .15s; }
  .nav a:hover, .nav a.active { color: #f1f5f9; border-bottom-color: #6366f1; }
  .nav-right { margin-left: auto; display: flex; align-items: center; gap: 16px; }
  .budget-pill { background: #1e2533; border: 1px solid #374151; border-radius: 99px;
                 padding: 4px 14px; font-size: 0.78rem; color: #94a3b8; }
  .budget-pill span { color: #f1f5f9; font-weight: 600; }
  .refresh-btn { background: #1e2533; border: 1px solid #374151; border-radius: 8px;
                 color: #94a3b8; font-size: 0.8rem; padding: 6px 12px; cursor: pointer; }
  .refresh-btn:hover { color: #f1f5f9; }

  /* ── layout ── */
  .page { max-width: 1280px; margin: 0 auto; padding: 28px 24px; }

  /* ── stat cards ── */
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 28px; }
  .card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 18px 20px; }
  .card-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                letter-spacing: .07em; color: #64748b; margin-bottom: 8px; }
  .card-value { font-size: 1.9rem; font-weight: 700; color: #f8fafc; line-height: 1; }
  .card-value.green  { color: #4ade80; }
  .card-value.red    { color: #f87171; }
  .card-value.yellow { color: #fbbf24; }
  .card-value.blue   { color: #60a5fa; }

  /* ── filter tabs ── */
  .tabs { display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }
  .tab { padding: 6px 16px; border-radius: 99px; font-size: 0.8rem; font-weight: 500;
         border: 1px solid #1f2937; background: transparent; color: #64748b; cursor: pointer; transition: all .15s; }
  .tab:hover   { color: #f1f5f9; border-color: #374151; }
  .tab.active  { background: #6366f1; border-color: #6366f1; color: #fff; }
  .tab .count  { background: rgba(255,255,255,.15); border-radius: 99px;
                 padding: 1px 7px; font-size: 0.72rem; margin-left: 6px; }

  /* ── table ── */
  .table-wrap { background: #111827; border: 1px solid #1f2937; border-radius: 12px; overflow: hidden; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  thead tr { background: #0f1623; }
  th { padding: 11px 14px; text-align: left; font-size: 0.72rem; font-weight: 600;
       text-transform: uppercase; letter-spacing: .06em; color: #475569;
       border-bottom: 1px solid #1f2937; white-space: nowrap; }
  td { padding: 12px 14px; border-bottom: 1px solid #151e2d; vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr.data-row:hover td { background: #141c2e; cursor: pointer; }

  /* ── status badges ── */
  .badge { display: inline-block; padding: 2px 10px; border-radius: 99px;
           font-size: 0.72rem; font-weight: 600; white-space: nowrap; }
  .s-idea              { background: #1e293b; color: #94a3b8; }
  .s-filtered          { background: #1e3a5f; color: #93c5fd; }
  .s-implementing      { background: #2d1f5e; color: #c4b5fd; animation: pulse-purple 2s infinite; }
  .s-awaiting-research { background: #3b2000; color: #fb923c; animation: pulse-orange 2s infinite; }
  .s-implemented       { background: #2d1f5e; color: #c4b5fd; }
  .s-quick-testing     { background: #1a3a2a; color: #4ade80; animation: pulse-green 2s infinite; }
  .s-quick-tested      { background: #1a3a2a; color: #4ade80; }
  .s-awaiting-review   { background: #1e1b4b; color: #818cf8; animation: pulse-indigo 2s infinite; }
  .s-backtesting       { background: #3b2f00; color: #fcd34d; animation: pulse-yellow 2s infinite; }
  .s-validating        { background: #1c3352; color: #67e8f9; animation: pulse-cyan 2s infinite; }
  .s-live              { background: #14532d; color: #86efac; }
  .s-done              { background: #14532d; color: #86efac; }
  .s-failed            { background: #450a0a; color: #fca5a5; }
  .s-rejected          { background: #450a0a; color: #fca5a5; }
  .s-archived          { background: #1c1c1c; color: #475569; }

  @keyframes pulse-indigo {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,.5); }
    50%       { box-shadow: 0 0 0 4px rgba(99,102,241,0); }
  }
  @keyframes pulse-purple {
    0%, 100% { box-shadow: 0 0 0 0 rgba(139,92,246,.5); }
    50%       { box-shadow: 0 0 0 4px rgba(139,92,246,0); }
  }
  @keyframes pulse-yellow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(251,191,36,.5); }
    50%       { box-shadow: 0 0 0 4px rgba(251,191,36,0); }
  }
  @keyframes pulse-cyan {
    0%, 100% { box-shadow: 0 0 0 0 rgba(103,232,249,.5); }
    50%       { box-shadow: 0 0 0 4px rgba(103,232,249,0); }
  }
  @keyframes pulse-orange {
    0%, 100% { box-shadow: 0 0 0 0 rgba(251,146,60,.5); }
    50%       { box-shadow: 0 0 0 4px rgba(251,146,60,0); }
  }
  @keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(74,222,128,.5); }
    50%       { box-shadow: 0 0 0 4px rgba(74,222,128,0); }
  }

  /* ── metric cells ── */
  .metric { font-family: "SF Mono", "Fira Code", monospace; font-size: 0.82rem; }
  .pos { color: #4ade80; } .neg { color: #f87171; } .neu { color: #94a3b8; }

  /* ── detail panel (centered fullscreen dialog) ── */
  .panel-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.75);
                   z-index: 100; display: none;
                   align-items: flex-start; justify-content: center;
                   padding: 24px 16px; overflow-y: auto; }
  .panel-overlay.open { display: flex; }
  .panel { position: relative; width: 100%; max-width: 960px; min-height: 0;
           background: #111827; border: 1px solid #1f2937; border-radius: 14px;
           overflow: visible; z-index: 101; padding: 32px 36px;
           margin: auto;
           opacity: 0; transform: scale(.97) translateY(8px);
           transition: opacity .2s ease, transform .2s ease; }
  .panel-overlay.open .panel { opacity: 1; transform: scale(1) translateY(0); }
  .panel-close { position: absolute; top: 16px; right: 18px;
                 background: #1e2533; border: 1px solid #374151; border-radius: 8px;
                 color: #94a3b8; font-size: 1.1rem; cursor: pointer;
                 padding: 4px 10px; line-height: 1; }
  .panel-close:hover { color: #f1f5f9; background: #374151; }
  .panel h2 { font-size: 1.2rem; font-weight: 700; color: #f8fafc;
              margin-bottom: 6px; padding-right: 48px; }
  .panel-hyp { color: #94a3b8; font-size: 0.875rem; margin-bottom: 20px; line-height: 1.6; }
  .panel-section { margin-bottom: 20px; }
  .panel-section h3 { font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                      letter-spacing: .07em; color: #475569; margin-bottom: 10px; }
  .kv-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .kv { background: #0f1623; border-radius: 8px; padding: 10px 12px; }
  .kv-label { font-size: 0.7rem; color: #64748b; margin-bottom: 3px; text-transform: uppercase; letter-spacing: .05em; }
  .kv-val   { font-size: 0.95rem; font-weight: 600; color: #f1f5f9;
              font-family: "SF Mono", monospace; }
  .kv-val.good { color: #4ade80; } .kv-val.bad { color: #f87171; }
  .error-box { background: #1f0a0a; border: 1px solid #7f1d1d; border-radius: 8px;
               padding: 12px; font-size: 0.8rem; color: #fca5a5; white-space: pre-wrap; }
  pre.params { background: #0f1623; border-radius: 8px; padding: 12px;
               font-size: 0.78rem; color: #a5b4fc; overflow-x: auto; }
  .empty { text-align: center; padding: 60px 0; color: #475569; font-size: 0.9rem; }
  .loading { text-align: center; padding: 40px 0; color: #475569; }
  .tag-pill { background:#1e293b;border-radius:99px;padding:3px 10px;font-size:.75rem;color:#93c5fd; }
  details summary::-webkit-details-marker { display:none; }
  details[open] summary { color: #94a3b8; }
</style>
</head>
<body>

<nav class="nav">
  <div class="nav-logo">Trading Research</div>
  <a href="/dashboard" class="active">Dashboard</a>
  <a href="/ideas">Ideas</a>
  <a href="/research">Research</a>
  <a href="/data">Data</a>

  <div class="nav-right">
    <div class="budget-pill">Today: <span id="spend">…</span> / <span id="limit">$8.00</span></div>
    <button class="refresh-btn" onclick="loadAll()">↻ Refresh</button>
  </div>
</nav>

<div class="page">
  <div class="cards" id="cards">
    <div class="card"><div class="card-label">Total</div><div class="card-value" id="c-total">…</div></div>
    <div class="card"><div class="card-label">Done</div><div class="card-value green" id="c-done">…</div></div>
    <div class="card"><div class="card-label">Failed</div><div class="card-value red" id="c-failed">…</div></div>
    <div class="card"><div class="card-label">In Progress</div><div class="card-value blue" id="c-progress">…</div></div>
    <div class="card"><div class="card-label">Queued</div><div class="card-value yellow" id="c-queued">…</div></div>
  </div>

  <div class="tabs" id="tabs"></div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Strategy</th>
          <th>Status</th>
          <th>Tags</th>
          <th>Sharpe</th>
          <th>OOS Sharpe</th>
          <th>Drawdown</th>
          <th>Win Rate</th>
          <th>Sig/Year</th>
          <th>Leakage</th>
          <th>Updated</th>
        </tr>
      </thead>
      <tbody id="tbody"><tr><td colspan="10" class="loading">Loading…</td></tr></tbody>
    </table>
  </div>
</div>

<!-- Detail panel — centered dialog -->
<div class="panel-overlay" id="overlay" onclick="if(event.target===this)closePanel()">
  <div class="panel" id="panel">
    <button class="panel-close" onclick="closePanel()">✕ close</button>
    <div id="panel-content"></div>
  </div>
</div>

<script>
const STATUS_ORDER = ['all','idea','filtered','implementing','awaiting_research','implemented','quick_testing','quick_tested','awaiting_review','backtesting','validating','live','failed','archived'];
let currentStatus = 'all';
let statsData = {};
let strategiesData = [];

async function loadStats() {
  const r = await fetch('/api/stats');
  statsData = await r.json();
  document.getElementById('c-total').textContent    = statsData.total ?? '0';
  document.getElementById('c-done').textContent     = statsData.done ?? '0';
  document.getElementById('c-failed').textContent   = statsData.failed ?? '0';
  document.getElementById('c-progress').textContent = statsData.in_progress ?? '0';
  document.getElementById('c-queued').textContent   = statsData.pending_ideas ?? '0';
  document.getElementById('spend').textContent      = '$' + (statsData.daily_spend_usd ?? 0).toFixed(2);
  renderTabs(statsData.status_breakdown || {});
}

async function loadStrategies() {
  const url = '/api/strategies?status=' + currentStatus + '&limit=100';
  const r = await fetch(url);
  const data = await r.json();
  strategiesData = data.strategies || [];
  renderTable(strategiesData);
}

function loadAll() { loadStats(); loadStrategies(); }

function renderTabs(breakdown) {
  const tabs = document.getElementById('tabs');
  const total = Object.values(breakdown).reduce((a,b) => a+b, 0);
  breakdown.all = total;
  tabs.innerHTML = STATUS_ORDER.map(s => {
    const cnt = breakdown[s] ?? 0;
    const active = s === currentStatus ? 'active' : '';
    return `<button class="tab ${active}" onclick="switchTab('${s}')">
      ${s.charAt(0).toUpperCase() + s.slice(1)}
      <span class="count">${cnt}</span>
    </button>`;
  }).join('');
}

function switchTab(s) {
  currentStatus = s;
  loadAll();
}

const STATUS_LABELS = {
  'idea':              'Idea',
  'filtered':          'Queued',
  'implementing':      '⚙ Implementing…',
  'awaiting_research': '⏳ Awaiting Research',
  'implemented':       'Dispatched',
  'quick_testing':     '⚙ Quick Test…',
  'quick_tested':      'Quick Tested',
  'awaiting_review':   '👁 Awaiting Review',
  'backtesting':       '⚙ Optimizing…',
  'validating':        '⚙ Validating…',
  'live':              '✓ Live',
  'done':              '✓ Done',
  'failed':            '✗ Failed',
  'rejected':          '✗ Rejected',
  'archived':          '↓ Archived',
};
function statusLabel(s) { return STATUS_LABELS[s] || s; }

function renderTable(rows) {
  const tbody = document.getElementById('tbody');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="empty">No strategies yet in this status.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => {
    const sharpe    = fmtNum(r.backtest_sharpe);
    const oosSharpe = fmtNum(r.oos_sharpe);
    const dd        = r.max_drawdown != null ? (r.max_drawdown * 100).toFixed(1) + '%' : '—';
    const wr        = r.win_rate != null ? (r.win_rate * 100).toFixed(1) + '%' : '—';
    const spy       = r.signals_per_year != null ? Math.round(r.signals_per_year) : '—';
    const leak      = r.leakage_score != null ? r.leakage_score.toFixed(1) : '—';
    const leakCls   = r.leakage_score >= 7 ? 'pos' : r.leakage_score >= 4 ? 'neu' : 'neg';
    const sharpeCls = r.backtest_sharpe > 1 ? 'pos' : r.backtest_sharpe > 0 ? 'neu' : 'neg';
    const oosCls    = r.oos_sharpe > 0.8 ? 'pos' : r.oos_sharpe > 0 ? 'neu' : 'neg';
    const updated   = r.updated_at ? r.updated_at.slice(0,16).replace('T',' ') : '—';
    const badgeCls  = 's-' + (r.status || 'idea').replace(/_/g,'-');
    const tagsHtml  = (r.tags || []).map(t =>
      `<span style="background:#1e293b;border-radius:99px;padding:1px 8px;font-size:.7rem;
                    color:#93c5fd;margin-right:3px;white-space:nowrap">${esc(t)}</span>`
    ).join('');
    return `<tr class="data-row" onclick="openPanel('${r.id}')">
      <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(r.name)}">${esc(r.name)}</td>
      <td><span class="badge ${badgeCls}">${statusLabel(r.status)}</span></td>
      <td style="max-width:140px">${tagsHtml || '<span style="color:#374151">—</span>'}</td>
      <td class="metric ${sharpeCls}">${sharpe}</td>
      <td class="metric ${oosCls}">${oosSharpe}</td>
      <td class="metric">${dd}</td>
      <td class="metric">${wr}</td>
      <td class="metric">${spy}</td>
      <td class="metric ${leakCls}">${leak}</td>
      <td style="color:#475569;font-size:.78rem;white-space:nowrap">${updated}</td>
    </tr>`;
  }).join('');
}

async function openPanel(id) {
  document.getElementById('overlay').classList.add('open');
  document.getElementById('panel-content').innerHTML = '<div class="loading">Loading…</div>';
  // Prevent body scroll while dialog is open
  document.body.style.overflow = 'hidden';
  const r = await fetch('/api/strategy/' + id);
  const s = await r.json();
  renderPanel(s);
}

function closePanel() {
  document.getElementById('overlay').classList.remove('open');
  document.body.style.overflow = '';
}

function renderPanel(s) {
  const badgeCls = 's-' + (s.status || 'idea').replace(/_/g,'-');
  const trainSharpe = fmtNum(s.backtest_sharpe);
  const oosSharpe   = fmtNum(s.oos_sharpe);
  const trainCls    = s.backtest_sharpe > 1 ? 'good' : s.backtest_sharpe > 0 ? '' : 'bad';
  const oosCls      = s.oos_sharpe > 0.8 ? 'good' : s.oos_sharpe > 0 ? '' : 'bad';
  const mc = s.monte_carlo_pvalue != null ? s.monte_carlo_pvalue.toFixed(4) : '—';
  const mcCls = s.monte_carlo_pvalue < 0.05 ? 'good' : 'bad';
  const dd = s.max_drawdown != null ? (s.max_drawdown * 100).toFixed(2) + '%' : '—';
  const wr = s.win_rate != null ? (s.win_rate * 100).toFixed(1) + '%' : '—';

  // Quick test results block — multi-timeframe table
  let quickTestHtml = '';
  if (s.quick_test_trades != null || s.quick_test_sharpe != null || s.quick_test_all_timeframes) {
    const bestTf   = s.best_timeframe || '—';
    const allTf    = s.quick_test_all_timeframes || {};
    const TF_ORDER = ['4h','1h','15m','5m','1m'];

    // Build rows for the per-TF table
    let tfRows = '';
    for (const tf of TF_ORDER) {
      if (!(tf in allTf)) continue;
      const m = allTf[tf];
      const isBest = tf === bestTf;
      const rowStyle = isBest ? 'background:#172554;font-weight:600' : '';
      if (m.error) {
        tfRows += `<tr style="${rowStyle}">
          <td>${isBest ? '▶ ' : ''}${tf}</td>
          <td colspan="6" style="color:#f87171;font-size:.75rem">${m.error.slice(0,80)}</td>
        </tr>`;
      } else {
        const sh  = m.sharpe   != null ? (m.sharpe >= 0 ? '+' : '') + m.sharpe.toFixed(4) : '—';
        const shCls = m.sharpe > 1 ? 'color:#34d399' : m.sharpe > 0 ? 'color:#94a3b8' : 'color:#f87171';
        const wr  = m.win_rate != null ? (m.win_rate*100).toFixed(1)+'%' : '—';
        const dd  = m.drawdown != null ? (m.drawdown*100).toFixed(1)+'%' : '—';
        const pf  = m.profit_factor != null ? m.profit_factor.toFixed(2) : '—';
        const spy = m.signals_per_year != null ? Math.round(m.signals_per_year) : '—';
        tfRows += `<tr style="${rowStyle}">
          <td style="font-weight:600">${isBest ? '▶ ' : ''}${tf}</td>
          <td style="${shCls}">${sh}</td>
          <td>${m.trades ?? '—'}</td>
          <td>${wr}</td>
          <td>${pf}</td>
          <td>${dd}</td>
          <td style="color:#64748b">${spy}</td>
        </tr>`;
      }
    }

    const tableStyle = 'width:100%;border-collapse:collapse;font-size:.8rem;margin-top:10px';
    const thStyle    = 'text-align:left;padding:4px 8px;border-bottom:1px solid #334155;color:#94a3b8;font-weight:400';
    const tdStyle    = 'padding:4px 8px;border-bottom:1px solid #1e293b';

    quickTestHtml = `<div class="panel-section">
      <h3>Multi-Timeframe Quick Test <span style="font-size:.7rem;font-weight:400;color:#64748b">(default params — best: <b style="color:#38bdf8">${bestTf}</b>)</span></h3>
      <style>
        .tf-table td { ${tdStyle} }
        .tf-table th { ${thStyle} }
        .tf-table tr:hover td { background:#1e293b }
      </style>
      <table class="tf-table" style="${tableStyle}">
        <thead><tr>
          <th>TF</th><th>Sharpe</th><th>Trades</th><th>Win%</th><th>PF</th><th>DD</th><th>Sig/yr</th>
        </tr></thead>
        <tbody>${tfRows}</tbody>
      </table>
      <div style="font-size:.72rem;color:#64748b;margin-top:6px">
        Sharpe = per-trade annualized. Full optimization will use <b>${bestTf}</b>.
      </div>
      ${(s.quick_test_trades === 0 || (allTf[bestTf] && allTf[bestTf].trades === 0))
        ? '<div style="color:#fb923c;font-size:.8rem;margin-top:6px">⚠ Zero trades on best timeframe — optimizer may still find signal</div>'
        : ''}
    </div>`;
  }

  let wfHtml = '';
  if (s.walk_forward_scores) {
    const scores = typeof s.walk_forward_scores === 'string'
      ? JSON.parse(s.walk_forward_scores) : s.walk_forward_scores;
    if (Array.isArray(scores)) {
      wfHtml = `<div class="panel-section">
        <h3>Walk-Forward OOS (${scores.length} folds)</h3>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          ${scores.map((v,i) => `<div class="kv" style="min-width:90px">
            <div class="kv-label">Fold ${i+1}</div>
            <div class="kv-val ${v>0.8?'good':v>0?'':'bad'}">${typeof v==='number'?v.toFixed(3):v}</div>
          </div>`).join('')}
        </div></div>`;
    }
  }

  let paramsHtml = '';
  if (s.hyperparams) {
    const p = typeof s.hyperparams === 'string' ? JSON.parse(s.hyperparams) : s.hyperparams;
    paramsHtml = `<div class="panel-section">
      <details>
        <summary style="cursor:pointer;font-size:.72rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:.07em;color:#475569;list-style:none;display:flex;
                        align-items:center;gap:6px;user-select:none"
                 onclick="this.parentElement.open ? this.querySelector('.arr').textContent='▶' : this.querySelector('.arr').textContent='▼'">
          <span class="arr" style="font-size:.9rem">▶</span> Best Hyperparameters
        </summary>
        <pre class="params" style="margin-top:8px">${JSON.stringify(p, null, 2)}</pre>
      </details>
    </div>`;
  }

  let codeHtml = '';
  if (s.backtest_code) {
    codeHtml = `<div class="panel-section">
      <details>
        <summary style="cursor:pointer;font-size:.72rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:.07em;color:#475569;list-style:none;display:flex;
                        align-items:center;gap:6px;user-select:none"
                 onclick="this.parentElement.open ? this.querySelector('.arr').textContent='▶' : this.querySelector('.arr').textContent='▼'">
          <span class="arr" style="font-size:.9rem">▶</span> Strategy Source Code
          <span style="font-size:.68rem;color:#374151;font-weight:400;text-transform:none;letter-spacing:0;margin-left:4px">(${s.backtest_code.split('\n').length} lines)</span>
        </summary>
        <pre style="margin-top:8px;background:#0a0f1a;border:1px solid #1e2d3d;border-radius:8px;
                    padding:14px;font-size:.75rem;color:#7dd3fc;overflow-x:auto;
                    max-height:500px;overflow-y:auto;line-height:1.55;
                    font-family:'SF Mono','Fira Code',monospace">${esc(s.backtest_code)}</pre>
      </details>
    </div>`;
  }

  let errHtml = '';
  if (s.error_log) {
    errHtml = `<div class="panel-section"><h3>Error Log</h3>
      <div class="error-box">${esc(s.error_log)}</div></div>`;
  }

  // Campaign panel — shown when strategy is the root of a variation campaign
  let campaignHtml = '';
  if (s.is_campaign_root) {
    const statusesRunning = ['filtering','implementing','implemented','quick_testing'];
    const allDone = !statusesRunning.includes(s.status); // rough check
    campaignHtml = `
    <div class="panel-section" style="background:#0c1220;border:2px solid #1e3a5f;
                border-radius:12px;padding:18px 20px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
        <div>
          <span style="font-size:.72rem;font-weight:600;text-transform:uppercase;
                       letter-spacing:.07em;color:#38bdf8">🔬 Strategy Campaign</span>
          <div style="font-size:.82rem;color:#94a3b8;margin-top:3px">
            Multiple implementations tested in parallel — pick the best performer.
          </div>
        </div>
        <button onclick="loadCampaignTable('${s.id}')"
          style="background:#1e2533;border:1px solid #374151;border-radius:8px;
                 color:#64748b;padding:5px 12px;font-size:.78rem;cursor:pointer">
          ↻ Refresh
        </button>
      </div>

      <div id="campaign-table-${s.id}">
        <div style="color:#64748b;font-size:.8rem;padding:8px 0">Loading variations…</div>
      </div>

      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;padding-top:14px;
                  border-top:1px solid #1f2937">
        <button onclick="approveTopCampaign('${s.id}', 3)"
          id="campaign-top3-${s.id}"
          style="background:#4f46e5;border:none;border-radius:8px;color:#fff;
                 padding:9px 20px;font-size:.85rem;font-weight:700;cursor:pointer">
          ✓ Optimize top 3
        </button>
        <button onclick="approveTopCampaign('${s.id}', 1)"
          id="campaign-top1-${s.id}"
          style="background:#059669;border:none;border-radius:8px;color:#fff;
                 padding:9px 20px;font-size:.85rem;font-weight:700;cursor:pointer">
          ✓ Optimize best only
        </button>
        <span style="font-size:.75rem;color:#374151;align-self:center">
          Approves for full Optuna optimization (~15–25 min each)
        </span>
      </div>
      <div id="campaign-result-${s.id}" style="margin-top:8px;font-size:.8rem;color:#64748b"></div>
    </div>`;

    // Load async after render
    setTimeout(() => loadCampaignTable('${s.id}'), 80);
  }

  // Review panel for awaiting_review strategies
  let reviewHtml = '';
  if (s.status === 'awaiting_review') {
    // Build a concise "what was implemented" summary from indicators JSONB
    let implSummary = '';
    const ind = s.indicators || {};
    const indicatorsUsed = (ind.indicators_used || []).join(', ') || '—';
    const symbolUsed     = (ind.symbols || [])[0] || s.symbol || 'EURUSD';
    const bestTfLabel    = s.best_timeframe || '—';
    implSummary = `
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px">
        <div style="background:#0f1623;border-radius:8px;padding:8px 12px;font-size:.8rem">
          <span style="color:#475569">Class: </span><span style="color:#7dd3fc">${esc(ind.strategy_class||'—')}</span>
        </div>
        <div style="background:#0f1623;border-radius:8px;padding:8px 12px;font-size:.8rem">
          <span style="color:#475569">Indicators: </span><span style="color:#7dd3fc">${esc(indicatorsUsed)}</span>
        </div>
        <div style="background:#0f1623;border-radius:8px;padding:8px 12px;font-size:.8rem">
          <span style="color:#475569">Symbol: </span><span style="color:#7dd3fc">${esc(symbolUsed)}</span>
        </div>
        <div style="background:#0f1623;border-radius:8px;padding:8px 12px;font-size:.8rem">
          <span style="color:#475569">Best TF: </span><span style="color:#38bdf8;font-weight:600">${esc(bestTfLabel)}</span>
        </div>
      </div>`;

    // Analyzer findings summary
    let analyzerSummary = '';
    if (s.analysis_notes && s.analysis_notes.llm) {
      const llm = s.analysis_notes.llm;
      if (llm.key_finding) {
        analyzerSummary = `
        <div style="background:#0c1a2e;border:1px solid #1e3a5f;border-radius:8px;
                    padding:10px 14px;margin-bottom:12px;font-size:.82rem;color:#93c5fd;line-height:1.5">
          <span style="font-size:.68rem;font-weight:600;text-transform:uppercase;
                       letter-spacing:.06em;color:#475569;display:block;margin-bottom:4px">Analyzer finding</span>
          ${esc(llm.key_finding)}
          ${llm.improvement_type && llm.improvement_type !== 'none'
            ? `<span style="background:#172554;border-radius:4px;padding:1px 7px;font-size:.7rem;
                            margin-left:8px;color:#38bdf8">${esc(llm.improvement_type)}</span>` : ''}
        </div>`;
      }
    }

    reviewHtml = `
    <div class="panel-section" style="background:#120f1f;border:2px solid #4f46e5;
                border-radius:10px;padding:16px 18px">
      <div style="font-size:.72rem;font-weight:600;text-transform:uppercase;
                  letter-spacing:.07em;color:#818cf8;margin-bottom:12px">
        👁 Strategy Review — Your Approval Required
      </div>
      ${implSummary}
      ${analyzerSummary}
      <div style="color:#94a3b8;font-size:.82rem;margin-bottom:14px">
        Review the quick test results above. Approve to start full optimization
        (Optuna 50 trials + walk-forward + OOS), or request a change to the code.
      </div>

      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px">
        <button onclick="approveStrategy('${s.id}')"
          id="approve-btn-${s.id}"
          style="background:#4f46e5;border:none;border-radius:8px;color:#fff;
                 padding:9px 22px;font-size:.9rem;font-weight:700;cursor:pointer">
          ✓ Approve &amp; Optimize
        </button>
        <span style="font-size:.78rem;color:#4b5563;align-self:center">
          ~15–25 min on Modal
        </span>
      </div>

      <div style="border-top:1px solid #1f2937;padding-top:14px">
        <div style="font-size:.72rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:.07em;color:#64748b;margin-bottom:8px">
          Request a Change
        </div>
        <div style="color:#64748b;font-size:.78rem;margin-bottom:8px">
          Examples: "add a volume filter", "limit to London session 8–17 UTC",
          "replace EMA with VWAP", "tighten stop loss to 1× ATR"
        </div>
        <textarea id="revise-input-${s.id}" rows="3"
          placeholder="Describe what you'd like changed…"
          style="width:100%;background:#0f1623;border:1px solid #374151;border-radius:8px;
                 color:#f1f5f9;padding:8px 12px;font-size:.85rem;outline:none;
                 resize:vertical;font-family:inherit;margin-bottom:8px"></textarea>
        <div style="display:flex;gap:10px;align-items:center">
          <button onclick="reviseStrategy('${s.id}')"
            id="revise-btn-${s.id}"
            style="background:#1e2533;border:1px solid #374151;border-radius:8px;
                   color:#94a3b8;padding:8px 18px;font-size:.85rem;font-weight:600;cursor:pointer">
            ↺ Request Changes &amp; Re-test
          </button>
          <span style="font-size:.75rem;color:#374151">Will re-run quick test after applying changes</span>
        </div>
      </div>
      <div id="review-result-${s.id}" style="margin-top:10px;font-size:.8rem;color:#64748b"></div>
    </div>`;
  }

  // "Run Full Optimization" action for quick_tested strategies (manual override)
  let quickOptimizeHtml = '';
  if (s.status === 'quick_tested') {
    quickOptimizeHtml = `
    <div class="panel-section" style="background:#0d1f16;border:1px solid #166534;
                border-radius:10px;padding:14px 16px">
      <div style="font-size:.72rem;font-weight:600;text-transform:uppercase;
                  letter-spacing:.07em;color:#4ade80;margin-bottom:10px">Ready for Full Optimization</div>
      <div style="color:#94a3b8;font-size:.82rem;margin-bottom:12px">
        Quick test complete — waiting for analyzer. Or start immediately:
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <button onclick="runFullOptimization('${s.id}')"
          style="background:#16a34a;border:none;border-radius:8px;color:#fff;
                 padding:9px 20px;font-size:.85rem;font-weight:600;cursor:pointer"
          id="optimize-btn-${s.id}">
          ▶ Run Full Optimization Now
        </button>
      </div>
      <div id="optimize-result-${s.id}" style="margin-top:8px;font-size:.8rem;color:#64748b"></div>
    </div>`;
  }

  // Modal job status block for in-progress strategies
  let modalHtml = '';
  const IN_PROGRESS = ['implemented','quick_testing','backtesting','validating'];
  if (IN_PROGRESS.includes(s.status)) {
    const elapsed = s.updated_at ? elapsedSince(s.updated_at) : '?';
    modalHtml = `
    <div class="panel-section" style="background:#0d1a2d;border:1px solid #1e3a5f;
                border-radius:10px;padding:14px 16px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
        <span style="font-size:.72rem;font-weight:600;text-transform:uppercase;
                     letter-spacing:.07em;color:#475569">Modal Job</span>
        <span style="font-size:.75rem;color:#64748b">Running for ${elapsed}</span>
      </div>
      ${s.modal_job_id ? `<div style="font-size:.72rem;color:#374151;margin-bottom:10px;
            font-family:monospace;word-break:break-all">${s.modal_job_id}</div>` : ''}
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button onclick="checkModalStatus('${s.id}')"
          style="background:#1e2533;border:1px solid #374151;border-radius:8px;
                 color:#93c5fd;padding:6px 14px;font-size:.8rem;cursor:pointer"
          id="modal-check-btn-${s.id}">
          ↻ Check status
        </button>
        <button onclick="restartStrategy('${s.id}')"
          style="background:#7c3aed;border:none;border-radius:8px;
                 color:#fff;padding:6px 14px;font-size:.8rem;font-weight:600;cursor:pointer"
          id="modal-restart-btn-${s.id}">
          ⟳ Restart job
        </button>
        <a href="https://modal.com/apps" target="_blank"
          style="background:#1e2533;border:1px solid #374151;border-radius:8px;
                 color:#64748b;padding:6px 14px;font-size:.8rem;text-decoration:none;
                 display:inline-block">
          ↗ Modal Dashboard
        </a>
      </div>
      <div id="modal-status-result-${s.id}" style="margin-top:10px;font-size:.8rem;color:#64748b"></div>
    </div>`;
  }

  let actionsHtml = '';
  if (s.status === 'failed') {
    const retryLabel = s.backtest_code ? 'Retry from Modal dispatch'
                     : s.pre_filter_score != null ? 'Retry from Implementer'
                     : 'Retry from Pre-filter';
    actionsHtml = `
    <div class="panel-section" id="actions-${s.id}">
      <h3>Actions</h3>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px">
        <button onclick="retryStrategy('${s.id}')"
          style="background:#6366f1;border:none;border-radius:8px;color:#fff;
                 padding:9px 18px;font-size:.85rem;font-weight:600;cursor:pointer">
          ↺ ${retryLabel}
        </button>
        <button onclick="toggleEditForm('${s.id}')"
          style="background:#1e2533;border:1px solid #374151;border-radius:8px;color:#94a3b8;
                 padding:9px 18px;font-size:.85rem;font-weight:600;cursor:pointer">
          ✎ Edit &amp; Retry from scratch
        </button>
      </div>
      <div id="edit-form-${s.id}" style="display:none">
        <div style="margin-bottom:12px">
          <label style="display:block;font-size:.72rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:.06em;color:#64748b;margin-bottom:6px">Strategy Name</label>
          <input id="edit-name-${s.id}" type="text" value="${esc(s.name)}"
            style="width:100%;background:#0f1623;border:1px solid #374151;border-radius:8px;
                   color:#f1f5f9;padding:8px 12px;font-size:.9rem;outline:none">
        </div>
        <div style="margin-bottom:12px">
          <label style="display:block;font-size:.72rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:.06em;color:#64748b;margin-bottom:6px">Description / Hypothesis</label>
          <textarea id="edit-hyp-${s.id}" rows="5"
            style="width:100%;background:#0f1623;border:1px solid #374151;border-radius:8px;
                   color:#f1f5f9;padding:8px 12px;font-size:.9rem;outline:none;
                   resize:vertical;font-family:inherit">${esc(s.hypothesis || '')}</textarea>
        </div>
        <button onclick="saveAndRetry('${s.id}')"
          style="background:#059669;border:none;border-radius:8px;color:#fff;
                 padding:9px 18px;font-size:.85rem;font-weight:600;cursor:pointer">
          Save &amp; Retry from scratch
        </button>
      </div>
    </div>`;
  }

  let reportHtml = '';
  if (s.report_text) {
    const isHtmlReport = s.report_text.trimStart().startsWith('<!') ||
                         s.report_text.trimStart().toLowerCase().startsWith('<html');
    if (isHtmlReport) {
      // Bokeh interactive equity-curve report — render in sandboxed iframe via blob URL
      const blobUrl = URL.createObjectURL(
        new Blob([s.report_text], {type: 'text/html'})
      );
      reportHtml = `<div class="panel-section">
        <h3>Equity Curve Report
          <a href="${blobUrl}" target="_blank"
            style="margin-left:10px;background:#1e2533;border:1px solid #374151;border-radius:6px;
                   color:#93c5fd;padding:3px 10px;font-size:.72rem;text-decoration:none">
            ↗ Fullscreen
          </a>
        </h3>
        <iframe src="${blobUrl}"
          style="width:100%;height:480px;border:none;border-radius:8px;background:#fff">
        </iframe>
        ${s.report_url ? `<a href="${s.report_url}" target="_blank"
          style="display:inline-block;margin-top:6px;color:#818cf8;font-size:.8rem;">↗ Raw file (R2)</a>` : ''}
      </div>`;
    } else {
      // Plain markdown report (from summariser agent after full validation)
      const md = esc(s.report_text)
        .replace(/^#{3} (.+)$/gm, '<strong style="color:#94a3b8;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em">$1</strong>')
        .replace(/^#{1,2} (.+)$/gm, '<strong style="color:#f1f5f9;font-size:.95rem">$1</strong>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
      reportHtml = `<div class="panel-section">
        <h3>Report</h3>
        <div style="background:#0f1623;border-radius:8px;padding:16px;font-size:.82rem;
                    line-height:1.65;color:#cbd5e1;max-height:360px;overflow-y:auto">${md}</div>
        ${s.report_url ? `<a href="${s.report_url}" target="_blank"
          style="display:inline-block;margin-top:8px;color:#818cf8;font-size:.8rem;">↗ Raw file (R2)</a>` : ''}
      </div>`;
    }
  } else if (s.report_url) {
    reportHtml = `<div class="panel-section">
      <a href="${s.report_url}" target="_blank"
         style="color:#818cf8;font-size:.85rem;">↗ View full report (R2)</a></div>`;
  }

  document.getElementById('panel-content').innerHTML = `
    <h2>${esc(s.name)}</h2>
    <div style="margin-bottom:12px"><span class="badge ${badgeCls}">${statusLabel(s.status)}</span>
      <span style="color:#475569;font-size:.78rem;margin-left:8px">
        ${s.source || 'user'} · ${(s.created_at||'').slice(0,10)}</span></div>
    ${s.entry_logic && s.entry_logic !== s.hypothesis ? `
    <div style="margin-bottom:14px">
      <div style="font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#475569;margin-bottom:5px">Original Idea</div>
      <div style="background:#0a0f1a;border-left:3px solid #374151;border-radius:0 8px 8px 0;
                  padding:10px 14px;font-size:.82rem;color:#64748b;line-height:1.55;white-space:pre-wrap">${esc(s.entry_logic)}</div>
    </div>` : ''}
    <div style="font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#475569;margin-bottom:5px">
      ${s.entry_logic && s.entry_logic !== s.hypothesis ? 'Refined Description' : 'Description'}
    </div>
    <p class="panel-hyp">${esc(s.hypothesis || '')}</p>

    <div class="panel-section">
      <h3>In-Sample Results</h3>
      <div class="kv-grid">
        <div class="kv"><div class="kv-label">Sharpe</div><div class="kv-val ${trainCls}">${trainSharpe}</div></div>
        <div class="kv"><div class="kv-label">Calmar</div><div class="kv-val">${fmtNum(s.backtest_calmar)}</div></div>
        <div class="kv"><div class="kv-label">Max Drawdown</div><div class="kv-val">${dd}</div></div>
        <div class="kv"><div class="kv-label">Win Rate</div><div class="kv-val">${wr}</div></div>
        <div class="kv"><div class="kv-label">Total Trades</div><div class="kv-val">${s.total_signals ?? '—'}</div></div>
        <div class="kv"><div class="kv-label">Signals/Year</div><div class="kv-val">${s.signals_per_year ? Math.round(s.signals_per_year) : '—'}</div></div>
        <div class="kv"><div class="kv-label">Profit Factor</div><div class="kv-val">${fmtNum(s.profit_factor)}</div></div>
        <div class="kv"><div class="kv-label">Leakage Score</div>
          <div class="kv-val ${s.leakage_score>=7?'good':s.leakage_score>=4?'':'bad'}">${s.leakage_score != null ? s.leakage_score.toFixed(1)+'/10' : '—'}</div></div>
      </div>
    </div>

    <div class="panel-section">
      <h3>OOS Results (2026+)</h3>
      <div class="kv-grid">
        <div class="kv"><div class="kv-label">OOS Sharpe</div><div class="kv-val ${oosCls}">${oosSharpe}</div></div>
        <div class="kv"><div class="kv-label">OOS Win Rate</div><div class="kv-val">${s.oos_win_rate ? (s.oos_win_rate*100).toFixed(1)+'%' : '—'}</div></div>
        <div class="kv"><div class="kv-label">OOS Trades</div><div class="kv-val">${s.oos_total_trades ?? '—'}</div></div>
        <div class="kv"><div class="kv-label">Monte Carlo p</div><div class="kv-val ${mcCls}">${mc}</div></div>
      </div>
    </div>

    ${campaignHtml}${reviewHtml}${quickOptimizeHtml}${modalHtml}${actionsHtml}${quickTestHtml}${wfHtml}${paramsHtml}${codeHtml}${errHtml}${reportHtml}
    <div class="panel-section" style="border-top:1px solid #1f2937;padding-top:16px;margin-top:4px">
      <h3>Comments</h3>
      <div id="comments-list-${s.id}" style="margin-bottom:10px">
        ${renderComments(s.comments)}
      </div>
      <div style="display:flex;gap:8px;align-items:flex-start">
        <textarea id="comment-input-${s.id}" rows="2" placeholder="Add a comment…"
          style="flex:1;background:#0f1623;border:1px solid #374151;border-radius:8px;
                 color:#f1f5f9;padding:7px 12px;font-size:.85rem;outline:none;
                 resize:none;font-family:inherit"></textarea>
        <button onclick="addComment('${s.id}')"
          style="background:#1e2533;border:1px solid #374151;border-radius:8px;
                 color:#94a3b8;padding:7px 14px;font-size:.82rem;cursor:pointer;white-space:nowrap;align-self:flex-start">
          Add
        </button>
      </div>
    </div>
    <div class="panel-section" style="border-top:1px solid #1f2937;padding-top:16px;margin-top:4px">
      <h3>Tags</h3>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px" id="tag-display-${s.id}">
        ${(s.tags||[]).map(t=>`<span class="tag-pill">${esc(t)}</span>`).join('')}
        ${!(s.tags||[]).length ? '<span style="color:#475569;font-size:.8rem">No tags yet</span>' : ''}
      </div>
      <div style="display:flex;gap:8px">
        <input id="tag-input-${s.id}" type="text" placeholder="Add tags (comma separated)"
          value="${esc((s.tags||[]).join(', '))}"
          style="flex:1;background:#0f1623;border:1px solid #374151;border-radius:8px;
                 color:#f1f5f9;padding:7px 12px;font-size:.85rem;outline:none">
        <button onclick="saveTags('${s.id}')"
          style="background:#1e2533;border:1px solid #374151;border-radius:8px;
                 color:#94a3b8;padding:7px 14px;font-size:.82rem;cursor:pointer;white-space:nowrap">
          Save tags
        </button>
      </div>
    </div>
    <div style="padding-top:12px;text-align:right">
      <button onclick="deleteStrategy('${s.id}', '${esc(s.name)}')"
        style="background:none;border:1px solid #7f1d1d;border-radius:8px;color:#f87171;
               padding:7px 16px;font-size:.82rem;cursor:pointer">
        Delete strategy
      </button>
    </div>`;
}

function elapsedSince(ts) {
  if (!ts) return '?';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return '?';
  const diff = Math.floor((Date.now() - d.getTime()) / 1000);
  if (diff < 60) return diff + 's';
  if (diff < 3600) return Math.floor(diff/60) + 'm ' + (diff%60) + 's';
  return Math.floor(diff/3600) + 'h ' + Math.floor((diff%3600)/60) + 'm';
}

async function checkModalStatus(id) {
  const btn = document.getElementById(`modal-check-btn-${id}`);
  const out = document.getElementById(`modal-status-result-${id}`);
  btn.textContent = 'Checking…';
  btn.disabled = true;
  try {
    const r = await fetch(`/api/strategy/${id}/modal-status`);
    const d = await r.json();
    if (d.status === 'running') {
      out.style.color = '#fcd34d';
      out.innerHTML = `⚙ Job running — ${d.age_minutes ?? '?'} min elapsed.`;
    } else if (d.status === 'done') {
      out.style.color = '#4ade80';
      out.textContent = '✓ Job finished. Refreshing…';
      setTimeout(() => { loadAll(); openPanel(id); }, 1500);
    } else if (d.status === 'stuck') {
      out.style.color = '#f87171';
      out.innerHTML = `⚠ Job stuck (${d.age_minutes} min). Likely timed out without updating DB. `
        + `<a href="#" style="color:#f87171;text-decoration:underline"
             onclick="restartStrategy('${id}');return false">Force restart</a>`;
    } else if (d.status === 'no_job') {
      out.style.color = '#64748b';
      out.textContent = 'No job ID — job may still be starting.';
    } else {
      out.style.color = '#f87171';
      out.textContent = '? ' + (d.error || d.message || d.status);
    }
  } catch(e) {
    out.style.color = '#f87171';
    out.textContent = 'Network error.';
  }
  btn.textContent = '↻ Check status';
  btn.disabled = false;
}

function renderComments(comments) {
  if (!comments || !comments.length) {
    return '<span style="color:#475569;font-size:.8rem">No comments yet</span>';
  }
  const list = typeof comments === 'string' ? JSON.parse(comments) : comments;
  return list.map(c => {
    const isPipeline = (c.text || '').startsWith('[pipeline]');
    const text = isPipeline ? esc(c.text.slice('[pipeline]'.length).trim()) : esc(c.text);
    const bg    = isPipeline ? '#0d1a2d' : '#0f1623';
    const border = isPipeline ? 'border-left:3px solid #334155' : 'border-left:3px solid #6366f1';
    const label = isPipeline
      ? `<span style="font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#475569;margin-bottom:5px;display:block">Pipeline</span>`
      : `<span style="font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#818cf8;margin-bottom:5px;display:block">You</span>`;
    return `
    <div style="background:${bg};border-radius:8px;padding:10px 14px;margin-bottom:8px;${border}">
      ${label}
      <div style="font-size:.82rem;color:${isPipeline ? '#94a3b8' : '#e2e8f0'};white-space:pre-wrap;word-break:break-word;line-height:1.55">${text}</div>
      <div style="font-size:.7rem;color:#374151;margin-top:5px">${c.ts ? c.ts.replace('T',' ') : ''}</div>
    </div>`;
  }).join('');
}

async function addComment(id) {
  const input = document.getElementById(`comment-input-${id}`);
  const text = input.value.trim();
  if (!text) return;
  const r = await fetch(`/api/strategy/${id}/comment`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text}),
  });
  const data = await r.json();
  if (data.ok) {
    input.value = '';
    document.getElementById(`comments-list-${id}`).innerHTML = renderComments(data.comments);
  } else {
    alert('Error: ' + (data.error || 'unknown'));
  }
}

function fmtNum(v) { return v != null ? parseFloat(v).toFixed(3) : '—'; }
function esc(s) {
  if (!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function saveTags(id) {
  const raw = document.getElementById(`tag-input-${id}`).value;
  const tags = raw.split(',').map(t=>t.trim()).filter(Boolean);
  const r = await fetch(`/api/strategy/${id}/tags`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({tags}),
  });
  const data = await r.json();
  if (data.ok) {
    const disp = document.getElementById(`tag-display-${id}`);
    disp.innerHTML = tags.length
      ? tags.map(t=>`<span class="tag-pill">${esc(t)}</span>`).join('')
      : '<span style="color:#475569;font-size:.8rem">No tags yet</span>';
    loadAll(); // refresh table
  }
}

async function loadCampaignTable(rootId) {
  const container = document.getElementById(`campaign-table-${rootId}`);
  if (!container) return;
  container.innerHTML = '<div style="color:#64748b;font-size:.8rem">Loading…</div>';
  try {
    const r = await fetch(`/api/strategy/${rootId}/campaign`);
    const d = await r.json();
    const variations = d.variations || [];
    if (!variations.length) {
      container.innerHTML = '<span style="color:#64748b;font-size:.8rem">No variations found yet.</span>';
      return;
    }
    const thStyle = 'padding:5px 10px;text-align:left;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#475569;border-bottom:1px solid #1e293b';
    const tdStyle = 'padding:6px 10px;border-bottom:1px solid #0f1623;font-size:.8rem;vertical-align:middle';
    const rows = variations.map(v => {
      const sh = v.quick_test_sharpe != null
        ? `<span style="${v.quick_test_sharpe>0.3?'color:#34d399':v.quick_test_sharpe>0?'color:#94a3b8':'color:#f87171'}">${v.quick_test_sharpe>=0?'+':''}${v.quick_test_sharpe.toFixed(3)}</span>`
        : '<span style="color:#374151">—</span>';
      const tr = v.quick_test_trades ?? '—';
      const wr = v.quick_test_win_rate != null ? (v.quick_test_win_rate*100).toFixed(0)+'%' : '—';
      const tf = v.best_timeframe || '—';
      const seedBadge = v.is_root ? ' <span style="background:#1e3a5f;color:#38bdf8;border-radius:4px;padding:1px 6px;font-size:.66rem">seed</span>' : '';
      const statusBadge = `<span class="badge s-${(v.status||'').replace(/_/g,'-')}" style="font-size:.66rem">${statusLabel(v.status||'')}</span>`;
      const passGate = (v.quick_test_sharpe||0) > 0 && (v.quick_test_trades||0) >= 30;
      const rowBg = passGate ? 'background:#0d1f16' : '';
      const actionBtn = (v.status === 'quick_tested' || v.status === 'awaiting_review') && passGate
        ? `<button onclick="event.stopPropagation();approveVariationFromCampaign('${v.id}','${rootId}')"
             style="background:#16a34a;border:none;border-radius:6px;color:#fff;
                    padding:3px 10px;font-size:.75rem;cursor:pointer;white-space:nowrap">
             Optimize
           </button>` : '';
      return `<tr onclick="openPanel('${v.id}')" style="cursor:pointer;${rowBg}">
        <td style="${tdStyle}">${esc(v.name)}${seedBadge}</td>
        <td style="${tdStyle}">${statusBadge}</td>
        <td style="${tdStyle}">${sh}</td>
        <td style="${tdStyle}">${tr}</td>
        <td style="${tdStyle}">${wr}</td>
        <td style="${tdStyle}">${tf}</td>
        <td style="${tdStyle}">${actionBtn}</td>
      </tr>`;
    }).join('');
    container.innerHTML = `
      <div style="font-size:.75rem;color:#475569;margin-bottom:8px">
        ${d.total} variations · <span style="color:#4ade80">${d.passed} passed quality gate</span>
        · click any row to open details
      </div>
      <table style="width:100%;border-collapse:collapse">
        <thead><tr>
          <th style="${thStyle}">Variation</th>
          <th style="${thStyle}">Status</th>
          <th style="${thStyle}">Sharpe</th>
          <th style="${thStyle}">Trades</th>
          <th style="${thStyle}">Win%</th>
          <th style="${thStyle}">Best TF</th>
          <th style="${thStyle}"></th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  } catch(e) {
    container.innerHTML = `<span style="color:#f87171;font-size:.8rem">Failed to load: ${e.message}</span>`;
  }
}

async function approveTopCampaign(rootId, topN) {
  const btn1 = document.getElementById(`campaign-top1-${rootId}`);
  const btn3 = document.getElementById(`campaign-top3-${rootId}`);
  const out  = document.getElementById(`campaign-result-${rootId}`);
  if (btn1) btn1.disabled = true;
  if (btn3) btn3.disabled = true;
  out.textContent = 'Fetching top variations…';
  try {
    const r = await fetch(`/api/strategy/${rootId}/campaign`);
    const d = await r.json();
    const eligible = (d.variations || [])
      .filter(v => (v.status === 'quick_tested' || v.status === 'awaiting_review')
               && (v.quick_test_sharpe || 0) > 0
               && (v.quick_test_trades || 0) >= 30)
      .slice(0, topN);
    if (!eligible.length) {
      out.style.color = '#f87171';
      out.textContent = 'No eligible variations (need Sharpe > 0 and trades ≥ 30).';
      if (btn1) btn1.disabled = false;
      if (btn3) btn3.disabled = false;
      return;
    }
    let approved = 0;
    for (const v of eligible) {
      const ar = await fetch(`/api/strategy/${v.id}/approve`, {method: 'POST'});
      const ad = await ar.json();
      if (ad.ok) approved++;
    }
    out.style.color = '#4ade80';
    out.textContent = `✓ Approved ${approved} variation${approved>1?'s':''} for full optimization.`;
    setTimeout(() => { loadAll(); openPanel(rootId); }, 2000);
  } catch(e) {
    out.style.color = '#f87171';
    out.textContent = 'Error: ' + e.message;
    if (btn1) btn1.disabled = false;
    if (btn3) btn3.disabled = false;
  }
}

async function approveVariationFromCampaign(variationId, rootId) {
  const r = await fetch(`/api/strategy/${variationId}/approve`, {method: 'POST'});
  const d = await r.json();
  if (d.ok) {
    loadAll();
    loadCampaignTable(rootId);
  } else {
    alert('Approve failed: ' + (d.error || 'unknown'));
  }
}

async function deleteStrategy(id, name) {
  if (!confirm(`Delete "${name}"?\nThis cannot be undone.`)) return;
  const r = await fetch(`/api/strategy/${id}`, {method:'DELETE'});
  const data = await r.json();
  if (data.ok) { closePanel(); loadAll(); }
  else alert('Delete failed: ' + (data.error || 'unknown error'));
}

async function approveStrategy(id) {
  const btn = document.getElementById(`approve-btn-${id}`);
  const out = document.getElementById(`review-result-${id}`);
  btn.disabled = true;
  btn.textContent = 'Approving…';
  try {
    const r = await fetch(`/api/strategy/${id}/approve`, {method: 'POST'});
    const d = await r.json();
    if (d.ok) {
      out.style.color = '#4ade80';
      out.textContent = '✓ Approved — full optimization dispatched. Refreshing…';
      setTimeout(() => { loadAll(); openPanel(id); }, 2000);
    } else {
      out.style.color = '#f87171';
      out.textContent = '✗ ' + (d.error || 'Error');
      btn.disabled = false;
      btn.textContent = '✓ Approve & Optimize';
    }
  } catch(e) {
    out.style.color = '#f87171';
    out.textContent = 'Network error.';
    btn.disabled = false;
    btn.textContent = '✓ Approve & Optimize';
  }
}

async function reviseStrategy(id) {
  const input = document.getElementById(`revise-input-${id}`);
  const btn   = document.getElementById(`revise-btn-${id}`);
  const out   = document.getElementById(`review-result-${id}`);
  const message = input.value.trim();
  if (!message) { out.style.color = '#f87171'; out.textContent = 'Please describe what to change.'; return; }
  btn.disabled = true;
  btn.textContent = 'Applying changes…';
  try {
    const r = await fetch(`/api/strategy/${id}/revise`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message}),
    });
    const d = await r.json();
    if (d.ok) {
      out.style.color = '#fbbf24';
      out.textContent = '↺ ' + d.message + ' Panel will update when done.';
      input.value = '';
      // Poll for status change back to awaiting_review
      let polls = 0;
      const poll = setInterval(async () => {
        polls++;
        const pr = await fetch('/api/strategy/' + id);
        const ps = await pr.json();
        if (ps.status === 'awaiting_review' || polls > 30) {
          clearInterval(poll);
          loadAll();
          openPanel(id);
        }
      }, 5000);
    } else {
      out.style.color = '#f87171';
      out.textContent = '✗ ' + (d.error || 'Error');
    }
  } catch(e) {
    out.style.color = '#f87171';
    out.textContent = 'Network error.';
  }
  btn.disabled = false;
  btn.textContent = '↺ Request Changes & Re-test';
}

async function runFullOptimization(id) {
  const btn = document.getElementById(`optimize-btn-${id}`);
  const out = document.getElementById(`optimize-result-${id}`);
  btn.disabled = true;
  btn.textContent = 'Dispatching…';
  try {
    const r = await fetch(`/api/strategy/${id}/restart`, {method: 'POST'});
    const d = await r.json();
    if (d.ok) {
      out.style.color = '#4ade80';
      out.textContent = '✓ Dispatched to full optimization. Refreshing…';
      setTimeout(() => { loadAll(); openPanel(id); }, 2000);
    } else {
      out.style.color = '#f87171';
      out.textContent = '✗ ' + (d.error || 'Error');
      btn.disabled = false;
      btn.textContent = '▶ Run Full Optimization';
    }
  } catch(e) {
    out.style.color = '#f87171';
    out.textContent = 'Network error.';
    btn.disabled = false;
    btn.textContent = '▶ Run Full Optimization';
  }
}

async function restartStrategy(id) {
  const btn = document.getElementById(`modal-restart-btn-${id}`);
  const out = document.getElementById(`modal-status-result-${id}`);
  btn.disabled = true;
  btn.textContent = 'Restarting…';
  try {
    const r = await fetch(`/api/strategy/${id}/restart`, {method: 'POST'});
    const d = await r.json();
    if (d.ok) {
      out.style.color = '#a78bfa';
      out.textContent = `⟳ Redispatched to ${d.dispatched_to} job. Refreshing…`;
      setTimeout(() => { loadAll(); openPanel(id); }, 2000);
    } else {
      out.style.color = '#f87171';
      out.textContent = '✗ ' + (d.error || 'Error');
      btn.disabled = false;
      btn.textContent = '⟳ Restart job';
    }
  } catch(e) {
    out.style.color = '#f87171';
    out.textContent = 'Network error.';
    btn.disabled = false;
    btn.textContent = '⟳ Restart job';
  }
}

async function retryStrategy(id) {
  const btn = event.target;
  btn.disabled = true;
  btn.textContent = 'Retrying…';
  try {
    const r = await fetch(`/api/strategy/${id}/retry`, {method: 'POST'});
    const data = await r.json();
    if (data.ok) {
      btn.textContent = '✓ Queued — refreshing…';
      setTimeout(() => { closePanel(); loadAll(); }, 1200);
    } else {
      btn.textContent = '✗ ' + (data.error || 'Error');
      btn.disabled = false;
    }
  } catch(e) {
    btn.textContent = '✗ Network error';
    btn.disabled = false;
  }
}

function toggleEditForm(id) {
  const el = document.getElementById(`edit-form-${id}`);
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

async function saveAndRetry(id) {
  const name = document.getElementById(`edit-name-${id}`).value;
  const hyp  = document.getElementById(`edit-hyp-${id}`).value;
  const btn  = event.target;
  btn.disabled = true;
  btn.textContent = 'Saving…';
  try {
    const r = await fetch(`/api/strategy/${id}/update`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, hypothesis: hyp}),
    });
    const data = await r.json();
    if (data.ok) {
      btn.textContent = '✓ Saved — refreshing…';
      setTimeout(() => { closePanel(); loadAll(); }, 1200);
    } else {
      btn.textContent = '✗ ' + (data.error || 'Error');
      btn.disabled = false;
    }
  } catch(e) {
    btn.textContent = '✗ Network error';
    btn.disabled = false;
  }
}

// Close panel on Escape
document.addEventListener('keydown', e => { if (e.key === 'Escape') closePanel(); });

// Auto-refresh every 30s
loadAll();
setInterval(loadAll, 30000);
</script>
</body>
</html>"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(_DASHBOARD_HTML)


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    """Redirect root to dashboard."""
    return HTMLResponse('<meta http-equiv="refresh" content="0;url=/dashboard">')


# ---------------------------------------------------------------------------
# Idea submission UI
# ---------------------------------------------------------------------------

_IDEAS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ideas — Trading Research</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e2e8f0; margin: 0; padding: 0; min-height: 100vh; }
  .nav { display: flex; align-items: center; gap: 0; background: #111827;
         border-bottom: 1px solid #1f2937; padding: 0 24px; }
  .nav-logo { font-weight: 700; font-size: 0.95rem; color: #f8fafc;
              padding: 14px 0; margin-right: 32px; letter-spacing: -.01em; }
  .nav a { display: block; padding: 14px 16px; font-size: 0.85rem; color: #94a3b8;
           text-decoration: none; border-bottom: 2px solid transparent; transition: color .15s; }
  .nav a:hover, .nav a.active { color: #f1f5f9; border-bottom-color: #6366f1; }
  .page { padding: 40px 24px; max-width: 720px; margin: 0 auto; }
  h1 { font-size: 1.5rem; font-weight: 600; margin: 0 0 6px; color: #f8fafc; }
  .subtitle { color: #64748b; font-size: 0.9rem; margin: 0 0 32px; line-height: 1.6; }
  .card { background: #1e2533; border: 1px solid #2d3748;
          border-radius: 12px; padding: 28px; margin-bottom: 36px; }
  textarea {
    width: 100%; background: #0f1117; border: 1px solid #374151;
    border-radius: 8px; color: #f1f5f9; padding: 14px;
    font-size: 1rem; outline: none; transition: border-color .15s;
    resize: vertical; min-height: 130px; font-family: inherit; line-height: 1.6;
  }
  textarea:focus { border-color: #6366f1; }
  .hint { font-size: .78rem; color: #475569; margin: 10px 0 20px; }
  button[type=submit] {
    background: #6366f1; color: #fff; border: none; border-radius: 8px;
    padding: 12px 28px; font-size: .95rem; font-weight: 600; cursor: pointer;
    transition: background .15s;
  }
  button[type=submit]:hover { background: #4f46e5; }
  .flash { margin-bottom: 24px; padding: 14px 18px; border-radius: 8px;
           font-size: 0.9rem; font-weight: 500; }
  .flash.success { background: #14532d; border: 1px solid #16a34a; color: #bbf7d0; }
  .flash.error   { background: #450a0a; border: 1px solid #dc2626; color: #fecaca; }
  .feed-hdr { font-size: .75rem; font-weight: 600; color: #475569;
              text-transform: uppercase; letter-spacing: .07em; margin: 0 0 14px; }
  .feed-row { background: #1e2533; border: 1px solid #2d3748;
              border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }
  .feed-title { font-weight: 600; color: #f1f5f9; font-size: .95rem; }
  .feed-desc  { color: #94a3b8; font-size: .82rem; margin: 4px 0; }
  .feed-meta  { font-size: .75rem; color: #475569; }
  .badge { display: inline-block; padding: 2px 9px; border-radius: 99px;
           font-size: .72rem; font-weight: 600; margin-left: 8px; vertical-align: middle; }
  .badge-pending   { background: #1e3a5f; color: #93c5fd; }
  .badge-running   { background: #1c3352; color: #67e8f9; }
  .badge-done      { background: #14532d; color: #86efac; }
  .badge-failed    { background: #450a0a; color: #fca5a5; }
  .badge-picked_up { background: #1c3352; color: #67e8f9; }
  .type-strategy { color: #818cf8; font-size: .72rem; font-weight: 600; }
  .type-research { color: #34d399; font-size: .72rem; font-weight: 600; }
</style>
</head>
<body>
<nav class="nav">
  <div class="nav-logo">Trading Research</div>
  <a href="/dashboard">Dashboard</a>
  <a href="/ideas" class="active">Ideas</a>
  <a href="/research">Research</a>
  <a href="/data">Data</a>
</nav>
<div class="page">
  <h1>Submit an Idea</h1>
  <p class="subtitle">Strategy idea, research question, market pattern to investigate — describe it in plain text.<br>
  The agent will classify it and route it to the right pipeline automatically.</p>

  {flash}

  <div class="card">
    <form method="post" action="/ideas">
      <textarea name="description" rows="5"
        placeholder="Examples:&#10;• Buy when RSI crosses above 30 and price is above 200 EMA on EURUSD&#10;• Does the London open spike reverse within 30 minutes?&#10;• Analyse how 1m volume predicts the next 1h candle direction&#10;• SuperTrend + MACD crossover with session filter" required></textarea>
      <div class="hint">Strategy ideas get implemented and backtested. Research questions get analysed by a data agent. The pipeline decides which is which.</div>
      <button type="submit">Submit</button>
    </form>
  </div>

  {feed_section}
</div>
</body>
</html>"""


def _render_ideas_page(flash: str = "", flash_type: str = "") -> str:
    flash_html = f'<div class="flash {flash_type}">{flash}</div>' if flash else ""

    feed_items = ""
    try:
        sb = db.get_client()
        # Recent strategy ideas (user_ideas inbox)
        idea_rows = (
            sb.table("user_ideas")
            .select("id, title, description, status, created_at")
            .order("created_at", desc=True)
            .limit(15)
            .execute()
        ).data or []
        for r in idea_rows:
            ts = (r.get("created_at") or "")[:10]
            preview = (r.get("description") or "")[:100].replace("<", "&lt;")
            if len(r.get("description") or "") > 100:
                preview += "…"
            badge = f'<span class="badge badge-{r["status"]}">{r["status"]}</span>'
            feed_items += f"""<div class="feed-row">
              <div class="feed-title"><span class="type-strategy">strategy</span> {badge}</div>
              <div class="feed-desc">{preview}</div>
              <div class="feed-meta">{ts}</div>
            </div>"""
    except Exception:
        pass

    try:
        tasks = db.get_research_tasks(status="all", limit=15)
        for t in tasks:
            ts = (t.get("created_at") or "")[:10]
            preview = (t.get("result_summary") or t.get("question") or "")[:100].replace("<", "&lt;")
            if len(preview) == 100:
                preview += "…"
            status = t.get("status", "pending")
            badge = f'<span class="badge badge-{status}">{status}</span>'
            feed_items += f"""<div class="feed-row">
              <div class="feed-title"><span class="type-research">research</span>
                {(t.get("title") or "")[:70]} {badge}</div>
              <div class="feed-desc">{preview}</div>
              <div class="feed-meta">{ts}</div>
            </div>"""
    except Exception:
        pass

    feed_section = ""
    if feed_items:
        feed_section = f'<div class="feed-hdr">Recent Submissions</div>{feed_items}'

    return (
        _IDEAS_HTML
        .replace("{flash}", flash_html)
        .replace("{feed_section}", feed_section)
    )


@app.get("/ideas", response_class=HTMLResponse)
def ideas_page() -> HTMLResponse:
    return HTMLResponse(_render_ideas_page())


@app.post("/ideas")
def submit_idea(
    background_tasks: BackgroundTasks,
    description: str = Form(...),
):
    description = description.strip()
    if not description:
        return HTMLResponse(_render_ideas_page("Description is required.", "error"))

    # Placeholder title from description — LLM renames it during pre-filter
    title = description[:80].rstrip() + ("…" if len(description) > 80 else "")

    try:
        sb = db.get_client()
        result = sb.table("user_ideas").insert({
            "title": title,
            "description": description,
            "status": "pending",
        }).execute()
        idea_id = result.data[0]["id"] if result.data else "?"
        log.info("idea_submitted", idea_id=idea_id, title=title)
        background_tasks.add_task(_scheduled_queue_worker)
        return RedirectResponse(url="/dashboard", status_code=303)
    except Exception as exc:
        log.error("idea_submit_failed", error=str(exc))
        return HTMLResponse(_render_ideas_page(f"Error saving idea: {exc}", "error"))


# ---------------------------------------------------------------------------
# Research page — AI-generated strategy ideas from scientific papers
# ---------------------------------------------------------------------------

_RESEARCH_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Indicator Research — Trading Research</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0a0d14; color: #e2e8f0; min-height: 100vh; }

  .nav { display: flex; align-items: center; gap: 0; background: #111827;
         border-bottom: 1px solid #1f2937; padding: 0 24px; }
  .nav-logo { font-weight: 700; font-size: 0.95rem; color: #f8fafc;
              padding: 14px 0; margin-right: 32px; letter-spacing: -.01em; }
  .nav a { display: block; padding: 14px 16px; font-size: 0.85rem; color: #94a3b8;
           text-decoration: none; border-bottom: 2px solid transparent; transition: color .15s; }
  .nav a:hover, .nav a.active { color: #f1f5f9; border-bottom-color: #6366f1; }
  .nav-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }

  .page { max-width: 1300px; margin: 0 auto; padding: 32px 24px; }
  .page-header { display: flex; align-items: flex-start; justify-content: space-between;
                 margin-bottom: 24px; flex-wrap: wrap; gap: 14px; }
  .page-title { font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
  .page-sub   { font-size: 0.85rem; color: #64748b; }

  /* ── Stats bar ── */
  .stats-bar { display: flex; gap: 14px; margin-bottom: 24px; flex-wrap: wrap; }
  .stat-pill { background: #111827; border: 1px solid #1f2937; border-radius: 10px;
               padding: 12px 20px; min-width: 110px; text-align: center; }
  .stat-pill .num { font-size: 1.6rem; font-weight: 700; color: #f8fafc;
                    font-family: "SF Mono","Fira Code",monospace; line-height: 1; }
  .stat-pill .lbl { font-size: 0.7rem; color: #475569; text-transform: uppercase;
                    letter-spacing: .06em; margin-top: 4px; }
  .stat-pill.works   .num { color: #4ade80; }
  .stat-pill.fails   .num { color: #f87171; }
  .stat-pill.partial .num { color: #fbbf24; }

  /* ── Filter bar ── */
  .filter-bar { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }
  .filter-chip { padding: 5px 14px; border-radius: 99px; font-size: 0.8rem; font-weight: 500;
                 border: 1px solid #1f2937; background: transparent; color: #64748b;
                 cursor: pointer; transition: all .15s; }
  .filter-chip:hover  { color: #f1f5f9; border-color: #374151; }
  .filter-chip.active { color: #fff; border-color: transparent; }
  .filter-chip.all     { }
  .filter-chip.all.active    { background: #6366f1; }
  .filter-chip.works.active  { background: #059669; }
  .filter-chip.fails.active  { background: #dc2626; }
  .filter-chip.partial.active { background: #b45309; }
  .search-input { flex: 1; max-width: 260px; background: #111827; border: 1px solid #1f2937;
                  border-radius: 8px; color: #f1f5f9; font-size: 0.85rem;
                  padding: 7px 14px; outline: none; }
  .search-input:focus { border-color: #4f46e5; }
  .search-input::placeholder { color: #374151; }

  /* ── Knowledge grid ── */
  .kb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px;
             margin-bottom: 40px; }

  .kb-card { background: #111827; border: 1px solid #1f2937; border-radius: 12px;
             padding: 18px; display: flex; flex-direction: column; gap: 10px; }
  .kb-card:hover { border-color: #374151; }

  .kb-card-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
  .kb-indicator { font-size: 0.95rem; font-weight: 700; color: #f1f5f9;
                  font-family: "SF Mono","Fira Code",monospace; }
  .cat-badge { padding: 3px 10px; border-radius: 99px; font-size: 0.68rem;
               font-weight: 700; text-transform: uppercase; letter-spacing: .04em;
               white-space: nowrap; flex-shrink: 0; }
  .cat-works   { background: #14532d; color: #86efac; }
  .cat-fails   { background: #7f1d1d; color: #fca5a5; }
  .cat-partial { background: #3b2f00; color: #fcd34d; }
  .cat-edge_case { background: #1e293b; color: #7dd3fc; }

  .kb-meta { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .kb-tag  { background: #1e2533; border-radius: 6px; padding: 2px 8px;
             font-size: 0.72rem; color: #64748b; font-family: "SF Mono",monospace; }
  .kb-sharpe { font-size: 0.72rem; font-weight: 600; margin-left: auto;
               font-family: "SF Mono",monospace; }
  .kb-sharpe.pos { color: #4ade80; }
  .kb-sharpe.neg { color: #f87171; }
  .kb-sharpe.neu { color: #64748b; }

  .kb-summary { font-size: 0.82rem; color: #94a3b8; line-height: 1.55; flex: 1; }

  /* ── Queue section ── */
  .section-hdr { font-size: 1rem; font-weight: 700; color: #f1f5f9; margin: 8px 0 16px;
                 display: flex; align-items: center; gap: 14px; }
  .queue-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
  .queue-table th { text-align: left; padding: 8px 14px; color: #475569; font-weight: 600;
                    font-size: 0.72rem; text-transform: uppercase; letter-spacing: .05em;
                    border-bottom: 1px solid #1f2937; }
  .queue-table td { padding: 10px 14px; border-bottom: 1px solid #111827; color: #94a3b8;
                    vertical-align: top; }
  .queue-table tr:hover td { background: #0d1117; }
  .queue-table tbody tr { cursor: pointer; }
  .q-title { color: #e2e8f0; font-size: 0.85rem; }
  .q-status { display: inline-block; padding: 2px 10px; border-radius: 99px;
              font-size: 0.7rem; font-weight: 600; white-space: nowrap; }
  .qs-pending  { background: #1e293b; color: #7dd3fc; }
  .qs-running  { background: #3b2f00; color: #fcd34d; }
  .qs-done     { background: #14532d; color: #86efac; }
  .qs-failed   { background: #7f1d1d; color: #fca5a5; }

  /* Task detail modal */
  .modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.7);
                   z-index:1000; align-items:center; justify-content:center; }
  .modal-overlay.open { display:flex; }
  .modal-box { background:#111827; border:1px solid #1f2937; border-radius:10px;
               width:min(720px,92vw); max-height:85vh; overflow-y:auto;
               padding:24px 28px; position:relative; }
  .modal-close { position:absolute; top:14px; right:16px; background:none; border:none;
                 color:#64748b; font-size:1.4rem; cursor:pointer; line-height:1; }
  .modal-close:hover { color:#e2e8f0; }
  .modal-title { font-size:1rem; font-weight:700; color:#f1f5f9; margin-bottom:4px; }
  .modal-meta  { font-size:0.75rem; color:#475569; margin-bottom:16px; }
  .modal-section { margin-top:16px; }
  .modal-section-hdr { font-size:0.7rem; font-weight:600; text-transform:uppercase;
                       letter-spacing:.05em; color:#475569; margin-bottom:6px; }
  .modal-body-text { font-size:0.82rem; color:#94a3b8; line-height:1.6;
                     white-space:pre-wrap; word-break:break-word; }
  .modal-error { font-size:0.82rem; color:#fca5a5; background:#1a0d0d;
                 border:1px solid #7f1d1d; border-radius:6px; padding:12px;
                 white-space:pre-wrap; word-break:break-word; font-family:monospace; }
  .modal-finding { background:#0d1117; border-left:3px solid #6366f1;
                   padding:8px 12px; margin-bottom:6px; border-radius:0 4px 4px 0; }
  .modal-finding-text { font-size:0.82rem; color:#c7d2fe; }
  .modal-finding-conf { font-size:0.7rem; color:#475569; margin-top:2px; }

  /* ── Buttons ── */
  .btn-primary { background: #6366f1; border: none; border-radius: 8px; color: #fff;
                 font-size: 0.85rem; font-weight: 600; padding: 9px 18px; cursor: pointer; }
  .btn-primary:hover { background: #4f46e5; }
  .btn-primary:disabled { opacity: .5; cursor: default; }
  .btn-secondary { background: #1e2533; border: 1px solid #374151; border-radius: 8px;
                   color: #94a3b8; font-size: 0.85rem; padding: 8px 16px; cursor: pointer; }
  .btn-secondary:hover { color: #f1f5f9; border-color: #6366f1; }

  .empty   { text-align: center; padding: 60px 0; color: #475569; font-size: 0.9rem; }
  .loading { text-align: center; padding: 40px 0; color: #475569; }
  .toast { position: fixed; bottom: 24px; right: 24px; background: #059669;
           color: #fff; border-radius: 10px; padding: 12px 20px;
           font-size: 0.85rem; font-weight: 600; z-index: 999;
           opacity: 0; transition: opacity .3s; pointer-events: none; }
  .toast.show { opacity: 1; }
  .toast.error { background: #dc2626; }
</style>
</head>
<body>
<nav class="nav">
  <div class="nav-logo">Trading Research</div>
  <a href="/dashboard">Dashboard</a>
  <a href="/ideas">Ideas</a>
  <a href="/research" class="active">Research</a>
  <a href="/data">Data</a>
  <div class="nav-right">
    <button class="btn-secondary" id="gen-btn" onclick="generateTasks()">
      ⚗ Run indicator research
    </button>
  </div>
</nav>

<div class="page">
  <div class="page-header">
    <div>
      <div class="page-title">Indicator Research Lab</div>
      <div class="page-sub">
        Systematic forward-return analysis of every pandas_ta indicator + price structure concept.
        Findings are injected into strategy generation automatically.
      </div>
    </div>
  </div>

  <!-- Stats -->
  <div class="stats-bar" id="stats-bar">
    <div class="stat-pill"><div class="num" id="s-total">…</div><div class="lbl">Total entries</div></div>
    <div class="stat-pill works"><div class="num" id="s-works">…</div><div class="lbl">Works</div></div>
    <div class="stat-pill fails"><div class="num" id="s-fails">…</div><div class="lbl">Fails</div></div>
    <div class="stat-pill partial"><div class="num" id="s-partial">…</div><div class="lbl">Partial</div></div>
    <div class="stat-pill" style="margin-left:auto">
      <div class="num" id="s-pending">…</div><div class="lbl">Queue pending</div>
    </div>
    <div class="stat-pill">
      <div class="num" id="s-running">…</div><div class="lbl">Running</div>
    </div>
  </div>

  <!-- Knowledge base -->
  <div class="filter-bar">
    <button class="filter-chip all active" onclick="setFilter('all',this)">All</button>
    <button class="filter-chip works"   onclick="setFilter('works',this)">Works</button>
    <button class="filter-chip fails"   onclick="setFilter('fails',this)">Fails</button>
    <button class="filter-chip partial" onclick="setFilter('partial',this)">Partial</button>
    <input class="search-input" id="search" placeholder="Filter by indicator…"
           oninput="onSearch()" />
  </div>
  <div id="kb-grid" class="loading">Loading knowledge base…</div>

  <!-- Queue -->
  <div class="section-hdr" style="margin-top:40px">
    Research Queue
    <span style="font-size:0.78rem;font-weight:400;color:#475569">
      (latest 30 indicator research tasks)
    </span>
  </div>
  <div id="queue-wrap" class="loading">Loading…</div>
</div>

<div class="toast" id="toast"></div>

<!-- Task detail modal -->
<div class="modal-overlay" id="task-modal" onclick="closeModal(event)">
  <div class="modal-box">
    <button class="modal-close" onclick="document.getElementById('task-modal').classList.remove('open')">&#x2715;</button>
    <div class="modal-title" id="m-title"></div>
    <div class="modal-meta"  id="m-meta"></div>
    <div class="modal-section" id="m-error-wrap" style="display:none">
      <div class="modal-section-hdr">Error</div>
      <div class="modal-error" id="m-error"></div>
    </div>
    <div class="modal-section" id="m-summary-wrap" style="display:none">
      <div class="modal-section-hdr">Summary</div>
      <div class="modal-body-text" id="m-summary"></div>
    </div>
    <div class="modal-section" id="m-findings-wrap" style="display:none">
      <div class="modal-section-hdr">Key Findings</div>
      <div id="m-findings"></div>
    </div>
    <div class="modal-section" id="m-report-wrap" style="display:none">
      <div class="modal-section-hdr">Full Report</div>
      <div class="modal-body-text" id="m-report"></div>
    </div>
  </div>
</div>

<script>
let allEntries = [];
let currentCat = 'all';
let searchVal  = '';

async function init() {
  await Promise.all([loadStats(), loadKnowledge(), loadQueue()]);
  setInterval(init, 30000);
}

async function loadStats() {
  try {
    const r = await fetch('/api/research/stats');
    const d = await r.json();
    document.getElementById('s-total').textContent   = d.total   ?? '0';
    document.getElementById('s-works').textContent   = d.works   ?? '0';
    document.getElementById('s-fails').textContent   = d.fails   ?? '0';
    document.getElementById('s-partial').textContent = d.partial ?? '0';
    document.getElementById('s-pending').textContent = d.queue_pending ?? '0';
    document.getElementById('s-running').textContent = d.queue_running ?? '0';
  } catch(e) {}
}

async function loadKnowledge() {
  try {
    const r = await fetch('/api/knowledge?limit=300');
    const d = await r.json();
    allEntries = d.entries || [];
    renderKnowledge();
  } catch(e) {
    document.getElementById('kb-grid').innerHTML = '<div class="empty">Failed to load knowledge base.</div>';
  }
}

async function loadQueue() {
  try {
    const r = await fetch('/api/research/tasks?type=indicator_research&limit=30');
    const d = await r.json();
    renderQueue(d.tasks || []);
  } catch(e) {
    document.getElementById('queue-wrap').innerHTML = '<div class="empty">Failed to load queue.</div>';
  }
}

function setFilter(cat, btn) {
  currentCat = cat;
  document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  renderKnowledge();
}

function onSearch() {
  searchVal = document.getElementById('search').value.toLowerCase();
  renderKnowledge();
}

function renderKnowledge() {
  const el = document.getElementById('kb-grid');
  let entries = allEntries;
  if (currentCat !== 'all') entries = entries.filter(e => e.category === currentCat);
  if (searchVal)             entries = entries.filter(e =>
    (e.indicator || '').toLowerCase().includes(searchVal) ||
    (e.summary   || '').toLowerCase().includes(searchVal));

  if (!entries.length) {
    el.className = '';
    el.innerHTML = '<div class="empty">No entries match the current filter.<br>'
      + (allEntries.length === 0 ? 'Click "Run indicator research" to generate and queue tests.' : '') + '</div>';
    return;
  }
  el.className = 'kb-grid';
  el.innerHTML = entries.map(renderCard).join('');
}

function renderCard(e) {
  const cat   = e.category || 'partial';
  const ind   = e.indicator || '?';
  const tf    = e.timeframe || '';
  const asset = e.asset     || '';
  const sharpe = e.sharpe_ref;
  const sharpeCls = sharpe === null || sharpe === undefined ? 'neu'
                  : sharpe > 0.3  ? 'pos'
                  : sharpe < -0.3 ? 'neg' : 'neu';
  const sharpeStr = sharpe !== null && sharpe !== undefined
    ? (sharpe > 0 ? '+' : '') + parseFloat(sharpe).toFixed(2) : '—';
  const summary = esc(e.summary || '');
  const ts = (e.created_at || '').slice(0, 10);
  return `
<div class="kb-card">
  <div class="kb-card-top">
    <div class="kb-indicator">${esc(ind)}</div>
    <span class="cat-badge cat-${cat}">${cat}</span>
  </div>
  <div class="kb-meta">
    ${asset ? `<span class="kb-tag">${esc(asset)}</span>` : ''}
    ${tf    ? `<span class="kb-tag">${esc(tf)}</span>`    : ''}
    <span class="kb-sharpe ${sharpeCls}">Sharpe ${sharpeStr}</span>
  </div>
  <div class="kb-summary">${summary}</div>
  <div style="font-size:.68rem;color:#374151;margin-top:2px">${ts}</div>
</div>`;
}

let _queueTasks = [];

function renderQueue(tasks) {
  _queueTasks = tasks;
  const el = document.getElementById('queue-wrap');
  if (!tasks.length) {
    el.className = '';
    el.innerHTML = '<div class="empty">No indicator research tasks yet.<br>Click "Run indicator research" to generate them.</div>';
    return;
  }
  el.className = '';
  const rows = tasks.map((t, i) => {
    const status  = t.status || 'pending';
    const title   = (t.title || '').replace('[Indicator] ', '');
    const ts      = (t.created_at || '').slice(0, 16).replace('T', ' ');
    const summary = esc((t.result_summary || '').slice(0, 100) + (t.result_summary && t.result_summary.length > 100 ? '…' : ''));
    return `<tr onclick="openTaskModal(${i})" title="Click to view details">
      <td class="q-title">${esc(title)}</td>
      <td><span class="q-status qs-${status}">${status}</span></td>
      <td>${summary}</td>
      <td style="white-space:nowrap;color:#374151">${ts}</td>
    </tr>`;
  }).join('');
  el.innerHTML = `
<table class="queue-table">
  <thead><tr>
    <th>Indicator / Signal</th><th>Status</th><th>Summary</th><th>Created</th>
  </tr></thead>
  <tbody>${rows}</tbody>
</table>`;
}

function openTaskModal(i) {
  const t = _queueTasks[i];
  if (!t) return;
  document.getElementById('m-title').textContent = (t.title || '').replace('[Indicator] ', '');
  document.getElementById('m-meta').textContent  =
    `Status: ${t.status || '—'}  ·  Created: ${(t.created_at || '').slice(0, 16).replace('T', ' ')}`;

  const errWrap = document.getElementById('m-error-wrap');
  if (t.error_log) {
    document.getElementById('m-error').textContent = t.error_log;
    errWrap.style.display = '';
  } else {
    errWrap.style.display = 'none';
  }

  const sumWrap = document.getElementById('m-summary-wrap');
  if (t.result_summary) {
    document.getElementById('m-summary').textContent = t.result_summary;
    sumWrap.style.display = '';
  } else {
    sumWrap.style.display = 'none';
  }

  const findingsWrap = document.getElementById('m-findings-wrap');
  const findings = t.key_findings;
  if (findings && findings.length) {
    document.getElementById('m-findings').innerHTML = findings.map(f => {
      const conf = f.confidence != null ? `Confidence: ${(f.confidence * 100).toFixed(0)}%` : '';
      return `<div class="modal-finding">
        <div class="modal-finding-text">${esc(f.finding || '')}</div>
        ${conf ? `<div class="modal-finding-conf">${conf}</div>` : ''}
      </div>`;
    }).join('');
    findingsWrap.style.display = '';
  } else {
    findingsWrap.style.display = 'none';
  }

  const reportWrap = document.getElementById('m-report-wrap');
  if (t.report_text) {
    document.getElementById('m-report').textContent = t.report_text;
    reportWrap.style.display = '';
  } else {
    reportWrap.style.display = 'none';
  }

  document.getElementById('task-modal').classList.add('open');
}

function closeModal(e) {
  if (e.target === document.getElementById('task-modal')) {
    document.getElementById('task-modal').classList.remove('open');
  }
}

async function generateTasks() {
  const btn = document.getElementById('gen-btn');
  btn.disabled = true;
  btn.textContent = 'Generating…';
  try {
    const r = await fetch('/api/research/generate', {method: 'POST'});
    const d = await r.json();
    showToast(`Created ${d.created} new research tasks. They will start shortly.`);
    setTimeout(() => { loadStats(); loadQueue(); }, 1500);
  } catch(e) {
    showToast('Failed to generate tasks', true);
  } finally {
    btn.disabled = false;
    btn.textContent = '⚗ Run indicator research';
  }
}

function showToast(msg, isError) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast' + (isError ? ' error' : '') + ' show';
  setTimeout(() => { t.className = 'toast' + (isError ? ' error' : ''); }, 4000);
}

function esc(s) {
  if (!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

init();
</script>
</body>
</html>"""


@app.get("/research", response_class=HTMLResponse)
def research_page() -> HTMLResponse:
    return HTMLResponse(_RESEARCH_HTML)


# ---------------------------------------------------------------------------
# Indicator Research Lab API
# ---------------------------------------------------------------------------

@app.post("/api/research/generate")
def api_generate_research_tasks(background_tasks: BackgroundTasks) -> JSONResponse:
    """Create pending research_task rows for all untested indicator specs."""
    def _run():
        from agents.indicator_researcher import generate_research_tasks
        return generate_research_tasks()
    background_tasks.add_task(_run)
    # Also run synchronously to get count immediately
    try:
        from agents.indicator_researcher import generate_research_tasks
        created = generate_research_tasks()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse({"ok": True, "created": created})


@app.get("/api/knowledge")
def api_get_knowledge(
    category: str = Query(default="all"),
    indicator: str = Query(default=""),
    limit: int = Query(default=200),
) -> JSONResponse:
    """Return knowledge_base entries with optional filters."""
    try:
        entries = db.get_knowledge_entries(
            category=category if category != "all" else None,
            indicator=indicator if indicator else None,
            limit=min(limit, 500),
        )
        return JSONResponse({"entries": entries})
    except Exception as exc:
        return JSONResponse({"entries": [], "error": str(exc)}, status_code=500)


@app.get("/api/research/stats")
def api_research_stats() -> JSONResponse:
    """Return knowledge base category counts + queue status counts."""
    try:
        kb_stats = db.get_knowledge_stats()
        pending_tasks = db.get_research_tasks(
            status="pending", task_type="indicator_research", limit=200)
        running_tasks = db.get_research_tasks(
            status="running", task_type="indicator_research", limit=200)
        return JSONResponse({
            **kb_stats,
            "queue_pending": len(pending_tasks),
            "queue_running": len(running_tasks),
        })
    except Exception as exc:
        return JSONResponse({"total": 0, "error": str(exc)}, status_code=500)


@app.get("/api/research/tasks")
def api_research_tasks(
    type: str = Query(default="indicator_research"),
    status: str = Query(default="all"),
    limit: int = Query(default=30),
) -> JSONResponse:
    """Return research tasks filtered by type and status."""
    try:
        tasks = db.get_research_tasks(
            status=status,
            task_type=type if type else None,
            limit=min(limit, 200),
        )
        return JSONResponse({"tasks": tasks})
    except Exception as exc:
        return JSONResponse({"tasks": [], "error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Data cache page
# ---------------------------------------------------------------------------

_DATA_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Cache — Trading Research</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0a0d14; color: #e2e8f0; min-height: 100vh; }

  .nav { display: flex; align-items: center; gap: 0; background: #111827;
         border-bottom: 1px solid #1f2937; padding: 0 24px; }
  .nav-logo { font-weight: 700; font-size: 0.95rem; color: #f8fafc;
              padding: 14px 0; margin-right: 32px; letter-spacing: -.01em; }
  .nav a { display: block; padding: 14px 16px; font-size: 0.85rem; color: #94a3b8;
           text-decoration: none; border-bottom: 2px solid transparent; transition: color .15s; }
  .nav a:hover, .nav a.active { color: #f1f5f9; border-bottom-color: #6366f1; }
  .nav-right { margin-left: auto; display: flex; align-items: center; gap: 12px; }

  .page { max-width: 1200px; margin: 0 auto; padding: 32px 24px; }
  .page-header { display: flex; align-items: flex-start; justify-content: space-between;
                 margin-bottom: 28px; flex-wrap: wrap; gap: 14px; }
  .page-title { font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
  .page-sub   { font-size: 0.85rem; color: #64748b; }

  .btn-primary { background: #6366f1; border: none; border-radius: 8px; color: #fff;
                 font-size: 0.85rem; font-weight: 600; padding: 9px 18px; cursor: pointer; }
  .btn-primary:hover { background: #4f46e5; }
  .btn-primary:disabled { opacity: .55; cursor: default; }

  .status-bar { background: #111827; border: 1px solid #1f2937; border-radius: 10px;
                padding: 12px 18px; margin-bottom: 24px; font-size: 0.83rem; color: #64748b;
                display: none; }
  .status-bar.visible { display: block; }
  .status-bar.running { border-color: #3b2f00; color: #fcd34d; background: #1a1500; }
  .status-bar.ok      { border-color: #14532d; color: #86efac; background: #0a1f0f; }
  .status-bar.err     { border-color: #7f1d1d; color: #fca5a5; background: #1f0a0a; }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 20px; }

  .ds-card { background: #111827; border: 1px solid #1f2937; border-radius: 14px; overflow: hidden; }
  .ds-header { padding: 16px 18px 12px; border-bottom: 1px solid #1f2937;
               display: flex; align-items: center; justify-content: space-between; }
  .ds-title { font-size: 1rem; font-weight: 700; color: #f8fafc; }
  .tf-badge { background: #1e2d3d; color: #7dd3fc; border-radius: 6px;
              padding: 3px 10px; font-size: 0.75rem; font-weight: 600;
              font-family: "SF Mono", monospace; }

  .chart-area { height: 80px; background: #0a0f1a; position: relative; }
  canvas.price-chart { width: 100%; height: 80px; display: block; }

  .ds-stats { padding: 14px 18px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .stat { }
  .stat-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: .06em;
                color: #475569; margin-bottom: 3px; }
  .stat-val   { font-size: 0.9rem; font-weight: 600; color: #e2e8f0;
                font-family: "SF Mono", "Fira Code", monospace; }
  .stat-val.good { color: #4ade80; }
  .stat-val.warn { color: #fbbf24; }
  .stat-val.bad  { color: #f87171; }

  .ds-footer { padding: 10px 18px; border-top: 1px solid #1f2937;
               display: flex; align-items: center; justify-content: space-between; }
  .cached-ts { font-size: 0.72rem; color: #374151; }
  .btn-refresh { background: none; border: 1px solid #374151; border-radius: 6px;
                 color: #64748b; font-size: 0.75rem; padding: 4px 10px; cursor: pointer; }
  .btn-refresh:hover { color: #f1f5f9; border-color: #6366f1; }
  .btn-refresh:disabled { opacity: .4; cursor: default; }

  .not-cached { padding: 32px 18px; text-align: center; color: #475569;
                font-size: 0.875rem; border-top: 1px solid #1f2937; }
  .not-cached button { margin-top: 12px; background: #6366f1; border: none;
                       border-radius: 8px; color: #fff; font-size: 0.82rem;
                       font-weight: 600; padding: 7px 16px; cursor: pointer; }

  .coverage-section { padding: 12px 18px; border-top: 1px solid #1f2937; }
  .coverage-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: .06em;
                    color: #475569; margin-bottom: 8px; }
  .coverage-bars { display: flex; gap: 3px; align-items: flex-end; height: 36px; }
  .cbar { flex: 1; border-radius: 2px 2px 0 0; min-width: 4px; position: relative;
          cursor: default; }
  .cbar:hover::after { content: attr(data-tip); position: absolute; bottom: 100%;
    left: 50%; transform: translateX(-50%); background: #1e2533; border: 1px solid #374151;
    border-radius: 6px; padding: 4px 8px; font-size: .7rem; white-space: nowrap;
    color: #e2e8f0; z-index: 10; pointer-events: none; margin-bottom: 4px; }
  .year-labels { display: flex; gap: 3px; margin-top: 3px; }
  .year-label  { flex: 1; font-size: 0.6rem; color: #374151; text-align: center;
                 overflow: hidden; min-width: 4px; }

  .loading { text-align: center; padding: 80px 0; color: #475569; }
</style>
</head>
<body>

<nav class="nav">
  <div class="nav-logo">Trading Research</div>
  <a href="/dashboard">Dashboard</a>
  <a href="/ideas">Ideas</a>
  <a href="/research">Research</a>
  <a href="/data" class="active">Data</a>

  <div class="nav-right">
    <button class="btn-primary" id="preload-btn" onclick="preloadAll()">
      ↓ Preload all data
    </button>
  </div>
</nav>

<div class="page">
  <div class="page-header">
    <div>
      <div class="page-title">OHLCV Data Cache</div>
      <div class="page-sub">Cached market data on the Modal Volume used for backtesting. 1m from 2024, 5m from 2022, 1h from 2015.</div>
    </div>
  </div>

  <div class="status-bar" id="status-bar"></div>
  <div id="grid-wrap"><div class="loading">Loading…</div></div>
</div>

<script>
// Datasets to show (always rendered as cards, even if not yet cached)
const DATASETS = [
  {symbol: 'EURUSD', timeframe: '1h',  label: '1 Hour',   color: '#6366f1'},
  {symbol: 'EURUSD', timeframe: '5m',  label: '5 Min',    color: '#22d3ee'},
  {symbol: 'EURUSD', timeframe: '1m',  label: '1 Min',    color: '#a78bfa'},
];

let cacheMap = {};   // key → metadata row from Supabase

async function loadCache() {
  try {
    const r = await fetch('/api/data/cache');
    const d = await r.json();
    cacheMap = {};
    for (const ds of (d.datasets || [])) {
      cacheMap[ds.symbol + '_' + ds.timeframe] = ds;
    }
    renderGrid();
  } catch(e) {
    document.getElementById('grid-wrap').innerHTML =
      '<div class="loading">Failed to load cache info.</div>';
  }
}

function renderGrid() {
  const wrap = document.getElementById('grid-wrap');
  wrap.innerHTML = `<div class="grid">${DATASETS.map(ds => renderCard(ds)).join('')}</div>`;
  // Load charts asynchronously
  DATASETS.forEach(ds => {
    const key = ds.symbol + '_' + ds.timeframe;
    if (cacheMap[key]) loadChart(ds.symbol, ds.timeframe, ds.color);
  });
}

function renderCard(ds) {
  const key  = ds.symbol + '_' + ds.timeframe;
  const meta = cacheMap[key];
  if (!meta) {
    return `
    <div class="ds-card" id="card-${key}">
      <div class="ds-header">
        <span class="ds-title">${ds.symbol}</span>
        <span class="tf-badge">${ds.timeframe}</span>
      </div>
      <div class="not-cached">
        Not cached yet
        <br><button onclick="preloadOne('${ds.symbol}','${ds.timeframe}',this)">↓ Load now</button>
      </div>
    </div>`;
  }

  const bars   = meta.bar_count ? meta.bar_count.toLocaleString() : '—';
  const first  = meta.first_date ? meta.first_date.slice(0,10) : '—';
  const last   = meta.last_date  ? meta.last_date.slice(0,10)  : '—';
  const sizeMb = meta.file_size_mb ? meta.file_size_mb.toFixed(1) + ' MB' : '—';
  const compl  = meta.completeness_pct != null ? meta.completeness_pct.toFixed(1) + '%' : '—';
  const complCls = meta.completeness_pct >= 90 ? 'good' : meta.completeness_pct >= 70 ? 'warn' : 'bad';

  // Warn if actual start date is more than 60 days later than expected
  let startWarn = '';
  if (meta.expected_start && meta.first_date) {
    const expectedMs = new Date(meta.expected_start).getTime();
    const actualMs   = new Date(meta.first_date).getTime();
    const driftDays  = Math.round((actualMs - expectedMs) / 86400000);
    if (driftDays > 60) {
      startWarn = `<div style="background:#1a1500;border:1px solid #3b2f00;border-radius:8px;
                               padding:8px 12px;font-size:.78rem;color:#fcd34d;margin:0 18px 12px">
        ⚠ Expected data from <b>${meta.expected_start}</b> but earliest bar is <b>${first}</b>
        (${driftDays} days gap). API may not have full history at this timeframe.
        <button onclick="preloadOne('${meta.symbol}','${meta.timeframe}',this)"
          style="margin-left:10px;background:#3b2f00;border:1px solid #fbbf24;border-radius:6px;
                 color:#fcd34d;font-size:.75rem;padding:3px 10px;cursor:pointer">↻ Retry fetch</button>
      </div>`;
    }
  }
  const mean   = meta.price_mean ? meta.price_mean.toFixed(5) : '—';
  const vol    = meta.avg_volume ? meta.avg_volume.toFixed(2) : '—';
  const cachedTs = meta.cached_at ? meta.cached_at.slice(0,16).replace('T',' ') + ' UTC' : '—';

  return `
  <div class="ds-card" id="card-${key}">
    <div class="ds-header">
      <span class="ds-title">${ds.symbol}</span>
      <span class="tf-badge">${ds.timeframe} · ${ds.label}</span>
    </div>
    <div class="chart-area">
      <canvas class="price-chart" id="chart-${key}" width="700" height="80"></canvas>
    </div>
    ${startWarn}
    <div class="ds-stats">
      <div class="stat"><div class="stat-label">Bars</div><div class="stat-val">${bars}</div></div>
      <div class="stat"><div class="stat-label">Size</div><div class="stat-val">${sizeMb}</div></div>
      <div class="stat"><div class="stat-label">From</div><div class="stat-val" style="font-size:.78rem">${first}</div></div>
      <div class="stat"><div class="stat-label">To</div><div class="stat-val" style="font-size:.78rem">${last}</div></div>
      <div class="stat"><div class="stat-label">Completeness</div><div class="stat-val ${complCls}">${compl}</div></div>
      <div class="stat"><div class="stat-label">Avg Close</div><div class="stat-val">${mean}</div></div>
      <div class="stat"><div class="stat-label">Min</div><div class="stat-val">${meta.price_min ? meta.price_min.toFixed(5) : '—'}</div></div>
      <div class="stat"><div class="stat-label">Max</div><div class="stat-val">${meta.price_max ? meta.price_max.toFixed(5) : '—'}</div></div>
      <div class="stat"><div class="stat-label">Std Dev</div><div class="stat-val">${meta.price_std ? meta.price_std.toFixed(5) : '—'}</div></div>
      <div class="stat"><div class="stat-label">Avg Volume</div><div class="stat-val">${vol}</div></div>
    </div>
    ${renderCoverage(meta.bars_by_year, meta.bar_count, ds.timeframe)}
    <div class="ds-footer">
      <span class="cached-ts">Cached: ${cachedTs}</span>
      <button class="btn-refresh" onclick="preloadOne('${ds.symbol}','${ds.timeframe}',this)">↻ Refresh</button>
    </div>
  </div>`;
}

// Expected bars per year per timeframe for FX (5 trading days × 52 weeks × bars_per_day)
const EXPECTED_PER_YEAR = {'1m': 374400, '5m': 74880, '1h': 6240, '4h': 1560, '1d': 260};

function renderCoverage(barsByYear, totalBars, timeframe) {
  if (!barsByYear || typeof barsByYear !== 'object') return '';
  const years = Object.keys(barsByYear).sort();
  if (!years.length) return '';

  const expected = EXPECTED_PER_YEAR[timeframe] || 6240;
  const maxBars  = Math.max(...Object.values(barsByYear), expected * 0.1);

  const cbars = years.map(y => {
    const n    = barsByYear[y];
    const pct  = Math.min(n / expected * 100, 100);
    const h    = Math.max(Math.round((n / maxBars) * 32), 2);
    const col  = pct >= 70 ? '#22d3ee' : pct >= 30 ? '#fbbf24' : '#f87171';
    const tip  = `${y}: ${n.toLocaleString()} bars (${pct.toFixed(0)}% of expected)`;
    return `<div class="cbar" style="height:${h}px;background:${col}" data-tip="${tip}"></div>`;
  }).join('');

  const labels = years.map(y =>
    `<div class="year-label">${y.slice(2)}</div>`
  ).join('');

  return `
  <div class="coverage-section">
    <div class="coverage-label">Data coverage by year</div>
    <div class="coverage-bars">${cbars}</div>
    <div class="year-labels">${labels}</div>
  </div>`;
}

async function loadChart(symbol, timeframe, color) {
  const key = symbol + '_' + timeframe;
  try {
    const r = await fetch(`/api/data/cache/${symbol}/${timeframe}`);
    const d = await r.json();
    const bars = d.bars || [];
    if (bars.length < 2) return;
    drawChart(key, bars, color);
  } catch(e) {}
}

function drawChart(key, bars, color) {
  const canvas = document.getElementById('chart-' + key);
  if (!canvas) return;

  // Use device pixel ratio for crisp rendering
  const dpr  = window.devicePixelRatio || 1;
  const rect  = canvas.getBoundingClientRect();
  const W     = canvas.width  = (rect.width  || 340) * dpr;
  const H     = canvas.height = 80 * dpr;
  canvas.style.width  = (rect.width  || 340) + 'px';
  canvas.style.height = '80px';

  const ctx   = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = W / dpr, h = H / dpr;

  const closes = bars.map(b => b.c);
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const range = max - min || 1;
  const pad   = 4;

  const xScale = v => (v / (closes.length - 1)) * w;
  const yScale = v => h - pad - ((v - min) / range) * (h - pad * 2);

  // Background
  ctx.fillStyle = '#0a0f1a';
  ctx.fillRect(0, 0, w, h);

  // Gradient fill under line
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color + '44');
  grad.addColorStop(1, color + '00');
  ctx.beginPath();
  closes.forEach((c, i) => {
    i === 0 ? ctx.moveTo(xScale(i), yScale(c)) : ctx.lineTo(xScale(i), yScale(c));
  });
  ctx.lineTo(xScale(closes.length - 1), h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Price line
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.lineJoin    = 'round';
  closes.forEach((c, i) => {
    i === 0 ? ctx.moveTo(xScale(i), yScale(c)) : ctx.lineTo(xScale(i), yScale(c));
  });
  ctx.stroke();

  // Min/max labels
  ctx.fillStyle = '#475569';
  ctx.font = `${9 * dpr / dpr}px -apple-system, sans-serif`;
  ctx.fillText(min.toFixed(4), 4, h - pad);
  ctx.fillText(max.toFixed(4), 4, pad + 9);
}

async function preloadAll() {
  const btn = document.getElementById('preload-btn');
  btn.disabled = true;
  btn.textContent = '↓ Preloading…';
  showStatus('running', '⚙ Preload job started on Modal — this takes 5–20 minutes for full data. Refresh the page once complete.');
  try {
    await fetch('/api/data/preload', {method: 'POST'});
  } catch(e) {
    showStatus('err', '✗ Failed to start preload job: ' + e.message);
  }
  setTimeout(() => {
    btn.disabled = false;
    btn.textContent = '↓ Preload all data';
  }, 5000);
}

async function preloadOne(symbol, timeframe, btn) {
  btn.disabled = true;
  btn.textContent = '…';
  showStatus('running', `⚙ Preload job started for ${symbol} ${timeframe} — check back in a few minutes.`);
  try {
    await fetch('/api/data/preload', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
    });
  } catch(e) {
    showStatus('err', '✗ Failed: ' + e.message);
  }
  setTimeout(() => {
    btn.disabled = false;
    btn.textContent = btn.className.includes('btn-refresh') ? '↻ Refresh' : '↓ Load now';
  }, 5000);
}

function showStatus(type, msg) {
  const el = document.getElementById('status-bar');
  el.textContent = msg;
  el.className = 'status-bar visible ' + type;
}

loadCache();
</script>
</body>
</html>"""


@app.get("/data", response_class=HTMLResponse)
def data_page() -> HTMLResponse:
    return HTMLResponse(_DATA_HTML)


# ---------------------------------------------------------------------------
# Scheduled jobs
# ---------------------------------------------------------------------------

def _scheduled_queue_worker() -> None:
    """Wrapper so APScheduler exceptions are logged rather than silently swallowed."""
    try:
        process_queue()
    except Exception:
        log.error("scheduled_queue_worker_crash", traceback=traceback.format_exc())


def _scheduled_research_cycle() -> None:
    """Fetch new papers from arXiv / Semantic Scholar and extract strategy ideas."""
    try:
        from agents.idea_generator import run_idea_generator
        inserted = run_idea_generator()
        log.info("research_cycle_complete", new_ideas=len(inserted))
    except Exception:
        log.error("research_cycle_failed", traceback=traceback.format_exc())


def _scheduled_budget_log() -> None:
    """Log current budget status so it shows up in Render log stream."""
    try:
        today_spend = db.get_daily_spend()
        remaining = get_remaining_budget()
        max_spend = float(os.environ.get("MAX_DAILY_SPEND_USD", 8.0))
        log.info(
            "budget_status",
            today_spend_usd=round(today_spend, 4),
            remaining_usd=round(remaining, 4),
            limit_usd=max_spend,
        )
    except Exception as exc:
        log.warning("budget_log_failed", error=str(exc))




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
