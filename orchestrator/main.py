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

    _scheduled_budget_log()

    scheduler.add_job(_scheduled_queue_worker, trigger="interval", minutes=10,
                      id="queue_worker", replace_existing=True)
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
        done = counts.get("done", 0)
        failed = counts.get("failed", 0)
        in_progress = total - done - failed - counts.get("idea", 0)

        today_spend = db.get_daily_spend()
        remaining = get_remaining_budget()
    except Exception as exc:
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
            "walk_forward_scores, hypothesis, hyperparams, best_session_hours, "
            "error_log, report_url, tags, comments, created_at, updated_at"
        )
        if status != "all":
            q = q.eq("status", status)
        result = q.order("updated_at", desc=True).range(offset, offset + limit - 1).execute()
        return JSONResponse({"strategies": result.data or [], "offset": offset, "limit": limit})
    except Exception as exc:
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
  .s-idea        { background: #1e293b; color: #94a3b8; }
  .s-filtered    { background: #1e3a5f; color: #93c5fd; }
  .s-implementing{ background: #2d1f5e; color: #c4b5fd; }
  .s-implemented { background: #3b2f00; color: #fcd34d; }
  .s-validating  { background: #1c3352; color: #67e8f9; }
  .s-done        { background: #14532d; color: #86efac; }
  .s-failed      { background: #450a0a; color: #fca5a5; }

  /* ── metric cells ── */
  .metric { font-family: "SF Mono", "Fira Code", monospace; font-size: 0.82rem; }
  .pos { color: #4ade80; } .neg { color: #f87171; } .neu { color: #94a3b8; }

  /* ── detail panel (slide-in) ── */
  .panel-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.6);
                   z-index: 100; display: none; }
  .panel-overlay.open { display: block; }
  .panel { position: fixed; right: 0; top: 0; bottom: 0; width: min(620px, 100vw);
           background: #111827; border-left: 1px solid #1f2937;
           overflow-y: auto; z-index: 101; padding: 28px;
           transform: translateX(100%); transition: transform .25s ease; }
  .panel.open { transform: translateX(0); }
  .panel-close { float: right; background: none; border: none; color: #64748b;
                 font-size: 1.4rem; cursor: pointer; line-height: 1; }
  .panel-close:hover { color: #f1f5f9; }
  .panel h2 { font-size: 1.1rem; font-weight: 600; color: #f8fafc;
              margin-bottom: 6px; padding-right: 32px; }
  .panel-hyp { color: #94a3b8; font-size: 0.875rem; margin-bottom: 20px; line-height: 1.5; }
  .panel-section { margin-bottom: 20px; }
  .panel-section h3 { font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                      letter-spacing: .07em; color: #475569; margin-bottom: 10px; }
  .kv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
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
</style>
</head>
<body>

<nav class="nav">
  <div class="nav-logo">Trading Research</div>
  <a href="/dashboard" class="active">Dashboard</a>
  <a href="/ideas">Ideas</a>
  <a href="/health">Health</a>
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

<!-- Detail panel -->
<div class="panel-overlay" id="overlay" onclick="closePanel()"></div>
<div class="panel" id="panel">
  <button class="panel-close" onclick="closePanel()">×</button>
  <div id="panel-content"></div>
</div>

<script>
const STATUS_ORDER = ['all','idea','filtered','implementing','implemented','validating','done','failed'];
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
      <td><span class="badge ${badgeCls}">${r.status}</span></td>
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
  document.getElementById('panel').classList.add('open');
  document.getElementById('panel-content').innerHTML = '<div class="loading">Loading…</div>';
  const r = await fetch('/api/strategy/' + id);
  const s = await r.json();
  renderPanel(s);
}

function closePanel() {
  document.getElementById('overlay').classList.remove('open');
  document.getElementById('panel').classList.remove('open');
}

function renderPanel(s) {
  const badgeCls = 's-' + (s.status || 'idea');
  const trainSharpe = fmtNum(s.backtest_sharpe);
  const oosSharpe   = fmtNum(s.oos_sharpe);
  const trainCls    = s.backtest_sharpe > 1 ? 'good' : s.backtest_sharpe > 0 ? '' : 'bad';
  const oosCls      = s.oos_sharpe > 0.8 ? 'good' : s.oos_sharpe > 0 ? '' : 'bad';
  const mc = s.monte_carlo_pvalue != null ? s.monte_carlo_pvalue.toFixed(4) : '—';
  const mcCls = s.monte_carlo_pvalue < 0.05 ? 'good' : 'bad';
  const dd = s.max_drawdown != null ? (s.max_drawdown * 100).toFixed(2) + '%' : '—';
  const wr = s.win_rate != null ? (s.win_rate * 100).toFixed(1) + '%' : '—';

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
    paramsHtml = `<div class="panel-section"><h3>Best Hyperparameters</h3>
      <pre class="params">${JSON.stringify(p, null, 2)}</pre></div>`;
  }

  let errHtml = '';
  if (s.error_log) {
    errHtml = `<div class="panel-section"><h3>Error Log</h3>
      <div class="error-box">${esc(s.error_log)}</div></div>`;
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
    // Render markdown-ish text: bold, headers, lists
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
  } else if (s.report_url) {
    reportHtml = `<div class="panel-section">
      <a href="${s.report_url}" target="_blank"
         style="color:#818cf8;font-size:.85rem;">↗ View full report (R2)</a></div>`;
  }

  document.getElementById('panel-content').innerHTML = `
    <h2>${esc(s.name)}</h2>
    <div style="margin-bottom:12px"><span class="badge ${badgeCls}">${s.status}</span>
      <span style="color:#475569;font-size:.78rem;margin-left:8px">
        ${s.source || 'user'} · ${(s.created_at||'').slice(0,10)}</span></div>
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

    ${actionsHtml}${wfHtml}${paramsHtml}${errHtml}${reportHtml}
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

function renderComments(comments) {
  if (!comments || !comments.length) {
    return '<span style="color:#475569;font-size:.8rem">No comments yet</span>';
  }
  const list = typeof comments === 'string' ? JSON.parse(comments) : comments;
  return list.map(c => `
    <div style="background:#0f1623;border-radius:8px;padding:10px 12px;margin-bottom:8px">
      <div style="font-size:.82rem;color:#e2e8f0;white-space:pre-wrap;word-break:break-word">${esc(c.text)}</div>
      <div style="font-size:.7rem;color:#475569;margin-top:4px">${c.ts ? c.ts.replace('T',' ') : ''}</div>
    </div>`).join('');
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

async function deleteStrategy(id, name) {
  if (!confirm(`Delete "${name}"?\nThis cannot be undone.`)) return;
  const r = await fetch(`/api/strategy/${id}`, {method:'DELETE'});
  const data = await r.json();
  if (data.ok) { closePanel(); loadAll(); }
  else alert('Delete failed: ' + (data.error || 'unknown error'));
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
<title>Trading Strategy Ideas</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117; color: #e2e8f0;
    margin: 0; padding: 24px;
    min-height: 100vh;
  }
  h1 { font-size: 1.5rem; font-weight: 600; margin: 0 0 4px; color: #f8fafc; }
  .subtitle { color: #94a3b8; font-size: 0.875rem; margin: 0 0 32px; }
  .card {
    background: #1e2533; border: 1px solid #2d3748;
    border-radius: 12px; padding: 28px;
    max-width: 680px; margin: 0 auto 32px;
  }
  label { display: block; font-size: 0.8rem; font-weight: 500;
          color: #94a3b8; margin-bottom: 6px; text-transform: uppercase; letter-spacing: .05em; }
  input[type=text], textarea, select {
    width: 100%; background: #0f1117; border: 1px solid #374151;
    border-radius: 8px; color: #f1f5f9; padding: 10px 14px;
    font-size: 0.95rem; outline: none;
    transition: border-color .15s;
  }
  input[type=text]:focus, textarea:focus, select:focus { border-color: #6366f1; }
  textarea { resize: vertical; min-height: 110px; font-family: inherit; }
  .field { margin-bottom: 20px; }
button[type=submit] {
    width: 100%; background: #6366f1; color: #fff;
    border: none; border-radius: 8px; padding: 12px;
    font-size: 1rem; font-weight: 600; cursor: pointer;
    transition: background .15s;
  }
  button[type=submit]:hover { background: #4f46e5; }
  .flash {
    max-width: 680px; margin: 0 auto 24px;
    padding: 14px 18px; border-radius: 8px;
    font-size: 0.9rem; font-weight: 500;
  }
  .flash.success { background: #14532d; border: 1px solid #16a34a; color: #bbf7d0; }
  .flash.error   { background: #450a0a; border: 1px solid #dc2626; color: #fecaca; }
  .ideas-list { max-width: 680px; margin: 0 auto; }
  .idea-row {
    background: #1e2533; border: 1px solid #2d3748;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
  }
  .idea-title { font-weight: 600; color: #f1f5f9; margin-bottom: 4px; }
  .idea-meta  { font-size: 0.8rem; color: #64748b; }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 99px;
    font-size: 0.75rem; font-weight: 600; margin-left: 8px;
  }
  .badge-pending    { background: #1e3a5f; color: #93c5fd; }
  .badge-picked_up  { background: #1c3352; color: #67e8f9; }
  .badge-done       { background: #14532d; color: #86efac; }
  .badge-failed     { background: #450a0a; color: #fca5a5; }
  .section-title { font-size: 1rem; font-weight: 600; color: #94a3b8;
                   max-width: 680px; margin: 0 auto 14px; }
</style>
</head>
<body>
<div style="max-width:680px;margin:0 auto 28px">
  <h1>Strategy Ideas</h1>
  <p class="subtitle">Describe your idea in plain English — the pipeline handles the rest.</p>
</div>

{flash}

<div class="card">
  <form method="post" action="/ideas">
    <div class="field">
      <label>Description *</label>
      <textarea name="description" placeholder="Describe the entry/exit logic, which indicators, timeframe, markets. The more detail the better — but vague ideas work too." required></textarea>
    </div>
    <div class="field">
      <label>Notes (optional)</label>
      <textarea name="notes" placeholder="Known caveats, inspiration papers, session preferences, etc." style="min-height:70px"></textarea>
    </div>
    <button type="submit">Submit for Analysis</button>
  </form>
</div>

{ideas_section}

</body>
</html>"""


def _render_ideas_page(flash: str = "", flash_type: str = "") -> str:
    flash_html = ""
    if flash:
        flash_html = f'<div class="flash {flash_type}">{flash}</div>'

    # Load recent ideas from DB
    ideas_html = ""
    try:
        sb = db.get_client()
        result = (
            sb.table("user_ideas")
            .select("id, title, description, status, created_at")
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        rows = result.data or []
        if rows:
            items = ""
            for r in rows:
                badge_cls = f"badge-{r['status'].replace('_', '_')}"
                ts = r["created_at"][:10] if r.get("created_at") else ""
                desc_preview = (r["description"] or "")[:120].replace("<", "&lt;")
                if len(r["description"] or "") > 120:
                    desc_preview += "…"
                items += f"""
                <div class="idea-row">
                  <div class="idea-title">
                    {r['title']}
                    <span class="badge {badge_cls}">{r['status']}</span>
                  </div>
                  <div style="color:#94a3b8;font-size:.875rem;margin:4px 0 4px">{desc_preview}</div>
                  <div class="idea-meta">{ts}</div>
                </div>"""
            ideas_html = f'<div class="section-title">Recent Ideas</div><div class="ideas-list">{items}</div>'
    except Exception:
        pass

    return _IDEAS_HTML.replace("{flash}", flash_html).replace("{ideas_section}", ideas_html)


@app.get("/ideas", response_class=HTMLResponse)
def ideas_page(submitted: str = "") -> HTMLResponse:
    if submitted:
        return HTMLResponse(_render_ideas_page(
            "Idea submitted! It will be picked up within 10 minutes.", "success"
        ))
    return HTMLResponse(_render_ideas_page())


@app.post("/ideas")
def submit_idea(
    background_tasks: BackgroundTasks,
    description: str = Form(...),
    notes: str = Form(""),
):
    description = description.strip()
    notes = notes.strip()

    if not description:
        return HTMLResponse(_render_ideas_page("Description is required.", "error"))

    # Auto-generate a placeholder title from description (LLM will rename it during pre-filter)
    title = description[:80].rstrip() + ("…" if len(description) > 80 else "")

    try:
        sb = db.get_client()
        result = sb.table("user_ideas").insert({
            "title": title,
            "description": description,
            "notes": notes or None,
            "status": "pending",
        }).execute()
        idea_id = result.data[0]["id"] if result.data else "?"
        log.info("idea_submitted", idea_id=idea_id, title=title)
        # Process immediately in background instead of waiting for the 10-min scheduler
        background_tasks.add_task(_scheduled_queue_worker)
        # POST → redirect → GET prevents double-submit on refresh
        return RedirectResponse(url="/ideas?submitted=1", status_code=303)
    except Exception as exc:
        log.error("idea_submit_failed", error=str(exc))
        return HTMLResponse(_render_ideas_page(f"Error saving idea: {exc}", "error"))


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
    """Placeholder for the future autonomous researcher agent."""
    log.info("research_cycle", msg="Research cycle tick — researcher agent not yet implemented.")


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
