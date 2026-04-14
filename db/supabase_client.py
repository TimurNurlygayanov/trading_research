"""
Supabase client wrapper — typed helpers used by all agents.
Never import supabase directly elsewhere; always use these helpers.
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_ANON_KEY"]
        _client = create_client(url, key)
    return _client


# ── strategies ──────────────────────────────────────────────────────────────

def insert_strategy(data: dict[str, Any]) -> dict[str, Any]:
    sb = get_client()
    result = sb.table("strategies").insert(data).execute()
    return result.data[0]


def update_strategy(strategy_id: str, updates: dict[str, Any]) -> None:
    sb = get_client()
    sb.table("strategies").update(updates).eq("id", strategy_id).execute()


def delete_strategy(strategy_id: str) -> None:
    sb = get_client()
    # Clear FK references first
    sb.table("user_ideas").update({"strategy_id": None}).eq("strategy_id", strategy_id).execute()
    sb.table("spend_log").delete().eq("strategy_id", strategy_id).execute()
    sb.table("knowledge_base").delete().eq("strategy_id", strategy_id).execute()
    sb.table("strategies").delete().eq("id", strategy_id).execute()


def get_strategy(strategy_id: str) -> dict[str, Any] | None:
    sb = get_client()
    result = sb.table("strategies").select("*").eq("id", strategy_id).execute()
    return result.data[0] if result.data else None


def get_strategies_by_status(status: str, limit: int = 10) -> list[dict[str, Any]]:
    sb = get_client()
    result = (
        sb.table("strategies")
        .select("*")
        .eq("status", status)
        .order("created_at")
        .limit(limit)
        .execute()
    )
    return result.data


def get_campaign_children(campaign_id: str) -> list[dict[str, Any]]:
    """Return all child strategies for a campaign, ordered by quick_test_sharpe desc."""
    sb = get_client()
    result = (
        sb.table("strategies")
        .select(
            "id, name, status, hypothesis, quick_test_sharpe, quick_test_trades, "
            "quick_test_win_rate, quick_test_drawdown, quick_test_signals_per_year, "
            "best_timeframe, error_log, analysis_notes, updated_at"
        )
        .eq("campaign_id", campaign_id)
        .order("quick_test_sharpe", desc=True, nullsfirst=False)
        .execute()
    )
    return result.data or []


# ── user_ideas ───────────────────────────────────────────────────────────────

def get_pending_user_ideas(limit: int = 5) -> list[dict[str, Any]]:
    sb = get_client()
    result = (
        sb.table("user_ideas")
        .select("*")
        .eq("status", "pending")
        .order("priority")
        .order("created_at")
        .limit(limit)
        .execute()
    )
    return result.data


def mark_idea_picked_up(idea_id: str, strategy_id: str) -> None:
    sb = get_client()
    sb.table("user_ideas").update({
        "status": "picked_up",
        "strategy_id": strategy_id,
    }).eq("id", idea_id).execute()


def mark_idea_done(idea_id: str) -> None:
    sb = get_client()
    sb.table("user_ideas").update({"status": "done"}).eq("id", idea_id).execute()


def mark_idea_failed(idea_id: str, error: str) -> None:
    sb = get_client()
    sb.table("user_ideas").update({"status": "failed"}).eq("id", idea_id).execute()


# ── knowledge_base ───────────────────────────────────────────────────────────

def insert_knowledge(data: dict[str, Any]) -> None:
    sb = get_client()
    sb.table("knowledge_base").insert(data).execute()


def get_knowledge_summary(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent knowledge entries for the Researcher/Pre-filter context."""
    sb = get_client()
    result = (
        sb.table("knowledge_base")
        .select("category, indicator, timeframe, asset, session, summary, sharpe_ref")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


# ── spend_log ────────────────────────────────────────────────────────────────

def log_spend(
    agent: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    strategy_id: str | None = None,
) -> None:
    sb = get_client()
    sb.table("spend_log").insert({
        "agent": agent,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "strategy_id": strategy_id,
    }).execute()


# ── generated_ideas ──────────────────────────────────────────────────────────

def insert_generated_idea(data: dict[str, Any]) -> dict[str, Any]:
    sb = get_client()
    result = sb.table("generated_ideas").insert(data).execute()
    return result.data[0]


def get_generated_ideas(status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
    sb = get_client()
    q = sb.table("generated_ideas").select("*")
    if status != "all":
        q = q.eq("status", status)
    result = q.order("created_at", desc=True).limit(limit).execute()
    return result.data


def get_generated_idea_urls() -> set[str]:
    sb = get_client()
    result = sb.table("generated_ideas").select("source_url").execute()
    return {r["source_url"] for r in result.data if r.get("source_url")}


def update_generated_idea(idea_id: str, updates: dict[str, Any]) -> None:
    sb = get_client()
    sb.table("generated_ideas").update(updates).eq("id", idea_id).execute()


# ── data_cache ───────────────────────────────────────────────────────────────

def upsert_data_cache(symbol: str, timeframe: str, data: dict[str, Any]) -> None:
    """Insert or update a data_cache row (keyed by symbol + timeframe)."""
    sb = get_client()
    data["cached_at"] = datetime.utcnow().isoformat()
    sb.table("data_cache").upsert(data, on_conflict="symbol,timeframe").execute()


def get_data_cache(symbol: str | None = None) -> list[dict[str, Any]]:
    """Return cache metadata for all datasets (without recent_bars to keep payload small)."""
    sb = get_client()
    q = sb.table("data_cache").select(
        "symbol, timeframe, bar_count, first_date, last_date, "
        "file_size_mb, price_min, price_max, price_mean, price_std, "
        "avg_volume, completeness_pct, bars_by_year, expected_start, cached_at"
    ).order("symbol").order("timeframe")
    if symbol:
        q = q.eq("symbol", symbol)
    return q.execute().data or []


def get_data_cache_bars(symbol: str, timeframe: str) -> list[dict[str, Any]]:
    """Return recent_bars for a single dataset (used by /data chart endpoint)."""
    sb = get_client()
    result = (
        sb.table("data_cache")
        .select("recent_bars")
        .eq("symbol", symbol)
        .eq("timeframe", timeframe)
        .execute()
    )
    if not result.data:
        return []
    raw = result.data[0].get("recent_bars") or []
    return raw if isinstance(raw, list) else []


# ── research_tasks ───────────────────────────────────────────────────────────

def insert_research_task(data: dict[str, Any]) -> dict[str, Any]:
    sb = get_client()
    result = sb.table("research_tasks").insert(data).execute()
    return result.data[0]


def update_research_task(task_id: str, updates: dict[str, Any]) -> None:
    sb = get_client()
    sb.table("research_tasks").update(updates).eq("id", task_id).execute()


def get_research_task(task_id: str) -> dict[str, Any] | None:
    sb = get_client()
    result = sb.table("research_tasks").select("*").eq("id", task_id).execute()
    return result.data[0] if result.data else None


def get_research_tasks(
    status: str = "pending",
    limit: int = 20,
    task_type: str | None = None,
) -> list[dict[str, Any]]:
    sb = get_client()
    q = sb.table("research_tasks").select("*")
    if status != "all":
        q = q.eq("status", status)
    if task_type:
        q = q.eq("type", task_type)
    result = q.order("created_at", desc=True).limit(limit).execute()
    return result.data or []


def get_knowledge_entries(
    category: str | None = None,
    indicator: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return knowledge_base entries with optional category/indicator filters."""
    sb = get_client()
    q = sb.table("knowledge_base").select(
        "id, category, indicator, timeframe, asset, session, summary, sharpe_ref, created_at"
    )
    if category and category != "all":
        q = q.eq("category", category)
    if indicator:
        q = q.ilike("indicator", f"%{indicator}%")
    result = q.order("created_at", desc=True).limit(limit).execute()
    return result.data or []


def get_knowledge_stats() -> dict[str, int]:
    """Return counts per category for the knowledge base."""
    sb = get_client()
    result = sb.table("knowledge_base").select("category").execute()
    rows = result.data or []
    stats: dict[str, int] = {"total": len(rows), "works": 0, "fails": 0, "partial": 0, "edge_case": 0}
    for r in rows:
        cat = r.get("category", "partial")
        stats[cat] = stats.get(cat, 0) + 1
    return stats


def get_strategies_awaiting_research(limit: int = 5) -> list[dict[str, Any]]:
    """Return strategies blocked waiting for research tasks to complete."""
    sb = get_client()
    result = (
        sb.table("strategies")
        .select("id, name, pending_research_ids, hypothesis, entry_logic, "
                "indicators, pre_filter_notes, hyperparams")
        .eq("status", "awaiting_research")
        .order("created_at")
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_daily_spend(for_date: date | None = None) -> float:
    sb = get_client()
    target = (for_date or date.today()).isoformat()
    result = (
        sb.table("spend_log")
        .select("cost_usd")
        .eq("date", target)
        .execute()
    )
    return sum(row["cost_usd"] for row in result.data)
