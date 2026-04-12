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
