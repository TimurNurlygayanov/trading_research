"""Shared utilities for all agents."""
from __future__ import annotations

import datetime
import json


def add_pipeline_note(strategy_id: str, text: str) -> None:
    """Append an automated pipeline note to the strategy's comments."""
    from db import supabase_client as db
    strategy = db.get_strategy(strategy_id)
    if not strategy:
        return
    existing = strategy.get("comments") or []
    if isinstance(existing, str):
        try:
            existing = json.loads(existing)
        except (ValueError, TypeError):
            existing = []
    ts = datetime.datetime.utcnow().isoformat()[:19]
    existing.append({"text": f"[pipeline] {text}", "ts": ts})
    db.update_strategy(strategy_id, {"comments": existing})


def full_description(strategy: dict) -> str:
    """
    Return the strategy's full description as seen by agents:
    original hypothesis + any user comments appended below.

    This ensures every agent in the pipeline has the same complete context
    regardless of when the user added comments.
    """
    parts: list[str] = []

    hypothesis = (strategy.get("hypothesis") or "").strip()
    if hypothesis:
        parts.append(hypothesis)

    comments = strategy.get("comments") or []
    if isinstance(comments, str):
        try:
            comments = json.loads(comments)
        except (ValueError, TypeError):
            comments = []

    if comments:
        lines = ["--- User comments ---"]
        for c in comments:
            ts = (c.get("ts") or "")[:10]
            text = (c.get("text") or "").strip()
            if text:
                lines.append(f"[{ts}] {text}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)
