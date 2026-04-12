"""Shared utilities for all agents."""
from __future__ import annotations

import json


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
