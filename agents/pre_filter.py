"""
Agent 2: Pre-Filter
Evaluates strategy ideas and scores them. Score >= 6 → proceed to Implementer.
User-submitted ideas get +2 bonus and are processed immediately.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

from db import supabase_client as db
from agents.prompts import PRE_FILTER_SYSTEM, PRE_FILTER_USER_TEMPLATE
from agents.utils import full_description, add_pipeline_note

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"  # Explicit 0-2 rubric + structured JSON output → Haiku sufficient; 12× cheaper than Sonnet
MIN_SCORE_TO_PROCEED = 6.0


def run_pre_filter(strategy_id: str) -> dict[str, Any]:
    """
    Score a strategy idea. Updates strategy status in DB.
    Returns the parsed scoring result.
    """
    strategy = db.get_strategy(strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")

    knowledge = db.get_knowledge_summary(limit=30)
    knowledge_text = _format_knowledge(knowledge)

    def _esc(s: str) -> str:
        """Escape curly braces in user content so str.format() doesn't choke on {self} etc."""
        return s.replace("{", "{{").replace("}", "}}")

    user_msg = PRE_FILTER_USER_TEMPLATE.format(
        title=_esc(strategy.get("name", "")),
        description=_esc(full_description(strategy)),
        notes=_esc(strategy.get("entry_logic", "")),
        source=_esc(strategy.get("source", "")),
        knowledge_base_context=knowledge_text,
    )

    from agents.utils import call_claude
    response = call_claude(
        model=MODEL,
        max_tokens=1024,
        system=PRE_FILTER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    # Track spend
    usage = response.usage
    cost = _estimate_cost(MODEL, usage.input_tokens, usage.output_tokens)
    db.log_spend("pre_filter", MODEL, usage.input_tokens, usage.output_tokens, cost, strategy_id)

    # Parse response
    raw_text = response.content[0].text.strip()
    result = _parse_json_response(raw_text, strategy_id)

    score = float(result.get("score", 0))
    verdict = result.get("verdict", "reject")
    submission_type = result.get("submission_type", "strategy")

    # ── Route research questions to the research pipeline ────────────────────
    if submission_type == "research":
        research_title    = result.get("research_title") or strategy.get("name") or "Research Task"
        research_question = result.get("research_question") or strategy.get("hypothesis") or ""
        task = db.insert_research_task({
            "title":    research_title,
            "question": research_question,
            "type":     "market_analysis",
            "status":   "pending",
            "created_by_strategy_id": None,
        })
        task_id = task.get("id", "?") if task else "?"
        log.info("idea_routed_as_research", strategy_id=strategy_id, task_id=task_id)
        # Delete the placeholder strategy record — it's not a strategy
        try:
            db.delete_strategy(strategy_id)
        except Exception as _del_err:
            log.warning("could_not_delete_placeholder_strategy", error=str(_del_err))
        return {"submission_type": "research", "task_id": task_id}

    # ── Strategy flow ────────────────────────────────────────────────────────
    # Apply user bonus
    if strategy.get("source") == "user":
        result["score"] = min(10.0, result.get("score", 0) + 2.0)
        result["score_breakdown"]["user_bonus"] = 2

    score = float(result.get("score", 0))

    if verdict == "proceed" and score >= MIN_SCORE_TO_PROCEED:
        new_status = "filtered"
    elif verdict == "modify":
        new_status = "filtered"  # Proceed with modifications noted
    else:
        new_status = "failed"

    updates: dict = {
        "status": new_status,
        "pre_filter_score": score,
        "pre_filter_notes": json.dumps(result),
        "error_log": result.get("rejection_reason") if new_status == "failed" else None,
    }
    if result.get("strategy_name"):
        updates["name"] = result["strategy_name"].strip()
    # Overwrite hypothesis with the refined description so the Implementer
    # gets a precise, actionable spec. The original text is preserved in entry_logic.
    if result.get("refined_description"):
        updates["hypothesis"] = result["refined_description"].strip()
    db.update_strategy(strategy_id, updates)

    log.info(f"Pre-filter: strategy={strategy_id} score={score} verdict={verdict} → {new_status}")

    if new_status == "failed":
        add_pipeline_note(strategy_id, f"Pre-filter REJECTED — score {score:.1f}/10.\n{result.get('rejection_reason', '')}")
    else:
        mods = result.get("suggested_modifications")
        note = f"Pre-filter passed — score {score:.1f}/10, verdict: {verdict}."
        if mods:
            import re
            mods_formatted = re.sub(r'\s+(\d+\.)\s+', r'\n\1 ', mods.strip())
            note += f"\n\nSuggestions:\n{mods_formatted}"
        add_pipeline_note(strategy_id, note)

    return result


def _format_knowledge(entries: list[dict]) -> str:
    if not entries:
        return "No knowledge base entries yet."
    lines = []
    for e in entries[:20]:  # Limit context
        lines.append(
            f"[{e['category']}] {e.get('indicator', '?')} on {e.get('timeframe', '?')} "
            f"({e.get('asset', '?')}): {e['summary']}"
        )
    return "\n".join(lines)


def _parse_json_response(text: str, strategy_id: str) -> dict:
    # Strip markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse pre-filter response for {strategy_id}: {e}\nText: {text[:200]}")
        return {"score": 0, "verdict": "reject", "rejection_reason": f"Parse error: {e}"}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    # Pricing as of early 2026 (update if needed)
    prices = {
        "claude-haiku-4-5-20251001": (0.00025, 0.00125),      # per 1K tokens in/out
        "claude-sonnet-4-6": (0.003, 0.015),
        "claude-opus-4-6": (0.015, 0.075),
    }
    in_price, out_price = prices.get(model, (0.003, 0.015))
    return (input_tokens * in_price + output_tokens * out_price) / 1000
