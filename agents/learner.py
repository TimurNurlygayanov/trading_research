"""
Agent 6: Learner
Extracts generalizable knowledge from completed strategy results and stores it
in the knowledge_base table. This knowledge is used by the Pre-Filter and
Researcher agents to improve future decisions.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic
from dotenv import load_dotenv

from db import supabase_client as db
from agents.prompts import LEARNER_SYSTEM, LEARNER_USER_TEMPLATE
from agents.utils import full_description

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"  # Knowledge extraction is lightweight


def run_learner(strategy_id: str) -> list[dict[str, Any]]:
    """
    Extract knowledge entries from a completed (or failed) strategy result.
    Inserts each entry into the knowledge_base table via db.insert_knowledge.
    Returns the list of inserted entries.
    """
    strategy = db.get_strategy(strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")

    user_msg = _build_user_message(strategy)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=LEARNER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    usage = response.usage
    cost = _estimate_cost(MODEL, usage.input_tokens, usage.output_tokens)
    db.log_spend("learner", MODEL, usage.input_tokens, usage.output_tokens, cost, strategy_id)

    raw_text = response.content[0].text.strip()
    entries = _parse_json_response(raw_text, strategy_id)

    if not entries:
        log.warning(f"Learner: no knowledge entries extracted for {strategy_id}")
        return []

    # Validate and insert each entry
    inserted: list[dict[str, Any]] = []
    for entry in entries:
        if not _validate_entry(entry):
            log.warning(f"Learner: skipping invalid entry: {entry}")
            continue

        # Enrich with strategy metadata for traceability
        row = {
            "category": entry["category"],
            "indicator": entry.get("indicator"),
            "timeframe": entry.get("timeframe"),
            "asset": entry.get("asset"),
            "session": entry.get("session"),
            "summary": entry["summary"],
            "sharpe_ref": _safe_float(entry.get("sharpe_ref")),
            "strategy_id": strategy_id,
        }

        try:
            db.insert_knowledge(row)
            inserted.append(row)
            log.debug(f"Learner: inserted knowledge entry [{entry['category']}]: {entry['summary'][:80]}")
        except Exception as e:
            log.error(f"Learner: failed to insert knowledge entry for {strategy_id}: {e}")

    log.info(f"Learner: strategy={strategy_id} inserted {len(inserted)}/{len(entries)} knowledge entries")
    return inserted


# ── Message builder ───────────────────────────────────────────────────────────

def _build_user_message(strategy: dict) -> str:
    """Format LEARNER_USER_TEMPLATE from the strategy record."""
    wf_scores = strategy.get("walk_forward_scores", "[]")
    if isinstance(wf_scores, str):
        try:
            wf_scores = json.loads(wf_scores)
        except (json.JSONDecodeError, TypeError):
            pass

    hyperparams = strategy.get("best_hyperparams", "{}")
    if isinstance(hyperparams, str):
        try:
            hyperparams = json.loads(hyperparams)
        except (json.JSONDecodeError, TypeError):
            pass

    indicators_used = strategy.get("indicators_used", "[]")
    if isinstance(indicators_used, str):
        try:
            indicators_used = json.loads(indicators_used)
        except (json.JSONDecodeError, TypeError):
            pass

    return LEARNER_USER_TEMPLATE.format(
        strategy_name=strategy.get("name", strategy.get("id", "Unknown")),
        hypothesis=full_description(strategy),
        status=strategy.get("status", "unknown"),
        reject_reason=strategy.get("error_log") or "N/A",
        sharpe=_fmt(strategy.get("sharpe_ratio")),
        calmar=_fmt(strategy.get("calmar_ratio")),
        win_rate=_fmt(strategy.get("win_rate")),
        total_trades=strategy.get("total_trades", "N/A"),
        signals_per_year=_fmt(strategy.get("signals_per_year")),
        oos_sharpe=_fmt(strategy.get("oos_sharpe")),
        monte_carlo_pvalue=_fmt(strategy.get("monte_carlo_pvalue")),
        walk_forward_scores=wf_scores,
        hyperparams=hyperparams,
        best_session_hours=strategy.get("best_session_hours", "N/A"),
        indicators_used=indicators_used,
        timeframe=strategy.get("timeframe", "N/A"),
        symbol=strategy.get("symbol", "N/A"),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_response(text: str, strategy_id: str) -> list[dict]:
    """Parse a JSON array from the LLM response, stripping markdown fences."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            stripped = part.strip()
            if stripped.startswith("["):
                text = stripped
                break

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        # LLM sometimes wraps the array in a dict key
        if isinstance(data, dict):
            for key in ("entries", "knowledge", "results", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        log.error(f"Learner: expected JSON array for {strategy_id}, got {type(data)}")
        return []
    except json.JSONDecodeError as e:
        log.error(f"Learner JSON parse error for {strategy_id}: {e}\nText prefix: {text[:300]}")
        return []


def _validate_entry(entry: dict) -> bool:
    """Ensure required fields are present and values are within allowed enums."""
    if not isinstance(entry, dict):
        return False
    if not entry.get("category") or not entry.get("summary"):
        return False
    valid_categories = {"works", "fails", "partial", "edge_case"}
    if entry["category"] not in valid_categories:
        log.warning(f"Learner: unknown category '{entry['category']}' — defaulting to 'partial'")
        entry["category"] = "partial"
    # summary must be non-trivial
    if len(entry["summary"].strip()) < 5:
        return False
    return True


def _safe_float(value: Any) -> float | None:
    """Convert to float or return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, decimals: int = 3) -> str:
    """Format a numeric value or return 'N/A'."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = {
        "claude-haiku-4-5-20251001": (0.00025, 0.00125),
        "claude-sonnet-4-6": (0.003, 0.015),
        "claude-opus-4-6": (0.015, 0.075),
    }
    in_price, out_price = prices.get(model, (0.003, 0.015))
    return (input_tokens * in_price + output_tokens * out_price) / 1000
