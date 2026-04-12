"""
Agent 3: Implementer
Translates a scored strategy idea into a complete backtesting.py Strategy class.
Uses Claude Sonnet for better code quality. Runs leakage detection and retries on failure.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic
from dotenv import load_dotenv

from db import supabase_client as db
from agents.prompts import IMPLEMENTER_SYSTEM, IMPLEMENTER_USER_TEMPLATE
from agents.utils import full_description
from backtest.leakage_detector import check_leakage

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"  # Needs better code quality than Haiku
MAX_RETRIES = 2
MIN_LEAKAGE_SCORE = 7.0


def run_implementer(strategy_id: str) -> dict[str, Any]:
    """
    Generate strategy code from a filtered idea.
    Retries up to MAX_RETRIES times if leakage is detected.
    Updates strategy status in DB and returns generated code + param_space.
    """
    strategy = db.get_strategy(strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")

    knowledge = db.get_knowledge_summary(limit=20)
    knowledge_text = _format_knowledge(knowledge)

    # Parse pre_filter_notes for context passed from pre-filter agent
    pre_filter_data = _parse_pre_filter_notes(strategy.get("pre_filter_notes", ""))

    user_msg = IMPLEMENTER_USER_TEMPLATE.format(
        title=strategy.get("name", ""),
        description=full_description(strategy),
        notes=strategy.get("entry_logic", ""),
        pre_filter_notes=pre_filter_data.get("notes", ""),
        indicators=", ".join(pre_filter_data.get("suggested_indicators", [])),
        timeframes=", ".join(pre_filter_data.get("suggested_timeframes", ["1h", "4h"])),
        symbols=", ".join(pre_filter_data.get("suggested_symbols", ["EURUSD", "GBPUSD"])),
        knowledge_base_context=knowledge_text,
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    result: dict[str, Any] | None = None
    last_leakage_result = None
    total_input_tokens = 0
    total_output_tokens = 0

    messages: list[dict[str, str]] = [{"role": "user", "content": user_msg}]

    for attempt in range(MAX_RETRIES + 1):
        log.info(f"Implementer: strategy={strategy_id} attempt={attempt + 1}/{MAX_RETRIES + 1}")

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=IMPLEMENTER_SYSTEM,
            messages=messages,
        )

        usage = response.usage
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens

        raw_text = response.content[0].text.strip()
        parsed = _parse_json_response(raw_text, strategy_id)

        if not parsed or "code" not in parsed:
            log.warning(f"Implementer: parse failed on attempt {attempt + 1}")
            if attempt < MAX_RETRIES:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response could not be parsed as valid JSON with a 'code' key. "
                        "Please return ONLY a JSON object with the keys: strategy_name, "
                        "strategy_class, code, param_space, hypothesis, indicators_used, "
                        "recommended_symbols, recommended_timeframes, notes."
                    ),
                })
                continue
            # All retries exhausted on parse failure
            _mark_failed(strategy_id, "Failed to parse implementer response after retries")
            _log_spend(strategy_id, total_input_tokens, total_output_tokens)
            raise RuntimeError(f"Implementer failed to produce valid JSON for {strategy_id}")

        # Run leakage detection on generated code
        code = parsed["code"]
        leakage_result = check_leakage(code)
        last_leakage_result = leakage_result

        log.info(
            f"Implementer: leakage score={leakage_result.score} "
            f"issues={len(leakage_result.issues)} attempt={attempt + 1}"
        )

        if leakage_result.score >= MIN_LEAKAGE_SCORE:
            # Code is clean enough — accept it
            result = parsed
            break

        if attempt < MAX_RETRIES:
            # Feed issues back to LLM for a fix
            issues_text = "\n".join(f"  - {issue}" for issue in leakage_result.issues)
            warnings_text = "\n".join(f"  - {w}" for w in leakage_result.warnings)
            fix_request = (
                f"The generated code has data leakage issues (score: {leakage_result.score}/10). "
                f"Please fix ALL of the following issues and return the corrected JSON:\n\n"
                f"CRITICAL ISSUES (must fix):\n{issues_text or '  None'}\n\n"
                f"WARNINGS (should fix):\n{warnings_text or '  None'}\n\n"
                "Remember: no shift(-N), no bfill(), no pandas_ta inside next(), "
                "all indicators in init() with self.I(), entry signals use [-2] not [-1]."
            )
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": fix_request})
        else:
            # Retries exhausted — save what we have with a warning
            log.warning(
                f"Implementer: leakage score {leakage_result.score} below threshold after "
                f"{MAX_RETRIES} retries — saving with warning for strategy={strategy_id}"
            )
            result = parsed

    # Save total spend
    cost = _estimate_cost(MODEL, total_input_tokens, total_output_tokens)
    db.log_spend("implementer", MODEL, total_input_tokens, total_output_tokens, cost, strategy_id)

    if result is None:
        _mark_failed(strategy_id, "Implementer produced no valid result")
        raise RuntimeError(f"Implementer produced no valid result for {strategy_id}")

    # Persist code and param_space to strategy record
    leakage_score = last_leakage_result.score if last_leakage_result else 0.0
    leakage_issues = last_leakage_result.issues if last_leakage_result else []

    db.update_strategy(strategy_id, {
        "status": "implemented",
        "backtest_code": result["code"],
        # hyperparams stores the param_space dict until Optuna replaces it with best params
        "hyperparams": result.get("param_space", {}),
        # indicators stores metadata: class name + indicator list
        "indicators": {
            "strategy_class": result.get("strategy_class", ""),
            "indicators_used": result.get("indicators_used", []),
            "symbols": result.get("recommended_symbols", ["EURUSD"]),
            "timeframes": result.get("recommended_timeframes", ["1h"]),
        },
        "leakage_score": leakage_score,
        "leakage_issues": leakage_issues,   # jsonb — pass list directly
        "hypothesis": result.get("hypothesis", strategy.get("hypothesis", "")),
        "error_log": None,
    })

    log.info(
        f"Implementer: strategy={strategy_id} → implemented "
        f"(leakage_score={leakage_score}, class={result.get('strategy_class', '?')})"
    )

    return {
        "code": result["code"],
        "param_space": result.get("param_space", {}),
        "strategy_class": result.get("strategy_class", ""),
        "leakage_score": leakage_score,
        "leakage_issues": leakage_issues,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_pre_filter_notes(raw: str | None) -> dict:
    """Parse the JSON stored in pre_filter_notes field."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_json_response(text: str, strategy_id: str) -> dict | None:
    """Extract and parse JSON from the LLM response, stripping markdown fences."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        # Find the first ``` block that looks like JSON
        parts = text.split("```")
        for part in parts[1::2]:  # Odd indices are code block content
            stripped = part.strip()
            if stripped.startswith("{"):
                text = stripped
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error for {strategy_id}: {e}\nText prefix: {text[:300]}")
        return None


def _format_knowledge(entries: list[dict]) -> str:
    if not entries:
        return "No knowledge base entries yet."
    lines = []
    for e in entries[:15]:
        lines.append(
            f"[{e['category']}] {e.get('indicator', '?')} on {e.get('timeframe', '?')} "
            f"({e.get('asset', '?')}): {e['summary']}"
        )
    return "\n".join(lines)


def _mark_failed(strategy_id: str, reason: str) -> None:
    db.update_strategy(strategy_id, {"status": "failed", "error_log": reason})


def _log_spend(strategy_id: str, input_tokens: int, output_tokens: int) -> None:
    cost = _estimate_cost(MODEL, input_tokens, output_tokens)
    db.log_spend("implementer", MODEL, input_tokens, output_tokens, cost, strategy_id)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = {
        "claude-haiku-4-5-20251001": (0.00025, 0.00125),
        "claude-sonnet-4-6": (0.003, 0.015),
        "claude-opus-4-6": (0.015, 0.075),
    }
    in_price, out_price = prices.get(model, (0.003, 0.015))
    return (input_tokens * in_price + output_tokens * out_price) / 1000
