"""
Agent 4: Validator
Reviews generated strategy code for data leakage and logic bugs using Claude Sonnet.
If the LLM provides corrected_code, runs leakage detection on that too.
Updates strategy status to "validating" (ready for Monte Carlo) on success.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic
from dotenv import load_dotenv

from db import supabase_client as db
from agents.prompts import VALIDATOR_SYSTEM, VALIDATOR_USER_TEMPLATE
from agents.utils import full_description
from backtest.leakage_detector import check_leakage

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"  # Code review needs strong reasoning — Sonnet not Haiku


def run_validator(strategy_id: str) -> dict[str, Any]:
    """
    Validate strategy code for leakage and logic bugs.
    Updates strategy status in DB.
    Returns the validation result dict.
    """
    strategy = db.get_strategy(strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")

    code = strategy.get("backtest_code", "")
    if not code:
        _mark_failed(strategy_id, "No backtest_code found — run Implementer first")
        raise ValueError(f"Strategy {strategy_id} has no code to validate")

    user_msg = VALIDATOR_USER_TEMPLATE.format(
        strategy_name=strategy.get("name", strategy_id),
        description=full_description(strategy),
        code=code,
    )

    from agents.utils import call_claude
    response = call_claude(
        model=MODEL,
        max_tokens=4096,
        system=VALIDATOR_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    usage = response.usage
    cost = _estimate_cost(MODEL, usage.input_tokens, usage.output_tokens)
    db.log_spend("validator", MODEL, usage.input_tokens, usage.output_tokens, cost, strategy_id)

    raw_text = response.content[0].text.strip()
    result = _parse_json_response(raw_text, strategy_id)

    if result is None:
        _mark_failed(strategy_id, "Validator returned unparseable response")
        raise RuntimeError(f"Validator could not parse LLM response for {strategy_id}")

    # ── Check corrected_code if LLM provided one ──────────────────────────────
    corrected_code: str | None = result.get("corrected_code")
    final_code = code
    final_leakage_result = check_leakage(code)  # Always run on original for baseline

    if corrected_code and corrected_code.strip():
        corrected_leakage = check_leakage(corrected_code)
        log.info(
            f"Validator: corrected_code leakage score={corrected_leakage.score} "
            f"(original={final_leakage_result.score})"
        )
        # Use corrected code if it's at least as clean as the original
        if corrected_leakage.score >= final_leakage_result.score:
            final_code = corrected_code
            final_leakage_result = corrected_leakage
        else:
            log.warning(
                f"Validator: corrected_code is WORSE (score {corrected_leakage.score} < "
                f"{final_leakage_result.score}) — keeping original"
            )
    else:
        log.info(f"Validator: no corrected_code provided; using original (score={final_leakage_result.score})")

    # ── Determine final pass/fail ─────────────────────────────────────────────
    llm_passed: bool = result.get("passed", False)
    static_passed: bool = final_leakage_result.passed  # score >= 7.0

    # Strategy must pass BOTH the LLM review and static leakage detection
    overall_passed = llm_passed and static_passed

    leakage_issues: list[str] = (
        result.get("leakage_issues", [])
        + final_leakage_result.issues
    )
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped_issues: list[str] = []
    for issue in leakage_issues:
        if issue not in seen:
            seen.add(issue)
            deduped_issues.append(issue)

    if overall_passed:
        new_status = "validating"
        error_log = None
    else:
        new_status = "failed"
        failure_reasons = []
        if not llm_passed:
            failure_reasons.append("LLM validator: failed")
        if not static_passed:
            failure_reasons.append(
                f"Static leakage detector: score={final_leakage_result.score}/10 "
                f"({len(final_leakage_result.issues)} critical issues)"
            )
        error_log = "; ".join(failure_reasons)

    db.update_strategy(strategy_id, {
        "status": new_status,
        "backtest_code": final_code,         # Save corrected code if improved
        "leakage_score": final_leakage_result.score,
        "leakage_issues": deduped_issues,
        "error_log": error_log,
    })

    log.info(
        f"Validator: strategy={strategy_id} passed={overall_passed} "
        f"leakage_score={final_leakage_result.score} → {new_status}"
    )

    return {
        "passed": overall_passed,
        "llm_passed": llm_passed,
        "static_passed": static_passed,
        "leakage_score": final_leakage_result.score,
        "leakage_issues": deduped_issues,
        "leakage_warnings": final_leakage_result.warnings,
        "logic_bugs": result.get("logic_bugs", []),
        "performance_issues": result.get("performance_issues", []),
        "structural_issues": result.get("structural_issues", []),
        "corrections_made": result.get("corrections_made", []),
        "confidence": result.get("confidence", 0.0),
        "validator_notes": result.get("validator_notes", ""),
        "final_code": final_code,
        "status": new_status,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_json_response(text: str, strategy_id: str) -> dict | None:
    """Strip markdown fences and parse JSON from the LLM response."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            stripped = part.strip()
            if stripped.startswith("{"):
                text = stripped
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"Validator JSON parse error for {strategy_id}: {e}\nText prefix: {text[:300]}")
        return None


def _mark_failed(strategy_id: str, reason: str) -> None:
    db.update_strategy(strategy_id, {"status": "failed", "error_log": reason})


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = {
        "claude-haiku-4-5-20251001": (0.00025, 0.00125),
        "claude-sonnet-4-6": (0.003, 0.015),
        "claude-opus-4-6": (0.015, 0.075),
    }
    in_price, out_price = prices.get(model, (0.003, 0.015))
    return (input_tokens * in_price + output_tokens * out_price) / 1000
