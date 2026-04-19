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

from dotenv import load_dotenv

from db import supabase_client as db
from agents.prompts import (
    IMPLEMENTER_SYSTEM,
    IMPLEMENTER_USER_TEMPLATE,
    IMPLEMENTER_USER_TEMPLATE_WITH_RESEARCH_OPTION,
)
from agents.utils import full_description, add_pipeline_note
from backtest.leakage_detector import check_leakage

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"  # Detailed template + leakage detector backstop → Sonnet quality is sufficient; 5× cheaper than Opus
MAX_RETRIES = 2
MIN_LEAKAGE_SCORE = 7.0


def run_implementer(
    strategy_id: str,
    research_results: list[dict] | None = None,
    allow_research_requests: bool = True,
) -> dict[str, Any]:
    """
    Generate strategy code from a filtered idea.

    Parameters
    ----------
    strategy_id           : DB strategy UUID
    research_results      : completed research task results to inject as context.
                            Populated when re-running after awaiting_research.
    allow_research_requests: if True, the LLM may respond with needs_research_first=true
                             instead of code. Queue worker handles that response.
                             Set False when re-running after research (always want code).

    Returns
    -------
    dict with either:
      - {"code": ..., "param_space": ..., ...}  — normal implementation
      - {"needs_research": True, "task_ids": [...]}  — research requested
    """
    strategy = db.get_strategy(strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")

    db.update_strategy(strategy_id, {"status": "implementing"})

    knowledge = db.get_knowledge_summary(limit=50)
    knowledge_text = _format_knowledge(knowledge)

    library = db.get_indicator_library(limit=50)
    library_text = _format_indicator_library(library)

    pre_filter_data = _parse_pre_filter_notes(strategy.get("pre_filter_notes", ""))

    def _esc(s: str) -> str:
        return s.replace("{", "{{").replace("}", "}}")

    # Build research context string if we have prior research results
    research_context = _format_research_results(research_results or [])

    # Choose prompt variant based on whether we allow research requests
    if allow_research_requests and not research_results:
        template = IMPLEMENTER_USER_TEMPLATE_WITH_RESEARCH_OPTION
        user_msg = template.format(
            title=_esc(strategy.get("name", "")),
            description=_esc(full_description(strategy)),
            notes=_esc(strategy.get("entry_logic", "")),
            pre_filter_notes=_esc(pre_filter_data.get("notes", "")),
            indicators=", ".join(pre_filter_data.get("suggested_indicators", [])),
            timeframes=", ".join(pre_filter_data.get("suggested_timeframes", ["1h", "4h"])),
            symbols=", ".join(pre_filter_data.get("suggested_symbols", ["EURUSD", "GBPUSD"])),
            knowledge_base_context=knowledge_text,
            indicator_library_context=library_text,
        )
    else:
        user_msg = IMPLEMENTER_USER_TEMPLATE.format(
            title=_esc(strategy.get("name", "")),
            description=_esc(full_description(strategy)),
            notes=_esc(strategy.get("entry_logic", "")),
            pre_filter_notes=_esc(pre_filter_data.get("notes", "")),
            indicators=", ".join(pre_filter_data.get("suggested_indicators", [])),
            timeframes=", ".join(pre_filter_data.get("suggested_timeframes", ["1h", "4h"])),
            symbols=", ".join(pre_filter_data.get("suggested_symbols", ["EURUSD", "GBPUSD"])),
            knowledge_base_context=knowledge_text,
            research_context=research_context,
            indicator_library_context=library_text,
        )

    from agents.utils import call_claude

    result: dict[str, Any] | None = None
    last_leakage_result = None
    total_input_tokens = 0
    total_output_tokens = 0

    messages: list[dict[str, str]] = [{"role": "user", "content": user_msg}]

    for attempt in range(MAX_RETRIES + 1):
        log.info(f"Implementer: strategy={strategy_id} attempt={attempt + 1}/{MAX_RETRIES + 1}")

        response = call_claude(
            model=MODEL,
            max_tokens=3000,
            system=IMPLEMENTER_SYSTEM,
            messages=messages,
        )

        usage = response.usage
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens

        raw_text = response.content[0].text.strip()
        parsed = _parse_json_response(raw_text, strategy_id)

        if not parsed:
            log.warning(f"Implementer: parse failed on attempt {attempt + 1}")
            if attempt < MAX_RETRIES:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response could not be parsed as valid JSON. "
                        "Please return ONLY a valid JSON object."
                    ),
                })
                continue
            _mark_failed(strategy_id, "Failed to parse implementer response after retries")
            _log_spend(strategy_id, total_input_tokens, total_output_tokens)
            raise RuntimeError(f"Implementer failed to produce valid JSON for {strategy_id}")

        # Handle research request — agent wants to gather data before coding
        if parsed.get("needs_research_first"):
            research_tasks_raw = parsed.get("research_tasks") or []
            if not research_tasks_raw:
                # LLM said needs_research but gave no tasks — treat as a bad response
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": (
                        "You set needs_research_first=true but provided no research_tasks. "
                        "Either provide research_tasks with at least one item, "
                        "or proceed directly with strategy implementation."
                    ),
                })
                continue

            _log_spend(strategy_id, total_input_tokens, total_output_tokens)
            task_ids = _create_research_tasks(strategy_id, research_tasks_raw)
            reason = parsed.get("reason", "Agent requested research before implementation.")
            db.update_strategy(strategy_id, {
                "status": "awaiting_research",
                "pending_research_ids": task_ids,
                "error_log": None,
            })
            add_pipeline_note(
                strategy_id,
                f"Implementer requested {len(task_ids)} research task(s) before coding. "
                f"Reason: {reason}. Task IDs: {task_ids}"
            )
            log.info(f"Implementer: research requested, strategy={strategy_id}, tasks={task_ids}")
            return {"needs_research": True, "task_ids": task_ids}

        if "code" not in parsed:
            log.warning(f"Implementer: no 'code' key on attempt {attempt + 1}")
            if attempt < MAX_RETRIES:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response is missing the 'code' key. "
                        "Please return ONLY a JSON object with the keys: strategy_name, "
                        "strategy_class, code, param_space, hypothesis, indicators_used, "
                        "recommended_symbols, recommended_timeframes, notes."
                    ),
                })
                continue
            _mark_failed(strategy_id, "Implementer produced no code after retries")
            _log_spend(strategy_id, total_input_tokens, total_output_tokens)
            raise RuntimeError(f"Implementer produced no code for {strategy_id}")

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
            # Retries exhausted and still leaky — reject rather than waste Modal compute
            issues_text = "; ".join(leakage_result.issues[:3])
            reason = (
                f"Leakage score {leakage_result.score}/10 still below {MIN_LEAKAGE_SCORE} "
                f"after {MAX_RETRIES} retries. Issues: {issues_text}"
            )
            log.warning(f"Implementer: {reason} — marking failed for strategy={strategy_id}")
            _mark_failed(strategy_id, reason)
            _log_spend(strategy_id, total_input_tokens, total_output_tokens)
            raise RuntimeError(reason)

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

    issues_summary = f" Issues: {'; '.join(leakage_issues[:3])}" if leakage_issues else ""
    leak_status = "clean" if leakage_score >= MIN_LEAKAGE_SCORE else f"WARNING score {leakage_score:.1f}/10"
    add_pipeline_note(
        strategy_id,
        f"Implementer generated {result.get('strategy_class', 'Strategy')} — "
        f"leakage {leak_status}.{issues_summary} Dispatching to Modal for backtest."
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


def _format_research_results(results: list[dict]) -> str:
    """Format completed research task results for injection into implementer context."""
    if not results:
        return "No prior research available."
    lines = []
    for r in results:
        lines.append(f"### {r.get('title', 'Research')}")
        lines.append(f"Question: {r.get('question', '')}")
        lines.append(f"Summary: {r.get('result_summary', 'No summary.')}")
        findings = r.get("key_findings") or []
        if findings:
            lines.append("Key findings:")
            for f in findings[:5]:
                finding_text = f if isinstance(f, str) else f.get("finding", "")
                lines.append(f"  - {finding_text}")
        lines.append("")
    return "\n".join(lines)


def _create_research_tasks(strategy_id: str, tasks_raw: list[dict]) -> list[str]:
    """Create research_tasks DB records and dispatch them to Modal. Returns task IDs."""
    task_ids = []
    for task_spec in tasks_raw[:3]:  # cap at 3 research tasks per strategy
        record = db.insert_research_task({
            "type": task_spec.get("type", "custom"),
            "title": task_spec.get("title", "Research Task"),
            "question": task_spec.get("question", ""),
            "data_requirements": task_spec.get("data_requirements"),
            "status": "pending",
            "created_by_strategy_id": strategy_id,
        })
        task_ids.append(record["id"])

    # Dispatch research tasks to Modal asynchronously
    try:
        import modal
        fn = modal.Function.from_name("trading-research-research", "run_research_task")
        for task_id in task_ids:
            call = fn.spawn(task_id)
            db.update_research_task(task_id, {"modal_job_id": call.object_id, "status": "running"})
        log.info(f"Research tasks dispatched to Modal: {task_ids}")
    except Exception as exc:
        log.warning(f"Could not dispatch research tasks to Modal: {exc}")
        # Tasks remain in 'pending' — queue worker will retry dispatch

    return task_ids


def _format_indicator_library(entries: list[dict]) -> str:
    if not entries:
        return "No indicator library entries yet."
    lines = ["Available indicator implementations (inline the code directly, do not import):"]
    for e in entries:
        sharpe_str = f" | best_sharpe={e['best_sharpe']:.2f}" if e.get("best_sharpe") else ""
        lines.append(
            f"- [{e['category']}] {e['display_name']} (spec_id={e['spec_id']}{sharpe_str}): "
            f"{e.get('description', '')[:120]}"
        )
        if e.get("best_params"):
            lines.append(f"  Best params: {e['best_params']}")
    return "\n".join(lines)


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
