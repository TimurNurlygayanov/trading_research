"""
Researcher agent: generates Python analysis code for a research task.

The output code is executed by the research Modal job (modal_jobs/research_job.py).
It must define run_analysis(data) -> dict with summary, key_findings, report_text.
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from db import supabase_client as db
from agents.prompts import RESEARCHER_SYSTEM, RESEARCHER_USER_TEMPLATE

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"


def generate_research_code(
    task_id: str,
    title: str,
    question: str,
    data_requirements: dict | None = None,
) -> str:
    """
    Call the LLM to generate Python analysis code for a research task.
    Returns the raw Python code string (no markdown fences).
    Logs spend to spend_log (associated with the task's strategy if known).
    """
    data_req = data_requirements or {}
    symbol = data_req.get("symbol", "EURUSD")
    timeframe = data_req.get("timeframe", "1h")
    start = data_req.get("start", "2018-01-01")
    end = data_req.get("end", "2026-01-01")

    user_msg = RESEARCHER_USER_TEMPLATE.format(
        title=title,
        question=question,
        task_type=_infer_task_type(question),
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )

    from agents.utils import call_claude
    response = call_claude(
        model=MODEL,
        max_tokens=8192,
        system=RESEARCHER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    usage = response.usage
    cost = _estimate_cost(MODEL, usage.input_tokens, usage.output_tokens)

    # Get strategy_id for spend logging (may be None for standalone tasks)
    task = db.get_research_task(task_id)
    strategy_id = task.get("created_by_strategy_id") if task else None
    db.log_spend("researcher", MODEL, usage.input_tokens, usage.output_tokens, cost, strategy_id)

    code = response.content[0].text.strip()
    # Strip markdown fences if the LLM wrapped in them
    if code.startswith("```python"):
        code = code[len("```python"):].lstrip("\n")
    if code.startswith("```"):
        code = code[3:].lstrip("\n")
    if code.endswith("```"):
        code = code[:-3].rstrip()

    log.info(
        f"Researcher: generated analysis code for task={task_id}, "
        f"tokens={usage.input_tokens}+{usage.output_tokens}, cost=${cost:.4f}"
    )
    return code


# ── Helpers ──────────────────────────────────────────────────────────────────

def _infer_task_type(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["indicator", "rsi", "macd", "ema", "atr", "bollinger"]):
        return "indicator_research"
    if any(w in q for w in ["predict", "correlat", "timeframe", "period", "regime"]):
        return "market_analysis"
    return "custom"


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = {
        "claude-haiku-4-5-20251001": (0.00025, 0.00125),
        "claude-sonnet-4-6": (0.003, 0.015),
        "claude-opus-4-6": (0.015, 0.075),
    }
    in_price, out_price = prices.get(model, (0.003, 0.015))
    return (input_tokens * in_price + output_tokens * out_price) / 1000
