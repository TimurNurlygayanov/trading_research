"""
Agent: Strategy Reviewer

Applies a user-requested code change to an existing backtesting.py strategy.
Called when the user is in the `awaiting_review` stage and submits a revision
request (e.g. "add a volume filter", "tighten the session to 8–17 UTC",
"replace EMA with VWAP").

Returns the complete updated Python code. The caller is responsible for
saving it to DB and re-dispatching the quick test.
"""
from __future__ import annotations

import logging
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"

_REVIEWER_SYSTEM = """You are an expert quantitative trading developer.
You will be given an existing backtesting.py Strategy subclass and a user's
revision request. Apply exactly what the user asks — nothing more.

Rules:
- Return ONLY the complete updated Python code. No markdown fences, no explanation.
- Preserve all existing logic except where the user explicitly asks to change it.
- Keep the same class name, same parameter names (unless renaming was requested).
- All self.I() calls in init(), never in next().
- Use pandas_ta for indicators (no manual calculations).
- Never access self.position.sl / self.position.tp — use trade.sl / trade.tp.
- Default session hours: start_hour=0, end_hour=23 (unless user specifies otherwise).
- If the user asks to "add X", add it cleanly without removing other logic.
- If the user asks to "remove X", remove only X.
- If the user asks to "change X to Y", change only X."""


def run_strategy_reviewer(strategy_id: str, user_message: str) -> dict[str, str]:
    """
    Apply user's revision request to the strategy code.
    Returns {"code": "<updated code>"} or {"error": "<message>"}.
    """
    from db import supabase_client as db
    from agents.utils import add_pipeline_note

    strategy = db.get_strategy(strategy_id)
    if not strategy:
        return {"error": f"Strategy {strategy_id} not found"}

    code = strategy.get("backtest_code", "")
    if not code:
        return {"error": "No backtest_code to revise"}

    description = strategy.get("hypothesis") or strategy.get("entry_logic") or ""

    user_msg = (
        f"ORIGINAL STRATEGY DESCRIPTION:\n{description[:400]}\n\n"
        f"CURRENT CODE:\n{code}\n\n"
        f"USER REVISION REQUEST:\n{user_message}"
    )

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=_REVIEWER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        usage = response.usage
        cost = (usage.input_tokens * 0.003 + usage.output_tokens * 0.015) / 1000
        try:
            db.log_spend("strategy_reviewer", MODEL, usage.input_tokens,
                         usage.output_tokens, cost, strategy_id)
        except Exception:
            pass

        updated = response.content[0].text.strip()

        # Strip markdown fences if present
        if updated.startswith("```python"):
            updated = updated[9:]
        elif updated.startswith("```"):
            updated = updated[3:]
        if updated.endswith("```"):
            updated = updated[:-3]
        updated = updated.strip()

        if not updated or "class" not in updated:
            log.warning("reviewer_invalid_output", strategy_id=strategy_id)
            return {"error": "LLM returned invalid code (no class found)"}

        # Save updated code, reset to implemented to trigger a new quick test
        db.update_strategy(strategy_id, {
            "backtest_code":  updated,
            "status":         "implemented",
            "modal_job_id":   None,
            "analysis_done":  False,   # re-run analyzer after the new quick test
            "error_log":      None,
        })
        add_pipeline_note(
            strategy_id,
            f"Code revised per user request: \"{user_message[:120]}\". "
            "Re-running quick test."
        )
        log.info("strategy_reviewer_done", strategy_id=strategy_id,
                 request=user_message[:60])
        return {"code": updated}

    except Exception as exc:
        log.error("strategy_reviewer_error", strategy_id=strategy_id, error=str(exc))
        return {"error": str(exc)}
