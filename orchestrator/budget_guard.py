"""
Budget guard: checks daily LLM spend before every agent call.
If daily spend >= MAX_DAILY_SPEND_USD, raises BudgetExceeded.
Called at the start of every agent that makes LLM calls.
"""
from __future__ import annotations

import os
from datetime import date

from db import supabase_client as db


class BudgetExceeded(Exception):
    pass


def check_budget(agent_name: str) -> None:
    """
    Raise BudgetExceeded if daily spend limit is reached.
    Call this at the start of any agent that uses LLM API.
    """
    max_spend = float(os.environ.get("MAX_DAILY_SPEND_USD", 8.0))
    today_spend = db.get_daily_spend()

    if today_spend >= max_spend:
        raise BudgetExceeded(
            f"Daily LLM budget exhausted: ${today_spend:.2f} spent today "
            f"(limit: ${max_spend:.2f}). Agent '{agent_name}' blocked. "
            f"Resets at midnight UTC."
        )


def get_remaining_budget() -> float:
    max_spend = float(os.environ.get("MAX_DAILY_SPEND_USD", 8.0))
    today_spend = db.get_daily_spend()
    return max(0.0, max_spend - today_spend)
