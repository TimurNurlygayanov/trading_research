"""
Agent: Code Fixer
Classifies backtest failures and attempts to repair code-level bugs using LLM.

Called by queue_worker when a strategy fails with a Python error (SyntaxError,
AttributeError, NameError, etc.) rather than a legitimate quality rejection
(bad Sharpe, leakage gate, OOS degradation, etc.).

Error classification:
  quality_rejection  → legitimate pipeline gate, do not retry
  infrastructure     → transient infra error, retry without code change
  code_bug           → Python error in strategy code, fix with LLM and retry
  unknown            → retry once without code change, then give up
"""
from __future__ import annotations

import logging
import os
import re
from typing import Literal

import anthropic
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"

ErrorClass = Literal["quality_rejection", "infrastructure", "code_bug", "unknown"]

# ── Error classification ──────────────────────────────────────────────────────

# These errors are intentional pipeline gates — never retry
_QUALITY_KEYWORDS = [
    "leakage check failed",
    "oos degradation",
    "walk-forward",
    "overfitting detected",
    "sharpe",
    "calmar",
    "unprofitable in recent",
    "correction limit reached",
    "corrections_applied",
    "not consistently profitable",
    "mean oos sharpe",
    "curve-fitted",
    "below minimum",
    "multi-timeframe validation failed",
]

# These are transient infra errors — retry without code change
_INFRA_KEYWORDS = [
    "no module named 'db'",
    "no module named 'agents'",
    "no module named 'backtest'",
    "no module named 'backtesting'",
    "timeouterror",
    "memoryerror",
    "sigkill",
    "killed",
    "container exited",
]

# Python runtime errors that indicate the generated strategy code is broken
_CODE_BUG_PATTERNS = [
    r"SyntaxError",
    r"IndentationError",
    r"NameError",
    r"AttributeError",
    r"TypeError",
    r"ImportError",
    r"ModuleNotFoundError",
    r"ValueError",
    r"KeyError",
    r"ZeroDivisionError",
    r"in <strategy>",          # exec() traceback marker
    r"No Strategy subclass found",
    r"strategy_classes",
    r"exec\(",
    r"compile\(",
]


def classify_error(error_log: str) -> ErrorClass:
    """
    Classify a strategy failure to decide how to respond.

    Returns one of: quality_rejection | infrastructure | code_bug | unknown
    """
    if not error_log:
        return "unknown"

    lower = error_log.lower()

    for kw in _QUALITY_KEYWORDS:
        if kw in lower:
            return "quality_rejection"

    for kw in _INFRA_KEYWORDS:
        if kw in lower:
            return "infrastructure"

    for pattern in _CODE_BUG_PATTERNS:
        if re.search(pattern, error_log):
            return "code_bug"

    return "unknown"


# ── LLM code repair ───────────────────────────────────────────────────────────

_FIX_SYSTEM = """You are an expert Python developer specialising in backtesting.py trading strategies.

You will be given:
1. Broken strategy code (a backtesting.py Strategy subclass)
2. The Python traceback / error message
3. The strategy description

Your task: return ONLY the fixed, complete Python code. No explanation, no markdown fences, no extra text — just the raw Python code starting with imports.

Rules you MUST follow:
- Use pandas_ta for all indicators (no manual RSI/ATR/EMA implementations)
- All indicator calls must be inside init() using self.I()
- next() must only read pre-computed arrays via [-1] or [-2] indexing
- No shift(-N), no bfill(), no fitting on full dataset
- The Strategy subclass must be at module level (not nested)
- Use backtesting.py Strategy API: self.buy(), self.sell(), self.position
- NEVER access self.position.sl or self.position.tp — Position has no .sl/.tp attributes.
  Set SL/TP only via self.buy(sl=, tp=) or self.sell(sl=, tp=).
  To modify stops on open trades: for trade in self.trades: trade.sl = new_value
- Preserve the original strategy logic — only fix the bug, don't redesign"""


def fix_strategy_code(
    code: str,
    error_log: str,
    strategy_description: str,
    strategy_id: str | None = None,
) -> str | None:
    """
    Use Claude Sonnet to fix a broken strategy.
    Returns the corrected code string, or None if the LLM can't fix it.
    """
    user_msg = (
        f"STRATEGY DESCRIPTION:\n{strategy_description[:500]}\n\n"
        f"ERROR:\n{error_log[:1000]}\n\n"
        f"BROKEN CODE:\n{code}"
    )

    try:
        from db import supabase_client as db  # noqa: import inside fn for testability
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=_FIX_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        usage = response.usage
        cost = (usage.input_tokens * 0.003 + usage.output_tokens * 0.015) / 1000
        try:
            db.log_spend("code_fixer", MODEL, usage.input_tokens, usage.output_tokens,
                         cost, strategy_id)
        except Exception:
            pass

        fixed = response.content[0].text.strip()

        # Strip markdown fences if LLM wrapped the code
        if fixed.startswith("```python"):
            fixed = fixed[9:]
        elif fixed.startswith("```"):
            fixed = fixed[3:]
        if fixed.endswith("```"):
            fixed = fixed[:-3]

        fixed = fixed.strip()
        if not fixed or "class" not in fixed:
            log.warning(f"code_fixer: LLM returned empty/invalid code for {strategy_id}")
            return None

        log.info(f"code_fixer: generated fix for {strategy_id} ({len(fixed)} chars)")
        return fixed

    except Exception as e:
        log.error(f"code_fixer: LLM call failed for {strategy_id}: {e}")
        return None
