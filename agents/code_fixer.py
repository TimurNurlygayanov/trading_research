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
    # Zero trades — entry logic never fires, session filter kills everything, or indicators are all NaN
    r"zero trades on all timeframes",
    r"0 trades on all timeframes",
    r"zero signals",
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
1. Strategy code (a backtesting.py Strategy subclass)
2. The error message or failure description
3. The original strategy description

Your task: return ONLY the fixed, complete Python code. No explanation, no markdown fences, no extra text.

═══════════════════════════════════
ZERO TRADES DIAGNOSIS CHECKLIST
(check these FIRST if error says "zero trades on all timeframes")
═══════════════════════════════════

1. SESSION FILTER TOO RESTRICTIVE
   Problem: `start_hour=7, end_hour=20` blocks too many bars on 4h/1d data,
   or the condition uses `< end_hour` which excludes the end_hour bar itself.
   Fix: Change defaults to `start_hour: int = 0` and `end_hour: int = 23` so the filter
   is effectively disabled with default params. The optimizer will find good session windows.
   The check in next() should be:
     hour = self.data.index[-1].hour
     if not (self.start_hour <= hour < self.end_hour): return

2. INDICATOR ALL NaN
   Problem: pandas_ta returns NaN for the first `length` bars, and if `length` is large
   relative to the dataset, most bars are NaN. Common with large EMA periods.
   Fix: Add a NaN guard before using any indicator in next():
     if np.isnan(self.ema[-1]) or np.isnan(self.atr[-1]): return
   Also reduce default periods (e.g. ema_period=50 not 200).

3. WRONG PANDAS_TA COLUMN NAME
   Problem: `_st["SUPERT_7_3.0"]` fails if the actual column is `SUPERT_7_3` (no trailing zero)
   or the period/multiplier don't match.
   Fix: Use dynamic column names built from the actual params:
     _st_col = f"SUPERT_{self.st_period}_{float(self.st_mult)}"
   Then verify the column exists: if _st_col not in _st.columns: raise KeyError(...)

4. ENTRY CONDITION USES [-1] INSTEAD OF [-2]
   Problem: using self.indicator[-1] (current unconfirmed bar) causes issues.
   Fix: Always use [-2] for the confirmed previous bar in entry signals.

5. LONG AND SHORT BOTH BLOCKED BY SAME CONDITION
   Problem: if `self.st_dir[-2] == 1` for long AND `self.st_dir[-2] == -1` for short,
   but the condition is never met because of a logic error.
   Fix: Add `else: self.position.close()` or remove conflicting conditions.

═══════════════════════════════════
GENERAL RULES
═══════════════════════════════════
- pandas_ta for all indicators (no manual RSI/ATR/EMA)
- All self.I() calls must be in init(), never in next()
- next() reads pre-computed arrays only via [-1] or [-2]
- No shift(-N), no bfill(), no fitting on full dataset
- NEVER access self.position.sl or self.position.tp (Position has no .sl/.tp)
  Set SL/TP only via self.buy(sl=..., tp=...) or self.sell(sl=..., tp=...)
  Modify open trade stops: for trade in self.trades: trade.sl = new_value
- Preserve the original strategy logic — only fix the bug, don't redesign
- Return 100% complete, runnable Python code"""


def fix_strategy_code(
    code: str,
    error_log: str,
    strategy_description: str,
    strategy_id: str | None = None,
    extra_context: str | None = None,
) -> str | None:
    """
    Use Claude Sonnet to fix a broken strategy.
    Returns the corrected code string, or None if the LLM can't fix it.

    extra_context: optional additional diagnostic info (e.g. multi-TF test results)
    """
    context_block = f"\nADDITIONAL CONTEXT:\n{extra_context}\n" if extra_context else ""
    user_msg = (
        f"STRATEGY DESCRIPTION:\n{strategy_description[:600]}\n\n"
        f"FAILURE REASON:\n{error_log[:1200]}\n"
        f"{context_block}\n"
        f"CODE TO FIX:\n{code}"
    )

    try:
        from db import supabase_client as db  # noqa: import inside fn for testability
        from agents.utils import call_claude
        response = call_claude(
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
