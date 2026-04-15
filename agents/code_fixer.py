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

    # Optimization regression: strategy produced trades in quick test but optimizer
    # found only 0-trade param combos.  Must be caught BEFORE quality_rejection because
    # the error message may also mention "walk-forward" or "sharpe".
    if "optimization_regression" in lower:
        return "code_bug"

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
OPTIMIZATION REGRESSION DIAGNOSIS
(check if error says "optimization_regression")
═══════════════════════════════════

The strategy DOES produce trades with its default params (proven by quick test),
but the Optuna optimizer found 0 trades when it varied the params.  The code is
not fundamentally broken — something goes wrong when params leave their defaults.

Common causes and fixes:

1. PARAM SPACE TOO WIDE — some param values disable the strategy entirely
   Example: ema_period=500 → all NaN for the first 500 bars of a short dataset.
   Fix: add a NaN guard in next() for every indicator:
     if np.isnan(self.ema[-1]): return
   And if the param class has a `hyperparams` field, tighten the ranges so the
   optimizer stays in a region where the indicator has data.

2. SESSION FILTER COLLAPSES WITH NON-DEFAULT HOURS
   Example: start_hour=21, end_hour=4 → wraps midnight, no bars match.
   Fix: change defaults to start_hour=0, end_hour=23 and add a wrap-around check:
     hour = self.data.index[-1].hour
     if self.start_hour <= self.end_hour:
         if not (self.start_hour <= hour < self.end_hour): return
     else:  # wraps midnight
         if not (hour >= self.start_hour or hour < self.end_hour): return

3. ENTRY THRESHOLD PARAM NEVER MET
   Example: rsi_buy=20, rsi_sell=80 with the optimizer pushing them to 5/95 →
   RSI never actually reaches those extremes in training data.
   Fix: clamp extreme thresholds in next() or tighten the param ranges.

4. CONFLICTING LONG/SHORT CONDITIONS WITH PARAM INTERACTION
   Example: long requires fast_ema > slow_ema AND rsi > rsi_buy, but optimizer
   sets rsi_buy so high that the combined condition never fires.
   Fix: relax conditions to use OR logic, or reduce compounding filters.

The error context will include the failed optimized params and quick-test results.
Use them to pinpoint which param caused 0 trades.

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


_OPT_ANALYSIS_SYSTEM = """You are an expert in quantitative trading strategy optimization.

You will be given:
1. Quick-test results: how the strategy performed with DEFAULT params across timeframes
2. Optimization failure reason: why Optuna + walk-forward rejected the strategy
3. Current parameter space (hyperparams): what the optimizer was allowed to vary

Your task: decide if the failure is a FIXABLE optimization problem or a GENUINE rejection.

FIXABLE means: the strategy has a real edge (proven by quick test) but the optimization
was set up in a way that caused overfitting, instability, or parameter interaction problems.
A good rule of thumb — if quick test shows Sharpe ≥ 0.3 and ≥ 10 trades on any timeframe,
the edge is real.  The optimizer may have simply searched too wide a space.

GENUINE REJECTION means: the edge doesn't generalize — quick test results are marginal,
or walk-forward OOS Sharpes are negative across ALL folds, or recent data is strongly negative.

═══════════════════════════════════════
IF FIXABLE: simplify the param space
═══════════════════════════════════════
Return a minimal hyperparams dict with ≤ 3-4 params.  Rules:
- Keep only the 2-3 most impactful params (main period length, key signal threshold)
- Remove secondary params (TP multipliers, secondary filters, session hours unless critical)
- Narrow int ranges to roughly 3x the default (default=14 → [7, 30] not [3, 200])
- Narrow float ranges to ≤ 2x the default
- Use categorical for natural breakpoints instead of wide continuous ranges
Format: {"param": ["int", min, max], "other": ["float", lo, hi], ...}

═══════════════════════════════════════
IF GENUINE REJECTION: explain clearly
═══════════════════════════════════════
The explanation will be shown as a pipeline note, so be concise and specific:
what in the optimization results proves the edge doesn't exist.

OUTPUT — return ONLY valid JSON, one of:
  {"action": "simplify", "hyperparams": {...}}
  {"action": "reject", "reason": "..."}
No explanation outside the JSON."""


def analyze_optimization_failure(
    strategy: dict,
    error_log: str,
) -> dict | None:
    """
    Analyze why a promising strategy failed optimization.

    Returns one of:
      {"action": "simplify", "hyperparams": {...}}  — caller should update DB + retry
      {"action": "reject", "reason": "..."}         — confirmed genuine rejection
      None                                           — LLM call failed
    """
    import json as _json

    quick_test = strategy.get("quick_test_all_timeframes") or {}
    hyperparams = strategy.get("hyperparams") or {}

    qt_lines = []
    for tf, m in quick_test.items():
        if isinstance(m, dict) and "error" not in m:
            qt_lines.append(
                f"  {tf}: trades={m.get('trades', 0)}, "
                f"sharpe={m.get('sharpe', 0):.3f}, "
                f"win_rate={m.get('win_rate', 0):.0%}"
            )

    user_msg = (
        "QUICK TEST RESULTS (default params):\n"
        + ("\n".join(qt_lines) if qt_lines else "  (no data)")
        + f"\n\nOPTIMIZATION FAILURE REASON:\n{error_log[:800]}"
        + f"\n\nCURRENT PARAM SPACE:\n{_json.dumps(hyperparams, indent=2)}"
        + f"\n\nSTRATEGY DESCRIPTION:\n"
        + (strategy.get("hypothesis") or strategy.get("entry_logic") or "")[:400]
    )

    try:
        from agents.utils import call_claude
        import db.supabase_client as _db

        response = call_claude(
            model=MODEL,
            max_tokens=1024,
            system=_OPT_ANALYSIS_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        usage = response.usage
        cost = (usage.input_tokens * 0.003 + usage.output_tokens * 0.015) / 1000
        try:
            _db.log_spend("optimization_analyst", MODEL,
                          usage.input_tokens, usage.output_tokens,
                          cost, strategy.get("id"))
        except Exception:
            pass

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

        result = _json.loads(raw)
        if result.get("action") not in ("simplify", "reject"):
            log.warning("optimization_analyst_bad_action strategy_id=%s", strategy.get("id"))
            return None
        return result

    except Exception as exc:
        log.error("optimization_analyst_failed strategy_id=%s error=%s",
                  strategy.get("id"), exc)
        return None


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
            max_tokens=3000,
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
