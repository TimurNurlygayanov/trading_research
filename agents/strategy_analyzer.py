"""
Agent: Strategy Analyzer

Runs after the multi-timeframe quick test. Analyses the actual trade-level data
to find systematic improvements:

  1. Session/time — which hours and days are profitable vs lossy
  2. Trade cap  — does limiting trades per day improve Sharpe?
  3. Regime     — LLM interprets patterns and suggests code-level patches

Outputs:
  - analysis_notes  (JSONB) stored in strategies table
  - Tighter Optuna search bounds written back to hyperparams
  - Optionally updated backtest_code (when max_daily_trades is missing or regime
    filter would materially improve the strategy)

Runs on Render (lightweight — just pandas + one LLM call). No Modal needed.
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for analysis interpretation


# ── Public entry point ────────────────────────────────────────────────────────

def run_strategy_analyzer(strategy_id: str) -> dict[str, Any]:
    """
    Load trades from DB, run all analyses, store findings, tighten Optuna bounds.
    Returns a summary dict.  Safe to call multiple times (idempotent writes).
    """
    from db import supabase_client as db
    from agents.utils import add_pipeline_note

    strategy = db.get_strategy(strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")

    trades_raw = strategy.get("quick_test_trade_records")
    if not trades_raw:
        log.info("strategy_analyzer_skip_no_trades", strategy_id=strategy_id)
        return {"skipped": True, "reason": "no quick_test_trade_records stored"}

    try:
        import pandas as pd
        import numpy as np

        trades = pd.DataFrame(trades_raw)
        if trades.empty or "PnL" not in trades.columns:
            return {"skipped": True, "reason": "trade data missing PnL column"}

        trades["EntryTime"] = pd.to_datetime(trades["EntryTime"], utc=True, errors="coerce")
        trades = trades.dropna(subset=["EntryTime", "PnL"])
        if len(trades) < 20:
            return {"skipped": True, "reason": f"too few trades for analysis ({len(trades)})"}

        # ── Run all three analyses ────────────────────────────────────────────
        session_findings  = _analyse_session(trades)
        tradecap_findings = _analyse_trade_cap(trades)
        code              = strategy.get("backtest_code") or ""
        description       = strategy.get("hypothesis") or strategy.get("entry_logic") or ""
        param_space       = strategy.get("hyperparams") or {}

        # ── LLM interpretation ────────────────────────────────────────────────
        llm_result = _llm_interpret(
            strategy_id=strategy_id,
            description=description,
            code=code,
            session=session_findings,
            tradecap=tradecap_findings,
        )

        # ── Build analysis_notes ──────────────────────────────────────────────
        analysis_notes = {
            "session":  session_findings,
            "tradecap": tradecap_findings,
            "llm":      llm_result,
        }

        # ── Apply improvements ────────────────────────────────────────────────
        updates: dict[str, Any] = {"analysis_notes": analysis_notes}
        improvements: list[str] = []

        # 1. Tighten start_hour / end_hour Optuna bounds
        new_param_space = dict(param_space) if isinstance(param_space, dict) else {}
        best_start = session_findings.get("best_start_hour")
        best_end   = session_findings.get("best_end_hour")
        session_sharpe_gain = session_findings.get("sharpe_improvement", 0)

        if best_start is not None and best_end is not None and session_sharpe_gain > 0.1:
            slack = 3  # ± hours Optuna can still explore around the found optimum
            new_param_space["start_hour"] = [
                "int",
                max(0, best_start - slack),
                min(best_start + slack, best_end - 1),
            ]
            new_param_space["end_hour"] = [
                "int",
                max(best_start + 1, best_end - slack),
                min(23, best_end + slack),
            ]
            improvements.append(
                f"Session filter: optimal window {best_start:02d}:00–{best_end:02d}:00 UTC "
                f"(Sharpe gain +{session_sharpe_gain:.3f}). "
                f"Optuna will search ±{slack}h around this."
            )

        # 2. Trade cap: if capping daily trades helps, tighten the Optuna bound
        best_cap         = tradecap_findings.get("best_daily_cap")
        tradecap_gain    = tradecap_findings.get("sharpe_improvement", 0)

        if best_cap is not None and tradecap_gain > 0.1:
            # Update Optuna bounds for max_daily_trades or max_daily_losses
            if "max_daily_trades" in code:
                new_param_space["max_daily_trades"] = ["int", 1, min(best_cap + 3, 10)]
            elif "max_daily_losses" in code:
                new_param_space["max_daily_losses"] = ["int", 1, min(best_cap + 2, 6)]
            improvements.append(
                f"Trade cap: max {best_cap} trades/day improves Sharpe by "
                f"+{tradecap_gain:.3f}. Optuna bound tightened."
            )

        if new_param_space != param_space:
            updates["hyperparams"] = new_param_space

        # 3. LLM code patch (only if LLM is confident and patch is non-empty)
        patch = (llm_result.get("code_patch") or "").strip()
        if patch and llm_result.get("confidence") == "high" and patch != code:
            updates["backtest_code"] = patch
            updates["status"] = "implemented"   # re-run quick test with improved code
            improvements.append(
                f"Code improved: {llm_result.get('key_finding', 'LLM applied code patch')}"
            )

        db.update_strategy(strategy_id, updates)

        note_lines = ["Strategy analysis complete."]
        if improvements:
            note_lines.append("Improvements applied:")
            note_lines.extend(f"  • {imp}" for imp in improvements)
        else:
            note_lines.append(
                "No significant improvements found "
                f"(session gain={session_sharpe_gain:.3f}, "
                f"cap gain={tradecap_gain:.3f}). "
                "Proceeding to full optimization with original params."
            )
        add_pipeline_note(strategy_id, "\n".join(note_lines))

        return {
            "improvements": improvements,
            "session": session_findings,
            "tradecap": tradecap_findings,
            "code_patched": "backtest_code" in updates,
        }

    except Exception as exc:
        log.error("strategy_analyzer_error", strategy_id=strategy_id, error=str(exc))
        return {"error": str(exc)}


# ── Analysis 1: Session / time-of-day ────────────────────────────────────────

def _analyse_session(trades) -> dict[str, Any]:
    """
    Find the contiguous hour window that maximises per-trade Sharpe.
    Returns best_start_hour, best_end_hour, and Sharpe improvement over all-day.
    """
    import numpy as np

    trades = trades.copy()
    trades["hour"] = trades["EntryTime"].dt.hour
    trades["dow"]  = trades["EntryTime"].dt.dayofweek   # 0=Mon

    baseline_sharpe = _trade_sharpe(trades["PnL"])

    # ── Hourly P&L profile ────────────────────────────────────────────────────
    hourly = (
        trades.groupby("hour")["PnL"]
        .agg(count="count", mean="mean", win_rate=lambda x: (x > 0).mean())
        .reset_index()
    )
    hourly_dict = hourly.set_index("hour")[["count", "mean", "win_rate"]].to_dict("index")
    # Fill missing hours with zeros
    hourly_full = {h: hourly_dict.get(h, {"count": 0, "mean": 0.0, "win_rate": 0.0})
                   for h in range(24)}

    # ── Day-of-week profile ───────────────────────────────────────────────────
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = (
        trades.groupby("dow")["PnL"]
        .agg(count="count", mean="mean", win_rate=lambda x: (x > 0).mean())
        .reset_index()
    )
    dow_dict = {
        dow_names[int(row["dow"])]: {
            "count": int(row["count"]),
            "mean":  round(float(row["mean"]), 4),
            "win_rate": round(float(row["win_rate"]), 3),
        }
        for _, row in dow.iterrows()
    }

    # ── Find best contiguous window ───────────────────────────────────────────
    # Slide windows of width 4–20 hours across the day; pick best Sharpe.
    best_sharpe  = baseline_sharpe
    best_start   = None
    best_end     = None
    MIN_TRADES   = max(10, len(trades) * 0.15)   # window must have ≥15% of all trades

    for width in range(4, 21):
        for start in range(24):
            hours_in_window = [(start + i) % 24 for i in range(width)]
            subset = trades[trades["hour"].isin(hours_in_window)]
            if len(subset) < MIN_TRADES:
                continue
            sh = _trade_sharpe(subset["PnL"])
            if sh > best_sharpe:
                best_sharpe = sh
                best_start  = start
                best_end    = (start + width) % 24

    return {
        "baseline_sharpe":   round(baseline_sharpe, 4),
        "best_start_hour":   best_start,
        "best_end_hour":     best_end,
        "best_sharpe":       round(best_sharpe, 4),
        "sharpe_improvement": round(best_sharpe - baseline_sharpe, 4),
        "hourly_profile":    {str(h): v for h, v in hourly_full.items()},
        "dow_profile":       dow_dict,
    }


# ── Analysis 2: Daily trade cap ───────────────────────────────────────────────

def _analyse_trade_cap(trades) -> dict[str, Any]:
    """
    Test max_daily_trades caps 1..10. Return best cap and Sharpe improvement.
    """
    trades = trades.copy()
    trades["date"] = trades["EntryTime"].dt.date

    baseline_sharpe = _trade_sharpe(trades["PnL"])
    daily_max = int(trades.groupby("date").size().max())

    results = {}
    for cap in range(1, min(11, daily_max + 1)):
        capped = trades.groupby("date").head(cap)
        if len(capped) < 10:
            continue
        sh = _trade_sharpe(capped["PnL"])
        results[cap] = round(sh, 4)

    if not results:
        return {"baseline_sharpe": round(baseline_sharpe, 4), "best_daily_cap": None,
                "sharpe_improvement": 0.0, "cap_results": {}}

    best_cap = max(results, key=results.get)
    best_sh  = results[best_cap]

    return {
        "baseline_sharpe":   round(baseline_sharpe, 4),
        "best_daily_cap":    best_cap if best_sh > baseline_sharpe else None,
        "best_sharpe":       round(best_sh, 4),
        "sharpe_improvement": round(best_sh - baseline_sharpe, 4),
        "cap_results":       results,
        "avg_trades_per_day": round(float(trades.groupby("date").size().mean()), 1),
        "max_trades_per_day": daily_max,
    }


# ── Analysis 3: LLM interpretation and optional code patch ───────────────────

_ANALYZER_SYSTEM = """You are an expert quantitative trading strategy analyst.

You receive:
1. Strategy description and code
2. Time-of-day performance profile (hourly mean P&L, win rate)
3. Day-of-week profile
4. Trade cap analysis (Sharpe at different max_daily_trades limits)

Your task: identify the single most impactful improvement and, if a code change
is needed, produce a complete corrected version of the code.

Rules:
- Only suggest improvements with clear statistical support (not random noise).
  A minimum of 20% Sharpe improvement is worth acting on.
- If the improvement is just tightening parameters (session hours, trade cap)
  and those params already exist in the code, set code_patch = null.
  The optimizer will find the right values.
- Only output code_patch when the strategy is MISSING a useful parameter or
  needs a NEW filter (e.g., max_daily_trades not in code, ATR regime filter).
- If outputting code_patch, return the COMPLETE fixed Python file — not a diff.
- Set confidence = "high" only if you are very certain the change will help
  and the code compiles correctly.

OUTPUT FORMAT (JSON only):
{
  "key_finding": "<1-2 sentence summary of the most important pattern found>",
  "improvement_type": "session_filter" | "trade_cap" | "code_change" | "none",
  "confidence": "high" | "medium" | "low",
  "reasoning": "<why this improvement should help — reference specific numbers>",
  "code_patch": "<complete corrected Python code, or null>"
}"""


def _llm_interpret(
    strategy_id: str,
    description: str,
    code: str,
    session: dict,
    tradecap: dict,
) -> dict[str, Any]:
    """Call Haiku to interpret analysis findings and optionally produce a code patch."""
    # Build a compact human-readable summary to pass to the LLM
    hourly = session.get("hourly_profile", {})
    top_hours = sorted(
        ((int(h), v["mean"]) for h, v in hourly.items() if v["count"] > 0),
        key=lambda x: x[1], reverse=True
    )[:6]
    worst_hours = sorted(
        ((int(h), v["mean"]) for h, v in hourly.items() if v["count"] > 0),
        key=lambda x: x[1]
    )[:6]
    cap_results = tradecap.get("cap_results", {})

    user_msg = f"""STRATEGY DESCRIPTION:
{description[:400]}

CURRENT CODE (key parts):
{code[:2000]}

TIME-OF-DAY ANALYSIS:
Baseline Sharpe (all hours): {session.get('baseline_sharpe')}
Best session window found: {session.get('best_start_hour')}:00–{session.get('best_end_hour')}:00 UTC (Sharpe: {session.get('best_sharpe')}, improvement: +{session.get('sharpe_improvement')})
Top profitable hours: {', '.join(f'{h:02d}:00 (mean PnL={v:.4f})' for h, v in top_hours)}
Worst hours: {', '.join(f'{h:02d}:00 (mean PnL={v:.4f})' for h, v in worst_hours)}

DAY-OF-WEEK:
{json.dumps(session.get('dow_profile', {}), indent=2)}

DAILY TRADE CAP ANALYSIS:
Avg trades/day: {tradecap.get('avg_trades_per_day')}, max: {tradecap.get('max_trades_per_day')}
Baseline Sharpe: {tradecap.get('baseline_sharpe')}
Best daily cap: {tradecap.get('best_daily_cap')} trades (Sharpe: {tradecap.get('best_sharpe')}, improvement: +{tradecap.get('sharpe_improvement')})
All cap results: {json.dumps(cap_results)}

Return JSON only."""

    try:
        from agents.utils import call_claude
        response = call_claude(
            model=MODEL,
            max_tokens=4096,
            system=_ANALYZER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        usage = response.usage
        cost = (usage.input_tokens * 0.00025 + usage.output_tokens * 0.00125) / 1000
        try:
            from db import supabase_client as db
            db.log_spend("strategy_analyzer", MODEL, usage.input_tokens,
                         usage.output_tokens, cost, strategy_id)
        except Exception:
            pass

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()
        return json.loads(raw)

    except Exception as exc:
        log.error("strategy_analyzer_llm_error", strategy_id=strategy_id, error=str(exc))
        return {"key_finding": f"LLM call failed: {exc}", "improvement_type": "none",
                "confidence": "low", "reasoning": "", "code_patch": None}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trade_sharpe(pnl_series) -> float:
    """Per-trade annualized Sharpe (same formula as engine._compute_trade_sharpe)."""
    import numpy as np
    pnl = pnl_series.dropna()
    if len(pnl) < 2:
        return 0.0
    std = float(pnl.std())
    if std == 0:
        return 0.0
    # Annualise assuming ~252 trading days, rough trades-per-year from count
    ann = max(len(pnl), 1) ** 0.5
    return float(pnl.mean() / std * ann)
