"""
Agent: Variation Planner

Given a seed strategy concept, generates N structurally diverse implementations
that explore the same underlying market edge from different angles.

Design philosophy:
  Step 1 — Decompose the seed into its fundamental components
            (what condition, what signal, what confirmation, what regime filter).
  Step 2 — Systematically explore the design space by varying each component
            independently using a concrete taxonomy of technical approaches.
  Step 3 — Inject knowledge base from past experiments so known-good approaches
            are weighted higher and known failures are avoided.

The variations are NOT parameter sweeps — they are architecturally different
strategies that all share the same core hypothesis.
"""
from __future__ import annotations

import json
import logging
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-opus-4-5"  # Best design decisions come from the strongest model

# Phase 1: decompose the seed idea into its components
_DECOMPOSE_SYSTEM = """You are a quantitative trading strategy researcher.

Analyze a seed strategy concept and decompose it into its fundamental components.

Return ONLY a JSON object with these fields:
{
  "core_hypothesis": "one sentence — the fundamental market inefficiency being exploited",
  "market_condition": "what overall market state the strategy needs (trending, ranging, volatile, quiet, etc.)",
  "entry_trigger": "the primary signal that fires an entry (price pattern, indicator cross, threshold breach, etc.)",
  "confirmation": "secondary evidence that validates the signal (volume, divergence, another indicator, etc.)",
  "regime_filter": "what market conditions to EXCLUDE (avoid trading when X)",
  "edge_summary": "why does this work? What behavioural or structural reason makes this edge exist?"
}

Be precise and technical. Name specific indicators where implied. No markdown, no explanation."""

# Phase 2: generate variations using the decomposition + taxonomy
_GENERATE_SYSTEM = """You are a quantitative trading strategy researcher.

You will receive:
1. A decomposed seed strategy (its components broken down explicitly)
2. A taxonomy of technical approaches for each component axis
3. Past experiment knowledge (what has worked and what has failed)

Your task: generate exactly {n} STRUCTURALLY DIFFERENT strategy implementations.

DESIGN SPACE TAXONOMY — use these to vary each axis:

MARKET CONDITION DETECTORS:
  - ATR-relative: current ATR vs N-bar ATR average (squeeze/expansion)
  - Bollinger Band width: bb_width < threshold → range; bb_width > threshold → trend
  - N-bar HH/LL: price range over last N bars relative to ATR
  - ADX threshold: ADX < 20 = range, ADX > 25 = trend
  - Keltner Channel: price inside/outside KC bands
  - VWAP deviation: distance from VWAP as % of price

ENTRY TRIGGER OPTIONS:
  - RSI extremes: RSI < 30 long / RSI > 70 short
  - MACD cross: histogram direction change or line crossover
  - Stochastic: %K crosses %D, with overbought/oversold filter
  - Bollinger touch: price touches outer band with reversal candle
  - EMA cross: fast EMA crosses slow EMA
  - Price action: engulfing, pin bar, inside bar breakout
  - Supertrend direction change: direction flips from -1 to +1
  - Kijun bounce: price crosses above/below Kijun line
  - VWAP cross: price crosses VWAP from below/above
  - Price levels & structure: entry near swing high/low (N-bar lookback), prior session high/low,
    round-number levels, pivot points (standard/Camarilla), weekly/daily open, or a manually
    identified S/R zone detected algorithmically (e.g. price clusters via rolling max/min)

CONFIRMATION SIGNAL OPTIONS:
  - Volume surge: bar volume > N× rolling average
  - RSI direction: RSI rising (for long) or falling (for short)
  - Momentum: ROC (rate of change) positive/negative
  - Second timeframe agreement: higher TF trend alignment
  - Divergence: price makes new extreme but indicator does not
  - MACD histogram: histogram in same direction as trade
  - ATR expansion: ATR rising (momentum confirmation)

REGIME FILTER OPTIONS:
  - Volatility: avoid trading when daily ATR > N× average (choppy)
  - Trend strength: require ADX > 20 OR ADX < 20 depending on strategy type
  - Session: only trade specific hours (London, NY, overlap)
  - Gap filter: skip if open gaps > N% from prior close
  - Volume: avoid low-volume periods (< 50% of average)

VARIATION RULES (must follow all):
  ✓ Each variation must use DIFFERENT combination of (entry trigger + confirmation + regime filter)
  ✓ Two variations must address the regime filter axis explicitly (one that restricts to trend, one to range)
  ✓ At least one variation must use volume as the confirmation signal
  ✓ At least one variation must use a price-action pattern (not just indicators) as entry trigger
  ✓ At least one variation must use price levels & structure as the primary entry condition
    (swing high/low, prior session high/low, pivot points, round numbers, or algorithmic S/R zones)
  ✗ Do NOT just change indicator parameters (RSI 14 → RSI 21 is NOT a variation)
  ✗ Do NOT repeat a combination already used by the seed strategy

PAST EXPERIMENT KNOWLEDGE (from this project's backtests — weight these heavily):
{knowledge}

OUTPUT FORMAT — JSON array of exactly {n} objects:
[
  {{
    "name": "CamelCaseStrategyName",
    "approach": "one-sentence label (e.g. 'Volume surge confirmation at range support')",
    "entry_trigger": "exact indicator/pattern used for entry",
    "confirmation": "what secondary signal confirms the entry",
    "regime_filter": "what market condition this variation requires or excludes",
    "description": "4-6 sentences. Name exact indicators and thresholds. Describe both long AND short entry triggers. The implementer will code exactly this — be precise."
  }},
  ...
]

Return ONLY the JSON array. No explanation, no markdown fences."""


def _format_knowledge(entries: list[dict]) -> str:
    """Format knowledge base entries into a compact summary for the prompt."""
    if not entries:
        return "No prior experiment data available yet."

    lines = []
    works = [e for e in entries if e.get("category") == "works"]
    fails = [e for e in entries if e.get("category") == "fails"]
    partial = [e for e in entries if e.get("category") in ("partial", "edge_case")]

    if works:
        lines.append("WHAT WORKS:")
        for e in works[:10]:
            sharpe = f" (Sharpe {e['sharpe_ref']:.2f})" if e.get("sharpe_ref") else ""
            lines.append(f"  + [{e.get('indicator','?')} / {e.get('timeframe','?')}]{sharpe}: {e['summary']}")

    if fails:
        lines.append("WHAT FAILS:")
        for e in fails[:10]:
            lines.append(f"  - [{e.get('indicator','?')} / {e.get('timeframe','?')}]: {e['summary']}")

    if partial:
        lines.append("PARTIAL / CONTEXT-DEPENDENT:")
        for e in partial[:5]:
            lines.append(f"  ~ [{e.get('indicator','?')} / {e.get('timeframe','?')}]: {e['summary']}")

    return "\n".join(lines) if lines else "No prior experiment data available yet."


def run_variation_planner(
    strategy_id: str,
    n_variations: int = 8,
) -> list[dict[str, str]]:
    """
    Generate n_variations - 1 additional strategy descriptions.
    The seed strategy (strategy_id) is variation #1 and runs independently.
    Returns list of {name, approach, description, ...} for variations #2..N.
    An empty list is returned on failure — caller falls back to single implementation.

    Two-phase process:
      Phase 1: Decompose seed into components (condition / trigger / confirmation / regime)
      Phase 2: Generate N-1 variations using taxonomy + knowledge base injection
    """
    from db import supabase_client as db
    from agents.utils import add_pipeline_note

    strategy = db.get_strategy(strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")

    n_extra = n_variations - 1  # root is #1; generate #2..N
    if n_extra <= 0:
        return []

    description = strategy.get("hypothesis") or strategy.get("entry_logic") or ""
    pre_filter  = strategy.get("pre_filter_notes") or {}

    suggestions = ""
    if isinstance(pre_filter, dict):
        suggestions = pre_filter.get("suggested_modifications", "") or ""
    elif isinstance(pre_filter, str):
        try:
            pf = json.loads(pre_filter)
            suggestions = pf.get("suggested_modifications", "") or ""
        except Exception:
            pass

    from agents.utils import call_claude
    total_input = 0
    total_output = 0

    # ── Phase 1: Decompose ─────────────────────────────────────────────────────
    try:
        decompose_msg = (
            f"SEED STRATEGY:\n{description}\n\n"
            + (f"PRE-FILTER SUGGESTIONS:\n{suggestions[:400]}" if suggestions else "")
        ).strip()

        r1 = call_claude(
            model=MODEL,
            max_tokens=1024,
            system=_DECOMPOSE_SYSTEM,
            messages=[{"role": "user", "content": decompose_msg}],
        )
        total_input  += r1.usage.input_tokens
        total_output += r1.usage.output_tokens

        raw1 = r1.content[0].text.strip()
        if raw1.startswith("```"):
            raw1 = raw1.split("```")[1]
            if raw1.startswith("json"):
                raw1 = raw1[4:]
            raw1 = raw1.strip().rstrip("```").strip()

        decomposed = json.loads(raw1)
        log.info("variation_planner_decompose_done", strategy_id=strategy_id,
                 hypothesis=decomposed.get("core_hypothesis", "?")[:80])
    except Exception as exc:
        log.warning("variation_planner_decompose_failed", error=str(exc))
        # Fall back to raw description if decomposition fails
        decomposed = {"core_hypothesis": description, "edge_summary": ""}

    # ── Phase 2: Load knowledge base + generate variations ────────────────────
    try:
        knowledge_entries = db.get_knowledge_summary(limit=50)
        knowledge_text = _format_knowledge(knowledge_entries)
    except Exception:
        knowledge_text = "No prior experiment data available yet."

    decompose_block = "\n".join(
        f"  {k.upper().replace('_', ' ')}: {v}"
        for k, v in decomposed.items()
        if v
    )

    user_msg = (
        f"SEED STRATEGY DECOMPOSITION:\n{decompose_block}\n\n"
        f"ORIGINAL DESCRIPTION (for reference):\n{description}\n\n"
        f"Generate exactly {n_extra} variations "
        f"(the seed itself is already variation #1 and will be implemented separately)."
    )

    try:
        r2 = call_claude(
            model=MODEL,
            max_tokens=6000,
            system=_GENERATE_SYSTEM.format(n=n_extra, knowledge=knowledge_text),
            messages=[{"role": "user", "content": user_msg}],
        )
        total_input  += r2.usage.input_tokens
        total_output += r2.usage.output_tokens

        cost = (total_input * 0.015 + total_output * 0.075) / 1000
        try:
            db.log_spend("variation_planner", MODEL, total_input, total_output,
                         cost, strategy_id)
        except Exception:
            pass

        raw2 = r2.content[0].text.strip()
        if raw2.startswith("```"):
            raw2 = raw2.split("```")[1]
            if raw2.startswith("json"):
                raw2 = raw2[4:]
            raw2 = raw2.strip().rstrip("```").strip()

        variations = json.loads(raw2)
        if not isinstance(variations, list):
            raise ValueError(f"Expected JSON list, got {type(variations).__name__}")

        # Trim to requested count in case LLM produced extras
        variations = variations[:n_extra]

        log.info("variation_planner_done", strategy_id=strategy_id,
                 n_generated=len(variations))

        add_pipeline_note(
            strategy_id,
            f"Campaign launched — {len(variations) + 1} variations will be explored:\n"
            f"  #1 (seed): {strategy.get('name', 'original')}\n"
            + "\n".join(
                f"  #{i+2}: {v.get('name','?')} — {v.get('approach','')}"
                for i, v in enumerate(variations)
            )
        )
        return variations

    except Exception as exc:
        log.error("variation_planner_error", strategy_id=strategy_id, error=str(exc))
        add_pipeline_note(strategy_id,
            f"Variation planner failed ({exc}). Proceeding with single implementation.")
        return []
