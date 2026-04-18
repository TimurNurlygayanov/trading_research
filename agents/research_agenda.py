"""
Research agenda system.

Instead of hardcoding individual indicator tests, agendas define high-level
research hypotheses. Claude generates specific, testable sub-questions from each
agenda, which become research_task rows in the DB.

Adding a new research direction:
  1. Append a dict to RESEARCH_AGENDAS below.
  2. The queue worker auto-generates tasks from it once the static catalogue is done.

Agenda spec_type controls which code generation template runs:
  "entry_research"      — signal → forward return (existing template)
  "exit_research"       — fixed entry → test exit rules (existing template)
  "limit_order_research"— signal → pending order level → fill simulation (new template)
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Research agendas
# Each entry is a dict with: agenda_id, title, description, spec_type, n_tasks
# ---------------------------------------------------------------------------

RESEARCH_AGENDAS: list[dict] = [
    {
        "agenda_id": "exit_timing",
        "title": "Trade exit timing optimization",
        "spec_type": "exit_research",
        "n_tasks": 20,
        "description": """
We cannot reliably predict future price direction, but AFTER placing a trade we have
much more information available — the trade is live, price is moving, indicators are
updating in real time relative to our entry.

Research question: given an open trade (entered via EMA 20/50 cross as baseline),
what signals or conditions most reliably tell us whether to:
  (A) exit early and bank gains before the TP is hit,
  (B) cut a losing trade before the SL is hit,
  (C) hold through a drawdown because the trade will recover,
  (D) trail the stop to lock in profits without capping upside.

Specific aspects to investigate:
  - Which technical indicators (RSI, MACD histogram, ADX, BB position, Stochastic)
    predict good exit points vs "noise" that causes premature exits?
  - Does volume behavior (spike, dry-up) distinguish "move over" from "pause"?
  - Does price reaching a prior swing high/low signal exit?
  - Do specific candle patterns (engulfing, doji) at key levels predict exit?
  - How does exit timing interact with session (London close, NY close)?
  - Is partial exit (close 50% at 1R, trail 50%) better than full exit rules?
  - Does implied volatility (ATR ratio vs recent avg ATR) affect optimal exit?
  - When is a drawdown "normal pullback" vs "trade thesis broken"?

Each research task should test ONE specific exit hypothesis with a clear measurable outcome:
improvement in avg R/trade vs baseline fixed 2:1 R:R, and MFE capture ratio.
""",
    },
    {
        "agenda_id": "pending_orders",
        "title": "Pending / limit order entry strategies",
        "spec_type": "limit_order_research",
        "n_tasks": 20,
        "description": """
Market order entry (buying immediately at signal) accepts whatever price is available.
Limit and stop orders let us define a target price and wait for the market to come to us.

Research question: for common entry signals, does placing a PENDING ORDER at a
calculated level (rather than entering at market) improve trade outcomes?

Two pending order types:
  LIMIT orders (counter-trend entry): place a buy limit BELOW current price,
    waiting for a pullback to a better level before entering.
    Example: BB lower band touch → place buy limit 0.3×ATR below the touch bar low.
    We get a better entry price if filled, but we may miss the move entirely.

  STOP orders (momentum entry): place a buy stop ABOVE current price,
    entering only when price confirms the breakout direction.
    Example: EMA cross → place buy stop 0.1×ATR above the cross bar high.
    We enter only if price "proves" it's going our direction.

Specific aspects to investigate:
  - Which signals benefit most from limit entry (mean-reversion setups)?
  - Which signals benefit most from stop entry (breakout/momentum setups)?
  - What is the optimal distance from signal bar to set the pending order?
  - How long should we wait for fill before cancelling (1, 5, 10, 20 bars)?
  - Does fill rate (how often price reaches our level) meaningfully hurt edge?
  - Which price levels make best pending order targets: ATR-based offset,
    prior bar high/low, swing level, round number, BB band, session high/low?
  - Does the pending order approach filter out false breakouts?
  - Is there a session effect: pending orders set during Asian session and
    filled during London open?

Each task should test ONE specific pending order setup with fill rate + return quality
metrics that can be compared to immediate market entry.
""",
    },
    {
        "agenda_id": "regime_filter_quality",
        "title": "Regime detection: when to trade vs stay flat",
        "spec_type": "entry_research",
        "n_tasks": 15,
        "description": """
Many strategies work in trending markets but give back all profits in ranging markets
(and vice versa). A regime filter that correctly identifies the current market state
could dramatically improve any strategy's Sharpe by simply switching it off in unfavourable conditions.

Research question: which regime detection methods most reliably identify periods when
a specific strategy type (trend-following OR mean-reversion) will be profitable?

Regimes to investigate:
  - Trend strength: ADX > threshold (strong trend vs chop)
  - Volatility regime: ATR vs rolling average ATR (expansion = trending, contraction = ranging)
  - Price structure: higher highs + higher lows (trend) vs mixed structure (range)
  - MA slope: EMA angle (fast-rising = strong trend, flat = range)
  - BB width: wide bands = trending/volatile, narrow = range/squeeze
  - Volume regime: above-average volume confirms trend, below-average = indecision
  - Multi-timeframe: daily trend direction vs intraday signal
  - Time-based: certain sessions (London open, NY open) consistently trend vs range

For each regime filter, measure:
  - Hit rate improvement when regime is "on" vs regime is "off"
  - Percentage of time the filter is "on" (if it's off 90% of time, it's too restrictive)
  - False negative cost: how often does the filter block profitable trades?
  - Does combining two regime filters improve or over-restrict?
""",
    },
    {
        "agenda_id": "price_microstructure",
        "title": "Price microstructure and order flow signals",
        "spec_type": "entry_research",
        "n_tasks": 15,
        "description": """
Beyond standard indicators, price microstructure contains signals from the raw
bar data itself: candle anatomy, gaps, inside/outside bars, volume distribution.

Research question: which price microstructure patterns carry statistically significant
predictive power for forward returns on FX and Gold (OHLCV data only, no order book)?

Patterns to investigate:
  - Candle body/wick ratio: large body = conviction, large wicks = rejection
  - Inside bars (current high < previous high AND current low > previous low):
    compression often precedes breakout — which direction and when?
  - Outside bars (engulfing the previous bar): directional signal?
  - Gap between close and next open (overnight, or between 5m bars):
    do gaps get filled? How quickly?
  - Consecutive same-direction bars: 3 green candles in a row — continuation or reversal?
  - Bar size relative to ATR: very large bar vs very small bar — what follows?
  - High/low distance from open: does a bar that opens near its high tend to close lower?
  - Volume on up-bars vs down-bars (up-volume ratio): institutional buying/selling proxy
  - Price velocity: rate of change over 3-5 bars vs long-term average
  - Pivot candles: a bar whose high is higher than the prior 2 AND next 2 bars (fractal)

Each pattern should be tested as a standalone signal for forward returns, and also
combined with one volume or trend confirmation filter.
""",
    },
]


# ---------------------------------------------------------------------------
# Task generation via LLM
# ---------------------------------------------------------------------------

_AGENDA_TASK_GEN_PROMPT = """You are a quantitative FX/commodities researcher.

You will receive a high-level RESEARCH AGENDA describing a category of trading questions.
Your job: generate {n} specific, testable research tasks from this agenda.

AGENDA TITLE: {title}
AGENDA TYPE: {spec_type}
AGENDA DESCRIPTION:
{description}

EXISTING TASKS (do NOT duplicate these):
{existing_summary}

Generate exactly {n} UNIQUE tasks that each test ONE specific, measurable hypothesis
within this agenda. Avoid generic tasks — each must have clear measurable outcomes.

Each task description MUST contain all four sections:
  ENTRY: what event/signal triggers the trade entry or study window
  STOP: initial stop loss placement (e.g. "1.5×ATR below entry")
  EXIT RULE: the specific exit condition or rule being tested
  METRIC: what we measure to determine if this is better than the baseline

For limit_order_research tasks, also include:
  ORDER TYPE: "limit" (counter-trend, wait for pullback) or "stop" (momentum, wait for confirm)
  ORDER LEVEL: how to calculate the pending order price
  MAX_WAIT: max bars to wait for fill before cancelling

Respond ONLY with a JSON array. Each element:
{{
  "spec_id": "unique_snake_case_id",
  "indicator": "primary_indicator_or_concept",
  "category": "exit_timing | limit_order | regime | microstructure | combo",
  "title": "under 60 chars",
  "description": "ENTRY: ...\\nSTOP: ...\\nEXIT RULE or ORDER LEVEL: ...\\nMETRIC: ..."
}}

No markdown. No explanation. Raw JSON array only."""


def generate_agenda_tasks(agenda: dict, n: int, existing_titles: set[str]) -> list[dict]:
    """
    Ask Claude to generate n specific research tasks from a high-level agenda.
    Returns list of spec dicts ready for insertion as research_task rows.
    """
    from agents.utils import call_claude

    # Compact summary of existing tasks to avoid duplicates
    existing_sample = sorted(existing_titles)[:40]  # show up to 40
    existing_summary = (
        "\n".join(f"  - {t}" for t in existing_sample)
        if existing_sample else "  (none yet)"
    )
    if len(existing_titles) > 40:
        existing_summary += f"\n  ... and {len(existing_titles) - 40} more"

    prompt = _AGENDA_TASK_GEN_PROMPT.format(
        n=n,
        title=agenda["title"],
        spec_type=agenda["spec_type"],
        description=agenda["description"].strip(),
        existing_summary=existing_summary,
    )

    response = call_claude(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    try:
        proposals = json.loads(raw)
    except Exception as exc:
        # Try to salvage partial array
        last_brace = raw.rfind("}")
        if last_brace != -1:
            candidate = raw[: last_brace + 1].rstrip().rstrip(",") + "\n]"
            if not candidate.lstrip().startswith("["):
                candidate = "[" + candidate
            try:
                proposals = json.loads(candidate)
            except Exception:
                proposals = []
        else:
            proposals = []
        log.warning("agenda_task_gen_parse_failed agenda=%s error=%s salvaged=%s",
                    agenda["agenda_id"], exc, len(proposals))

    # Enrich each proposal with spec_type from the agenda
    for p in proposals:
        if isinstance(p, dict):
            p.setdefault("spec_type", agenda["spec_type"])

    log.info("agenda_task_gen_done agenda=%s generated=%s requested=%s",
             agenda["agenda_id"], len(proposals), n)
    return proposals


def process_all_agendas(limit_per_agenda: int | None = None) -> int:
    """
    For each agenda in RESEARCH_AGENDAS, generate and insert tasks that don't exist yet.
    Only generates tasks for agendas that still have room (existing < target n_tasks).

    Called by the queue worker when the static catalogue is exhausted.
    Returns total number of new tasks created.
    """
    from db import supabase_client as db

    existing = db.get_research_tasks(status="all", limit=5000, task_type="indicator_research")
    existing_titles: set[str] = {t["title"].lower() for t in existing}

    total_created = 0
    for agenda in RESEARCH_AGENDAS:
        agenda_id  = agenda["agenda_id"]
        target_n   = limit_per_agenda or agenda["n_tasks"]
        spec_type  = agenda["spec_type"]

        # Count how many tasks already exist from this agenda
        # (identified by agenda_id prefix in spec_id)
        already = sum(
            1 for t in existing
            if (t.get("research_spec") or {}).get("agenda_id") == agenda_id
        )
        remaining = target_n - already
        if remaining <= 0:
            log.info("agenda_saturated agenda=%s existing=%s target=%s",
                     agenda_id, already, target_n)
            continue

        log.info("agenda_generating agenda=%s need=%s", agenda_id, remaining)
        proposals = generate_agenda_tasks(agenda, n=remaining, existing_titles=existing_titles)

        created = 0
        for p in proposals:
            if not isinstance(p, dict) or not p.get("title"):
                continue
            title = f"[{agenda_id}] {p['title']}"
            if title.lower() in existing_titles:
                continue
            try:
                spec = {
                    "agenda_id":   agenda_id,
                    "spec_id":     p.get("spec_id", f"{agenda_id}_{created}"),
                    "spec_type":   spec_type,
                    "indicator":   p.get("indicator", "custom"),
                    "category":    p.get("category", "custom"),
                    "title":       p.get("title", ""),
                    "description": p.get("description", ""),
                }
                db.insert_research_task({
                    "type":          "indicator_research",
                    "title":         title,
                    "question":      spec["description"],
                    "research_spec": spec,
                    "status":        "pending",
                })
                existing_titles.add(title.lower())
                created += 1
            except Exception as exc:
                log.warning("agenda_task_insert_failed agenda=%s error=%s", agenda_id, exc)

        log.info("agenda_tasks_created agenda=%s created=%s", agenda_id, created)
        total_created += created

    return total_created
