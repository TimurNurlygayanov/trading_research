"""
Agent: Indicator Researcher

Systematically tests all pandas_ta indicators plus price structure concepts
(swing levels, session H/L, pivots, round numbers) and writes findings into
the knowledge_base table.  Future strategy generation uses this knowledge via
get_knowledge_summary() in variation_planner and pre_filter.

Public interface:
  generate_research_tasks(limit)   →  int   (create pending research_task rows)
  run_indicator_research(task_id, cache_dir)  →  dict  (called from Modal)

Analysis method: statistical forward-return test.
  For each indicator signal (+1 long / -1 short):
    measure 5-bar and 20-bar forward returns,
    compute hit-rate and t-statistic,
    categorise as works / fails / partial.
  No full backtesting.py strategy needed — pure pandas analysis.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"   # generates code AND interprets results

# Asset/timeframe combos to test each indicator on
_TEST_ASSETS = [("EURUSD", "1h"), ("EURUSD", "4h"), ("XAUUSD", "1h")]

# ---------------------------------------------------------------------------
# Indicator spec catalogue
# ---------------------------------------------------------------------------

INDICATOR_SPECS: list[dict[str, str]] = [
    # ── Momentum ──────────────────────────────────────────────────────────────
    {
        "spec_id": "RSI_mean_reversion",
        "indicator": "RSI", "category": "momentum",
        "title": "RSI mean-reversion (oversold/overbought cross)",
        "description": (
            "14-period RSI. Long entry: RSI crosses above 30 from below (oversold bounce). "
            "Short entry: RSI crosses below 70 from above (overbought reversal)."
        ),
    },
    {
        "spec_id": "RSI_trend",
        "indicator": "RSI", "category": "momentum",
        "title": "RSI trend-following (50-line cross)",
        "description": (
            "14-period RSI. Long entry: RSI crosses above 50 from below. "
            "Short entry: RSI crosses below 50 from above. Trend-following interpretation."
        ),
    },
    {
        "spec_id": "MACD_signal_cross",
        "indicator": "MACD", "category": "momentum",
        "title": "MACD signal-line cross (12, 26, 9)",
        "description": (
            "MACD(12,26,9). Long: MACD line crosses above signal line. "
            "Short: MACD line crosses below signal line."
        ),
    },
    {
        "spec_id": "MACD_histogram",
        "indicator": "MACD", "category": "momentum",
        "title": "MACD histogram direction flip",
        "description": (
            "MACD(12,26,9) histogram. Long: histogram flips from negative to positive "
            "(hist[t-1] < 0 and hist[t] >= 0). Short: histogram flips from positive to negative. "
            "Fires one bar earlier than the signal-line cross."
        ),
    },
    {
        "spec_id": "STOCH_cross",
        "indicator": "STOCH", "category": "momentum",
        "title": "Stochastic %K oversold/overbought cross (14, 3, 3)",
        "description": (
            "Stochastic(14,3,3). Long: %K crosses above 20 from below (oversold bounce). "
            "Short: %K crosses below 80 from above (overbought reversal)."
        ),
    },
    {
        "spec_id": "WILLR_cross",
        "indicator": "WILLR", "category": "momentum",
        "title": "Williams %R oversold/overbought cross (14)",
        "description": (
            "Williams %R(14). Long: crosses above -80 from below. "
            "Short: crosses below -20 from above."
        ),
    },
    {
        "spec_id": "CCI_cross",
        "indicator": "CCI", "category": "momentum",
        "title": "CCI ±100 level cross (20)",
        "description": (
            "CCI(20). Long: crosses above -100 from below. "
            "Short: crosses below +100 from above."
        ),
    },
    {
        "spec_id": "ROC_zero",
        "indicator": "ROC", "category": "momentum",
        "title": "ROC zero-line cross — momentum shift (14)",
        "description": (
            "14-period Rate of Change. Long: ROC crosses above 0. "
            "Short: ROC crosses below 0. Simple momentum-direction signal."
        ),
    },
    # ── Trend ─────────────────────────────────────────────────────────────────
    {
        "spec_id": "EMA_20_50",
        "indicator": "EMA_cross", "category": "trend",
        "title": "EMA 20/50 crossover",
        "description": (
            "EMA(20) and EMA(50). Long: EMA20 crosses above EMA50. "
            "Short: EMA20 crosses below EMA50."
        ),
    },
    {
        "spec_id": "EMA_50_200",
        "indicator": "EMA_cross", "category": "trend",
        "title": "EMA 50/200 crossover (golden / death cross)",
        "description": (
            "EMA(50) and EMA(200). Long: EMA50 crosses above EMA200. "
            "Short: EMA50 crosses below EMA200. Long-term trend signal."
        ),
    },
    {
        "spec_id": "SUPERTREND",
        "indicator": "SUPERTREND", "category": "trend",
        "title": "SuperTrend direction flip (7, 3.0)",
        "description": (
            "SuperTrend(7, 3.0). Long: direction flips from -1 to +1. "
            "Short: direction flips from +1 to -1."
        ),
    },
    {
        "spec_id": "ADX_DI_cross",
        "indicator": "ADX", "category": "trend",
        "title": "ADX DI+/DI- cross with ADX > 20 filter (14)",
        "description": (
            "ADX(14), DI+(14), DI-(14). Long: DI+ crosses above DI- AND ADX > 20. "
            "Short: DI- crosses above DI+ AND ADX > 20. Directional movement system."
        ),
    },
    {
        "spec_id": "PSAR_flip",
        "indicator": "PSAR", "category": "trend",
        "title": "Parabolic SAR flip (step=0.02, max=0.2)",
        "description": (
            "Parabolic SAR(0.02, 0.2). Long: price crosses above PSAR (was below, now above). "
            "Short: price crosses below PSAR (was above, now below)."
        ),
    },
    # ── Volatility ─────────────────────────────────────────────────────────────
    {
        "spec_id": "BBANDS_bounce",
        "indicator": "BBANDS", "category": "volatility",
        "title": "Bollinger Bands mean-reversion bounce (20, 2.0)",
        "description": (
            "BBands(20, 2.0). Long: price closes below lower band. "
            "Short: price closes above upper band. Mean-reversion signal."
        ),
    },
    {
        "spec_id": "BBANDS_breakout",
        "indicator": "BBANDS", "category": "volatility",
        "title": "Bollinger Bands breakout after inside period (20, 2.0)",
        "description": (
            "BBands(20, 2.0). Long: price closes above upper band after 3+ consecutive bars "
            "where price was inside the bands. Short: closes below lower band after 3+ inside bars."
        ),
    },
    {
        "spec_id": "BBANDS_squeeze",
        "indicator": "BBANDS", "category": "volatility",
        "title": "Bollinger Bands squeeze-release breakout",
        "description": (
            "BBands(20, 2.0). Band width = (upper - lower) / middle. "
            "Squeeze: current width < its 20-bar rolling mean. "
            "Signal: first bar where squeeze ends AND price closes outside the bands — "
            "long if above upper, short if below lower."
        ),
    },
    {
        "spec_id": "KC_bounce",
        "indicator": "KC", "category": "volatility",
        "title": "Keltner Channel bounce (20, 2.0 ATR)",
        "description": (
            "Keltner Channel(20, 2.0). Long: price touches or closes below lower KC band. "
            "Short: price touches or closes above upper KC band."
        ),
    },
    {
        "spec_id": "ATR_expansion",
        "indicator": "ATR", "category": "volatility",
        "title": "ATR expansion breakout — volatility surge with direction (14)",
        "description": (
            "ATR(14). Long: ATR > 1.5 × its 14-bar rolling mean AND price closes above "
            "its 20-bar rolling high. Short: ATR expands AND price closes below 20-bar rolling low."
        ),
    },
    # ── Volume ────────────────────────────────────────────────────────────────
    {
        "spec_id": "OBV_ema_cross",
        "indicator": "OBV", "category": "volume",
        "title": "OBV vs its EMA crossover (20)",
        "description": (
            "OBV and its 20-bar EMA. Long: OBV crosses above its EMA. "
            "Short: OBV crosses below its EMA. Volume flow direction change."
        ),
    },
    {
        "spec_id": "CMF_zero",
        "indicator": "CMF", "category": "volume",
        "title": "Chaikin Money Flow zero-line cross (20)",
        "description": (
            "CMF(20). Long: CMF crosses above 0. Short: CMF crosses below 0. "
            "Accumulation vs distribution."
        ),
    },
    {
        "spec_id": "MFI_cross",
        "indicator": "MFI", "category": "volume",
        "title": "Money Flow Index oversold/overbought cross (14)",
        "description": (
            "MFI(14). Long: MFI crosses above 20 from below. "
            "Short: MFI crosses below 80 from above. Volume-weighted RSI signal."
        ),
    },
    {
        "spec_id": "VWAP_cross",
        "indicator": "VWAP", "category": "volume",
        "title": "VWAP daily cross",
        "description": (
            "Daily VWAP (reset each day using pandas_ta). "
            "Long: price crosses above VWAP. Short: price crosses below VWAP."
        ),
    },
    # ── Price structure ────────────────────────────────────────────────────────
    {
        "spec_id": "swing_level_bounce",
        "indicator": "swing_levels", "category": "structure",
        "title": "Swing level bounce (20-bar local extremes)",
        "description": (
            "Identify swing lows as 20-bar local minimums (argrelextrema or rolling min check). "
            "Long: price comes within 0.3% of a recent swing low AND next bar closes above it. "
            "Short: mirror for swing highs. Use the most recent swing level within 50 bars."
        ),
    },
    {
        "spec_id": "prior_session_hl",
        "indicator": "prior_session_hl", "category": "structure",
        "title": "Prior session high/low breakout",
        "description": (
            "Compute prior day's high and low using daily resampling. "
            "Long: price closes above prior day's high. "
            "Short: price closes below prior day's low."
        ),
    },
    {
        "spec_id": "round_numbers",
        "indicator": "round_numbers", "category": "structure",
        "title": "Round number support/resistance bounce",
        "description": (
            "Round 100-pip levels (e.g. 1.0800, 1.0900 for EURUSD). "
            "Detect the nearest round level = round(price, -2) / 10000 for 4-decimal pairs. "
            "Long: price comes within 0.1% below a round level and closes above it. "
            "Short: price comes within 0.1% above and closes below it."
        ),
    },
    {
        "spec_id": "pivot_points",
        "indicator": "pivot_points", "category": "structure",
        "title": "Daily pivot point bounce (standard)",
        "description": (
            "Daily standard pivot: P=(H+L+C)/3, S1=2P-H, R1=2P-L. "
            "Compute using prior day OHLC, apply to current day bars. "
            "Long: price bounces off S1 (touches S1 ±0.1% then closes above S1). "
            "Short: price bounces off R1 (touches R1 ±0.1% then closes below R1)."
        ),
    },
]

_SPEC_MAP = {s["spec_id"]: s for s in INDICATOR_SPECS}


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_research_tasks(limit: int = len(INDICATOR_SPECS)) -> int:
    """
    Create research_task rows for all unresearched indicator specs.
    Idempotent — skips specs that already have a matching research_task.
    Returns count of new tasks created.
    """
    from db import supabase_client as db

    existing = db.get_research_tasks(status="all", limit=500, task_type="indicator_research")
    existing_titles = {t["title"] for t in existing}

    created = 0
    for spec in INDICATOR_SPECS[:limit]:
        title = f"[Indicator] {spec['title']}"
        if title in existing_titles:
            continue
        try:
            db.insert_research_task({
                "type": "indicator_research",
                "title": title,
                "question": spec["description"],
                "research_spec": spec,
                "status": "pending",
            })
            created += 1
            log.info("indicator_task_created", spec_id=spec["spec_id"])
        except Exception as exc:
            log.warning("indicator_task_insert_failed", spec_id=spec["spec_id"], error=str(exc))

    log.info("generate_research_tasks_done", created=created,
             already_existed=len(INDICATOR_SPECS) - created)
    return created


# ---------------------------------------------------------------------------
# Analysis pipeline (runs on Modal)
# ---------------------------------------------------------------------------

def run_indicator_research(task_id: str, cache_dir: str = "/ohlcv_cache") -> dict:
    """
    Entry point called from Modal.
    1. Generates pandas analysis code for the indicator spec.
    2. Runs it on each test asset/TF combo.
    3. Interprets combined results → knowledge_base entries.
    4. Updates research_task to done.
    """
    import traceback

    from db import supabase_client as db

    task = db.get_research_task(task_id)
    if not task:
        raise ValueError(f"Research task {task_id} not found")

    spec = task.get("research_spec") or {}
    if not spec:
        raise ValueError(f"Task {task_id} has no research_spec — use run_research_task instead")

    try:
        db.update_research_task(task_id, {"status": "running"})

        # 1. Generate analysis code once (shared across all combos)
        code = _generate_analysis_code(spec["indicator"], spec["description"])

        # 2. Run on each test combo
        results_by_combo: dict[str, Any] = {}
        for symbol, tf in _TEST_ASSETS:
            df = _load_data(symbol, tf, cache_dir)
            if df is None or len(df) < 200:
                continue
            try:
                result = _run_analysis(code, df)
                if result and result.get("fwd_5", {}).get("count", 0) >= 10:
                    results_by_combo[f"{symbol}_{tf}"] = result
                    log.info("combo_done", indicator=spec["indicator"],
                             symbol=symbol, tf=tf, count=result["fwd_5"]["count"])
            except Exception as exc:
                log.warning("combo_failed", indicator=spec["indicator"],
                            symbol=symbol, tf=tf, error=str(exc))

        if not results_by_combo:
            raise ValueError(f"No successful analysis for {spec['indicator']} — check generated code")

        # 3. Interpret results → knowledge entries
        knowledge_entries = _interpret_and_save(spec, results_by_combo, task_id, db)

        # 4. Mark done
        summary = _build_summary(spec, results_by_combo, knowledge_entries)
        db.update_research_task(task_id, {
            "status": "done",
            "modal_job_id": None,
            "result_summary": summary,
            "key_findings": [
                {"finding": e["summary"], "confidence": 0.8}
                for e in knowledge_entries
            ],
            "generated_code": code,
            "error_log": None,
        })

        return {
            "passed": True,
            "task_id": task_id,
            "indicator": spec["indicator"],
            "combos_tested": len(results_by_combo),
            "knowledge_entries": len(knowledge_entries),
        }

    except Exception as exc:
        tb = traceback.format_exc()
        db.update_research_task(task_id, {
            "status": "failed",
            "modal_job_id": None,
            "error_log": f"{type(exc).__name__}: {exc}\n{tb[:600]}",
        })
        raise


# ---------------------------------------------------------------------------
# Step 1: generate analysis code via LLM
# ---------------------------------------------------------------------------

_CODE_GEN_SYSTEM = r"""You write Python indicator analysis functions.

Given an indicator description, write ONLY a function `analyze_indicator(df)`.

The function MUST:
1. Accept a DataFrame with columns: Open, High, Low, Close, Volume (title-cased)
2. Compute the indicator(s) using pandas_ta
3. Set df['signal'] = +1 (long), -1 (short), or 0 (no signal)
   — only mark signal on the bar where the condition first becomes true
   — use .shift(1) to check the previous bar's value for crosses
4. Compute forward returns and return stats dict exactly as shown below

REQUIRED RETURN FORMAT (do not change the stats block):
```python
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy import stats as _sp

def analyze_indicator(df: pd.DataFrame) -> dict:
    df = df.copy()
    df.columns = [c.title() for c in df.columns]

    # === YOUR INDICATOR CODE HERE ===
    # e.g. df.ta.rsi(length=14, append=True)


    # === YOUR SIGNAL CODE HERE ===
    df['signal'] = 0
    # e.g. df.loc[(df['RSI_14'] < 30) & (df['RSI_14'].shift(1) >= 30), 'signal'] = 1


    # === STATS (do not modify) ===
    df['fwd_5']  = df['Close'].pct_change(5).shift(-5)
    df['fwd_20'] = df['Close'].pct_change(20).shift(-20)
    longs  = df[df['signal'] ==  1].dropna(subset=['fwd_5','fwd_20'])
    shorts = df[df['signal'] == -1].dropna(subset=['fwd_5','fwd_20'])
    rets_5  = pd.concat([longs['fwd_5'],  -shorts['fwd_5']]).dropna()
    rets_20 = pd.concat([longs['fwd_20'], -shorts['fwd_20']]).dropna()
    def _stat(r):
        if len(r) < 10:
            return dict(count=int(len(r)), hit_rate=None, avg_return=None, tstat=None, pval=None)
        t, p = _sp.ttest_1samp(r.values, 0)
        return dict(count=int(len(r)), hit_rate=float((r>0).mean()),
                    avg_return=float(r.mean()), tstat=float(t), pval=float(p))
    return dict(fwd_5=_stat(rets_5), fwd_20=_stat(rets_20),
                long_count=int(len(longs)), short_count=int(len(shorts)))
```

Return ONLY the complete Python function, no markdown fences, no explanation."""


def _generate_analysis_code(indicator: str, description: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=_CODE_GEN_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"INDICATOR: {indicator}\n"
                f"SIGNAL LOGIC: {description}\n\n"
                "Write analyze_indicator(df) implementing exactly this signal logic."
            ),
        }],
    )
    code = response.content[0].text.strip()
    # Strip markdown fences if LLM adds them anyway
    if code.startswith("```"):
        code = code.split("```")[1]
        if code.startswith("python"):
            code = code[6:]
        code = code.strip().rstrip("`").strip()

    _log_spend("indicator_code_gen", response.usage, strategy_id=None)
    return code


# ---------------------------------------------------------------------------
# Step 2: load data
# ---------------------------------------------------------------------------

def _load_data(symbol: str, tf: str, cache_dir: str):
    """Load OHLCV DataFrame from cache or fetch if missing."""
    import os as _os
    import pandas as pd

    cache_file = f"{cache_dir}/{symbol}_{tf}.parquet"
    if _os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    # Fallback: fetch live
    try:
        from backtest.data_fetcher import fetch_ohlcv
        df = fetch_ohlcv(symbol, tf, start="2015-01-01", end="2026-12-31")
        return df
    except Exception as exc:
        log.warning("data_load_failed", symbol=symbol, tf=tf, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Step 3: execute analysis code
# ---------------------------------------------------------------------------

def _run_analysis(code: str, df) -> dict | None:
    """Execute generated analyze_indicator(df) and return stats dict."""
    import pandas as pd
    import numpy as np
    import pandas_ta as ta
    from scipy import stats as _sp

    namespace = {
        "pd": pd, "np": np, "ta": ta, "_sp": _sp,
        "__builtins__": __builtins__,
    }
    exec(compile(code, "<indicator_analysis>", "exec"), namespace)

    if "analyze_indicator" not in namespace:
        raise ValueError("Generated code did not define analyze_indicator(df)")

    result = namespace["analyze_indicator"](df.copy())
    if not isinstance(result, dict):
        return None
    return result


# ---------------------------------------------------------------------------
# Step 4: interpret results → knowledge_base entries
# ---------------------------------------------------------------------------

_INTERP_SYSTEM = """You analyze trading indicator statistical test results and write knowledge base entries.

Given forward-return statistics for a trading indicator signal across multiple asset/timeframe combinations,
produce a list of knowledge entries — one per combination that has enough data (count >= 20).

Output JSON array:
[
  {
    "combo": "EURUSD_1h",
    "category": "works" | "fails" | "partial",
    "summary": "2-3 sentence summary: what the signal achieves, what conditions it works in, caveats."
  },
  ...
]

CATEGORIZATION RULES (apply per combo):
  "works":   hit_rate_5 >= 0.54  AND abs(tstat_5) >= 1.8  AND count >= 20
  "fails":   hit_rate_5 <= 0.46  AND tstat_5     <= -1.8  AND count >= 20
  "partial": everything else (moderate signal, mixed results, or borderline stats)

SUMMARY GUIDANCE:
  - Include the hit rate and t-stat in the summary (be specific)
  - Mention whether long or short side drives the edge if asymmetric
  - For "partial": note what conditions might improve it
  - Be honest about noise — don't overstate weak signals

Return ONLY the JSON array. No explanation."""


def _interpret_and_save(spec: dict, results_by_combo: dict, task_id: str, db) -> list[dict]:
    """Call LLM to interpret combined results, save to knowledge_base, return entries."""
    # Format results table for LLM
    lines = [f"INDICATOR: {spec['title']}", ""]
    for combo, r in results_by_combo.items():
        f5  = r.get("fwd_5", {})
        f20 = r.get("fwd_20", {})
        lines.append(f"  {combo}:")
        lines.append(f"    signals: {r.get('long_count',0)} long + {r.get('short_count',0)} short = {f5.get('count','?')} combined")
        lines.append(f"    5-bar:  hit_rate={_pct(f5.get('hit_rate'))}  avg_return={_pct(f5.get('avg_return'))}  tstat={_fmt(f5.get('tstat'))}  pval={_fmt(f5.get('pval'))}")
        lines.append(f"    20-bar: hit_rate={_pct(f20.get('hit_rate'))}  avg_return={_pct(f20.get('avg_return'))}  tstat={_fmt(f20.get('tstat'))}  pval={_fmt(f20.get('pval'))}")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=_INTERP_SYSTEM,
        messages=[{"role": "user", "content": "\n".join(lines)}],
    )
    _log_spend("indicator_interpretation", response.usage, strategy_id=None)

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    try:
        entries_raw = json.loads(raw)
    except Exception:
        entries_raw = []

    saved = []
    for entry in entries_raw:
        combo  = entry.get("combo", "")
        parts  = combo.split("_", 1)
        asset  = parts[0] if parts else None
        tf     = parts[1] if len(parts) > 1 else None
        r      = results_by_combo.get(combo, {})
        sharpe = _estimate_sharpe(r)

        kb_entry = {
            "category":    entry.get("category", "partial"),
            "indicator":   spec["indicator"],
            "timeframe":   tf,
            "asset":       asset,
            "session":     None,
            "summary":     entry.get("summary", ""),
            "sharpe_ref":  sharpe,
            "strategy_id": None,
        }
        try:
            db.insert_knowledge(kb_entry)
            saved.append(kb_entry)
        except Exception as exc:
            log.warning("knowledge_insert_failed", error=str(exc))

    return saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_summary(spec: dict, results: dict, knowledge: list[dict]) -> str:
    works   = [e for e in knowledge if e["category"] == "works"]
    fails   = [e for e in knowledge if e["category"] == "fails"]
    partial = [e for e in knowledge if e["category"] == "partial"]
    return (
        f"{spec['title']}: tested on {len(results)} asset/TF combos. "
        f"Works: {len(works)}  Fails: {len(fails)}  Partial: {len(partial)}. "
        + (knowledge[0]["summary"] if knowledge else "No significant signal found.")
    )


def _estimate_sharpe(r: dict) -> float | None:
    """Rough Sharpe estimate from t-stat and signal count."""
    f5 = r.get("fwd_5", {})
    tstat = f5.get("tstat")
    count = f5.get("count") or 0
    if tstat is None or count < 10:
        return None
    import math
    return round(tstat / math.sqrt(count) * math.sqrt(252), 2)


def _pct(v) -> str:
    return f"{v*100:.1f}%" if v is not None else "N/A"


def _fmt(v) -> str:
    return f"{v:.2f}" if v is not None else "N/A"


def _log_spend(agent: str, usage, strategy_id) -> None:
    try:
        from db import supabase_client as db
        cost = (usage.input_tokens * 0.003 + usage.output_tokens * 0.015) / 1000
        db.log_spend(agent, MODEL, usage.input_tokens, usage.output_tokens, cost, strategy_id)
    except Exception:
        pass
