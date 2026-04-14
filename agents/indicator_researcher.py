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

def _build_all_specs() -> list[dict]:
    """
    Programmatically generate the full indicator spec catalogue.
    Covers: parameter sweeps for each indicator family, 2-indicator combos
    with confirmation filters, and price-structure variants.
    Returns a deduplicated list ordered by category then spec_id.
    """
    specs: list[dict] = []

    # ── RSI ───────────────────────────────────────────────────────────────────
    for period in [7, 10, 14, 21, 28]:
        # mean-reversion at different threshold pairs
        for os_lvl, ob_lvl in [(20, 80), (25, 75), (30, 70)]:
            specs.append({
                "spec_id": f"RSI_mr_{period}_{os_lvl}",
                "indicator": "RSI", "category": "momentum",
                "title": f"RSI mean-reversion ({period}, {os_lvl}/{ob_lvl})",
                "description": (
                    f"{period}-period RSI. Long: crosses above {os_lvl} from below. "
                    f"Short: crosses below {ob_lvl} from above."
                ),
            })
        # trend-following via 50-line
        specs.append({
            "spec_id": f"RSI_trend_{period}",
            "indicator": "RSI", "category": "momentum",
            "title": f"RSI 50-line trend ({period})",
            "description": (
                f"{period}-period RSI. Long: crosses above 50 from below. "
                f"Short: crosses below 50 from above."
            ),
        })

    # ── MACD ──────────────────────────────────────────────────────────────────
    for fast, slow, sig in [(12, 26, 9), (8, 17, 9), (5, 35, 5), (19, 39, 9)]:
        specs.append({
            "spec_id": f"MACD_signal_{fast}_{slow}_{sig}",
            "indicator": "MACD", "category": "momentum",
            "title": f"MACD signal-line cross ({fast},{slow},{sig})",
            "description": (
                f"MACD({fast},{slow},{sig}). Long: MACD crosses above signal. "
                f"Short: MACD crosses below signal."
            ),
        })
        specs.append({
            "spec_id": f"MACD_hist_{fast}_{slow}_{sig}",
            "indicator": "MACD", "category": "momentum",
            "title": f"MACD histogram flip ({fast},{slow},{sig})",
            "description": (
                f"MACD({fast},{slow},{sig}) histogram. Long: flips from negative to positive. "
                f"Short: flips from positive to negative."
            ),
        })

    # ── Stochastic ────────────────────────────────────────────────────────────
    for k, d, smooth in [(5, 3, 3), (9, 3, 3), (14, 3, 3), (21, 3, 3)]:
        for os_lvl, ob_lvl in [(20, 80), (25, 75)]:
            specs.append({
                "spec_id": f"STOCH_{k}_{d}_{os_lvl}",
                "indicator": "STOCH", "category": "momentum",
                "title": f"Stochastic %K cross ({k},{d},{smooth}) {os_lvl}/{ob_lvl}",
                "description": (
                    f"Stochastic({k},{d},{smooth}). Long: %K crosses above {os_lvl}. "
                    f"Short: %K crosses below {ob_lvl}."
                ),
            })

    # ── Williams %R ───────────────────────────────────────────────────────────
    for period in [10, 14, 21]:
        specs.append({
            "spec_id": f"WILLR_{period}",
            "indicator": "WILLR", "category": "momentum",
            "title": f"Williams %R cross ({period})",
            "description": (
                f"Williams %R({period}). Long: crosses above -80 from below. "
                f"Short: crosses below -20 from above."
            ),
        })

    # ── CCI ───────────────────────────────────────────────────────────────────
    for period in [14, 20, 30]:
        for level in [100, 150]:
            specs.append({
                "spec_id": f"CCI_{period}_{level}",
                "indicator": "CCI", "category": "momentum",
                "title": f"CCI ±{level} cross ({period})",
                "description": (
                    f"CCI({period}). Long: crosses above -{level} from below. "
                    f"Short: crosses below +{level} from above."
                ),
            })

    # ── ROC ───────────────────────────────────────────────────────────────────
    for period in [9, 14, 20]:
        specs.append({
            "spec_id": f"ROC_zero_{period}",
            "indicator": "ROC", "category": "momentum",
            "title": f"ROC zero-line cross ({period})",
            "description": (
                f"{period}-period ROC. Long: crosses above 0. Short: crosses below 0."
            ),
        })

    # ── EMA cross ─────────────────────────────────────────────────────────────
    for fast, slow in [(5, 20), (8, 21), (9, 21), (10, 30), (13, 34), (20, 50), (50, 200)]:
        specs.append({
            "spec_id": f"EMA_{fast}_{slow}",
            "indicator": "EMA_cross", "category": "trend",
            "title": f"EMA {fast}/{slow} crossover",
            "description": (
                f"EMA({fast}) and EMA({slow}). Long: EMA{fast} crosses above EMA{slow}. "
                f"Short: EMA{fast} crosses below EMA{slow}."
            ),
        })

    # ── SMA cross ─────────────────────────────────────────────────────────────
    for fast, slow in [(10, 30), (20, 50), (50, 200)]:
        specs.append({
            "spec_id": f"SMA_{fast}_{slow}",
            "indicator": "SMA_cross", "category": "trend",
            "title": f"SMA {fast}/{slow} crossover",
            "description": (
                f"SMA({fast}) and SMA({slow}). Long: SMA{fast} crosses above SMA{slow}. "
                f"Short: SMA{fast} crosses below SMA{slow}."
            ),
        })

    # ── SuperTrend ────────────────────────────────────────────────────────────
    for period in [7, 10, 14]:
        for mult in [2.0, 3.0, 4.0]:
            specs.append({
                "spec_id": f"SUPERTREND_{period}_{int(mult*10)}",
                "indicator": "SUPERTREND", "category": "trend",
                "title": f"SuperTrend flip ({period}, {mult})",
                "description": (
                    f"SuperTrend({period}, {mult}). Long: direction flips to +1. "
                    f"Short: direction flips to -1."
                ),
            })

    # ── ADX / DI ──────────────────────────────────────────────────────────────
    for period in [10, 14, 20]:
        for adx_thresh in [20, 25]:
            specs.append({
                "spec_id": f"ADX_{period}_{adx_thresh}",
                "indicator": "ADX", "category": "trend",
                "title": f"ADX DI cross (period={period}, ADX>{adx_thresh})",
                "description": (
                    f"ADX({period}), DI+, DI-. Long: DI+ crosses above DI- AND ADX > {adx_thresh}. "
                    f"Short: DI- crosses above DI+ AND ADX > {adx_thresh}."
                ),
            })

    # ── Parabolic SAR ─────────────────────────────────────────────────────────
    for step, max_af in [(0.02, 0.2), (0.01, 0.1), (0.03, 0.3)]:
        specs.append({
            "spec_id": f"PSAR_{int(step*100)}_{int(max_af*10)}",
            "indicator": "PSAR", "category": "trend",
            "title": f"Parabolic SAR flip (step={step}, max={max_af})",
            "description": (
                f"PSAR(step={step}, max={max_af}). "
                f"Long: price crosses above SAR. Short: price crosses below SAR."
            ),
        })

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    for period in [10, 20, 30]:
        for std in [1.5, 2.0, 2.5]:
            specs.append({
                "spec_id": f"BB_bounce_{period}_{int(std*10)}",
                "indicator": "BBANDS", "category": "volatility",
                "title": f"BB mean-reversion bounce ({period}, {std}σ)",
                "description": (
                    f"BBands({period}, {std}). Long: closes below lower band. "
                    f"Short: closes above upper band."
                ),
            })
            specs.append({
                "spec_id": f"BB_break_{period}_{int(std*10)}",
                "indicator": "BBANDS", "category": "volatility",
                "title": f"BB breakout after inside period ({period}, {std}σ)",
                "description": (
                    f"BBands({period}, {std}). Long: closes above upper band after 3+ inside bars. "
                    f"Short: closes below lower band after 3+ inside bars."
                ),
            })
    # BB squeeze-release (fixed at 20,2)
    specs.append({
        "spec_id": "BB_squeeze_20_20",
        "indicator": "BBANDS", "category": "volatility",
        "title": "BB squeeze-release breakout (20, 2.0σ)",
        "description": (
            "BBands(20, 2.0). Band width = (upper-lower)/middle. "
            "Squeeze: width < 20-bar rolling mean. "
            "Signal: first bar squeeze ends AND price closes outside bands."
        ),
    })

    # ── Keltner Channel ───────────────────────────────────────────────────────
    for period in [10, 20]:
        for mult in [1.5, 2.0, 2.5]:
            specs.append({
                "spec_id": f"KC_bounce_{period}_{int(mult*10)}",
                "indicator": "KC", "category": "volatility",
                "title": f"Keltner Channel bounce ({period}, {mult}×ATR)",
                "description": (
                    f"KC({period}, {mult}). Long: price touches/closes below lower band. "
                    f"Short: price touches/closes above upper band."
                ),
            })

    # ── ATR expansion ─────────────────────────────────────────────────────────
    for period in [7, 10, 14, 20]:
        for mult in [1.5, 2.0]:
            specs.append({
                "spec_id": f"ATR_exp_{period}_{int(mult*10)}",
                "indicator": "ATR", "category": "volatility",
                "title": f"ATR expansion breakout ({period}, {mult}×avg)",
                "description": (
                    f"ATR({period}). Long: ATR > {mult}× rolling mean AND price > 20-bar high. "
                    f"Short: ATR expands AND price < 20-bar low."
                ),
            })

    # ── OBV ───────────────────────────────────────────────────────────────────
    for ema_period in [10, 20, 30]:
        specs.append({
            "spec_id": f"OBV_ema_{ema_period}",
            "indicator": "OBV", "category": "volume",
            "title": f"OBV vs EMA({ema_period}) crossover",
            "description": (
                f"OBV and its {ema_period}-bar EMA. Long: OBV crosses above EMA. "
                f"Short: OBV crosses below EMA."
            ),
        })

    # ── CMF ───────────────────────────────────────────────────────────────────
    for period in [10, 20]:
        specs.append({
            "spec_id": f"CMF_zero_{period}",
            "indicator": "CMF", "category": "volume",
            "title": f"CMF zero-line cross ({period})",
            "description": (
                f"CMF({period}). Long: crosses above 0. Short: crosses below 0."
            ),
        })

    # ── MFI ───────────────────────────────────────────────────────────────────
    for period in [10, 14]:
        for os_lvl, ob_lvl in [(20, 80), (25, 75)]:
            specs.append({
                "spec_id": f"MFI_{period}_{os_lvl}",
                "indicator": "MFI", "category": "volume",
                "title": f"MFI cross ({period}, {os_lvl}/{ob_lvl})",
                "description": (
                    f"MFI({period}). Long: crosses above {os_lvl}. Short: crosses below {ob_lvl}."
                ),
            })

    # ── VWAP ──────────────────────────────────────────────────────────────────
    specs.append({
        "spec_id": "VWAP_cross",
        "indicator": "VWAP", "category": "volume",
        "title": "VWAP daily cross",
        "description": (
            "Daily VWAP (reset each day). "
            "Long: price crosses above VWAP. Short: price crosses below VWAP."
        ),
    })
    specs.append({
        "spec_id": "VWAP_band_bounce",
        "indicator": "VWAP", "category": "volume",
        "title": "VWAP ±1σ band bounce",
        "description": (
            "Daily VWAP ± 1 std-dev bands. "
            "Long: price touches lower band and closes back above it. "
            "Short: price touches upper band and closes back below it."
        ),
    })

    # ── Price structure ────────────────────────────────────────────────────────
    for lookback in [10, 20, 30]:
        specs.append({
            "spec_id": f"swing_bounce_{lookback}",
            "indicator": "swing_levels", "category": "structure",
            "title": f"Swing level bounce ({lookback}-bar extremes)",
            "description": (
                f"Identify swing lows as {lookback}-bar local minimums. "
                f"Long: price comes within 0.3% of swing low AND next bar closes above it. "
                f"Short: mirror for swing highs."
            ),
        })
    specs.append({
        "spec_id": "prior_session_hl",
        "indicator": "prior_session_hl", "category": "structure",
        "title": "Prior session high/low breakout",
        "description": (
            "Prior day high/low via daily resampling. "
            "Long: closes above prior day high. Short: closes below prior day low."
        ),
    })
    specs.append({
        "spec_id": "round_numbers",
        "indicator": "round_numbers", "category": "structure",
        "title": "Round number bounce",
        "description": (
            "Round 100-pip levels (e.g. 1.0800). "
            "Long: price within 0.1% below level then closes above it. "
            "Short: mirror."
        ),
    })
    specs.append({
        "spec_id": "pivot_points",
        "indicator": "pivot_points", "category": "structure",
        "title": "Daily pivot point bounce (standard)",
        "description": (
            "Daily standard pivot: P=(H+L+C)/3, S1=2P-H, R1=2P-L. "
            "Long: bounces off S1. Short: bounces off R1."
        ),
    })

    # ── 2-indicator combos: RSI momentum + EMA trend filter ───────────────────
    for rsi_period in [10, 14]:
        for ema_period in [50, 200]:
            specs.append({
                "spec_id": f"RSI_EMA_filter_{rsi_period}_{ema_period}",
                "indicator": "RSI+EMA", "category": "combo",
                "title": f"RSI({rsi_period}) mean-reversion + EMA({ema_period}) trend filter",
                "description": (
                    f"RSI({rsi_period}) + EMA({ema_period}) trend filter. "
                    f"Long: RSI crosses above 30 AND price > EMA{ema_period} (uptrend only). "
                    f"Short: RSI crosses below 70 AND price < EMA{ema_period} (downtrend only)."
                ),
            })

    # ── 2-indicator combos: MACD + ADX strength filter ────────────────────────
    for adx_thresh in [20, 25]:
        specs.append({
            "spec_id": f"MACD_ADX_{adx_thresh}",
            "indicator": "MACD+ADX", "category": "combo",
            "title": f"MACD signal cross + ADX>{adx_thresh} filter",
            "description": (
                f"MACD(12,26,9) signal cross only when ADX(14) > {adx_thresh}. "
                f"Long: MACD crosses above signal AND ADX > {adx_thresh}. "
                f"Short: MACD crosses below signal AND ADX > {adx_thresh}."
            ),
        })

    # ── 2-indicator combos: BB + RSI confirmation ─────────────────────────────
    specs.append({
        "spec_id": "BB_RSI_confirm",
        "indicator": "BBANDS+RSI", "category": "combo",
        "title": "BB bounce + RSI oversold/overbought confirmation",
        "description": (
            "BBands(20,2) + RSI(14). "
            "Long: price closes below lower BB AND RSI < 35. "
            "Short: price closes above upper BB AND RSI > 65."
        ),
    })

    # ── 2-indicator combos: EMA cross + volume confirmation ───────────────────
    specs.append({
        "spec_id": "EMA_OBV_confirm",
        "indicator": "EMA+OBV", "category": "combo",
        "title": "EMA 20/50 cross + OBV confirmation",
        "description": (
            "EMA(20/50) cross + OBV trend. "
            "Long: EMA20 crosses above EMA50 AND OBV > its 20-bar EMA. "
            "Short: EMA20 crosses below EMA50 AND OBV < its 20-bar EMA."
        ),
    })

    # ── 2-indicator combos: SuperTrend + RSI pullback ─────────────────────────
    specs.append({
        "spec_id": "SUPERTREND_RSI_pullback",
        "indicator": "SUPERTREND+RSI", "category": "combo",
        "title": "SuperTrend direction + RSI pullback entry",
        "description": (
            "SuperTrend(7,3) for direction + RSI(14) for entry timing. "
            "Long: SuperTrend is +1 AND RSI dips below 40 then crosses back above 40. "
            "Short: SuperTrend is -1 AND RSI rises above 60 then crosses back below 60."
        ),
    })

    # ── 2-indicator combos: Stochastic + EMA trend ────────────────────────────
    for ema_period in [50, 100]:
        specs.append({
            "spec_id": f"STOCH_EMA_{ema_period}",
            "indicator": "STOCH+EMA", "category": "combo",
            "title": f"Stochastic cross + EMA({ema_period}) trend filter",
            "description": (
                f"Stochastic(14,3,3) + EMA({ema_period}). "
                f"Long: %K crosses above 20 AND price > EMA{ema_period}. "
                f"Short: %K crosses below 80 AND price < EMA{ema_period}."
            ),
        })

    # ── 3-indicator combos ────────────────────────────────────────────────────
    specs.append({
        "spec_id": "RSI_BB_volume_triple",
        "indicator": "RSI+BB+OBV", "category": "combo",
        "title": "RSI oversold + BB lower band + OBV rising (triple confirmation)",
        "description": (
            "RSI(14) + BBands(20,2) + OBV(20 EMA). "
            "Long: RSI < 30 AND close < lower BB AND OBV > its EMA (volume supports move). "
            "Short: RSI > 70 AND close > upper BB AND OBV < its EMA."
        ),
    })
    specs.append({
        "spec_id": "EMA_MACD_ADX_triple",
        "indicator": "EMA+MACD+ADX", "category": "combo",
        "title": "EMA trend + MACD momentum + ADX strength (triple)",
        "description": (
            "EMA(20/50) + MACD(12,26,9) + ADX(14). "
            "Long: EMA20 > EMA50 AND MACD > signal AND ADX > 20. "
            "Short: EMA20 < EMA50 AND MACD < signal AND ADX > 20."
        ),
    })

    # Deduplicate by spec_id (keep first occurrence)
    seen: set[str] = set()
    result: list[dict] = []
    for s in specs:
        if s["spec_id"] not in seen:
            seen.add(s["spec_id"])
            result.append(s)
    return result


def _build_exit_specs() -> list[dict]:
    """
    Exit strategy / risk management research specs.
    These test WHEN and HOW to exit, not whether to enter.
    The generated code enters at a baseline signal and compares exit rules.
    """
    specs: list[dict] = []

    # ── ATR-based stop placement ───────────────────────────────────────────────
    for entry_signal in ["ema_cross", "rsi_mr"]:
        for atr_period in [10, 14]:
            specs.append({
                "spec_id":    f"exit_atr_stop_{entry_signal}_{atr_period}",
                "indicator":  "ATR_stop",
                "category":   "exit_research",
                "spec_type":  "exit_research",
                "title":      f"ATR stop placement ({atr_period}) on {entry_signal} entries",
                "description": (
                    f"Entry signal: {entry_signal}. "
                    f"Test stop placement at 1.0×, 1.5×, 2.0×, 2.5×, 3.0× ATR({atr_period}) from entry. "
                    f"Measure: actual MAE at time of eventual exit, win rate, avg R:R achieved vs theoretical, "
                    f"capture ratio (actual PnL / MFE). "
                    f"Baseline comparison: fixed 20-pip stop."
                ),
                "param_space": {
                    "atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
                    "atr_period": [atr_period],
                },
            })

    # ── Trailing stop strategies ────────────────────────────────────────────────
    for trail_type in ["atr_trail", "highest_close_trail", "chandelier"]:
        specs.append({
            "spec_id":    f"exit_trail_{trail_type}",
            "indicator":  "trailing_stop",
            "category":   "exit_research",
            "spec_type":  "exit_research",
            "title":      f"Trailing stop: {trail_type}",
            "description": (
                f"Entry: EMA(20/50) cross. Exit rule: {trail_type}. "
                f"ATR trail: stop moves up by ATR(14)×mult each bar. "
                f"Highest-close trail: stop = highest close over N bars × (1 - X%). "
                f"Chandelier: highest high over N bars minus ATR(14)×mult. "
                f"Compare vs fixed 2:1 R:R baseline. "
                f"Measure: MFE capture ratio, avg trade PnL, profit factor."
            ),
            "param_space": {
                "trail_mult": [1.5, 2.0, 2.5, 3.0],
                "trail_period": [10, 14, 20],
            },
        })

    # ── Indicator-based exits ──────────────────────────────────────────────────
    # Test: "does indicator X crossing level Y predict a good exit point?"
    for exit_indicator, exit_spec in [
        ("RSI",   "RSI(14) crossing above 60 (exit long) / below 40 (exit short)"),
        ("RSI",   "RSI(14) crossing above 70 (exit long) / below 30 (exit short)"),
        ("MACD",  "MACD histogram flipping direction — first sign of momentum fading"),
        ("BB",    "Price touching upper BB (exit long) / lower BB (exit short)"),
        ("ADX",   "ADX(14) falling below 20 — trend weakening, exit trend trades"),
        ("STOCH", "Stochastic %K above 80 (exit long) / below 20 (exit short)"),
    ]:
        spec_id = f"exit_{exit_indicator.lower()}_{exit_spec[:20].replace(' ','_').replace('/','').lower()}"
        specs.append({
            "spec_id":    spec_id,
            "indicator":  f"{exit_indicator}_exit",
            "category":   "exit_research",
            "spec_type":  "exit_research",
            "title":      f"Exit signal: {exit_indicator} ({exit_spec[:40]})",
            "description": (
                f"Entry: EMA(20/50) cross (representative trend-following entry). "
                f"Exit rule: {exit_spec}. "
                f"Compare vs: (a) fixed 2:1 R:R, (b) fixed 20-bar time exit. "
                f"Measure: avg trade PnL improvement vs baselines, profit factor, "
                f"MFE capture ratio, trade duration distribution."
            ),
            "param_space": {},  # LLM will generate the param space
        })

    # ── Time-based exits ───────────────────────────────────────────────────────
    for horizon in [5, 10, 20, 40]:
        specs.append({
            "spec_id":    f"exit_time_{horizon}bar",
            "indicator":  "time_exit",
            "category":   "exit_research",
            "spec_type":  "exit_research",
            "title":      f"Time-based exit at {horizon} bars (MAE/MFE profile)",
            "description": (
                f"For EMA(20/50) cross entries: exit after exactly {horizon} bars. "
                f"Run MAE/MFE analysis: how far against entry (max adverse) and for entry (max favorable) "
                f"does price move during the {horizon}-bar holding period? "
                f"Key metrics: MAE distribution (where to set stop), MFE distribution (where to set TP), "
                f"MFE/MAE ratio, fraction of trades where MAE > 1%, > 2%, > 3%."
            ),
            "param_space": {},
        })

    # ── Partial exit strategies ────────────────────────────────────────────────
    specs.append({
        "spec_id":    "exit_partial_scale_out",
        "indicator":  "partial_exit",
        "category":   "exit_research",
        "spec_type":  "exit_research",
        "title":      "Partial exit: scale out at 1R then trail remainder",
        "description": (
            "Entry: EMA(20/50) cross with ATR(14) stop. "
            "Strategy A (baseline): exit full position at 2R. "
            "Strategy B: exit 50% at 1R, trail remaining 50% with 1.5×ATR trailing stop. "
            "Strategy C: exit 33% at 1R, 33% at 2R, trail 33% with chandelier. "
            "Measure: avg PnL per unit risked, profit factor, Sharpe, max drawdown per strategy. "
            "Vary split ratio: 25/75, 50/50, 75/25."
        ),
        "param_space": {"scale_out_pct": [0.25, 0.50, 0.75], "first_target_r": [0.75, 1.0, 1.5]},
    })

    # ── Breakeven stop research ────────────────────────────────────────────────
    specs.append({
        "spec_id":    "exit_breakeven_trigger",
        "indicator":  "breakeven_stop",
        "category":   "exit_research",
        "spec_type":  "exit_research",
        "title":      "Breakeven stop: when to move stop to entry",
        "description": (
            "Entry: EMA(20/50) cross with 1.5×ATR(14) initial stop. "
            "Test: at what profit level (0.5R, 0.75R, 1.0R, 1.25R) does moving stop to breakeven "
            "improve or hurt overall performance? "
            "Measure: impact on win rate, avg trade PnL, how often breakeven stop triggers "
            "before TP vs eventual win if held."
        ),
        "param_space": {"breakeven_trigger_r": [0.5, 0.75, 1.0, 1.25, 1.5]},
    })

    # ── Volatility-regime exit adjustment ─────────────────────────────────────
    specs.append({
        "spec_id":    "exit_vol_regime_adjust",
        "indicator":  "vol_regime_exit",
        "category":   "exit_research",
        "spec_type":  "exit_research",
        "title":      "Exit adjustment based on volatility regime (ATR expansion/contraction)",
        "description": (
            "Entry: RSI(14) mean-reversion (oversold/overbought). "
            "Classify bars as high-vol (ATR > 1.5×20-bar avg ATR) or low-vol. "
            "High-vol regime: use wider stop (3×ATR) and larger TP (3×ATR). "
            "Low-vol regime: use tighter stop (1×ATR) and smaller TP (1.5×ATR). "
            "Compare vs single fixed stop/TP across all regimes. "
            "Measure: win rate, avg PnL, PF per regime and combined."
        ),
        "param_space": {
            "high_vol_stop_mult": [2.0, 3.0],
            "low_vol_stop_mult":  [0.75, 1.0, 1.5],
        },
    })

    # ── Session-based exits ────────────────────────────────────────────────────
    specs.append({
        "spec_id":    "exit_session_close",
        "indicator":  "session_exit",
        "category":   "exit_research",
        "spec_type":  "exit_research",
        "title":      "Exit at session close (London/NY close) vs holding overnight",
        "description": (
            "Entry: any intraday signal. "
            "Compare: (A) exit at London close (17:00 UTC) if trade open, "
            "(B) exit at NY close (22:00 UTC), "
            "(C) hold until TP/SL regardless of session. "
            "Measure: avg trade PnL per exit rule, overnight gap impact, "
            "which session has worse overnight holds."
        ),
        "param_space": {},
    })

    return specs


EXIT_SPECS: list[dict] = _build_exit_specs()
INDICATOR_SPECS: list[dict] = _build_all_specs() + EXIT_SPECS


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


def generate_llm_combo_tasks(n: int = 15) -> int:
    """
    Ask Claude to propose n novel indicator combo ideas not yet in research_tasks.
    Returns the number of new tasks created.

    Called by the queue_worker when the pending queue runs low, keeping the
    research pipeline running indefinitely.
    """
    from db import supabase_client as db

    # Build context: what indicators have shown edge (top by sharpe_ref)
    try:
        kb_entries = db.get_knowledge_entries(limit=50)
    except Exception:
        kb_entries = []

    works = [e for e in kb_entries if e.get("category") == "works"][:20]
    fails = [e for e in kb_entries if e.get("category") == "fails"][:10]
    works_summary = "; ".join(
        f"{e['indicator']} ({e.get('timeframe','?')}): sharpe~{e.get('sharpe_ref',0):.2f}"
        for e in works
    ) or "none yet"
    fails_summary = ", ".join(e["indicator"] for e in fails) or "none yet"

    # What titles already exist (to avoid duplicates)
    existing = db.get_research_tasks(status="all", limit=1000, task_type="indicator_research")
    existing_titles = {t["title"].lower() for t in existing}

    prompt = f"""You are a quantitative FX/commodities researcher.

Based on statistical forward-return tests on EURUSD and XAUUSD (1h, 4h), here is what we know:

INDICATORS WITH EDGE: {works_summary}

INDICATORS THAT FAILED: {fails_summary}

Generate exactly {n} NEW indicator combination ideas to test next. Aim for:
- Novel combos not obviously covered above
- Mix of mean-reversion, trend-following, and volatility-based ideas
- 2–3 indicator combinations where one provides the signal and another provides confirmation or a filter
- Specific parameter values in the description (exact periods, thresholds, entry/exit rules)

Respond ONLY with a JSON array. Each element must have:
  "spec_id": unique snake_case identifier (e.g. "RSI_STOCH_divergence_14_14")
  "indicator": short indicator combo name (e.g. "RSI+STOCH")
  "category": one of: momentum, trend, volatility, volume, structure, combo
  "title": short human-readable title (under 60 chars)
  "description": precise entry/exit rules with exact parameter values. Entry must produce +1 (long) or -1 (short) signals for pandas forward-return testing.

No markdown, no explanation — only the raw JSON array."""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        proposals = json.loads(raw)
    except Exception as exc:
        log.warning("llm_combo_parse_failed error=%s raw=%s", exc, raw[:200])
        return 0

    created = 0
    for p in proposals:
        if not isinstance(p, dict):
            continue
        title = f"[Indicator] {p.get('title', '')}"
        if title.lower() in existing_titles:
            continue
        try:
            spec = {
                "spec_id":    p.get("spec_id", f"llm_{created}"),
                "indicator":  p.get("indicator", "unknown"),
                "category":   p.get("category", "combo"),
                "title":      p.get("title", ""),
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
            log.warning("llm_combo_insert_failed error=%s", exc)

    log.info("llm_combo_tasks_created created=%s requested=%s", created, n)
    return created


# ---------------------------------------------------------------------------
# Analysis pipeline (runs on Modal)
# ---------------------------------------------------------------------------

def run_indicator_research(task_id: str, cache_dir: str = "/ohlcv_cache") -> dict:
    """
    Entry point called from Modal.

    Pipeline per spec:
      1. LLM generates parameterized analyze_indicator(df, **params) + PARAM_SPACE
      2. Sweep PARAM_SPACE on each test asset/TF combo → find best params
      3. Run MCPT (bar permutation) on best params → p-value
      4. Interpret full results → knowledge_base entries with p-value + optimal params
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

        # Determine research type
        spec_type = spec.get("spec_type", "entry_research")
        is_exit = spec_type == "exit_research"
        fn_name  = "analyze_exit_strategy" if is_exit else "analyze_indicator"

        # 1. Generate parameterized analysis code + param space
        code, param_space = _generate_analysis_code(
            spec["indicator"], spec["description"], spec_type=spec_type
        )
        # Merge spec-level param_space (from catalogue) with LLM-generated one
        # (spec takes priority for explicitly defined ranges)
        if spec.get("param_space"):
            param_space = {**param_space, **spec["param_space"]}

        # 2. Parameter sweep across all combos
        sweep_results: dict[str, Any] = {}   # combo → {params_str: stats_dict}
        best_by_combo: dict[str, Any] = {}   # combo → {params, stats}

        for symbol, tf in _TEST_ASSETS:
            df = _load_data(symbol, tf, cache_dir)
            if df is None or len(df) < 200:
                continue
            combo_key = f"{symbol}_{tf}"
            try:
                sweep, best = _sweep_params(code, df, param_space, fn_name=fn_name)
                # Exit research: need at least 10 trades; check via "baseline" count
                min_count = (
                    (best["stats"].get("baseline") or {}).get("count", 0)
                    if is_exit else
                    best["stats"].get("fwd_5", {}).get("count", 0)
                ) if best else 0
                if best and min_count >= 10:
                    sweep_results[combo_key] = sweep
                    best_by_combo[combo_key] = best
                    pf_key = ("baseline" if is_exit else "fwd_5")
                    log.info("sweep_done indicator=%s combo=%s best_pf=%.3f count=%s",
                             spec["indicator"], combo_key,
                             (best["stats"].get(pf_key) or {}).get("profit_factor", 0),
                             min_count)
            except Exception as exc:
                log.warning("sweep_failed indicator=%s combo=%s error=%s",
                            spec["indicator"], combo_key, exc)

        if not best_by_combo:
            raise ValueError(f"No successful sweep for {spec['indicator']}")

        # 3. MCPT — pick the globally best-performing combo for permutation test
        best_combo = max(best_by_combo,
                         key=lambda c: best_by_combo[c]["stats"].get("profit_factor", 0))
        best_entry = best_by_combo[best_combo]
        symbol_mc, tf_mc = best_combo.rsplit("_", 1)
        df_mc = _load_data(symbol_mc, tf_mc, cache_dir)

        mcpt_pvalue: float | None = None
        if df_mc is not None:
            try:
                mcpt_pvalue = _mcpt_test(code, df_mc, best_entry["params"], n_perms=300,
                                         fn_name=fn_name)
                log.info("mcpt_done indicator=%s combo=%s p_value=%.4f",
                         spec["indicator"], best_combo, mcpt_pvalue)
            except Exception as exc:
                log.warning("mcpt_failed indicator=%s error=%s", spec["indicator"], exc)

        # 4. Interpret → knowledge entries
        knowledge_entries = _interpret_and_save(
            spec, best_by_combo, sweep_results, mcpt_pvalue, task_id, db
        )

        # 5. Mark done
        summary = _build_summary(spec, best_by_combo, knowledge_entries, mcpt_pvalue)
        db.update_research_task(task_id, {
            "status":       "done",
            "modal_job_id": None,
            "result_summary": summary,
            "key_findings": [
                {"finding": e["summary"],
                 "confidence": _pvalue_to_confidence(e.get("p_value"))}
                for e in knowledge_entries
            ],
            "generated_code": code,
            "error_log":    None,
        })

        return {
            "passed":            True,
            "task_id":           task_id,
            "indicator":         spec["indicator"],
            "combos_tested":     len(best_by_combo),
            "knowledge_entries": len(knowledge_entries),
            "mcpt_pvalue":       mcpt_pvalue,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        db.update_research_task(task_id, {
            "status":       "failed",
            "modal_job_id": None,
            "error_log":    f"{type(exc).__name__}: {exc}\n{tb[:600]}",
        })
        raise


# ---------------------------------------------------------------------------
# Step 1: generate parameterized analysis code via LLM
# ---------------------------------------------------------------------------

_CODE_GEN_SYSTEM = r"""You write parameterized Python indicator analysis functions for forward-return testing.

Given an indicator description, output TWO things separated by the delimiter "###PARAM_SPACE###":

PART 1 — The function `analyze_indicator(df, **params)`.
PART 2 — A JSON dict `PARAM_SPACE` defining the parameter sweep grid.

=== PART 1 REQUIREMENTS ===
The function MUST:
1. Accept df (Open, High, Low, Close, Volume title-cased) and **params with sensible defaults
2. Compute indicators using pandas_ta
3. Set df['signal'] = +1 (long), -1 (short), or 0
   — signal fires only on the bar where condition becomes true (use .shift(1) for prior value)
4. Return the stats block EXACTLY as shown (do not modify stats code)

```python
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy import stats as _sp

def analyze_indicator(df: pd.DataFrame, **params) -> dict:
    df = df.copy()
    df.columns = [c.title() for c in df.columns]

    # Extract params with defaults
    period = params.get('period', 14)
    # ... extract other params ...

    # === INDICATOR CODE ===

    # === SIGNAL CODE ===
    df['signal'] = 0

    # === STATS BLOCK — do not modify ===
    for horizon in [5, 10, 20, 50]:
        df[f'fwd_{horizon}'] = np.log(df['Close'].shift(-horizon) / df['Close'])
    longs  = df[df['signal'] ==  1].copy()
    shorts = df[df['signal'] == -1].copy()

    def _stat(rets):
        rets = rets.dropna()
        if len(rets) < 10:
            return dict(count=int(len(rets)), hit_rate=None, avg_log_return=None,
                        profit_factor=None, tstat=None, pval=None)
        wins   = rets[rets > 0]
        losses = rets[rets < 0]
        pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else None
        t, p = _sp.ttest_1samp(rets.values, 0)
        return dict(
            count=int(len(rets)),
            hit_rate=float((rets > 0).mean()),
            avg_log_return=float(rets.mean()),
            profit_factor=pf,
            tstat=float(t),
            pval=float(p),
        )

    horizons = {}
    for h in [5, 10, 20, 50]:
        col = f'fwd_{h}'
        rets_long  = longs[col] if col in longs.columns else pd.Series(dtype=float)
        rets_short = -shorts[col] if col in shorts.columns else pd.Series(dtype=float)
        combined   = pd.concat([rets_long, rets_short])
        horizons[f'fwd_{h}'] = _stat(combined)
        horizons[f'fwd_{h}_long']  = _stat(rets_long.dropna())
        horizons[f'fwd_{h}_short'] = _stat(rets_short.dropna())

    return dict(
        long_count=int(len(longs)),
        short_count=int(len(shorts)),
        **horizons,
    )
```

=== PART 2 — PARAM_SPACE JSON ===
A dict mapping param name → list of values to sweep.
Keep it under 50 total combinations (product of all lists).
Example:
{"period": [10, 14, 21], "threshold": [25, 30, 35]}

=== OUTPUT FORMAT ===
<function code here>
###PARAM_SPACE###
{"period": [...], ...}

No markdown fences. No explanation."""


_EXIT_CODE_GEN_SYSTEM = r"""You write Python exit strategy / trade management analysis functions.

Unlike entry research (which measures if a signal predicts forward returns), exit research:
1. Enters trades using a FIXED baseline entry rule
2. Tests different EXIT rules / stop strategies as the variable
3. Measures trade outcomes: avg PnL per trade, profit factor, MAE/MFE capture, win rate

You will output TWO parts separated by "###PARAM_SPACE###":

PART 1 — Function `analyze_exit_strategy(df, **params)`.
PART 2 — JSON dict PARAM_SPACE defining the sweep grid.

=== PART 1 REQUIREMENTS ===
```python
import pandas as pd
import pandas_ta as ta
import numpy as np

def analyze_exit_strategy(df: pd.DataFrame, **params) -> dict:
    df = df.copy()
    df.columns = [c.title() for c in df.columns]

    # === BASELINE ENTRY SIGNAL ===
    # Use EMA(20/50) cross as the representative entry (unless spec says otherwise)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df['entry_long']  = (df['EMA_20'] > df['EMA_50']) & (df['EMA_20'].shift(1) <= df['EMA_50'].shift(1))
    df['entry_short'] = (df['EMA_20'] < df['EMA_50']) & (df['EMA_20'].shift(1) >= df['EMA_50'].shift(1))

    # === YOUR EXIT LOGIC HERE ===
    # Simulate trades: for each entry, apply the exit rule and measure PnL
    # ... (implement per-trade simulation) ...

    # === STATS BLOCK — compute per exit rule ===
    # For each simulated trade record: {entry_price, exit_price, direction, pnl_r, mae_r, mfe_r, bars_held}
    # pnl_r = PnL in R (units of initial risk). mae_r = max adverse in R. mfe_r = max favorable in R.

    def _trade_stats(trades: list[dict]) -> dict:
        if len(trades) < 10:
            return dict(count=len(trades), win_rate=None, avg_pnl_r=None,
                        profit_factor=None, avg_mfe_r=None, avg_mae_r=None,
                        mfe_capture=None, avg_bars=None)
        pnls = [t['pnl_r'] for t in trades]
        mfes = [t.get('mfe_r', 0) for t in trades]
        maes = [t.get('mae_r', 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        pf = sum(wins) / abs(sum(losses)) if losses else None
        mfe_cap = np.mean([p/m if m > 0 else 0 for p, m in zip(pnls, mfes)]) if mfes else None
        return dict(
            count=len(trades),
            win_rate=float(sum(1 for p in pnls if p > 0) / len(pnls)),
            avg_pnl_r=float(np.mean(pnls)),
            profit_factor=float(pf) if pf is not None else None,
            avg_mfe_r=float(np.mean(mfes)),
            avg_mae_r=float(np.mean(maes)),
            mfe_capture=float(mfe_cap) if mfe_cap is not None else None,
            avg_bars=float(np.mean([t.get('bars_held', 0) for t in trades])),
        )

    # Return results for each exit rule tested
    return {
        "baseline": _trade_stats(baseline_trades),   # fixed 2:1 R:R
        "exit_rule": _trade_stats(exit_rule_trades),  # your exit rule
        "improvement": float(np.mean([t['pnl_r'] for t in exit_rule_trades]) -
                             np.mean([t['pnl_r'] for t in baseline_trades]))
        if exit_rule_trades and baseline_trades else None,
    }
```

=== TRADE SIMULATION PATTERN ===
```python
# For each entry signal, simulate the trade:
atr = df.ta.atr(length=14)
trades = []
for i in df.index[df['entry_long']]:
    iloc_i = df.index.get_loc(i)
    entry_price = df.at[i, 'Close']
    stop = entry_price - atr.at[i] * stop_mult
    risk = entry_price - stop  # initial risk in price
    if risk <= 0: continue
    mae_r = 0.0
    mfe_r = 0.0
    exit_price = None
    bars_held = 0
    for j in range(iloc_i + 1, min(iloc_i + max_bars + 1, len(df))):
        bar = df.iloc[j]
        # track MAE/MFE
        mae_r = max(mae_r, (entry_price - bar['Low']) / risk)
        mfe_r = max(mfe_r, (bar['High'] - entry_price) / risk)
        bars_held += 1
        # check your exit rule here
        if <exit_condition>:
            exit_price = bar['Close']
            break
    if exit_price is None:
        exit_price = df.iloc[min(iloc_i + max_bars, len(df)-1)]['Close']
    pnl_r = (exit_price - entry_price) / risk
    trades.append(dict(pnl_r=pnl_r, mae_r=mae_r, mfe_r=mfe_r, bars_held=bars_held))
```

=== OUTPUT FORMAT ===
<function code>
###PARAM_SPACE###
{"param_name": [...]}

No markdown. No explanation."""


def _generate_analysis_code(indicator: str, description: str, spec_type: str = "entry_research") -> tuple[str, dict]:
    """Returns (code_str, param_space_dict)."""
    system = _EXIT_CODE_GEN_SYSTEM if spec_type == "exit_research" else _CODE_GEN_SYSTEM
    user_msg = (
        f"EXIT STRATEGY: {indicator}\n"
        f"DESCRIPTION: {description}\n\n"
        "Write analyze_exit_strategy(df, **params) testing this exit rule vs baseline."
    ) if spec_type == "exit_research" else (
        f"INDICATOR: {indicator}\n"
        f"SIGNAL LOGIC: {description}\n\n"
        "Write the parameterized analyze_indicator(df, **params) and PARAM_SPACE."
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    _log_spend("indicator_code_gen", response.usage, strategy_id=None)

    # Split on delimiter
    if "###PARAM_SPACE###" in raw:
        code_part, ps_part = raw.split("###PARAM_SPACE###", 1)
    else:
        # Fallback: treat entire response as code, no param sweep
        code_part, ps_part = raw, "{}"

    code = code_part.strip()
    if code.startswith("```"):
        code = code.split("```")[1]
        if code.startswith("python"):
            code = code[6:]
        code = code.strip().rstrip("`").strip()

    ps_raw = ps_part.strip()
    if ps_raw.startswith("```"):
        ps_raw = ps_raw.split("```")[1]
        if ps_raw.startswith("json"):
            ps_raw = ps_raw[4:]
        ps_raw = ps_raw.strip().rstrip("`").strip()

    try:
        param_space = json.loads(ps_raw) if ps_raw else {}
    except Exception:
        param_space = {}

    return code, param_space


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

    try:
        from backtest.data_fetcher import fetch_ohlcv
        df = fetch_ohlcv(symbol, tf, start="2015-01-01", end="2026-12-31")
        return df
    except Exception as exc:
        log.warning("data_load_failed symbol=%s tf=%s error=%s", symbol, tf, exc)
        return None


# ---------------------------------------------------------------------------
# Step 3: parameter sweep
# ---------------------------------------------------------------------------

def _sweep_params(
    code: str, df, param_space: dict, fn_name: str = "analyze_indicator"
) -> tuple[dict, dict | None]:
    """
    Run the generated function over all param combinations.
    Returns (sweep_dict, best_entry) where:
      sweep_dict: {params_str: stats_dict}
      best_entry: {"params": {...}, "stats": {...}} for the best combo by profit factor
    """
    import itertools
    import pandas as pd
    import numpy as np
    import pandas_ta as ta
    from scipy import stats as _sp

    namespace = {
        "pd": pd, "np": np, "ta": ta, "_sp": _sp,
        "__builtins__": __builtins__,
    }
    exec(compile(code, "<indicator_analysis>", "exec"), namespace)
    fn = namespace.get(fn_name)
    if fn is None:
        raise ValueError(f"Generated code missing {fn_name}")

    # Build combinations
    if param_space:
        keys   = list(param_space.keys())
        values = list(param_space.values())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        combos = [{}]  # single run with defaults

    sweep: dict[str, dict] = {}
    best_pf   = -1.0
    best_entry: dict | None = None

    for params in combos:
        try:
            result = fn(df.copy(), **params)
            if not isinstance(result, dict):
                continue
            pf = result.get("fwd_5", {}).get("profit_factor") or 0.0
            params_str = json.dumps(params, sort_keys=True)
            sweep[params_str] = result
            if pf > best_pf:
                best_pf   = pf
                best_entry = {"params": params, "stats": result}
        except Exception:
            pass

    return sweep, best_entry


# ---------------------------------------------------------------------------
# Step 4: Monte Carlo Permutation Test (MCPT)
# ---------------------------------------------------------------------------

def _bar_permute(df):
    """
    Shuffle OHLCV bars while preserving intra-bar structure.
    Method from mcpt repo (neurotrader888):
      - Compute log prices
      - Separate gap moves (open[t] - close[t-1]) from intra-bar moves (H/L/C relative to O)
      - Shuffle gaps and intra-bar vectors independently
      - Reconstruct price series
    """
    import numpy as np
    import pandas as pd

    df = df.copy()
    log_o = np.log(df["Open"].values)
    log_h = np.log(df["High"].values)
    log_l = np.log(df["Low"].values)
    log_c = np.log(df["Close"].values)

    # Intra-bar relative to open (preserves H/L/C relationships within bar)
    dh = log_h - log_o
    dl = log_l - log_o
    dc = log_c - log_o

    # Gap moves: open[t] - close[t-1]  (index 1..n)
    gaps = log_o[1:] - log_c[:-1]

    # Shuffle independently
    intra_idx = np.random.permutation(len(df))
    gap_idx   = np.random.permutation(len(gaps))

    dh_s = dh[intra_idx]
    dl_s = dl[intra_idx]
    dc_s = dc[intra_idx]
    gaps_s = gaps[gap_idx]

    # Reconstruct log open series from shuffled gaps
    new_log_o = np.empty(len(df))
    new_log_o[0] = log_o[0]
    for i in range(1, len(df)):
        new_log_o[i] = new_log_o[i - 1] + dc_s[i - 1] + gaps_s[i - 1]

    new_log_h = new_log_o + dh_s
    new_log_l = new_log_o + dl_s
    new_log_c = new_log_o + dc_s

    result = df.copy()
    result["Open"]  = np.exp(new_log_o)
    result["High"]  = np.exp(new_log_h)
    result["Low"]   = np.exp(new_log_l)
    result["Close"] = np.exp(new_log_c)
    return result


def _mcpt_test(
    code: str, df, best_params: dict, n_perms: int = 300, fn_name: str = "analyze_indicator"
) -> float | None:
    """
    Monte Carlo Permutation Test.
    Returns p-value = fraction of permutations whose profit_factor >= real PF.
    Lower p-value = more significant (p < 0.05 means <5% of random shuffles beat it).
    """
    import pandas as pd
    import numpy as np
    import pandas_ta as ta
    from scipy import stats as _sp

    namespace = {
        "pd": pd, "np": np, "ta": ta, "_sp": _sp,
        "__builtins__": __builtins__,
    }
    exec(compile(code, "<indicator_analysis>", "exec"), namespace)
    fn = namespace.get(fn_name)
    if fn is None:
        return None

    # Real profit factor — entry research uses fwd_5, exit research uses baseline
    def _get_pf(result: dict) -> float | None:
        if "baseline" in result:
            return (result.get("baseline") or {}).get("profit_factor")
        return (result.get("fwd_5") or {}).get("profit_factor")

    try:
        real_result = fn(df.copy(), **best_params)
        real_pf = _get_pf(real_result)
        if real_pf is None or real_result.get("fwd_5", {}).get("count", 0) < 10:
            return None
    except Exception:
        return None

    # Permuted profit factors
    perm_pfs: list[float] = []
    for _ in range(n_perms):
        try:
            df_perm = _bar_permute(df)
            perm_result = fn(df_perm, **best_params)
            pf = _get_pf(perm_result)
            if pf is not None:
                perm_pfs.append(pf)
        except Exception:
            pass

    if not perm_pfs:
        return None

    # p-value: fraction of permutations that matched or beat the real PF
    p_value = (sum(1 for pf in perm_pfs if pf >= real_pf) + 1) / (len(perm_pfs) + 1)
    return round(p_value, 4)


# ---------------------------------------------------------------------------
# Step 5: interpret results → knowledge_base entries
# ---------------------------------------------------------------------------

_INTERP_SYSTEM = """You analyze trading research results and write knowledge base entries.

You will receive EITHER:
(A) Entry research: forward-return statistics (hit rate, PF, t-stat across multiple horizons)
(B) Exit research: trade simulation results comparing an exit rule vs a fixed 2:1 R:R baseline

In both cases also receive: best params found via sweep, MCPT p-value.

Produce a JSON array — one entry per asset/TF combination (count >= 10).

Output format:
[
  {
    "combo": "EURUSD_1h",
    "category": "works" | "fails" | "partial",
    "summary": "3-4 sentences. For entry: hit rate, PF, best horizon, long/short asymmetry, p-value. For exit: improvement over baseline in R/trade, MFE capture ratio, when it works best, p-value."
  },
  ...
]

ENTRY CATEGORIZATION:
  "works":   PF >= 1.4 AND hit_rate >= 0.53 AND tstat >= 1.8 AND p_value < 0.10
  "fails":   PF <= 0.8 OR (tstat <= -1.5 AND hit_rate <= 0.47)
  "partial": everything else

EXIT CATEGORIZATION:
  "works":   exit_rule PF > baseline PF by >= 0.2 AND improvement >= +0.15 R/trade AND p_value < 0.10
  "fails":   exit_rule PF < baseline PF (exit rule hurts performance)
  "partial": mixed or marginal improvement

If p_value >= 0.10, cap at "partial" — cannot rule out random chance.
Always include p-value in summaries.

Return ONLY the JSON array."""


def _interpret_and_save(
    spec: dict,
    best_by_combo: dict,
    sweep_results: dict,
    mcpt_pvalue: float | None,
    task_id: str,
    db,
) -> list[dict]:
    is_exit = spec.get("spec_type") == "exit_research"
    lines = [
        f"{'EXIT STRATEGY' if is_exit else 'INDICATOR'}: {spec['title']}",
        f"MCPT p-value: {_fmt(mcpt_pvalue)}",
        "",
    ]

    for combo, entry in best_by_combo.items():
        params  = entry["params"]
        r       = entry["stats"]
        n_sweep = len(sweep_results.get(combo, {}))
        lines.append(f"  {combo} (best of {n_sweep} param combos, params={params}):")

        if is_exit:
            # Exit research result format
            baseline = r.get("baseline") or {}
            rule     = r.get("exit_rule") or {}
            impr     = r.get("improvement")
            lines.append(
                f"    baseline (fixed 2:1):  count={baseline.get('count')}  "
                f"win={_pct(baseline.get('win_rate'))}  avg_pnl_r={_fmt(baseline.get('avg_pnl_r'))}  "
                f"PF={_fmt(baseline.get('profit_factor'))}  "
                f"mfe_capture={_pct(baseline.get('mfe_capture'))}"
            )
            lines.append(
                f"    exit_rule:            count={rule.get('count')}  "
                f"win={_pct(rule.get('win_rate'))}  avg_pnl_r={_fmt(rule.get('avg_pnl_r'))}  "
                f"PF={_fmt(rule.get('profit_factor'))}  "
                f"mfe_capture={_pct(rule.get('mfe_capture'))}  "
                f"avg_bars={_fmt(rule.get('avg_bars'))}"
            )
            lines.append(f"    improvement vs baseline: {_fmt(impr)} R/trade")
        else:
            # Entry research result format
            lines.append(f"    signals: {r.get('long_count',0)} long + {r.get('short_count',0)} short")
            for h in [5, 10, 20, 50]:
                fk = f"fwd_{h}"
                s  = r.get(fk, {})
                if not s or s.get("count", 0) < 5:
                    continue
                sl = r.get(f"{fk}_long", {})
                ss = r.get(f"{fk}_short", {})
                lines.append(
                    f"    {h}-bar: count={s.get('count')}  hit={_pct(s.get('hit_rate'))}  "
                    f"PF={_fmt(s.get('profit_factor'))}  avg_logret={_fmt(s.get('avg_log_return'))}  "
                    f"t={_fmt(s.get('tstat'))}  p={_fmt(s.get('pval'))}"
                )
                if sl.get("count", 0) >= 5 and ss.get("count", 0) >= 5:
                    lines.append(
                        f"         long:  hit={_pct(sl.get('hit_rate'))}  PF={_fmt(sl.get('profit_factor'))}"
                        f" | short: hit={_pct(ss.get('hit_rate'))}  PF={_fmt(ss.get('profit_factor'))}"
                    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
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
        combo = entry.get("combo", "")
        parts = combo.rsplit("_", 1)
        asset = parts[0] if parts else None
        tf    = parts[1] if len(parts) > 1 else None
        best  = best_by_combo.get(combo, {})
        r     = best.get("stats", {})
        sharpe = _estimate_sharpe(r)

        kb_entry = {
            "category":      entry.get("category", "partial"),
            "indicator":     spec["indicator"],
            "timeframe":     tf,
            "asset":         asset,
            "session":       None,
            "summary":       entry.get("summary", ""),
            "sharpe_ref":    sharpe,
            "strategy_id":   None,
            "p_value":       mcpt_pvalue,
            "profit_factor": (
                (r.get("exit_rule") or {}).get("profit_factor")
                if spec.get("spec_type") == "exit_research"
                else (r.get("fwd_5") or {}).get("profit_factor")
            ),
            "t_stat":        (r.get("fwd_5") or {}).get("tstat"),
            "optimal_params": best.get("params"),
        }
        try:
            db.insert_knowledge(kb_entry)
            saved.append({**kb_entry, "p_value": mcpt_pvalue})
        except Exception as exc:
            log.warning("knowledge_insert_failed error=%s", exc)

    return saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_summary(
    spec: dict,
    best_by_combo: dict,
    knowledge: list[dict],
    mcpt_pvalue: float | None,
) -> str:
    works   = [e for e in knowledge if e.get("category") == "works"]
    fails   = [e for e in knowledge if e.get("category") == "fails"]
    partial = [e for e in knowledge if e.get("category") == "partial"]
    pval_str = f"  MCPT p={mcpt_pvalue:.4f}" if mcpt_pvalue is not None else ""
    return (
        f"{spec['title']}: tested on {len(best_by_combo)} combos.{pval_str}  "
        f"Works: {len(works)}  Fails: {len(fails)}  Partial: {len(partial)}.  "
        + (knowledge[0]["summary"] if knowledge else "No significant signal found.")
    )


def _estimate_sharpe(r: dict) -> float | None:
    f5    = r.get("fwd_5", {})
    tstat = f5.get("tstat")
    count = f5.get("count") or 0
    if tstat is None or count < 10:
        return None
    import math
    return round(tstat / math.sqrt(count) * math.sqrt(252), 2)


def _pvalue_to_confidence(p_value: float | None) -> float:
    if p_value is None:
        return 0.5
    # p=0.01 → 0.99 confidence, p=0.10 → 0.90, p=0.50 → 0.50
    return round(max(0.0, min(1.0, 1.0 - p_value)), 2)


def _pct(v) -> str:
    return f"{v*100:.1f}%" if v is not None else "N/A"


def _fmt(v) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def _log_spend(agent: str, usage, strategy_id) -> None:
    try:
        from db import supabase_client as db
        cost = (usage.input_tokens * 0.003 + usage.output_tokens * 0.015) / 1000
        db.log_spend(agent, MODEL, usage.input_tokens, usage.output_tokens, cost, strategy_id)
    except Exception:
        pass
