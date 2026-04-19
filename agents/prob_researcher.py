"""
Statistical probability analysis of market conditions.
Tests whether specific conditions produce a statistically significant
forward-return bias (bullish or bearish skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from dataclasses import dataclass, field


@dataclass
class ConditionSpec:
    condition_id: str
    description: str
    category: str  # candle | ema | session | volatility | momentum
    forward_bars: list[int] = field(default_factory=lambda: [1, 4, 12, 24])
    params: dict = field(default_factory=dict)  # condition hyper-params for reference


# ── Technical helpers ────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR using Wilder's method."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using ewm."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(close: pd.Series, period: int) -> pd.Series:
    """pandas ewm EMA."""
    return close.ewm(span=period, adjust=False).mean()


def _bb_upper(close: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
    """Bollinger upper band."""
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return ma + std_dev * std


def _bb_lower(close: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
    """Bollinger lower band."""
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return ma - std_dev * std


# ── Condition catalogue ──────────────────────────────────────────────────────

CONDITION_CATALOGUE: list[dict] = [
    # candle
    {"id": "prev_candle_bullish",   "desc": "Previous candle is bullish (close > open)",                   "cat": "candle"},
    {"id": "prev_candle_bearish",   "desc": "Previous candle is bearish (close < open)",                   "cat": "candle"},
    {"id": "prev_candle_bull_large","desc": "Previous candle bullish and body > 1× ATR(14)",               "cat": "candle"},
    {"id": "prev_candle_bear_large","desc": "Previous candle bearish and body > 1× ATR(14)",               "cat": "candle"},
    {"id": "prev_2_bullish",        "desc": "Last 2 candles both bullish",                                  "cat": "candle"},
    {"id": "prev_2_bearish",        "desc": "Last 2 candles both bearish",                                  "cat": "candle"},
    {"id": "prev_3_bullish",        "desc": "Last 3 candles all bullish",                                   "cat": "candle"},
    {"id": "prev_3_bearish",        "desc": "Last 3 candles all bearish",                                   "cat": "candle"},
    {"id": "close_near_high",       "desc": "Close in top 25% of current candle range",                    "cat": "candle"},
    {"id": "close_near_low",        "desc": "Close in bottom 25% of current candle range",                 "cat": "candle"},
    {"id": "inside_bar",            "desc": "Current bar is an inside bar (high < prev high, low > prev low)", "cat": "candle"},
    {"id": "doji",                  "desc": "Doji: body < 10% of full candle range",                       "cat": "candle"},
    {"id": "large_body",            "desc": "Large body: body > 70% of full candle range",                 "cat": "candle"},
    # ema
    {"id": "above_ema20",           "desc": "Close above EMA(20)",                                         "cat": "ema"},
    {"id": "below_ema20",           "desc": "Close below EMA(20)",                                         "cat": "ema"},
    {"id": "above_ema50",           "desc": "Close above EMA(50)",                                         "cat": "ema"},
    {"id": "below_ema50",           "desc": "Close below EMA(50)",                                         "cat": "ema"},
    {"id": "above_ema200",          "desc": "Close above EMA(200)",                                        "cat": "ema"},
    {"id": "below_ema200",          "desc": "Close below EMA(200)",                                        "cat": "ema"},
    {"id": "ema20_above_ema50",     "desc": "EMA(20) above EMA(50) — golden cross regime",                 "cat": "ema"},
    {"id": "ema20_below_ema50",     "desc": "EMA(20) below EMA(50) — death cross regime",                  "cat": "ema"},
    {"id": "cross_ema20_up",        "desc": "Close just crossed above EMA(20)",                            "cat": "ema"},
    {"id": "cross_ema20_dn",        "desc": "Close just crossed below EMA(20)",                            "cat": "ema"},
    # session
    {"id": "session_asian",         "desc": "Asian session (00:00–06:59 UTC)",                             "cat": "session"},
    {"id": "session_london",        "desc": "London session (07:00–12:59 UTC)",                            "cat": "session"},
    {"id": "session_overlap",       "desc": "London/NY overlap (13:00–16:59 UTC)",                        "cat": "session"},
    {"id": "session_ny",            "desc": "New York session (13:00–21:59 UTC)",                         "cat": "session"},
    {"id": "hour_10_12",            "desc": "Hour 10–11 UTC",                                              "cat": "session"},
    {"id": "hour_13_15",            "desc": "Hour 13–14 UTC",                                              "cat": "session"},
    {"id": "hour_18_22",            "desc": "Hour 18–21 UTC",                                              "cat": "session"},
    {"id": "weekday_monday",        "desc": "Monday",                                                       "cat": "session"},
    {"id": "weekday_friday",        "desc": "Friday",                                                       "cat": "session"},
    {"id": "weekday_mid",           "desc": "Tuesday–Thursday",                                             "cat": "session"},
    # volatility
    {"id": "high_atr",              "desc": "ATR above 75th percentile (high volatility regime)",          "cat": "volatility"},
    {"id": "low_atr",               "desc": "ATR below 25th percentile (low volatility regime)",           "cat": "volatility"},
    {"id": "above_bb_upper",        "desc": "Close above Bollinger upper band",                            "cat": "volatility"},
    {"id": "below_bb_lower",        "desc": "Close below Bollinger lower band",                            "cat": "volatility"},
    {"id": "inside_bb",             "desc": "Close inside Bollinger bands",                                "cat": "volatility"},
    # momentum
    {"id": "rsi_oversold",          "desc": "RSI(14) < 30 — oversold",                                    "cat": "momentum"},
    {"id": "rsi_overbought",        "desc": "RSI(14) > 70 — overbought",                                  "cat": "momentum"},
    {"id": "rsi_mid_bull",          "desc": "RSI(14) in [45, 55] and rising",                             "cat": "momentum"},
    {"id": "rsi_mid_bear",          "desc": "RSI(14) in [45, 55] and falling",                            "cat": "momentum"},
    {"id": "rsi_above_50",          "desc": "RSI(14) > 50",                                               "cat": "momentum"},
    {"id": "rsi_below_50",          "desc": "RSI(14) < 50",                                               "cat": "momentum"},
    {"id": "rsi_cross_50_up",       "desc": "RSI(14) crossed above 50",                                   "cat": "momentum"},
    {"id": "rsi_cross_50_dn",       "desc": "RSI(14) crossed below 50",                                   "cat": "momentum"},
    # combo: BB + RSI — high-conviction mean-reversion signals
    {"id": "bb_lower_rsi_oversold", "desc": "Close below BB lower band AND RSI(14) < 30",                 "cat": "momentum"},
    {"id": "bb_upper_rsi_overbought","desc": "Close above BB upper band AND RSI(14) > 70",                "cat": "momentum"},
    {"id": "bb_lower_rsi_mid",      "desc": "Close below BB lower band AND RSI(14) in [30, 50]",          "cat": "momentum"},
    # volatility squeeze → expansion
    {"id": "bb_squeeze",            "desc": "BB width < 25th percentile (volatility squeeze)",             "cat": "volatility"},
    {"id": "bb_expansion",          "desc": "BB width > 75th percentile (volatility expansion)",           "cat": "volatility"},
    # EMA distance — extended vs compressed
    {"id": "far_above_ema50",       "desc": "Close > 1.5× ATR above EMA(50) (extended up)",               "cat": "ema"},
    {"id": "far_below_ema50",       "desc": "Close < 1.5× ATR below EMA(50) (extended down)",             "cat": "ema"},
    {"id": "near_ema50",            "desc": "Close within 0.3× ATR of EMA(50) (compressed near MA)",      "cat": "ema"},
    # calendar / time
    {"id": "weekday_tuesday",       "desc": "Tuesday",                                                      "cat": "session"},
    {"id": "weekday_wednesday",     "desc": "Wednesday",                                                    "cat": "session"},
    {"id": "weekday_thursday",      "desc": "Thursday",                                                     "cat": "session"},
    {"id": "month_end",             "desc": "Last 3 trading days of the month",                            "cat": "session"},
    {"id": "month_start",           "desc": "First 3 trading days of the month",                           "cat": "session"},
    # price structure
    {"id": "new_5bar_high",         "desc": "Current close is highest of last 5 bars",                    "cat": "candle"},
    {"id": "new_5bar_low",          "desc": "Current close is lowest of last 5 bars",                     "cat": "candle"},
    {"id": "new_20bar_high",        "desc": "Current close is highest of last 20 bars",                   "cat": "candle"},
    {"id": "new_20bar_low",         "desc": "Current close is lowest of last 20 bars",                    "cat": "candle"},
    {"id": "gap_up",                "desc": "Open > previous close by > 0.1× ATR (gap up)",               "cat": "candle"},
    {"id": "gap_down",              "desc": "Open < previous close by > 0.1× ATR (gap down)",             "cat": "candle"},
]


def get_all_specs(forward_bars: list[int] | None = None) -> list[ConditionSpec]:
    """Return a ConditionSpec for every entry in CONDITION_CATALOGUE."""
    fb = forward_bars or [1, 4, 12, 24]
    return [
        ConditionSpec(
            condition_id=entry["id"],
            description=entry["desc"],
            category=entry["cat"],
            forward_bars=fb,
        )
        for entry in CONDITION_CATALOGUE
    ]


# ── Condition dispatcher ─────────────────────────────────────────────────────

def compute_condition(df: pd.DataFrame, condition_id: str) -> pd.Series:
    """
    Compute a boolean Series for the given condition_id, aligned to df.index.
    Returns a Series of False on any exception.
    """
    try:
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        open_ = df["Open"]
        hour  = df.index.hour
        dow   = df.index.dayofweek

        # ── candle ──────────────────────────────────────────────────────────
        if condition_id == "prev_candle_bullish":
            return (close.shift(1) > open_.shift(1))

        if condition_id == "prev_candle_bearish":
            return (close.shift(1) < open_.shift(1))

        if condition_id == "prev_candle_bull_large":
            atr = _atr(df)
            body = (close.shift(1) - open_.shift(1)).abs()
            return (close.shift(1) > open_.shift(1)) & (body > atr)

        if condition_id == "prev_candle_bear_large":
            atr = _atr(df)
            body = (close.shift(1) - open_.shift(1)).abs()
            return (close.shift(1) < open_.shift(1)) & (body > atr)

        if condition_id == "prev_2_bullish":
            bull1 = close.shift(1) > open_.shift(1)
            bull2 = close.shift(2) > open_.shift(2)
            return bull1 & bull2

        if condition_id == "prev_2_bearish":
            bear1 = close.shift(1) < open_.shift(1)
            bear2 = close.shift(2) < open_.shift(2)
            return bear1 & bear2

        if condition_id == "prev_3_bullish":
            return (
                (close.shift(1) > open_.shift(1))
                & (close.shift(2) > open_.shift(2))
                & (close.shift(3) > open_.shift(3))
            )

        if condition_id == "prev_3_bearish":
            return (
                (close.shift(1) < open_.shift(1))
                & (close.shift(2) < open_.shift(2))
                & (close.shift(3) < open_.shift(3))
            )

        if condition_id == "close_near_high":
            ratio = (close - low) / (high - low).replace(0, np.nan)
            return ratio.fillna(0.5) >= 0.75

        if condition_id == "close_near_low":
            ratio = (close - low) / (high - low).replace(0, np.nan)
            return ratio.fillna(0.5) <= 0.25

        if condition_id == "inside_bar":
            return (high < high.shift(1)) & (low > low.shift(1))

        if condition_id == "doji":
            body = (close - open_).abs()
            rng  = (high - low).replace(0, np.nan)
            return body < 0.1 * rng

        if condition_id == "large_body":
            body = (close - open_).abs()
            rng  = (high - low).replace(0, np.nan)
            return body > 0.7 * rng

        # ── ema ─────────────────────────────────────────────────────────────
        if condition_id == "above_ema20":
            return close > _ema(close, 20)

        if condition_id == "below_ema20":
            return close < _ema(close, 20)

        if condition_id == "above_ema50":
            return close > _ema(close, 50)

        if condition_id == "below_ema50":
            return close < _ema(close, 50)

        if condition_id == "above_ema200":
            return close > _ema(close, 200)

        if condition_id == "below_ema200":
            return close < _ema(close, 200)

        if condition_id == "ema20_above_ema50":
            return _ema(close, 20) > _ema(close, 50)

        if condition_id == "ema20_below_ema50":
            return _ema(close, 20) < _ema(close, 50)

        if condition_id == "cross_ema20_up":
            ema20 = _ema(close, 20)
            return (close > ema20) & (close.shift(1) <= ema20.shift(1))

        if condition_id == "cross_ema20_dn":
            ema20 = _ema(close, 20)
            return (close < ema20) & (close.shift(1) >= ema20.shift(1))

        # ── session ─────────────────────────────────────────────────────────
        if condition_id == "session_asian":
            return pd.Series(hour < 7, index=df.index)

        if condition_id == "session_london":
            return pd.Series((hour >= 7) & (hour < 13), index=df.index)

        if condition_id == "session_overlap":
            return pd.Series((hour >= 13) & (hour < 17), index=df.index)

        if condition_id == "session_ny":
            return pd.Series((hour >= 13) & (hour < 22), index=df.index)

        if condition_id == "hour_10_12":
            return pd.Series((hour == 10) | (hour == 11), index=df.index)

        if condition_id == "hour_13_15":
            return pd.Series((hour == 13) | (hour == 14), index=df.index)

        if condition_id == "hour_18_22":
            return pd.Series((hour >= 18) & (hour <= 21), index=df.index)

        if condition_id == "weekday_monday":
            return pd.Series(dow == 0, index=df.index)

        if condition_id == "weekday_friday":
            return pd.Series(dow == 4, index=df.index)

        if condition_id == "weekday_mid":
            return pd.Series((dow >= 1) & (dow <= 3), index=df.index)

        # ── volatility ──────────────────────────────────────────────────────
        if condition_id == "high_atr":
            atr = _atr(df)
            return atr > atr.rolling(252, min_periods=30).quantile(0.75)

        if condition_id == "low_atr":
            atr = _atr(df)
            return atr < atr.rolling(252, min_periods=30).quantile(0.25)

        if condition_id == "above_bb_upper":
            return close > _bb_upper(close)

        if condition_id == "below_bb_lower":
            return close < _bb_lower(close)

        if condition_id == "inside_bb":
            return (close <= _bb_upper(close)) & (close >= _bb_lower(close))

        # ── momentum ────────────────────────────────────────────────────────
        if condition_id == "rsi_oversold":
            return _rsi(close) < 30

        if condition_id == "rsi_overbought":
            return _rsi(close) > 70

        if condition_id == "rsi_mid_bull":
            rsi = _rsi(close)
            return (rsi >= 45) & (rsi <= 55) & (rsi > rsi.shift(1))

        if condition_id == "rsi_mid_bear":
            rsi = _rsi(close)
            return (rsi >= 45) & (rsi <= 55) & (rsi < rsi.shift(1))

        if condition_id == "rsi_above_50":
            return _rsi(close) > 50

        if condition_id == "rsi_below_50":
            return _rsi(close) < 50

        if condition_id == "rsi_cross_50_up":
            rsi = _rsi(close)
            return (rsi > 50) & (rsi.shift(1) <= 50)

        if condition_id == "rsi_cross_50_dn":
            rsi = _rsi(close)
            return (rsi < 50) & (rsi.shift(1) >= 50)

        # ── combo: BB + RSI ─────────────────────────────────────────────────
        if condition_id == "bb_lower_rsi_oversold":
            return (close < _bb_lower(close)) & (_rsi(close) < 30)

        if condition_id == "bb_upper_rsi_overbought":
            return (close > _bb_upper(close)) & (_rsi(close) > 70)

        if condition_id == "bb_lower_rsi_mid":
            rsi = _rsi(close)
            return (close < _bb_lower(close)) & (rsi >= 30) & (rsi <= 50)

        # ── volatility squeeze / expansion ──────────────────────────────────
        if condition_id == "bb_squeeze":
            width = _bb_upper(close) - _bb_lower(close)
            return width < width.rolling(252, min_periods=30).quantile(0.25)

        if condition_id == "bb_expansion":
            width = _bb_upper(close) - _bb_lower(close)
            return width > width.rolling(252, min_periods=30).quantile(0.75)

        # ── EMA distance ────────────────────────────────────────────────────
        if condition_id == "far_above_ema50":
            atr = _atr(df)
            return (close - _ema(close, 50)) > 1.5 * atr

        if condition_id == "far_below_ema50":
            atr = _atr(df)
            return (_ema(close, 50) - close) > 1.5 * atr

        if condition_id == "near_ema50":
            atr = _atr(df)
            return (close - _ema(close, 50)).abs() < 0.3 * atr

        # ── calendar ────────────────────────────────────────────────────────
        if condition_id == "weekday_tuesday":
            return pd.Series(dow == 1, index=df.index)

        if condition_id == "weekday_wednesday":
            return pd.Series(dow == 2, index=df.index)

        if condition_id == "weekday_thursday":
            return pd.Series(dow == 3, index=df.index)

        if condition_id == "month_end":
            # Last 3 calendar days of the month (works on all timeframes)
            return pd.Series(df.index.to_series().apply(
                lambda ts: ts.day >= (ts.to_period('M').to_timestamp('M').day - 2)
            ).values, index=df.index)

        if condition_id == "month_start":
            return pd.Series(df.index.day <= 3, index=df.index)

        # ── price structure ─────────────────────────────────────────────────
        if condition_id == "new_5bar_high":
            return close >= close.rolling(5).max()

        if condition_id == "new_5bar_low":
            return close <= close.rolling(5).min()

        if condition_id == "new_20bar_high":
            return close >= close.rolling(20).max()

        if condition_id == "new_20bar_low":
            return close <= close.rolling(20).min()

        if condition_id == "gap_up":
            atr = _atr(df)
            return (open_ - close.shift(1)) > 0.1 * atr

        if condition_id == "gap_down":
            atr = _atr(df)
            return (close.shift(1) - open_) > 0.1 * atr

        # Unknown condition — return all-False
        print(f"[prob_researcher] Unknown condition_id: {condition_id}")
        return pd.Series(False, index=df.index)

    except Exception as exc:
        print(f"[prob_researcher] compute_condition({condition_id}) error: {exc}")
        return pd.Series(False, index=df.index)


# ── Bars-per-year lookup ─────────────────────────────────────────────────────

BARS_PER_YEAR: dict[str, int] = {
    "1m":  374400,
    "5m":  74880,
    "15m": 24960,
    "1h":  6240,
    "4h":  1560,
    "1d":  260,
}


# ── Main analysis function ───────────────────────────────────────────────────

def run_analysis(
    df: pd.DataFrame,
    condition_id: str,
    timeframe: str,
    forward_bars: list[int] | None = None,
) -> list[dict]:
    """
    Compute statistical metrics for a condition across multiple forward horizons.

    For each fwd horizon where n >= 30 samples:
      - mean_return, std_return, median_return, hit_rate
      - t_stat, p_value (one-sample t-test vs 0)
      - annualised Sharpe (trade-style)
      - is_significant flag (p < 0.05 and n >= 30)

    Returns a list of result dicts (one per forward horizon), or [] on error.
    All numeric values are converted to Python float for JSONB compatibility.
    """
    if forward_bars is None:
        forward_bars = [1, 4, 12, 24]

    try:
        condition_series = compute_condition(df, condition_id)
    except Exception as exc:
        print(f"[prob_researcher] run_analysis({condition_id}) condition error: {exc}")
        return []

    bpy = BARS_PER_YEAR.get(timeframe, 6240)
    results: list[dict] = []

    for fwd in forward_bars:
        try:
            fwd_return = df["Close"].shift(-fwd) / df["Close"] - 1
            mask = condition_series & fwd_return.notna()
            sample = fwd_return[mask]

            if len(sample) < 30:
                continue

            mean_ret   = float(sample.mean())
            std_ret    = float(sample.std())
            median_ret = float(sample.median())
            hit_rate   = float((sample > 0).mean())
            n_samples  = int(len(sample))

            t_stat, p_value = scipy_stats.ttest_1samp(sample, 0)
            t_stat  = float(t_stat)
            p_value = float(p_value)

            sharpe = float(mean_ret / std_ret * np.sqrt(bpy / fwd)) if std_ret > 0 else 0.0

            results.append({
                "condition_id":  condition_id,
                "forward_bars":  fwd,
                "n_samples":     n_samples,
                "mean_return":   mean_ret,
                "std_return":    std_ret,
                "median_return": median_ret,
                "hit_rate":      hit_rate,
                "t_stat":        t_stat,
                "p_value":       p_value,
                "sharpe":        sharpe,
                "is_significant": bool(p_value < 0.05 and n_samples >= 30),
            })
        except Exception as exc:
            print(f"[prob_researcher] run_analysis({condition_id}, fwd={fwd}) error: {exc}")
            continue

    return results
