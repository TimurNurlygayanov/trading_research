"""
Triple-barrier labeling.

For each TF bar t we hypothetically:
  - enter at open(t+1) (the next bar's open) at the ask (long) or bid (short)
  - place SL at entry ± sl_mult * ATR(t)        (ATR uses bars <= t)
  - place TP at entry ± sl_mult * ATR(t) * rr
  - exit when SL or TP is touched first, or at timeout after max_hold_bars

Costs are baked in: long entry pays half-spread above mid; long exit pays half-spread below.
Mirrored for short. Commission is applied in the simulator (not the labeler).

Resolution:
  - If df_1m is provided, we drill into 1m bars between TF bars to break "both touched in
    same TF bar" ambiguity correctly.
  - If df_1m is None, we use a conservative rule: SL wins ties within a bar.

Returns y_long, y_short, t_exit (timestamp), tp_long, sl_long, tp_short, sl_short.
y = 1 if TP hit first, 0 if SL hit first or timeout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _atr_for_labels(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _resolve_with_1m(
    df_1m: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    tp: float,
    sl: float,
    side: str,
) -> int:
    """Return 1 if TP hit first, 0 if SL hit first, -1 if neither in [start, end]."""
    sub = df_1m.loc[start_ts:end_ts]
    if sub.empty:
        return -1
    highs = sub["high"].values
    lows = sub["low"].values
    if side == "long":
        for i in range(len(sub)):
            hit_sl = lows[i] <= sl
            hit_tp = highs[i] >= tp
            if hit_sl and hit_tp:
                return 0  # conservative tie-break inside a 1m bar
            if hit_sl:
                return 0
            if hit_tp:
                return 1
    else:
        for i in range(len(sub)):
            hit_sl = highs[i] >= sl
            hit_tp = lows[i] <= tp
            if hit_sl and hit_tp:
                return 0
            if hit_sl:
                return 0
            if hit_tp:
                return 1
    return -1


def build_labels(
    df: pd.DataFrame,
    df_1m: pd.DataFrame | None,
    sl_atr: float = 1.5,
    rr: float = 3.0,
    max_hold_bars: int = 48,
    spread: float = 0.00008,
    tf_minutes: int = 5,
) -> pd.DataFrame:
    """
    Returns DataFrame indexed like df, with columns:
        y_long, y_short      ∈ {0, 1, NaN}
        entry_long, entry_short, tp_long, sl_long, tp_short, sl_short
        t_exit_long, t_exit_short
        hold_long, hold_short (bars)

    NaN where unresolved (would be unsafe to label — typically the tail of the series).
    """
    n = len(df)
    atr = _atr_for_labels(df, 14)
    half_sp = spread / 2.0

    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    atr_arr = atr.values
    idx = df.index

    y_l = np.full(n, np.nan)
    y_s = np.full(n, np.nan)
    e_l = np.full(n, np.nan)
    e_s = np.full(n, np.nan)
    tp_l = np.full(n, np.nan)
    sl_l = np.full(n, np.nan)
    tp_s = np.full(n, np.nan)
    sl_s = np.full(n, np.nan)
    hold_l = np.full(n, np.nan)
    hold_s = np.full(n, np.nan)

    for t in range(n - 1):
        a = atr_arr[t]
        if np.isnan(a) or a <= 0:
            continue

        entry_mid = open_arr[t + 1]
        entry_long = entry_mid + half_sp
        entry_short = entry_mid - half_sp
        sl_dist = sl_atr * a
        tp_dist = sl_dist * rr

        # Levels are quoted in mid; we set them so that on touch, the net pnl
        # equals ±sl_dist (modulo spread already paid on entry/exit).
        tp_long = entry_long + tp_dist
        sl_long = entry_long - sl_dist
        tp_short = entry_short - tp_dist
        sl_short = entry_short + sl_dist

        e_l[t] = entry_long
        e_s[t] = entry_short
        tp_l[t] = tp_long
        sl_l[t] = sl_long
        tp_s[t] = tp_short
        sl_s[t] = sl_short

        end = min(t + 1 + max_hold_bars, n - 1)
        outcome_long = -1
        outcome_short = -1
        bars_long = max_hold_bars
        bars_short = max_hold_bars

        for j in range(t + 1, end + 1):
            if df_1m is not None:
                bar_end = idx[j]
                bar_start = idx[j] - pd.Timedelta(minutes=tf_minutes) + pd.Timedelta("1ns")
                if outcome_long == -1:
                    res = _resolve_with_1m(df_1m, bar_start, bar_end, tp_long, sl_long, "long")
                    if res != -1:
                        outcome_long = res
                        bars_long = j - t
                if outcome_short == -1:
                    res = _resolve_with_1m(df_1m, bar_start, bar_end, tp_short, sl_short, "short")
                    if res != -1:
                        outcome_short = res
                        bars_short = j - t
            else:
                if outcome_long == -1:
                    hit_sl = low_arr[j] <= sl_long
                    hit_tp = high_arr[j] >= tp_long
                    if hit_sl:
                        outcome_long = 0
                        bars_long = j - t
                    elif hit_tp:
                        outcome_long = 1
                        bars_long = j - t
                if outcome_short == -1:
                    hit_sl = high_arr[j] >= sl_short
                    hit_tp = low_arr[j] <= tp_short
                    if hit_sl:
                        outcome_short = 0
                        bars_short = j - t
                    elif hit_tp:
                        outcome_short = 1
                        bars_short = j - t
            if outcome_long != -1 and outcome_short != -1:
                break

        y_l[t] = outcome_long if outcome_long != -1 else 0  # timeout = loss
        y_s[t] = outcome_short if outcome_short != -1 else 0
        hold_l[t] = bars_long
        hold_s[t] = bars_short

    out = pd.DataFrame(
        {
            "y_long": y_l,
            "y_short": y_s,
            "entry_long": e_l,
            "entry_short": e_s,
            "tp_long": tp_l,
            "sl_long": sl_l,
            "tp_short": tp_s,
            "sl_short": sl_s,
            "hold_long": hold_l,
            "hold_short": hold_s,
            "atr": atr_arr,
        },
        index=df.index,
    )
    return out
