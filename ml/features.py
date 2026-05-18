"""
Feature builders — all past-only.

Pure-function contract: at row index t, the feature value depends only on
df.iloc[:t+1]. Verified by ml.leakage.assert_no_lookahead and by the static scan.

Two profiles:
  build_features(df, df_htf, fast=True)  -> ~20 essentials
  build_features(df, df_htf, fast=False) -> full set
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# pandas_ta indicators are all past-only.
import pandas_ta as ta  # noqa: F401  (used via DataFrame.ta accessor)


# ── primitives ──────────────────────────────────────────────────────────────


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    roll_dn = down.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _ema(close: pd.Series, length: int) -> pd.Series:
    return close.ewm(span=length, adjust=False, min_periods=length).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _linreg_slope(s: pd.Series, length: int) -> pd.Series:
    """Slope of linear regression over rolling window. Past-only."""
    x = np.arange(length)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def slope(window):
        y = window
        y_mean = y.mean()
        return ((x - x_mean) * (y - y_mean)).sum() / x_var

    return s.rolling(length, min_periods=length).apply(slope, raw=True)


def _supertrend(df: pd.DataFrame, length: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Returns DataFrame with columns: st_dir (1=up, -1=down), st_dist (close-ST in ATR units)."""
    atr = _atr(df, length)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    n = len(df)
    final_upper = upper.values.copy()
    final_lower = lower.values.copy()
    direction = np.ones(n, dtype=int)
    st = np.full(n, np.nan)

    close = df["close"].values
    for i in range(1, n):
        # carry forward bands
        if not np.isnan(upper.iloc[i]) and not np.isnan(final_upper[i - 1]):
            if upper.iloc[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                final_upper[i] = upper.iloc[i]
            else:
                final_upper[i] = final_upper[i - 1]
            if lower.iloc[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                final_lower[i] = lower.iloc[i]
            else:
                final_lower[i] = final_lower[i - 1]
        # direction
        if direction[i - 1] == 1 and close[i] < final_lower[i]:
            direction[i] = -1
        elif direction[i - 1] == -1 and close[i] > final_upper[i]:
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]
        st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    out = pd.DataFrame(index=df.index)
    out["st_dir"] = direction
    out["st_dist"] = (df["close"].values - st) / atr.values
    return out


def _session_tag(idx: pd.DatetimeIndex) -> pd.Series:
    """0=other, 1=Asia, 2=London, 3=NY, 4=London+NY overlap. UTC hours."""
    h = idx.hour
    out = pd.Series(0, index=idx, dtype="int8")
    out[(h >= 0) & (h < 7)] = 1
    out[(h >= 7) & (h < 12)] = 2
    out[(h >= 12) & (h < 16)] = 4
    out[(h >= 16) & (h < 21)] = 3
    return out


def _bars_since_session_open(idx: pd.DatetimeIndex) -> pd.Series:
    """Counts bars since most recent session change."""
    sess = _session_tag(idx).values
    out = np.zeros(len(idx), dtype=int)
    for i in range(1, len(idx)):
        out[i] = 0 if sess[i] != sess[i - 1] else out[i - 1] + 1
    return pd.Series(out, index=idx)


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP that resets at session open (defined by _session_tag changes)."""
    sess = _session_tag(df.index)
    group = (sess != sess.shift(1)).cumsum()
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"].clip(lower=1)
    vol = df["volume"].clip(lower=1)
    return pv.groupby(group).cumsum() / vol.groupby(group).cumsum()


def _failed_extreme_bars(df: pd.DataFrame, lookback: int) -> tuple[pd.Series, pd.Series]:
    """
    For each bar t, count bars since the rolling max-high (resp. min-low) was set
    *and not exceeded*. Larger value = older extreme that has held — i.e., a stronger
    resistance/support level.
    """
    n = len(df)
    high = df["high"].values
    low = df["low"].values
    bars_since_high = np.zeros(n, dtype=int)
    bars_since_low = np.zeros(n, dtype=int)
    for i in range(1, n):
        start = max(0, i - lookback + 1)
        window_high = high[start:i + 1]
        window_low = low[start:i + 1]
        argmax_off = len(window_high) - 1 - int(np.argmax(window_high[::-1]))
        argmin_off = len(window_low) - 1 - int(np.argmin(window_low[::-1]))
        bars_since_high[i] = i - (start + argmax_off)
        bars_since_low[i] = i - (start + argmin_off)
    return (
        pd.Series(bars_since_high, index=df.index),
        pd.Series(bars_since_low, index=df.index),
    )


def _rolling_naive_winrate(
    df: pd.DataFrame, atr: pd.Series, lookback: int, rr: float, sl_mult: float, side: str
) -> pd.Series:
    """
    For each bar t, compute the fraction of bars in [t-lookback, t-1] where a naive
    entry-on-close, TP/SL=rr would have hit TP within the next H bars (H=24 for 5m).
    Conservative: uses subsequent bar highs/lows on the SAME timeframe.

    This is a *past-only* feature — at bar t we look at outcomes that have already
    resolved by bar t. We resolve up to bar t-H to be safe.
    """
    n = len(df)
    H = 24
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    atr_v = atr.values
    won = np.full(n, np.nan)

    for i in range(H + 1, n):
        a = atr_v[i - H - 1]
        if np.isnan(a) or a == 0:
            continue
        entry = close[i - H - 1]
        if side == "long":
            tp = entry + sl_mult * a * rr
            sl = entry - sl_mult * a
        else:
            tp = entry - sl_mult * a * rr
            sl = entry + sl_mult * a
        outcome = 0
        for j in range(i - H, i + 1):
            if side == "long":
                if low[j] <= sl:
                    outcome = 0
                    break
                if high[j] >= tp:
                    outcome = 1
                    break
            else:
                if high[j] >= sl:
                    outcome = 0
                    break
                if low[j] <= tp:
                    outcome = 1
                    break
        won[i] = outcome

    s = pd.Series(won, index=df.index)
    return s.rolling(lookback, min_periods=max(20, lookback // 4)).mean()


# ── higher TF merge ────────────────────────────────────────────────────────


def _daily_features(df: pd.DataFrame, atr14: pd.Series) -> pd.DataFrame:
    """
    Daily-candle features. All past-only:
      d_today_green   : 1 if today's running close > today's open
      d_today_body    : (close - open) / today's running range
      d_today_pos     : (close - today_low) / today's running range
      d_prev_green    : 1 if yesterday's full daily candle is green
      d_prev_body     : yesterday's signed body / range
      d_streak        : signed count of consecutive same-direction daily closes
                        (positive = green run, negative = red run)
      d_close_vs_pdh  : (close - prev_day_high) / atr   (negative = below PDH)
      d_close_vs_pdl  : (close - prev_day_low)  / atr   (positive = above PDL)
    """
    out = pd.DataFrame(index=df.index)
    day = pd.Index(df.index.date)

    # ── today running (past-only via groupby + cummax/cummin/transform-first) ──
    g_open = df.groupby(day)["open"].transform("first")
    g_high = df.groupby(day)["high"].cummax()
    g_low = df.groupby(day)["low"].cummin()
    close = df["close"]
    rng = (g_high - g_low).replace(0, np.nan)

    out["d_today_green"] = (close > g_open).astype("int8")
    out["d_today_body"] = (close - g_open) / rng
    out["d_today_pos"] = (close - g_low) / rng

    # ── daily resample, then asof-merge yesterday's row into base index ──
    daily = df.resample("1D", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()
    daily["green"] = (daily["close"] > daily["open"]).astype("int8")
    daily["body_signed"] = (
        (daily["close"] - daily["open"])
        / (daily["high"] - daily["low"]).replace(0, np.nan)
    )

    signs = np.where(daily["green"].values == 1, 1, -1)
    streak = np.zeros(len(daily), dtype=int)
    if len(daily) > 0:
        streak[0] = int(signs[0])
        for i in range(1, len(daily)):
            if signs[i] == signs[i - 1]:
                streak[i] = streak[i - 1] + int(signs[i])
            else:
                streak[i] = int(signs[i])
    daily["streak"] = streak

    prev_df = pd.DataFrame({
        "d_prev_green": daily["green"].astype("int8"),
        "d_prev_body": daily["body_signed"],
        "d_streak": daily["streak"],
        "_pdh": daily["high"],
        "_pdl": daily["low"],
    })

    out = _merge_htf(out, prev_df, "")
    atr_safe = atr14.replace(0, np.nan)
    out["d_close_vs_pdh"] = (close - out["_pdh"]) / atr_safe
    out["d_close_vs_pdl"] = (close - out["_pdl"]) / atr_safe
    out = out.drop(columns=["_pdh", "_pdl"], errors="ignore")
    return out


def _merge_htf(base: pd.DataFrame, htf: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Asof-merge HTF features into base index. The HTF bar timestamped at time T
    closes at exactly T (label='right', closed='right' in our resampler), so it is
    known at base bar t whenever T <= t — pandas merge_asof direction='backward'.
    No future leakage: a 5m bar at 10:00 sees the 1h bar that just closed at 10:00.
    """
    base = base.sort_index().copy()
    htf = htf.sort_index().copy()
    # Normalize datetime precision so merge_asof doesn't error on ns vs ms
    base.index = pd.DatetimeIndex(base.index).as_unit("ns")
    htf.index = pd.DatetimeIndex(htf.index).as_unit("ns")
    out = pd.merge_asof(
        base, htf,
        left_index=True, right_index=True,
        direction="backward",
        suffixes=("", suffix),
    )
    return out


# ── main entry point ───────────────────────────────────────────────────────


def build_features(
    df: pd.DataFrame,
    df_htf: pd.DataFrame | None = None,
    fast: bool = False,
) -> pd.DataFrame:
    """
    Build feature matrix indexed identically to `df`. All features are past-only.
    Caller is responsible for dropping initial NaN rows after warmup.
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Trend / momentum
    out["rsi_14"] = _rsi(close, 14)
    out["ema_20"] = (close / _ema(close, 20)) - 1.0
    out["ema_50"] = (close / _ema(close, 50)) - 1.0
    out["ema_200"] = (close / _ema(close, 200)) - 1.0
    out["ema50_slope"] = _linreg_slope(_ema(close, 50), 10) / close
    out["ema200_slope"] = _linreg_slope(_ema(close, 200), 20) / close

    # Volatility
    atr14 = _atr(df, 14)
    atr50 = _atr(df, 50)
    out["atr_14"] = atr14 / close
    out["atr_ratio"] = atr14 / atr50

    # Position in range
    for N in (20, 50, 100, 200):
        rng_max = high.rolling(N, min_periods=N).max()
        rng_min = low.rolling(N, min_periods=N).min()
        out[f"pos_{N}"] = (close - rng_min) / (rng_max - rng_min).replace(0, np.nan)

    # Time
    h = df.index.hour
    out["hour_sin"] = np.sin(2 * np.pi * h / 24)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    out["weekday"] = df.index.weekday.astype("int8")
    out["session"] = _session_tag(df.index).astype("int8")

    if fast:
        # adaptive context — cheap version
        out["naive_long_wr_200"] = _rolling_naive_winrate(df, atr14, 200, 3.0, 1.5, "long")
        return out

    # ── heavy-only features ────────────────────────────────────────────────
    out["rsi_50"] = _rsi(close, 50)

    macd = df.ta.macd(fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out["macd"] = macd.iloc[:, 0] / close
        out["macd_signal"] = macd.iloc[:, 2] / close
        out["macd_hist"] = macd.iloc[:, 1] / close

    st = _supertrend(df, 10, 3.0)
    out["st_dir"] = st["st_dir"].astype("int8")
    out["st_dist"] = st["st_dist"]

    bb = df.ta.bbands(length=20, std=2.0)
    if bb is not None and not bb.empty:
        upper = bb.iloc[:, 2]
        lower = bb.iloc[:, 0]
        out["bb_width"] = (upper - lower) / close
        out["bb_pos"] = (close - lower) / (upper - lower).replace(0, np.nan)

    # VWAP
    vwap = _session_vwap(df)
    out["vwap_dist"] = (close - vwap) / (atr14.replace(0, np.nan))
    out["hl2_sma20_dist"] = (close - ((high + low) / 2).rolling(20, min_periods=20).mean()) / atr14

    # Structure
    bars_h, bars_l = _failed_extreme_bars(df, lookback=200)
    out["bars_since_hi"] = bars_h
    out["bars_since_lo"] = bars_l

    # Candle anatomy
    body = (close - df["open"]).abs()
    rng = (high - low).replace(0, np.nan)
    out["body_ratio"] = body / rng
    out["upper_wick"] = (high - df[["open", "close"]].max(axis=1)) / rng
    out["lower_wick"] = (df[["open", "close"]].min(axis=1) - low) / rng

    # Engulfing / inside / pin (simple boolean flags)
    prev_body = body.shift(1)
    prev_o = df["open"].shift(1)
    prev_c = close.shift(1)
    bull_eng = ((close > df["open"]) & (prev_c < prev_o) & (close > prev_o) & (df["open"] < prev_c)).astype("int8")
    bear_eng = ((close < df["open"]) & (prev_c > prev_o) & (close < prev_o) & (df["open"] > prev_c)).astype("int8")
    out["bull_engulf"] = bull_eng
    out["bear_engulf"] = bear_eng
    out["inside_bar"] = ((high < high.shift(1)) & (low > low.shift(1))).astype("int8")

    # Time-since-session-open
    out["bars_since_sess"] = _bars_since_session_open(df.index)

    # Adaptive rolling win-rate (both sides)
    out["naive_long_wr_200"] = _rolling_naive_winrate(df, atr14, 200, 3.0, 1.5, "long")
    out["naive_short_wr_200"] = _rolling_naive_winrate(df, atr14, 200, 3.0, 1.5, "short")

    # Daily candle features
    daily_df = _daily_features(df, atr14)
    out = pd.concat([out, daily_df.reindex(out.index)], axis=1)

    # ── HTF features ───────────────────────────────────────────────────────
    if df_htf is not None and not df_htf.empty:
        htf_feats = pd.DataFrame(index=df_htf.index)
        atr_h = _atr(df_htf, 14)
        htf_feats["rsi"] = _rsi(df_htf["close"], 14)
        htf_feats["ema50_slope"] = _linreg_slope(_ema(df_htf["close"], 50), 10) / df_htf["close"]
        htf_feats["pos_50"] = (
            (df_htf["close"] - df_htf["low"].rolling(50, min_periods=50).min())
            / (df_htf["high"].rolling(50, min_periods=50).max() - df_htf["low"].rolling(50, min_periods=50).min()).replace(0, np.nan)
        )
        st_h = _supertrend(df_htf, 10, 3.0)
        htf_feats["st_dir"] = st_h["st_dir"].astype("int8")
        body_h = (df_htf["close"] - df_htf["open"]).abs()
        rng_h = (df_htf["high"] - df_htf["low"]).replace(0, np.nan)
        htf_feats["body_ratio"] = body_h / rng_h
        out = _merge_htf(out, htf_feats, "_h")

    return out
