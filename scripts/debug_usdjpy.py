"""Quick debug: how many LONG signals fire for USDJPY in 2026?"""
import sys
sys.path.insert(0, r"C:\Users\Тимур\Documents\GitHub\trading_research\my")

import numpy as np
import pandas as pd
from strategy025 import get_data, add_daily_candles, find_swing_points, find_previous_swing

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

for ticker in ["EURUSD", "USDJPY", "GBPUSD"]:
    print(f"\n=== {ticker} ===")
    df = get_data(ticker=ticker, timeframe="5m", start="2026-01-01", end="2026-05-16")
    df = add_daily_candles(df)

    high = df["High"].values
    low = df["Low"].values
    daily_color = df["daily_color"].values
    daily_color_prev = df["daily_color_prev"].values
    n = len(df)

    # Count green/red days
    daily_unique_dates = df.index.normalize().unique()
    green_days = sum(1 for d in daily_unique_dates if daily_color[df.index.normalize() == d][0] == 1)
    red_days = sum(1 for d in daily_unique_dates if daily_color[df.index.normalize() == d][0] == -1)
    print(f"  Days: total={len(daily_unique_dates)}, green={green_days}, red={red_days}")

    # Count signals
    swing_highs, swing_lows = find_swing_points(high, low, 10)
    n_swing_lows = sum(1 for x in swing_lows if not np.isnan(x))
    n_swing_highs = sum(1 for x in swing_highs if not np.isnan(x))
    print(f"  Swing lows={n_swing_lows}, swing highs={n_swing_highs}")

    # Count LONG signals
    n_long = 0
    for i in range(11, n):
        if not np.isnan(swing_lows[i]):
            prev = find_previous_swing(swing_lows, i)
            if not np.isnan(prev) and swing_lows[i] > prev:
                if daily_color_prev[i] == 1 and daily_color[i] == 1:
                    n_long += 1
    print(f"  LONG signals (higher low + 2 green days): {n_long}")

    # Count SHORT signals
    n_short = 0
    for i in range(11, n):
        if not np.isnan(swing_highs[i]):
            prev = find_previous_swing(swing_highs, i)
            if not np.isnan(prev) and swing_highs[i] < prev:
                if daily_color_prev[i] == -1 and daily_color[i] == -1:
                    n_short += 1
    print(f"  SHORT signals (lower high + 2 red days): {n_short}")
