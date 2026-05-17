"""
Debug script: compare signal/entry bars between strategy025.py logic and backtrader version.

Generates two sets of entry bar timestamps - if they match, the swing/daily_color
detection is identical. If they differ, the discrepancy is in detection.
If they match but trade counts differ - the issue is in execution.
"""
import sys
sys.path.insert(0, r"C:\Users\Тимур\Documents\GitHub\trading_research\my")

import numpy as np
import pandas as pd
from strategy025 import get_data, add_daily_candles, find_swing_points, find_previous_swing

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOOKBACK = 10
DIRECTION = 1  # LONG only

# Load same data
print("Loading EURUSD 5m data...")
df = get_data(ticker="EURUSD", timeframe="5m", start="2026-01-01", end="2026-05-16")
df = add_daily_candles(df)

# Detect signals using strategy025.py logic
high = df["High"].values
low = df["Low"].values
close = df["Close"].values
daily_color = df["daily_color"].values
daily_color_prev = df["daily_color_prev"].values
n = len(df)

swing_highs_arr, swing_lows_arr = find_swing_points(high, low, LOOKBACK)

# Find entry signal bars (using original logic exactly)
original_signals = []  # list of (signal_bar_idx, entry_bar_idx, swing_low, prev_swing)
for i in range(LOOKBACK + 1, n):
    if DIRECTION == 1:  # LONG
        if not np.isnan(swing_lows_arr[i]):
            prev_swing = find_previous_swing(swing_lows_arr, i)
            if not np.isnan(prev_swing) and swing_lows_arr[i] > prev_swing:
                if daily_color_prev[i] == 1 and daily_color[i] == 1:
                    original_signals.append((i, i + 1, swing_lows_arr[i], prev_swing))

print(f"\nOriginal logic: {len(original_signals)} signals (entry triggers)")

# Now use MY backtrader-style detection logic (without backtrader)
# Detect swings the same way, but step through bar-by-bar like next()
my_signals = []
my_swing_lows = []  # list of (bar_idx, price)

for bar_idx in range(LOOKBACK + 1, n):
    # Detect if PREVIOUS bar (bar_idx - 1) was a swing low
    prev_bar_idx = bar_idx - 1
    prev_bar_low = low[prev_bar_idx]
    # Lookback window for the previous bar: bars [prev_bar_idx - lookback ... prev_bar_idx - 1]
    win_start = max(0, prev_bar_idx - LOOKBACK)
    win_end = prev_bar_idx  # exclusive
    prev_window_lows = low[win_start:win_end]

    is_prev_swing_low = prev_bar_low < np.min(prev_window_lows)

    if is_prev_swing_low:
        prev_swing_low_price = my_swing_lows[-1][1] if my_swing_lows else None
        my_swing_lows.append((prev_bar_idx, prev_bar_low))

        if (DIRECTION == 1 and
            prev_swing_low_price is not None and
            prev_bar_low > prev_swing_low_price and
            daily_color_prev[prev_bar_idx] == 1 and
            daily_color[prev_bar_idx] == 1):
            # Entry trigger fires on bar_idx (current)
            my_signals.append((prev_bar_idx, bar_idx, prev_bar_low, prev_swing_low_price))

print(f"My logic     : {len(my_signals)} signals")

# Compare
orig_set = set(s[0] for s in original_signals)  # signal bar indices
mine_set = set(s[0] for s in my_signals)

print(f"\nIn original but not mine: {sorted(orig_set - mine_set)[:20]}")
print(f"In mine but not original: {sorted(mine_set - orig_set)[:20]}")
print(f"Common: {len(orig_set & mine_set)} signals")
