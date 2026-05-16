"""
Strategy 025 — Swing Point + Daily Color Filter (EURUSD 1H)

Entry Logic:
  LONG:  New Higher Low (prev swing low) + both yesterday and today daily candles green
  SHORT: New Lower High (prev swing high) + both yesterday and today daily candles red

Features:
  - Test 10 and 20 candle lookback for swing detection
  - Two SL options: (a) at new swing point, (b) at previous swing point
  - 1:3 risk:reward ratio
  - Entry on bar after confirmation

Daily Color:
  Green = close > open (bullish)
  Red = close < open (bearish)
"""
import sys
import time

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


_MT5_TF_MAP = {
    "1m":  1,
    "5m":  5,
    "15m": 15,
    "30m": 30,
    "1h":  16385,
    "4h":  16388,
    "1d":  16408,
}

PIP_SIZE    = 0.0001
SPREAD_PIPS = 1.0


def get_data(ticker="EURUSD", timeframe="1h", start="2026-01-01", end="2026-05-16"):
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end   = pd.Timestamp(end, tz="UTC")
    mt5.initialize()
    mt5.symbol_select(ticker, True)

    tf_const = _MT5_TF_MAP.get(timeframe, 16385)
    CHUNK = 50_000
    frames = []
    pos = 0

    while True:
        chunk = mt5.copy_rates_from_pos(ticker, tf_const, pos, CHUNK)
        if chunk is None or len(chunk) < CHUNK:
            break
        frames.append(pd.DataFrame(chunk))
        oldest_ts = int(frames[-1]["time"].min())
        oldest_utc = pd.Timestamp(oldest_ts, unit="s", tz="UTC")
        if oldest_utc <= ts_start:
            break
        pos += CHUNK

    df = pd.concat(frames[::-1])
    df = df.drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(columns={
        "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "tick_volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]]


def add_daily_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to daily to get daily OHLC and detect candle color."""
    # Create a date column for easier grouping
    df["date"] = df.index.normalize()

    # Group by date and aggregate
    daily_data = df.groupby("date").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).reset_index()

    # Mark candles as green (1) if close > open, red (-1) if close < open, neutral (0) if equal
    daily_data["color"] = np.where(
        daily_data["Close"] > daily_data["Open"],
        1,
        np.where(daily_data["Close"] < daily_data["Open"], -1, 0)
    )

    # Create a mapping of date to color and previous day color
    color_map = dict(zip(daily_data["date"], daily_data["color"]))
    dates_sorted = sorted(color_map.keys())

    # For each 1H bar, find its daily color and previous day color
    daily_colors = []
    daily_colors_prev = []

    for idx in df.index:
        current_date = idx.normalize()

        # Current day color
        daily_colors.append(color_map.get(current_date, 0))

        # Previous day color
        prev_date = current_date - pd.Timedelta(days=1)
        daily_colors_prev.append(color_map.get(prev_date, 0))

    df["daily_color"] = daily_colors
    df["daily_color_prev"] = daily_colors_prev
    df = df.drop(columns=["date"])
    return df


def find_swing_points(high: np.ndarray, low: np.ndarray, lookback: int) -> tuple:
    """
    Detect swing points with NO LOOKAHEAD (realistic).

    A swing low is confirmed at bar i when:
      - low[i] is the minimum of low[i-lookback:i+1]
      - AND we've seen higher lows before it

    This only uses past/current data, no future bars.
    """
    n = len(high)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    # Find local minima using only past data
    for i in range(lookback, n):
        # Check if this bar's low is lower than all previous lookback bars
        min_low_in_window = min(low[max(0, i - lookback):i])
        if low[i] < min_low_in_window:
            swing_lows[i] = low[i]

        # Check if this bar's high is higher than all previous lookback bars
        max_high_in_window = max(high[max(0, i - lookback):i])
        if high[i] > max_high_in_window:
            swing_highs[i] = high[i]

    return swing_highs, swing_lows


def find_previous_swing(swings: np.ndarray, current_idx: int) -> float:
    """Find the previous swing point before current_idx."""
    for i in range(current_idx - 1, -1, -1):
        if not np.isnan(swings[i]):
            return swings[i]
    return np.nan


def backtest_strategy(
    df: pd.DataFrame,
    lookback: int = 10,
    sl_at_prev_swing: bool = False,
    direction: int = 1,  # 1 for long, -1 for short
):
    """
    Backtest swing strategy.

    Args:
        df: DataFrame with Open, High, Low, Close, daily_color, daily_color_prev
        lookback: Number of bars to look back for swing points (10 or 20)
        sl_at_prev_swing: If True, SL at previous swing. If False, SL at current swing.
        direction: 1 for long, -1 for short
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(df)

    daily_color = df["daily_color"].values
    daily_color_prev = df["daily_color_prev"].values

    # Detect swings
    swing_highs, swing_lows = find_swing_points(high, low, lookback)

    # Track which swing triggered entry (index and level)
    entry_triggers = {}  # maps i -> (swing_type, swing_level, prev_swing_level)

    if direction == 1:  # LONG
        for i in range(lookback + 1, n):
            if not np.isnan(swing_lows[i]):
                prev_swing = find_previous_swing(swing_lows, i)
                if not np.isnan(prev_swing) and swing_lows[i] > prev_swing:
                    if daily_color_prev[i] == 1 and daily_color[i] == 1:
                        # Entry on next bar
                        entry_triggers[i + 1] = ("long", swing_lows[i], prev_swing)

    else:  # SHORT
        for i in range(lookback + 1, n):
            if not np.isnan(swing_highs[i]):
                prev_swing = find_previous_swing(swing_highs, i)
                if not np.isnan(prev_swing) and swing_highs[i] < prev_swing:
                    if daily_color_prev[i] == -1 and daily_color[i] == -1:
                        # Entry on next bar
                        entry_triggers[i + 1] = ("short", swing_highs[i], prev_swing)

    # Simulate trades with 1:3 risk:reward
    trades = []
    pos = False
    entry_px = 0.0
    entry_i = 0
    sl_px = 0.0
    tp_px = 0.0

    for i in range(n):
        if pos:
            # Check stop loss
            if direction == 1:
                if low[i] <= sl_px:
                    # Spread deducted: 1 pip at entry + 1 pip at exit = 2 pips total
                    pips = (sl_px - entry_px) / PIP_SIZE - 2 * SPREAD_PIPS
                    trades.append({"pips": pips, "hold": i - entry_i, "exit": "sl"})
                    pos = False
                    continue
                # Check take profit
                if high[i] >= tp_px:
                    pips = (tp_px - entry_px) / PIP_SIZE - 2 * SPREAD_PIPS
                    trades.append({"pips": pips, "hold": i - entry_i, "exit": "tp"})
                    pos = False
                    continue
            else:  # SHORT
                if high[i] >= sl_px:
                    pips = (entry_px - sl_px) / PIP_SIZE - 2 * SPREAD_PIPS
                    trades.append({"pips": pips, "hold": i - entry_i, "exit": "sl"})
                    pos = False
                    continue
                # Check take profit
                if low[i] <= tp_px:
                    pips = (entry_px - tp_px) / PIP_SIZE - 2 * SPREAD_PIPS
                    trades.append({"pips": pips, "hold": i - entry_i, "exit": "tp"})
                    pos = False
                    continue

        # Check for entry
        if not pos and i in entry_triggers:
            trigger_type, current_swing, prev_swing = entry_triggers[i]
            pos = True
            # Entry price includes spread (we buy at ask)
            entry_px = close[i]
            entry_i = i

            if direction == 1:  # LONG
                if sl_at_prev_swing:
                    sl_px = prev_swing
                else:
                    sl_px = current_swing
                sl_distance = entry_px - sl_px
                tp_px = entry_px + sl_distance * 3

            else:  # SHORT
                if sl_at_prev_swing:
                    sl_px = prev_swing
                else:
                    sl_px = current_swing
                sl_distance = sl_px - entry_px
                tp_px = entry_px - sl_distance * 3

    # Compute stats
    if not trades:
        return {
            "n": 0,
            "pips": 0,
            "win_pct": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "sharpe": 0,
            "max_dd": 0,
            "avg_hold": 0,
        }

    pips = np.array([t["pips"] for t in trades])
    cum = np.cumsum(pips)
    days = max(1, (df.index[-1] - df.index[0]).days)
    wins = pips > 0

    sharpe = 0
    if len(trades) > 1 and pips.std() > 0:
        sharpe = (pips.mean() / pips.std() *
                  np.sqrt(len(trades) / (days / 365.25)))

    return {
        "n": len(trades),
        "pips": float(pips.sum()),
        "win_pct": float(wins.mean() * 100) if len(trades) > 0 else 0,
        "avg_win": float(pips[wins].mean()) if wins.any() else 0.0,
        "avg_loss": float(pips[~wins].mean()) if (~wins).any() else 0.0,
        "sharpe": float(sharpe),
        "max_dd": float((cum - np.maximum.accumulate(cum)).min()) if len(cum) > 0 else 0,
        "avg_hold": float(np.mean([t["hold"] for t in trades])) if len(trades) > 0 else 0,
        "tp_pct": float(np.mean([t["exit"] == "tp" for t in trades]) * 100),
        "sl_pct": float(np.mean([t["exit"] == "sl" for t in trades]) * 100),
    }


def main():
    print("=" * 100)
    print("STRATEGY 025 — Swing Point + Daily Color Filter (EURUSD 1H)")
    print("=" * 100)

    # Load data
    print("\nLoading EURUSD 5m data...")
    df = get_data(ticker="EURUSD", timeframe="5m",
                  start="2026-01-01", end="2026-05-16")
    print(f"  Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    # Add daily candle info
    print("Computing daily candle colors...")
    df = add_daily_candles(df)

    # Test configurations
    configs = [
        {"lookback": 10, "sl_at_prev": False, "direction": 1, "name": "LONG 10-bar lookback, SL at current"},
        {"lookback": 10, "sl_at_prev": True,  "direction": 1, "name": "LONG 10-bar lookback, SL at prev"},
        {"lookback": 20, "sl_at_prev": False, "direction": 1, "name": "LONG 20-bar lookback, SL at current"},
        {"lookback": 20, "sl_at_prev": True,  "direction": 1, "name": "LONG 20-bar lookback, SL at prev"},
        {"lookback": 10, "sl_at_prev": False, "direction": -1, "name": "SHORT 10-bar lookback, SL at current"},
        {"lookback": 10, "sl_at_prev": True,  "direction": -1, "name": "SHORT 10-bar lookback, SL at prev"},
        {"lookback": 20, "sl_at_prev": False, "direction": -1, "name": "SHORT 20-bar lookback, SL at current"},
        {"lookback": 20, "sl_at_prev": True,  "direction": -1, "name": "SHORT 20-bar lookback, SL at prev"},
    ]

    print("\n" + "=" * 100)
    print(f"{'Config':<50} {'Trades':>8} {'Pips':>10} {'Win%':>8} {'Sharpe':>8} {'AvgW':>8} {'AvgL':>8}")
    print("=" * 100)

    for config in configs:
        result = backtest_strategy(
            df,
            lookback=config["lookback"],
            sl_at_prev_swing=config["sl_at_prev"],
            direction=config["direction"],
        )

        if result["n"] > 0:
            print(f"{config['name']:<50} {result['n']:>8} {result['pips']:>+10.1f} "
                  f"{result['win_pct']:>7.1f}% {result['sharpe']:>+8.2f} "
                  f"{result['avg_win']:>+8.1f} {result['avg_loss']:>+8.1f}")
        else:
            print(f"{config['name']:<50} {'No trades':>8}")

    print("=" * 100)


if __name__ == "__main__":
    main()
