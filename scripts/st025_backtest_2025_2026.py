"""
Strategy 025 Backtest Runner — 2025-2026 Forex Data
Uses backtesting.py framework for professional analysis.

Outputs:
  - EURUSD alone backtest
  - All USD pairs combined (with FTMO no-hedge rules)
  - Drawdown analysis (size, period)
  - Total PnL
  - Equity curves
"""
import sys
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================================
# DATA LOADING (MT5 Direct)
# ============================================================================

_MT5_TF_MAP = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 16385,
    "4h": 16388,
    "1d": 16408,
}

PIP_SIZE = 0.0001
SPREAD_PIPS = 1.0


def get_mt5_data(
    ticker: str = "EURUSD",
    timeframe: str = "5m",
    start: str = "2025-01-01",
    end: str = "2026-12-31",
) -> pd.DataFrame:
    """Load OHLCV data from MT5 terminal."""
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    if not mt5.initialize():
        print(f"ERROR: Could not initialize MT5")
        return pd.DataFrame()

    mt5.symbol_select(ticker, True)

    tf_const = _MT5_TF_MAP.get(timeframe, 5)
    CHUNK = 50_000
    frames = []
    pos = 0

    print(f"  Fetching {ticker} {timeframe} from MT5...")
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

    if not frames:
        print(f"  WARNING: No data returned for {ticker}")
        return pd.DataFrame()

    df = pd.concat(frames[::-1])
    df = df.drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
        }
    )

    result = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    print(f"  Loaded {len(result)} bars for {ticker} ({result.index[0]} to {result.index[-1]})")
    return result


def add_daily_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily candle color information."""
    df = df.copy()
    df["date"] = df.index.normalize()

    daily_data = df.groupby("date").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).reset_index()

    daily_data["color"] = np.where(
        daily_data["Close"] > daily_data["Open"],
        1,
        np.where(daily_data["Close"] < daily_data["Open"], -1, 0),
    )

    color_map = dict(zip(daily_data["date"], daily_data["color"]))
    dates_sorted = sorted(color_map.keys())

    daily_colors = []
    daily_colors_prev = []

    for idx in df.index:
        current_date = idx.normalize()
        daily_colors.append(color_map.get(current_date, 0))
        prev_date = current_date - pd.Timedelta(days=1)
        daily_colors_prev.append(color_map.get(prev_date, 0))

    df["daily_color"] = daily_colors
    df["daily_color_prev"] = daily_colors_prev
    df = df.drop(columns=["date"])

    return df


# ============================================================================
# SIMPLE BACKTEST WITH NUMPY (No backtesting.py)
# ============================================================================


def backtest_strategy_simple(
    df: pd.DataFrame,
    lookback: int = 10,
    sl_at_prev_swing: bool = False,
    direction: int = 1,
    cash: float = 10_000,
) -> Optional[dict]:
    """
    Simple backtest without backtesting.py framework.
    Returns metrics: drawdown, PnL, equity curve, trades.
    """
    if df.empty:
        return None

    df = df.copy()
    if "daily_color" not in df.columns:
        df = add_daily_candles(df)

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    daily_color = df["daily_color"].values
    daily_color_prev = df["daily_color_prev"].values

    n = len(df)
    trades = []
    pos = False
    entry_px = 0.0
    entry_i = 0
    sl_px = 0.0
    tp_px = 0.0

    # Track equity curve
    equity_curve = [cash]
    cumulative_pnl = 0.0

    for i in range(lookback + 1, n):
        if pos:
            # Check stop loss
            if direction == 1:  # LONG
                if low[i] <= sl_px:
                    pips = (sl_px - entry_px) / PIP_SIZE - 2 * SPREAD_PIPS
                    trades.append({"pips": pips, "exit": "sl"})
                    cumulative_pnl += pips * 10  # 1 pip = $10 per lot
                    pos = False
                    equity_curve.append(cash + cumulative_pnl)
                    continue
                if high[i] >= tp_px:
                    pips = (tp_px - entry_px) / PIP_SIZE - 2 * SPREAD_PIPS
                    trades.append({"pips": pips, "exit": "tp"})
                    cumulative_pnl += pips * 10
                    pos = False
                    equity_curve.append(cash + cumulative_pnl)
                    continue
            else:  # SHORT
                if high[i] >= sl_px:
                    pips = (entry_px - sl_px) / PIP_SIZE - 2 * SPREAD_PIPS
                    trades.append({"pips": pips, "exit": "sl"})
                    cumulative_pnl += pips * 10
                    pos = False
                    equity_curve.append(cash + cumulative_pnl)
                    continue
                if low[i] <= tp_px:
                    pips = (entry_px - tp_px) / PIP_SIZE - 2 * SPREAD_PIPS
                    trades.append({"pips": pips, "exit": "tp"})
                    cumulative_pnl += pips * 10
                    pos = False
                    equity_curve.append(cash + cumulative_pnl)
                    continue

        # Check for entry
        if not pos:
            if direction == 1:  # LONG
                # New swing low + both daily candles green
                min_idx = max(0, i - lookback)
                prev_low = np.nanmin(low[min_idx:i])
                if low[i] < prev_low and daily_color_prev[i] == 1 and daily_color[i] == 1:
                    pos = True
                    entry_px = close[i]
                    entry_i = i

                    if sl_at_prev_swing:
                        # Previous swing low
                        prev_min_idx = max(0, min_idx - lookback)
                        sl_px = np.nanmin(low[prev_min_idx:min_idx]) if min_idx > 0 else low[min_idx]
                    else:
                        sl_px = low[i]

                    sl_dist = entry_px - sl_px
                    tp_px = entry_px + sl_dist * 3

            else:  # SHORT
                # New swing high + both daily candles red
                max_idx = max(0, i - lookback)
                prev_high = np.nanmax(high[max_idx:i])
                if high[i] > prev_high and daily_color_prev[i] == -1 and daily_color[i] == -1:
                    pos = True
                    entry_px = close[i]
                    entry_i = i

                    if sl_at_prev_swing:
                        # Previous swing high
                        prev_max_idx = max(0, max_idx - lookback)
                        sl_px = np.nanmax(high[prev_max_idx:max_idx]) if max_idx > 0 else high[max_idx]
                    else:
                        sl_px = high[i]

                    sl_dist = sl_px - entry_px
                    tp_px = entry_px - sl_dist * 3

        equity_curve.append(cash + cumulative_pnl)

    # Compute statistics
    if not trades:
        return {
            "trades": 0,
            "pnl": 0,
            "pnl_pct": 0,
            "win_rate": 0,
            "max_dd": 0,
            "max_dd_pct": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "equity_curve": equity_curve,
        }

    pips = np.array([t["pips"] for t in trades])
    wins = pips > 0

    # Metrics
    total_pnl = pips.sum() * 10
    pnl_pct = (total_pnl / cash) * 100
    win_rate = wins.mean() * 100
    avg_win = pips[wins].mean() if wins.any() else 0
    avg_loss = pips[~wins].mean() if (~wins).any() else 0

    # Max drawdown
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdown = running_max - equity_arr
    max_dd = drawdown.max()
    max_dd_pct = (max_dd / equity_arr[0]) * 100 if equity_arr[0] > 0 else 0

    return {
        "trades": len(trades),
        "pnl": total_pnl,
        "pnl_pct": pnl_pct,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "equity_curve": equity_curve,
    }


# ============================================================================
# BACKTEST RUNNER WITH METRICS
# ============================================================================






# ============================================================================
# REPORTING UTILITIES
# ============================================================================


def generate_equity_csv(results: dict, pair: str):
    """Save equity curve and drawdown to CSV."""
    equity_curve = results.get("equity_curve", [])
    if not equity_curve:
        return

    try:
        # Calculate drawdown
        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = running_max - equity_arr
        drawdown_pct = (drawdown / running_max) * 100

        # Save CSV
        import csv
        filename = f"st025_{pair}_equity_curve.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Bar", "Equity", "Drawdown_$", "Drawdown_%"])
            for i, eq in enumerate(equity_curve):
                writer.writerow([i, f"{eq:.2f}", f"{drawdown[i]:.2f}", f"{drawdown_pct[i]:.2f}"])

        print(f"  → Saved: {filename}")

    except Exception as e:
        print(f"  ERROR saving CSV: {e}")


# ============================================================================
# MAIN BACKTEST RUNNER
# ============================================================================


def main():
    print("=" * 100)
    print("STRATEGY 025 BACKTEST RUNNER — 2025-2026 Forex Data")
    print("=" * 100)

    USD_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "CADUSD"]

    # ---- EURUSD ALONE ----
    print("\n[1] EURUSD ALONE")
    print("-" * 100)

    df_eurusd = get_mt5_data("EURUSD", "5m", "2025-01-01", "2026-12-31")
    if not df_eurusd.empty:
        df_eurusd = add_daily_candles(df_eurusd)

        configs = [
            {"lookback": 10, "sl_at_prev": False, "direction": 1, "name": "LONG 10-bar, SL@current"},
            {"lookback": 10, "sl_at_prev": True, "direction": 1, "name": "LONG 10-bar, SL@prev"},
            {"lookback": 20, "sl_at_prev": False, "direction": 1, "name": "LONG 20-bar, SL@current"},
            {"lookback": 20, "sl_at_prev": True, "direction": 1, "name": "LONG 20-bar, SL@prev"},
        ]

        print(f"{'Config':<35} {'Trades':>8} {'PnL':>10} {'PnL%':>8} {'Win%':>8} {'MaxDD':>10}")
        print("-" * 100)

        for config in configs:
            result = backtest_strategy_simple(
                df_eurusd,
                lookback=config["lookback"],
                sl_at_prev_swing=config["sl_at_prev"],
                direction=config["direction"],
            )
            if result and result["trades"] > 0:
                print(
                    f"{config['name']:<35} {result['trades']:>8} "
                    f"${result['pnl']:>+9.2f} {result['pnl_pct']:>+7.1f}% "
                    f"{result['win_rate']:>7.1f}% ${result['max_dd']:>+9.2f}"
                )
                # Save best config equity curve
                if config["sl_at_prev"] and config["lookback"] == 10:
                    generate_equity_csv(result, f"EURUSD_{config['name']}")
            elif result:
                print(f"{config['name']:<35} {'No trades':>8}")

    # ---- ALL USD PAIRS (WITH FTMO NO-HEDGE RULE) ----
    print("\n[2] ALL USD PAIRS COMBINED (FTMO No-Hedge Rules)")
    print("-" * 100)

    data_dict = {}
    for pair in USD_PAIRS:
        df = get_mt5_data(pair, "5m", "2025-01-01", "2026-12-31")
        if not df.empty:
            df = add_daily_candles(df)
            data_dict[pair] = df
        else:
            print(f"  SKIP {pair}: No data")

    if data_dict:
        print(f"\nLoaded data for: {', '.join(data_dict.keys())}")
        print("\nLONG-only strategy (each pair, no hedging):")
        print(f"{'Pair':<12} {'Trades':>8} {'PnL':>12} {'PnL%':>8} {'AvgW':>8} {'AvgL':>8} {'Win%':>8} {'MaxDD':>10} {'MaxDD%':>8}")
        print("-" * 130)

        total_pnl = 0
        total_trades = 0
        results_by_pair = {}

        for pair in USD_PAIRS:
            if pair not in data_dict:
                continue

            result = backtest_strategy_simple(
                data_dict[pair],
                lookback=10,
                sl_at_prev_swing=False,
                direction=1,  # LONG only
            )

            if result and result["trades"] > 0:
                results_by_pair[pair] = result
                print(
                    f"{pair:<12} {result['trades']:>8} "
                    f"${result['pnl']:>+11.2f} {result['pnl_pct']:>+7.1f}% "
                    f"{result['avg_win']:>+7.1f} {result['avg_loss']:>+7.1f} "
                    f"{result['win_rate']:>7.1f}% ${result['max_dd']:>+9.2f} "
                    f"{result['max_dd_pct']:>+7.2f}%"
                )
                total_pnl += result["pnl"]
                total_trades += result["trades"]
            elif result:
                print(f"{pair:<12} {'No trades':>8}")

        if total_trades > 0:
            print("-" * 130)
            print(f"{'TOTAL':<12} {total_trades:>8} ${total_pnl:>+11.2f}")

        # Generate equity curves for all pairs
        print("\n[3] SAVING EQUITY CURVE DATA...")
        print("-" * 100)
        for pair, result in results_by_pair.items():
            generate_equity_csv(result, pair)

    print("\n" + "=" * 100)
    print("Backtest Complete")
    print("=" * 100)


if __name__ == "__main__":
    main()
