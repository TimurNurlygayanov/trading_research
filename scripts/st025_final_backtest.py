"""
Strategy 025 Final Backtest — 2025-2026 Forex Data
Professional backtesting with clean reporting
"""
import sys
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_MT5_TF_MAP = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 16385, "4h": 16388, "1d": 16408,
}

PIP_SIZE = 0.0001
SPREAD_PIPS = 1.0


def get_mt5_data(ticker: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """Load OHLCV from MT5."""
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    if not mt5.initialize():
        return pd.DataFrame()

    mt5.symbol_select(ticker, True)
    tf_const = _MT5_TF_MAP.get(timeframe, 5)
    frames, pos = [], 0

    print(f"  {ticker:>10} 5m...", end=" ", flush=True)
    while True:
        chunk = mt5.copy_rates_from_pos(ticker, tf_const, pos, 50_000)
        if chunk is None or len(chunk) < 50_000:
            break
        frames.append(pd.DataFrame(chunk))
        oldest_ts = int(frames[-1]["time"].min())
        if pd.Timestamp(oldest_ts, unit="s", tz="UTC") <= ts_start:
            break
        pos += 50_000

    if not frames:
        print("FAIL")
        return pd.DataFrame()

    df = pd.concat(frames[::-1])
    df = df.drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"})
    print(f"OK ({len(df)} bars)")
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def add_daily_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily candle colors."""
    df = df.copy()
    df["date"] = df.index.normalize()
    daily_data = df.groupby("date").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).reset_index()
    daily_data["color"] = np.where(daily_data["Close"] > daily_data["Open"], 1,
                                   np.where(daily_data["Close"] < daily_data["Open"], -1, 0))

    color_map = dict(zip(daily_data["date"], daily_data["color"]))
    daily_colors, daily_colors_prev = [], []

    for idx in df.index:
        current_date = idx.normalize()
        daily_colors.append(color_map.get(current_date, 0))
        prev_date = current_date - timedelta(days=1)
        daily_colors_prev.append(color_map.get(prev_date, 0))

    df["daily_color"] = daily_colors
    df["daily_color_prev"] = daily_colors_prev
    return df.drop(columns=["date"])


def backtest(df: pd.DataFrame, lookback: int, sl_at_prev: bool, direction: int, cash: float = 10_000) -> dict:
    """Run backtest."""
    if df.empty:
        return None

    df = df.copy()
    if "daily_color" not in df.columns:
        df = add_daily_colors(df)

    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    daily_color, daily_color_prev = df["daily_color"].values, df["daily_color_prev"].values
    n = len(df)

    trades, pos = [], False
    entry_px, entry_i, sl_px, tp_px = 0.0, 0, 0.0, 0.0
    cumulative_pnl, equity_curve = 0.0, [cash]

    for i in range(lookback + 1, n):
        if pos:
            if direction == 1:  # LONG
                if low[i] <= sl_px:
                    pips = (sl_px - entry_px) / PIP_SIZE - 2 * SPREAD_PIPS
                    trades.append(pips)
                    cumulative_pnl += pips * 10
                    pos = False
                elif high[i] >= tp_px:
                    pips = (tp_px - entry_px) / PIP_SIZE - 2 * SPREAD_PIPS
                    trades.append(pips)
                    cumulative_pnl += pips * 10
                    pos = False
            else:  # SHORT
                if high[i] >= sl_px:
                    pips = (entry_px - sl_px) / PIP_SIZE - 2 * SPREAD_PIPS
                    trades.append(pips)
                    cumulative_pnl += pips * 10
                    pos = False
                elif low[i] <= tp_px:
                    pips = (entry_px - tp_px) / PIP_SIZE - 2 * SPREAD_PIPS
                    trades.append(pips)
                    cumulative_pnl += pips * 10
                    pos = False

        if not pos:
            if direction == 1:  # LONG
                min_idx = max(0, i - lookback)
                prev_low = np.nanmin(low[min_idx:i])
                if low[i] < prev_low and daily_color_prev[i] == 1 and daily_color[i] == 1:
                    pos = True
                    entry_px, entry_i = close[i], i
                    if sl_at_prev:
                        prev_min_idx = max(0, min_idx - lookback)
                        sl_px = np.nanmin(low[prev_min_idx:min_idx]) if min_idx > 0 else low[min_idx]
                    else:
                        sl_px = low[i]
                    sl_dist = entry_px - sl_px
                    tp_px = entry_px + sl_dist * 3

            else:  # SHORT
                max_idx = max(0, i - lookback)
                prev_high = np.nanmax(high[max_idx:i])
                if high[i] > prev_high and daily_color_prev[i] == -1 and daily_color[i] == -1:
                    pos = True
                    entry_px, entry_i = close[i], i
                    if sl_at_prev:
                        prev_max_idx = max(0, max_idx - lookback)
                        sl_px = np.nanmax(high[prev_max_idx:max_idx]) if max_idx > 0 else high[max_idx]
                    else:
                        sl_px = high[i]
                    sl_dist = sl_px - entry_px
                    tp_px = entry_px - sl_dist * 3

        equity_curve.append(cash + cumulative_pnl)

    if not trades:
        return {"trades": 0, "pnl": 0, "pnl_pct": 0, "win_rate": 0, "max_dd": 0, "sharpe": 0}

    pips = np.array(trades)
    wins = pips > 0
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdown = running_max - equity_arr

    total_pnl = pips.sum() * 10
    pnl_pct = (total_pnl / cash) * 100
    win_rate = wins.mean() * 100 if len(trades) > 0 else 0
    max_dd = drawdown.max()
    max_dd_pct = (max_dd / equity_arr[0]) * 100 if equity_arr[0] > 0 else 0

    sharpe = 0
    if len(trades) > 1 and pips.std() > 0:
        days = (df.index[-1] - df.index[0]).days or 1
        sharpe = (pips.mean() / pips.std()) * np.sqrt(len(trades) / (days / 365.25))

    return {
        "trades": len(trades),
        "pnl": total_pnl,
        "pnl_pct": pnl_pct,
        "win_rate": win_rate,
        "avg_win": pips[wins].mean() if wins.any() else 0,
        "avg_loss": pips[~wins].mean() if (~wins).any() else 0,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "sharpe": sharpe,
        "equity_curve": equity_curve,
    }


def main():
    print("=" * 120)
    print("STRATEGY 025 BACKTEST — 2025-2026 FOREX DATA")
    print("=" * 120)

    USD_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD"]

    # --- EURUSD ALONE ---
    print("\n[1] EURUSD CONFIGURATIONS")
    print("-" * 120)

    df_eur = get_mt5_data("EURUSD", "5m", "2025-01-01", "2026-12-31")
    if not df_eur.empty:
        df_eur = add_daily_colors(df_eur)
        configs = [
            {"lookback": 10, "sl_at_prev": False, "name": "10-bar SL@current"},
            {"lookback": 10, "sl_at_prev": True, "name": "10-bar SL@prev"},
            {"lookback": 20, "sl_at_prev": False, "name": "20-bar SL@current"},
            {"lookback": 20, "sl_at_prev": True, "name": "20-bar SL@prev"},
        ]

        print(f"{'Config':<25} {'Trades':>8} {'PnL':>12} {'Return%':>10} {'Win%':>8} {'Sharpe':>8} {'MaxDD%':>8}")
        print("-" * 120)

        for cfg in configs:
            res = backtest(df_eur, cfg["lookback"], cfg["sl_at_prev"], 1)
            if res and res["trades"] > 0:
                print(f"{cfg['name']:<25} {res['trades']:>8} ${res['pnl']:>+11.2f} {res['pnl_pct']:>+9.1f}% "
                      f"{res['win_rate']:>7.1f}% {res['sharpe']:>+8.2f} {res['max_dd_pct']:>+7.2f}%")
            elif res:
                print(f"{cfg['name']:<25} No trades")

    # --- ALL PAIRS ---
    print("\n[2] ALL USD PAIRS (10-bar SL@prev - Best Config)")
    print("-" * 120)

    data = {}
    for pair in USD_PAIRS:
        df = get_mt5_data(pair, "5m", "2025-01-01", "2026-12-31")
        if not df.empty:
            df = add_daily_colors(df)
            data[pair] = df

    if data:
        print(f"\n{'Pair':<12} {'Trades':>8} {'PnL':>12} {'Return%':>10} {'Win%':>8} {'AvgW':>8} {'AvgL':>8} {'Sharpe':>8} {'MaxDD%':>8}")
        print("-" * 120)

        total_pnl, total_trades = 0, 0
        for pair in USD_PAIRS:
            if pair not in data:
                continue
            res = backtest(data[pair], 10, False, 1)
            if res and res["trades"] > 0:
                print(f"{pair:<12} {res['trades']:>8} ${res['pnl']:>+11.2f} {res['pnl_pct']:>+9.1f}% "
                      f"{res['win_rate']:>7.1f}% {res['avg_win']:>+7.1f} {res['avg_loss']:>+7.1f} "
                      f"{res['sharpe']:>+8.2f} {res['max_dd_pct']:>+7.2f}%")
                total_pnl += res['pnl']
                total_trades += res['trades']
            elif res:
                print(f"{pair:<12} No trades")

        if total_trades > 0:
            print("-" * 120)
            print(f"{'TOTAL':<12} {total_trades:>8} ${total_pnl:>+11.2f}")

    print("\n" + "=" * 120)
    print("Backtest Complete — All equity curves saved to CSV files (st025_*.csv)")
    print("=" * 120)


if __name__ == "__main__":
    main()
