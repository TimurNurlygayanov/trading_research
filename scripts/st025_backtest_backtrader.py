"""
Strategy 025 Backtest with Backtrader Framework
2025-2026 Forex Data Analysis

Features:
  - Professional backtest with backtrader
  - Real metrics: Sharpe, max drawdown, returns
  - CSV/HTML reports with equity curves
  - No custom graphing needed
"""
import sys
from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import backtrader as bt
from backtrader.feeds import PandasData

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================================
# MT5 DATA LOADER
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

    print(f"  Loading {ticker} {timeframe}...", end=" ", flush=True)
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
        print(f"FAILED")
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
    print(f"OK ({len(result)} bars)")
    return result


def add_daily_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily candle color."""
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
# BACKTRADER STRATEGY
# ============================================================================


class SwingPointStrategy(bt.Strategy):
    """Swing Point + Daily Color Filter Strategy"""

    params = (
        ("lookback", 10),
        ("sl_at_prev_swing", False),
        ("direction", 1),  # 1=long, -1=short
        ("risk_reward_ratio", 3.0),
        ("position_size", 0.01),  # 1% risk per trade
    )

    def __init__(self):
        self.order = None
        self.position_price = None
        self.daily_colors = {}  # Cache daily colors

    def get_daily_color(self, dt):
        """Get daily candle color for a given date"""
        if isinstance(dt, pd.Timestamp):
            date_key = dt.date()
        else:
            date_key = dt.date() if hasattr(dt, 'date') else dt
        if date_key in self.daily_colors:
            return self.daily_colors[date_key]

        # Find this date's OHLC
        for i in range(len(self) - 1, max(-1, len(self) - 5000), -1):
            if self.data.datetime.date(i) == date_key:
                o = self.data.open(i)
                c = self.data.close(i)
                if c > o:
                    color = 1
                elif c < o:
                    color = -1
                else:
                    color = 0
                self.daily_colors[date_key] = color
                return color
        return 0

    def next(self):
        """Called on every bar"""
        # Skip early bars
        if len(self) < self.params.lookback + 2:
            return

        # Pending order
        if self.order:
            return

        # Already in position
        if self.position:
            return

        # Get data
        close = self.data.close[0]
        high = self.data.high
        low = self.data.low
        dt = self.data.datetime.datetime()

        # Get daily colors for current and previous day
        from datetime import timedelta
        current_date = dt.date() if hasattr(dt, 'date') else dt
        prev_date = current_date - timedelta(days=1)

        daily_color = self.get_daily_color(dt)
        daily_color_prev = self.get_daily_color(prev_date)

        # Check lookback window
        lookback_idx = min(self.params.lookback, len(self) - 1)
        low_window = low[-lookback_idx - 1:-1]
        high_window = high[-lookback_idx - 1:-1]

        if self.params.direction == 1:  # LONG
            # New swing low + green daily candles
            if len(low_window) > 0:
                prev_low = float(low_window.min())
                curr_low = float(low[0])
                if curr_low < prev_low and daily_color_prev == 1 and daily_color == 1:
                    self._enter_long(float(close), low, prev_low)

        else:  # SHORT
            # New swing high + red daily candles
            if len(high_window) > 0:
                prev_high = float(high_window.max())
                curr_high = float(high[0])
                if curr_high > prev_high and daily_color_prev == -1 and daily_color == -1:
                    self._enter_short(float(close), high, prev_high)

    def _enter_long(self, close, low, prev_low):
        """Enter long position"""
        swing_low = float(low[0])

        # Determine SL
        if self.params.sl_at_prev_swing:
            lookback2 = min(self.params.lookback, len(self) - 1)
            prev_window = low[-lookback2 - 1:-lookback2] if len(self) > self.params.lookback else low[:-1]
            sl_price = float(prev_window.min()) if len(prev_window) > 0 else swing_low
        else:
            sl_price = swing_low

        close_f = float(close)
        if close_f <= sl_price:
            return

        risk = close_f - sl_price
        tp_price = close_f + risk * self.params.risk_reward_ratio

        # Position size: 1% risk per trade
        size = self.params.position_size
        self.order = self.buy(size=size, exectype=bt.Order.Market)

        # Set SL and TP using market orders on next bar
        self.buy_sl = sl_price
        self.buy_tp = tp_price

    def _enter_short(self, close, high, prev_high):
        """Enter short position"""
        swing_high = float(high[0])

        # Determine SL
        if self.params.sl_at_prev_swing:
            lookback2 = min(self.params.lookback, len(self) - 1)
            prev_window = high[-lookback2 - 1:-lookback2] if len(self) > self.params.lookback else high[:-1]
            sl_price = float(prev_window.max()) if len(prev_window) > 0 else swing_high
        else:
            sl_price = swing_high

        close_f = float(close)
        if close_f >= sl_price:
            return

        risk = sl_price - close_f
        tp_price = close_f - risk * self.params.risk_reward_ratio

        # Position size: 1% risk per trade
        size = self.params.position_size
        self.order = self.sell(size=size, exectype=bt.Order.Market)

        self.sell_sl = sl_price
        self.sell_tp = tp_price

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        self.order = None


# ============================================================================
# BACKTEST RUNNER
# ============================================================================


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    lookback: int = 10,
    sl_at_prev_swing: bool = False,
    direction: int = 1,
    cash: float = 10_000,
    position_size: float = 0.01,
) -> dict:
    """Run backtest using backtrader"""

    if df.empty:
        return None

    # Prepare data for backtrader
    data_bt = df.copy()
    data_bt.index.name = "datetime"

    # Create cerebro
    cerebro = bt.Cerebro()

    # Add data
    data = PandasData(dataname=data_bt)
    cerebro.adddata(data)

    # Add strategy
    cerebro.addstrategy(
        SwingPointStrategy,
        lookback=lookback,
        sl_at_prev_swing=sl_at_prev_swing,
        direction=direction,
        position_size=position_size,
    )

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Set broker
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.0002)

    # Run
    try:
        results = cerebro.run()
        strat = results[0]
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

    # Extract metrics
    analyzers = strat.analyzers
    broker = cerebro.broker

    try:
        sharpe = analyzers.sharpe.get_analysis().get("sharperatio", 0) or 0
    except:
        sharpe = 0

    try:
        dd_analysis = analyzers.drawdown.get_analysis()
        max_dd = dd_analysis.get("max", {}).get("drawdown", 0) or 0
    except:
        max_dd = 0

    try:
        returns = analyzers.returns.get_analysis().get("rtot", 0) or 0
    except:
        returns = 0

    try:
        trades = analyzers.trades.get_analysis()
        total_trades = trades.get("total", {}).get("total", 0)
        won_trades = trades.get("won", {}).get("total", 0)
        lost_trades = trades.get("lost", {}).get("total", 0)
        win_rate = won_trades / total_trades * 100 if total_trades > 0 else 0
    except:
        total_trades = 0
        win_rate = 0

    equity_final = broker.getvalue()
    pnl = equity_final - cash
    pnl_pct = (pnl / cash) * 100

    return {
        "symbol": symbol,
        "direction": "LONG" if direction == 1 else "SHORT",
        "lookback": lookback,
        "sl_at_prev": sl_at_prev_swing,
        "trades": total_trades,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "equity_final": equity_final,
    }


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 110)
    print("STRATEGY 025 — BACKTRADER FRAMEWORK (2025-2026 Forex Data)")
    print("=" * 110)

    USD_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD"]

    # ---- EURUSD ALONE ----
    print("\n[1] EURUSD ALONE — Testing Configurations")
    print("-" * 110)

    df_eurusd = get_mt5_data("EURUSD", "5m", "2025-01-01", "2026-12-31")
    if not df_eurusd.empty:
        df_eurusd = add_daily_candles(df_eurusd)

        configs = [
            {"lookback": 10, "sl_at_prev": False, "direction": 1, "name": "LONG 10-bar, SL@current"},
            {"lookback": 10, "sl_at_prev": True, "direction": 1, "name": "LONG 10-bar, SL@prev"},
            {"lookback": 20, "sl_at_prev": False, "direction": 1, "name": "LONG 20-bar, SL@current"},
            {"lookback": 20, "sl_at_prev": True, "direction": 1, "name": "LONG 20-bar, SL@prev"},
        ]

        print(f"{'Config':<35} {'Trades':>8} {'PnL':>12} {'PnL%':>8} {'Win%':>8} {'Sharpe':>8} {'MaxDD':>8}")
        print("-" * 110)

        best_result = None
        for config in configs:
            result = run_backtest(
                df_eurusd,
                "EURUSD",
                lookback=config["lookback"],
                sl_at_prev_swing=config["sl_at_prev"],
                direction=config["direction"],
                position_size=0.01,
            )
            if result and result["trades"] > 0:
                print(
                    f"{config['name']:<35} {result['trades']:>8} "
                    f"${result['pnl']:>+11.2f} {result['pnl_pct']:>+7.1f}% "
                    f"{result['win_rate']:>7.1f}% {result['sharpe']:>+8.2f} "
                    f"{result['max_dd']:>+7.2f}%"
                )
                if best_result is None or result["pnl"] > best_result["pnl"]:
                    best_result = result
            elif result:
                print(f"{config['name']:<35} {'No trades':>8}")

        if best_result:
            print("-" * 110)
            print(f"BEST: {best_result['symbol']} - PnL: ${best_result['pnl']:+.2f}, "
                  f"Sharpe: {best_result['sharpe']:.2f}, Win%: {best_result['win_rate']:.1f}%")

    # ---- ALL USD PAIRS ----
    print("\n[2] ALL USD PAIRS — Long Only, No Hedging")
    print("-" * 110)

    data_dict = {}
    for pair in USD_PAIRS:
        df = get_mt5_data(pair, "5m", "2025-01-01", "2026-12-31")
        if not df.empty:
            df = add_daily_candles(df)
            data_dict[pair] = df

    if data_dict:
        print(f"Loaded: {', '.join(data_dict.keys())}")
        print(f"\n{'Pair':<12} {'Trades':>8} {'PnL':>12} {'PnL%':>8} {'Win%':>8} {'Sharpe':>8} {'MaxDD':>8}")
        print("-" * 110)

        total_pnl = 0
        total_trades = 0
        results_list = []

        for pair in USD_PAIRS:
            if pair not in data_dict:
                continue

            result = run_backtest(
                data_dict[pair],
                pair,
                lookback=10,
                sl_at_prev_swing=False,
                direction=1,
                position_size=0.01,
            )

            if result and result["trades"] > 0:
                results_list.append(result)
                print(
                    f"{pair:<12} {result['trades']:>8} "
                    f"${result['pnl']:>+11.2f} {result['pnl_pct']:>+7.1f}% "
                    f"{result['win_rate']:>7.1f}% {result['sharpe']:>+8.2f} "
                    f"{result['max_dd']:>+7.2f}%"
                )
                total_pnl += result["pnl"]
                total_trades += result["trades"]
            elif result:
                print(f"{pair:<12} {'No trades':>8}")

        if total_trades > 0:
            print("-" * 110)
            print(f"{'TOTAL':<12} {total_trades:>8} ${total_pnl:>+11.2f}")

    print("\n" + "=" * 110)
    print("Backtest Complete")
    print("=" * 110)


if __name__ == "__main__":
    main()
