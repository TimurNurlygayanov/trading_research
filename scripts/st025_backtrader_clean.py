"""
Strategy 025 with Backtrader - Clean Implementation
2025-2026 Forex Data Backtesting
"""
import sys
from datetime import timedelta

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import backtrader as bt
from backtrader.feeds import PandasData


class CustomPandasData(PandasData):
    """Custom feed that includes daily_color columns."""
    lines = ('daily_color', 'daily_color_prev')

    params = (
        ('dtformat', '%Y-%m-%d'),
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('daily_color', 'daily_color'),
        ('daily_color_prev', 'daily_color_prev'),
    )

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_MT5_TF_MAP = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 16385, "4h": 16388, "1d": 16408,
}


def get_mt5_data(ticker: str, timeframe: str = "5m", start: str = "2026-01-01", end: str = "2026-05-16") -> pd.DataFrame:
    """Load data from MT5."""
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    if not mt5.initialize():
        print(f"ERROR: MT5 init failed")
        return pd.DataFrame()

    mt5.symbol_select(ticker, True)
    tf_const = _MT5_TF_MAP.get(timeframe, 5)
    frames, pos = [], 0

    print(f"  {ticker:>10}...", end=" ", flush=True)
    while True:
        chunk = mt5.copy_rates_from_pos(ticker, tf_const, pos, 50_000)
        if chunk is None or len(chunk) < 50_000:
            break
        frames.append(pd.DataFrame(chunk))
        if pd.Timestamp(int(frames[-1]["time"].min()), unit="s", tz="UTC") <= ts_start:
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


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily color indicator."""
    df = df.copy()
    df["date"] = df.index.normalize()

    daily = df.groupby("date").agg({"Open": "first", "Close": "last"}).reset_index()
    daily["color"] = np.where(daily["Close"] > daily["Open"], 1,
                              np.where(daily["Close"] < daily["Open"], -1, 0))

    color_map = dict(zip(daily["date"], daily["color"]))

    df["daily_color"] = df["date"].map(color_map)
    df["daily_color_prev"] = df["date"].map(lambda d: color_map.get(d - timedelta(days=1), 0))

    return df.drop(columns=["date"])


# ============================================================================
# BACKTRADER STRATEGY
# ============================================================================

class SwingPointStrategyBT(bt.Strategy):
    """
    Swing Point + Daily Color Filter Strategy.

    Mirrors strategy025.py logic exactly:
    - Pre-detect ALL swing points (LONG-only or SHORT-only based on direction)
    - On signal bar, schedule entry for NEXT bar at its close
    - Manual SL/TP exit checks each bar
    """
    params = (
        ("lookback", 10),
        ("sl_at_prev", False),
        ("direction", 1),  # 1=long, -1=short
        ("risk_reward_ratio", 3.0),
        ("position_size", 10000),
    )

    def __init__(self):
        # State for current open trade
        self.entry_bar = None
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None

        # Track swing lows / highs detected so far (bar_idx, price)
        self.swing_lows = []
        self.swing_highs = []

        # Pending entry trigger: maps bar_idx -> ("long"/"short", swing, prev_swing)
        self.entry_triggers = {}

    def next(self):
        lookback = self.params.lookback

        if len(self) < lookback + 2:
            return

        bar_idx = len(self) - 1

        # Bracket orders handle exits automatically intra-bar at exact SL/TP price.
        # No manual exit check needed.

        # ---- DETECT SWING AT PREVIOUS BAR (bar_idx - 1) ----
        # Note: a "swing low at bar i" means low[i] < min(low[i-lookback:i])
        # Detect this AT bar i (current), then trigger entry at i+1.
        # Here we evaluate whether the PREVIOUS bar was a swing — so entry fires on THIS bar.
        prev_bar_low = float(self.data.low[-1])
        prev_bar_high = float(self.data.high[-1])

        # Lookback window for the previous bar: bars [bar_idx - lookback - 1 .. bar_idx - 2]
        # That's data.low[-2] to data.low[-(lookback + 1)]
        prev_window_lows = [float(self.data.low[-i]) for i in range(2, lookback + 2)]
        prev_window_highs = [float(self.data.high[-i]) for i in range(2, lookback + 2)]

        is_prev_swing_low = prev_bar_low < min(prev_window_lows)
        is_prev_swing_high = prev_bar_high > max(prev_window_highs)

        prev_bar_idx = bar_idx - 1

        # Daily colors for the SIGNAL bar (prev bar in our terms)
        daily_color_signal = int(self.data.daily_color[-1])
        daily_color_prev_signal = int(self.data.daily_color_prev[-1])

        # ---- ALWAYS TRACK SWINGS & TRIGGERS (regardless of position state) ----
        if is_prev_swing_low:
            prev_swing_low_price = self.swing_lows[-1][1] if self.swing_lows else None
            self.swing_lows.append((prev_bar_idx, prev_bar_low))

            if (self.params.direction == 1 and
                prev_swing_low_price is not None and
                prev_bar_low > prev_swing_low_price and
                daily_color_prev_signal == 1 and
                daily_color_signal == 1):
                self.entry_triggers[bar_idx] = ("long", prev_bar_low, prev_swing_low_price)

        if is_prev_swing_high:
            prev_swing_high_price = self.swing_highs[-1][1] if self.swing_highs else None
            self.swing_highs.append((prev_bar_idx, prev_bar_high))

            if (self.params.direction == -1 and
                prev_swing_high_price is not None and
                prev_bar_high < prev_swing_high_price and
                daily_color_prev_signal == -1 and
                daily_color_signal == -1):
                self.entry_triggers[bar_idx] = ("short", prev_bar_high, prev_swing_high_price)

        # ---- ENTRY ----
        # If we're in a position (bracket order active), skip new entries.
        if self.position:
            return

        if bar_idx in self.entry_triggers:
            trigger_type, current_swing, prev_swing = self.entry_triggers[bar_idx]
            entry_px = float(self.data.close[0])

            if trigger_type == "long":
                sl_price = prev_swing if self.params.sl_at_prev else current_swing
                if entry_px > sl_price:
                    risk = entry_px - sl_price
                    tp_price = entry_px + risk * self.params.risk_reward_ratio
                    self.buy_bracket(
                        size=self.params.position_size,
                        exectype=bt.Order.Market,
                        stopprice=sl_price,
                        limitprice=tp_price,
                    )
                    self.entry_bar = bar_idx
                    self.entry_price = entry_px
                    self.sl_price = sl_price
                    self.tp_price = tp_price

            elif trigger_type == "short":
                sl_price = prev_swing if self.params.sl_at_prev else current_swing
                if entry_px < sl_price:
                    risk = sl_price - entry_px
                    tp_price = entry_px - risk * self.params.risk_reward_ratio
                    self.sell_bracket(
                        size=self.params.position_size,
                        exectype=bt.Order.Market,
                        stopprice=sl_price,
                        limitprice=tp_price,
                    )
                    self.entry_bar = bar_idx
                    self.entry_price = entry_px
                    self.sl_price = sl_price
                    self.tp_price = tp_price

    def _reset_state(self):
        self.entry_bar = None
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None



# ============================================================================
# BACKTEST RUNNER
# ============================================================================

def run_backtest(df: pd.DataFrame, symbol: str, lookback: int = 10,
                 sl_at_prev: bool = False, direction: int = 1,
                 cash: float = 10_000) -> dict:
    """Run backtest with backtrader."""

    if df.empty:
        return None

    # Prepare data
    df_prep = prepare_data(df)

    # Create cerebro
    cerebro = bt.Cerebro()

    # Add data with custom feed
    data_feed = CustomPandasData(dataname=df_prep)
    cerebro.adddata(data_feed)

    # Add strategy (position_size in units: 10000 = 1 mini lot)
    cerebro.addstrategy(
        SwingPointStrategyBT,
        lookback=lookback,
        sl_at_prev=sl_at_prev,
        direction=direction,
        position_size=10000,
    )

    # Add analyzers for metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                        timeframe=bt.TimeFrame.Days, riskfreerate=0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return',
                        timeframe=bt.TimeFrame.Days)

    # Broker settings - forex with leverage
    # Realistic execution: market orders fill at next bar's open (no cheating).
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.0002, leverage=100, margin=False)

    # Run
    try:
        results = cerebro.run()
        strat = results[0]
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

    # Extract metrics
    broker_value = cerebro.broker.getvalue()
    pnl = broker_value - cash
    pnl_pct = (pnl / cash) * 100

    # Get analyzers
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    trades_analysis = strat.analyzers.trades.get_analysis()

    try:
        sharpe = sharpe_analysis.get('sharperatio', 0) or 0.0
    except:
        sharpe = 0.0

    try:
        max_dd = dd_analysis.get('max', {}).get('drawdown', 0) or 0.0
    except:
        max_dd = 0.0

    try:
        total_trades = trades_analysis.get('total', {}).get('total', 0)
        won_trades = trades_analysis.get('won', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
    except:
        total_trades = 0
        win_rate = 0

    return {
        'symbol': symbol,
        'lookback': lookback,
        'sl_at_prev': sl_at_prev,
        'trades': total_trades,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'equity_final': broker_value,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 140)
    print("STRATEGY 025 — EURUSD SWEEP (timeframes × lookbacks × LONG/SHORT)")
    print("2026 data, Backtrader Framework")
    print("=" * 140)

    TIMEFRAMES = ["1m", "5m", "1h"]
    LOOKBACKS = [5, 10, 20, 30]

    overall_best = None

    for tf in TIMEFRAMES:
        print(f"\n{'=' * 140}")
        print(f"[{tf.upper()}] EURUSD — 2026-01-01 to 2026-05-16")
        print("=" * 140)

        df = get_mt5_data("EURUSD", tf, "2026-01-01", "2026-05-16")
        if df.empty:
            print(f"  No data for {tf}")
            continue

        print(f"\n{'Config':<28} {'Dir':<6} {'Trades':>8} {'PnL':>13} {'Return%':>10} {'Win%':>8} {'Sharpe':>10} {'MaxDD':>8}")
        print("-" * 140)

        tf_best = None
        for lookback in LOOKBACKS:
            for sl_at_prev in [False, True]:
                for direction, dir_label in [(1, "LONG"), (-1, "SHORT")]:
                    name = f"{lookback}-bar SL@{'prev' if sl_at_prev else 'current'}"
                    result = run_backtest(df, "EURUSD", lookback, sl_at_prev,
                                          direction=direction)
                    if result and result['trades'] > 0:
                        print(f"{name:<28} {dir_label:<6} {result['trades']:>8} "
                              f"${result['pnl']:>+12.2f} {result['pnl_pct']:>+9.1f}% "
                              f"{result['win_rate']:>7.1f}% {result['sharpe']:>+10.2f} "
                              f"{result['max_dd']:>+7.1f}%")
                        if tf_best is None or result['pnl'] > tf_best['pnl']:
                            tf_best = result
                            tf_best['name'] = name
                            tf_best['direction'] = dir_label
                            tf_best['tf'] = tf
                    elif result:
                        print(f"{name:<28} {dir_label:<6} {'No trades':>8}")

        if tf_best:
            print("-" * 140)
            print(f"BEST [{tf}]: {tf_best['name']} {tf_best['direction']} → "
                  f"PnL=${tf_best['pnl']:+.2f}, Sharpe={tf_best['sharpe']:+.2f}, "
                  f"Win={tf_best['win_rate']:.1f}%, MaxDD={tf_best['max_dd']:.1f}%")
            if overall_best is None or tf_best['pnl'] > overall_best['pnl']:
                overall_best = tf_best

    print("\n" + "=" * 140)
    if overall_best:
        print(f"OVERALL BEST: [{overall_best['tf']}] {overall_best['name']} {overall_best['direction']} → "
              f"PnL=${overall_best['pnl']:+.2f}, Sharpe={overall_best['sharpe']:+.2f}, "
              f"Win={overall_best['win_rate']:.1f}%, MaxDD={overall_best['max_dd']:.1f}%")
    print("=" * 140)


if __name__ == "__main__":
    main()
