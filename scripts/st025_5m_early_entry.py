"""
Strategy 025 — 5m Strategy with Early-Entry (Backtrader)

Idea:
  - Detect swing-low/high signals on 5m candles (built from 1m bars)
  - But don't wait for the 5m candle to fully close — at minute 4
    (i.e., 4 of 5 minutes elapsed), "imagine" the candle closes now
    using the partial OHLC of minutes 0-3, and enter at that 1m bar's close.
  - This saves up to 1 minute vs waiting for the 5m close.

Compared to strategy025.py (which only sees completed 5m bars and enters at
the close of the NEXT 5m bar), this version trades on the SAME 5m bar where
the signal forms, 1 minute before that bar closes.

EURUSD only, 2026 data.
"""
import sys
from datetime import timedelta

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import backtrader as bt
from backtrader.feeds import PandasData

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================================
# MT5 DATA LOADING (1m data)
# ============================================================================

_MT5_TF_MAP = {"1m": 1, "5m": 5, "15m": 15, "1h": 16385}


def get_mt5_data(ticker: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    if not mt5.initialize():
        return pd.DataFrame()

    mt5.symbol_select(ticker, True)
    tf_const = _MT5_TF_MAP[timeframe]

    print(f"  {ticker} {timeframe} from {start} to {end}...", flush=True)

    # Chunked scan backward via copy_rates_from_pos until we cross ts_start.
    CHUNK = 50_000
    frames = []
    pos = 0
    iterations = 0
    while True:
        chunk = mt5.copy_rates_from_pos(ticker, tf_const, pos, CHUNK)
        if chunk is None or len(chunk) == 0:
            print(f"    chunk {iterations}: empty (mt5 error: {mt5.last_error()})")
            break
        frames.append(pd.DataFrame(chunk))
        oldest_utc = pd.Timestamp(int(frames[-1]["time"].min()), unit="s", tz="UTC")
        newest_utc = pd.Timestamp(int(frames[-1]["time"].max()), unit="s", tz="UTC")
        print(f"    chunk {iterations}: {len(chunk)} bars, {oldest_utc} -> {newest_utc}")
        iterations += 1
        if oldest_utc <= ts_start:
            break
        if len(chunk) < CHUNK:
            print(f"    chunk smaller than {CHUNK}, MT5 likely out of history")
            break
        pos += len(chunk)
        if iterations > 30:
            print(f"    safety break at 30 iterations")
            break

    if not frames:
        print("  No data loaded")
        return pd.DataFrame()

    df = pd.concat(frames[::-1])
    df = df.drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "tick_volume": "Volume"})
    if len(df) == 0:
        print(f"  After date filter: 0 bars (requested {ts_start} to {ts_end})")
        return pd.DataFrame()
    print(f"  Final: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def add_daily_colors_to_1m(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily candle color to each 1m bar."""
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
# DATA FEED WITH DAILY COLORS
# ============================================================================

class CustomPandasData(PandasData):
    lines = ('daily_color', 'daily_color_prev')
    params = (
        ('open', 'Open'), ('high', 'High'), ('low', 'Low'),
        ('close', 'Close'), ('volume', 'Volume'),
        ('openinterest', None),
        ('daily_color', 'daily_color'),
        ('daily_color_prev', 'daily_color_prev'),
    )


# ============================================================================
# EARLY-ENTRY STRATEGY (5m signals built from 1m data)
# ============================================================================

class EarlyEntryStrategy(bt.Strategy):
    """
    AGGRESSIVE early-entry: at minute 4 of the FORMING 5m candle, build a
    partial 5m candle from minutes 0-3 and check if it would qualify as a
    signal bar. If yes, enter immediately at this 1m bar's close.

    This means we enter on the SAME 5m bar as the signal, 1 minute before
    that bar would actually close. The original strategy025.py waits for
    the next 5m bar to close after the signal — we save ~6 minutes total.
    """
    params = (
        ("lookback", 10),
        ("sl_at_prev", False),
        ("direction", 1),
        ("risk_reward_ratio", 3.0),
        ("position_size", 10000),
    )

    def __init__(self):
        # Completed 5m bars (dicts of OHLC)
        self.bars_5m = []

        # Partial 5m candle being formed from current 1m bars
        self.partial_open = None
        self.partial_high = -np.inf
        self.partial_low = np.inf
        self.partial_close = None

        # Track swing low/high prices detected on completed 5m bars
        # (used for "higher low" / "lower high" comparison)
        self.swing_lows = []
        self.swing_highs = []

    def next(self):
        dt = self.data.datetime.datetime(0)
        minute_in_5m = dt.minute % 5  # 0..4

        o = float(self.data.open[0])
        h = float(self.data.high[0])
        l = float(self.data.low[0])
        c = float(self.data.close[0])

        # Update partial 5m candle
        if minute_in_5m == 0:
            self.partial_open = o
            self.partial_high = h
            self.partial_low = l
            self.partial_close = c
        else:
            if self.partial_open is None:
                self.partial_open = o
                self.partial_high = h
                self.partial_low = l
            else:
                self.partial_high = max(self.partial_high, h)
                self.partial_low = min(self.partial_low, l)
            self.partial_close = c

        # ---- AT MINUTE 4 (i.e., after the 4th 1m bar closes = position 3) ----
        # The partial 5m candle now contains data from minutes 0-3 (4 minutes).
        # Imagine this partial candle is the close of a 5m bar and check for
        # a signal. If valid, enter immediately at this 1m bar's close.
        if minute_in_5m == 3:
            self._check_signal_and_enter(c)

        # ---- AT MINUTE 5 (position 4) — actual 5m candle closes ----
        # Save the completed 5m bar for future signal detection.
        if minute_in_5m == 4:
            bar_5m = {
                "open": self.partial_open,
                "high": self.partial_high,
                "low": self.partial_low,
                "close": self.partial_close,
            }
            self.bars_5m.append(bar_5m)

            # Did this completed 5m bar form a NEW swing low/high?
            # (Used as the previous-swing reference for future comparisons.)
            self._record_swing_on_completed_5m()

            # Reset partial for next 5m period
            self.partial_open = None
            self.partial_high = -np.inf
            self.partial_low = np.inf
            self.partial_close = None

    def _check_signal_and_enter(self, current_price):
        """At minute 4, treat the partial 5m candle as if it just closed."""
        if self.position:
            return

        lookback = self.params.lookback
        if len(self.bars_5m) < lookback:
            return

        # Partial OHLC (so far, minutes 0-3)
        partial_high = self.partial_high
        partial_low = self.partial_low

        # Lookback window = last N COMPLETED 5m bars
        window = self.bars_5m[-lookback:]

        is_swing_low = partial_low < min(b["low"] for b in window)
        is_swing_high = partial_high > max(b["high"] for b in window)

        daily_color = int(self.data.daily_color[0])
        daily_color_prev = int(self.data.daily_color_prev[0])

        if self.params.direction == 1 and is_swing_low:
            prev_swing = self.swing_lows[-1] if self.swing_lows else None
            if (prev_swing is not None and
                    partial_low > prev_swing and
                    daily_color_prev == 1 and
                    daily_color == 1):
                self._enter_long(current_price, partial_low, prev_swing)

        elif self.params.direction == -1 and is_swing_high:
            prev_swing = self.swing_highs[-1] if self.swing_highs else None
            if (prev_swing is not None and
                    partial_high < prev_swing and
                    daily_color_prev == -1 and
                    daily_color == -1):
                self._enter_short(current_price, partial_high, prev_swing)

    def _record_swing_on_completed_5m(self):
        """After a 5m bar completes, record it if it's a new swing low/high
        relative to the past `lookback` completed bars."""
        lookback = self.params.lookback
        if len(self.bars_5m) < lookback + 1:
            return
        current = self.bars_5m[-1]
        window = self.bars_5m[-lookback - 1:-1]
        if current["low"] < min(b["low"] for b in window):
            self.swing_lows.append(current["low"])
        if current["high"] > max(b["high"] for b in window):
            self.swing_highs.append(current["high"])

    def _enter_long(self, entry_px, current_swing, prev_swing):
        sl_price = prev_swing if self.params.sl_at_prev else current_swing
        if entry_px <= sl_price:
            return
        risk = entry_px - sl_price
        tp_price = entry_px + risk * self.params.risk_reward_ratio
        self.buy_bracket(
            size=self.params.position_size,
            exectype=bt.Order.Market,
            stopprice=sl_price,
            limitprice=tp_price,
        )

    def _enter_short(self, entry_px, current_swing, prev_swing):
        sl_price = prev_swing if self.params.sl_at_prev else current_swing
        if entry_px >= sl_price:
            return
        risk = sl_price - entry_px
        tp_price = entry_px - risk * self.params.risk_reward_ratio
        self.sell_bracket(
            size=self.params.position_size,
            exectype=bt.Order.Market,
            stopprice=sl_price,
            limitprice=tp_price,
        )


# ============================================================================
# RUNNER
# ============================================================================

def run_backtest(df, lookback, sl_at_prev, direction, cash=10_000):
    if df.empty:
        return None

    cerebro = bt.Cerebro()
    cerebro.adddata(CustomPandasData(dataname=df))
    cerebro.addstrategy(
        EarlyEntryStrategy,
        lookback=lookback,
        sl_at_prev=sl_at_prev,
        direction=direction,
        position_size=10000,
    )
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                        timeframe=bt.TimeFrame.Days, riskfreerate=0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.0002, leverage=100, margin=False)

    try:
        results = cerebro.run()
    except Exception as e:
        print(f"    ERROR: {e}")
        return None
    strat = results[0]

    pnl = cerebro.broker.getvalue() - cash
    pnl_pct = (pnl / cash) * 100
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0.0
    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get('max', {}).get('drawdown', 0) or 0.0
    tr = strat.analyzers.trades.get_analysis()
    total_trades = tr.get('total', {}).get('total', 0)
    won_trades = tr.get('won', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    return {
        'trades': total_trades, 'pnl': pnl, 'pnl_pct': pnl_pct,
        'win_rate': win_rate, 'sharpe': sharpe, 'max_dd': max_dd,
    }


def main():
    print("=" * 140)
    print("STRATEGY 025 — EURUSD 5m EARLY ENTRY (signal+entry on same partial 5m bar)")
    print("SL@prev only — 2026 data — Backtrader")
    print("=" * 140)

    LOOKBACKS = [5, 10, 20, 30]

    df = get_mt5_data("EURUSD", "1m", "2026-01-01", "2026-12-31")
    if df.empty:
        print("No data")
        return
    df = add_daily_colors_to_1m(df)

    print(f"\n{'Config':<28} {'Dir':<6} {'Trades':>8} {'PnL':>13} {'Return%':>10} {'Win%':>8} {'Sharpe':>10} {'MaxDD':>8}")
    print("-" * 140)

    overall_best = None
    for lookback in LOOKBACKS:
        for direction, dir_label in [(1, "LONG"), (-1, "SHORT")]:
            name = f"{lookback}-bar SL@prev"
            result = run_backtest(df, lookback, sl_at_prev=True, direction=direction)
            if result and result['trades'] > 0:
                print(f"{name:<28} {dir_label:<6} {result['trades']:>8} "
                      f"${result['pnl']:>+12.2f} {result['pnl_pct']:>+9.1f}% "
                      f"{result['win_rate']:>7.1f}% {result['sharpe']:>+10.2f} "
                      f"{result['max_dd']:>+7.1f}%")
                if overall_best is None or result['pnl'] > overall_best['pnl']:
                    overall_best = {**result, 'name': name, 'dir': dir_label}
            elif result:
                print(f"{name:<28} {dir_label:<6} {'No trades':>8}")

    print("-" * 140)
    if overall_best:
        print(f"OVERALL BEST: {overall_best['name']} {overall_best['dir']} → "
              f"PnL=${overall_best['pnl']:+.2f}, Sharpe={overall_best['sharpe']:+.2f}, "
              f"Win={overall_best['win_rate']:.1f}%, MaxDD={overall_best['max_dd']:.1f}%")
    print("=" * 140)


if __name__ == "__main__":
    main()
