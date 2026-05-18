"""
Bar-by-bar simulator with spread, commission, and FTMO gates.

For each bar t in the trade window, the model emits p_long and p_short. If either crosses
its threshold and we have no open position, we enter at open(t+1) and place SL/TP.

FTMO rules enforced live:
  - Daily loss limit: if cumulative day PnL <= -daily_loss_pct * starting_equity, halt
    new entries until next session day. Close open positions at day end (defensive).
  - Max overall loss: if equity <= start_equity * (1 - max_loss_pct), terminate run.

We use 1m data inside each TF bar (when provided) to resolve which barrier hit first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.leakage import assert_tp_sl_invariant


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str               # "long" / "short"
    entry: float
    exit: float
    sl: float
    tp: float
    size: float             # units
    pnl: float              # net of commission
    outcome: str            # "tp" / "sl" / "timeout" / "day_close" / "max_loss"
    bars_held: int
    proba: float


@dataclass
class SimResult:
    trades: list[Trade]
    equity_curve: pd.Series
    start_equity: float
    end_equity: float
    halted: bool
    halt_reason: str | None = None
    daily_pnl: dict = field(default_factory=dict)


def _half_spread(spread: float) -> float:
    return spread / 2.0


def _resolve_intra_bar_1m(
    df_1m: pd.DataFrame,
    bar_start: pd.Timestamp,
    bar_end: pd.Timestamp,
    tp: float,
    sl: float,
    side: str,
) -> tuple[str, float] | None:
    """Inside one TF bar, scan 1m bars in order. Returns (outcome, exit_price) or None."""
    sub = df_1m.loc[bar_start:bar_end]
    if sub.empty:
        return None
    highs = sub["high"].values
    lows = sub["low"].values
    if side == "long":
        for i in range(len(sub)):
            if lows[i] <= sl:
                return ("sl", sl)
            if highs[i] >= tp:
                return ("tp", tp)
    else:
        for i in range(len(sub)):
            if highs[i] >= sl:
                return ("sl", sl)
            if lows[i] <= tp:
                return ("tp", tp)
    return None


def simulate(
    df: pd.DataFrame,                    # trade-window OHLCV, indexed by UTC
    df_1m: pd.DataFrame | None,          # 1m bars covering the trade window
    p_long: np.ndarray,                  # per-bar long probability
    p_short: np.ndarray,                 # per-bar short probability
    threshold_long: float,
    threshold_short: float,
    atr_series: pd.Series,               # ATR aligned to df
    *,
    sl_atr: float = 1.5,
    rr: float = 3.0,
    max_hold_bars: int = 48,
    spread: float = 0.00008,
    commission: float = 0.0002,
    start_equity: float = 100_000.0,
    risk_pct: float = 0.005,             # 0.5% per trade
    daily_loss_pct: float = 0.04,
    max_loss_pct: float = 0.08,
    tf_minutes: int = 5,
    pip_value_quote: float = 1.0,        # 1.0 for USD-quoted FX pairs
    pip_size: float = 0.0001,            # 1 pip in price units
    slippage_pips_sl: float = 0.0,       # extra adverse fill on SL (pips)
    slippage_pips_tp: float = 0.0,       # adverse fill on TP / market orders (pips)
) -> SimResult:
    """
    Returns SimResult. Assumes the prices in df are mid quotes.
    """
    assert len(p_long) == len(df) == len(p_short)
    n = len(df)

    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    idx = df.index
    atr_arr = atr_series.reindex(df.index).values
    half_sp = _half_spread(spread)
    slip_sl = slippage_pips_sl * pip_size
    slip_tp = slippage_pips_tp * pip_size

    equity = start_equity
    peak_equity = start_equity
    max_loss_floor = start_equity * (1 - max_loss_pct)
    daily_pnl: dict[pd.Timestamp, float] = {}
    halted = False
    halt_reason = None
    trades: list[Trade] = []
    equity_points: list[tuple[pd.Timestamp, float]] = [(idx[0], equity)]

    open_trade: dict | None = None
    halted_day: pd.Timestamp | None = None

    def _day_of(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.normalize()

    def _close_trade(trade_info: dict, exit_time: pd.Timestamp, exit_price: float, outcome: str):
        nonlocal equity, peak_equity
        side = trade_info["side"]
        entry = trade_info["entry"]
        size = trade_info["size"]
        if side == "long":
            gross = (exit_price - half_sp - entry) * size
        else:
            gross = (entry - (exit_price + half_sp)) * size
        gross *= pip_value_quote
        notional = entry * size
        comm = notional * commission * 2.0  # round trip
        pnl = gross - comm
        equity += pnl
        peak_equity = max(peak_equity, equity)
        day = _day_of(exit_time)
        daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl
        bars = trade_info.get("bars_held", 0)
        trades.append(
            Trade(
                entry_time=trade_info["entry_time"],
                exit_time=exit_time,
                side=side,
                entry=entry,
                exit=exit_price,
                sl=trade_info["sl"],
                tp=trade_info["tp"],
                size=size,
                pnl=pnl,
                outcome=outcome,
                bars_held=bars,
                proba=trade_info["proba"],
            )
        )

    for t in range(n - 1):
        ts = idx[t]
        day = _day_of(ts)
        equity_points.append((ts, equity))

        # FTMO max loss
        if equity <= max_loss_floor:
            halted = True
            halt_reason = "ftmo_max_loss"
            if open_trade is not None:
                _close_trade(open_trade, ts, open_arr[t], "max_loss")
                open_trade = None
            break

        # FTMO daily loss halt
        if daily_pnl.get(day, 0.0) <= -daily_loss_pct * start_equity:
            if halted_day != day:
                halted_day = day
                if open_trade is not None:
                    _close_trade(open_trade, ts, open_arr[t], "day_close")
                    open_trade = None

        # Manage open trade
        if open_trade is not None:
            open_trade["bars_held"] += 1
            bar_start = ts + pd.Timedelta("1ns")
            bar_end = ts + pd.Timedelta(minutes=tf_minutes)
            resolved = None
            if df_1m is not None:
                resolved = _resolve_intra_bar_1m(
                    df_1m, bar_start, bar_end,
                    open_trade["tp"], open_trade["sl"], open_trade["side"],
                )
            if resolved is None:
                # 5m fallback. Slippage applied at fill: SL fills WORSE than nominal,
                # TP fills WORSE than nominal (we never get better than the level).
                if open_trade["side"] == "long":
                    hit_sl = low_arr[t] <= open_trade["sl"]
                    hit_tp = high_arr[t] >= open_trade["tp"]
                    if hit_sl and hit_tp:
                        resolved = ("sl", open_trade["sl"] - slip_sl)
                    elif hit_sl:
                        resolved = ("sl", open_trade["sl"] - slip_sl)
                    elif hit_tp:
                        resolved = ("tp", open_trade["tp"] - slip_tp)
                else:
                    hit_sl = high_arr[t] >= open_trade["sl"]
                    hit_tp = low_arr[t] <= open_trade["tp"]
                    if hit_sl and hit_tp:
                        resolved = ("sl", open_trade["sl"] + slip_sl)
                    elif hit_sl:
                        resolved = ("sl", open_trade["sl"] + slip_sl)
                    elif hit_tp:
                        resolved = ("tp", open_trade["tp"] + slip_tp)
            if resolved is not None:
                _close_trade(open_trade, ts, resolved[1], resolved[0])
                open_trade = None
            elif open_trade["bars_held"] >= max_hold_bars:
                _close_trade(open_trade, ts, open_arr[t], "timeout")
                open_trade = None

        # Day end: close open trade defensively if next bar is next day
        if open_trade is not None and t + 1 < n:
            if _day_of(idx[t + 1]) != day:
                _close_trade(open_trade, ts, df["close"].values[t], "day_close")
                open_trade = None

        # New entry?
        if (
            open_trade is None
            and not (halted_day == day)
            and t + 1 < n
            and not np.isnan(atr_arr[t])
            and atr_arr[t] > 0
        ):
            want_long = p_long[t] >= threshold_long
            want_short = p_short[t] >= threshold_short
            if want_long and want_short:
                # take the stronger signal
                if p_long[t] >= p_short[t]:
                    want_short = False
                else:
                    want_long = False
            if want_long or want_short:
                entry_mid = open_arr[t + 1]
                a = atr_arr[t]
                sl_dist = sl_atr * a
                tp_dist = sl_dist * rr
                if want_long:
                    entry_price = entry_mid + half_sp
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist
                    side = "long"
                    proba = float(p_long[t])
                else:
                    entry_price = entry_mid - half_sp
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - tp_dist
                    side = "short"
                    proba = float(p_short[t])
                assert_tp_sl_invariant(entry_price, sl_price, tp_price, side)
                risk_usd = risk_pct * equity
                size = risk_usd / (sl_dist * pip_value_quote)
                open_trade = {
                    "entry_time": idx[t + 1],
                    "side": side,
                    "entry": entry_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "size": size,
                    "bars_held": 0,
                    "proba": proba,
                }

    # Close any trailing open trade
    if open_trade is not None and not halted:
        _close_trade(open_trade, idx[-1], df["close"].values[-1], "timeout")

    equity_curve = pd.Series(
        [e for _, e in equity_points] + [equity],
        index=[t for t, _ in equity_points] + [idx[-1]],
    )
    equity_curve = equity_curve[~equity_curve.index.duplicated(keep="last")]

    return SimResult(
        trades=trades,
        equity_curve=equity_curve,
        start_equity=start_equity,
        end_equity=equity,
        halted=halted,
        halt_reason=halt_reason,
        daily_pnl=daily_pnl,
    )
