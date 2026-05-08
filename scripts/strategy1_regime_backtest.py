"""
Regime-switching backtest — compares pure fade / pure breakout / three hybrid
regime detectors on 1m FX data (2024-2026).

Strategies
----------
pure_fade       Always fade every range breakout (current live strategy)
pure_breakout   Always follow every range breakout
option_a        breakout_dist < threshold×ATR → fade; ≥ threshold×ATR → follow
option_b        Breakout WITH EMA trend → follow; against EMA trend → fade
option_c        Per-pair rolling win-rate; switch mode when WR drops below threshold

Cost model (FTMO MT5, confirmed from 2026-05-06 live trades)
-------------------------------------------------------------
  Entry  : market order, crosses half-spread
  TP exit: limit order, fills at TP price (no spread)
  SL exit: stop→market, crosses half-spread
  Gap SL : if bar opens past SL price, fill at bar open (not SL)
  Commission: $5 per round-trip at 1 lot (FTMO Swing observed)

Usage
-----
  python -m scripts.strategy1_regime_backtest
  python -m scripts.strategy1_regime_backtest --start 2024-01-01 --end 2026-05-06 \\
      --oos-start 2026-01-01 --lots 1.0 --lookback 10 --no-plot
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.data_fetcher import fetch_ohlcv


# ── MT5 data loader ───────────────────────────────────────────────────────────

_MT5_TF_MAP = {
    "1m":  1,       # TIMEFRAME_M1
    "5m":  5,       # TIMEFRAME_M5
    "15m": 15,      # TIMEFRAME_M15
    "30m": 30,      # TIMEFRAME_M30
    "1h":  16385,   # TIMEFRAME_H1
    "4h":  16388,   # TIMEFRAME_H4
    "1d":  16408,   # TIMEFRAME_D1
}

# Minutes per bar for each timeframe (used to estimate fetch count)
_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def _load_data_mt5(
    pairs: list[str],
    start: str,
    end: str,
    timeframe: str = "1m",
    mt5_path: str | None = None,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
    symbol_suffix: str = "",
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV history from a running MT5 terminal for all pairs.
    Converts broker-local timestamps to UTC automatically.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise RuntimeError("MetaTrader5 package not installed.  pip install MetaTrader5")

    kwargs: dict = {}
    if mt5_path:
        kwargs["path"] = mt5_path
    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

    if login is not None:
        ok = mt5.login(login=int(login), password=password or "", server=server or "")
        if not ok:
            mt5.shutdown()
            raise RuntimeError(f"mt5.login() failed: {mt5.last_error()}")

    # Detect broker UTC offset from the first available tick
    first_sym = (pairs[0] + symbol_suffix) if pairs else "EURUSD"
    mt5.symbol_select(first_sym, True)
    tick = mt5.symbol_info_tick(first_sym)
    if tick is not None:
        from datetime import datetime, timezone
        broker_ts = tick.time
        utc_ts    = datetime.now(tz=timezone.utc).timestamp()
        tz_offset = round((broker_ts - utc_ts) / 3600)
    else:
        tz_offset = 0
    log.info("MT5 broker UTC offset: %+dh", tz_offset)

    ts_start  = pd.Timestamp(start, tz="UTC")
    ts_end    = pd.Timestamp(end,   tz="UTC")
    tf_const  = _MT5_TF_MAP.get(timeframe, 1)
    tf_min    = _TF_MINUTES.get(timeframe, 1)

    # MT5 caps copy_rates_from_pos at ~100K bars per call; fetch in chunks.
    # For 5m data, 50K bars ~ 6 months; for 1m data ~ 5 weeks — loop as needed.
    CHUNK = 50_000

    result: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        sym_mt5 = pair + symbol_suffix
        mt5.symbol_select(sym_mt5, True)

        frames = []
        pos = 0
        while True:
            chunk = mt5.copy_rates_from_pos(sym_mt5, tf_const, pos, CHUNK)
            if chunk is None or len(chunk) == 0:
                break
            frames.append(pd.DataFrame(chunk))
            oldest_broker_ts = int(frames[-1]["time"].min())
            oldest_utc = pd.Timestamp(oldest_broker_ts, unit="s", tz="UTC") \
                         - pd.Timedelta(hours=tz_offset)
            if oldest_utc <= ts_start:
                break                    # reached far enough back
            if len(chunk) < CHUNK:
                break                    # terminal ran out of history
            pos += CHUNK

        if not frames:
            log.warning("  %s (%s): no data  last_error=%s",
                        pair, sym_mt5, mt5.last_error())
            continue

        df = pd.concat(frames[::-1])     # oldest first
        df = df.drop_duplicates(subset="time").sort_values("time")
        broker_dt = pd.to_datetime(df["time"], unit="s", utc=True)
        df.index  = broker_dt - pd.Timedelta(hours=tz_offset)
        df = df[(df.index >= ts_start) & (df.index < ts_end)]
        if len(df) == 0:
            log.warning("  %s: fetched data but none falls in %s → %s", pair, start, end)
            continue
        df = df.rename(columns={
            "open": "Open", "high": "High",
            "low":  "Low",  "close": "Close", "tick_volume": "Volume",
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        result[pair] = df
        log.info("  %s: %d bars", pair, len(df))

    mt5.shutdown()
    return result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("regime_backtest")

# ── Constants ─────────────────────────────────────────────────────────────────

PAIRS = ["EURUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD",
         "GBPUSD", "USDJPY", "EURJPY"]

# Half-spread in pips (bid-ask / 2), calibrated from FTMO MT5 live data
HALF_SPREAD_PIPS = {
    "EURUSD": 0.20,
    "AUDUSD": 0.30,
    "NZDUSD": 0.50,
    "USDCHF": 0.50,
    "USDCAD": 0.40,
    "GBPUSD": 0.60,  # estimated from FTMO live data
    "USDJPY": 0.30,
    "EURGBP": 0.70,
    "EURJPY": 0.50,
}

# $-value of 1 pip at 1 standard lot
PIP_VALUE = {
    "EURUSD": 10.0,
    "AUDUSD": 10.0,
    "NZDUSD": 10.0,
    "USDCHF": 9.8,   # approx — varies with USDCHF rate
    "USDCAD": 7.5,   # approx — varies with USDCAD rate
    "GBPUSD": 10.0,
    "USDJPY": 7.0,   # approx — varies with JPY rate
    "EURGBP": 12.0,  # approx — varies with GBPUSD rate
    "EURJPY": 7.0,   # approx — varies with JPY rate
}

COMMISSION_PER_LOT = 7.0   # $ per round-trip, 1 lot (FTMO Swing — verify from account history)
PIP_SIZE = 0.0001           # 4-decimal pairs (non-JPY)

# Per-pair pip size: JPY pairs use 0.01 (2-decimal), all others 0.0001
PIP_SIZE_MAP = {
    "EURUSD": 0.0001, "AUDUSD": 0.0001, "NZDUSD": 0.0001,
    "USDCHF": 0.0001, "USDCAD": 0.0001, "GBPUSD": 0.0001,
    "EURGBP": 0.0001, "USDJPY": 0.01,   "EURJPY": 0.01,
}

# ── Indicator helpers ─────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int) -> np.ndarray:
    hi  = df["High"].values
    lo  = df["Low"].values
    cl  = df["Close"].values
    tr  = np.maximum(hi - lo,
          np.maximum(np.abs(hi - np.roll(cl, 1)),
                     np.abs(lo  - np.roll(cl, 1))))
    tr[0] = hi[0] - lo[0]
    # Wilder EMA
    atr = np.empty(len(tr))
    atr[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, len(tr)):
        atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
    return atr


def _ema(closes: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    out = np.empty(len(closes))
    out[0] = closes[0]
    for i in range(1, len(closes)):
        out[i] = out[i-1] * (1 - alpha) + closes[i] * alpha
    return out


# ── Cost model ────────────────────────────────────────────────────────────────

def _entry_cost(pair: str, lots: float) -> float:
    """Dollar cost at entry (market order crosses half spread)."""
    return HALF_SPREAD_PIPS[pair] * PIP_VALUE[pair] * lots


def _exit_cost_sl(pair: str, lots: float) -> float:
    """Dollar cost at SL exit (stop→market crosses half spread)."""
    return HALF_SPREAD_PIPS[pair] * PIP_VALUE[pair] * lots


def _commission(lots: float) -> float:
    return COMMISSION_PER_LOT * lots


# ── Signal detection ──────────────────────────────────────────────────────────

def _detect_breakout(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr_vals: np.ndarray,
    i: int,
    lookback: int,
    tight_atr: float | None,
) -> dict | None:
    """
    At bar i, check if close crossed outside the rolling range.
    Range = [min(Low[i-lookback:i]), max(High[i-lookback:i])] — no look-ahead.
    Returns signal dict or None.
    """
    if i < lookback + 1:
        return None
    window_hi = highs[i - lookback: i]
    window_lo = lows [i - lookback: i]
    range_high = float(window_hi.max())
    range_low  = float(window_lo.min())
    close      = closes[i]
    atr_e      = atr_vals[i]
    if atr_e <= 0:
        return None

    if tight_atr is not None and (range_high - range_low) >= tight_atr * atr_e:
        return None

    if close > range_high:
        breakout_dist = close - range_high
        raw_direction = 1     # raw breakout direction (up = +1)
    elif close < range_low:
        breakout_dist = range_low - close
        raw_direction = -1    # raw breakout direction (down = −1)
    else:
        return None

    return {
        "i":             i,
        "close":         close,
        "range_high":    range_high,
        "range_low":     range_low,
        "breakout_dist": breakout_dist,
        "raw_direction": raw_direction,   # +1 = upward break, −1 = downward break
        "atr":           atr_e,
    }


# ── Regime selectors ──────────────────────────────────────────────────────────

def _mode_to_direction(raw_direction: int, mode: str) -> int:
    """
    Convert breakout direction + mode into trade direction.
    Fade: trade opposite the breakout (mean-reversion).
    Breakout: trade same as breakout (momentum).
    """
    if mode == "fade":
        return -raw_direction
    else:
        return raw_direction


def regime_pure_fade(sig, *_) -> str:
    return "fade"


def regime_pure_breakout(sig, *_) -> str:
    return "breakout"


def make_regime_a(threshold: float):
    """Option A: small breakout → fade; large breakout → follow."""
    def _fn(sig, closes, ema_vals, _state) -> str:
        if sig["breakout_dist"] < threshold * sig["atr"]:
            return "fade"
        return "breakout"
    return _fn


def make_regime_b(ema_period: int, min_slope_atr: float = 0.1):
    """
    Option B: EMA slope tells us the trend.
    Breakout WITH trend → follow (breakout mode).
    Breakout AGAINST trend → fade.
    Flat EMA → fade (default).

    min_slope_atr: EMA slope must exceed this fraction of ATR to count as trending.
    """
    def _fn(sig, closes, ema_vals, _state) -> str:
        i = sig["i"]
        if i < 2 or ema_vals is None:
            return "fade"
        slope = ema_vals[i] - ema_vals[i - 1]
        atr   = sig["atr"]
        if abs(slope) < min_slope_atr * atr:
            return "fade"                          # EMA flat → range regime
        ema_up   = slope > 0
        break_up = sig["raw_direction"] > 0
        if ema_up == break_up:
            return "breakout"                      # breakout aligns with EMA trend
        return "fade"                              # breakout against EMA trend
    return _fn


def make_regime_c(window: int, switch_threshold: float):
    """
    Option C: per-pair rolling win-rate.
    Tracks last `window` trade outcomes in the current mode.
    Switches mode when WR < switch_threshold; switches back when the new
    mode's WR also drops below switch_threshold (avoids infinite flip).

    Returns a tuple (regime_fn, state_dict) so callers can reset state
    between pairs.
    """
    def make_state():
        return {
            "mode":      "fade",
            "outcomes":  deque(maxlen=window),   # 1=win, 0=loss in current mode
            "cooldown":  0,                       # bars before we can switch again
        }

    def _fn(sig, closes, ema_vals, state) -> str:
        if state["cooldown"] > 0:
            state["cooldown"] -= 1
            return state["mode"]
        outcomes = state["outcomes"]
        if len(outcomes) >= max(3, window // 2):
            wr = sum(outcomes) / len(outcomes)
            if wr < switch_threshold:
                # flip mode, reset outcomes, set cooldown
                state["mode"]     = "breakout" if state["mode"] == "fade" else "fade"
                state["outcomes"] = deque(maxlen=window)
                state["cooldown"] = window        # don't flip again immediately
        return state["mode"]

    return _fn, make_state


# ── TP/SL calculation ─────────────────────────────────────────────────────────

def _compute_levels(
    close: float,
    direction: int,
    atr: float,
    mode: str,
    tp_atr_fade: float,
    sl_atr_fade: float,
    tp_atr_break: float,
    sl_atr_break: float,
) -> tuple[float, float]:
    """Return (tp_price, sl_price) for the given mode and direction."""
    if mode == "fade":
        tp_mult, sl_mult = tp_atr_fade, sl_atr_fade
    else:
        tp_mult, sl_mult = tp_atr_break, sl_atr_break
    tp = close + tp_mult * atr * direction
    sl = close - sl_mult * atr * direction
    return tp, sl


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate_pair(
    pair: str,
    df: pd.DataFrame,
    lookback: int,
    atr_period: int,
    ema_period: int,
    tp_atr_fade: float,
    sl_atr_fade: float,
    tp_atr_break: float,
    sl_atr_break: float,
    lots: float,
    tight_atr: float | None,
    regime_fn,
    regime_state_factory,    # callable → fresh state dict; None if stateless
    cb_window: int = 0,      # circuit breaker: rolling window (0 = disabled)
    cb_min_losses: int = 6,  # losses out of cb_window that trigger pause
    cb_break_bars: int = 0,  # bars to pause after trigger (60 = 1h at 1m)
) -> list[dict]:
    """
    Bar-by-bar simulation for one pair.
    Returns list of closed trade dicts.
    """
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    opens  = df["Open"].values
    times  = df.index

    atr_vals = _atr(df, atr_period)
    ema_vals = _ema(closes, ema_period)

    warmup = lookback + atr_period + 1
    in_trade: dict | None = None
    trades: list[dict] = []
    state = regime_state_factory() if regime_state_factory else {}
    cb_outcomes: deque = deque(maxlen=cb_window if cb_window > 0 else 1)
    cooldown_until: int = 0

    half_sp  = HALF_SPREAD_PIPS[pair] * PIP_VALUE[pair] * lots / 10  # $/0.1pip
    half_sp_val = HALF_SPREAD_PIPS[pair] * PIP_SIZE     # price units

    for i in range(warmup, len(df)):
        bar_open  = opens[i]
        bar_high  = highs[i]
        bar_low   = lows[i]
        bar_close = closes[i]
        bar_time  = times[i]

        # ── Check exit for open trade ──────────────────────────────────────
        if in_trade is not None:
            tp      = in_trade["tp"]
            sl      = in_trade["sl"]
            entry   = in_trade["entry"]
            dir_    = in_trade["direction"]
            mode_   = in_trade["mode"]
            tick_val = PIP_VALUE[pair] * lots / 10   # $/0.1pip = $/pip*0.1

            exit_price = None
            exit_type  = None

            if dir_ == 1:   # BUY position
                if bar_open <= sl:                   # gap down through SL
                    exit_price, exit_type = bar_open, "sl_gap"
                elif bar_low <= sl:                  # normal SL
                    exit_price, exit_type = sl,      "sl"
                elif bar_high >= tp:                 # TP
                    exit_price, exit_type = tp,      "tp"
            else:            # SELL position
                if bar_open >= sl:                   # gap up through SL
                    exit_price, exit_type = bar_open, "sl_gap"
                elif bar_high >= sl:                 # normal SL
                    exit_price, exit_type = sl,      "sl"
                elif bar_low <= tp:                  # TP
                    exit_price, exit_type = tp,      "tp"

            if exit_price is not None:
                pip_move = (exit_price - entry) * dir_ / PIP_SIZE
                gross    = pip_move * PIP_VALUE[pair] * lots
                # Cost: entry half-spread already deducted; add exit cost for SL
                sl_exit_cost = _exit_cost_sl(pair, lots) if "sl" in exit_type else 0.0
                net = gross - in_trade["entry_cost"] - sl_exit_cost - _commission(lots)
                won = net > 0

                trade = {
                    "time":       in_trade["time"],
                    "close_time": bar_time,
                    "pair":       pair,
                    "mode":       mode_,
                    "direction":  dir_,
                    "entry":      entry,
                    "tp":         tp,
                    "sl":         sl,
                    "exit":       exit_price,
                    "exit_type":  exit_type,
                    "gross":      gross,
                    "net":        net,
                    "won":        won,
                    "atr":        in_trade["atr"],
                }
                trades.append(trade)
                in_trade = None

                # Option C: record outcome for regime tracking
                if regime_state_factory is not None and state:
                    state["outcomes"].append(1 if won else 0)

                # Circuit breaker: pause if too many recent losses
                if cb_window > 0:
                    cb_outcomes.append(0 if won else 1)   # 1 = loss
                    if (len(cb_outcomes) == cb_window
                            and sum(cb_outcomes) >= cb_min_losses):
                        cooldown_until = i + cb_break_bars
                        cb_outcomes.clear()

        # ── Check entry ────────────────────────────────────────────────────
        if in_trade is None and i >= cooldown_until:
            sig = _detect_breakout(closes, highs, lows, atr_vals, i,
                                   lookback, tight_atr)
            if sig is not None:
                mode = regime_fn(sig, closes, ema_vals, state)
                dir_ = _mode_to_direction(sig["raw_direction"], mode)

                # Entry price: fill at close ± half-spread (market order)
                entry_spread_adj = half_sp_val * dir_   # buy: +spread, sell: -spread
                # For BUY we pay ask = close + half_sp; for SELL we get bid = close - half_sp
                # SL/TP are anchored to the bar close (same as current strategy)
                tp, sl = _compute_levels(bar_close, dir_, sig["atr"], mode,
                                         tp_atr_fade, sl_atr_fade,
                                         tp_atr_break, sl_atr_break)

                entry_cost = _entry_cost(pair, lots)

                in_trade = {
                    "time":       bar_time,
                    "entry":      bar_close + entry_spread_adj,
                    "tp":         tp,
                    "sl":         sl,
                    "direction":  dir_,
                    "mode":       mode,
                    "atr":        sig["atr"],
                    "entry_cost": entry_cost,
                }

    return trades


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[dict], label: str, period: str) -> dict:
    if not trades:
        return {"label": label, "period": period, "n": 0, "pnl": 0.0,
                "win_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "sharpe": 0.0, "max_dd": 0.0, "ev_per_trade": 0.0}

    nets  = np.array([t["net"] for t in trades])
    won   = np.array([t["won"] for t in trades], dtype=bool)
    n     = len(trades)
    pnl   = nets.sum()
    wr    = won.mean()
    aw    = nets[won].mean()  if won.any()  else 0.0
    al    = nets[~won].mean() if (~won).any() else 0.0

    # Per-trade Sharpe (annualised): mean/std × sqrt(trades_per_year)
    # Approximate trades/year from the dataset span
    if n > 1 and nets.std() > 0:
        # Count trading days in dataset
        days_span = max(1, (trades[-1]["time"] - trades[0]["time"]).days)
        tpy = n / (days_span / 365.25)
        sharpe = (nets.mean() / nets.std()) * np.sqrt(tpy)
    else:
        sharpe = 0.0

    # Max drawdown
    cum = np.cumsum(nets)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak)
    max_dd = dd.min()

    ev = pnl / n

    return {"label": label, "period": period, "n": n,
            "pnl": pnl, "win_pct": round(wr * 100, 1),
            "avg_win": round(aw, 2), "avg_loss": round(al, 2),
            "sharpe": round(sharpe, 2), "max_dd": round(max_dd, 2),
            "ev_per_trade": round(ev, 2)}


# ── Option D: Random Forest classifier ───────────────────────────────────────

def _signal_features(sig: dict, closes: np.ndarray, ema_vals: np.ndarray) -> list:
    """Feature vector from signal context (no look-ahead)."""
    i = sig["i"]
    ema_slope = (ema_vals[i] - ema_vals[i - 1]) / sig["atr"] if i > 0 else 0.0
    return [
        sig["breakout_dist"] / sig["atr"],
        (sig["range_high"] - sig["range_low"]) / sig["atr"],
        ema_slope,
        abs(closes[i] - ema_vals[i]) / sig["atr"],
        sig["atr"] / closes[i] * 10000,
        float(sig["raw_direction"]),
    ]


def collect_is_features_labels(
    pairs_data: dict[str, pd.DataFrame],
    oos_start: pd.Timestamp,
    lookback: int,
    atr_period: int,
    ema_period: int,
    tp_atr: float,
    sl_atr: float,
    tight_atr: float | None,
    max_forward_bars: int = 100,
) -> tuple[list, list]:
    """
    Forward-look only on IS data to produce supervised labels.
    label=1 → fade won, label=0 → breakout won.
    """
    X, y = [], []
    for pair, df_full in pairs_data.items():
        df = df_full[df_full.index < oos_start]
        if len(df) < lookback + atr_period + max_forward_bars + 10:
            continue
        closes   = df["Close"].values
        highs    = df["High"].values
        lows     = df["Low"].values
        atr_vals = _atr(df, atr_period)
        ema_vals = _ema(closes, ema_period)
        warmup   = lookback + atr_period + 1
        for i in range(warmup, len(df) - max_forward_bars):
            sig = _detect_breakout(closes, highs, lows, atr_vals, i, lookback, tight_atr)
            if sig is None:
                continue
            fade_dir = -sig["raw_direction"]
            tp_fade  = closes[i] + fade_dir * tp_atr  * sig["atr"]
            sl_fade  = closes[i] - fade_dir * sl_atr  * sig["atr"]
            label = None
            for j in range(i + 1, min(i + max_forward_bars, len(df))):
                if fade_dir == 1:
                    if lows[j]  <= sl_fade: label = 0; break
                    if highs[j] >= tp_fade: label = 1; break
                else:
                    if highs[j] >= sl_fade: label = 0; break
                    if lows[j]  <= tp_fade: label = 1; break
            if label is None:
                continue
            X.append(_signal_features(sig, closes, ema_vals))
            y.append(label)
    return X, y


def train_regime_rf(X: list, y: list):
    """Train RandomForestClassifier on IS features/labels. Returns model or None."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        Xa = np.array(X)
        ya = np.array(y)
        model = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            random_state=42, n_jobs=1,
        )
        model.fit(Xa, ya)
        log.info("RF trained: %d samples, %.1f%% fade-wins", len(ya), ya.mean() * 100)
        feat_names = ["bo_dist_atr", "range_atr", "ema_slope", "price_ema_dist",
                      "atr_norm", "direction"]
        importances = sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
        log.info("  Importances: %s", "  ".join(f"{n}={v:.3f}" for n, v in importances))
        return model
    except ImportError:
        log.warning("sklearn not available — skipping Option D")
        return None
    except Exception as exc:
        log.error("RF training failed: %s", exc)
        return None


def make_regime_d(model):
    """Option D: Random Forest selects fade vs breakout per signal."""
    def _fn(sig, closes, ema_vals, _state) -> str:
        pred = model.predict([_signal_features(sig, closes, ema_vals)])[0]
        return "fade" if pred == 1 else "breakout"
    return _fn


# ── Strategy runner ───────────────────────────────────────────────────────────

def run_strategy(
    label: str,
    pairs_data: dict[str, pd.DataFrame],
    lookback: int,
    atr_period: int,
    ema_period: int,
    tp_atr_fade: float,
    sl_atr_fade: float,
    tp_atr_break: float,
    sl_atr_break: float,
    lots: float,
    tight_atr: float | None,
    regime_fn,
    regime_state_factory,
    oos_start: pd.Timestamp,
    cb_window: int = 0,
    cb_min_losses: int = 6,
    cb_break_bars: int = 0,
) -> tuple[dict, dict]:
    """Run strategy on all pairs, split IS/OOS. Returns (is_metrics, oos_metrics)."""
    all_is, all_oos = [], []

    for pair, df in pairs_data.items():
        trades = simulate_pair(
            pair, df, lookback, atr_period, ema_period,
            tp_atr_fade, sl_atr_fade, tp_atr_break, sl_atr_break,
            lots, tight_atr, regime_fn, regime_state_factory,
            cb_window, cb_min_losses, cb_break_bars,
        )
        for t in trades:
            if t["time"] < oos_start:
                all_is.append(t)
            else:
                all_oos.append(t)

    is_m  = compute_metrics(all_is,  label, "IS")
    oos_m = compute_metrics(all_oos, label, "OOS")
    return is_m, oos_m


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--start",       default="2024-01-01")
    ap.add_argument("--end",         default="2026-05-06")
    ap.add_argument("--oos-start",   default="2026-01-01",
                    help="Start of out-of-sample window")
    ap.add_argument("--lookback",    type=int,   default=10)
    ap.add_argument("--atr-period",  type=int,   default=14)
    ap.add_argument("--ema-period",  type=int,   default=20,
                    help="EMA period for Option B regime detector")
    ap.add_argument("--lots",        type=float, default=1.0)
    ap.add_argument("--timeframe",      default="1m",
                    choices=list(_MT5_TF_MAP.keys()),
                    help="Bar timeframe (default 1m)")
    ap.add_argument("--no-tight",       action="store_true", default=True)
    ap.add_argument("--no-plot",        action="store_true", default=False)
    ap.add_argument("--quick",          action="store_true", default=False,
                    help="Run only pure_fade and pure_breakout (skip A/B/C/D/E)")
    ap.add_argument("--pairs",          nargs="+",  default=None)
    # Data source
    ap.add_argument("--source",         default="api", choices=["api", "mt5"],
                    help="Data source: 'api' (default) or 'mt5' (live terminal)")
    ap.add_argument("--mt5-path",       default=None, help="Path to terminal64.exe")
    ap.add_argument("--login",          default=None, type=int)
    ap.add_argument("--password",       default=None)
    ap.add_argument("--server",         default=None)
    ap.add_argument("--symbol-suffix",  default="", help="MT5 symbol suffix, e.g. .raw")
    args = ap.parse_args()

    pairs = args.pairs or PAIRS
    tight_atr = None if args.no_tight else 1.5
    oos_start = pd.Timestamp(args.oos_start, tz="UTC")

    # ── Load data ────────────────────────────────────────────────────────────
    log.info("Loading %s data (%s) for %s  (%s → %s)",
             args.timeframe, args.source, pairs, args.start, args.end)
    pairs_data: dict[str, pd.DataFrame] = {}

    if args.source == "mt5":
        try:
            pairs_data = _load_data_mt5(
                pairs, args.start, args.end,
                timeframe=args.timeframe,
                mt5_path=args.mt5_path, login=args.login,
                password=args.password, server=args.server,
                symbol_suffix=args.symbol_suffix,
            )
        except Exception as exc:
            log.error("MT5 data load failed: %s", exc)
    else:
        for pair in pairs:
            try:
                df = fetch_ohlcv(pair, args.timeframe, args.start, args.end)
                pairs_data[pair] = df
                log.info("  %s: %d bars", pair, len(df))
            except Exception as exc:
                log.error("  Failed to load %s: %s", pair, exc)

    if not pairs_data:
        log.error("No data loaded.")
        return

    # Common params
    base = dict(
        pairs_data=pairs_data,
        lookback=args.lookback,
        atr_period=args.atr_period,
        ema_period=args.ema_period,
        tp_atr_fade=1.0,   sl_atr_fade=2.0,    # validated config
        tp_atr_break=2.0,  sl_atr_break=1.0,   # breakout: wide TP, tight SL
        lots=args.lots,
        tight_atr=tight_atr,
        oos_start=oos_start,
    )

    results_is, results_oos = [], []

    def _run(label, regime_fn, regime_state_factory=None):
        log.info("Running: %s", label)
        m_is, m_oos = run_strategy(
            label, **base,
            regime_fn=regime_fn,
            regime_state_factory=regime_state_factory,
        )
        results_is.append(m_is)
        results_oos.append(m_oos)

    # ── 0. Baselines ─────────────────────────────────────────────────────────
    _run("pure_fade",     regime_pure_fade)
    _run("pure_breakout", regime_pure_breakout)

    if not args.quick:
        # ── Option A: breakout magnitude threshold ────────────────────────────
        for thresh in [0.1, 0.25, 0.5, 0.75, 1.0]:
            _run(f"A_thresh={thresh}", make_regime_a(thresh))

        # ── Option B: EMA slope regime ────────────────────────────────────────
        for ema_p in [9, 20, 50]:
            for min_slope in [0.05, 0.15, 0.30]:
                _run(f"B_ema={ema_p}_slope={min_slope}",
                     make_regime_b(ema_p, min_slope))

        # ── Option C: rolling win-rate switch ─────────────────────────────────
        for window in [5, 10, 20]:
            for thresh in [0.40, 0.45, 0.50, 0.55]:
                regime_fn, make_state = make_regime_c(window, thresh)
                _run(f"C_win={window}_thr={thresh}", regime_fn, make_state)

        # ── Option E: circuit breaker ─────────────────────────────────────────
        tf_min = _TF_MINUTES.get(args.timeframe, 1)
        for break_hours in [1, 2, 3]:
            lbl = f"E_cb6of10_{break_hours}h"
            log.info("Running: %s", lbl)
            m_is, m_oos = run_strategy(
                lbl, **base,
                regime_fn=regime_pure_fade,
                regime_state_factory=None,
                cb_window=10, cb_min_losses=6,
                cb_break_bars=break_hours * 60 // tf_min,
            )
            results_is.append(m_is)
            results_oos.append(m_oos)

        # ── Option D: Random Forest ───────────────────────────────────────────
        log.info("Collecting IS features for Option D (RF)...")
        X_rf, y_rf = collect_is_features_labels(
            pairs_data, oos_start,
            lookback=args.lookback,
            atr_period=args.atr_period,
            ema_period=args.ema_period,
            tp_atr=base["tp_atr_fade"],
            sl_atr=base["sl_atr_fade"],
            tight_atr=tight_atr,
        )
        if X_rf:
            rf_model = train_regime_rf(X_rf, y_rf)
            if rf_model is not None:
                _run("D_random_forest", make_regime_d(rf_model))

    # ── Results table ─────────────────────────────────────────────────────────
    cols = ["label", "n", "pnl", "win_pct", "avg_win", "avg_loss",
            "ev_per_trade", "max_dd", "sharpe"]

    df_is  = pd.DataFrame(results_is)[cols].sort_values("pnl", ascending=False)
    df_oos = pd.DataFrame(results_oos)[cols].sort_values("pnl", ascending=False)

    def _fmt(df, title):
        print(f"\n{'='*90}")
        print(f"  {title}  (lots={args.lots})")
        print(f"{'='*90}")
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    _fmt(df_is,  f"IN-SAMPLE  ({args.start} → {args.oos_start})")
    _fmt(df_oos, f"OUT-OF-SAMPLE  ({args.oos_start} → {args.end})")

    # Show top-5 OOS with IS comparison side-by-side
    print(f"\n{'='*90}")
    print("  TOP-5 OOS — IS vs OOS side-by-side")
    print(f"{'='*90}")
    top5 = df_oos.head(5)["label"].tolist()
    rows = []
    for lbl in top5:
        is_row  = df_is[df_is["label"]  == lbl].iloc[0]
        oos_row = df_oos[df_oos["label"] == lbl].iloc[0]
        rows.append({
            "label":       lbl,
            "IS_pnl":      is_row["pnl"],
            "IS_win%":     is_row["win_pct"],
            "IS_sharpe":   is_row["sharpe"],
            "OOS_pnl":     oos_row["pnl"],
            "OOS_win%":    oos_row["win_pct"],
            "OOS_sharpe":  oos_row["sharpe"],
            "OOS_maxDD":   oos_row["max_dd"],
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    # ── Optional equity-curve plot ─────────────────────────────────────────────
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
            for period, ax, res_list in [("IS", axes[0], results_is),
                                         ("OOS", axes[1], results_oos)]:
                ax.set_title(f"{period} — P&L by strategy")
                labels = [r["label"] for r in res_list]
                pnls   = [r["pnl"]   for r in res_list]
                colors = ["green" if p > 0 else "red" for p in pnls]
                ax.barh(labels, pnls, color=colors)
                ax.axvline(0, color="black", lw=0.8)
                ax.set_xlabel("Net P&L ($)")
            plt.tight_layout()
            out_path = ROOT / "data" / "backtest_curves" / "regime_comparison.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=120)
            log.info("Plot saved: %s", out_path)
            plt.show()
        except Exception as exc:
            log.warning("Plot failed: %s", exc)


if __name__ == "__main__":
    main()
