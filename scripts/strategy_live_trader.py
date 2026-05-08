"""
Live trader: Combined 1h (EMA9) + 5m (EMA20) double-EMA extrema strategy.

Signal logic (same as backtest):
  - Double EMA of HL/2: DEMA = EMA(EMA((H+L)/2, p), p)
  - argmax/argmin in sliding window → peak = SHORT, valley = LONG
  - 1h signal is the anchor; 5m signal opposing active 1h direction is blocked
  - Same direction: 1h + 5m positions may be open simultaneously (max 0.75 lots)

Order management:
  - Entry:  market order using Ask (BUY) or Bid (SELL)
  - TP:     embedded in entry order (broker-side limit), fired when
            Bid >= TP (long) or Ask <= TP (short)
  - Exit:   market close on opposite signal; or broker closes via TP/SL
  - SL:     1h positions get 40-pip hard SL by default (caps tail risk);
            5m positions have no SL — close_profit handles 5m exits

Daily circuit-breaker (portfolio-level):
  - Tracks realized + unrealized P&L across all pairs every poll cycle
  - If total daily P&L < -daily_stop_usd: close ALL positions, halt until midnight UTC

Risk note (FTMO $100k swing — challenge-safe defaults):
  - Default lots: 0.5 (1h) + 0.25 (5m) — quarter of backtest lot sizes
  - OOS scaled worst day: ~-$64;  daily stop at -$1000 = 5x FTMO buffer
  - Scale to 1.0 + 0.5 lots only after passing the challenge phase

Usage
-----
  python -m scripts.strategy_live_trader \\
      --login 1513313327 --server FTMO-Demo \\
      [--pairs EURUSD AUDUSD NZDUSD USDCHF USDCAD GBPUSD USDJPY EURJPY] \\
      [--ema-1h 9] [--ema-5m 20] [--window 20] \\
      [--lots-1h 0.5] [--lots-5m 0.25] \\
      [--tp-1h-pips 10] [--tp-5m-pips 3] \\
      [--sl-pips-1h 40] [--sl-pips-5m 0] \\
      [--max-dist-pips 0] \\
      [--daily-stop 1000] \\
      [--min-bars-5m 3] [--min-bars-1h 2] \\
      [--partial-usd 0.0]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not found — pip install MetaTrader5")
    sys.exit(1)

from scripts.strategy1_regime_backtest import _ema, HALF_SPREAD_PIPS

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("live_trader.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("live_trader")

# ── Constants ─────────────────────────────────────────────────────────────────

MAGIC             = 202601  # unique identifier for this bot's orders
POLL_SECS         = 10      # normal sleep between polls
POLL_SECS_NEAR    = 1       # tight polling within NEAR_BAR_SECS of bar close
NEAR_BAR_SECS     = 12      # switch to tight polling this many seconds before bar
BARS_NEEDED       = 120     # bars fetched per signal computation (warmup + window)
MAX_SLIPPAGE      = 20      # max deviation in points (~2 pips for 5-digit broker)
MAX_SPREAD_FACTOR = 3.0     # skip entry if spread > 3× typical full spread

# Friday/weekend schedule (UTC)
FRIDAY_NO_ENTRY_HOUR = 14   # no new positions from Friday 14:00 UTC
FRIDAY_CLOSE_HOUR    = 20   # close all open positions at Friday 20:00 UTC
SUNDAY_REOPEN_HOUR   = 21   # markets re-open Sunday ~21:00 UTC

PAIRS_DEFAULT = ["EURUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD",
                 "GBPUSD", "USDJPY", "EURJPY"]

# USD direction when BUYing a pair: -1 = buying this pair sells USD; +1 = buys USD
USD_SIDE = {
    "EURUSD": -1, "AUDUSD": -1, "NZDUSD": -1, "GBPUSD": -1,
    "USDCHF": +1, "USDCAD": +1, "USDJPY": +1,
    "EURGBP":  0, "EURJPY":  0,
}

TF_MT5: dict[str, int] = {}   # populated after mt5 import in main()


# ── MT5 connection ────────────────────────────────────────────────────────────

def mt5_connect(login: int, password: str, server: str,
                mt5_path: Optional[str] = None) -> bool:
    kwargs: dict = {}
    if mt5_path:
        kwargs["path"] = mt5_path
    if not mt5.initialize(**kwargs):
        log.error("mt5.initialize() failed: %s", mt5.last_error())
        return False
    if not mt5.login(login, password=password, server=server):
        log.error("mt5.login() failed: %s", mt5.last_error())
        mt5.shutdown()
        return False
    info = mt5.account_info()
    log.info("Connected  login=%s  balance=%.2f  server=%s",
             info.login, info.balance, info.server)
    return True


# ── Bar data ──────────────────────────────────────────────────────────────────

def get_bars(symbol: str, tf: str, count: int) -> Optional[pd.DataFrame]:
    """
    Fetch the last `count` CLOSED bars.
    Requests count+1 from MT5 (position 0 = current forming bar) and
    drops the last row so callers always work with confirmed closes.
    """
    rates = mt5.copy_rates_from_pos(symbol, TF_MT5[tf], 0, count + 1)
    if rates is None or len(rates) < 2:
        log.warning("%s %s: got %s bars", symbol, tf,
                    len(rates) if rates is not None else "None")
        return None
    df = pd.DataFrame(rates)[["time", "open", "high", "low", "close"]]
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = (df.rename(columns={"open": "Open", "high": "High",
                              "low": "Low", "close": "Close"})
            .set_index("time")
            .sort_index()
            .iloc[:-1])   # drop current incomplete bar
    return df


# ── Signal ────────────────────────────────────────────────────────────────────

def compute_signal(df: pd.DataFrame, ema_period: int, window: int,
                   early_entry: bool = False) -> int:
    """
    Double-EMA of HL/2, then argmax/argmin over last `window` bars.
    Returns +1 (LONG after valley), -1 (SHORT after peak), 0 (no signal).

    early_entry=False: peak/valley must NOT be the current bar (confirmed signal).
    early_entry=True:  also fires when current bar IS the peak/valley (1 bar earlier).
    """
    if len(df) < ema_period * 2 + window + 5:
        return 0
    hl2   = (df["High"].values + df["Low"].values) / 2.0
    ema_v = _ema(_ema(hl2, ema_period), ema_period)
    dw    = ema_v[-window:]

    pos_max = int(np.argmax(dw))
    pos_min = int(np.argmin(dw))
    if pos_max == pos_min:
        return 0
    max_pos = window - 1 if early_entry else window - 2
    if pos_max > pos_min and pos_max <= max_pos:
        return -1   # peak recently → expect reversion down → SHORT
    if pos_min > pos_max and pos_min <= max_pos:
        return 1    # valley recently → expect reversion up → LONG
    return 0


def dema_distance_pips(df: pd.DataFrame, ema_period: int, symbol: str) -> float:
    """Distance in pips between last close and last DEMA value."""
    if len(df) < ema_period * 2 + 5:
        return 0.0
    hl2   = (df["High"].values + df["Low"].values) / 2.0
    ema_v = _ema(_ema(hl2, ema_period), ema_period)
    return abs(float(df["Close"].iloc[-1]) - float(ema_v[-1])) / pip_size(symbol)


# ── Symbol info cache ─────────────────────────────────────────────────────────

_sym_cache: dict[str, object] = {}

def sym_info(symbol: str):
    if symbol not in _sym_cache:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info({symbol}) failed")
        _sym_cache[symbol] = info
    return _sym_cache[symbol]

def pip_size(symbol: str) -> float:
    """pip = 10 × point  (0.0001 for 5-digit EURUSD, 0.01 for 3-digit USDJPY)."""
    return sym_info(symbol).point * 10


# ── Schedule helpers ──────────────────────────────────────────────────────────

def _secs_to_next_5m_bar() -> float:
    """Seconds remaining until the next 5m bar closes."""
    now = datetime.now(timezone.utc)
    secs_in_bar = (now.minute % 5) * 60 + now.second + now.microsecond / 1e6
    return 300.0 - secs_in_bar

def _is_friday_no_new_entry() -> bool:
    """True from Friday 14:00 UTC — no new positions until Monday."""
    now = datetime.now(timezone.utc)
    return now.weekday() == 4 and now.hour >= FRIDAY_NO_ENTRY_HOUR

def _is_weekend() -> bool:
    """True from Friday 20:00 UTC through Sunday 21:00 UTC (market closed)."""
    now = datetime.now(timezone.utc)
    wd = now.weekday()
    return (
        (wd == 4 and now.hour >= FRIDAY_CLOSE_HOUR) or  # Friday night
        wd == 5 or                                        # Saturday
        (wd == 6 and now.hour < SUNDAY_REOPEN_HOUR)      # Sunday before open
    )


# ── Orders ────────────────────────────────────────────────────────────────────

def open_order(symbol: str, direction: int, lots: float,
               tp_pips: float, sl_pips: float,
               comment: str) -> Optional[int]:
    """
    Place a market order.
    direction: +1 = BUY (entry at Ask), -1 = SELL (entry at Bid).
    TP and SL are given in pips from the entry price.
    Returns the position ticket on success, None on failure.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("%s: no tick — cannot open order", symbol)
        return None

    # Spread guard: skip entry if spread is abnormally wide (news, thin market)
    pip          = pip_size(symbol)
    spread_pips  = (tick.ask - tick.bid) / pip
    typical_full = 2.0 * HALF_SPREAD_PIPS.get(symbol, 0.5)
    max_spread   = MAX_SPREAD_FACTOR * typical_full
    if spread_pips > max_spread:
        log.warning("SKIP   %-8s  spread=%.2f pips > max=%.2f (%.1fx normal) — wide spread",
                    symbol, spread_pips, max_spread, spread_pips / typical_full)
        return None

    info   = sym_info(symbol)
    digits = info.digits
    tp_d   = round(tp_pips * pip, digits)
    sl_d   = round(sl_pips * pip, digits) if sl_pips > 0 else 0.0

    if direction == 1:       # BUY: enter at Ask, TP above, SL below
        price      = tick.ask
        tp         = round(price + tp_d, digits) if tp_d else 0.0
        sl         = round(price - sl_d, digits) if sl_d else 0.0
        order_type = mt5.ORDER_TYPE_BUY
    else:                    # SELL: enter at Bid, TP below, SL above
        price      = tick.bid
        tp         = round(price - tp_d, digits) if tp_d else 0.0
        sl         = round(price + sl_d, digits) if sl_d else 0.0
        order_type = mt5.ORDER_TYPE_SELL

    req: dict = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lots,
        "type":         order_type,
        "price":        price,
        "deviation":    MAX_SLIPPAGE,
        "magic":        MAGIC,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if tp:
        req["tp"] = tp
    if sl:
        req["sl"] = sl

    result = mt5.order_send(req)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else mt5.last_error()
        log.error("OPEN FAILED  %s  %s  %.2f lots  code=%s",
                  symbol, "BUY" if direction == 1 else "SELL", lots, code)
        return None

    actual_tp = tp if tp else 0.0
    log.info("OPEN   %-8s  %s  %.2f lots @ %.5f  TP=%.5f  ticket=%d",
             symbol, "BUY " if direction == 1 else "SELL",
             lots, result.price, actual_tp, result.order)
    return result.order


def close_order(ticket: int, symbol: str, direction: int,
                lots: float, reason: str) -> bool:
    """Close a position (or partial lots) by ticket via market order."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("%s: no tick — cannot close ticket=%d", symbol, ticket)
        return False

    if direction == 1:       # long → sell to close
        price      = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
    else:                    # short → buy to close
        price      = tick.ask
        order_type = mt5.ORDER_TYPE_BUY

    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lots,
        "type":         order_type,
        "position":     ticket,
        "price":        price,
        "deviation":    MAX_SLIPPAGE,
        "magic":        MAGIC,
        "comment":      f"close_{reason}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(req)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else mt5.last_error()
        log.error("CLOSE FAILED  ticket=%d  %s  code=%s", ticket, symbol, code)
        return False

    log.info("CLOSE  %-8s  %s  %.2f lots @ %.5f  reason=%-16s  ticket=%d",
             symbol, "LONG" if direction == 1 else "SHORT",
             lots, result.price, reason, ticket)
    return True


# ── Daily P&L ─────────────────────────────────────────────────────────────────

def portfolio_daily_pnl() -> float:
    """
    Realized P&L (closed deals today) + unrealized (open positions)
    for all orders placed by this bot (magic == MAGIC).
    """
    day_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    now = datetime.now(timezone.utc)

    realized = 0.0
    deals = mt5.history_deals_get(day_start, now)
    if deals:
        for d in deals:
            if d.magic == MAGIC and d.entry in (
                    mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT):
                realized += d.profit + d.commission + d.swap

    unrealized = 0.0
    positions = mt5.positions_get()
    if positions:
        unrealized = sum(p.profit for p in positions if p.magic == MAGIC)

    return realized + unrealized


# ── Per-pair state ────────────────────────────────────────────────────────────

class PairState:
    """Tracks bot-owned position state for one symbol."""
    __slots__ = ("symbol", "pos_1h", "pos_5m", "last_bar_1h", "last_bar_5m",
                 "bars_held_1h", "bars_held_5m", "partial_done_5m")

    def __init__(self, symbol: str):
        self.symbol      = symbol
        # each pos: {"ticket": int, "dir": ±1, "lots": float}
        self.pos_1h: Optional[dict] = None
        self.pos_5m: Optional[dict] = None
        self.last_bar_1h: Optional[pd.Timestamp] = None
        self.last_bar_5m: Optional[pd.Timestamp] = None
        self.bars_held_1h    = 0   # 1h bars elapsed since pos_1h opened
        self.bars_held_5m    = 0   # 5m bars elapsed since pos_5m opened
        self.partial_done_5m = False  # True once partial close fired on pos_5m


# ── Live trader ───────────────────────────────────────────────────────────────

class LiveTrader:
    def __init__(self, args):
        self.pairs        = args.pairs
        self.ema_1h       = args.ema_1h
        self.ema_5m       = args.ema_5m
        self.window       = args.window
        self.lots_1h      = args.lots_1h
        self.lots_5m      = args.lots_5m
        self.tp_1h        = args.tp_1h_pips
        self.tp_5m        = args.tp_5m_pips
        # SL: per-timeframe values; fall back to unified --sl-pips if not set
        self.sl_1h        = args.sl_pips_1h if args.sl_pips_1h > 0 else args.sl_pips
        self.sl_5m        = args.sl_pips_5m if args.sl_pips_5m > 0 else args.sl_pips
        self.daily_stop   = args.daily_stop
        self.min_bars_5m  = args.min_bars_5m
        self.min_bars_1h  = args.min_bars_1h
        self.partial_usd  = args.partial_usd
        self.max_dist_pips = args.max_dist_pips
        self.usd_filter    = args.usd_filter
        self.early_1h      = args.early_1h

        self.states    = {p: PairState(p) for p in self.pairs}
        self._halt_day = None   # date of daily-stop trigger

    # ── Halt / daily stop ─────────────────────────────────────────────────────

    def _halted(self) -> bool:
        return self._halt_day == datetime.now(timezone.utc).date()

    def _check_daily_stop(self) -> bool:
        if self.daily_stop <= 0:
            return False
        pnl = portfolio_daily_pnl()
        if pnl < -self.daily_stop:
            log.warning("DAILY STOP  pnl=%.2f < -%.0f  closing all positions",
                        pnl, self.daily_stop)
            self._close_all("daily_stop")
            self._halt_day = datetime.now(timezone.utc).date()
            return True
        return False

    def _close_all(self, reason: str) -> None:
        for st in self.states.values():
            for attr in ("pos_1h", "pos_5m"):
                pos = getattr(st, attr)
                if pos:
                    close_order(pos["ticket"], st.symbol,
                                pos["dir"], pos["lots"], reason)
                    setattr(st, attr, None)

    # ── State reconciliation ──────────────────────────────────────────────────

    def _sync(self) -> None:
        """
        Reconcile internal state with actual MT5 open positions.
        Called every poll cycle to detect broker-side TP hits.
        On startup also restores state from positions opened in a previous run.
        Keeps pos["lots"] current so partial closes are reflected correctly.
        """
        open_tickets: set[int] = set()
        positions = mt5.positions_get() or []
        for p in positions:
            if p.magic != MAGIC:
                continue
            open_tickets.add(p.ticket)
            if p.symbol not in self.states:
                continue
            st  = self.states[p.symbol]
            dir = 1 if p.type == mt5.POSITION_TYPE_BUY else -1
            cmt = p.comment or ""
            if "1h" in cmt:
                if st.pos_1h is None:
                    st.pos_1h = {"ticket": p.ticket, "dir": dir, "lots": p.volume}
                    log.info("Restored 1h  %-8s  ticket=%d  dir=%+d",
                             p.symbol, p.ticket, dir)
                else:
                    st.pos_1h["lots"] = p.volume   # keep lots current after any partial
            elif "5m" in cmt:
                if st.pos_5m is None:
                    st.pos_5m = {"ticket": p.ticket, "dir": dir, "lots": p.volume}
                    log.info("Restored 5m  %-8s  ticket=%d  dir=%+d",
                             p.symbol, p.ticket, dir)
                else:
                    st.pos_5m["lots"] = p.volume   # keep lots current after any partial

        # Clear state for positions no longer in MT5 (TP hit or manual close)
        for st in self.states.values():
            for attr in ("pos_1h", "pos_5m"):
                pos = getattr(st, attr)
                if pos and pos["ticket"] not in open_tickets:
                    log.info("%-8s  %s ticket=%d closed (TP/manual)",
                             st.symbol, attr, pos["ticket"])
                    setattr(st, attr, None)
                    if attr == "pos_1h":
                        st.bars_held_1h = 0
                    else:
                        st.bars_held_5m = 0
                        st.partial_done_5m = False

    # ── Partial close ─────────────────────────────────────────────────────────

    def _check_partial_close(self) -> None:
        """Close half the 5m position when its unrealized P&L >= partial_usd."""
        if self.partial_usd <= 0:
            return
        for st in self.states.values():
            if not st.pos_5m or st.partial_done_5m:
                continue
            mt5_pos = mt5.positions_get(ticket=st.pos_5m["ticket"])
            if not mt5_pos:
                continue
            profit = mt5_pos[0].profit
            if profit >= self.partial_usd:
                half = round(st.pos_5m["lots"] / 2, 2)
                log.info("PARTIAL  %-8s  5m  profit=%.2f >= %.2f  closing %.2f lots",
                         st.symbol, profit, self.partial_usd, half)
                if close_order(st.pos_5m["ticket"], st.symbol,
                               st.pos_5m["dir"], half, "partial_close"):
                    st.pos_5m["lots"] = round(st.pos_5m["lots"] - half, 2)
                    st.partial_done_5m = True

    # ── USD correlation filter ────────────────────────────────────────────────

    def _net_usd_exposure(self) -> int:
        """Net USD direction across all open positions (>0 = net long USD, <0 = net short)."""
        net = 0
        for st in self.states.values():
            usd = USD_SIDE.get(st.symbol, 0)
            if st.pos_1h:
                net += st.pos_1h["dir"] * usd
            if st.pos_5m:
                net += st.pos_5m["dir"] * usd
        return net

    def _usd_allowed(self, symbol: str, direction: int) -> bool:
        """Return False if opening this position would put us on both sides of USD."""
        if not self.usd_filter:
            return True
        net = self._net_usd_exposure()
        if net == 0:
            return True
        new_usd = direction * USD_SIDE.get(symbol, 0)
        if new_usd == 0:
            return True  # non-USD pair or neutral
        return (net > 0) == (new_usd > 0)

    # ── Signal handling ───────────────────────────────────────────────────────

    def _handle_1h(self, st: PairState, sig: int, bars: pd.DataFrame) -> None:
        if sig == 0:
            return
        if st.pos_1h and sig == st.pos_1h["dir"]:
            return  # already positioned in this direction

        # Close existing 1h position on direction flip
        if st.pos_1h:
            if st.bars_held_1h < self.min_bars_1h:
                log.debug("%-8s  1h exit blocked (bars_held=%d < min=%d)",
                          st.symbol, st.bars_held_1h, self.min_bars_1h)
                return
            log.info("%-8s  1h flip  %+d → %+d", st.symbol, st.pos_1h["dir"], sig)
            close_order(st.pos_1h["ticket"], st.symbol,
                        st.pos_1h["dir"], st.pos_1h["lots"], "flip_1h")
            st.pos_1h = None
            st.bars_held_1h = 0

            # 1h flip also clears a 5m position that was in the OLD 1h direction
            if st.pos_5m and st.pos_5m["dir"] == -sig:
                close_order(st.pos_5m["ticket"], st.symbol,
                            st.pos_5m["dir"], st.pos_5m["lots"], "flip_1h_clear")
                st.pos_5m = None
                st.bars_held_5m = 0
                st.partial_done_5m = False

        # Distance filter: skip entry if close is too far from DEMA
        if self.max_dist_pips > 0:
            dist = dema_distance_pips(bars, self.ema_1h, st.symbol)
            if dist > self.max_dist_pips:
                log.info("%-8s  1h entry skipped — dist=%.1f pips > max=%.1f",
                         st.symbol, dist, self.max_dist_pips)
                return

        if not self._usd_allowed(st.symbol, sig):
            log.info("%-8s  1h entry blocked — USD filter  net_usd=%+d",
                     st.symbol, self._net_usd_exposure())
            return

        if _is_friday_no_new_entry():
            log.info("%-8s  1h entry skipped — Friday cutoff (≥%02d:00 UTC)",
                     st.symbol, FRIDAY_NO_ENTRY_HOUR)
            return

        ticket = open_order(st.symbol, sig, self.lots_1h,
                            self.tp_1h, self.sl_1h, f"ema_1h_{'+' if sig==1 else '-'}")
        if ticket:
            st.pos_1h = {"ticket": ticket, "dir": sig, "lots": self.lots_1h}
            st.bars_held_1h = 0

    def _handle_5m(self, st: PairState, sig: int, bars: pd.DataFrame) -> None:
        if sig == 0:
            return

        # Block: 5m signal opposing active 1h direction
        if st.pos_1h and sig == -st.pos_1h["dir"]:
            log.debug("%-8s  5m %+d blocked by 1h %+d", st.symbol, sig, st.pos_1h["dir"])
            return

        if st.pos_5m:
            if sig == st.pos_5m["dir"]:
                return  # already in this direction
            # min_bars guard: don't exit on a whipsaw signal too soon after entry
            if st.bars_held_5m < self.min_bars_5m:
                log.debug("%-8s  5m exit blocked (bars_held=%d < min=%d)",
                          st.symbol, st.bars_held_5m, self.min_bars_5m)
                return
            # 5m direction flip — close and reopen (unless blocked by 1h)
            close_order(st.pos_5m["ticket"], st.symbol,
                        st.pos_5m["dir"], st.pos_5m["lots"], "flip_5m")
            st.pos_5m = None
            st.bars_held_5m = 0
            st.partial_done_5m = False

        # Distance filter: skip entry if close is too far from DEMA
        if self.max_dist_pips > 0:
            dist = dema_distance_pips(bars, self.ema_5m, st.symbol)
            if dist > self.max_dist_pips:
                log.debug("%-8s  5m entry skipped — dist=%.1f pips > max=%.1f",
                          st.symbol, dist, self.max_dist_pips)
                return

        if not self._usd_allowed(st.symbol, sig):
            log.debug("%-8s  5m entry blocked — USD filter  net_usd=%+d",
                      st.symbol, self._net_usd_exposure())
            return

        if _is_friday_no_new_entry():
            log.debug("%-8s  5m entry skipped — Friday cutoff (≥%02d:00 UTC)",
                      st.symbol, FRIDAY_NO_ENTRY_HOUR)
            return

        ticket = open_order(st.symbol, sig, self.lots_5m,
                            self.tp_5m, self.sl_5m, f"ema_5m_{'+' if sig==1 else '-'}")
        if ticket:
            st.pos_5m = {"ticket": ticket, "dir": sig, "lots": self.lots_5m}
            st.bars_held_5m = 0
            st.partial_done_5m = False

    # ── Per-pair processing ───────────────────────────────────────────────────

    def _process_pair(self, st: PairState) -> None:
        # 1h: check for new closed bar
        bars_1h = get_bars(st.symbol, "1h", BARS_NEEDED)
        if bars_1h is not None and not bars_1h.empty:
            bar_ts = bars_1h.index[-1]
            if bar_ts != st.last_bar_1h:
                st.last_bar_1h = bar_ts
                if st.pos_1h:
                    st.bars_held_1h += 1
                sig = compute_signal(bars_1h, self.ema_1h, self.window,
                                     early_entry=self.early_1h)
                log.debug("%-8s  1h bar %s  sig=%+d  bars_held=%d",
                          st.symbol, bar_ts, sig, st.bars_held_1h)
                self._handle_1h(st, sig, bars_1h)

        # 5m: check for new closed bar
        bars_5m = get_bars(st.symbol, "5m", BARS_NEEDED)
        if bars_5m is not None and not bars_5m.empty:
            bar_ts = bars_5m.index[-1]
            if bar_ts != st.last_bar_5m:
                st.last_bar_5m = bar_ts
                if st.pos_5m:
                    st.bars_held_5m += 1
                sig = compute_signal(bars_5m, self.ema_5m, self.window)
                log.debug("%-8s  5m bar %s  sig=%+d  bars_held=%d",
                          st.symbol, bar_ts, sig, st.bars_held_5m)
                self._handle_5m(st, sig, bars_5m)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info(
            "Starting  pairs=%s  ema=%d/%d  w=%d  "
            "lots=%.1f/%.1f  tp=%.0f/%.0f pips  "
            "sl_1h=%.0f sl_5m=%.0f pips  stop=$%.0f  "
            "min_bars=%d/%d  partial=$%.0f  max_dist=%.1f pips  usd_filter=%s",
            self.pairs, self.ema_1h, self.ema_5m, self.window,
            self.lots_1h, self.lots_5m,
            self.tp_1h, self.tp_5m,
            self.sl_1h, self.sl_5m, self.daily_stop,
            self.min_bars_5m, self.min_bars_1h,
            self.partial_usd, self.max_dist_pips,
            "on" if self.usd_filter else "off",
        )

        self._sync()   # restore any positions from a previous run

        while True:
            try:
                # Weekend: close any open positions and sleep until market re-opens
                if _is_weekend():
                    has_open = any(st.pos_1h or st.pos_5m
                                   for st in self.states.values())
                    if has_open:
                        log.warning("Weekend close  %s — closing all positions",
                                    datetime.now(timezone.utc).strftime("%A %H:%M UTC"))
                        self._close_all("weekend")
                    time.sleep(300)   # check every 5 min over weekend
                    continue

                if self._halted():
                    time.sleep(60)
                    continue

                if self._check_daily_stop():
                    continue

                self._sync()              # detect TP hits; reconcile with broker state
                self._check_partial_close()  # close half 5m pos when profitable

                for st in self.states.values():
                    self._process_pair(st)

            except KeyboardInterrupt:
                log.info("Keyboard interrupt")
                break
            except Exception:
                log.exception("Unhandled error in main loop")

            # Smart sleep: tight polling near 5m bar boundaries for fast detection
            stt = _secs_to_next_5m_bar()
            time.sleep(POLL_SECS_NEAR if stt < NEAR_BAR_SECS
                       else min(stt - NEAR_BAR_SECS, POLL_SECS))

        log.info("Shutdown — closing all open positions")
        self._close_all("shutdown")
        mt5.shutdown()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Live trader: combined 1h+5m double-EMA strategy",
    )
    ap.add_argument("--login",        type=int,   required=True)
    ap.add_argument("--password",     default="",
                    help="MT5 password (or set MT5_PASSWORD env var)")
    ap.add_argument("--server",       required=True)
    ap.add_argument("--mt5-path",     default=None,
                    help="Path to terminal64.exe (auto-detected if omitted)")
    ap.add_argument("--pairs",        nargs="+",  default=PAIRS_DEFAULT)
    ap.add_argument("--ema-1h",       type=int,   default=9)
    ap.add_argument("--ema-5m",       type=int,   default=20)
    ap.add_argument("--window",       type=int,   default=20)
    ap.add_argument("--lots-1h",      type=float, default=0.5,
                    help="Lots per 1h entry (start here, scale after live validation)")
    ap.add_argument("--lots-5m",      type=float, default=0.25,
                    help="Lots per 5m entry")
    ap.add_argument("--tp-1h-pips",   type=float, default=10.0,
                    help="Take-profit in pips for 1h positions (~90%% hit rate)")
    ap.add_argument("--tp-5m-pips",   type=float, default=3.0,
                    help="Take-profit in pips for 5m positions (~85%% hit rate)")
    ap.add_argument("--sl-pips",      type=float, default=0.0,
                    help="Hard stop-loss in pips for all positions (0 = use per-TF values)")
    ap.add_argument("--sl-pips-1h",   type=float, default=40.0,
                    help="Hard SL for 1h positions in pips (recommended: 40; 0=disabled)")
    ap.add_argument("--sl-pips-5m",   type=float, default=0.0,
                    help="Hard SL for 5m positions in pips (0=disabled; close_profit handles exits)")
    ap.add_argument("--max-dist-pips", type=float, default=0.0,
                    help="Skip entry if close > N pips from DEMA (0=disabled; backtest shows 10+ hurts)")
    ap.add_argument("--daily-stop",   type=float, default=1000.0,
                    help="Portfolio daily loss limit USD — closes all and halts (0=off)")
    ap.add_argument("--min-bars-5m",  type=int,   default=3,
                    help="Min 5m bars held before an opposite signal can exit the position")
    ap.add_argument("--min-bars-1h",  type=int,   default=2,
                    help="Min 1h bars held before an opposite signal can exit the position")
    ap.add_argument("--partial-usd",  type=float, default=0.0,
                    help="Close half of 5m position when unrealized P&L >= USD (0=off)")
    ap.add_argument("--usd-filter",   action="store_true", default=False,
                    help="Block new entries that would oppose net USD exposure across pairs")
    ap.add_argument("--early-1h",     action="store_true", default=True,
                    help="Fire 1h signal on the peak/valley bar itself (1 bar earlier; default ON)")
    ap.add_argument("--no-early-1h",  action="store_false", dest="early_1h",
                    help="Disable early-1h (revert to bar-after-peak confirmation)")
    args = ap.parse_args()

    # Resolve password from env if not provided
    import os
    if not args.password:
        args.password = os.environ.get("MT5_PASSWORD", "")

    # Populate timeframe constants after mt5 is imported
    TF_MT5["1h"] = mt5.TIMEFRAME_H1
    TF_MT5["5m"] = mt5.TIMEFRAME_M5

    if not mt5_connect(args.login, args.password, args.server, args.mt5_path):
        sys.exit(1)

    LiveTrader(args).run()


if __name__ == "__main__":
    main()
