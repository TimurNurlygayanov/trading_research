"""
Live trading bot for Strategy 1 (range fade) — MetaTrader 5 / FTMO integration.

Mirrors the semantics of `scripts/strategy1_live_ib.py` 1:1 — same range/ATR
evaluation, same risk caps, same overnight rules — but uses the MT5 Python
package to talk to a MetaTrader 5 terminal instead of IB. Designed for FTMO
prop-firm challenge / funded accounts (also works with any broker that
provides MT5 API access).

Sizing (pass exactly one; default = --risk-usd 100, --max-lot 1.0)
  --lot 1.0           : fixed lot size
  --risk-usd 100      : risk-based — lots chosen so SL distance × lots = $100
                        (uses broker tick_value, so account currency is handled).

Risk caps
  --max-lot 1.0             : hard cap on position size in lots (default 1.0 per
                              user request — "1 lot max").
  --daily-dd-limit-pair 150 : per-pair daily loss limit (USD account-ccy units).
                              Once today's realised PnL on a pair drops below
                              -limit, block new entries on that pair until the
                              next UTC day. Resets at 00:00 UTC.

Overnight / weekend rules (UTC hours; --no-overnight applies sensible defaults)
  --no-overnight              : close-by-hour=21, no-entry-after-hour=20,
                                no-friday-after-hour=17
  --close-by-hour H           : force-close ALL open positions when bar.hour ≥ H
  --no-entry-after-hour H     : block new entries when bar.hour ≥ H
  --no-friday-after-hour H    : on Friday only, block new entries when bar.hour ≥ H

Spread guard (filters wide bid-ask spreads — typically news-minute spikes)
  --spread-guard
  --max-spread-pips EURUSD=0.6 NZDUSD=2.5    (or single global value)

Slippage guard (filters trades where the market has moved away from the signal's
intended entry price between bar close and our submission — i.e. live ask for a
BUY is far above the signal close, or live bid for a SELL is far below it).
This catches both wide-spread *and* fast-market situations in a single check.
  --slippage-guard
  --max-slippage-pips EURUSD=0.5 NZDUSD=2.0  (or single global value)

FTMO / broker symbol naming
  Most FTMO MT5 accounts expose pairs as plain "EURUSD". Some brokers add a
  suffix like ".raw", ".pro", ".m". Use --symbol-suffix to add one globally:
    --symbol-suffix .raw    →  EURUSD.raw, AUDUSD.raw, …
  Or pass per-pair mappings via --symbol-map EURUSD=EURUSDx NZDUSD=NZDUSDx

Setup
  pip install MetaTrader5
  Install MetaTrader 5 terminal and log into your FTMO account.
  Tools → Options → Expert Advisors → "Allow algorithmic trading".
  Start MT5 terminal, then run this script.

Usage
  # Recommended live config (matches the validated backtest)
  python -m scripts.strategy1_live_mt5 \\
      --pairs EURUSD AUDUSD NZDUSD USDCHF USDCAD \\
      --risk-usd 100 --max-lot 1.0 --daily-dd-limit-pair 150 \\
      --no-overnight --spread-guard --slippage-guard

  # Dry-run (logs signals + computed orders, never calls order_send)
  python -m scripts.strategy1_live_mt5 --dry-run --no-overnight --spread-guard --slippage-guard

  # Connect to a non-default MT5 install / specific account
  python -m scripts.strategy1_live_mt5 \\
      --mt5-path "C:/Program Files/MetaTrader 5/terminal64.exe" \\
      --login 12345678 --password "xxx" --server "FTMO-Demo"

For offline replay / signal-logic sanity check, use the IB module's --simulate
mode (works without any broker connection):
  python -m scripts.strategy1_live_ib --simulate --pairs EURUSD AUDUSD NZDUSD USDCHF USDCAD
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the broker-agnostic strategy logic & helpers from the IB module.
# (evaluate_entry imports `_atr` from strategy1; no IB-specific deps at module load.)
import strategy1_live_ib as _ib_mod
from strategy1_live_ib import (
    evaluate_entry,
    _resolve_overnight,
    _resolve_max_spreads,
    _pip_size,
    _is_usd_quote,
    DEFAULT_MAX_SPREAD_PIPS,
    FADE,
)
from strategy1 import _atr  # ATR helper reused for pre-computing intra-bar levels

CONFIG_PATH = Path(__file__).parent / "strategy1_config.json"

# Default pair allow-list — the validated 5 from TRADING_STRATEGY.md (no GBPUSD).
DEFAULT_PAIRS = ["EURUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD"]

# Magic number tags our trades in the MT5 deal history.
MAGIC_NUMBER = 20240501

# Per-pair slippage caps (pips) when --slippage-guard is on but no explicit
# --max-slippage-pips was passed. Tighter than the spread caps because they
# bound how far the live fillable price has drifted from the signal's intended
# entry — a more meaningful "do we still want this trade?" measure than raw
# bid-ask width. Roughly 3× the typical half-spread per pair.
DEFAULT_MAX_SLIPPAGE_PIPS = {
    "EURUSD": 0.5,
    "GBPUSD": 1.0,
    "AUDUSD": 0.8,
    "NZDUSD": 2.0,
    "USDCHF": 1.5,
    "USDCAD": 1.5,
    "USDJPY": 0.8,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("strategy1_live_mt5")


# ── MT5 helpers ───────────────────────────────────────────────────────────────

def _resolve_max_slippages(specs: list[str] | None, pairs: list[str]) -> dict[str, float]:
    """
    Build {pair: max_slippage_pips}. Same parsing rules as --max-spread-pips:
    a single global value or a list of pair=value overrides. Pairs not specified
    fall back to DEFAULT_MAX_SLIPPAGE_PIPS.
    """
    out = {sym: DEFAULT_MAX_SLIPPAGE_PIPS.get(sym, 1.5) for sym in pairs}
    if not specs:
        return out
    if len(specs) == 1 and "=" not in specs[0]:
        try:
            global_lim = float(specs[0])
            return {sym: global_lim for sym in pairs}
        except ValueError:
            pass
    for s in specs:
        if "=" not in s:
            log.warning("Ignoring malformed --max-slippage-pips entry: %r", s)
            continue
        sym, val = s.split("=", 1)
        try:
            out[sym.upper()] = float(val)
        except ValueError:
            log.warning("Ignoring non-numeric value: %r", s)
    return out


def _format_stats(s: dict) -> str:
    pw      = s.get("signals_price_wait", 0)
    pw_exp  = s.get("signals_price_wait_expired", 0)
    pw_part = f"  price_wait={pw}  price_wait_exp={pw_exp}" if pw or pw_exp else ""
    return (f"signals={s['signals_total']}  "
            f"placed={s['trades_placed']}  "
            f"skip_inpos={s['signals_skipped_inpos']}  "
            f"skip_session={s.get('signals_skipped_session', 0)}  "
            f"skip_dailydd={s.get('signals_skipped_dailydd', 0)}  "
            f"skip_spread={s['signals_skipped_spread']}  "
            f"skip_slippage={s.get('signals_skipped_slippage', 0)}  "
            f"skip_nodata={s['signals_skipped_nodata']}  "
            f"force_closed={s.get('force_closes', 0)}"
            + pw_part)


def _resolve_symbol_map(args, pairs: list[str]) -> dict[str, str]:
    """Build {logical_pair → broker_symbol}. CLI mappings override suffix."""
    suffix = args.symbol_suffix or ""
    out = {sym: sym + suffix for sym in pairs}
    for spec in (args.symbol_map or []):
        if "=" not in spec:
            log.warning("Ignoring malformed --symbol-map entry: %r", spec)
            continue
        k, v = spec.split("=", 1)
        out[k.upper()] = v
    return out


def _detect_broker_offset(mt5, sym_mt5: str) -> int:
    """Broker server timezone offset (hours) — broker_time minus UTC.

    Many MT5 brokers run on EET (GMT+2 / +3 with DST). Bar timestamps from
    copy_rates_from_pos are in broker time, so we subtract this offset to get
    UTC for the strategy logic. Rounded to whole hours.
    """
    tick = mt5.symbol_info_tick(sym_mt5)
    if tick is None or not tick.time:
        log.warning("Cannot detect broker offset — no tick for %s; assuming 0 (UTC)", sym_mt5)
        return 0
    # tick.time is broker server time as a unix-style int. Treating it as UTC
    # gives us a wall-clock comparison against now_utc.
    broker_wall = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    offset_hours = round((broker_wall - now_utc).total_seconds() / 3600)
    log.info("Broker timezone offset detected: %+d hours (broker − UTC)", offset_hours)
    return offset_hours


def _resolve_filling_mode(mt5, info) -> int:
    """Pick the first filling mode the symbol supports (FTMO requires IOC)."""
    fm = getattr(info, "filling_mode", 0) or 0
    # Bitmask: 1 = FOK, 2 = IOC, 4 = RETURN
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def _round_to_step(value: float, step: float) -> float:
    """Round value DOWN to the nearest multiple of step."""
    if step <= 0:
        return value
    return math.floor(value / step + 1e-9) * step


def _round_price(price: float, point: float, digits: int) -> float:
    """Snap a price to the symbol's tick grid."""
    if point > 0:
        price = round(price / point) * point
    return round(price, digits)


# ── MT5 trader ────────────────────────────────────────────────────────────────

class MT5Trader:
    """Wraps MetaTrader5 Python API for the range-fade strategy."""

    def __init__(self, mt5_module, symbol_map: dict[str, str], dry_run: bool,
                 daily_dd_limit: float | None, broker_offset_hours: int):
        self.mt5 = mt5_module
        self.symbol_map = symbol_map  # logical → broker symbol
        self.dry_run = dry_run
        self.daily_dd_limit = daily_dd_limit
        self.broker_offset_hours = broker_offset_hours
        self._last_bar_time: dict[str, datetime] = {}  # logical → UTC bar time
        self._symbol_info: dict[str, object] = {}
        # sym → (ticket, placed_monotonic_time) for pending limit orders
        self._pending_orders: dict[str, tuple[int, float]] = {}
        self.stats = {
            "signals_total":               0,
            "signals_skipped_inpos":       0,
            "signals_skipped_spread":      0,
            "signals_skipped_slippage":    0,
            "signals_skipped_nodata":      0,
            "signals_skipped_dailydd":     0,
            "signals_skipped_session":     0,
            "trades_placed":               0,
            "force_closes":                0,
            "signals_price_wait":          0,
            "signals_price_wait_expired":  0,
        }

    # — connection lifecycle —

    def connect(self, mt5_path: str | None, login: int | None,
                password: str | None, server: str | None) -> None:
        kwargs = {}
        if mt5_path:
            kwargs["path"] = mt5_path
        if not self.mt5.initialize(**kwargs):
            raise RuntimeError(f"mt5.initialize failed: {self.mt5.last_error()}")
        if login is not None:
            ok = self.mt5.login(login=int(login), password=password or "", server=server or "")
            if not ok:
                raise RuntimeError(f"mt5.login failed: {self.mt5.last_error()}")
        acct = self.mt5.account_info()
        term = self.mt5.terminal_info()
        if acct is None:
            raise RuntimeError("mt5.account_info() returned None — not logged in?")
        log.info("MT5 connected. account=%s server=%s currency=%s balance=%.2f trade_allowed=%s",
                 acct.login, acct.server, acct.currency, acct.balance,
                 getattr(term, "trade_allowed", "?"))

    def shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    # — symbol housekeeping —

    def prime_symbols(self, logical_pairs: list[str],
                      warmup_required: int) -> dict[str, pd.DataFrame]:
        """Select each pair in MarketWatch, verify the terminal has at least
        `warmup_required` 1m bars cached, and return the fetched bars
        (closed history + the partial current bar).

        Retries briefly so cold-cache symbols get a chance to pull history
        from the broker on first connect.
        """
        out: dict[str, pd.DataFrame] = {}
        for sym in logical_pairs:
            sym_mt5 = self.symbol_map[sym]
            if not self.mt5.symbol_select(sym_mt5, True):
                raise RuntimeError(f"Cannot select symbol {sym_mt5} in MarketWatch — "
                                   f"check the symbol name (try --symbol-suffix or "
                                   f"--symbol-map)")
            info = self.mt5.symbol_info(sym_mt5)
            if info is None:
                raise RuntimeError(f"symbol_info({sym_mt5}) returned None")
            self._symbol_info[sym] = info

            # Prime + verify history. Up to ~5 s budget per pair (10 × 0.5 s).
            df = None
            for _ in range(10):
                df = self.fetch_bars(sym, 300)
                if df is not None and len(df) >= warmup_required + 1:
                    break
                time.sleep(0.5)
            if df is None or len(df) < warmup_required + 1:
                n = 0 if df is None else len(df)
                raise RuntimeError(
                    f"{sym} ({sym_mt5}): terminal returned only {n} 1m bars "
                    f"(need ≥ {warmup_required + 1}). Open the M1 chart for "
                    f"{sym_mt5} in MT5 to load history, then retry.")
            out[sym] = df
            log.info("Symbol %s → %s  bars_avail=%d  (point=%.5f digits=%d "
                     "step=%.2f min=%.2f max=%.2f tick_val=%.4f)",
                     sym, sym_mt5, len(df), info.point, info.digits,
                     info.volume_step, info.volume_min, info.volume_max,
                     info.trade_tick_value)
        return out

    # — bars —

    def fetch_bars(self, sym: str, count: int) -> pd.DataFrame | None:
        """Return last `count` 1m bars (incl. partial current bar) with UTC index."""
        sym_mt5 = self.symbol_map[sym]
        rates = self.mt5.copy_rates_from_pos(sym_mt5, self.mt5.TIMEFRAME_M1, 0, count)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        broker_dt = pd.to_datetime(df["time"], unit="s", utc=True)
        utc_dt = broker_dt - pd.Timedelta(hours=self.broker_offset_hours)
        df.index = utc_dt
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "tick_volume": "Volume",
        })
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def latest_tick(self, sym: str) -> object | None:
        """Latest cached tick (non-blocking memory read). Returns the raw tick
        struct (with .bid, .ask, .time, .time_msc fields) or None.
        """
        return self.mt5.symbol_info_tick(self.symbol_map[sym])

    # — quotes / spread —

    def get_bid_ask(self, sym: str) -> tuple[float, float] | None:
        """Live (bid, ask) tuple, or None if no valid quote."""
        sym_mt5 = self.symbol_map[sym]
        tick = self.mt5.symbol_info_tick(sym_mt5)
        if tick is None or not (tick.bid > 0 and tick.ask > 0 and tick.ask >= tick.bid):
            return None
        return float(tick.bid), float(tick.ask)

    def current_spread_pips(self, sym: str) -> float | None:
        ba = self.get_bid_ask(sym)
        if ba is None:
            return None
        bid, ask = ba
        return (ask - bid) / _pip_size(sym)

    def entry_slippage_pips(self, sym: str, sig: dict) -> tuple[float, float] | None:
        """
        Distance from signal entry price to the live fillable price, in pips,
        plus the live fillable price itself. Returns (slippage_pips, fillable).
        Positive slippage = market moved AGAINST us (worse fill than signal close);
        negative = market moved IN our favor.
        For BUY we use ask; for SELL we use bid.
        """
        ba = self.get_bid_ask(sym)
        if ba is None:
            return None
        bid, ask = ba
        if sig["action"] == "BUY":
            fillable = ask
            slip = ask - sig["entry_price"]
        else:
            fillable = bid
            slip = sig["entry_price"] - bid
        return slip / _pip_size(sym), fillable

    # — positions / orders —

    def has_open_position(self, sym: str) -> bool:
        sym_mt5 = self.symbol_map[sym]
        pos = self.mt5.positions_get(symbol=sym_mt5)
        if pos:
            return True
        # Pending orders count too (in case a deferred order is sitting)
        pend = self.mt5.orders_get(symbol=sym_mt5)
        return bool(pend)

    def daily_pnl(self, sym: str) -> float:
        """Realised PnL today (UTC) for this symbol, summed over all closed deals."""
        sym_mt5 = self.symbol_map[sym]
        now_utc = datetime.now(timezone.utc)
        midnight_utc = datetime.combine(now_utc.date(), datetime.min.time(),
                                        tzinfo=timezone.utc)
        # mt5.history_deals_get accepts datetime in broker time; convert from UTC.
        from_dt = midnight_utc + timedelta(hours=self.broker_offset_hours)
        to_dt   = now_utc       + timedelta(hours=self.broker_offset_hours)
        deals = self.mt5.history_deals_get(from_dt, to_dt, group=sym_mt5)
        if not deals:
            return 0.0
        total = 0.0
        for d in deals:
            total += float(d.profit) + float(d.swap) + float(d.commission)
        return total

    def is_daily_blocked(self, sym: str) -> bool:
        if self.daily_dd_limit is None:
            return False
        return self.daily_pnl(sym) <= -self.daily_dd_limit

    # — sizing —

    def size_lots(self, sym: str, sig: dict, args) -> tuple[float, bool]:
        """Resolve volume from --lot or --risk-usd, capped by --max-lot.
        Returns (lots, was_capped). MT5 tick_value/tick_size handle account-ccy
        conversion automatically.
        """
        info = self._symbol_info[sym]
        cap_lots = args.max_lot if args.max_lot is not None else 5.0

        if args.risk_usd is None:
            raw_lots = float(args.lot)
        else:
            sl_distance = abs(sig["entry_price"] - sig["sl_price"])
            if sl_distance <= 0 or info.trade_tick_size <= 0:
                return 0.0, False
            ticks = sl_distance / info.trade_tick_size
            loss_per_lot = ticks * info.trade_tick_value
            if loss_per_lot <= 0:
                return 0.0, False
            raw_lots = float(args.risk_usd) / loss_per_lot

        was_capped = raw_lots > cap_lots
        lots = min(raw_lots, cap_lots)
        lots = _round_to_step(lots, info.volume_step)
        lots = max(info.volume_min, min(info.volume_max, lots))
        return lots, was_capped

    # — order placement —

    def submit_market(self, sym: str, sig: dict, lots: float) -> bool:
        info = self._symbol_info[sym]
        sym_mt5 = self.symbol_map[sym]
        action_buy = (sig["action"] == "BUY")
        order_type = self.mt5.ORDER_TYPE_BUY if action_buy else self.mt5.ORDER_TYPE_SELL

        tick = self.mt5.symbol_info_tick(sym_mt5)
        if tick is None:
            log.warning("SKIP %s: no tick available for entry price", sym)
            return False
        entry_px = tick.ask if action_buy else tick.bid

        tp = _round_price(sig["tp_price"], info.point, info.digits)
        sl = _round_price(sig["sl_price"], info.point, info.digits)

        # Validate SL/TP against the broker's minimum stop level (retcode 10016).
        # MT5 checks SL/TP vs the CLOSING side of the spread, not the fill side:
        #   BUY position closes via SELL at bid  → SL must be < bid - min_dist
        #                                          TP must be > bid + min_dist
        #   SELL position closes via BUY  at ask  → SL must be > ask + min_dist
        #                                           TP must be < ask - min_dist
        # Using entry_px (ask for BUY, bid for SELL) misses the spread on the
        # stop side — a narrow SL on SELL can be below ask even when above bid.
        stops = int(getattr(info, "trade_stops_level", 0) or 0)
        min_dist = max(stops, 1) * info.point
        if action_buy:
            if sl >= tick.bid - min_dist:
                log.warning("SKIP %s BUY: SL %.5f ≥ bid %.5f - min_dist %.5f "
                            "(stops_level=%d pts)",
                            sym, sl, tick.bid, min_dist, stops)
                return False
            if tp <= tick.bid + min_dist:
                log.warning("SKIP %s BUY: TP %.5f ≤ bid %.5f + min_dist %.5f "
                            "(stops_level=%d pts)",
                            sym, tp, tick.bid, min_dist, stops)
                return False
        else:
            if sl <= tick.ask + min_dist:
                log.warning("SKIP %s SELL: SL %.5f ≤ ask %.5f + min_dist %.5f "
                            "(stops_level=%d pts)",
                            sym, sl, tick.ask, min_dist, stops)
                return False
            if tp >= tick.ask - min_dist:
                log.warning("SKIP %s SELL: TP %.5f ≥ ask %.5f - min_dist %.5f "
                            "(stops_level=%d pts)",
                            sym, tp, tick.ask, min_dist, stops)
                return False

        if self.dry_run:
            log.info("[DRY-RUN] %s %s lots=%.2f  entry≈%.5f  TP=%.5f  SL=%.5f",
                     sym, sig["action"], lots, entry_px, tp, sl)
            return True

        request = {
            "action":       self.mt5.TRADE_ACTION_DEAL,
            "symbol":       sym_mt5,
            "volume":       float(lots),
            "type":         order_type,
            "price":        entry_px,
            "sl":           sl,
            "tp":           tp,
            "deviation":    20,  # max slippage in points (0.0002 for 5-digit FX)
            "magic":        MAGIC_NUMBER,
            "comment":      "strategy1_fade",
            "type_time":    self.mt5.ORDER_TIME_GTC,
            "type_filling": _resolve_filling_mode(self.mt5, info),
        }
        result = self.mt5.order_send(request)
        if result is None:
            log.error("ORDER FAILED %s: order_send returned None  err=%s",
                      sym, self.mt5.last_error())
            return False
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            log.error("ORDER REJECT %s  retcode=%d  comment=%s",
                      sym, result.retcode, result.comment)
            return False
        log.info("PLACED %s %s lots=%.2f  fill=%.5f  TP=%.5f  SL=%.5f  ticket=%d",
                 sym, sig["action"], lots, result.price, tp, sl, result.order)
        return True

    def submit_pending(self, sym: str, sig: dict, lots: float,
                       expiry_minutes: int) -> bool:
        """Place a pending order at the midpoint of TP and SL (1:1 R:R, 1.5×ATR each side).

        Order type is chosen by comparing limit_px to current market price:
          fade  BUY  → midpoint below current price → BUY  LIMIT
          fade  SELL → midpoint above current price → SELL LIMIT
          break BUY  → midpoint above current price → BUY  STOP
          break SELL → midpoint below current price → SELL STOP
        """
        info = self._symbol_info[sym]
        sym_mt5 = self.symbol_map[sym]
        action_buy = (sig["action"] == "BUY")

        # Midpoint of TP and SL → 1:1 R:R from fill price regardless of mode.
        limit_px = _round_price((sig["tp_price"] + sig["sl_price"]) / 2,
                                info.point, info.digits)
        tp = _round_price(sig["tp_price"], info.point, info.digits)
        sl = _round_price(sig["sl_price"], info.point, info.digits)

        # Need current market to classify and validate the pending order.
        tick = self.mt5.symbol_info_tick(sym_mt5)
        if tick is None:
            log.warning("SKIP PENDING %s: no live tick for price validation", sym)
            return False

        # If the market has already reached or passed the pending price, enter
        # at market immediately — we'll get an equal or better fill than waiting.
        #   BUY:  ask already ≤ limit_px → price came down to/past our entry
        #   SELL: bid already ≥ limit_px → price came up to/past our entry
        # This also covers breakout stops: if price already broke through the
        # stop level (ask ≥ limit for BUY STOP, bid ≤ limit for SELL STOP),
        # the pending would fill instantly anyway — skip the round-trip.
        if action_buy and tick.ask <= limit_px:
            log.info("MARKET FALLBACK %s BUY: ask %.5f ≤ limit %.5f",
                     sym, tick.ask, limit_px)
            return self.submit_market(sym, sig, lots)
        if not action_buy and tick.bid >= limit_px:
            log.info("MARKET FALLBACK %s SELL: bid %.5f ≥ limit %.5f",
                     sym, tick.bid, limit_px)
            return self.submit_market(sym, sig, lots)

        # Decide LIMIT vs STOP based on where limit_px sits relative to market:
        #   BUY  LIMIT → price < ask  (retracement entry below market)
        #   BUY  STOP  → price > ask  (breakout entry above market)
        #   SELL LIMIT → price > bid  (retracement entry above market)
        #   SELL STOP  → price < bid  (breakout entry below market)
        if action_buy:
            order_type = (self.mt5.ORDER_TYPE_BUY_LIMIT
                          if limit_px < tick.ask
                          else self.mt5.ORDER_TYPE_BUY_STOP)
            order_label = "BUY LIMIT" if limit_px < tick.ask else "BUY STOP"
        else:
            order_type = (self.mt5.ORDER_TYPE_SELL_LIMIT
                          if limit_px > tick.bid
                          else self.mt5.ORDER_TYPE_SELL_STOP)
            order_label = "SELL LIMIT" if limit_px > tick.bid else "SELL STOP"

        # MT5 enforces trade_stops_level points minimum distance from market.
        # Violating it gives retcode 10015 (INVALID_PRICE). Skip rather than crash.
        stops = int(getattr(info, "trade_stops_level", 0) or 0)
        min_dist = max(stops, 1) * info.point
        if action_buy:
            ref_price = tick.ask
            too_close = abs(limit_px - ref_price) < min_dist
        else:
            ref_price = tick.bid
            too_close = abs(limit_px - ref_price) < min_dist
        if too_close:
            log.warning("SKIP PENDING %s %s: limit %.5f too close to %.5f "
                        "(need ≥%.5f gap, stops_level=%d pts)",
                        sym, order_label, limit_px, ref_price, min_dist, stops)
            return False

        if self.dry_run:
            log.info("[DRY-RUN PENDING] %s %s lots=%.2f  limit=%.5f  TP=%.5f  SL=%.5f"
                     "  expiry=%dm",
                     sym, order_label, lots, limit_px, tp, sl, expiry_minutes)
            return True

        # ORDER_TIME_GTC — we cancel manually after expiry_minutes.
        # ORDER_TIME_SPECIFIED is widely rejected by brokers (retcode 10022).
        request = {
            "action":       self.mt5.TRADE_ACTION_PENDING,
            "symbol":       sym_mt5,
            "volume":       float(lots),
            "type":         order_type,
            "price":        limit_px,
            "sl":           sl,
            "tp":           tp,
            "deviation":    20,
            "magic":        MAGIC_NUMBER,
            "comment":      "strategy1_pending",
            "type_time":    self.mt5.ORDER_TIME_GTC,
            "type_filling": _resolve_filling_mode(self.mt5, info),
        }
        result = self.mt5.order_send(request)
        if result is None:
            log.error("PENDING ORDER FAILED %s: order_send returned None  err=%s",
                      sym, self.mt5.last_error())
            return False
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            log.error("PENDING ORDER REJECT %s  retcode=%d  comment=%s",
                      sym, result.retcode, result.comment)
            return False
        self._pending_orders[sym] = (result.order, time.monotonic())
        log.info("PENDING PLACED %s %s lots=%.2f  limit=%.5f  TP=%.5f  SL=%.5f"
                 "  expiry=%dm  ticket=%d",
                 sym, order_label, lots, limit_px, tp, sl,
                 expiry_minutes, result.order)
        return True

    def cancel_expired_pending(self, expiry_minutes: int) -> None:
        """Cancel any pending limit orders older than expiry_minutes."""
        if not self._pending_orders:
            return
        now = time.monotonic()
        expired = [sym for sym, (_, placed) in self._pending_orders.items()
                   if (now - placed) >= expiry_minutes * 60]
        for sym in expired:
            ticket, _ = self._pending_orders.pop(sym)
            sym_mt5 = self.symbol_map[sym]
            orders = self.mt5.orders_get(symbol=sym_mt5)
            if not orders or not any(o.ticket == ticket for o in orders):
                # Already filled or gone — nothing to cancel
                continue
            result = self.mt5.order_send({
                "action": self.mt5.TRADE_ACTION_REMOVE,
                "order":  ticket,
            })
            if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
                log.info("PENDING CANCELLED %s ticket=%d (expired after %dm)",
                         sym, ticket, expiry_minutes)
                self.stats["force_closes"] += 1
            else:
                rc = getattr(result, "retcode", None)
                log.warning("PENDING CANCEL FAILED %s ticket=%d retcode=%s",
                            sym, ticket, rc)

    def force_close_all(self, logical_pairs: list[str], reason: str) -> int:
        """Cancel all positions across the given pairs. Returns count closed."""
        if self.dry_run:
            return 0
        n = 0
        for sym in logical_pairs:
            sym_mt5 = self.symbol_map[sym]
            positions = self.mt5.positions_get(symbol=sym_mt5)
            if not positions:
                continue
            info = self._symbol_info[sym]
            for pos in positions:
                tick = self.mt5.symbol_info_tick(sym_mt5)
                if tick is None:
                    continue
                is_buy_pos = (pos.type == self.mt5.POSITION_TYPE_BUY)
                close_type = self.mt5.ORDER_TYPE_SELL if is_buy_pos else self.mt5.ORDER_TYPE_BUY
                close_price = tick.bid if is_buy_pos else tick.ask
                request = {
                    "action":       self.mt5.TRADE_ACTION_DEAL,
                    "symbol":       sym_mt5,
                    "volume":       float(pos.volume),
                    "type":         close_type,
                    "position":     pos.ticket,
                    "price":        close_price,
                    "deviation":    20,
                    "magic":        MAGIC_NUMBER,
                    "comment":      f"force_close:{reason}",
                    "type_time":    self.mt5.ORDER_TIME_GTC,
                    "type_filling": _resolve_filling_mode(self.mt5, info),
                }
                result = self.mt5.order_send(request)
                if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
                    log.warning("FORCE-CLOSE %s ticket=%d vol=%.2f reason=%s",
                                sym, pos.ticket, pos.volume, reason)
                    n += 1
                else:
                    rc = getattr(result, "retcode", None)
                    log.error("FORCE-CLOSE FAILED %s ticket=%d retcode=%s",
                              sym, pos.ticket, rc)
        if n:
            self.stats["force_closes"] += n
        return n


# ── Tick-driven 1m bar builder ────────────────────────────────────────────────
#
# Why this exists: MT5's `copy_rates_from_pos` blocks inside the call when the
# terminal needs to fetch history from the broker, so it can't drive tight
# 1m-bar timing reliably. `symbol_info_tick` is a non-blocking memory read of
# the latest cached tick (sub-ms). We poll ticks at ~10 Hz and aggregate them
# into 1m bars locally — the same pattern an MQL5 EA's OnTick() uses. Bar
# closes are detected by minute-rollover in tick timestamps.

class TickBarBuilder:
    def __init__(self, mt5_module, symbol_map: dict[str, str],
                 broker_offset_hours: int, history_keep: int = 500):
        self.mt5 = mt5_module
        self.symbol_map = symbol_map
        self.broker_offset_hours = broker_offset_hours
        self.history_keep = history_keep
        self.history: dict[str, pd.DataFrame] = {}
        self.current_bar: dict[str, dict | None] = {}
        self.last_tick_time_msc: dict[str, int] = {}
        self.last_tick_wall_ts: dict[str, float] = {}  # monotonic seconds, for staleness logs
        self.last_mid: dict[str, float] = {}           # latest tick mid-price
        self.last_bid: dict[str, float] = {}           # latest tick bid
        self.last_ask: dict[str, float] = {}           # latest tick ask

    def seed(self, sym: str, df_with_partial: pd.DataFrame) -> None:
        """Seed builder with historical bars + the partial current bar.
        df_with_partial: OHLCV index UTC; df.iloc[-1] is the in-progress bar.
        """
        if df_with_partial is None or len(df_with_partial) < 2:
            self.history[sym] = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
            self.current_bar[sym] = None
        else:
            self.history[sym] = df_with_partial.iloc[:-1].copy()
            partial = df_with_partial.iloc[-1]
            self.current_bar[sym] = {
                "time":  df_with_partial.index[-1],
                "open":  float(partial["Open"]),
                "high":  float(partial["High"]),
                "low":   float(partial["Low"]),
                "close": float(partial["Close"]),
            }
        self.last_tick_time_msc[sym] = 0
        self.last_tick_wall_ts[sym] = 0.0

    def _append_to_history(self, sym: str, bar: dict) -> None:
        row = pd.DataFrame(
            [{"Open": bar["open"], "High": bar["high"],
              "Low":  bar["low"],  "Close": bar["close"], "Volume": 0}],
            index=pd.DatetimeIndex([bar["time"]], tz="UTC"),
        )
        h = pd.concat([self.history[sym], row])
        h = h[~h.index.duplicated(keep="last")].sort_index().iloc[-self.history_keep:]
        self.history[sym] = h

    def poll(self, sym: str) -> datetime | None:
        """Read latest tick, update bars. Returns the closed bar's UTC time if
        a minute boundary was just crossed by an incoming tick, else None.
        """
        tick = self.mt5.symbol_info_tick(self.symbol_map[sym])
        if tick is None:
            return None
        # Prefer ms timestamp when available (MT5 5.0.37+); fall back to seconds.
        time_msc = int(getattr(tick, "time_msc", 0) or (int(tick.time) * 1000))
        if time_msc <= self.last_tick_time_msc.get(sym, 0):
            return None  # No new tick since last poll
        self.last_tick_time_msc[sym] = time_msc
        self.last_tick_wall_ts[sym] = time.monotonic()

        bid = float(getattr(tick, "bid", 0) or 0)
        ask = float(getattr(tick, "ask", 0) or 0)
        if not (bid > 0 and ask > 0 and ask >= bid):
            return None
        mid = (bid + ask) / 2.0
        self.last_mid[sym] = mid
        self.last_bid[sym] = bid
        self.last_ask[sym] = ask

        # Broker time → UTC. tick.time/time_msc are in broker timezone.
        broker_dt = datetime.fromtimestamp(time_msc / 1000.0, tz=timezone.utc)
        utc_dt = broker_dt - timedelta(hours=self.broker_offset_hours)
        bar_minute = utc_dt.replace(second=0, microsecond=0)

        cur = self.current_bar.get(sym)
        closed_time: datetime | None = None

        if cur is None:
            self.current_bar[sym] = {"time": bar_minute, "open": mid,
                                     "high": mid, "low": mid, "close": mid}
        elif cur["time"] == bar_minute:
            if mid > cur["high"]: cur["high"] = mid
            if mid < cur["low"]:  cur["low"]  = mid
            cur["close"] = mid
        else:
            # Minute rolled over → finalize previous bar
            self._append_to_history(sym, cur)
            closed_time = cur["time"]
            # Start new partial bar
            self.current_bar[sym] = {"time": bar_minute, "open": mid,
                                     "high": mid, "low": mid, "close": mid}

        return closed_time

    def force_close_stale_bar(self, sym: str, now_utc: datetime) -> datetime | None:
        """If the current partial bar's minute is in the past (no tick came in
        to roll it over), finalize it now using its last observed mid as the
        close. Returns the closed bar's UTC time, else None.

        Keeps bar emission on the wall clock for pairs with sparse ticks. The
        next real tick will start a fresh partial at its own minute.
        """
        cur = self.current_bar.get(sym)
        if cur is None:
            return None
        current_minute = now_utc.replace(second=0, microsecond=0)
        if cur["time"] >= current_minute:
            return None
        self._append_to_history(sym, cur)
        closed_time = cur["time"]
        self.current_bar[sym] = None
        return closed_time

    def closed_bars(self, sym: str) -> pd.DataFrame:
        return self.history.get(sym, pd.DataFrame(
            columns=["Open","High","Low","Close","Volume"]))

    def bars_with_partial(self, sym: str) -> pd.DataFrame:
        """Return closed history + a partial bar appended at the end.

        Callers (`_evaluate_one_symbol`) strip `iloc[-1]` to get the just-closed
        bar at `iloc[-2]`. We must always end with *some* partial; if there is
        no live partial (e.g. right after `force_close_stale_bar`), synthesize
        a flat carry-forward bar one minute after the latest close so the
        strip-last contract stays correct.
        """
        h = self.closed_bars(sym)
        cur = self.current_bar.get(sym)
        if cur is not None:
            partial = pd.DataFrame(
                [{"Open": cur["open"], "High": cur["high"],
                  "Low": cur["low"], "Close": cur["close"], "Volume": 0}],
                index=pd.DatetimeIndex([cur["time"]], tz="UTC"),
            )
            return pd.concat([h, partial])
        if len(h) == 0:
            return h
        last_close = float(h.iloc[-1]["Close"])
        partial_time = h.index[-1] + timedelta(minutes=1)
        partial = pd.DataFrame(
            [{"Open": last_close, "High": last_close,
              "Low": last_close, "Close": last_close, "Volume": 0}],
            index=pd.DatetimeIndex([partial_time], tz="UTC"),
        )
        return pd.concat([h, partial])

    def tick_age_ms(self, sym: str) -> float | None:
        wall = self.last_tick_wall_ts.get(sym, 0.0)
        if wall <= 0:
            return None
        return (time.monotonic() - wall) * 1000.0


# ── Live driver ───────────────────────────────────────────────────────────────

def _check_and_submit(
    trader: MT5Trader,
    sym: str,
    sig: dict,
    args,
    max_spread_pips,
    max_slippage_pips,
    overnight: dict,
    tag: str = "",
) -> bool:
    """Apply all entry guards and submit if clear. Returns True if order placed."""
    if trader.has_open_position(sym):
        trader.stats["signals_skipped_inpos"] += 1
        return False

    bar_t = sig["time"]
    no_entry_after_hour  = overnight.get("no_entry_after_hour")
    no_friday_after_hour = overnight.get("no_friday_after_hour")
    if no_entry_after_hour is not None and bar_t.hour >= no_entry_after_hour:
        trader.stats["signals_skipped_session"] += 1
        log.info("SKIP %s%s: hour %d ≥ no_entry_after_hour=%d",
                 sym, tag, bar_t.hour, no_entry_after_hour)
        return False
    if (no_friday_after_hour is not None
            and bar_t.dayofweek == 4
            and bar_t.hour >= no_friday_after_hour):
        trader.stats["signals_skipped_session"] += 1
        log.info("SKIP %s%s: Friday %d:00 ≥ no_friday_after_hour=%d",
                 sym, tag, bar_t.hour, no_friday_after_hour)
        return False

    if trader.is_daily_blocked(sym):
        trader.stats["signals_skipped_dailydd"] += 1
        log.info("SKIP %s%s: daily DD limit hit (today PnL=$%+.2f, limit=$-%.0f)",
                 sym, tag, trader.daily_pnl(sym), args.daily_dd_limit_pair)
        return False

    trader.stats["signals_total"] += 1
    log.info("SIGNAL%s %s @ %s  %s  entry=%.5f  TP=%.5f  SL=%.5f",
             tag, sym, sig["time"].strftime("%Y-%m-%d %H:%M"),
             sig["action"], sig["entry_price"], sig["tp_price"], sig["sl_price"])

    if max_spread_pips is not None:
        limit  = max_spread_pips.get(sym)
        spread = trader.current_spread_pips(sym)
        if spread is None:
            log.info("SKIP %s%s: no live quote yet (spread guard)", sym, tag)
            trader.stats["signals_skipped_nodata"] += 1
            return False
        if limit is not None and spread > limit:
            log.info("SKIP %s%s: spread %.2f pips > limit %.2f", sym, tag, spread, limit)
            trader.stats["signals_skipped_spread"] += 1
            return False
        log.info("PASS %s%s: spread %.2f pips ≤ limit %.2f", sym, tag, spread,
                 limit if limit is not None else -1)

    if max_slippage_pips is not None:
        limit = max_slippage_pips.get(sym)
        result = trader.entry_slippage_pips(sym, sig)
        if result is None:
            log.info("SKIP %s%s: no live quote yet (slippage guard)", sym, tag)
            trader.stats["signals_skipped_nodata"] += 1
            return False
        slip, fillable = result
        if limit is not None and slip > limit:
            log.info("SKIP %s%s: slippage %+.2f pips > limit %.2f "
                     "(signal=%.5f live=%.5f, against us)",
                     sym, tag, slip, limit, sig["entry_price"], fillable)
            trader.stats["signals_skipped_slippage"] += 1
            return False
        if slip < 0:
            log.info("PASS %s%s: slippage %+.2f pips (in our favor, "
                     "signal=%.5f live=%.5f)", sym, tag, slip,
                     sig["entry_price"], fillable)
        else:
            log.info("PASS %s%s: slippage %+.2f pips ≤ limit %.2f "
                     "(signal=%.5f live=%.5f)", sym, tag, slip,
                     limit if limit is not None else -1,
                     sig["entry_price"], fillable)

    lots, was_capped = trader.size_lots(sym, sig, args)
    if lots <= 0:
        log.info("SKIP %s%s: computed lots=0 (zero SL distance / step rounding)",
                 sym, tag)
        return False
    if was_capped:
        log.info("CAP %s%s: would have wanted >max-lot at risk=$%.0f, "
                 "capping at %.2f lots", sym, tag,
                 args.risk_usd if args.risk_usd else 0, lots)

    if getattr(args, "pending_entry", False):
        if trader.submit_pending(sym, sig, lots, args.expiry_minutes):
            trader.stats["trades_placed"] += 1
            return True
    else:
        if trader.submit_market(sym, sig, lots):
            trader.stats["trades_placed"] += 1
            return True
    return False


def _evaluate_one_symbol(trader: MT5Trader, sym: str, p: dict, pair_cfg: dict,
                         args, max_spread_pips, max_slippage_pips,
                         overnight: dict, logical_pairs: list[str],
                         df: pd.DataFrame) -> dict | None:
    """
    Evaluate bar-close signal. Returns the signal dict when --price-improvement
    is active (caller handles submission). Otherwise submits via _check_and_submit
    and returns None.
    """
    warmup = p["lookback"] + p["atr_period"] + 1
    if df is None or len(df) < warmup + 1:
        trader.stats["signals_skipped_nodata"] += 1
        return None

    # Drop the partial (current) last bar — same as IB live & backtest.
    closed_df = df.iloc[:-1]
    closed_bar_time = closed_df.index[-1]

    # Idempotency: only act once per closed bar per symbol.
    if trader._last_bar_time.get(sym) == closed_bar_time:
        return None
    trader._last_bar_time[sym] = closed_bar_time

    sig = evaluate_entry(closed_df, p, pair_cfg,
                         use_filters=not args.no_filters)
    if sig is None:
        return None

    if getattr(args, "price_improvement", False):
        return sig  # caller handles price-gating and submission

    _check_and_submit(trader, sym, sig, args, max_spread_pips,
                      max_slippage_pips, overnight)
    return None




def _enqueue_price_wait(
    price_wait: dict,
    trader: MT5Trader,
    sym: str,
    sig: dict,
    args,
    overnight: dict,
) -> None:
    """
    Apply pre-entry guards and either enter immediately (if price already
    favourable) or park the signal in `price_wait` for tick-level monitoring.

    Called only when --price-improvement is active.

    Good price:
      BUY  → enter when ask ≤ sig["entry_price"]  (price came back to signal close)
      SELL → enter when bid ≥ sig["entry_price"]
    """
    bar_t = sig["time"]
    no_entry = overnight.get("no_entry_after_hour")
    no_fri   = overnight.get("no_friday_after_hour")

    if no_entry is not None and bar_t.hour >= no_entry:
        trader.stats["signals_skipped_session"] += 1
        return
    if no_fri is not None and bar_t.dayofweek == 4 and bar_t.hour >= no_fri:
        trader.stats["signals_skipped_session"] += 1
        return
    if trader.is_daily_blocked(sym):
        trader.stats["signals_skipped_dailydd"] += 1
        return
    if trader.has_open_position(sym):
        trader.stats["signals_skipped_inpos"] += 1
        return

    lots, was_capped = trader.size_lots(sym, sig, args)
    if lots <= 0:
        return
    if was_capped:
        log.info("CAP %s: risk-based lots capped at %.2f", sym, lots)

    trader.stats["signals_total"] += 1
    log.info("SIGNAL %s @ %s  %s  entry=%.5f  TP=%.5f  SL=%.5f",
             sym, sig["time"].strftime("%Y-%m-%d %H:%M"),
             sig["action"], sig["entry_price"], sig["tp_price"], sig["sl_price"])

    ba = trader.get_bid_ask(sym)
    target = sig["entry_price"]
    already_good = False
    if ba is not None:
        bid_now, ask_now = ba
        already_good = ((sig["action"] == "BUY"  and ask_now <= target) or
                        (sig["action"] == "SELL" and bid_now >= target))

    if already_good:
        log.info("PRICE-IMPR %s %s: price already at/past target=%.5f — immediate entry",
                 sym, sig["action"], target)
        if trader.submit_market(sym, sig, lots):
            trader.stats["trades_placed"] += 1
    else:
        price_wait[sym] = {
            "sig":       sig,
            "lots":      lots,
            "placed_at": time.monotonic(),
        }
        trader.stats["signals_price_wait"] += 1
        ref = "ask" if sig["action"] == "BUY" else "bid"
        log.info("PRICE-WAIT %s %s  target=%s≤%.5f  lots=%.2f  expires=%dm",
                 sym, sig["action"], ref, target, lots, args.expiry_minutes)


def run_live(args, cfg) -> None:
    if args.lot is None and args.risk_usd is None:
        # User asked for "1 lot max OR $100 max risk" → risk-based default.
        args.risk_usd = 100.0
    if args.lot is not None and args.risk_usd is not None:
        raise SystemExit("Pass exactly one of --lot or --risk-usd")
    if args.max_lot is None:
        args.max_lot = 1.0  # user-requested hard cap

    p = dict(cfg["params"])  # copy — we mutate below
    p["lookback"] = args.lookback
    if args.no_tight:
        p["tight_atr"] = None
    p["max_breakout_atr"] = args.max_breakout_atr  # None = disabled
    if args.mode == "breakout":
        # Flip direction: long above range high, short below range low.
        # Swap TP/SL distances so TP=2×ATR (wide target) and SL=1×ATR (tight stop).
        _ib_mod.FADE = False
        p["tp_atr"], p["sl_atr"] = p["sl_atr"], p["tp_atr"]
        log.info("MODE: breakout  (FADE=False, tp_atr=%.1f, sl_atr=%.1f)",
                 p["tp_atr"], p["sl_atr"])
    else:
        _ib_mod.FADE = True  # explicit, in case process is reused
        log.info("MODE: fade  (FADE=True, tp_atr=%.1f, sl_atr=%.1f)",
                 p["tp_atr"], p["sl_atr"])
    logical_pairs = args.pairs or DEFAULT_PAIRS
    # Validate against config — pairs must exist there for filter / usd_quote info
    for sym in logical_pairs:
        if sym not in cfg["pairs"]:
            raise SystemExit(f"Pair {sym} not in {CONFIG_PATH.name}; add it or use --pairs")

    max_spread_pips = (_resolve_max_spreads(args.max_spread_pips, logical_pairs)
                       if args.spread_guard else None)
    max_slippage_pips = (_resolve_max_slippages(args.max_slippage_pips, logical_pairs)
                         if args.slippage_guard else None)
    overnight = _resolve_overnight(args)

    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        log.error("MetaTrader5 package is not installed.  Run:  pip install MetaTrader5")
        raise SystemExit(1) from exc

    symbol_map = _resolve_symbol_map(args, logical_pairs)
    trader = MT5Trader(mt5, symbol_map, dry_run=args.dry_run,
                       daily_dd_limit=args.daily_dd_limit_pair,
                       broker_offset_hours=0)
    try:
        trader.connect(args.mt5_path, args.login, args.password, args.server)
        warmup_required = p["lookback"] + p["atr_period"] + 1
        # Select first symbol up-front so we can detect broker tz BEFORE any
        # bar-history fetch (fetch_bars uses broker_offset_hours to convert).
        first_mt5_sym = symbol_map[logical_pairs[0]]
        if not mt5.symbol_select(first_mt5_sym, True):
            raise RuntimeError(f"Cannot select {first_mt5_sym} — check the "
                               f"symbol name (--symbol-suffix / --symbol-map)")
        trader.broker_offset_hours = _detect_broker_offset(mt5, first_mt5_sym)
        seed_dfs = trader.prime_symbols(logical_pairs, warmup_required)

        sizing_str = (f"--lot {args.lot}" if args.lot is not None
                      else f"--risk-usd ${args.risk_usd:.0f}")
        cap_str = f"max_lot={args.max_lot}"
        dd_str  = (f"daily_dd_per_pair=$-{args.daily_dd_limit_pair:.0f}"
                   if args.daily_dd_limit_pair is not None else "daily_dd=off")
        on_str  = (", ".join(f"{k}={v}" for k, v in overnight.items())
                   if overnight else "off")
        spread_str = ("OFF" if max_spread_pips is None
                      else "ON " + " ".join(f"{s}<={pips:g}p"
                                            for s, pips in max_spread_pips.items()))
        slip_str = ("OFF" if max_slippage_pips is None
                    else "ON " + " ".join(f"{s}<={pips:g}p"
                                          for s, pips in max_slippage_pips.items()))
        log.info("Live config: pairs=%s  sizing=%s  %s  %s  overnight={%s}  "
                 "spread_guard=%s  slippage_guard=%s  dry_run=%s",
                 logical_pairs, sizing_str, cap_str, dd_str, on_str,
                 spread_str, slip_str, args.dry_run)

        # ── Seed the tick-driven bar builder with the bars from prime_symbols ─
        builder = TickBarBuilder(mt5, symbol_map, trader.broker_offset_hours)
        for sym in logical_pairs:
            df0 = seed_dfs[sym]
            builder.seed(sym, df0)
            log.info("Builder seeded %s: %d closed bars + 1 partial "
                     "(latest closed=%s)",
                     sym, len(df0) - 1,
                     df0.index[-2].strftime("%Y-%m-%d %H:%M UTC"))

        # ── Tick-poll main loop ──────────────────────────────────────────────
        # Ticks build the in-progress 1m OHLCV bar continuously (10 Hz poll).
        # Evaluation fires once per minute at a fixed wall-clock offset
        # (EVAL_OFFSET_MS ms into the minute) so ALL pairs are processed at the
        # same moment regardless of tick frequency. Pairs with sparse ticks are
        # force-closed at the evaluation point; pairs that already rolled over
        # via a tick have their bar in history and evaluate correctly too.
        POLL_INTERVAL_S   = 0.1   # 10 Hz tick poll for OHLCV building
        EVAL_OFFSET_MS    = 5     # evaluate all pairs 5ms into each new minute
        HEARTBEAT_S       = 60.0
        STATS_S           = 600.0
        STALE_TICK_WARN_S = 180.0

        last_heartbeat    = time.monotonic()
        last_stats_log    = time.time()
        last_eval_minute: datetime | None = None
        last_minute_tick: int | None = None
        bars_emitted      = {s: 0 for s in logical_pairs}
        # pending_levels[sym]: range+ATR for the bar currently forming, used
        # by --intra-bar to enter as soon as a tick crosses the range boundary.
        # Refreshed after each bar close. None = no valid signal level yet.
        pending_levels: dict[str, dict | None] = {s: None for s in logical_pairs}

        # price_wait[sym]: signal queued for --price-improvement mode. Cleared when
        # the price condition is met, position opened externally, or it expires.
        # {sig, lots, placed_at (monotonic seconds)}
        price_wait: dict[str, dict] = {}

        close_by_hour = overnight.get("close_by_hour")

        while True:
            loop_start = time.monotonic()
            now_utc = datetime.now(timezone.utc)

            # 1. Poll ticks — updates in-progress OHLCV bars for all pairs.
            #    Tick rollovers update history but do NOT trigger evaluation;
            #    the scheduled pass below handles all pairs uniformly.
            if args.pending_entry:
                trader.cancel_expired_pending(args.expiry_minutes)

            for sym in logical_pairs:
                try:
                    builder.poll(sym)
                except Exception:
                    log.exception("tick poll failed for %s", sym)

            # 1b. Intra-bar entry (--intra-bar, fade mode only).
            #     After each bar close we pre-compute range+ATR from the closed
            #     bars (pending_levels). On every tick we check whether the new
            #     mid has crossed the range boundary. If so, we enter immediately
            #     — at the range edge rather than at the bar close, which can be
            #     1-2 pips further away. The scheduled bar-close eval still runs
            #     as a fallback, but has_open_position will block double-entry.
            if args.intra_bar and args.mode == "fade":
                for sym in logical_pairs:
                    lvl = pending_levels.get(sym)
                    if not lvl:
                        continue
                    mid = builder.last_mid.get(sym)
                    if mid is None:
                        continue
                    crossed_high = mid > lvl["range_high"]
                    crossed_low  = mid < lvl["range_low"]
                    if not crossed_high and not crossed_low:
                        continue
                    # Consume the level — one entry per bar.
                    pending_levels[sym] = None
                    direction = -1 if crossed_high else 1
                    atr_e     = lvl["atr"]
                    sig = {
                        "time":        lvl["next_bar_time"],
                        "action":      "BUY" if direction == 1 else "SELL",
                        "direction":   direction,
                        "entry_price": mid,
                        "tp_price":    mid + p["tp_atr"] * atr_e * direction,
                        "sl_price":    mid - p["sl_atr"] * atr_e * direction,
                        "atr":         atr_e,
                        "range_high":  lvl["range_high"],
                        "range_low":   lvl["range_low"],
                    }
                    # Mark bar as handled so bar-close eval skips it.
                    trader._last_bar_time[sym] = lvl["next_bar_time"]
                    log.info("INTRA-BAR %s  %s  mid=%.5f  range=%.5f–%.5f",
                             sym, sig["action"], mid,
                             lvl["range_low"], lvl["range_high"])
                    try:
                        _check_and_submit(trader, sym, sig, args,
                                          max_spread_pips, max_slippage_pips,
                                          overnight, tag=" [intra]")
                    except Exception:
                        log.exception("intra-bar submit failed for %s", sym)

            # 1c. Price-improvement: check waiting signals against current bid/ask.
            if args.price_improvement and price_wait:
                for pw_sym in list(price_wait):
                    pw = price_wait[pw_sym]
                    # Expire
                    if time.monotonic() - pw["placed_at"] >= args.expiry_minutes * 60:
                        log.info("PRICE-WAIT EXPIRED %s %s (no improvement in %dm)",
                                 pw_sym, pw["sig"]["action"], args.expiry_minutes)
                        trader.stats["signals_price_wait_expired"] += 1
                        del price_wait[pw_sym]
                        continue
                    # Drop if a position opened externally (intra-bar, forced, etc.)
                    if trader.has_open_position(pw_sym):
                        del price_wait[pw_sym]
                        continue
                    # Check price condition
                    ba = trader.get_bid_ask(pw_sym)
                    if ba is None:
                        continue
                    bid_now, ask_now = ba
                    pw_sig  = pw["sig"]
                    target  = pw_sig["entry_price"]
                    reached = ((pw_sig["action"] == "BUY"  and ask_now <= target) or
                               (pw_sig["action"] == "SELL" and bid_now >= target))
                    if not reached:
                        continue
                    ref_str = f"ask={ask_now:.5f}" if pw_sig["action"] == "BUY" else f"bid={bid_now:.5f}"
                    log.info("PRICE-WAIT HIT %s %s  %s ≤/≥ target=%.5f",
                             pw_sym, pw_sig["action"], ref_str, target)
                    del price_wait[pw_sym]
                    if trader.submit_market(pw_sym, pw_sig, pw["lots"]):
                        trader.stats["trades_placed"] += 1

            # 2. Scheduled evaluation: once per minute at T+EVAL_OFFSET_MS.
            #    Force-close any pair whose bar hasn't been rolled over by a
            #    tick yet, then evaluate all pairs together. Every pair is
            #    processed within EVAL_OFFSET_MS of the minute boundary.
            ms_into = now_utc.second * 1000 + now_utc.microsecond // 1000
            this_eval_minute = now_utc.replace(second=0, microsecond=0)

            if ms_into >= EVAL_OFFSET_MS and this_eval_minute != last_eval_minute:
                last_eval_minute = this_eval_minute
                prev_minute = this_eval_minute - timedelta(minutes=1)

                for sym in logical_pairs:
                    try:
                        forced = builder.force_close_stale_bar(sym, now_utc)
                        source = "deadline" if forced else "tick"
                        bars_emitted[sym] += 1
                        df = builder.bars_with_partial(sym)
                        log.info("BAR-CLOSE %s @ %s  closed_bars=%d  src=%s",
                                 sym, prev_minute.strftime("%Y-%m-%d %H:%M UTC"),
                                 len(df) - 1 if len(df) > 0 else 0, source)
                        sig = _evaluate_one_symbol(
                            trader, sym, p, cfg["pairs"][sym], args,
                            max_spread_pips, max_slippage_pips,
                            overnight, logical_pairs, df)
                        if sig is not None:
                            # --price-improvement path: gate on bid/ask before submitting.
                            _enqueue_price_wait(
                                price_wait, trader, sym, sig, args, overnight)
                        # Refresh pending levels for the next bar so intra-bar
                        # entry can fire during the bar that is now forming.
                        if args.intra_bar and args.mode == "fade":
                            closed = builder.closed_bars(sym)
                            min_len = p["lookback"] + p["atr_period"]
                            if len(closed) >= min_len:
                                try:
                                    atr_e = float(_atr(closed, p["atr_period"])[-1])
                                    window = closed.iloc[-p["lookback"]:]
                                    rh = float(window["High"].max())
                                    rl = float(window["Low"].min())
                                    tight = p.get("tight_atr")
                                    if tight is None or (rh - rl) < tight * atr_e:
                                        pending_levels[sym] = {
                                            "range_high":   rh,
                                            "range_low":    rl,
                                            "atr":          atr_e,
                                            "next_bar_time": this_eval_minute,
                                        }
                                    else:
                                        pending_levels[sym] = None
                                except Exception:
                                    pending_levels[sym] = None
                    except Exception:
                        log.exception("evaluate failed for %s", sym)

            # 3. Wall-clock minute boundary: close_by_hour.
            this_minute = now_utc.minute + now_utc.hour * 60
            if last_minute_tick is None:
                last_minute_tick = this_minute
            elif this_minute != last_minute_tick:
                last_minute_tick = this_minute
                if close_by_hour is not None and now_utc.hour >= close_by_hour \
                        and not args.dry_run:
                    trader.force_close_all(logical_pairs,
                                           reason=f"close_by_hour={close_by_hour}")

            # 4. Periodic heartbeat — once per minute.
            if time.monotonic() - last_heartbeat >= HEARTBEAT_S:
                parts = []
                for sym in logical_pairs:
                    df = builder.closed_bars(sym)
                    last_close = (df.index[-1].strftime("%H:%M")
                                  if len(df) else "—")
                    age_ms = builder.tick_age_ms(sym)
                    age_str = ("—" if age_ms is None
                               else f"{age_ms:.0f}ms" if age_ms < 60_000
                               else f"{age_ms/1000:.0f}s")
                    parts.append(f"{sym}=last:{last_close}/tick:{age_str}/"
                                 f"emit:{bars_emitted[sym]}")
                    if age_ms is not None and age_ms / 1000 > STALE_TICK_WARN_S:
                        log.warning("STALE-TICKS %s: no tick in %.0fs",
                                    sym, age_ms / 1000)
                log.info("HEARTBEAT %s  %s",
                         now_utc.strftime("%H:%M:%S UTC"), "  ".join(parts))
                last_heartbeat = time.monotonic()
                bars_emitted = {s: 0 for s in logical_pairs}

            # 5. Stats every 10 minutes.
            if time.time() - last_stats_log >= STATS_S:
                log.info("STATS  %s", _format_stats(trader.stats))
                last_stats_log = time.time()

            # 6. Sleep: wake at T+EVAL_OFFSET_MS when approaching, else normal poll.
            elapsed = time.monotonic() - loop_start
            _now = datetime.now(timezone.utc)
            ms_now = _now.second * 1000 + _now.microsecond // 1000
            if ms_now < EVAL_OFFSET_MS:
                sleep_s = (EVAL_OFFSET_MS - ms_now) / 1000.0 - elapsed
            else:
                sleep_s = POLL_INTERVAL_S - elapsed
            if sleep_s > 0.001:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down.")
    finally:
        log.info("FINAL STATS  %s", _format_stats(trader.stats))
        trader.shutdown()


# ── CLI ───────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # MT5 connection
    ap.add_argument("--mt5-path", default=None,
                    help="Path to terminal64.exe (default: MT5 looks up the running terminal)")
    ap.add_argument("--login",    type=int, default=None,
                    help="MT5 account login. If omitted, uses currently logged-in terminal.")
    ap.add_argument("--password", default=None)
    ap.add_argument("--server",   default=None,
                    help='MT5 server name, e.g. "FTMO-Demo" or "FTMO-Server"')
    # Symbol naming
    ap.add_argument("--symbol-suffix", default="",
                    help="Append this to every pair name (e.g. '.raw'). "
                         "Most FTMO accounts: leave empty.")
    ap.add_argument("--symbol-map", nargs="+", default=None,
                    help="Per-pair overrides, e.g. EURUSD=EURUSDx NZDUSD=NZDUSDx")
    # Sizing
    ap.add_argument("--lot",       type=float, default=None,
                    help="Fixed lot size. Mutually exclusive with --risk-usd. "
                         "Default: --risk-usd $100 if neither given.")
    ap.add_argument("--risk-usd",  type=float, default=None,
                    help="Risk-based sizing: lots chosen so SL distance × position "
                         "= this many account-ccy units (default 100).")
    ap.add_argument("--max-lot",   type=float, default=None,
                    help="Hard cap on position size in lots (default 1.0).")
    ap.add_argument("--daily-dd-limit-pair", type=float, default=None,
                    help="Per-pair daily loss limit in account currency. When today's "
                         "realised PnL on a pair drops to -limit, block new entries "
                         "on that pair until 00:00 UTC.")
    # Pair selection & filters
    ap.add_argument("--pairs",    nargs="+", default=None,
                    help=f"Pairs to trade. Default: {' '.join(DEFAULT_PAIRS)}")
    ap.add_argument("--intra-bar", action="store_true", default=False,
                    help="(fade mode only) Enter as soon as a tick crosses the range "
                         "boundary during the bar, rather than waiting for bar close. "
                         "Gives a better entry price — you fade at the range edge "
                         "instead of 1-2 pips past it. Bar-close evaluation still "
                         "runs as a fallback if no intra-bar entry fired.")
    ap.add_argument("--mode", choices=["fade", "breakout"], default="fade",
                    help="'fade' (default): short above range high, long below range low. "
                         "'breakout': long above range high, short below range low, "
                         "with TP and SL distances swapped (TP=2×ATR, SL=1×ATR).")
    ap.add_argument("--no-filters", action="store_true", default=True,
                    help="Ignore the hour/day filters from config (validated default).")
    ap.add_argument("--use-filters", dest="no_filters", action="store_false",
                    help="Force-enable hour/day filters from config (legacy).")
    ap.add_argument("--no-tight", action="store_true", default=True,
                    help="Disable the range-tightness filter (validated default).")
    ap.add_argument("--use-tight", dest="no_tight", action="store_false",
                    help="Re-enable the range-tightness filter (legacy).")
    ap.add_argument("--max-breakout-atr", type=float, default=None,
                    dest="max_breakout_atr",
                    help="Skip fade entries where the bar closed more than N×ATR past the "
                         "range edge. E.g. 0.5 means ignore large momentum breakouts. "
                         "Default: disabled (fade all breakouts).")
    ap.add_argument("--lookback", type=int, default=10,
                    help="Range lookback in bars (validated default: 10).")
    # Overnight
    ap.add_argument("--no-overnight", action="store_true",
                    help="Apply intraday-only defaults: close-by-hour=21, "
                         "no-entry-after-hour=20, no-friday-after-hour=17 (UTC).")
    ap.add_argument("--close-by-hour",       type=int, default=None)
    ap.add_argument("--no-entry-after-hour", type=int, default=None)
    ap.add_argument("--no-friday-after-hour", type=int, default=None)
    # Spread guard
    ap.add_argument("--spread-guard", action="store_true",
                    help="Skip trades when current bid-ask spread exceeds the per-pair limit.")
    ap.add_argument("--max-spread-pips", nargs="+", default=None,
                    help="Spread limits in pips. Single global value (e.g. '1.5') "
                         "or per-pair (e.g. 'EURUSD=0.6 NZDUSD=2.5').")
    ap.add_argument("--slippage-guard", action="store_true",
                    help="Skip trades when the live fillable price (ask for BUY, "
                         "bid for SELL) is more than --max-slippage-pips above "
                         "the signal's intended entry price. Favorable moves "
                         "(price came our way) always pass through.")
    ap.add_argument("--max-slippage-pips", nargs="+", default=None,
                    help="Slippage limits in pips. Single global value or "
                         "per-pair (e.g. 'EURUSD=0.5 NZDUSD=2.0'). Defaults from "
                         "DEFAULT_MAX_SLIPPAGE_PIPS when --slippage-guard is set.")
    # Price-improvement entry
    ap.add_argument("--price-improvement", action="store_true", default=False,
                    help="After a bar-close signal, wait for the market to reach the "
                         "signal's entry price before submitting a market order. "
                         "BUY: enter when ask ≤ bar-close. SELL: enter when bid ≥ bar-close. "
                         "If price never improves within --expiry-minutes, the signal is dropped.")
    # Pending limit entry
    ap.add_argument("--pending-entry", action="store_true", default=False,
                    help="Place a SELL LIMIT / BUY LIMIT at the midpoint of TP and SL "
                         "instead of a market order. Entry price = (tp + sl) / 2, which "
                         "is 0.5×ATR past the range edge — gives 1:1 R:R (1.5×ATR to "
                         "each side). The order cancels automatically via MT5 expiry if "
                         "not filled within --expiry-minutes. Pending orders also block "
                         "double-entry (has_open_position checks orders too).")
    ap.add_argument("--expiry-minutes", type=int, default=3,
                    help="Minutes before an unfilled pending limit order is cancelled "
                         "(default: 3). Only used with --pending-entry.")
    # Misc
    ap.add_argument("--dry-run", action="store_true",
                    help="Log signals + computed orders, never call order_send.")
    args = ap.parse_args()

    cfg = load_config()
    run_live(args, cfg)


if __name__ == "__main__":
    main()
