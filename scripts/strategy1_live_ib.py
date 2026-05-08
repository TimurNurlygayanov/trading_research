"""
Live trading bot for Strategy 1 (range fade) — Interactive Brokers integration.

Subscribes to live 1m bars per pair via the IB API. On each *closed* bar,
evaluates the range-fade entry rule and, when triggered, submits a bracket
order (market entry + stop-loss + take-profit).

Sizing (pass exactly one)
  --lot 0.5           : fixed 0.5 lot (50,000 units)
  --risk-usd 50       : risk-based sizing — units chosen so SL distance × position = $50

Risk caps
  --max-lot 0.5             : hard cap on position size in lots (works with both sizing modes)
  --daily-dd-limit-pair 150 : per-pair daily loss limit; once exceeded, no new entries on
                              that pair until the next UTC day. Resets at 00:00 UTC.

Overnight / weekend rules (UTC hours; --no-overnight applies sensible defaults)
  --no-overnight              : turns on close-by-hour=21, no-entry-after-hour=20,
                                no-friday-after-hour=17
  --close-by-hour 21          : force-close ALL open positions when bar.hour ≥ 21
  --no-entry-after-hour 20    : block new entries when bar.hour ≥ 20
  --no-friday-after-hour 17   : on Friday only, block new entries when bar.hour ≥ 17

Spread guard
  --spread-guard
  --max-spread-pips EURUSD=0.6 NZDUSD=2.5    (or single global value)

Look-ahead audit (verified)
  • ATR/range at bar i use only bars ≤ i, range slice excludes i.
  • df.iloc[-1] is the just-closed bar; partial bar dropped before evaluate.

Cost model (paid by IB on real fills; mirrored in the backtest)
  • Market entry crosses ~half the spread.
  • Stop becomes market on trigger → ~half spread on SL exit.
  • TP limit fills at limit (no extra spread).
  • Commission (IB Pro): max($2, 0.20 bp × notional) per side.

Setup
  pip install ib_async
  Run TWS or IB Gateway, enable API access in Configure → API → Settings,
  set socket port (7497 paper / 7496 live).

Usage
  # Recommended live config (matches the validated backtest)
  python -m scripts.strategy1_live_ib --port 7497 \\
      --pairs EURUSD AUDUSD NZDUSD USDCHF USDCAD \\
      --risk-usd 50 --max-lot 0.5 --daily-dd-limit-pair 150 \\
      --no-overnight --spread-guard

  # Paper-trade dry-run (no orders submitted)
  python -m scripts.strategy1_live_ib --dry-run --no-overnight --spread-guard

  # Offline replay (no IB connection)
  python -m scripts.strategy1_live_ib --simulate --pairs AUDUSD NZDUSD
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
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

from strategy1 import _atr  # reuse the exact same ATR formula
from backtest.data_fetcher import fetch_ohlcv

CONFIG_PATH = Path(__file__).parent / "strategy1_config.json"
LOT_UNITS = 100_000  # 1 standard lot in base currency
FADE = True          # config strategy is "range fade"
_HARD_MAX_UNITS = 5 * LOT_UNITS  # safety cap for risk-based sizing if no --max-lot

# Per-pair max spread (pips) when --spread-guard is enabled. Set to roughly 2×
# the typical IB spread so normal-hour ticks pass and news spikes are skipped.
DEFAULT_MAX_SPREAD_PIPS = {
    "EURUSD": 0.6,
    "GBPUSD": 1.0,
    "AUDUSD": 1.0,
    "NZDUSD": 2.5,
    "USDCHF": 1.5,
    "USDCAD": 1.5,
    "USDJPY": 0.8,
}


def _pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol else 0.0001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("strategy1_live")


# ── Strategy evaluation (shared between live & simulate) ──────────────────────

def evaluate_entry(df: pd.DataFrame, p: dict, pair_cfg: dict,
                   use_filters: bool = True) -> dict | None:
    """
    Evaluate entry rule on the *last closed bar* of df.

    df: OHLCV with DatetimeIndex (UTC), at least lookback + atr_period + 1 rows.
    Returns a signal dict or None.
    """
    lookback         = p["lookback"]
    atr_period       = p["atr_period"]
    tp_atr           = p["tp_atr"]
    sl_atr           = p["sl_atr"]
    tight_atr        = p.get("tight_atr")
    max_breakout_atr = p.get("max_breakout_atr")  # skip if bar closed too far past range

    if len(df) < lookback + atr_period + 1:
        return None

    bar = df.iloc[-1]
    bar_time = df.index[-1]

    if use_filters:
        if bar_time.hour not in set(pair_cfg["hours"]):
            return None
        if bar_time.dayofweek not in set(pair_cfg["days"]):
            return None

    atr_vals = _atr(df, atr_period)
    atr_e = float(atr_vals[-1])

    # Range over the lookback bars *preceding* the current closed bar
    window = df.iloc[-(lookback + 1):-1]
    range_high = float(window["High"].max())
    range_low  = float(window["Low"].min())

    if tight_atr is not None and (range_high - range_low) >= tight_atr * atr_e:
        return None

    close = float(bar["Close"])
    if close > range_high:
        direction = -1 if FADE else 1
        breakout_dist = close - range_high
    elif close < range_low:
        direction = 1 if FADE else -1
        breakout_dist = range_low - close
    else:
        return None

    # Skip if the bar blew far past the range edge — that's momentum, not noise.
    # Large breakouts (>max_breakout_atr × ATR past the range) tend to continue
    # rather than revert; the fade strategy loses edge on them.
    if max_breakout_atr is not None and atr_e > 0 and breakout_dist > max_breakout_atr * atr_e:
        return None

    tp_price = close + tp_atr * atr_e * direction
    sl_price = close - sl_atr * atr_e * direction
    action   = "BUY" if direction == 1 else "SELL"

    return {
        "time":        bar_time,
        "action":      action,
        "direction":   direction,
        "entry_price": close,
        "tp_price":    tp_price,
        "sl_price":    sl_price,
        "atr":         atr_e,
        "range_high":  range_high,
        "range_low":   range_low,
    }


# ── IB connection helpers ─────────────────────────────────────────────────────

def _round_for_pair(symbol: str, price: float) -> float:
    """Round prices to the pair's pip precision (FX TWS expects ≤5 dp)."""
    if symbol.endswith("JPY") or symbol.startswith("JPY"):
        return round(price, 3)
    return round(price, 5)


def _qty_for_lot(lot: float) -> int:
    return int(round(lot * LOT_UNITS))


def _is_usd_quote(symbol: str, pair_cfg: dict) -> bool:
    """Whether USD is the quote currency (XXX/USD) for this pair."""
    if "usd_quote" in pair_cfg:
        return bool(pair_cfg["usd_quote"])
    return symbol.endswith("USD")


def _size_units(sig: dict, symbol: str, pair_cfg: dict, args) -> tuple[int, bool]:
    """Resolve position size from --lot, or --risk-usd capped by --max-lot.

    Returns (units, was_capped).
    """
    cap_units = int(args.max_lot * LOT_UNITS) if args.max_lot is not None else _HARD_MAX_UNITS

    if args.risk_usd is None:
        return min(_qty_for_lot(args.lot), cap_units), False

    sl_distance = abs(sig["entry_price"] - sig["sl_price"])
    if sl_distance <= 0:
        return 0, False
    usd_quote = _is_usd_quote(symbol, pair_cfg)
    if usd_quote:
        u = args.risk_usd / sl_distance
    else:
        u = args.risk_usd * sig["entry_price"] / sl_distance
    if u > cap_units:
        return cap_units, True
    return int(round(u)), False


class DailyPairTracker:
    """Per-pair realized PnL today, with auto-reset at 00:00 UTC."""

    def __init__(self, limit_usd: float | None):
        self.limit = limit_usd  # None = disabled
        self.pnl_per_pair: dict[str, float] = {}
        self.day: object = None  # date

    def _maybe_reset(self, now: datetime) -> None:
        d = now.astimezone(timezone.utc).date()
        if self.day != d:
            if self.day is not None:
                log.info("Daily PnL reset (was %s, now %s)", self.day, d)
            self.day = d
            self.pnl_per_pair.clear()

    def add(self, symbol: str, pnl_usd: float, when: datetime) -> None:
        self._maybe_reset(when)
        self.pnl_per_pair[symbol] = self.pnl_per_pair.get(symbol, 0.0) + pnl_usd

    def is_blocked(self, symbol: str, when: datetime) -> bool:
        if self.limit is None:
            return False
        self._maybe_reset(when)
        return self.pnl_per_pair.get(symbol, 0.0) <= -self.limit

    def get(self, symbol: str) -> float:
        return self.pnl_per_pair.get(symbol, 0.0)


class IBTrader:
    """Wraps ib_async for the strategy."""

    def __init__(self, host: str, port: int, client_id: int, dry_run: bool,
                 daily_dd_limit: float | None = None):
        from ib_async import IB
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.dry_run = dry_run
        self._contracts: dict[str, object] = {}
        self._bars: dict[str, object] = {}
        self._tickers: dict[str, object] = {}
        self._last_bar_time: dict[str, datetime] = {}
        self._processed_fill_ids: set[str] = set()
        self.daily = DailyPairTracker(daily_dd_limit)
        self.stats = {
            "signals_total":           0,
            "signals_skipped_inpos":   0,
            "signals_skipped_spread":  0,
            "signals_skipped_nodata":  0,
            "signals_skipped_dailydd": 0,
            "signals_skipped_session": 0,
            "trades_placed":           0,
            "force_closes":            0,
        }

    async def connect(self) -> None:
        log.info("Connecting to IB at %s:%d (clientId=%d)…", self.host, self.port, self.client_id)
        await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
        log.info("Connected. Server version: %s", self.ib.client.serverVersion())

    def make_forex(self, symbol: str):
        from ib_async import Forex
        if symbol not in self._contracts:
            base, quote = symbol[:3], symbol[3:]
            c = Forex(f"{base}{quote}")
            self.ib.qualifyContracts(c)
            self._contracts[symbol] = c
        return self._contracts[symbol]

    def subscribe_quotes(self, symbol: str) -> None:
        """Streaming bid/ask via reqMktData; ib_async caches into ticker.bid/ask."""
        if symbol in self._tickers:
            return
        contract = self.make_forex(symbol)
        ticker = self.ib.reqMktData(contract, "", False, False)
        self._tickers[symbol] = ticker

    def current_spread_pips(self, symbol: str) -> float | None:
        """Return current (ask - bid) in pips, or None if no live quote yet."""
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return None
        bid = getattr(ticker, "bid", None)
        ask = getattr(ticker, "ask", None)
        # ib_async sets these to nan until first tick; treat as missing
        if bid is None or ask is None:
            return None
        try:
            if not (bid > 0 and ask > 0 and ask >= bid):
                return None
        except TypeError:
            return None
        return (ask - bid) / _pip_size(symbol)

    def has_open_position(self, symbol: str) -> bool:
        c = self._contracts.get(symbol)
        if c is None:
            return False
        for pos in self.ib.positions():
            if pos.contract.localSymbol.replace(".", "") == symbol and pos.position != 0:
                return True
        # Also check open orders we placed but that haven't filled yet
        for trade in self.ib.openTrades():
            if (trade.contract.localSymbol.replace(".", "") == symbol
                and not trade.isDone()):
                return True
        return False

    def submit_bracket(self, symbol: str, sig: dict, qty: int,
                       usd_quote: bool) -> None:
        from ib_async import MarketOrder, LimitOrder, StopOrder

        contract = self.make_forex(symbol)
        action = sig["action"]
        opp    = "SELL" if action == "BUY" else "BUY"
        tp     = _round_for_pair(symbol, sig["tp_price"])
        sl     = _round_for_pair(symbol, sig["sl_price"])

        if self.dry_run:
            log.info("[DRY-RUN] %s %s qty=%d entry≈%.5f  TP=%.5f  SL=%.5f",
                     symbol, action, qty, sig["entry_price"], tp, sl)
            return

        parent = MarketOrder(action, qty)
        parent.orderId  = self.ib.client.getReqId()
        parent.transmit = False

        tp_order = LimitOrder(opp, qty, tp)
        tp_order.orderId  = self.ib.client.getReqId()
        tp_order.parentId = parent.orderId
        tp_order.transmit = False

        sl_order = StopOrder(opp, qty, sl)
        sl_order.orderId  = self.ib.client.getReqId()
        sl_order.parentId = parent.orderId
        sl_order.transmit = True  # final leg transmits the whole bracket

        self.ib.placeOrder(contract, parent)
        tp_trade = self.ib.placeOrder(contract, tp_order)
        sl_trade = self.ib.placeOrder(contract, sl_order)
        log.info("PLACED  %s %s qty=%d  TP=%.5f  SL=%.5f  parentId=%d",
                 symbol, action, qty, tp, sl, parent.orderId)

        # Wire up realized-PnL tracking on the closing legs. Whichever fires
        # (TP or SL), the same handler picks it up and updates the daily
        # circuit breaker.
        entry = {
            "symbol":      symbol,
            "entry_price": sig["entry_price"],
            "direction":   1 if action == "BUY" else -1,
            "usd_quote":   usd_quote,
        }
        self._wire_close_fills(tp_trade, entry)
        self._wire_close_fills(sl_trade, entry)

    def _wire_close_fills(self, trade, entry: dict) -> None:
        """Add a fillEvent handler that converts each closing fill to USD PnL."""
        def _on_fill(_t, fill):
            try:
                exec_id = fill.execution.execId
                if exec_id in self._processed_fill_ids:
                    return
                self._processed_fill_ids.add(exec_id)
                close_px   = float(fill.execution.price)
                close_qty  = float(fill.execution.shares)
                raw = (close_px - entry["entry_price"]) * close_qty * entry["direction"]
                pnl_usd = raw if entry["usd_quote"] else raw / entry["entry_price"]
                # Subtract IB-reported commission on this fill if available
                try:
                    cr = fill.commissionReport
                    if cr and cr.commission:
                        pnl_usd -= float(cr.commission)
                except AttributeError:
                    pass
                self.daily.add(entry["symbol"], pnl_usd,
                               datetime.now(timezone.utc))
                log.info("REALIZED  %s  $%+.2f  daily=$%+.2f",
                         entry["symbol"], pnl_usd, self.daily.get(entry["symbol"]))
            except Exception as exc:
                log.warning("close-fill handler failed for %s: %s",
                            entry["symbol"], exc)
        trade.fillEvent += _on_fill

    def force_close_all(self, reason: str) -> int:
        """Cancel all open bracket legs and market-out any positions. Returns count closed."""
        from ib_async import MarketOrder
        n = 0
        # Cancel pending bracket legs first so the OCO doesn't fight us
        for trade in list(self.ib.openTrades()):
            if not trade.isDone():
                try:
                    self.ib.cancelOrder(trade.order)
                except Exception:
                    pass
        for pos in self.ib.positions():
            if pos.position == 0:
                continue
            sym = pos.contract.localSymbol.replace(".", "")
            qty = abs(int(pos.position))
            action = "SELL" if pos.position > 0 else "BUY"
            order = MarketOrder(action, qty)
            order.orderId = self.ib.client.getReqId()
            self.ib.placeOrder(pos.contract, order)
            log.warning("FORCE-CLOSE %s  qty=%d %s  reason=%s",
                        sym, qty, action, reason)
            n += 1
        if n:
            self.stats["force_closes"] += n
        return n

    def subscribe_1m(self, symbol: str, on_closed_bar) -> None:
        """Request 1m bars with live updates; callback fires on each *closed* bar."""
        contract = self.make_forex(symbol)
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="1 min",
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=2,
            keepUpToDate=True,
        )
        self._bars[symbol] = bars

        def _on_update(_bars, has_new):
            if not has_new or len(_bars) < 2:
                return
            # The last bar in the list is the still-forming one; the one
            # before it is the most recently *closed* bar.
            closed = _bars[-2]
            ts = closed.date
            if isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if self._last_bar_time.get(symbol) == ts:
                return
            self._last_bar_time[symbol] = ts
            df = pd.DataFrame(
                [{"Open": b.open, "High": b.high, "Low": b.low,
                  "Close": b.close, "Volume": b.volume}
                 for b in list(_bars)[:-1]],  # drop the partial bar
                index=pd.DatetimeIndex(
                    [b.date if isinstance(b.date, datetime) else
                     datetime.combine(b.date, datetime.min.time(), tzinfo=timezone.utc)
                     for b in list(_bars)[:-1]],
                    tz="UTC",
                ),
            )
            on_closed_bar(symbol, df)

        bars.updateEvent += _on_update


# ── Live driver ───────────────────────────────────────────────────────────────

def _resolve_overnight(args) -> dict:
    """Build overnight kwargs from --no-overnight + fine-grained overrides."""
    out = {}
    if args.no_overnight:
        out["close_by_hour"]        = 21
        out["no_entry_after_hour"]  = 20
        out["no_friday_after_hour"] = 17
    if args.close_by_hour is not None:
        out["close_by_hour"] = args.close_by_hour
    if args.no_entry_after_hour is not None:
        out["no_entry_after_hour"] = args.no_entry_after_hour
    if args.no_friday_after_hour is not None:
        out["no_friday_after_hour"] = args.no_friday_after_hour
    return out


async def run_live(args, cfg) -> None:
    if args.lot is None and args.risk_usd is None:
        args.lot = 0.5  # default
    if args.lot is not None and args.risk_usd is not None:
        raise SystemExit("Pass exactly one of --lot or --risk-usd")

    p = dict(cfg["params"])  # copy — CLI flags may override
    p["max_breakout_atr"] = args.max_breakout_atr
    pairs = args.pairs or list(cfg["pairs"].keys())
    max_spread_pips = _resolve_max_spreads(args.max_spread_pips, pairs) \
                      if args.spread_guard else None
    overnight = _resolve_overnight(args)
    close_by_hour        = overnight.get("close_by_hour")
    no_entry_after_hour  = overnight.get("no_entry_after_hour")
    no_friday_after_hour = overnight.get("no_friday_after_hour")

    trader = IBTrader(args.host, args.port, args.client_id,
                      dry_run=args.dry_run,
                      daily_dd_limit=args.daily_dd_limit_pair)
    await trader.connect()

    # On startup, prime contracts so position checks work
    for sym in pairs:
        trader.make_forex(sym)
        if max_spread_pips is not None:
            trader.subscribe_quotes(sym)

    def on_closed(symbol: str, df: pd.DataFrame) -> None:
        try:
            sig = evaluate_entry(df, p, cfg["pairs"][symbol],
                                 use_filters=not args.no_filters)

            # Force-close trigger: any open position should exit by close_by_hour.
            # Run on every bar so even when no signal fires, we still enforce it.
            if close_by_hour is not None:
                bar_t = df.index[-1]
                if bar_t.hour >= close_by_hour and not trader.dry_run:
                    if trader.force_close_all(reason=f"close_by_hour={close_by_hour}"):
                        return  # let market-out flow finish before any new entries

            if sig is None:
                return

            if trader.has_open_position(symbol):
                trader.stats["signals_skipped_inpos"] += 1
                return

            # Time-of-day blocks — entry-side overnight rules
            bar_t = sig["time"]
            if no_entry_after_hour is not None and bar_t.hour >= no_entry_after_hour:
                trader.stats["signals_skipped_session"] += 1
                log.info("SKIP %s: hour %d ≥ no_entry_after_hour=%d",
                         symbol, bar_t.hour, no_entry_after_hour)
                return
            if (no_friday_after_hour is not None
                    and bar_t.dayofweek == 4
                    and bar_t.hour >= no_friday_after_hour):
                trader.stats["signals_skipped_session"] += 1
                log.info("SKIP %s: Friday %d:00 ≥ no_friday_after_hour=%d",
                         symbol, bar_t.hour, no_friday_after_hour)
                return

            # Per-pair daily-DD circuit breaker
            now_utc = datetime.now(timezone.utc)
            if trader.daily.is_blocked(symbol, now_utc):
                trader.stats["signals_skipped_dailydd"] += 1
                log.info("SKIP %s: daily DD limit hit (today PnL=$%+.2f, limit=$-%.0f)",
                         symbol, trader.daily.get(symbol), args.daily_dd_limit_pair)
                return

            trader.stats["signals_total"] += 1
            log.info("SIGNAL  %s @ %s  %s  entry=%.5f  TP=%.5f  SL=%.5f",
                     symbol, sig["time"].strftime("%Y-%m-%d %H:%M"),
                     sig["action"], sig["entry_price"],
                     sig["tp_price"], sig["sl_price"])

            if max_spread_pips is not None:
                limit = max_spread_pips.get(symbol)
                spread = trader.current_spread_pips(symbol)
                if spread is None:
                    log.info("SKIP %s: no live quote yet (spread guard)", symbol)
                    trader.stats["signals_skipped_nodata"] += 1
                    return
                if limit is not None and spread > limit:
                    log.info("SKIP %s: spread %.2f pips > limit %.2f",
                             symbol, spread, limit)
                    trader.stats["signals_skipped_spread"] += 1
                    return
                log.info("PASS %s: spread %.2f pips ≤ limit %.2f",
                         symbol, spread, limit if limit is not None else -1)

            # Compute units (fixed-lot or risk-based, capped by max-lot)
            pair_cfg = cfg["pairs"][symbol]
            qty, was_capped = _size_units(sig, symbol, pair_cfg, args)
            if qty <= 0:
                log.info("SKIP %s: computed qty=0 (zero SL distance?)", symbol)
                return
            if was_capped:
                log.info("CAP %s: would have wanted >max-lot at risk=$%.0f, "
                         "capping at %d units", symbol, args.risk_usd, qty)

            trader.submit_bracket(symbol, sig, qty,
                                  usd_quote=_is_usd_quote(symbol, pair_cfg))
            trader.stats["trades_placed"] += 1
        except Exception as exc:
            log.exception("on_closed(%s) failed: %s", symbol, exc)

    for sym in pairs:
        log.info("Subscribing 1m bars: %s", sym)
        trader.subscribe_1m(sym, on_closed)

    guard_str = "OFF" if max_spread_pips is None else \
                "ON " + " ".join(f"{s}<={pips:g}p" for s, pips in max_spread_pips.items())
    sizing_str = f"--lot {args.lot}" if args.lot is not None else f"--risk-usd ${args.risk_usd}"
    cap_str = f"max_lot={args.max_lot}" if args.max_lot is not None else "max_lot=none"
    dd_str  = (f"daily_dd_per_pair=$-{args.daily_dd_limit_pair:.0f}"
               if args.daily_dd_limit_pair is not None else "daily_dd=off")
    on_str  = (", ".join(f"{k}={v}" for k, v in overnight.items())
               if overnight else "off")
    log.info("Live config: pairs=%s  sizing=%s  %s  %s  overnight={%s}  spread_guard=%s  dry_run=%s",
             pairs, sizing_str, cap_str, dd_str, on_str, guard_str, args.dry_run)

    # Periodic stats log every 10 minutes
    async def _stats_loop():
        while True:
            await trader.ib.sleepAsync(600)
            log.info("STATS  %s", _format_stats(trader.stats))

    import asyncio as _aio
    _aio.create_task(_stats_loop())

    try:
        await trader.ib.runAsync()
    finally:
        log.info("FINAL STATS  %s", _format_stats(trader.stats))


def _format_stats(stats: dict) -> str:
    return (f"signals={stats['signals_total']}  "
            f"placed={stats['trades_placed']}  "
            f"skip_inpos={stats['signals_skipped_inpos']}  "
            f"skip_session={stats.get('signals_skipped_session', 0)}  "
            f"skip_dailydd={stats.get('signals_skipped_dailydd', 0)}  "
            f"skip_spread={stats['signals_skipped_spread']}  "
            f"skip_nodata={stats['signals_skipped_nodata']}  "
            f"force_closed={stats.get('force_closes', 0)}")


def _resolve_max_spreads(specs: list[str] | None, pairs: list[str]) -> dict[str, float]:
    """
    Build {pair: max_spread_pips}. specs is a list like
    ["EURUSD=0.6", "GBPUSD=1.0"] OR a single global value like ["1.5"].
    Pairs not specified fall back to DEFAULT_MAX_SPREAD_PIPS.
    """
    out = {sym: DEFAULT_MAX_SPREAD_PIPS.get(sym, 1.5) for sym in pairs}
    if not specs:
        return out
    # Single global value form
    if len(specs) == 1 and "=" not in specs[0]:
        try:
            global_lim = float(specs[0])
            return {sym: global_lim for sym in pairs}
        except ValueError:
            pass
    for s in specs:
        if "=" not in s:
            log.warning("Ignoring malformed --max-spread-pips entry: %r", s)
            continue
        sym, val = s.split("=", 1)
        try:
            out[sym.upper()] = float(val)
        except ValueError:
            log.warning("Ignoring non-numeric value: %r", s)
    return out


# ── Offline simulate driver (no IB) ───────────────────────────────────────────

def run_simulate(args, cfg) -> None:
    """
    Replay the last `--sim-hours` of historical 1m bars, feeding each newly
    'closed' bar into evaluate_entry exactly as the live loop would.
    Prints what would have been traded. No IB connection.
    """
    p     = cfg["params"]
    pairs = args.pairs or list(cfg["pairs"].keys())
    # Simulate mode just logs would-be entries; sizing isn't actually used,
    # but we report a representative qty for context.
    lot_for_log = args.lot if args.lot is not None else 0.5
    qty = _qty_for_lot(lot_for_log)

    end   = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(hours=args.sim_hours)
    start_s = start.strftime("%Y-%m-%d")
    end_s   = (end + timedelta(days=1)).strftime("%Y-%m-%d")

    log.info("SIMULATE  pairs=%s  lot=%.2f  qty=%d  window=%s → %s",
             pairs, lot_for_log, qty, start, end)

    for sym in pairs:
        log.info("Fetching %s 1m %s → %s", sym, start_s, end_s)
        df_full = fetch_ohlcv(sym, "1m", start_s, end_s)
        if df_full.empty:
            log.warning("no data for %s", sym)
            continue
        df_full = df_full[(df_full.index >= start) & (df_full.index <= end)]
        if df_full.empty:
            log.warning("no in-window bars for %s", sym)
            continue

        warmup = p["lookback"] + p["atr_period"] + 1
        in_trade = False
        trades_simulated = 0

        for i in range(warmup, len(df_full)):
            df = df_full.iloc[: i + 1]   # bars 0..i, where bar i is "just closed"
            if in_trade:
                # SL/TP would be checked by IB; here we just stop emitting
                # new signals until the next bar that satisfies our toy "exit"
                # — for simulate, treat each signal as immediately resolved.
                in_trade = False
                continue
            sig = evaluate_entry(df, p, cfg["pairs"][sym],
                                 use_filters=not args.no_filters)
            if sig is None:
                continue
            log.info("[SIM] %s @ %s  %s  entry=%.5f  TP=%.5f  SL=%.5f",
                     sym, sig["time"].strftime("%Y-%m-%d %H:%M"),
                     sig["action"], sig["entry_price"],
                     sig["tp_price"], sig["sl_price"])
            trades_simulated += 1
            in_trade = True

        log.info("%s: %d signals over %d bars",
                 sym, trades_simulated, len(df_full) - warmup)


# ── CLI ───────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host",      default="127.0.0.1")
    ap.add_argument("--port",      type=int, default=7497,
                    help="7497 = TWS paper, 7496 = TWS live, 4002/4001 = Gateway")
    ap.add_argument("--client-id", type=int, default=42)
    ap.add_argument("--lot",       type=float, default=None,
                    help="Fixed lot size (e.g. 0.5 = 50,000 units). Mutually "
                         "exclusive with --risk-usd. Default 0.5 if neither given.")
    ap.add_argument("--risk-usd",  type=float, default=None,
                    help="Risk-based sizing: position chosen so SL distance × "
                         "position = this many USD. Mutually exclusive with --lot.")
    ap.add_argument("--max-lot",   type=float, default=None,
                    help="Hard cap on position size in lots, applied regardless "
                         "of sizing mode (recommended with --risk-usd to bound size).")
    ap.add_argument("--daily-dd-limit-pair", type=float, default=None,
                    help="Per-pair daily loss limit (USD). When today's realised "
                         "PnL on a pair drops to -limit, block new entries on "
                         "that pair until 00:00 UTC.")
    ap.add_argument("--pairs",     nargs="+", default=None,
                    help="Subset of pairs from strategy1_config.json")
    ap.add_argument("--no-filters", action="store_true",
                    help="Ignore the hour/day filters from config")
    ap.add_argument("--max-breakout-atr", type=float, default=None,
                    dest="max_breakout_atr",
                    help="Skip fade entries where the bar closed more than N×ATR past the "
                         "range edge. E.g. 0.5 skips large momentum breakouts. "
                         "Default: disabled (all breakouts are faded).")
    ap.add_argument("--no-overnight", action="store_true",
                    help="Apply intraday-only defaults: close-by-hour=21, "
                         "no-entry-after-hour=20, no-friday-after-hour=17 (UTC). "
                         "Fine-grained overrides below.")
    ap.add_argument("--close-by-hour",       type=int, default=None,
                    help="Force-close all open positions when bar.hour ≥ this (UTC).")
    ap.add_argument("--no-entry-after-hour", type=int, default=None,
                    help="Block new entries when bar.hour ≥ this (UTC).")
    ap.add_argument("--no-friday-after-hour", type=int, default=None,
                    help="On Friday only, block new entries when bar.hour ≥ this.")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Log signals but do not submit orders to IB")
    ap.add_argument("--spread-guard", action="store_true",
                    help="Skip trades when current bid-ask spread exceeds the "
                         "per-pair limit. Default limits in DEFAULT_MAX_SPREAD_PIPS.")
    ap.add_argument("--max-spread-pips", nargs="+", default=None,
                    help="Spread limits in pips. Either a single global number "
                         "(e.g. '1.5') or per-pair (e.g. 'EURUSD=0.6 NZDUSD=2.5'). "
                         "Only used when --spread-guard is on.")
    ap.add_argument("--simulate",  action="store_true",
                    help="Offline replay of recent historical bars (no IB connection)")
    ap.add_argument("--sim-hours", type=int, default=48,
                    help="Hours of recent history to replay in --simulate (default 48)")
    args = ap.parse_args()

    cfg = load_config()

    if args.simulate:
        run_simulate(args, cfg)
        return

    try:
        asyncio.run(run_live(args, cfg))
    except ImportError as exc:
        log.error("ib_async is not installed.  Run:  pip install ib_async")
        log.error("Or use --simulate for offline replay without an IB connection.")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down.")


if __name__ == "__main__":
    main()
