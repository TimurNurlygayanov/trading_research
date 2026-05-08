"""
Live EMA-touch strategy for MT5.

Entry (fires intra-bar as soon as price enters the zone — no waiting for close):
  BUY:  EMA rising  AND  ask > EMA  AND  (ask - EMA) < max_ema_dist × ATR
        entry = ask
        SL    = bid − 1×ATR          (clears spread automatically)
        TP    = ask + rr×ATR

  SELL: EMA declining  AND  bid < EMA  AND  (EMA - bid) < max_ema_dist × ATR
        entry = bid
        SL    = ask + 1×ATR          (clears spread automatically)
        TP    = bid − rr×ATR

EMA state is computed once per bar from closed bars. On every tick (~10 Hz) the
current bid/ask is checked against the EMA zone. The bar-close evaluation also
runs as a fallback in case no tick crossed the zone during the bar.

Default: EMA=9, ATR=14, max_ema_dist=1.0, R:R=2.0.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from strategy1_live_mt5 import (  # noqa: E402
    MT5Trader,
    TickBarBuilder,
    _check_and_submit,
    _resolve_symbol_map,
    _resolve_max_slippages,
    _detect_broker_offset,
    _format_stats,
    DEFAULT_PAIRS,
)
from strategy1_live_ib import (  # noqa: E402
    _resolve_overnight,
    _resolve_max_spreads,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("strategy2_live_mt5")


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift(1)).abs()
    lpc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ── Signal evaluation ─────────────────────────────────────────────────────────

def evaluate_ema_entry(
    df: pd.DataFrame,
    ema_period: int,
    atr_period: int,
    rr: float,
    max_ema_dist: float,
    price: float | None = None,
    spread: float = 0.0,
) -> dict | None:
    """
    Evaluate for an EMA-touch entry.
    df must be closed bars only (no partial bar).

    price:  execution price — pass ask for BUY, bid for SELL.
            Defaults to df["Close"].iloc[-1] (bar mid) when None.
    spread: current bid-ask spread. Used to push SL outside the spread:
              BUY  SL = entry(ask) − ATR − spread  = bid − ATR
              SELL SL = entry(bid) + ATR + spread  = ask + ATR

    BUY:  EMA rising,    ask > EMA,  (ask − EMA) < max_ema_dist × ATR
          entry = ask
          SL    = ask − ATR − spread   (= bid − ATR)
          TP    = ask + rr × ATR

    SELL: EMA declining, bid < EMA,  (EMA − bid) < max_ema_dist × ATR
          entry = bid
          SL    = bid + ATR + spread   (= ask + ATR)
          TP    = bid − rr × ATR
    """
    warmup = ema_period * 3 + atr_period
    if len(df) < warmup:
        return None

    ema_vals = _ema(df["Close"], ema_period)
    atr_vals = _atr(df, atr_period)

    ema_cur  = float(ema_vals.iloc[-1])
    ema_prev = float(ema_vals.iloc[-2])
    atr_cur  = float(atr_vals.iloc[-1])
    if atr_cur <= 0:
        return None

    entry    = float(df["Close"].iloc[-1]) if price is None else price
    bar_time = df.index[-1]

    ema_rising    = ema_cur > ema_prev
    ema_declining = ema_cur < ema_prev

    if ema_rising and entry > ema_cur and (entry - ema_cur) < max_ema_dist * atr_cur:
        action = "BUY"
        sl     = entry - atr_cur - spread
        tp     = entry + rr * atr_cur
    elif ema_declining and entry < ema_cur and (ema_cur - entry) < max_ema_dist * atr_cur:
        action = "SELL"
        sl     = entry + atr_cur + spread
        tp     = entry - rr * atr_cur
    else:
        return None

    return {
        "time":        bar_time,
        "action":      action,
        "direction":   1 if action == "BUY" else -1,
        "entry_price": entry,
        "tp_price":    tp,
        "sl_price":    sl,
        "atr":         atr_cur,
    }


def _compute_ema_state(
    closed: pd.DataFrame,
    ema_period: int,
    atr_period: int,
) -> dict | None:
    """
    Compute EMA + ATR state from closed bars for intra-bar tick checking.
    Returns None if not enough bars.
    """
    warmup = ema_period * 3 + atr_period
    if len(closed) < warmup:
        return None
    ema_vals = _ema(closed["Close"], ema_period)
    atr_vals = _atr(closed, atr_period)
    ema_cur  = float(ema_vals.iloc[-1])
    ema_prev = float(ema_vals.iloc[-2])
    atr_cur  = float(atr_vals.iloc[-1])
    if atr_cur <= 0:
        return None
    return {
        "ema_cur":      ema_cur,
        "ema_declining": ema_cur < ema_prev,
        "ema_rising":   ema_cur > ema_prev,
        "atr":          atr_cur,
    }


def _evaluate_one_symbol(
    trader: MT5Trader,
    sym: str,
    ema_period: int,
    atr_period: int,
    rr: float,
    max_ema_dist: float,
    args,
    max_spread_pips,
    max_slippage_pips,
    overnight: dict,
    df: pd.DataFrame,
    bid: float | None = None,
    ask: float | None = None,
) -> None:
    warmup = ema_period * 3 + atr_period
    if df is None or len(df) < warmup + 1:
        trader.stats["signals_skipped_nodata"] += 1
        return

    closed_df = df.iloc[:-1]
    closed_bar_time = closed_df.index[-1]

    if trader._last_bar_time.get(sym) == closed_bar_time:
        return
    trader._last_bar_time[sym] = closed_bar_time

    # Use ask (BUY) / bid (SELL) as the execution reference price so that
    # SL and TP are computed relative to the actual fill price, not mid.
    # Determine EMA direction first so we pick the right side.
    ema_vals  = _ema(closed_df["Close"], ema_period)
    ema_cur   = float(ema_vals.iloc[-1])
    ema_prev  = float(ema_vals.iloc[-2])
    if ema_cur > ema_prev and ask is not None:
        exec_price = ask   # EMA rising → potential BUY: fill at ask
    elif ema_cur < ema_prev and bid is not None:
        exec_price = bid   # EMA declining → potential SELL: fill at bid
    else:
        exec_price = None  # no live quote — fall back to bar close (mid)

    spread = (ask - bid) if (bid is not None and ask is not None) else 0.0
    sig = evaluate_ema_entry(closed_df, ema_period, atr_period, rr, max_ema_dist,
                             price=exec_price, spread=spread)
    if sig is None:
        return

    _check_and_submit(trader, sym, sig, args,
                      max_spread_pips, max_slippage_pips, overnight)


# ── Live driver ───────────────────────────────────────────────────────────────

def run_live(args) -> None:
    if args.lot is None and args.risk_usd is None:
        args.risk_usd = 100.0
    if args.lot is not None and args.risk_usd is not None:
        raise SystemExit("Pass exactly one of --lot or --risk-usd")
    if args.max_lot is None:
        args.max_lot = 1.0

    logical_pairs     = args.pairs or DEFAULT_PAIRS
    max_spread_pips   = (_resolve_max_spreads(args.max_spread_pips, logical_pairs)
                         if args.spread_guard else None)
    max_slippage_pips = (_resolve_max_slippages(args.max_slippage_pips, logical_pairs)
                         if args.slippage_guard else None)
    overnight         = _resolve_overnight(args)

    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        log.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
        raise SystemExit(1) from exc

    symbol_map = _resolve_symbol_map(args, logical_pairs)
    trader = MT5Trader(mt5, symbol_map, dry_run=args.dry_run,
                       daily_dd_limit=args.daily_dd_limit_pair,
                       broker_offset_hours=0)

    ema_period      = args.ema_period
    atr_period      = args.atr_period
    rr              = args.rr
    max_ema_dist    = args.max_ema_dist
    warmup_required = ema_period * 3 + atr_period + 1

    try:
        trader.connect(args.mt5_path, args.login, args.password, args.server)
        first_sym = symbol_map[logical_pairs[0]]
        if not mt5.symbol_select(first_sym, True):
            raise RuntimeError(f"Cannot select {first_sym} — check symbol name")
        trader.broker_offset_hours = _detect_broker_offset(mt5, first_sym)
        seed_dfs = trader.prime_symbols(logical_pairs, warmup_required)

        sizing_str = (f"lot={args.lot}" if args.lot is not None
                      else f"risk=${args.risk_usd:.0f}")
        on_str = (", ".join(f"{k}={v}" for k, v in overnight.items())
                  if overnight else "off")
        log.info("EMA-touch live  pairs=%s  EMA%d  ATR%d  RR=%.1f  "
                 "max_ema_dist=%.1f×ATR  %s  max_lot=%.2f  overnight={%s}  dry_run=%s",
                 logical_pairs, ema_period, atr_period, rr,
                 max_ema_dist, sizing_str, args.max_lot, on_str, args.dry_run)

        builder = TickBarBuilder(mt5, symbol_map, trader.broker_offset_hours)
        for sym in logical_pairs:
            df0 = seed_dfs[sym]
            builder.seed(sym, df0)
            log.info("Seeded %s: %d closed bars + 1 partial", sym, len(df0) - 1)

        # Seed the per-symbol EMA state from the initial closed bars.
        # ema_state[sym]: EMA/ATR values from last closed bar, used to check
        # intra-bar entry conditions on every tick without waiting for bar close.
        ema_state: dict[str, dict | None] = {}
        for sym in logical_pairs:
            closed = builder.closed_bars(sym)
            ema_state[sym] = _compute_ema_state(closed, ema_period, atr_period)

        POLL_INTERVAL_S   = 0.1
        EVAL_OFFSET_MS    = 5
        HEARTBEAT_S       = 60.0
        STATS_S           = 600.0
        STALE_TICK_WARN_S = 180.0

        last_heartbeat    = time.monotonic()
        last_stats_log    = time.time()
        last_eval_minute: datetime | None = None
        last_minute_tick: int | None = None
        bars_emitted      = {s: 0 for s in logical_pairs}
        close_by_hour     = overnight.get("close_by_hour")

        while True:
            loop_start = time.monotonic()
            now_utc    = datetime.now(timezone.utc)

            # 1. Cancel expired pending limit/stop orders.
            if args.pending_entry:
                trader.cancel_expired_pending(args.expiry_minutes)

            # 2. Poll ticks — build OHLCV bars at 10 Hz.
            for sym in logical_pairs:
                try:
                    builder.poll(sym)
                except Exception:
                    log.exception("tick poll failed for %s", sym)

            # 3. Intra-bar entry: check EMA zone on every tick.
            #    Uses EMA state from the last closed bar. Fires as soon as the
            #    current mid crosses into the zone — no waiting for bar close.
            #    The bar-close eval (step 4) still runs as a fallback but
            #    _last_bar_time prevents double-entry.
            for sym in logical_pairs:
                state = ema_state.get(sym)
                if not state:
                    continue
                bid = builder.last_bid.get(sym)
                ask = builder.last_ask.get(sym)
                if bid is None or ask is None:
                    continue

                ema_cur  = state["ema_cur"]
                atr_cur  = state["atr"]
                action   = None
                entry    = 0.0
                sl_dist  = 0.0

                if state["ema_rising"] and ask > ema_cur:
                    # BUY: EMA rising, ask just above EMA
                    gap = ask - ema_cur
                    if gap < max_ema_dist * atr_cur:
                        action = "BUY"
                        entry  = ask
                elif state["ema_declining"] and bid < ema_cur:
                    # SELL: EMA declining, bid just below EMA
                    gap = ema_cur - bid
                    if gap < max_ema_dist * atr_cur:
                        action = "SELL"
                        entry  = bid

                if action is None:
                    continue

                # Consume the state so we don't fire again this bar.
                ema_state[sym] = None
                bar_time = now_utc.replace(second=0, microsecond=0)
                trader._last_bar_time[sym] = bar_time

                # BUY:  entry=ask, SL=bid−ATR (=ask−ATR−spread), TP=ask+rr×ATR
                # SELL: entry=bid, SL=ask+ATR (=bid+ATR+spread), TP=bid−rr×ATR
                # SL is always outside the spread — no extra floor needed.
                spread = ask - bid
                if action == "BUY":
                    sl = entry - atr_cur - spread   # entry=ask → sl = bid − ATR
                    tp = entry + rr * atr_cur
                else:
                    sl = entry + atr_cur + spread   # entry=bid → sl = ask + ATR
                    tp = entry - rr * atr_cur

                sig = {
                    "time":        bar_time,
                    "action":      action,
                    "direction":   1 if action == "BUY" else -1,
                    "entry_price": entry,
                    "tp_price":    tp,
                    "sl_price":    sl,
                    "atr":         atr_cur,
                }
                log.info("INTRA-BAR %s %s  entry=%.5f(ask/bid)  EMA=%.5f  dist=%.5f",
                         sym, action, entry, ema_cur, sl_dist)
                try:
                    _check_and_submit(trader, sym, sig, args,
                                      max_spread_pips, max_slippage_pips,
                                      overnight, tag=" [intra]")
                except Exception:
                    log.exception("intra-bar submit failed for %s", sym)

            # 4. Scheduled bar-close evaluation at T+EVAL_OFFSET_MS each minute.
            ms_into          = now_utc.second * 1000 + now_utc.microsecond // 1000
            this_eval_minute = now_utc.replace(second=0, microsecond=0)

            if ms_into >= EVAL_OFFSET_MS and this_eval_minute != last_eval_minute:
                last_eval_minute = this_eval_minute
                prev_minute      = this_eval_minute - timedelta(minutes=1)

                for sym in logical_pairs:
                    try:
                        forced = builder.force_close_stale_bar(sym, now_utc)
                        source = "deadline" if forced else "tick"
                        bars_emitted[sym] += 1
                        df = builder.bars_with_partial(sym)
                        log.info("BAR-CLOSE %s @ %s  closed_bars=%d  src=%s",
                                 sym,
                                 prev_minute.strftime("%Y-%m-%d %H:%M UTC"),
                                 len(df) - 1 if len(df) > 0 else 0,
                                 source)
                        _evaluate_one_symbol(
                            trader, sym, ema_period, atr_period, rr, max_ema_dist,
                            args, max_spread_pips, max_slippage_pips, overnight, df,
                            bid=builder.last_bid.get(sym),
                            ask=builder.last_ask.get(sym))
                        # Refresh EMA state for the bar now forming.
                        closed = builder.closed_bars(sym)
                        ema_state[sym] = _compute_ema_state(
                            closed, ema_period, atr_period)
                    except Exception:
                        log.exception("evaluate failed for %s", sym)

            # 5. Force-close by hour.
            this_minute = now_utc.minute + now_utc.hour * 60
            if last_minute_tick is None:
                last_minute_tick = this_minute
            elif this_minute != last_minute_tick:
                last_minute_tick = this_minute
                if (close_by_hour is not None
                        and now_utc.hour >= close_by_hour
                        and not args.dry_run):
                    trader.force_close_all(logical_pairs,
                                           reason=f"close_by_hour={close_by_hour}")

            # 6. Heartbeat every minute.
            if time.monotonic() - last_heartbeat >= HEARTBEAT_S:
                parts = []
                for sym in logical_pairs:
                    df  = builder.closed_bars(sym)
                    lc  = df.index[-1].strftime("%H:%M") if len(df) else "—"
                    age = builder.tick_age_ms(sym)
                    age_s = ("—" if age is None
                             else f"{age:.0f}ms" if age < 60_000
                             else f"{age/1000:.0f}s")
                    st = ema_state.get(sym)
                    ema_s = (f"EMA={st['ema_cur']:.5f}({'↑' if st['ema_rising'] else '↓'})"
                             if st else "EMA=—")
                    parts.append(
                        f"{sym}=last:{lc}/tick:{age_s}/emit:{bars_emitted[sym]}/{ema_s}")
                    if age is not None and age / 1000 > STALE_TICK_WARN_S:
                        log.warning("STALE-TICKS %s: no tick in %.0fs",
                                    sym, age / 1000)
                log.info("HEARTBEAT %s  %s",
                         now_utc.strftime("%H:%M:%S UTC"), "  ".join(parts))
                last_heartbeat = time.monotonic()
                bars_emitted   = {s: 0 for s in logical_pairs}

            # 7. Stats every 10 minutes.
            if time.time() - last_stats_log >= STATS_S:
                log.info("STATS  %s", _format_stats(trader.stats))
                last_stats_log = time.time()

            # 8. Sleep: snap to T+EVAL_OFFSET_MS when close, else normal poll.
            elapsed = time.monotonic() - loop_start
            _now    = datetime.now(timezone.utc)
            ms_now  = _now.second * 1000 + _now.microsecond // 1000
            sleep_s = ((EVAL_OFFSET_MS - ms_now) / 1000.0 - elapsed
                       if ms_now < EVAL_OFFSET_MS
                       else POLL_INTERVAL_S - elapsed)
            if sleep_s > 0.001:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        log.info("Interrupted; shutting down.")
    finally:
        log.info("FINAL STATS  %s", _format_stats(trader.stats))
        trader.shutdown()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="EMA-touch mean-reversion live bot for MT5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # MT5 connection
    ap.add_argument("--mt5-path",  default=None,
                    help="Path to terminal64.exe (auto-detect if omitted)")
    ap.add_argument("--login",     type=int, default=None)
    ap.add_argument("--password",  default=None)
    ap.add_argument("--server",    default=None,
                    help='MT5 server name, e.g. "FTMO-Demo"')
    # Symbol naming
    ap.add_argument("--symbol-suffix", default="",
                    help="Append to every pair name, e.g. '.raw'")
    ap.add_argument("--symbol-map",    nargs="+", default=None,
                    help="Per-pair overrides: EURUSD=EURUSDx")
    # Strategy parameters
    ap.add_argument("--ema-period",    type=int,   default=9,
                    help="EMA period")
    ap.add_argument("--atr-period",    type=int,   default=14,
                    help="ATR period (distance filter + sizing)")
    ap.add_argument("--rr",            type=float, default=2.0,
                    help="Risk:reward — TP = entry ± rr × sl_dist")
    ap.add_argument("--max-ema-dist",  type=float, default=1.0,
                    help="Max allowed distance between price and EMA, in ATR multiples. "
                         "E.g. 0.5 = only enter when price is within 0.5×ATR of EMA.")
    # Sizing
    ap.add_argument("--lot",      type=float, default=None,
                    help="Fixed lot size. Mutually exclusive with --risk-usd.")
    ap.add_argument("--risk-usd", type=float, default=None,
                    help="Risk-based sizing: lots so SL_dist × lots = $N (default $100).")
    ap.add_argument("--max-lot",  type=float, default=None,
                    help="Hard cap on lots (default 1.0).")
    ap.add_argument("--daily-dd-limit-pair", type=float, default=None,
                    help="Per-pair daily loss limit. Block new entries once hit.")
    # Pairs
    ap.add_argument("--pairs", nargs="+", default=None,
                    help=f"Pairs to trade. Default: {' '.join(DEFAULT_PAIRS)}")
    # Overnight rules
    ap.add_argument("--no-overnight",          action="store_true",
                    help="close-by-hour=21, no-entry-after=20, no-friday-after=17 (UTC).")
    ap.add_argument("--close-by-hour",         type=int, default=None)
    ap.add_argument("--no-entry-after-hour",   type=int, default=None)
    ap.add_argument("--no-friday-after-hour",  type=int, default=None)
    # Spread / slippage guards
    ap.add_argument("--spread-guard",          action="store_true",
                    help="Skip trades when spread exceeds per-pair limit.")
    ap.add_argument("--max-spread-pips",       nargs="+", default=None)
    ap.add_argument("--slippage-guard",        action="store_true",
                    help="Skip trades when live fill price is too far from signal price.")
    ap.add_argument("--max-slippage-pips",     nargs="+", default=None)
    # Pending entry
    ap.add_argument("--pending-entry",         action="store_true", default=False,
                    help="Place a pending order at midpoint of TP/SL instead of market.")
    ap.add_argument("--expiry-minutes",        type=int, default=3,
                    help="Cancel unfilled pending order after N minutes.")
    # Misc
    ap.add_argument("--dry-run",               action="store_true",
                    help="Log signals without placing orders.")
    args = ap.parse_args()
    run_live(args)


if __name__ == "__main__":
    main()
