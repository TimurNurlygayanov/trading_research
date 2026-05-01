"""
Strategy 2: Range breakout — EUR/USD 1m (and other timeframes).

Range  : rolling max(high) and min(low) over the last `lookback` bars
Entry  : Long  when close breaks above range high
         Short when close breaks below range low
Filter : optional range-tightness gate — only trade when
         (range_high - range_low) < tight_atr_mult * ATR(14)
Exit   : TP when price reaches entry ± tp_atr * ATR
         SL when price reaches entry ∓ sl_atr * ATR
Size   : 1 lot (100,000 units); 1 pip = $10

One trade at a time.  Entry at the close of the breakout bar.

Usage
  python -m scripts.strategy1
  python -m scripts.strategy1 --sweep
  python -m scripts.strategy1 --lookback 30 --tight-atr 2.0 --tp-atr 3 --sl-atr 2
  python -m scripts.strategy1 --multi --tf 1m 5m
  python -m scripts.strategy1 --multi --sweep --tf 1m
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv

LOT_SIZE    = 100_000
SWEEP_MULTS = [1.0, 2.0, 3.0, 5.0, 10.0]
YEARS       = [2021, 2022, 2023, 2024]
TIMEFRAMES  = ["1m", "5m", "15m", "1h", "4h"]

# Pairs available on USD accounts.
# usd_per_pip: dollar value of 1 pip (0.0001) per lot.
# For XXX/USD pairs: 1 pip = 0.0001 * 100,000 = $10 (USD is quote).
# For USD/YYY pairs: 1 pip = 0.0001 * 100,000 / price ≈ varies; we use
#   price at trade entry to convert, approximated as avg_price constant here.
# jpy_pairs use pip = 0.01 instead of 0.0001.
SYMBOLS = {
    "EURUSD": {"pip": 0.0001, "usd_quote": True},
    "GBPUSD": {"pip": 0.0001, "usd_quote": True},
    "AUDUSD": {"pip": 0.0001, "usd_quote": True},
    "NZDUSD": {"pip": 0.0001, "usd_quote": True},
    "USDJPY": {"pip": 0.01,   "usd_quote": False},  # P&L in JPY → convert /price
    "USDCHF": {"pip": 0.0001, "usd_quote": False},  # P&L in CHF → convert /price
    "USDCAD": {"pip": 0.0001, "usd_quote": False},  # P&L in CAD → convert /price
}


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(close: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(close).ewm(span=span, adjust=False).mean().values


def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n     = len(close)
    tr    = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    atr    = np.empty(n)
    atr[0] = tr[0]
    alpha  = 1.0 / period
    for i in range(1, n):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


# ── Rolling range (excluding current bar) ────────────────────────────────────

def _rolling_range(high: np.ndarray, low: np.ndarray,
                   lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """range_high[i] = max(high[i-lookback : i])  (excludes bar i)."""
    n           = len(high)
    range_high  = np.full(n, np.nan)
    range_low   = np.full(n, np.nan)
    for i in range(lookback, n):
        range_high[i] = high[i - lookback : i].max()
        range_low[i]  = low[i  - lookback : i].min()
    return range_high, range_low


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate(close: np.ndarray,
             low: np.ndarray,
             high: np.ndarray,
             atr_vals: np.ndarray,
             range_high: np.ndarray,
             range_low: np.ndarray,
             tp_atr_mult: float,
             sl_atr_mult: float,
             commission_usd: float,
             tight_atr_mult: float | None,
             warmup: int,
             fade: bool = False,
             usd_quote: bool = True,
             bar_hours: np.ndarray | None = None,
             bar_dows: np.ndarray | None = None,
             hour_filter: set | None = None,
             dow_filter: set | None = None) -> dict:
    """
    fade=False: trade WITH the breakout.  fade=True: fade (mean-revert) it.
    usd_quote=True:  pair is XXX/USD — P&L already in USD (EURUSD, GBPUSD…).
    usd_quote=False: pair is USD/YYY — raw P&L is in foreign currency;
                     divide by entry price to get USD (USDJPY, USDCHF…).
    """

    n = len(close)
    n_tp_l = n_sl_l = n_tp_s = n_sl_s = n_open = 0
    tp_pnl = sl_pnl = open_pnl = 0.0
    bars_tp: list[int] = []
    worst_dd_list: list[float] = []

    in_trade    = False
    direction   = 0        # +1 long, -1 short
    entry_price = 0.0
    tp_price    = 0.0
    sl_price    = 0.0
    entry_bar   = 0
    worst_low   = 0.0

    def _pnl(exit_price: float) -> float:
        raw = (exit_price - entry_price) * LOT_SIZE * direction
        usd = raw if usd_quote else raw / entry_price
        return usd - commission_usd

    for i in range(warmup, n):
        if np.isnan(range_high[i]):
            continue

        if in_trade:
            unrealized = _pnl(close[i])
            worst_low  = min(worst_low, unrealized)

            sl_hit = (low[i]  <= sl_price) if direction ==  1 else (high[i] >= sl_price)
            tp_hit = (high[i] >= tp_price) if direction ==  1 else (low[i]  <= tp_price)
            if sl_hit and tp_hit:
                sl_hit, tp_hit = True, False

            if sl_hit:
                pnl = _pnl(sl_price)
                sl_pnl += pnl
                if direction ==  1: n_sl_l += 1
                else:               n_sl_s += 1
                worst_dd_list.append(worst_low)
                in_trade = False
            elif tp_hit:
                pnl = _pnl(tp_price)
                tp_pnl += pnl
                if direction ==  1: n_tp_l += 1
                else:               n_tp_s += 1
                bars_tp.append(i - entry_bar)
                worst_dd_list.append(worst_low)
                in_trade = False

        else:
            if hour_filter is not None and bar_hours[i] not in hour_filter:
                continue
            if dow_filter is not None and bar_dows[i] not in dow_filter:
                continue
            # Tightness filter: skip wide ranges
            if tight_atr_mult is not None:
                if (range_high[i] - range_low[i]) >= tight_atr_mult * atr_vals[i]:
                    continue

            atr_e = atr_vals[i]

            if close[i] > range_high[i]:
                # Breakout above range: follow it (long) or fade it (short)
                in_trade    = True
                direction   = -1 if fade else 1
                entry_price = close[i]
                entry_bar   = i
                worst_low   = 0.0
                tp_price    = close[i] + tp_atr_mult * atr_e * direction
                sl_price    = close[i] - sl_atr_mult * atr_e * direction

            elif close[i] < range_low[i]:
                # Breakout below range: follow it (short) or fade it (long)
                in_trade    = True
                direction   = 1 if fade else -1
                entry_price = close[i]
                entry_bar   = i
                worst_low   = 0.0
                tp_price    = close[i] + tp_atr_mult * atr_e * direction
                sl_price    = close[i] - sl_atr_mult * atr_e * direction

    if in_trade:
        pnl = _pnl(close[n - 1])
        n_open += 1
        open_pnl += pnl
        worst_dd_list.append(min(worst_low, pnl))

    n_tp    = n_tp_l + n_tp_s
    n_sl    = n_sl_l + n_sl_s
    total   = n_tp + n_sl + n_open
    if total == 0:
        return {"total": 0}

    win_rate    = n_tp / (n_tp + n_sl) if (n_tp + n_sl) > 0 else 0.0
    net_pnl     = tp_pnl + sl_pnl + open_pnl
    avg_tp_bars = float(np.mean(bars_tp)) if bars_tp else 0.0
    worst_dd    = float(np.min(worst_dd_list)) if worst_dd_list else 0.0
    avg_dd      = float(np.mean(worst_dd_list)) if worst_dd_list else 0.0

    return {
        "total": total, "n_tp": n_tp, "n_sl": n_sl, "n_open": n_open,
        "n_tp_l": n_tp_l, "n_sl_l": n_sl_l,
        "n_tp_s": n_tp_s, "n_sl_s": n_sl_s,
        "win_rate": win_rate, "net_pnl": net_pnl,
        "tp_pnl": tp_pnl, "sl_pnl": sl_pnl,
        "avg_tp_pnl": tp_pnl / n_tp if n_tp else 0.0,
        "avg_sl_pnl": sl_pnl / n_sl if n_sl else 0.0,
        "avg_tp_bars": avg_tp_bars,
        "worst_dd": worst_dd, "avg_dd": avg_dd,
    }


# ── Prepare arrays ────────────────────────────────────────────────────────────

def _prepare(df: pd.DataFrame, atr_period: int,
             lookback: int) -> tuple:
    close       = df["Close"].values.astype(float)
    low         = df["Low"].values.astype(float)
    high        = df["High"].values.astype(float)
    atr_vals    = _atr(df, atr_period)
    range_high, range_low = _rolling_range(high, low, lookback)
    warmup      = atr_period + lookback + 1
    bar_hours   = df.index.hour.values.astype(np.int8)
    bar_dows    = df.index.dayofweek.values.astype(np.int8)  # 0=Mon…6=Sun
    return close, low, high, atr_vals, range_high, range_low, warmup, bar_hours, bar_dows


# ── Output ────────────────────────────────────────────────────────────────────

def _print_single(r: dict, args: argparse.Namespace,
                  df: pd.DataFrame, tf: str) -> None:
    if r["total"] == 0:
        print("  No trades triggered."); return

    tight_str = f"{args.tight_atr}×ATR" if args.tight_atr else "none"
    print(f"\n{'='*64}")
    mode = "range FADE (mean reversion)" if args.fade else "range breakout"
    print(f"  Strategy 2: {mode}  |  EUR/USD {tf}")
    print(f"  Range   : {args.lookback}-bar rolling high/low")
    print(f"  Filter  : range size < {tight_str}")
    print(f"  TP      : entry ± {args.tp_atr}×ATR  |  SL: entry ∓ {args.sl_atr}×ATR")
    print(f"  Comm    : ${args.commission:.1f}")
    print(f"  Data    : {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")
    print(f"{'='*64}")
    print(f"  Total trades  : {r['total']}")
    print(f"  TP hits       : {r['n_tp']}  ({r['win_rate']*100:.1f}%)")
    print(f"    Long  TP/SL : {r['n_tp_l']} / {r['n_sl_l']}")
    print(f"    Short TP/SL : {r['n_tp_s']} / {r['n_sl_s']}")
    print(f"  SL hits       : {r['n_sl']}  ({r['n_sl']/r['total']*100:.1f}%)")
    print(f"  Still open    : {r['n_open']}")
    if r["n_tp"]:
        print(f"\n  TP avg bars   : {r['avg_tp_bars']:.1f}"
              f"   TP avg PnL : ${r['avg_tp_pnl']:+.2f}")
    if r["n_sl"]:
        print(f"  SL avg PnL    : ${r['avg_sl_pnl']:+.2f}")
    print(f"\n  Worst unrealized (single trade) : ${r['worst_dd']:+.2f}")
    print(f"  Avg worst unrealized            : ${r['avg_dd']:+.2f}")
    print(f"\n  TP PnL  : ${r['tp_pnl']:+.2f}")
    if r["n_sl"]:
        print(f"  SL PnL  : ${r['sl_pnl']:+.2f}")
    print(f"  Net PnL : ${r['net_pnl']:+.2f}")


def _print_sweep(results: dict, mults: list[float],
                 args: argparse.Namespace, tf: str) -> None:
    tight_str = f"{args.tight_atr}×ATR" if args.tight_atr else "none"
    mode      = "FADE" if args.fade else "breakout"
    header    = (f"\n  EUR/USD {tf}  |  {args.lookback}-bar range {mode}"
                 f"  |  tightness: {tight_str}")
    col_w     = 10
    sl_labels = [f"SL {m}×" for m in mults]

    def _row(label: str, vals: list[str]) -> str:
        return f"  {label:<12}" + "".join(f"{v:>{col_w}}" for v in vals)

    for metric, title, fmt in [
        ("net_pnl",     "Net PnL ($)",  lambda v: f"${v:+.0f}"),
        ("win_rate",    "Win rate (%)", lambda v: f"{v*100:.1f}%"),
        ("total",       "# Trades",     lambda v: str(v)),
        ("avg_tp_bars", "Avg TP bars",  lambda v: f"{v:.1f}"),
    ]:
        print(header)
        print(f"  [{title}]")
        print(_row("", sl_labels))
        print(f"  {'-' * (12 + col_w * len(mults))}")
        for tp in mults:
            row_vals = []
            for sl in mults:
                r = results.get((tp, sl), {})
                row_vals.append(fmt(r[metric]) if r and r["total"] > 0 else "n/a")
            print(_row(f"TP {tp}×", row_vals))


def _print_multi_summary(data: dict, years: list[int], tfs: list[str],
                         args: argparse.Namespace) -> None:
    tight_str = f"{args.tight_atr}×ATR" if args.tight_atr else "none"
    config    = (f"{args.lookback}-bar range  TP {args.tp_atr}×  SL {args.sl_atr}×"
                 f"  tight:{tight_str}")
    col_w = 14

    def _row(label: str, vals: list[str]) -> str:
        return f"  {label:<8}" + "".join(f"{v:>{col_w}}" for v in vals)

    for metric, title, fmt in [
        ("net_pnl",  "Net PnL ($)",  lambda v: f"${v:+.0f}"),
        ("win_rate", "Win rate (%)", lambda v: f"{v*100:.1f}%"),
        ("total",    "# Trades",     lambda v: str(v)),
    ]:
        print(f"\n  [{title}]  {config}")
        print(_row("", tfs))
        print(f"  {'-' * (8 + col_w * len(tfs))}")
        for yr in years:
            row_vals = []
            for tf in tfs:
                r = data.get((yr, tf), {})
                row_vals.append(fmt(r[metric]) if r and r["total"] > 0 else "n/a")
            print(_row(str(yr), row_vals))


def _print_multi_sweep_summary(data: dict, years: list[int], tfs: list[str],
                                mults: list[float], args: argparse.Namespace) -> None:
    tight_str = f"{args.tight_atr}×ATR" if args.tight_atr else "none"
    col_w = 18

    def _row(label: str, vals: list[str]) -> str:
        return f"  {label:<8}" + "".join(f"{v:>{col_w}}" for v in vals)

    print(f"\n  [Best Net PnL across all TP/SL combos]"
          f"  {args.lookback}-bar range  tight:{tight_str}")
    print(_row("", tfs))
    print(f"  {'-' * (8 + col_w * len(tfs))}")
    for yr in years:
        row_vals = []
        for tf in tfs:
            best_pnl = None
            best_cfg = ""
            for tp in mults:
                for sl in mults:
                    r = data.get((yr, tf, tp, sl), {})
                    if r and r["total"] > 0 and (best_pnl is None or r["net_pnl"] > best_pnl):
                        best_pnl = r["net_pnl"]
                        best_cfg = f"tp{tp}sl{sl}"
            row_vals.append(f"${best_pnl:+.0f}({best_cfg})" if best_pnl is not None else "n/a")
        print(_row(str(yr), row_vals))


# ── Run modes ─────────────────────────────────────────────────────────────────

def _usd_quote(symbol: str) -> bool:
    return SYMBOLS.get(symbol, {}).get("usd_quote", True)


def run_single(df: pd.DataFrame, args: argparse.Namespace, tf: str) -> None:
    close, low, high, atr_vals, range_high, range_low, warmup, bh, bd = _prepare(
        df, args.atr_period, args.lookback)
    hf = set(args.hours) if args.hours else None
    df_ = set(args.days)  if args.days  else None
    r = simulate(close, low, high, atr_vals, range_high, range_low,
                 args.tp_atr, args.sl_atr, args.commission,
                 args.tight_atr, warmup, args.fade, _usd_quote(args.symbol),
                 bar_hours=bh, bar_dows=bd, hour_filter=hf, dow_filter=df_)
    _print_single(r, args, df, tf)


def run_sweep(df: pd.DataFrame, args: argparse.Namespace, tf: str) -> None:
    close, low, high, atr_vals, range_high, range_low, warmup, bh, bd = _prepare(
        df, args.atr_period, args.lookback)
    uq = _usd_quote(args.symbol)
    hf = set(args.hours) if args.hours else None
    df_ = set(args.days)  if args.days  else None
    results: dict[tuple, dict] = {}
    for tp in SWEEP_MULTS:
        for sl in SWEEP_MULTS:
            results[(tp, sl)] = simulate(
                close, low, high, atr_vals, range_high, range_low,
                tp, sl, args.commission, args.tight_atr, warmup, args.fade, uq,
                bar_hours=bh, bar_dows=bd, hour_filter=hf, dow_filter=df_)
    _print_sweep(results, SWEEP_MULTS, args, tf)


def _year_range(yr: int) -> tuple[str, str]:
    start = f"{yr}-01-01"
    end   = min(date(yr + 1, 1, 1), date.today()).isoformat()
    return start, end


def run_multi(args: argparse.Namespace) -> None:
    """Loop years × symbols, fixed tf=1m. Prints a year×symbol net PnL table."""
    symbols = args.pairs if args.pairs else list(SYMBOLS.keys())
    years   = args.years if args.years else YEARS
    tf      = (args.tf or ["1m"])[0]

    data: dict = {}
    for sym in symbols:
        uq = _usd_quote(sym)
        for yr in years:
            start, end = _year_range(yr)
            print(f"  Fetching {sym} {tf}  {start} ...", end=" ", flush=True)
            df = fetch_ohlcv(sym, tf, start, end)
            if df.empty:
                print("no data"); continue
            print(f"{len(df):,} bars")
            close, low, high, atr_vals, rh, rl, warmup, bh, bd = _prepare(
                df, args.atr_period, args.lookback)
            hf  = set(args.hours) if args.hours else None
            df_ = set(args.days)  if args.days  else None
            data[(sym, yr)] = simulate(
                close, low, high, atr_vals, rh, rl,
                args.tp_atr, args.sl_atr, args.commission,
                args.tight_atr, warmup, args.fade, uq,
                bar_hours=bh, bar_dows=bd, hour_filter=hf, dow_filter=df_)

    # Print tables: rows=years, cols=symbols
    col_w = 14

    def _row(label: str, vals: list[str]) -> str:
        return f"  {label:<8}" + "".join(f"{v:>{col_w}}" for v in vals)

    tight_str = f"{args.tight_atr}×ATR" if args.tight_atr else "none"
    config    = (f"{args.lookback}-bar range FADE  TP {args.tp_atr}×  SL {args.sl_atr}×"
                 f"  tight:{tight_str}  {tf}")
    for metric, title, fmt in [
        ("net_pnl",  "Net PnL ($)",  lambda v: f"${v:+.0f}"),
        ("win_rate", "Win rate (%)", lambda v: f"{v*100:.1f}%"),
        ("total",    "# Trades",     lambda v: str(v)),
    ]:
        print(f"\n  [{title}]  {config}")
        print(_row("", symbols))
        print(f"  {'-' * (8 + col_w * len(symbols))}")
        for yr in years:
            row_vals = []
            for sym in symbols:
                r = data.get((sym, yr), {})
                row_vals.append(fmt(r[metric]) if r and r["total"] > 0 else "n/a")
            print(_row(str(yr), row_vals))


# ── Time-of-day analysis ──────────────────────────────────────────────────────

def run_time_analysis(df: pd.DataFrame, args: argparse.Namespace, symbol: str) -> None:
    close, low, high, atr_vals, rh, rl, warmup, bar_hours, bar_dows = _prepare(
        df, args.atr_period, args.lookback)
    uq = _usd_quote(symbol)

    def _sim(hour_filter=None, dow_filter=None):
        return simulate(close, low, high, atr_vals, rh, rl,
                        args.tp_atr, args.sl_atr, args.commission,
                        args.tight_atr, warmup, args.fade, uq,
                        bar_hours=bar_hours, bar_dows=bar_dows,
                        hour_filter=hour_filter, dow_filter=dow_filter)

    DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    tight_str = f"{args.tight_atr}×ATR" if args.tight_atr else "none"
    header    = (f"\n  Time-of-day analysis  |  {symbol}  "
                 f"TP {args.tp_atr}×  SL {args.sl_atr}×  tight:{tight_str}"
                 f"  {args.lookback}-bar range")
    print(header)
    print(f"  Data: {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")

    print(f"\n  [UTC Hour of Day]")
    print(f"  {'Hour':<6}  {'Net PnL':>11}  {'Win%':>7}  {'Trades':>7}")
    print(f"  {'-'*38}")
    for h in range(24):
        r = _sim(hour_filter={h})
        if r["total"] == 0:
            print(f"  {h:02d}:xx   {'n/a':>11}  {'':>7}  {'0':>7}")
        else:
            print(f"  {h:02d}:xx   ${r['net_pnl']:>+10.0f}  "
                  f"{r['win_rate']*100:>6.1f}%  {r['total']:>7}")

    print(f"\n  [Day of Week (UTC)]")
    print(f"  {'Day':<6}  {'Net PnL':>11}  {'Win%':>7}  {'Trades':>7}")
    print(f"  {'-'*38}")
    for d in range(7):
        r = _sim(dow_filter={d})
        if r["total"] == 0:
            print(f"  {DOW_NAMES[d]:<6}  {'n/a':>11}  {'':>7}  {'0':>7}")
        else:
            print(f"  {DOW_NAMES[d]:<6}  ${r['net_pnl']:>+10.0f}  "
                  f"{r['win_rate']*100:>6.1f}%  {r['total']:>7}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start",       default="2024-01-01")
    ap.add_argument("--end",         default="2025-01-01")
    ap.add_argument("--symbol",      default="EURUSD",
                    help="Single symbol for non-multi runs (default: EURUSD)")
    ap.add_argument("--pairs",       nargs="+", default=None,
                    help="Symbols for --multi (default: all 7 USD pairs)")
    ap.add_argument("--tf",          nargs="+", default=None,
                    help="Timeframe(s) (default: 1m)")
    ap.add_argument("--atr-period",  type=int,   default=14)
    ap.add_argument("--lookback",    type=int,   default=20,
                    help="Bars to define the range (default 20)")
    ap.add_argument("--tight-atr",   type=float, default=None,
                    help="Only trade when range size < N×ATR (e.g. 2.0); omit = no filter")
    ap.add_argument("--commission",  type=float, default=7.0,
                    help="Round-trip commission USD (default 7 = 0.7 pip)")
    ap.add_argument("--tp-atr",      type=float, default=3.0,
                    help="TP in ATR multiples from entry (default 3)")
    ap.add_argument("--sl-atr",      type=float, default=2.0,
                    help="SL in ATR multiples from entry (default 2)")
    ap.add_argument("--fade",          action="store_true",
                    help="Fade the breakout: short when price breaks above range, long below")
    ap.add_argument("--sweep",         action="store_true",
                    help="Sweep SL/TP grid [1,2,3,5,10]×ATR")
    ap.add_argument("--multi",         action="store_true",
                    help="Run across years × timeframes")
    ap.add_argument("--years",         nargs="+", type=int, default=None,
                    help="Years to include in --multi (default: 2021-2024); partial years capped at today")
    ap.add_argument("--time-analysis", action="store_true",
                    help="Show per-UTC-hour and per-DOW PnL breakdown for --symbol over --start/--end")
    ap.add_argument("--hours",         nargs="+", type=int, default=None,
                    help="Restrict entries to these UTC hours (e.g. --hours 8 9 10 11)")
    ap.add_argument("--days",          nargs="+", type=int, default=None,
                    help="Restrict entries to these days-of-week 0=Mon…6=Sun (e.g. --days 0 1 2 3 4)")
    args = ap.parse_args()

    if args.multi:
        run_multi(args)
        return

    if args.time_analysis:
        tf = (args.tf or ["1m"])[0]
        print(f"Fetching {args.symbol} {tf}  {args.start} → {args.end} ...")
        df = fetch_ohlcv(args.symbol, tf, args.start, args.end)
        if df.empty:
            print("  No data returned."); return
        print(f"  {len(df):,} bars")
        run_time_analysis(df, args, args.symbol)
        return

    for tf in (args.tf or ["1m"]):
        print(f"Fetching {args.symbol} {tf}  {args.start} → {args.end} ...")
        df = fetch_ohlcv(args.symbol, tf, args.start, args.end)
        if df.empty:
            print("  No data returned."); continue
        print(f"  {len(df):,} bars")
        if args.sweep:
            run_sweep(df, args, tf)
        else:
            run_single(df, args, tf)


if __name__ == "__main__":
    main()
