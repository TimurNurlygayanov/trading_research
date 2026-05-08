"""
Fetch today's 1m bars from a live MT5 terminal and replay pure-fade strategy.

Usage
-----
  python -m scripts.fetch_mt5_today
  python -m scripts.fetch_mt5_today --login 12345678 --password xxx --server FTMO-Demo
  python -m scripts.fetch_mt5_today --mt5-path "C:/Program Files/MetaTrader 5/terminal64.exe"
  python -m scripts.fetch_mt5_today --output today_1m.csv   # also save raw bars to CSV
  python -m scripts.fetch_mt5_today --no-replay             # just fetch & print, no simulation
  python -m scripts.fetch_mt5_today --symbol-suffix .raw    # for brokers that append suffixes
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, timezone, datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIRS = ["EURUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD"]
LOOKBACK   = 10
ATR_PERIOD = 14
EMA_PERIOD = 20
# Need at least lookback + atr_period + ema_period + 1 warmup bars before today
WARMUP_BARS = max(LOOKBACK + ATR_PERIOD + LOOKBACK, LOOKBACK + EMA_PERIOD + 10)
# Fetch enough bars to cover warmup + full trading day (1440 min = 24h)
FETCH_COUNT = WARMUP_BARS + 1500


def _detect_tz_offset(mt5, sym_mt5: str) -> int:
    """
    Detect broker server UTC offset from the most recent tick timestamp.
    Returns integer hours (e.g. 2 for EET winter, 3 for EET summer).
    """
    tick = mt5.symbol_info_tick(sym_mt5)
    if tick is None:
        return 0
    broker_ts = tick.time                          # broker-local unix seconds
    utc_ts    = datetime.now(tz=timezone.utc).timestamp()
    diff      = broker_ts - utc_ts
    return round(diff / 3600)


def fetch_today_bars(mt5, sym_mt5: str, tz_offset: int) -> pd.DataFrame | None:
    """
    Fetch last FETCH_COUNT 1m bars and return a UTC-indexed DataFrame
    containing ONLY bars from today (UTC date).
    """
    rates = mt5.copy_rates_from_pos(sym_mt5, mt5.TIMEFRAME_M1, 0, FETCH_COUNT)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    # Broker timestamps are in broker-local time; subtract offset to get UTC
    broker_dt = pd.to_datetime(df["time"], unit="s", utc=True)
    utc_dt    = broker_dt - pd.Timedelta(hours=tz_offset)
    df.index  = utc_dt
    df = df.rename(columns={
        "open":  "Open", "high": "High",
        "low":   "Low",  "close": "Close", "tick_volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df[df.index.date == date.today()]
    return df if len(df) > 0 else None


def fetch_bars_with_warmup(mt5, sym_mt5: str, tz_offset: int) -> pd.DataFrame | None:
    """
    Fetch FETCH_COUNT bars and return ALL (warmup + today) for simulation.
    """
    rates = mt5.copy_rates_from_pos(sym_mt5, mt5.TIMEFRAME_M1, 0, FETCH_COUNT)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    broker_dt = pd.to_datetime(df["time"], unit="s", utc=True)
    utc_dt    = broker_dt - pd.Timedelta(hours=tz_offset)
    df.index  = utc_dt
    df = df.rename(columns={
        "open":  "Open", "high": "High",
        "low":   "Low",  "close": "Close", "tick_volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]]


def run_fade_replay(pairs_data: dict[str, pd.DataFrame]) -> None:
    """Replay pure-fade on today's bars using the backtest engine."""
    from scripts.strategy1_regime_backtest import (
        simulate_pair, regime_pure_fade,
    )

    today = date.today()
    all_trades: list[dict] = []

    print("\n=== PER-PAIR (today only) ===")
    for pair, df in pairs_data.items():
        trades = simulate_pair(
            pair, df,
            lookback=LOOKBACK, atr_period=ATR_PERIOD, ema_period=EMA_PERIOD,
            tp_atr_fade=1.0, sl_atr_fade=2.0, tp_atr_break=2.0, sl_atr_break=1.0,
            lots=1.0, tight_atr=None,
            regime_fn=regime_pure_fade, regime_state_factory=None,
        )
        today_t = [t for t in trades if t["time"].date() == today]
        all_trades.extend(today_t)

        n   = len(today_t)
        if n == 0:
            print(f"  {pair}: no trades")
            continue
        wr  = sum(1 for t in today_t if t["won"]) / n * 100
        pnl = sum(t["net"] for t in today_t)
        print(f"  {pair}: {n:3d} trades  win={wr:4.0f}%  pnl=${pnl:8.1f}")

    if not all_trades:
        print("  No trades today across all pairs.")
        return

    nets = np.array([t["net"] for t in all_trades])
    won  = np.array([t["won"] for t in all_trades])

    print(f"\n=== SUMMARY — pure fade, {date.today()} ===")
    print(f"  Trades:    {len(all_trades)}")
    print(f"  Win rate:  {won.mean()*100:.1f}%  ({won.sum()} W / {(~won).sum()} L)")
    print(f"  Avg win:   ${nets[won].mean():.2f}" if won.any() else "  Avg win:   n/a")
    print(f"  Avg loss:  ${nets[~won].mean():.2f}" if (~won).any() else "  Avg loss:  n/a")
    print(f"  Net P&L:   ${nets.sum():,.2f}")
    print(f"  EV/trade:  ${nets.mean():.2f}")

    print(f"\n=== HOURLY BREAKDOWN ===")
    print(f"  Hour  | Trades | Win%  | P&L")
    print(f"  ------+--------+-------+----------")
    by_hour: dict[int, list] = {}
    for t in all_trades:
        by_hour.setdefault(t["time"].hour, []).append(t)
    for h in sorted(by_hour):
        g    = by_hour[h]
        wr_h = sum(1 for t in g if t["won"]) / len(g) * 100
        pnl_h = sum(t["net"] for t in g)
        flag = "  <<< trend hour" if wr_h < 50 else ""
        print(f"  {h:02d}h   | {len(g):6d} | {wr_h:4.0f}% | ${pnl_h:8.1f}{flag}")

    # Circuit breaker simulation
    print(f"\n=== CIRCUIT BREAKER — 6/10 losses -> N-hour pause ===")
    from scripts.strategy1_regime_backtest import simulate_pair as _sp
    print(f"  {'Break':8s} | Trades | Skipped | Win%  | P&L")
    print(f"  ---------+--------+---------+-------+----------")
    for break_hours in [1, 2, 3]:
        cb_trades: list[dict] = []
        for pair, df in pairs_data.items():
            trades = _sp(
                pair, df,
                lookback=LOOKBACK, atr_period=ATR_PERIOD, ema_period=EMA_PERIOD,
                tp_atr_fade=1.0, sl_atr_fade=2.0, tp_atr_break=2.0, sl_atr_break=1.0,
                lots=1.0, tight_atr=None,
                regime_fn=regime_pure_fade, regime_state_factory=None,
                cb_window=10, cb_min_losses=6, cb_break_bars=break_hours * 60,
            )
            cb_trades.extend(t for t in trades if t["time"].date() == today)
        n_cb     = len(cb_trades)
        skipped  = len(all_trades) - n_cb
        wr_cb    = sum(1 for t in cb_trades if t["won"]) / n_cb * 100 if n_cb else 0
        pnl_cb   = sum(t["net"] for t in cb_trades)
        print(f"  {break_hours}h break  | {n_cb:6d} | {skipped:7d} | {wr_cb:4.0f}% | ${pnl_cb:8.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mt5-path",      default=None,  help="Path to terminal64.exe")
    ap.add_argument("--login",         default=None,  type=int)
    ap.add_argument("--password",      default=None)
    ap.add_argument("--server",        default=None)
    ap.add_argument("--symbol-suffix", default="",    help="Appended to symbol names (e.g. .raw)")
    ap.add_argument("--output",        default=None,  help="Save today bars to CSV path")
    ap.add_argument("--no-replay",     action="store_true", default=False,
                    help="Skip fade simulation, just print bar counts")
    ap.add_argument("--pairs",         nargs="+", default=None)
    args = ap.parse_args()

    pairs = args.pairs or PAIRS

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 package not installed.  pip install MetaTrader5")
        sys.exit(1)

    # Connect
    kwargs: dict = {}
    if args.mt5_path:
        kwargs["path"] = args.mt5_path
    if not mt5.initialize(**kwargs):
        print(f"ERROR: mt5.initialize() failed: {mt5.last_error()}")
        sys.exit(1)

    if args.login is not None:
        ok = mt5.login(login=args.login, password=args.password or "",
                       server=args.server or "")
        if not ok:
            print(f"ERROR: mt5.login() failed: {mt5.last_error()}")
            mt5.shutdown()
            sys.exit(1)

    info = mt5.terminal_info()
    account = mt5.account_info()
    print(f"Connected: {info.name}  |  "
          f"account #{account.login if account else '?'}  "
          f"({account.server if account else '?'})")

    # Detect broker timezone offset from first available symbol
    first_sym = (pairs[0] + args.symbol_suffix) if pairs else "EURUSD"
    tz_offset = _detect_tz_offset(mt5, first_sym)
    print(f"Broker UTC offset: {tz_offset:+d}h  (today UTC = {date.today()})\n")

    # Fetch data
    today_dfs:  dict[str, pd.DataFrame] = {}   # today only (for display / CSV)
    full_dfs:   dict[str, pd.DataFrame] = {}   # warmup + today (for simulation)
    csv_frames: list[pd.DataFrame] = []

    print("=== BAR COUNTS — today ===")
    for pair in pairs:
        sym_mt5 = pair + args.symbol_suffix
        mt5.symbol_select(sym_mt5, True)

        today_df = fetch_today_bars(mt5, sym_mt5, tz_offset)
        full_df  = fetch_bars_with_warmup(mt5, sym_mt5, tz_offset)

        if today_df is None or full_df is None:
            print(f"  {pair}: no data (symbol={sym_mt5})")
            continue

        today_dfs[pair] = today_df
        full_dfs[pair]  = full_df
        print(f"  {pair}: {len(today_df):4d} bars today  "
              f"({today_df.index[0].strftime('%H:%M')} — "
              f"{today_df.index[-1].strftime('%H:%M')} UTC)  "
              f"[{len(full_df)} total loaded for warmup]")

        if args.output:
            tmp = today_df.copy()
            tmp["pair"] = pair
            csv_frames.append(tmp)

    if args.output and csv_frames:
        out = pd.concat(csv_frames).sort_index()
        out.to_csv(args.output)
        print(f"\nSaved {len(out)} bars to {args.output}")

    mt5.shutdown()

    if not args.no_replay and full_dfs:
        run_fade_replay(full_dfs)


if __name__ == "__main__":
    main()
