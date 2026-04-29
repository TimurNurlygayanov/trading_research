"""
EMA pullback after cross — 5M EURUSD US-session strategy.

Long signal (mirror for short)
  1. Bullish cross: EMA20 crosses above EMA50 → "long armed"
  2. First in-session bar after the cross whose range touches the EMA zone
     (bar.low ≤ max(EMA20, EMA50)  AND  bar.high ≥ min(EMA20, EMA50))
     → BUY at the next bar's open. Disarm long until next cross.
  3. SL = entry − sl_atr × ATR(14)        # default 2.0
     TP = entry + tp_atr × ATR(14)        # default 2.0

Note on the spec
  Original wording said "EMA 20 < EMA 50, ... after EMA 20 crossed over EMA 50",
  which is internally inconsistent (a bullish cross of 20 over 50 means 20 > 50).
  This script uses the canonical reading: long after the bullish cross while
  EMA20 > EMA50. Reverse for shorts.

What the script produces
  • all_signals_*.csv         every signal independently simulated → features
                              + outcome label. This is the dataset for the
                              Phase-2 Random Forest filter.
  • sequential_trades_*.csv   one-position-at-a-time realistic backtest stats
                              (drops signals that fire while a trade is open).
  • Console report             summary of both views.

Phase 2 (separate)
  Train RandomForestClassifier on (ema20_slope, ema50_slope, rsi, atr)
  → predict tp_hit on 2021-2025 signals; apply as filter to 2026 signals.

Usage
  python -m scripts.ema_pullback_backtest
  python -m scripts.ema_pullback_backtest --start 2021-01-01 --end 2025-12-31
  python -m scripts.ema_pullback_backtest --commission 0.00005 --max-bars 300
  python -m scripts.ema_pullback_backtest --long-only
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv

PIP = 0.0001  # EURUSD pip value


# ── feature + signal computation ────────────────────────────────────────────
def add_features(df: pd.DataFrame, fast: int, slow: int,
                 atr_len: int, rsi_len: int, slope_n: int) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ta.ema(out["Close"], length=fast)
    out["ema50"] = ta.ema(out["Close"], length=slow)
    out["atr"]   = ta.atr(out["High"], out["Low"], out["Close"], length=atr_len)
    out["rsi"]   = ta.rsi(out["Close"], length=rsi_len)
    out["ema20_slope_bps"] = (out["ema20"] / out["ema20"].shift(slope_n) - 1) * 10_000
    out["ema50_slope_bps"] = (out["ema50"] / out["ema50"].shift(slope_n) - 1) * 10_000
    return out


def detect_signals(df: pd.DataFrame, session_start: int, session_end: int,
                   include_short: bool = True) -> pd.DataFrame:
    """One-pass state machine: every (cross → first zone touch in session) event."""
    e20, e50  = df["ema20"].values, df["ema50"].values
    high, low = df["High"].values,  df["Low"].values
    atr_v     = df["atr"].values
    rsi_v     = df["rsi"].values
    e20s      = df["ema20_slope_bps"].values
    e50s      = df["ema50_slope_bps"].values
    hours     = df.index.hour.values
    n = len(df)

    bull_cross = (e20[1:] > e50[1:]) & (e20[:-1] <= e50[:-1])
    bear_cross = (e20[1:] < e50[1:]) & (e20[:-1] >= e50[:-1])
    bull_cross = np.concatenate([[False], bull_cross])
    bear_cross = np.concatenate([[False], bear_cross])

    armed_long = False
    armed_short = False
    rows = []
    for i in range(n):
        # Regime updates on cross
        if bull_cross[i]:
            armed_long, armed_short = True, False
        if bear_cross[i]:
            armed_short, armed_long = True, False

        # Need valid indicators + in-session
        if (np.isnan(e20[i]) or np.isnan(e50[i]) or np.isnan(atr_v[i])
            or np.isnan(rsi_v[i]) or atr_v[i] <= 0):
            continue
        if not (session_start <= hours[i] < session_end):
            continue

        # Zone-touch check (works regardless of which EMA is higher)
        zone_top = max(e20[i], e50[i])
        zone_bot = min(e20[i], e50[i])
        touches  = (low[i] <= zone_top) and (high[i] >= zone_bot)
        if not touches:
            continue

        if armed_long:
            rows.append((i, "long", e20[i], e50[i], e20s[i], e50s[i],
                         rsi_v[i], atr_v[i], int(hours[i])))
            armed_long = False
        elif armed_short and include_short:
            rows.append((i, "short", e20[i], e50[i], e20s[i], e50s[i],
                         rsi_v[i], atr_v[i], int(hours[i])))
            armed_short = False

    sig = pd.DataFrame(rows, columns=[
        "bar_idx", "side", "ema20", "ema50",
        "ema20_slope_bps", "ema50_slope_bps", "rsi", "atr", "session_hour"])
    sig["signal_time"] = df.index[sig["bar_idx"].values] if not sig.empty else pd.NaT
    return sig


# ── per-signal forward simulation ────────────────────────────────────────────
def simulate_signal(df: pd.DataFrame, sig_idx: int, side: str,
                    atr: float, sl_atr: float, tp_atr: float,
                    max_bars: int) -> dict:
    """
    Walk forward from sig_idx+1 (entry on the next bar's open) until SL or TP hits,
    or max_bars reached. Conservative on ambiguous bars (SL wins if both touched).
    """
    n = len(df)
    if sig_idx + 1 >= n:
        return dict(outcome="no_entry", entry_price=np.nan, exit_price=np.nan,
                    sl_price=np.nan, tp_price=np.nan,
                    bars_held=0, pnl_pips=0.0, exit_idx=-1, tp_hit=np.nan)

    entry = float(df["Open"].iloc[sig_idx + 1])
    if side == "long":
        sl_price = entry - sl_atr * atr
        tp_price = entry + tp_atr * atr
    else:
        sl_price = entry + sl_atr * atr
        tp_price = entry - tp_atr * atr

    end = min(sig_idx + 1 + max_bars, n)
    high = df["High"].values
    low  = df["Low"].values

    for j in range(sig_idx + 1, end):
        h, l = high[j], low[j]
        if side == "long":
            hit_sl, hit_tp = (l <= sl_price), (h >= tp_price)
        else:
            hit_sl, hit_tp = (h >= sl_price), (l <= tp_price)

        if hit_sl:        # conservative on overlap — SL wins
            pnl = (sl_price - entry) if side == "long" else (entry - sl_price)
            return dict(outcome="sl", entry_price=entry, exit_price=sl_price,
                        sl_price=sl_price, tp_price=tp_price,
                        bars_held=j - sig_idx, pnl_pips=pnl / PIP,
                        exit_idx=j, tp_hit=0)
        if hit_tp:
            pnl = (tp_price - entry) if side == "long" else (entry - tp_price)
            return dict(outcome="tp", entry_price=entry, exit_price=tp_price,
                        sl_price=sl_price, tp_price=tp_price,
                        bars_held=j - sig_idx, pnl_pips=pnl / PIP,
                        exit_idx=j, tp_hit=1)

    last_close = float(df["Close"].iloc[end - 1])
    pnl = (last_close - entry) if side == "long" else (entry - last_close)
    return dict(outcome="timeout", entry_price=entry, exit_price=last_close,
                sl_price=sl_price, tp_price=tp_price,
                bars_held=end - 1 - sig_idx, pnl_pips=pnl / PIP,
                exit_idx=end - 1, tp_hit=np.nan)


def simulate_all(df: pd.DataFrame, signals: pd.DataFrame,
                 sl_atr: float, tp_atr: float, max_bars: int,
                 commission_pips: float) -> pd.DataFrame:
    if signals.empty:
        return signals
    out = []
    for r in signals.itertuples(index=False):
        sim = simulate_signal(df, r.bar_idx, r.side, r.atr,
                              sl_atr, tp_atr, max_bars)
        sim["pnl_pips_net"] = sim["pnl_pips"] - 2 * commission_pips  # round-trip
        out.append({**r._asdict(), **sim})
    return pd.DataFrame(out)


# ── sequential (realistic) backtest ─────────────────────────────────────────
def sequential_take(simulated: pd.DataFrame) -> pd.DataFrame:
    """Drop signals that fire while an earlier trade is still open."""
    if simulated.empty:
        return simulated
    s = simulated.sort_values("bar_idx").reset_index(drop=True)
    last_exit = -1
    keep = []
    for r in s.itertuples():
        if r.bar_idx <= last_exit:
            continue
        keep.append(r.Index)
        last_exit = max(last_exit, r.exit_idx)
    return s.loc[keep].reset_index(drop=True)


def report(name: str, trades: pd.DataFrame, df_len: int, tf: str):
    if trades.empty:
        print(f"\n── {name}: 0 trades ──")
        return
    n   = len(trades)
    pnl = trades["pnl_pips_net"].sum()
    avg = trades["pnl_pips_net"].mean()
    win = (trades["tp_hit"] == 1).sum()
    los = (trades["tp_hit"] == 0).sum()
    out = (trades["tp_hit"].isna()).sum()
    win_rate = win / max(n, 1) * 100

    eq = trades["pnl_pips_net"].cumsum()
    dd = (eq - eq.cummax()).min()

    # Per-trade Sharpe (project convention)
    bpy = bars_per_year_for_tf(tf)
    tpy = n * bpy / max(df_len, 1)
    r = trades["pnl_pips_net"]
    tsr = float(r.mean() / r.std() * np.sqrt(max(tpy, 1))) if r.std() > 0 else 0.0

    print(f"\n── {name} ──")
    print(f"  Signals/Trades : {n}")
    print(f"  TP hit         : {win}  ({win_rate:.1f}%)")
    print(f"  SL hit         : {los}")
    print(f"  Timeout        : {out}")
    print(f"  Total P&L      : {pnl:+.1f} pips")
    print(f"  Avg per trade  : {avg:+.2f} pips")
    print(f"  Max DD (pips)  : {dd:+.1f}")
    print(f"  Trade Sharpe   : {tsr:+.3f}")


def bars_per_year_for_tf(tf: str) -> float:
    tf = tf.lower().strip()
    n, unit = int(tf[:-1]), tf[-1]
    minutes = {"m": n, "h": n * 60, "d": n * 1440}[unit]
    return 365 * 24 * 60 / minutes


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbol",        default="EURUSD")
    ap.add_argument("--tf",            default="5m")
    ap.add_argument("--start",         default="2021-01-01")
    ap.add_argument("--end",           default="2025-12-31")
    ap.add_argument("--fast",          type=int,   default=20)
    ap.add_argument("--slow",          type=int,   default=50)
    ap.add_argument("--atr-len",       type=int,   default=14)
    ap.add_argument("--rsi-len",       type=int,   default=14)
    ap.add_argument("--slope-n",       type=int,   default=5)
    ap.add_argument("--sl-atr",        type=float, default=2.0)
    ap.add_argument("--tp-atr",        type=float, default=2.0)
    ap.add_argument("--max-bars",      type=int,   default=500,
                    help="forward simulation horizon (timeout if SL/TP not hit)")
    ap.add_argument("--session-start", type=int,   default=13)
    ap.add_argument("--session-end",   type=int,   default=21)
    ap.add_argument("--commission",    type=float, default=0.00005,
                    help="per-side commission as price fraction (0.00005 = 0.5 pip)")
    ap.add_argument("--long-only",     action="store_true")
    ap.add_argument("--side-policy",   default="as-signaled",
                    choices=("as-signaled", "flip-all", "flip-long-only", "flip-short-only"),
                    help="map detected signal side to actual trade side. "
                         "'flip-all' inverts every signal; 'flip-long-only' inverts "
                         "only the bullish-cross-pullback signal (keeps shorts as-is); "
                         "'flip-short-only' is the mirror.")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent

    print(f"Fetching {args.symbol} {args.tf}  {args.start} → {args.end}")
    raw = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  fetched {len(raw):,} bars\n")
    if raw.empty:
        return

    print("Computing features and signals…")
    df = add_features(raw, args.fast, args.slow, args.atr_len, args.rsi_len, args.slope_n)
    sigs = detect_signals(df, args.session_start, args.session_end,
                          include_short=not args.long_only)
    print(f"  signals: {len(sigs)}  (long={ (sigs['side']=='long').sum() }, "
          f"short={ (sigs['side']=='short').sum() })")

    if sigs.empty:
        print("No signals — aborting.")
        return

    # ── apply side policy ──────────────────────────────────────────────────
    if args.side_policy != "as-signaled":
        flip_long  = args.side_policy in ("flip-all", "flip-long-only")
        flip_short = args.side_policy in ("flip-all", "flip-short-only")
        new_side = sigs["side"].copy()
        if flip_long:
            new_side[sigs["side"] == "long"]  = "short"
        if flip_short:
            new_side[sigs["side"] == "short"] = "long"
        n_flipped = (new_side != sigs["side"]).sum()
        sigs["side"] = new_side
        print(f"  side policy: {args.side_policy}  → {n_flipped} signals flipped")

    commission_pips = args.commission / PIP  # convert to pips per side
    print(f"\nSimulating each signal independently (max_bars={args.max_bars}, "
          f"SL={args.sl_atr}×ATR, TP={args.tp_atr}×ATR, "
          f"cost={2*commission_pips:.2f} pips round-trip)")

    sim = simulate_all(df, sigs, args.sl_atr, args.tp_atr,
                       args.max_bars, commission_pips)

    seq = sequential_take(sim)

    # ── reports ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"RESULTS — {args.symbol} {args.tf}  {args.start} → {args.end}")
    print(f"  EMA({args.fast})/EMA({args.slow})  ATR({args.atr_len})  "
          f"RSI({args.rsi_len})  session={args.session_start:02d}-{args.session_end:02d}UTC")
    print("=" * 80)
    report("ALL signals (independent, for RF training)", sim, len(df), args.tf)
    report("SEQUENTIAL (realistic, one trade at a time)", seq, len(df), args.tf)

    # ── breakdown by side ────────────────────────────────────────────────
    if not sim.empty:
        print("\nBy side (all signals):")
        for side in ("long", "short"):
            sub = sim[sim["side"] == side]
            if sub.empty:
                continue
            wr = (sub["tp_hit"] == 1).mean() * 100
            print(f"  {side:5s}  n={len(sub):4d}  win%={wr:5.1f}  "
                  f"pnl={sub['pnl_pips_net'].sum():+8.1f}  "
                  f"avg={sub['pnl_pips_net'].mean():+5.2f}")

    # ── save datasets ────────────────────────────────────────────────────
    suffix = "" if args.side_policy == "as-signaled" else f"_{args.side_policy}"
    sig_path = out_dir / f"all_signals_{args.symbol}_{args.tf}{suffix}.csv"
    seq_path = out_dir / f"sequential_trades_{args.symbol}_{args.tf}{suffix}.csv"
    sim.drop(columns=["bar_idx"], errors="ignore").to_csv(sig_path, index=False)
    seq.drop(columns=["bar_idx"], errors="ignore").to_csv(seq_path, index=False)
    print(f"\n→ saved {sig_path}  ({len(sim)} signals — RF training set)")
    print(f"→ saved {seq_path}  ({len(seq)} taken trades — realistic backtest)")
    print("\nFor Phase 2: load all_signals_*.csv, train RandomForest with features")
    print("  (ema20_slope_bps, ema50_slope_bps, rsi, atr) → tp_hit, "
          "split on 2026-01-01.")


if __name__ == "__main__":
    main()
