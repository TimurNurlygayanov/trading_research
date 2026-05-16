"""
Strategy 006 — Breakout-pullback continuation, 1h, 7 majors.

LONG  signal at bar i:  argmin(Low [i-LOOKBACK : i+1]) == 0
                        i.e. lowest Low of the last 21 bars sits at the OLDEST
                        bar — no fresh low for LOOKBACK candles.
SHORT signal at bar i:  argmax(High[i-LOOKBACK : i+1]) == 0

Entry:  at Close[i].
SL:     at the anchor's Low (long) / High (short) — that 20-bars-ago extreme.
TP:     Entry ± 3 * |Entry - SL|.  Fixed 1:3 R:R.
Exit:   first intra-bar touch of TP or SL.  Same-bar conflict → SL wins.

One position per pair. After exit, re-enter on the next qualifying bar.

Period: 2025-01-01 .. 2026-05-15.  Spread 1 pip.
"""
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


_MT5_TF_MAP = {"1h": 16385}


def get_data(ticker, timeframe, start, end):
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end   = pd.Timestamp(end,   tz="UTC")
    mt5.initialize()
    mt5.symbol_select(ticker, True)

    tf_const = _MT5_TF_MAP[timeframe]
    CHUNK    = 50_000

    frames, pos = [], 0
    while True:
        chunk = mt5.copy_rates_from_pos(ticker, tf_const, pos, CHUNK)
        if chunk is None or len(chunk) == 0:
            break
        frames.append(pd.DataFrame(chunk))
        oldest = pd.Timestamp(int(frames[-1]["time"].min()), unit="s", tz="UTC")
        if oldest <= ts_start or len(chunk) < CHUNK:
            break
        pos += CHUNK

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames[::-1])
    df = df.drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(columns={"open": "Open", "high": "High",
                            "low": "Low", "close": "Close", "tick_volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


SPREAD_PIPS = 1.0
LOOKBACK_SWEEP = [10, 20, 30, 50]
RR_SWEEP       = [1.0, 2.0, 3.0, 5.0]
# Session filter: trade only during these broker hours (inclusive sets).
SESSION_HOURS = set(range(3, 11)) | set(range(17, 24))   # 03-10 and 17-23


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def entry_signals(df: pd.DataFrame, lookback: int):
    """Returns long_sig, short_sig, anchor_low, anchor_high arrays.

    long_sig[i]  True iff Low [i-lookback] is the min over Low [i-lookback..i]
                 AND df.index[i].hour is in SESSION_HOURS.
    short_sig[i] True iff High[i-lookback] is the max over High[i-lookback..i]
                 AND df.index[i].hour is in SESSION_HOURS.
    """
    low  = df["Low"].values
    high = df["High"].values
    hrs  = df.index.hour.to_numpy()
    n    = len(df)
    long_sig    = np.zeros(n, dtype=bool)
    short_sig   = np.zeros(n, dtype=bool)
    anchor_low  = np.full(n, np.nan)
    anchor_high = np.full(n, np.nan)

    sess_mask = np.array([h in SESSION_HOURS for h in hrs], dtype=bool)
    for i in range(lookback, n):
        if not sess_mask[i]:
            continue
        lo_win = low[i - lookback: i + 1]
        hi_win = high[i - lookback: i + 1]
        if np.argmin(lo_win) == 0:
            long_sig[i]   = True
            anchor_low[i] = lo_win[0]
        if np.argmax(hi_win) == 0:
            short_sig[i]   = True
            anchor_high[i] = hi_win[0]
    return long_sig, short_sig, anchor_low, anchor_high


def backtest(df: pd.DataFrame, direction: int,
             sig: np.ndarray, anchor: np.ndarray, pip: float,
             rr: float):
    """Event-driven. For each signal bar, find first intra-bar TP/SL touch.

    Returns (trades, n_open, sum_open_mtm_pips).
    """
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    n     = len(close)
    entry_ix = np.flatnonzero(sig)

    trades       = []
    n_open       = 0
    open_mtm     = 0.0
    next_allowed = 0

    for ei in entry_ix:
        if ei < next_allowed:
            continue
        if ei + 1 >= n:
            break
        entry_px = close[ei]
        sl_px    = anchor[ei]
        risk     = (entry_px - sl_px) * direction          # >0 for valid setup
        if not np.isfinite(risk) or risk <= 0:
            continue
        tp_px    = entry_px + direction * rr * risk

        fut_high = high[ei + 1:]
        fut_low  = low[ei + 1:]
        if direction == 1:
            tp_mask = fut_high >= tp_px
            sl_mask = fut_low  <= sl_px
        else:
            tp_mask = fut_low  <= tp_px
            sl_mask = fut_high >= sl_px

        exit_mask = tp_mask | sl_mask
        if not exit_mask.any():
            n_open   += 1
            open_mtm += (close[-1] - entry_px) * direction / pip
            next_allowed = ei + 1
            continue

        off  = int(np.argmax(exit_mask))
        ex_i = ei + 1 + off
        if sl_mask[off]:
            pips_gross = (sl_px - entry_px) * direction / pip      # = -risk/pip
            kind = "sl"
        else:
            pips_gross = (tp_px - entry_px) * direction / pip      # = +RR*risk/pip
            kind = "tp"

        trades.append({"pips":      float(pips_gross),
                       "hold":      ex_i - ei,
                       "exit":      kind,
                       "entry_i":   int(ei),
                       "dir":       direction,
                       "risk_pips": float(risk / pip)})
        next_allowed = ex_i + 1
    return trades, n_open, open_mtm


def summarize(trades, df):
    if not trades:
        return {"n": 0}
    pips_gross = np.array([t["pips"] for t in trades])
    pips_net   = pips_gross - SPREAD_PIPS
    wins       = pips_net > 0
    days       = max(1, (df.index[-1] - df.index[0]).days)
    sharpe     = (pips_net.mean() / pips_net.std()
                  * np.sqrt(len(trades) / (days / 365.25))
                  if len(trades) > 1 and pips_net.std() > 0 else 0.0)
    risks = [t["risk_pips"] for t in trades]
    return {
        "n":         len(trades),
        "pips_net":  float(pips_net.sum()),
        "win_pct":   float(wins.mean() * 100),
        "avg_win":   float(pips_net[wins].mean())  if wins.any()  else 0.0,
        "avg_loss":  float(pips_net[~wins].mean()) if (~wins).any() else 0.0,
        "avg_hold":  float(np.mean([t["hold"] for t in trades])),
        "sharpe":    float(sharpe),
        "n_tp":      sum(t["exit"] == "tp" for t in trades),
        "n_sl":      sum(t["exit"] == "sl" for t in trades),
        "avg_risk":  float(np.mean(risks)),
    }


PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
START = "2025-01-01"
END   = "2026-05-15"


def run_pair_combo(df: pd.DataFrame, pip: float, lookback: int, rr: float):
    long_sig, short_sig, a_low, a_high = entry_signals(df, lookback)
    longs_t,  o_l, mtm_l = backtest(df, +1, long_sig,  a_low,  pip, rr)
    shorts_t, o_s, mtm_s = backtest(df, -1, short_sig, a_high, pip, rr)
    trades = longs_t + shorts_t
    s = summarize(trades, df)
    s["n_open"]    = o_l + o_s
    s["open_mtm"]  = mtm_l + mtm_s
    s["n_long"]    = len(longs_t)
    s["n_short"]   = len(shorts_t)
    return {"summary": s, "trades": trades, "df": df}


# ----- run -----
print(f"Strategy 006 — breakout-pullback (lookback × R:R sweep)")
print(f"Period: {START} .. {END}")
print(f"Session filter: broker hours {sorted(SESSION_HOURS)} only")
print(f"Pairs: {', '.join(PAIRS)}\n")

# Load all pair data once
print("Loading data ...")
pair_data = {}
for pair in PAIRS:
    df = get_data(pair, "1h", START, END)
    if df.empty:
        print(f"  {pair}: missing data; skip")
        continue
    pair_data[pair] = (df, pip_size(pair))
    print(f"  {pair}: {len(df)} bars")

# Sweep
print(f"\nSWEEP AGGREGATE (across {len(pair_data)} pairs)")
print(f"{'Lookback':>8} {'R:R':>5} {'TotN':>5} {'TP':>4} {'SL':>4} "
      f"{'Win%':>6} {'NetP':>9} {'+MTM':>9} {'AvgHold':>7} {'+/-Pairs':>9}")
print("-" * 80)

best_combo = None
best_net   = -1e9
all_combo_results = {}   # (lookback, rr) -> {pair: result}

for lookback in LOOKBACK_SWEEP:
    for rr in RR_SWEEP:
        per_pair = {}
        for pair, (df, pip) in pair_data.items():
            per_pair[pair] = run_pair_combo(df, pip, lookback, rr)
        all_combo_results[(lookback, rr)] = per_pair

        tot_n   = sum(r["summary"]["n"]        for r in per_pair.values())
        tot_net = sum(r["summary"]["pips_net"] for r in per_pair.values())
        tot_mtm = sum(r["summary"]["open_mtm"] for r in per_pair.values())
        tot_tp  = sum(r["summary"]["n_tp"]     for r in per_pair.values())
        tot_sl  = sum(r["summary"]["n_sl"]     for r in per_pair.values())
        avg_hold = (sum(r["summary"]["avg_hold"] * r["summary"]["n"]
                        for r in per_pair.values()) / max(tot_n, 1))
        pos = sum(1 for r in per_pair.values() if r["summary"]["pips_net"] > 0)
        neg = sum(1 for r in per_pair.values() if r["summary"]["pips_net"] <= 0)
        win_pct = tot_tp / max(tot_n, 1) * 100

        print(f"{lookback:>8} {rr:>5.1f} {tot_n:>5} {tot_tp:>4} {tot_sl:>4} "
              f"{win_pct:>5.1f}% {tot_net:>+9.1f} {tot_net + tot_mtm:>+9.1f} "
              f"{avg_hold:>7.1f} {pos:>3}+/{neg:<3}-")

        if tot_net > best_net:
            best_net   = tot_net
            best_combo = (lookback, rr)

# Per-pair breakdown for best combo
lb, rr = best_combo
print(f"\nBEST COMBO: lookback={lb}, R:R 1:{rr:.0f}  →  net {best_net:+.1f} pips")
print(f"{'Pair':<8} {'N':>4} {'L/S':>7} {'TP':>4} {'SL':>4} {'NetP':>8} "
      f"{'Sharpe':>7} {'Win%':>6} {'AvgW':>6} {'AvgL':>7} {'Hold':>5} "
      f"{'AvgR':>5} | {'Open':>4}")
print("-" * 100)
for pair, res in all_combo_results[best_combo].items():
    s = res["summary"]
    if s.get("n", 0) == 0:
        print(f"{pair:<8} {'-':>4}")
        continue
    ls = f"{s['n_long']}/{s['n_short']}"
    print(f"{pair:<8} {s['n']:>4} {ls:>7} {s['n_tp']:>4} {s['n_sl']:>4} "
          f"{s['pips_net']:>+8.1f} {s['sharpe']:>+7.2f} {s['win_pct']:>5.1f}% "
          f"{s['avg_win']:>+6.1f} {s['avg_loss']:>+7.1f} {s['avg_hold']:>5.1f} "
          f"{s['avg_risk']:>5.1f} | {s['n_open']:>4}")
