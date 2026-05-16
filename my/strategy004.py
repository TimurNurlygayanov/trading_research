"""
Strategy 004 — MACD crossover entries with higher-timeframe confirmation.

Entry (long):  MACD(12,26,9) line crosses ABOVE signal on the entry timeframe.
Entry (short): MACD(12,26,9) line crosses BELOW signal on the entry timeframe.

Exit (first rule to fire, both intra-bar, both ATR-scaled):
  1) TP at Entry ± 1·ATR(14 @ entry bar) → fill at the TP price (+1·ATR gross)
  2) SL at Entry ∓ 3·ATR(14 @ entry bar) → fill at the stop price (-3·ATR gross)
  Same-bar conflict: SL wins (conservative — stop fires before the close).

Trades that hit neither by end-of-data are reported as `Open` with their MTM,
so the picture stays honest about hidden drawdown.

Variants per pair:
  A. 5m raw                  (no HTF filter)
  B. 5m + 1h MACD direction  (long needs 1h MACD > Signal, short needs <)
  C. 1h raw                  (no HTF filter)
  D. 1h + daily MACD direction

HTF state is taken from the *previously closed* HTF bar — shift(1) before forward-
filling onto the entry-TF index — so the filter cannot peek into an open HTF bar.

Period: 2025-01-01 .. 2026-05-14.
Pairs:  EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD.
"""
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


_MT5_TF_MAP = {
    "5m": 5,
    "1h": 16385,
    "1d": 16408,
}


def get_data(ticker, timeframe, start, end):
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end   = pd.Timestamp(end,   tz="UTC")
    mt5.initialize()

    tf_const = _MT5_TF_MAP[timeframe]
    mt5.symbol_select(ticker, True)
    CHUNK = 50_000

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
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIG    = 9
ATR_LEN     = 14
# (tp_mult, sl_mult) — breakeven win% = sl / (tp + sl)
RR_SWEEP = [
    (1.0, 1.0),   # 1:1   BE 50%
    (1.0, 2.0),   # 1:2   BE 67%
    (1.0, 3.0),   # 1:3   BE 75%
    (2.0, 2.0),   # 1:1   BE 50% (wider)
    (2.0, 3.0),   # 2:3   BE 60%
]
SESSION_HOUR_LO = 3    # inclusive — London open
SESSION_HOUR_HI = 14   # inclusive — early NY


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    macd = ta.macd(df["Close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIG)
    df["macd"]  = macd[f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIG}"]
    df["macds"] = macd[f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIG}"]
    return df


def add_atr(df: pd.DataFrame, length: int = ATR_LEN) -> pd.DataFrame:
    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=length)
    return df


def macd_crosses(df: pd.DataFrame):
    m, s = df["macd"].values, df["macds"].values
    prev_m = np.concatenate([[np.nan], m[:-1]])
    prev_s = np.concatenate([[np.nan], s[:-1]])
    valid  = np.isfinite(prev_m) & np.isfinite(prev_s) & np.isfinite(m) & np.isfinite(s)
    cross_up   = valid & (prev_m <= prev_s) & (m > s)
    cross_down = valid & (prev_m >= prev_s) & (m < s)
    return cross_up, cross_down


def htf_bullish(htf_df: pd.DataFrame, ltf_index: pd.DatetimeIndex) -> np.ndarray:
    """For each LTF bar, is the previously-closed HTF bar in MACD>Signal state?"""
    htf = add_macd(htf_df.copy())
    bull = (htf["macd"] > htf["macds"]).astype(int).shift(1).fillna(0)
    aligned = bull.reindex(ltf_index, method="ffill").fillna(0)
    return aligned.to_numpy(dtype=bool)


def backtest(df: pd.DataFrame, direction: int,
             entry_sig: np.ndarray, pip: float,
             tp_atr_mult: float, sl_atr_mult: float,
             htf_ok: np.ndarray | None = None):
    """Event-driven 1h backtest with ATR-scaled TP/SL.

    TP/SL are sized per-entry from ATR(14) at the entry bar (no lookahead — that
    bar has just closed when we enter at its Close). Both exits are intra-bar.
    Same-bar SL/TP conflict: SL wins.

    Returns (trades, n_open, sum_open_mtm_pips).
    """
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    atr   = df["atr"].values
    n     = len(close)
    sig   = entry_sig & htf_ok if htf_ok is not None else entry_sig
    sig   = sig & np.isfinite(atr)
    entry_ix = np.flatnonzero(sig)

    trades: list[dict] = []
    n_open = 0
    open_mtm = 0.0
    next_allowed = 0

    for ei in entry_ix:
        if ei < next_allowed:
            continue
        if ei + 1 >= n:
            break
        entry_px = close[ei]
        atr_e    = atr[ei]
        tp_px    = entry_px + direction * tp_atr_mult * atr_e
        sl_px    = entry_px - direction * sl_atr_mult * atr_e

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
            n_open += 1
            open_mtm += (close[-1] - entry_px) * direction / pip
            next_allowed = ei + 1
            continue

        off  = int(np.argmax(exit_mask))
        ex_i = ei + 1 + off

        if sl_mask[off]:
            pips_gross, kind = -sl_atr_mult * atr_e / pip, "sl"
        else:
            pips_gross, kind = tp_atr_mult * atr_e / pip, "tp"

        trades.append({"pips":      float(pips_gross),
                       "hold":      ex_i - ei,
                       "exit":      kind,
                       "entry_i":   int(ei),
                       "dir":       direction,
                       "atr_pips":  float(atr_e / pip)})
        next_allowed = ex_i + 1
    return trades, n_open, open_mtm


def summarize(trades: list, df: pd.DataFrame) -> dict:
    if not trades:
        return {"n": 0}
    pips_gross = np.array([t["pips"] for t in trades])
    pips_net   = pips_gross - SPREAD_PIPS
    wins       = pips_net > 0
    days       = max(1, (df.index[-1] - df.index[0]).days)
    sharpe     = (pips_net.mean() / pips_net.std()
                  * np.sqrt(len(trades) / (days / 365.25))
                  if len(trades) > 1 and pips_net.std() > 0 else 0.0)
    kinds = [t["exit"] for t in trades]
    atr_ps = [t["atr_pips"] for t in trades]
    return {
        "n":        len(trades),
        "pips_net": float(pips_net.sum()),
        "win_pct":  float(wins.mean() * 100),
        "avg_win":  float(pips_net[wins].mean())  if wins.any()      else 0.0,
        "avg_loss": float(pips_net[~wins].mean()) if (~wins).any()   else 0.0,
        "avg_hold": float(np.mean([t["hold"] for t in trades])),
        "sharpe":   float(sharpe),
        "n_tp":     sum(k == "tp" for k in kinds),
        "n_sl":     sum(k == "sl" for k in kinds),
        "avg_atr":  float(np.mean(atr_ps)),
    }


PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
START = "2025-01-01"   # full 2025 + 2026 YTD (~16 months)
END   = "2026-05-14"


def run_one(pair: str):
    pip = pip_size(pair)
    df_1h = add_atr(add_macd(get_data(pair, "1h", START, END)))
    df_1d = get_data(pair, "1d", START, END)

    if df_1h.empty or df_1d.empty:
        print(f"{pair}: missing data; skip")
        return None

    cu_1h, cd_1h  = macd_crosses(df_1h)
    bull_1d_on_1h = htf_bullish(df_1d, df_1h.index)

    # Session filter: only allow entries during broker hours [LO, HI] inclusive.
    hrs = df_1h.index.hour.to_numpy()
    session_ok = (hrs >= SESSION_HOUR_LO) & (hrs <= SESSION_HOUR_HI)
    cu_1h = cu_1h & session_ok
    cd_1h = cd_1h & session_ok

    base_variants = [
        ("C 1h raw         ", df_1h, cu_1h, cd_1h, None),
        ("D 1h + 1d filter ", df_1h, cu_1h, cd_1h, bull_1d_on_1h),
    ]
    out = {}
    for base_label, df_v, sig_l, sig_s, htf in base_variants:
        for tp_mult, sl_mult in RR_SWEEP:
            label = f"{base_label} TP{tp_mult:g}/SL{sl_mult:g}"
            htf_long  = htf
            htf_short = (~htf) if htf is not None else None
            longs_t,  o_l, mtm_l = backtest(df_v, +1, sig_l, pip, tp_mult, sl_mult, htf_long)
            shorts_t, o_s, mtm_s = backtest(df_v, -1, sig_s, pip, tp_mult, sl_mult, htf_short)
            trades = longs_t + shorts_t
            s = summarize(trades, df_v)
            s["n_open"]   = o_l + o_s
            s["open_mtm"] = mtm_l + mtm_s
            s["tp_mult"]  = tp_mult
            s["sl_mult"]  = sl_mult
            out[label] = {"summary": s, "trades": trades, "df": df_v}
    return out


def print_header(title):
    print(f"\n{title}")
    print(f"{'Variant':<32} {'N':>4} {'TP':>4} {'SL':>4} {'NetP':>8} "
          f"{'Sharpe':>7} {'Win%':>6} {'AvgW':>6} {'AvgL':>7} {'Hold':>5} "
          f"{'AvgATR':>6} | {'Open':>4} {'OpenMTM':>8}")
    print("-" * 116)


def print_row(label, s):
    if s.get("n", 0) == 0:
        n_open = s.get("n_open", 0)
        mtm    = s.get("open_mtm", 0.0)
        print(f"{label:<32} {'-':>4} {'-':>4} {'-':>4} {'-':>8} {'-':>7} "
              f"{'-':>6} {'-':>6} {'-':>7} {'-':>5} {'-':>6} | "
              f"{n_open:>4} {mtm:>+8.1f}")
        return
    print(f"{label:<32} {s['n']:>4} {s['n_tp']:>4} {s['n_sl']:>4} "
          f"{s['pips_net']:>+8.1f} {s['sharpe']:>+7.2f} {s['win_pct']:>5.1f}% "
          f"{s['avg_win']:>+6.1f} {s['avg_loss']:>+7.1f} {s['avg_hold']:>5.1f} "
          f"{s['avg_atr']:>6.1f} | "
          f"{s.get('n_open', 0):>4} {s.get('open_mtm', 0.0):>+8.1f}")


def hour_table(rows, title):
    if not rows:
        print(f"\n{title}: no trades")
        return
    df = pd.DataFrame(rows)
    g = df.groupby("hour")["pips_net"].agg(
        n=("count"), sum=("sum"), mean=("mean"),
        win_pct=lambda x: (x > 0).mean() * 100,
    )
    print(f"\n{title}")
    print(g.round(2).to_string())
    best = g["sum"].idxmax()
    worst = g["sum"].idxmin()
    print(f"  best  hour: {best:02d}  (sum={g.loc[best, 'sum']:+.1f} pips, "
          f"n={int(g.loc[best, 'n'])})")
    print(f"  worst hour: {worst:02d}  (sum={g.loc[worst, 'sum']:+.1f} pips, "
          f"n={int(g.loc[worst, 'n'])})")


# ----- run -----
all_results = {}
for pair in PAIRS:
    print(f"\n=== {pair} ===")
    res = run_one(pair)
    if res is None:
        continue
    all_results[pair] = res
    print_header(f"{pair}  {START} .. {END}")
    for label, payload in res.items():
        print_row(label, payload["summary"])

# Cross-pair aggregate per (variant, R:R) — total realized P&L + total stuck MTM
print("\n" + "=" * 70)
print("CROSS-PAIR AGGREGATE")
print("=" * 70)
print(f"{'Variant':<32} {'TotN':>5} {'NetP':>9} {'+MTM':>9} {'Win%':>6} "
      f"{'Open':>5} {'+/-Pairs':>9}")
print("-" * 78)

# Collect variant labels in order from any pair (they all share the same set)
sample = next(iter(all_results.values()))
for label in sample.keys():
    tot_n = 0
    tot_net = 0.0
    tot_open = 0
    tot_mtm = 0.0
    pair_results = []
    win_pcts = []
    for pair, res in all_results.items():
        s = res[label]["summary"]
        if s.get("n", 0) == 0:
            continue
        tot_n   += s["n"]
        tot_net += s["pips_net"]
        tot_open+= s["n_open"]
        tot_mtm += s["open_mtm"]
        pair_results.append(s["pips_net"])
        win_pcts.append(s["win_pct"])
    if tot_n == 0:
        continue
    pos = sum(1 for p in pair_results if p > 0)
    neg = sum(1 for p in pair_results if p <= 0)
    avg_win_pct = sum(win_pcts) / len(win_pcts)
    print(f"{label:<32} {tot_n:>5} {tot_net:>+9.1f} {tot_net+tot_mtm:>+9.1f} "
          f"{avg_win_pct:>5.1f}% {tot_open:>5} {pos:>3}+/{neg:<3}-")
