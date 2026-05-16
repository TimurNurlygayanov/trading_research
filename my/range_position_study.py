"""
Multi-pair, multi-TF, multi-N range-position study.

For each bar i, with range = past N bars (strictly before i):
  range_pos = (Close[i] - range_lo) / (range_hi - range_lo)
Scan forward K bars: which of Close[i]±M·ATR is touched first?

Aggregate across pairs to find structural patterns that survive different
per-pair drift directions. Each pair has its own baseline edge (P_up − P_down
across all observations); we report bucket excess edge = bucket_edge − baseline,
then average excess across pairs.
"""
import sys
import time

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


_MT5_TF_MAP = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
               "1h": 16385, "4h": 16388, "1d": 16408}


def get_data(ticker: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end   = pd.Timestamp(end,   tz="UTC")
    mt5.initialize()
    tf_const = _MT5_TF_MAP.get(timeframe, 1)
    mt5.symbol_select(ticker, True)
    CHUNK = 50_000
    frames, pos = [], 0
    while True:
        chunk = mt5.copy_rates_from_pos(ticker, tf_const, pos, CHUNK)
        if chunk is None or len(chunk) < CHUNK:
            if chunk is not None and len(chunk):
                frames.append(pd.DataFrame(chunk))
            break
        frames.append(pd.DataFrame(chunk))
        oldest = pd.Timestamp(int(frames[-1]["time"].min()), unit="s", tz="UTC")
        if oldest <= ts_start:
            break
        pos += CHUNK
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames[::-1])
    df = df.drop_duplicates(subset="time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df.index >= ts_start) & (df.index < ts_end)]
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "tick_volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ── Config ──
PAIRS    = ["EURUSD", "AUDUSD", "NZDUSD", "USDCHF",
            "USDCAD", "GBPUSD", "USDJPY", "EURJPY"]
TFS      = ["1h", "5m"]
NS       = [10, 20, 50]
START    = "2025-01-01"
END      = "2026-05-01"
ATR_P    = 14
ATR_MULT = 2.0
K_FWD    = 50


def study(df: pd.DataFrame, n_range: int, atr_p: int,
          mult: float, k_fwd: int) -> pd.DataFrame:
    """Vectorized forward-scan. Returns one row per valid bar."""
    n = len(df)
    if n < max(n_range, atr_p) + k_fwd + 5:
        return pd.DataFrame()

    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values

    atr_v    = pandas_ta.atr(df["High"], df["Low"], df["Close"], length=atr_p).values
    range_lo = pd.Series(low ).rolling(n_range, min_periods=n_range).min().shift(1).values
    range_hi = pd.Series(high).rolling(n_range, min_periods=n_range).max().shift(1).values

    span = range_hi - range_lo
    rp   = np.where(span > 0, (close - range_lo) / span, np.nan)
    upper = close + mult * atr_v
    lower = close - mult * atr_v

    # outcome_code: 0=unresolved/none, 1=up, 2=down, 3=tie
    outcome_code = np.zeros(n, dtype=np.int8)
    bars_to      = np.full(n, k_fwd, dtype=np.int32)

    for k in range(1, k_fwd + 1):
        idx_max = n - k
        unresolved = outcome_code[:idx_max] == 0
        up_hit = (high[k:k + idx_max] >= upper[:idx_max]) & unresolved
        dn_hit = (low [k:k + idx_max] <= lower[:idx_max]) & unresolved
        tie     = up_hit & dn_hit
        up_only = up_hit & ~tie
        dn_only = dn_hit & ~tie

        oc_view = outcome_code[:idx_max]
        oc_view[tie]     = 3
        oc_view[up_only] = 1
        oc_view[dn_only] = 2

        any_hit = up_only | dn_only | tie
        bt_view = bars_to[:idx_max]
        bt_view[any_hit] = k

    fwd_idx  = np.minimum(np.arange(n) + k_fwd, n - 1)
    fwd_atr  = np.where(atr_v > 0, (close[fwd_idx] - close) / atr_v, np.nan)

    valid = np.isfinite(rp) & np.isfinite(atr_v)
    valid[max(0, n - k_fwd):] = False  # not enough forward bars

    code_to_str = np.array(["none", "up", "down", "tie"])
    return pd.DataFrame({
        "range_pos":   rp[valid],
        "outcome":     code_to_str[outcome_code[valid]],
        "bars_to":     bars_to[valid],
        "fwd_ret_atr": fwd_atr[valid],
    })


def bucket(rp: float) -> str:
    if rp < 0: return "<0 brkdn"
    if rp > 1: return ">1 brkup"
    idx = min(int(rp * 10), 9)
    return f"{idx/10:.1f}-{(idx+1)/10:.1f}"


BIN_ORDER = ["<0 brkdn"] + [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)] + [">1 brkup"]


# ── Step 1: load ──
print(f"Loading {len(PAIRS)} pairs × {len(TFS)} TFs  ({START} → {END})…")
t0 = time.time()
data: dict = {}
for tf in TFS:
    for pair in PAIRS:
        df = get_data(pair, tf, START, END)
        data[(tf, pair)] = df
        print(f"  {pair} {tf}: {len(df):>7} bars")
print(f"  ({time.time()-t0:.1f}s)")


# ── Step 2: studies ──
print(f"\nRunning {len(PAIRS) * len(TFS) * len(NS)} studies…")
t0 = time.time()
results: dict = {(tf, n): [] for tf in TFS for n in NS}
baselines: dict = {}
for tf in TFS:
    for pair in PAIRS:
        df = data[(tf, pair)]
        if df.empty:
            continue
        for n_range in NS:
            obs = study(df, n_range, ATR_P, ATR_MULT, K_FWD)
            if obs.empty:
                continue
            obs["bin"] = obs["range_pos"].apply(bucket)
            p_up = (obs["outcome"] == "up"  ).mean() * 100
            p_dn = (obs["outcome"] == "down").mean() * 100
            base = p_up - p_dn
            baselines[(tf, pair)] = base  # same regardless of n_range (overall stats)

            g = obs.groupby("bin", observed=True)
            stats = pd.DataFrame({
                "n":           g.size(),
                "p_up":        g["outcome"].apply(lambda s: (s == "up"  ).mean() * 100),
                "p_down":      g["outcome"].apply(lambda s: (s == "down").mean() * 100),
                "avg_bars":    g["bars_to"].mean(),
                "avg_fwd_atr": g["fwd_ret_atr"].mean(),
            })
            stats["edge"]   = stats["p_up"] - stats["p_down"]
            stats["excess"] = stats["edge"] - base
            results[(tf, n_range)].append((pair, stats, base))
print(f"  ({time.time()-t0:.1f}s)")


# ── Step 3: baseline drift table ──
print("\n\n══════ Per-pair baseline drift  (P_up − P_down %, overall) ══════")
print(f"{'Pair':<8}  " + "  ".join(f"{tf:>8}" for tf in TFS))
for pair in PAIRS:
    cells = []
    for tf in TFS:
        b = baselines.get((tf, pair))
        cells.append(f"{b:+8.1f}" if b is not None else "    n/a")
    print(f"{pair:<8}  " + "  ".join(cells))


# ── Step 4: per-(TF, N) aggregated tables ──
for (tf, n_range), runs in sorted(results.items()):
    if not runs:
        continue
    n_pairs = len(runs)
    print(f"\n\n══════ {tf} TF · N_RANGE={n_range} · K_FWD={K_FWD} · "
          f"avg across {n_pairs} pairs ══════")
    hdr = f"{'bin':<10} {'pooled_n':>9} {'mean_edge':>10} {'mean_excess':>12} {'agree':>7} {'avg_bars':>9} {'avg_fwd_atr':>12}"
    print(hdr)
    print("-" * len(hdr))

    for b in BIN_ORDER:
        ns, edges, excesses, bars_list, fwd_list = [], [], [], [], []
        for pair, stats, _ in runs:
            if b in stats.index:
                row = stats.loc[b]
                ns.append(int(row["n"]))
                edges.append(float(row["edge"]))
                excesses.append(float(row["excess"]))
                bars_list.append(float(row["avg_bars"]))
                fwd_list.append(float(row["avg_fwd_atr"]))
        if not ns:
            continue
        pooled_n     = sum(ns)
        mean_edge    = float(np.mean(edges))
        mean_excess  = float(np.mean(excesses))
        mean_sign    = np.sign(mean_excess) if mean_excess != 0 else 1
        pairs_agree  = sum(1 for e in excesses if (e > 0) == (mean_sign > 0))
        agree_str    = f"{pairs_agree}/{len(excesses)}"
        avg_bars_m   = float(np.mean(bars_list))
        avg_fwd_atr_m= float(np.mean(fwd_list))
        print(f"{b:<10} {pooled_n:>9} {mean_edge:>+10.1f} {mean_excess:>+12.1f} "
              f"{agree_str:>7} {avg_bars_m:>9.1f} {avg_fwd_atr_m:>+12.2f}")
