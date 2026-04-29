"""
EMA-angle vs forward-return correlation study.

Question: does the slope ("angle") of EMA{21,50,200} have predictive power
          for whether price closes higher/lower N candles ahead?

Features
  ang_e{P}_n{N} = (EMA_P[t] - EMA_P[t-N]) / EMA_P[t-N] * 100   (% slope)
  for P in {21, 50, 200}, N in {1, 5, 10}        → 9 features

Targets
  ret_h{H} = close[t+H] / close[t] - 1                          (forward return)
  up_h{H}  = 1 if close[t+H] > close[t] else 0                  (direction hit)
  for H in {20, 50}                              → 4 targets

Outputs
  - Pearson + Spearman correlations (with p-values, sample size)
  - Quintile-binned hit-rate tables: P(up | feature ∈ quintile)
  - CSV files alongside this script

Usage
  python -m scripts.ema_angle_correlation                       # default EURUSD 1h
  python -m scripts.ema_angle_correlation --symbol GBPUSD --tf 1h
  python -m scripts.ema_angle_correlation --start 2018-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv

EMAS      = (21, 50, 200)
LOOKBACKS = (1, 5, 10)
HORIZONS  = (2, 3, 10, 20)
QUANTILES = 5  # quintiles


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["Close"]].copy()
    for p in EMAS:
        out[f"ema{p}"] = out["Close"].ewm(span=p, adjust=False).mean()
    for p in EMAS:
        for n in LOOKBACKS:
            out[f"ang_e{p}_n{n}"] = (out[f"ema{p}"] / out[f"ema{p}"].shift(n) - 1) * 100
    for h in HORIZONS:
        out[f"ret_h{h}"] = out["Close"].shift(-h) / out["Close"] - 1
        out[f"up_h{h}"]  = (out["Close"].shift(-h) > out["Close"]).astype(int)
    return out


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    feats   = [f"ang_e{p}_n{n}" for p in EMAS for n in LOOKBACKS]
    targets = [f"ret_h{h}" for h in HORIZONS] + [f"up_h{h}" for h in HORIZONS]
    rows = []
    for f in feats:
        for t in targets:
            sub = df[[f, t]].dropna()
            if len(sub) < 100:
                continue
            pr, pp = stats.pearsonr(sub[f], sub[t])
            sr, sp = stats.spearmanr(sub[f], sub[t])
            rows.append({
                "feature":     f,
                "target":      t,
                "n":           len(sub),
                "pearson_r":   pr,
                "pearson_p":   pp,
                "spearman_r":  sr,
                "spearman_p":  sp,
            })
    return pd.DataFrame(rows)


def quantile_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """For each feature, bin into quintiles and report hit rate / mean return."""
    feats = [f"ang_e{p}_n{n}" for p in EMAS for n in LOOKBACKS]
    rows = []
    for f in feats:
        sub = df[[f, target]].dropna().copy()
        if len(sub) < 100:
            continue
        sub["q"] = pd.qcut(sub[f], QUANTILES, labels=False, duplicates="drop")
        baseline = sub[target].mean()
        for q in sorted(sub["q"].dropna().unique()):
            grp = sub[sub["q"] == q]
            rows.append({
                "feature":  f,
                "quintile": int(q) + 1,
                "n":        len(grp),
                "feat_lo":  grp[f].min(),
                "feat_hi":  grp[f].max(),
                "metric":   grp[target].mean(),
                "baseline": baseline,
                "lift":     grp[target].mean() - baseline,
            })
    return pd.DataFrame(rows)


def fmt_corr(corr: pd.DataFrame) -> str:
    """Pretty-print correlation table sorted by |spearman_r| desc."""
    c = corr.copy()
    c["abs_sp"] = c["spearman_r"].abs()
    c = c.sort_values("abs_sp", ascending=False).drop(columns="abs_sp")
    fmt = lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v)
    return c.to_string(index=False, formatters={col: fmt for col in c.columns
                                                 if c[col].dtype.kind == "f"})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--tf",     default="1h", help="timeframe (1h, 5m, etc.)")
    ap.add_argument("--start",  default="2018-01-01")
    ap.add_argument("--end",    default="2025-12-31")
    ap.add_argument("--out",    default=None,
                    help="output dir for CSVs (default: same dir as this script)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {args.symbol} {args.tf}  {args.start} → {args.end}")
    df = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  fetched {len(df):,} bars\n")
    if df.empty:
        print("No data — aborting.")
        return

    feat = compute_features(df).dropna()
    print(f"Usable rows after warmup + forward-shift: {len(feat):,}\n")

    # ── correlations ─────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"CORRELATIONS — {args.symbol} {args.tf}  (sorted by |Spearman|)")
    print("=" * 72)
    corr = correlation_table(feat)
    print(fmt_corr(corr))
    corr_path = out_dir / f"ema_angle_corr_{args.symbol}_{args.tf}.csv"
    corr.to_csv(corr_path, index=False)
    print(f"\n→ saved {corr_path}")

    # ── direction quantile tables ────────────────────────────────────────────
    for h in HORIZONS:
        target = f"up_h{h}"
        print("\n" + "=" * 72)
        print(f"HIT-RATE BY FEATURE QUINTILE — target = {target}")
        print("=" * 72)
        qt = quantile_table(feat, target)
        # show only top-3 features by |lift| in extreme quintiles, plus their full bins
        if not qt.empty:
            extreme = qt[qt["quintile"].isin([1, QUANTILES])]
            top_feats = (extreme.assign(abs_lift=lambda d: d["lift"].abs())
                                 .groupby("feature")["abs_lift"].max()
                                 .sort_values(ascending=False).head(3).index.tolist())
            shown = qt[qt["feature"].isin(top_feats)]
            fmt = lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v)
            print(shown.to_string(index=False,
                                  formatters={c: fmt for c in shown.columns
                                              if shown[c].dtype.kind == "f"}))
        path = out_dir / f"ema_angle_hitrate_h{h}_{args.symbol}_{args.tf}.csv"
        qt.to_csv(path, index=False)
        print(f"\n→ saved {path}")

    # ── headline summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("TOP 5 FEATURE × TARGET PAIRS BY |SPEARMAN|")
    print("=" * 72)
    top = (corr.assign(abs_sp=lambda d: d["spearman_r"].abs())
               .sort_values("abs_sp", ascending=False).head(5))
    for _, r in top.iterrows():
        sig = "***" if r["spearman_p"] < 0.001 else "**" if r["spearman_p"] < 0.01 else \
              "*"   if r["spearman_p"] < 0.05  else ""
        print(f"  {r['feature']:<14s} → {r['target']:<10s}  "
              f"ρ={r['spearman_r']:+.4f}  p={r['spearman_p']:.2e}  n={r['n']:,} {sig}")
    print()


if __name__ == "__main__":
    main()
