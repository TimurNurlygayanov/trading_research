"""
Exploratory analysis of 5m MT5 data to identify tradeable edges.

Sections
--------
  1. Data overview — bar counts, ATR levels, session hours
  2. Return autocorrelation — is there momentum or mean-reversion?
  3. Directional persistence — P(up | last N bars up), P(up | EMA rising)
  4. EMA alignment — how reliably does EMA slope predict next-bar direction?
  5. Move-size distribution — how far does price travel in next K bars (in ATR)?
  6. Hourly stats — trend score, avg ATR, win% for simple follow/fade rules
  7. Day-of-week patterns
  8. Simple strategy screening — forward-test several rules, show EV

Usage
-----
  python -m scripts.analyze_mt5_5m \\
      --login 1513313327 --server FTMO-Demo \\
      --start 2026-01-01 --end 2026-05-06
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.strategy1_regime_backtest import (
    _load_data_mt5, _atr, _ema, PAIRS,
    HALF_SPREAD_PIPS, PIP_VALUE, PIP_SIZE, COMMISSION_PER_LOT,
)

SEP  = "=" * 80
SEP2 = "-" * 60


def _cost_per_rt(pair: str, lots: float = 1.0) -> float:
    """Total round-trip cost: 2 × half-spread + commission."""
    spread_cost = 2 * HALF_SPREAD_PIPS[pair] * PIP_VALUE[pair] * lots
    return spread_cost + COMMISSION_PER_LOT * lots


# ── 1. Data overview ──────────────────────────────────────────────────────────

def section_overview(all_df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  SECTION 1 — DATA OVERVIEW")
    print(SEP)
    print(f"  Bars:        {len(all_df):,}")
    print(f"  Date range:  {all_df.index.min().date()}  to  {all_df.index.max().date()}")
    print(f"  Pairs:       {all_df['pair'].unique().tolist()}")

    print(f"\n  {'Pair':<8} {'Bars':>7} {'ATR_avg(pip)':>14} {'ATR_p10':>9} {'ATR_p90':>9} "
          f"{'TypSprd(pip)':>14}")
    for pair, g in all_df.groupby("pair"):
        atr_pip = g["atr"] / PIP_SIZE
        spread  = HALF_SPREAD_PIPS[pair] * 2
        print(f"  {pair:<8} {len(g):>7,} {atr_pip.mean():>14.2f} "
              f"{atr_pip.quantile(0.1):>9.2f} {atr_pip.quantile(0.9):>9.2f} "
              f"{spread:>14.2f}")


# ── 2. Return autocorrelation ─────────────────────────────────────────────────

def section_autocorr(all_df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  SECTION 2 — RETURN AUTOCORRELATION")
    print("  Positive = momentum (trend-following edge)")
    print("  Negative = mean-reversion (fade edge)")
    print(SEP)
    print(f"  {'Pair':<8} {'lag1':>8} {'lag2':>8} {'lag3':>8} {'lag5':>8} {'lag10':>8}  interpretation")

    for pair, g in all_df.groupby("pair"):
        g  = g.sort_index()
        r  = g["ret"]
        acs = [r.autocorr(lag=k) for k in [1, 2, 3, 5, 10]]
        mean_ac = np.mean(acs[:3])
        interp  = ("MOMENTUM  >>>  trend-follow" if mean_ac > 0.02
                   else "MEAN-REV  >>>  fade/reversal" if mean_ac < -0.02
                   else "RANDOM    >>>  no clear edge")
        print(f"  {pair:<8} " + " ".join(f"{a:>8.4f}" for a in acs) + f"  {interp}")


# ── 3. Directional persistence ────────────────────────────────────────────────

def section_persistence(all_df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  SECTION 3 — DIRECTIONAL PERSISTENCE")
    print("  P(next bar same direction as last N bars)")
    print("  50% = coin-flip.  >55% = exploitable.  <45% = fade edge.")
    print(SEP)

    for pair, g in all_df.groupby("pair"):
        g = g.sort_index()
        up = (g["ret"] > 0).astype(int)

        probs = {}
        for n in [1, 2, 3, 5]:
            # last N bars all up → next bar up?
            run_up   = up.rolling(n).min() == 1
            run_down = (1 - up).rolling(n).min() == 1
            next_up  = up.shift(-1)
            p_up_after_up   = next_up[run_up  & next_up.notna()].mean()
            p_up_after_down = next_up[run_down & next_up.notna()].mean()
            probs[n] = (p_up_after_up, 1 - p_up_after_down)

        print(f"\n  {pair}:")
        print(f"    {'streak':<10} {'P(cont after N up)':>22} {'P(cont after N dn)':>22}")
        for n, (pu, pd_) in probs.items():
            flag_u = " <<<" if abs(pu - 0.5) > 0.04 else ""
            flag_d = " <<<" if abs(pd_ - 0.5) > 0.04 else ""
            print(f"    N={n:<8} {pu:>22.3f}{flag_u}   {pd_:>22.3f}{flag_d}")


# ── 4. EMA alignment ──────────────────────────────────────────────────────────

def section_ema(all_df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  SECTION 4 — EMA SLOPE PREDICTS NEXT BAR DIRECTION?")
    print("  When EMA is rising, does the next bar close up?")
    print(SEP)
    print(f"  {'Pair':<8} {'EMA':>5} {'P(up|EMA rise)':>16} {'P(dn|EMA fall)':>16} "
          f"{'edge':>8}  note")

    for pair, g in all_df.groupby("pair"):
        g = g.sort_index()
        for ema_p in [9, 20, 50]:
            col = f"ema{ema_p}"
            if col not in g.columns:
                continue
            slope   = g[col].diff()
            next_up = (g["ret"].shift(-1) > 0)
            ema_up  = slope > 0
            ema_dn  = slope < 0

            p_up_when_ema_up = next_up[ema_up  & next_up.notna()].mean()
            p_dn_when_ema_dn = (~next_up)[ema_dn & next_up.notna()].mean()
            edge = (p_up_when_ema_up + p_dn_when_ema_dn) / 2 - 0.5

            note = ("strong signal <<<" if abs(edge) > 0.03
                    else "weak signal" if abs(edge) > 0.01
                    else "no signal")
            print(f"  {pair:<8} {ema_p:>5}  {p_up_when_ema_up:>16.3f}  "
                  f"{p_dn_when_ema_dn:>16.3f}  {edge:>8.4f}  {note}")


# ── 5. Move-size distribution ─────────────────────────────────────────────────

def section_moves(all_df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  SECTION 5 — FORWARD MOVE DISTRIBUTION (in ATR units)")
    print("  Max favorable / Max adverse excursion in next K bars from bar close")
    print("  Helps set realistic TP and SL targets")
    print(SEP)

    for pair, g in all_df.groupby("pair"):
        g    = g.sort_index().reset_index(drop=False)
        cls  = g["Close"].values
        hi   = g["High"].values
        lo   = g["Low"].values
        atr  = g["atr"].values
        n    = len(g)

        print(f"\n  {pair}:")
        print(f"    {'K bars':>8} {'MFE_p25':>9} {'MFE_p50':>9} {'MFE_p75':>9} "
              f"{'MAE_p25':>9} {'MAE_p50':>9} {'MAE_p75':>9}")

        for k in [1, 3, 6, 12, 24]:
            mfe_list, mae_list = [], []
            for i in range(n - k):
                if atr[i] <= 0:
                    continue
                future_hi = hi[i+1: i+k+1].max()
                future_lo = lo[i+1: i+k+1].min()
                mfe = (future_hi - cls[i]) / atr[i]   # best case move up
                mae = (cls[i] - future_lo) / atr[i]   # worst case move down
                mfe_list.append(mfe)
                mae_list.append(mae)
            mfe_a = np.array(mfe_list)
            mae_a = np.array(mae_list)
            print(f"    {k:>8}  "
                  f"{np.percentile(mfe_a,25):>9.2f} {np.percentile(mfe_a,50):>9.2f} "
                  f"{np.percentile(mfe_a,75):>9.2f}  "
                  f"{np.percentile(mae_a,25):>9.2f} {np.percentile(mae_a,50):>9.2f} "
                  f"{np.percentile(mae_a,75):>9.2f}")


# ── 6. Hourly stats ───────────────────────────────────────────────────────────

def section_hourly(all_df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  SECTION 6 — HOURLY PATTERN (all pairs combined)")
    print("  trend_score = |mean(ret)| / std(ret) × 100  (higher = more directional)")
    print(SEP)
    print(f"  {'Hour':>5} {'Bars':>7} {'ATR(pip)':>10} {'trend_score':>13} "
          f"{'P(follow)':>11} {'P(fade)':>9}  verdict")

    g = all_df.copy()
    g["next_ret"] = g.groupby("pair")["ret"].shift(-1)
    g = g.dropna(subset=["next_ret"])

    for h, grp in g.groupby("hour"):
        n   = len(grp)
        atr = (grp["atr"] / PIP_SIZE).mean()
        r   = grp["ret"]
        nr  = grp["next_ret"]

        trend_score = abs(r.mean()) / (r.std() + 1e-10) * 100

        # Follow = next bar goes same direction as current bar
        follow_wr = ((r > 0) & (nr > 0) | (r < 0) & (nr < 0)).mean()
        fade_wr   = 1 - follow_wr

        verdict = ("TREND  >>>" if follow_wr > 0.54
                   else "FADE   <<<" if fade_wr > 0.54
                   else "neutral")
        print(f"  {h:>5}h {n:>7,} {atr:>10.2f} {trend_score:>13.2f} "
              f"{follow_wr:>11.3f} {fade_wr:>9.3f}  {verdict}")


# ── 7. Day-of-week ────────────────────────────────────────────────────────────

def section_dow(all_df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  SECTION 7 — DAY OF WEEK (all pairs combined)")
    print(SEP)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    g = all_df.copy()
    g["next_ret"] = g.groupby("pair")["ret"].shift(-1)
    g = g.dropna(subset=["next_ret"])
    print(f"  {'Day':>5} {'Bars':>7} {'ATR(pip)':>10} {'follow_wr':>11} {'fade_wr':>9}")
    for d, grp in g.groupby("dow"):
        n   = len(grp)
        atr = (grp["atr"] / PIP_SIZE).mean()
        r   = grp["ret"];  nr = grp["next_ret"]
        follow = ((r > 0) & (nr > 0) | (r < 0) & (nr < 0)).mean()
        print(f"  {days[d]:>5} {n:>7,} {atr:>10.2f} {follow:>11.3f} {1-follow:>9.3f}")


# ── 8. Simple strategy screening ─────────────────────────────────────────────

def section_strategies(all_df: pd.DataFrame, lots: float = 1.0) -> None:
    """
    Forward-test a set of simple 1-bar and multi-bar rules.
    Each rule specifies entry condition at bar close, exit after K bars.
    Net P&L accounts for spread + commission.
    """
    print(f"\n{SEP}")
    print("  SECTION 8 — SIMPLE STRATEGY SCREEN  (1 lot, enter at close, exit after K bars)")
    print("  Tests entry signal → hold K bars → exit at close")
    print(SEP)

    results = []

    for pair, g in all_df.groupby("pair"):
        g    = g.sort_index().reset_index(drop=False)
        cls  = g["Close"].values
        atr  = g["atr"].values
        ret  = g["ret"].values          # close[i] - close[i-1]
        ema9 = g["ema9"].values
        ema20= g["ema20"].values
        n    = len(g)
        cost = _cost_per_rt(pair, lots) / (PIP_VALUE[pair] * lots / 10)  # in price units

        for K in [1, 3, 6, 12]:
            for name, signal in [
                # Momentum signals
                ("follow_1bar",    lambda i: ret[i] > 0),
                ("follow_1bar_dn", lambda i: ret[i] < 0),
                ("ema9_rising",    lambda i: ema9[i] > ema9[i-1] if i > 0 else False),
                ("ema9_falling",   lambda i: ema9[i] < ema9[i-1] if i > 0 else False),
                ("above_ema9",     lambda i: cls[i] > ema9[i]),
                ("below_ema9",     lambda i: cls[i] < ema9[i]),
                ("above_ema20",    lambda i: cls[i] > ema20[i]),
                # Volatility
                ("big_bar_up",     lambda i: ret[i] > 0.5 * atr[i]),
                ("big_bar_dn",     lambda i: ret[i] < -0.5 * atr[i]),
                ("small_bar",      lambda i: abs(ret[i]) < 0.2 * atr[i]),
            ]:
                trades_net = []
                for i in range(1, n - K):
                    if atr[i] <= 0:
                        continue
                    try:
                        sig = signal(i)
                    except Exception:
                        continue
                    if not sig:
                        continue
                    # direction based on signal name
                    direction = -1 if ("_dn" in name or "falling" in name
                                       or "below" in name) else 1
                    entry = cls[i]
                    exit_ = cls[i + K]
                    gross = (exit_ - entry) * direction / PIP_SIZE * PIP_VALUE[pair] * lots
                    # simplified cost: always pay spread both sides + commission
                    net   = gross - _cost_per_rt(pair, lots)
                    trades_net.append(net)

                if len(trades_net) < 30:
                    continue
                arr = np.array(trades_net)
                n_t = len(arr)
                wr  = (arr > 0).mean()
                ev  = arr.mean()
                sr  = arr.mean() / (arr.std() + 1e-9) * np.sqrt(n_t)
                results.append({
                    "pair": pair, "signal": name, "hold_bars": K,
                    "n": n_t, "win_pct": round(wr*100, 1),
                    "ev_per_trade": round(ev, 3),
                    "total_pnl": round(arr.sum(), 1),
                    "sharpe": round(sr, 2),
                })

    if not results:
        print("  No results.")
        return

    df = pd.DataFrame(results)
    # Show top 15 by Sharpe, then bottom 5 (to see what clearly doesn't work)
    df_pos = df[df["ev_per_trade"] > 0].sort_values("sharpe", ascending=False)
    df_neg = df[df["ev_per_trade"] < 0].sort_values("sharpe").head(5)

    cols = ["pair", "signal", "hold_bars", "n", "win_pct", "ev_per_trade", "total_pnl", "sharpe"]

    print("\n  >>> POSITIVE EV SIGNALS (sorted by Sharpe):")
    if len(df_pos):
        print(df_pos[cols].head(20).to_string(index=False))
    else:
        print("  None found — all signals negative EV after costs")

    print("\n  >>> WORST SIGNALS (avoid these):")
    print(df_neg[cols].to_string(index=False))

    # Summary by signal name (aggregated across pairs and hold periods)
    print(f"\n  >>> SIGNAL SUMMARY (avg EV across pairs × hold periods):")
    summary = (df.groupby("signal")
                 .agg(avg_ev=("ev_per_trade","mean"),
                      avg_sharpe=("sharpe","mean"),
                      configs=("n","count"))
                 .sort_values("avg_ev", ascending=False))
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--start",         default="2026-01-01")
    ap.add_argument("--end",           default="2026-05-06")
    ap.add_argument("--mt5-path",      default=None)
    ap.add_argument("--login",         type=int, default=None)
    ap.add_argument("--password",      default=None)
    ap.add_argument("--server",        default=None)
    ap.add_argument("--symbol-suffix", default="")
    ap.add_argument("--pairs",         nargs="+", default=None)
    ap.add_argument("--lots",          type=float, default=1.0)
    args = ap.parse_args()

    pairs = args.pairs or PAIRS

    print(f"Loading 5m MT5 data {args.start} → {args.end} ...")
    try:
        pairs_data = _load_data_mt5(
            pairs, args.start, args.end,
            timeframe="5m",
            mt5_path=args.mt5_path, login=args.login,
            password=args.password, server=args.server,
            symbol_suffix=args.symbol_suffix,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not pairs_data:
        print("No data loaded.")
        sys.exit(1)

    # ── Build master dataframe ────────────────────────────────────────────────
    frames = []
    for pair, df in pairs_data.items():
        d = df.copy()
        d["pair"]  = pair
        d["ret"]   = d["Close"].diff()           # raw price change
        d["atr"]   = _atr(d, 14)
        d["ema9"]  = _ema(d["Close"].values, 9)
        d["ema20"] = _ema(d["Close"].values, 20)
        d["ema50"] = _ema(d["Close"].values, 50)
        d["hour"]  = d.index.hour
        d["dow"]   = d.index.dayofweek
        frames.append(d)

    all_df = pd.concat(frames).dropna(subset=["ret", "atr"])

    section_overview(all_df)
    section_autocorr(all_df)
    section_persistence(all_df)
    section_ema(all_df)
    section_moves(all_df)
    section_hourly(all_df)
    section_dow(all_df)
    section_strategies(all_df, lots=args.lots)

    print(f"\n{SEP}")
    print("  DONE")
    print(SEP)


if __name__ == "__main__":
    main()
