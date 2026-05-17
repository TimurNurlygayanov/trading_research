"""
Visualize ST34 winners/losers to audit the detection logic.

Reimplements ST34 in plain numpy (no backtesting.py) so the detection is
auditable, then renders sample trades with:
  - candles around the entry
  - pattern bars highlighted (S1: green→red pair / S2: lookback window + wicks)
  - S2 level line, SL line, TP line
  - entry arrow at the FILL bar (signal_bar + 1, open price)
  - actual exit bar

Usage:
  python scripts/debug_st034_visualize.py --pair EURUSD --tf 5m --signal both --n 6
  python scripts/debug_st034_visualize.py --pair EURUSD --tf 5m --signal s1
  python scripts/debug_st034_visualize.py --pair EURUSD --tf 5m --signal s2 --winners-only
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.st034_backtest_2026 import get_mt5_data


# ============================================================================
# DETECTION + SIMULATION (plain numpy mirror of the Strategy class)
# ============================================================================

def detect_and_simulate(
    df: pd.DataFrame,
    rr_ratio: float = 2.0,
    atr_len: int = 14,
    s1_tol_atr: float = 0.1,
    s1_body_clear: int = 20,
    s2_lookback: int = 10,
    s2_body_clear: int = 20,
    s2_min_wicks: int = 3,
    s2_sl_atr_mult: float = 1.0,
    s1_enable: bool = True,
    s2_enable: bool = True,
    sl_mode: str = "new",   # "old" = wick-top (S1) / level+ATR (S2)
                            # "new" = level_s1 (S1) / peak_high (S2)
) -> list[dict]:
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    n = len(df)

    atr = ta.atr(pd.Series(h), pd.Series(l), pd.Series(c), length=atr_len).values
    body_top = np.maximum(o, c)
    level = pd.Series(body_top).rolling(s2_lookback).max().shift(1).values
    body_max_wide = pd.Series(body_top).rolling(s2_body_clear).max().shift(1).values
    body_max_s1 = pd.Series(body_top).rolling(s1_body_clear).max().values

    warmup = max(atr_len, s2_lookback, s2_body_clear, s1_body_clear) + 2
    trades: list[dict] = []
    in_pos = False
    pending_exit_at: int | None = None

    i = warmup
    while i < n - 1:   # need bar i+1 for fill
        if in_pos:
            i += 1
            continue

        sig_type = None
        entry_price = None
        sl = tp = None
        meta: dict = {}

        # ── SIGNAL 1: twin tops ───────────────────────────────────────────
        if s1_enable and i >= 1 and not np.isnan(atr[i]):
            green_prev = c[i - 1] > o[i - 1]
            red_curr = c[i] < o[i]
            wick_prev = h[i - 1] > max(o[i - 1], c[i - 1])
            wick_curr = h[i] > max(o[i], c[i])
            match_mid = abs(c[i - 1] - o[i]) <= s1_tol_atr * atr[i]
            level_s1 = max(c[i - 1], o[i])
            bms1 = body_max_s1[i]
            s1_clean = (not np.isnan(bms1)) and bms1 <= level_s1
            if (green_prev and red_curr and wick_prev and wick_curr
                    and match_mid and s1_clean):
                sl_cand = level_s1 if sl_mode == "new" else max(h[i - 1], h[i])
                entry_cand = c[i]
                if sl_cand > entry_cand:
                    sig_type = "S1"
                    sl = sl_cand
                    entry_price = entry_cand
                    tp = entry_price - rr_ratio * (sl - entry_price)
                    meta = {"prev_idx": i - 1, "match_diff": abs(c[i - 1] - o[i]),
                            "tol": s1_tol_atr * atr[i],
                            "level_s1": level_s1,
                            "body_clear_start": i - s1_body_clear + 1}

        # ── SIGNAL 2: wick cluster (only if S1 didn't fire) ───────────────
        if (s2_enable and sig_type is None and not np.isnan(atr[i])
                and not np.isnan(level[i])):
            lvl = level[i]
            bmw = body_max_wide[i]
            body_clear = (not np.isnan(bmw)) and bmw <= lvl
            wicks_idx = [j for j in range(i - s2_lookback, i) if h[j] > lvl]
            wc = len(wicks_idx)
            if wc >= s2_min_wicks and body_clear:
                red_curr = c[i] < o[i]
                no_new_hi = h[i] < lvl
                if red_curr and no_new_hi:
                    if sl_mode == "new":
                        sl_cand = float(np.max(h[i - s2_lookback : i]))
                    else:
                        sl_cand = lvl + s2_sl_atr_mult * atr[i]
                    entry_cand = c[i]
                    if sl_cand > entry_cand:
                        sig_type = "S2"
                        sl = sl_cand
                        entry_price = entry_cand
                        tp = entry_price - rr_ratio * (sl - entry_price)
                        meta = {"level": lvl, "wick_count": wc,
                                "wicks_idx": wicks_idx,
                                "lookback_start": i - s2_lookback,
                                "body_clear_start": i - s2_body_clear,
                                "atr": atr[i]}

        if sig_type is None:
            i += 1
            continue

        # ── FILL at bar i+1 open (same as backtesting.py trade_on_close=False)
        fill_idx = i + 1
        fill_price = o[fill_idx]

        # Walk forward: find first bar where SL or TP hit (short trade)
        exit_idx = None
        exit_price = None
        exit_reason = None
        for j in range(fill_idx, n):
            # For shorts: SL triggers when high >= sl ; TP triggers when low <= tp
            sl_hit = h[j] >= sl
            tp_hit = l[j] <= tp
            if sl_hit and tp_hit:
                # ambiguous intrabar — assume SL first (conservative)
                exit_idx = j
                exit_price = sl
                exit_reason = "SL_AMBIG"
                break
            if sl_hit:
                exit_idx = j
                exit_price = sl
                exit_reason = "SL"
                break
            if tp_hit:
                exit_idx = j
                exit_price = tp
                exit_reason = "TP"
                break

        if exit_idx is None:
            # Trade never closed in the data window — skip
            i += 1
            continue

        pnl = fill_price - exit_price   # short: profit when exit < fill
        won = pnl > 0
        trades.append({
            "sig_type": sig_type,
            "signal_idx": i,
            "fill_idx": fill_idx,
            "exit_idx": exit_idx,
            "fill_price": fill_price,
            "exit_price": exit_price,
            "sl": sl,
            "tp": tp,
            "pnl": pnl,
            "won": won,
            "exit_reason": exit_reason,
            "entry_t": df.index[fill_idx],
            "exit_t": df.index[exit_idx],
            "meta": meta,
        })

        # Jump past exit (no pyramiding)
        i = exit_idx + 1

    return trades


# ============================================================================
# CANDLESTICK DRAWING
# ============================================================================

def draw_candles(ax, df_slice: pd.DataFrame) -> None:
    for x, (_, row) in enumerate(df_slice.iterrows()):
        bull = row["Close"] >= row["Open"]
        body_c = "#26a69a" if bull else "#ef5350"
        wick_c = "#666666"
        lo_body = min(row["Open"], row["Close"])
        hi_body = max(row["Open"], row["Close"])
        ax.bar(x, hi_body - lo_body, bottom=lo_body,
               width=0.7, color=body_c, linewidth=0)
        ax.plot([x, x], [row["Low"], lo_body], color=wick_c, linewidth=0.7)
        ax.plot([x, x], [hi_body, row["High"]], color=wick_c, linewidth=0.7)


# ============================================================================
# PER-TRADE CHART
# ============================================================================

def plot_trade(ax, trade: dict, df: pd.DataFrame) -> None:
    sig_i = trade["signal_idx"]
    fill_i = trade["fill_idx"]
    exit_i = trade["exit_idx"]
    sig_type = trade["sig_type"]

    pad_before = 20
    pad_after = 6
    start = max(0, sig_i - pad_before)
    end = min(len(df), exit_i + pad_after + 1)

    df_slice = df.iloc[start:end].copy()
    n_slice = len(df_slice)

    sig_x = sig_i - start
    fill_x = fill_i - start
    exit_x = exit_i - start

    # Highlight pattern bars
    if sig_type == "S1":
        # 20-bar clean window
        bc_start_x = max(0, trade["meta"]["body_clear_start"] - start)
        ax.axvspan(bc_start_x - 0.4, sig_x + 0.4, alpha=0.06,
                   color="#90caf9", label="_nolegend_", zorder=0)
        # Pattern pair
        prev_x = (sig_i - 1) - start
        ax.axvspan(prev_x - 0.4, sig_x + 0.4, alpha=0.20,
                   color="#ffe082", label="_nolegend_", zorder=0)
        # S1 body-match level line
        ax.axhline(trade["meta"]["level_s1"], color="#6a1b9a", linewidth=0.9,
                   linestyle="-.", alpha=0.7,
                   label=f"Level {trade['meta']['level_s1']:.5f}")
    else:  # S2
        # Wider body-clear window (lighter)
        bc_start_x = max(0, trade["meta"].get("body_clear_start", trade["meta"]["lookback_start"]) - start)
        ax.axvspan(bc_start_x - 0.4, sig_x + 0.4, alpha=0.05,
                   color="#90caf9", label="_nolegend_", zorder=0)
        # Narrow wick window (darker)
        lb_start_x = max(0, trade["meta"]["lookback_start"] - start)
        ax.axvspan(lb_start_x - 0.4, sig_x + 0.4, alpha=0.10,
                   color="#1976d2", label="_nolegend_", zorder=0)
        # Mark each rejection wick bar
        for w_idx in trade["meta"]["wicks_idx"]:
            wx = w_idx - start
            if 0 <= wx < n_slice:
                ax.plot(wx, df.iloc[w_idx]["High"], marker="v",
                        color="#1565c0", markersize=5, zorder=4)

    draw_candles(ax, df_slice)

    # SL / TP / level lines
    ax.axhline(trade["sl"], color="#c62828", linewidth=1.0,
               linestyle="--", alpha=0.85, label=f"SL {trade['sl']:.5f}")
    ax.axhline(trade["tp"], color="#2e7d32", linewidth=1.0,
               linestyle="--", alpha=0.85, label=f"TP {trade['tp']:.5f}")
    ax.axhline(trade["fill_price"], color="#424242", linewidth=0.7,
               linestyle=":", alpha=0.7, label=f"Fill {trade['fill_price']:.5f}")
    if sig_type == "S2":
        ax.axhline(trade["meta"]["level"], color="#1565c0", linewidth=0.9,
                   linestyle="-.", alpha=0.7, label=f"Level {trade['meta']['level']:.5f}")

    # Entry arrow at fill bar
    fill_high = df.iloc[fill_i]["High"]
    span = df_slice["High"].max() - df_slice["Low"].min()
    y_top = fill_high + 0.06 * span
    ax.annotate("", xy=(fill_x, fill_high), xytext=(fill_x, y_top),
                arrowprops=dict(arrowstyle="->", color="#d50000", lw=1.6))
    ax.text(fill_x, y_top + 0.01 * span, f"SHORT ({sig_type})",
            ha="center", va="bottom", fontsize=6.5,
            color="#d50000", fontweight="bold")

    # Exit marker
    if 0 <= exit_x < n_slice:
        result_color = "#2e7d32" if trade["won"] else "#c62828"
        ax.axvline(exit_x, color=result_color, linewidth=1.0,
                   linestyle="--", alpha=0.9)
        ax.plot(exit_x, trade["exit_price"], marker="o",
                color=result_color, markersize=6, zorder=5)

    # Title
    won_str = "WIN" if trade["won"] else "LOSS"
    color = "#2e7d32" if trade["won"] else "#c62828"
    t_str = trade["entry_t"].strftime("%Y-%m-%d %H:%M")
    hold = exit_i - fill_i
    pnl_pips = trade["pnl"] * 10000
    extra = ""
    if sig_type == "S2":
        extra = f"  wicks={trade['meta']['wick_count']}"
    ax.set_title(
        f"{sig_type} {won_str} ({trade['exit_reason']})  {t_str}  "
        f"hold={hold}bars  pnl={pnl_pips:+.1f}pips{extra}",
        fontsize=7.5, color=color, pad=3,
    )

    # X-axis: show timestamps
    tick_positions = sorted({0, sig_x, fill_x, exit_x, n_slice - 1})
    tick_labels = []
    for p in tick_positions:
        idx = start + p
        if 0 <= idx < len(df):
            tick_labels.append(df.index[idx].strftime("%m-%d %H:%M"))
        else:
            tick_labels.append("")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=5, rotation=25, ha="right")
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=5, loc="lower left", framealpha=0.7)
    ax.set_xlim(-1, n_slice)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="EURUSD")
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-05-15")
    ap.add_argument("--signal", choices=["s1", "s2", "both"], default="both")
    ap.add_argument("--n", type=int, default=6, help="winners + losers each")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading {args.pair} {args.tf}  {args.start} → {args.end}…")
    df = get_mt5_data(args.pair, args.tf, args.start, args.end)
    if df.empty:
        print("No data.")
        return

    print("Detecting + simulating…")
    trades = detect_and_simulate(df, rr_ratio=args.rr)
    if not trades:
        print("No trades detected.")
        return

    if args.signal != "both":
        trades = [t for t in trades if t["sig_type"].lower() == args.signal]

    winners = [t for t in trades if t["won"]]
    losers = [t for t in trades if not t["won"]]
    print(f"  total={len(trades)}  winners={len(winners)}  losers={len(losers)}")

    s1_n = sum(1 for t in trades if t["sig_type"] == "S1")
    s2_n = sum(1 for t in trades if t["sig_type"] == "S2")
    print(f"  S1={s1_n}  S2={s2_n}")
    if trades:
        win_rate = 100 * len(winners) / len(trades)
        sum_pips = sum(t["pnl"] for t in trades) * 10000
        print(f"  win_rate={win_rate:.1f}%  sum_pips={sum_pips:+.1f}")

    def _pick(pool: list[dict], k: int) -> list[dict]:
        if len(pool) <= k:
            return pool
        return random.sample(pool, k)

    sel_winners = _pick(winners, args.n)
    sel_losers = _pick(losers, args.n)

    def _render(sample: list[dict], label: str) -> str:
        if not sample:
            print(f"  no {label}, skipping")
            return ""
        rows = (len(sample) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(18, 4.2 * rows))
        axes = np.atleast_2d(axes)
        for k, tr in enumerate(sample):
            ax = axes[k // 3, k % 3]
            plot_trade(ax, tr, df)
        # Hide unused
        for k in range(len(sample), rows * 3):
            axes[k // 3, k % 3].axis("off")
        fig.suptitle(
            f"ST34  {args.pair} {args.tf}  {args.signal.upper()}  "
            f"{label.upper()}  (RR={args.rr})",
            fontsize=11, y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        out = Path(args.out_dir) / f"st034_debug_{args.pair}_{args.tf}_{args.signal}_{label}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")
        return str(out)

    _render(sel_winners, "winners")
    _render(sel_losers, "losers")


if __name__ == "__main__":
    main()
