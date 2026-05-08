"""
Visualize sample trades from EMA-extrema strategy.

Plots candlestick charts with EMA, entry/exit markers, and the lookback
window that generated the signal.

Usage
-----
  python -m scripts.visualize_ema_trades \\
      --login 1513313327 --server FTMO-Demo \\
      --pair EURUSD --timeframe 1h \\
      --winning   (show winners instead of losers)
      --n 6       (how many charts)
"""
from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.strategy1_regime_backtest import (
    _load_data_mt5,
    _ema, _atr,
    HALF_SPREAD_PIPS, PIP_VALUE, PIP_SIZE, COMMISSION_PER_LOT,
)

# ── Simulation (records bar indices + direction) ──────────────────────────────

def simulate(
    pair: str, df: pd.DataFrame,
    ema_period:        int   = 20,
    window:            int   = 20,
    lots:              float = 1.0,
    close_profit:    bool  = False,
    min_swing_pips:  float = 0.0,
) -> tuple[list[dict], np.ndarray]:
    """Returns (trades, ema_values). Trades include entry_idx / exit_idx."""
    opens  = df["Open"].values
    lows   = df["Low"].values
    highs  = df["High"].values
    closes = df["Close"].values
    times  = df.index

    hl2   = (highs + lows) / 2.0
    ema_v = _ema(_ema(hl2, ema_period), ema_period)   # double EMA of HL/2
    atr_v = _atr(df, 14)

    pip_val = PIP_VALUE[pair]
    sp_cost = HALF_SPREAD_PIPS[pair] * pip_val * lots
    comm    = COMMISSION_PER_LOT * lots

    def _net(gross: float) -> float:
        return gross - 2.0 * sp_cost - comm

    min_swing = min_swing_pips * PIP_SIZE

    def _sig(i: int) -> int:
        dw      = ema_v[i - window + 1 : i + 1]
        pos_max = int(np.argmax(dw))
        pos_min = int(np.argmin(dw))
        if pos_max == pos_min:
            return 0
        if (dw[pos_max] - dw[pos_min]) < min_swing:
            return 0
        if pos_max > pos_min and pos_max < window - 1:
            return -1
        if pos_min > pos_max and pos_min < window - 1:
            return 1
        return 0

    warmup   = ema_period + window + 15
    position = 0
    trade: dict | None = None
    trades: list[dict] = []

    for i in range(warmup, len(df)):
        bar_high  = highs[i]
        bar_low   = lows[i]
        bar_close = closes[i]
        bar_time  = times[i]

        if trade is not None:
            # Intrabar profit-close: check bar_high (LONG) or bar_low (SHORT)
            if close_profit:
                check_price = bar_high if position == 1 else bar_low
                pip_move    = (check_price - trade["entry"]) * position / PIP_SIZE
                gross       = pip_move * pip_val * lots
                if _net(gross) > 0:
                    trades.append({
                        "pair":      pair,
                        "direction": position,
                        "entry_idx": trade["bar_idx"],
                        "exit_idx":  i,
                        "entry_t":   trade["time"],
                        "exit_t":    bar_time,
                        "entry":     trade["entry"],
                        "exit":      check_price,
                        "net":       _net(gross),
                        "won":       True,
                        "hold_bars": i - trade["bar_idx"],
                        "signal_start": trade["bar_idx"] - window + 1,
                    })
                    position = 0
                    trade    = None
                    continue

            sig = _sig(i)
            if sig != 0 and sig != position:
                pip_move = (bar_close - trade["entry"]) * position / PIP_SIZE
                gross    = pip_move * pip_val * lots
                net      = _net(gross)
                trades.append({
                    "pair":      pair,
                    "direction": position,
                    "entry_idx": trade["bar_idx"],
                    "exit_idx":  i,
                    "entry_t":   trade["time"],
                    "exit_t":    bar_time,
                    "entry":     trade["entry"],
                    "exit":      bar_close,
                    "net":       net,
                    "won":       net > 0,
                    "hold_bars": i - trade["bar_idx"],
                    "signal_start": trade["bar_idx"] - window + 1,
                })
                position = sig
                trade = {"time": bar_time, "bar_idx": i, "entry": bar_close,
                         "atr": atr_v[i]}
            continue

        sig = _sig(i)
        if sig != 0:
            position = sig
            trade = {"time": bar_time, "bar_idx": i, "entry": bar_close,
                     "atr": atr_v[i]}

    return trades, ema_v


# ── Candlestick drawing ───────────────────────────────────────────────────────

def draw_candles(ax: plt.Axes, df_slice: pd.DataFrame) -> None:
    for x, (_, row) in enumerate(df_slice.iterrows()):
        bull   = row["Close"] >= row["Open"]
        body_c = "#26a69a" if bull else "#ef5350"
        wick_c = "#888888"
        lo_body = min(row["Open"], row["Close"])
        hi_body = max(row["Open"], row["Close"])
        ax.bar(x, hi_body - lo_body, bottom=lo_body,
               width=0.7, color=body_c, linewidth=0)
        ax.plot([x, x], [row["Low"],  lo_body], color=wick_c, linewidth=0.8)
        ax.plot([x, x], [hi_body, row["High"]], color=wick_c, linewidth=0.8)


# ── Per-trade chart ───────────────────────────────────────────────────────────

def plot_trade(ax: plt.Axes, trade: dict, df: pd.DataFrame,
               ema_v: np.ndarray, ema_period: int, window: int) -> None:
    entry_i = trade["entry_idx"]
    exit_i  = trade["exit_idx"]

    pad_before = max(window + 5, 30)
    pad_after  = 8
    start = max(0, entry_i - pad_before)
    end   = min(len(df), exit_i + pad_after + 1)

    df_slice  = df.iloc[start:end].copy()
    ema_slice = ema_v[start:end]
    xs        = np.arange(len(df_slice))

    entry_x = entry_i - start
    exit_x  = exit_i  - start
    sig_x0  = max(0, trade["signal_start"] - start)   # start of lookback window

    # Signal lookback window (light background)
    ax.axvspan(sig_x0, entry_x, alpha=0.08, color="#aaaaff", label="_nolegend_")

    # Candles + EMA
    draw_candles(ax, df_slice)
    ax.plot(xs, ema_slice, color="#2962ff", linewidth=1.3, label=f"EMA{ema_period}", zorder=3)

    # Trade region shading
    shade_color = "#ff3333" if not trade["won"] else "#33cc33"
    ax.axvspan(entry_x, exit_x, alpha=0.15, color=shade_color, zorder=1)

    # Entry marker
    direction = trade["direction"]
    pip = PIP_SIZE
    if direction == 1:   # long
        y_arrow = df.iloc[entry_i]["Low"] - 4 * pip
        ax.annotate("", xy=(entry_x, df.iloc[entry_i]["Low"]),
                    xytext=(entry_x, y_arrow),
                    arrowprops=dict(arrowstyle="->", color="#00c853", lw=1.5))
        ax.text(entry_x, y_arrow - 2 * pip, "BUY", ha="center", va="top",
                fontsize=6, color="#00c853", fontweight="bold")
    else:                # short
        y_arrow = df.iloc[entry_i]["High"] + 4 * pip
        ax.annotate("", xy=(entry_x, df.iloc[entry_i]["High"]),
                    xytext=(entry_x, y_arrow),
                    arrowprops=dict(arrowstyle="->", color="#d50000", lw=1.5))
        ax.text(entry_x, y_arrow + 2 * pip, "SELL", ha="center", va="bottom",
                fontsize=6, color="#d50000", fontweight="bold")

    # Entry price line
    ax.axhline(trade["entry"], color="#888888", linewidth=0.6,
               linestyle="--", xmax=(exit_x + 0.5) / len(df_slice), alpha=0.6)

    # MFE line: max favorable excursion during the trade (shows if profit was reachable)
    trade_slice = df.iloc[entry_i:exit_i + 1]
    if direction == 1:
        mfe_price = trade_slice["High"].max()
    else:
        mfe_price = trade_slice["Low"].min()
    ax.axhline(mfe_price, color="#ffeb3b", linewidth=0.8,
               linestyle=":", xmax=(exit_x + 0.5) / len(df_slice), alpha=0.8)

    # Exit marker
    ax.axvline(exit_x, color="#ff6f00", linewidth=1.2, linestyle="--", alpha=0.9)
    result_color = "#2e7d32" if trade["won"] else "#c62828"

    # X-axis: show timestamps at entry and exit
    tick_positions = sorted({0, entry_x, exit_x, len(df_slice) - 1})
    tick_labels    = []
    for p in tick_positions:
        idx = start + p
        if 0 <= idx < len(df):
            tick_labels.append(df.index[idx].strftime("%m-%d %Hh"))
        else:
            tick_labels.append("")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=5, rotation=20, ha="right")
    ax.tick_params(axis="y", labelsize=6)

    # Title
    d_str  = "LONG" if direction == 1 else "SHORT"
    net    = trade["net"]
    t_str  = trade["entry_t"].strftime("%Y-%m-%d %H:%M")
    hold   = trade["hold_bars"]
    ax.set_title(
        f"{trade['pair']}  {d_str}  |  entry: {t_str}  |  "
        f"hold: {hold}h  |  net: ${net:+.0f}",
        fontsize=7.5, pad=3,
        color=result_color,
    )
    ax.set_xlim(-1, len(df_slice))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair",       default="EURUSD")
    ap.add_argument("--timeframe",  default="1h")
    ap.add_argument("--start",      default="2026-01-01")
    ap.add_argument("--end",        default="2026-05-07")
    ap.add_argument("--ema-period", type=int,   default=20)
    ap.add_argument("--window",     type=int,   default=20)
    ap.add_argument("--n",          type=int,   default=6,  help="charts to draw")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--winning",       action="store_true", help="show winners instead")
    ap.add_argument("--close-profit",  action="store_true", help="exit intrabar when high/low is profitable")
    ap.add_argument("--min-swing",  type=float, default=0.0, help="min EMA swing from prior opposite extremum (pips) to qualify signal")
    ap.add_argument("--login",      type=int,   default=None)
    ap.add_argument("--password",   default=None)
    ap.add_argument("--server",     default=None)
    ap.add_argument("--out",        default="trade_samples.png")
    args = ap.parse_args()

    print(f"Loading {args.timeframe} data for {args.pair}…")
    try:
        pairs_data = _load_data_mt5(
            [args.pair], args.start, args.end,
            timeframe=args.timeframe,
            login=args.login, password=args.password, server=args.server,
        )
    except Exception as e:
        print(f"MT5 load failed: {e}"); return

    df = pairs_data.get(args.pair)
    if df is None or df.empty:
        print("No data."); return
    print(f"  {len(df)} bars loaded.")

    trades, ema_v = simulate(
        args.pair, df,
        ema_period=args.ema_period, window=args.window,
        close_profit=args.close_profit,
        min_swing_pips=args.min_swing,
    )

    subset = [t for t in trades if t["won"] == args.winning]
    label  = "WINNING" if args.winning else "LOSING"
    print(f"  Total trades: {len(trades)}  |  {label}: {len(subset)}")
    if not subset:
        print("No matching trades."); return

    random.seed(args.seed)
    sample = random.sample(subset, min(args.n, len(subset)))
    sample.sort(key=lambda t: t["entry_idx"])   # chronological order

    ncols  = min(3, len(sample))
    nrows  = (len(sample) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(7 * ncols, 5 * nrows),
                              facecolor="#1a1a2e")
    axes = np.array(axes).flatten() if len(sample) > 1 else [axes]

    for ax in axes:
        ax.set_facecolor("#131722")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    for ax, trade in zip(axes, sample):
        plot_trade(ax, trade, df, ema_v, args.ema_period, args.window)

    # Hide unused subplots
    for ax in axes[len(sample):]:
        ax.set_visible(False)

    ema_patch  = mpatches.Patch(color="#2962ff",  label=f"EMA {args.ema_period}")
    win_patch  = mpatches.Patch(color="#26a69a",  label="Bullish candle")
    lose_patch = mpatches.Patch(color="#ef5350",  label="Bearish candle")
    sig_patch  = mpatches.Patch(color="#aaaaff",  alpha=0.4, label=f"Signal window ({args.window}b)")
    mfe_patch  = mpatches.Patch(color="#ffeb3b",  label="MFE (max favorable excursion)")
    fig.legend(handles=[ema_patch, win_patch, lose_patch, sig_patch, mfe_patch],
               loc="lower center", ncol=4, fontsize=8,
               facecolor="#1a1a2e", labelcolor="#cccccc", framealpha=0.5)

    title = (f"{args.pair} {args.timeframe}  |  EMA{args.ema_period} extrema, "
             f"window={args.window}  |  {label} trades  (seed={args.seed})")
    fig.suptitle(title, fontsize=10, color="#dddddd", y=1.01)
    fig.tight_layout()

    out = Path(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved → {out.resolve()}")


if __name__ == "__main__":
    main()
