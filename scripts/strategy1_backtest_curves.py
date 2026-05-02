"""
Detailed backtest with equity curves & metrics for Strategy 1 (range fade).

Runs the configured range-fade strategy on every pair in strategy1_config.json
for two lot sizes (0.1 and 1.0 by default) and produces:
  - per-pair PnL/metrics table  (Sharpe, max DD, profit factor, expectancy)
  - per-pair equity curve PNGs
  - portfolio aggregate equity & drawdown PNGs
  - trades CSV for further analysis

Outputs go to data/backtest_curves/<run-stamp>/.

──────────────────────────────────────────────────────────────────────────────
LOOK-AHEAD AUDIT  (verified on 2026-05-02)

At the decision point (close of bar i), every input is causal:

  • range_high[i] = high[i-lookback : i].max()    (slice ends at i-1, excludes i)
  • range_low[i]  = low[i-lookback : i].min()                     "
  • atr[i]        depends on TR[i] = f(high[i], low[i], close[i-1])
                   and the prior atr[i-1]              — all known at close of i
  • close[i], bar_hours[i], bar_dows[i]            — known at close of i

Order of operations inside the bar loop is:
   if in_trade:  check SL/TP for this bar               (uses high[i], low[i])
   else:         check entry for this bar               (uses close[i], range[i])

Entry flips `in_trade=True`, but the SL/TP block already ran above, so the
SL/TP check for an entry made at bar i first happens at bar i+1 — never on the
same bar.  ✓ No future leakage.

──────────────────────────────────────────────────────────────────────────────
TRANSACTION COST MODEL  (IB-realistic)

  Entry  : market order — crosses HALF the spread (taker)
  TP exit: limit order  — fills at the limit price, no spread cost (maker)
  SL exit: stop order   — becomes market on trigger, crosses HALF the spread

  Commission per side = max($2, 0.20 bp × notional_usd)              [IB Pro]
                       ≈ $2 minimum dominates for trades ≤ ~1 lot

Default per-pair spreads (pips, conservative end of typical IB ranges):

       EURUSD  GBPUSD  AUDUSD  NZDUSD  USDCHF  USDCAD  USDJPY
        0.30    0.60    0.50    1.50    1.00    1.00    0.40

Tunable via --spread-mult (e.g. 0.5 for tight regimes, 2.0 for stress).
Use --no-costs to compare against ideal (mid-price, no fees).

Usage
  python -m scripts.strategy1_backtest_curves
  python -m scripts.strategy1_backtest_curves --start 2023-01-01 --end 2025-01-01
  python -m scripts.strategy1_backtest_curves --years 2022 2023 2024
  python -m scripts.strategy1_backtest_curves --lots 0.1 1.0
  python -m scripts.strategy1_backtest_curves --spread-mult 1.5    # stress
  python -m scripts.strategy1_backtest_curves --no-costs           # frictionless
  python -m scripts.strategy1_backtest_curves --no-filters
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from strategy1 import _atr, _rolling_range  # noqa: E402  reuse exact indicators
from backtest.data_fetcher import fetch_ohlcv  # noqa: E402

CONFIG_PATH = Path(__file__).parent / "strategy1_config.json"
LOT_UNITS = 100_000  # 1 standard lot in base currency
FADE = True

# IB-typical FX spreads (pips) — conservative end of normal-hours range.
# JPY pairs use 0.01 pip size; everything else uses 0.0001.
DEFAULT_SPREADS_PIPS = {
    "EURUSD": 0.30,
    "GBPUSD": 0.60,
    "AUDUSD": 0.50,
    "NZDUSD": 1.50,
    "USDCHF": 1.00,
    "USDCAD": 1.00,
    "USDJPY": 0.40,
}
IB_COMMISSION_BPS    = 0.20e-4   # 0.20 basis points of notional, per side
IB_COMMISSION_MIN    = 2.00      # USD per side, IBKR Pro tier-1


def _pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol else 0.0001


def _ib_commission_per_side(units: float, entry_price: float,
                            usd_quote: bool) -> float:
    """Notional in USD: for XXX/USD, units*price; for USD/YYY, units (base=USD)."""
    notional_usd = units * entry_price if usd_quote else units
    return max(IB_COMMISSION_MIN, IB_COMMISSION_BPS * notional_usd)


def _half_spread_cost_usd(spread_pips: float, pip_size: float, units: float,
                          entry_price: float, usd_quote: bool) -> float:
    """Cost in USD of crossing half the bid-ask spread once."""
    raw = 0.5 * spread_pips * pip_size * units
    return raw if usd_quote else raw / entry_price


# ── Trade-by-trade simulation ─────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    direction:   int           # +1 long, -1 short
    entry_price: float
    exit_price:  float
    pnl:         float         # USD, after commission
    bars_held:   int
    exit_kind:   str           # "tp", "sl", "open", "forced"
    units:       float = 0.0   # actual position size (varies for fixed-risk)
    gap_slip_usd: float = 0.0  # actual_pnl - planned_pnl. Negative for SL gap-through; positive for favorable TP gap.


@dataclass
class SimResult:
    trades:               list[Trade] = field(default_factory=list)
    final_pnl:            float       = 0.0
    n_open:               int         = 0
    n_blocked_streak:     int         = 0   # entries skipped by loss-streak gate
    n_blocked_daily_dd:   int         = 0   # entries skipped by per-pair daily-DD gate
    n_capped_size:        int         = 0   # trades where risk-sized position hit max_lot cap
    n_sl_gapped:          int         = 0   # SL trades where open gapped through stop
    n_tp_gapped:          int         = 0   # TP trades that filled at favorable gap
    total_sl_slip_usd:    float       = 0.0  # extra loss from SL gap-throughs (negative)
    total_tp_bonus_usd:   float       = 0.0  # bonus from favorable TP gaps (positive)


# Hard cap to prevent insane fixed-risk sizing if ATR collapses near zero.
# 5 standard lots of base currency = $500K notional for major pairs.
_MAX_UNITS_CAP = 5 * LOT_UNITS


def simulate_with_trades(df: pd.DataFrame, p: dict, pair_cfg: dict,
                         symbol: str,
                         lot: float | None = None,
                         risk_usd: float | None = None,
                         max_lot: float | None = None,
                         daily_dd_limit_pair: float | None = None,
                         loss_streak_stop: int | None = None,
                         close_by_hour: int | None = None,
                         no_entry_after_hour: int | None = None,
                         no_friday_after_hour: int | None = None,
                         use_filters: bool = True,
                         spread_pips: float | None = None,
                         apply_costs: bool = True) -> SimResult:
    """
    Drop-in equivalent of strategy1.simulate with risk-aware position sizing.

    Sizing (pass exactly one):
      - lot: fixed-lot mode (0.1 → 10,000 units, 1.0 → 100,000 units)
      - risk_usd: fixed-dollar-risk mode. Each trade is sized so the SL
        distance × position size = risk_usd. Position varies with ATR.

    Loss-streak cooldown:
      - loss_streak_stop=N: after N consecutive losing trades on the same
        UTC day, block new entries until the next day. A TP exit resets
        the streak. Only closed trades count (still-open trades don't).

    Overnight / weekend rules (UTC hours):
      - close_by_hour=H: force-close any open position at the close of the
        first bar with hour >= H (records exit_kind="forced")
      - no_entry_after_hour=H: block new entries when bar.hour >= H
      - no_friday_after_hour=H: block new entries on Friday when bar.hour >= H

    Cost model: commission + half-spread on entry, half-spread on SL exit,
    none on TP exit (limit order fills at limit). See module docstring.
    """
    if (lot is None) == (risk_usd is None):
        raise ValueError("Pass exactly one of `lot` or `risk_usd`")

    lookback   = p["lookback"]
    atr_period = p["atr_period"]
    tp_atr     = p["tp_atr"]
    sl_atr     = p["sl_atr"]
    tight_atr  = p.get("tight_atr")
    usd_quote  = pair_cfg["usd_quote"]
    pip_size   = _pip_size(symbol)
    spread_pips = (DEFAULT_SPREADS_PIPS.get(symbol, 1.0)
                   if spread_pips is None else spread_pips)

    close    = df["Close"].values.astype(float)
    high     = df["High"].values.astype(float)
    low      = df["Low"].values.astype(float)
    open_    = df["Open"].values.astype(float)
    atr_vals = _atr(df, atr_period)
    rh, rl   = _rolling_range(high, low, lookback)
    warmup   = atr_period + lookback + 1
    bar_hours = df.index.hour.values.astype(np.int8)
    bar_dows  = df.index.dayofweek.values.astype(np.int8)

    hour_filter = set(pair_cfg["hours"]) if use_filters else None
    dow_filter  = set(pair_cfg["days"])  if use_filters else None

    n = len(close)
    res = SimResult()
    bar_dates = df.index.normalize()  # UTC date per bar (for streak / daily-DD gates)

    in_trade    = False
    direction   = 0
    entry_price = 0.0
    tp_price    = 0.0
    sl_price    = 0.0
    entry_idx   = 0
    units       = 0.0      # set per-trade

    # Loss-streak state
    streak       = 0
    streak_day: pd.Timestamp | None = None

    # Per-pair daily-DD state — sum of trade PnL realised today on this pair
    daily_pair_pnl = 0.0
    pnl_day: pd.Timestamp | None = None

    def _pnl(exit_price: float, exit_kind: str) -> float:
        raw = (exit_price - entry_price) * units * direction
        gross = raw if usd_quote else raw / entry_price
        if not apply_costs:
            return gross
        half_spread = _half_spread_cost_usd(spread_pips, pip_size, units,
                                            entry_price, usd_quote)
        commission = 2 * _ib_commission_per_side(units, entry_price, usd_quote)
        spread_cost = half_spread + (half_spread if exit_kind == "sl" else 0.0)
        return gross - spread_cost - commission

    cap_units = (max_lot * LOT_UNITS) if max_lot is not None else _MAX_UNITS_CAP

    def _size_for(entry: float, sl: float) -> tuple[float, bool]:
        """Return (units, was_capped)."""
        if lot is not None:
            return lot * LOT_UNITS, False
        sl_distance = abs(entry - sl)
        if sl_distance <= 0:
            return 0.0, False
        # risk_usd = sl_distance * units * (1 if usd_quote else 1/entry)
        u = (risk_usd / sl_distance) if usd_quote \
            else (risk_usd * entry / sl_distance)
        if u > cap_units:
            return cap_units, True
        return u, False

    for i in range(warmup, n):
        if np.isnan(rh[i]):
            continue

        # Day rollover: reset per-pair daily PnL counter at midnight UTC
        if pnl_day != bar_dates[i]:
            pnl_day        = bar_dates[i]
            daily_pair_pnl = 0.0

        if in_trade:
            sl_hit = (low[i] <= sl_price) if direction == 1 else (high[i] >= sl_price)
            tp_hit = (high[i] >= tp_price) if direction == 1 else (low[i] <= tp_price)

            # Both-hit: use the bar's open to disambiguate. If it gapped clearly
            # through one of the levels at the open, that's where we filled.
            # Otherwise (intra-bar both touched), conservative tiebreaker = SL.
            if sl_hit and tp_hit:
                if direction == 1:
                    if open_[i] <= sl_price:
                        tp_hit = False           # gap-down through SL
                    elif open_[i] >= tp_price:
                        sl_hit = False           # gap-up through TP
                    else:
                        tp_hit = False           # ambiguous → SL
                else:
                    if open_[i] >= sl_price:
                        tp_hit = False           # gap-up through SL (short)
                    elif open_[i] <= tp_price:
                        sl_hit = False           # gap-down through TP (short)
                    else:
                        tp_hit = False

            forced = (close_by_hour is not None
                      and not (sl_hit or tp_hit)
                      and bar_hours[i] >= close_by_hour)

            if sl_hit or tp_hit or forced:
                gap_slip = 0.0
                if sl_hit:
                    kind = "sl"
                    # Stop becomes a MARKET order on trigger. If the bar gapped
                    # past the stop at the open, real fill is the open price.
                    # This is the dominant gap risk in retail FX (weekend opens,
                    # news minutes).
                    if direction == 1:
                        fill_price = min(sl_price, open_[i])
                    else:
                        fill_price = max(sl_price, open_[i])
                    pnl_planned = _pnl(sl_price, "sl")
                    pnl         = _pnl(fill_price, "sl")
                    gap_slip    = pnl - pnl_planned   # negative if gapped
                    if fill_price != sl_price:
                        res.n_sl_gapped += 1
                        res.total_sl_slip_usd += gap_slip
                elif tp_hit:
                    kind = "tp"
                    # TP is a LIMIT order. In retail FX with tight spreads and
                    # deep liquidity, limits fill at the limit price — favorable
                    # gaps don't materially improve fills. Conservative model:
                    # TP always fills at exactly tp_price.
                    fill_price = tp_price
                    pnl = _pnl(fill_price, "tp")
                else:
                    kind = "forced"
                    fill_price = close[i]
                    pnl = _pnl(fill_price, "forced")
                exit_day = bar_dates[i]
                res.trades.append(Trade(
                    entry_time=df.index[entry_idx], exit_time=df.index[i],
                    direction=direction, entry_price=entry_price,
                    exit_price=fill_price, pnl=pnl,
                    bars_held=i - entry_idx, exit_kind=kind, units=units,
                    gap_slip_usd=gap_slip))
                # Track today's pair PnL for the daily-DD circuit breaker
                daily_pair_pnl += pnl
                # Update streak based on the exit day. Forced-close trades
                # don't count toward win/loss streak — they're administrative.
                if loss_streak_stop is not None and kind in ("sl", "tp"):
                    if kind == "sl":
                        if streak_day == exit_day:
                            streak += 1
                        else:
                            streak = 1
                            streak_day = exit_day
                    else:  # TP win → reset streak on this day
                        if streak_day == exit_day:
                            streak = 0
                            streak_day = None
                in_trade = False
        else:
            if hour_filter is not None and bar_hours[i] not in hour_filter:
                continue
            if dow_filter is not None and bar_dows[i] not in dow_filter:
                continue
            if tight_atr is not None and (rh[i] - rl[i]) >= tight_atr * atr_vals[i]:
                continue
            # Overnight gate (don't open trades close to daily close)
            if no_entry_after_hour is not None and bar_hours[i] >= no_entry_after_hour:
                continue
            # Friday-evening gate (don't open trades close to weekend close).
            # bar_dows: 0=Mon … 4=Fri.
            if (no_friday_after_hour is not None and bar_dows[i] == 4
                    and bar_hours[i] >= no_friday_after_hour):
                continue
            # Per-pair daily-DD circuit breaker: stop trading this pair
            # for the rest of the UTC day if today's realised loss hit the limit.
            if (daily_dd_limit_pair is not None
                    and daily_pair_pnl <= -daily_dd_limit_pair):
                res.n_blocked_daily_dd += 1
                continue

            if close[i] > rh[i]:
                d = -1 if FADE else 1
            elif close[i] < rl[i]:
                d = 1 if FADE else -1
            else:
                continue

            # Loss-streak gate: only count actual would-be entries as "blocked".
            # Checked AFTER all other entry filters so the count reflects real
            # trades skipped, not every quiet bar.
            if (loss_streak_stop is not None
                    and streak_day == bar_dates[i]
                    and streak >= loss_streak_stop):
                res.n_blocked_streak += 1
                continue

            atr_e   = atr_vals[i]
            entry_p = close[i]
            tp_p    = entry_p + tp_atr * atr_e * d
            sl_p    = entry_p - sl_atr * atr_e * d
            u, was_capped = _size_for(entry_p, sl_p)
            if u <= 0:
                continue
            if was_capped:
                res.n_capped_size += 1

            in_trade    = True
            direction   = d
            entry_price = entry_p
            tp_price    = tp_p
            sl_price    = sl_p
            entry_idx   = i
            units       = u

    if in_trade:
        pnl = _pnl(close[n - 1], "open")
        res.trades.append(Trade(
            entry_time=df.index[entry_idx], exit_time=df.index[n - 1],
            direction=direction, entry_price=entry_price,
            exit_price=close[n - 1], pnl=pnl,
            bars_held=n - 1 - entry_idx, exit_kind="open", units=units))
        res.n_open = 1

    res.final_pnl = sum(t.pnl for t in res.trades)
    return res


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[Trade], df: pd.DataFrame) -> dict:
    if not trades:
        return {
            "n_trades": 0, "n_tp": 0, "n_sl": 0, "n_open": 0, "n_forced": 0,
            "win_rate": 0.0, "net_pnl": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "max_dd": 0.0, "sharpe": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
            "trades_per_year": 0.0,
        }

    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    closed = [t for t in trades if t.exit_kind != "open"]
    n_tp = sum(1 for t in closed if t.exit_kind == "tp")
    n_sl = sum(1 for t in closed if t.exit_kind == "sl")

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    gross_win  = float(wins.sum()) if len(wins)  else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400 if len(df) > 1 else 1
    span_years = max(span_days / 365.25, 1e-9)
    trades_per_year = len(trades) / span_years
    mean = float(pnls.mean())
    std  = float(pnls.std(ddof=1)) if len(pnls) > 1 else 0.0
    sharpe = (mean / std * np.sqrt(trades_per_year)) if std > 0 else 0.0

    return {
        "n_trades":      len(trades),
        "n_tp":          n_tp,
        "n_sl":          n_sl,
        "n_open":        sum(1 for t in trades if t.exit_kind == "open"),
        "n_forced":      sum(1 for t in trades if t.exit_kind == "forced"),
        "win_rate":      (n_tp / (n_tp + n_sl)) if (n_tp + n_sl) > 0 else 0.0,
        "net_pnl":       float(pnls.sum()),
        "avg_win":       float(wins.mean()) if len(wins) else 0.0,
        "avg_loss":      float(losses.mean()) if len(losses) else 0.0,
        "expectancy":    mean,
        "profit_factor": profit_factor,
        "max_dd":        max_dd,
        "sharpe":        sharpe,
        "best_trade":    float(pnls.max()),
        "worst_trade":   float(pnls.min()),
        "trades_per_year": trades_per_year,
    }


# ── Equity curve & plots ──────────────────────────────────────────────────────

def equity_curve(trades: list[Trade]) -> pd.Series:
    """Cumulative PnL indexed by trade-exit timestamp."""
    if not trades:
        return pd.Series(dtype=float)
    rows = [(t.exit_time, t.pnl) for t in trades]
    s = pd.DataFrame(rows, columns=["time", "pnl"]).sort_values("time")
    s["cum"] = s["pnl"].cumsum()
    return pd.Series(s["cum"].values, index=pd.DatetimeIndex(s["time"]))


def drawdown_series(eq: pd.Series) -> pd.Series:
    if eq.empty:
        return eq
    peak = eq.cummax()
    return eq - peak


def plot_pair(symbol: str, eqs: dict[str, pd.Series], out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for label, eq in eqs.items():
        if eq.empty:
            continue
        ax1.plot(eq.index, eq.values, label=label)
        ax2.plot(eq.index, drawdown_series(eq).values, label=label)
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_title(f"{symbol} — equity curve (range fade)")
    ax1.set_ylabel("Cum PnL ($)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Time (UTC)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_portfolio(eqs_per_scenario: dict[str, pd.Series], out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for label, eq in eqs_per_scenario.items():
        if eq.empty:
            continue
        ax1.plot(eq.index, eq.values, label=label)
        ax2.plot(eq.index, drawdown_series(eq).values, label=label)
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_title("Portfolio — aggregate equity curve")
    ax1.set_ylabel("Cum PnL ($)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Time (UTC)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ── Reporting ─────────────────────────────────────────────────────────────────

METRIC_COLS = [
    ("net_pnl",       "Net PnL",      lambda v: f"${v:+,.0f}"),
    ("n_trades",      "# Trades",     lambda v: f"{int(v):,}"),
    ("win_rate",      "Win%",         lambda v: f"{v*100:.1f}%"),
    ("avg_win",       "Avg Win",      lambda v: f"${v:+,.0f}"),
    ("avg_loss",      "Avg Loss",     lambda v: f"${v:+,.0f}"),
    ("expectancy",    "Expect.",      lambda v: f"${v:+,.2f}"),
    ("profit_factor", "PF",           lambda v: f"{v:.2f}" if np.isfinite(v) else "∞"),
    ("max_dd",        "Max DD",       lambda v: f"${v:+,.0f}"),
    ("sharpe",        "Sharpe",       lambda v: f"{v:.2f}"),
]


def print_metrics_table(label: str, rows: dict[str, dict]) -> None:
    print(f"\n  {label}")
    headers = ["Pair"] + [c[1] for c in METRIC_COLS]
    widths  = [10] + [12] * len(METRIC_COLS)
    fmt = lambda vals: "  ".join(f"{v:>{w}}" for v, w in zip(vals, widths))
    print("  " + fmt(headers))
    print("  " + "-" * (sum(widths) + 2 * len(widths)))
    for sym, m in rows.items():
        cells = [sym] + [fn(m[k]) for k, _, fn in METRIC_COLS]
        print("  " + fmt(cells))


def metrics_to_df(rows: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index")


# ── Driver ────────────────────────────────────────────────────────────────────

def _resolve_dates(args) -> tuple[str, str]:
    if args.start and args.end:
        return args.start, args.end
    years = args.years or [2022, 2023, 2024]
    start = f"{min(years)}-01-01"
    end   = min(date(max(years) + 1, 1, 1), date.today()).isoformat()
    return start, end


@dataclass
class Scenario:
    key:        str          # safe filename token
    label:      str          # human-readable for tables
    sim_kwargs: dict         # passed to simulate_with_trades


def build_scenarios(args) -> list[Scenario]:
    """Cartesian product of sizing × loss-streak settings."""
    sizing: list[tuple[str, dict]] = []
    for lot in (args.lots or []):
        sizing.append((f"{lot:g}lot", {"lot": lot}))
    for risk in (args.risk_usd or []):
        sizing.append((f"risk${int(risk)}", {"risk_usd": risk}))
    if not sizing:
        sizing = [("0.1lot", {"lot": 0.1}), ("1lot", {"lot": 1.0})]

    streaks: list[int | None] = [None]
    if args.loss_streak_stop:
        streaks.append(int(args.loss_streak_stop))

    out = []
    for size_key, size_kw in sizing:
        for streak in streaks:
            kw = dict(size_kw)
            if streak is None:
                key, label = size_key, size_key
            else:
                kw["loss_streak_stop"] = streak
                key   = f"{size_key}_streak{streak}"
                label = f"{size_key}+streak≤{streak}"
            out.append(Scenario(key=key, label=label, sim_kwargs=kw))
    return out


def run_backtest(args, cfg) -> None:
    p = dict(cfg["params"])
    if args.lookback  is not None: p["lookback"]   = args.lookback
    if args.tp_atr    is not None: p["tp_atr"]     = args.tp_atr
    if args.sl_atr    is not None: p["sl_atr"]     = args.sl_atr
    if args.tight_atr is not None: p["tight_atr"]  = args.tight_atr
    if args.no_tight:              p["tight_atr"]  = None

    pairs = [s for s in cfg["pairs"].keys() if s not in (args.exclude or [])]
    tf    = args.tf or p["timeframe"]
    use_filters = not args.no_filters
    apply_costs = not args.no_costs
    spread_mult = float(args.spread_mult)

    # Per-trade and per-pair-per-day risk caps — applied to every scenario
    risk_kwargs: dict = {}
    if args.max_lot is not None:
        risk_kwargs["max_lot"] = args.max_lot
    if args.daily_dd_limit_pair is not None:
        risk_kwargs["daily_dd_limit_pair"] = args.daily_dd_limit_pair

    # Overnight / weekend rules — applied uniformly across all scenarios
    overnight_kwargs = {}
    if args.no_overnight:
        # Sensible defaults: close all by 21 UTC, no entries after 20 UTC,
        # Friday entries cut off at 17 UTC. Fine-grained flags override.
        overnight_kwargs["close_by_hour"]        = 21
        overnight_kwargs["no_entry_after_hour"]  = 20
        overnight_kwargs["no_friday_after_hour"] = 17
    if args.close_by_hour is not None:
        overnight_kwargs["close_by_hour"] = args.close_by_hour
    if args.no_entry_after_hour is not None:
        overnight_kwargs["no_entry_after_hour"] = args.no_entry_after_hour
    if args.no_friday_after_hour is not None:
        overnight_kwargs["no_friday_after_hour"] = args.no_friday_after_hour

    scenarios = build_scenarios(args)

    start, end = _resolve_dates(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "data" / "backtest_curves" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput dir: {out_dir}")
    print(f"Window: {start} → {end}   tf={tf}   filters={use_filters}")
    print(f"Params: lookback={p['lookback']}  atr_period={p['atr_period']}  "
          f"tp={p['tp_atr']}×ATR  sl={p['sl_atr']}×ATR  "
          f"tight={p['tight_atr'] if p['tight_atr'] is not None else 'off'}")
    print(f"Costs : "
          + ("none" if not apply_costs
             else f"IB (spread×{spread_mult:g}, comm=max($2, 0.20bp))"))
    if risk_kwargs:
        print("Risk  : "
              + ", ".join(f"{k}={v}" for k, v in risk_kwargs.items()))
    if overnight_kwargs:
        print("Rules : "
              + ", ".join(f"{k}={v}" for k, v in overnight_kwargs.items()))
    print(f"Scenarios ({len(scenarios)}): " + ", ".join(s.label for s in scenarios))

    # Fetch once per pair, reuse across all scenarios
    pair_data: dict[str, pd.DataFrame] = {}
    for sym in pairs:
        print(f"  Fetching {sym} {tf}  {start} → {end} ...", end=" ", flush=True)
        df = fetch_ohlcv(sym, tf, start, end)
        if df.empty:
            print("no data")
            continue
        print(f"{len(df):,} bars")
        pair_data[sym] = df

    # results[scenario.key][sym] = metrics dict
    # eq_per_pair[sym][scenario.label] = equity series
    results: dict[str, dict[str, dict]] = {sc.key: {} for sc in scenarios}
    eq_per_pair: dict[str, dict[str, pd.Series]] = {}
    all_trades: list[dict] = []

    for sym, df in pair_data.items():
        eq_per_pair[sym] = {}
        spread_pips = DEFAULT_SPREADS_PIPS.get(sym, 1.0) * spread_mult
        for sc in scenarios:
            sim = simulate_with_trades(
                df, p, cfg["pairs"][sym], sym,
                use_filters=use_filters,
                spread_pips=spread_pips,
                apply_costs=apply_costs,
                **overnight_kwargs,
                **risk_kwargs,
                **sc.sim_kwargs,
            )
            mt = compute_metrics(sim.trades, df)
            avg_lot = (np.mean([t.units for t in sim.trades]) / LOT_UNITS
                       if sim.trades else 0.0)
            max_lot = (max(t.units for t in sim.trades) / LOT_UNITS
                       if sim.trades else 0.0)
            mt["avg_lot"]            = avg_lot
            mt["max_lot"]            = max_lot
            mt["n_blocked_streak"]   = sim.n_blocked_streak
            mt["n_blocked_daily_dd"] = sim.n_blocked_daily_dd
            mt["n_capped_size"]      = sim.n_capped_size
            mt["n_sl_gapped"]        = sim.n_sl_gapped
            mt["total_sl_slip"]      = sim.total_sl_slip_usd
            mt["n_tp_gapped"]        = sim.n_tp_gapped
            mt["total_tp_bonus"]     = sim.total_tp_bonus_usd
            results[sc.key][sym] = mt
            eq_per_pair[sym][sc.label] = equity_curve(sim.trades)
            for t in sim.trades:
                all_trades.append({
                    "scenario":    sc.key,
                    "symbol":      sym,
                    "entry_time":  t.entry_time,
                    "exit_time":   t.exit_time,
                    "direction":   t.direction,
                    "entry_price": t.entry_price,
                    "exit_price":  t.exit_price,
                    "units":       t.units,
                    "pnl":         t.pnl,
                    "bars_held":   t.bars_held,
                    "exit_kind":   t.exit_kind,
                })

    # Per-pair equity-curve PNGs (all scenarios on one chart per pair)
    for sym, eqs in eq_per_pair.items():
        plot_pair(sym, eqs, out_dir / f"equity_{sym}.png")

    # Portfolio aggregate per scenario
    portfolio_eqs: dict[str, pd.Series] = {}
    for sc in scenarios:
        merged = []
        for sym, eqs in eq_per_pair.items():
            eq = eqs.get(sc.label)
            if eq is None or eq.empty:
                continue
            merged.append(eq.diff().fillna(eq.iloc[0]))
        if merged:
            evt = pd.concat(merged).sort_index()
            portfolio_eqs[sc.label] = evt.cumsum()
        else:
            portfolio_eqs[sc.label] = pd.Series(dtype=float)
    plot_portfolio(portfolio_eqs, out_dir / "equity_portfolio.png")

    # ── PORTFOLIO COMPARISON TABLE (the headline) ───────────────────────────
    print(f"\n  === PORTFOLIO COMPARISON ===")
    print(f"  {'Scenario':<22}  {'Trades':>7}  {'BlkStrk':>7}  {'BlkDD':>6}  "
          f"{'Capped':>6}  {'Win%':>6}  {'Net PnL':>12}  {'MaxDD':>11}  "
          f"{'AvgLot':>7}  {'MaxLot':>7}")
    print(f"  {'-' * 115}")
    for sc in scenarios:
        n_trades   = sum(m["n_trades"]            for m in results[sc.key].values())
        n_tp       = sum(m["n_tp"]                for m in results[sc.key].values())
        n_sl       = sum(m["n_sl"]                for m in results[sc.key].values())
        n_strk     = sum(m["n_blocked_streak"]    for m in results[sc.key].values())
        n_dd       = sum(m["n_blocked_daily_dd"]  for m in results[sc.key].values())
        n_capped   = sum(m["n_capped_size"]       for m in results[sc.key].values())
        wr         = (n_tp / (n_tp + n_sl)) if (n_tp + n_sl) else 0.0
        eq         = portfolio_eqs[sc.label]
        net        = float(eq.iloc[-1]) if not eq.empty else 0.0
        dd         = float(drawdown_series(eq).min()) if not eq.empty else 0.0
        avg_lots   = [m["avg_lot"] for m in results[sc.key].values() if m["n_trades"]]
        max_lots_  = [m["max_lot"] for m in results[sc.key].values() if m["n_trades"]]
        avg_lot    = float(np.mean(avg_lots)) if avg_lots else 0.0
        max_lot_   = float(max(max_lots_))    if max_lots_ else 0.0
        print(f"  {sc.label:<22}  {n_trades:>7,}  {n_strk:>7,}  {n_dd:>6,}  "
              f"{n_capped:>6,}  {wr*100:>5.1f}%  ${net:>+11,.0f}  ${dd:>+10,.0f}  "
              f"{avg_lot:>7.2f}  {max_lot_:>7.2f}")

    # ── GAP IMPACT (asymmetric: SL stops can gap through; TP limits don't) ──
    print(f"\n  === SL GAP IMPACT ===")
    print(f"  Stops that gapped through (real fill at next-bar open, "
          f"not the stop price)")
    print(f"  {'Scenario':<22}  {'SL trades':>10}  {'SL gapped':>11}  "
          f"{'% gapped':>10}  {'Slip cost $':>13}  {'Avg slip $':>11}")
    print(f"  {'-' * 85}")
    for sc in scenarios:
        n_sl     = sum(m["n_sl"]          for m in results[sc.key].values())
        n_sl_gap = sum(m["n_sl_gapped"]   for m in results[sc.key].values())
        sl_slip  = sum(m["total_sl_slip"] for m in results[sc.key].values())
        pct = (n_sl_gap / n_sl * 100) if n_sl else 0.0
        avg = (sl_slip / n_sl_gap)     if n_sl_gap else 0.0
        print(f"  {sc.label:<22}  {n_sl:>10,}  {n_sl_gap:>11,}  "
              f"{pct:>9.1f}%  ${sl_slip:>+12,.0f}  ${avg:>+10,.0f}")

    # Per-scenario per-pair tables
    for sc in scenarios:
        print_metrics_table(f"=== Per-pair metrics  [{sc.label}] ===", results[sc.key])
        metrics_to_df(results[sc.key]).to_csv(out_dir / f"metrics_{sc.key}.csv")

    # ── ACCOUNT SIZE ESTIMATE (IB FX, fixed-lot scenarios only) ─────────────
    fixed_lot_scenarios = [(sc, sc.sim_kwargs.get("lot")) for sc in scenarios
                           if "lot" in sc.sim_kwargs]
    if fixed_lot_scenarios:
        margin_pct = args.account_margin_pct / 100.0
        # Per-pair mean price → notional in USD per 1 unit of base
        notional_per_unit_usd: dict[str, float] = {}
        for sym, df in pair_data.items():
            mean_px = float(df["Close"].mean())
            usd_quote = cfg["pairs"][sym].get("usd_quote", True)
            # XXX/USD: notional/unit (USD) = mean_price.
            # USD/YYY: notional/unit (USD) = 1 (because base=USD).
            notional_per_unit_usd[sym] = mean_px if usd_quote else 1.0

        # Pick the largest fixed-lot scenario for the headline estimate
        max_lot = max(lot for _, lot in fixed_lot_scenarios)

        # Worst-case portfolio MaxDD across the fixed-lot scenarios
        max_dd_observed = 0.0
        for sc, _ in fixed_lot_scenarios:
            eq = portfolio_eqs.get(sc.label, pd.Series(dtype=float))
            if not eq.empty:
                dd = float(drawdown_series(eq).min())
                max_dd_observed = min(max_dd_observed, dd)
        max_dd_abs = abs(max_dd_observed)

        print(f"\n  === ACCOUNT SIZE ESTIMATE (IB FX, {max_lot:g} lot per pair, "
              f"margin {args.account_margin_pct:g}%) ===")
        print(f"  {'Pair':<8}  {'Mean px':>10}  {'Notional/lot':>14}  {'Margin/lot':>12}")
        total_margin = 0.0
        for sym, df in pair_data.items():
            mean_px = float(df["Close"].mean())
            usd_quote = cfg["pairs"][sym].get("usd_quote", True)
            units = max_lot * LOT_UNITS
            notional = units * notional_per_unit_usd[sym]
            margin   = notional * margin_pct
            total_margin += margin
            px_str = f"{mean_px:.4f}" if usd_quote else f"{mean_px:.4f}*"
            print(f"  {sym:<8}  {px_str:>10}  ${notional:>13,.0f}  ${margin:>11,.0f}")
        print(f"  {'─' * 50}")
        print(f"  Concurrent margin if all {len(pair_data)} pairs in trade: ${total_margin:,.0f}")

        print(f"\n  Backtest portfolio MaxDD (worst across fixed-lot scenarios): ${max_dd_abs:,.0f}")
        rec_aggressive = total_margin + 1.0 * max_dd_abs
        rec_moderate   = total_margin + 3.0 * max_dd_abs
        rec_safe       = (total_margin + 3.0 * max_dd_abs) * 1.30
        print(f"\n  Recommended minimum account size:")
        print(f"    Aggressive (margin + 1× DD)              : ${rec_aggressive:>10,.0f}")
        print(f"    Moderate   (margin + 3× DD)              : ${rec_moderate:>10,.0f}")
        print(f"    Safe       (margin + 3× DD + 30% buffer) : ${rec_safe:>10,.0f}")
        print(f"\n  Notes:")
        print(f"    * Mean price is computed over the backtest window. Actual margin")
        print(f"      will vary intraday with FX prices.")
        print(f"    * IB FX margin varies by leverage tier (~2-5%); tune --account-margin-pct.")
        print(f"    * MaxDD is in-sample. Real-world DD can be 1.5-2× backtest, hence the 3× rule.")

    # Trades CSV
    if all_trades:
        pd.DataFrame(all_trades).to_csv(out_dir / "trades.csv", index=False)
        print(f"\n  Wrote {len(all_trades):,} trades → {out_dir / 'trades.csv'}")

    # Summary JSON
    summary = {
        "window":      {"start": start, "end": end},
        "params":      p,
        "filters":     use_filters,
        "scenarios":   [{"key": s.key, "label": s.label, "kwargs": s.sim_kwargs}
                        for s in scenarios],
        "costs":       {
            "applied":     apply_costs,
            "spread_mult": spread_mult,
            "ib_commission_min_per_side": IB_COMMISSION_MIN,
            "ib_commission_bps":          IB_COMMISSION_BPS,
        },
        "results": {sc.key: {sym: {k: (v if np.isfinite(v) else None)
                                   for k, v in m.items()}
                             for sym, m in results[sc.key].items()}
                    for sc in scenarios},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Plots & metrics in {out_dir}")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start",      default=None)
    ap.add_argument("--end",        default=None)
    ap.add_argument("--years",      nargs="+", type=int, default=None,
                    help="Calendar years (default: 2022 2023 2024)")
    ap.add_argument("--lots",       nargs="+", type=float, default=[],
                    help="Fixed-lot scenarios (e.g. --lots 0.1 1.0). If neither "
                         "--lots nor --risk-usd is given, defaults to 0.1 and 1.0.")
    ap.add_argument("--risk-usd",   nargs="+", type=float, default=[],
                    help="Fixed-risk scenarios in USD (e.g. --risk-usd 50 100). "
                         "Each trade is sized so SL distance × position = risk.")
    ap.add_argument("--loss-streak-stop", type=int, default=None,
                    help="Stop entries on a pair after N consecutive losses on "
                         "the same UTC day (resets at midnight or on a TP). "
                         "Adds a parallel set of streak-gated scenarios for "
                         "side-by-side comparison.")
    ap.add_argument("--tf",         default=None,
                    help="Timeframe override (e.g. 1m, 5m, 15m, 1h, 4h). "
                         "Default: value from strategy1_config.json (1m).")
    ap.add_argument("--lookback",   type=int, default=None,
                    help="Range lookback bars (default: from config)")
    ap.add_argument("--tight-atr",  type=float, default=None,
                    help="Tightness filter: only trade when range < N×ATR. "
                         "Default: from config (1.5).")
    ap.add_argument("--no-tight",   action="store_true",
                    help="Disable the tightness filter entirely")
    ap.add_argument("--tp-atr",     type=float, default=None,
                    help="TP in ATR multiples (default: from config)")
    ap.add_argument("--sl-atr",     type=float, default=None,
                    help="SL in ATR multiples (default: from config)")
    ap.add_argument("--no-filters", action="store_true",
                    help="Ignore hour/day filters")
    ap.add_argument("--exclude",    nargs="+", default=None,
                    help="Pairs to drop from the run (e.g. --exclude GBPUSD)")
    ap.add_argument("--no-overnight", action="store_true",
                    help="Apply intraday-only defaults: close any open trade "
                         "at 21:00 UTC, block new entries after 20:00 UTC, "
                         "and block all Friday entries after 17:00 UTC.")
    ap.add_argument("--close-by-hour", type=int, default=None,
                    help="Force-close all open trades at the close of the "
                         "first bar with hour ≥ N (UTC). Overrides --no-overnight.")
    ap.add_argument("--no-entry-after-hour", type=int, default=None,
                    help="Block new entries when bar.hour ≥ N (UTC). "
                         "Overrides --no-overnight.")
    ap.add_argument("--no-friday-after-hour", type=int, default=None,
                    help="On Friday only, block new entries when bar.hour ≥ N. "
                         "Overrides --no-overnight.")
    ap.add_argument("--account-margin-pct", type=float, default=5.0,
                    help="IB FX margin requirement to assume in the account-size "
                         "estimate (default 5%% — conservative).")
    ap.add_argument("--max-lot", type=float, default=None,
                    help="Hard cap on position size in lots, applied on top of "
                         "risk-based sizing. Trades that would need >cap to hit "
                         "the risk target are capped (and risk less than target).")
    ap.add_argument("--daily-dd-limit-pair", type=float, default=None,
                    help="Per-pair daily loss limit in USD. When today's "
                         "realised PnL on a pair drops to -limit, block any new "
                         "entries on THAT pair until the next UTC day.")
    ap.add_argument("--spread-mult", type=float, default=1.0,
                    help="Multiply default per-pair spreads (1.0=normal, "
                         "0.5=tight regime, 2.0=stress; default 1.0)")
    ap.add_argument("--no-costs",   action="store_true",
                    help="Frictionless mode: no spread, no commission")
    args = ap.parse_args()

    cfg = load_config()
    run_backtest(args, cfg)


if __name__ == "__main__":
    main()
