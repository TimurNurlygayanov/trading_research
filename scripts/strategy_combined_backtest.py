"""
1h double-EMA extrema strategy — backtest.

(File historically named "combined" because it modelled 1h + 5m together.
 5m has been removed for FTMO compliance and to lower trade frequency.
 The filename is preserved for shell-script compatibility.)

Position rules per pair:
  - 1h signal fires from double-EMA argmax/argmin in a sliding window
  - Entry on signal; exit on opposite signal (with min-hold), profit-close, or hard SL

FTMO compliance (joint simulation, default):
  - Hedge filter: blocks any new entry that would create OPPOSING exposure on
    a currency already held (e.g. LONG EURUSD blocks SHORT GBPUSD, LONG USDCHF,
    SHORT EURJPY).
  - Cumulative correlation cap: --max-ccy-exposure limits how many concurrent
    positions can share the same currency direction.

Daily circuit-breaker:
  - Tracks daily P&L per pair; pauses pair if day P&L < -daily_stop

Usage
-----
  python -m scripts.strategy_combined_backtest \\
      --login 1513313327 --server FTMO-Demo \\
      --start 2026-01-01 --end 2026-05-07 --oos-start 2026-04-01 \\
      --ema-1h 9 --window 20 --lots-1h 1 \\
      --max-ccy-exposure 2

  # Diagnostic per-pair mode (no cross-pair FTMO filter):
  python -m scripts.strategy_combined_backtest ... --per-pair
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.strategy1_regime_backtest import (
    _load_data_mt5, _ema, _atr,
    PAIRS, HALF_SPREAD_PIPS, PIP_VALUE, PIP_SIZE, PIP_SIZE_MAP, COMMISSION_PER_LOT,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("backtest_1h")


# ── Currency exposure (FTMO hedge & correlation filter) ──────────────────────

def position_currencies(symbol: str, direction: int) -> dict[str, int]:
    """
    Decompose a position into per-currency exposure.
      LONG  EURUSD  → {EUR: +1, USD: -1}
      SHORT USDJPY  → {USD: -1, JPY: +1}
    Assumes a 6-char BASEQUO symbol.
    """
    return {symbol[:3]: direction, symbol[3:6]: -direction}


# ── Signal precomputation ─────────────────────────────────────────────────────

def _compute_signals(df: pd.DataFrame, ema_period: int, window: int,
                     min_swing_pips: float, early_entry: bool = False,
                     pip_size: float = PIP_SIZE) -> np.ndarray:
    """
    Return signal array (−1/0/+1) per bar using double-EMA of HL/2.

    early_entry=False (default): fire only when peak/valley is at least 1 bar old
                                 (pos < window-1).  Standard confirmed signal.
    early_entry=True:            also fire when current bar IS the peak/valley
                                 (pos == window-1).  One bar earlier — enter at
                                 the extremum bar's close rather than the bar after.
    """
    highs = df["High"].values
    lows  = df["Low"].values
    hl2   = (highs + lows) / 2.0
    ema_v = _ema(_ema(hl2, ema_period), ema_period)
    n     = len(ema_v)
    sigs  = np.zeros(n, dtype=int)
    min_sw = min_swing_pips * pip_size
    warmup = ema_period * 2 + window + 5
    max_pos = window - 1 if early_entry else window - 2

    for i in range(warmup, n):
        dw      = ema_v[i - window + 1 : i + 1]
        pos_max = int(np.argmax(dw))
        pos_min = int(np.argmin(dw))
        if pos_max == pos_min:
            continue
        if (dw[pos_max] - dw[pos_min]) < min_sw:
            continue
        if pos_max > pos_min and pos_max <= max_pos:
            sigs[i] = -1
        elif pos_min > pos_max and pos_min <= max_pos:
            sigs[i] = 1
    return sigs


# ── Per-pair simulation (no cross-pair FTMO filter) ──────────────────────────

def simulate_pair(
    pair:           str,
    df_1h:          pd.DataFrame,
    ema_1h:         int,
    window:         int,
    min_swing:      float,
    lots_1h:        float,
    close_profit:   bool  = True,
    daily_stop_usd: float = 0.0,
    min_bars_1h:    int   = 2,
    partial_usd:    float = 0.0,
    n_exit:         int   = 0,
    sl_pips_1h:     float = 0.0,
    tp_pips_1h:     float = 0.0,
    max_dist_pips:  float = 0.0,
    early_1h:       bool  = True,
    reverse_signals: bool  = False,
    atr_period:     int   = 14,
    sl_atr_mult:    float = 0.0,
    tp_atr_mult:    float = 0.0,
) -> list[dict]:
    """
    Backtest a single pair on 1h bars.  No cross-pair correlation filter —
    use simulate_joint() for FTMO-compliant multi-pair simulation.
    """
    pip_val = PIP_VALUE[pair]
    sp      = HALF_SPREAD_PIPS[pair] * pip_val
    comm    = COMMISSION_PER_LOT
    _ps     = PIP_SIZE_MAP.get(pair, PIP_SIZE)
    atr_on  = sl_atr_mult > 0 or tp_atr_mult > 0

    def _net(gross: float, lots: float) -> float:
        return gross - 2.0 * sp * lots - comm * lots

    sigs = _compute_signals(df_1h, ema_1h, window, min_swing,
                            early_entry=early_1h, pip_size=_ps)
    dema = _ema(_ema((df_1h["High"].values + df_1h["Low"].values) / 2.0,
                     ema_1h), ema_1h)
    atr_v = _atr(df_1h, atr_period) if atr_on else None

    def _empty_pos() -> dict:
        return {"dir": 0, "entry": 0.0, "idx": -1, "time": None,
                "tf": "1h", "lots_rem": 0.0, "partial_done": False,
                "atr_entry": 0.0}

    pos: dict       = _empty_pos()
    trades: list[dict] = []
    cur_day         = None
    day_pnl         = 0.0
    day_pause       = False

    def _close(p: dict, exit_px: float, exit_ts, lots: float,
               exit_type: str, hold_bars: int) -> dict:
        pm  = (exit_px - p["entry"]) * p["dir"] / _ps
        net = _net(pm * pip_val * lots, lots)
        return {"time": p["time"], "close_time": exit_ts, "pair": pair,
                "tf": "1h", "direction": p["dir"], "entry": p["entry"],
                "exit": exit_px, "exit_type": exit_type, "net": net,
                "won": net > 0, "hold_bars": hold_bars}

    if n_exit > 0:
        prev_lo = pd.Series(df_1h["Low"].values).rolling(n_exit).min().shift(1).fillna(np.inf).values
        prev_hi = pd.Series(df_1h["High"].values).rolling(n_exit).max().shift(1).fillna(-np.inf).values
    else:
        prev_lo = prev_hi = None

    def _exit_confirmed(direction: int, i: int) -> bool:
        if prev_lo is None:
            return True
        if direction == 1:
            return df_1h["Low"].iloc[i] < prev_lo[i]
        return df_1h["High"].iloc[i] > prev_hi[i]

    for i in range(len(df_1h)):
        ts = df_1h.index[i]
        h  = float(df_1h["High"].iloc[i])
        l  = float(df_1h["Low"].iloc[i])
        c  = float(df_1h["Close"].iloc[i])

        # New day — reset daily P&L
        day = ts.date()
        if day != cur_day:
            cur_day = day
            day_pnl = 0.0
            day_pause = False

        # Hard SL — checked intrabar via bar's H/L
        if sl_pips_1h > 0 and pos["dir"] != 0:
            sl_px = pos["entry"] - pos["dir"] * sl_pips_1h * _ps
            sl_hit = (pos["dir"] == 1 and l <= sl_px) or \
                     (pos["dir"] == -1 and h >= sl_px)
            if sl_hit:
                t = _close(pos, sl_px, ts, pos["lots_rem"], "sl_1h",
                           i - pos["idx"])
                trades.append(t); day_pnl += t["net"]
                pos = _empty_pos()

        # Hard TP — checked intrabar via bar's H/L (SL above takes priority)
        if tp_pips_1h > 0 and pos["dir"] != 0:
            tp_px = pos["entry"] + pos["dir"] * tp_pips_1h * _ps
            tp_hit = (pos["dir"] == 1 and h >= tp_px) or \
                     (pos["dir"] == -1 and l <= tp_px)
            if tp_hit:
                t = _close(pos, tp_px, ts, pos["lots_rem"], "tp_1h",
                           i - pos["idx"])
                trades.append(t); day_pnl += t["net"]
                pos = _empty_pos()

        # ATR-based SL/TP — checked intrabar via bar's H/L.
        # SL has priority over TP when both could hit on the same bar.
        if atr_on and pos["dir"] != 0 and pos["atr_entry"] > 0:
            if sl_atr_mult > 0:
                sl_px = pos["entry"] - pos["dir"] * sl_atr_mult * pos["atr_entry"]
                sl_hit = (pos["dir"] == 1 and l <= sl_px) or \
                         (pos["dir"] == -1 and h >= sl_px)
                if sl_hit:
                    t = _close(pos, sl_px, ts, pos["lots_rem"], "sl_atr",
                               i - pos["idx"])
                    trades.append(t); day_pnl += t["net"]
                    pos = _empty_pos()
            if pos["dir"] != 0 and tp_atr_mult > 0:
                tp_px = pos["entry"] + pos["dir"] * tp_atr_mult * pos["atr_entry"]
                tp_hit = (pos["dir"] == 1 and h >= tp_px) or \
                         (pos["dir"] == -1 and l <= tp_px)
                if tp_hit:
                    t = _close(pos, tp_px, ts, pos["lots_rem"], "tp_atr",
                               i - pos["idx"])
                    trades.append(t); day_pnl += t["net"]
                    pos = _empty_pos()

        # Partial close: lock half when half-lot net >= partial_usd
        _partial = False
        if partial_usd > 0 and pos["dir"] != 0 and not pos["partial_done"]:
            chk  = h if pos["dir"] == 1 else l
            half = pos["lots_rem"] / 2
            pm   = (chk - pos["entry"]) * pos["dir"] / _ps
            if _net(pm * pip_val * half, half) >= partial_usd:
                t = _close(pos, chk, ts, half, "partial_close", i - pos["idx"])
                trades.append(t); day_pnl += t["net"]
                pos["lots_rem"]    -= half
                pos["partial_done"] = True
                _partial = True

        # close_profit: close remaining lots on any profitable intrabar move
        if close_profit and pos["dir"] != 0 and not _partial:
            chk = h if pos["dir"] == 1 else l
            pm  = (chk - pos["entry"]) * pos["dir"] / _ps
            if _net(pm * pip_val * pos["lots_rem"], pos["lots_rem"]) > 0:
                t = _close(pos, chk, ts, pos["lots_rem"], "profit_close",
                           i - pos["idx"])
                trades.append(t); day_pnl += t["net"]
                pos = _empty_pos()

        # Signal exit / entry
        sig = int(sigs[i]) if i < len(sigs) else 0
        if reverse_signals:
            sig = -sig
        if sig != 0 and sig != pos["dir"]:
            too_early = pos["dir"] != 0 and (i - pos["idx"]) < min_bars_1h
            exit_conf = pos["dir"] == 0 or _exit_confirmed(pos["dir"], i)
            if not too_early and exit_conf:
                if pos["dir"] != 0:
                    t = _close(pos, c, ts, pos["lots_rem"], "flip_1h",
                               i - pos["idx"])
                    trades.append(t); day_pnl += t["net"]
                if not day_pause and (daily_stop_usd <= 0 or day_pnl > -daily_stop_usd):
                    if max_dist_pips > 0 and abs(c - dema[i]) / _ps > max_dist_pips:
                        pos = _empty_pos()
                    else:
                        pos = {"dir": sig, "entry": c, "idx": i, "time": ts,
                               "tf": "1h", "lots_rem": lots_1h,
                               "partial_done": False,
                               "atr_entry": float(atr_v[i]) if atr_on else 0.0}
                else:
                    day_pause = True
                    pos       = _empty_pos()

    return trades


# ── Joint simulation (all pairs together, with FTMO hedge & cumulative cap) ──

def simulate_joint(
    pairs:            list[str],
    data_1h:          dict[str, pd.DataFrame],
    ema_1h:           int,
    window:           int,
    min_swing:        float,
    lots_1h:          float,
    close_profit:     bool  = True,
    daily_stop_usd:   float = 0.0,
    min_bars_1h:      int   = 2,
    partial_usd:      float = 0.0,
    sl_pips_1h:       float = 0.0,
    tp_pips_1h:       float = 0.0,
    max_dist_pips:    float = 0.0,
    max_ccy_exposure: int   = 2,
    early_1h:         bool  = True,
    reverse_signals:  bool  = False,
    atr_period:       int   = 14,
    sl_atr_mult:      float = 0.0,
    tp_atr_mult:      float = 0.0,
) -> list[dict]:
    """
    Run all pairs through a single merged 1h timeline with FTMO compliance:
      - Hedge filter: refuses any entry that would create opposing exposure
        on a currency already held by another open position.
      - Cumulative cap: refuses any entry that would push the net per-currency
        exposure above max_ccy_exposure (in number of positions).
    """
    atr_on = sl_atr_mult > 0 or tp_atr_mult > 0

    def _ep() -> dict:
        return {"dir": 0, "entry": 0.0, "idx": -1, "time": None,
                "tf": "1h", "lots_rem": 0.0, "partial_done": False,
                "atr_entry": 0.0}

    # Precompute per-pair data
    pp: dict[str, dict] = {}
    for pair in pairs:
        df = data_1h.get(pair)
        if df is None or df.empty:
            continue
        _ps = PIP_SIZE_MAP.get(pair, PIP_SIZE)
        pp[pair] = {
            "df":     df,
            "sigs":   _compute_signals(df, ema_1h, window, min_swing,
                                       early_entry=early_1h, pip_size=_ps),
            "dema":   _ema(_ema((df["High"].values + df["Low"].values) / 2.0,
                                ema_1h), ema_1h),
            "atr":    _atr(df, atr_period) if atr_on else None,
            "ts_map": {ts: i for i, ts in enumerate(df.index)},
            "pv":     PIP_VALUE[pair],
            "sp":     HALF_SPREAD_PIPS[pair] * PIP_VALUE[pair],
            "ps":     _ps,
            "pos":    _ep(),
            "day_pnl":   0.0,
            "day_pause": False,
        }

    def _trade(p: dict, exit_px: float, ts, lots: float,
               exit_type: str, hold_bars: int, pair: str) -> dict:
        d   = pp[pair]
        pm  = (exit_px - p["entry"]) * p["dir"] / d["ps"]
        net = pm * d["pv"] * lots - 2.0 * d["sp"] * lots - COMMISSION_PER_LOT * lots
        return {"time": p["time"], "close_time": ts, "pair": pair,
                "tf": "1h", "direction": p["dir"], "entry": p["entry"],
                "exit": exit_px, "exit_type": exit_type, "net": net,
                "won": net > 0, "hold_bars": hold_bars}

    def _net_ccy() -> dict[str, int]:
        net: dict[str, int] = {}
        for pair, d in pp.items():
            if d["pos"]["dir"] != 0:
                for ccy, side in position_currencies(
                        pair, d["pos"]["dir"]).items():
                    net[ccy] = net.get(ccy, 0) + side
        return net

    def _entry_blocked(pair: str, direction: int) -> Optional[str]:
        """Return reason if entry blocked by hedge or cumulative cap; else None."""
        new_exp = position_currencies(pair, direction)
        cur_net = _net_ccy()
        for ccy, side in new_exp.items():
            cur = cur_net.get(ccy, 0)
            if cur != 0 and (cur > 0) != (side > 0):
                return f"hedge_{ccy}"
            if abs(cur + side) > max_ccy_exposure:
                return f"max_exp_{ccy}"
        return None

    # Merged 1h timeline across all pairs
    all_ts = sorted(set(ts for d in pp.values() for ts in d["df"].index))
    all_trades: list[dict] = []
    cur_day = None

    for ts in all_ts:
        day = ts.date()
        if day != cur_day:
            cur_day = day
            for d in pp.values():
                d["day_pnl"]   = 0.0
                d["day_pause"] = False

        for pair, d in pp.items():
            if ts not in d["ts_map"]:
                continue

            i  = d["ts_map"][ts]
            df = d["df"]
            h  = float(df["High"].iloc[i])
            l  = float(df["Low"].iloc[i])
            c  = float(df["Close"].iloc[i])
            p  = d["pos"]

            # Hard SL
            if sl_pips_1h > 0 and p["dir"] != 0:
                sl_px = p["entry"] - p["dir"] * sl_pips_1h * d["ps"]
                if (p["dir"] == 1 and l <= sl_px) or (p["dir"] == -1 and h >= sl_px):
                    t = _trade(p, sl_px, ts, p["lots_rem"], "sl_1h",
                               i - p["idx"], pair)
                    all_trades.append(t); d["day_pnl"] += t["net"]
                    d["pos"] = _ep(); p = d["pos"]

            # Hard TP (SL above takes priority)
            if tp_pips_1h > 0 and p["dir"] != 0:
                tp_px = p["entry"] + p["dir"] * tp_pips_1h * d["ps"]
                if (p["dir"] == 1 and h >= tp_px) or (p["dir"] == -1 and l <= tp_px):
                    t = _trade(p, tp_px, ts, p["lots_rem"], "tp_1h",
                               i - p["idx"], pair)
                    all_trades.append(t); d["day_pnl"] += t["net"]
                    d["pos"] = _ep(); p = d["pos"]

            # ATR-based SL/TP — SL has priority over TP on same bar
            if atr_on and p["dir"] != 0 and p["atr_entry"] > 0:
                if sl_atr_mult > 0:
                    sl_px = p["entry"] - p["dir"] * sl_atr_mult * p["atr_entry"]
                    if (p["dir"] == 1 and l <= sl_px) or (p["dir"] == -1 and h >= sl_px):
                        t = _trade(p, sl_px, ts, p["lots_rem"], "sl_atr",
                                   i - p["idx"], pair)
                        all_trades.append(t); d["day_pnl"] += t["net"]
                        d["pos"] = _ep(); p = d["pos"]
                if p["dir"] != 0 and tp_atr_mult > 0:
                    tp_px = p["entry"] + p["dir"] * tp_atr_mult * p["atr_entry"]
                    if (p["dir"] == 1 and h >= tp_px) or (p["dir"] == -1 and l <= tp_px):
                        t = _trade(p, tp_px, ts, p["lots_rem"], "tp_atr",
                                   i - p["idx"], pair)
                        all_trades.append(t); d["day_pnl"] += t["net"]
                        d["pos"] = _ep(); p = d["pos"]

            """
            # Partial close
            _partial = False
            if partial_usd > 0 and p["dir"] != 0 and not p["partial_done"]:
                chk  = h if p["dir"] == 1 else l
                half = p["lots_rem"] / 2
                pm   = (chk - p["entry"]) * p["dir"] / d["ps"]
                if pm * d["pv"] * half - 2.0 * d["sp"] * half - COMMISSION_PER_LOT * half >= partial_usd:
                    t = _trade(p, chk, ts, half, "partial_close",
                               i - p["idx"], pair)
                    all_trades.append(t); d["day_pnl"] += t["net"]
                    d["pos"]["lots_rem"]    -= half
                    d["pos"]["partial_done"] = True
                    _partial = True
                    p = d["pos"]

            # close_profit
            if close_profit and p["dir"] != 0 and not _partial:
                chk = h if p["dir"] == 1 else l
                lr  = p["lots_rem"]
                pm  = (chk - p["entry"]) * p["dir"] / d["ps"]
                if pm * d["pv"] * lr - 2.0 * d["sp"] * lr - COMMISSION_PER_LOT * lr > 0:
                    t = _trade(p, chk, ts, lr, "profit_close",
                               i - p["idx"], pair)
                    all_trades.append(t); d["day_pnl"] += t["net"]
                    d["pos"] = _ep(); p = d["pos"]
            """

            # Signal exit / entry
            sig = int(d["sigs"][i]) if i < len(d["sigs"]) else 0
            if reverse_signals:
                sig = -sig
            if sig != 0 and sig != p["dir"]:
                too_early = p["dir"] != 0 and (i - p["idx"]) < min_bars_1h
                if not too_early:
                    if p["dir"] != 0:
                        t = _trade(p, c, ts, p["lots_rem"], "flip_1h",
                                   i - p["idx"], pair)
                        all_trades.append(t); d["day_pnl"] += t["net"]
                        d["pos"] = _ep()
                    if not d["day_pause"] and (daily_stop_usd <= 0 or d["day_pnl"] > -daily_stop_usd):
                        if max_dist_pips > 0 and abs(c - d["dema"][i]) / d["ps"] > max_dist_pips:
                            d["pos"] = _ep()
                        elif _entry_blocked(pair, sig) is None:
                            d["pos"] = {"dir": sig, "entry": c, "idx": i,
                                        "time": ts, "tf": "1h",
                                        "lots_rem": lots_1h,
                                        "partial_done": False,
                                        "atr_entry": float(d["atr"][i]) if atr_on else 0.0}
                        else:
                            d["pos"] = _ep()
                    else:
                        d["day_pause"] = True
                        d["pos"]       = _ep()

    return sorted(all_trades, key=lambda t: t["close_time"])


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(trades: list[dict], label: str, period: str) -> dict:
    base = {"label": label, "period": period, "n": 0,
            "pnl": 0.0, "win_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "ev": 0.0, "max_dd": 0.0, "sharpe": 0.0, "avg_hold": 0.0}
    if not trades:
        return base
    nets  = np.array([t["net"] for t in trades])
    won   = np.array([t["won"] for t in trades], dtype=bool)
    holds = np.array([t["hold_bars"] for t in trades])
    n     = len(trades)
    pnl   = nets.sum()
    aw    = nets[won].mean()  if won.any()    else 0.0
    al    = nets[~won].mean() if (~won).any() else 0.0
    if n > 1 and nets.std() > 0:
        days   = max(1, (trades[-1]["close_time"] - trades[0]["time"]).days)
        sharpe = nets.mean() / nets.std() * np.sqrt(n / (days / 365.25))
    else:
        sharpe = 0.0
    cum    = np.cumsum(nets)
    max_dd = (cum - np.maximum.accumulate(cum)).min()
    return {
        "label": label, "period": period, "n": n,
        "pnl": round(pnl, 1), "win_pct": round(won.mean() * 100, 1),
        "avg_win": round(aw, 2), "avg_loss": round(al, 2),
        "sharpe": round(sharpe, 2), "max_dd": round(max_dd, 1),
        "ev": round(pnl / n, 2), "avg_hold": round(holds.mean(), 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--start",        default="2026-01-01")
    ap.add_argument("--end",          default="2026-05-07")
    ap.add_argument("--oos-start",    default="2026-04-01")
    ap.add_argument("--ema-1h",       type=int,   default=9)
    ap.add_argument("--window",       type=int,   default=20)
    ap.add_argument("--min-swing",    type=float, default=0.0)
    ap.add_argument("--lots-1h",      type=float, default=2.0)
    ap.add_argument("--close-profit", action="store_true", default=True)
    ap.add_argument("--no-close-profit", dest="close_profit", action="store_false")
    ap.add_argument("--daily-stop",   type=float, default=0.0,
                    help="Daily loss limit per pair in USD (0=disabled)")
    ap.add_argument("--min-bars-1h",  type=int,   default=2,
                    help="Min 1h bars to hold before signal exit")
    ap.add_argument("--partial-usd",  type=float, default=0.0,
                    help="Close half position when half-lot net >= USD (0=off)")
    ap.add_argument("--n-exit",       type=int,   default=0,
                    help="New-extreme confirmation lookback bars (0=off; per-pair sim only)")
    ap.add_argument("--sl-pips-1h",   type=float, default=40.0,
                    help="Hard SL for 1h positions in pips (0=disabled)")
    ap.add_argument("--tp-pips-1h",   type=float, default=0.0,
                    help="Hard TP for 1h positions in pips (0=disabled; SL takes priority on same bar)")
    ap.add_argument("--max-dist-pips", type=float, default=0.0,
                    help="Skip entry if close > N pips from DEMA (0=disabled)")
    ap.add_argument("--max-ccy-exposure", type=int, default=2,
                    help="FTMO cumulative-correlation cap: max simultaneous positions "
                         "sharing the same currency direction (joint sim only)")
    ap.add_argument("--per-pair",     action="store_true", default=False,
                    help="Run isolated per-pair simulation (no FTMO filter — diagnostic)")
    ap.add_argument("--early-1h",     action="store_true", default=True,
                    help="Fire 1h signal on the peak/valley bar itself (1 bar earlier; default ON)")
    ap.add_argument("--no-early-1h",  action="store_false", dest="early_1h",
                    help="Disable early-1h (revert to bar-after-peak confirmation)")
    ap.add_argument("--reverse-signals", action="store_true", default=False,
                    help="Flip every signal: LONG↔SHORT. Tests trend-follow instead of fade.")
    ap.add_argument("--atr-period",   type=int,   default=14,
                    help="ATR lookback period for ATR-based SL/TP")
    ap.add_argument("--sl-atr-mult",  type=float, default=0.0,
                    help="SL distance = N × ATR_at_entry (0 = use fixed --sl-pips-1h)")
    ap.add_argument("--tp-atr-mult",  type=float, default=0.0,
                    help="TP distance = N × ATR_at_entry (0 = no TP, use signal flip / close_profit)")
    ap.add_argument("--pairs",        nargs="+",  default=None)
    ap.add_argument("--login",        type=int,   default=None)
    ap.add_argument("--password",     default=None)
    ap.add_argument("--server",       default=None)
    ap.add_argument("--mt5-path",     default=None)
    args = ap.parse_args()

    pairs     = args.pairs or PAIRS
    oos_start = pd.Timestamp(args.oos_start, tz="UTC")

    # When any TP is active, the 1-2 pip close_profit ratchet would override
    # it on almost every bar — auto-disable so the TP test is clean.
    if (args.tp_atr_mult > 0 or args.tp_pips_1h > 0) and args.close_profit:
        log.info("TP active → auto-disabling close_profit "
                 "(use --close-profit explicitly to keep both)")
        args.close_profit = False

    log.info("Loading 1h data…")
    data_1h = _load_data_mt5(pairs, args.start, args.end, timeframe="1h",
                              login=args.login, password=args.password,
                              server=args.server, mt5_path=args.mt5_path)

    all_is, all_oos = [], []

    if args.per_pair:
        log.info("Per-pair simulation — no cross-pair FTMO filter (diagnostic mode)")
        for pair in pairs:
            df = data_1h.get(pair)
            if df is None or df.empty:
                log.warning("  %s: missing data", pair); continue
            log.info("  %s: %d 1h bars", pair, len(df))
            for t in simulate_pair(
                pair, df,
                ema_1h=args.ema_1h, window=args.window, min_swing=args.min_swing,
                lots_1h=args.lots_1h, close_profit=args.close_profit,
                daily_stop_usd=args.daily_stop,
                min_bars_1h=args.min_bars_1h,
                partial_usd=args.partial_usd,
                n_exit=args.n_exit,
                sl_pips_1h=args.sl_pips_1h,
                tp_pips_1h=args.tp_pips_1h,
                max_dist_pips=args.max_dist_pips,
                early_1h=args.early_1h,
                reverse_signals=args.reverse_signals,
                atr_period=args.atr_period,
                sl_atr_mult=args.sl_atr_mult,
                tp_atr_mult=args.tp_atr_mult,
            ):
                (all_is if t["time"] < oos_start else all_oos).append(t)
    else:
        log.info("Joint simulation across %d pairs — FTMO hedge & cumulative cap on",
                 len(pairs))
        all_trades = simulate_joint(
            pairs, data_1h,
            ema_1h=args.ema_1h, window=args.window, min_swing=args.min_swing,
            lots_1h=args.lots_1h, close_profit=args.close_profit,
            daily_stop_usd=args.daily_stop,
            min_bars_1h=args.min_bars_1h,
            partial_usd=args.partial_usd,
            sl_pips_1h=args.sl_pips_1h,
            tp_pips_1h=args.tp_pips_1h,
            max_dist_pips=args.max_dist_pips,
            max_ccy_exposure=args.max_ccy_exposure,
            early_1h=args.early_1h,
            reverse_signals=args.reverse_signals,
            atr_period=args.atr_period,
            sl_atr_mult=args.sl_atr_mult,
            tp_atr_mult=args.tp_atr_mult,
        )
        for t in all_trades:
            (all_is if t["time"] < oos_start else all_oos).append(t)

    W = 115
    for period, trades in [("IS", all_is), ("OOS", all_oos)]:
        m = metrics(trades, f"ema{args.ema_1h}_w{args.window}", period)
        print(f"\n{'='*W}")
        oos_label = "IN-SAMPLE" if period == "IS" else "OUT-OF-SAMPLE"
        mode = "per-pair" if args.per_pair else f"joint (max_ccy_exp={args.max_ccy_exposure})"
        tags = ""
        if args.early_1h: tags += "  early_1h=on"
        if args.reverse_signals: tags += "  REVERSED"
        if args.sl_atr_mult > 0 or args.tp_atr_mult > 0:
            tags += f"  atr({args.atr_period}):sl={args.sl_atr_mult}x/tp={args.tp_atr_mult}x"
        print(f"  {oos_label}  ema_1h={args.ema_1h}  w={args.window}  "
              f"sw={args.min_swing}  lots_1h={args.lots_1h}  "
              f"min_bars={args.min_bars_1h}  partial=${args.partial_usd:.0f}  "
              f"sl1h={args.sl_pips_1h:.0f}p  tp1h={args.tp_pips_1h:.0f}p  "
              f"dist={args.max_dist_pips:.0f}p  "
              f"cp={args.close_profit}  mode={mode}{tags}")
        print(f"{'='*W}")
        print(f"  Trades: {m['n']}")
        print(f"  P&L:    ${m['pnl']:,.1f}")
        print(f"  Win%:   {m['win_pct']:.1f}%")
        print(f"  Avg win: ${m['avg_win']:.2f}  Avg loss: ${m['avg_loss']:.2f}")
        print(f"  EV/trade: ${m['ev']:.2f}  Max DD: ${m['max_dd']:,.1f}")
        print(f"  Sharpe: {m['sharpe']:.2f}  Avg hold: {m['avg_hold']:.1f} bars")

        # Daily P&L distribution
        if trades:
            daily: dict = {}
            for t in trades:
                d = t["close_time"].date()
                daily[d] = daily.get(d, 0.0) + t["net"]
            pnls = list(daily.values())
            pos_days = sum(1 for p in pnls if p > 0)
            neg_days = sum(1 for p in pnls if p < 0)
            print(f"\n  Daily P&L:  avg=${np.mean(pnls):+,.0f}  "
                  f"best=${max(pnls):+,.0f}  worst=${min(pnls):+,.0f}  "
                  f"green={pos_days}d  red={neg_days}d")
            print(f"  Days exceeding -$1000: {sum(1 for p in pnls if p < -1000)}")
            print(f"  Days exceeding -$2000: {sum(1 for p in pnls if p < -2000)}")
            print(f"  Days exceeding -$5000: {sum(1 for p in pnls if p < -5000)}")

            # ── Breakdowns: by pair, by exit type, by entry hour ──
            def _group(key_fn):
                g: dict = {}
                for t in trades:
                    k = key_fn(t)
                    b = g.setdefault(k, {"n": 0, "pnl": 0.0, "wins": 0})
                    b["n"]   += 1
                    b["pnl"] += t["net"]
                    b["wins"] += 1 if t["won"] else 0
                return g

            def _fmt(label_w: int, rows: list[tuple]) -> None:
                # rows: list of (key, n, pnl, wins) sorted by pnl ascending (worst first)
                for k, n, pnl, wins in rows:
                    win_pct = (wins / n * 100) if n else 0.0
                    ev      = (pnl / n)        if n else 0.0
                    print(f"    {str(k):<{label_w}}  n={n:>4}  "
                          f"pnl=${pnl:>+9,.0f}  win%={win_pct:>4.1f}  ev=${ev:>+7.1f}")

            by_pair = _group(lambda t: t["pair"])
            print(f"\n  By pair:")
            _fmt(8, sorted([(k, v["n"], v["pnl"], v["wins"])
                            for k, v in by_pair.items()],
                           key=lambda r: r[2]))

            by_exit = _group(lambda t: t["exit_type"])
            print(f"\n  By exit type:")
            _fmt(14, sorted([(k, v["n"], v["pnl"], v["wins"])
                             for k, v in by_exit.items()],
                            key=lambda r: r[2]))

            by_hour = _group(lambda t: t["time"].hour)
            print(f"\n  By entry hour (broker time, UTC+3):")
            _fmt(2, sorted([(f"{k:02d}", v["n"], v["pnl"], v["wins"])
                            for k, v in by_hour.items()],
                           key=lambda r: int(r[0])))


if __name__ == "__main__":
    main()
