"""
Combined 1h + 5m double-EMA extrema strategy.

Position rules per pair:
  - 1h and 5m each fire independent signals on their own data
  - Same direction: both positions open simultaneously (max 2 lots per pair)
  - Opposite direction: newer 5m signal that contradicts 1h is BLOCKED
  - 1h direction flip: closes BOTH positions, then opens new 1h position

Exits:
  - close_profit: each position checks intrabar H/L; closes when net P&L > 0
  - no_sl: closes only on opposite signal (bar-close fill)

Daily circuit-breaker:
  - Tracks daily P&L; pauses pair if day P&L < -daily_stop

Usage
-----
  python -m scripts.strategy_combined_backtest \\
      --login 1513313327 --server FTMO-Demo \\
      --start 2026-01-01 --end 2026-05-07 --oos-start 2026-04-01 \\
      --ema-1h 20 --ema-5m 20 --window 20 \\
      --lots-1h 2 --lots-5m 1 --close-profit
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

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
log = logging.getLogger("combined")

# USD direction when BUYing a pair: -1 = sells USD; +1 = buys USD
USD_SIDE_MAP = {
    "EURUSD": -1, "AUDUSD": -1, "NZDUSD": -1, "GBPUSD": -1,
    "USDCHF": +1, "USDCAD": +1, "USDJPY": +1,
    "EURGBP":  0, "EURJPY":  0,  # cross pairs — no direct USD exposure
}


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
    # pos must be ≤ this to fire; default keeps current bar blocked (W-2), early allows it (W-1)
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


# ── Combined simulation ───────────────────────────────────────────────────────

def simulate_combined(
    pair:            str,
    df_1h:           pd.DataFrame,
    df_5m:           pd.DataFrame,
    ema_1h:          int,
    ema_5m:          int,
    window:          int,
    min_swing:       float,
    lots_1h:         float,
    lots_5m:         float,
    close_profit:    bool  = True,
    daily_stop_usd:  float = 0.0,
    min_bars_5m:     int   = 3,    # min 5m bars held before a signal can exit
    min_bars_1h:     int   = 2,    # min 1h bars held before a signal can exit
    partial_usd:     float = 0.0,  # close half when half-lot net P&L >= USD (0=off)
    n_exit:          int   = 0,    # new-extreme confirmation: N-bar lookback (0=off)
    sl_pips:         float = 0.0,  # hard SL for both timeframes (0=off)
    sl_pips_1h:      float = 0.0,  # hard SL for 1h positions only (overrides sl_pips for 1h)
    sl_pips_5m:      float = 0.0,  # hard SL for 5m positions only (overrides sl_pips for 5m)
    half_sl_pips_1h: float = 0.0,  # close half the 1h position when adverse move >= this (0=off)
    max_dist_pips:   float = 0.0,  # skip entry if close > N pips from DEMA (0=off)
    no_block:        bool  = False, # True = 5m runs independently of 1h (separate accounts mode)
    early_1h:        bool  = True,  # fire 1h signal on peak/valley bar itself (1 bar earlier)
    early_5m:        bool  = False, # fire 5m signal on peak/valley bar itself (1 bar earlier)
) -> list[dict]:

    pip_val_h = PIP_VALUE[pair]
    sp_h      = HALF_SPREAD_PIPS[pair] * pip_val_h
    comm_h    = COMMISSION_PER_LOT
    _ps       = PIP_SIZE_MAP.get(pair, PIP_SIZE)

    def _net(gross: float, lots: float) -> float:
        return gross - 2.0 * sp_h * lots - comm_h * lots

    sigs_1h   = _compute_signals(df_1h, ema_1h, window, min_swing, early_entry=early_1h, pip_size=_ps)
    sigs_5m   = _compute_signals(df_5m, ema_5m, window, min_swing, early_entry=early_5m, pip_size=_ps)
    ts_1h_map = {ts: i for i, ts in enumerate(df_1h.index)}

    # DEMA arrays for distance-from-DEMA entry filter
    dema_1h_arr = _ema(_ema((df_1h["High"].values + df_1h["Low"].values) / 2.0, ema_1h), ema_1h)
    dema_5m_arr = _ema(_ema((df_5m["High"].values + df_5m["Low"].values) / 2.0, ema_5m), ema_5m)

    def _empty_pos(tf: str = "") -> dict:
        return {"dir": 0, "entry": 0.0, "idx": -1, "time": None,
                "tf": tf, "lots_rem": 0.0, "partial_done": False}

    pos_1h = _empty_pos("1h")
    pos_5m = _empty_pos("5m")
    trades: list[dict] = []
    cur_day   = None
    day_pnl   = 0.0
    day_pause = False

    def _close(pos: dict, exit_price: float, exit_time, lots: float,
                exit_type: str, hold_bars: int) -> dict:
        pm  = (exit_price - pos["entry"]) * pos["dir"] / _ps
        net = _net(pm * pip_val_h * lots, lots)
        return {
            "time":       pos["time"],
            "close_time": exit_time,
            "pair":       pair,
            "tf":         pos["tf"],
            "direction":  pos["dir"],
            "entry":      pos["entry"],
            "exit":       exit_price,
            "exit_type":  exit_type,
            "net":        net,
            "won":        net > 0,
            "hold_bars":  hold_bars,
        }

    # Precompute rolling min/max for new-extreme exit filter
    _lo5 = df_5m["Low"].values
    _hi5 = df_5m["High"].values
    _lo1 = df_1h["Low"].values
    _hi1 = df_1h["High"].values
    if n_exit > 0:
        _prev_lo5 = pd.Series(_lo5).rolling(n_exit).min().shift(1).fillna(np.inf).values
        _prev_hi5 = pd.Series(_hi5).rolling(n_exit).max().shift(1).fillna(-np.inf).values
        _prev_lo1 = pd.Series(_lo1).rolling(n_exit).min().shift(1).fillna(np.inf).values
        _prev_hi1 = pd.Series(_hi1).rolling(n_exit).max().shift(1).fillna(-np.inf).values
    else:
        _prev_lo5 = _prev_hi5 = _prev_lo1 = _prev_hi1 = None

    def _exit_confirmed_5m(direction: int, i: int) -> bool:
        if _prev_lo5 is None:
            return True
        return _lo5[i] < _prev_lo5[i] if direction == 1 else _hi5[i] > _prev_hi5[i]

    def _exit_confirmed_1h(direction: int, i: int) -> bool:
        if _prev_lo1 is None:
            return True
        return _lo1[i] < _prev_lo1[i] if direction == 1 else _hi1[i] > _prev_hi1[i]

    for i5 in range(len(df_5m)):
        ts = df_5m.index[i5]
        h5 = df_5m["High"].iloc[i5]
        l5 = df_5m["Low"].iloc[i5]
        c5 = df_5m["Close"].iloc[i5]

        # ─── Hard SL: checked at 5m resolution for all open positions ──────────
        _sl1h = sl_pips_1h if sl_pips_1h > 0 else sl_pips
        _sl5m = sl_pips_5m if sl_pips_5m > 0 else sl_pips
        if _sl1h > 0 and pos_1h["dir"] != 0:
            sl_px  = pos_1h["entry"] - pos_1h["dir"] * _sl1h * _ps
            sl_hit = (pos_1h["dir"] == 1 and l5 <= sl_px) or \
                     (pos_1h["dir"] == -1 and h5 >= sl_px)
            if sl_hit:
                t = _close(pos_1h, sl_px, ts, pos_1h["lots_rem"], "sl_1h", 0)
                trades.append(t); day_pnl += t["net"]
                # clear same-direction 5m position — anchor is gone
                if pos_5m["dir"] == pos_1h["dir"]:
                    t2 = _close(pos_5m, sl_px, ts, pos_5m["lots_rem"], "sl_1h_clear", 0)
                    trades.append(t2); day_pnl += t2["net"]
                    pos_5m = _empty_pos("5m")
                pos_1h = _empty_pos("1h")
        if _sl5m > 0 and pos_5m["dir"] != 0:
            sl_px  = pos_5m["entry"] - pos_5m["dir"] * _sl5m * _ps
            sl_hit = (pos_5m["dir"] == 1 and l5 <= sl_px) or \
                     (pos_5m["dir"] == -1 and h5 >= sl_px)
            if sl_hit:
                t = _close(pos_5m, sl_px, ts, pos_5m["lots_rem"], "sl_5m", 0)
                trades.append(t); day_pnl += t["net"]
                pos_5m = _empty_pos("5m")

        day = ts.date()
        if day != cur_day:
            cur_day   = day
            day_pnl   = 0.0
            day_pause = False

        # ── 1h bar close ──────────────────────────────────────────────────
        if ts in ts_1h_map:
            i1 = ts_1h_map[ts]
            h1 = df_1h["High"].iloc[i1]
            l1 = df_1h["Low"].iloc[i1]
            c1 = df_1h["Close"].iloc[i1]

            # Partial close: lock half when half-lot net >= partial_usd
            _partial_1h = False
            if partial_usd > 0 and pos_1h["dir"] != 0 and not pos_1h["partial_done"]:
                chk  = h1 if pos_1h["dir"] == 1 else l1
                pm   = (chk - pos_1h["entry"]) * pos_1h["dir"] / _ps
                half = pos_1h["lots_rem"] / 2
                if _net(pm * pip_val_h * half, half) >= partial_usd:
                    t = _close(pos_1h, chk, ts, half, "partial_close",
                               i1 - pos_1h["idx"])
                    trades.append(t); day_pnl += t["net"]
                    pos_1h["lots_rem"]    -= half
                    pos_1h["partial_done"] = True
                    _partial_1h = True  # skip close_profit this bar; let remaining half run

            # close_profit: close remaining lots on any profitable intrabar move
            # Skip on the same bar partial fired — remaining half should run further
            if close_profit and pos_1h["dir"] != 0 and not _partial_1h:
                chk = h1 if pos_1h["dir"] == 1 else l1
                pm  = (chk - pos_1h["entry"]) * pos_1h["dir"] / _ps
                if _net(pm * pip_val_h * pos_1h["lots_rem"], pos_1h["lots_rem"]) > 0:
                    t = _close(pos_1h, chk, ts, pos_1h["lots_rem"], "profit_close",
                               i1 - pos_1h["idx"])
                    trades.append(t); day_pnl += t["net"]
                    pos_1h = _empty_pos("1h")

            # 1h signal exit — requires min_bars hold AND new-extreme confirmation
            sig1 = sigs_1h[i1] if i1 < len(sigs_1h) else 0
            if sig1 != 0 and sig1 != pos_1h["dir"]:
                too_early = pos_1h["dir"] != 0 and (i1 - pos_1h["idx"]) < min_bars_1h
                # new-extreme filter only applied when closing an existing position
                exit_conf = pos_1h["dir"] == 0 or _exit_confirmed_1h(pos_1h["dir"], i1)
                if not too_early and exit_conf:
                    if pos_1h["dir"] != 0:
                        t = _close(pos_1h, c1, ts, pos_1h["lots_rem"], "flip_1h",
                                   i1 - pos_1h["idx"])
                        trades.append(t); day_pnl += t["net"]
                    # 1h flip clears same-direction 5m (unless running as separate accounts)
                    if not no_block and pos_5m["dir"] != 0 and pos_5m["dir"] == -sig1:
                        t = _close(pos_5m, c1, ts, pos_5m["lots_rem"],
                                   "flip_1h_clear", 0)
                        trades.append(t); day_pnl += t["net"]
                        pos_5m = _empty_pos("5m")
                    if daily_stop_usd <= 0 or day_pnl > -daily_stop_usd:
                        # distance filter: skip entry if close too far from DEMA
                        if max_dist_pips > 0 and \
                                abs(c1 - dema_1h_arr[i1]) / _ps > max_dist_pips:
                            pos_1h = _empty_pos("1h")
                        else:
                            pos_1h = {"dir": sig1, "entry": c1, "idx": i1, "time": ts,
                                      "tf": "1h", "lots_rem": lots_1h, "partial_done": False}
                    else:
                        day_pause = True
                        pos_1h    = _empty_pos("1h")

        # ── 5m processing ─────────────────────────────────────────────────
        if i5 >= len(sigs_5m):
            continue

        # Partial close for 5m
        _partial_5m = False
        if partial_usd > 0 and pos_5m["dir"] != 0 and not pos_5m["partial_done"]:
            chk  = h5 if pos_5m["dir"] == 1 else l5
            pm   = (chk - pos_5m["entry"]) * pos_5m["dir"] / _ps
            half = pos_5m["lots_rem"] / 2
            if _net(pm * pip_val_h * half, half) >= partial_usd:
                t = _close(pos_5m, chk, ts, half, "partial_close",
                           i5 - pos_5m["idx"])
                trades.append(t); day_pnl += t["net"]
                pos_5m["lots_rem"]    -= half
                pos_5m["partial_done"] = True
                _partial_5m = True  # skip close_profit this bar; let remaining half run

        # close_profit for 5m
        # Skip on the same bar partial fired — remaining half should run further
        if close_profit and pos_5m["dir"] != 0 and not _partial_5m:
            chk = h5 if pos_5m["dir"] == 1 else l5
            pm  = (chk - pos_5m["entry"]) * pos_5m["dir"] / _ps
            if _net(pm * pip_val_h * pos_5m["lots_rem"], pos_5m["lots_rem"]) > 0:
                t = _close(pos_5m, chk, ts, pos_5m["lots_rem"], "profit_close",
                           i5 - pos_5m["idx"])
                trades.append(t); day_pnl += t["net"]
                pos_5m = _empty_pos("5m")

        # 5m signal — ignored if position has not been held min_bars_5m yet
        sig5 = sigs_5m[i5]
        if sig5 == 0:
            continue

        if pos_5m["dir"] != 0 and sig5 != pos_5m["dir"]:
            if (i5 - pos_5m["idx"]) < min_bars_5m:
                pass  # too early — skip exit signal, keep holding
            elif not _exit_confirmed_5m(pos_5m["dir"], i5):
                pass  # no new-extreme confirmation — keep holding
            elif not no_block and pos_1h["dir"] != 0 and pos_1h["dir"] == pos_5m["dir"]:
                # new 5m signal would oppose 1h → close 5m, don't reopen opposite
                t = _close(pos_5m, c5, ts, pos_5m["lots_rem"], "flip_5m_blocked",
                           i5 - pos_5m["idx"])
                trades.append(t); day_pnl += t["net"]
                pos_5m = _empty_pos("5m")
            else:
                t = _close(pos_5m, c5, ts, pos_5m["lots_rem"], "flip_5m",
                           i5 - pos_5m["idx"])
                trades.append(t); day_pnl += t["net"]
                if not day_pause:
                    allowed_reopen = no_block or (pos_1h["dir"] == 0 or pos_1h["dir"] == sig5)
                    if allowed_reopen:
                        if daily_stop_usd <= 0 or day_pnl > -daily_stop_usd:
                            if max_dist_pips <= 0 or \
                                    abs(c5 - dema_5m_arr[i5]) / _ps <= max_dist_pips:
                                pos_5m = {"dir": sig5, "entry": c5, "idx": i5, "time": ts,
                                          "tf": "5m", "lots_rem": lots_5m, "partial_done": False}
                        else:
                            day_pause = True

        elif pos_5m["dir"] == 0 and sig5 != 0:
            allowed = no_block or (pos_1h["dir"] == 0 or pos_1h["dir"] == sig5)
            if allowed and not day_pause:
                if daily_stop_usd <= 0 or day_pnl > -daily_stop_usd:
                    if max_dist_pips <= 0 or \
                            abs(c5 - dema_5m_arr[i5]) / _ps <= max_dist_pips:
                        pos_5m = {"dir": sig5, "entry": c5, "idx": i5, "time": ts,
                                  "tf": "5m", "lots_rem": lots_5m, "partial_done": False}

    return trades


# ── Joint simulation (all pairs together, with USD exposure filter) ───────────

def simulate_joint(
    pairs:          list[str],
    data_1h:        dict[str, pd.DataFrame],
    data_5m:        dict[str, pd.DataFrame],
    ema_1h:         int,
    ema_5m:         int,
    window:         int,
    min_swing:      float,
    lots_1h:        float,
    lots_5m:        float,
    close_profit:   bool  = True,
    daily_stop_usd: float = 0.0,
    min_bars_5m:    int   = 3,
    min_bars_1h:    int   = 2,
    partial_usd:    float = 0.0,
    sl_pips:        float = 0.0,
    sl_pips_1h:     float = 0.0,
    sl_pips_5m:     float = 0.0,
    max_dist_pips:  float = 0.0,
) -> list[dict]:
    """
    Run all pairs through a single merged 5m timeline.
    Before any new entry, checks net USD exposure to prevent holding
    simultaneously long and short USD positions across different pairs.
    """
    _sl1h = sl_pips_1h if sl_pips_1h > 0 else sl_pips
    _sl5m = sl_pips_5m if sl_pips_5m > 0 else sl_pips

    def _ep(tf: str = "") -> dict:
        return {"dir": 0, "entry": 0.0, "idx": -1, "time": None,
                "tf": tf, "lots_rem": 0.0, "partial_done": False}

    # Precompute per-pair data
    pp: dict[str, dict] = {}
    for pair in pairs:
        df1 = data_1h.get(pair)
        df5 = data_5m.get(pair)
        if df1 is None or df5 is None or df1.empty or df5.empty:
            continue
        _ps = PIP_SIZE_MAP.get(pair, PIP_SIZE)
        pp[pair] = {
            "df1": df1, "df5": df5,
            "sigs_1h":  _compute_signals(df1, ema_1h, window, min_swing, pip_size=_ps),
            "sigs_5m":  _compute_signals(df5, ema_5m, window, min_swing, pip_size=_ps),
            "dema_1h":  _ema(_ema((df1["High"].values + df1["Low"].values) / 2.0, ema_1h), ema_1h),
            "dema_5m":  _ema(_ema((df5["High"].values + df5["Low"].values) / 2.0, ema_5m), ema_5m),
            "ts_1h_map": {ts: i for i, ts in enumerate(df1.index)},
            "ts_5m_map": {ts: i for i, ts in enumerate(df5.index)},
            "pv":  PIP_VALUE[pair],
            "sp":  HALF_SPREAD_PIPS[pair] * PIP_VALUE[pair],
            "ps":  _ps,
            "pos_1h":    _ep("1h"),
            "pos_5m":    _ep("5m"),
            "day_pnl":   0.0,
            "day_pause": False,
        }

    def _trade(pos: dict, exit_px: float, ts, lots: float,
               exit_type: str, hold_bars: int, pair: str) -> dict:
        d   = pp[pair]
        pm  = (exit_px - pos["entry"]) * pos["dir"] / d["ps"]
        net = pm * d["pv"] * lots - 2.0 * d["sp"] * lots - COMMISSION_PER_LOT * lots
        return {"time": pos["time"], "close_time": ts, "pair": pair,
                "tf": pos["tf"], "direction": pos["dir"],
                "entry": pos["entry"], "exit": exit_px,
                "exit_type": exit_type, "net": net, "won": net > 0,
                "hold_bars": hold_bars}

    def _net_usd() -> int:
        net = 0
        for p, d in pp.items():
            usd = USD_SIDE_MAP.get(p, 0)
            net += d["pos_1h"]["dir"] * usd + d["pos_5m"]["dir"] * usd
        return net

    def _usd_ok(pair: str, direction: int) -> bool:
        net = _net_usd()
        if net == 0:
            return True
        new_usd = direction * USD_SIDE_MAP.get(pair, 0)
        if new_usd == 0:
            return True
        return (net > 0) == (new_usd > 0)

    # Merged 5m timeline across all pairs
    all_ts = sorted(set(ts for d in pp.values() for ts in d["df5"].index))
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
            if ts not in d["ts_5m_map"]:
                continue

            i5  = d["ts_5m_map"][ts]
            df5 = d["df5"]
            h5  = float(df5["High"].iloc[i5])
            l5  = float(df5["Low"].iloc[i5])
            c5  = float(df5["Close"].iloc[i5])
            p1  = d["pos_1h"]
            p5  = d["pos_5m"]

            # ── Hard SL checks ────────────────────────────────────────────────
            if _sl1h > 0 and p1["dir"] != 0:
                sl_px = p1["entry"] - p1["dir"] * _sl1h * d["ps"]
                if (p1["dir"] == 1 and l5 <= sl_px) or (p1["dir"] == -1 and h5 >= sl_px):
                    t = _trade(p1, sl_px, ts, p1["lots_rem"], "sl_1h", 0, pair)
                    all_trades.append(t); d["day_pnl"] += t["net"]
                    if p5["dir"] == p1["dir"]:
                        t2 = _trade(p5, sl_px, ts, p5["lots_rem"], "sl_1h_clear", 0, pair)
                        all_trades.append(t2); d["day_pnl"] += t2["net"]
                        d["pos_5m"] = _ep("5m")
                    d["pos_1h"] = _ep("1h")
                    p1 = d["pos_1h"]; p5 = d["pos_5m"]

            if _sl5m > 0 and p5["dir"] != 0:
                sl_px = p5["entry"] - p5["dir"] * _sl5m * d["ps"]
                if (p5["dir"] == 1 and l5 <= sl_px) or (p5["dir"] == -1 and h5 >= sl_px):
                    t = _trade(p5, sl_px, ts, p5["lots_rem"], "sl_5m", 0, pair)
                    all_trades.append(t); d["day_pnl"] += t["net"]
                    d["pos_5m"] = _ep("5m"); p5 = d["pos_5m"]

            # ── 1h bar close ──────────────────────────────────────────────────
            if ts in d["ts_1h_map"]:
                i1  = d["ts_1h_map"][ts]
                df1 = d["df1"]
                h1  = float(df1["High"].iloc[i1])
                l1  = float(df1["Low"].iloc[i1])
                c1  = float(df1["Close"].iloc[i1])
                p1  = d["pos_1h"]

                _partial_1h = False
                if partial_usd > 0 and p1["dir"] != 0 and not p1["partial_done"]:
                    chk  = h1 if p1["dir"] == 1 else l1
                    half = p1["lots_rem"] / 2
                    pm   = (chk - p1["entry"]) * p1["dir"] / d["ps"]
                    if pm * d["pv"] * half - 2.0 * d["sp"] * half - COMMISSION_PER_LOT * half >= partial_usd:
                        t = _trade(p1, chk, ts, half, "partial_close", i1 - p1["idx"], pair)
                        all_trades.append(t); d["day_pnl"] += t["net"]
                        d["pos_1h"]["lots_rem"] -= half
                        d["pos_1h"]["partial_done"] = True
                        _partial_1h = True
                        p1 = d["pos_1h"]

                if close_profit and p1["dir"] != 0 and not _partial_1h:
                    chk = h1 if p1["dir"] == 1 else l1
                    lr  = p1["lots_rem"]
                    pm  = (chk - p1["entry"]) * p1["dir"] / d["ps"]
                    if pm * d["pv"] * lr - 2.0 * d["sp"] * lr - COMMISSION_PER_LOT * lr > 0:
                        t = _trade(p1, chk, ts, lr, "profit_close", i1 - p1["idx"], pair)
                        all_trades.append(t); d["day_pnl"] += t["net"]
                        d["pos_1h"] = _ep("1h"); p1 = d["pos_1h"]

                sigs_1h = d["sigs_1h"]
                sig1    = int(sigs_1h[i1]) if i1 < len(sigs_1h) else 0
                if sig1 != 0 and sig1 != p1["dir"]:
                    too_early = p1["dir"] != 0 and (i1 - p1["idx"]) < min_bars_1h
                    if not too_early:
                        if p1["dir"] != 0:
                            t = _trade(p1, c1, ts, p1["lots_rem"], "flip_1h",
                                       i1 - p1["idx"], pair)
                            all_trades.append(t); d["day_pnl"] += t["net"]
                        p5 = d["pos_5m"]
                        if p5["dir"] != 0 and p5["dir"] == -sig1:
                            t = _trade(p5, c1, ts, p5["lots_rem"], "flip_1h_clear", 0, pair)
                            all_trades.append(t); d["day_pnl"] += t["net"]
                            d["pos_5m"] = _ep("5m")
                        if not d["day_pause"] and (daily_stop_usd <= 0 or d["day_pnl"] > -daily_stop_usd):
                            if max_dist_pips > 0 and abs(c1 - d["dema_1h"][i1]) / d["ps"] > max_dist_pips:
                                d["pos_1h"] = _ep("1h")
                            elif _usd_ok(pair, sig1):
                                d["pos_1h"] = {"dir": sig1, "entry": c1, "idx": i1,
                                               "time": ts, "tf": "1h",
                                               "lots_rem": lots_1h, "partial_done": False}
                            else:
                                d["pos_1h"] = _ep("1h")
                        else:
                            d["day_pause"] = True
                            d["pos_1h"]    = _ep("1h")

            # ── 5m processing ─────────────────────────────────────────────────
            sigs_5m = d["sigs_5m"]
            sig5    = int(sigs_5m[i5]) if i5 < len(sigs_5m) else 0
            if sig5 == 0:
                continue

            p5 = d["pos_5m"]
            p1 = d["pos_1h"]

            _partial_5m = False
            if partial_usd > 0 and p5["dir"] != 0 and not p5["partial_done"]:
                chk  = h5 if p5["dir"] == 1 else l5
                half = p5["lots_rem"] / 2
                pm   = (chk - p5["entry"]) * p5["dir"] / d["ps"]
                if pm * d["pv"] * half - 2.0 * d["sp"] * half - COMMISSION_PER_LOT * half >= partial_usd:
                    t = _trade(p5, chk, ts, half, "partial_close", i5 - p5["idx"], pair)
                    all_trades.append(t); d["day_pnl"] += t["net"]
                    d["pos_5m"]["lots_rem"]    -= half
                    d["pos_5m"]["partial_done"] = True
                    _partial_5m = True
                    p5 = d["pos_5m"]

            if close_profit and p5["dir"] != 0 and not _partial_5m:
                chk = h5 if p5["dir"] == 1 else l5
                lr  = p5["lots_rem"]
                pm  = (chk - p5["entry"]) * p5["dir"] / d["ps"]
                if pm * d["pv"] * lr - 2.0 * d["sp"] * lr - COMMISSION_PER_LOT * lr > 0:
                    t = _trade(p5, chk, ts, lr, "profit_close", i5 - p5["idx"], pair)
                    all_trades.append(t); d["day_pnl"] += t["net"]
                    d["pos_5m"] = _ep("5m"); p5 = d["pos_5m"]

            if p5["dir"] != 0 and sig5 != p5["dir"]:
                if (i5 - p5["idx"]) < min_bars_5m:
                    pass  # whipsaw guard
                elif p1["dir"] != 0 and p1["dir"] == p5["dir"]:
                    t = _trade(p5, c5, ts, p5["lots_rem"], "flip_5m_blocked",
                               i5 - p5["idx"], pair)
                    all_trades.append(t); d["day_pnl"] += t["net"]
                    d["pos_5m"] = _ep("5m")
                else:
                    t = _trade(p5, c5, ts, p5["lots_rem"], "flip_5m",
                               i5 - p5["idx"], pair)
                    all_trades.append(t); d["day_pnl"] += t["net"]
                    d["pos_5m"] = _ep("5m")
                    if (p1["dir"] == 0 or p1["dir"] == sig5) and not d["day_pause"]:
                        if daily_stop_usd <= 0 or d["day_pnl"] > -daily_stop_usd:
                            if max_dist_pips <= 0 or abs(c5 - d["dema_5m"][i5]) / d["ps"] <= max_dist_pips:
                                if _usd_ok(pair, sig5):
                                    d["pos_5m"] = {"dir": sig5, "entry": c5, "idx": i5,
                                                   "time": ts, "tf": "5m",
                                                   "lots_rem": lots_5m, "partial_done": False}
                        else:
                            d["day_pause"] = True

            elif p5["dir"] == 0 and sig5 != 0:
                if (p1["dir"] == 0 or p1["dir"] == sig5) and not d["day_pause"]:
                    if daily_stop_usd <= 0 or d["day_pnl"] > -daily_stop_usd:
                        if max_dist_pips <= 0 or abs(c5 - d["dema_5m"][i5]) / d["ps"] <= max_dist_pips:
                            if _usd_ok(pair, sig5):
                                d["pos_5m"] = {"dir": sig5, "entry": c5, "idx": i5,
                                               "time": ts, "tf": "5m",
                                               "lots_rem": lots_5m, "partial_done": False}

    return sorted(all_trades, key=lambda t: t["close_time"])


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(trades: list[dict], label: str, period: str) -> dict:
    base = {"label": label, "period": period, "n": 0, "n_1h": 0, "n_5m": 0,
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
    n_1h   = sum(1 for t in trades if t["tf"] == "1h")
    n_5m   = sum(1 for t in trades if t["tf"] == "5m")
    return {
        "label": label, "period": period,
        "n": n, "n_1h": n_1h, "n_5m": n_5m,
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
    ap.add_argument("--ema-5m",       type=int,   default=20)
    ap.add_argument("--window",       type=int,   default=20)
    ap.add_argument("--min-swing",    type=float, default=0.0)
    ap.add_argument("--lots-1h",      type=float, default=2.0)
    ap.add_argument("--lots-5m",      type=float, default=1.0)
    ap.add_argument("--close-profit", action="store_true", default=True)
    ap.add_argument("--no-close-profit", dest="close_profit", action="store_false")
    ap.add_argument("--daily-stop",   type=float, default=0.0,
                    help="Daily loss limit per pair in USD (0=disabled)")
    ap.add_argument("--min-bars-5m",  type=int,   default=3,
                    help="Min 5m bars to hold before signal exit")
    ap.add_argument("--min-bars-1h",  type=int,   default=2,
                    help="Min 1h bars to hold before signal exit")
    ap.add_argument("--partial-usd",  type=float, default=0.0,
                    help="Close half position when half-lot net >= USD (0=off)")
    ap.add_argument("--n-exit",       type=int,   default=0,
                    help="New-extreme confirmation lookback bars (0=off)")
    ap.add_argument("--sl-pips",      type=float, default=0.0,
                    help="Hard stop-loss in pips for all positions (0=disabled)")
    ap.add_argument("--sl-pips-1h",   type=float, default=40.0,
                    help="Hard SL for 1h positions only (overrides --sl-pips for 1h)")
    ap.add_argument("--sl-pips-5m",   type=float, default=0.0,
                    help="Hard SL for 5m positions only (overrides --sl-pips for 5m)")
    ap.add_argument("--max-dist-pips",type=float, default=0.0,
                    help="Skip entry if close > N pips from DEMA (0=disabled)")
    ap.add_argument("--usd-filter",   action="store_true", default=False,
                    help="Run joint simulation with USD exposure filter (prevents opposing-USD entries)")
    ap.add_argument("--no-block",     action="store_true", default=False,
                    help="5m runs independently of 1h — simulates two separate accounts")
    ap.add_argument("--early-1h",     action="store_true", default=True,
                    help="Fire 1h signal on the peak/valley bar itself (1 bar earlier; default ON)")
    ap.add_argument("--no-early-1h",  action="store_false", dest="early_1h",
                    help="Disable early-1h (revert to bar-after-peak confirmation)")
    ap.add_argument("--early-5m",     action="store_true", default=False,
                    help="Fire 5m signal on the peak/valley bar itself (1 bar earlier than default)")
    ap.add_argument("--pairs",        nargs="+",  default=None)
    ap.add_argument("--login",        type=int,   default=None)
    ap.add_argument("--password",     default=None)
    ap.add_argument("--server",       default=None)
    ap.add_argument("--mt5-path",     default=None)
    args = ap.parse_args()

    pairs     = args.pairs or PAIRS
    oos_start = pd.Timestamp(args.oos_start, tz="UTC")

    log.info("Loading 1h data…")
    data_1h = _load_data_mt5(pairs, args.start, args.end, timeframe="1h",
                              login=args.login, password=args.password,
                              server=args.server, mt5_path=args.mt5_path)
    log.info("Loading 5m data…")
    data_5m = _load_data_mt5(pairs, args.start, args.end, timeframe="5m",
                              login=args.login, password=args.password,
                              server=args.server, mt5_path=args.mt5_path)

    all_is, all_oos = [], []
    common_kw = dict(
        ema_1h=args.ema_1h, ema_5m=args.ema_5m,
        window=args.window, min_swing=args.min_swing,
        lots_1h=args.lots_1h, lots_5m=args.lots_5m,
        close_profit=args.close_profit,
        daily_stop_usd=args.daily_stop,
        min_bars_5m=args.min_bars_5m,
        min_bars_1h=args.min_bars_1h,
        partial_usd=args.partial_usd,
        sl_pips=args.sl_pips,
        sl_pips_1h=args.sl_pips_1h,
        sl_pips_5m=args.sl_pips_5m,
        max_dist_pips=args.max_dist_pips,
    )
    if args.usd_filter:
        log.info("Running joint simulation with USD filter across %d pairs", len(pairs))
        all_trades = simulate_joint(pairs, data_1h, data_5m, **common_kw)
        for t in all_trades:
            (all_is if t["time"] < oos_start else all_oos).append(t)
    else:
        for pair in pairs:
            df1 = data_1h.get(pair)
            df5 = data_5m.get(pair)
            if df1 is None or df5 is None or df1.empty or df5.empty:
                log.warning("  %s: missing data", pair); continue
            log.info("  %s: %d 1h bars, %d 5m bars", pair, len(df1), len(df5))
            for t in simulate_combined(
                pair, df1, df5, n_exit=args.n_exit,
                no_block=args.no_block,
                early_1h=args.early_1h, early_5m=args.early_5m,
                **common_kw
            ):
                (all_is if t["time"] < oos_start else all_oos).append(t)

    # Sweep ema combinations
    combos = [
        (args.ema_1h, args.ema_5m, args.window, args.min_swing,
         args.lots_1h, args.lots_5m),
    ]

    W = 115
    for period, trades in [("IS", all_is), ("OOS", all_oos)]:
        m = metrics(trades, f"ema{args.ema_1h}1h_ema{args.ema_5m}5m", period)
        print(f"\n{'='*W}")
        oos_label = "IN-SAMPLE" if period == "IS" else "OUT-OF-SAMPLE"
        tags = ""
        if args.usd_filter: tags += "  usd_filter=on"
        if args.no_block:   tags += "  no_block=on"
        if args.early_1h:   tags += "  early_1h=on"
        if args.early_5m:   tags += "  early_5m=on"
        print(f"  {oos_label}  ema_1h={args.ema_1h}  ema_5m={args.ema_5m}  "
              f"w={args.window}  sw={args.min_swing}  "
              f"lots: 1h={args.lots_1h}  5m={args.lots_5m}  "
              f"min_bars={args.min_bars_1h}/{args.min_bars_5m}  "
              f"partial=${args.partial_usd:.0f}  "
              f"sl1h={args.sl_pips_1h or args.sl_pips:.0f}p  "
              f"sl5m={args.sl_pips_5m or args.sl_pips:.0f}p  "
              f"dist={args.max_dist_pips:.0f}p  cp={args.close_profit}{tags}")
        print(f"{'='*W}")
        print(f"  Trades: {m['n']}  (1h: {m['n_1h']}  5m: {m['n_5m']})")
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
            print(f"  Days exceeding -$1000: "
                  f"{sum(1 for p in pnls if p < -1000)}")
            print(f"  Days exceeding -$2000: "
                  f"{sum(1 for p in pnls if p < -2000)}")
            print(f"  Days exceeding -$5000: "
                  f"{sum(1 for p in pnls if p < -5000)}")


if __name__ == "__main__":
    main()
