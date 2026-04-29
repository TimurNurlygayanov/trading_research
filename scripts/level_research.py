"""
Level research pipeline -- EURUSD, any timeframe.

Phase 1  Build a causal level map as we scan through history:
  - Confirmed swing pivot highs/lows (pivot_n bars each side)
  - Previous day H/L
  - Round numbers (50-pip grid)
  - Nearby levels merged into zones (zone_pips half-width)
  - Each zone tracks its own touch history

Phase 2  Detect touch events (causal, no lookahead):
  A touch fires on the FIRST bar where price enters a zone after being
  outside it.  Features recorded per touch:
    touch_num          -- 0 = first ever touch of this zone
    bars_since_last    -- bars elapsed since previous touch (NaN on first)
    approach_side      -- "support" (from above) | "resistance" (from below)
    approach_atr       -- ATRs traveled from last local extreme to the zone
    approach_bars      -- consecutive same-direction bars into the touch
    candle_range_atr   -- (H-L)/ATR of the touch bar
    wick_ratio         -- rejection wick / total range  (direction-aware)
    body_pct           -- |close-open| / (H-L)
    close_inside       -- did the bar CLOSE inside the zone? (vs wick only)
    level_type         -- swing_high | swing_low | round | prev_day_h | prev_day_l
    cy_hour, dow

Phase 3  Measure outcomes (forward-looking, NOT causal -- research only):
  For each touch look forward fwd_bars:
    max_bounce_atr     -- max move in expected direction / ATR
    max_against_atr    -- max move opposite direction / ATR
    did_bounce_1atr    -- reached 1 ATR in expected direction
    did_bounce_2atr    -- reached 2 ATR in expected direction
    did_break          -- closed through the zone in opposite direction

Phase 4  Console stats:
  Bounce rate by touch_num, approach_side, wick_ratio quartile,
  approach_atr quartile, close_inside.

Usage:
  python -m scripts.level_research
  python -m scripts.level_research --tf 1h --start 2022-01-01
  python -m scripts.level_research --fwd-bars 20 --zone-pips 8
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.data_fetcher import fetch_ohlcv

PIP = 0.0001


# ── timezone ------------------------------------------------------------------
def _cy_hours(df: pd.DataFrame) -> np.ndarray:
    idx = df.index
    try:
        idx_utc = idx.tz_localize("UTC")
    except TypeError:
        idx_utc = idx.tz_convert("UTC")
    return idx_utc.tz_convert("Asia/Nicosia").hour.values


# ── indicators ----------------------------------------------------------------
def add_features(df: pd.DataFrame, atr_len: int = 14) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = ta.atr(out["High"], out["Low"], out["Close"], length=atr_len)
    return out


# ── pivot detection (vectorised, centered window) ----------------------------
def _pivot_value_arrays(df: pd.DataFrame,
                        pivot_n: int) -> tuple[np.ndarray, np.ndarray]:
    high_s = pd.Series(df["High"].values)
    low_s  = pd.Series(df["Low"].values)
    w = 2 * pivot_n + 1
    ph = (high_s == high_s.rolling(w, center=True, min_periods=pivot_n + 1).max()).values
    pl = (low_s  == low_s.rolling( w, center=True, min_periods=pivot_n + 1).min()).values
    return (np.where(ph, df["High"].values, np.nan),
            np.where(pl, df["Low"].values,  np.nan))


# ── daily H/L arrays ---------------------------------------------------------
def _daily_hl_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n   = len(df)
    pdh = np.full(n, np.nan)
    pdl = np.full(n, np.nan)
    daily     = df.resample("1D").agg({"High": "max", "Low": "min"})
    bar_dates = df.index.normalize()
    for i in range(1, len(daily)):
        mask     = bar_dates == daily.index[i]
        pdh[mask] = float(daily["High"].iloc[i - 1])
        pdl[mask] = float(daily["Low"].iloc[i - 1])
    return pdh, pdl


# ── zone management ----------------------------------------------------------
class Zone:
    """One price zone with its own touch history."""
    __slots__ = ("price", "half", "level_type", "formed_at",
                 "touch_times", "touch_sides")

    def __init__(self, price: float, half: float,
                 level_type: str, formed_at: int):
        self.price      = price
        self.half       = half          # half-width in price
        self.level_type = level_type
        self.formed_at  = formed_at
        self.touch_times: list[int] = []
        self.touch_sides: list[str] = []

    @property
    def top(self) -> float: return self.price + self.half
    @property
    def bot(self) -> float: return self.price - self.half

    def overlaps(self, other_price: float) -> bool:
        return abs(self.price - other_price) <= self.half * 2

    def contains(self, price: float) -> bool:
        return self.bot <= price <= self.top

    def touch_count(self) -> int:
        return len(self.touch_times)

    def bars_since_last(self, i: int) -> float:
        if not self.touch_times:
            return np.nan
        return i - self.touch_times[-1]

    def record_touch(self, bar_idx: int, side: str):
        self.touch_times.append(bar_idx)
        self.touch_sides.append(side)


def _merge_or_add(zones: list[Zone], price: float, half: float,
                  level_type: str, formed_at: int):
    """Add a new zone, or skip if one already exists within merge distance."""
    for z in zones:
        if z.overlaps(price):
            return  # absorbed into existing zone
    zones.append(Zone(price, half, level_type, formed_at))


# ── approach features --------------------------------------------------------
def _approach_features(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                       atr_v: np.ndarray, i: int,
                       side: str, lookback: int = 30
                       ) -> tuple[float, int]:
    """
    Returns (approach_atr, approach_bars).
    approach_atr: how many ATRs price traveled from its last local extreme
                  to reach the level (rough momentum measure).
    approach_bars: consecutive same-direction bars just before touch.
    """
    atr = atr_v[i]
    if atr <= 0 or np.isnan(atr):
        return np.nan, 0

    start = max(0, i - lookback)

    if side == "support":   # price falling into support
        # Find highest high in lookback
        recent_high = np.max(high[start:i + 1])
        approach_atr = (recent_high - close[i]) / atr
        # Count consecutive down-bars (close < prev close)
        bars = 0
        for j in range(i - 1, max(0, i - lookback) - 1, -1):
            if close[j] < close[j - 1] if j > 0 else False:
                bars += 1
            else:
                break
    else:   # price rising into resistance
        recent_low = np.min(low[start:i + 1])
        approach_atr = (close[i] - recent_low) / atr
        bars = 0
        for j in range(i - 1, max(0, i - lookback) - 1, -1):
            if close[j] > close[j - 1] if j > 0 else False:
                bars += 1
            else:
                break

    return float(approach_atr), int(bars)


# ── outcome measurement (forward-looking) ------------------------------------
def _measure_outcome(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     n: int, i: int, side: str,
                     zone_bot: float, zone_top: float,
                     atr: float, fwd_bars: int) -> dict:
    end = min(i + fwd_bars + 1, n)
    if i + 1 >= n:
        return dict(max_bounce_atr=np.nan, max_against_atr=np.nan,
                    did_bounce_1atr=False, did_bounce_2atr=False,
                    did_break=False)

    future_high  = high[i + 1:end]
    future_low   = low[i + 1:end]
    future_close = close[i + 1:end]
    ref          = close[i]

    if side == "support":     # expected: price goes UP
        bounce_moves  = future_high - ref
        against_moves = ref - future_low
        # break: any close below the zone bottom
        broke = bool(np.any(future_close < zone_bot))
    else:                     # resistance: price goes DOWN
        bounce_moves  = ref - future_low
        against_moves = future_high - ref
        broke = bool(np.any(future_close > zone_top))

    max_bounce  = float(np.max(bounce_moves))  if bounce_moves.size  else 0.0
    max_against = float(np.max(against_moves)) if against_moves.size else 0.0

    return dict(
        max_bounce_atr   = max_bounce  / atr if atr > 0 else np.nan,
        max_against_atr  = max_against / atr if atr > 0 else np.nan,
        did_bounce_1atr  = max_bounce  >= 1.0 * atr,
        did_bounce_2atr  = max_bounce  >= 2.0 * atr,
        did_break        = broke,
    )


# ── main scan ----------------------------------------------------------------
def run_research(df: pd.DataFrame, cy_hours_arr: np.ndarray,
                 pivot_n: int, zone_pips: float, round_step: float,
                 fwd_bars: int, approach_lookback: int) -> pd.DataFrame:

    open_  = df["Open"].values
    high   = df["High"].values
    low    = df["Low"].values
    close  = df["Close"].values
    atr_v  = df["atr"].values
    dow    = df.index.dayofweek.values
    ts     = df.index
    n      = len(df)

    piv_h, piv_l = _pivot_value_arrays(df, pivot_n)
    pdh, pdl     = _daily_hl_arrays(df)

    half_price = zone_pips * PIP   # zone half-width in price

    zones: list[Zone] = []
    rows:  list[dict] = []

    # Track which zones price was inside on the PREVIOUS bar
    # to detect zone ENTRIES (first bar inside, not every bar inside)
    prev_inside: set[int] = set()   # indices into `zones`

    for i in range(1, n - fwd_bars):
        if np.isnan(atr_v[i]) or atr_v[i] <= 0:
            continue

        # ── 1. Add newly confirmed pivot levels ──────────────────────────────
        # Pivot at j is confirmed once bar j+pivot_n is seen, i.e. j = i-pivot_n
        j = i - pivot_n
        if j >= 0:
            if not np.isnan(piv_h[j]):
                _merge_or_add(zones, piv_h[j], half_price, "swing_high", j)
            if not np.isnan(piv_l[j]):
                _merge_or_add(zones, piv_l[j], half_price, "swing_low", j)

        # ── 2. Add prev-day H/L (already arrays, just check for new values) --
        if not np.isnan(pdh[i]):
            _merge_or_add(zones, pdh[i], half_price, "prev_day_h", i)
        if not np.isnan(pdl[i]):
            _merge_or_add(zones, pdl[i], half_price, "prev_day_l", i)

        # ── 3. Add round number levels near current price --------------------
        c    = close[i]
        base = round(c / round_step) * round_step
        for k in range(-4, 5):
            _merge_or_add(zones, round(base + k * round_step, 5),
                          half_price, "round", i)

        # ── 4. Detect zone entries -------------------------------------------
        atr   = atr_v[i]
        cur_inside: set[int] = set()

        for zi, z in enumerate(zones):
            # Only consider levels formed before this bar
            if z.formed_at >= i:
                continue
            # Is this bar inside the zone?
            bar_inside = (low[i] <= z.top) and (high[i] >= z.bot)
            if not bar_inside:
                continue
            cur_inside.add(zi)

            # Touch = first bar entering the zone (was outside on prev bar)
            if zi in prev_inside:
                continue

            # Determine approach side from the close of the bar BEFORE entry
            prev_close = close[i - 1]
            if prev_close > z.top:
                side = "support"      # coming from above, level acts as support
            elif prev_close < z.bot:
                side = "resistance"   # coming from below, level acts as resistance
            else:
                side = "inside"       # was already near level -- skip ambiguous
                continue

            # ── Features at touch ────────────────────────────────────────────
            touch_num       = z.touch_count()
            bars_since      = z.bars_since_last(i)

            approach_atr, approach_bars = _approach_features(
                close, high, low, atr_v, i, side, approach_lookback)

            candle_range = high[i] - low[i]
            range_atr    = candle_range / atr if atr > 0 else np.nan

            if candle_range > 0:
                body_pct = abs(close[i] - open_[i]) / candle_range
                # Rejection wick in expected bounce direction
                if side == "support":    # expect bounce up -> lower wick matters
                    wick = min(open_[i], close[i]) - low[i]
                else:                    # expect bounce down -> upper wick matters
                    wick = high[i] - max(open_[i], close[i])
                wick_ratio = max(wick, 0) / candle_range
            else:
                body_pct   = np.nan
                wick_ratio = np.nan

            # Did the bar CLOSE inside the zone (vs only wick touching it)?
            close_inside = z.bot <= close[i] <= z.top

            # ── Outcomes (forward-looking) ───────────────────────────────────
            outcome = _measure_outcome(
                high, low, close, n, i, side,
                z.bot, z.top, atr, fwd_bars)

            rows.append({
                "bar_idx":         i,
                "signal_time":     ts[i],
                "level_price":     z.price,
                "level_type":      z.level_type,
                "approach_side":   side,
                "touch_num":       touch_num,       # 0 = virgin touch
                "bars_since_last": bars_since,
                "approach_atr":    round(float(approach_atr), 3) if not np.isnan(approach_atr) else np.nan,
                "approach_bars":   approach_bars,
                "candle_range_atr": round(float(range_atr), 3) if not np.isnan(range_atr) else np.nan,
                "wick_ratio":      round(float(wick_ratio), 3) if not np.isnan(wick_ratio) else np.nan,
                "body_pct":        round(float(body_pct), 3)   if not np.isnan(body_pct)   else np.nan,
                "close_inside":    int(close_inside),
                "cy_hour":         int(cy_hours_arr[i]),
                "dow":             int(dow[i]),
                "atr":             atr,
                **outcome,
            })

            z.record_touch(i, side)

        prev_inside = cur_inside

    return pd.DataFrame(rows)


# ── statistical summary -------------------------------------------------------
def _rate(df: pd.DataFrame, col: str = "did_bounce_1atr") -> str:
    if df.empty:
        return "n=0"
    n   = len(df)
    hit = df[col].sum()
    return f"n={n:4d}  bounce1={hit/n*100:5.1f}%  bounce2={df['did_bounce_2atr'].sum()/n*100:5.1f}%  break={df['did_break'].sum()/n*100:5.1f}%"


def print_stats(ev: pd.DataFrame) -> None:
    if ev.empty:
        print("No touch events.")
        return

    print(f"\nTotal touch events : {len(ev)}")
    print(f"  support          : {(ev['approach_side']=='support').sum()}")
    print(f"  resistance       : {(ev['approach_side']=='resistance').sum()}")

    # -- By touch number (first touch vs re-tests) ----------------------------
    print("\n--- By touch number (0 = first ever touch of this zone) ---")
    for t in sorted(ev["touch_num"].unique()):
        sub = ev[ev["touch_num"] == t]
        label = f"touch #{t}" if t < 5 else "touch 5+"
        print(f"  {label:10s}  {_rate(sub)}")

    # -- By approach side -------------------------------------------------------
    print("\n--- By approach side ---")
    for side in ("support", "resistance"):
        sub = ev[ev["approach_side"] == side]
        print(f"  {side:12s}  {_rate(sub)}")

    # -- By close_inside -------------------------------------------------------
    print("\n--- Close inside zone vs wick-only touch ---")
    for ci, label in [(1, "close inside "), (0, "wick only    ")]:
        sub = ev[ev["close_inside"] == ci]
        print(f"  {label}  {_rate(sub)}")

    # -- Wick ratio quartiles --------------------------------------------------
    print("\n--- Wick ratio quartiles (rejection wick / candle range) ---")
    ev2 = ev.dropna(subset=["wick_ratio"])
    if not ev2.empty:
        ev2 = ev2.copy()
        ev2["wr_q"] = pd.qcut(ev2["wick_ratio"], 4,
                               labels=["Q1 small", "Q2", "Q3", "Q4 large"])
        for q in ["Q1 small", "Q2", "Q3", "Q4 large"]:
            sub = ev2[ev2["wr_q"] == q]
            print(f"  {q:10s}  {_rate(sub)}")

    # -- Approach ATR quartiles ------------------------------------------------
    print("\n--- Approach ATR quartiles (how far price traveled to reach level) ---")
    ev3 = ev.dropna(subset=["approach_atr"])
    if not ev3.empty:
        ev3 = ev3.copy()
        ev3["aa_q"] = pd.qcut(ev3["approach_atr"], 4,
                               labels=["Q1 slow", "Q2", "Q3", "Q4 fast"])
        for q in ["Q1 slow", "Q2", "Q3", "Q4 fast"]:
            sub = ev3[ev3["aa_q"] == q]
            print(f"  {q:10s}  {_rate(sub)}")

    # -- By level type ---------------------------------------------------------
    print("\n--- By level type ---")
    for lt in sorted(ev["level_type"].unique()):
        sub = ev[ev["level_type"] == lt]
        print(f"  {lt:15s}  {_rate(sub)}")

    # -- By session hour -------------------------------------------------------
    print("\n--- By Cyprus hour ---")
    for h in sorted(ev["cy_hour"].unique()):
        sub = ev[ev["cy_hour"] == h]
        if len(sub) >= 10:
            print(f"  hour {h:02d}  {_rate(sub)}")

    # -- Touch 0 vs 1 breakdown by wick ----------------------------------------
    print("\n--- First touch (0) vs second touch (1) x close_inside ---")
    for t in (0, 1):
        for ci, lbl in ((1, "close_inside"), (0, "wick_only")):
            sub = ev[(ev["touch_num"] == t) & (ev["close_inside"] == ci)]
            if len(sub) >= 5:
                print(f"  touch#{t} {lbl:12s}  {_rate(sub)}")


# ── main ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Level touch research pipeline")
    ap.add_argument("--symbol",       default="EURUSD")
    ap.add_argument("--tf",           default="1h")
    ap.add_argument("--start",        default="2022-01-01")
    ap.add_argument("--end",          default="2025-12-31")
    ap.add_argument("--atr-len",      type=int,   default=14)
    ap.add_argument("--pivot-n",      type=int,   default=5,
                    help="bars each side for swing pivot confirmation")
    ap.add_argument("--zone-pips",    type=float, default=10,
                    help="half-width of each zone in pips (default 10)")
    ap.add_argument("--round-step",   type=float, default=0.0050,
                    help="round-number grid in price (0.005 = 50 pips)")
    ap.add_argument("--fwd-bars",     type=int,   default=20,
                    help="bars to look forward for outcome measurement")
    ap.add_argument("--approach-lb",  type=int,   default=30,
                    help="lookback bars for approach feature calculation")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent

    print(f"Fetching {args.symbol} {args.tf}  {args.start} -> {args.end}")
    raw = fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    print(f"  {len(raw):,} bars\n")
    if raw.empty:
        return

    df = add_features(raw, atr_len=args.atr_len)
    cy = _cy_hours(df)

    print("Scanning levels and touch events...")
    ev = run_research(df, cy,
                      pivot_n=args.pivot_n,
                      zone_pips=args.zone_pips,
                      round_step=args.round_step,
                      fwd_bars=args.fwd_bars,
                      approach_lookback=args.approach_lb)

    print(f"  touch events found: {len(ev)}")

    print("\n" + "=" * 65)
    print(f"LEVEL TOUCH ANALYSIS  {args.symbol} {args.tf}  "
          f"zone=+-{args.zone_pips}pip  fwd={args.fwd_bars}bars")
    print("=" * 65)
    print_stats(ev)

    out_path = out_dir / f"level_touches_{args.symbol}_{args.tf}.csv"
    ev.to_csv(out_path, index=False)
    print(f"\n-> {out_path}  ({len(ev)} touch events)")


if __name__ == "__main__":
    main()
