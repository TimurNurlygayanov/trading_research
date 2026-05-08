# EMA Combined Strategy — Reference Document

**Strategy:** Double-EMA Extrema with Combined 1h + 5m Signals
**Account:** FTMO Swing $100 000
**Pairs:** EURUSD, AUDUSD, NZDUSD, USDCHF, USDCAD, GBPUSD, USDJPY, EURJPY
**Status:** Live (challenge-safe defaults: 0.5/0.25 lots, $1 000 daily stop)

---

## Table of Contents

1. [Core Idea](#1-core-idea)
2. [Signal Logic](#2-signal-logic)
3. [Position Rules (1h + 5m)](#3-position-rules-1h--5m)
4. [Take-Profit & Stop-Loss](#4-take-profit--stop-loss)
5. [Parameters](#5-parameters)
6. [Backtest Results](#6-backtest-results)
7. [Risk Management](#7-risk-management)
8. [Files & Entry Points](#8-files--entry-points)
9. [Known Caveats](#9-known-caveats)

---

## 1. Core Idea

Price reverts after a clear local peak or valley in the smoothed trend line. The strategy:

1. Computes a **Double EMA of (High+Low)/2** — two EMA passes suppress noise without adding lag.
2. Looks for the **peak or valley** of that DEMA within a sliding window of recent bars.
3. Fires a SHORT when the peak is on the current or previous bar, a LONG when the valley is. Entry is on the same bar — no waiting for confirmation.
4. Runs on **two timeframes simultaneously**: 1h is the trend anchor, 5m provides higher-frequency entries in the same direction.

---

## 2. Signal Logic

### Sample winning trades — EURUSD 1h (real backtest)

![Winning 1h trades](diagrams/winners_1h_EURUSD.png)

The blue line is the DEMA, the shaded blue band is the 20-bar lookback window, and the green/red shading marks the held trade. BUY arrows fire on a DEMA valley; SELL arrows on a peak. Yellow dotted line = max favorable excursion during the trade.

### Formula

```
hl2      = (High + Low) / 2
DEMA     = EMA( EMA(hl2, period), period )
window   = last W bars of DEMA
pos_max  = argmax(window)          # position of highest DEMA value
pos_min  = argmin(window)          # position of lowest DEMA value
```

### Signal rules

| Condition | Signal |
|-----------|--------|
| `pos_max > pos_min` AND `pos_max <= W − 1` | **SHORT** — peak is at or before current bar |
| `pos_min > pos_max` AND `pos_min <= W − 1` | **LONG** — valley is at or before current bar |
| `pos_max == pos_min` | No signal (flat EMA) |

The guard is `<= W − 1`, which allows the signal to fire on the same bar as the peak/valley — entering at the actual turning point. No look-ahead bias: the bar is confirmed closed before the signal is evaluated.

### EMA periods

| Timeframe | `ema_period` | `window` |
|-----------|-------------|----------|
| 1h        | 9           | 20       |
| 5m        | 20          | 20       |

EMA9 on 1h reacts faster; EMA20 on 5m stays smoother — best OOS combination (Sharpe 57 vs 25 for EMA20/EMA20).

---

## 3. Position Rules (1h + 5m)

| 1h state | 5m signal | Action |
|----------|-----------|--------|
| Flat | Any | Open 5m position freely |
| LONG active | LONG signal | Open 5m LONG (stack — both run simultaneously) |
| LONG active | SHORT signal | **Block** — 5m signal ignored |
| SHORT active | SHORT signal | Open 5m SHORT (stack) |
| SHORT active | LONG signal | **Block** |
| 1h flips direction | — | Close 1h + close any same-old-direction 5m → open new 1h |

The 1h flip clears the entire pair's exposure in the old direction before re-entering. This prevents holding both sides of the same pair simultaneously.

### Lot sizes (live default — challenge-safe)

| Timeframe | Lots | Max per pair (stacked) |
|-----------|------|------------------------|
| 1h        | 0.5  | 0.5                    |
| 5m        | 0.25 | 0.25                   |
| **Total** | —    | **0.75 lots / pair**   |

Scale up only after live performance matches backtest:
- Phase 2 (validated): `--lots-1h 1.0 --lots-5m 0.5`
- Phase 3 (full backtest size): `--lots-1h 2.0 --lots-5m 1.0`

---

## 4. Take-Profit & Stop-Loss

### Sample losing trades — EURUSD 1h (real backtest)

![Losing 1h trades](diagrams/losers_1h_EURUSD.png)

Only **5 of 308** EURUSD 1h trades lost — the −40 pip SL caps each loss at ~$100 per 0.5 lot. The pattern: signal fires, price keeps moving against the entry past the lookback window's range, SL hits.

### Entry

| Direction | Entry price |
|-----------|-------------|
| LONG (BUY)  | **Ask** |
| SHORT (SELL)| **Bid** |

### TP and SL placement

| Position | Take-Profit | Stop-Loss |
|----------|-------------|-----------|
| 1h LONG  | `Ask + 10 pips` | `Ask − 40 pips` |
| 1h SHORT | `Bid − 10 pips` | `Bid + 40 pips` |
| 5m LONG  | `Ask + 3 pips`  | none — `close_profit` exits |
| 5m SHORT | `Bid − 3 pips`  | none |

Both TP and SL are submitted with the entry order, so the broker manages them as resting limits — no monitoring loop required.

### Exit priority

1. **TP hit** (broker-side): position auto-closes when Bid ≥ TP (long) or Ask ≤ TP (short).
2. **SL hit** (broker-side, 1h only): −40 pips caps tail risk at ~$100/trade per 0.5 lot.
3. **Opposite signal**: client polls every 10 s; closes at market on flip.
4. **Daily stop**: client closes ALL positions if portfolio P&L < −$1 000.

### Cost model (EURUSD, 1 lot)

```
cost per trade = 2 × half_spread × pip_value + commission
               = 2 × 0.2 pip × $10 + $7  =  $11

Required move to break even (1 lot): 1.1 pips
Default 1h TP (10 pips) → ~$89 net per win
Default 5m TP (3 pips)  → ~$19 net per win
```

---

## 5. Parameters

### Signal parameters

| Parameter | Value |
|-----------|-------|
| `ema_1h` | 9 |
| `ema_5m` | 20 |
| `window` | 20 |

### Risk parameters (live defaults — challenge-safe)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lots_1h` | **0.5** | Half of Phase 2; cuts daily DD by 50% |
| `lots_5m` | **0.25** | |
| `tp_1h_pips` | 10 | ~90% MFE hit rate on 1h |
| `tp_5m_pips` | 3 | ~85% MFE hit rate on 5m |
| `sl_pips_1h` | 40 | Caps 1h tail risk at ~$100/trade per 0.5 lot |
| `sl_pips_5m` | 0 | No SL — `close_profit` handles 5m exits |
| `min_bars_5m` | 3 | Whipsaw guard before allowing exit on opposite signal |
| `min_bars_1h` | 2 | Same, on 1h |
| `early_1h` | True | Enter on the peak/valley bar itself (Sharpe 57 vs 35) |
| `partial_usd` | 0 | Disabled — partial closes reduced OOS P&L by 36% |
| `max_dist_pips` | 0 | Disabled — distance filter reduced Sharpe from 42 → 22 |
| `daily_stop` | **1 000** | Portfolio USD daily loss limit (5× FTMO buffer) |

### Run with all defaults

```bash
python -m scripts.strategy_live_trader --login <LOGIN> --server FTMO-Demo
```

That's it — no other flags needed. Set `MT5_PASSWORD` env var or pass `--password`.

---

## 6. Backtest Results

**Period:** Jan 1 – May 7 2026 · IS: Jan–Apr · OOS: Apr–May 2026
**Backtest lots:** 2.0 (1h) + 1.0 (5m) — quadruple the live default

### Recommended config (`early_1h=True, sl_pips_1h=40, partial_usd=0, min_bars=2/3, close_profit=True`)

| Period | Trades | P&L | Win% | Sharpe | Max DD | Worst day |
|--------|--------|-----|------|--------|--------|-----------|
| IS  | 13 555 | +$738 400 | 91.9% | 65.47 | −$5 215 | −$2 270 |
| **OOS** | **4 945** | **+$252 352** | **91.4%** | **57.36** | **−$2 940** | **−$255** |

### Scaled to live defaults (0.5/0.25 lots — 1/4 of backtest)

| Metric | Backtest (2/1) | Live default (0.5/0.25) |
|--------|----------------|--------------------------|
| OOS P&L | +$252 352 | ~+$63 100 |
| Max DD | −$2 940 | ~−$735 |
| Worst day | −$255 | ~−$64 |

Hitting FTMO's $10 000 profit target at live defaults: **~2 weeks** at the OOS daily-average pace, with ~16× headroom on the daily stop and ~14× on total drawdown.

---

## 7. Risk Management

### FTMO Swing $100k limits

| Limit | FTMO threshold | Strategy buffer (live defaults) |
|-------|----------------|---------------------------------|
| Daily drawdown | $5 000 | Stop at $1 000 (5× buffer) |
| Total drawdown | $10 000 | OOS scaled max DD: ~$735 (14× buffer) |
| Profit target | $10 000 | ~3 weeks at OOS pace |

### Scale-up sequence

```
Phase 1 — Live defaults (current):
  All 8 pairs, 0.5 lot 1h + 0.25 lot 5m, daily stop $1 000
  Goal: prove live fills match backtest. Pass FTMO challenge.

Phase 2 — Validated (after profitable challenge phase):
  Same pairs, 1.0 lot 1h + 0.5 lot 5m, daily stop $2 500
  Expected daily P&L: ~$650–$1 000 OOS-pace

Phase 3 — Full backtest size (after 4+ profitable weeks):
  Same pairs, 2.0 lot 1h + 1.0 lot 5m, daily stop $5 000
  Expected daily P&L: ~$1 300–$2 000 OOS-pace
```

### Daily circuit-breaker

```
Every 10 seconds:
  pnl = realized_today + unrealized_open_positions
  if pnl < -daily_stop_usd:
      close ALL positions (all pairs) at market
      halt new entries until midnight UTC
```

Uses `history_deals_get()` for realized P&L and live `positions_get()` for unrealized — reacts to floating losses, not just closed trades.

---

## 8. Files & Entry Points

| File | Purpose |
|------|---------|
| `scripts/strategy_ema_extrema_backtest.py` | Single-TF parameter sweep (1h or 5m) |
| `scripts/strategy_combined_backtest.py` | Combined 1h+5m backtest — IS/OOS split |
| `scripts/analyze_mfe.py` | MFE distribution — calibrate TP levels |
| `scripts/strategy_live_trader.py` | Live MT5 trader |
| `scripts/visualize_ema_trades.py` | Chart viewer with DEMA overlay |
| `docs/diagrams/winners_1h_EURUSD.png` | 6 sample winning 1h trades (EURUSD) |
| `docs/diagrams/losers_1h_EURUSD.png` | All 5 losing 1h trades (EURUSD, SL hit) |
| `docs/diagrams/winners_5m_EURUSD.png` | 6 sample winning 5m trades (EURUSD) |
| `docs/diagrams/losers_5m_EURUSD.png` | 6 sample losing 5m trades (EURUSD) |

### Re-run backtest

```bash
python -m scripts.strategy_combined_backtest \
    --start 2026-01-01 --end 2026-05-07 --oos-start 2026-04-01
```

All recommended params (ema_1h=9, sl_pips_1h=40, etc.) are baked into defaults.

### Start live trader

```bash
# Live default — challenge-safe (0.5 + 0.25 lots, $1 000 daily stop)
python -m scripts.strategy_live_trader --login <LOGIN> --server FTMO-Demo

# Single-pair validation (most conservative)
python -m scripts.strategy_live_trader --login <LOGIN> --server FTMO-Demo \
    --pairs EURUSD --lots-1h 0.25 --lots-5m 0.1 --daily-stop 500
```

---

## 9. Known Caveats

### `close_profit` fill assumption
Backtest exits use the intrabar High (long) or Low (short) the moment it crosses breakeven, assuming a resting limit order. Live execution depends on broker latency and spread at the TP touch. If live win% drops below 80% (vs 90% backtest), the TP is being missed and should be widened by 1–2 pips.

### `avg_hold` display bug in combined backtest
`avg_hold` shows nonsense numbers for combined runs because `flip_1h_clear` exits subtract a 5m bar index from a 1h bar index. P&L, win%, and Sharpe are correct — only `avg_hold` is wrong.

### OOS period is short
Apr–May 2026 = 31 trading days. Enough to validate, not enough to rule out lucky streaks. Re-run after 3–6 months of live data.

### Stop-loss (1h positions)
`--sl-pips-1h 40` caps tail risk at ~$100/trade per 0.5 lot. Backtest-optimal is no SL (Sharpe 51) but SL=40 gives Sharpe 42 with capped worst-case loss — preferable for challenge-safe trading.

### Distance filter (not recommended)
`--max-dist-pips 10` reduced OOS Sharpe from 42 → 22. Blocks entries during fast momentum that then continues — strategy performs better entering slightly late than not at all.

### `partial_usd` is harmful — keep disabled
Tested partial closes at half-position-profit threshold: OOS P&L dropped 36% ($67 737 → $43 152) with no improvement to max DD. Default is 0.

### USD correlation filter (not recommended)
`--usd-filter` reduced OOS P&L by 36% with worse max DD. The strategy's edge is independent mean-reversion per pair — cross-pair USD correlation does not matter at 5-bar hold times.

### 1m timeframe excluded
1m needs ~2.9 pips to cover costs but median MFE is only 3.2 pips — razor-thin edge, highly sensitive to spread.
