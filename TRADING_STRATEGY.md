# Range Fade Strategy

Mean-reversion strategy on 1m forex bars. When price breaks out of a tight rolling range, fade the breakout (short above range high, long below range low). Time-of-day and day-of-week filters remove low-quality sessions.

---

## Logic

1. **Range**: rolling `max(high)` and `min(low)` over the last 20 bars (excludes current bar — no lookahead).
2. **Tightness filter**: skip entry if `(range_high − range_low) ≥ 1.5 × ATR(14)`. Only trade when the market has been consolidating tightly.
3. **Entry**: at the close of the breakout bar, take the opposite direction.
   - Close > range_high → **short**
   - Close < range_low  → **long**
4. **Exit**: ATR-based TP and SL, set at entry.
   - TP = entry ± 1.0 × ATR (in trade direction)
   - SL = entry ∓ 2.0 × ATR
5. **Size**: 1 lot (100,000 units). P&L in USD.
   - XXX/USD pairs: 1 pip = $10 directly.
   - USD/YYY pairs: raw P&L divided by entry price.
6. **Commission**: $7 round-trip (≈ 0.7 pip).
7. **One trade at a time per pair**. New signals while in a trade are ignored.

---

## Time Filters

Derived from per-hour / per-DOW analysis on 2022–2024 data (stored in `scripts/strategy1_config.json`). Filters remove sessions with consistently negative PnL.

| Pair   | Excluded UTC hours | Excluded days |
|--------|--------------------|---------------|
| EURUSD | 1, 7, 8, 9, 11, 12, 14, 15, 18 | — |
| GBPUSD | 0, 1, 2, 6, 8, 9, 11, 12, 13, 20 | Wed (2), Thu (3) |
| AUDUSD | 0, 1 | — |
| NZDUSD | 14, 16 | — |
| USDCHF | 14 | — |
| USDCAD | 13, 14, 16 | — |

**Universal best hours** (positive for all 6 pairs): 3–5 UTC (Asian pre-market) and 19, 21–23 UTC (NY evening).  
**GBPUSD** is the most session-sensitive pair — avoid Wed/Thu entirely.  
Note: USDJPY is excluded from the portfolio (inconsistent results).

---

## Files

| File | Purpose |
|------|---------|
| `scripts/strategy1.py` | Core backtest — single pair, sweep, multi-year, `--time-analysis` |
| `scripts/strategy1_config.json` | Per-pair params + hour/day filters |
| `scripts/strategy1_portfolio.py` | Portfolio backtest — all 6 pairs with config filters |

---

## How to Run

### Single pair
```bash
# Default: EURUSD, 2024, 1m
python -m scripts.strategy1

# Custom pair and date range
python -m scripts.strategy1 --symbol NZDUSD --start 2023-01-01 --end 2025-01-01 \
    --fade --tight-atr 1.5 --tp-atr 1 --sl-atr 2

# With hour/day filters
python -m scripts.strategy1 --symbol NZDUSD --start 2024-01-01 --end 2025-01-01 \
    --fade --tight-atr 1.5 --tp-atr 1 --sl-atr 2 \
    --hours 3 4 5 19 21 22 23 --days 0 1 2 3 4

# TP/SL grid sweep
python -m scripts.strategy1 --symbol NZDUSD --start 2024-01-01 --end 2025-01-01 \
    --fade --tight-atr 1.5 --sweep
```

### Time-of-day analysis (per-pair)
```bash
python -m scripts.strategy1 --symbol NZDUSD --start 2022-01-01 --end 2025-01-01 \
    --fade --tight-atr 1.5 --tp-atr 1 --sl-atr 2 --time-analysis
```
Prints a table of net PnL / win rate / trades for each UTC hour (0–23) and each day of week.

### Portfolio (all 6 pairs with config filters)
```bash
# Default: 2022, 2023, 2024
python -m scripts.strategy1_portfolio

# Specific years
python -m scripts.strategy1_portfolio --years 2022 2023 2024 2025

# Specific date range
python -m scripts.strategy1_portfolio --start 2025-01-01 --end 2025-06-01

# Compare without time filters
python -m scripts.strategy1_portfolio --years 2022 2023 2024 --no-filters
```

---

## Backtest Results

### Portfolio (6 pairs, 1m, TP 1×ATR, SL 2×ATR, tight 1.5×ATR, with filters)

| Year | EURUSD | GBPUSD | AUDUSD | NZDUSD | USDCHF | USDCAD | **Total** | Win rate |
|------|--------|--------|--------|--------|--------|--------|-----------|----------|
| 2022 | +$1,882 | +$4,401 | +$6,130 | +$11,358 | +$5,150 | +$71 | **+$28,992** | 83.9% |
| 2023 | +$2,827 | +$609 | +$15,426 | +$25,761 | +$13,478 | +$2,712 | **+$60,813** | 87.5% |
| 2024 | +$2,678 | +$1,534 | +$25,421 | +$42,895 | +$17,542 | +$10,557 | **+$100,627** | 89.9% |
| 2025* | +$86 | +$655 | +$3,435 | +$16,364 | +$10,786 | +$1,703 | **+$33,030** | 88.7% |

*2025 is Jan–Apr only (partial year, out-of-sample for filters).

### Without time filters (comparison)

| Year | Total | Win rate | Trades |
|------|-------|----------|--------|
| 2022 | +$32,036 | 82.8% | 2,212 |
| 2023 | +$63,735 | 86.4% | 3,594 |
| 2024 | +$97,893 | 88.5% | 4,957 |

Filters reduce trades ~12% while maintaining similar or better PnL, with a 1–2pp higher win rate.

### Best individual pairs (unfiltered, 2022–2024)

| Pair | 2022 | 2023 | 2024 |
|------|------|------|------|
| NZDUSD | +$12,049 | +$27,071 | +$41,414 |
| AUDUSD | +$8,894 | +$16,467 | +$25,477 |
| USDCHF | +$5,857 | +$14,048 | +$17,616 |
| GBPUSD | +$3,240 | +$526 | +$1,185 |
| EURUSD | +$1,869 | +$2,378 | +$2,226 |
| USDJPY | negative — excluded | | |

---

## Key Parameters (strategy1_config.json)

```json
{
  "lookback":   20,    // bars for rolling range
  "atr_period": 14,    // ATR smoothing period
  "tp_atr":     1.0,   // TP = 1× ATR from entry
  "sl_atr":     2.0,   // SL = 2× ATR from entry
  "tight_atr":  1.5,   // max range size in ATR units
  "commission": 7.0,   // USD round-trip
  "timeframe":  "1m"
}
```

To edit filters or add/remove pairs, update `scripts/strategy1_config.json`.

---

## Caveats

- Commission is fixed at $7 (≈ 0.7 pip). Real slippage on 1m bars may be higher.
- USD/YYY P&L conversion uses entry price as approximation.
- Time filters were optimized on 2022–2024 (in-sample). 2025 is the first clean OOS year.
- Results assume unlimited concurrent positions across pairs (no margin/allocation model).
