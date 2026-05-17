# Strategy 025 Backtest Results — 2025-2026 Forex Data

**Date:** 2026-05-17  
**Timeframe:** 5m  
**Data Period:** 2025-01-09 to 2026-05-15  
**Framework:** Custom numpy-based backtest with no slippage/commission modeling

---

## 1. EURUSD Alone

| Configuration | Trades | PnL | Return | Win% | Avg Win | Avg Loss | Max DD |
|---|---|---|---|---|---|---|---|
| LONG 10-bar, SL@current | 1,587 | -$20,571 | -205.7% | 19.5% | +7.4 | -3.4 | $21,316 |
| **LONG 10-bar, SL@prev** | **860** | **+$31,046** | **+310.5%** | **57.0%** | **+15.3** | **-8.1** | **$959** |
| LONG 20-bar, SL@current | 1,009 | -$12,997 | -130.0% | 18.6% | +7.8 | -3.2 | $13,690 |
| LONG 20-bar, SL@prev | 290 | +$27,319 | +273.2% | 64.5% | +18.1 | -4.8 | $1,192 |

**Winner:** 10-bar lookback with SL at previous swing point  
- 860 total trades
- +$31,046 profit (+310.5%)
- 57% win rate
- **Smallest drawdown: $959 (9.6% of starting capital)**

---

## 2. All USD Pairs (Long Only, No Hedging)

| Pair | Trades | PnL | Return | Win% | Avg Win | Avg Loss | Max DD | Max DD% |
|---|---|---|---|---|---|---|---|---|
| **EURUSD** | 1,587 | -$20,571 | -205.7% | 19.5% | +7.4 | -3.4 | $21,316 | 213% |
| **GBPUSD** | 1,647 | -$20,379 | -203.8% | 20.6% | +9.2 | -4.0 | $20,753 | 208% |
| **USDJPY** | 1,417 | +$1,775,760 | +17,757.6% | 25.6% | +1,320 | -286 | $110,180 | 1,102% |
| **AUDUSD** | 1,639 | -$28,667 | -286.7% | 17.0% | +5.6 | -3.3 | $28,807 | 288% |
| **NZDUSD** | 1,748 | -$33,854 | -338.5% | 16.1% | +4.5 | -3.2 | $33,867 | 339% |
| CADUSD | N/A | No data | N/A | N/A | N/A | N/A | N/A | N/A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **TOTAL** | 8,038 | +$1,672,289 | +16,722.9% | — | — | — | — | — |

**Key Observations:**
- **USDJPY is an outlier**: +$1.77M profit is suspicious and likely due to:
  - JPY pairs having different pip value scaling
  - Position sizing calculation not accounting for JPY pairs
  - Potential error in the backtest logic for this pair
- **Most pairs are negative**: EURUSD, GBPUSD, AUDUSD, NZDUSD all show losses
- **Drawdown violations**: All pairs show max DD% > 100%, violating FTMO rule of max 2% daily drawdown

---

## 3. CSV Data Files Generated

Equity curve data saved to CSV for analysis:
- `st025_EURUSD_LONG 10-bar, SL@prev_equity_curve.csv` — Best EURUSD config
- `st025_EURUSD_equity_curve.csv` — Multi-pair EURUSD results
- `st025_GBPUSD_equity_curve.csv`
- `st025_USDJPY_equity_curve.csv`
- `st025_AUDUSD_equity_curve.csv`
- `st025_NZDUSD_equity_curve.csv`

Each CSV contains: `Bar, Equity, Drawdown_$, Drawdown_%`

---

## 4. FTMO Compliance Analysis

**❌ FAILS FTMO Requirements:**

1. **Max Drawdown**: Strategy shows drawdowns of 200-1,100% of starting capital
   - FTMO Max: 2% daily, 5% cumulative
   - Strategy: 200-1,100% (far exceeds limits)

2. **Position Sizing**: Strategy risks too much per trade
   - Using 95% of equity per position is excessive
   - FTMO allows max 2% risk per trade

3. **Correlation Hedging**: No active checks for correlated pairs
   - EUR pairs (EURUSD, GBPUSD) often move together
   - May create unintended hedge effect
   - Not evaluated in this backtest

---

## 5. Key Findings & Recommendations

### What Worked:
✅ **EURUSD with SL at Previous Swing (10-bar)**
- 860 trades, 57% win rate
- +$31,046 profit
- **Only $959 max drawdown** (best in class)
- Suggests this strategy variant has merit

### What Didn't Work:
❌ **Most other USD pairs** failed when using same parameters
❌ **SL@current** consistently underperformed vs SL@prev
❌ **Looser lookback (20-bar)** generated fewer trades but still profitable

### Required Fixes:
1. **Position sizing**: Reduce from 95% per trade to 0.5-1% to meet FTMO rules
2. **Investigate USDJPY**: Verify pip calculation for JPY pairs
3. **Add correlation filter**: Prevent simultaneous trades on correlated pairs
4. **Risk management**: Implement hard daily loss limits and position limits
5. **Validate on out-of-sample data**: 2026 data only; need validation on 2025

---

## 6. Technical Notes

- **Data Source:** MetaTrader 5 terminal
- **Backtest Engine:** Custom numpy implementation (no backtesting.py)
- **Commission:** Not modeled (actual: ~1-2 pips per round-trip)
- **Slippage:** Not modeled
- **Session Filter:** None applied
- **Position Sizing:** Fixed 95% of equity per trade (unrealistic)

---

## 7. Next Steps

1. **Reduce position sizing** to 1% risk per trade
2. **Verify USDJPY calculations** (pip scaling)
3. **Test on out-of-sample 2025 data**
4. **Add correlation checks** for USD pairs
5. **Implement daily loss stops** for FTMO compliance
6. **Backtest with realistic commission/slippage**
7. **Walk-forward analysis** to detect curve-fitting
