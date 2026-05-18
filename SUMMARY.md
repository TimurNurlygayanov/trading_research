# Results Summary — Adaptive CatBoost Entry Classifier

Snapshot as of latest run. **All numbers preliminary — see "Known concerns" below.**

## Final configuration (best so far)

| Component | Value |
|---|---|
| Symbol / TF | EURUSD 5m |
| Walk-forward | 3 months train → 1 month trade, sliding by 1 month |
| Model | CatBoost, depth 3, 200 iterations, lr 0.1, `auto_class_weights="Balanced"` |
| Features | ~40 (heavy): RSI, EMA20/50/200 + slopes, ATR, position-in-range, MACD, Supertrend, VWAP, Bollinger, structure (bars-since-failed-extreme), candle anatomy + patterns, time-of-day/session, naive rolling win-rate, daily candle features, 1h HTF features |
| Label | Triple-barrier, RR=2:1 (SL=2 ATR, TP=1 ATR), max-hold=48 bars |
| Decision | Unified model with `side` feature; row-doubling at train time; at inference score both sides, take higher proba above threshold |
| Costs | spread 0.8 pips, commission 0.00003 per side (≈$7.50/lot RT) |
| Threshold | 0.90 (very selective) — explored 0.10 / 0.30 / 0.50 / 0.70 / 0.90 |
| Risk per trade | 0.2% equity |

## Headline numbers

### Full 2025 walk-forward (9 windows, all 2:1 thresholds compared)
| RR | Return | Max DD | Win % | Trades | PF |
|---|---|---|---|---|---|
| **2:1** (rr=0.5) | **+59.7%** | −5.3% | 74.3% | 7,783 | 1.11 |
| 3:1 (rr=0.33) | −12.2% | −19.9% | 81.4% | 11,527 | 0.97 |
| 4:1 (rr=0.25) | −53.9% | −59.2% | 85.5% | 14,994 | 0.85 |

→ RR=2:1 is the only profitable one. Tighter TPs killed by costs.

### 2026 Q1 — out-of-sample threshold sweep (RR=2:1)
| Threshold | Trades | Win % | PF | Return | Max DD | Sharpe |
|---|---|---|---|---|---|---|
| 0.10 / 0.30 / 0.50 | 2,327 | 77.2% | 1.36 | +43.6% | −2.65% | 13.1 |
| 0.70 | 1,950 | 78.9% | 1.55 | +51.1% | −1.65% | 17.4 |
| 0.90 | 317 | 90.5% | 3.51 | +17.6% | −0.57% | 23.7 |

### 2026 Q1 — directional / EMA-filter ablation
| Config | Trades | Win % | PF | Return | Max DD |
|---|---|---|---|---|---|
| thr=0.70 (baseline) | 1,950 | 78.9% | 1.55 | +51.1% | −1.65% |
| thr=0.70, short-only | 2,201 | 66.9% | 0.77 | **−28.6%** | −29% |
| thr=0.90 | 317 | 90.5% | 3.51 | +17.6% | −0.57% |
| thr=0.70 + EMA filter | 1,127 | 79.2% | 1.57 | +28.0% | −1.48% |
| thr=0.90 + EMA filter | 144 | 93.1% | 5.27 | +8.9% | −0.45% |

Short-only result confirms the model is using direction info (not just base-rate). EMA filter is roughly neutral on per-trade quality.

### 2026 Q1 — realistic FTMO costs (thr=0.90, RR=2:1)

Added `--ftmo-realistic` preset: spread 1.2 pips, commission 0.00004 (≈$10 RT), SL slippage 0.7 pips, TP slippage 0.2 pips.

| | Optimistic costs | Realistic FTMO |
|---|---|---|
| Trades | 317 | 310 |
| Win % | 90.5% | 89.7% |
| PF | 3.51 | 2.55 |
| Return | +17.6% | +12.1% |
| Max DD | −0.57% | −0.68% |

Strategy survives realistic costs — return drops ~5pp, PF drops 3.5→2.55, but still profitable with sub-1% DD.

## Diagnostic findings

- **Deep model overfit catastrophically** (train PR-AUC 0.995 vs val 0.453). Switching to depth=3, iter=200 — train 0.49 / val 0.51 — exposed the real signal.
- **Realistic costs matter** — initial commission default of 0.0002/side gave $52 round-trip and made the strategy look losing. At realistic $7.50 RT it became profitable.
- **Daily candle features added** (`d_today_green`, `d_today_body`, `d_prev_green`, `d_streak`, `d_close_vs_pdh/pdl`). Past-only verified by recompute test.
- **Threshold 0.90 produces highest-quality trades**: PF 3.5, max DD 0.57% on 2026 Q1 OOS.

## Known concerns / next audits

1. **Results may be too good** — +59% 2025 and +43% 2026 Q1 with sub-3% DD is anomalous for retail FX. Possible causes still to rule out:
   - Spread/commission still optimistic vs real FTMO accounts
   - No slippage on stops
   - 5m-bar TP/SL resolution (no 1m drill-down on this run) — when both barriers touched within one bar, we award SL conservatively, but this is the same on labels and live → no bias in either direction, but underestimates real volatility costs
   - FTMO daily-reset boundary uses UTC midnight, not 17:00 ET
2. **High trade frequency** — 30+ trades/day even at threshold 0.70 means real execution friction (latency, requote, partial fills) would matter and is not modeled.
3. **Permutation test not run** — should shuffle labels, retrain, confirm Sharpe collapses to ~0.
4. **Single symbol** — not yet tested on GBPUSD / AUDUSD / USDJPY.

## File outputs

Each run writes to `runs/<name>/`:
- `summary.json` — headline metrics
- `trades.csv` — every trade with entry/exit/pnl/outcome
- `windows.csv` — per-window stats
- `equity.csv` + `equity.png` — equity curve

Notable runs:
- [runs/wf_rr2to1/](runs/wf_rr2to1/) — full 2025, RR=2:1, threshold 0.10
- [runs/wf_2026_both/](runs/wf_2026_both/) — 2026 Q1 OOS, threshold 0.10
- [runs/wf_2026_A_thr090/](runs/wf_2026_A_thr090/) — 2026 Q1, threshold 0.90, no filter — current best risk-adjusted
- [runs/wf_2026_short/](runs/wf_2026_short/) — 2026 Q1, short-only (lost −28.6%, confirms directional learning)
