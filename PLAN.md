# Implementation Plan — Adaptive CatBoost Entry Classifier with Walk-Forward Retraining

This is the master plan for the project. Updated as we learn.

## 1. Goal & success criteria

A script that, for a given symbol+timeframe, simulates "learn on the latest data, trade the next month, repeat" over 2025–2026 and reports realistic PnL under FTMO rules and real costs.

Default focus: **EURUSD 5m**, 3 months train / 1 month trade rolling.

**Success = all of:**
1. OOS (forward) Sharpe ≥ 1.0 over the full 2025–2026 walk-forward
2. Max drawdown ≤ 8% (FTMO overall) and no single day < −4% (FTMO daily)
3. Performance must not collapse if commissions are doubled or spread widened 50%
4. Random-label permutation test: AUC drops to ~0.5 (proves the model isn't memorizing time-based artifacts)
5. Trade count > 200 across the OOS period (avoid lucky-streak conclusions)

If any of these fail, v1 is not ready to escalate.

## 2. Architecture

```
scripts/ml_adaptive_entries.py    ← CLI entrypoint, walk-forward loop
ml/
  features.py     ← all feature builders, pure functions of past-only data
  labels.py       ← triple-barrier labeling with TP/SL/timeout
  dataset.py      ← assemble (X, y, t); class-imbalance aware
  model.py        ← CatBoost wrapper: train/predict/calibrate/save
  walkforward.py  ← rolling-window driver
  simulator.py    ← bar-by-bar trade simulator with spread/commission/FTMO gates
  leakage.py      ← runtime + offline leakage assertions
  data.py         ← parquet loader (data/cache)
  report.py       ← equity curve, per-window stats, regime breakdown
```

## 3. Libraries

| Purpose | Library |
|---|---|
| Model | CatBoost |
| Indicators | pandas-ta |
| Data | pandas, pyarrow |
| Validation | scikit-learn (TimeSeriesSplit only) |
| Plots | matplotlib |

## 4. CLI

```
python scripts/ml_adaptive_entries.py \
  --symbol EURUSD --timeframe 5m \
  --train-months 3 --trade-months 1 \
  --start 2025-01-01 --end 2026-05-01 \
  --rr 3.0 --sl-atr 1.5 --risk-pct 0.5 \
  --spread-pips 0.8 --commission 0.0002 \
  --fast | --heavy
  --min-proba 0.55 \
  --out runs/ml_adaptive/<run_id>/
```

## 5. Feature set

All past-only. Asserted by `leakage.py`.

- Trend/momentum: RSI(14, 50), EMA(20/50/200), slopes, MACD, Supertrend
- Volatility: ATR(14, 50), Bollinger width, vol percentile
- Position-in-range: `(close - minN) / (maxN - minN)` for N ∈ {20, 50, 100, 200}, distance to last swing high/low in ATR units
- VWAP, HL/2 SMA(20), distance from VWAP
- Structure: bars since last failed high / failed low, mirror-touch count, last 3 swing points
- Higher TF versions of the above (5m → 1h)
- Candle patterns: body/range, wick ratios, engulfing/inside/pin flags (current + higher TF)
- Time: hour sin/cos, weekday, session tag (categorical), bars since session open
- Adaptive: rolling win-rate of naive long/short RR=3 over last N=200 bars

## 6. Labels — triple-barrier

For each bar `t`, label hypothetical long and short entries:
- SL = entry ± sl_atr × ATR(t)
- TP = entry shifted by sl_atr × ATR(t) × rr
- Timeout: max_holding_bars (default 48 for 5m → 4h)
- Resolve which barrier hits first using **1m data** (available in data/cache)
- Subtract spread+commission from TP, add to SL before labeling

Train two binary classifiers: long-entry and short-entry.

## 7. Class imbalance

- `CatBoostClassifier(auto_class_weights="Balanced")`
- No SMOTE
- Eval with PR-AUC, precision@k — not accuracy/ROC-AUC
- Tune decision threshold on val slice (purged), target precision ≥ 0.45

## 8. Walk-forward driver

```
windows = []
cur = start
while cur + 3mo + 1mo <= end:
    train = df[cur : cur + 3mo]
    trade = df[cur + 3mo : cur + 3mo + 1mo]
    windows.append((train, trade))
    cur += 1mo
```

Per window: build features → label train → purged train/val split with embargo = max_holding_bars → fit → tune threshold on val → freeze → simulate trade slice bar-by-bar → save trades + equity. Aggregate.

## 9. Leakage defenses (5 layers)

1. Static scan of `ml/features.py` for `shift(-N)`, `iloc[i+...]`, `min_periods=0`
2. Pure-function contract per feature: `f(df[:t]) == f(df[:t+10][:t])`
3. `X.index.is_monotonic_increasing`; label resolution time ≤ trade-window start − embargo
4. Purge + embargo in train/val split
5. Permutation test (heavy mode only): shuffle labels, refit, Sharpe should collapse

**Common stupid mistakes — explicit asserts:**
- `long_tp > long_entry > long_sl`, mirror for short
- Long entry at `ask`, exit at `bid` (and vice versa)
- ATR(t-1) used when entering at close of t? Use bar-aligned correctly — never use bar `t`'s own future inside its own indicator
- Session VWAP resets at session open
- CLAUDE.md `or 0` trap

## 10. FTMO constraints (live during simulation)

| Rule | Action |
|---|---|
| Daily loss −4% | Halt new entries until next session; close open at day end |
| Overall loss −8% | Terminate run, reason `ftmo_max_loss` |
| Profit target +8% | Reported, not a halt |

Sizing: `qty = (risk_pct × equity) / (sl_distance × pip_value)`.

## 11. Cost model

```
entry = mid + (spread/2) × side
exit  = mid - (spread/2) × side
commission = notional × 0.0002 × 2  (round trip)
```

Default spreads (pips): EURUSD 0.8, GBPUSD 1.0, USDJPY 1.0, AUDUSD 1.0.

Heavy mode runs sensitivity sweep at spread × {0.5, 1, 1.5, 2}.

## 12. Fast vs heavy

| Aspect | Fast | Heavy |
|---|---|---|
| Features | ~20 essentials | full ~60-80 |
| CatBoost | depth=4, iter=300, lr=0.1 | depth=6-8, iter=2000, lr=0.03, early stop |
| Walk-forward | every 2nd window | every window |
| Calibration | none | isotonic |
| Validation | basic | + permutation + spread sweep + regime breakdown |
| Wall time on EURUSD 5m 2025-2026 | < 5 min | 30-90 min |

## 13. Tuning order

1. Tune the label first (TP/SL/timeout/RR) — if no config gives >40% win rate, no model saves it.
2. Filter aggressively, trade less — high `min_proba`.
3. Scan symbols+TFs in fast mode, escalate top 2-3 to heavy.
4. Feature ablation — drop groups, keep contributors.
5. Per-window threshold tuning.
6. Two-stage gate: model + rolling naive win-rate filter.
7. Tighter intra-day stop than FTMO (−2.5% halt for day).

## 14. Reporting

- Equity curve over full WF
- Per-window Sharpe / win rate / trades
- R-multiple distribution
- FTMO compliance summary
- Feature importance (avg across windows)
- Regime breakdown (vol bins)
- Permutation baseline (heavy)
- Spread sensitivity (heavy)

## 15. Build order

1. `data.py` + `labels.py` — eyeball labels first
2. `features.py` fast subset + pure-function tests
3. `simulator.py` standalone — replay hardcoded signals to verify costs/FTMO
4. `model.py` + `walkforward.py` — end-to-end fast run
5. Inspect results before adding complexity
6. Heavy mode + permutation + spread sweep
