# CLAUDE.md — Adaptive ML Trading Pipeline

Behavioral guidelines for this codebase.

---

## 1. Project scope

A single-purpose research codebase: walk-forward CatBoost classifier for trade entries, with strict leakage controls and FTMO-compliant simulation. See `PLAN.md` for the full design.

Default target: **EURUSD 5m**, 3-month train → 1-month trade, rolling.

---

## 2. Think before coding

- State assumptions. If uncertain, ask first.
- Push back when a simpler approach exists.
- Don't guess and implement.

---

## 3. Surgical changes

- Touch only what the task requires.
- Match existing style.
- Remove imports/variables you make unused. Don't remove pre-existing dead code unless asked.

---

## 4. Simplicity first

- No speculative abstractions.
- No "future flexibility" features.
- Senior-engineer test: if they'd call it overcomplicated, simplify.

---

## 5. Verify before finishing

Always syntax-check edited Python files:
```
python -c "import ast; ast.parse(open('file.py').read())"
```

---

## 6. Hard rules — leakage & realism

The most important rules in this repo. Violating them silently invalidates results.

### 6a. Feature pure-function contract
Every feature builder takes `df_up_to_t` and returns a value depending **only** on rows ≤ t. Unit test:
```python
assert f(df.iloc[:t]) == f(df.iloc[:t+10]).iloc[t]
```

### 6b. No `shift(-N)`, no `iloc[i+...]`, no `min_periods=0` in features
Caught by `ml/leakage.py` static scan. Run gates whenever `ml/features.py` changes.

### 6c. Label resolution ≤ train-window end − embargo
Embargo = `max_holding_bars`. Asserted in `walkforward.py`.

### 6d. Cost model is non-negotiable
- Long entry pays half-spread above mid; exits at mid − half-spread.
- Commission round-trip = `notional × commission × 2`.
- Labels are computed **net of costs**.

### 6e. TP/SL invariants
At order time:
```
long:  tp > entry > sl
short: sl > entry > tp
```

### 6f. The `or 0` trap
```python
# WRONG — 0.0 is falsy
x = float(stats.get("k", 0) or 0)
# RIGHT
v = stats.get("k")
x = float(v) if v is not None else 0.0
```

### 6g. ATR alignment
When entering at close of bar `t`, SL/TP must use `ATR(t-1)`. Don't include `t`'s own range.

### 6h. Session VWAP resets at session open
Never roll continuously across sessions.

---

## 7. Class imbalance

- CatBoost `auto_class_weights="Balanced"`.
- No SMOTE / oversampling — breaks time order.
- Evaluate with PR-AUC and precision@k. Not accuracy.

---

## 8. Walk-forward discipline

- Train and trade windows are **adjacent**, never overlap.
- Slide by `trade_months`.
- Each window's model is independent. No parameter carryover.

---

## 9. Tuning order

1. Re-check leakage gates.
2. Tune the label (RR, SL multiplier, timeout).
3. Raise `min_proba`.
4. Symbol/TF scan.
5. Feature ablation.

Don't reach for deeper trees or more features before #1-3.

---

## 10. Files

| File | Role |
|---|---|
| `scripts/ml_adaptive_entries.py` | CLI entrypoint |
| `ml/data.py` | Parquet loader from `data/cache/` |
| `ml/features.py` | Past-only feature builders |
| `ml/labels.py` | Triple-barrier labeling, 1m resolution |
| `ml/dataset.py` | (X, y, t) assembly |
| `ml/model.py` | CatBoost wrapper |
| `ml/simulator.py` | Bar-by-bar sim with spread/commission/FTMO |
| `ml/walkforward.py` | Rolling-window driver |
| `ml/leakage.py` | Static + runtime checks |
| `ml/report.py` | Per-run report |
| `data/cache/` | Historical parquet files |
| `PLAN.md` | Full design document |
