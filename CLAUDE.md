# CLAUDE.md — Trading Research Pipeline

Behavioral guidelines for working on this codebase. Read before making changes.

---

## 1. Think Before Coding

- State assumptions explicitly. If uncertain, ask first.
- If multiple approaches exist, present them — don't pick silently.
- If something is unclear, say so. Ask. Don't guess and implement.
- Push back when a simpler approach exists.

---

## 2. Surgical Changes

- Touch only what the task requires. Don't "improve" adjacent code.
- Don't refactor things that aren't broken.
- Match existing style even if you'd do it differently.
- Every changed line should trace directly to the request.
- When your changes make imports/variables unused, remove them. Don't remove pre-existing dead code unless asked.

---

## 3. Simplicity First

- No features beyond what was asked.
- No speculative abstractions or "future flexibility".
- If you write 200 lines and 50 would do, rewrite it.
- Ask: "Would a senior engineer call this overcomplicated?" If yes, simplify.

---

## 4. Verify Before Finishing

Always run `python -c "import ast; ast.parse(open('file.py').read())"` on every Python file you edit.
State a brief plan for multi-step tasks before starting.

---

## 5. Project-Specific: Known Bug Patterns

These bugs have already been fixed. Do not reintroduce them.

### 5a. The `or 0` trap
```python
# WRONG — 0.0 is falsy in Python, so `0.0 or 0` → 0 (int)
sharpe = float(stats.get("Sharpe Ratio", 0) or 0)

# RIGHT — use _safe_float() from backtest/engine.py
sharpe = _safe_float(stats.get("Sharpe Ratio"))
```

### 5b. `setattr` race condition in Optuna parallel trials
```python
# WRONG — mutates the shared class object; all parallel trials stomp each other
for k, v in params.items():
    setattr(strategy_class, k, v)
bt = Backtest(df, strategy_class, ...)

# RIGHT — create a throw-away subclass per trial
parameterized = type(strategy_class.__name__, (strategy_class,), {k: v for k, v in params.items()})
bt = Backtest(df, parameterized, ...)
```

### 5c. Equity-curve Sharpe is always ~0 for intraday strategies
backtesting.py computes Sharpe from *daily* equity returns. Intraday strategies that open and close within the same day produce near-zero daily returns, making Sharpe ≈ 0.
- `result.sharpe` in `BacktestResult` is the **per-trade annualized Sharpe** (`mean_pnl/std_pnl * sqrt(trades/year)`) — this is the metric used everywhere.
- `result.equity_sharpe` is the backtesting.py value — diagnostic only, never used for gates or optimization.

### 5d. Python dataclass field ordering
Fields with defaults must come after fields without defaults.
```python
# WRONG — non-default field after default field → TypeError at class definition
trade_sharpe: float = 0.0
raw_stats: dict          # no default

# RIGHT
raw_stats: dict
trade_sharpe: float = 0.0
```

### 5e. modal_job_id must be cleared on failure
Every `except` block in a Modal function must include `"modal_job_id": None` in the DB update, or the UI will show "job stuck" forever.
```python
except Exception as e:
    db.update_strategy(strategy_id, {
        "status": "failed",
        "modal_job_id": None,   # ← always include this
        "error_log": ...,
    })
```

### 5f. pandas_ta column names with float params
```python
# WRONG — pandas_ta may format 3.0 as "3.0" or "3" depending on version
col = "SUPERT_7_3.0"

# RIGHT — build column name dynamically from the actual params
col = f"SUPERT_{self.st_period}_{float(self.st_mult)}"
```

### 5g. Session filter blocks all trades on higher timeframes
Default `start_hour=7, end_hour=20` works for 1h/15m data but can block all 4h/1d bars.
Always default to `start_hour=0, end_hour=23` (no filter). Let the optimizer find good windows.

---

## 6. Project Architecture

### Pipeline stages (in order)
```
idea → filtered → implementing → [awaiting_research] → implemented
     → quick_testing → quick_tested → backtesting → validating → live
```
Failed strategies are auto-retried by `_process_failed_strategies()` in `queue_worker.py`.

### Key files
| File | Role |
|------|------|
| `orchestrator/queue_worker.py` | 10-min loop, dispatches Modal jobs, auto-fixes |
| `orchestrator/main.py` | FastAPI dashboard + API endpoints |
| `modal_jobs/backtest_job.py` | `run_quick_backtest`, `run_backtest_pipeline` on Modal |
| `backtest/engine.py` | `run_backtest()` wrapper, `BacktestResult` dataclass |
| `backtest/optimizer.py` | Optuna optimizer (uses `result.sharpe` = trade Sharpe) |
| `backtest/walk_forward.py` | Walk-forward validation (uses `result.sharpe`) |
| `agents/pre_filter.py` | Classifies submission as strategy or research, scores ideas |
| `agents/implementer.py` | Generates backtesting.py Strategy class from description |
| `agents/code_fixer.py` | Diagnoses failures, LLM-repairs broken strategy code |
| `db/supabase_client.py` | All DB helpers — never use supabase directly elsewhere |
| `db/schema.sql` | Source of truth for DB schema — always update when adding columns |

### Multi-timeframe quick test
Every strategy is tested on `["4h", "1h", "15m", "5m", "1m"]` with default params.
The best timeframe (by trade Sharpe) is stored in `best_timeframe` and used for full optimization.
Zero trades on ALL timeframes → `failed` immediately, routed to `code_fixer`.

### Research tasks
Submissions that look like questions ("how does X affect Y?") are classified as `research` by pre-filter and inserted into `research_tasks` table instead of `strategies`.
The `Modal` researcher agent runs the analysis and stores findings.

### Sharpe metric policy
- **Never** use backtesting.py's "Sharpe Ratio" stat as the primary metric for intraday strategies.
- **Always** use `_compute_trade_sharpe(trades_df, signals_per_year)` from `engine.py`.
- Quality gates, Optuna objective, walk-forward IS/OOS comparison — all use `result.sharpe` which is the trade Sharpe.

---

## 7. DB Schema Rules

- Always add `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for new columns (idempotent).
- Always add new tables to the RLS policy cleanup block in `schema.sql`:
  ```sql
  WHERE tablename IN ('strategies', 'user_ideas', ..., 'your_new_table')
  ```
- Run the full schema in Supabase SQL editor to verify before shipping.
