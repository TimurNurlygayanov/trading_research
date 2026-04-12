# Trading Research — Autonomous Strategy Pipeline

An autonomous multi-agent system that takes a trading idea from plain text to a validated, backtested strategy with no manual coding required.

## How It Works

```
User Idea → Pre-Filter → Implementer → Backtest → Validator → Summariser → Learner
```

1. **Pre-Filter** — LLM screens the idea for viability (novelty, edge plausibility, data requirements). Rejects junk early.
2. **Implementer** — LLM generates a complete `backtesting.py` Strategy class with proper `init()`/`next()` structure, ATR-based risk management, and a hyperparameter search space.
3. **Backtest** (Modal) — Runs on Modal.com (4 CPUs, 8 GB):
   - Leakage detection (AST + regex scan for lookahead bias)
   - Optuna hyperparameter optimization — 50 trials, TPE sampler
   - Walk-forward validation — 5 folds on training data
   - Monte Carlo permutation test — 1000 permutations
   - OOS backtest on 2026+ data (never used for fitting)
4. **Validator** (Modal) — LLM reviews results, checks edge coherence, flags curve-fitting.
5. **Summariser** — Writes a concise research report and stores it.
6. **Learner** — Extracts lessons (what worked, what didn't) into a shared knowledge base used by future Implementer calls.

## Key Design Decisions

- **No data leakage**: hard OOS cutoff at 2026-01-01; leakage detector rejects `shift(-N)`, `bfill`, rolling/pandas_ta calls inside `next()`.
- **Minimum bar**: 100 signals/year required to pass backtest; strategies with fewer are rejected regardless of Sharpe.
- **Heavy compute on Modal**: Render.com runs the always-on orchestrator; Modal.com handles CPU-intensive backtests/validation in isolated containers.
- **Self-improving**: the Learner agent writes structured notes back to the knowledge base, so the Implementer gets better prompts over time.

## Stack

| Layer | Tool |
|---|---|
| Orchestrator | FastAPI on Render.com |
| LLM | Claude (Anthropic) |
| Backtesting | backtesting.py + pandas_ta |
| Optimization | Optuna (TPE, 50 trials) |
| Heavy compute | Modal.com |
| Database | Supabase (Postgres) |
| Market data | Massive (formerly Polygon.io) |
| Storage | Cloudflare R2 |

## Running Locally

```bash
cp .env.example .env  # fill in API keys
pip install -r requirements.txt
python -m uvicorn orchestrator.main:app --reload
```

Submit ideas at `http://localhost:8000/ideas`, monitor at `http://localhost:8000/dashboard`.

## Deploying

1. Push to GitHub → Render auto-deploys the orchestrator.
2. Deploy Modal jobs once:
   ```bash
   modal deploy modal_jobs/backtest_job.py
   modal deploy modal_jobs/validator_job.py
   ```
3. Set all secrets in Render dashboard and Modal secret store (`trading-research-secrets`).
