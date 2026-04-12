# Autonomous Multi-Agent Trading Strategy Research Pipeline

> **Goal:** A self-improving, always-on (24/7) cloud agent pipeline that researches papers & ideas, implements strategies, backtests them, validates for bugs/leakage, and reports results — while you do anything else. You can also drop your own strategy ideas in at any time and the pipeline picks them up automatically. Zero laptop dependency after a ~1-hour setup. Everything visible on a board online.

---

## TL;DR: Best Stack for Easy 24/7 + Premium Research

| What you want | Best pick | Why |
|---|---|---|
| **Always-on 24/7, easy deploy** | **Render.com** | You already have an account. Web services run 24/7 on paid tier, auto-deploy from GitHub, env vars UI. $7/month. |
| **Heavy compute: backtests & ML** | **Render.com Background Worker** (same account) or **Modal.com** | Backtests + Optuna + RandomForest need CPU + time. Separate service or Modal burst compute keeps the orchestrator light. |
| **LLM backbone** | **Claude Sonnet 4.5 / Haiku 3.5** | Best at code + reasoning, long context for papers |
| **Premium strategy research** | **Quantpedia API** | 700+ curated strategies with Sharpe, drawdown, rules, papers — all via JSON API |
| **Academic papers** | **Semantic Scholar API + SSRN** | 200M papers, free API, no paywalls on preprints |
| **Your own strategy ideas** | **Supabase table + simple web form** | Drop ideas in plain English → pipeline picks them up automatically |
| **Progress board** | **Linear.app** | Free Kanban, API, mobile app, Slack notifications |
| **Results DB** | **Supabase** | Free Postgres, REST API, built-in dashboard |
| **File storage** | **Cloudflare R2** | Free 10 GB, S3-compatible, free egress |

---

## Why 24/7 Instead of Just Nightly?

A nightly cron processes ~5 strategies/night → ~150/month.  
A 24/7 queue-based worker processes strategies as fast as they come in → **600–2000+/month**.  
The pipeline compounds: the more strategies tried, the smarter the Researcher gets (via the self-learning knowledge base). Running 24/7 means you hit a critical mass of knowledge in **days, not weeks**.

---

## Platform Comparison: Always-On

| Platform | 24/7? | Setup difficulty | Cost/month | Notes |
|----------|:-----:|:-----------:|:---:|---|
| **Render.com** ✅ | Yes | ⭐ Very easy | $7 (Starter) | **Recommended — you already have an account.** Git push = deployed. Auto-restart. Env vars UI. |
| **Railway.app** ✅ | Yes | ⭐ Very easy | $5 (Hobby) | Great alternative if you want to try it. |
| **Fly.io** ✅ | Yes | ⭐⭐ Medium | ~$3–10 | More control, slightly more config |
| **Modal.com** 🔶 | On-demand burst | ⭐⭐ Medium | Pay per run | **Best for heavy backtest compute** — spin up 8 CPUs for 10 minutes, pay nothing when idle |
| **LangGraph Cloud** ✅ | Yes, managed | ⭐ Easiest | $39 | Adds visual agent trace UI (LangSmith). Worth it if you want observability. |
| **AWS Lambda / GCP Run** ❌ | Cron only, 15min limit | ⭐⭐⭐ Hard | ~$1–5 | Too limited for long-running backtest agents |

### ✅ Recommended Setup: Render.com + Modal.com (burst compute)
- **Render.com** (you already have it): hosts the orchestrator + all lightweight agents (Researcher, Pre-filter, Summariser, Learner) — runs 24/7, $7/month.
- **Modal.com** (free $30 credits to start): handles the **heavy compute** — backtest runs, Optuna hyperparameter search, ML model training. Render calls Modal when it needs compute. You pay only for actual CPU time used (~$0.10/hr), not 24/7.
- This split is important: backtests + ML can run for 10–30 minutes and need real CPU. You don't want to pay for a beefy always-on Render instance just for that.

---

## Where Each Agent Runs (Compute Architecture)

This is the most important thing to get right — running everything on one $7 server means backtests will be slow and block other agents.

```
┌──────────────────────────────────────────────────────────────┐
│              Render.com — Always-on Orchestrator              │
│              (Starter $7/month, 0.5 CPU, 512 MB RAM)         │
│                                                               │
│  Agent 1: Researcher    — API calls only, very light         │
│  Agent 2: Pre-filter    — one LLM call per strategy, light   │
│  Agent 5: Summariser    — writes text + uploads files, light │
│  Agent 6: Learner       — reads/writes DB, light             │
│  Orchestrator           — queue polling every 10min          │
└──────────────────────────────┬───────────────────────────────┘
                               │ dispatches jobs via HTTP
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              Modal.com — On-Demand Burst Compute              │
│              (spins up in ~3s, 4–8 CPUs, auto-scales)        │
│                                                               │
│  Agent 3: Implementer   — generates code + runs backtest     │
│    └─ backtesting.py    — processes years of OHLCV data      │
│    └─ Optuna (50 trials)— CPU-intensive hyperopt             │
│    └─ Walk-forward OOS  — 5× backtest runs per strategy      │
│    └─ RandomForest/XGB  — ML signal filter training          │
│                                                               │
│  Agent 4: Validator     — Monte Carlo (1000 permutations)    │
│    └─ Permutation test  — CPU-intensive statistical test     │
└──────────────────────────────────────────────────────────────┘
```

### Why this split matters
| Task | Time on 0.5 CPU | Time on 4 CPU (Modal) | Cost |
|------|:-----------:|:---:|---:|
| 1 backtest (5yr 1H data) | ~4 min | ~20 sec | ~$0.003 |
| Optuna 50 trials | ~3.5 hrs | ~10 min | ~$0.10 |
| RandomForest training | ~45 min | ~3 min | ~$0.05 |
| Monte Carlo 1000× | ~2 hrs | ~8 min | ~$0.08 |
| **Full strategy run** | **~6 hrs (blocks everything)** | **~25 min** | **~$0.25** |

With the split: Render orchestrator stays responsive 24/7, and Modal runs each strategy in ~25 min for $0.25. At 15 strategies/day: ~$3.75/day in compute.

---

## Premium Research Sources

The free sources (ArXiv, forums) are good but incomplete. Here's what serious quant researchers actually use:

### 🥇 Tier 1 — Must Have

| Source | What's inside | Cost | API? |
|--------|--------------|------|:----:|
| **Quantpedia** | 700+ curated strategies: trading rules, Sharpe, drawdown, code, related papers. Filter by asset class, Sharpe, time period. THE best structured source. | $60/month (Premium) | ✅ Full JSON API |
| **SSRN** | 1M+ finance preprints. Filter by "algorithmic trading", "FX momentum", "EUR/USD". Most are free PDFs. | Free to read | Scrape OK |
| **ArXiv q-fin** | Latest academic preprints (q-fin.TR, q-fin.ST). Often 6–12mo ahead of journals. | Free | ✅ Free RSS + API |
| **Semantic Scholar** | 200M papers with abstracts, citations. Free API key (email signup). | Free | ✅ Free API key |
| **NBER Working Papers** | Top macro/finance economists. Often cited strategies before journals. | Free | Scrape OK |

### 🥈 Tier 2 — High Value

| Source | What's inside | Cost |
|--------|--------------|------|
| **Quantpedia Pro** | Live backtestable signals, factor data, strategy alerts | $200/month |
| **Two Sigma / AQR white papers** | Free PDFs published on their own sites. Very high quality. | Free |
| **Alpha Architect blog** | Deep factor/momentum research. Mostly equities but principles transfer to FX. | Free |
| **Wilmott Magazine** | Classic quant practitioner strategies and models | $80/year |
| **Journal of Portfolio Management** | Peer-reviewed, best OOS-tested strategies | ~$40/month (or via Sci-Hub for preprints) |

### 🥉 Tier 3 — Community & Signals

| Source | What's inside | Cost |
|--------|--------------|------|
| **QuantConnect** research notebooks | Thousands of community-written strategy implementations with live results | Free |
| **GitHub awesome-quant** | Curated open-source quant libraries and strategies | Free |
| **r/algotrading** top posts | Practical community strategies, real backtest screenshots | Free |
| **Kaggle finance competitions** | ML-based trading notebooks | Free |

### 💡 Best bang for buck starting point
**Quantpedia API at $60/month** is the single highest-leverage spend. It gives your Researcher agent structured, already-curated strategies with known Sharpe ratios — you skip 80% of the noise filtering. Combined with free ArXiv + Semantic Scholar, your pipeline has both depth (academic) and breadth (curated practitioner).

---

## Architecture: 24/7 Queue-Based

```
┌─────────────────────────────────────────────────────────────────┐
│              Render.com — Orchestrator (always running)          │
│                                                                  │
│  ┌─────────────────┐    ┌──────────────────────────────────┐    │
│  │  APScheduler    │───▶│  Orchestrator + queue manager    │    │
│  │ every 4h:       │    │  budget guard + idea ingestion   │    │
│  │  research cycle │    └──────────────┬───────────────────┘    │
│  │ every 10min:    │                   │                         │
│  │  process queue  │   ┌───────────────┼──────────────────┐     │
│  └─────────────────┘   ▼               ▼                  ▼     │
│                    Agent 1          Agent 2           Agent 5+6  │
│                  Researcher        Pre-filter      Summariser+   │
│                 (API calls)      (Haiku, light)      Learner     │
│                        │               │                         │
│                        └───────────────┘                         │
│                                │ dispatches heavy jobs           │
└────────────────────────────────┼────────────────────────────────┘
                                 │ HTTP call to Modal
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│           Modal.com — On-Demand Compute (spins up in ~3s)        │
│                                                                  │
│    Agent 3: Implementer     Agent 4: Validator                   │
│    ├── Generate code        ├── Look-ahead bias scan             │
│    ├── Run backtest         ├── Monte Carlo 1000×                │
│    ├── Optuna 50 trials     └── Walk-forward OOS check           │
│    ├── Walk-forward OOS                                          │
│    └── RF/XGBoost overlay                                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
         Supabase DB        Linear.app          Cloudflare R2
       (strategies,         (Kanban board,     (reports, charts,
        knowledge_base,     visible online,     equity curves,
        spend_log,          mobile app)         backtest code)
        user_ideas)
```

### How the queue works (not just nightly cron)
1. **Every 4 hours** → Researcher queries Quantpedia API + ArXiv + Semantic Scholar → inserts new strategy `idea` rows into Supabase
2. **Always** → Queue worker also checks `user_ideas` table — if you added your own idea, it gets picked up immediately, no wait
3. **Every 10 minutes** → Queue worker picks up `idea` rows → Pre-filter scores them → moves good ones to `filtered`
4. **On demand (Modal)** → Implementer picks up `filtered` rows → generates code → dispatches to Modal for backtest + ML
5. **On demand (Modal)** → Validator picks up passing strategies → dispatches Monte Carlo + OOS check to Modal
6. **On each Modal completion** → Summariser writes report to R2, updates Linear ticket
7. **After each cycle** → Learner updates knowledge base so next Researcher call is smarter
8. **Always** → Budget guard checks daily spend before every LLM call; pauses if limit exceeded

---

## Step-by-Step Setup (~1 Hour Total)

### Step 1 — Accounts (20 min)

1. **GitHub** → create private repo `trading-agents`. This is your deploy source — Render watches it.
2. **Render.com** → you already have an account → "New +" → "Background Worker" → connect GitHub → select `trading-agents` → Starter plan ($7/month). Done. It auto-deploys every push.
3. **Modal.com** → `modal.com` → sign up (free, $30 credits included) → note: no deploy step needed, Render calls Modal via its Python SDK at runtime.
4. **Anthropic Console** → `console.anthropic.com` → create API key → set a billing alert at **$100** as safety net (Settings → Billing → Alerts).
5. **Supabase** → `supabase.com` → new project → copy `Project URL` and `anon key`.
6. **Cloudflare R2** → `dash.cloudflare.com` → R2 → create bucket `trading-research` → API tokens → create token with R2 Read+Write permissions.
7. **Linear.app** → `linear.app` → free workspace → new project "Trading Research" → add workflow states: `Idea | Filtered | Backtesting | Validating | Done | Failed` → Settings → API → create personal API key, note your Team ID.
8. **Quantpedia** → `quantpedia.com` → Premium plan ($60/month) → API key. *(Optional but strongly recommended)*
9. **Semantic Scholar** → `api.semanticscholar.org/graph/v1` → request free API key (just email form).

### Step 2 — Render Environment Variables (5 min)

In Render dashboard → your Background Worker → **Environment** tab, add all of these (pure UI, no terminal):

```
ANTHROPIC_API_KEY        = sk-ant-...
SUPABASE_URL             = https://xxxx.supabase.co
SUPABASE_ANON_KEY        = eyJ...
R2_ACCOUNT_ID            = ...
R2_ACCESS_KEY_ID         = ...
R2_SECRET_ACCESS_KEY     = ...
R2_BUCKET_NAME           = trading-research
LINEAR_API_KEY           = lin_api_...
LINEAR_TEAM_ID           = ...
MODAL_TOKEN_ID           = ak-...        # from modal.com → Settings → API Tokens
MODAL_TOKEN_SECRET       = as-...
QUANTPEDIA_API_KEY       = ...           # optional
SEMANTIC_SCHOLAR_API_KEY = ...           # free
MAX_DAILY_SPEND_USD      = 8.0           # hard budget cap
BACKTEST_MIN_SHARPE      = 0.8
RESEARCH_INTERVAL_HOURS  = 4
```

No secrets ever touch your laptop code or git history.

### Step 3 — Supabase Schema (10 min)

In Supabase → SQL Editor → run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE strategies (
  id               uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  name             text NOT NULL,
  source           text,          -- 'quantpedia' | 'arxiv' | 'ssrn' | 'semantic_scholar' | 'user'
  source_url       text,
  hypothesis       text,
  entry_logic      text,
  exit_logic       text,
  indicators       jsonb,
  timeframes       text[],
  status           text DEFAULT 'idea',
  -- flow: idea → filtered → implementing → implemented → validating → done | failed
  pre_filter_score float,
  backtest_sharpe  float,
  backtest_calmar  float,
  max_drawdown     float,
  oos_sharpe       float,
  leakage_score    float,          -- 0 (bad) to 10 (clean)
  hyperparams      jsonb,
  backtest_code    text,
  report_url       text,
  equity_curve_url text,
  linear_issue_id  text,
  retry_count      int DEFAULT 0,
  error_log        text,
  created_at       timestamptz DEFAULT now(),
  updated_at       timestamptz DEFAULT now()
);

-- ✏️ User-submitted strategy ideas (you fill this in plain English)
CREATE TABLE user_ideas (
  id          uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  title       text NOT NULL,           -- short name, e.g. "RSI + Volume spike on 4H"
  description text NOT NULL,           -- plain English, as detailed or vague as you want
  notes       text,                    -- optional: known edge cases, inspiration, papers
  priority    int DEFAULT 5,           -- 1 (highest) to 10 (lowest) — pipeline processes higher first
  status      text DEFAULT 'pending',  -- pending | picked_up | done
  strategy_id uuid REFERENCES strategies(id),  -- set once pipeline creates the strategy
  created_at  timestamptz DEFAULT now()
);

CREATE TABLE knowledge_base (
  id          uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  category    text,    -- 'works' | 'fails' | 'partial'
  indicator   text,
  timeframe   text,
  summary     text,
  embedding   vector(1536),   -- for semantic dedup
  created_at  timestamptz DEFAULT now()
);

CREATE TABLE spend_log (
  id            uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  date          date DEFAULT CURRENT_DATE,
  agent         text,
  model         text,
  input_tokens  int,
  output_tokens int,
  cost_usd      float,
  created_at    timestamptz DEFAULT now()
);

-- Fast queue polling
CREATE INDEX strategies_status_idx  ON strategies(status);
CREATE INDEX strategies_sharpe_idx  ON strategies(backtest_sharpe DESC NULLS LAST);
CREATE INDEX spend_log_date_idx     ON spend_log(date);
CREATE INDEX user_ideas_status_idx  ON user_ideas(status, priority);
```

### Step 4 — How to Submit Your Own Strategy Ideas

You don't need to write code. Just insert a row into `user_ideas` in Supabase with plain English — the pipeline picks it up within 10 minutes.

**Option A — Supabase Table Editor (simplest, no code):**
Go to Supabase → Table Editor → `user_ideas` → "Insert row":
```
title:       "4H momentum + session open breakout"
description: "Buy when price breaks above the London open high with RSI > 55 and ATR 
              expanding. Exit at 2× ATR profit target or end of NY session. 
              I think this works because London breakouts tend to continue into NY."
notes:       "Maybe add a volatility filter — only trade if ATR > 20-day average ATR"
priority:    2
```
Done. The orchestrator checks this table every 10 minutes and processes `pending` rows immediately, bypassing the 4-hour research cycle — your ideas get priority.

**Option B — Tiny Python script (run once from anywhere):**
```python
# submit_idea.py — run this anywhere, even your laptop
import os
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
sb.table("user_ideas").insert({
    "title": "RSI divergence + VWAP bounce on 1H",
    "description": """
        Look for bullish RSI divergence (price makes lower low, RSI makes higher low) 
        on the 1H chart while price is within 0.1% of VWAP. 
        Entry: next candle open after divergence confirmed.
        Exit: RSI > 70 or price -1.5× ATR from entry.
        Hypothesis: VWAP acts as a magnet, divergence shows exhaustion of sellers.
    """,
    "notes": "Try on both EUR/USD and GBP/USD. Check if works better in NY session.",
    "priority": 1
}).execute()
print("Idea submitted!")
```

**What happens next (automatic):**
1. Within 10 min → Pre-filter agent scores your idea (it gets a +2 "novelty" bonus since it's user-submitted)
2. If score ≥ 6 → Implementer agent translates your plain-English description into a `backtesting.py` strategy
3. Dispatched to Modal → runs Optuna tuning + walk-forward OOS validation
4. Validator checks for look-ahead bias
5. Summariser creates a Linear ticket with full results + equity curve
6. You get notified (Linear email/Slack) with the results

**How detailed should your description be?**  
Any level of detail works. The agent interprets your intent:
- Vague: `"something with Bollinger Bands and volume"` → agent picks reasonable parameters, tests multiple variants
- Detailed: full entry/exit rules → agent implements exactly what you described, then also explores variants
- With paper: `"implement the strategy from https://arxiv.org/..."` → agent reads the paper and implements it

---

### Step 5 — Project Structure

```
trading-agents/                  ← GitHub repo root
├── main.py                      ← entry point: starts scheduler + queue workers
├── orchestrator.py              ← queue management, pipeline logic
├── agents/
│   ├── researcher.py            ← Agent 1: Quantpedia + ArXiv + Semantic Scholar
│   ├── prefilter.py             ← Agent 2: Claude Haiku scoring rubric
│   ├── implementer.py           ← Agent 3 (light): generates code, dispatches to Modal
│   ├── implementer_modal.py     ← Agent 3 (heavy): Modal function — backtest + Optuna + ML
│   ├── validator.py             ← Agent 4 (light): dispatches to Modal
│   ├── validator_modal.py       ← Agent 4 (heavy): Modal function — Monte Carlo + OOS
│   ├── summariser.py            ← Agent 5: Markdown reports + Linear tickets
│   └── learner.py               ← Agent 6: knowledge base updates
├── shared/
│   ├── llm.py                   ← Anthropic client + cost tracking
│   ├── db.py                    ← Supabase wrapper (strategies + user_ideas)
│   ├── storage.py               ← Cloudflare R2 wrapper
│   ├── board.py                 ← Linear.app GraphQL API
│   └── budget.py                ← Hard daily spend cap enforcement
├── prompts/
│   ├── researcher_system.txt
│   ├── prefilter_system.txt
│   ├── implementer_system.txt   ← includes user idea translation instructions
│   ├── validator_system.txt
│   └── summariser_system.txt
├── requirements.txt
└── render.yaml                  ← Render config (optional, can configure in UI instead)
```

### Step 6 — Core Files

#### `render.yaml` (optional — can also configure via Render UI)
```yaml
services:
  - type: worker
    name: trading-agents-orchestrator
    env: python
    plan: starter        # $7/month, 0.5 CPU, 512 MB RAM — enough for light agents
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py
    autoDeploy: true     # redeploys on every GitHub push
```

#### `requirements.txt`
```
anthropic>=0.25
supabase>=2.4
boto3>=1.34
pandas>=2.2
numpy>=1.26
backtesting>=0.3.3
optuna>=3.6
scikit-learn>=1.4
xgboost>=2.0
lightgbm>=4.3
ta>=0.11
apscheduler>=3.10
requests>=2.31
feedparser>=6.0
matplotlib>=3.8
modal>=0.62          # Modal SDK — Render calls Modal at runtime
```

#### `agents/implementer_modal.py` — Heavy Compute on Modal
```python
# This file runs ON Modal (4 CPUs, not on your $7 Render box)
import modal

app   = modal.App("trading-implementer")
image = modal.Image.debian_slim().pip_install(
    "backtesting", "optuna", "scikit-learn", "xgboost",
    "lightgbm", "pandas", "numpy", "ta", "matplotlib", "anthropic"
)

@app.function(image=image, cpu=4, timeout=1800,   # 4 CPUs, 30min max
              secrets=[modal.Secret.from_name("trading-secrets")])
def run_backtest_and_optimize(strategy: dict) -> dict:
    """
    Runs entirely on Modal's 4-CPU machines.
    Called remotely by Render orchestrator via: run_backtest_and_optimize.remote(strategy)
    """
    import json, os
    from backtesting import Backtest, Strategy
    import optuna, pandas as pd
    from shared.llm import claude

    # 1. Ask Claude Sonnet to write the backtesting.py Strategy class
    code_prompt = f"""Write a complete backtesting.py Strategy subclass for this strategy:

Name: {strategy['name']}
Hypothesis: {strategy['hypothesis']}
Entry logic: {strategy['entry_logic']}
Exit logic: {strategy['exit_logic']}
Indicators: {strategy['indicators']}
Timeframe: {strategy['timeframes'][0] if strategy.get('timeframes') else '1H'}

{'USER NOTES: ' + strategy.get('user_notes', '') if strategy.get('source') == 'user' else ''}

Requirements:
- Use ONLY the backtesting.py API (no future data access)
- Apply commission=0.0001 (1 pip spread equivalent)  
- No fractional lots (size must be integer)
- All indicator values must use .shift(1) before use as signals
- Include clearly named hyperparameter variables at class level
- Return the complete Python code, nothing else."""

    code = claude(code_prompt, model="claude-sonnet-4-5", max_tokens=3000)

    # 2. Execute the generated code
    exec_globals = {}
    exec(code, exec_globals)
    StrategyClass = [v for v in exec_globals.values()
                     if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy][0]

    # 3. Load your EUR/USD data (fetched from R2 or rebuilt)
    data = _load_eurusd_data(strategy.get("timeframes", ["1H"])[0])

    split = int(len(data) * 0.8)
    train_data, oos_data = data.iloc[:split], data.iloc[split:]

    # 4. Optuna hyperparameter search (50 trials on training data)
    def objective(trial):
        params = _suggest_params(trial, StrategyClass)
        bt = Backtest(train_data, StrategyClass, commission=0.0001, cash=10000)
        stats = bt.run(**params)
        return stats["Sharpe Ratio"] if stats["# Trades"] > 10 else -999

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, n_jobs=4)   # parallel on 4 CPUs
    best_params = study.best_params

    # 5. Walk-forward OOS validation (5 folds)
    oos_sharpes = _walk_forward(data, StrategyClass, best_params, n_folds=5)

    # 6. Final OOS run with best params
    bt_oos = Backtest(oos_data, StrategyClass, commission=0.0001, cash=10000)
    oos_stats = bt_oos.run(**best_params)

    # 7. Optional: RandomForest signal filter
    rf_sharpe = _try_rf_filter(train_data, oos_data, StrategyClass, best_params)

    return {
        "backtest_code":    code,
        "hyperparams":      best_params,
        "backtest_sharpe":  oos_stats["Sharpe Ratio"],
        "backtest_calmar":  oos_stats["Calmar Ratio"],
        "max_drawdown":     oos_stats["Max. Drawdown [%]"],
        "oos_sharpe":       float(pd.Series(oos_sharpes).mean()),
        "rf_sharpe":        rf_sharpe,
        "equity_curve":     oos_stats["_equity_curve"].to_csv(),
    }
```

#### `agents/validator_modal.py` — Monte Carlo on Modal
```python
import modal

app   = modal.App("trading-validator")
image = modal.Image.debian_slim().pip_install(
    "backtesting", "numpy", "pandas", "scipy", "anthropic"
)

@app.function(image=image, cpu=4, timeout=900,
              secrets=[modal.Secret.from_name("trading-secrets")])
def run_validation(strategy: dict) -> dict:
    """
    Monte Carlo permutation test + look-ahead bias audit.
    Runs on Modal 4-CPU. Called remotely from Render.
    """
    import numpy as np
    from backtesting import Backtest
    from shared.llm import claude

    # 1. LLM look-ahead bias audit (fast)
    audit_prompt = f"""Review this backtesting.py Strategy code for look-ahead bias:

{strategy['backtest_code']}

Check for:
1. Any iloc[i+1] or shift(-N) — future bar access
2. Indicators calculated on the full series before the loop
3. .fillna() or .ffill() that could propagate future values
4. Any pandas operations that use future data

For each issue found, give the line and explain why it's a leak.
Score the code 0–10 for cleanliness (10 = no leakage found).
Return JSON: {{"leakage_score": int, "issues": [str], "verdict": str}}"""

    audit = claude(audit_prompt, model="claude-sonnet-4-5", max_tokens=1000)
    audit_result = json.loads(audit)

    # 2. Monte Carlo permutation test (1000 shuffles, parallel)
    data    = _load_eurusd_data(strategy["timeframes"][0])
    oos     = data.iloc[int(len(data) * 0.8):]
    params  = strategy["hyperparams"]

    real_sharpe = strategy["backtest_sharpe"]
    null_sharpes = _permutation_test(oos, strategy["backtest_code"], params, n=1000)

    p_value = float(np.mean(np.array(null_sharpes) >= real_sharpe))

    # Combine: both LLM audit AND statistical test must pass
    passes = audit_result["leakage_score"] >= 7 and p_value < 0.05

    return {
        "leakage_score":   audit_result["leakage_score"],
        "leakage_issues":  audit_result["issues"],
        "mc_p_value":      p_value,
        "mc_null_sharpes": null_sharpes[:20],   # sample for reporting
        "passes_validation": passes,
    }
```

#### `orchestrator.py` — Updated Queue with User Ideas + Modal dispatch
```python
from shared.db import get_pending_strategies, get_pending_user_ideas, update_strategy, mark_user_idea_picked_up
from shared.budget import check_budget
from agents.researcher import run as research
from agents.prefilter import score as prefilter
from agents.implementer_modal import run_backtest_and_optimize
from agents.validator_modal import run_validation
from agents.summariser import run as summarise
from agents.learner import update_knowledge_base
import logging

log = logging.getLogger(__name__)

def research_cycle():
    if not check_budget():
        return
    try:
        ideas = research()
        _score_and_queue(ideas)
    except Exception as e:
        log.error(f"Research cycle failed: {e}", exc_info=True)

def process_queue():
    if not check_budget():
        return

    # ⭐ User ideas get priority — picked up before researcher ideas
    for user_idea in get_pending_user_ideas(limit=3):
        strategy = _user_idea_to_strategy(user_idea)   # creates strategy row with source='user'
        mark_user_idea_picked_up(user_idea["id"], strategy["id"])
        score = prefilter(strategy, bonus=2)             # +2 bonus for user-submitted
        if score >= 6.0:
            update_strategy(strategy["id"], status="filtered", pre_filter_score=score)

    # Then researcher-sourced filtered strategies
    for strategy in get_pending_strategies(status="filtered", limit=2):
        try:
            # Dispatch to Modal — non-blocking, Render stays light
            result = run_backtest_and_optimize.remote(strategy)   # Modal call
            sharpe = result.get("sharpe", 0) if result else 0
            if sharpe >= float(os.environ.get("BACKTEST_MIN_SHARPE", "0.8")):
                update_strategy(strategy["id"], status="implemented", **result)
                log.info(f"✅ {strategy['name']} | Sharpe={sharpe:.2f}")
            else:
                update_strategy(strategy["id"], status="failed",
                                error_log=f"Sharpe too low: {sharpe:.2f}")
        except Exception as e:
            update_strategy(strategy["id"], status="failed", error_log=str(e))

    for strategy in get_pending_strategies(status="implemented", limit=2):
        try:
            validated = run_validation.remote(strategy)           # Modal call
            if validated["passes_validation"]:
                summarise({**strategy, **validated})
                update_strategy(strategy["id"], status="done", **validated)
                log.info(f"🏆 DONE: {strategy['name']} | p={validated['mc_p_value']:.3f}")
            else:
                update_strategy(strategy["id"], status="failed",
                                error_log=f"Leakage={validated['leakage_score']}, p={validated['mc_p_value']:.3f}")
        except Exception as e:
            update_strategy(strategy["id"], status="failed", error_log=str(e))

    update_knowledge_base()
```

#### `agents/researcher.py` — Multi-Source Research Agent
```python
import os, json, requests, feedparser
from shared.llm import claude
from shared.db import insert_strategy, get_knowledge_base_summary

QUANTPEDIA_KEY    = os.environ.get("QUANTPEDIA_API_KEY")
SEMANTIC_KEY      = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

def fetch_quantpedia() -> list[dict]:
    """Curated strategies with trading rules + known Sharpe. Best source."""
    if not QUANTPEDIA_KEY:
        return []
    r = requests.get(
        "https://quantpedia.com/api/strategies/",
        params={"asset_class": "currencies", "api_key": QUANTPEDIA_KEY, "page_size": 20},
        timeout=30
    )
    return r.json().get("results", []) if r.ok else []

def fetch_arxiv() -> list[str]:
    """Latest q-fin.TR + q-fin.ST preprints."""
    feed = feedparser.parse("https://export.arxiv.org/rss/q-fin.TR+q-fin.ST")
    return [f"{e.title}: {e.summary[:500]}" for e in feed.entries[:15]]

def fetch_semantic_scholar(query: str) -> list[dict]:
    """Free API — 200M papers."""
    headers = {"x-api-key": SEMANTIC_KEY} if SEMANTIC_KEY else {}
    r = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers=headers,
        params={"query": query,
                "fields": "title,abstract,year,externalIds",
                "limit": 10,
                "publicationTypes": "JournalArticle"},
        timeout=30
    )
    return r.json().get("data", []) if r.ok else []

def run() -> list[dict]:
    kb_summary   = get_knowledge_base_summary()
    quantpedia   = fetch_quantpedia()
    arxiv        = fetch_arxiv()
    papers_fx    = fetch_semantic_scholar("EUR USD FX algorithmic trading strategy momentum 2022 2023 2024")
    papers_ml    = fetch_semantic_scholar("machine learning forex regime switching strategy out-of-sample")

    prompt = f"""You are a quantitative finance researcher specialising in FX algorithmic trading.

KNOWLEDGE BASE — already tried, do NOT repeat these:
{kb_summary}

QUANTPEDIA STRATEGIES (curated, with known backtests):
{json.dumps([{{"name": s.get("name"), "description": s.get("description","")[:300],
               "sharpe": s.get("sharpe_ratio"), "asset": s.get("asset_class")}}
              for s in quantpedia[:10]], indent=2)}

ARXIV PREPRINTS (latest):
{chr(10).join(arxiv)}

ACADEMIC PAPERS (Semantic Scholar):
{json.dumps([{{"title": p.get("title"), "abstract": (p.get("abstract") or "")[:300]}}
              for p in papers_fx + papers_ml], indent=2)}

Extract the 5 most promising strategies NOT already in the knowledge base.
Focus on: EUR/USD, timeframes 1H–1D, Sharpe > 0.8 in published results.
Prefer: regime-switching, ML-enhanced, microstructure, carry + momentum combos.

Return ONLY a valid JSON array. Each object must have:
- name (string)
- hypothesis (string, 1–2 sentences)
- entry_logic (string, plain English step-by-step)
- exit_logic (string)
- indicators (array of strings)
- timeframes (array, e.g. ["4H", "1D"])
- source_url (string)
- estimated_sharpe (float or null)
- source (string: "quantpedia" | "arxiv" | "semantic_scholar")
- why_promising (string, 1 sentence)"""

    response = claude(prompt, model="claude-sonnet-4-5", max_tokens=4000)
    ideas = json.loads(response)
    for idea in ideas:
        insert_strategy(idea)
    return ideas
```

#### `shared/board.py` — Linear Kanban API
```python
import os, requests

LINEAR_URL = "https://api.linear.app/graphql"

def _headers():
    return {"Authorization": os.environ["LINEAR_API_KEY"],
            "Content-Type": "application/json"}

def create_issue(title: str, description: str) -> str:
    """Creates a Linear ticket, returns issue ID."""
    mutation = """
    mutation($title: String!, $desc: String!, $teamId: String!) {
      issueCreate(input: {title: $title, description: $desc, teamId: $teamId}) {
        issue { id url }
      }
    }"""
    r = requests.post(LINEAR_URL,
                      json={"query": mutation,
                            "variables": {"title": title, "desc": description,
                                          "teamId": os.environ["LINEAR_TEAM_ID"]}},
                      headers=_headers(), timeout=15)
    return r.json()["data"]["issueCreate"]["issue"]["id"]

def move_issue(issue_id: str, state_name: str):
    """Moves ticket to a column by workflow state name."""
    # 1. Query workflow state ID by name
    # 2. Call issueUpdate with stateId
    # (full implementation: ~15 lines of standard GraphQL calls)
    pass
```

### Step 7 — Deploy (5 min, one time only)

```bash
# On your laptop — just once:
git clone https://github.com/YOUR_USERNAME/trading-agents.git
cd trading-agents
# add all files from structure above
git add . && git commit -m "Initial 24/7 agent pipeline"
git push origin main
```

Render auto-detects the push, builds the image, starts `python main.py`. **Done. Your laptop is out of the loop forever.**

Every future change: edit any file in the GitHub web editor → commit → Render auto-redeploys in ~2 minutes. No terminal needed ever again.

---

## Agent Prompts Summary

### Agent 2 — Pre-filter Scoring Rubric (Claude Haiku, cheap)
Score 0–10 on 5 dimensions (2 pts each):
- **Realism**: executable on OANDA/IB with typical EUR/USD spreads (0.5–1.5 pip)?
- **Data availability**: requires only OHLCV + standard indicators? (no order book, no alt data)
- **Novelty**: not already in knowledge base?
- **Published OOS evidence**: peer-reviewed or credible practitioner backtest?
- **Simplicity**: ≤ 5 parameters? (low overfitting risk)

Threshold: **≥ 6/10** passes to implementation.

### Agent 3 — Implementer (Claude Sonnet)
Generates in one pass:
1. `backtesting.py` Strategy subclass — realistic: spread applied, no fractional lots
2. Optuna hyperparameter search (50 trials, maximise Sharpe on in-sample period)
3. Walk-forward OOS validation (5 anchored folds, last 20% of data = final OOS)
4. Optional: Random Forest signal filter overlay (`sklearn`)
5. Output: equity curve PNG (→ R2), metrics dict (→ Supabase)

### Agent 4 — Validator Checklist (Claude Sonnet)
Checks each strategy implementation for:
- [ ] Any `iloc[i+1]`, `.shift(-n)`, or future bar reference in `calculate()`?
- [ ] All indicator values `.shift(1)` before use as signals?
- [ ] `commission` and `margin` set to realistic OANDA values?
- [ ] Monte Carlo: shuffle signal randomly 1000× — does Sharpe drop to noise?
- [ ] Walk-forward OOS Sharpe within 50% of in-sample Sharpe? (>50% drop = overfit flag)

### Agent 5 — Summariser Output (Claude Haiku)
Per strategy, writes to R2:
```
reports/
  {strategy_name}/
    report.md          ← full human-readable analysis
    equity_curve.png   ← equity curve chart
    metrics.json       ← all numbers
```
Creates/updates Linear ticket with metrics table in description.  
Sends Slack/email notification (via Linear) if Sharpe > 1.0.

### Agent 6 — Learner (runs after every queue cycle)
Reads all `done` and `failed` strategies from Supabase.
Writes compact entries to `knowledge_base` table, e.g.:
- ✅ `"RSI(14) divergence on 4H + ATR(2) trailing stop → Sharpe 1.31, works in trending regimes (ADX > 25)"`
- ❌ `"MACD(12,26,9) crossover on 1H → Sharpe -0.21, overtrades in consolidation — avoid"`
- ⚠️ `"BB squeeze on 4H → Sharpe 0.61, only works during low-volatility periods"`

This summary is injected into the Researcher system prompt each cycle → progressively smarter targeting.

---

## Cost Estimates

### 🟢 Budget Mode — ~$25/month
`MAX_DAILY_SPEND_USD=0.80`, Claude Haiku for all agents except Implementer, ~1 research cycle/day, ~5 strategies/day. Still 24/7, still self-improving.

| Item | Monthly |
|------|---------|
| Claude Haiku (most tasks) | ~$5 |
| Claude Sonnet (Implementer only) | ~$8 |
| Modal compute (~5 strategies/day × $0.25 × 30 days) | ~$37 |
| Render.com Starter | $7 |
| Supabase, R2, Linear | $0 (all free tiers) |
| **Total** | **~$57** |

> 💡 To get to true ~$25: set `MAX_DAILY_SPEND_USD=0.50` and `MAX_STRATEGIES_PER_RUN=2` — runs 2 strategies/day, Modal cost drops to ~$15/month.

### 🟡 Balanced Mode — ~$120–180/month ← recommended starting point
`MAX_DAILY_SPEND_USD=3.00`, Sonnet for research + implement + validate, Haiku for filter + summarise. ~3 cycles/day, ~15 strategies/day.

| Item | Monthly |
|------|---------|
| Claude Sonnet (heavy agents) | ~$40 |
| Claude Haiku (light agents) | ~$8 |
| Modal compute (~15 strategies/day × $0.25 × 30) | ~$112 |
| Render.com Starter | $7 |
| Quantpedia API *(optional but recommended)* | $60 |
| Supabase, R2, Linear | $0 |
| **Total with Quantpedia** | **~$227** |
| **Total without Quantpedia** | **~$167** |

> 💡 Reduce Modal costs by capping `n_trials=20` in Optuna (instead of 50) and `n=200` in Monte Carlo (instead of 1000). Drops compute per strategy from $0.25 to ~$0.08.

### 🔴 Power Mode — ~$350+/month
`MAX_DAILY_SPEND_USD=12.0`, Sonnet for everything, Opus for final validation, 30+ strategies/day.

---

## Progress Board: Linear.app

Your board has 6 columns. Cards move automatically as the pipeline runs:

```
💡 Idea  →  ✅ Filtered  →  ⚗️ Backtesting  →  🔍 Validating  →  🏆 Done  →  ❌ Failed
```

Each card is auto-filled by the Summariser agent with:
- Strategy name + link to source paper / Quantpedia page
- Sharpe (in-sample + OOS), Calmar ratio, max drawdown
- Leakage score (0–10)
- Direct link to full Markdown report + equity curve image in R2
- Date discovered, date completed

**Mobile**: Linear has iOS + Android apps. Wake up → check app → see what moved to 🏆 Done.  
**Notifications**: configure Slack or email alert in Linear for any card → Done column.

*Alternative free board*: **GitHub Projects** works identically if you prefer — free, has API, already have GitHub for deploys.

---

## Self-Learning Loop

```
Day 1 (cycles 1–6):
  Researcher: 10 ideas from Quantpedia + ArXiv
  4 pass pre-filter → implement → 1 has Sharpe > 0.8 + leakage ≥ 7
  Learner writes to knowledge_base:
    ✅ "4H RSI divergence + ATR trailing stop → Sharpe 1.2"
    ❌ "1H MACD cross → overtrades, Sharpe -0.3"
    ⚠️  "BB squeeze → 0.6 Sharpe, only low-vol regimes"

Day 3 (cycles 15–20):
  Researcher reads KB → prioritises RSI variants & regime filters
  Pre-filter auto-boosts RSI-based ideas, penalises MACD
  Now 3–4 ideas pass per cycle instead of 1

Week 2:
  KB has 50+ entries. Researcher targets very specific hypothesis space.
  Most ideas score ≥ 7. Pipeline efficiency is 4× higher than Day 1.
  Profitable strategies start clustering around a theme → insights emerge.
```

---

## Guardrails Summary

```python
MAX_DAILY_SPEND_USD    = 8.0    # hard stop checked before every LLM call
MAX_STRATEGIES_PER_RUN = 5      # researcher cap per cycle
BACKTEST_MIN_SHARPE    = 0.8    # don't pass below this to validator
LEAKAGE_MIN_SCORE      = 7      # don't mark "done" below this
MAX_RETRY_COUNT        = 2      # don't retry failing strategies more than twice
```

Also: set a **$100 billing alert** in Anthropic Console as a second safety net (Settings → Billing → Alerts).

---

## Quick-Start Checklist

- [ ] Create private GitHub repo `trading-agents`
- [ ] **Render.com** → Background Worker → connect repo → Starter plan ($7/month)
- [ ] **Modal.com** → sign up (free) → get Token ID + Secret → add to Render env vars
- [ ] Get Anthropic API key → set $100 billing alert in Anthropic Console
- [ ] Create Supabase project → run SQL schema from Step 3 above
- [ ] Create Cloudflare R2 bucket `trading-research`
- [ ] Create Linear workspace → "Trading Research" project → 6 workflow states
- [ ] *(Optional but recommended)* Subscribe Quantpedia API — $60/month
- [ ] Get free Semantic Scholar API key (email form on their site)
- [ ] Add all env vars to Render Environment tab — no terminal needed, pure UI
- [ ] Create all project files → push to GitHub
- [ ] Render auto-deploys → pipeline is live 24/7 ✅
- [ ] Open Supabase Table Editor → drop your first idea into `user_ideas`
- [ ] Check Linear app 30 minutes later to see your idea being processed ☕

**Total setup time: ~1 hour.**  
**Total ongoing effort: check Linear board whenever you feel like it.**  
**Everything else: fully autonomous, 24/7.**

---

## FAQ

**Q: What if an agent crashes mid-run?**  
A: Render auto-restarts the worker service. APScheduler re-runs missed jobs. All state is in Supabase — nothing is lost. Incomplete `implementing` rows get retried on the next queue cycle (up to `MAX_RETRY_COUNT=2`).

**Q: What if a Modal function times out?**  
A: Modal has a 30-min timeout configured. If a backtest exceeds that (unlikely), the strategy gets marked `failed` with the error logged. You can increase the timeout or reduce Optuna trials.

**Q: Can I see what each agent is actually thinking?**  
A: Two options: (1) Free — add LangSmith tracing (wrap each `claude()` call, free tier 1K traces/month). (2) Paid — convert to LangGraph nodes, host on LangGraph Cloud ($39/month) for a full live visual trace graph.

**Q: How do I submit my own strategy idea?**  
A: Simplest: open Supabase Table Editor in your browser → `user_ideas` table → Insert row → fill in `title` + `description` in plain English. Done. The pipeline picks it up within 10 minutes. See Step 4 for details and examples.

**Q: How do I change a prompt or threshold?**  
A: Edit the file in the GitHub web editor → commit → Render auto-redeploys in ~2 minutes. Zero local setup ever.

**Q: What if Quantpedia is too expensive initially?**  
A: Skip it — ArXiv + Semantic Scholar alone provide hundreds of strategy ideas. Estimated cost without Quantpedia: ~$57/month budget mode. Add Quantpedia after week 2 once you see the pipeline is working.

**Q: How many profitable strategies can I expect?**  
A: Budget mode (~5 strategies/day): 2–5 passing validation per week. Balanced mode (~15/day): 5–15/week. The self-learning loop means week 4 is significantly more efficient than week 1. After 1 month you'll have 15–50 validated, leak-checked strategies to review.
