-- =============================================================================
-- Trading Research — complete database schema
-- =============================================================================
-- Idempotent: safe to run on a fresh Supabase project OR on top of an
-- existing database (uses CREATE TABLE IF NOT EXISTS + ADD COLUMN IF NOT EXISTS).
--
-- To redeploy from scratch:
--   1. Open Supabase SQL editor
--   2. Paste and run this entire file
--
-- Tables:
--   strategies       — core pipeline records
--   user_ideas       — ideas submitted via /ideas form
--   generated_ideas  — ideas auto-generated from arXiv / Semantic Scholar
--   knowledge_base   — insights extracted by Learner agent
--   spend_log        — per-call LLM cost tracking
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Utility trigger function: auto-update updated_at on every row write
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION _set_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;


-- =============================================================================
-- Table: strategies
-- =============================================================================

CREATE TABLE IF NOT EXISTS strategies (
  id                    uuid         PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Identity
  name                  text         NOT NULL DEFAULT 'Untitled Strategy',
  source                text         NOT NULL DEFAULT 'user',   -- 'user' | 'researcher'

  -- Pipeline state
  -- idea → filtered → implementing → implemented → backtesting → validating → live | failed
  status                text         NOT NULL DEFAULT 'idea',

  -- Strategy content (entry_logic = original user text, never changed)
  hypothesis            text,
  entry_logic           text,
  exit_logic            text,

  -- Pre-filter output
  pre_filter_score      float8,
  pre_filter_notes      jsonb,        -- {verdict, score_breakdown, suggested_modifications, ...}

  -- Generated code & parameter space
  backtest_code         text,         -- Python Strategy subclass (backtesting.py)
  indicators            jsonb,        -- {strategy_class, indicators_used[], symbols[], timeframes[]}
  hyperparams           jsonb,        -- param_space during optimisation; best params after backtest
  best_hyperparams      jsonb,
  best_session_hours    text,
  risk_params           jsonb,        -- {max_daily_losses, sl_atr, tp_atr}

  -- Leakage detection
  leakage_score         float8,       -- 0–10 (10 = clean)
  leakage_issues        jsonb,        -- list of issue strings

  -- In-sample backtest results
  backtest_sharpe       float8,
  backtest_calmar       float8,
  max_drawdown          float8,
  total_signals         int,
  signals_per_year      float8,
  win_rate              float8,
  profit_factor         float8,
  avg_trade_pnl         float8,

  -- Out-of-sample results (2026+)
  oos_sharpe            float8,
  oos_win_rate          float8,
  oos_total_trades      int,

  -- Statistical validation
  walk_forward_scores   jsonb,        -- list[float] — OOS Sharpe per WF fold
  monte_carlo_pvalue    float8,

  -- Modal compute tracking
  modal_job_id          text,

  -- Retry / auto-fix counters
  retry_count           int          DEFAULT 0,
  auto_fix_count        int          DEFAULT 0,  -- incremented by code_fixer agent
  validator_corrections int          DEFAULT 0,  -- incremented by validator correction loop

  -- Report artefacts
  report_url            text,         -- Cloudflare R2 URL
  equity_curve_url      text,
  report_text           text,         -- markdown content (stored inline for dashboard)

  -- UX / metadata
  comments              jsonb        DEFAULT '[]'::jsonb,  -- [{text: str, ts: str}]
  tags                  jsonb        DEFAULT '[]'::jsonb,
  error_log             text,

  -- Misc fields used by Learner / Summariser
  timeframes            jsonb,
  symbol                text,
  indicators_used       jsonb,

  -- Legacy / compatibility
  source_url            text,
  linear_issue_id       text,

  created_at            timestamptz  NOT NULL DEFAULT now(),
  updated_at            timestamptz  NOT NULL DEFAULT now()
);

-- Add any columns that may be missing in an existing deployment
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS modal_job_id          text;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS auto_fix_count         int  DEFAULT 0;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS validator_corrections  int  DEFAULT 0;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS retry_count            int  DEFAULT 0;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS comments               jsonb DEFAULT '[]'::jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS tags                   jsonb DEFAULT '[]'::jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS best_hyperparams       jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS best_session_hours     text;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS risk_params            jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS timeframes             jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS symbol                 text;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS indicators_used        jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS avg_trade_pnl          float8;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS oos_win_rate           float8;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS oos_total_trades       int;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS walk_forward_scores    jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS monte_carlo_pvalue     float8;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS report_text            text;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS equity_curve_url       text;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS exit_logic             text;

-- Quick test results (raw run with default params, no optimization)
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS quick_test_sharpe              float8;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS quick_test_calmar              float8;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS quick_test_drawdown            float8;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS quick_test_trades              int;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS quick_test_win_rate            float8;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS quick_test_signals_per_year    float8;
-- Multi-symbol × multi-timeframe quick test: best combo + full results grid
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS best_timeframe                 text;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS best_symbol                    text;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS quick_test_all_timeframes      jsonb;  -- best symbol's per-TF dict
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS quick_test_all_symbols         jsonb;  -- {symbol: {tf: metrics}}
-- Trade-level data from best-TF quick test (for strategy_analyzer)
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS quick_test_trade_records       jsonb;
-- Analysis findings from strategy_analyzer (session, trade cap, LLM interpretation)
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS analysis_notes                 jsonb;
-- Flag: analysis already run for this quick_test cycle
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS analysis_done                  boolean DEFAULT false;

-- Research task IDs blocking this strategy (set when awaiting research results)
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS pending_research_ids     jsonb DEFAULT '[]'::jsonb;

-- Campaign / variation exploration
-- campaign_id: NULL for standalone strategies and campaign roots.
--              Set to the root strategy's id for all variation children.
-- is_campaign_root: true on the original strategy that spawned the campaign.
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS campaign_id          uuid    REFERENCES strategies(id) ON DELETE SET NULL;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS is_campaign_root     boolean DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_strategies_campaign_id ON strategies(campaign_id);

CREATE INDEX IF NOT EXISTS idx_strategies_status     ON strategies(status);
CREATE INDEX IF NOT EXISTS idx_strategies_updated_at ON strategies(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_strategies_source     ON strategies(source);
CREATE INDEX IF NOT EXISTS idx_strategies_sharpe     ON strategies(backtest_sharpe DESC NULLS LAST);

DROP TRIGGER IF EXISTS trg_strategies_updated_at ON strategies;
CREATE TRIGGER trg_strategies_updated_at
  BEFORE UPDATE ON strategies
  FOR EACH ROW EXECUTE FUNCTION _set_updated_at();


-- =============================================================================
-- Table: user_ideas
-- =============================================================================

CREATE TABLE IF NOT EXISTS user_ideas (
  id          uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  title       text         NOT NULL,
  description text         NOT NULL,
  notes       text,
  status      text         NOT NULL DEFAULT 'pending',
  -- pending → picked_up → done | failed
  priority    int          DEFAULT 5,
  strategy_id uuid         REFERENCES strategies(id) ON DELETE SET NULL,
  created_at  timestamptz  NOT NULL DEFAULT now(),
  updated_at  timestamptz  NOT NULL DEFAULT now()
);

ALTER TABLE user_ideas ADD COLUMN IF NOT EXISTS notes       text;
ALTER TABLE user_ideas ADD COLUMN IF NOT EXISTS priority    int DEFAULT 5;
ALTER TABLE user_ideas ADD COLUMN IF NOT EXISTS updated_at  timestamptz DEFAULT now();
ALTER TABLE user_ideas ADD COLUMN IF NOT EXISTS strategy_id uuid REFERENCES strategies(id) ON DELETE SET NULL;
ALTER TABLE user_ideas ADD COLUMN IF NOT EXISTS source      text DEFAULT 'user';  -- 'user' | 'indicator_library' | 'seed_agent'

CREATE INDEX IF NOT EXISTS idx_user_ideas_status      ON user_ideas(status);
CREATE INDEX IF NOT EXISTS idx_user_ideas_priority    ON user_ideas(priority, created_at);
CREATE INDEX IF NOT EXISTS idx_user_ideas_strategy_id ON user_ideas(strategy_id);

DROP TRIGGER IF EXISTS trg_user_ideas_updated_at ON user_ideas;
CREATE TRIGGER trg_user_ideas_updated_at
  BEFORE UPDATE ON user_ideas
  FOR EACH ROW EXECUTE FUNCTION _set_updated_at();


-- =============================================================================
-- Table: generated_ideas  (auto-generated from arXiv / Semantic Scholar)
-- =============================================================================

CREATE TABLE IF NOT EXISTS generated_ideas (
  id           uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  title        text         NOT NULL,
  summary      text,        -- LLM-extracted strategy description (entry/exit rules)
  source_type  text,        -- 'arxiv' | 'semantic_scholar'
  source_title text,        -- original paper title
  source_url   text,        -- link to paper (used for deduplication)
  asset_class  text         DEFAULT 'multi',   -- 'forex'|'equity'|'crypto'|'multi'
  confidence   text         DEFAULT 'medium',  -- 'high'|'medium'|'low'
  status       text         NOT NULL DEFAULT 'pending',
  -- pending → approved | dismissed
  user_idea_id uuid         REFERENCES user_ideas(id) ON DELETE SET NULL,
  created_at   timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_generated_ideas_status     ON generated_ideas(status);
CREATE INDEX IF NOT EXISTS idx_generated_ideas_source_url ON generated_ideas(source_url);
CREATE INDEX IF NOT EXISTS idx_generated_ideas_created_at ON generated_ideas(created_at DESC);


-- =============================================================================
-- Table: knowledge_base  (Learner agent output)
-- =============================================================================

CREATE TABLE IF NOT EXISTS knowledge_base (
  id          uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  category    text         NOT NULL,   -- 'works'|'fails'|'partial'|'edge_case'
  indicator   text,
  timeframe   text,
  asset       text,
  session     text,
  summary     text         NOT NULL,
  sharpe_ref  float8,
  strategy_id uuid         REFERENCES strategies(id) ON DELETE SET NULL,
  created_at  timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_base_category    ON knowledge_base(category, indicator);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_strategy_id ON knowledge_base(strategy_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_created_at  ON knowledge_base(created_at DESC);

ALTER TABLE knowledge_base ADD COLUMN IF NOT EXISTS p_value        float8;
ALTER TABLE knowledge_base ADD COLUMN IF NOT EXISTS profit_factor  float8;
ALTER TABLE knowledge_base ADD COLUMN IF NOT EXISTS t_stat         float8;
ALTER TABLE knowledge_base ADD COLUMN IF NOT EXISTS optimal_params jsonb;


-- =============================================================================
-- Table: spend_log  (LLM cost tracking for daily budget enforcement)
-- =============================================================================

CREATE TABLE IF NOT EXISTS spend_log (
  id            uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  agent         text         NOT NULL,
  -- 'pre_filter'|'implementer'|'validator'|'summariser'|'learner'|'code_fixer'|'idea_generator'
  model         text         NOT NULL,
  input_tokens  int          NOT NULL DEFAULT 0,
  output_tokens int          NOT NULL DEFAULT 0,
  cost_usd      float8       NOT NULL DEFAULT 0,
  strategy_id   uuid         REFERENCES strategies(id) ON DELETE SET NULL,
  date          date         NOT NULL DEFAULT CURRENT_DATE,
  created_at    timestamptz  NOT NULL DEFAULT now()
);

ALTER TABLE spend_log ADD COLUMN IF NOT EXISTS date date NOT NULL DEFAULT CURRENT_DATE;

CREATE INDEX IF NOT EXISTS idx_spend_log_date         ON spend_log(date);
CREATE INDEX IF NOT EXISTS idx_spend_log_strategy_id  ON spend_log(strategy_id);
CREATE INDEX IF NOT EXISTS idx_spend_log_agent        ON spend_log(agent);


-- =============================================================================
-- Row Level Security
-- The anon key (used by supabase-py) requires explicit RLS policies.
-- This is a private backend tool — simple "allow all" policies are fine.
-- Tighten these if you add user-facing auth later.
-- =============================================================================

ALTER TABLE strategies      ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_ideas       ENABLE ROW LEVEL SECURITY;
ALTER TABLE generated_ideas  ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_base   ENABLE ROW LEVEL SECURITY;
ALTER TABLE spend_log        ENABLE ROW LEVEL SECURITY;

-- Drop and recreate policies so re-runs are safe
DO $$
DECLARE pol RECORD;
BEGIN
  FOR pol IN
    SELECT policyname, tablename FROM pg_policies
    WHERE tablename IN (
      'strategies', 'user_ideas', 'generated_ideas', 'knowledge_base', 'spend_log',
      'research_tasks', 'indicator_library', 'system_config', 'prob_research_results'
    )
  LOOP
    EXECUTE format('DROP POLICY IF EXISTS %I ON %I', pol.policyname, pol.tablename);
  END LOOP;
END $$;

CREATE POLICY allow_all ON strategies      FOR ALL USING (true) WITH CHECK (true);


-- =============================================================================
-- Table: research_tasks  (standalone research + those spawned by agents)
-- =============================================================================

CREATE TABLE IF NOT EXISTS research_tasks (
  id                      uuid         PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Task identity
  type                    text         NOT NULL DEFAULT 'market_analysis',
  -- 'market_analysis' | 'indicator_research' | 'custom'
  title                   text         NOT NULL,
  question                text         NOT NULL,  -- the research question / objective

  -- Data requirements (optional — what OHLCV data the analysis needs)
  data_requirements       jsonb,        -- {symbol: str, timeframe: str, start: str, end: str}

  -- Pipeline state
  status                  text         NOT NULL DEFAULT 'pending',
  -- pending → running → done | failed

  -- Results
  result_summary          text,         -- 2-4 sentence plain text summary of findings
  report_text             text,         -- full markdown report
  report_url              text,         -- Cloudflare R2 URL (if uploaded)
  key_findings            jsonb,        -- [{finding: str, confidence: float}]
  generated_code          text,         -- the Python analysis code that was executed

  -- Provenance
  created_by_strategy_id  uuid         REFERENCES strategies(id) ON DELETE SET NULL,
  -- null = standalone research task submitted directly

  -- Compute tracking
  modal_job_id            text,
  error_log               text,

  created_at              timestamptz  NOT NULL DEFAULT now(),
  updated_at              timestamptz  NOT NULL DEFAULT now()
);

ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS generated_code  text;
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS key_findings    jsonb;
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS research_spec   jsonb;
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS retry_count     int  DEFAULT 0;
-- Multi-symbol fan-out: fanned_out=true once symbol variants have been queued;
-- parent_task_id links variant tasks back to the original.
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS fanned_out      boolean DEFAULT false;
ALTER TABLE research_tasks ADD COLUMN IF NOT EXISTS parent_task_id  uuid REFERENCES research_tasks(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_research_tasks_status     ON research_tasks(status);
CREATE INDEX IF NOT EXISTS idx_research_tasks_type       ON research_tasks(type);
CREATE INDEX IF NOT EXISTS idx_research_tasks_created_at ON research_tasks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_tasks_strategy   ON research_tasks(created_by_strategy_id);

DROP TRIGGER IF EXISTS trg_research_tasks_updated_at ON research_tasks;
CREATE TRIGGER trg_research_tasks_updated_at
  BEFORE UPDATE ON research_tasks
  FOR EACH ROW EXECUTE FUNCTION _set_updated_at();

ALTER TABLE research_tasks ENABLE ROW LEVEL SECURITY;
CREATE POLICY allow_all ON research_tasks FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY allow_all ON user_ideas       FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY allow_all ON generated_ideas  FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY allow_all ON knowledge_base   FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY allow_all ON spend_log        FOR ALL USING (true) WITH CHECK (true);

-- =============================================================================
-- Table: indicator_library  (reusable indicator implementations from research)
-- =============================================================================
CREATE TABLE IF NOT EXISTS indicator_library (
  id             uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  spec_id        text         UNIQUE NOT NULL,   -- matches research_spec.spec_id
  name           text         NOT NULL,          -- indicator family name, e.g. "wick_ratio"
  display_name   text         NOT NULL,          -- human title
  category       text         NOT NULL,          -- momentum/trend/volatility/volume/structure/custom
  description    text,                           -- ENTRY/STOP/RR/EXIT spec description
  code           text,                           -- full analyze_indicator(df, **params) function
  best_params    jsonb,                          -- optimal params found during research
  best_sharpe    float,                          -- best sharpe_ref from knowledge_base
  source_task_id uuid         REFERENCES research_tasks(id) ON DELETE SET NULL,
  created_at     timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_indicator_library_category ON indicator_library(category);
CREATE INDEX IF NOT EXISTS idx_indicator_library_name     ON indicator_library(name);

ALTER TABLE indicator_library ADD COLUMN IF NOT EXISTS strategy_generated boolean DEFAULT false;

ALTER TABLE indicator_library ENABLE ROW LEVEL SECURITY;
CREATE POLICY allow_all ON indicator_library FOR ALL USING (true) WITH CHECK (true);

-- =============================================================================
-- Table: prob_research_results  (statistical probability research)
-- =============================================================================
CREATE TABLE IF NOT EXISTS prob_research_results (
  id               uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  condition_id     text         NOT NULL,
  condition_desc   text         NOT NULL,
  category         text         NOT NULL,
  symbol           text         NOT NULL,
  timeframe        text         NOT NULL,
  forward_bars     int          NOT NULL,
  n_samples        int,
  hit_rate         float8,
  mean_return      float8,
  std_return       float8,
  median_return    float8,
  t_stat           float8,
  p_value          float8,
  sharpe           float8,
  is_significant   boolean      DEFAULT false,
  last_updated     timestamptz  NOT NULL DEFAULT now(),
  UNIQUE (condition_id, symbol, timeframe, forward_bars)
);

CREATE INDEX IF NOT EXISTS idx_prob_results_symbol    ON prob_research_results(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_prob_results_category  ON prob_research_results(category);
CREATE INDEX IF NOT EXISTS idx_prob_results_sig       ON prob_research_results(is_significant, p_value);

ALTER TABLE prob_research_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY allow_all ON prob_research_results FOR ALL USING (true) WITH CHECK (true);

-- =============================================================================
-- Table: system_config  (single-row key/value store for persistent system state)
-- =============================================================================
CREATE TABLE IF NOT EXISTS system_config (
  key        text PRIMARY KEY,
  value      text,
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE system_config ENABLE ROW LEVEL SECURITY;
CREATE POLICY allow_all ON system_config FOR ALL USING (true) WITH CHECK (true);
