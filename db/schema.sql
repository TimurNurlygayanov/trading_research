-- Enable vector extension for semantic dedup
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- strategies: the core pipeline table
-- Status flow: idea → filtered → implementing → implemented → validating → done | failed
-- ============================================================
CREATE TABLE IF NOT EXISTS strategies (
  id                  uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  name                text NOT NULL,
  source              text,          -- 'user' | 'quantpedia' | 'arxiv' | 'ssrn' | 'semantic_scholar'
  source_url          text,
  hypothesis          text,
  entry_logic         text,
  exit_logic          text,
  indicators          jsonb,
  timeframes          text[],
  status              text DEFAULT 'idea',
  pre_filter_score    float,
  pre_filter_notes    text,
  -- Backtest results (in-sample)
  backtest_sharpe     float,
  backtest_calmar     float,
  max_drawdown        float,
  total_signals       int,
  signals_per_year    float,
  win_rate            float,
  profit_factor       float,
  avg_trade_pnl       float,
  -- OOS results (2026+)
  oos_sharpe          float,
  oos_win_rate        float,
  oos_total_trades    int,
  -- Validation
  leakage_score       float,         -- 0 (bad) to 10 (clean)
  leakage_issues      jsonb,         -- list of detected issues
  monte_carlo_pvalue  float,         -- p-value from permutation test
  walk_forward_scores jsonb,         -- array of OOS Sharpe per fold
  -- Optimization
  hyperparams         jsonb,         -- best params from Optuna
  best_session_hours  jsonb,         -- {start_hour: N, end_hour: M}
  risk_params         jsonb,         -- {max_daily_losses: N, sl_atr: X, tp_atr: Y}
  -- Code & reports
  backtest_code       text,
  report_url          text,
  equity_curve_url    text,
  -- Tracking
  linear_issue_id     text,
  retry_count         int DEFAULT 0,
  error_log           text,
  created_at          timestamptz DEFAULT now(),
  updated_at          timestamptz DEFAULT now()
);

-- ============================================================
-- user_ideas: plain-English strategy ideas submitted by you
-- ============================================================
CREATE TABLE IF NOT EXISTS user_ideas (
  id          uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  title       text NOT NULL,
  description text NOT NULL,
  notes       text,
  priority    int DEFAULT 5,          -- 1 (highest) to 10 (lowest)
  status      text DEFAULT 'pending', -- pending | picked_up | done | failed
  strategy_id uuid REFERENCES strategies(id),
  created_at  timestamptz DEFAULT now()
);

-- ============================================================
-- knowledge_base: what the Learner agent accumulates
-- ============================================================
CREATE TABLE IF NOT EXISTS knowledge_base (
  id          uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  category    text NOT NULL,  -- 'works' | 'fails' | 'partial' | 'edge_case'
  indicator   text,
  timeframe   text,
  asset       text,
  session     text,           -- 'london' | 'newyork' | 'asian' | 'overlap' | null
  summary     text NOT NULL,
  sharpe_ref  float,          -- reference Sharpe when this was recorded
  strategy_id uuid REFERENCES strategies(id),
  embedding   vector(1536),   -- for semantic search/dedup
  created_at  timestamptz DEFAULT now()
);

-- ============================================================
-- spend_log: track every LLM API call cost
-- ============================================================
CREATE TABLE IF NOT EXISTS spend_log (
  id            uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  date          date DEFAULT CURRENT_DATE,
  agent         text NOT NULL,
  model         text NOT NULL,
  input_tokens  int NOT NULL,
  output_tokens int NOT NULL,
  cost_usd      float NOT NULL,
  strategy_id   uuid REFERENCES strategies(id),
  created_at    timestamptz DEFAULT now()
);

-- ============================================================
-- Indexes for fast queue polling and reporting
-- ============================================================
CREATE INDEX IF NOT EXISTS strategies_status_idx       ON strategies(status);
CREATE INDEX IF NOT EXISTS strategies_sharpe_idx       ON strategies(backtest_sharpe DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS strategies_updated_idx      ON strategies(updated_at DESC);
CREATE INDEX IF NOT EXISTS spend_log_date_idx          ON spend_log(date);
CREATE INDEX IF NOT EXISTS user_ideas_status_prio_idx  ON user_ideas(status, priority);
CREATE INDEX IF NOT EXISTS knowledge_base_category_idx ON knowledge_base(category, indicator);

-- ============================================================
-- Auto-update updated_at
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER strategies_updated_at
  BEFORE UPDATE ON strategies
  FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
