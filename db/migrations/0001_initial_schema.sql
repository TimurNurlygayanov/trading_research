-- Initial schema: strategies, user_ideas, knowledge_base, spend_log
-- Safe to run on existing DB (IF NOT EXISTS throughout)

CREATE OR REPLACE FUNCTION _set_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$;

CREATE TABLE IF NOT EXISTS strategies (
  id                  uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  name                text         NOT NULL DEFAULT 'Untitled Strategy',
  source              text         NOT NULL DEFAULT 'user',
  status              text         NOT NULL DEFAULT 'idea',
  hypothesis          text,
  entry_logic         text,
  exit_logic          text,
  indicators          jsonb,
  hyperparams         jsonb,
  backtest_code       text,
  pre_filter_score    float8,
  pre_filter_notes    jsonb,
  leakage_score       float8,
  leakage_issues      jsonb,
  backtest_sharpe     float8,
  backtest_calmar     float8,
  max_drawdown        float8,
  total_signals       int,
  signals_per_year    float8,
  win_rate            float8,
  profit_factor       float8,
  avg_trade_pnl       float8,
  oos_sharpe          float8,
  oos_win_rate        float8,
  oos_total_trades    int,
  walk_forward_scores jsonb,
  monte_carlo_pvalue  float8,
  error_log           text,
  report_url          text,
  report_text         text,
  retry_count         int          DEFAULT 0,
  created_at          timestamptz  NOT NULL DEFAULT now(),
  updated_at          timestamptz  NOT NULL DEFAULT now()
);

DROP TRIGGER IF EXISTS trg_strategies_updated_at ON strategies;
CREATE TRIGGER trg_strategies_updated_at
  BEFORE UPDATE ON strategies
  FOR EACH ROW EXECUTE FUNCTION _set_updated_at();

CREATE INDEX IF NOT EXISTS idx_strategies_status     ON strategies(status);
CREATE INDEX IF NOT EXISTS idx_strategies_updated_at ON strategies(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_strategies_sharpe     ON strategies(backtest_sharpe DESC NULLS LAST);

CREATE TABLE IF NOT EXISTS user_ideas (
  id          uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  title       text         NOT NULL,
  description text         NOT NULL,
  notes       text,
  priority    int          DEFAULT 5,
  status      text         NOT NULL DEFAULT 'pending',
  strategy_id uuid         REFERENCES strategies(id) ON DELETE SET NULL,
  created_at  timestamptz  NOT NULL DEFAULT now(),
  updated_at  timestamptz  NOT NULL DEFAULT now()
);

DROP TRIGGER IF EXISTS trg_user_ideas_updated_at ON user_ideas;
CREATE TRIGGER trg_user_ideas_updated_at
  BEFORE UPDATE ON user_ideas
  FOR EACH ROW EXECUTE FUNCTION _set_updated_at();

CREATE INDEX IF NOT EXISTS idx_user_ideas_status   ON user_ideas(status);
CREATE INDEX IF NOT EXISTS idx_user_ideas_priority ON user_ideas(priority, created_at);

CREATE TABLE IF NOT EXISTS knowledge_base (
  id          uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  category    text         NOT NULL,
  indicator   text,
  timeframe   text,
  asset       text,
  session     text,
  summary     text         NOT NULL,
  sharpe_ref  float8,
  strategy_id uuid         REFERENCES strategies(id) ON DELETE SET NULL,
  created_at  timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_base_category ON knowledge_base(category, indicator);

CREATE TABLE IF NOT EXISTS spend_log (
  id            uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  agent         text         NOT NULL,
  model         text         NOT NULL,
  input_tokens  int          NOT NULL DEFAULT 0,
  output_tokens int          NOT NULL DEFAULT 0,
  cost_usd      float8       NOT NULL DEFAULT 0,
  strategy_id   uuid         REFERENCES strategies(id) ON DELETE SET NULL,
  date          date         NOT NULL DEFAULT CURRENT_DATE,
  created_at    timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_spend_log_date ON spend_log(date);

-- RLS: allow anon key full access (private backend tool)
ALTER TABLE strategies    ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_ideas     ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_base ENABLE ROW LEVEL SECURITY;
ALTER TABLE spend_log      ENABLE ROW LEVEL SECURITY;

DO $$ DECLARE pol RECORD;
BEGIN
  FOR pol IN SELECT policyname, tablename FROM pg_policies
    WHERE tablename IN ('strategies','user_ideas','knowledge_base','spend_log')
  LOOP
    EXECUTE format('DROP POLICY IF EXISTS %I ON %I', pol.policyname, pol.tablename);
  END LOOP;
END $$;

CREATE POLICY allow_all ON strategies    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY allow_all ON user_ideas     FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY allow_all ON knowledge_base FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY allow_all ON spend_log      FOR ALL USING (true) WITH CHECK (true);
