-- Add columns introduced during pipeline development:
-- comments, tags, modal_job_id, validator_corrections,
-- best_hyperparams, best_session_hours, timeframes, symbol,
-- indicators_used, risk_params, report_text, equity_curve_url

ALTER TABLE strategies ADD COLUMN IF NOT EXISTS comments              jsonb DEFAULT '[]'::jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS tags                  jsonb DEFAULT '[]'::jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS modal_job_id          text;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS validator_corrections int   DEFAULT 0;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS best_hyperparams      jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS best_session_hours    text;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS risk_params           jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS timeframes            jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS symbol                text;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS indicators_used       jsonb;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS report_text           text;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS equity_curve_url      text;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS source_url            text;

ALTER TABLE user_ideas ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();
