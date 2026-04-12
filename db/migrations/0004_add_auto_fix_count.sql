-- Track automatic code-fix attempts by the code_fixer agent

ALTER TABLE strategies ADD COLUMN IF NOT EXISTS auto_fix_count int DEFAULT 0;
