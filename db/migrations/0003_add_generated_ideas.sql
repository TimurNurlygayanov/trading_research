-- Research page: AI-generated strategy ideas from arXiv / Semantic Scholar

CREATE TABLE IF NOT EXISTS generated_ideas (
  id           uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  title        text         NOT NULL,
  summary      text,
  source_type  text,        -- 'arxiv' | 'semantic_scholar'
  source_title text,
  source_url   text,        -- used for deduplication
  asset_class  text         DEFAULT 'multi',
  confidence   text         DEFAULT 'medium',
  status       text         NOT NULL DEFAULT 'pending',
  user_idea_id uuid         REFERENCES user_ideas(id) ON DELETE SET NULL,
  created_at   timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_generated_ideas_status     ON generated_ideas(status);
CREATE INDEX IF NOT EXISTS idx_generated_ideas_source_url ON generated_ideas(source_url);
CREATE INDEX IF NOT EXISTS idx_generated_ideas_created_at ON generated_ideas(created_at DESC);

ALTER TABLE generated_ideas ENABLE ROW LEVEL SECURITY;

DO $$ DECLARE pol RECORD;
BEGIN
  FOR pol IN SELECT policyname, tablename FROM pg_policies
    WHERE tablename = 'generated_ideas'
  LOOP
    EXECUTE format('DROP POLICY IF EXISTS %I ON %I', pol.policyname, pol.tablename);
  END LOOP;
END $$;

CREATE POLICY allow_all ON generated_ideas FOR ALL USING (true) WITH CHECK (true);
