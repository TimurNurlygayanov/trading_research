-- Migration 0006: add bars_by_year and expected_start to data_cache
-- bars_by_year: bar count per calendar year for coverage visualisation.
-- expected_start: the configured start date requested during preload,
--   so the UI can warn when actual data starts much later.

ALTER TABLE data_cache
    ADD COLUMN IF NOT EXISTS bars_by_year   jsonb;

ALTER TABLE data_cache
    ADD COLUMN IF NOT EXISTS expected_start text;
