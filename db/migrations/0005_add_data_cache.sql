-- Migration 0005: OHLCV data cache metadata table
-- Stores per-dataset stats + recent bars for the /data dashboard page.
-- Written by the preload Modal job; read by the orchestrator API.

CREATE TABLE IF NOT EXISTS data_cache (
    id               serial       PRIMARY KEY,
    symbol           text         NOT NULL,
    timeframe        text         NOT NULL,
    bar_count        int,
    first_date       timestamptz,
    last_date        timestamptz,
    file_size_mb     float,
    price_min        float,
    price_max        float,
    price_mean       float,
    price_std        float,
    avg_volume       float,
    completeness_pct float,
    bars_by_year     jsonb,
    recent_bars      jsonb,
    cached_at        timestamptz  NOT NULL DEFAULT now(),
    UNIQUE (symbol, timeframe)
);

-- Index for dashboard query (order by symbol, timeframe)
CREATE INDEX IF NOT EXISTS idx_data_cache_lookup
    ON data_cache (symbol, timeframe);
