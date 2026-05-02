-- migrations/init_institutional_trades_db.sql
-- Schema for institutional_trades.db (per-transaction institutional flow from Massive).
-- This file is for reference/manual restores. The schema is also embedded in
-- features/institutional_ingest.py and applied idempotently on first run.

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS institutional_trades (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    trade_ts            TEXT    NOT NULL,           -- ISO 8601 UTC
    trade_date          TEXT    NOT NULL,           -- denorm YYYY-MM-DD
    sip_ts_ns           INTEGER,                    -- Polygon/Massive sip_timestamp (ns)
    side                TEXT    NOT NULL DEFAULT 'UNKNOWN',  -- BUY/SELL/UNKNOWN
    shares              REAL    NOT NULL,
    price               REAL    NOT NULL,
    notional_usd        REAL    NOT NULL,
    exchange_code       INTEGER,                    -- Massive exchange ID
    exchange_name       TEXT,                       -- resolved name
    is_dark_pool        INTEGER NOT NULL DEFAULT 0, -- 1 if FINRA TRF/ADF
    is_block            INTEGER NOT NULL DEFAULT 0, -- 1 if shares >= 10,000
    is_sweep            INTEGER NOT NULL DEFAULT 0, -- 1 if intermarket sweep condition
    is_cross            INTEGER NOT NULL DEFAULT 0, -- 1 if cross trade condition
    conditions_raw      TEXT,                       -- JSON array of condition codes
    provider            TEXT    NOT NULL DEFAULT 'massive',
    fetched_at          TEXT    NOT NULL,
    UNIQUE(ticker, sip_ts_ns, shares, price)
);

CREATE INDEX IF NOT EXISTS idx_inst_ticker_ts    ON institutional_trades(ticker, trade_ts DESC);
CREATE INDEX IF NOT EXISTS idx_inst_date_ticker  ON institutional_trades(trade_date, ticker);
CREATE INDEX IF NOT EXISTS idx_inst_notional     ON institutional_trades(notional_usd DESC);
CREATE INDEX IF NOT EXISTS idx_inst_dark_pool    ON institutional_trades(is_dark_pool, trade_date DESC) WHERE is_dark_pool = 1;
CREATE INDEX IF NOT EXISTS idx_inst_block        ON institutional_trades(is_block, trade_date DESC) WHERE is_block = 1;

CREATE TABLE IF NOT EXISTS institutional_scraper_state (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    last_poll_at        TEXT,
    last_provider       TEXT,
    last_row_count      INTEGER,
    last_ticker_count   INTEGER,
    last_error          TEXT,
    updated_at          TEXT
);
INSERT OR IGNORE INTO institutional_scraper_state (id) VALUES (1);

CREATE TABLE IF NOT EXISTS ingest_cursor (
    ticker              TEXT PRIMARY KEY,
    last_trade_ts       TEXT NOT NULL,
    last_sip_ts_ns      INTEGER,
    rows_total          INTEGER NOT NULL DEFAULT 0,
    updated_at          TEXT NOT NULL
);
