-- DuckDB schema for institutional_trades migration
-- Source: SQLite institutional_trades.db, 2.1M rows, 125 tickers, 60 days

CREATE TABLE IF NOT EXISTS institutional_trades (
    id                  BIGINT PRIMARY KEY,
    tracking_id         BIGINT UNIQUE,
    ticker              VARCHAR NOT NULL,
    trade_ts            TIMESTAMPTZ NOT NULL,
    trade_date          DATE NOT NULL,
    sip_ts_ns           BIGINT,
    side                VARCHAR NOT NULL DEFAULT 'UNKNOWN',
    shares              DOUBLE NOT NULL,
    price               DOUBLE NOT NULL,
    notional_usd        DOUBLE NOT NULL,
    nbbo_bid            DOUBLE,
    nbbo_ask            DOUBLE,
    exchange_code       VARCHAR,
    exchange_name       VARCHAR,
    is_dark_pool        BOOLEAN NOT NULL DEFAULT TRUE,
    is_block            BOOLEAN NOT NULL DEFAULT FALSE,
    is_sweep            BOOLEAN NOT NULL DEFAULT FALSE,
    is_cross            BOOLEAN NOT NULL DEFAULT FALSE,
    is_algo             BOOLEAN NOT NULL DEFAULT FALSE,
    is_closing_auction  BOOLEAN NOT NULL DEFAULT FALSE,
    is_canceled         BOOLEAN NOT NULL DEFAULT FALSE,
    sale_cond_codes     VARCHAR,
    provider            VARCHAR NOT NULL DEFAULT 'uw',
    fetched_at          TIMESTAMPTZ NOT NULL
);

CREATE SEQUENCE IF NOT EXISTS seq_institutional_trades_id START 1;

CREATE INDEX IF NOT EXISTS idx_inst_ticker_ts ON institutional_trades(ticker, trade_ts);
CREATE INDEX IF NOT EXISTS idx_inst_date_ticker ON institutional_trades(trade_date, ticker);
CREATE INDEX IF NOT EXISTS idx_inst_notional ON institutional_trades(notional_usd);
CREATE INDEX IF NOT EXISTS idx_inst_block_date ON institutional_trades(is_block, trade_date);

CREATE TABLE IF NOT EXISTS ingest_cursor (
    ticker           VARCHAR PRIMARY KEY,
    last_trade_ts    TIMESTAMPTZ NOT NULL,
    last_tracking_id BIGINT,
    rows_total       BIGINT NOT NULL DEFAULT 0,
    updated_at       TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS institutional_scraper_state (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    last_poll_at      TIMESTAMPTZ,
    last_provider     VARCHAR,
    last_row_count    BIGINT,
    last_ticker_count BIGINT,
    last_error        VARCHAR,
    updated_at        TIMESTAMPTZ
);
