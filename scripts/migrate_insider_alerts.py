#!/usr/bin/env python3
# scripts/migrate_insider_alerts.py
# ─────────────────────────────────────────────────────────────────────────────
# Idempotent schema migration for the insider alert system.
# Adds insider_filings_raw and insider_alerts tables to insider_trades.db.
# Does NOT touch existing insider_flows table.
#
# Safe to run multiple times. Uses CREATE TABLE IF NOT EXISTS.
#
# Usage:
#   python scripts/migrate_insider_alerts.py
#   python scripts/migrate_insider_alerts.py --verify
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import argparse
import os
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(os.getenv("INSIDER_DB_PATH", "insider_trades.db"))


SCHEMA = """
-- Raw per-filing insider data, one row per (insider, transaction).
-- Populated by data/etl_insider_realtime.py (Phase 2 — not built yet).
-- This is the canonical source. insider_flows is rebuilt from this.
CREATE TABLE IF NOT EXISTS insider_filings_raw (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    accession           TEXT    NOT NULL,        -- SEC filing ID
    filing_date         TEXT    NOT NULL,        -- when SEC received it
    trade_date          TEXT    NOT NULL,        -- periodOfReport from XML
    insider_name        TEXT,                    -- "Cook, Timothy D."
    insider_title       TEXT,                    -- raw role string
    role_weight         REAL    NOT NULL DEFAULT 1.0,
    transaction_code    TEXT,                    -- P/S/A/G/F/M
    shares              REAL    NOT NULL,
    price_per_share     REAL,
    notional_usd        REAL,                    -- shares * price (NULL if price unknown)
    acquired_disposed   TEXT    NOT NULL,        -- 'A' (acquired/buy) or 'D' (disposed/sell)
    is_csuite           INTEGER NOT NULL DEFAULT 0,  -- 1 if role_weight >= 2.5
    fetched_at          TEXT    NOT NULL,        -- ISO timestamp when our scraper saw it
    UNIQUE(accession, insider_name, transaction_code, trade_date, shares)
);

CREATE INDEX IF NOT EXISTS idx_raw_ticker_date     ON insider_filings_raw(ticker, trade_date);
CREATE INDEX IF NOT EXISTS idx_raw_filing_date     ON insider_filings_raw(filing_date);
CREATE INDEX IF NOT EXISTS idx_raw_fetched_at      ON insider_filings_raw(fetched_at);
CREATE INDEX IF NOT EXISTS idx_raw_csuite_buy      ON insider_filings_raw(is_csuite, acquired_disposed)
    WHERE is_csuite = 1 AND acquired_disposed = 'A';


-- Alert log — one row per alert classified, regardless of notification channel.
-- Used for audit, threshold tuning, and "did we alert on this?" lookups.
CREATE TABLE IF NOT EXISTS insider_alerts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_filing_id       INTEGER NOT NULL,        -- FK to insider_filings_raw.id
    ticker              TEXT    NOT NULL,
    signal              TEXT    NOT NULL,        -- 'GREEN_STRONG'|'GREEN_WEAK'|'RED_STRONG'|'RED_WEAK'
    rationale           TEXT,                    -- human-readable why
    notional_usd        REAL,                    -- denormalized for dashboard queries
    insider_title       TEXT,
    sent_at             TEXT    NOT NULL,        -- ISO timestamp when classified
    notification_sent   INTEGER NOT NULL DEFAULT 0,  -- 1 if macOS/email fired
    notification_channels TEXT,                  -- comma-separated: 'macos,email'
    FOREIGN KEY (raw_filing_id) REFERENCES insider_filings_raw(id)
);

CREATE INDEX IF NOT EXISTS idx_alerts_ticker_time  ON insider_alerts(ticker, sent_at);
CREATE INDEX IF NOT EXISTS idx_alerts_signal       ON insider_alerts(signal, sent_at);


-- Scraper state — tracks last poll time and last processed filing id.
-- Single-row table. Lets the alert checker know what's new since last run.
CREATE TABLE IF NOT EXISTS insider_scraper_state (
    id                       INTEGER PRIMARY KEY CHECK (id = 1),  -- only one row
    last_poll_at             TEXT,
    last_alert_check_id      INTEGER NOT NULL DEFAULT 0,  -- max raw.id processed by alert checker
    last_aggregate_at        TEXT,                        -- when we last rebuilt insider_flows
    updated_at               TEXT
);

INSERT OR IGNORE INTO insider_scraper_state (id, last_alert_check_id) VALUES (1, 0);
"""


def migrate(db_path: Path) -> dict:
    """Apply schema. Returns dict of table_name → row_count after migration."""
    if not db_path.exists():
        print(f"  ⓘ  {db_path} doesn't exist yet — creating fresh DB.")

    conn = sqlite3.connect(db_path)
    try:
        # Run each statement separately so a failure in one doesn't silently abort others.
        for statement in [s.strip() for s in SCHEMA.split(";") if s.strip()]:
            conn.execute(statement)
        conn.commit()

        counts = {}
        for table in ["insider_filings_raw", "insider_alerts", "insider_scraper_state",
                      "insider_flows"]:
            try:
                counts[table] = conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = "MISSING"
        return counts
    finally:
        conn.close()


def verify(db_path: Path) -> bool:
    """Read back the schema and confirm all expected objects exist."""
    if not db_path.exists():
        print(f"  ✗  DB not found: {db_path}")
        return False

    expected_tables = {
        "insider_flows",            # pre-existing, must still be there
        "insider_filings_raw",      # new
        "insider_alerts",           # new
        "insider_scraper_state",    # new
    }
    expected_indexes = {
        "idx_ticker_date",          # pre-existing on insider_flows
        "idx_raw_ticker_date",
        "idx_raw_filing_date",
        "idx_raw_fetched_at",
        "idx_raw_csuite_buy",
        "idx_alerts_ticker_time",
        "idx_alerts_signal",
    }

    conn = sqlite3.connect(db_path)
    try:
        actual_tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        actual_indexes = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }

        missing_tables  = expected_tables  - actual_tables
        missing_indexes = expected_indexes - actual_indexes

        if missing_tables:
            print(f"  ✗  Missing tables: {missing_tables}")
        if missing_indexes:
            print(f"  ✗  Missing indexes: {missing_indexes}")

        if not missing_tables and not missing_indexes:
            print("  ✓  All tables and indexes present.")
            return True
        return False
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Insider-alert schema migration.")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify existing schema; don't migrate.")
    parser.add_argument("--db", default=str(DB_PATH),
                        help=f"DB path (default: {DB_PATH})")
    args = parser.parse_args()

    db_path = Path(args.db)
    print(f"Insider-alert schema migration")
    print(f"  DB: {db_path}")
    print()

    if args.verify:
        return 0 if verify(db_path) else 1

    counts = migrate(db_path)
    print("After migration, row counts:")
    for table, count in counts.items():
        print(f"  {table:30s}  {count}")
    print()

    if verify(db_path):
        print("Migration successful.")
        return 0
    print("Migration FAILED verification.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
