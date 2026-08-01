"""
scripts/migrations/migrate_inst_to_duckdb.py
─────────────────────────────────────────────────────────────────────────────
One-time migration: institutional_trades.db (SQLite) -> institutional_trades.duckdb (DuckDB).

Usage:
  python -m scripts.migrations.migrate_inst_to_duckdb [--dry-run]

Idempotent: safe to re-run. Drops + recreates target.
"""
from __future__ import annotations
import argparse
import sqlite3
import sys
import time
from pathlib import Path

import duckdb

SRC_DB = Path("institutional_trades.db")
DST_DB = Path("institutional_trades.duckdb")
SCHEMA_FILE = Path("scripts/migrations/duckdb_schema.sql")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Verify counts only")
    args = ap.parse_args()

    if not SRC_DB.exists():
        print(f"ERROR: {SRC_DB} not found")
        sys.exit(1)
    if not SCHEMA_FILE.exists():
        print(f"ERROR: {SCHEMA_FILE} not found")
        sys.exit(1)

    # Source row counts
    print(f"=== Reading source: {SRC_DB} ===")
    src = sqlite3.connect(SRC_DB, timeout=30)
    src_main = src.execute("SELECT COUNT(*) FROM institutional_trades").fetchone()[0]
    src_cursor = src.execute("SELECT COUNT(*) FROM ingest_cursor").fetchone()[0]
    src_state = src.execute("SELECT COUNT(*) FROM institutional_scraper_state").fetchone()[0]
    src.close()
    print(f"  institutional_trades: {src_main:,} rows")
    print(f"  ingest_cursor: {src_cursor} rows")
    print(f"  institutional_scraper_state: {src_state} rows")

    if args.dry_run:
        print("\n[DRY RUN] Skipping migration")
        return

    # Drop existing DuckDB file for clean slate
    if DST_DB.exists():
        print(f"\nRemoving existing {DST_DB}")
        DST_DB.unlink()

    # Create DuckDB schema
    print(f"\n=== Creating DuckDB: {DST_DB} ===")
    dst = duckdb.connect(str(DST_DB))
    schema_sql = SCHEMA_FILE.read_text()
    dst.execute(schema_sql)
    print("Schema applied")

    # Attach SQLite as read-only and bulk copy
    print(f"\n=== Bulk copy via sqlite_scanner ===")
    dst.execute("INSTALL sqlite_scanner")
    dst.execute("LOAD sqlite_scanner")
    dst.execute(f"ATTACH '{SRC_DB}' AS src (TYPE sqlite, READ_ONLY)")

    # Main table — cast types explicitly for boolean conversion
    t0 = time.time()
    print("Copying institutional_trades...")
    dst.execute("""
        INSERT INTO institutional_trades
        SELECT 
            id,
            tracking_id,
            ticker,
            CAST(trade_ts AS TIMESTAMPTZ) AS trade_ts,
            CAST(trade_date AS DATE) AS trade_date,
            sip_ts_ns,
            side,
            shares,
            price,
            notional_usd,
            nbbo_bid,
            nbbo_ask,
            exchange_code,
            exchange_name,
            CAST(is_dark_pool AS BOOLEAN) AS is_dark_pool,
            CAST(is_block AS BOOLEAN) AS is_block,
            CAST(is_sweep AS BOOLEAN) AS is_sweep,
            CAST(is_cross AS BOOLEAN) AS is_cross,
            CAST(is_algo AS BOOLEAN) AS is_algo,
            CAST(is_closing_auction AS BOOLEAN) AS is_closing_auction,
            CAST(is_canceled AS BOOLEAN) AS is_canceled,
            sale_cond_codes,
            provider,
            CAST(fetched_at AS TIMESTAMPTZ) AS fetched_at
        FROM src.institutional_trades
    """)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s")

    # Cursor table
    t0 = time.time()
    print("Copying ingest_cursor...")
    dst.execute("""
        INSERT INTO ingest_cursor
        SELECT 
            ticker,
            CAST(last_trade_ts AS TIMESTAMPTZ),
            last_tracking_id,
            rows_total,
            CAST(updated_at AS TIMESTAMPTZ)
        FROM src.ingest_cursor
    """)
    print(f"  done in {time.time()-t0:.1f}s")

    # Scraper state
    t0 = time.time()
    print("Copying institutional_scraper_state...")
    dst.execute("""
        INSERT INTO institutional_scraper_state
        SELECT 
            id,
            CAST(last_poll_at AS TIMESTAMPTZ),
            last_provider,
            last_row_count,
            last_ticker_count,
            last_error,
            CAST(updated_at AS TIMESTAMPTZ)
        FROM src.institutional_scraper_state
    """)
    print(f"  done in {time.time()-t0:.1f}s")

    # Set the sequence to current max(id)+1 for future inserts
    max_id = dst.execute("SELECT COALESCE(MAX(id), 0) FROM institutional_trades").fetchone()[0]
    print(f"\nSetting sequence start to {max_id + 1}")
    dst.execute(f"DROP SEQUENCE IF EXISTS seq_institutional_trades_id")
    dst.execute(f"CREATE SEQUENCE seq_institutional_trades_id START {max_id + 1}")

    # Verify
    print(f"\n=== Verification ===")
    dst_main = dst.execute("SELECT COUNT(*) FROM institutional_trades").fetchone()[0]
    dst_cursor = dst.execute("SELECT COUNT(*) FROM ingest_cursor").fetchone()[0]
    dst_state = dst.execute("SELECT COUNT(*) FROM institutional_scraper_state").fetchone()[0]
    print(f"  institutional_trades: src={src_main:,} dst={dst_main:,} match={src_main==dst_main}")
    print(f"  ingest_cursor: src={src_cursor} dst={dst_cursor} match={src_cursor==dst_cursor}")
    print(f"  institutional_scraper_state: src={src_state} dst={dst_state} match={src_state==dst_state}")

    assert src_main == dst_main, "Row count mismatch on main table"
    assert src_cursor == dst_cursor, "Row count mismatch on cursor"
    assert src_state == dst_state, "Row count mismatch on scraper_state"

    # Sample data spot-check
    print(f"\n=== Sample verification ===")
    src_sample = sqlite3.connect(SRC_DB, timeout=30).execute(
        "SELECT id, ticker, trade_ts, notional_usd FROM institutional_trades ORDER BY id LIMIT 3"
    ).fetchall()
    dst_sample = dst.execute(
        "SELECT id, ticker, trade_ts, notional_usd FROM institutional_trades ORDER BY id LIMIT 3"
    ).fetchall()
    print("SQLite first 3:")
    for r in src_sample:
        print(f"  {r}")
    print("DuckDB first 3:")
    for r in dst_sample:
        print(f"  {r}")

    # Date range
    src_range = sqlite3.connect(SRC_DB, timeout=30).execute(
        "SELECT MIN(trade_ts), MAX(trade_ts) FROM institutional_trades"
    ).fetchone()
    dst_range = dst.execute(
        "SELECT MIN(trade_ts), MAX(trade_ts) FROM institutional_trades"
    ).fetchone()
    print(f"\nDate range SQLite: {src_range[0]} to {src_range[1]}")
    print(f"Date range DuckDB: {dst_range[0]} to {dst_range[1]}")

    # File size
    src_size_mb = SRC_DB.stat().st_size / 1024 / 1024
    dst_size_mb = DST_DB.stat().st_size / 1024 / 1024
    print(f"\nFile sizes: SQLite={src_size_mb:.1f}MB DuckDB={dst_size_mb:.1f}MB (ratio: {dst_size_mb/src_size_mb:.2f}x)")

    dst.close()
    print("\nMigration complete.")


if __name__ == "__main__":
    main()
