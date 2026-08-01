"""
scripts/migrations/sync_inst_to_duckdb.py
─────────────────────────────────────────────────────────────────────────────
Incremental sync of institutional_trades.db (SQLite) → institutional_trades.duckdb.

Reads max id from DuckDB, copies all SQLite rows with id > that.
Idempotent and safe to re-run.

Runs nightly as part of Pipeline A after institutional_ingest completes.

Usage:
  python -m scripts.migrations.sync_inst_to_duckdb
"""
from __future__ import annotations
import sqlite3
import sys
import time
from pathlib import Path

import duckdb

SRC_DB = Path("institutional_trades.db")
DST_DB = Path("institutional_trades.duckdb")


def main():
    if not SRC_DB.exists():
        print(f"ERROR: {SRC_DB} not found")
        sys.exit(1)
    if not DST_DB.exists():
        print(f"ERROR: {DST_DB} not found. Run migrate_inst_to_duckdb.py first.")
        sys.exit(1)

    # Get max id in DuckDB (already-synced watermark)
    dst = duckdb.connect(str(DST_DB))
    max_id = dst.execute("SELECT COALESCE(MAX(id), 0) FROM institutional_trades").fetchone()[0]
    print(f"DuckDB max id: {max_id:,}")

    # Source count of new rows
    src = sqlite3.connect(SRC_DB, timeout=30)
    new_count = src.execute(
        "SELECT COUNT(*) FROM institutional_trades WHERE id > ?",
        (max_id,)
    ).fetchone()[0]
    src.close()
    print(f"SQLite new rows since: {new_count:,}")

    if new_count == 0:
        print("Already up to date.")
        dst.close()
        return

    # Copy new rows via sqlite_scanner
    dst.execute("INSTALL sqlite_scanner")
    dst.execute("LOAD sqlite_scanner")
    dst.execute(f"ATTACH '{SRC_DB}' AS src (TYPE sqlite, READ_ONLY)")

    t0 = time.time()
    dst.execute(f"""
        INSERT INTO institutional_trades
        SELECT
            id,
            tracking_id,
            ticker,
            CAST(trade_ts AS TIMESTAMPTZ),
            CAST(trade_date AS DATE),
            sip_ts_ns,
            side,
            shares,
            price,
            notional_usd,
            nbbo_bid,
            nbbo_ask,
            exchange_code,
            exchange_name,
            CAST(is_dark_pool AS BOOLEAN),
            CAST(is_block AS BOOLEAN),
            CAST(is_sweep AS BOOLEAN),
            CAST(is_cross AS BOOLEAN),
            CAST(is_algo AS BOOLEAN),
            CAST(is_closing_auction AS BOOLEAN),
            CAST(is_canceled AS BOOLEAN),
            sale_cond_codes,
            provider,
            CAST(fetched_at AS TIMESTAMPTZ)
        FROM src.institutional_trades
        WHERE id > {max_id}
    """)
    dt = time.time() - t0
    print(f"Synced {new_count:,} rows in {dt:.1f}s")

    # Refresh cursor and state tables (small, just overwrite)
    dst.execute("DELETE FROM ingest_cursor")
    dst.execute("""
        INSERT INTO ingest_cursor
        SELECT ticker, CAST(last_trade_ts AS TIMESTAMPTZ), last_tracking_id, rows_total, CAST(updated_at AS TIMESTAMPTZ)
        FROM src.ingest_cursor
    """)
    dst.execute("DELETE FROM institutional_scraper_state")
    dst.execute("""
        INSERT INTO institutional_scraper_state
        SELECT id, CAST(last_poll_at AS TIMESTAMPTZ), last_provider, last_row_count, last_ticker_count, last_error, CAST(updated_at AS TIMESTAMPTZ)
        FROM src.institutional_scraper_state
    """)

    # Verify
    new_dst_count = dst.execute("SELECT COUNT(*) FROM institutional_trades").fetchone()[0]
    src = sqlite3.connect(SRC_DB, timeout=30)
    new_src_count = src.execute("SELECT COUNT(*) FROM institutional_trades").fetchone()[0]
    src.close()
    print(f"Total rows: SQLite={new_src_count:,} DuckDB={new_dst_count:,}")
    assert new_src_count == new_dst_count, f"Mismatch: src={new_src_count} dst={new_dst_count}"

    dst.close()
    print("Sync complete.")


if __name__ == "__main__":
    main()
