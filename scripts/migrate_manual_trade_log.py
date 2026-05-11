#!/usr/bin/env python3
"""
scripts/migrate_manual_trade_log.py
─────────────────────────────────────────────────────────────────
Creates manual_trade_log table for tracking executed trades
(May 11 2026, Sprint 1 Day 4).

Idempotent: skips creation if table already exists. Safe to run 
multiple times. Safe to run on existing DB without manual_trade_log.

Schema per docs/ROADMAP_HYBRID_ADVISOR.md §6.1:
  - id              INTEGER PRIMARY KEY AUTOINCREMENT
  - ticker          TEXT    NOT NULL
  - side            TEXT    NOT NULL CHECK(BUY|SELL|TRIM)
  - shares          REAL    NOT NULL
  - price           REAL    NOT NULL  (fill price)
  - fill_time       TEXT    NOT NULL  (ISO 8601)
  - notes           TEXT              (nullable)
  - signal_id       INTEGER           (nullable FK to predictions.id)
  - suggested_price REAL              (price suggested by system)
  - created_at      TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP

Indexes:
  - idx_mtl_ticker      (per-ticker history queries)
  - idx_mtl_fill_time   (time-range queries)
  - idx_mtl_side        (filter by BUY/SELL/TRIM)

Usage:
  python scripts/migrate_manual_trade_log.py [db_path]
  
  Default db_path: accuracy.db
─────────────────────────────────────────────────────────────────
"""
import sqlite3
import sys
from pathlib import Path

TABLE_NAME = "manual_trade_log"

CREATE_TABLE_DDL = """
CREATE TABLE manual_trade_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    side            TEXT    NOT NULL CHECK(side IN ('BUY', 'SELL', 'TRIM')),
    shares          REAL    NOT NULL,
    price           REAL    NOT NULL,
    fill_time       TEXT    NOT NULL,
    notes           TEXT,
    signal_id       INTEGER,
    suggested_price REAL,
    created_at      TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (signal_id) REFERENCES predictions(id)
)
"""

INDEXES = [
    ("idx_mtl_ticker",    "manual_trade_log(ticker)"),
    ("idx_mtl_fill_time", "manual_trade_log(fill_time)"),
    ("idx_mtl_side",      "manual_trade_log(side)"),
]


def table_exists(conn, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,)
    )
    return cur.fetchone() is not None


def index_exists(conn, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (name,)
    )
    return cur.fetchone() is not None


def migrate(db_path: str) -> dict:
    """
    Apply migration to db_path.
    Returns dict with stats: {table_created, indexes_created, indexes_skipped}.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    try:
        table_created = False
        indexes_created = 0
        indexes_skipped = 0
        
        # Step 1: Create table if not exists
        if table_exists(conn, TABLE_NAME):
            print(f"  ✓ Table {TABLE_NAME} already exists")
        else:
            conn.execute(CREATE_TABLE_DDL)
            print(f"  + Created table {TABLE_NAME}")
            table_created = True
        
        # Step 2: Create indexes if not exist
        for idx_name, idx_def in INDEXES:
            if index_exists(conn, idx_name):
                print(f"  ✓ Index {idx_name:25s} already exists")
                indexes_skipped += 1
            else:
                conn.execute(f"CREATE INDEX {idx_name} ON {idx_def}")
                print(f"  + Created index {idx_name:25s} on {idx_def}")
                indexes_created += 1
        
        conn.commit()
        
        # Verify final state
        return {
            "table_created":    table_created,
            "indexes_created":  indexes_created,
            "indexes_skipped":  indexes_skipped,
            "table_exists":     table_exists(conn, TABLE_NAME),
            "all_indexes":      all(index_exists(conn, name) for name, _ in INDEXES),
        }
    finally:
        conn.close()


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "accuracy.db"
    
    print(f"=" * 60)
    print(f"  Manual Trade Log Migration — May 11 2026")
    print(f"  DB: {db_path}")
    print(f"=" * 60)
    print()
    
    stats = migrate(db_path)
    
    print()
    print(f"Result:")
    print(f"  Table created:   {'✅ YES' if stats['table_created'] else 'pre-existing'}")
    print(f"  Indexes added:   {stats['indexes_created']}/3")
    print(f"  Already existed: {stats['indexes_skipped']}/3")
    print(f"  Table present:   {'✅ YES' if stats['table_exists'] else '❌ NO'}")
    print(f"  All indexes:     {'✅ YES' if stats['all_indexes'] else '❌ NO'}")
    print()
    
    if stats['table_created'] or stats['indexes_created'] > 0:
        print(f"Migration applied. Run again to verify idempotency.")
    else:
        print(f"DB already migrated. No changes needed.")


if __name__ == "__main__":
    main()
