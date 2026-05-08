#!/usr/bin/env python3
"""
scripts/migrate_predictions_v2.py
─────────────────────────────────────────────────────────────────
Adds 9 columns to predictions table for multiplier audit + 
validator reconstruction (May 8 2026).

Idempotent: skips columns that already exist. Safe to run multiple
times. Safe to run on partially-migrated DB.

Columns added:
  - risk_mult, sent_mult, regime_mult, options_mult, squeeze_mult,
    intraday_mult, fg_mult  (7 multipliers, REAL, NULL-able)
  - gate_block              (INTEGER 0/1, NULL-able)
  - prob_eff_uncapped       (REAL, NULL-able)

All columns NULL-able so old rows have NULL values.
Validator must skip rows where gate_block IS NULL.

Usage:
  python scripts/migrate_predictions_v2.py [db_path]
  
  Default db_path: accuracy.db
─────────────────────────────────────────────────────────────────
"""
import sqlite3
import sys
from pathlib import Path

NEW_COLUMNS = [
    ("risk_mult",         "REAL"),
    ("sent_mult",         "REAL"),
    ("regime_mult",       "REAL"),
    ("options_mult",      "REAL"),
    ("squeeze_mult",      "REAL"),
    ("intraday_mult",     "REAL"),
    ("fg_mult",           "REAL"),
    ("gate_block",        "INTEGER"),
    ("prob_eff_uncapped", "REAL"),
]


def get_existing_columns(conn) -> set:
    """Read current schema via PRAGMA table_info."""
    cur = conn.execute("PRAGMA table_info(predictions)")
    return {row[1] for row in cur.fetchall()}


def migrate(db_path: str) -> dict:
    """
    Apply migration to db_path.
    Returns dict with stats: {added, skipped, total_after}.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    try:
        # Verify predictions table exists
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        )
        if not cur.fetchone():
            raise RuntimeError("predictions table doesn't exist in this DB")
        
        existing = get_existing_columns(conn)
        added = 0
        skipped = 0
        
        for col, typ in NEW_COLUMNS:
            if col in existing:
                print(f"  ✓ {col:20s} already exists ({typ})")
                skipped += 1
                continue
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")
                print(f"  + Added {col:20s} {typ}")
                added += 1
            except sqlite3.OperationalError as e:
                print(f"  ✗ Failed to add {col}: {e}")
                raise
        
        conn.commit()
        
        # Verify final state
        final_columns = get_existing_columns(conn)
        return {
            "added": added,
            "skipped": skipped,
            "total_columns_after": len(final_columns),
            "all_present": all(col in final_columns for col, _ in NEW_COLUMNS),
        }
    finally:
        conn.close()


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "accuracy.db"
    
    print(f"=" * 60)
    print(f"  Predictions Schema Migration v2 — May 8 2026")
    print(f"  DB: {db_path}")
    print(f"=" * 60)
    print()
    
    stats = migrate(db_path)
    
    print()
    print(f"Result:")
    print(f"  Columns added:   {stats['added']}/9")
    print(f"  Already existed: {stats['skipped']}/9")
    print(f"  Total columns:   {stats['total_columns_after']}")
    print(f"  All 9 present:   {'✅ YES' if stats['all_present'] else '❌ NO'}")
    print()
    
    if stats['added'] > 0:
        print(f"Migration applied. Run again to verify idempotency (should add 0).")
    else:
        print(f"DB already at v2. No changes needed.")


if __name__ == "__main__":
    main()
