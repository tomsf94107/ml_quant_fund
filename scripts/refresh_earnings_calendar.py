#!/usr/bin/env python3
"""
scripts/refresh_earnings_calendar.py

Materializes accuracy.db.earnings_calendar from accuracy.db.earnings_cache.

earnings_cache (populated by daily_uw_snapshot.py) holds both past and
future earnings dates per ticker. This script aggregates a per-ticker
"next upcoming earnings" view for fast lookup by:
  • ui/pages/4_Events.py (Events dashboard)
  • signals/risk_gate.py (earnings-week BUY suppression)
  • features/builder.py (days_to_earnings fallback)

Refresh strategy: derived view, idempotent UPSERT. Safe to run multiple
times per day. Should run AFTER daily_uw_snapshot.py.

Usage:
    python scripts/refresh_earnings_calendar.py
    python scripts/refresh_earnings_calendar.py --verbose
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "accuracy.db"


def today_str() -> str:
    """Return today as YYYY-MM-DD (ET — convention per Memory #27)."""
    return date.today().isoformat()


def init_table(db_path: Path = DB_PATH) -> None:
    """Create earnings_calendar table if it doesn't exist."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS earnings_calendar (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker        TEXT    NOT NULL,
                next_date     TEXT    NOT NULL,
                next_time     TEXT,
                expected_move REAL,
                days_until    INTEGER,
                updated_at    TEXT    NOT NULL,
                UNIQUE(ticker)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_earncal_ticker ON earnings_calendar(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_earncal_date ON earnings_calendar(next_date)"
        )
        conn.commit()


def refresh(db_path: Path = DB_PATH, verbose: bool = False) -> tuple[int, int]:
    """
    Derive earnings_calendar from earnings_cache.

    For each ticker with at least one future earnings date in
    earnings_cache, write the MIN(report_date) row to earnings_calendar.

    Returns (inserted, total_attempted).
    """
    init_table(db_path)
    today = today_str()
    now_iso = datetime.now().isoformat()

    with sqlite3.connect(db_path) as conn:
        # Source query: per-ticker FRESHEST future entry.
        #
        # UW maintains multiple rows per ticker when earnings dates shift
        # (tentative date pre-announced, then confirmed/moved). The OLDER
        # tentative row stays in earnings_cache, so picking MIN(report_date)
        # gives the STALE date. Verified May 15 2026: 13/114 tickers had
        # duplicate future rows (CRWD, SNOW, QUBT, AI, ASAN, AVGO, CAVA,
        # MRVL, NIO, PL, ROST, S, ZM).
        #
        # Fix: pick row with MAX(updated_at) per ticker. Tiebreaker is
        # soonest report_date (rare, only when same daily_uw_snapshot
        # writes both rows in one run).
        rows = conn.execute("""
            SELECT ticker, report_date, report_time, expected_move
            FROM (
                SELECT ticker, report_date, report_time, expected_move,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker
                           ORDER BY updated_at DESC, report_date ASC
                       ) AS rn
                FROM earnings_cache
                WHERE report_date > ?
            )
            WHERE rn = 1
        """, (today,)).fetchall()

        # Compute days_until and upsert
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        inserted = 0
        for ticker, next_date, next_time, expected_move in rows:
            try:
                next_dt = datetime.strptime(next_date, "%Y-%m-%d")
                days_until = (next_dt - today_dt).days
            except Exception:
                if verbose:
                    print(f"  SKIP {ticker}: bad date {next_date}")
                continue

            try:
                conn.execute("""
                    INSERT OR REPLACE INTO earnings_calendar
                        (ticker, next_date, next_time, expected_move,
                         days_until, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (ticker, next_date, next_time, expected_move,
                      days_until, now_iso))
                inserted += 1
                if verbose:
                    em_str = f"{expected_move:.4f}" if expected_move else "—"
                    print(f"  {ticker:6s}  {next_date}  {next_time or '—':10s}  "
                          f"em={em_str:8s}  in {days_until}d")
            except Exception as ex:
                if verbose:
                    print(f"  SKIP {ticker}: {ex}")

        # Clean up stale: remove rows where next_date <= today
        # (in case ticker's earnings already happened since last refresh)
        stale = conn.execute(
            "DELETE FROM earnings_calendar WHERE next_date <= ?",
            (today,),
        )
        stale_count = stale.rowcount

        conn.commit()

    if verbose and stale_count > 0:
        print(f"  Removed {stale_count} stale (past-dated) rows")

    return inserted, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true",
                        help="Print each ticker written")
    args = parser.parse_args()

    print(f"[refresh_earnings_calendar] Today: {today_str()}")
    inserted, total = refresh(DB_PATH, verbose=args.verbose)
    print(f"[refresh_earnings_calendar] Wrote {inserted}/{total} tickers")

    # Verify
    with sqlite3.connect(DB_PATH) as conn:
        n_rows, earliest, latest = conn.execute("""
            SELECT COUNT(*), MIN(next_date), MAX(next_date)
            FROM earnings_calendar
        """).fetchone()
        print(f"[refresh_earnings_calendar] DB now has {n_rows} tickers "
              f"({earliest} → {latest})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
