#!/usr/bin/env python3
"""
scripts/refresh_economic_calendar.py

Refreshes accuracy.db.economic_calendar with next 45 days of UW economic
events. Designed to be cron-scheduled M/W/F so risk_gate.py and Events page
read from DB instead of hitting UW per-call.

Usage:
    python scripts/refresh_economic_calendar.py
    python scripts/refresh_economic_calendar.py --days 45
    python scripts/refresh_economic_calendar.py --verbose
"""
import argparse
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.uw_client import uw_get

DB_PATH = Path(__file__).parent.parent / "accuracy.db"


def fetch_uw_events(start: str, end: str) -> list[dict]:
    """Fetch raw UW economic-calendar events between start and end (YYYY-MM-DD)."""
    payload = uw_get(
        "/api/market/economic-calendar",
        params={"from": start, "to": end},
    )
    if payload is None:
        return []
    return payload.get("data", [])


def upsert_events(events: list[dict], verbose: bool = False) -> tuple[int, int]:
    """
    Insert/replace events into economic_calendar.
    Returns (inserted, total_attempted).
    """
    if not events:
        return 0, 0

    now_iso = datetime.now().isoformat()
    inserted = 0
    skipped = 0

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for e in events:
            # UW returns "time" as full ISO datetime: "2026-05-15T13:15:00Z"
            time_str = e.get("time") or ""
            event_date = time_str[:10] if time_str else (e.get("date") or "")[:10]
            title      = e.get("event") or e.get("name") or "Economic Event"
            if not event_date or not title:
                skipped += 1
                continue

            # Extract HH:MM from ISO datetime
            event_time = time_str[11:16] if len(time_str) >= 16 else ""

            # UW doesn't tag impact — derive from title keywords
            title_upper = title.upper()
            HIGH_KEYWORDS = ("FOMC", "FED ", "FEDERAL RESERVE", "CPI", "CONSUMER PRICE",
                             "PPI", "PRODUCER PRICE", "NONFARM", "UNEMPLOYMENT RATE",
                             "GDP", "JOBLESS", "RETAIL SALES", "PCE")
            MEDIUM_KEYWORDS = ("JOBS", "EMPLOYMENT", "INFLATION", "TRADE", "MANUFACTURING",
                               "ISM", "PMI", "CONSUMER SENTIMENT", "HOUSING", "DURABLE",
                               "INDUSTRIAL PRODUCTION", "POWELL", "FED SPEAKER", "SPEECH",
                               "TESTIMONY", "TREASURY")
            if any(k in title_upper for k in HIGH_KEYWORDS):
                impact = "High"
            elif any(k in title_upper for k in MEDIUM_KEYWORDS):
                impact = "Medium"
            else:
                impact = "Low"

            country  = e.get("country") or "US"  # UW endpoint is US-only
            forecast = e.get("forecast") or ""
            previous = e.get("prev") or e.get("previous") or "" 

            try:
                cur.execute("""
                    INSERT OR REPLACE INTO economic_calendar
                        (event_date, event_time, title, impact, country,
                         forecast, previous, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (event_date, event_time, title, impact, country,
                      forecast, previous, now_iso))
                inserted += 1
                if verbose:
                    print(f"  {event_date} [{impact:6s}] {title[:60]}")
            except Exception as ex:
                skipped += 1
                if verbose:
                    print(f"  SKIP {event_date} {title[:40]}: {ex}")

        conn.commit()
    return inserted, len(events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=45,
                        help="Days forward to fetch (default: 45)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each event written")
    args = parser.parse_args()

    today    = date.today()
    end_date = today + timedelta(days=args.days)
    start_s  = today.isoformat()
    end_s    = end_date.isoformat()

    print(f"[refresh_economic_calendar] Fetching UW events {start_s} → {end_s}")
    events = fetch_uw_events(start_s, end_s)
    print(f"[refresh_economic_calendar] UW returned {len(events)} events")

    if not events:
        print("[refresh_economic_calendar] No events — keeping existing DB rows")
        return 0

    inserted, total = upsert_events(events, verbose=args.verbose)
    print(f"[refresh_economic_calendar] Inserted/replaced {inserted}/{total} events")

    # Verify
    with sqlite3.connect(DB_PATH) as conn:
        n_rows, earliest, latest = conn.execute("""
            SELECT COUNT(*), MIN(event_date), MAX(event_date)
            FROM economic_calendar
            WHERE event_date >= ?
        """, (start_s,)).fetchone()
        print(f"[refresh_economic_calendar] DB now has {n_rows} forward events "
              f"({earliest} → {latest})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
