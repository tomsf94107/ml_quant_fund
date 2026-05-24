"""
data/etl_polygon_revenue.py — Backfill rev_actual into earnings.db.earnings_surprises
                              using Polygon Financials API.

Polygon's reference/financials endpoint returns quarterly + TTM data with
revenues, EPS, etc. Free on existing Polygon API key.

Matches Polygon quarter `end_date` to existing earnings_surprises.report_date
via fuzzy ±30 day window (fiscal calendars can differ).

Rate limited: 5 calls/min → ~25 min for 125 tickers.
Foreign filers (ASML, AZN, NIO, NOK, NVO, TSM, ARM, FVRR, MNDY, NVMI) excluded.
ETFs (SPY, XLE, etc) excluded.

Session E Phase 2, May 22 2026.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

# Logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

DB_PATH = Path(os.environ.get("EARNINGS_DB_PATH", "earnings.db"))

# Foreign filers + ETFs: no quarterly Polygon data
SKIP_TICKERS = {
    "ASML", "AZN", "NIO", "NOK", "NVO", "TSM", "ARM",
    "FVRR", "MNDY", "NVMI",
    "SPY", "XLE", "XLF", "XLI", "XLU", "XLV", "XLK", "XLP", "XLY",
}


def _load_api_key() -> str:
    """Load Polygon API key from .env or env."""
    key = os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY")
    if not key and Path(".env").exists():
        for line in open(".env"):
            for var in ("POLYGON_API_KEY", "MASSIVE_API_KEY"):
                if line.startswith(var):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return key or ""


def fetch_quarterly(ticker: str, api_key: str, limit: int = 100) -> list[dict]:
    """Fetch last `limit` quarterly financials. Returns [] on failure."""
    url = f"https://api.polygon.io/vX/reference/financials"
    params = {"ticker": ticker, "timeframe": "quarterly", "limit": limit, "apiKey": api_key}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 429:
            log.warning(f"  {ticker}: rate limited, sleeping 15s")
            time.sleep(15)
            r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            log.error(f"  {ticker}: HTTP {r.status_code}: {r.text[:200]}")
            return []
        return r.json().get("results", [])
    except requests.RequestException as e:
        log.error(f"  {ticker}: request failed: {e}")
        return []


def update_revenue(conn: sqlite3.Connection, ticker: str, quarter_data: dict) -> int:
    """
    Match Polygon quarter to earnings_surprises.report_date within ±30 days.
    If match: UPDATE rev_actual where NULL.
    If no match: INSERT new row using Polygon end_date as report_date.

    Returns 1 if any row was inserted or updated, 0 otherwise.

    Updated May 24 2026: now upserts so historical quarters (2009-2024) get
    INSERTED if not already present. Previously only updated existing rows.
    """
    inc = quarter_data.get("financials", {}).get("income_statement", {})
    rev = inc.get("revenues", {}).get("value")
    if rev is None:
        return 0

    end_date_str = quarter_data.get("end_date")
    if not end_date_str:
        return 0

    end_date = datetime.fromisoformat(end_date_str.replace("Z", ""))
    window_start = end_date - timedelta(days=30)
    window_end = end_date + timedelta(days=30)

    # First: try to UPDATE existing matching row
    cur = conn.execute(
        """
        UPDATE earnings_surprises
        SET rev_actual = ?
        WHERE ticker = ?
          AND date(report_date) >= date(?)
          AND date(report_date) <= date(?)
          AND rev_actual IS NULL
        """,
        (float(rev), ticker, window_start.date().isoformat(), window_end.date().isoformat())
    )
    if cur.rowcount > 0:
        return cur.rowcount

    # No matching row OR existing row already has rev_actual: check existence
    # before INSERT to avoid creating duplicate of a row that already has rev
    exists = conn.execute(
        """
        SELECT 1 FROM earnings_surprises
        WHERE ticker = ?
          AND date(report_date) >= date(?)
          AND date(report_date) <= date(?)
        LIMIT 1
        """,
        (ticker, window_start.date().isoformat(), window_end.date().isoformat())
    ).fetchone()

    if exists:
        # Row exists with rev_actual already populated — skip
        return 0

    # INSERT new row with Polygon end_date as report_date
    conn.execute(
        """
        INSERT INTO earnings_surprises (
            ticker, report_date, eps_actual, eps_estimate, eps_surprise,
            eps_surprise_pct, rev_actual, rev_estimate, rev_surprise, created_at
        ) VALUES (?, ?, NULL, NULL, NULL, NULL, ?, NULL, NULL, ?)
        """,
        (
            ticker,
            end_date.date().isoformat() + " 00:00:00",
            float(rev),
            datetime.utcnow().isoformat(),
        )
    )
    return 1


def ingest_ticker(conn: sqlite3.Connection, ticker: str, api_key: str) -> tuple[int, int]:
    """Returns (quarters_received, rows_updated)."""
    if ticker in SKIP_TICKERS:
        return 0, 0

    quarters = fetch_quarterly(ticker, api_key)
    if not quarters:
        return 0, 0

    n_updated = 0
    for q in quarters:
        n_updated += update_revenue(conn, ticker, q)

    conn.commit()
    return len(quarters), n_updated


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m data.etl_polygon_revenue <ticker1> [ticker2] ...")
        print("       python -m data.etl_polygon_revenue --all")
        sys.exit(1)

    api_key = _load_api_key()
    if not api_key:
        log.error("No POLYGON_API_KEY or MASSIVE_API_KEY found in env or .env")
        sys.exit(1)

    if sys.argv[1] == "--all":
        tickers_file = Path("tickers.txt")
        if not tickers_file.exists():
            log.error("tickers.txt not found")
            sys.exit(1)
        tickers = [t.strip() for t in tickers_file.read_text().splitlines() if t.strip()]
    else:
        tickers = sys.argv[1:]

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    total_quarters = 0
    total_updated = 0
    start = time.time()

    for i, t in enumerate(tickers, 1):
        log.info(f"[{i}/{len(tickers)}] {t}")
        q, u = ingest_ticker(conn, t, api_key)
        total_quarters += q
        total_updated += u
        # Rate limit: 5/min = wait 12.5s between requests
        if i < len(tickers) and t not in SKIP_TICKERS:
            time.sleep(13)

    conn.close()
    elapsed = time.time() - start
    log.info(f"DONE — {total_quarters} quarters fetched, {total_updated} rows updated, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
