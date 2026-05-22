"""
data/etl_eightk_items.py — Backfill 8-K Item codes for all tracked tickers.

For each ticker, fetches recent 8-K filings via edgartools and stores
the Item codes (2.02, 5.02, 1.01, etc) into earnings.db.eightk_items.

Backfill takes ~20-30 min for 125 tickers (SEC rate limit 10 req/sec).
Idempotent — re-runs only insert new (ticker, accession, item) rows.

Foreign filers (ASML, AZN, NIO, etc) don't file 8-Ks. Skipped.

Session E Phase 3, May 22 2026.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# edgartools requires identity
from edgar import set_identity, Company

set_identity("atom_research atom_quant@research.local")

# Logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

DB_PATH = Path(os.environ.get("EARNINGS_DB_PATH", "earnings.db"))

# Foreign filers (file 6-K instead of 8-K) — skip
FOREIGN_TICKERS = {
    "ASML", "AZN", "NIO", "NOK", "NVO", "TSM", "ARM",
    "FVRR", "MNDY", "NVMI",
}


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eightk_items (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT    NOT NULL,
            accession       TEXT    NOT NULL,
            filing_date     TEXT    NOT NULL,
            item_code       TEXT    NOT NULL,
            created_at      TEXT    NOT NULL,
            UNIQUE(ticker, accession, item_code)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eightk_ticker_date ON eightk_items(ticker, filing_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eightk_item ON eightk_items(item_code)")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()


def ingest_ticker(conn: sqlite3.Connection, ticker: str, max_filings: int = 50) -> tuple[int, int]:
    """
    Fetch recent 8-K filings for ticker and store Item codes.
    Returns (filings_processed, items_inserted).
    """
    if ticker in FOREIGN_TICKERS:
        log.info(f"  skip {ticker} (foreign filer, files 6-K)")
        return 0, 0

    try:
        c = Company(ticker)
        filings = c.get_filings(form="8-K").head(max_filings)
    except Exception as e:
        log.error(f"  {ticker}: SEC fetch failed: {type(e).__name__}: {e}")
        return 0, 0

    n_filings = 0
    n_items = 0
    now = datetime.utcnow().isoformat(timespec="seconds")

    for f in filings:
        try:
            obj = f.obj()
            if obj is None:
                continue
            items = getattr(obj, 'items', None)
            if not items:
                continue
            filing_date_str = f.filing_date.isoformat() if hasattr(f.filing_date, 'isoformat') else str(f.filing_date)
            accession = f.accession_no
            n_filings += 1
            for raw_item in items:
                # 'Item 2.02' -> '2.02'
                item_code = raw_item.replace("Item ", "").strip()
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO eightk_items "
                        "(ticker, accession, filing_date, item_code, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (ticker, accession, filing_date_str, item_code, now)
                    )
                    if conn.total_changes > 0:
                        n_items += 1
                except sqlite3.Error as e:
                    log.error(f"  {ticker} insert fail: {e}")
        except Exception as e:
            log.error(f"  {ticker} filing parse fail: {type(e).__name__}: {e}")
            continue

    conn.commit()
    return n_filings, n_items


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m data.etl_eightk_items <ticker1> [ticker2] ...")
        print("       python -m data.etl_eightk_items --all  (uses tickers.txt)")
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
    init_db(conn)

    total_filings = 0
    total_items = 0
    start = time.time()
    for i, t in enumerate(tickers, 1):
        log.info(f"[{i}/{len(tickers)}] {t}")
        f_count, i_count = ingest_ticker(conn, t)
        total_filings += f_count
        total_items += i_count

    conn.close()
    elapsed = time.time() - start
    log.info(f"DONE — {total_filings} filings, {total_items} items, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
