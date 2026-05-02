# data/etl_insider_raw.py
# ─────────────────────────────────────────────────────────────────────────────
# Per-transaction Form 4 scraper writing to insider_filings_raw.
#
# This module is SEPARATE from data/etl_insider.py:
#
#   etl_insider.py        → aggregates daily, writes to insider_flows
#                           (feeds the existing signal/alert pipeline)
#   etl_insider_raw.py    → preserves every transaction, writes to
#                           insider_filings_raw (feeds the page-14 BI table)
#
# Both can run independently. Both reuse the CIK cache helpers from
# etl_insider.py so we don't duplicate the SEC ticker→CIK lookup logic.
#
# Form 4 XML parsing notes:
#   - One filing = one reporting owner (insider) + one issuer (company)
#   - One filing can contain MULTIPLE transactions in nonDerivativeTable
#     and/or derivativeTable — each becomes its own row here
#   - We capture ALL transaction codes (P/S/A/M/F/G/D/C/X) — page filters
#   - Derivative transactions (option grants, exercises) ARE captured
#     because they're often the most informative signal
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import requests

# Reuse helpers from the existing aggregating scraper
from data.etl_insider import (
    DB_PATH,
    SEC_HEADERS,
    DATA_HEADERS,
    REQUEST_DELAY,
    _get_cik,
    _list_form4_filings,
    _find_primary_xml,
    _load_cik_map,
)

# ─── Role weight mapping ─────────────────────────────────────────────────────
# Mirror the weights baked into the existing scraper / journal so the BI
# table's role_weight values are consistent with what insider_flows shows.
ROLE_WEIGHTS = {
    "CEO": 3.0, "CHIEF EXECUTIVE OFFICER": 3.0,
    "CFO": 3.0, "CHIEF FINANCIAL OFFICER": 3.0,
    "PRESIDENT": 3.0,
    "COO": 2.5, "CHIEF OPERATING OFFICER": 2.5,
    "CTO": 2.0, "CHIEF TECHNOLOGY OFFICER": 2.0,
    "CIO": 2.0, "CHIEF INFORMATION OFFICER": 2.0,
    "GENERAL COUNSEL": 1.5, "GC": 1.5,
    "EVP": 1.5, "EXECUTIVE VICE PRESIDENT": 1.5,
    "SVP": 1.5, "SENIOR VICE PRESIDENT": 1.5,
    "VP": 1.5, "VICE PRESIDENT": 1.5,
    "OFFICER": 1.5,
    "DIRECTOR": 1.0,
    "10% OWNER": 1.0, "10 PERCENT OWNER": 1.0,
}
CSUITE_THRESHOLD = 2.5  # is_csuite = 1 if role_weight >= this


def _resolve_role_weight(title: str) -> float:
    """Return weight for a title string. Picks max of any matching keyword."""
    if not title:
        return 0.5
    upper = title.upper().strip()
    best = 0.0
    for kw, weight in ROLE_WEIGHTS.items():
        if kw in upper:
            best = max(best, weight)
    return best if best > 0 else 0.5


# ─── Schema ──────────────────────────────────────────────────────────────────
# This table is created by an earlier migration (the page already references
# it). We only init it if missing, idempotently — same shape as what's
# currently in insider_trades.db.
SCHEMA_SQL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS insider_filings_raw (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    accession           TEXT    NOT NULL,
    filing_date         TEXT    NOT NULL,
    trade_date          TEXT    NOT NULL,
    insider_name        TEXT,
    insider_title       TEXT,
    role_weight         REAL    NOT NULL DEFAULT 1.0,
    transaction_code    TEXT,
    shares              REAL    NOT NULL,
    price_per_share     REAL,
    notional_usd        REAL,
    acquired_disposed   TEXT    NOT NULL,
    is_csuite           INTEGER NOT NULL DEFAULT 0,
    fetched_at          TEXT    NOT NULL,
    UNIQUE(accession, insider_name, transaction_code, trade_date, shares)
);

CREATE INDEX IF NOT EXISTS idx_raw_ticker_date ON insider_filings_raw(ticker, trade_date);
CREATE INDEX IF NOT EXISTS idx_raw_filing_date ON insider_filings_raw(filing_date);
CREATE INDEX IF NOT EXISTS idx_raw_fetched_at  ON insider_filings_raw(fetched_at);
CREATE INDEX IF NOT EXISTS idx_raw_csuite_buy  ON insider_filings_raw(is_csuite, acquired_disposed)
    WHERE is_csuite = 1 AND acquired_disposed = 'A';

-- Per-ticker high-water mark for incremental refresh
CREATE TABLE IF NOT EXISTS insider_raw_cursor (
    ticker            TEXT PRIMARY KEY,
    last_filing_date  TEXT NOT NULL,
    rows_total        INTEGER NOT NULL DEFAULT 0,
    updated_at        TEXT NOT NULL
);

-- Scraper state for the page's "last refresh" indicator
CREATE TABLE IF NOT EXISTS insider_raw_scraper_state (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    last_poll_at      TEXT,
    last_row_count    INTEGER,
    last_ticker_count INTEGER,
    last_error        TEXT,
    updated_at        TEXT
);
INSERT OR IGNORE INTO insider_raw_scraper_state (id) VALUES (1);
"""


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.executescript(SCHEMA_SQL)
    return conn


# ─── XML parser — preserves per-transaction detail ───────────────────────────

def _findtext(parent: ET.Element, path: str) -> str:
    el = parent.find(path)
    return (el.text or "").strip() if el is not None and el.text else ""


def _parse_form4_full(xml_url: str, accession: str, filing_date: str) -> list[dict]:
    """
    Parse a Form 4 XML, returning ONE dict per transaction (not aggregated).
    Captures both nonDerivative (real stock) and derivative (options) tables.

    Returns empty list on parse failure.
    """
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(xml_url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception:
        return []

    # Reporting owner (the insider). Form 4 may list multiple owners but
    # the standard case is one — take the first.
    owner = root.find("reportingOwner")
    if owner is None:
        return []

    name_el = owner.find("reportingOwnerId/rptOwnerName")
    insider_name = (name_el.text or "").strip() if name_el is not None and name_el.text else ""

    rel = owner.find("reportingOwnerRelationship")
    insider_title = ""
    if rel is not None:
        # Pick the most specific role: explicit officerTitle first, else booleans
        officer_title = _findtext(rel, "officerTitle")
        is_director = _findtext(rel, "isDirector") == "1"
        is_officer = _findtext(rel, "isOfficer") == "1"
        is_ten = _findtext(rel, "isTenPercentOwner") == "1"
        is_other = _findtext(rel, "isOther") == "1"

        if officer_title:
            insider_title = officer_title
        elif is_director:
            insider_title = "Director"
        elif is_officer:
            insider_title = "Officer"
        elif is_ten:
            insider_title = "10% Owner"
        elif is_other:
            other_text = _findtext(rel, "otherText")
            insider_title = other_text or "Other"

    role_weight = _resolve_role_weight(insider_title)
    is_csuite = 1 if role_weight >= CSUITE_THRESHOLD else 0

    # Period of report (trade date if no per-tx date)
    default_trade_date = _findtext(root, "periodOfReport")

    rows: list[dict] = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    # Walk both transaction tables
    for table_path, is_derivative in (
        ("nonDerivativeTable/nonDerivativeTransaction", False),
        ("derivativeTable/derivativeTransaction", True),
    ):
        for tx in root.findall(table_path):
            # Per-transaction date overrides filing-level
            tx_date = _findtext(tx, "transactionDate/value") or default_trade_date
            if not tx_date:
                continue

            shares_el = tx.find("transactionAmounts/transactionShares/value")
            if shares_el is None or not shares_el.text:
                continue
            try:
                shares = float(shares_el.text)
            except ValueError:
                continue
            if shares <= 0:
                continue

            price_el = tx.find("transactionAmounts/transactionPricePerShare/value")
            price = None
            if price_el is not None and price_el.text:
                try:
                    price = float(price_el.text)
                except ValueError:
                    price = None

            ad_el = tx.find("transactionAmounts/transactionAcquiredDisposedCode/value")
            ad = (ad_el.text or "").strip().upper() if ad_el is not None and ad_el.text else ""
            if ad not in ("A", "D"):
                continue

            code_el = tx.find("transactionCoding/transactionCode")
            code = (code_el.text or "").strip().upper() if code_el is not None and code_el.text else ""
            # Mark derivative-table entries with same code; the page can
            # filter on derivative-related codes (M, C) anyway
            if not code:
                continue

            notional = (shares * price) if (price is not None) else None

            rows.append({
                "accession": accession,
                "filing_date": filing_date,
                "trade_date": tx_date,
                "insider_name": insider_name,
                "insider_title": insider_title,
                "role_weight": role_weight,
                "is_csuite": is_csuite,
                "transaction_code": code,
                "shares": shares,
                "price_per_share": price,
                "notional_usd": notional,
                "acquired_disposed": ad,
                "fetched_at": fetched_at,
            })

    return rows


# ─── Cursor + state ──────────────────────────────────────────────────────────

def get_cursor(conn: sqlite3.Connection, ticker: str) -> Optional[str]:
    row = conn.execute(
        "SELECT last_filing_date FROM insider_raw_cursor WHERE ticker = ?", (ticker,)
    ).fetchone()
    return row[0] if row and row[0] else None


def update_cursor(conn: sqlite3.Connection, ticker: str,
                  last_filing_date: str, n_new_rows: int) -> None:
    conn.execute(
        """
        INSERT INTO insider_raw_cursor (ticker, last_filing_date, rows_total, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            last_filing_date = excluded.last_filing_date,
            rows_total       = insider_raw_cursor.rows_total + excluded.rows_total,
            updated_at       = excluded.updated_at
        """,
        (ticker, last_filing_date, n_new_rows,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def update_scraper_state(conn: sqlite3.Connection, total_rows: int,
                         total_tickers: int, error: Optional[str] = None) -> None:
    conn.execute(
        """
        UPDATE insider_raw_scraper_state
           SET last_poll_at      = ?,
               last_row_count    = ?,
               last_ticker_count = ?,
               last_error        = ?,
               updated_at        = ?
         WHERE id = 1
        """,
        (datetime.now(timezone.utc).isoformat(),
         total_rows, total_tickers, error,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


# ─── DB write ────────────────────────────────────────────────────────────────

_INSERT_COLS = [
    "ticker", "accession", "filing_date", "trade_date",
    "insider_name", "insider_title", "role_weight",
    "transaction_code", "shares", "price_per_share", "notional_usd",
    "acquired_disposed", "is_csuite", "fetched_at",
]
_INSERT_SQL = (
    f"INSERT OR IGNORE INTO insider_filings_raw "
    f"({','.join(_INSERT_COLS)}) "
    f"VALUES ({','.join('?' * len(_INSERT_COLS))})"
)


def upsert_rows(conn: sqlite3.Connection, ticker: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    payload = [
        tuple([ticker.upper()] + [r.get(c) for c in _INSERT_COLS[1:]])
        for r in rows
    ]
    cur = conn.executemany(_INSERT_SQL, payload)
    conn.commit()
    return max(cur.rowcount or 0, 0)


# ─── Per-ticker ingest ───────────────────────────────────────────────────────

def ingest_ticker(conn: sqlite3.Connection, ticker: str,
                  since: str,
                  inner_progress_cb: Optional[Callable[[int, int], None]] = None,
                  update_cursor_after: bool = True,
                  ) -> int:
    """
    Pull all Form 4 filings for `ticker` since ISO date `since`, parse them,
    write per-transaction rows. Returns count of new rows inserted.
    """
    ticker = ticker.upper().strip()
    cik = _get_cik(ticker)
    if not cik:
        print(f"  ⚠ {ticker}: no CIK")
        return 0

    filings = _list_form4_filings(cik, since)
    if not filings:
        return 0

    total_inserted = 0
    latest_filing_date = since
    n_filings = len(filings)

    for idx, f in enumerate(filings):
        accession = f["accession"]
        filing_date = f["filing_date"]
        xml_url = _find_primary_xml(cik, accession)
        if not xml_url:
            continue

        rows = _parse_form4_full(xml_url, accession, filing_date)
        if rows:
            n = upsert_rows(conn, ticker, rows)
            total_inserted += n

        if filing_date > latest_filing_date:
            latest_filing_date = filing_date

        if inner_progress_cb:
            try:
                inner_progress_cb(idx + 1, n_filings)
            except Exception:
                pass

    if update_cursor_after:
        update_cursor(conn, ticker, latest_filing_date, total_inserted)

    return total_inserted


# ─── Top-level entrypoint ────────────────────────────────────────────────────

def run_insider_raw_etl(
    tickers: list[str],
    days_back: int = 730,
    db_path: Path = DB_PATH,
    use_cursor: bool = True,
    progress_cb: Optional[Callable[[str, int, int, float], None]] = None,
    since_date: Optional[str] = None,
) -> dict[str, int]:
    """
    Main entrypoint.

    Args:
        tickers:       list of ticker symbols
        days_back:     lookback window if no cursor exists for a ticker (default 2y)
        db_path:       path to insider_trades.db
        use_cursor:    if True, resume from insider_raw_cursor.last_filing_date
        progress_cb:   callable(ticker, ticker_idx, total_tickers, ticker_progress_0_to_1)
        since_date:    explicit ISO date "YYYY-MM-DD" (overrides days_back, bypasses cursor)

    Returns:
        dict mapping ticker → rows inserted
    """
    conn = init_db(db_path)
    _load_cik_map()  # prime CIK cache

    # Decide window
    if since_date is not None:
        use_cursor = False
        default_since = since_date
    else:
        default_since = (datetime.now(timezone.utc).date() - timedelta(days=days_back)).isoformat()

    results: dict[str, int] = {}
    n_tickers = len(tickers)
    last_error: Optional[str] = None

    try:
        for idx, ticker in enumerate(tickers):
            ticker = ticker.upper().strip()

            if use_cursor:
                cursor_date = get_cursor(conn, ticker)
                since = cursor_date if cursor_date else default_since
            else:
                since = default_since

            try:
                def _icb(done: int, total: int) -> None:
                    if progress_cb and total > 0:
                        progress_cb(ticker, idx, n_tickers, done / total)

                n = ingest_ticker(conn, ticker, since,
                                  inner_progress_cb=_icb,
                                  update_cursor_after=use_cursor)
                results[ticker] = n
            except Exception as e:
                last_error = f"{ticker}: {e}"
                print(f"  ⚠ {ticker} failed: {e}")
                results[ticker] = 0

            if progress_cb:
                progress_cb(ticker, idx, n_tickers, 1.0)

    finally:
        total_rows = sum(results.values())
        active = sum(1 for v in results.values() if v > 0)
        update_scraper_state(conn, total_rows, active, last_error)
        conn.close()

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="SEC Form 4 per-transaction scraper → insider_filings_raw"
    )
    parser.add_argument("--tickers-file", default="tickers.txt")
    parser.add_argument("--days-back", type=int, default=730,
                        help="lookback window if no cursor (default 730 = 2 years)")
    parser.add_argument("--since", help="explicit start date YYYY-MM-DD (overrides --days-back, bypasses cursor)")
    parser.add_argument("--no-cursor", action="store_true",
                        help="ignore insider_raw_cursor; force full days-back window")
    parser.add_argument("--ticker", help="single ticker (overrides tickers-file)")
    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = [
            line.strip().upper()
            for line in Path(args.tickers_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    if args.since:
        window_str = f"since {args.since}"
    else:
        window_str = f"last {args.days_back}d"
    print(f"Form 4 raw ingest: {len(tickers)} tickers, {window_str}")

    def cb(t: str, i: int, n: int, frac: float) -> None:
        bar_len = 30
        filled = int(bar_len * frac)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{i+1}/{n}] {t:8s} [{bar}] {frac*100:.0f}%   ",
              end="", flush=True)

    res = run_insider_raw_etl(
        tickers,
        days_back=args.days_back,
        since_date=args.since,
        use_cursor=not args.no_cursor,
        progress_cb=cb,
    )
    print()
    total = sum(res.values())
    active = sum(1 for v in res.values() if v > 0)
    print(f"Done: {total:,} new rows across {active} tickers.")
