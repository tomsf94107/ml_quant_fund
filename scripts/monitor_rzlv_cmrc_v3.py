#!/usr/bin/env python3
"""
monitor_rzlv_cmrc.py — Daily merger-vote signal tracker for RZLV / CMRC.

Tracks insider, institutional, dark pool, options, and short interest activity
on both tickers ahead of CMRC's May 14, 2026 annual meeting / director election
(the de facto proxy fight on Rezolve's takeover bid).

What this gives you:
  - A console section per signal type, per ticker
  - Day-over-day deltas vs the previous run (institutional positions, short
    interest, insider net flow, large dark pool prints)
  - "FLAG" lines on anything that exceeds the configured thresholds
  - All raw data persisted to merger_monitor.db so you can re-query later

Usage:
    export UW_API_KEY="..."           # required
    export MASSIVE_API_KEY="..."      # optional; enables Massive cross-check
    python monitor_rzlv_cmrc.py                   # default since=2026-03-01
    python monitor_rzlv_cmrc.py --since 2026-02-01
    python monitor_rzlv_cmrc.py --tickers CMRC    # CMRC only
    python monitor_rzlv_cmrc.py --csv             # also dump today's snapshot

Cron-compatible. Returns nonzero on hard failure; soft-fails per-endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

UW_BASE = "https://api.unusualwhales.com"
MASSIVE_BASE = "https://api.massive.com"

DB_PATH = Path(os.environ.get("MERGER_DB", "merger_monitor.db"))

DEFAULT_TICKERS = ["CMRC", "RZLV"]

# Global today's-flags tracker. Sections call flag(...) to push events here;
# main() prints a consolidated summary at the end so signals don't get lost
# in the per-section verbose output.
TODAY_FLAGS: list[tuple[str, str, str]] = []  # (severity, ticker, message)


def flag(severity: str, ticker: str, message: str) -> None:
    """Severity: 'HIGH', 'MED', 'INFO'. Ticker can be '*' for cross-cutting."""
    TODAY_FLAGS.append((severity, ticker, message))

# Vote-relevant event timeline (informational; printed in the header)
DEAL_TIMELINE = [
    ("2026-02-??", "Rezolve's first private overture to CMRC board (rejected)"),
    ("2026-04-02", "DBLP Sea Cow (Wagner-affiliated) increases RZLV stake"),
    ("2026-04-08", "Rezolve goes public: 1 RZLV for 2 CMRC exchange offer"),
    ("2026-04-14", "CMRC adopts poison pill (10% / 20% passive thresholds)"),
    ("2026-04-15", "Rezolve investor call to CMRC shareholders"),
    ("2026-04-27", "Record date for CMRC rights distribution"),
    ("2026-05-07", "CMRC Q1 2026 earnings"),
    ("2026-05-14", "CMRC annual meeting / director election (proxy fight)"),
]

# Signal thresholds (calibrated for thin small-cap names; tune as needed)
DARKPOOL_BLOCK_MIN_USD = 250_000        # individual print
DARKPOOL_DAILY_AGGREGATE_USD = 1_000_000  # any day above this = noteworthy
INSIDER_MIN_USD = 100_000               # single insider trade
INSIDER_FLOW_DAYS = 30                  # rolling window
NEW_INST_MIN_VALUE_USD = 1_000_000      # new 13F position threshold
OPTIONS_PREMIUM_USD = 100_000           # single option trade premium
SHORT_INTEREST_DELTA_PCT = 0.05         # 5pp move in % of float short

REQUEST_TIMEOUT = 30
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class APIError(Exception):
    pass


def _uw_headers() -> dict[str, str]:
    key = os.environ.get("UW_API_KEY")
    if not key:
        raise APIError("UW_API_KEY environment variable not set")
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def uw_get(path: str, params: Optional[dict] = None) -> Any:
    """GET against Unusual Whales. Soft-fails: returns None on HTTP error and
    prints the reason. 403 is treated as a tier-access issue and printed once,
    quietly, instead of as an error stack."""
    url = f"{UW_BASE}{path}"
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=_uw_headers(),
                             params=params or {}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 403:
                print(f"  [info] {path} returned 403 (not in your UW plan); skipping")
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(1 + attempt)
                continue
    print(f"  [warn] UW {path} failed: {last_err}")
    return None


# ---------------------------------------------------------------------------
# Field-extraction helpers — UW response shapes vary across endpoints, so
# we try a list of plausible field names and take the first non-null hit.
# ---------------------------------------------------------------------------

def _first(obj: dict, *keys: str) -> Any:
    for k in keys:
        v = obj.get(k)
        if v not in (None, "", "0", 0, 0.0):
            return v
    return None


def get_shares(r: dict) -> float:
    v = _first(r, "shares", "units", "quantity", "qty", "holding_size",
               "position_size", "total_shares", "share_count", "size")
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def get_value_usd(r: dict) -> float:
    v = _first(r, "value", "value_usd", "market_value", "total_value",
               "dollar_value", "usd_value", "position_value")
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def get_date(r: dict) -> Optional[str]:
    v = _first(r, "record_date", "settlement_date", "date", "report_date",
               "as_of_date", "effective_date", "period_date", "filing_date")
    return str(v) if v else None


def get_inst_name(r: dict) -> str:
    v = _first(r, "institution_name", "name", "institution", "manager_name",
               "filer_name", "holder_name")
    return str(v) if v else "?"


# ---------------------------------------------------------------------------
# SEC EDGAR helpers — free, no auth, but require a User-Agent header
# ---------------------------------------------------------------------------

EDGAR_UA = "ML Quant Fund Research atom@example.com"
_CIK_CACHE: Optional[dict[str, int]] = None


def edgar_get(url: str, params: Optional[dict] = None) -> Any:
    """GET against SEC EDGAR. SEC requires a User-Agent identifying the
    requester or they 403."""
    headers = {"User-Agent": EDGAR_UA, "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, params=params or {},
                         timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        # SEC sometimes returns JSON, sometimes plain text — caller handles
        ct = r.headers.get("content-type", "")
        if "json" in ct:
            return r.json()
        return r.text
    except requests.RequestException as e:
        print(f"  [warn] EDGAR {url} failed: {e}")
        return None


def get_cik(ticker: str) -> Optional[int]:
    """Look up SEC CIK for a ticker symbol (cached after first call)."""
    global _CIK_CACHE
    if _CIK_CACHE is None:
        data = edgar_get("https://www.sec.gov/files/company_tickers.json")
        if not data or not isinstance(data, dict):
            return None
        # File format is {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
        _CIK_CACHE = {}
        for entry in data.values():
            t = entry.get("ticker", "").upper()
            cik = entry.get("cik_str")
            if t and cik:
                _CIK_CACHE[t] = int(cik)
    return _CIK_CACHE.get(ticker.upper())


def massive_get(path: str, params: Optional[dict] = None) -> Any:
    """GET against Massive. Soft-fails like uw_get."""
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        return None
    url = f"{MASSIVE_BASE}{path}"
    try:
        r = requests.get(url, params={**(params or {}), "apiKey": key},
                         timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"  [warn] Massive {path} failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS run_log (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts TEXT NOT NULL,
    ticker TEXT NOT NULL,
    section TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_log_ticker_section_ts
    ON run_log(ticker, section, run_ts);

CREATE TABLE IF NOT EXISTS insider_trades (
    accession_number TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    transaction_date TEXT,
    filing_date TEXT,
    insider_name TEXT,
    insider_title TEXT,
    transaction_code TEXT,
    shares REAL,
    price REAL,
    value_usd REAL,
    shares_owned_after REAL,
    raw TEXT,
    seen_ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS institutional_holdings (
    ticker TEXT NOT NULL,
    institution_name TEXT NOT NULL,
    report_date TEXT NOT NULL,
    shares REAL,
    value_usd REAL,
    pct_of_float REAL,
    seen_ts TEXT NOT NULL,
    PRIMARY KEY (ticker, institution_name, report_date)
);

CREATE TABLE IF NOT EXISTS darkpool_prints (
    ticker TEXT NOT NULL,
    executed_at TEXT NOT NULL,
    size REAL,
    price REAL,
    value_usd REAL,
    venue TEXT,
    tracking_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS short_interest_snapshots (
    ticker TEXT NOT NULL,
    record_date TEXT NOT NULL,
    short_interest REAL,
    pct_of_float REAL,
    days_to_cover REAL,
    seen_ts TEXT NOT NULL,
    PRIMARY KEY (ticker, record_date)
);

CREATE TABLE IF NOT EXISTS edgar_filings (
    accession_number TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    cik INTEGER,
    form_type TEXT,
    filing_date TEXT,
    primary_doc TEXT,
    is_filed_by_company INTEGER,
    seen_ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edgar_filings_ticker_date
    ON edgar_filings(ticker, filing_date);
"""


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Section: insider activity (Form 4)
# ---------------------------------------------------------------------------

def section_insiders(conn: sqlite3.Connection, ticker: str, since: str) -> None:
    print(f"\n=== INSIDER ACTIVITY — {ticker} (since {since}) ===")

    # Primary endpoint: /api/insider/{ticker}. May return EITHER:
    #   Shape 1: a flat list of transaction rows
    #   Shape 2: a list of insider profile objects with embedded transactions
    data = uw_get(f"/api/insider/{ticker}", params={"limit": 200})
    raw = (data or {}).get("data") or []

    ticker_upper = ticker.upper()
    rows: list[dict] = []

    for item in raw:
        # Shape 1: already a transaction row
        if "transaction_date" in item or "transaction_code" in item:
            # Defensive: if the API returned a row with an explicit ticker, only
            # accept rows that match the one we asked for. Some UW endpoints
            # ignore the path filter and return market-wide data.
            row_ticker = (item.get("ticker") or item.get("symbol") or "").upper()
            if row_ticker and row_ticker != ticker_upper:
                continue
            item["ticker"] = ticker_upper  # tag for storage
            rows.append(item)
            continue

        # Shape 2: insider profile object with embedded transactions
        embedded = (item.get("transactions") or item.get("recent_transactions") or
                    item.get("trades") or [])
        if isinstance(embedded, list):
            insider_name = (item.get("name") or item.get("insider_name") or
                            item.get("reporting_name"))
            insider_title = (item.get("title") or item.get("insider_title") or
                             item.get("position"))
            for t in embedded:
                row_ticker = (t.get("ticker") or t.get("symbol") or "").upper()
                if row_ticker and row_ticker != ticker_upper:
                    continue
                t.setdefault("insider_name", insider_name)
                t.setdefault("insider_title", insider_title)
                t["ticker"] = ticker_upper
                rows.append(t)

    # NOTE: removed the broken /api/insider/transactions fallback. That endpoint
    # ignores the ticker filter and pollutes results with unrelated names.

    # Filter to the window
    def in_window(r: dict) -> bool:
        d = (r.get("transaction_date") or r.get("filing_date") or
             r.get("trade_date") or "")[:10]
        return bool(d) and d >= since

    rows = [r for r in rows if in_window(r)]
    if not rows:
        # Fall back to the aggregated buy/sell endpoint as the only signal
        print("  No per-trade insider data returned for this ticker.")
        aggr = uw_get(f"/api/stock/{ticker}/insider-buy-sells")
        if aggr and aggr.get("data"):
            data_list = aggr["data"] if isinstance(aggr["data"], list) else [aggr["data"]]
            # Display only the first/most-recent entry (was showing duplicates)
            entry = data_list[0] if data_list else {}
            period = entry.get("period") or entry.get("date") or "latest"
            buys = entry.get("purchases", entry.get("buy_count", "?"))
            sells = entry.get("sales", entry.get("sell_count", "?"))
            print(f"  UW aggregated [{period}]: buys={buys}  sells={sells}")
            # Track for flags: insider buying is the most predictive signal
            try:
                if int(buys) > 0:
                    flag("MED", ticker, f"UW reports {buys} aggregated insider buy(s) "
                                        f"in latest period — verify on Form 4")
            except (TypeError, ValueError):
                pass
        return

    # Helpers for messy field names
    def get_shares_from_row(r: dict) -> float:
        for k in ("shares", "quantity", "qty", "share_count", "transaction_shares",
                  "amount", "size"):
            v = r.get(k)
            if v not in (None, "", 0, 0.0, "0"):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def get_price_from_row(r: dict) -> float:
        for k in ("price", "price_per_share", "transaction_price",
                  "average_price", "avg_price"):
            v = r.get(k)
            if v not in (None, "", 0, 0.0, "0"):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def get_value_from_row(r: dict, sh: float, px: float) -> float:
        if sh and px:
            return sh * px
        for k in ("transaction_value", "value", "value_usd", "total_value",
                  "dollar_value"):
            v = r.get(k)
            if v not in (None, "", 0, 0.0, "0"):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return 0.0

    # Persist + summarize
    cur = conn.cursor()
    new_count = 0
    enriched: list[dict] = []
    for r in rows:
        sh = get_shares_from_row(r)
        px = get_price_from_row(r)
        v = get_value_from_row(r, sh, px)
        r["_shares"], r["_price"], r["_value"] = sh, px, v
        enriched.append(r)
        try:
            cur.execute("""
                INSERT OR IGNORE INTO insider_trades
                (accession_number, ticker, transaction_date, filing_date,
                 insider_name, insider_title, transaction_code, shares, price,
                 value_usd, shares_owned_after, raw, seen_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r.get("accession_number") or
                    f"{ticker_upper}-{r.get('transaction_date')}-"
                    f"{r.get('insider_name')}-{sh}-{px}",
                ticker_upper,
                r.get("transaction_date") or r.get("filing_date") or r.get("trade_date"),
                r.get("filing_date"),
                r.get("insider_name") or r.get("owner_name") or r.get("reporting_name"),
                r.get("insider_title") or r.get("officer_title") or r.get("position"),
                r.get("transaction_code") or r.get("code"),
                sh, px, v,
                float(r.get("shares_owned_after") or r.get("shares_owned") or 0),
                json.dumps(r, default=str),
                now_iso(),
            ))
            if cur.rowcount:
                new_count += 1
        except sqlite3.Error as e:
            print(f"  [warn] insert failed: {e}")
    conn.commit()

    # Buy / sell aggregates by Form 4 transaction code:
    #   P = open-market purchase (high signal)
    #   S = open-market sale
    #   A = grant/award (compensation, low signal — exclude from net)
    #   M = exercise of derivative (mostly cashless, low signal alone)
    #   F = tax-withholding (forced, low signal)
    #   D = disposition (varies)
    real_buys = sum(r["_value"] for r in enriched
                    if (r.get("transaction_code") or "").upper() == "P")
    real_sells = sum(r["_value"] for r in enriched
                     if (r.get("transaction_code") or "").upper() == "S")
    print(f"  Trades in window: {len(enriched)} (new since last run: {new_count})")
    print(f"  Open-market BUYS  (P):  ${real_buys:>14,.0f}")
    print(f"  Open-market SELLS (S):  ${real_sells:>14,.0f}")
    print(f"  Net (P - S):            ${real_buys - real_sells:>14,.0f}")
    print(f"  (Excludes A/M/F/D codes which are compensation/exercise/tax)")

    # Top trades by absolute value
    rows_sorted = sorted(enriched, key=lambda r: r["_value"], reverse=True)[:15]
    print("  Top trades:")
    for r in rows_sorted:
        code = (r.get("transaction_code") or r.get("code") or "?").upper()
        who = (r.get("insider_name") or r.get("owner_name") or
               r.get("reporting_name") or "?")[:32]
        title = (r.get("insider_title") or r.get("officer_title") or
                 r.get("position") or "")[:30]
        tag = "FLAG" if r["_value"] >= INSIDER_MIN_USD and code in ("P", "S") else "    "
        d = (r.get("transaction_date") or r.get("filing_date") or
             r.get("trade_date") or "?")[:10]
        print(f"  {tag}  {d}  {code}  {who:<32}  {title:<30}"
              f"  {r['_shares']:>10,.0f} @ ${r['_price']:>7,.2f}  = ${r['_value']:>12,.0f}")

    # Aggregated context endpoint
    aggr = uw_get(f"/api/stock/{ticker}/insider-buy-sells")
    if aggr and aggr.get("data"):
        last = aggr["data"][0] if isinstance(aggr["data"], list) else aggr["data"]
        print(f"  UW aggregated last period: "
              f"buys={last.get('purchases', last.get('buy_count', '?'))}  "
              f"sells={last.get('sales', last.get('sell_count', '?'))}")


# ---------------------------------------------------------------------------
# Section: institutional ownership (13F)
# ---------------------------------------------------------------------------

def section_institutional(conn: sqlite3.Connection, ticker: str) -> None:
    print(f"\n=== INSTITUTIONAL OWNERSHIP — {ticker} ===")

    data = uw_get(f"/api/institution/{ticker}/ownership", params={"limit": 200})
    rows = (data or {}).get("data") or []
    if not rows:
        print("  No institutional ownership data returned.")
        return

    # Pull previous snapshot from DB to compute deltas
    cur = conn.cursor()
    prev = {row[0]: row[1] for row in cur.execute(
        "SELECT institution_name, shares FROM institutional_holdings "
        "WHERE ticker = ? AND report_date = ("
        "  SELECT MAX(report_date) FROM institutional_holdings WHERE ticker = ?"
        ")", (ticker, ticker))}

    # Save current snapshot
    for r in rows:
        try:
            cur.execute("""
                INSERT OR REPLACE INTO institutional_holdings
                (ticker, institution_name, report_date, shares, value_usd, pct_of_float, seen_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                get_inst_name(r),
                get_date(r) or now_iso()[:10],   # fall back to today's date if API didn't return one
                get_shares(r),
                get_value_usd(r),
                float(r.get("pct_of_float") or r.get("percent_of_float") or 0),
                now_iso(),
            ))
        except sqlite3.Error as e:
            print(f"  [warn] insert failed: {e}")
    conn.commit()

    # Top holders by shares (with value as tiebreak in case shares=0)
    rows_sorted = sorted(rows, key=lambda r: (get_shares(r), get_value_usd(r)), reverse=True)[:15]
    print(f"  Total holders reported: {len(rows)}")
    print(f"  {'Institution':<40} {'Shares':>14} {'Δ vs prev':>14} {'Value $':>16}")
    for r in rows_sorted:
        name = get_inst_name(r)[:40]
        sh = get_shares(r)
        v = get_value_usd(r)
        delta = sh - prev.get(name, sh)
        is_new = name not in prev and bool(prev)
        tag = ""
        if is_new and v >= NEW_INST_MIN_VALUE_USD:
            tag = "  NEW"
            flag("HIGH", ticker, f"NEW institutional holder: {name} (${v:,.0f})")
        elif sh > 0 and abs(delta) > 0.05 * sh:
            tag = "  CHG"
            pct = (delta / sh) * 100 if sh else 0
            if abs(pct) > 20:  # only flag substantial moves to summary
                flag("MED", ticker,
                     f"{name} position change: {delta:+,.0f} shares ({pct:+.1f}%)")
        print(f"  {name:<40} {sh:>14,.0f} {delta:>+14,.0f} {v:>16,.0f}{tag}")


# ---------------------------------------------------------------------------
# Section: dark pool prints
# ---------------------------------------------------------------------------

def section_darkpool(conn: sqlite3.Connection, ticker: str, since: str) -> None:
    print(f"\n=== DARK POOL PRINTS — {ticker} (since {since}) ===")

    # UW's per-ticker darkpool endpoint 403s with `newer_than` parameter.
    # The /recent endpoint works fine without it. Hypothesis: the per-ticker
    # endpoint expects a single explicit `date=YYYY-MM-DD` parameter.
    # Workaround: loop dates ourselves, hitting one trading day at a time.
    # We persist into the DB so we only fetch each date once across runs.

    cur = conn.cursor()
    existing_dates = {row[0] for row in cur.execute(
        "SELECT DISTINCT substr(executed_at, 1, 10) "
        "FROM darkpool_prints WHERE ticker = ?", (ticker,)
    )}

    today = date.today()
    try:
        start_date = date.fromisoformat(since)
    except ValueError:
        start_date = today - timedelta(days=60)

    # Build list of trading days to fetch (skip weekends + already-fetched)
    dates_to_fetch: list[str] = []
    d = start_date
    while d <= today:
        if d.weekday() < 5:  # Mon-Fri
            iso = d.isoformat()
            if iso not in existing_dates:
                dates_to_fetch.append(iso)
        d += timedelta(days=1)

    # Cap new fetches per run to be polite on rate limits
    # (40k daily limit is huge but we share it with other endpoints)
    MAX_NEW_DATES_PER_RUN = 30
    if len(dates_to_fetch) > MAX_NEW_DATES_PER_RUN:
        # Prioritize the most recent dates first
        dates_to_fetch = dates_to_fetch[-MAX_NEW_DATES_PER_RUN:]

    if dates_to_fetch:
        print(f"  Fetching {len(dates_to_fetch)} new trading day(s)...")
    elif existing_dates:
        print(f"  All trading days in window already fetched ({len(existing_dates)} cached).")

    # Fetch each new date
    fetch_failed = 0
    fetch_succeeded = 0
    for iso in dates_to_fetch:
        data = uw_get(f"/api/darkpool/{ticker}",
                      params={"date": iso, "limit": 500})
        if data is None:
            fetch_failed += 1
            # If first 3 fail, assume tier issue and stop hammering
            if fetch_failed >= 3 and fetch_succeeded == 0:
                print(f"  [info] Aborting darkpool fetch after {fetch_failed} consecutive "
                      f"failures (likely a UW plan/access issue on per-ticker darkpool).")
                break
            continue
        fetch_succeeded += 1
        rows_today = data.get("data") or []
        for r in rows_today:
            try:
                sz = float(r.get("size") or 0)
                px = float(r.get("price") or 0)
                v = sz * px
                cur.execute("""
                    INSERT OR IGNORE INTO darkpool_prints
                    (ticker, executed_at, size, price, value_usd, venue, tracking_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    r.get("executed_at") or r.get("trf_executed_at") or r.get("trade_time"),
                    sz, px, v,
                    r.get("market_center") or r.get("venue"),
                    str(r.get("tracking_id")) if r.get("tracking_id") else
                    f"{ticker}-{r.get('executed_at')}-{sz}-{px}",
                ))
            except sqlite3.Error as e:
                print(f"  [warn] insert failed: {e}")
        conn.commit()

    # Now query the DB for the full window and analyze
    rows = list(cur.execute("""
        SELECT executed_at, size, price, value_usd, venue
        FROM darkpool_prints
        WHERE ticker = ? AND substr(executed_at, 1, 10) >= ?
        ORDER BY executed_at DESC
    """, (ticker, since)))

    if not rows:
        # Either fetch failed entirely or nothing in window
        if fetch_failed and fetch_succeeded == 0:
            print(f"  No dark pool data accessible for this ticker on your UW plan.")
        else:
            print(f"  No dark pool prints found in window.")
        return

    print(f"  Prints in window: {len(rows)}")

    # Aggregate by day
    by_day: dict[str, dict] = {}
    for executed_at, size, price, value, venue in rows:
        day = (executed_at or "")[:10]
        if not day:
            continue
        b = by_day.setdefault(day, {"prints": 0, "shares": 0.0,
                                    "value": 0.0, "max_print": 0.0})
        b["prints"] += 1
        b["shares"] += float(size or 0)
        b["value"] += float(value or 0)
        b["max_print"] = max(b["max_print"], float(value or 0))

    print(f"  {'Date':<12} {'Prints':>7} {'Shares':>14} {'Value $':>16} {'Max Print $':>14}")
    sorted_days = sorted(by_day.keys(), reverse=True)
    for day in sorted_days[:15]:
        b = by_day[day]
        tags = ""
        if b["value"] >= DARKPOOL_DAILY_AGGREGATE_USD:
            tags += "  HEAVY"
            flag("MED", ticker,
                 f"Heavy dark pool day {day}: ${b['value']:,.0f} on {b['prints']} prints")
        if b["max_print"] >= DARKPOOL_BLOCK_MIN_USD:
            tags += "  BLOCK"
            if "HEAVY" not in tags:  # don't double-flag
                flag("MED", ticker,
                     f"Dark pool block {day}: max single print ${b['max_print']:,.0f}")
        print(f"  {day:<12} {b['prints']:>7d} {b['shares']:>14,.0f} "
              f"{b['value']:>16,.0f} {b['max_print']:>14,.0f}{tags}")

    # Trend signal: today's vs 5-day avg vs 20-day avg
    if len(sorted_days) >= 6:
        today_val = by_day[sorted_days[0]]["value"]
        five_day_avg = sum(by_day[d]["value"] for d in sorted_days[1:6]) / 5
        if five_day_avg > 0 and today_val > 2 * five_day_avg:
            flag("MED", ticker,
                 f"Today's dark pool value (${today_val:,.0f}) is "
                 f"{today_val/five_day_avg:.1f}x prior 5-day avg")

    # Top 10 single prints in window
    big = sorted(rows, key=lambda r: float(r[3] or 0), reverse=True)[:10]
    print("  Top 10 single prints:")
    for executed_at, size, price, value, venue in big:
        if (value or 0) < DARKPOOL_BLOCK_MIN_USD:
            continue
        print(f"    {(executed_at or '')[:19]}  {size:>10,.0f} @ ${price:>7,.2f}  "
              f"= ${value:>12,.0f}  ({venue or '?'})")


# ---------------------------------------------------------------------------
# Section: options flow (note: thin for CMRC; useful for RZLV)
# ---------------------------------------------------------------------------

def section_options_flow(conn: sqlite3.Connection, ticker: str, since: str) -> None:
    print(f"\n=== OPTIONS FLOW — {ticker} (recent) ===")

    data = uw_get(f"/api/stock/{ticker}/flow-alerts", params={"limit": 100})
    rows = (data or {}).get("data") or []
    rows = [r for r in rows if (r.get("executed_at") or r.get("created_at") or "")[:10] >= since]

    if not rows:
        print(f"  No flow alerts returned (small-caps like {ticker} often have thin options).")
        return

    print(f"  Flow alerts in window: {len(rows)}")

    # Aggregate calls vs puts by premium
    call_prem = sum(float(r.get("total_premium") or r.get("premium") or 0)
                    for r in rows if (r.get("type") or r.get("option_type") or "").lower() == "call")
    put_prem = sum(float(r.get("total_premium") or r.get("premium") or 0)
                   for r in rows if (r.get("type") or r.get("option_type") or "").lower() == "put")
    pc_prem = (put_prem / call_prem) if call_prem else float("inf") if put_prem else 0

    print(f"  Call premium: ${call_prem:,.0f}")
    print(f"  Put premium:  ${put_prem:,.0f}")
    print(f"  Put/Call premium ratio: {pc_prem:.2f}")

    # Top single alerts
    big = sorted(rows, key=lambda r: float(r.get("total_premium") or r.get("premium") or 0), reverse=True)[:10]
    print("  Top alerts by premium:")
    for r in big:
        p = float(r.get("total_premium") or r.get("premium") or 0)
        if p < OPTIONS_PREMIUM_USD:
            continue
        otype = (r.get("type") or r.get("option_type") or "?").upper()
        strike = r.get("strike")
        exp = r.get("expiry") or r.get("expiration")
        print(f"    {(r.get('executed_at') or r.get('created_at') or '')[:19]}  "
              f"{otype:<5} ${strike} {exp}  prem=${p:,.0f}  "
              f"vol={r.get('volume') or '?'}  oi={r.get('open_interest') or '?'}")


# ---------------------------------------------------------------------------
# Section: short interest
# ---------------------------------------------------------------------------

def section_short_interest(conn: sqlite3.Connection, ticker: str) -> None:
    print(f"\n=== SHORT INTEREST — {ticker} ===")

    data = uw_get(f"/api/shorts/{ticker}/interest-float/v2")
    rows = (data or {}).get("data") or []
    if not rows:
        print("  No short interest data returned.")
        return

    cur = conn.cursor()
    # Persist all snapshots
    for r in rows:
        d = get_date(r)
        if not d:
            # Synthesize a stable date from index if API omitted it; without
            # a date we can't dedup, so skip persistence but still display.
            continue
        try:
            cur.execute("""
                INSERT OR REPLACE INTO short_interest_snapshots
                (ticker, record_date, short_interest, pct_of_float, days_to_cover, seen_ts)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                d,
                float(r.get("short_interest") or r.get("short_volume") or 0),
                float(r.get("short_percent_of_float") or r.get("pct_of_float") or
                      r.get("percent_of_float") or 0),
                float(r.get("days_to_cover") or r.get("dtc") or 0),
                now_iso(),
            ))
        except sqlite3.Error as e:
            print(f"  [warn] insert failed: {e}")
    conn.commit()

    print(f"  {'Date':<12} {'Short Int':>14} {'% Float':>9} {'DTC':>7}")
    # Sort newest first by whichever date field works
    sorted_rows = sorted(rows, key=lambda x: (get_date(x) or ""), reverse=True)[:8]
    for r in sorted_rows:
        d = get_date(r) or "?"
        si = float(r.get("short_interest") or r.get("short_volume") or 0)
        pct = float(r.get("short_percent_of_float") or r.get("pct_of_float") or
                    r.get("percent_of_float") or 0)
        dtc = float(r.get("days_to_cover") or r.get("dtc") or 0)
        print(f"  {d[:12]:<12} {si:>14,.0f} {pct:>8.2%} {dtc:>7.1f}")

    # Trend signal
    dated = [r for r in rows if get_date(r)]
    if len(dated) >= 2:
        srt = sorted(dated, key=lambda x: get_date(x))
        latest_pct = float(srt[-1].get("short_percent_of_float") or
                           srt[-1].get("pct_of_float") or
                           srt[-1].get("percent_of_float") or 0)
        prev_pct = float(srt[-2].get("short_percent_of_float") or
                         srt[-2].get("pct_of_float") or
                         srt[-2].get("percent_of_float") or 0)
        delta = latest_pct - prev_pct
        tag = "  FLAG" if abs(delta) >= SHORT_INTEREST_DELTA_PCT else ""
        print(f"  Δ % of float (latest vs prior snapshot): {delta:+.2%}{tag}")
        if tag:
            flag("MED", ticker, f"Short % of float moved {delta:+.2%} period-over-period")

    # Also compute share-count trend (works even when dates are missing)
    if len(rows) >= 2:
        si_latest = float(rows[0].get("short_interest") or rows[0].get("short_volume") or 0)
        si_prev = float(rows[1].get("short_interest") or rows[1].get("short_volume") or 0)
        if si_prev:
            pct_change = (si_latest - si_prev) / si_prev * 100
            print(f"  Δ short shares (latest vs prior snapshot): "
                  f"{si_latest - si_prev:+,.0f} ({pct_change:+.1f}%)")
            if abs(pct_change) >= 10:
                direction = "covering wave" if pct_change < 0 else "short build"
                flag("MED", ticker,
                     f"Significant {direction}: {pct_change:+.1f}% ({si_latest - si_prev:+,.0f} shares)")


# ---------------------------------------------------------------------------
# Section: SEC EDGAR filings (highest-signal source for proxy fights)
# ---------------------------------------------------------------------------

# Forms we care about, with severity tags for the flag summary
EDGAR_FORM_PRIORITY = {
    # Form         (severity, human-readable description)
    "SC 13D":     ("HIGH", "13D — new 5%+ position with intent to influence"),
    "SC 13D/A":   ("HIGH", "13D/A — amendment to active 13D position"),
    "SC 13G":     ("MED",  "13G — passive 5%+ position"),
    "SC 13G/A":   ("MED",  "13G/A — passive position amendment"),
    "DEFC14A":    ("HIGH", "DEFC14A — definitive contested proxy materials"),
    "DFAN14A":    ("HIGH", "DFAN14A — definitive additional contesting materials"),
    "PRRN14A":    ("HIGH", "PRRN14A — preliminary contesting proxy materials"),
    "DEFA14A":    ("MED",  "DEFA14A — additional definitive proxy materials"),
    "DEF 14A":    ("MED",  "DEF 14A — definitive proxy statement"),
    "PRE 14A":    ("MED",  "PRE 14A — preliminary proxy statement"),
    "8-K":        ("INFO", "8-K — material event"),
    "4":          ("INFO", "Form 4 — insider transaction"),
    "4/A":        ("INFO", "Form 4/A — insider transaction amendment"),
    "144":        ("INFO", "Form 144 — proposed insider sale"),
    "10-Q":       ("INFO", "10-Q — quarterly report"),
    "10-K":       ("INFO", "10-K — annual report"),
    "S-4":        ("MED",  "S-4 — registration of securities for M&A"),
    "425":        ("MED",  "425 — communications during pending M&A"),
}


def section_edgar(conn: sqlite3.Connection, ticker: str, since: str) -> None:
    print(f"\n=== SEC EDGAR FILINGS — {ticker} (since {since}) ===")

    cik = get_cik(ticker)
    if cik is None:
        print(f"  Could not find CIK for {ticker} in EDGAR ticker file.")
        return
    print(f"  CIK: {cik}")

    cur = conn.cursor()
    seen_accessions = {row[0] for row in cur.execute(
        "SELECT accession_number FROM edgar_filings WHERE ticker = ?", (ticker,)
    )}

    # Part 1: filings BY this company (its own 10-Q, DEF 14A, 8-K, etc.)
    sub_url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    data = edgar_get(sub_url)
    if not isinstance(data, dict):
        print(f"  Could not fetch EDGAR submissions for CIK {cik}.")
        return

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    fdates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    primaries = recent.get("primaryDocument", [])

    company_hits: list[dict] = []
    for i, form in enumerate(forms):
        if i >= len(fdates) or i >= len(accs):
            break
        fdate = fdates[i]
        if fdate < since:
            continue
        if form not in EDGAR_FORM_PRIORITY:
            continue
        company_hits.append({
            "form": form, "date": fdate,
            "accession": accs[i],
            "primary": primaries[i] if i < len(primaries) else "",
            "by_company": True,
        })

    # Part 2: filings ABOUT this company by THIRD PARTIES (the killer signal:
    # an activist filing a 13D ON CMRC won't show up in CMRC's submission feed).
    # Use the full-text search API.
    third_party_hits: list[dict] = []
    fts_url = "https://efts.sec.gov/LATEST/search-index"
    # Search specifically for 13D and contested-proxy forms naming this CIK
    for form_filter in ["SC 13D", "SC 13D/A", "DEFC14A", "DFAN14A", "PRRN14A"]:
        params = {
            "q": f'"{ticker}"',  # ticker symbol in filing text
            "forms": form_filter,
            "dateRange": "custom",
            "startdt": since,
            "enddt": date.today().isoformat(),
        }
        result = edgar_get(fts_url, params=params)
        if not isinstance(result, dict):
            continue
        hits = result.get("hits", {}).get("hits", [])
        for h in hits:
            src = h.get("_source", {}) or {}
            acc = (src.get("adsh") or h.get("_id") or "").replace("-", "").replace(":", "")[:30]
            if not acc:
                continue
            # Filter to filings that actually reference our company's CIK
            cik_list = src.get("ciks", []) or []
            if cik_list and str(cik) not in [str(c).lstrip("0") for c in cik_list] and \
               cik not in [int(c) for c in cik_list if str(c).isdigit()]:
                # The full-text search hit something but the filing isn't
                # actually about this CIK — skip it
                continue
            third_party_hits.append({
                "form": src.get("form") or form_filter,
                "date": src.get("file_date") or src.get("filed") or "",
                "accession": src.get("adsh") or acc,
                "primary": (src.get("display_names") or [""])[0] if src.get("display_names")
                           else "",
                "by_company": False,
            })

    all_hits = company_hits + third_party_hits

    if not all_hits:
        print(f"  No EDGAR filings found in window for {ticker}.")
        return

    # Persist all + flag NEW ones since last run
    new_hits: list[dict] = []
    for h in all_hits:
        acc = h["accession"]
        is_new = acc not in seen_accessions
        try:
            cur.execute("""
                INSERT OR IGNORE INTO edgar_filings
                (accession_number, ticker, cik, form_type, filing_date,
                 primary_doc, is_filed_by_company, seen_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (acc, ticker, cik, h["form"], h["date"], h["primary"],
                  1 if h["by_company"] else 0, now_iso()))
        except sqlite3.Error as e:
            print(f"  [warn] insert failed: {e}")
        if is_new:
            new_hits.append(h)
    conn.commit()

    # Sort by date desc and print
    print(f"  Filings in window: {len(all_hits)}  (new since last run: {len(new_hits)})")
    print(f"  {'Date':<12} {'Form':<10} {'By':<6} {'Severity':<8} Description")
    sorted_hits = sorted(all_hits, key=lambda h: h["date"], reverse=True)
    for h in sorted_hits[:25]:
        sev, desc = EDGAR_FORM_PRIORITY.get(h["form"], ("INFO", h["form"]))
        by = "self" if h["by_company"] else "3rd"
        is_new_marker = "  NEW" if h["accession"] not in seen_accessions else ""
        print(f"  {h['date']:<12} {h['form']:<10} {by:<6} {sev:<8} {desc[:50]}{is_new_marker}")
        # Flag new HIGH-severity events
        if h["accession"] not in seen_accessions:
            if sev == "HIGH":
                flag("HIGH", ticker,
                     f"NEW {h['form']} filing on {h['date']} — {desc}")
            elif sev == "MED" and not h["by_company"]:
                # 3rd-party MED filings (like 13G) are more interesting than self-filed
                flag("MED", ticker,
                     f"NEW 3rd-party {h['form']} on {h['date']} — {desc}")

    if new_hits:
        # Print URLs for the new high-severity filings
        print("  New filing URLs:")
        for h in new_hits[:10]:
            sev, _ = EDGAR_FORM_PRIORITY.get(h["form"], ("INFO", ""))
            if sev in ("HIGH", "MED"):
                acc_clean = h["accession"].replace("-", "")
                url = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                       f"&CIK={cik}&type={h['form'].replace(' ', '+')}&dateb=&owner=include&count=10")
                print(f"    [{h['form']}] {h['date']}: {url}")


# ---------------------------------------------------------------------------
# Section: peer-relative price (signal hidden in your existing OHLC data)
# ---------------------------------------------------------------------------

def _ohlc_returns(rows: list[dict]) -> dict:
    """Given OHLC rows sorted oldest-first, compute 1d/5d/20d returns from
    the latest close. Returns empty dict if not enough data."""
    if not rows:
        return {}
    closes = [float(r.get("close") or 0) for r in rows if r.get("close")]
    if len(closes) < 2:
        return {}
    out = {"latest": closes[-1]}
    out["ret_1d"] = (closes[-1] / closes[-2]) - 1 if closes[-2] else 0
    if len(closes) >= 6:
        out["ret_5d"] = (closes[-1] / closes[-6]) - 1 if closes[-6] else 0
    if len(closes) >= 21:
        out["ret_20d"] = (closes[-1] / closes[-21]) - 1 if closes[-21] else 0
    return out


def section_peer_relative(ticker: str) -> None:
    """Compares ticker returns to small-cap (IWM) and broad-market (SPY)
    benchmarks. Relative strength is signal: a stock holding flat while
    its peer set drops 2% is showing institutional bid."""
    print(f"\n=== PEER-RELATIVE PRICE — {ticker} ===")

    def fetch_returns(t: str) -> dict:
        data = uw_get(f"/api/stock/{t}/ohlc/1d", params={"limit": 30})
        rows = (data or {}).get("data") or []
        # Sort oldest first by whatever date field is populated
        rows.sort(key=lambda r: r.get("date") or r.get("market_time") or "")
        return _ohlc_returns(rows)

    t = fetch_returns(ticker)
    iwm = fetch_returns("IWM")
    spy = fetch_returns("SPY")

    if not t:
        print(f"  Insufficient OHLC data for {ticker}.")
        return

    print(f"  {'':<14} {'1d':>8} {'5d':>8} {'20d':>8}")
    def row(label: str, d: dict) -> None:
        cells = []
        for key in ("ret_1d", "ret_5d", "ret_20d"):
            v = d.get(key)
            cells.append(f"{v*100:+.2f}%" if v is not None else "    -")
        print(f"  {label:<14} {cells[0]:>8} {cells[1]:>8} {cells[2]:>8}")

    row(f"{ticker}", t)
    row("IWM (R2K)", iwm)
    row("SPY", spy)
    print("  " + "-" * 42)

    # Relative strength rows
    rel_iwm = {k: t.get(k, 0) - iwm.get(k, 0) for k in ("ret_1d", "ret_5d", "ret_20d")
               if t.get(k) is not None and iwm.get(k) is not None}
    rel_spy = {k: t.get(k, 0) - spy.get(k, 0) for k in ("ret_1d", "ret_5d", "ret_20d")
               if t.get(k) is not None and spy.get(k) is not None}
    row(f"vs IWM", rel_iwm)
    row(f"vs SPY", rel_spy)

    # Flag meaningful relative strength
    # 5d rel strength of +5% on small-cap = institutional bid pattern
    if rel_iwm.get("ret_5d", 0) >= 0.05:
        flag("MED", ticker,
             f"5-day relative strength vs IWM: +{rel_iwm['ret_5d']*100:.1f}% — "
             f"institutional accumulation pattern")
    if rel_iwm.get("ret_5d", 0) <= -0.05:
        flag("MED", ticker,
             f"5-day relative weakness vs IWM: {rel_iwm['ret_5d']*100:.1f}% — "
             f"distribution or thesis decay")
    if rel_iwm.get("ret_1d", 0) >= 0.03:
        flag("INFO", ticker,
             f"Today's relative strength vs IWM: +{rel_iwm['ret_1d']*100:.1f}%")

# ---------------------------------------------------------------------------
# Section: price + volume context
# ---------------------------------------------------------------------------

def section_price_volume(ticker: str, since: str) -> None:
    print(f"\n=== PRICE / VOLUME CONTEXT — {ticker} (since {since}) ===")

    data = uw_get(f"/api/stock/{ticker}/ohlc/1d", params={"limit": 90})
    rows = (data or {}).get("data") or []
    rows = [r for r in rows if (r.get("date") or r.get("market_time") or "")[:10] >= since]
    if not rows:
        print("  No OHLC returned.")
        return

    # Recent volume vs 20d avg
    rows_sorted = sorted(rows, key=lambda r: r.get("date") or r.get("market_time") or "")
    vols = [float(r.get("volume") or 0) for r in rows_sorted]
    closes = [float(r.get("close") or 0) for r in rows_sorted]
    if len(vols) < 21:
        print(f"  Only {len(vols)} bars; volume baseline weak.")
    else:
        avg20 = sum(vols[-21:-1]) / 20
        latest_vol = vols[-1]
        ratio = latest_vol / avg20 if avg20 else 0
        print(f"  Latest close: ${closes[-1]:.2f}   "
              f"Latest volume: {latest_vol:,.0f}   "
              f"20d avg vol: {avg20:,.0f}   "
              f"Ratio: {ratio:.2f}x")
        if ratio >= 3.0:
            print("  FLAG: volume >= 3x 20d average")
            flag("MED", ticker,
                 f"Today's volume {ratio:.1f}x 20d avg ({latest_vol:,.0f} vs {avg20:,.0f})")
        elif ratio >= 2.0:
            print("  FLAG: volume >= 2x 20d average")

    # High-volume days in window
    big_days = sorted(rows_sorted,
                      key=lambda r: float(r.get("volume") or 0), reverse=True)[:5]
    print("  Top 5 volume days in window:")
    for r in big_days:
        d = (r.get("date") or r.get("market_time") or "")[:10]
        v = float(r.get("volume") or 0)
        c = float(r.get("close") or 0)
        o = float(r.get("open") or 0)
        chg = (c - o) / o * 100 if o else 0
        print(f"    {d}  vol={v:>14,.0f}  open=${o:>6.2f} close=${c:>6.2f}  ({chg:+.2f}%)")


# ---------------------------------------------------------------------------
# Optional: Massive cross-check on session block trades
# ---------------------------------------------------------------------------

def section_massive_blocks(ticker: str, since: str) -> None:
    if not os.environ.get("MASSIVE_API_KEY"):
        return
    print(f"\n=== MASSIVE LIT-BLOCK CROSS-CHECK — {ticker} (since {since}) ===")
    # Daily aggregates to find unusually heavy days; tick-level scan would be
    # rate-prohibitive on Starter. This is a coarse confirmation pass.
    data = massive_get(f"/v2/aggs/ticker/{ticker}/range/1/day/{since}/{date.today().isoformat()}",
                       params={"adjusted": "true", "limit": 120})
    if not data or not data.get("results"):
        print("  Massive returned no aggregates.")
        return
    bars = data["results"]
    bars_sorted = sorted(bars, key=lambda b: b.get("v") or 0, reverse=True)[:5]
    print("  Top 5 volume days from Massive (all venues, including TRFs):")
    for b in bars_sorted:
        ts = datetime.fromtimestamp((b.get("t") or 0) / 1000, tz=timezone.utc).date().isoformat()
        v = b.get("v") or 0
        c = b.get("c") or 0
        o = b.get("o") or 0
        chg = (c - o) / o * 100 if o else 0
        print(f"    {ts}  vol={v:>14,.0f}  open=${o:>6.2f} close=${c:>6.2f}  ({chg:+.2f}%)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def print_header(args) -> None:
    print("=" * 78)
    print(f"  RZLV / CMRC merger-vote signal monitor")
    print(f"  Run UTC: {now_iso()}")
    print(f"  Window:  since {args.since}    Tickers: {', '.join(args.tickers)}")
    print(f"  DB:      {DB_PATH}")
    print("=" * 78)
    print("  Deal timeline (informational):")
    for d, ev in DEAL_TIMELINE:
        print(f"    {d}  {ev}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", default="2026-03-01",
                   help="ISO date; lookback start for time-windowed sections (default 2026-03-01)")
    p.add_argument("--darkpool-since", default="2026-04-01",
                   help="Separate (later) lookback for dark pool — looping per-day "
                        "fetches make a longer window expensive (default 2026-04-01)")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                   help="Tickers to scan (default: CMRC RZLV)")
    p.add_argument("--csv", action="store_true",
                   help="Also dump today's signals to CSV alongside the DB")
    p.add_argument("--skip", nargs="+", default=[],
                   choices=["insiders", "institutional", "darkpool", "options",
                            "shorts", "ohlc", "massive", "edgar", "peer"],
                   help="Skip one or more sections")
    args = p.parse_args()

    try:
        _uw_headers()  # fail fast if no key
    except APIError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    print_header(args)
    conn = get_db()

    for ticker in args.tickers:
        print("\n" + "#" * 78)
        print(f"#  {ticker}")
        print("#" * 78)

        if "edgar" not in args.skip:
            section_edgar(conn, ticker, args.since)
        if "insiders" not in args.skip:
            section_insiders(conn, ticker, args.since)
        if "institutional" not in args.skip:
            section_institutional(conn, ticker)
        if "darkpool" not in args.skip:
            section_darkpool(conn, ticker, args.darkpool_since)
        if "options" not in args.skip:
            section_options_flow(conn, ticker, args.since)
        if "shorts" not in args.skip:
            section_short_interest(conn, ticker)
        if "ohlc" not in args.skip:
            section_price_volume(ticker, args.since)
        if "peer" not in args.skip:
            section_peer_relative(ticker)
        if "massive" not in args.skip:
            section_massive_blocks(ticker, args.since)

    conn.close()

    # ----- Consolidated flags summary -----
    # Sort flags by severity (HIGH > MED > INFO), then by ticker
    sev_order = {"HIGH": 0, "MED": 1, "INFO": 2}
    sorted_flags = sorted(TODAY_FLAGS,
                          key=lambda f: (sev_order.get(f[0], 99), f[1], f[2]))
    print("\n" + "=" * 78)
    print("  TODAY'S FLAGS")
    print("=" * 78)
    if not sorted_flags:
        print("  (no flags raised — all sections within thresholds)")
    else:
        # Group by severity
        for severity in ("HIGH", "MED", "INFO"):
            group = [f for f in sorted_flags if f[0] == severity]
            if not group:
                continue
            print(f"\n  [{severity}]  {len(group)} flag(s)")
            for sev, ticker, msg in group:
                print(f"    • {ticker:<5} {msg}")

    print("\n" + "=" * 78)
    print("  Done. Re-run daily during US market hours.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
