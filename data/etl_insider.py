# data/etl_insider.py
# ─────────────────────────────────────────────────────────────────────────────
# Insider trades ETL. Pulls Form 4 filings from SEC EDGAR, parses the primary
# XML document of each filing for real share counts, writes to SQLite.
#
# Flow:
#   1. Resolve ticker → CIK via company_tickers.json
#   2. Fetch CIK{cik}.json submissions → filter to Form 4 in date window
#   3. For each Form 4: fetch the index JSON, find the primary XML, parse it
#   4. Aggregate by date, weight by role, upsert to insider_flows table
#
# Respects SEC fair-use: User-Agent with email, ~6 req/sec ceiling.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH      = Path(os.getenv("INSIDER_DB_PATH", "insider_trades.db"))
SEC_HEADERS  = {
    # SEC requires a User-Agent with a real email. Set SEC_USER_AGENT env var
    # to your contact email, e.g. "atom research atom.v.nguyen@gmail.com"
    "User-Agent": os.getenv("SEC_USER_AGENT", "ML Quant Fund research@example.com"),
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}
DATA_HEADERS = {**SEC_HEADERS, "Host": "data.sec.gov"}
REQUEST_DELAY = 0.18  # ~5.5 req/sec, under SEC's 10 req/sec ceiling

# Cache ticker→CIK map so we only fetch it once per run
_CIK_CACHE: dict[str, str] = {}

ROLE_WEIGHTS: dict[str, float] = {
    "CEO": 3.0, "CHIEF EXECUTIVE": 3.0,
    "CFO": 3.0, "CHIEF FINANCIAL": 3.0,
    "PRESIDENT": 3.0,
    "COO": 2.5, "CHIEF OPERATING": 2.5,
    "CTO": 2.0, "CHIEF TECHNOLOGY": 2.0,
    "CIO": 2.0, "CHIEF INFORMATION": 2.0,
    "GENERAL COUNSEL": 1.5,
    "SVP": 1.5, "EVP": 1.5, "VP": 1.5,
    "OFFICER": 1.5,
    "DIRECTOR": 1.0,
    "TRUSTEE": 1.0,
}
DEFAULT_WEIGHT = 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def _init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS insider_flows (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            net_shares  REAL    NOT NULL DEFAULT 0,
            buy_shares  REAL    NOT NULL DEFAULT 0,
            sell_shares REAL    NOT NULL DEFAULT 0,
            num_buy_tx  INTEGER NOT NULL DEFAULT 0,
            num_sell_tx INTEGER NOT NULL DEFAULT 0,
            role_weight REAL    NOT NULL DEFAULT 1.0,
            UNIQUE(ticker, date)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ticker_date ON insider_flows(ticker, date)"
    )
    conn.commit()
    return conn


# ══════════════════════════════════════════════════════════════════════════════
#  SEC EDGAR
# ══════════════════════════════════════════════════════════════════════════════

def _load_cik_map() -> dict[str, str]:
    """Fetch ticker → CIK map once and cache it in-process."""
    if _CIK_CACHE:
        return _CIK_CACHE
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS, timeout=15,
        )
        resp.raise_for_status()
        for entry in resp.json().values():
            tkr = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            if tkr and cik:
                _CIK_CACHE[tkr] = cik
    except Exception as e:
        print(f"  ⚠ CIK map fetch failed: {e}")
    return _CIK_CACHE


def _get_cik(ticker: str) -> Optional[str]:
    return _load_cik_map().get(ticker.upper())


def _get_role_weight(relationship_text: str) -> float:
    if not relationship_text:
        return DEFAULT_WEIGHT
    upper = relationship_text.upper()
    for role, weight in ROLE_WEIGHTS.items():
        if role in upper:
            return weight
    return DEFAULT_WEIGHT


def _list_form4_filings(cik: str, since: str) -> list[dict]:
    """Return list of {accession, filing_date, report_date} for Form 4 filings since `since`."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(url, headers=DATA_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ⚠ submissions fetch failed for CIK {cik}: {e}")
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms  = recent.get("form", [])
    fdates = recent.get("filingDate", [])
    rdates = recent.get("reportDate", [])
    accs   = recent.get("accessionNumber", [])

    out = []
    for form, fd, rd, acc in zip(forms, fdates, rdates, accs):
        if form != "4":
            continue
        if fd < since:
            continue
        out.append({"accession": acc, "filing_date": fd, "report_date": rd or fd})
    return out


def _find_primary_xml(cik: str, accession: str) -> Optional[str]:
    """Given a Form 4 accession, return the URL of the primary XML document."""
    acc_clean = accession.replace("-", "")
    idx_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{acc_clean}/index.json"
    )
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(idx_url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("directory", {}).get("item", [])
    except Exception:
        return None

    # Form 4 primary doc is usually the only .xml file in the filing
    for item in items:
        name = item.get("name", "")
        if name.endswith(".xml") and not name.endswith("index.xml"):
            return (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_clean}/{name}"
            )
    return None


def _parse_form4_xml(xml_url: str) -> Optional[dict]:
    """
    Parse a Form 4 XML. Returns:
      {
        "report_date": "YYYY-MM-DD",
        "role": "<relationship text>",
        "buy_shares": float,
        "sell_shares": float,
      }
    or None on failure.
    """
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(xml_url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception:
        return None

    def _findtext(path: str, parent: ET.Element = root) -> str:
        el = parent.find(path)
        return (el.text or "").strip() if el is not None and el.text else ""

    # Relationship — multiple booleans may be set; pick the most senior.
    rel = root.find("reportingOwner/reportingOwnerRelationship")
    role = ""
    if rel is not None:
        officer_title = _findtext("officerTitle", rel)
        if officer_title:
            role = officer_title
        elif _findtext("isDirector", rel) == "1":
            role = "DIRECTOR"
        elif _findtext("isOfficer", rel) == "1":
            role = "OFFICER"
        elif _findtext("isTenPercentOwner", rel) == "1":
            role = "10% OWNER"

    # Report date (period of report) — this is the actual trade date, not filing date.
    report_date = _findtext("periodOfReport") or ""

    # Sum non-derivative transactions (actual stock buys/sells).
    # Derivative transactions (options grants/exercises) are in a separate block
    # and we skip them — they're noise for our signal.
    buy_shares = 0.0
    sell_shares = 0.0
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        amt_el = tx.find("transactionAmounts/transactionShares/value")
        code_el = tx.find("transactionCoding/transactionCode")
        ad_el   = tx.find("transactionAmounts/transactionAcquiredDisposedCode/value")
        if amt_el is None or ad_el is None:
            continue
        try:
            shares = float(amt_el.text or 0)
        except ValueError:
            continue
        ad = (ad_el.text or "").strip().upper()
        # Open-market P/S only. Skip grants (A code with no cost), gifts (G), etc.
        # that aren't real economic signal.
        code = (code_el.text or "").strip().upper() if code_el is not None else ""
        if code not in {"P", "S"}:
            continue
        if ad == "A":
            buy_shares += shares
        elif ad == "D":
            sell_shares += shares

    if buy_shares == 0 and sell_shares == 0:
        return None

    return {
        "report_date": report_date,
        "role": role,
        "buy_shares": buy_shares,
        "sell_shares": sell_shares,
    }


def _fetch_form4_trades(ticker: str, days_back: int = 365) -> pd.DataFrame:
    """
    Fetch and parse all Form 4 filings for a ticker in the last `days_back` days.
    Returns aggregated DataFrame indexed by trade date.
    """
    cik = _get_cik(ticker)
    if not cik:
        print(f"  ⚠ no CIK for {ticker}")
        return pd.DataFrame()

    since = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    filings = _list_form4_filings(cik, since)
    if not filings:
        return pd.DataFrame()

    rows = []
    for f in filings:
        xml_url = _find_primary_xml(cik, f["accession"])
        if not xml_url:
            continue
        parsed = _parse_form4_xml(xml_url)
        if not parsed:
            continue

        weight = _get_role_weight(parsed["role"])
        net = parsed["buy_shares"] - parsed["sell_shares"]
        rows.append({
            "ds":          parsed["report_date"] or f["filing_date"],
            "net_shares":  net * weight,  # role-weighted
            "buy_shares":  parsed["buy_shares"] * weight,
            "sell_shares": parsed["sell_shares"] * weight,
            "num_buy_tx":  1 if parsed["buy_shares"] > 0 else 0,
            "num_sell_tx": 1 if parsed["sell_shares"] > 0 else 0,
            "role_weight": weight,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return (
        df.groupby("ds")
          .agg(
              net_shares  = ("net_shares",  "sum"),
              buy_shares  = ("buy_shares",  "sum"),
              sell_shares = ("sell_shares", "sum"),
              num_buy_tx  = ("num_buy_tx",  "sum"),
              num_sell_tx = ("num_sell_tx", "sum"),
              role_weight = ("role_weight", "mean"),
          )
          .reset_index()
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SQLITE SINK
# ══════════════════════════════════════════════════════════════════════════════

def _upsert_to_db(ticker: str, df: pd.DataFrame, conn: sqlite3.Connection) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["ticker"] = ticker.upper()
    df["date"] = df["ds"].astype(str)
    rows = df[["ticker", "date", "net_shares", "buy_shares", "sell_shares",
               "num_buy_tx", "num_sell_tx", "role_weight"]].to_dict("records")
    conn.executemany("""
        INSERT OR REPLACE INTO insider_flows
            (ticker, date, net_shares, buy_shares, sell_shares,
             num_buy_tx, num_sell_tx, role_weight)
        VALUES
            (:ticker, :date, :net_shares, :buy_shares, :sell_shares,
             :num_buy_tx, :num_sell_tx, :role_weight)
    """, rows)
    conn.commit()
    return len(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC
# ══════════════════════════════════════════════════════════════════════════════

def run_insider_etl(
    tickers: list[str],
    days_back: int = 365,
    db_path: Path = DB_PATH,
    verbose: bool = True,
) -> dict[str, int]:
    conn = _init_db(db_path)
    results = {}
    _load_cik_map()  # prime cache

    for ticker in tickers:
        ticker = ticker.upper().strip()
        if verbose:
            print(f"  {ticker} — fetching Form 4 filings...")
        try:
            df = _fetch_form4_trades(ticker, days_back=days_back)
        except Exception as e:
            print(f"    ⚠ {ticker} failed: {e}")
            results[ticker] = 0
            continue

        if df.empty:
            if verbose:
                print(f"    · no Form 4 trades in last {days_back}d")
            results[ticker] = 0
            continue

        n = _upsert_to_db(ticker, df, conn)
        results[ticker] = n
        if verbose:
            print(f"    ✓ {n} trade-days written")

    conn.close()
    return results


def load_insider_flows(
    ticker: str,
    start_date: str | date | None = None,
    end_date:   str | date | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    q = "SELECT date, ticker, net_shares, buy_shares, sell_shares, num_buy_tx, num_sell_tx, role_weight FROM insider_flows WHERE ticker = ?"
    params: list = [ticker.upper()]
    if start_date:
        q += " AND date >= ?"; params.append(str(start_date))
    if end_date:
        q += " AND date <= ?"; params.append(str(end_date))
    q += " ORDER BY date"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(q, conn, params=params, parse_dates=["date"])


if __name__ == "__main__":
    # When invoked as `python -m data.etl_insider`, read the ticker universe
    # from the same place your retrain uses. Adjust this import if your
    # universe lives somewhere else.
    import argparse as _ap
    from pathlib import Path as _P

    _parser = _ap.ArgumentParser(description="Run insider Form 4 ETL.")
    _parser.add_argument("--days-back", type=int, default=365,
                         help="How many days of history to pull (default: 365 for full rebuild).")
    _args = _parser.parse_args()

    _tf = _P("tickers.txt")
    if _tf.exists():
        TICKERS = [ln.strip().upper() for ln in _tf.read_text().splitlines()
                   if ln.strip() and not ln.strip().startswith("#")]
    else:
        TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]
        print(f"  tickers.txt not found; using test tickers: {TICKERS}")

    print(f"Running insider ETL for {len(TICKERS)} tickers, {_args.days_back} days back...")
    results = run_insider_etl(TICKERS, days_back=_args.days_back, verbose=True)

    total = sum(results.values())
    print(f"\nDone. {total} total trade-days written to {DB_PATH}")
