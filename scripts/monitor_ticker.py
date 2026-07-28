#!/usr/bin/env python3
"""
monitor_earnings.py — Daily pre-earnings signal monitor.

Runs a positioning-and-flow playbook on each ticker leading up to its earnings
date. For each ticker, surfaces:
  - Earnings calendar context (date, days-until, BMO/AMC, last quarter results)
  - Implied earnings move (from straddle pricing)
  - Historical earnings reactions (last 4-8 prints)
  - Volatility regime (IV vs 30/90/365-day range)
  - SEC EDGAR filings (Form 4, 8-K, 10-Q — quiet-period insider trades flagged)
  - Form 4 XML parsing (real insider buy/sell with code, shares, price, filer)
  - Dark pool prints (per-day, looped — institutional accumulation footprint)
  - Options flow alerts (call/put premium, big single trades)
  - Short interest snapshots
  - Institutional ownership snapshots
  - Peer-relative price action (vs sector ETF + SPY, 1d / 5d / 20d)
  - Consolidated TODAY'S FLAGS summary (HIGH / MED / INFO)

All raw data is persisted to earnings_monitor.db so subsequent runs only fetch
what's new.

Usage:
    export UW_API_KEY="..."
    export EDGAR_USER_AGENT="Your Name your.real.email@yourdomain.com"
    export MASSIVE_API_KEY="..."   # optional

    python scripts/monitor_earnings.py                       # default 8 tickers
    python scripts/monitor_earnings.py --tickers NVDA DDOG   # subset
    python scripts/monitor_earnings.py --tickers NVDA --since 2026-04-01

Default ticker universe: NVDA, DDOG, SMCI, OKLO, QUBT, CRWD, SNOW, NVMI.
Cron-compatible. Returns nonzero on hard failure; soft-fails per-endpoint.
"""

from __future__ import annotations

import statistics
import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

UW_BASE = "https://api.unusualwhales.com"
MASSIVE_BASE = "https://api.massive.com"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _today_et():
    """ET-anchored 'today'. _today_et() on a UTC+7 machine is a day ahead of
    the US session from ~11:00 VN onward -- the class bug behind the AMZN
    days-to-earnings off-by-one and every stale news-cutoff window."""
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _Z
    return _dt.now(_Z("America/New_York")).date()
DB_PATH = Path(os.environ.get("EARNINGS_DB",
                              str(_REPO_ROOT / "earnings_monitor.db")))

# Default ticker universe with sector benchmark and earnings context.
# Confirm earnings dates via the company's IR page before relying on them.
TICKER_CONFIG: dict[str, dict] = {
    "MSFT": {
        "sector_etf": "IGV",
        "earnings_date": "2026-07-29",
        "earnings_time": "AMC",
        "news_search_term": "Microsoft MSFT",
    },
    "GOOG": {
        "sector_etf": "XLC",
        "earnings_date": "2026-11-04",
        "earnings_time": "AMC",
        "news_search_term": "Alphabet Google GOOG",
    },
    "AMZN": {
        "sector_etf": "XLY",
        "earnings_date": "2026-07-30",
        "earnings_time": "AMC",
        "news_search_term": "Amazon AMZN",
    },
    "BYND": {
        "earnings_date": "2026-08-05",  # AMC, CONFIRMED by company PR Jul 22 (GlobeNewswire) -- Q2 ended Jun 27
    },
    "NVDA": {
        "sector_etf": "SMH",     # VanEck Semiconductors
        "earnings_date": "2026-08-26",  # AMC, CONFIRMED (Wall Street Horizon + TipRanks, Jul 28) -- Q2 FY27
        "earnings_time": "AMC",
        "fiscal_q": "Q1 FY27",
        "shares_out": 24_300_000_000,  # ~24.3B (verify quarterly)
        "news_search_term": "NVIDIA NVDA",
    },
    "PLTR": {
        "sector_etf": "IGV",
        "earnings_date": "2026-08-03",  # AMC, confirmed via company press release Jul 13
        "earnings_time": "AMC",
        "news_search_term": "Palantir PLTR",
    },
    "SMCI": {
        "sector_etf": "SMH",     # Server hardware tracks semis
        "earnings_date": "2026-08-11",  # AMC, Q4 FY26 (was 2026-05-05 -- stale, anchored implied move on a weekly)
        "earnings_time": "AMC",
        "fiscal_q": "Q3 FY26",
        "shares_out": 600_000_000,
        "news_search_term": "Super Micro Computer SMCI",
    },
    "DDOG": {
        "sector_etf": "IGV",     # iShares Software ETF
        "earnings_date": "2026-08-06",  # BMO, CONFIRMED company IR PR Jul 16 (call 8:00 AM ET) -- was 2026-05-07
        "earnings_time": "BMO",  # BMO 8 AM ET, confirmed
        "earnings_time": "BMO",
        "fiscal_q": "Q1 2026",
        "shares_out": 345_000_000,
        "news_search_term": "Datadog DDOG",
    },
    "OKLO": {
        "sector_etf": "URA",     # Global X Uranium ETF (closest sector proxy)
        "earnings_date": "2026-05-12",  # AMC, confirmed (Q1 2026)
        "earnings_time": "AMC",
        "fiscal_q": "Q1 2026",
        "shares_out": 150_000_000,  # ~150M (verify; share count grows post-SPAC)
        "news_search_term": "Oklo nuclear OKLO stock",
    },
    "QUBT": {
        "sector_etf": "QTUM",    # Defiance Quantum ETF
        "earnings_date": "2026-05-11",  # AMC 4:30 PM ET, confirmed via PR
        "earnings_time": "AMC",
        "fiscal_q": "Q1 2026",
        "shares_out": 245_000_000,  # ~245M (mkt cap ~$2.23B / ~$9 price)
        "news_search_term": "Quantum Computing QUBT stock",
    },
    "CRWD": {
        "sector_etf": "CIBR",    # First Trust Cybersecurity ETF
        "earnings_date": "2026-06-03",  # AMC, estimate (UW: Jun 2, others Jun 9)
        "earnings_time": "AMC",
        "fiscal_q": "Q1 FY27",
        "shares_out": 260_000_000,  # ~260M diluted (per company guidance)
        "news_search_term": "CrowdStrike CRWD",
    },
    "SNOW": {
        "sector_etf": "IGV",     # iShares Software ETF
        "earnings_date": "2026-05-27",  # AMC 2 PM PT, confirmed via PR
        "earnings_time": "AMC",
        "fiscal_q": "Q1 FY27",
        "shares_out": 335_000_000,
        "news_search_term": "Snowflake SNOW stock",
    },
    "NVMI": {
        "sector_etf": "SMH",     # VanEck Semiconductors (NVMI = semi metrology)
        "earnings_date": "2026-05-14",  # BMO 8:30 AM ET, confirmed via 6-K
        "earnings_time": "BMO",
        "fiscal_q": "Q1 2026",
        "shares_out": 33_000_000,
        "news_search_term": "Nova metrology NVMI",
    },
    

}

DEFAULT_TICKERS = list(TICKER_CONFIG.keys())

# Global today's-flags tracker. Sections call flag(...) to push events here;
# main() prints a consolidated summary at the end so signals don't get lost
# in the per-section verbose output.
TODAY_FLAGS: list[tuple[str, str, str]] = []  # (severity, ticker, message)
_STALE_WARNED: set[str] = set()  # tickers already warned about stale earnings_date this run
# EPS rows known contaminated by GAAP equity marks (vendor feeds GAAP for some
# names, adjusted for others). MIRROR of the set in daily_uw_snapshot.py --
# update BOTH. List, not threshold: the 04-29 row (+94%) evades any sane bound.
EPS_QUARANTINE: set[tuple[str, str]] = {("GOOG", "2026-07-22"), ("GOOG", "2026-04-29")}
# Ticker-level: basis mismatch is SYSTEMATIC (GAAP actual vs adjusted consensus,
# every quarter). DDOG specimen: DB -66% "miss" on the May-7 +30% day; actual
# non-GAAP $0.60 BEAT $0.51 [fact, 3 sources Jul 26]. A date list would grow
# forever -- quarantine the ticker until vendor basis is fixed.
EPS_QUARANTINE_TICKERS: set[str] = {"DDOG"}


def flag(severity: str, ticker: str, message: str) -> None:
    """Severity: 'HIGH', 'MED', 'INFO'. Ticker can be '*' for cross-cutting.
    Dedupes exact repeats -- the stale-earnings warning fired 4-5x/run
    (one per days_until_earnings call site). Class fix, not site fix."""
    if (severity, ticker, message) in TODAY_FLAGS:
        return
    TODAY_FLAGS.append((severity, ticker, message))


def days_until_earnings(ticker: str) -> Optional[int]:
    """Returns None if no earnings date configured for this ticker."""
    cfg = TICKER_CONFIG.get(ticker.upper())
    if not cfg or not cfg.get("earnings_date"):
        return None
    try:
        ed = date.fromisoformat(cfg["earnings_date"])
        if (ed - _today_et()).days < -2:
            if ticker.upper() not in _STALE_WARNED:
                _STALE_WARNED.add(ticker.upper())
                print(f"  [warn] {ticker.upper()}: configured earnings_date {ed} is in the PAST "
                      f"-- config STALE; treating as unconfigured (SMCI-class bug: a past date "
                      f"silently anchors the implied move on the nearest weekly)")
            flag("MED", ticker.upper(), f"earnings_date {ed} in config is STALE -- update TICKER_CONFIG")
            return None
        return (ed - _today_et()).days
    except ValueError:
        return None


def is_in_quiet_period(ticker: str) -> bool:
    """SEC and corporate quiet period typically runs ~14 days before earnings.
    Insider Form 4 trades during this window are unusual and elevate signal."""
    d = days_until_earnings(ticker)
    return d is not None and 0 <= d <= 14

# Signal thresholds. These names range from megacap (NVDA $200+, ~225M shares/day)
# to small-cap (NVMI, low ADV). Thresholds below are
# absolute USD; sections that compute relative metrics use ratios instead.
DARKPOOL_BLOCK_MIN_USD = 1_000_000      # individual print to be "block" worthy
DARKPOOL_DAILY_AGGREGATE_USD = 25_000_000  # heavy day on a typical mid/large
INSIDER_MIN_USD = 250_000               # single insider trade to flag
INSIDER_FLOW_DAYS = 30                  # rolling window
NEW_INST_MIN_VALUE_USD = 50_000_000     # new 13F position threshold
OPTIONS_PREMIUM_USD = 500_000           # single option trade premium
SHORT_INTEREST_DELTA_PCT = 0.02         # 2pp move in % of float short

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
    v = _first(r, "market_date", "record_date", "settlement_date", "date",
               "report_date", "as_of_date", "effective_date", "period_date",
               "filing_date")
    return str(v) if v else None


def get_inst_name(r: dict) -> str:
    v = _first(r, "institution_name", "name", "institution", "manager_name",
               "filer_name", "holder_name")
    return str(v) if v else "?"


# ---------------------------------------------------------------------------
# SEC EDGAR helpers — free, no auth, but require a User-Agent header
# ---------------------------------------------------------------------------

# SEC requires a User-Agent that identifies the requester with real contact
# info — they actively block User-Agents containing example.com or other
# obvious placeholders. Set EDGAR_USER_AGENT env var to your real info, e.g.
#   export EDGAR_USER_AGENT="Atom Nguyen atom@yourdomain.com"
# Falls back to a generic-but-not-fake string if unset.
EDGAR_UA = os.environ.get(
    "EDGAR_USER_AGENT",
    "Individual Research atomnguyen.research@gmail.com"
)
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


# ---------------------------------------------------------------------------
# Form 4 (insider transaction) XML fetching and parsing
# ---------------------------------------------------------------------------
# SEC rate limit is 10 req/sec. We sleep 0.15s between SEC requests to be polite.

import xml.etree.ElementTree as ET
import re

_LAST_SEC_REQUEST_TS = 0.0
SEC_MIN_INTERVAL_SEC = 0.15


def edgar_throttle() -> None:
    """Block until SEC_MIN_INTERVAL_SEC has passed since the last SEC request."""
    global _LAST_SEC_REQUEST_TS
    now = time.time()
    elapsed = now - _LAST_SEC_REQUEST_TS
    if elapsed < SEC_MIN_INTERVAL_SEC:
        time.sleep(SEC_MIN_INTERVAL_SEC - elapsed)
    _LAST_SEC_REQUEST_TS = time.time()


def edgar_get_text(url: str) -> Optional[str]:
    """GET against SEC for raw text/XML (Form 4 documents are XML, not JSON)."""
    edgar_throttle()
    try:
        r = requests.get(url, headers={"User-Agent": EDGAR_UA}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        print(f"  [warn] EDGAR fetch {url}: {e}")
        return None


def build_form4_url(cik: int, accession: str, primary_doc: str) -> str:
    """Construct the URL to a Form 4's primary XML document.

    Accession numbers come back in two formats from EDGAR — with dashes
    (0001193125-26-192318) and without (000119312526192318). The Archives
    URL needs the no-dash form."""
    acc_clean = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{primary_doc}"


def find_form4_xml_url(cik: int, accession: str) -> Optional[str]:
    """Locate a Form 4's actual XML document via SEC's index.json.

    Necessary because the `primaryDocument` field from submissions JSON is
    sometimes empty or points at the HTML rendering. index.json reliably
    lists every file in the filing's archive directory, and we pick the
    XML one that looks like a Form 4 ownership document."""
    acc_clean = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}"
    idx_url = f"{base}/index.json"

    edgar_throttle()
    try:
        r = requests.get(idx_url, headers={"User-Agent": EDGAR_UA},
                         timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return None

    items = data.get("directory", {}).get("item", []) or []
    xml_candidates: list[str] = []
    for item in items:
        name = (item.get("name") or "").strip()
        if not name.lower().endswith(".xml"):
            continue
        # Skip the filing's metadata index XML (different from the doc itself)
        if name.lower().endswith("-index.xml"):
            continue
        if "filingsummary" in name.lower():
            continue
        xml_candidates.append(name)

    if not xml_candidates:
        return None

    # Preference order, most specific first:
    #   1. primary_doc.xml — SEC's standard Form 4 name
    #   2. anything matching ownership / form4 / edgar patterns
    #   3. anything else
    for preferred in ("primary_doc.xml",):
        if preferred in xml_candidates:
            return f"{base}/{preferred}"
    for name in xml_candidates:
        nl = name.lower()
        if "form4" in nl or "ownership" in nl or "edgardoc" in nl:
            return f"{base}/{name}"
    return f"{base}/{xml_candidates[0]}"


def fetch_form4_xml(cik: int, accession: str, primary_doc: Optional[str]) -> Optional[str]:
    """Fetch Form 4 XML, trying primary_doc first then falling back to
    index.json discovery if the response isn't well-formed XML."""

    def looks_like_xml(text: str) -> bool:
        if not text:
            return False
        head = text.lstrip()[:200].lower()
        if head.startswith("<!doctype html") or head.startswith("<html"):
            return False
        # Look for <ownershipDocument> or <?xml prefix
        return ("<ownershipdocument" in head or
                head.startswith("<?xml") or
                "<edgarsubmission" in head)

    # Attempt 1: primary_doc if we have one
    if primary_doc:
        url = build_form4_url(cik, accession, primary_doc)
        text = edgar_get_text(url)
        if text and looks_like_xml(text):
            return text

    # Attempt 2: index.json discovery
    xml_url = find_form4_xml_url(cik, accession)
    if not xml_url:
        return None
    text = edgar_get_text(xml_url)
    if text and looks_like_xml(text):
        return text

    return None


def _xml_text(elem: Optional[ET.Element], path: str) -> str:
    """Find a sub-element by XPath and return its text, with .//value support."""
    if elem is None:
        return ""
    found = elem.find(path)
    if found is None:
        return ""
    # Form 4 XML uses <field><value>X</value></field> for most fields
    val = found.find("value")
    if val is not None and val.text:
        return val.text.strip()
    return (found.text or "").strip()


def extract_10b5_1_plan_date(text: str) -> Optional[str]:
    """Find a 10b5-1 plan adoption date inside footnote text.

    Form 4 footnotes use phrasing like:
      "Shares sold pursuant to a 10b5-1 plan dated June 13, 2025."
      "...Rule 10b5-1 trading plan dated December 8, 2025."
      "...pre-arranged 10b5-1 trading plan dated August 22, 2025."

    Returns the date in ISO format (YYYY-MM-DD), or None if no plan
    reference is found. If a plan IS referenced but the date can't be
    parsed, returns the empty string '' so callers can distinguish
    "no plan mentioned" (None) from "plan mentioned but unparseable" ('').
    """
    if not text:
        return None

    # Catch any reference to a 10b5-1 plan first; return None if absent
    if "10b5-1" not in text and "10b5–1" not in text:
        return None

    # Try to extract "Month DD, YYYY" or "Month D, YYYY" after "dated"
    pattern = r"dated\s+([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})"
    m = re.search(pattern, text)
    if not m:
        return ""  # plan referenced but date not findable

    date_str = m.group(1).replace(".", "").strip()
    # Try a few date formats that show up in real Form 4 footnotes
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(date_str, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


# Role tiers for insider signal weighting:
#   "OFFICER"   = current operational executive (CEO, CFO, COO, CTO, etc.)
#                 — highest signal weight; their selling actually means
#                 something operational
#   "DIRECTOR"  = board director only, no current officer role — much lower
#                 signal weight (board members see less operational data,
#                 plus former executives often retain board seats post-departure
#                 while diversifying their stake)
#   "10PCT"     = 10%+ owner with no officer/director role — institutional
#                 holder activity, generally low signal
#   "OTHER"     = none of the above (rare)

def filer_role_tier(parsed_form4: dict) -> str:
    """Classify a Form 4 filer into a role tier for signal weighting."""
    if parsed_form4.get("is_officer"):
        # Some filings keep ex-officers as officers with "Former" in title;
        # we treat those as DIRECTOR-tier signal-wise
        title = (parsed_form4.get("officer_title") or "").lower()
        if "former" in title or "outgoing" in title:
            return "DIRECTOR"
        return "OFFICER"
    if parsed_form4.get("is_director"):
        return "DIRECTOR"
    if parsed_form4.get("is_ten_percent_owner"):
        return "10PCT"
    return "OTHER"


def classify_10b5_1_safety(plan_date_iso: str, transaction_date_iso: str) -> str:
    """Given a plan adoption date and a transaction date, return:
       "CLEAN"   — plan adopted >= 90 days before transaction (safe harbor)
       "TIGHT"   — plan adopted < 90 days before (suspicious / potentially
                   inside the cooling-off period — REGULATORY RED FLAG)
       "UNKNOWN" — plan referenced in footnote but date not parseable
       "NONE"    — no 10b5-1 plan referenced at all (discretionary trade)
    """
    if plan_date_iso is None:
        return "NONE"
    if plan_date_iso == "":
        return "UNKNOWN"
    try:
        pd = date.fromisoformat(plan_date_iso)
        td = date.fromisoformat(transaction_date_iso[:10])
    except (ValueError, TypeError):
        return "UNKNOWN"
    cooling_off_days = (td - pd).days
    if cooling_off_days >= 90:
        return "CLEAN"
    return "TIGHT"


def parse_form4_xml(xml_text: str) -> dict:
    """Parse a Form 4 XML document into structured transaction data.

    Returns a dict with:
      - filer_name, filer_cik, is_director, is_officer, is_ten_percent_owner,
        officer_title
      - transactions: list of dicts, each with date, code, shares, price, value,
        acquired_disposed, post_holding, security_title, is_derivative,
        plan_date (ISO date or empty if no 10b5-1 plan referenced),
        footnote_text (concatenated text of any footnotes referenced)
    """
    out: dict = {
        "filer_name": "", "filer_cik": "",
        "is_director": False, "is_officer": False,
        "is_ten_percent_owner": False, "officer_title": "",
        "issuer_symbol": "", "issuer_cik": "",
        "transactions": [],
    }
    try:
        # Form 4 XML often has no namespace
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  [warn] Form 4 XML parse failed: {e}")
        return out
    # Issuer identity. EDGAR surfaces Form 4s under BOTH issuer and reporting-
    # owner CIKs, so a venture arm's portfolio dispositions (GV -> Alphabet)
    # appear in the parent's feed. Without these fields the consumer cannot
    # tell GV selling a portfolio company from insiders selling GOOG.
    _iss = root.find("issuer")
    if _iss is not None:
        out["issuer_symbol"] = (_iss.findtext("issuerTradingSymbol") or "").strip().upper()
        out["issuer_cik"] = (_iss.findtext("issuerCik") or "").strip()

    # Reporting owner
    owner = root.find(".//reportingOwner")
    if owner is not None:
        out["filer_name"] = _xml_text(owner, "reportingOwnerId/rptOwnerName")
        out["filer_cik"] = _xml_text(owner, "reportingOwnerId/rptOwnerCik")
        rel = owner.find("reportingOwnerRelationship")
        if rel is not None:
            out["is_director"] = _xml_text(rel, "isDirector") in ("1", "true", "True")
            out["is_officer"] = _xml_text(rel, "isOfficer") in ("1", "true", "True")
            out["is_ten_percent_owner"] = _xml_text(rel, "isTenPercentOwner") in ("1", "true", "True")
            out["officer_title"] = _xml_text(rel, "officerTitle")

    # Build a footnote_id -> text map. Footnotes contain the 10b5-1 plan
    # adoption date when one applies to the transaction. Form 4 schema:
    #   <footnotes>
    #     <footnote id="F1">Shares sold pursuant to a 10b5-1 plan dated June 13, 2025.</footnote>
    #   </footnotes>
    footnotes: dict[str, str] = {}
    fns_elem = root.find(".//footnotes")
    if fns_elem is not None:
        for f in fns_elem.findall("footnote"):
            fid = f.get("id", "")
            text = (f.text or "").strip()
            if fid and text:
                footnotes[fid] = text

    def collect_footnote_ids(tx_elem: ET.Element) -> list[str]:
        """Return all <footnoteId id="..."/> references found anywhere inside
        a transaction element. Form 4 places footnoteIds inside nested fields
        (transactionShares, transactionPricePerShare, etc.)."""
        ids = []
        for fid_elem in tx_elem.iter("footnoteId"):
            fid = fid_elem.get("id", "")
            if fid:
                ids.append(fid)
        return ids

    def parse_tx(tx_elem: ET.Element, is_derivative: bool) -> Optional[dict]:
        code = _xml_text(tx_elem, "transactionCoding/transactionCode")
        date = _xml_text(tx_elem, "transactionDate")
        sec_title = _xml_text(tx_elem, "securityTitle")
        try:
            shares = float(_xml_text(tx_elem, "transactionAmounts/transactionShares") or 0)
        except ValueError:
            shares = 0.0
        try:
            price = float(_xml_text(tx_elem, "transactionAmounts/transactionPricePerShare") or 0)
        except ValueError:
            price = 0.0
        ad = _xml_text(tx_elem, "transactionAmounts/transactionAcquiredDisposedCode")
        try:
            post = float(_xml_text(tx_elem, "postTransactionAmounts/sharesOwnedFollowingTransaction") or 0)
        except ValueError:
            post = 0.0
        if not code and not shares:
            return None

        # Pull footnote text(s) referenced by this transaction
        ref_ids = collect_footnote_ids(tx_elem)
        ref_texts = [footnotes[fid] for fid in ref_ids if fid in footnotes]
        # De-dupe while preserving order
        seen = set()
        ref_texts = [t for t in ref_texts if not (t in seen or seen.add(t))]
        footnote_text = " | ".join(ref_texts)

        # Look for "10b5-1 plan dated <date>" in any of the footnote texts
        plan_date = extract_10b5_1_plan_date(footnote_text)

        return {
            "date": date, "code": code, "shares": shares, "price": price,
            "value": shares * price, "acquired_disposed": ad,
            "post_holding": post, "security_title": sec_title,
            "is_derivative": is_derivative,
            "plan_date": plan_date,  # None=no plan, ""=plan but unparseable, ISO=parsed
            "footnote_text": footnote_text[:500],  # cap for storage
        }

    # Non-derivative (common stock) transactions
    for tx in root.findall(".//nonDerivativeTransaction"):
        d = parse_tx(tx, is_derivative=False)
        if d:
            out["transactions"].append(d)

    # Derivative transactions (options, RSUs) — usually low-signal but worth tracking
    for tx in root.findall(".//derivativeTransaction"):
        d = parse_tx(tx, is_derivative=True)
        if d:
            out["transactions"].append(d)

    return out


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

CREATE TABLE IF NOT EXISTS form4_parsed (
    accession_number TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    parsed_at TEXT NOT NULL,
    filer_name TEXT,
    filer_cik TEXT,
    officer_title TEXT,
    is_director INTEGER,
    is_officer INTEGER,
    is_ten_percent_owner INTEGER,
    transaction_count INTEGER,
    aggregate_p_value REAL,
    aggregate_s_value REAL
);

CREATE TABLE IF NOT EXISTS form4_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    accession_number TEXT NOT NULL,
    ticker TEXT NOT NULL,
    transaction_date TEXT,
    code TEXT,
    shares REAL,
    price REAL,
    value_usd REAL,
    acquired_disposed TEXT,
    post_holding REAL,
    security_title TEXT,
    is_derivative INTEGER,
    filer_name TEXT,
    officer_title TEXT,
    plan_date TEXT,           -- ISO date if 10b5-1 plan referenced in footnote
    role_tier TEXT            -- 'OFFICER', 'DIRECTOR', '10PCT', 'OTHER'
);
CREATE INDEX IF NOT EXISTS idx_form4_tx_ticker_date
    ON form4_transactions(ticker, transaction_date);
"""


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.executescript(SCHEMA)
    # Add new columns if upgrading from an older schema version. SQLite
    # is happy to silently ignore the failure if columns already exist.
    for col, decl in [("plan_date", "TEXT"), ("role_tier", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE form4_transactions ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass  # already exists
    conn.commit()
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

def _get_shares_outstanding(ticker: str) -> Optional[float]:
    """Pull approximate shares outstanding for the ticker. Cached on the
    config dict to avoid re-fetching. Hardcoded value in TICKER_CONFIG
    takes precedence — UW endpoints for this seem unreliable on Basic plan."""
    cfg = TICKER_CONFIG.get(ticker.upper(), {})
    # Cached from prior call
    cached = cfg.get("_shares_out")
    if cached:
        return cached
    # Hardcoded in config (preferred — verified once, doesn't change often)
    hardcoded = cfg.get("shares_out")
    if hardcoded:
        cfg["_shares_out"] = float(hardcoded)
        return cfg["_shares_out"]

    # Fallback: try UW endpoints
    for path in [
        f"/api/stock/{ticker}/info",
        f"/api/stock/{ticker}/stock-state",
        f"/api/companies/{ticker}/profile",
    ]:
        try:
            data = uw_get(path)
        except Exception:
            data = None
        if not data:
            continue
        d = data.get("data") if isinstance(data, dict) else data
        if isinstance(d, list) and d:
            d = d[0]
        if not isinstance(d, dict):
            continue
        for key in ("shares_outstanding", "sharesOutstanding", "shares_out",
                    "outstanding_shares", "total_shares"):
            v = d.get(key)
            if v:
                try:
                    so = float(v)
                    if so > 1_000_000:  # sanity check
                        cfg["_shares_out"] = so
                        return so
                except (TypeError, ValueError):
                    continue
    return None


def section_institutional(conn: sqlite3.Connection, ticker: str) -> None:
    print(f"\n=== INSTITUTIONAL OWNERSHIP — {ticker} ===")

    data = uw_get(f"/api/institution/{ticker}/ownership", params={"limit": 200})
    rows = (data or {}).get("data") or []
    if not rows:
        print("  No institutional ownership data returned.")
        return

    # Pull previous snapshot from DB to compute deltas
    cur = conn.cursor()
    # 13F DELTA FIX (Jul 11 2026): the old query read prev at MAX(report_date),
    # but the CURRENT snapshot is written with that SAME report_date (13F dates are
    # quarter-end and static between runs) -- so prev WAS the filing being displayed
    # and every delta came out +0 across all 200 holders. Compare to the PREVIOUS
    # report_date instead, and say so plainly when there is no prior snapshot.
    _cur_rd = max([(get_date(r) or now_iso()[:10]) for r in rows] or ["9999-12-31"])
    prev = {row[0]: row[1] for row in cur.execute(
        "SELECT institution_name, shares FROM institutional_holdings "
        "WHERE ticker = ? AND report_date = ("
        "  SELECT MAX(report_date) FROM institutional_holdings "
        "  WHERE ticker = ? AND report_date < ?"
        ")", (ticker, ticker, _cur_rd))}
    _pr = cur.execute("SELECT MAX(report_date) FROM institutional_holdings "
                      "WHERE ticker = ? AND report_date < ?",
                      (ticker, _cur_rd)).fetchone()
    _prev_rd = _pr[0] if _pr else None
    if not prev:
        print(f"  [note] no prior 13F snapshot (current report_date {_cur_rd}); "
              f"'Delta vs prev' reads +0 until the next filing lands")
    else:
        print(f"  [note] deltas vs report_date {_prev_rd}")

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
                get_date(r) or now_iso()[:10],
                get_shares(r),
                get_value_usd(r),
                float(r.get("pct_of_float") or r.get("percent_of_float") or 0),
                now_iso(),
            ))
        except sqlite3.Error as e:
            print(f"  [warn] insert failed: {e}")
    conn.commit()

    # Get shares outstanding for percent-of-float threshold
    shares_out = _get_shares_outstanding(ticker)

    # Calibrate the "meaningful" threshold per ticker. A $50M position is
    # routine for NVDA (3T market cap) but huge for NVMI (~$7B market cap).
    # New rule: position is "meaningful" if it represents >= 0.5% of shares
    # outstanding, OR >= $50M for tickers where shares-out couldn't be fetched.
    NEW_HOLDER_PCT_THRESHOLD = 0.005  # 0.5% of shares out

    rows_sorted = sorted(rows, key=lambda r: (get_shares(r), get_value_usd(r)), reverse=True)[:15]
    print(f"  Total holders reported: {len(rows)}")
    if shares_out:
        print(f"  Shares outstanding: ~{shares_out:,.0f}  "
              f"(new-holder threshold: {NEW_HOLDER_PCT_THRESHOLD*100:.1f}% = "
              f"~{shares_out * NEW_HOLDER_PCT_THRESHOLD:,.0f} shares)")
    print(f"  {'Institution':<40} {'Shares':>14} {'Δ vs prev':>14} {'Value $':>16} {'% s/o':>6}")
    for r in rows_sorted:
        name = get_inst_name(r)[:40]
        sh = get_shares(r)
        v = get_value_usd(r)
        delta = sh - prev.get(name, sh)
        is_new = name not in prev and bool(prev)
        pct_so = (sh / shares_out * 100) if shares_out and shares_out > 0 else None

        tag = ""
        if is_new:
            # Flag only if position is meaningful by relative or absolute measure
            meaningful_relative = (pct_so is not None and pct_so >= NEW_HOLDER_PCT_THRESHOLD * 100)
            meaningful_absolute = (pct_so is None and v >= NEW_INST_MIN_VALUE_USD)
            if meaningful_relative:
                tag = "  NEW"
                flag("MED", ticker,
                     f"New institutional holder: {name} ({pct_so:.2f}% s/o, ${v:,.0f})")
            elif meaningful_absolute:
                tag = "  NEW"
                flag("MED", ticker,
                     f"New institutional holder: {name} (${v:,.0f})")
            else:
                # Suppress — small position, likely 13F rebalance noise
                pass
        elif sh > 0 and abs(delta) > 0.05 * sh:
            tag = "  CHG"
            pct_chg = (delta / sh) * 100 if sh else 0
            # Only flag substantial moves (>20% AND >0.5% of s/o or >$50M)
            material = abs(pct_chg) > 20 and (
                (pct_so is not None and pct_so >= NEW_HOLDER_PCT_THRESHOLD * 100)
                or (pct_so is None and v >= NEW_INST_MIN_VALUE_USD)
            )
            if material:
                flag("MED", ticker,
                     f"{name} position change: {delta:+,.0f} shares ({pct_chg:+.1f}%)")
        pct_so_s = f"{pct_so:.2f}%" if pct_so is not None else "?"
        print(f"  {name:<40} {sh:>14,.0f} {delta:>+14,.0f} {v:>16,.0f} {pct_so_s:>6}{tag}")


# ---------------------------------------------------------------------------
# Section: dark pool prints
# ---------------------------------------------------------------------------

def section_darkpool(conn: sqlite3.Connection, ticker: str, since: str) -> None:
    print(f"\n=== DARK POOL PRINTS — {ticker} (since {since}) ===")

    # CURSOR WALK (Jul 14 2026). UW serves newest-500 descending; `older_than`
    # is an EXCLUSIVE cursor (verified). Step +1s and let the tracking_id PK
    # absorb re-served boundary rows. Replaces the per-date loop, which froze
    # each day at first fetch (9.2% of tape stored), re-fetched holidays
    # forever (weekday()<5), and hardcoded EDT (utcnow()-4h).
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
    cur = conn.cursor()
    try:
        since_date = date.fromisoformat(str(since))
    except (ValueError, TypeError):
        since_date = datetime.now(ET_TZ).date() - timedelta(days=60)
    older = None
    prev_oldest = None
    pages = 0
    fetch_failed = 0
    fetch_succeeded = 0
    MAX_PAGES   = int(os.environ.get("ML_QUANT_DP_MAX_PAGES", "800"))
    MAX_SECONDS = float(os.environ.get("ML_QUANT_DP_MAX_SECONDS", "600"))
    _t0 = time.time()
    _reached = None
    _dry = 0
    _exhausted = False
    while pages < MAX_PAGES:
        params = {"limit": 500}
        if older:
            params["older_than"] = older
        data = uw_get(f"/api/darkpool/{ticker}", params=params)
        if data is None:
            fetch_failed += 1
            break
        rows_page = data.get("data") or []
        if not rows_page:
            break
        fetch_succeeded += 1
        _before = conn.total_changes
        for r in rows_page:
            ts = r.get("executed_at") or r.get("trf_executed_at") or r.get("trade_time")
            if not ts:
                continue
            try:
                et_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(ET_TZ)
                sz = float(r.get("size") or 0)
                px = float(r.get("price") or 0)
            except (TypeError, ValueError):
                continue
            try:
                cur.execute(
                    "INSERT OR IGNORE INTO darkpool_prints "
                    "(ticker, executed_at, size, price, value_usd, venue, tracking_id, "
                    " nbbo_bid, nbbo_ask, canceled, ext_hours, et_date) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (ticker, ts, sz, px, sz * px,
                     r.get("market_center") or r.get("venue"),
                     str(r.get("tracking_id")) if r.get("tracking_id")
                         else f"{ticker}-{ts}-{sz}-{px}",
                     float(r["nbbo_bid"]) if r.get("nbbo_bid") else None,
                     float(r["nbbo_ask"]) if r.get("nbbo_ask") else None,
                     1 if r.get("canceled") else 0,
                     r.get("ext_hour_sold_codes"),
                     et_dt.date().isoformat()))
            except sqlite3.Error as e:
                print(f"  [warn] insert failed: {e}")
        pages += 1
        oldest = min((r.get("executed_at") for r in rows_page if r.get("executed_at")),
                     default=None)
        if oldest is not None:
            _reached = oldest
        if oldest is None or len(rows_page) < 500:
            _exhausted = True
            break
        oldest_dt = datetime.fromisoformat(oldest.replace("Z", "+00:00"))
        if oldest_dt.astimezone(ET_TZ).date() < since_date:
            break
        if prev_oldest is not None and oldest >= prev_oldest:
            print(f"    [warn] cursor reset at page {pages} (feed re-served newer rows); stopping")
            break
        if time.time() - _t0 > MAX_SECONDS:
            print(f"    [warn] {MAX_SECONDS:.0f}s budget hit at page {pages}")
            break
        if conn.total_changes == _before:
            _dry += 1
            if _dry >= 2:
                print(f"    2 pages, 0 new prints -- window already cached, stopping.")
                break
        else:
            _dry = 0
        if pages % 25 == 0:
            print(f"    ... {pages} pages, {time.time()-_t0:.0f}s, back to {oldest[:10]}")
        prev_oldest = oldest
        older = (oldest_dt + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    if pages:
        _r = (datetime.fromisoformat(_reached.replace("Z", "+00:00")).astimezone(ET_TZ).date()
              if _reached else None)
        _done = "complete" if (_r and _r <= since_date) else "PARTIAL (feed exhausted)" if _exhausted else "PARTIAL"
        print(f"  Dark pool: {pages} page(s) in {time.time()-_t0:.0f}s; reached {_r}, "
              f"requested {since_date} -- {_done}.")
        if _r and _r > since_date:
            print(f"  [note] days before {_r} are whatever was cached earlier -- "
                  f"heal with: python scripts/repair_darkpool_days.py --ticker {ticker} "
                  f"(if the daily table below is gapless, the window is already healed -- "
                  f"PARTIAL describes the live walk, which stops at cached data by design)")
    conn.commit()
    # Now query the DB for the full window and analyze
    rows = list(cur.execute("""
        SELECT executed_at, size, price, value_usd, venue,
               nbbo_bid, nbbo_ask, ext_hours, et_date
        FROM darkpool_prints
        WHERE ticker = ? AND et_date >= ? AND COALESCE(canceled, 0) = 0
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
    for executed_at, size, price, value, venue, nbbo_bid, nbbo_ask, ext_hours, et_date in rows:
        day = et_date or (executed_at or "")[:10]
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

    # Compute the rolling 20-day average dark pool value for this ticker.
    # Then "heavy" = today is >2x that average. Switching from absolute USD
    # threshold to relative ratio so liquid names (NVDA, DDOG) don't flag
    # every single day; they only flag genuinely unusual days.
    # PARTIAL-DAY EXCLUSION (Jul 18 2026). Days fetched before the cursor-walk fix
    # are frozen ~500-print slices (~9% of tape). Averaging them with full days
    # understates the baseline ~4x and fires false HEAVY flags. Signature: <=550
    # prints on a ticker whose full days run into the thousands.
    _maxp = max((by_day[d]["prints"] for d in sorted_days), default=0)
    def _partial(day):
        return _maxp >= 5000 and by_day[day]["prints"] <= 550
    _clean_days = [d for d in sorted_days[1:21] if not _partial(d)]
    _n_part = len([d for d in sorted_days[1:21] if _partial(d)])
    all_values = [by_day[d]["value"] for d in _clean_days]
    avg_20d = (sum(all_values) / len(all_values)) if all_values else 0
    if _n_part:
        print(f"  [note] {_n_part} partial day(s) excluded from the HEAVY baseline "
              f"(pre-fix ~500-print slices); run repair_darkpool_days.py to heal them")
    HEAVY_THRESHOLD = max(DARKPOOL_DAILY_AGGREGATE_USD, avg_20d * 2)

    for day in sorted_days[:15]:
        b = by_day[day]
        tags = ""
        if b["value"] >= HEAVY_THRESHOLD:
            tags += "  HEAVY"
            # Only flag the most recent 5 heavy days to avoid summary spam,
            # and only if today's day is one of them (most actionable signal)
            if day == sorted_days[0] or day in sorted_days[:3]:
                ratio = b["value"] / avg_20d if avg_20d else 0
                flag("MED", ticker,
                     f"Heavy dark pool day {day}: ${b['value']:,.0f} on {b['prints']} prints"
                     + (f" ({ratio:.1f}x 20d avg)" if ratio else ""))
        if b["max_print"] >= DARKPOOL_BLOCK_MIN_USD:
            tags += "  BLOCK"
            # Only flag block prints in the last 3 days
            if day in sorted_days[:3] and "HEAVY" not in tags:
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
    # TAG, don't hide (SMCI same-universe lesson): AH closing crosses and
    # known-mechanical dates rank as "top prints" but are auction plumbing,
    # not organic blocks -- AMZN's entire Top-10 was eight Jun-26 Russell
    # crosses at one price. They stay listed, labeled.
    try:
        from scripts.mechanical_block_filter import KNOWN_MECHANICAL_DATES as _MECH_DATES
    except Exception:
        _MECH_DATES = {"2026-06-26"}  # fallback: Russell recon
    for executed_at, size, price, value, venue, _nb, _na, _xh, _ed in big:
        if (value or 0) < DARKPOOL_BLOCK_MIN_USD:
            continue
        _tags = ""
        if _xh:
            _tags += " AH"
        if _ed in _MECH_DATES:
            _tags += " MECH"
        print(f"    {(executed_at or '')[:19]}  {size:>10,.0f} @ ${price:>7,.2f}  "
              f"= ${value:>12,.0f}  ({venue or '?'}{_tags})")

    # ----- Signed dark pool flow (VWAP heuristic) -----
    # True Lee-Ready needs NBBO at print time (UW Premium tier). As an
    # approximation: compute each day's VWAP from all dark prints, then
    # classify each print:
    #   price > VWAP * 1.0005 → buy-side (paid up)
    #   price < VWAP * 0.9995 → sell-side (took a haircut)
    #   else → neutral (at VWAP)
    # Report the dollar-weighted skew per day for the most recent 7.
    print(f"\n  Signed flow estimate (NBBO midpoint; VWAP fallback for pre-May-29 rows; RTH only):")
    # COVERAGE GUARD (Jul 11 2026). A skew without its coverage is not a number.
    # Gate on ABSOLUTE classified flow, not the ratio: MSFT Jun 26 had $161M
    # classified (a normal sample) but 0.8% coverage only because the denominator
    # carried $20B of Russell-reconstitution auction prints.
    MIN_CLASSIFIED_USD = 5_000_000
    MIN_CLASSIFIED_PRINTS = 20
    MIN_COVERAGE_PCT = 50.0
    n_usable = 0
    n_days = 0
    print(f"  {'Date':<12} {'Buy $':>16} {'Sell $':>16} {'Net':>14} {'Skew':>8} {'Cover':>7}")
    by_day_signed: dict[str, dict] = {}
    for executed_at, size, price, value, venue, nbbo_bid, nbbo_ask, ext_hours, et_date in rows:
        day = et_date or (executed_at or "")[:10]
        if not day:
            continue
        if ext_hours:
            continue  # RTH-only signing: AH/pre-market tape is thin and the
                      # 16:00+ closing-cross mega-prints land here by venue rule
        d = by_day_signed.setdefault(day, {"prints": [], "total_val": 0.0,
                                           "total_sh": 0.0})
        try:
            sh = float(size or 0)
            px = float(price or 0)
            v = float(value or 0)
        except (TypeError, ValueError):
            continue
        if sh and px:
            d["prints"].append((px, v, nbbo_bid, nbbo_ask))
            d["total_val"] += v
            d["total_sh"] += sh

    skew_buy_total = 0.0
    skew_sell_total = 0.0
    for day in sorted_days[:7]:
        d = by_day_signed.get(day, {})
        if not d.get("prints"):
            continue
        # Volume-weighted average price for the day
        vwap = d["total_val"] / d["total_sh"] if d["total_sh"] else 0
        if not vwap:
            continue
        buy_val = sell_val = neutral_val = 0.0
        for px, v, _bid, _ask in d["prints"]:
            # NBBO MIDPOINT SIGNING (Jul 14 2026). Lee-Ready-lite: the print's
            # own quote context, not the day's VWAP (which mega-prints dominate
            # by construction). VWAP band kept only as fallback for rows
            # missing NBBO (pre-2026-05-29 unhealable slices).
            if _bid and _ask and _ask >= _bid:
                mid = (_bid + _ask) / 2.0
                if px > mid:
                    buy_val += v
                elif px < mid:
                    sell_val += v
                else:
                    neutral_val += v
            elif px > vwap * 1.0005:
                buy_val += v
            elif px < vwap * 0.9995:
                sell_val += v
            else:
                neutral_val += v
        net = buy_val - sell_val
        denom = buy_val + sell_val
        total_dp = denom + neutral_val
        coverage = (denom / total_dp * 100) if total_dp else 0
        skew_pct = (net / denom * 100) if denom else 0
        usable = (denom >= MIN_CLASSIFIED_USD
                  and len(d["prints"]) >= MIN_CLASSIFIED_PRINTS
                  and coverage >= MIN_COVERAGE_PCT
                  and not _partial(day))   # coverage is classified/stored, NOT
                  # stored/actual -- a frozen 500-print slice reports high coverage
                  # on ~2% of the tape. Exclude it from the aggregate outright.
        n_days += 1
        if usable:
            n_usable += 1
            skew_buy_total += buy_val
            skew_sell_total += sell_val
        mark = "" if usable else ("  PARTIAL" if _partial(day) else "  THIN")
        print(f"  {day:<12} {buy_val:>16,.0f} {sell_val:>16,.0f} "
              f"{net:>+14,.0f} {skew_pct:>+7.1f}% {coverage:>6.1f}%{mark}")

    if skew_buy_total or skew_sell_total:
        denom = skew_buy_total + skew_sell_total
        agg_skew = ((skew_buy_total - skew_sell_total) / denom * 100) if denom else 0
        # SIGNED-UNIVERSE LINE (SMCI specimen: +0.9% skew silently described
        # only 57% of a $407M day). Headline and signed values on the same
        # line so no report can mistake the skew's universe again.
        _headline7 = sum(by_day.get(_d, {}).get("value", 0.0) for _d in sorted_days[:7])
        if _headline7:
            print(f"  Signed universe: ${denom:,.0f} of ${_headline7:,.0f} headline "
                  f"({denom / _headline7 * 100:.0f}%) -- remainder is AH/neutral/unclassified")
        # Flag persistent skew > 30% across the 7-day window
        if abs(agg_skew) >= 30 and denom >= 10_000_000 and n_usable >= 3:
            direction = "BUY" if agg_skew > 0 else "SELL"
            flag("MED", ticker,
                 f"7d signed dark pool skew: {agg_skew:+.1f}% ({direction}-side, "
                 f"VWAP heuristic — confirm with Lee-Ready)")
        # HONEST LABEL (Jul 11 2026): the aggregate is computed on USABLE sessions
        # only. Saying "7-day" when 4 of 7 were THIN invites over-trust in what may
        # be a single day's number. Say what it actually rests on.
        print(f"\n  {n_days}-day aggregate skew ({n_usable} usable): {agg_skew:+.1f}%  "
              f"(buy ${skew_buy_total:,.0f} vs sell ${skew_sell_total:,.0f})")
        if n_usable < 3:
            print(f"  ⚠️  only {n_usable} usable session(s) — treat the aggregate as noise")
        print("  [note] NBBO-midpoint signing (Lee-Ready-lite, no tick test). At-mid prints neutral; pre-May-29 rows fall back to VWAP.")


# ---------------------------------------------------------------------------
# Section: options flow (most actionable signal pre-earnings)
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

    # AGGRESSOR TILT (2026-07-24). The UW payload carries per-alert
    # total_ask_side_prem / total_bid_side_prem -- side-of-trade premium the
    # monitor previously dropped, forcing every position-type read to [inf].
    # Ask-side = paid the offer = aggressive buying of that contract type.
    # DESCRIPTIVE ONLY: zero validation history; inherits the dark-pool scope
    # rule verbatim (never leads a signal; feature-sink only via full RULE-1
    # checklist after a null-controlled per-date IC test).
    def _sides(sub):
        a = sum(float(x.get("total_ask_side_prem") or 0) for x in sub)
        b = sum(float(x.get("total_bid_side_prem") or 0) for x in sub)
        return a, b
    _calls = [x for x in rows if (x.get("type") or x.get("option_type") or "").lower() == "call"]
    _puts = [x for x in rows if (x.get("type") or x.get("option_type") or "").lower() == "put"]
    _ca, _cb = _sides(_calls)
    _pa, _pb = _sides(_puts)
    _tot = call_prem + put_prem
    _cov = ((_ca + _cb + _pa + _pb) / _tot * 100) if _tot else 0.0
    def _tilt(a, b):
        return ((a - b) / (a + b) * 100) if (a + b) else 0.0
    print(f"  Aggressor tilt (ask=aggr buy): "
          f"CALLS {_tilt(_ca, _cb):+.1f}% (ask ${_ca:,.0f} / bid ${_cb:,.0f})  "
          f"PUTS {_tilt(_pa, _pb):+.1f}% (ask ${_pa:,.0f} / bid ${_pb:,.0f})")
    print(f"  Side coverage: {_cov:.1f}% of premium classified "
          f"(rest at-mid/unclassified){'  [LOW -- wide error bars]' if _cov < 70 else ''}")
    if _cov > 105:
        print(f"  [warn] side sums exceed total premium by {_cov - 100:.0f}% -- "
              f"field scaling suspect, do not trust tilt until reconciled")

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
        _a = float(r.get("total_ask_side_prem") or 0)
        _b = float(r.get("total_bid_side_prem") or 0)
        _ashare = f"A%={_a / p * 100:.0f}" if p and (_a or _b) else "A%=?"
        _op = r.get("all_opening_trades")
        _optag = "OPENING" if _op in (True, "true", 1) else ("mixed" if _op in (False, "false", 0) else "?")
        print(f"    {(r.get('executed_at') or r.get('created_at') or '')[:19]}  "
              f"{otype:<5} ${strike} {exp}  prem=${p:,.0f}  "
              f"vol={r.get('volume') or '?'}  oi={r.get('open_interest') or '?'}  "
              f"{_ashare}  [{_optag}]")


# ---------------------------------------------------------------------------
# Section: short interest
# ---------------------------------------------------------------------------

def section_short_interest(conn: sqlite3.Connection, ticker: str) -> None:
    print(f"\n=== SHORT INTEREST — {ticker} ===")

    data = uw_get(f"/api/shorts/{ticker}/interest-float/v2")
    rows = (data or {}).get("data") or []
    if not rows:
        import os as _os
        _si_db = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "short_interest.db")
        fin = []
        if _os.path.exists(_si_db):
            try:
                _con = sqlite3.connect(_si_db)
                fin = list(_con.execute(
                    "SELECT settlement_date, current_short, avg_daily_vol, days_to_cover "
                    "FROM short_interest WHERE ticker=? ORDER BY settlement_date DESC LIMIT 8",
                    (ticker,)))
                _con.close()
            except sqlite3.Error as e:
                print(f"  [warn] FINRA fallback read failed: {e}")
        if not fin:
            print("  No short interest data returned (UW empty; no FINRA rows either).")
            return
        print("  UW endpoint empty -- FINRA fallback (short_interest.db, bi-monthly settlements):")
        print(f"  {'Settlement':<12} {'Short shares':>14} {'Avg daily vol':>14} {'DTC':>7}")
        for i, (sd, cs, adv, dtc) in enumerate(fin):
            chg = ""
            if i + 1 < len(fin) and fin[i+1][1]:
                chg = f"   ({(cs - fin[i+1][1]) / fin[i+1][1] * 100:+.1f}% shares vs prior)"
            print(f"  {sd:<12} {cs:>14,.0f} {adv:>14,.0f} {dtc:>7.2f}{chg}")
        print("  [note] DTC = shares/ADV -- check BOTH columns before reading a DTC move as positioning.")
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
                float(r.get("si_float") or r.get("short_percent_of_float") or
                      r.get("pct_of_float") or r.get("percent_of_float") or 0),
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
        pct = float(r.get("si_float") or r.get("short_percent_of_float") or
                    r.get("pct_of_float") or r.get("percent_of_float") or 0)
        dtc = float(r.get("days_to_cover") or r.get("dtc") or 0)
        print(f"  {d[:12]:<12} {si:>14,.0f} {pct:>8.2%} {dtc:>7.1f}")

    # Trend signal
    dated = [r for r in rows if get_date(r)]
    if len(dated) >= 2:
        srt = sorted(dated, key=lambda x: get_date(x))
        latest_pct = float(srt[-1].get("si_float") or
                           srt[-1].get("short_percent_of_float") or
                           srt[-1].get("pct_of_float") or
                           srt[-1].get("percent_of_float") or 0)
        prev_pct = float(srt[-2].get("si_float") or
                         srt[-2].get("short_percent_of_float") or
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

# Forms we care about for earnings-monitoring context.
# Severity tunes the flag-summary noise level; we want material events
# (8-K), insider activity (Form 4), and quarterly reports (10-Q) prominent.
EDGAR_FORM_PRIORITY = {
    # Form         (severity, human-readable description)
    "8-K":        ("MED",  "8-K — material event (could be pre-announce or guidance)"),
    "10-Q":       ("INFO", "10-Q — quarterly report"),
    "10-K":       ("INFO", "10-K — annual report"),
    "10-Q/A":     ("MED",  "10-Q/A — quarterly amendment (often material)"),
    "10-K/A":     ("HIGH", "10-K/A — annual amendment (highest severity if recent)"),
    "4":          ("INFO", "Form 4 — insider transaction (parsed below)"),
    "4/A":        ("INFO", "Form 4/A — insider amendment"),
    "144":        ("INFO", "Form 144 — proposed insider sale"),
    "NT 10-Q":    ("HIGH", "NT 10-Q — late filing notification (RED FLAG)"),
    "NT 10-K":    ("HIGH", "NT 10-K — late filing notification (RED FLAG)"),
    "S-3":        ("INFO", "S-3 — securities registration"),
    "S-8":        ("INFO", "S-8 — employee securities registration"),
    "424B":       ("INFO", "424B — prospectus supplement"),
    "SC 13G":     ("MED",  "13G — passive 5%+ position"),
    "SC 13G/A":   ("INFO", "13G/A — passive position amendment"),
    "SC 13D":     ("HIGH", "13D — activist 5%+ position"),
    "SC 13D/A":   ("HIGH", "13D/A — activist position amendment"),
    "DEF 14A":    ("INFO", "DEF 14A — definitive proxy statement"),
    "8-K/A":      ("MED",  "8-K/A — amendment to material event"),
}


# ---------------------------------------------------------------------------
# Section: earnings calendar context (date, days-until, history, reactions)
# ---------------------------------------------------------------------------

def _try_uw_endpoints(paths: list[str], description: str = "") -> Optional[dict]:
    """Try a list of UW endpoint paths in order; return the first that
    returns a non-empty payload."""
    for path in paths:
        try:
            data = uw_get(path)
        except Exception:
            data = None
        if data:
            payload = data.get("data") if isinstance(data, dict) else data
            if payload:
                return data
    return None


def _f(d: dict, *keys, default=None):
    """Return the first non-empty value from `d` matching any of `keys`.
    Lets us cope with UW's inconsistent field names across endpoints."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        v = d.get(k)
        if v not in (None, "", "null"):
            return v
    return default


def section_earnings_calendar(ticker: str) -> None:
    print(f"\n=== EARNINGS CALENDAR — {ticker} ===")

    cfg = TICKER_CONFIG.get(ticker.upper(), {})
    earnings_date = cfg.get("earnings_date")
    earnings_time = cfg.get("earnings_time", "?")
    fiscal_q = cfg.get("fiscal_q", "?")

    if earnings_date:
        d_until = days_until_earnings(ticker)
        if d_until is not None:
            if d_until < 0:
                status = f"({-d_until} day(s) AGO — likely already reported)"
            elif d_until == 0:
                status = "📍 TODAY"
            elif d_until <= 7:
                status = f"⚠️  {d_until} days — EARNINGS WEEK"
            elif d_until <= 14:
                status = f"⚠️  {d_until} days — quiet period"
            else:
                status = f"{d_until} days out"
            print(f"  Next earnings:   {earnings_date} {earnings_time}  ({fiscal_q})")
            print(f"  Status:          {status}")
            if 0 <= d_until <= 14:
                flag("INFO", ticker,
                     f"Earnings in {d_until} day(s): {earnings_date} {earnings_time}")
    else:
        print(f"  No earnings date configured for {ticker}.")

    # Try multiple UW earnings endpoints — different plans expose different
    # paths and field names, so we cast a wide net.
    data = _try_uw_endpoints([
        f"/api/earnings/{ticker.upper()}",
        f"/api/stock/{ticker.upper()}/earnings",
        f"/api/earnings/historical/{ticker.upper()}",
        f"/api/stock/{ticker.upper()}/earnings-history",
        f"/api/earnings/afterhours/{ticker.upper()}",
        f"/api/earnings/premarket/{ticker.upper()}",
    ])

    rows = (data or {}).get("data") or (data if isinstance(data, list) else None) or []
    if not rows:
        print(f"  (No historical earnings data returned by UW for {ticker}.)")
        return

    print(f"\n  Historical earnings reactions (last 8 prints if available):")
    print(f"  {'Date':<12} {'EPS Act':>10} {'EPS Est':>10} {'Surprise':>10} "
          f"{'Rev Act':>14} {'Rev Surp':>10} {'1d move':>10}")

    sorted_rows = sorted(rows, key=lambda r: (
        _f(r, "report_date", "date", "earnings_date", "fiscal_period_end",
           "report_period", default="")
    ), reverse=True)[:8]

    moves: list[float] = []
    # 1D-MOVE BACKFILL (Jul 25 2026). UW's move field is absent for most rows
    # (8 of 8 rendered "?" on PLTR/GOOG) -- the implied-vs-realized ledger had
    # no denominator. Compute from canonical closes when the vendor is silent:
    # AMC print -> next session close / print-day close; BMO -> print-day /
    # prior. Computed values carry a * marker.
    _px: dict = {}
    try:
        import sqlite3 as _sq3
        _c = _sq3.connect(f"file:{_REPO_ROOT / 'prices.db'}?mode=ro", uri=True)
        for _d, _cl in _c.execute(
                "SELECT date, adj_close FROM daily_prices WHERE ticker=? "
                "AND adj_close IS NOT NULL ORDER BY date", (ticker.upper(),)):
            _px[str(_d)[:10]] = float(_cl)
        _c.close()
    except Exception as _e:
        print(f"  [warn] backfill closes unavailable: {_e}")
    _pdates = sorted(_px)
    def _move_from_prices(rd: str, rtime: str):
        if not _px or not rd or rd not in _px:
            return None
        import bisect as _bi
        _i = _bi.bisect_left(_pdates, rd)
        _amc = "post" in (rtime or "").lower() or "amc" in (rtime or "").lower()                or "after" in (rtime or "").lower()
        try:
            if _amc:
                if _i + 1 >= len(_pdates):
                    return None
                return (_px[_pdates[_i + 1]] / _px[rd] - 1) * 100
            if _i == 0:
                return None
            return (_px[rd] / _px[_pdates[_i - 1]] - 1) * 100
        except (KeyError, ZeroDivisionError):
            return None
    _any_backfilled = False
    for r in sorted_rows:
        rdate = (_f(r, "report_date", "date", "earnings_date",
                    "fiscal_period_end", "report_period", default="?") or "?")[:10]

        eps_act = _f(r, "actual_eps", "eps_actual", "eps", "earnings_per_share",
                     "eps_act")
        eps_est = _f(r, "expected_eps", "eps_estimate", "eps_consensus",
                     "consensus_eps", "estimated_eps", "eps_est",
                     "eps_consensus_estimate", "estimate_eps",
                     "street_mean_est")  # actual UW key -- without it every Est
                                         # rendered "?" and no surprise computed
        if (ticker.upper(), rdate) in EPS_QUARANTINE or ticker.upper() in EPS_QUARANTINE_TICKERS:
            eps_act = None  # renders "?" -- see EPS_QUARANTINE note
        rev_act = _f(r, "actual_revenue", "revenue_actual", "revenue",
                     "total_revenue", "revenue_act")
        rev_est = _f(r, "expected_revenue", "revenue_estimate",
                     "revenue_consensus", "consensus_revenue",
                     "estimated_revenue", "rev_est")
        move = _f(r, "post_earnings_move", "price_change_pct", "reaction_1d",
                  "price_reaction", "post_earnings_drift", "next_day_return",
                  "1d_change", "post_earnings_change_pct", "next_day_change_pct")
        _computed_move = False
        if move is None:
            _bf = _move_from_prices(rdate, str(_f(r, "report_time", "time", default="") or ""))
            if _bf is not None:
                move = _bf
                _computed_move = True
                _any_backfilled = True

        # Compute surprises if not provided directly
        surprise = _f(r, "eps_surprise", "surprise_pct", "eps_surprise_pct",
                      "surprise")
        if surprise is None:
            try:
                if eps_act is not None and eps_est not in (None, 0):
                    surprise = (float(eps_act) - float(eps_est)) / abs(float(eps_est)) * 100
            except (TypeError, ValueError):
                pass
        # Sanity tripwire, not a defense: 5.3% of rows exceed |100%| legitimately
        # (near-zero estimates), and the GOOG 04-29 contamination sat at +94%,
        # under any bound. Known-bad rows are quarantined by list in
        # daily_uw_snapshot.py; this only surfaces NEW suspects for review.
        if surprise is not None and abs(float(surprise)) > 100:
            flag("MED", ticker,
                 f"EPS surprise {float(surprise):+.0f}% on {rdate} exceeds sanity bound "
                 f"-- GAAP/equity-mark contamination? Verify basis before trusting")
        rev_surprise = _f(r, "revenue_surprise", "rev_surprise_pct")
        if rev_surprise is None:
            try:
                if rev_act is not None and rev_est not in (None, 0):
                    rev_surprise = (float(rev_act) - float(rev_est)) / abs(float(rev_est)) * 100
            except (TypeError, ValueError):
                pass

        try:
            move_f = float(move) if move is not None else None
            if move_f is not None and abs(move_f) < 1.5 and not _computed_move:
                # Normalize decimal (0.05) to percent (5.0). Vendor fields are
                # decimals; the prices.db backfill is ALREADY percent -- without
                # the _computed_move gate a real +1.4% computed move would
                # render +140%.
                move_f *= 100
            if move_f is not None:
                moves.append(move_f)
        except (TypeError, ValueError):
            move_f = None

        eps_act_s = f"${float(eps_act):.2f}" if eps_act is not None else "?"
        eps_est_s = f"${float(eps_est):.2f}" if eps_est is not None else "?"
        sur_s = f"{float(surprise):+.1f}%" if surprise is not None else "?"
        try:
            ra = float(rev_act) if rev_act else 0
            rev_act_s = (f"${ra/1e9:.2f}B" if ra > 1e9 else
                         f"${ra/1e6:.0f}M" if ra > 1e6 else
                         "?" if not ra else f"${ra:,.0f}")
        except (TypeError, ValueError):
            rev_act_s = "?"
        rev_sur_s = f"{float(rev_surprise):+.1f}%" if rev_surprise is not None else "?"
        move_s = ((f"{move_f:+.2f}%*" if _computed_move else f"{move_f:+.2f}%")
                  if move_f is not None else "?")

        print(f"  {rdate:<12} {eps_act_s:>10} {eps_est_s:>10} {sur_s:>10} "
              f"{rev_act_s:>14} {rev_sur_s:>10} {move_s:>10}")

    if _any_backfilled:
        print(f"  (* = 1d move computed from prices.db closes, not vendor-supplied)")
    if moves:
        avg_abs = sum(abs(m) for m in moves) / len(moves)
        max_up = max(moves)
        max_dn = min(moves)
        print(f"\n  Historical 1-day move stats: avg |move| = {avg_abs:.1f}%,  "
              f"max up = {max_up:+.1f}%,  max down = {max_dn:+.1f}%")
        # Stash so the implied move section can compute rich/cheap ratio
        cfg["_avg_abs_move"] = avg_abs


# ---------------------------------------------------------------------------
# Section: implied earnings move (from straddle pricing)
# ---------------------------------------------------------------------------
# Buying the ATM straddle (call + put at same strike, nearest expiry after
# earnings) costs roughly stock_price × implied_move%. So:
#     implied_move ≈ (call_mid + put_mid) / spot
# This tells us how much the options market thinks the stock will move on
# the print, regardless of direction.

def _extract_atm_straddle(chain_rows: list, on_or_after: str, spot: float) -> Optional[tuple]:
    """Given option chain rows from any UW endpoint shape, return
    (expiry, strike, call_mid, put_mid) for the ATM straddle of the
    nearest expiry on or after earnings_date. Returns None if shape
    doesn't yield a usable straddle."""
    import re as _re
    def _occ(c):
        """Parse OCC option_symbol -> (expiry 'YYYY-MM-DD', 'C'|'P', strike float).
        UW rows carry strike/type ONLY in option_symbol, e.g.
        MSFT260828C00345000 -> ('2026-08-28','C',345.0). Root up to 6 chars."""
        sym = _f(c, "option_symbol", "symbol", "osi")
        if not isinstance(sym, str):
            return (None, None, None)
        m = _re.match(r'^([A-Z]{1,6})(\d{6})([CP])(\d{8})$', sym.strip())
        if not m:
            return (None, None, None)
        _root, ymd, cp, strike = m.groups()
        return (f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}", cp, int(strike) / 1000.0)
    if not chain_rows:
        return None

    # Group by expiry
    by_expiry: dict = {}
    for c in chain_rows:
        if not isinstance(c, dict):
            continue
        exp = _f(c, "expires", "expiry", "expiration", "expiration_date", "expires_at")
        if not exp:
            exp = _occ(c)[0]
        if not exp:
            continue
        exp_str = exp[:10] if isinstance(exp, str) else str(exp)[:10]
        if exp_str < on_or_after:
            continue
        by_expiry.setdefault(exp_str, []).append(c)

    if not by_expiry:
        return None

    chosen_exp = min(by_expiry.keys())
    expiry_rows = by_expiry[chosen_exp]

    def get_strike(c):
        _e, _cp, _k = _occ(c)
        if _k is not None:
            return _k
        s = _f(c, "strike", "strike_price")
        try:
            return float(s) if s is not None else None
        except (TypeError, ValueError):
            return None

    rows_with_strikes = [(c, get_strike(c)) for c in expiry_rows]
    rows_with_strikes = [(c, s) for c, s in rows_with_strikes if s is not None]
    if not rows_with_strikes:
        return None

    # Schema 1: each row contains BOTH call_mid and put_mid for a strike
    rows_with_strikes.sort(key=lambda x: abs(x[1] - spot))
    chosen_row, chosen_strike = rows_with_strikes[0]
    call_mid = _f(chosen_row, "call_mid", "c_mid", "call_price",
                  "call_last", "call_bid_ask_mid", "callMid")
    put_mid = _f(chosen_row, "put_mid", "p_mid", "put_price",
                 "put_last", "put_bid_ask_mid", "putMid")
    try:
        call_mid = float(call_mid) if call_mid is not None else 0.0
        put_mid = float(put_mid) if put_mid is not None else 0.0
    except (TypeError, ValueError):
        call_mid = put_mid = 0.0

    if call_mid and put_mid:
        return (chosen_exp, chosen_strike, call_mid, put_mid)

    # Schema 2: rows are individual call/put options, with type field
    def _cp(r):
        _e, cp, _k = _occ(r)
        return cp or (_f(r, "type", "option_type") or "")[:1].upper()
    calls = [(r, s) for r, s in rows_with_strikes if _cp(r) == "C"]
    puts = [(r, s) for r, s in rows_with_strikes if _cp(r) == "P"]
    if calls and puts:
        calls.sort(key=lambda x: abs(x[1] - spot))
        puts.sort(key=lambda x: abs(x[1] - spot))
        cr, c_strike = calls[0]
        pr, p_strike = puts[0]
        # Try mid (single field) or bid/ask average
        c_mid = _f(cr, "mid", "last", "bid_ask_mid")
        p_mid = _f(pr, "mid", "last", "bid_ask_mid")
        try:
            c_mid = float(c_mid) if c_mid is not None else 0.0
            p_mid = float(p_mid) if p_mid is not None else 0.0
        except (TypeError, ValueError):
            c_mid = p_mid = 0.0
        if not c_mid or not p_mid:
            try:
                cb = float(_f(cr, "bid", "best_bid") or 0)
                ca = float(_f(cr, "ask", "best_ask") or 0)
                pb = float(_f(pr, "bid", "best_bid") or 0)
                pa = float(_f(pr, "ask", "best_ask") or 0)
                if cb and ca:
                    c_mid = (cb + ca) / 2
                if pb and pa:
                    p_mid = (pb + pa) / 2
            except (TypeError, ValueError):
                pass
        if c_mid and p_mid:
            return (chosen_exp, c_strike, c_mid, p_mid)

    return None


def section_implied_move(ticker: str) -> None:
    print(f"\n=== IMPLIED EARNINGS MOVE — {ticker} ===")

    cfg = TICKER_CONFIG.get(ticker.upper(), {})
    earnings_date = cfg.get("earnings_date")
    # Route through the stale-date guard (days_until_earnings returns None for
    # unconfigured AND stale dates). Before this, a stale config date silently
    # anchored the straddle on the front weekly -- NVDA specimen, Jul 24 report:
    # guard warned but this section read cfg directly and mis-anchored anyway.
    _generic_prior = False
    if earnings_date and days_until_earnings(ticker) is None:
        earnings_date = None
    _du = days_until_earnings(ticker) if earnings_date else None
    if not earnings_date or (_du is not None and _du > 15):
        earnings_date = (_today_et() + timedelta(days=5)).isoformat()
        print(f"  Generic prior (no earnings within 15d): anchor {earnings_date} = today ET +5d.")
        _generic_prior = True

    # Live spot price — try realtime first, fall back to most recent OHLC
    spot = None
    realtime = uw_get(f"/api/stock/{ticker}/stock-state") or uw_get(f"/api/stock/{ticker}/realtime")
    if realtime:
        rt = realtime.get("data") if isinstance(realtime, dict) else realtime
        if isinstance(rt, list) and rt:
            rt = rt[0]
        if isinstance(rt, dict):
            try:
                spot = float(_f(rt, "last", "price", "market_price",
                                "close", "last_price") or 0) or None
            except (TypeError, ValueError):
                spot = None

    if not spot:
        spot_data = uw_get(f"/api/stock/{ticker}/ohlc/1d", params={"limit": 5})
        spot_rows = (spot_data or {}).get("data") or []
        if spot_rows:
            spot_rows.sort(key=lambda r: (r.get("date") or r.get("market_time") or
                                          r.get("start_time") or ""), reverse=True)
            try:
                spot = float(_f(spot_rows[0], "close", "market_price", "last") or 0)
            except (TypeError, ValueError):
                spot = None

    if not spot:
        print(f"  Spot price unavailable.")
        return

    # SPOT = latest completed close from canonical daily series (was quote-first:
    # produced quote-anchored implied ranges in 4/4 Jul-21 reports; the OHLC
    # fallback was worse -- reverse-stable sort landed on the PRE-MARKET row).
    _quote_ctx = spot
    _dr = _daily_returns(ticker) or {}
    _dc = _dr.get("latest")
    _spot_bar = _dr.get("bar_date") or "?"
    if _dc:
        spot = float(_dc)
    if _quote_ctx and abs(_quote_ctx - spot) > 0.005:
        print(f"  Spot:           ${spot:.2f}  (close {_spot_bar}; live quote ${_quote_ctx:.2f})")
    else:
        print(f"  Spot:           ${spot:.2f}  (close {_spot_bar})")

    # Step 1: discover available expiries via expiry-breakdown (which returns
    # the list of expirations with volume/OI per expiry — no extra param needed)
    expiry_data = uw_get(f"/api/stock/{ticker}/expiry-breakdown")
    expiry_rows = (expiry_data or {}).get("data") or []
    expiries = []
    for r in expiry_rows:
        if not isinstance(r, dict):
            continue
        exp = _f(r, "expires", "expiry", "expiration", "expiration_date")
        if exp:
            exp_str = exp[:10] if isinstance(exp, str) else str(exp)[:10]
            expiries.append(exp_str)
    expiries = sorted(set(expiries))

    if not expiries:
        print(f"  Could not discover available expiries.")
        print(f"  → Manual lookup: nearest expiry after {earnings_date} on UW.")
        return

    # Pick the nearest expiry on or after earnings
    target_expiry = next((e for e in expiries if e >= earnings_date), None)
    if not target_expiry:
        print(f"  No expiries available on or after earnings date {earnings_date}.")
        return
    print(f"  Target expiry:  {target_expiry}  (nearest >= {earnings_date} of {len(expiries)} available)")

    # Step 2: call atm-chains with the expiration parameter explicitly set.
    # UW's pattern across multiple endpoints is `expirations[]=YYYY-MM-DD`.
    chains = uw_get(f"/api/stock/{ticker}/atm-chains",
                    params={"expirations[]": target_expiry})
    chain_data = (chains or {}).get("data") if isinstance(chains, dict) else chains
    if not chain_data:
        # Try the singular form some endpoints use
        chains = uw_get(f"/api/stock/{ticker}/atm-chains",
                        params={"expirations": target_expiry})
        chain_data = (chains or {}).get("data") if isinstance(chains, dict) else chains
    if not chain_data:
        # Last resort: try option-chains with the same param
        chains = uw_get(f"/api/stock/{ticker}/option-chains",
                        params={"expirations[]": target_expiry})
        chain_data = (chains or {}).get("data") if isinstance(chains, dict) else chains

    if not chain_data:
        print(f"  Could not fetch ATM chain for {target_expiry}.")
        print(f"  → Manual lookup: ATM straddle for {target_expiry} on UW.")
        return

    if not isinstance(chain_data, list):
        chain_data = [chain_data]

    result = _extract_atm_straddle(chain_data, earnings_date, spot)
    if not result:
        # Show what we got back so we can adapt the parser
        sample = chain_data[0] if chain_data else {}
        if isinstance(sample, dict):
            print(f"  Got chain data but couldn't extract straddle.")
            print(f"  Sample row keys: {list(sample.keys())[:15]}")
        else:
            print(f"  Got chain data in unexpected shape: {type(sample).__name__}")
        return

    chosen_exp, chosen_strike, call_mid, put_mid = result
    straddle = call_mid + put_mid
    implied_pct = (straddle / spot) * 100
    print(f"  Strike:         ${chosen_strike:.2f}")
    print(f"  Call mid:       ${call_mid:.2f}")
    print(f"  Put mid:        ${put_mid:.2f}")
    print(f"  Straddle cost:  ${straddle:.2f}")
    print(f"  Implied move:   ±{implied_pct:.2f}%")
    print(f"  Implied range:  ${spot - straddle:.2f}  to  ${spot + straddle:.2f}")
    # CARRY-ADJUSTED PARITY (Jul 26 2026). C - P = S - K*e^(-rT), not S - K:
    # the undiscounted check flagged pure cost-of-carry as failure (GOOG 6m
    # "FAIL 7.59" vs K*(1-e^(-rT)) ~= $8.1 at r=4.5% -- the "failure" WAS the
    # carry). r approximates the 3m bill [fact? Jul 2026]; dividend yield
    # ignored (GOOG/MSFT <1%, absorbed by tolerance). Refresh r if regime moves.
    import math as _m
    _RF = 0.045
    _T0 = max((date.fromisoformat(str(chosen_exp)[:10]) - _today_et()).days, 0) / 365.0
    _fpar = abs((call_mid - put_mid) - (spot - chosen_strike * _m.exp(-_RF * _T0)))
    if _fpar > max(0.015 * spot, 0.50):
        print(f"  [warn] Put-call parity gap ${_fpar:.2f} -- stale/crossed chain; do NOT anchor to this straddle.")
        flag("MED", ticker, f"Implied-move chain fails parity by ${_fpar:.2f} at {chosen_exp}")

    hist = cfg.get("_avg_abs_move")
    if hist and _generic_prior:
        print(f"  Historical avg: ±{hist:.1f}% is EARNINGS-DAY history -- not comparable "
              f"to a non-earnings weekly straddle; RICH/CHEAP comparison suppressed.")
        hist = None
    if hist:
        ratio = implied_pct / hist
        if ratio >= 1.4:
            print(f"  Historical avg: ±{hist:.1f}% — implied is {ratio:.2f}x (RICH)")
            flag("INFO", ticker,
                 f"Options pricing rich: ±{implied_pct:.1f}% implied vs "
                 f"{hist:.1f}% historical ({ratio:.2f}x)")
        elif ratio <= 0.7:
            print(f"  Historical avg: ±{hist:.1f}% — implied is {ratio:.2f}x (CHEAP)")
            flag("INFO", ticker,
                 f"Options pricing cheap: ±{implied_pct:.1f}% implied vs "
                 f"{hist:.1f}% historical ({ratio:.2f}x)")
        else:
            print(f"  Historical avg: ±{hist:.1f}% — implied is {ratio:.2f}x (fair)")
    print(f"  Term structure (ATM straddle per horizon):")
    _seen_exp = {chosen_exp}
    for _lbl, _days in (("1m", 30), ("3m", 91), ("6m", 182), ("12m", 365)):
        _anchor = (_today_et() + timedelta(days=_days)).isoformat()
        _texp = next((e for e in expiries if e >= _anchor), None)
        if not _texp:
            print(f"    {_lbl:>3}: no listed expiry >= {_anchor}")
            continue
        if _texp in _seen_exp:
            print(f"    {_lbl:>3}: exp {_texp} (same as prior bucket)")
            continue
        _ch = uw_get(f"/api/stock/{ticker}/atm-chains", params={"expirations[]": _texp})
        _cd = (_ch or {}).get("data") if isinstance(_ch, dict) else _ch
        if not _cd:
            _ch = uw_get(f"/api/stock/{ticker}/atm-chains", params={"expirations": _texp})
            _cd = (_ch or {}).get("data") if isinstance(_ch, dict) else _ch
        if not _cd:
            print(f"    {_lbl:>3}: exp {_texp} -- no chain data")
            continue
        if not isinstance(_cd, list):
            _cd = [_cd]
        _res = _extract_atm_straddle(_cd, _texp, spot)
        if not _res:
            print(f"    {_lbl:>3}: exp {_texp} -- no usable straddle")
            continue
        _e, _k, _c, _pm = _res
        _st = _c + _pm
        _Tt = max((date.fromisoformat(str(_e)[:10]) - _today_et()).days, 0) / 365.0
        _par = abs((_c - _pm) - (spot - _k * _m.exp(-_RF * _Tt)))
        _ptag = "" if _par <= max(0.015 * spot, 0.50) else f"  [PARITY FAIL {_par:.2f} carry-adj]"
        _seen_exp.add(_e)
        print(f"    {_lbl:>3}: exp {_e}  K ${_k:.2f}  straddle ${_st:.2f}  ±{(_st/spot)*100:.2f}%{_ptag}")


# ---------------------------------------------------------------------------
# Section: SEC EDGAR filings (filings BY this company, earnings-relevant)
# ---------------------------------------------------------------------------

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

    # For earnings monitoring we only care about filings BY this company.
    # (For event-driven situations like proxy fights, we'd also pull 3rd-party
    # 13Ds via full-text search — but here the company's own 8-K cadence is
    # what matters: pre-announcements, guidance updates, surprise filings.)
    all_hits = company_hits

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
# Section: 8-K content snippet (pre-announcement detection)
# ---------------------------------------------------------------------------
# 8-K filings can contain material info: pre-announcements, guidance updates,
# investor day announcements, M&A. Without reading the content, we only see
# the form code. This section fetches recent 8-Ks, extracts the first ~600
# words, and surfaces pre-announcement language patterns.

# Stylized phrases that indicate the 8-K contains earnings/guidance content
PREANNOUNCE_PATTERNS = [
    "expected results", "preliminary results", "preliminary financial",
    "updates its outlook", "updates outlook", "revised guidance",
    "raises guidance", "lowers guidance", "lowers its guidance",
    "raises its guidance", "withdraws guidance", "withdrawing its outlook",
    "below previous guidance", "above previous guidance",
    "expects revenue", "expects earnings", "anticipates revenue",
    "anticipated revenue", "now expects", "no longer expects",
    "preliminary unaudited", "fourth quarter results", "first quarter results",
    "second quarter results", "third quarter results",
    "weaker than expected", "stronger than expected",
    "below the company's prior guidance", "above the company's prior guidance",
]


def section_eightk_content(conn: sqlite3.Connection, ticker: str) -> None:
    """For each 8-K filed in the last 30 days, fetch and scan for
    pre-announcement language. Only flag if material patterns matched."""
    print(f"\n=== 8-K CONTENT SCAN — {ticker} ===")

    cik = get_cik(ticker)
    if cik is None:
        print(f"  No CIK for {ticker}; skipping.")
        return

    cur = conn.cursor()
    cutoff = (_today_et() - timedelta(days=30)).isoformat()
    rows = list(cur.execute("""
        SELECT accession_number, filing_date, primary_doc
        FROM edgar_filings
        WHERE ticker = ? AND form_type IN ('8-K', '8-K/A')
          AND filing_date >= ?
        ORDER BY filing_date DESC
    """, (ticker, cutoff)))

    if not rows:
        print(f"  No 8-K filings in the last 30 days.")
        return

    print(f"  Scanning {len(rows)} 8-K filing(s) from last 30 days...")
    for acc, fdate, primary in rows[:5]:  # cap at 5 most recent
        # Prefer EX-99 exhibits (the press release with actual financials);
        # filer naming varies: ex99, ex-99, exhibit99, exhibit991q22026...
        text = None
        _src = None
        import re as _re9
        acc_clean = acc.replace("-", "")
        _base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}"
        edgar_throttle()
        try:
            _r = requests.get(f"{_base}/index.json",
                              headers={"User-Agent": EDGAR_UA}, timeout=REQUEST_TIMEOUT)
            _r.raise_for_status()
            _items = (_r.json().get("directory") or {}).get("item") or []
            _exs = [it for it in _items
                    if _re9.search(r"ex(hibit)?[-_.]?99", (it.get("name") or "").lower())
                    and (it.get("name") or "").lower().endswith((".htm", ".html", ".txt"))]
            _exs.sort(key=lambda it: int(it.get("size") or 0), reverse=True)
            if _exs:
                _src = _exs[0].get("name")
                text = edgar_get_text(f"{_base}/{_src}")
        except Exception:
            pass
        if not text and primary:
            url = build_form4_url(cik, acc, primary)  # reuse the URL builder
            text = edgar_get_text(url)
            _src = _src or primary
        if not text:
            xml_url = find_form4_xml_url(cik, acc)
            if xml_url:
                text = edgar_get_text(xml_url)
                _src = _src or "xml"
        if not text:
            continue
        if _src:
            print(f"  [source: {_src}]")

        # Crude HTML strip — drop tags, collapse whitespace
        import re as _re
        cleaned = _re.sub(r"<[^>]+>", " ", text)
        cleaned = _re.sub(r"\s+", " ", cleaned).strip()
        # Skip the SEC document boilerplate at the top
        cleaned_lower = cleaned.lower()

        # Check for pre-announcement language patterns
        matched_patterns = [p for p in PREANNOUNCE_PATTERNS if p in cleaned_lower]

        # First ~80 words — usually the most signal-dense content
        words = cleaned.split()
        snippet = " ".join(words[:80]) if words else ""

        print(f"\n  --- 8-K {fdate} (accession {acc}) ---")
        if matched_patterns:
            print(f"  ⚠️  Pre-announcement language detected: {matched_patterns[:3]}")
            flag("HIGH", ticker,
                 f"8-K {fdate} contains pre-announcement language: {matched_patterns[0]}")
        if snippet:
            print(f"  Snippet: {snippet[:500]}{'...' if len(snippet) > 500 else ''}")


# ---------------------------------------------------------------------------
# Section: sector cohort (peer comparison — distinguishes idiosyncratic
# moves from sector-wide rotation)
# ---------------------------------------------------------------------------

# Define peer cohorts. Each ticker maps to ~3-4 peers in the same business.
# Used to differentiate "is this a stock-specific move or a sector move?"
SECTOR_COHORTS: dict[str, list[str]] = {
    "MSFT": ["GOOG", "AMZN", "META"],
    "PLTR": ["NOW", "SNOW", "DDOG"],
    "GOOG": ["MSFT", "AMZN", "META"],
    # AMZN (Jul 11 2026): XLY is 28.5% AMZN (Tesla 18.3%; the top two are ~47% of
    # the fund), so "AMZN vs XLY" is partly Amazon measured against itself and the
    # excess return is mechanically compressed. Amazon's economics live in cloud +
    # ads + platform scale, not in Home Depot and McDonald's.
    "AMZN": ["MSFT", "GOOG", "META"],
    "NVDA": ["AVGO", "AMD", "MRVL", "TSM"],
    "SMCI": ["DELL", "HPE", "ANET", "AVGO"],
    "DDOG": ["MDB", "SNOW", "NET", "TEAM"],
    "OKLO": ["SMR", "NNE", "LEU", "BWXT"],
    "QUBT": ["IONQ", "QBTS", "RGTI", "ARQQ"],
    "CRWD": ["PANW", "ZS", "S", "FTNT"],
    "SNOW": ["DDOG", "MDB", "NET", "TEAM"],
    "NVMI": ["KLAC", "AMAT", "LRCX", "ONTO"],
}


def section_sector_cohort(ticker: str) -> None:
    print(f"\n=== SECTOR COHORT — {ticker} ===")

    cohort = SECTOR_COHORTS.get(ticker.upper(), [])
    if not cohort:
        print(f"  No cohort defined for {ticker}.")
        return

    cfg = TICKER_CONFIG.get(ticker.upper(), {})
    sector_etf = cfg.get("sector_etf", "")

    def fetch_returns(t: str) -> dict:
        return _daily_returns(t)

    print(f"  {'Ticker':<10} {'1d':>8} {'5d':>8} {'20d':>8}")
    self_ret = fetch_returns(ticker)

    def render(label: str, d: dict) -> None:
        cells = []
        for key in ("ret_1d", "ret_5d", "ret_20d"):
            v = d.get(key)
            cells.append(f"{v*100:+.2f}%" if v is not None else "    -")
        print(f"  {label:<10} {cells[0]:>8} {cells[1]:>8} {cells[2]:>8}")

    render(ticker, self_ret)
    if sector_etf:
        render(f"{sector_etf}", fetch_returns(sector_etf))

    # Cohort
    cohort_rets = []
    for peer in cohort:
        d = fetch_returns(peer)
        if d:
            cohort_rets.append(d)
            render(peer, d)

    # Cohort average vs ticker — if ticker diverges from cohort, that's signal
    if cohort_rets:
        for window in ("ret_5d", "ret_20d"):
            avg = sum(d.get(window, 0) for d in cohort_rets) / len(cohort_rets)
            self_v = self_ret.get(window, 0)
            divergence = self_v - avg
            window_label = "5d" if window == "ret_5d" else "20d"
            if abs(divergence) >= 0.04:  # 4pp divergence
                direction = "OUTPERFORMING" if divergence > 0 else "UNDERPERFORMING"
                print(f"  {window_label} divergence vs cohort avg: "
                      f"{divergence*100:+.2f}pp ({direction})")
                flag("MED", ticker,
                     f"{window_label} cohort divergence: {ticker} {self_v*100:+.1f}% vs "
                     f"{', '.join(cohort)} avg {avg*100:+.1f}% — {direction.lower()}")


# ---------------------------------------------------------------------------
# Section: recent news (UW news endpoint)
# ---------------------------------------------------------------------------

URGENT_NEWS_PATTERNS = [
    "preannounce", "pre-announce", "preliminary results", "guidance",
    "downgrade", "upgrade", "investigation", "lawsuit", "subpoena",
    "ceo step down", "cfo step down", "ceo resigns", "cfo resigns",
    "warns", "raises target", "lowers target", "price target",
    "raises guidance", "lowers guidance", "withdraws guidance",
    "weaker than expected", "stronger than expected",
    "sec investigation", "doj", "fraud", "restatement",
]

# Keywords that indicate macro news likely to move broad markets in the
# near term. Different from ticker-level urgency — these are about the
# overall regime (rates, geopolitics, trade, recession risk).
MACRO_URGENT_PATTERNS = [
    "rate cut", "rate hike", "rate decision", "fomc", "powell speaks",
    "tariff", "export controls", "sanctions", "decouple", "trade deal",
    "escalat", "missile strike", "ceasefire", "invasion", "shutdown",
    "default", "recession", "inflation surge", "cpi above", "cpi below",
    "ppi above", "ppi below", "jobs report beat", "jobs report miss",
    "hawkish", "dovish", "yield curve", "credit crunch",
    "election", "executive order", "geopolitical risk",
]

# Top-level macro queries. Run ONCE per script invocation, shared across
# all tickers. Each tuple is (display_label, RSS query).
MACRO_NEWS_QUERIES: list[tuple[str, str]] = [
    ("FED",      "Federal Reserve interest rate FOMC Powell"),
    ("ECON",     "US CPI inflation jobs report unemployment GDP"),
    ("TRADE",    "China tariffs trade war export controls"),
    ("GEOPOL",   "Ukraine Russia Israel Iran Taiwan conflict"),
    ("MARKET",   "stock market today S&P 500 NASDAQ"),
]

# Ticker-specific macro topics. Macro news is most actionable when it's
# about the thing that moves THIS ticker. NVDA cares about chip export
# controls; NVMI cares about semi-cycle headlines. Keep queries narrow.
TICKER_MACRO_CONTEXT: dict[str, list[str]] = {
    "NVDA": [
        "AI chip China export controls",
        "Taiwan TSMC semiconductor",
    ],
    "SMCI": [
        "AI server demand hyperscaler capex",
    ],
    "DDOG": [
        "cloud spending enterprise software AI budget",
    ],
    "OKLO": [
        "small modular reactor SMR nuclear policy",
        "AI data center power demand",
    ],
    "QUBT": [
        "quantum computing breakthrough government funding",
    ],
    "CRWD": [
        "cybersecurity spending CISO budget",
        "ransomware breach incident",
    ],
    "SNOW": [
        "cloud spending enterprise software AI budget",
    ],
    "NVMI": [
        "semiconductor capex foundry equipment",
        "TSMC Samsung Intel fab investment",
    ],
}

# In-memory cache of RSS pulls within a single run. Avoids refetching the
# same query when multiple tickers share macro queries.
_RSS_CACHE: dict[str, list[dict]] = {}


def _fetch_google_news_rss(query: str, max_items: int = 15) -> list[dict]:
    """Fetch news headlines from Google News RSS. Returns list of dicts
    with title, link, date (ISO), source. No auth, no API key. Cached
    per-run to avoid duplicate fetches.
    """
    cache_key = f"google|{query}|{max_items}"
    if cache_key in _RSS_CACHE:
        return _RSS_CACHE[cache_key]

    import urllib.request

    q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_bytes = resp.read()
    except Exception as e:
        print(f"  [warn] Google News RSS failed: {e}")
        _RSS_CACHE[cache_key] = []
        return []

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"  [warn] Google News RSS parse failed: {e}")
        _RSS_CACHE[cache_key] = []
        return []

    items = []
    for item in root.findall(".//item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        src_elem = item.find("source")
        source = src_elem.text.strip() if src_elem is not None and src_elem.text else ""
        try:
            dt = parsedate_to_datetime(pub) if pub else None
            iso = dt.date().isoformat() if dt else ""
        except (TypeError, ValueError):
            iso = ""
        # Google News titles end with " - SourceName"; strip if no separate source
        if " - " in title and not source:
            t, _, s = title.rpartition(" - ")
            title, source = t.strip(), s.strip()
        items.append({"title": title, "link": link, "date": iso, "source": source})

    # Be polite to Google between RSS pulls
    time.sleep(0.5)
    _RSS_CACHE[cache_key] = items
    return items


def _fetch_yahoo_news_rss(ticker: str, max_items: int = 15) -> list[dict]:
    """Yahoo Finance RSS — backup news source if Google News fails."""
    cache_key = f"yahoo|{ticker}|{max_items}"
    if cache_key in _RSS_CACHE:
        return _RSS_CACHE[cache_key]

    import urllib.request

    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_bytes = resp.read()
    except Exception as e:
        print(f"  [warn] Yahoo Finance RSS failed: {e}")
        _RSS_CACHE[cache_key] = []
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"  [warn] Yahoo Finance RSS parse failed: {e}")
        _RSS_CACHE[cache_key] = []
        return []
    items = []
    for item in root.findall(".//item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        try:
            dt = parsedate_to_datetime(pub) if pub else None
            iso = dt.date().isoformat() if dt else ""
        except (TypeError, ValueError):
            iso = ""
        items.append({"title": title, "link": link, "date": iso, "source": "Yahoo Finance"})
    _RSS_CACHE[cache_key] = items
    return items


def _render_news_items(
    items: list[dict],
    cutoff_iso: str,
    urgent_kw: list[str],
    ticker: Optional[str] = None,
    flag_severity: str = "MED",
    flag_label: str = "News alert",
    max_show: int = 6,
    flag_max: int = 2,
) -> int:
    """Print news items, flag urgent ones, return count shown."""
    shown = 0
    flagged = 0
    for it in items:
        published = (it.get("date") or "")[:10]
        if published and published < cutoff_iso:
            continue
        headline = it.get("title") or "(no headline)"
        source = it.get("source") or ""
        h_lower = headline.lower()
        urgent = any(p in h_lower for p in urgent_kw)
        marker = " ⚠️" if urgent else ""
        print(f"  {published:<12} [{source[:20]:<20}] {headline[:88]}{marker}")
        # Cap flags per call so a single Reuters keyword-rich day doesn't
        # produce 8 MED flags from one query
        if urgent and ticker and flagged < flag_max:
            flag(flag_severity, ticker, f"{flag_label}: {headline[:120]}")
            flagged += 1
        shown += 1
        if shown >= max_show:
            break
    return shown


def section_macro_news() -> None:
    """Global macro/market news context. Runs ONCE per script run, shared
    across all tickers. Surfaces market-moving stories (Fed, trade,
    geopolitical, economic data, market mood) from the last 3 days."""
    print(f"\n{'#' * 78}")
    print(f"#  MACRO / MARKET CONTEXT — global news from the last 3 days")
    print(f"#  (single fetch, applies to all tickers in this run)")
    print(f"{'#' * 78}")

    cutoff_iso = (_today_et() - timedelta(days=3)).isoformat()

    for label, query in MACRO_NEWS_QUERIES:
        items = _fetch_google_news_rss(query, max_items=8)
        print(f"\n  [{label}]  query: {query}")
        if not items:
            print(f"  (no items returned)")
            continue
        shown = _render_news_items(
            items,
            cutoff_iso=cutoff_iso,
            urgent_kw=MACRO_URGENT_PATTERNS,
            ticker=None,  # macro flags handled separately below
            max_show=4,
        )
        if shown == 0:
            print(f"  (no items within last 3 days)")
            continue

        # If multiple urgent macro items appeared in this query, raise a
        # single global flag tagged as MACRO. We use a synthetic ticker
        # "MACRO" so the flag summary clearly separates macro from per-ticker.
        urgent_titles = [it["title"] for it in items
                         if any(p in (it["title"] or "").lower()
                                for p in MACRO_URGENT_PATTERNS)
                         and (it.get("date") or "") >= cutoff_iso]
        if urgent_titles:
            flag("MED", "MACRO",
                 f"[{label}] {urgent_titles[0][:140]}")


def section_news(ticker: str) -> None:
    print(f"\n=== RECENT NEWS — {ticker} ===")

    cfg = TICKER_CONFIG.get(ticker.upper(), {})
    cutoff = (_today_et() - timedelta(days=14)).isoformat()

    # Try UW first (will likely 404 on Basic plan for these endpoints —
    # the section_news function used to give up here. Now we fall through
    # to Google News RSS, which is free and reliable.)
    uw_rows = []
    for path in [
        f"/api/news/{ticker}",
        f"/api/stock/{ticker}/news",
        f"/api/stock/{ticker}/news-headlines",
        f"/api/news/headlines",  # global headlines, may filter by ticker
    ]:
        try:
            data = uw_get(path, params={"ticker": ticker, "limit": 15})
        except Exception:
            data = None
        if data:
            rows = (data or {}).get("data") or (data if isinstance(data, list) else [])
            if rows:
                uw_rows = rows
                break

    items: list[dict] = []
    if uw_rows:
        for r in uw_rows[:15]:
            if not isinstance(r, dict):
                continue
            items.append({
                "title": _f(r, "headline", "title", "summary") or "",
                "link": _f(r, "url", "link") or "",
                "date": (_f(r, "published_at", "publishedAt", "date",
                            "created_at") or "")[:10],
                "source": _f(r, "source", "publisher") or "UW",
            })
        source_label = "UnusualWhales"
    else:
        # Fallback chain: Google News → Yahoo Finance
        query = cfg.get("news_search_term") or f"{ticker} stock"
        items = _fetch_google_news_rss(query, max_items=15)
        source_label = "Google News"
        if not items:
            items = _fetch_yahoo_news_rss(ticker, max_items=15)
            source_label = "Yahoo Finance"

    if not items:
        print(f"  No news available for {ticker}.")
    else:
        print(f"  Source: {source_label}  (showing items within last 14 days)")
        shown = _render_news_items(
            items,
            cutoff_iso=cutoff,
            urgent_kw=URGENT_NEWS_PATTERNS,
            ticker=ticker,
            flag_severity="MED",
            flag_label="News alert",
            max_show=8,
            flag_max=2,
        )
        if shown == 0:
            print(f"  No news within last 14 days.")

    # Ticker-specific macro context. Different topic stream from "ticker
    # stock" — picks up macro stories that affect this name uniquely.
    # E.g. NVDA gets China chip export controls; NVMI gets semi-cycle headlines.
    macro_queries = TICKER_MACRO_CONTEXT.get(ticker.upper(), [])
    macro_cutoff = (_today_et() - timedelta(days=7)).isoformat()
    for query in macro_queries:
        macro_items = _fetch_google_news_rss(query, max_items=6)
        print(f"\n  [Macro context: {query}]")
        if not macro_items:
            print(f"  (no items returned)")
            continue
        shown = _render_news_items(
            macro_items,
            cutoff_iso=macro_cutoff,
            urgent_kw=MACRO_URGENT_PATTERNS + URGENT_NEWS_PATTERNS,
            ticker=ticker,
            flag_severity="MED",
            flag_label=f"Macro alert ({query[:30]})",
            max_show=4,
            flag_max=1,
        )
        if shown == 0:
            print(f"  (no items within last 7 days)")


# ---------------------------------------------------------------------------
# Section: Form 4 transaction details (parses raw XML from EDGAR)
# ---------------------------------------------------------------------------
# UW's per-ticker insider endpoint is unreliable on small-caps. EDGAR has the
# truth — every Form 4 is filed there as structured XML with transaction code,
# shares, and price. This section pulls each Form 4 we've discovered, parses
# it, and persists the transactions so we can compute real buy/sell aggregates.

def section_form4_details(conn: sqlite3.Connection, ticker: str, since: str) -> None:
    print(f"\n=== FORM 4 DETAILS — {ticker} (since {since}) ===")

    cik = get_cik(ticker)
    if cik is None:
        print(f"  No CIK for {ticker}; skipping.")
        return

    cur = conn.cursor()
    # Self-cleaning: any form4_parsed rows with 0 transactions are leftovers
    # from a previous broken run; delete them so we retry.
    cleanup = cur.execute(
        "DELETE FROM form4_parsed "
        "WHERE ticker = ? AND (transaction_count = 0 OR transaction_count IS NULL)",
        (ticker,)
    )
    if cleanup.rowcount:
        print(f"  Cleared {cleanup.rowcount} stale parse record(s) from prior run.")
    conn.commit()

    # Schema upgrade: any filings parsed by the older code path don't have
    # plan_date or role_tier on their transactions. Force a one-time re-parse
    # of those filings so the new fields get populated.
    upgraded = cur.execute(
        """
        DELETE FROM form4_parsed
        WHERE ticker = ? AND accession_number IN (
            SELECT DISTINCT t.accession_number
            FROM form4_transactions t
            WHERE t.ticker = ?
              AND (t.role_tier IS NULL OR t.role_tier = '')
        )
        """,
        (ticker, ticker),
    )
    if upgraded.rowcount:
        # Also delete the corresponding old transactions so we don't dupe
        cur.execute(
            """
            DELETE FROM form4_transactions
            WHERE ticker = ? AND (role_tier IS NULL OR role_tier = '')
            """,
            (ticker,),
        )
        print(f"  Re-parsing {upgraded.rowcount} filing(s) to capture "
              f"plan_date / role_tier (new in this script version).")
    conn.commit()

    # Find all Form 4 accessions for this ticker that we have NOT yet parsed
    # successfully (transaction_count > 0). The JOIN uses the AND clause to
    # ensure we retry filings that were stored with 0 transactions.
    rows = list(cur.execute("""
        SELECT f.accession_number, f.filing_date, f.primary_doc
        FROM edgar_filings f
        LEFT JOIN form4_parsed p ON f.accession_number = p.accession_number
                                AND p.transaction_count > 0
        WHERE f.ticker = ?
          AND f.form_type IN ('4', '4/A')
          AND f.filing_date >= ?
          AND p.accession_number IS NULL
        ORDER BY f.filing_date ASC
    """, (ticker, since)))

    # Cap fetches per run for politeness (each takes ~0.15s + network)
    MAX_FETCHES = 25
    if len(rows) > MAX_FETCHES:
        # Most recent N first — those matter most for the upcoming vote
        rows = rows[-MAX_FETCHES:]

    if rows:
        print(f"  Fetching and parsing {len(rows)} Form 4 filing(s) from EDGAR...")
    # (other-issuer exclusion note printed after the loop, see below)
    new_parsed = 0
    parse_failures = 0
    other_issuer_excluded = 0
    for acc, fdate, primary in rows:
        xml_text = fetch_form4_xml(cik, acc, primary)
        if not xml_text:
            parse_failures += 1
            continue

        parsed = parse_form4_xml(xml_text)
        txs = parsed.get("transactions", [])

        # If parser ran but got nothing, treat as failure too (don't mark
        # the filing as parsed, so we'll retry next run with better logic)
        if not parsed.get("filer_name") and not txs:
            parse_failures += 1
            continue

        # OTHER-ISSUER FILTER (Jul 21 2026). EDGAR lists Form 4s under BOTH
        # issuer and reporting-owner CIKs, so a venture arm's dispositions of
        # PORTFOLIO-COMPANY stock surface in the parent's feed (specimen: GV
        # 2019 GP $1.8M flagged as a GOOG quiet-period insider SELL, T-1 to
        # earnings). Exclude non-matching issuers BEFORE persist/flags. If the
        # XML carried no issuer symbol, keep the row (fail-open, old behavior).
        # CIK match, NOT symbol match: Alphabet files under GOOGL while the
        # monitor tracks GOOG (dual class) -- symbol comparison condemned all
        # 50 legitimate insider filings on 2026-07-21. Issuer CIK is class-
        # invariant: Alphabet=1652044 for both GOOG and GOOGL.
        _isym = parsed.get("issuer_symbol") or ""
        _icik = (parsed.get("issuer_cik") or "").strip().lstrip("0")
        _want = str(cik).lstrip("0")
        if _icik and _want and _icik != _want:
            other_issuer_excluded += 1
            try:
                cur.execute("""
                    INSERT OR REPLACE INTO form4_parsed
                    (accession_number, ticker, parsed_at, filer_name, filer_cik,
                     officer_title, is_director, is_officer, is_ten_percent_owner,
                     transaction_count, aggregate_p_value, aggregate_s_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (acc, ticker, now_iso(),
                      f"[OTHER ISSUER {_isym}] " + (parsed.get("filer_name") or ""),
                      parsed.get("filer_cik"), parsed.get("officer_title"),
                      0, 0, 0, 0, 0.0, 0.0))
            except sqlite3.Error as e:
                print(f"  [warn] form4_parsed insert: {e}")
            continue

        # Aggregate P (open-market buy) and S (open-market sell) values
        agg_p = sum(t["value"] for t in txs if t["code"] == "P")
        agg_s = sum(t["value"] for t in txs if t["code"] == "S")

        # Persist the parsed-filings row
        try:
            cur.execute("""
                INSERT OR REPLACE INTO form4_parsed
                (accession_number, ticker, parsed_at, filer_name, filer_cik,
                 officer_title, is_director, is_officer, is_ten_percent_owner,
                 transaction_count, aggregate_p_value, aggregate_s_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                acc, ticker, now_iso(),
                parsed.get("filer_name"), parsed.get("filer_cik"),
                parsed.get("officer_title"),
                1 if parsed.get("is_director") else 0,
                1 if parsed.get("is_officer") else 0,
                1 if parsed.get("is_ten_percent_owner") else 0,
                len(txs), agg_p, agg_s,
            ))
        except sqlite3.Error as e:
            print(f"  [warn] form4_parsed insert: {e}")

        # Compute role tier once per filing (filer is constant within a filing)
        role_tier = filer_role_tier(parsed)

        # Persist each individual transaction
        for t in txs:
            try:
                cur.execute("""
                    INSERT INTO form4_transactions
                    (accession_number, ticker, transaction_date, code, shares,
                     price, value_usd, acquired_disposed, post_holding,
                     security_title, is_derivative, filer_name, officer_title,
                     plan_date, role_tier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    acc, ticker, t["date"], t["code"], t["shares"],
                    t["price"], t["value"], t["acquired_disposed"],
                    t["post_holding"], t["security_title"],
                    1 if t["is_derivative"] else 0,
                    parsed.get("filer_name"), parsed.get("officer_title"),
                    t.get("plan_date"),  # leave None as SQL NULL (= "no 10b5-1 ref")
                    role_tier,
                ))
            except sqlite3.Error as e:
                print(f"  [warn] form4_tx insert: {e}")

        new_parsed += 1
    conn.commit()

    if rows:
        _skipped = len(rows) - new_parsed - parse_failures
        print(f"  Successfully parsed: {new_parsed}/{len(rows)} "
              f"(failures: {parse_failures}, already parsed/skipped: {_skipped})")

    # Now query the database for everything in window and analyze
    all_txs = list(cur.execute("""
        SELECT t.transaction_date, t.code, t.shares, t.price, t.value_usd,
               t.acquired_disposed, t.is_derivative, t.filer_name,
               t.officer_title, t.accession_number,
               t.plan_date, t.role_tier
        FROM form4_transactions t
        WHERE t.ticker = ? AND t.transaction_date >= ?
        ORDER BY t.transaction_date DESC
    """, (ticker, since)))

    if not all_txs:
        print(f"  No Form 4 transactions parsed in window.")
        return

    # Tuple positions for clarity:
    #   0 date, 1 code, 2 shares, 3 price, 4 value, 5 ad, 6 is_d,
    #   7 who, 8 title, 9 acc, 10 plan_date, 11 role_tier

    # Split into non-derivative vs derivative for cleaner analysis
    non_deriv = [t for t in all_txs if not t[6]]
    deriv = [t for t in all_txs if t[6]]

    # Aggregate by code (non-derivative only — that's where the real signal is)
    by_code: dict[str, dict] = {}
    for tx in non_deriv:
        tx_date, code, shares, price, value, ad, is_d, who, title, acc, plan_date, role_tier = tx
        c = code or "?"
        b = by_code.setdefault(c, {"count": 0, "value": 0.0, "shares": 0.0,
                                   "filers": set()})
        b["count"] += 1
        b["value"] += value or 0
        b["shares"] += shares or 0
        b["filers"].add(who or "?")

    print(f"  Total transactions parsed: {len(all_txs)} "
          f"(non-derivative: {len(non_deriv)}, derivative: {len(deriv)})")
    print(f"  New filings parsed this run: {new_parsed}")
    print()
    print("  Aggregate by transaction code (non-derivative only):")
    print(f"  {'Code':<5} {'Description':<35} {'Filings':>8} {'Filers':>7} "
          f"{'Shares':>14} {'Value $':>16}")
    code_descriptions = {
        "P": "Open-market purchase (HIGH)",
        "S": "Open-market sale",
        "A": "Grant/award (compensation)",
        "M": "Exercise of derivative",
        "F": "Tax withholding",
        "D": "Disposition (varies)",
        "G": "Gift",
        "I": "Discretionary transaction",
        "J": "Other (see filing)",
        "C": "Conversion of derivative",
        "K": "Equity swap",
        "X": "Exercise (oblig)",
    }
    # Sort by absolute value desc
    sorted_codes = sorted(by_code.items(), key=lambda kv: -kv[1]["value"])
    for code, b in sorted_codes:
        desc = code_descriptions.get(code, "(unknown)")
        print(f"  {code:<5} {desc:<35} {b['count']:>8} {len(b['filers']):>7} "
              f"{b['shares']:>14,.0f} {b['value']:>16,.0f}")

    p_value = by_code.get("P", {}).get("value", 0)
    s_value = by_code.get("S", {}).get("value", 0)
    p_count = by_code.get("P", {}).get("count", 0)
    # FIXED-WINDOW TOTALS (Jul 25 2026). The rolling window silently rolled
    # past PLTR's May-22 filings and the printed total fell $154.5M -> $29.0M
    # while selling never stopped -- a denominator artifact reading as a
    # regime change. Fixed 90d/365d anchors don't move between runs.
    try:
        _fw = {}
        for _lbl, _days in (("90d", 90), ("365d", 365)):
            _cut = (_today_et() - timedelta(days=_days)).isoformat()
            _row = cur.execute(
                "SELECT SUM(CASE WHEN code='S' THEN value_usd END), "
                "       SUM(CASE WHEN code='P' THEN value_usd END) "
                "FROM form4_transactions WHERE ticker=? AND is_derivative=0 "
                "AND transaction_date >= ?", (ticker, _cut)).fetchone()
            _fw[_lbl] = (_row[0] or 0.0, _row[1] or 0.0)
        print(f"  Fixed windows (window-roll-proof): "
              f"90d SELL ${_fw['90d'][0]:,.0f} / BUY ${_fw['90d'][1]:,.0f}   "
              f"365d SELL ${_fw['365d'][0]:,.0f} / BUY ${_fw['365d'][1]:,.0f}")
    except Exception as _e:
        print(f"  [warn] fixed-window totals failed: {_e}")
    s_count = by_code.get("S", {}).get("count", 0)
    p_filers = len(by_code.get("P", {}).get("filers", set()))
    s_filers = len(by_code.get("S", {}).get("filers", set()))

    print()
    print(f"  Open-market BUYS  (P): {p_count} filings by {p_filers} filer(s), "
          f"${p_value:,.0f}")
    print(f"  Open-market SELLS (S): {s_count} filings by {s_filers} filer(s), "
          f"${s_value:,.0f}")
    print(f"  Net (P - S):           ${p_value - s_value:>+15,.0f}")

    # Flag meaningful events. Insider open-market BUYING is the high-value
    # signal we care most about. Selling on its own is largely noise for
    # megacap tickers where S>>P is the steady-state — only escalate when
    # it's quiet-period selling or unusually concentrated.
    if p_value >= INSIDER_MIN_USD:
        flag("HIGH", ticker,
             f"Insider open-market BUYING: ${p_value:,.0f} across "
             f"{p_count} filing(s) by {p_filers} insider(s)")
    if p_filers >= 3:
        flag("HIGH", ticker,
             f"CLUSTER BUYING: {p_filers} different insiders bought in window "
             f"(strong bullish signal)")
    # Aggregate-selling flag: only emit if the seller list is concentrated
    # (≤3 filers) and the dollar amount is large. Diffuse selling across
    # many officers is normal 10b5-1 plan activity, not signal.
    # STALENESS DECAY (Jul 11 2026). s_value aggregates the whole lookback window,
    # so this flag kept firing on trades 31+ days old -- long past any information
    # content. An insider sale is a signal for a few weeks, not a quarter.
    # Gate on the MOST RECENT sale in the concentrated set.
    INSIDER_STALE_DAYS = 21
    _s_dates = [str(t[0])[:10] for t in non_deriv
                if str(t[1] or "").upper().startswith("S") and t[0]]
    _s_last = max(_s_dates) if _s_dates else None
    _age = None
    if _s_last:
        try:
            _age = (_today_et() - date.fromisoformat(_s_last)).days
        except Exception:
            _age = None
    if (s_value >= 5_000_000 and s_filers <= 3 and
            s_value > p_value * 4 and
            _age is not None and _age <= INSIDER_STALE_DAYS):
        flag("MED", ticker,
             f"Concentrated insider selling: ${s_value:,.0f} from "
             f"only {s_filers} filer(s), most recent {_s_last} ({_age}d ago) "
             f"(potential signal vs 10b5-1 noise)")
    elif s_value >= 5_000_000 and s_filers <= 3 and _age is not None:
        print(f"  [note] concentrated selling present but STALE "
              f"(most recent {_s_last}, {_age}d ago > {INSIDER_STALE_DAYS}d) — flag suppressed")

    # Top 15 most material transactions (non-derivative, sorted by value desc)
    print()
    print("  Top non-derivative transactions by value:")
    print(f"  {'Date':<12} {'Code':<5} {'Filer':<28} {'Title/Role':<22} "
          f"{'Shares':>10} {'Price $':>8} {'Value $':>12} {'10b5-1 plan':<14}")
    sorted_tx = sorted(non_deriv, key=lambda t: -float(t[4] or 0))[:15]
    for tx in sorted_tx:
        tx_date, code, shares, price, value, ad, is_d, who, title, acc, plan_date, role_tier = tx
        if (value or 0) < 1000:
            continue  # skip noise
        tag = ""
        if code == "P" and (value or 0) >= 50_000:
            tag = "  ★BUY"
        elif code == "S" and (value or 0) >= 100_000:
            tag = "  ↓SELL"
        # Show role tier in brackets when filer isn't a current officer
        title_display = (title or "")[:22]
        if not title_display and role_tier == "DIRECTOR":
            title_display = "[Director only]"
        elif not title_display and role_tier == "10PCT":
            title_display = "[10%+ owner]"
        plan_display = plan_date[:10] if plan_date else "-"
        print(f"  {(tx_date or '?')[:10]:<12} {code or '?':<5} "
              f"{(who or '?')[:28]:<28} {title_display:<22} "
              f"{shares or 0:>10,.0f} {price or 0:>8,.2f} "
              f"{value or 0:>12,.0f}  {plan_display:<14}{tag}")

    # Recent material transactions (last 14 days, regardless of size)
    today = _today_et().isoformat()
    fourteen_days_ago = (_today_et() - timedelta(days=14)).isoformat()
    recent_material = [t for t in non_deriv
                       if t[0] and t[0] >= fourteen_days_ago
                       and t[1] in ("P", "S")
                       and (t[4] or 0) > 1000]
    in_quiet = is_in_quiet_period(ticker)
    days_to_e = days_until_earnings(ticker)

    if recent_material:
        print()
        if in_quiet:
            print(f"  Recent open-market transactions (last 14 days) "
                  f"— IN QUIET PERIOD ({days_to_e}d until earnings):")
        else:
            print(f"  Recent open-market transactions (last 14 days):")
        # Display column headers including 10b5-1 status
        for tx in recent_material:
            date_, code, shares, price, value, ad, is_d, who, title, acc, plan_date, role_tier = tx
            tag = "★BUY" if code == "P" else "↓SELL"
            safety = classify_10b5_1_safety(plan_date, date_ or "")
            # Build a status marker
            if safety == "CLEAN":
                status = f"[10b5-1 {plan_date}]"
            elif safety == "TIGHT":
                status = f"[10b5-1 {plan_date} ⚠️ TIGHT]"
            elif safety == "UNKNOWN":
                status = "[10b5-1 ref but date unparsed]"
            else:
                status = "[discretionary]"
            role_marker = ""
            if role_tier == "DIRECTOR":
                role_marker = " (board only)"
            elif role_tier == "10PCT":
                role_marker = " (10%+ owner)"
            print(f"    {date_:<12} {tag:<6} {(who or '?')[:24]:<24}{role_marker:<14} "
                  f"{shares or 0:>10,.0f} @ ${price or 0:>7.2f} "
                  f"= ${value or 0:>11,.0f}  {status}")

        # Aggregate flags by (filer, code) so we don't spam the summary with
        # many individual rows when an insider does a chunked sell on one day.
        agg: dict[tuple[str, str], dict] = {}
        for tx in recent_material:
            date_, code, shares, price, value, ad, is_d, who, title, acc, plan_date, role_tier = tx
            key = (who or "unknown", code)
            a = agg.setdefault(key, {"count": 0, "value": 0.0,
                                      "shares": 0.0, "title": title,
                                      "role_tier": role_tier or "OTHER",
                                      "safeties": []})
            a["count"] += 1
            a["value"] += value or 0
            a["shares"] += shares or 0
            a["safeties"].append(classify_10b5_1_safety(plan_date, date_ or ""))

        # Emit flags with role-tier and 10b5-1 safety taken into account.
        for (who, code), a in agg.items():
            tx_word = "BUY" if code == "P" else "SELL"
            role_tier = a["role_tier"]
            # Aggregate safety: if all transactions are CLEAN 10b5-1, it's
            # confirmed routine. If any are TIGHT, that's a red flag. If any
            # are NONE during quiet period, that's discretionary in quiet.
            safeties = set(a["safeties"])
            all_clean = safeties == {"CLEAN"}
            any_tight = "TIGHT" in safeties
            any_discretionary = "NONE" in safeties

            # Role-tier qualifier for messages
            role_qual = ""
            if role_tier == "DIRECTOR":
                role_qual = " (board director)"
            elif role_tier == "10PCT":
                role_qual = " (10%+ owner)"

            # ---------- BUYS ----------
            if code == "P":
                if in_quiet and role_tier == "OFFICER":
                    flag("HIGH", ticker,
                         f"QUIET-PERIOD insider BUY by current officer: {who} "
                         f"purchased ${a['value']:,.0f} across {a['count']} trade(s) "
                         f"({days_to_e}d before earnings)")
                elif in_quiet:
                    flag("MED", ticker,
                         f"Quiet-period insider BUY: {who}{role_qual} purchased "
                         f"${a['value']:,.0f} ({days_to_e}d before earnings)")
                elif a["value"] >= INSIDER_MIN_USD and role_tier == "OFFICER":
                    flag("HIGH", ticker,
                         f"Recent insider BUY by officer: {who} purchased ${a['value']:,.0f}")
                elif a["value"] >= INSIDER_MIN_USD:
                    flag("MED", ticker,
                         f"Recent insider BUY: {who}{role_qual} ${a['value']:,.0f}")
                continue

            # ---------- SELLS ----------
            # Tighter cooling-off than 90 days = potential 10b5-1 abuse.
            # This is the most regulator-relevant signal we can produce.
            if any_tight:
                flag("HIGH", ticker,
                     f"⚠️  TIGHT 10b5-1 sale by {who}{role_qual}: plan adopted "
                     f"<90 days before sale (${a['value']:,.0f} sold across "
                     f"{a['count']} trade(s)) — REGULATORY RED FLAG")
                continue

            # Quiet-period selling without a 10b5-1 reference = discretionary
            # selling in a window when the filer isn't normally allowed to.
            # That's a stronger signal than scheduled selling.
            if in_quiet and any_discretionary and a["value"] >= 250_000:
                if role_tier == "OFFICER":
                    flag("HIGH", ticker,
                         f"Discretionary insider SELL in quiet period: {who} sold "
                         f"${a['value']:,.0f} with NO 10b5-1 plan reference "
                         f"({days_to_e}d before earnings)")
                else:
                    flag("MED", ticker,
                         f"Discretionary SELL in quiet period: {who}{role_qual} "
                         f"sold ${a['value']:,.0f}")
                continue

            # Confirmed clean 10b5-1 sales — these are LOW signal. Only emit
            # if the dollar amount is very large AND filer is a current officer.
            if all_clean and role_tier == "OFFICER" and a["value"] >= 5_000_000:
                flag("INFO", ticker,
                     f"Routine 10b5-1 SELL by officer {who}: ${a['value']:,.0f} "
                     f"(plan adopted >=90d before sale — confirmed clean)")
            elif all_clean:
                # Suppress entirely — confirmed routine, low signal
                pass
            elif in_quiet and a["value"] >= 1_000_000:
                # Quiet-period sale, plan reference exists but unverifiable
                # — soft flag
                flag("MED", ticker,
                     f"Quiet-period SELL: {who}{role_qual} sold ${a['value']:,.0f} "
                     f"({a['count']} trade(s), 10b5-1 status unclear)")
            elif a["value"] >= INSIDER_MIN_USD * 4:
                flag("MED", ticker,
                     f"Insider SELL: {who}{role_qual} sold ${a['value']:,.0f} "
                     f"across {a['count']} trade(s)")


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


def _daily_returns(t: str) -> dict:
    """1d/5d/20d from CANONICAL daily closes (prices.db adj_close), never UW
    /ohlc/1d. That endpoint returns THREE rows per session (pr/po/r, order
    varying by ticker) and ignores limit -> 1d was regular-close vs post-close
    (sign-flipping), closes[-21] was 7 sessions. Verified Jul 21 probe.
    Massive daily aggs fallback for tickers absent from daily_prices."""
    closes: list = []
    _bar_date = None
    try:
        import sqlite3 as _sq
        _con = _sq.connect(f"file:{_REPO_ROOT / 'prices.db'}?mode=ro", uri=True)
        _rows = _con.execute(
            "SELECT date, adj_close FROM daily_prices WHERE ticker=? "
            "AND adj_close IS NOT NULL ORDER BY date DESC LIMIT 22",
            (t.upper(),)).fetchall()
        _con.close()
        closes = [float(r[1]) for r in _rows][::-1]
        _bar_date = str(_rows[0][0])[:10] if _rows else None
    except Exception as _e:
        print(f"  [warn] daily_prices read failed for {t}: {_e}")
    # STALENESS FALL-THROUGH (Jul 28 2026). Fallback used to fire only on
    # len<21, never on stale bars -- a pull before Pipeline A ingested the
    # latest session served 2-session-stale spot/peer/cohort while the header
    # was current (NVDA: 206.84/196.51/194.25, three prices in one run,
    # on a -4.99% day).
    _exp = None
    try:
        import sys as _sy, os as _os
        _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _R not in _sy.path:
            _sy.path.insert(0, _R)
        from features import massive_client as _mc2
        _e2 = _mc2._last_completed_session()
        _exp = _e2.strftime("%Y-%m-%d") if hasattr(_e2, "strftime") else str(_e2)[:10]
    except Exception:
        _exp = None
    _stale = bool(_exp and _bar_date and _bar_date < _exp)
    if len(closes) < 21 or _stale:
        _end = _today_et().isoformat()
        _start = (_today_et() - timedelta(days=45)).isoformat()
        _data = massive_get(f"/v2/aggs/ticker/{t}/range/1/day/{_start}/{_end}",
                            params={"adjusted": "true"})
        _res = (_data or {}).get("results") or []
        if len(_res) > len(closes) or _stale:
            _new = [float(r.get("c") or 0) for r in _res if r.get("c")]
            if _new:
                closes = _new
                _bar_date = _exp or _bar_date
    if _exp and _bar_date and _bar_date < _exp:
        print(f"  [warn] {t}: daily bar {_bar_date} is STALE vs last completed "
              f"session {_exp} -- returns/spot computed on old data")
    out = _ohlc_returns([{"close": c} for c in closes])
    out["bar_date"] = _bar_date
    return out


def section_peer_relative(ticker: str) -> None:
    """Compares ticker returns to its sector ETF + SPY. Pre-earnings drift is
    a documented anomaly: stocks tend to drift in the direction of the upcoming
    surprise in the 5-10 days before the print, so 5d relative strength
    matters more than absolute price action."""
    print(f"\n=== PEER-RELATIVE PRICE — {ticker} ===")

    cfg = TICKER_CONFIG.get(ticker.upper(), {})
    sector_etf = cfg.get("sector_etf", "SPY")  # default SPY. The old IWM
    # fallback benchmarked megacaps against small-caps (10 of 12 tickers).

    def fetch_returns(t: str) -> dict:
        return _daily_returns(t)

    t = fetch_returns(ticker)
    sec = fetch_returns(sector_etf)
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
    row(f"{sector_etf} (sector)", sec)
    row("SPY", spy)
    print("  " + "-" * 42)

    rel_sec = {k: t.get(k, 0) - sec.get(k, 0) for k in ("ret_1d", "ret_5d", "ret_20d")
               if t.get(k) is not None and sec.get(k) is not None}
    rel_spy = {k: t.get(k, 0) - spy.get(k, 0) for k in ("ret_1d", "ret_5d", "ret_20d")
               if t.get(k) is not None and spy.get(k) is not None}
    row(f"vs {sector_etf}", rel_sec)
    row("vs SPY", rel_spy)

    # Pre-earnings drift: 5d relative strength vs SECTOR is the cleanest signal
    # because it filters out broad-market beta and idiosyncratic sector moves.
    days_to_e = days_until_earnings(ticker)
    in_drift_window = days_to_e is not None and 0 <= days_to_e <= 10

    if rel_sec.get("ret_5d", 0) >= 0.05:
        sev = "HIGH" if in_drift_window else "MED"
        msg = (f"5-day relative strength vs {sector_etf}: "
               f"+{rel_sec['ret_5d']*100:.1f}% — pre-earnings drift / accumulation pattern")
        if in_drift_window:
            msg += f" ({days_to_e}d before earnings)"
        flag(sev, ticker, msg)
    if rel_sec.get("ret_5d", 0) <= -0.05:
        sev = "HIGH" if in_drift_window else "MED"
        msg = (f"5-day relative WEAKNESS vs {sector_etf}: "
               f"{rel_sec['ret_5d']*100:.1f}% — distribution or thesis decay")
        if in_drift_window:
            msg += f" ({days_to_e}d before earnings)"
        flag(sev, ticker, msg)
    if rel_sec.get("ret_1d", 0) >= 0.03:
        flag("INFO", ticker,
             f"Today's relative strength vs {sector_etf}: "
             f"+{rel_sec['ret_1d']*100:.1f}%")
    if rel_sec.get("ret_1d", 0) <= -0.03:
        flag("INFO", ticker,
             f"Today's relative weakness vs {sector_etf}: "
             f"{rel_sec['ret_1d']*100:.1f}%")

# ---------------------------------------------------------------------------
# Section: price + volume context
# ---------------------------------------------------------------------------

def section_price_volume(ticker: str, since: str) -> None:
    print(f"\n=== PRICE / VOLUME CONTEXT — {ticker} (since {since}) ===")

    # Live intraday quote (if market is open) — gives the most current view
    realtime = uw_get(f"/api/stock/{ticker}/stock-state") or uw_get(f"/api/stock/{ticker}/realtime")
    if realtime:
        rt = realtime.get("data") if isinstance(realtime, dict) else realtime
        if isinstance(rt, list) and rt:
            rt = rt[0]
        if isinstance(rt, dict):
            try:
                live_price = float(_f(rt, "last", "price", "market_price",
                                      "close", "last_price") or 0)
                prev_close = float(_f(rt, "previous_close", "prev_close",
                                      "prevClose") or 0)
                live_vol = _f(rt, "volume", "today_volume")
            except (TypeError, ValueError):
                live_price = prev_close = 0
                live_vol = None
            if live_price:
                pct = ((live_price - prev_close) / prev_close * 100) if prev_close else 0
                vol_str = f"   vol={float(live_vol):,.0f}" if live_vol else ""
                import os as _os, sys as _sys
                _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                if _R not in _sys.path:
                    _sys.path.insert(0, _R)
                from utils.market_calendar import is_market_open as _rth_open
                _lbl = "Live (intraday)" if _rth_open() else "Last quote (market CLOSED)"
                print(f"  {_lbl}: ${live_price:.2f}   "
                      f"({pct:+.2f}% from prev close ${prev_close:.2f}){vol_str}")

    # VOLUME-DENOMINATOR FIX (Jul 11 2026)
    # WAS: uw_get("/api/stock/{ticker}/ohlc/1d"). That UW feed carries duplicate /
    # partial bars (MSFT showed 2026-06-26 TWICE: 186.2M and 115.3M), which
    # understated the 20d average (13.5M vs a true 47.0M) and INVERTED every
    # volume-ratio flag -- it reported "1.83x" on a day that was really 0.52x,
    # so the "volume >= 2x/3x" alerts fired on QUIET days.
    # squeeze_scan, reading massive_client in the SAME process, reported 0.51,
    # and FINRA's avg_daily_vol (36.5M) agreed. The monitor was the wrong one.
    # NOW: reads massive_client -> prices.db raw_bars, the same clean source
    # builder.py uses. End is clamped to the last completed session upstream.
    rows = []
    try:
        import sys as _sys, os as _os
        _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _ROOT not in _sys.path:
            _sys.path.insert(0, _ROOT)
        from features import massive_client as _mc
        _end = _mc._last_completed_session()
        _end_s = _end.strftime("%Y-%m-%d") if hasattr(_end, "strftime") else str(_end)[:10]
        _df = _mc.download(ticker, start=since, end=_end_s)
        if _df is not None and not _df.empty:
            for _ix, _r in _df.iterrows():
                rows.append({
                    "date":   _ix.strftime("%Y-%m-%d") if hasattr(_ix, "strftime") else str(_ix)[:10],
                    "open":   float(_r.get("Open")   or 0),
                    "high":   float(_r.get("High")   or 0),
                    "low":    float(_r.get("Low")    or 0),
                    "close":  float(_r.get("Close")  or 0),
                    "volume": float(_r.get("Volume") or 0),
                })
    except Exception as _e:
        print(f"  [warn] massive volume fetch failed ({_e}); falling back to UW OHLC")
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
        # MECHANICAL-DAY EXCLUSION (Jul 11 2026).
        # The plain 20d mean includes index-reconstitution days. MSFT's Jun 26
        # (Russell recon) printed 181.0M against a ~40.5M baseline and dragged the
        # denominator to 47.5M, understating true rvol by ~17% (0.52x vs 0.61x).
        # Mechanical flow is not a volume regime -- strip it from the BASELINE.
        # Rule: drop any day above 3x the window median (a recon/rebalance print,
        # not a normal heavy day). Generalises across market caps; no dollar floor.
        _w = vols[-21:-1]
        _med = statistics.median(_w) if _w else 0.0
        _clean = [v for v in _w if _med <= 0 or v <= 3.0 * _med]
        _n_excl = len(_w) - len(_clean)
        avg20 = (sum(_clean) / len(_clean)) if _clean else 0.0
        latest_vol = vols[-1]
        ratio = latest_vol / avg20 if avg20 else 0
        print(f"  Latest close: ${closes[-1]:.2f}   "
              f"Latest volume: {latest_vol:,.0f}   "
              f"20d avg vol: {avg20:,.0f}"
              + (f" ({_n_excl} mechanical day(s) excluded)" if _n_excl else "") + "   "
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
    data = massive_get(f"/v2/aggs/ticker/{ticker}/range/1/day/{since}/{_today_et().isoformat()}",
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

# ---------------------------------------------------------------------------
# Section: SQUEEZE FUEL — cost-to-borrow + short interest setup readout.
#
# HONEST SCOPE (Rule 1):
#   This identifies squeeze *candidates* (loaded ingredients), NOT timing and
#   NOT a buy signal. Your own validated SI brick found high short interest ->
#   LOWER forward returns on average (per-date IC -0.054, NW-t -4.46). Crowded
#   shorts underperform on average; the squeeze is the rare tail. So a HIGH fuel
#   reading means "this CAN squeeze if a catalyst hits", never "this will go up".
#
#   Borrow fee (fee_rate) + shares-available come from UW /api/shorts/{t}/data,
#   daily/intraday — fresher than the bi-monthly FINRA short-interest cycle.
# ---------------------------------------------------------------------------
def get_borrow_fee(ticker: str) -> Optional[dict]:
    """Latest borrow fee + shares-available from UW, with a 5-snapshot-back
    delta for a freshness read. Returns None if unavailable (e.g. not on plan).
    fee_rate arrives as a STRING and is cast to float here."""
    data = uw_get(f"/api/shorts/{ticker}/data")
    rows = (data or {}).get("data") or []
    if not rows:
        return None
    latest = rows[0]
    try:
        fee = float(latest.get("fee_rate") or 0)
    except (TypeError, ValueError):
        fee = 0.0
    avail = latest.get("short_shares_available")
    avail = int(avail) if avail is not None else None
    ts = latest.get("timestamp")
    # Delta vs a snapshot ~5 back (intraday depth varies; best-effort freshness).
    fee_prev = None
    if len(rows) > 5:
        try:
            fee_prev = float(rows[5].get("fee_rate") or 0)
        except (TypeError, ValueError):
            fee_prev = None
    return {"fee": fee, "avail": avail, "ts": ts, "fee_prev": fee_prev}


def get_squeeze_data(ticker: str, older_than: Optional[str] = None) -> dict:
    """Single source of truth for squeeze inputs. Merges the two UW endpoints:
      - /api/shorts/{t}/data          -> LIVE fee + shares-available (intraday)
      - /api/shorts/{t}/interest-float/v2 -> settlement SI% + float + DTC + date

    Returns ONE merged dict. section_squeeze just displays this; the model
    backfill (Phase 0) will call this with older_than= for point-in-time rows.

    Field facts (verified live 2026-07): fee_rate + si_float + days_to_cover are
    STRINGS; si_float is a FRACTION (0.2855 == 28.55%); date key is market_date.

    NOTE: older_than is plumbed through for the future backfill but its support
    on these endpoints is UNVERIFIED for interest-float/v2 -- do not rely on the
    historical path until it is tested (Rule 1). The live path (older_than=None)
    is what monitor uses today and is confirmed working.
    """
    # ---- live borrow half (reuse the existing fetcher) ----
    bf = get_borrow_fee(ticker) if older_than is None else None
    fee = bf["fee"] if bf else None
    avail = bf["avail"] if bf else None
    fee_prev = bf.get("fee_prev") if bf else None

    # ---- settlement SI half ----
    si_pct = dtc = mkt_date = total_float = None
    params = {"older_than": older_than} if older_than else None
    sidata = uw_get(f"/api/shorts/{ticker}/interest-float/v2", params=params)
    srows = (sidata or {}).get("data") or []
    if srows:
        # market_date is now a recognized get_date key; sort newest-first.
        srt = sorted(srows, key=lambda x: (get_date(x) or ""), reverse=True)
        r0 = srt[0]
        si_pct = float(r0.get("si_float") or r0.get("short_percent_of_float") or
                       r0.get("pct_of_float") or r0.get("percent_of_float") or 0) or None
        dtc = float(r0.get("days_to_cover") or r0.get("dtc") or 0) or None
        mkt_date = get_date(r0)
        try:
            total_float = float(r0.get("total_float") or 0) or None
        except (TypeError, ValueError):
            total_float = None
        # If the live /data call was skipped (backfill mode), fall back to the
        # settlement fee/avail from this same row so the dict is still complete.
        if fee is None:
            try:
                fee = float(r0.get("fee_rate") or 0) or None
            except (TypeError, ValueError):
                fee = None
        if avail is None:
            av = r0.get("short_shares_available")
            avail = int(av) if av is not None else None

    if dtc is None and older_than is None:
        import os as _os
        _si_db = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "short_interest.db")
        if _os.path.exists(_si_db):
            try:
                _con = sqlite3.connect(_si_db)
                _row = _con.execute(
                    "SELECT settlement_date, days_to_cover FROM short_interest "
                    "WHERE ticker=? ORDER BY settlement_date DESC LIMIT 1",
                    (ticker,)).fetchone()
                _con.close()
                if _row:
                    mkt_date = mkt_date or _row[0]
                    dtc = float(_row[1]) if _row[1] else None
            except sqlite3.Error:
                pass
    return {"fee": fee, "avail": avail, "fee_prev": fee_prev,
            "si_pct": si_pct, "dtc": dtc, "date": mkt_date, "float": total_float}




def assess_squeeze(fee: Optional[float], si_pct: Optional[float],
                   dtc: Optional[float], avail: Optional[int]) -> dict:
    """Pure, tunable, no network. Grades each ingredient and returns an overall
    fuel level. Thresholds are HEURISTIC (not validated like the SI brick).
    si_pct is a FRACTION (0.38 == 38%), matching how monitor stores it."""
    ingredients = []
    loaded = 0

    # Borrow fee (baseline easy-to-borrow ~0.3%)
    if fee is None:
        fee_tag = "n/a"
    elif fee >= 20:
        fee_tag = "EXTREME"; loaded += 2
    elif fee >= 5:
        fee_tag = "HIGH"; loaded += 1
    elif fee >= 1:
        fee_tag = "MODERATE"
    else:
        fee_tag = "easy"
    ingredients.append(("Borrow fee", f"{fee:.2f}%" if fee is not None else "n/a", fee_tag))

    # Short interest (% float, FRACTION)
    if si_pct is None:
        si_tag = "n/a"
    elif si_pct >= 0.30:
        si_tag = "HIGH"; loaded += 1
    elif si_pct >= 0.20:
        si_tag = "ELEVATED"
    else:
        si_tag = "—"
    ingredients.append(("Short interest", f"{si_pct:.1%} float" if si_pct is not None else "n/a", si_tag))

    # Days to cover (clip FINRA OTC junk > 50)
    dtc_disp = dtc
    if dtc is not None and dtc > 50:
        dtc_disp = None
    if dtc_disp is None:
        dtc_tag = "n/a"
    elif dtc_disp >= 10:
        dtc_tag = "HIGH"; loaded += 1
    elif dtc_disp >= 5:
        dtc_tag = "ELEVATED"
    else:
        dtc_tag = "—"
    ingredients.append(("Days to cover", f"{dtc_disp:.1f}" if dtc_disp is not None else "n/a", dtc_tag))

    # Shares available (context; weak alone without float normalization)
    ingredients.append(("Shares available", f"{avail:,}" if avail is not None else "n/a", "context"))

    if loaded >= 4:
        level = "HIGH"
    elif loaded >= 2:
        level = "MODERATE"
    elif loaded >= 1:
        level = "LOW"
    else:
        level = "NONE"
    return {"level": level, "loaded": loaded, "ingredients": ingredients}


def section_squeeze(conn: sqlite3.Connection, ticker: str) -> dict:
    """Prints the squeeze-fuel block and returns a summary dict for the
    cross-ticker roundup. Re-fetches SI from the same endpoint the SI section
    uses, so it is self-contained even under `--skip shorts`."""
    print(f"\n=== SQUEEZE FUEL — {ticker} ===")
    sq = get_squeeze_data(ticker)
    fee = sq["fee"]
    avail = sq["avail"]
    si_pct = sq["si_pct"]
    dtc = sq["dtc"]

    a = assess_squeeze(fee, si_pct, dtc, avail)
    for label, val, tag in a["ingredients"]:
        base = "  (baseline ~0.3%)" if label == "Borrow fee" else ""
        print(f"  {label:<17} {val:>14}   [{tag}]{base}")

    # Freshness: intraday fee drift the bi-monthly FINRA cycle can't see.
    if sq.get("fee_prev") is not None and fee is not None:
        d = fee - sq["fee_prev"]
        if abs(d) >= 0.01:
            print(f"  Fee drift (intraday):  {d:+.2f}pp  (fresher than bi-monthly SI)")

    print(f"  {'-'*40}")
    if a["level"] == "NONE":
        print(f"  FUEL: NONE — easy to borrow, not a squeeze candidate")
    else:
        print(f"  FUEL: {a['level']} — squeeze *candidate* ({a['loaded']} ingredients loaded)")
        print(f"  \u26a0 Fuel != buy. High SI -> underperformance on avg (your SI brick).")
        print(f"    Squeeze is the tail; price RAMP/IGNITION is the trigger to watch.")
        if a["level"] == "HIGH":
            flag("MED", ticker, f"Squeeze fuel HIGH (fee {fee:.1f}% / SI {si_pct:.0%} / DTC {dtc:.1f})"
                 if (fee is not None and si_pct is not None and dtc is not None)
                 else "Squeeze fuel HIGH (partial data)")

    return {"ticker": ticker, "level": a["level"], "fee": fee,
            "si_pct": si_pct, "dtc": dtc, "avail": avail}


def print_squeeze_summary(results: list) -> None:
    """Cross-ticker roundup printed once after the per-ticker loop."""
    rows = [r for r in results if r]
    if not rows:
        return
    order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "NONE": 3}
    rows.sort(key=lambda r: order.get(r["level"], 9))
    print("\n" + "=" * 78)
    print("  SQUEEZE SUMMARY (fuel = candidate strength, NOT a buy signal)")
    print("=" * 78)
    for r in rows:
        fee = f"fee {r['fee']:.1f}%" if r["fee"] is not None else "fee n/a"
        si = f"SI {r['si_pct']:.0%}" if r["si_pct"] is not None else "SI n/a"
        dtc = f"DTC {r['dtc']:.1f}" if r["dtc"] is not None else "DTC n/a"
        print(f"  {r['ticker']:<6} FUEL: {r['level']:<9} {fee:<11} {si:<8} {dtc}")
    print("=" * 78)


def print_header(args) -> None:
    print("=" * 78)
    print(f"  Pre-earnings signal monitor")
    print(f"  Run UTC: {now_iso()}")
    print(f"  Window:  since {args.since}    Tickers: {', '.join(args.tickers)}")
    print(f"  DB:      {DB_PATH}")
    print("=" * 78)
    print("  Upcoming earnings dates (from config):")
    for t in args.tickers:
        cfg = TICKER_CONFIG.get(t.upper(), {})
        ed = cfg.get("earnings_date", "?")
        et = cfg.get("earnings_time", "?")
        fq = cfg.get("fiscal_q", "?")
        d_until = days_until_earnings(t)
        d_str = f"{d_until}d" if d_until is not None else "?"
        print(f"    {t:<6}  {ed} {et:<5} ({fq:<10})  in {d_str}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Default lookbacks are computed relative to today
    default_since = (_today_et() - timedelta(days=60)).isoformat()
    default_dp_since = (_today_et() - timedelta(days=30)).isoformat()
    p.add_argument("--since", default=default_since,
                   help=f"ISO date; lookback start for time-windowed sections "
                        f"(default: 60 days back = {default_since})")
    p.add_argument("--darkpool-since", default=default_dp_since,
                   help=f"Lookback for dark pool — looping per-day fetches make "
                        f"longer windows expensive (default: 30 days back = {default_dp_since})")
    p.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS,
                   help=f"Tickers to scan (default: {' '.join(DEFAULT_TICKERS)})")
    p.add_argument("--csv", action="store_true",
                   help="Also dump today's signals to CSV alongside the DB")
    p.add_argument("--skip", nargs="+", default=[],
                   choices=["insiders", "institutional", "darkpool", "options",
                            "shorts", "ohlc", "massive", "edgar", "peer", "form4",
                            "calendar", "implied", "eightk", "cohort", "news",
                            "macro", "squeeze"],
                   help="Skip one or more sections")
    args = p.parse_args()

    try:
        _uw_headers()  # fail fast if no key
    except APIError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    print_header(args)
    conn = get_db()

    # Global macro news runs ONCE, before the per-ticker loop. The macro
    # picture (Fed, trade, geopolitics) shifts the regime for everything
    # downstream, so it makes sense to lead with it.
    if "macro" not in args.skip:
        section_macro_news()

    squeeze_results = []
    for ticker in args.tickers:
        print("\n" + "#" * 78)
        print(f"#  {ticker}")
        print("#" * 78)
        # LIVE-BOOK LINE (owed since the report-audit batch; every report's
        # 16.2 was blocked on it). Reads si_live_ledger.csv; states membership
        # so no report ever has to say "not verifiable from the report layer".
        try:
            _led = (_REPO_ROOT / "si_live_ledger.csv").read_text().strip().splitlines()
            _gen = _led[1].split(",")[0] if len(_led) > 1 else "?"
            _mine = [l.split(",") for l in _led[1:]
                     if len(l.split(",")) > 6 and l.split(",")[2].upper() == ticker.upper()]
            if _mine:
                _usd = sum(float(r[5]) for r in _mine)
                _sh = sum(float(r[6]) for r in _mine)
                print(f"  LIVE BOOK: HELD ${_usd:,.0f} / {_sh:,.0f} sh "
                      f"({_mine[0][3].strip()}) -- ledger generated {_gen}")
            else:
                print(f"  LIVE BOOK: not in si_live_ledger (generated {_gen})")
        except Exception as _e:
            print(f"  LIVE BOOK: ledger unreadable ({_e})")

        # Earnings-specific sections come first — calendar context frames
        # everything that follows.
        if "calendar" not in args.skip:
            section_earnings_calendar(ticker)
        if "implied" not in args.skip:
            section_implied_move(ticker)
        # Then the standard signal stack
        if "edgar" not in args.skip:
            section_edgar(conn, ticker, args.since)
        if "eightk" not in args.skip:
            section_eightk_content(conn, ticker)
        if "form4" not in args.skip:
            section_form4_details(conn, ticker, args.since)
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
        if "squeeze" not in args.skip:
            squeeze_results.append(section_squeeze(conn, ticker))
        if "ohlc" not in args.skip:
            section_price_volume(ticker, args.since)
        if "peer" not in args.skip:
            section_peer_relative(ticker)
        if "cohort" not in args.skip:
            section_sector_cohort(ticker)
        if "news" not in args.skip:
            section_news(ticker)
        if "massive" not in args.skip:
            section_massive_blocks(ticker, args.since)

    conn.close()

    # ----- Squeeze fuel roundup (candidate strength, NOT a buy signal) -----
    if "squeeze" not in args.skip:
        print_squeeze_summary(squeeze_results)

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
    print("  Done. Re-run daily during US market hours, especially within 7 days")
    print("  of any ticker's earnings date.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
