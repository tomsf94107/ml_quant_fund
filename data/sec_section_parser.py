"""
data/sec_section_parser.py
─────────────────────────────────────────────────────────────────────────────
SEC filing section-aware text extraction.

Session A (May 21 2026): 8-K + NT-10Q/NT-10K only.
Future sessions extend with 10-Q, 10-K, S-*, DEF 14A.

Design:
  - Fetch filing from EDGAR (throttled)
  - Parse HTML with BeautifulSoup
  - Strip tables (financial noise)
  - Extract section text by item-anchor regex
  - Return dict of section_name → cleaned_text
"""

from __future__ import annotations
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup


SEC_BASE = "https://www.sec.gov"
SEC_HEADERS = {
    "User-Agent": "ML Quant Fund research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}
EDGAR_HEADERS = {
    "User-Agent": "ML Quant Fund research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# SEC rate limit: ~10 req/sec
_LAST_REQ_TS = 0.0
_REQ_MIN_INTERVAL = 0.12


def _throttled_get(url: str, headers: dict = SEC_HEADERS, timeout: int = 15) -> Optional[requests.Response]:
    """SEC-rate-limited GET. Returns None on error."""
    global _LAST_REQ_TS
    elapsed = time.time() - _LAST_REQ_TS
    if elapsed < _REQ_MIN_INTERVAL:
        time.sleep(_REQ_MIN_INTERVAL - elapsed)
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        _LAST_REQ_TS = time.time()
        if r.status_code == 200:
            return r
        return None
    except Exception:
        return None


def get_cik(ticker: str) -> Optional[str]:
    """Look up CIK from ticker via EDGAR."""
    url = (f"{SEC_BASE}/cgi-bin/browse-edgar?company=&CIK={ticker}"
           f"&type=&dateb=&owner=include&count=1&action=getcompany&output=atom")
    r = _throttled_get(url)
    if r is None:
        return None
    m = re.search(r'CIK=(\d+)', r.text)
    if not m:
        return None
    return m.group(1).zfill(10)


def list_filings(cik: str, days_back: int = 365,
                 form_types: Optional[list] = None) -> list:
    """
    Return list of filings within the last N days for the given CIK.
    Each: {accession, filing_date, form, primary_document}
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = _throttled_get(url, headers=EDGAR_HEADERS)
    if r is None:
        return []
    try:
        data = r.json()
    except Exception:
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accns = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()

    results = []
    for form, d, accn, doc in zip(forms, dates, accns, docs):
        if d < cutoff:
            continue
        if form_types and form not in form_types:
            continue
        results.append({
            "accession": accn,
            "filing_date": d,
            "form": form,
            "primary_document": doc,
        })
    return results


def fetch_filing_html(cik: str, accession: str, primary_document: str) -> Optional[str]:
    """Fetch primary document of a filing as HTML text."""
    accn_clean = accession.replace("-", "")
    url = f"{SEC_BASE}/Archives/edgar/data/{int(cik)}/{accn_clean}/{primary_document}"
    r = _throttled_get(url)
    return r.text if r else None


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\x00", "", text)
    return text.strip()


def _strip_tables(soup: BeautifulSoup) -> BeautifulSoup:
    for t in soup.find_all("table"):
        t.decompose()
    return soup


# ─── 8-K extraction ──────────────────────────────────────────────────────────

ITEM_8K_PATTERN = re.compile(
    r"Item\s+(\d+\.\d+)\.?\s*[—–\-]?\s*([A-Z][^\n.]{5,150})",
    re.IGNORECASE,
)


def _strip_xbrl_header(text: str) -> str:
    """
    Strip XBRL machine-readable header data that appears before the actual
    filing prose. The real 8-K text starts at "UNITED STATES" or "FORM 8-K".
    """
    markers = [
        "UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION",
        "SECURITIES AND EXCHANGE COMMISSION",
        "CURRENT REPORT",
        "FORM 8-K",
        "FORM\xa08-K",
    ]
    for m in markers:
        idx = text.find(m)
        if idx > 0:
            return text[idx:]
    return text


def extract_8k_items(html: str) -> dict:
    """
    Extract 8-K items as {item_number: text}.
    Returns e.g. {"2.02": "Results of Operations...", "7.01": "Reg FD...", ...}
    Filings with no parseable items return {"_whole": cleaned_text}.
    """
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    soup = _strip_tables(soup)
    raw_text = soup.get_text(separator="\n")
    # Strip XBRL machine data preamble
    raw_text = _strip_xbrl_header(raw_text)

    items = {}
    matches = list(ITEM_8K_PATTERN.finditer(raw_text))
    if not matches:
        cleaned = _clean_text(raw_text)
        if len(cleaned) > 100:
            items["_whole"] = cleaned[:50000]
        return items

    for i, match in enumerate(matches):
        item_no = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        section_text = _clean_text(raw_text[start:end])
        if len(section_text) > 100:
            items[item_no] = section_text[:20000]
    return items


def find_8k_press_release_exhibit(cik: str, accession: str) -> Optional[str]:
    """Look at filing index for EX-99.1 (press release exhibit)."""
    accn_clean = accession.replace("-", "")
    detail_url = f"{SEC_BASE}/Archives/edgar/data/{int(cik)}/{accn_clean}/index.json"
    r = _throttled_get(detail_url)
    if r is None:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    for item in data.get("directory", {}).get("item", []):
        name = item.get("name", "").lower()
        if "ex-99" in name or "ex99" in name or "exhibit99" in name:
            return f"{SEC_BASE}/Archives/edgar/data/{int(cik)}/{accn_clean}/{item['name']}"
    return None


def extract_8k_press_release(cik: str, accession: str) -> Optional[str]:
    """Fetch and extract press release exhibit text."""
    url = find_8k_press_release_exhibit(cik, accession)
    if not url:
        return None
    r = _throttled_get(url)
    if r is None:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    soup = _strip_tables(soup)
    text = soup.get_text(separator="\n")
    cleaned = _clean_text(text)
    return cleaned[:50000] if len(cleaned) > 100 else None


def is_earnings_8k(items: dict) -> bool:
    """Item 2.02 = Results of Operations (earnings)."""
    return "2.02" in items


# ─── NT-10-Q / NT-10-K extraction ────────────────────────────────────────────

def extract_nt_notice(html: str) -> Optional[str]:
    """Extract reason text from late-filing notice."""
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    soup = _strip_tables(soup)
    text = soup.get_text(separator="\n")
    cleaned = _clean_text(text)
    return cleaned[:20000] if len(cleaned) >= 100 else None


# ─── Future sessions (stubs) ─────────────────────────────────────────────────

def extract_10q_sections(html: str) -> dict:
    """Session B: MD&A + Risk Factors + Quantitative Risk."""
    raise NotImplementedError("Session B")


def extract_10k_sections(html: str) -> dict:
    """Session C: Business + Risk Factors + MD&A + Legal."""
    raise NotImplementedError("Session C")


def extract_s_filing_sections(html: str) -> dict:
    """Session D: S-1/S-3 — Risk Factors + Use of Proceeds + Plan of Distribution."""
    raise NotImplementedError("Session D")


def extract_def14a_sections(html: str) -> dict:
    """Session D: DEF 14A — Comp Discussion + Risk Oversight."""
    raise NotImplementedError("Session D")
