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


# Lazy-loaded SEC ticker -> CIK map (downloaded once per session from
# https://www.sec.gov/files/company_tickers.json — authoritative source).
# This replaces the unreliable cgi-bin atom-feed regex approach which silently
# failed for ~20% of tickers (foreign filers, IPOs, sector-specific names).
_TICKER_CIK_MAP: Optional[dict] = None


def _load_ticker_cik_map() -> dict:
    """Fetch and cache the SEC ticker -> CIK JSON map."""
    global _TICKER_CIK_MAP
    if _TICKER_CIK_MAP is not None:
        return _TICKER_CIK_MAP
    url = "https://www.sec.gov/files/company_tickers.json"
    r = _throttled_get(url, headers=EDGAR_HEADERS, timeout=30)
    if r is None:
        _TICKER_CIK_MAP = {}
        return _TICKER_CIK_MAP
    try:
        data = r.json()
        _TICKER_CIK_MAP = {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in data.values()
            if "ticker" in v and "cik_str" in v
        }
    except Exception:
        _TICKER_CIK_MAP = {}
    return _TICKER_CIK_MAP


def get_cik(ticker: str) -> Optional[str]:
    """Look up CIK from ticker via SEC's authoritative ticker JSON."""
    tmap = _load_ticker_cik_map()
    return tmap.get(ticker.upper())


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
    """
    Look at filing index for press release exhibit. Returns largest matching
    candidate.

    Matching strategies (in order):
      1. Standard EX-99.x naming (ex-99.htm, ex99.htm, exhibit99.htm, etc)
      2. Press-release pattern in filename (pr.htm, press, presrel, etc)
      3. Largest .htm file other than the primary document (heuristic
         fallback — earnings exhibits are usually the biggest file)

    NVDA uses "q1fy27pr.htm" — strategy 2 catches that.
    AAPL uses "a8-kex991q2202603282026.htm" — strategy 1 catches it.
    """
    accn_clean = accession.replace("-", "")
    detail_url = f"{SEC_BASE}/Archives/edgar/data/{int(cik)}/{accn_clean}/index.json"
    r = _throttled_get(detail_url)
    if r is None:
        return None
    try:
        data = r.json()
    except Exception:
        return None

    items = data.get("directory", {}).get("item", [])

    # Strategy 1: explicit EX-99 naming
    for item in items:
        name = item.get("name", "").lower()
        if "ex-99" in name or "ex99" in name or "exhibit99" in name:
            return f"{SEC_BASE}/Archives/edgar/data/{int(cik)}/{accn_clean}/{item['name']}"

    # Strategy 2: press-release in filename
    # Common patterns: q1fy27pr.htm, q3pressrelease.htm, earnings-release.htm
    for item in items:
        name = item.get("name", "").lower()
        if not name.endswith((".htm", ".html")):
            continue
        # Match: pr, press, presrel, earnings-release, earningsrelease
        if (("pr" in name and ("q" in name or "fy" in name or "earnings" in name))
            or "press" in name
            or "presrel" in name
            or "earnings-release" in name
            or "earningsrelease" in name):
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
    """
    Extract 10-K sections for the Lazy Prices signal: business (Item 1),
    risk_factors (Item 1A), mda (Item 7).

    Headers sit on their own line after get_text(separator="\\n"), so anchor with
    re.MULTILINE ^\\s*Item — this excludes the inline prose mentions ("see Item 8")
    that match bare "Item N" everywhere. Sections via start/end anchor PAIRS, choose
    the LONGEST valid span (real body beats TOC line). Per Loughran-McDonald-style
    replication: drop sections below a min length (corrupt/partial parse) rather than
    feed garbage to the similarity calc. ~30% of filings may still fail to parse
    cleanly (every filing structure differs) — those return {} and are excluded.
    """
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    soup = _strip_tables(soup)
    raw = _strip_xbrl_header(soup.get_text(separator="\n"))

    def _starts(pat):
        return [m.start() for m in re.finditer(pat, raw, re.IGNORECASE | re.MULTILINE)]

    def _longest_span(start_pat, end_pats, min_len):
        s_pos = _starts(start_pat)
        e_pos = sorted(sum([_starts(ep) for ep in end_pats], []))
        best = ""
        for sp in s_pos:
            ea = [e for e in e_pos if e > sp + 50]
            if not ea:
                continue
            seg = _clean_text(raw[sp:min(ea)])
            if len(seg) > len(best):
                best = seg
        return best if len(best) >= min_len else None

    out = {}
    biz = _longest_span(r"^\s*Item\s+1\.?\s", [r"^\s*Item\s+1A\.?\s", r"^\s*Item\s+2\.?\s"], 1500)
    risk = _longest_span(r"^\s*Item\s+1A\.?\s", [r"^\s*Item\s+1B\.?\s", r"^\s*Item\s+2\.?\s"], 1500)
    mda = _longest_span(r"^\s*Item\s+7\.?\s", [r"^\s*Item\s+7A\.?\s", r"^\s*Item\s+8\.?\s"], 2500)
    if biz:
        out["business"] = biz[:100000]
    if risk:
        out["risk_factors"] = risk[:100000]
    if mda:
        out["mda"] = mda[:100000]
    return out


def extract_s_filing_sections(html: str) -> dict:
    """Session D: S-1/S-3 — Risk Factors + Use of Proceeds + Plan of Distribution."""
    raise NotImplementedError("Session D")


def extract_def14a_sections(html: str) -> dict:
    """Session D: DEF 14A — Comp Discussion + Risk Oversight."""
    raise NotImplementedError("Session D")
