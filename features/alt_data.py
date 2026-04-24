# features/alt_data.py
# ─────────────────────────────────────────────────────────────────────────────
# Alternative data sources — FREE, no API keys required
#   - Wikipedia pageviews: retail attention signal
#   - SEC 8-K filings: material events (M&A, bankruptcy, executive changes)
#
# Both cached in DB to minimize requests.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import sqlite3
import requests
from datetime import date, datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "accuracy.db"

# Ticker → Wikipedia article name mapping for major tickers
WIKI_MAPPING = {
    "AAPL": "Apple_Inc.",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet_Inc.",
    "GOOG":  "Alphabet_Inc.",
    "AMZN":  "Amazon_(company)",
    "META":  "Meta_Platforms",
    "NVDA":  "Nvidia",
    "TSLA":  "Tesla,_Inc.",
    "NFLX":  "Netflix",
    "AMD":   "Advanced_Micro_Devices",
    "INTC":  "Intel",
    "ORCL":  "Oracle_Corporation",
    "CRM":   "Salesforce",
    "ADBE":  "Adobe_Inc.",
    "PYPL":  "PayPal",
    "DIS":   "The_Walt_Disney_Company",
    "BA":    "Boeing",
    "JPM":   "JPMorgan_Chase",
    "V":     "Visa_Inc.",
    "MA":    "Mastercard",
    "WMT":   "Walmart",
    "TGT":   "Target_Corporation",
    "HD":    "The_Home_Depot",
    "COST":  "Costco",
    "PFE":   "Pfizer",
    "JNJ":   "Johnson_%26_Johnson",
    "MRNA":  "Moderna",
    "ABNB":  "Airbnb",
    "UBER":  "Uber",
    "LYFT":  "Lyft",
    "SHOP":  "Shopify",
    "SPOT":  "Spotify",
    "ROKU":  "Roku,_Inc.",
    "SQ":    "Block,_Inc.",
    "PLTR":  "Palantir_Technologies",
    "COIN":  "Coinbase",
    "RIVN":  "Rivian",
    "LCID":  "Lucid_Motors",
    "NIO":   "Nio_Inc.",
    "BYND":  "Beyond_Meat",
    "GME":   "GameStop",
    "AMC":   "AMC_Theatres",
}


# ══════════════════════════════════════════════════════════════════════════════
# WIKIPEDIA PAGEVIEWS
# ══════════════════════════════════════════════════════════════════════════════

def init_wiki_table():
    """Create wiki_pageviews_cache table."""
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wiki_pageviews_cache (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                date        TEXT NOT NULL,
                views       INTEGER,
                updated_at  TEXT NOT NULL,
                UNIQUE(ticker, date)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_wiki_ticker ON wiki_pageviews_cache(ticker)")
        conn.commit()


def fetch_wiki_pageviews(ticker: str, days_back: int = 30) -> list:
    """
    Fetch daily Wikipedia pageviews for a ticker.
    Returns list of {date, views} dicts.
    """
    article = WIKI_MAPPING.get(ticker.upper())
    if not article:
        return []  # No mapping — skip unmapped tickers

    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=days_back)
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/all-agents/{article}/daily/"
        f"{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
    )
    try:
        r = requests.get(url, headers={"User-Agent": "ML-Quant-Fund/1.0"}, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json().get("items", [])
        return [
            {"date": d["timestamp"][:8], "views": d["views"]}
            for d in data
        ]
    except Exception:
        return []


def save_wiki_to_db(ticker: str, pageviews: list):
    """Save fetched pageviews to DB."""
    if not pageviews:
        return
    now = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        for pv in pageviews:
            dt = f"{pv['date'][:4]}-{pv['date'][4:6]}-{pv['date'][6:8]}"
            conn.execute("""
                INSERT OR REPLACE INTO wiki_pageviews_cache
                    (ticker, date, views, updated_at)
                VALUES (?, ?, ?, ?)
            """, (ticker, dt, pv["views"], now))
        conn.commit()


def get_wiki_attention_score(ticker: str) -> dict:
    """
    Get Wikipedia attention signal — z-score of recent pageviews.
    High score = unusual attention spike = potential retail move.
    Reads from DB only.
    """
    result = {
        "ticker":         ticker,
        "attention_zscore": 0.0,
        "attention_signal": "NORMAL",
        "recent_views":    0,
        "avg_views":       0,
        "error":           None,
    }

    if ticker.upper() not in WIKI_MAPPING:
        result["error"] = "No Wikipedia mapping"
        return result

    try:
        cutoff = str(date.today() - timedelta(days=30))
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            rows = conn.execute("""
                SELECT views FROM wiki_pageviews_cache
                WHERE ticker=? AND date>=?
                ORDER BY date DESC
            """, (ticker, cutoff)).fetchall()

        if len(rows) < 10:
            return result

        views = [r[0] for r in rows if r[0] is not None]
        if len(views) < 10:
            return result

        recent_avg = sum(views[:3])  / 3  # last 3 days
        long_avg   = sum(views[3:]) / max(len(views[3:]), 1)
        std        = (sum((v - long_avg) ** 2 for v in views[3:]) / max(len(views[3:]), 1)) ** 0.5

        if std > 0:
            zscore = (recent_avg - long_avg) / std
        else:
            zscore = 0.0

        result["attention_zscore"] = round(zscore, 2)
        result["recent_views"]     = int(recent_avg)
        result["avg_views"]        = int(long_avg)

        if zscore > 2:
            result["attention_signal"] = "SPIKE_HIGH"
        elif zscore > 1:
            result["attention_signal"] = "ELEVATED"
        elif zscore < -1:
            result["attention_signal"] = "BELOW_AVG"

    except Exception as e:
        result["error"] = str(e)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SEC 8-K FILINGS
# ══════════════════════════════════════════════════════════════════════════════

def init_sec_table():
    """Create sec_8k_cache table."""
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sec_8k_cache (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker       TEXT NOT NULL,
                filing_date  TEXT NOT NULL,
                form         TEXT,
                items        TEXT,
                url          TEXT,
                updated_at   TEXT NOT NULL,
                UNIQUE(ticker, filing_date, form)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sec_ticker ON sec_8k_cache(ticker)")
        conn.commit()


def get_cik_for_ticker(ticker: str) -> str:
    """Lookup SEC CIK number for a ticker. Cached locally."""
    # SEC company tickers file is free
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "ML-Quant-Fund research@example.com"},
            timeout=10,
        )
        if r.status_code != 200:
            return ""
        data = r.json()
        for _, info in data.items():
            if info.get("ticker", "").upper() == ticker.upper():
                return str(info["cik_str"]).zfill(10)
    except Exception:
        pass
    return ""


def fetch_recent_8k(ticker: str, days_back: int = 30) -> list:
    """
    Fetch recent 8-K filings for a ticker from SEC EDGAR.
    Returns list of {filing_date, form, items, url}.
    """
    cik = get_cik_for_ticker(ticker)
    if not cik:
        return []

    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers={"User-Agent": "ML-Quant-Fund research@example.com"},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        filings = data.get("filings", {}).get("recent", {})

        cutoff = (date.today() - timedelta(days=days_back)).isoformat()
        results = []
        for i, form in enumerate(filings.get("form", [])):
            if form != "8-K":
                continue
            fd = filings.get("filingDate", [""])[i]
            if fd < cutoff:
                continue
            acc = filings.get("accessionNumber", [""])[i].replace("-", "")
            primary_doc = filings.get("primaryDocument", [""])[i]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{primary_doc}"
            items = filings.get("items", [""])[i]
            results.append({
                "filing_date": fd,
                "form":        form,
                "items":       items,
                "url":         url,
            })
        return results
    except Exception:
        return []


def save_8k_to_db(ticker: str, filings: list):
    """Save 8-K filings to DB."""
    if not filings:
        return
    now = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        for f in filings:
            conn.execute("""
                INSERT OR REPLACE INTO sec_8k_cache
                    (ticker, filing_date, form, items, url, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ticker, f["filing_date"], f["form"], f["items"], f["url"], now))
        conn.commit()


def get_8k_event_score(ticker: str) -> dict:
    """
    Check if ticker has recent 8-K filings and classify severity.
    Reads from DB.
    """
    result = {
        "ticker":        ticker,
        "has_recent_8k": False,
        "days_since_8k": 999,
        "event_type":    "NONE",
        "event_signal":  "NEUTRAL",
        "error":         None,
    }

    try:
        cutoff = str(date.today() - timedelta(days=14))
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            row = conn.execute("""
                SELECT filing_date, items FROM sec_8k_cache
                WHERE ticker=? AND filing_date>=?
                ORDER BY filing_date DESC LIMIT 1
            """, (ticker, cutoff)).fetchone()
        if row:
            filing_date, items = row
            result["has_recent_8k"] = True
            result["days_since_8k"] = (date.today() - date.fromisoformat(filing_date)).days
            items_str = str(items).lower()

            # Classify by item codes
            if "1.01" in items_str or "2.01" in items_str:
                result["event_type"]   = "M&A_ACQUISITION"
                result["event_signal"] = "BULLISH"
            elif "2.04" in items_str or "3.01" in items_str:
                result["event_type"]   = "DELISTING_WARNING"
                result["event_signal"] = "BEARISH"
            elif "4.01" in items_str or "4.02" in items_str:
                result["event_type"]   = "AUDITOR_CHANGE"
                result["event_signal"] = "BEARISH"
            elif "5.02" in items_str:
                result["event_type"]   = "EXECUTIVE_CHANGE"
                result["event_signal"] = "NEUTRAL"
            elif "1.03" in items_str:
                result["event_type"]   = "BANKRUPTCY"
                result["event_signal"] = "BEARISH"
            elif "2.02" in items_str:
                result["event_type"]   = "EARNINGS"
                result["event_signal"] = "NEUTRAL"
            else:
                result["event_type"]   = "OTHER"
                result["event_signal"] = "NEUTRAL"

    except Exception as e:
        result["error"] = str(e)

    return result
