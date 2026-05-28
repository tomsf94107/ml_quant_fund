"""
scripts/backfill_earnings_uw_new_tickers.py
────────────────────────────────────────────
Backfill earnings (EPS actual/estimate/surprise) for NEWLY ADDED tickers
using UW /api/stock/{ticker}/earnings.

UW provides deeper, more reliable earnings than yfinance (30yr history,
pre-computed surprise, street consensus estimates).

SCOPE: only the 30 new tickers, from 2019-01-01 onward.
Writes to earnings.db.earnings_surprises (EPS columns only — revenue is
owned by Polygon via etl_polygon_revenue.py).

Rule #1 compliant:
  (b) fails loud on UW errors — no silent 0-fill
  (h) chain: UW API → earnings_surprises → builder → model

Usage:
    python scripts/backfill_earnings_uw_new_tickers.py [--dry-run]
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The 30 newly added tickers (25 universe + 5 watchlist)
NEW_TICKERS = [
    # Universe (25)
    "KLAC","TXN","MCHP","NXPI","DELL","GLW","FLEX","JBL","GFS","NEE","D","CHTR","BIO",
    "HOOD","RDDT","PINS","MDB","CYBR","CRWV","NBIS","RKLB","STLA","CRCL","FIG","AMC",
    # Watchlist (5)
    "ALT","SANA","SENS","VXRT","RC",
]

CUTOFF_DATE = "2019-01-01"  # 1yr lookback for 2020 training YoY calcs
EARNINGS_DB = ROOT / "earnings.db"


def _load_uw_key() -> str:
    key = os.environ.get("UW_API_KEY")
    if not key and (ROOT / ".env").exists():
        for line in open(ROOT / ".env"):
            if line.startswith("UW_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("UW_API_KEY not found in env or .env")
    return key


def fetch_uw_earnings(ticker: str, key: str) -> list[dict]:
    """Fetch earnings history from UW. Returns list of normalized rows.
    Fails loud on HTTP errors per Rule #1(b)."""
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/earnings"
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"UW earnings {ticker}: HTTP {r.status_code}: {r.text[:120]}")
    rows = r.json().get("data", [])
    return rows


def normalize_row(ticker: str, row: dict) -> dict | None:
    """Convert UW earnings row to earnings_surprises schema. None if unusable."""
    report_date = row.get("report_date")
    if not report_date or report_date < CUTOFF_DATE:
        return None
    # Only rows with actual EPS (past earnings) — future estimates skipped here
    reported = row.get("reported_eps")
    if reported is None:
        return None
    
    def _f(v):
        try:
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None
    
    eps_actual = _f(reported)
    eps_estimate = _f(row.get("estimated_eps"))
    eps_surprise = _f(row.get("surprise"))
    eps_surprise_pct = _f(row.get("surprise_percentage"))
    # Compute surprise if UW didn't provide but has both actual+est
    if eps_surprise is None and eps_actual is not None and eps_estimate is not None:
        eps_surprise = eps_actual - eps_estimate
    if eps_surprise_pct is None and eps_surprise is not None and eps_estimate not in (None, 0):
        eps_surprise_pct = (eps_surprise / abs(eps_estimate)) * 100
    
    return {
        "ticker": ticker,
        "report_date": report_date,
        "eps_actual": eps_actual,
        "eps_estimate": eps_estimate,
        "eps_surprise": eps_surprise,
        "eps_surprise_pct": eps_surprise_pct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Fetch + report, don't write")
    args = ap.parse_args()

    key = _load_uw_key()
    conn = sqlite3.connect(EARNINGS_DB)
    cur = conn.cursor()

    total_written = 0
    print(f"Backfilling earnings for {len(NEW_TICKERS)} new tickers (>= {CUTOFF_DATE})")
    print(f"{'ticker':8s} {'fetched':>8s} {'usable':>7s} {'written':>8s}  range")
    print("-" * 60)

    for ticker in NEW_TICKERS:
        try:
            rows = fetch_uw_earnings(ticker, key)
        except Exception as e:
            print(f"{ticker:8s}  ERROR: {e}")
            continue

        normalized = [normalize_row(ticker, r) for r in rows]
        normalized = [n for n in normalized if n is not None]

        if not normalized:
            print(f"{ticker:8s} {len(rows):>8d} {0:>7d} {0:>8d}  (no usable rows)")
            continue

        dates = sorted(n["report_date"] for n in normalized)
        written = 0
        if not args.dry_run:
            now = datetime.now().isoformat()
            for n in normalized:
                cur.execute("""
                    INSERT INTO earnings_surprises
                      (ticker, report_date, eps_actual, eps_estimate,
                       eps_surprise, eps_surprise_pct, created_at)
                    VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(ticker, report_date) DO UPDATE SET
                       eps_actual=excluded.eps_actual,
                       eps_estimate=excluded.eps_estimate,
                       eps_surprise=excluded.eps_surprise,
                       eps_surprise_pct=excluded.eps_surprise_pct
                """, (n["ticker"], n["report_date"], n["eps_actual"],
                      n["eps_estimate"], n["eps_surprise"], n["eps_surprise_pct"], now))
                written += 1
            conn.commit()
        total_written += written
        print(f"{ticker:8s} {len(rows):>8d} {len(normalized):>7d} {written:>8d}  {dates[0]} → {dates[-1]}")

    conn.close()
    print("-" * 60)
    print(f"{'DRY RUN — nothing written' if args.dry_run else f'Total written: {total_written}'}")


if __name__ == "__main__":
    main()
