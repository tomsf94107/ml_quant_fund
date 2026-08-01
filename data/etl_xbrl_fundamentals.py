#!/usr/bin/env python3
"""
data/etl_xbrl_fundamentals.py — EDGAR companyfacts -> fundamentals.db (PIT-honest).

Track C: unlocks Quality (Novy-Marx GP = (Revenues - COGS) / Assets) + Value (B/M, E/P).

Source: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json  (free, no key)
PIT DISCIPLINE: every fact stored with filed_date; downstream features MUST only use
facts where filed_date < as_of_date. Period dates are NOT availability dates.

Usage:
  python -m data.etl_xbrl_fundamentals                      # full universe from tickers.txt
  python -m data.etl_xbrl_fundamentals --tickers AAPL MSFT  # subset
"""
import argparse, json, sqlite3, sys, time, urllib.request
from pathlib import Path

SEC_UA = "ML_Quant_Fund research atomnguyen@example.com"  # SEC fair-use requires contact UA
CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
DB_PATH = "fundamentals.db"
THROTTLE_S = 0.15  # ~6 req/s, under SEC's 10/s

# tag -> concept; aliases map to the same concept, feature layer picks best-populated
TAGS = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    "CostOfRevenue": "cogs",
    "CostOfGoodsAndServicesSold": "cogs",
    "CostOfGoodsSold": "cogs",
    "GrossProfit": "gross_profit",
    "Assets": "total_assets",
    "StockholdersEquity": "equity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "equity",
    "NetIncomeLoss": "net_income",
    "OperatingIncomeLoss": "operating_income",
    "EarningsPerShareDiluted": "eps_diluted",
    "CommonStockSharesOutstanding": "shares_out",
    "EntityCommonStockSharesOutstanding": "shares_out",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS xbrl_facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    cik          INTEGER NOT NULL,
    tag          TEXT NOT NULL,
    concept      TEXT NOT NULL,          -- normalized: revenue/cogs/total_assets/...
    unit         TEXT NOT NULL,
    fy           INTEGER,                -- fiscal year of the fact
    fp           TEXT,                   -- FY/Q1/Q2/Q3/Q4
    period_start TEXT,
    period_end   TEXT NOT NULL,
    value        REAL NOT NULL,
    form         TEXT,                   -- 10-K/10-Q/8-K...
    accession    TEXT,
    filed_date   TEXT NOT NULL,          -- PIT availability date. THE date that matters.
    fetched_at   TEXT NOT NULL,
    UNIQUE(cik, tag, unit, period_end, filed_date, form, value)
);
CREATE INDEX IF NOT EXISTS idx_xbrl_tkr_concept ON xbrl_facts(ticker, concept, filed_date);
CREATE INDEX IF NOT EXISTS idx_xbrl_filed ON xbrl_facts(filed_date);
CREATE TABLE IF NOT EXISTS xbrl_cursor (
    ticker TEXT PRIMARY KEY, cik INTEGER, rows_total INTEGER, status TEXT, updated_at TEXT
);
"""


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def load_cik_map():
    data = _get(CIK_MAP_URL)
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}


def extract_rows(ticker, cik, facts_json, now):
    """Pull only TAGS-listed facts; one row per (fact instance)."""
    rows = []
    gaap = facts_json.get("facts", {}).get("us-gaap", {})
    dei = facts_json.get("facts", {}).get("dei", {})
    for tag, concept in TAGS.items():
        node = gaap.get(tag) or dei.get(tag)
        if not node:
            continue
        for unit, items in node.get("units", {}).items():
            for it in items:
                if "val" not in it or "end" not in it or "filed" not in it:
                    continue
                rows.append((
                    ticker, cik, tag, concept, unit,
                    it.get("fy"), it.get("fp"),
                    it.get("start"), it["end"], float(it["val"]),
                    it.get("form"), it.get("accn"), it["filed"], now,
                ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", help="subset; default reads tickers.txt")
    ap.add_argument("--tickers-file", default="tickers.txt")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = [l.strip().upper() for l in open(args.tickers_file)
                   if l.strip() and not l.startswith("#")]

    conn = sqlite3.connect(args.db, timeout=30)
    conn.executescript(SCHEMA)
    print(f"XBRL companyfacts ingest: {len(tickers)} tickers -> {args.db}")

    cik_map = load_cik_map()
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    ok = miss = 0
    for i, t in enumerate(tickers, 1):
        cik = cik_map.get(t)
        if cik is None:
            conn.execute("INSERT OR REPLACE INTO xbrl_cursor VALUES (?,?,?,?,?)",
                         (t, None, 0, "NO_CIK", now))
            miss += 1
            print(f"[{i}/{len(tickers)}] {t:6s} NO CIK (foreign/delisted?)")
            continue
        try:
            fj = _get(FACTS_URL.format(cik=cik))
            rows = extract_rows(t, cik, fj, now)
            conn.executemany(
                "INSERT OR IGNORE INTO xbrl_facts (ticker,cik,tag,concept,unit,fy,fp,"
                "period_start,period_end,value,form,accession,filed_date,fetched_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            conn.execute("INSERT OR REPLACE INTO xbrl_cursor VALUES (?,?,?,?,?)",
                         (t, cik, len(rows), "OK", now))
            conn.commit()
            ok += 1
            if i % 10 == 0 or i == len(tickers):
                n = conn.execute("SELECT COUNT(*) FROM xbrl_facts").fetchone()[0]
                print(f"[{i}/{len(tickers)}] {t:6s} +{len(rows):5d} rows (db total {n})")
        except Exception as e:
            conn.execute("INSERT OR REPLACE INTO xbrl_cursor VALUES (?,?,?,?,?)",
                         (t, cik, 0, f"ERR:{type(e).__name__}", now))
            conn.commit()
            print(f"[{i}/{len(tickers)}] {t:6s} ERROR {e}")
        time.sleep(THROTTLE_S)

    n = conn.execute("SELECT COUNT(*) FROM xbrl_facts").fetchone()[0]
    rng = conn.execute("SELECT MIN(filed_date), MAX(filed_date) FROM xbrl_facts").fetchone()
    print(f"\nDONE: {ok} ok, {miss} no-CIK. {n} facts, filed {rng[0]} -> {rng[1]}")
    print("PIT REMINDER: features must filter filed_date < as_of_date. Never period_end.")


if __name__ == "__main__":
    main()
