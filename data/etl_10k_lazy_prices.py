"""
data/etl_10k_lazy_prices.py — Hunt #7 ingestion + similarity (Jun 1 2026).
Step 2: fetch each ticker's multi-year 10-Ks, extract sections, cache to SQLite.
Step 3: YoY cosine + Jaccard similarity per ticker per filing-year.
Lazy Prices (CMN 2020): LOW YoY similarity (big change) = SELL signal.
    python -m data.etl_10k_lazy_prices --fetch
    python -m data.etl_10k_lazy_prices --similarity
"""
import argparse, sqlite3, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.sec_section_parser import (get_cik, list_filings, fetch_filing_html,
                                     extract_10k_sections)

DB = ROOT / "sec_filings.db"
SECTIONS = ["business", "risk_factors", "mda"]

DDL = """
CREATE TABLE IF NOT EXISTS sec_10k_sections (
    ticker TEXT NOT NULL, filing_date TEXT NOT NULL, accession TEXT,
    section TEXT NOT NULL, text TEXT NOT NULL, n_chars INTEGER,
    UNIQUE(ticker, filing_date, section));
CREATE INDEX IF NOT EXISTS idx_10k_tk ON sec_10k_sections(ticker);
CREATE TABLE IF NOT EXISTS sec_10k_similarity (
    ticker TEXT NOT NULL, filing_date TEXT NOT NULL, prev_filing_date TEXT NOT NULL,
    section TEXT NOT NULL, cosine REAL, jaccard REAL,
    UNIQUE(ticker, filing_date, section));
CREATE INDEX IF NOT EXISTS idx_sim_tk ON sec_10k_similarity(ticker);
"""


def fetch_all(tickers, days_back=2600):
    con = sqlite3.connect(str(DB), timeout=30); con.executescript(DDL)
    for i, tk in enumerate(tickers, 1):
        cik = get_cik(tk)
        if not cik:
            print(f"  [{i}/{len(tickers)}] {tk}: no CIK"); continue
        try:
            fs = list_filings(cik, days_back=days_back, form_types=["10-K"])
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {tk}: list failed {e}"); continue
        kept = 0
        for f in fs:
            have = con.execute("SELECT COUNT(*) FROM sec_10k_sections WHERE ticker=? AND filing_date=?",
                               (tk, f["filing_date"])).fetchone()[0]
            if have:
                kept += 1; continue
            html = fetch_filing_html(cik, f["accession"], f["primary_document"])
            secs = extract_10k_sections(html) if html else {}
            for name, txt in secs.items():
                con.execute("""INSERT OR REPLACE INTO sec_10k_sections
                    (ticker,filing_date,accession,section,text,n_chars) VALUES (?,?,?,?,?,?)""",
                    (tk, f["filing_date"], f["accession"], name, txt, len(txt)))
            if secs:
                kept += 1
        con.commit()
        print(f"  [{i}/{len(tickers)}] {tk}: {kept} filings cached", flush=True)
    con.close()


def _cosine(a, b):
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        v = TfidfVectorizer(stop_words="english", max_features=20000).fit_transform([a, b])
        num = (v[0] @ v[1].T).toarray()[0, 0]
        den = np.sqrt((v[0].multiply(v[0])).sum()) * np.sqrt((v[1].multiply(v[1])).sum())
        return float(num / den) if den > 0 else np.nan
    except Exception:
        return np.nan


def _jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return np.nan
    return len(sa & sb) / len(sa | sb)


def compute_similarity():
    con = sqlite3.connect(str(DB), timeout=30); con.executescript(DDL)
    tickers = [r[0] for r in con.execute("SELECT DISTINCT ticker FROM sec_10k_sections").fetchall()]
    n_pairs = 0
    for tk in tickers:
        for section in SECTIONS:
            rows = con.execute("""SELECT filing_date, text FROM sec_10k_sections
                WHERE ticker=? AND section=? ORDER BY filing_date""", (tk, section)).fetchall()
            for k in range(1, len(rows)):
                prev_d, prev_t = rows[k - 1]; cur_d, cur_t = rows[k]
                con.execute("""INSERT OR REPLACE INTO sec_10k_similarity
                    (ticker,filing_date,prev_filing_date,section,cosine,jaccard) VALUES (?,?,?,?,?,?)""",
                    (tk, cur_d, prev_d, section, _cosine(prev_t, cur_t), _jaccard(prev_t, cur_t)))
                n_pairs += 1
        con.commit()
    print(f"  computed {n_pairs} YoY similarity pairs across {len(tickers)} tickers")
    import pandas as pd
    df = pd.read_sql("SELECT * FROM sec_10k_similarity", con)
    if len(df):
        print("\n  cosine by section:")
        print(df.groupby("section")["cosine"].describe()[["count","mean","min","25%","75%"]].to_string())
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--similarity", action="store_true")
    ap.add_argument("--days-back", type=int, default=2600)
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in open(ROOT / "tickers.txt")
               if t.strip() and not t.startswith("#")]
    if args.fetch:
        print(f"Fetching 10-Ks for {len(tickers)} tickers...")
        fetch_all(tickers, days_back=args.days_back)
    if args.similarity:
        print("Computing YoY similarity...")
        compute_similarity()
    if not (args.fetch or args.similarity):
        ap.error("pass --fetch and/or --similarity")


if __name__ == "__main__":
    main()
