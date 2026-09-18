#!/usr/bin/env python3
"""
ingest_congress.py — populate congress_trades.db from Unusual Whales.

WHAT WAS THERE BEFORE
    congress_trades.db existed at 0 bytes, created 2026-04-28, with no tables.
    features/builder.py:1250 `_load_congress` queried it, hit the bare
    `except Exception: return zeros`, and returned a constant 0.0 silently. The
    loader has never returned anything else.

    It never reached the model -- congress_net_shares is not in FEATURE_COLUMNS
    and has no rows in feature_importance_history -- so this is unbuilt
    plumbing rather than a corrupted feature. Worth stating precisely, because
    the same shape (an empty source, a silent fallback, a plausible-looking
    zero) WAS a live defect in vix_ret and dxy_ret for three months.

POINT-IN-TIME, AND WHY IT IS THE WHOLE DESIGN
    Members must disclose within 45 days. `transaction_date` is when the trade
    executed; `filed_at_date` is when the public could first know. A feature
    keyed on transaction_date is LOOK-AHEAD by up to 45 days and any backtest
    using it is contaminated.

    The congress_flows VIEW is therefore keyed on filed_at_date. That also
    matches the only construction with surviving evidence: post-2012 alphas for
    congressional leaders hold "even when portfolios are constructed using
    public disclosure dates rather than actual execution dates" (CEPR 2025).

WHAT THE EVIDENCE SAYS, SO THE PAGE IS NOT BUILT ON A DEAD SIGNAL
    Ziobrowski (2004, 2011) found the Senate beating the market by ~12%/yr and
    the House by 55bps/month -- on 1985-2001 data, BEFORE the STOCK Act.

    After 2012 that reverses. A study of 17,859 disclosures found the typical
    congressional purchase LAGGED the S&P 500 by 5.36 points after one year,
    and that eliminating the disclosure lag still does not create
    market-beating returns. Senate CARs measure -0.15% and +0.43%. A 2020 NBER
    study finds senators' portfolios do not meaningfully beat the market.

    The exception is narrow and specific: for LEADERS -- the small group at the
    top of the political hierarchy -- trade frequency falls after 2012 but
    average trade size and risk-adjusted returns remain largely intact.

    So an aggregate "what Congress is buying" list is a signal the literature
    says is dead. The subsets worth testing are leadership, committee-
    jurisdiction overlap, and multi-member clusters. This script stores the
    FULL record so those subsets are constructible; the old congress_flows
    shape (ticker, date, net) threw away member identity and made them
    impossible.

DOLLARS, NOT SHARES
    The builder's column is named congress_net_shares. UW reports an amount
    RANGE ("$1,001 - $15,000"), never a share count, so this stores the range
    midpoint in dollars under that name to keep the existing loader working.
    The name is wrong and the value is a dollar figure. Flagged rather than
    silently papered over; renaming would require touching the loader and
    FEATURE_COLUMNS, which is a separate change.

USAGE
    python analysis/ingest_congress.py                 # incremental
    python analysis/ingest_congress.py --full          # walk all pages
    python analysis/ingest_congress.py --status
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request as ureq
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = Path(os.getenv("CONGRESS_DB_PATH", ROOT / "congress_trades.db"))
BASE = "https://api.unusualwhales.com"

DDL = """
CREATE TABLE IF NOT EXISTS congress_trades (
    ticker            TEXT,
    transaction_date  TEXT NOT NULL,   -- when it executed
    filed_at_date     TEXT NOT NULL,   -- when the public could first know
    txn_type          TEXT,
    amounts           TEXT,            -- the raw range string, kept verbatim
    amount_mid        REAL,            -- parsed midpoint, dollars
    name              TEXT,            -- member
    politician_id     TEXT,
    member_type       TEXT,            -- house / senate / other
    reporter          TEXT,            -- who filed: member, spouse, child
    issuer            TEXT,
    notes             TEXT,
    is_active         INTEGER,
    fetched_at        TEXT NOT NULL,
    PRIMARY KEY (ticker, transaction_date, filed_at_date, name, txn_type, amounts)
);
CREATE INDEX IF NOT EXISTS idx_ct_filed  ON congress_trades(filed_at_date);
CREATE INDEX IF NOT EXISTS idx_ct_ticker ON congress_trades(ticker, filed_at_date);
CREATE INDEX IF NOT EXISTS idx_ct_name   ON congress_trades(name);

-- The view features/builder.py:1256 already queries, unchanged in shape.
-- Keyed on filed_at_date, NOT transaction_date: a member's trade is not
-- knowable until it is filed, up to 45 days later, and keying on execution
-- would put up to 45 days of look-ahead into every backtest that touches it.
DROP VIEW IF EXISTS congress_flows;
CREATE VIEW congress_flows AS
SELECT ticker,
       filed_at_date AS ds,
       SUM(CASE
             WHEN LOWER(txn_type) LIKE '%buy%'      THEN  amount_mid
             WHEN LOWER(txn_type) LIKE '%purchase%' THEN  amount_mid
             WHEN LOWER(txn_type) LIKE '%sell%'     THEN -amount_mid
             WHEN LOWER(txn_type) LIKE '%sale%'     THEN -amount_mid
             ELSE 0.0
           END) AS congress_net_shares
FROM congress_trades
WHERE ticker IS NOT NULL AND ticker <> '' AND amount_mid IS NOT NULL
GROUP BY ticker, filed_at_date;
"""

_NUM = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)")


def parse_amount(s):
    """Range string -> midpoint in dollars. UW never reports a share count.

    '$1,001 - $15,000' -> 8000.5 ; '$50,000,000 +' -> 50000000.0
    Returns None rather than 0.0 when nothing parses: 0.0 would be a stated
    'no money moved', which is a different claim from 'unknown'.
    """
    if not s:
        return None
    vals = [float(m.replace(",", "")) for m in _NUM.findall(str(s))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fetch(endpoint, key, params=None, timeout=30):
    url = BASE + endpoint
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = ureq.Request(url, headers={"Authorization": f"Bearer {key}",
                                     "Accept": "application/json"})
    with ureq.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d.get("data") if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="walk pages until exhausted; default stops early once "
                         "a page yields no new rows")
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument("--tickers", action="store_true",
                    help="backfill per ticker from tickers.txt -- deeper "
                         "history (NVDA reaches 2024-11-01) but 422 calls")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    con = sqlite3.connect(DB, timeout=30)
    con.executescript(DDL)
    con.commit()

    if a.status:
        n, tk, lo, hi = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(filed_at_date), "
            "MAX(filed_at_date) FROM congress_trades").fetchone()
        print(f"congress_trades: {n:,} rows, {tk} tickers, filed {lo} .. {hi}")
        for row in con.execute(
                "SELECT member_type, COUNT(*) FROM congress_trades "
                "GROUP BY 1 ORDER BY 2 DESC"):
            print(f"  {row[0] or '?':12} {row[1]:>7,}")
        lag = con.execute(
            "SELECT ROUND(AVG(julianday(filed_at_date)-julianday("
            "transaction_date)),1) FROM congress_trades "
            "WHERE filed_at_date >= transaction_date").fetchone()[0]
        print(f"  mean disclosure lag: {lag} days "
              f"(statutory limit 45; the reason congress_flows keys on "
              f"filed_at_date)")
        con.close()
        return

    key = os.environ.get("UW_API_KEY")
    if not key:
        print("UW_API_KEY not set -- run: set -a; . ./.env; set +a")
        sys.exit(1)

    before = con.execute("SELECT COUNT(*) FROM congress_trades").fetchone()[0]
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    total_seen, pages = 0, 0

    # PAGINATION DOES NOT EXIST ON THIS ENDPOINT. Probed 2026-09-15: `page`
    # and `offset` are accepted and SILENTLY IGNORED -- every variant returns
    # the same first row. So are `filed_at_date`, `start_date`/`end_date` and
    # `older_than`. Only two parameters do anything:
    #   limit=  caps at 200 (500 returns 422)
    #   date=   returns a window AROUND that date, not from it: date=2026-06-15
    #           gave filings 2026-06-01 .. 2026-07-19
    #   ticker= returns that name's own history, NVDA reaching 2024-11-01
    #
    # A parameter that is accepted and ignored is worse than one that errors:
    # the first backfill attempt "succeeded" 200 times and stored one window.
    if a.tickers:
        _names = [t.strip().upper() for t in
                  open(ROOT / "tickers.txt") if t.strip()]
        _reqs = [{"limit": 200, "ticker": t} for t in _names]
        print(f"  backfill by ticker: {len(_reqs)} names from tickers.txt")
    else:
        # Step `date` backwards in ~6-week strides, since each call returns
        # roughly that much of a window.
        import datetime as _dt
        _d = _dt.date.today()
        _reqs = [{"limit": 200}]
        for _ in range(a.max_pages - 1):
            _d -= _dt.timedelta(days=42)
            _reqs.append({"limit": 200, "date": _d.isoformat()})

    for page, _req in enumerate(_reqs):
        try:
            rows = fetch("/api/congress/recent-trades", key, _req)
        except Exception as e:
            print(f"  req {page}: {type(e).__name__}: {str(e)[:70]}")
            continue
        if not rows:
            continue
        pages += 1
        total_seen += len(rows)
        cur = con.execute("SELECT COUNT(*) FROM congress_trades").fetchone()[0]
        con.executemany("""
            INSERT OR IGNORE INTO congress_trades
            (ticker, transaction_date, filed_at_date, txn_type, amounts,
             amount_mid, name, politician_id, member_type, reporter, issuer,
             notes, is_active, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [(
            (r.get("ticker") or "").upper() or None,
            r.get("transaction_date") or "",
            r.get("filed_at_date") or r.get("transaction_date") or "",
            r.get("txn_type"), r.get("amounts"), parse_amount(r.get("amounts")),
            r.get("name"), str(r.get("politician_id") or ""),
            r.get("member_type"), r.get("reporter"), r.get("issuer"),
            r.get("notes"), 1 if r.get("is_active") else 0, now,
        ) for r in rows if r.get("transaction_date")])
        con.commit()
        after = con.execute("SELECT COUNT(*) FROM congress_trades").fetchone()[0]
        new = after - cur
        print(f"  page {page:>3}: {len(rows):>4} rows, {new:>4} new "
              f"(total {after:,})")
        # Incremental stops when a whole page is already stored. --full keeps
        # going: UW pages are filed-date ordered, and a backfill can cross a
        # stretch that is already complete before reaching older gaps.
        if new == 0 and not a.full and not a.tickers:
            print("  window yielded nothing new -- stopping "
                  "(--full walks on, --tickers backfills per name)")
            break
        time.sleep(0.25)

    after = con.execute("SELECT COUNT(*) FROM congress_trades").fetchone()[0]
    flows = con.execute("SELECT COUNT(*) FROM congress_flows").fetchone()[0]
    print(f"\n{pages} pages, {total_seen:,} rows seen, "
          f"{after - before:,} new, {after:,} stored")
    print(f"congress_flows view: {flows:,} ticker-days, keyed on filed_at_date")
    unparsed = con.execute(
        "SELECT COUNT(*) FROM congress_trades WHERE amount_mid IS NULL").fetchone()[0]
    if unparsed:
        print(f"[note] {unparsed:,} rows have an unparseable amount and are "
              f"excluded from the view rather than counted as zero")
    con.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
