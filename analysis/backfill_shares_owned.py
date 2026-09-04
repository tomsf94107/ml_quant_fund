#!/usr/bin/env python3
"""
backfill_shares_owned.py — add sharesOwnedFollowingTransaction to the insider DB.

WHY
    Two tests have now found no predictive content in insider selling at h=5:
    five window constructions (analysis/insider_construction_test.py) and five
    trajectory constructions across h=5/20/60
    (analysis/insider_timeseries_test.py). Both were clean nulls.

    Both tested FLOW -- what was already sold. The literature definition of
    overhang is a STOCK: "a large block of securities that the market knows or
    suspects will be sold in the near future". What is LEFT to sell.

    Those are opposite in the case that motivated this. A holder who has just
    finished distributing and one who is 10% through produce the SAME 90-day
    sum, the same persistence, the same breadth. Only the remaining stake
    separates them. For CRWV that difference is 32.9M shares -- about 7% of
    outstanding -- still to come.

    Form 4 reports it in every transaction block as
    sharesOwnedFollowingTransaction. insider_filings_raw does not store it
    (verified 2026-09-03 against the schema). This backfills it.

WHAT IT DOES
    For each accession already in insider_filings_raw, fetches the Form 4 XML
    from EDGAR and extracts, per (insider, transaction), the post-transaction
    holding. Writes to a NEW table -- insider_holdings -- rather than altering
    insider_filings_raw, so nothing that reads the existing table changes
    behaviour and a failed run cannot corrupt what works.

    Resumable: accessions already present are skipped, so an interrupted run
    continues where it stopped.

RATE LIMITS AND SCALE
    SEC asks for fewer than 10 requests/second and a descriptive User-Agent
    with a real email; requests without one are refused. This defaults to ~7/s.

    383,355 filing ROWS exist but they share accessions -- one filing carries
    many transaction rows. Run --count first to see the real number of unique
    accessions before committing to a long scrape.

HONEST LIMITS
    1. sharesOwnedFollowingTransaction EXCLUDES unvested RSUs and unexercised
       options. It is the reported direct/indirect holding, not economic
       exposure.
    2. A holder files per account. One beneficial owner can appear under
       several names -- CRWV showed MICHAEL INTRATOR, "10b5-1 Sales for MICHAEL
       INTRATOR" and OMNADORA CAPITAL LLC at one address. Summing by raw name
       splits one person into several; summing across all names may double-count
       where entities report the same shares. Both raw name and a normalised
       name are stored so the choice stays with the analysis.
    3. Only Section 16 filers appear -- officers, directors and 10%+ owners. A
       large holder below 10% who is not an officer files nothing here.
    4. This backfills only accessions ALREADY ingested. Tickers or periods the
       scraper never covered stay missing.

USAGE
    python analysis/backfill_shares_owned.py --count
    python analysis/backfill_shares_owned.py --ua "name real@email.com" --limit 500
    python analysis/backfill_shares_owned.py --ua "name real@email.com"
"""
import argparse
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request

DDL = """
CREATE TABLE IF NOT EXISTS insider_holdings (
    accession        TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    filing_date      TEXT NOT NULL,
    trade_date       TEXT,
    insider_name     TEXT,
    insider_norm     TEXT,          -- 10b5-1 wrapper stripped
    is_ten_pct       INTEGER,
    is_officer       INTEGER,
    is_director      INTEGER,
    transaction_code TEXT,
    shares           REAL,
    shares_owned_after REAL,        -- sharesOwnedFollowingTransaction
    seq              INTEGER,       -- order within the filing
    fetched_at       TEXT NOT NULL,
    PRIMARY KEY (accession, insider_name, seq)
)
"""


def norm_name(n):
    """Strip the 10b5-1 wrapper so one person is not counted as several.

    Deliberately does NOT merge distinct entities at the same address --
    OMNADORA CAPITAL LLC stays separate from MICHAEL INTRATOR. Merging those
    would be an attribution guess.
    """
    if not n:
        return None
    n = re.sub(r"^\s*10b5-1\s+Sales?\s+for\s+", "", n, flags=re.I)
    return " ".join(n.split()).upper()


def blocks(xml, tag):
    return re.findall(rf"<{tag}>(.*?)</{tag}>", xml, re.S)


def val(block, tag):
    """Form 4 wraps most fields as <tag><value>X</value></tag>."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.S)
    if not m:
        return None
    inner = m.group(1)
    v = re.search(r"<value>([^<]*)</value>", inner)
    return (v.group(1) if v else inner).strip()


def fnum(s):
    if s is None:
        return None
    s = re.sub(r"[,$\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def fetch(cik, accession, ua, timeout=30):
    """Form 4 filenames vary by agent; find the XML in the index."""
    acc = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}"
    req = urllib.request.Request(base + "/", headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        idx = r.read().decode("utf-8", "replace")
    cands = [f for f in re.findall(r'[a-zA-Z0-9_\.\-]+\.xml', idx)
             if "primary_doc" in f or "seq" in f or f.endswith("_4.xml")]
    if not cands:
        cands = re.findall(r'[a-zA-Z0-9_\.\-]+\.xml', idx)
    if not cands:
        return None
    req = urllib.request.Request(f"{base}/{cands[0]}",
                                 headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse(xml, ticker, accession):
    """-> list of rows. One per non-derivative transaction."""
    out = []
    owners = blocks(xml, "reportingOwner")
    name = None
    ten = off = dirn = 0
    if owners:
        name = val(owners[0], "rptOwnerName")
        rel = blocks(owners[0], "reportingOwnerRelationship")
        if rel:
            ten = 1 if (val(rel[0], "isTenPercentOwner") or "0") == "1" else 0
            off = 1 if (val(rel[0], "isOfficer") or "0") == "1" else 0
            dirn = 1 if (val(rel[0], "isDirector") or "0") == "1" else 0
    fdate = val(xml, "periodOfReport")

    for i, b in enumerate(blocks(xml, "nonDerivativeTransaction")):
        out.append({
            "accession": accession, "ticker": ticker,
            "trade_date": val(b, "transactionDate") or fdate,
            "insider_name": name, "insider_norm": norm_name(name),
            "is_ten_pct": ten, "is_officer": off, "is_director": dirn,
            "transaction_code": val(b, "transactionCode"),
            "shares": fnum(val(b, "transactionShares")),
            "shares_owned_after":
                fnum(val(b, "sharesOwnedFollowingTransaction")),
            "seq": i,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="insider_trades.db")
    ap.add_argument("--ua", help="descriptive User-Agent with a real email; "
                                 "EDGAR returns 403 without one")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N accessions; 0 = all")
    ap.add_argument("--rps", type=float, default=7.0)
    ap.add_argument("--count", action="store_true",
                    help="report how many accessions need fetching, then exit")
    ap.add_argument("--since", default="2021-01-01")
    ap.add_argument("--tickers",
                    help="comma-separated list; restricts the backfill. The "
                         "full set is 134,315 accessions (~10.7h). The recent-"
                         "IPO cohort where a large holder is plausibly "
                         "distributing is ~3,200 (~15 min), which is the cheap "
                         "way to see whether the variable separates at all "
                         "before committing to the full scrape.")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.execute(DDL)
    con.commit()

    done = {r[0] for r in con.execute(
        "SELECT DISTINCT accession FROM insider_holdings")}
    if args.tickers:
        tk = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
        q = ",".join("?" * len(tk))
        cur = con.execute(
            "SELECT DISTINCT accession, ticker, filing_date "
            f"FROM insider_filings_raw WHERE filing_date >= ? "
            f"AND ticker IN ({q}) ORDER BY filing_date", [args.since] + tk)
        print(f"restricted to {len(tk)} tickers")
    else:
        cur = con.execute(
            "SELECT DISTINCT accession, ticker, filing_date "
            "FROM insider_filings_raw WHERE filing_date >= ? "
            "ORDER BY filing_date", (args.since,))
    todo = [(a, t, fd) for a, t, fd in cur if a not in done]

    print(f"{len(done):,} accessions already backfilled")
    print(f"{len(todo):,} remaining since {args.since}")
    if todo:
        secs = len(todo) * 2 / args.rps          # two requests per accession
        print(f"at {args.rps:.0f} req/s and 2 requests each: "
              f"~{secs/3600:.1f} hours"
              + (" for this subset" if args.tickers else " for the full set"))
    if args.count:
        con.close()
        return
    if not args.ua:
        con.close()
        raise SystemExit("--ua required. EDGAR refuses requests without a "
                         "descriptive User-Agent containing a real email.")

    # ticker -> issuer CIK from SEC's official map.
    #
    # The first version searched browse-edgar with company=<TICKER>, which is a
    # company NAME search: "CRWV" is not CoreWeave's name, so it returned an
    # empty feed and every accession failed. 100/100 failed silently on the
    # first trial run. company_tickers.json is the canonical mapping, one
    # download for all 10,412 tickers, and removes a request per ticker.
    ciks = {}
    try:
        req = urllib.request.Request(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": args.ua})
        with urllib.request.urlopen(req, timeout=45) as r:
            import json as _json
            _m = _json.loads(r.read().decode())
        ciks = {v["ticker"].upper(): str(v["cik_str"]) for v in _m.values()}
        print(f"resolved {len(ciks):,} ticker->CIK mappings")
    except Exception as e:
        raise SystemExit(f"could not fetch company_tickers.json: {e}")
    delay = 1.0 / max(args.rps, 0.5)
    n_ok = n_fail = n_rows = 0
    unmapped = set()
    first_error_shown = False
    t0 = time.time()

    for i, (acc, tk, fd) in enumerate(todo):
        if args.limit and i >= args.limit:
            break
        cik = ciks.get(tk.upper())
        if not cik:
            if tk not in unmapped:
                unmapped.add(tk)
                print(f"  {tk}: no CIK in company_tickers.json -- skipped")
            n_fail += 1
            continue
        try:
            xml = fetch(cik, acc, args.ua)
            if not xml:
                n_fail += 1
                continue
            rows = parse(xml, tk, acc)
            if rows:
                con.executemany(
                    "INSERT OR IGNORE INTO insider_holdings "
                    "(accession, ticker, filing_date, trade_date, insider_name,"
                    " insider_norm, is_ten_pct, is_officer, is_director,"
                    " transaction_code, shares, shares_owned_after, seq,"
                    " fetched_at) VALUES "
                    "(:accession,:ticker,:filing_date,:trade_date,:insider_name,"
                    ":insider_norm,:is_ten_pct,:is_officer,:is_director,"
                    ":transaction_code,:shares,:shares_owned_after,:seq,"
                    ":fetched_at)",
                    [dict(r, filing_date=fd,
                          fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                     for r in rows])
                n_rows += len(rows)
            n_ok += 1
            if n_ok % 200 == 0:
                con.commit()
                el = time.time() - t0
                print(f"  {n_ok:,} ok  {n_fail:,} failed  {n_rows:,} rows  "
                      f"{el/60:.1f} min  ({n_ok/max(el,1):.1f}/s)")
        except urllib.error.HTTPError as e:
            n_fail += 1
            if e.code == 403:
                con.commit(); con.close()
                raise SystemExit("EDGAR returned 403 -- the User-Agent was "
                                 "refused. Use a real name and email.")
            if e.code == 429:
                print("  rate limited; backing off 10s")
                time.sleep(10)
        except Exception as e:
            n_fail += 1
            # Print the FIRST failure with its reason. The trial run reported
            # "100 failed" with no explanation, which is the same silent-failure
            # shape this codebase keeps producing: it ran, wrote nothing, and
            # said nothing about why.
            if not first_error_shown:
                first_error_shown = True
                print(f"  first failure: {tk} {acc} -- "
                      f"{type(e).__name__}: {e}")
        time.sleep(delay)

    con.commit()
    got = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker), "
                      "COALESCE(SUM(shares_owned_after IS NOT NULL),0) "
                      "FROM insider_holdings").fetchone()
    con.close()
    print(f"\n  {n_ok:,} accessions fetched, {n_fail:,} failed")
    # COALESCE above: SUM over an empty table returns NULL, and formatting NULL
    # with :, raised TypeError on the first trial -- an error report that itself
    # errored.
    print(f"  insider_holdings: {got[0]:,} rows, {got[1] or 0} tickers, "
          f"{got[2]:,} with shares_owned_after")
    print("\n  Resumable -- rerun to continue. Then test overhang with the "
          "latest\n  shares_owned_after per insider, summed per ticker, "
          "normalised by ADV\n  or shares outstanding.")


if __name__ == "__main__":
    sys.exit(main() or 0)
