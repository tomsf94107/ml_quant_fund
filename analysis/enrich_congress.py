#!/usr/bin/env python3
"""
enrich_congress.py — who they are, what they oversee, and what they bought.

Builds `congress_members` (roster) and the `congress_360` view, which joins
every disclosure to the filer's party, chamber, state, leadership role and
committee assignments, plus the traded name's sector and the filing lag.

WHY THE ROSTER IS A SEPARATE FETCH
    The UW feed carries name, member_type, ticker, amounts, txn_type,
    transaction_date and filed_at_date. It does NOT carry party, title,
    seniority or committee membership, and those are exactly what the only
    surviving angle needs: post-2012 abnormal returns hold for the small group
    of LEADERS at the top of the political hierarchy, not for the median
    member. Without a roster there is no way to construct that subset.

    Source: github.com/unitedstates/congress-legislators -- public domain,
    no key, maintained continuously, and the standard dataset for this.

THE THREE COLUMNS THAT CARRY THE ACTUAL ANGLES

    is_leadership   Party leaders, whips, and committee chairs/ranking members.
                    The one slice with post-STOCK-Act evidence.

    lag_days        filed_at_date minus transaction_date. The statutory limit
                    is 45 days and the measured mean on this data is 60.1, so
                    late filing is routine rather than exceptional. Filing late
                    is a choice, and a long lag on a large trade is a different
                    object from a long lag on a routine one.

    jurisdiction    Whether the filer sits on a committee whose remit covers
                    the traded name's sector. An Armed Services member buying a
                    defense contractor is the specific information channel the
                    literature describes; a Small Business member buying the
                    same name is not.

                    THIS COLUMN IS INFERENCE, NOT DATA. The committee-to-sector
                    map below is a judgement about which committees plausibly
                    see information about which sectors. It is not published by
                    anyone and it is not validated. Treat a jurisdiction hit as
                    a reason to look, never as evidence.

BRANCH, NOT JUST CHAMBER
    650 of the first 3,690 rows are `executive` -- cabinet officers and
    White House filers under a different disclosure regime (OGE Form 278e,
    different thresholds, different timing). They are kept in the same table
    with a `branch` column rather than dropped, because they are interesting,
    but they must not be pooled into a "Congress" statistic. Linda E McMahon
    alone shows $141.6M across 82 filings.

A CAUTION ABOUT THE HEAVIEST FILERS
    The highest trade counts in this data -- Ro Khanna at 537, Gilbert
    Cisneros at 319 -- are widely reported as advisor-managed accounts where
    the member has no trade-level discretion. High count, low information.
    `reporter` (self / spouse / child) is carried through so those can be
    separated, but it does not identify managed accounts; no field does.
    Any ranking by trade COUNT will be dominated by these filers.

USAGE
    python analysis/enrich_congress.py --refresh-roster
    python analysis/enrich_congress.py --status
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import urllib.request as ureq
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = Path(os.getenv("CONGRESS_DB_PATH", ROOT / "congress_trades.db"))

# The repo stores YAML at main and PUBLISHES the built JSON to GitHub Pages.
# The raw.githubusercontent.com/...main/*.json paths 404 -- those files do not
# exist in the tree. Verified 2026-09-18: all three Pages URLs return 200.
_PAGES = "https://unitedstates.github.io/congress-legislators/"
LEG_URL = _PAGES + "legislators-current.json"
CMTE_MEM_URL = _PAGES + "committee-membership-current.json"
CMTE_URL = _PAGES + "committees-current.json"

# COMMITTEE -> SECTOR ETF. Inference, not data. Which committees plausibly see
# information about which sectors, by their published jurisdiction. Committee
# codes are THOMAS IDs: H = House, S = Senate.
CMTE_SECTOR = {
    "HSAS": ["XLI"],            # Armed Services -- defense primes
    "SSAS": ["XLI"],
    "HSIF": ["XLV", "XLC", "XLE"],   # Energy & Commerce -- health, telecom, power
    "SSCM": ["XLC", "XLI"],     # Commerce, Science & Transportation
    "HSBA": ["XLF"],            # Financial Services
    "SSBK": ["XLF"],            # Banking, Housing & Urban Affairs
    "HSWM": ["XLV", "XLF"],     # Ways & Means -- tax, health financing
    "SSFI": ["XLV", "XLF"],     # Finance
    "HSAG": ["XLP"],            # Agriculture
    "SSAF": ["XLP"],
    "HSII": ["XLE", "XLB"],     # Natural Resources
    "SSEG": ["XLE", "XLU"],     # Energy & Natural Resources
    "HSSY": ["XLK"],            # Science, Space & Technology
    "HSVR": ["XLV"],            # Veterans' Affairs
    "SSVA": ["XLV"],
    "SSHR": ["XLV"],            # Health, Education, Labor & Pensions
    "HSPW": ["XLI", "XLB"],     # Transportation & Infrastructure
    "SSEV": ["XLU", "XLB"],     # Environment & Public Works
    "HSJU": ["XLK", "XLC"],     # Judiciary -- antitrust
    "SSJU": ["XLK", "XLC"],
    "HLIG": ["XLK", "XLI"],     # Intelligence
    "SLIN": ["XLK", "XLI"],
}

LEADERSHIP_TITLES = ("Speaker", "Majority Leader", "Minority Leader",
                     "Majority Whip", "Minority Whip", "President Pro Tempore",
                     "Chair", "Ranking Member")

DDL = """
CREATE TABLE IF NOT EXISTS congress_members (
    match_key     TEXT PRIMARY KEY,   -- normalized name, the join key
    full_name     TEXT,
    party         TEXT,
    state         TEXT,
    chamber       TEXT,               -- house / senate
    leadership    TEXT,               -- role string, NULL if none
    is_leadership INTEGER DEFAULT 0,
    committees    TEXT,               -- comma-separated committee names
    cmte_codes    TEXT,               -- comma-separated THOMAS ids
    sectors       TEXT,               -- sector ETFs their committees cover
    bioguide      TEXT
);
"""


def norm(name):
    """Normalize a name for joining. UW gives 'Ro Khanna' and
    'Michael McCaul'; the roster gives first/last separately, sometimes with
    middle names and suffixes. Lowercase, strip punctuation and suffixes, keep
    first and last token only -- middle names and initials are the main source
    of mismatch and dropping them costs little."""
    if not name:
        return ""
    s = re.sub(r"[^\w\s]", " ", str(name).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|dr|mr|mrs|ms)\b", " ", s)
    toks = [t for t in s.split() if len(t) > 1]
    if len(toks) >= 2:
        return f"{toks[0]} {toks[-1]}"
    return " ".join(toks)


def get_json(url, timeout=60):
    with ureq.urlopen(ureq.Request(
            url, headers={"User-Agent": "ml-quant-fund/1.0"}), timeout=timeout) as r:
        return json.load(r)


def refresh_roster(con):
    print("  fetching legislators-current.json ...")
    legs = get_json(LEG_URL)
    print(f"    {len(legs)} sitting members")
    print("  fetching committee data ...")
    cmte_mem = get_json(CMTE_MEM_URL)
    cmtes = {c["thomas_id"]: c for c in get_json(CMTE_URL) if c.get("thomas_id")}
    print(f"    {len(cmte_mem)} committees with membership, {len(cmtes)} defined")

    # bioguide -> [(code, name, title)]
    by_bio = {}
    for code, members in cmte_mem.items():
        cname = cmtes.get(code, {}).get("name", code)
        for m in members:
            b = m.get("bioguide")
            if b:
                by_bio.setdefault(b, []).append((code, cname, m.get("title")))

    rows = []
    for L in legs:
        term = (L.get("terms") or [{}])[-1]
        bio = (L.get("id") or {}).get("bioguide")
        nm = L.get("name") or {}
        full = f"{nm.get('first','')} {nm.get('last','')}".strip()
        mine = by_bio.get(bio, [])
        codes = [c for c, _, _ in mine]
        cnames = [n for _, n, _ in mine]
        titles = [t for _, _, t in mine if t]
        # Leadership = a chamber role on the term, or a committee chair /
        # ranking membership. Both are "top of the hierarchy" in the sense the
        # evidence uses.
        # FULL COMMITTEES ONLY. committee-membership-current.json covers 230
        # committees of which only 49 are defined in committees-current.json --
        # the other 181 are SUBcommittees, and their chairs and ranking members
        # carry the same title strings. Counting those flagged 360 of 539
        # members as leadership, two thirds of Congress, which discriminates
        # nothing.
        #
        # The evidence is about "the small group of leaders at the top of the
        # political hierarchy" (CEPR 2025), not anyone with a gavel. THOMAS ids
        # are 4 characters for a full committee and longer for a subcommittee,
        # so the length test separates them.
        _full = [(c, t) for c, _, t in mine if len(c) <= 4 and t]
        _chair = [t for _, t in _full
                  if "chair" in t.lower() or "ranking" in t.lower()]
        _chamber = term.get("leadership_role")   # Speaker, Whip, Leader
        if _chamber:
            lead = _chamber
        elif _chair:
            lead = "; ".join(sorted(set(_chair)))
        else:
            lead = None
        secs = sorted({s for c in codes for s in CMTE_SECTOR.get(c, [])})
        rows.append((
            norm(full), full, term.get("party"), term.get("state"),
            term.get("type") == "sen" and "senate" or "house",
            lead, 1 if lead else 0,
            ", ".join(sorted(set(cnames)))[:800],
            ",".join(sorted(set(codes))),
            ",".join(secs), bio,
        ))
    con.execute("DELETE FROM congress_members")
    con.executemany(
        "INSERT OR REPLACE INTO congress_members VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows)
    con.commit()
    print(f"    stored {len(rows)} members, "
          f"{sum(1 for r in rows if r[6])} flagged leadership")


def build_view(con):
    """congress_360 -- one row per disclosure with everything joined.

    LEFT JOIN on the roster, never INNER: executive-branch filers and former
    members have no roster row and must still appear, with NULL party. An
    INNER JOIN would silently delete them, and 650 of the first 3,690 rows are
    executive.
    """
    con.executescript("""
    DROP VIEW IF EXISTS congress_360;
    CREATE VIEW congress_360 AS
    SELECT
      t.filed_at_date,
      t.transaction_date,
      CAST(julianday(t.filed_at_date) - julianday(t.transaction_date) AS INT)
          AS lag_days,
      CASE WHEN julianday(t.filed_at_date) - julianday(t.transaction_date) > 45
           THEN 1 ELSE 0 END AS filed_late,
      CASE WHEN t.member_type = 'executive' THEN 'executive'
           ELSE 'legislative' END AS branch,
      t.member_type            AS chamber,
      t.name,
      m.party,
      m.state,
      m.leadership,
      COALESCE(m.is_leadership, 0) AS is_leadership,
      m.committees,
      m.sectors                AS oversees_sectors,
      t.reporter,
      t.ticker,
      t.issuer,
      t.txn_type,
      t.amounts,
      t.amount_mid,
      CASE WHEN m.sectors IS NOT NULL AND m.sectors <> '' THEN 1 ELSE 0 END
          AS has_jurisdiction_data
    FROM congress_trades t
    LEFT JOIN congress_members m ON m.match_key = LOWER(
        TRIM(SUBSTR(t.name, 1, INSTR(t.name || ' ', ' ') - 1)) || ' ' ||
        TRIM(REPLACE(t.name, RTRIM(t.name, REPLACE(t.name, ' ', '')), ''))
    );
    """)
    con.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-roster", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    con = sqlite3.connect(DB, timeout=30)
    con.executescript(DDL)

    if a.refresh_roster:
        try:
            refresh_roster(con)
        except Exception as e:
            print(f"  roster fetch FAILED: {type(e).__name__}: {e}")
            print("  (the view still builds; party and committee will be NULL)")

    build_view(con)

    n = con.execute("SELECT COUNT(*) FROM congress_360").fetchone()[0]
    matched = con.execute(
        "SELECT COUNT(*) FROM congress_360 WHERE party IS NOT NULL").fetchone()[0]
    print(f"\ncongress_360: {n:,} rows, {matched:,} joined to a roster entry "
          f"({100*matched/max(n,1):.0f}%)")
    print("  unmatched rows are executive-branch filers, former members, and "
          "name-format misses -- they are KEPT with NULL party, not dropped")

    if a.status:
        print("\n  by branch and leadership:")
        for r in con.execute(
                "SELECT branch, is_leadership, COUNT(*), "
                "ROUND(AVG(lag_days),1), ROUND(SUM(amount_mid)/1e6,1) "
                "FROM congress_360 GROUP BY 1,2 ORDER BY 3 DESC"):
            print(f"    {r[0]:12} leadership={r[1]}  {r[2]:>6,} rows  "
                  f"lag {r[3]}d  ${r[4]}M")
    con.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
