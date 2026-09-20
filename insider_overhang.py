#!/usr/bin/env python3
"""
insider_overhang.py
Answer "how much is LEFT to sell", not just "how much was sold".

The monitor's `90d SELL $X` is EXECUTED sales (rear-view). To size the remaining
overhang you need each insider's CURRENT holdings, which live in the Form-4 field
`sharesOwnedFollowingTransaction` (the "owned after" balance) that the monitor
parses but does not surface. This pulls that field from EDGAR and computes:

  per insider:  shares_sold_window, $_sold_window, latest_shares_owned_after,
                remaining_$ (= owned_after * price), as_of date, is_10pct flag
  totals:       total sold (reconciles vs the monitor's window) + total remaining overhang

HONEST LIMITS (read these):
  * `owned_after` for a 10%+ institutional holder (e.g. Magnetar) can be a per-account
    /direct-only figure and may NOT equal total beneficial ownership. For funds the
    13D/G or 13F is the better remaining-stake source. Flagged per-owner.
  * `owned_after` excludes unvested RSUs/unexercised options -> sellable can differ.
  * Form 144 (noticed-but-unexecuted intent) is a separate, cleaner forward number;
    stubbed below as a follow-on (different schema).
  * SEC EDGAR is NOT reachable from the build sandbox, so the live fetch here is
    UNTESTED end-to-end -- validate on your box. The XML PARSER is unit-tested offline.

USAGE
  python insider_overhang.py --ticker CRWV --cik 1769628 --price 84.26 \
         --since 2026-06-01 --ua "atom your_email@domain.com"
  python insider_overhang.py --self-test        # offline parser check, no network
"""

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date

import requests

EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
EDGAR_INDEX = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/index.json"


# ----------------------------------------------------------------------------- #
# Form-4 XML parser (ownershipDocument is un-namespaced) -- the tested core
# ----------------------------------------------------------------------------- #
def _txt(el, path):
    x = el.find(path)
    return x.text.strip() if x is not None and x.text else None


def parse_form4(xml_text):
    if "<ownershipDocument" not in xml_text:
        raise RuntimeError("not ownershipDocument (EDGAR error page / wrong doc)")
    """Return dict: issuer + reporting owners + non-derivative transactions/holdings,
    each carrying sharesOwnedFollowingTransaction. Field names are the fixed EDGAR
    ownershipDocument schema."""
    root = ET.fromstring(xml_text)

    issuer = {
        "cik": _txt(root, "./issuer/issuerCik"),
        "name": _txt(root, "./issuer/issuerName"),
        "symbol": _txt(root, "./issuer/issuerTradingSymbol"),
    }

    owners = []
    for ro in root.findall("./reportingOwner"):
        owners.append({
            "cik": _txt(ro, "./reportingOwnerId/rptOwnerCik"),
            "name": _txt(ro, "./reportingOwnerId/rptOwnerName"),
            "is_director": _txt(ro, "./reportingOwnerRelationship/isDirector") in ("1", "true"),
            "is_officer": _txt(ro, "./reportingOwnerRelationship/isOfficer") in ("1", "true"),
            "is_10pct": _txt(ro, "./reportingOwnerRelationship/isTenPercentOwner") in ("1", "true"),
            "title": _txt(ro, "./reportingOwnerRelationship/officerTitle"),
        })
    # a filing usually names one reporting owner; join name for attribution
    owner_name = owners[0]["name"] if owners else None
    is_10pct = any(o["is_10pct"] for o in owners)
    title = next((o["title"] for o in owners if o["title"]), None)

    lines = []
    # transactions (have transactionAmounts) AND holdings (no transaction, still carry owned-after)
    for kind, xp in (("txn", ".//nonDerivativeTransaction"),
                     ("hold", ".//nonDerivativeHolding")):
        for e in root.findall(xp):
            owned_after = _txt(e, "./postTransactionAmounts/sharesOwnedFollowingTransaction/value")
            lines.append({
                "kind": kind,
                "date": _txt(e, "./transactionDate/value"),
                "code": _txt(e, "./transactionCoding/transactionCode"),
                "shares": _num(_txt(e, "./transactionAmounts/transactionShares/value")),
                "price": _num(_txt(e, "./transactionAmounts/transactionPricePerShare/value")),
                "acq_disp": _txt(e, "./transactionAmounts/transactionAcquiredDisposedCode/value"),
                "owned_after": _num(owned_after),
                "d_or_i": _txt(e, "./ownershipNature/directOrIndirectOwnership/value") or "D",
            })
    return {"issuer": issuer, "owner_name": owner_name, "is_10pct": is_10pct,
            "title": title, "lines": lines}


def _num(s):
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ----------------------------------------------------------------------------- #
# EDGAR fetch (UNTESTED from sandbox -- sec.gov not reachable here)
# ----------------------------------------------------------------------------- #
def edgar_get(url, ua, tries=4):  # RATE_HARDEN
    """EDGAR throttles bursts with 503/429 and returns an HTML error page.
    Retry with backoff and surface the real status, instead of letting an HTML
    body reach the XML parser (which showed up as 'mismatched tag')."""
    import time as _t
    last = None
    for i in range(tries):
        r = requests.get(url, headers={"User-Agent": ua,
                                       "Accept-Encoding": "gzip, deflate"}, timeout=30)
        if r.status_code == 200:
            return r
        last = r.status_code
        if r.status_code in (429, 503, 403):
            _t.sleep(1.0 * (2 ** i))
            continue
        r.raise_for_status()
    raise RuntimeError(f"EDGAR {last} after {tries} tries: {url}")


def list_form4(cik, since, ua):
    j = edgar_get(EDGAR_SUBMISSIONS.format(cik=int(cik)), ua).json()
    rec = j["filings"]["recent"]
    out = []
    for form, acc, doc, fdate in zip(rec["form"], rec["accessionNumber"],
                                     rec["primaryDocument"], rec["filingDate"]):
        if form == "4" and fdate >= since:
            out.append((acc, doc, fdate))
    return out


def fetch_form4_xml(cik, acc, doc, ua):  # INDEX_FIRST
    """ALWAYS resolve the raw xml from the filing index. EDGAR's primaryDocument
    for Form 4 is the xsl-RENDERED path (xslF345X05/form4.xml) which also ends in
    .xml -- trusting it fetched HTML and the parser reported 'mismatched tag'."""
    acc_nodash = acc.replace("-", "")
    idx = edgar_get(EDGAR_INDEX.format(cik=int(cik), acc_nodash=acc_nodash), ua).json()
    names = [i.get("name", "") for i in idx.get("directory", {}).get("item", [])]
    for n in names:
        low = n.lower()
        if low.endswith(".xml") and "xsl" not in low and "/" not in n:
            txt = edgar_get(EDGAR_ARCHIVE.format(cik=int(cik), acc_nodash=acc_nodash, doc=n), ua).text
            if "<ownershipDocument" in txt:
                return txt
    raise RuntimeError(f"no ownership xml in {acc}; files={names}")




# ----------------------------------------------------------------------------- #
# overhang builder
# ----------------------------------------------------------------------------- #
def build(cik, ticker, price, since, ua, sleep=0.5):
    filings = list_form4(cik, since, ua)
    if not filings:
        sys.exit(f"[finding] no Form 4 filings for CIK {cik} since {since}")
    parsed = []
    for acc, doc, fdate in filings:
        try:
            xml = fetch_form4_xml(cik, acc, doc, ua)
            p = parse_form4(xml)
            p["filing_date"] = fdate
            p["accession"] = acc
            parsed.append(p)
        except Exception as e:
            print(f"  [warn] {acc}: {e}", file=sys.stderr)
        time.sleep(sleep)   # be polite to EDGAR
    return summarize(parsed, price)


def summarize(parsed, price):
    sold_sh = defaultdict(float)
    sold_val = defaultdict(float)
    is10 = {}
    title = {}
    # latest owned_after per (owner, direct/indirect), keyed by most recent date
    owned = defaultdict(lambda: (None, None))   # (owner,d_or_i) -> (date, shares)

    for p in parsed:
        owner = p["owner_name"] or "?"
        is10[owner] = is10.get(owner, False) or p["is_10pct"]
        if p["title"]:
            title[owner] = p["title"]
        for ln in p["lines"]:
            if ln["code"] == "S" and ln["shares"]:
                sold_sh[owner] += ln["shares"]
                sold_val[owner] += ln["shares"] * (ln["price"] or 0.0)
            if ln["owned_after"] is not None:
                key = (owner, ln["d_or_i"])
                d = ln["date"] or p["filing_date"]
                prev_d, _ = owned[key]
                if prev_d is None or (d and d >= prev_d):
                    owned[key] = (d, ln["owned_after"])

    # sum latest owned_after across direct+indirect buckets per owner
    remain_sh = defaultdict(float)
    as_of = {}
    for (owner, _di), (d, sh) in owned.items():
        remain_sh[owner] += sh or 0.0
        if d and (owner not in as_of or d >= as_of[owner]):
            as_of[owner] = d

    rows = []
    for owner in sorted(set(list(sold_sh) + list(remain_sh)),
                        key=lambda o: -(remain_sh.get(o, 0) * price)):
        rows.append({
            "insider": owner,
            "is_10pct": is10.get(owner, False),
            "title": title.get(owner, ""),
            "sold_shares": sold_sh.get(owner, 0.0),
            "sold_usd": sold_val.get(owner, 0.0),
            "owned_after_shares": remain_sh.get(owner, 0.0),
            "remaining_usd": remain_sh.get(owner, 0.0) * price,
            "as_of": as_of.get(owner, ""),
        })
    return rows


def print_report(rows, price, monitor_sold=None):
    tot_sold = sum(r["sold_usd"] for r in rows)
    tot_remain = sum(r["remaining_usd"] for r in rows)
    print(f"\n{'insider':32} {'10%?':5} {'sold $M':>9} {'owned_after':>13} "
          f"{'remaining $M':>13} {'as_of':>11}")
    print("-" * 92)
    for r in rows:
        print(f"{r['insider'][:32]:32} {'YES' if r['is_10pct'] else '':5} "
              f"{r['sold_usd']/1e6:9.1f} {r['owned_after_shares']:13,.0f} "
              f"{r['remaining_usd']/1e6:13.1f} {r['as_of']:>11}")
    print("-" * 92)
    print(f"{'TOTAL':32} {'':5} {tot_sold/1e6:9.1f} {'':13} {tot_remain/1e6:13.1f}")
    print(f"\n# sold (window, executed): ${tot_sold/1e6:,.1f}M")
    if monitor_sold:
        print(f"# monitor window figure:   ${monitor_sold/1e6:,.1f}M  "
              f"(diff = window mismatch; both are EXECUTED)")
    print(f"# REMAINING overhang (owned_after x ${price}): ${tot_remain/1e6:,.1f}M")
    if any(r["is_10pct"] for r in rows):
        print("# WARNING: 10%+ owner present -> their owned_after may be direct/per-account only, "
              "NOT total beneficial. Cross-check 13D/G for the real remaining stake.")
    print("# NOTE: owned_after excludes unvested RSUs/unexercised options. Form 144 "
          "noticed-but-unsold is a separate forward number (not pulled here).")


# ----------------------------------------------------------------------------- #
# offline self-test (no network) -- verifies the parser against real schema
# ----------------------------------------------------------------------------- #
SAMPLE = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerCik>0001769628</issuerCik><issuerName>CoreWeave, Inc.</issuerName>
    <issuerTradingSymbol>CRWV</issuerTradingSymbol></issuer>
  <reportingOwner><reportingOwnerId><rptOwnerCik>0001</rptOwnerCik>
    <rptOwnerName>Intrator Michael N</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>1</isOfficer>
    <isTenPercentOwner>0</isTenPercentOwner><officerTitle>CEO and President</officerTitle>
    </reportingOwnerRelationship></reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Class A Common Stock</value></securityTitle>
      <transactionDate><value>2026-08-25</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType>
        <transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>134067</value></transactionShares>
        <transactionPricePerShare><value>88.31</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>1250000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def self_test():
    p = parse_form4(SAMPLE)
    assert p["issuer"]["symbol"] == "CRWV", p["issuer"]
    assert p["owner_name"] == "Intrator Michael N"
    assert p["is_10pct"] is False
    ln = p["lines"][0]
    assert ln["code"] == "S" and ln["shares"] == 134067.0 and ln["price"] == 88.31
    assert ln["owned_after"] == 1250000.0, ln
    rows = summarize([{**p, "filing_date": "2026-08-25"}], price=84.26)
    r = rows[0]
    assert abs(r["sold_usd"] - 134067 * 88.31) < 1, r
    assert r["owned_after_shares"] == 1250000.0
    assert abs(r["remaining_usd"] - 1250000 * 84.26) < 1, r
    print("SELF-TEST PASSED: parser extracts sold + sharesOwnedFollowingTransaction correctly.")
    print_report(rows, 84.26, monitor_sold=1_206_059_575)


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Insider remaining-overhang from Form-4 owned-after.")
    ap.add_argument("--self-test", action="store_true", help="offline parser check, no network")
    ap.add_argument("--ticker")
    ap.add_argument("--cik")
    ap.add_argument("--price", type=float)
    ap.add_argument("--since", default="2026-06-01")
    ap.add_argument("--ua", help="EDGAR User-Agent, e.g. 'atom you@domain.com' (required by SEC)")
    ap.add_argument("--monitor-sold", type=float, default=None,
                    help="the monitor's window SELL $ to reconcile against")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return
    for req in ("cik", "price", "ua"):
        if not getattr(a, req):
            sys.exit(f"[finding] --{req} required (or use --self-test). "
                     "SEC requires a descriptive --ua or it returns 403.")
    rows = build(a.cik, a.ticker, a.price, a.since, a.ua)
    print_report(rows, a.price, a.monitor_sold)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"# wrote {a.out}")


if __name__ == "__main__":
    main()
