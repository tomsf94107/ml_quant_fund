#!/usr/bin/env python3
"""
insider_144_overhang.py — "how many shares have insiders filed INTENT to sell?"

Form 4  = executed (rear-view).  Form 144 = notice of PROPOSED sale, filed BEFORE
selling: shares + approx market value the insider intends to sell (~90d window),
PLUS a "securities sold in past 3 months" table on the same doc. That intended
number is what `90d SELL $1.2B` cannot give you.

Overhang ≈  Σ(Form-144 shares-to-be-sold, open notices)  −  executed since notice.
Ceiling  =  Form-4 sharesOwnedFollowingTransaction (use insider_overhang.py).
For a 10%+ FUND (Magnetar on CRWV) the 144/Form-4 figure is per-account and
UNDERCOUNTS total beneficial — cross-check 13D/G. Flagged in output.

I do NOT hardcode Form-144 tag names (I'm not certain of them and won't fake it).
The parser is namespace-agnostic and matches fields by local-name HINTS, printing
which tags it used. Run --inspect FIRST to see the real schema and lock the hints.

USAGE (on a box that can reach sec.gov; needs a descriptive --ua email):
  python insider_144_overhang.py --inspect  --cik 1769628 --ua "atom you@dom.com"
  python insider_144_overhang.py --overhang --cik 1769628 --since 2026-06-01 --ua "atom you@dom.com"
  python insider_144_overhang.py --self-test        # offline parser check, no network
"""
import argparse, json, sys, time
import xml.etree.ElementTree as ET
from collections import defaultdict

import requests

SUBM = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCH = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
IDX  = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json"

# local-name substring hints. VERIFY against --inspect output and edit if needed.  # CONFIRM
HINTS = {  # SCHEMA_LOCKED to real Form-144 tags (verified CRWV 0001950047-26-008947)
    "shares_to_sell": ["noofunitssold"],                 # under securitiesinformation = PROPOSED
    "market_value":   ["aggregatemarketvalue"],
    "sale_date":      ["approxsaledate"],
    "class_title":    ["securitiesclasstitle"],
    "issuer_name":    ["issuername"],
    "person_name":    ["nameofpersonforwhoseaccount"],
    "relationship":   ["relationshiptoissuer"],
    "past3m_amount":  ["amountofsecuritiessold"],         # itemized EXECUTED ledger
    "past3m_proceeds":["grossproceeds"],
    "plan_date":      ["planadoptiondate"],
}


def local(tag):
    return tag.split("}")[-1].lower()


def num(s):
    if s is None:
        return None
    t = str(s).replace(",", "").replace("$", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def edgar_get(url, ua):
    r = requests.get(url, headers={"User-Agent": ua,
                                   "Accept-Encoding": "gzip, deflate"}, timeout=30)
    r.raise_for_status()
    return r


def list_144(cik, since, ua):
    j = edgar_get(SUBM.format(cik=int(cik)), ua).json()
    rec = j["filings"]["recent"]
    out = []
    for form, acc, doc, fdate in zip(rec["form"], rec["accessionNumber"],
                                     rec["primaryDocument"], rec["filingDate"]):
        if form == "144" and fdate >= since:
            out.append((acc, doc, fdate))
    return out


def fetch_xml(cik, acc, doc, ua):
    """Fetch the RAW primary_doc.xml. EDGAR's primaryDocument for 144s points at
    the xsl-RENDERED HTML (xslF144X01/primary_doc.xml), which is not parseable as
    XML -- so always resolve the raw file from the filing index."""
    accn = acc.replace("-", "")
    idx = edgar_get(IDX.format(cik=int(cik), acc=accn), ua).json()
    items = idx.get("directory", {}).get("item", [])
    for it in items:
        if it.get("name", "").lower() == "primary_doc.xml":
            return edgar_get(ARCH.format(cik=int(cik), acc=accn, doc=it["name"]), ua).text
    for it in items:
        n = it.get("name", "").lower()
        if n.endswith(".xml") and "xsl" not in n:
            return edgar_get(ARCH.format(cik=int(cik), acc=accn, doc=it["name"]), ua).text
    raise RuntimeError(f"no raw xml in {acc}; files={[i.get('name') for i in items]}")


def inspect(xml):
    """Print the real element tree (local tag + value) so we lock the true schema."""
    root = ET.fromstring(xml)
    def walk(el, d=0):
        t = local(el.tag)
        v = (el.text or "").strip()
        v = (v[:60] + "…") if len(v) > 60 else v
        print(f"{'  '*d}{t}{('  = '+v) if v else ''}")
        for c in el:
            walk(c, d + 1)
    walk(root)


def parse_144(xml):
    """Namespace-agnostic. Returns dict of matched fields + which tags matched."""
    root = ET.fromstring(xml)
    hit = {k: [] for k in HINTS}
    used = defaultdict(set)
    for el in root.iter():
        ln = local(el.tag)
        txt = (el.text or "").strip()
        for field, subs in HINTS.items():
            if any(s in ln for s in subs):
                hit[field].append(txt)
                used[field].add(ln)
    # take first non-empty for scalars; sum numeric for shares/value
    def first(field):
        return next((x for x in hit[field] if x), None)
    shares = sum(v for v in (num(x) for x in hit["shares_to_sell"]) if v)
    value  = sum(v for v in (num(x) for x in hit["market_value"]) if v)
    return {
        "issuer": first("issuer_name"),
        "person": first("person_name"),
        "class": first("class_title"),
        "sale_date": first("sale_date"),
        "shares_to_sell": shares or None,
        "market_value": value or None,
        "has_past3m": bool(hit.get("past3m_amount")),  # PAST3M_FIX
        "_used_tags": {k: sorted(v) for k, v in used.items()},
    }


def overhang(cik, since, ua, sleep=0.15):
    fs = list_144(cik, since, ua)
    if not fs:
        sys.exit(f"[finding] no Form 144 filings for CIK {cik} since {since}")
    rows, tag_report = [], defaultdict(set)
    for acc, doc, fdate in fs:
        try:
            p = parse_144(fetch_xml(cik, acc, doc, ua))
            p.update(accession=acc, filing_date=fdate)
            rows.append(p)
            for k, v in p["_used_tags"].items():
                tag_report[k].update(v)
        except Exception as e:
            print(f"  [warn] {acc}: {e}", file=sys.stderr)
        time.sleep(sleep)
    return rows, tag_report


def report(rows, tag_report):
    tot_sh = sum(r["shares_to_sell"] or 0 for r in rows)
    tot_v  = sum(r["market_value"] or 0 for r in rows)
    print(f"\n{'filing':12} {'date':11} {'person':26} {'shares→sell':>13} {'$mkt (M)':>10}")
    print("-" * 78)
    for r in sorted(rows, key=lambda x: -(x["market_value"] or 0)):
        print(f"{r['accession'][:12]:12} {r['filing_date']:11} "
              f"{(r['person'] or '?')[:26]:26} "
              f"{(r['shares_to_sell'] or 0):13,.0f} {(r['market_value'] or 0)/1e6:10.1f}")
    print("-" * 78)
    print(f"{'TOTAL noticed intent':50} {tot_sh:13,.0f} {tot_v/1e6:10.1f}")
    print(f"\n# INTENDED overhang (open Form-144 notices, {len(rows)} filings): "
          f"{tot_sh:,.0f} shares / ${tot_v/1e6:,.1f}M")
    print("# subtract Form-4 executed-since-notice for the NET remaining intent.")
    print("# ceiling on total sellable = Form-4 sharesOwnedFollowingTransaction "
          "(run insider_overhang.py).")
    print("# 10%+ FUND (e.g. Magnetar): its 144 is per-account, UNDERCOUNTS beneficial "
          "-> cross-check 13D/G.")
    # surface which tags actually matched, so we can confirm the schema
    print("\n# TAGS MATCHED (verify vs --inspect; edit HINTS if a field is empty):")
    for k in HINTS:
        got = sorted(tag_report.get(k, []))
        flag = "" if got else "   <-- NOTHING MATCHED, fix HINTS"
        print(f"#   {k:16} {got}{flag}")


# --------------------------------------------------------------------------- #
# offline self-test: verifies namespace-stripping + numeric sum + hint match
# (SAMPLE tags are a PLAUSIBLE guess for testing MECHANICS, not authoritative.)
SAMPLE = """<?xml version="1.0"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/form144">
  <formData>
    <issuerInfo><issuerName>CoreWeave, Inc.</issuerName></issuerInfo>
    <securitiesToBeSoldInfo>
      <securitiesClassTitle>Class A Common Stock</securitiesClassTitle>
      <numberOfUnitsToBeSold>250000</numberOfUnitsToBeSold>
      <aggregateMarketValue>21000000</aggregateMarketValue>
      <approxSaleDate>2026-09-02</approxSaleDate>
    </securitiesToBeSoldInfo>
    <personForWhoseAccountInfo><personName>Intrator Michael N</personName></personForWhoseAccountInfo>
    <securitiesSoldInPast3MonthsInfo><noOfUnitsSold>100000</noOfUnitsSold></securitiesSoldInPast3MonthsInfo>
  </formData>
</edgarSubmission>"""


def self_test():
    p = parse_144(SAMPLE)
    assert p["issuer"] == "CoreWeave, Inc.", p
    assert p["person"] == "Intrator Michael N", p
    assert p["shares_to_sell"] == 250000.0, p
    assert p["market_value"] == 21000000.0, p
    assert p["has_past3m"] is True, p
    print("SELF-TEST PASSED: namespace-stripped, matched shares/value/person, summed correctly.")
    report([{**p, "accession": "0001-sample", "filing_date": "2026-09-02"}],
           {k: set(v) for k, v in p["_used_tags"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="dump one 144's real schema")
    ap.add_argument("--overhang", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--cik"); ap.add_argument("--ua")
    ap.add_argument("--since", default="2026-06-01")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    if not a.cik or not a.ua:
        sys.exit("[finding] --cik and --ua required (SEC needs a descriptive --ua email).")
    if a.inspect:
        fs = list_144(a.cik, a.since, a.ua)
        if not fs:
            sys.exit("[finding] no 144s found in window")
        acc, doc, fdate = fs[0]
        print(f"# inspecting newest 144: {acc} ({fdate})\n")
        inspect(fetch_xml(a.cik, acc, doc, a.ua))
        print("\n# ^ lock the real tag names into HINTS, then run --overhang")
        return
    if a.overhang:
        rows, tags = overhang(a.cik, a.since, a.ua)
        report(rows, tags)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
