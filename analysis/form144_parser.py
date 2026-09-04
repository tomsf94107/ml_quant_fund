#!/usr/bin/env python3
"""
form144_parser.py — parse SEC Form 144 into noticed intent + trailing sales.

WHAT FORM 144 IS, AND WHY IT MATTERS
    Form 4 reports what an insider ALREADY SOLD. Form 144 is filed BEFORE
    selling: it notices the shares and approximate value the filer intends to
    sell within roughly the next 90 days. Noticed minus executed is forward
    overhang -- the supply of stock a known seller has told the market it plans
    to deliver.

    That is a signal class no price-based model can see. CRWV is the case in
    point: the direction model scored 9.1% accuracy on 11 high-confidence h=5
    predictions -- confidently long -- while insiders executed $1.2B of sales
    over 90 days with zero buys and the stock fell from ~$106 to ~$84.

SCHEMA -- VERIFIED, NOT ASSUMED
    Confirmed 2026-09-03 against accession 0001950047-26-008947 (CoreWeave,
    filed 2026-09-01). Namespace http://www.sec.gov/edgar/ownership.

    Two of the field names carried in the prior handoff memo were WRONG:

        memo guessed                      actual
        amountOfSecuritiesToBeSold   ->   noOfUnitsSold
        approximateSaleDate          ->   approxSaleDate
        aggregateMarketValue         ->   aggregateMarketValue   (correct)

    That is why the memo said to verify against a real filing first. Every tag
    below was read out of a real document; none is inferred.

    THE NOTICE (one per filing, under securitiesInformation):
        noOfUnitsSold            shares noticed for sale        e.g. 200000
        aggregateMarketValue     approximate value              e.g. 16978000.00
        approxSaleDate           MM/DD/YYYY                     e.g. 09/01/2026
        noOfUnitsOutstanding     shares outstanding             e.g. 458871690
        securitiesExchangeName   venue

    THE FILER (under issuerInfo):
        nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold
        relationshipToIssuer     REPEATING -- one filer can be Director,
                                 Officer AND 10% Stockholder simultaneously
        issuerCik, issuerName

    TRAILING EXECUTIONS (repeating securitiesSoldInPast3Months -- 29 blocks in
    the verified sample):
        sellerDetails/name       e.g. "10b5-1 Sales for MICHAEL INTRATOR",
                                 "OMNADORA CAPITAL LLC"
        saleDate                 MM/DD/YYYY
        amountOfSecuritiesSold
        grossProceeds
        nothingToReportFlagOnSecuritiesSoldInPast3Months   'Y' when empty

HONEST LIMITS -- these bound what the output can be used to claim
    1. Form 144 notices INTENT for roughly one 90-day window and is re-filed on
       a rolling basis. It is NOT a lifetime cap. Summing rolling notices gives
       intent-to-date, never total remaining stake.
    2. The trailing-sales block covers the past three months only, and is
       self-reported by the filer.
    3. Sales appear under several seller names for one beneficial owner -- the
       verified sample shows MICHAEL INTRATOR, "10b5-1 Sales for MICHAEL
       INTRATOR" and OMNADORA CAPITAL LLC at the same address. Naive grouping
       by name will split one person into three. Names are normalised here, but
       entity attribution remains a judgement call and is reported raw
       alongside.
    4. A 10%-plus holder's per-account filings may understate total beneficial
       ownership. 13D/G is the cross-check and this parser does not do it.

USAGE
    python analysis/form144_parser.py --file /tmp/f144.xml
    python analysis/form144_parser.py --cik 1769628 --ua "name email@domain" --limit 10
    python analysis/form144_parser.py --cik 1769628 --ua "..." --limit 20 --csv out.csv
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict

NS = re.compile(r"</?(?:\w+:)?")


def _text(block, tag):
    m = re.search(rf"<(?:\w+:)?{tag}>(.*?)</(?:\w+:)?{tag}>", block, re.S)
    return m.group(1).strip() if m else None


def _all(block, tag):
    return re.findall(rf"<(?:\w+:)?{tag}>(.*?)</(?:\w+:)?{tag}>", block, re.S)


def _num(s):
    if s is None:
        return None
    s = re.sub(r"[,$\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def normalise_seller(name):
    """Strip the 10b5-1 wrapper so one person is not counted as several.

    The verified sample carries 'MICHAEL INTRATOR', '10b5-1 Sales for MICHAEL
    INTRATOR' and 'OMNADORA CAPITAL LLC' -- the first two are the same person.
    OMNADORA is a distinct entity at the same address and is NOT merged: that
    would be an attribution guess, and the raw name is preserved so a human can
    make the call.
    """
    if not name:
        return None
    n = re.sub(r"^\s*10b5-1\s+Sales?\s+for\s+", "", name, flags=re.I)
    return " ".join(n.split()).upper()


def parse(xml):
    """-> dict with the notice, the filer, and the trailing sales."""
    out = {}
    out["issuer_cik"] = _text(xml, "issuerCik")
    out["issuer_name"] = _text(xml, "issuerName")
    out["filer"] = _text(xml,
                         "nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold")
    out["relationships"] = _all(xml, "relationshipToIssuer")

    # NAMESPACE PREFIXES. Two generators produce Form 144 and they differ:
    #   Direct EDGAR:  <securitiesInformation>      (default namespace)
    #   Workiva:       <own:securitiesInformation>  (prefixed)
    # The _text/_all helpers already allowed an optional prefix, but these two
    # BLOCK searches did not -- so every Workiva filing parsed as "NOTICED n/a"
    # while looking like a successful parse. Found 2026-09-03 on two McVeety
    # filings; a plausible value from a failed parse, which is the failure mode
    # this codebase keeps producing.
    info = re.search(
        r"<(?:\w+:)?securitiesInformation>(.*?)</(?:\w+:)?securitiesInformation>",
        xml, re.S)
    blk = info.group(1) if info else ""
    out["noticed_shares"] = _num(_text(blk, "noOfUnitsSold"))
    out["noticed_value"] = _num(_text(blk, "aggregateMarketValue"))
    out["approx_sale_date"] = _text(blk, "approxSaleDate")
    out["shares_outstanding"] = _num(_text(blk, "noOfUnitsOutstanding"))
    out["exchange"] = _text(blk, "securitiesExchangeName")
    out["broker"] = _text(blk, "name")

    out["nothing_to_report"] = _text(
        xml, "nothingToReportFlagOnSecuritiesSoldInPast3Months")

    sales = []
    for b in re.findall(
            r"<(?:\w+:)?securitiesSoldInPast3Months>(.*?)"
            r"</(?:\w+:)?securitiesSoldInPast3Months>", xml, re.S):
        seller = _text(b, "name")
        sales.append({
            "seller_raw": seller,
            "seller": normalise_seller(seller),
            "sale_date": _text(b, "saleDate"),
            "shares": _num(_text(b, "amountOfSecuritiesSold")),
            "proceeds": _num(_text(b, "grossProceeds")),
        })
    out["past_3m_sales"] = sales
    return out


def fetch_filings(cik, ua, limit, timeout=45):
    cik10 = str(int(cik)).zfill(10)
    req = urllib.request.Request(
        f"https://data.sec.gov/submissions/CIK{cik10}.json",
        headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    rec = d["filings"]["recent"]
    out = []
    for i, form in enumerate(rec["form"]):
        if form != "144":
            continue
        out.append((rec["filingDate"][i], rec["accessionNumber"][i]))
        if len(out) >= limit:
            break
    return out


def fetch_doc(cik, accession, ua, timeout=45):
    acc = accession.replace("-", "")
    url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/"
           f"primary_doc.xml")
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def report(parsed, filing_date=None, accession=None):
    p = parsed
    hdr = f"{p.get('issuer_name')} ({p.get('issuer_cik')})"
    if filing_date:
        hdr = f"{filing_date}  {hdr}"
    print(f"\n{hdr}")
    if accession:
        print(f"  accession {accession}")
    print(f"  filer        {p.get('filer')}  "
          f"[{', '.join(p.get('relationships') or []) or 'n/a'}]")
    sh = p.get("noticed_shares")
    val = p.get("noticed_value")
    out = p.get("shares_outstanding")
    pct = f"  ({100*sh/out:.3f}% of shares out)" if sh and out else ""
    print(f"  NOTICED      {sh:,.0f} shares" if sh else "  NOTICED      n/a", end="")
    print(f"  ${val:,.0f}{pct}" if val else pct)
    print(f"  approx sale  {p.get('approx_sale_date')}   broker "
          f"{(p.get('broker') or '')[:44]}")

    sales = p.get("past_3m_sales") or []
    if p.get("nothing_to_report") == "Y" or not sales:
        print("  past 3 months: nothing reported")
        return
    tot_sh = sum(s["shares"] or 0 for s in sales)
    tot_pr = sum(s["proceeds"] or 0 for s in sales)
    print(f"  past 3 months: {len(sales)} sales, {tot_sh:,.0f} shares, "
          f"${tot_pr:,.0f}")
    byseller = defaultdict(lambda: [0, 0.0, 0])
    for s in sales:
        k = s["seller"] or "?"
        byseller[k][0] += s["shares"] or 0
        byseller[k][1] += s["proceeds"] or 0.0
        byseller[k][2] += 1
    print(f"    {'seller':<34}{'sales':>7}{'shares':>12}{'proceeds':>16}")
    for k, (sh_, pr_, n_) in sorted(byseller.items(), key=lambda x: -x[1][1]):
        print(f"    {k[:34]:<34}{n_:>7}{sh_:>12,.0f}{pr_:>16,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="parse a local primary_doc.xml")
    ap.add_argument("--cik")
    ap.add_argument("--ua", help="EDGAR requires a descriptive User-Agent "
                                 "with a real email or it returns 403")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--csv")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rows = []
    if args.file:
        p = parse(open(args.file, encoding="utf-8", errors="replace").read())
        report(p, accession=args.file)
        rows.append((None, args.file, p))
    elif args.cik:
        if not args.ua:
            raise SystemExit("--ua required: EDGAR 403s without a descriptive "
                             "User-Agent containing a real email")
        filings = fetch_filings(args.cik, args.ua, args.limit)
        print(f"{len(filings)} Form 144 filings for CIK {args.cik}")
        for date_, acc in filings:
            try:
                xml = fetch_doc(args.cik, acc, args.ua)
                p = parse(xml)
                report(p, date_, acc)
                rows.append((date_, acc, p))
            except Exception as e:
                print(f"\n  {date_} {acc}: FAILED {type(e).__name__}: {e}")
            time.sleep(0.15)          # SEC asks for <10 requests/second
    else:
        raise SystemExit("give --file or --cik")

    if rows:
        tot = sum((p.get("noticed_shares") or 0) for _, _, p in rows)
        totv = sum((p.get("noticed_value") or 0) for _, _, p in rows)
        print(f"\n{'='*66}")
        print(f"  {len(rows)} filings, {tot:,.0f} shares noticed, "
              f"${totv:,.0f}")
        print("  This is INTENT ACROSS ROLLING 90-DAY WINDOWS, not a lifetime\n"
              "  cap and not remaining stake. Overlapping notices from one\n"
              "  filer may re-notice unsold shares. Cross-check 13D/G for any\n"
              "  10%+ holder before treating this as total overhang.")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filing_date", "accession", "issuer", "filer",
                        "relationships", "noticed_shares", "noticed_value",
                        "approx_sale_date", "shares_outstanding",
                        "past3m_n", "past3m_shares", "past3m_proceeds"])
            for d, a, p in rows:
                s = p.get("past_3m_sales") or []
                w.writerow([d, a, p.get("issuer_name"), p.get("filer"),
                            "|".join(p.get("relationships") or []),
                            p.get("noticed_shares"), p.get("noticed_value"),
                            p.get("approx_sale_date"),
                            p.get("shares_outstanding"), len(s),
                            sum(x["shares"] or 0 for x in s),
                            sum(x["proceeds"] or 0 for x in s)])
        print(f"\nwrote {args.csv}")


SAMPLE = """<?xml version="1.0" encoding="UTF-8"?><edgarSubmission xmlns="http://www.sec.gov/edgar/ownership" xmlns:ns2="http://www.sec.gov/edgar/common">
<headerData><submissionType>144</submissionType></headerData>
<formData><issuerInfo><issuerCik>0001769628</issuerCik>
<issuerName>CoreWeave, Inc.</issuerName>
<nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>MICHAEL INTRATOR</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>
<relationshipsToIssuer><relationshipToIssuer>Director</relationshipToIssuer>
<relationshipToIssuer>Officer</relationshipToIssuer>
<relationshipToIssuer>10% Stockholder</relationshipToIssuer></relationshipsToIssuer>
</issuerInfo>
<securitiesInformation><securitiesClassTitle>Common</securitiesClassTitle>
<brokerOrMarketmakerDetails><name>Morgan Stanley Smith Barney LLC</name></brokerOrMarketmakerDetails>
<noOfUnitsSold>200000</noOfUnitsSold>
<aggregateMarketValue>16978000.00</aggregateMarketValue>
<noOfUnitsOutstanding>458871690</noOfUnitsOutstanding>
<approxSaleDate>09/01/2026</approxSaleDate>
<securitiesExchangeName>NASDAQ</securitiesExchangeName></securitiesInformation>
<nothingToReportFlagOnSecuritiesSoldInPast3Months>N</nothingToReportFlagOnSecuritiesSoldInPast3Months>
<securitiesSoldInPast3Months><sellerDetails><name>10b5-1 Sales for MICHAEL INTRATOR</name></sellerDetails>
<saleDate>08/25/2026</saleDate><amountOfSecuritiesSold>200000</amountOfSecuritiesSold>
<grossProceeds>17720940.00</grossProceeds></securitiesSoldInPast3Months>
<securitiesSoldInPast3Months><sellerDetails><name>OMNADORA CAPITAL LLC</name></sellerDetails>
<saleDate>08/25/2026</saleDate><amountOfSecuritiesSold>107692</amountOfSecuritiesSold>
<grossProceeds>9542017.35</grossProceeds></securitiesSoldInPast3Months>
<securitiesSoldInPast3Months><sellerDetails><name>MICHAEL INTRATOR</name></sellerDetails>
<saleDate>08/20/2026</saleDate><amountOfSecuritiesSold>13129</amountOfSecuritiesSold>
<grossProceeds>1206292.52</grossProceeds></securitiesSoldInPast3Months>
</formData></edgarSubmission>"""


WORKIVA_SAMPLE = SAMPLE.replace("<edgarSubmission", "<own:edgarSubmission")
for _t in ("headerData", "submissionType", "formData", "issuerInfo",
           "issuerCik", "issuerName",
           "nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold",
           "relationshipsToIssuer", "relationshipToIssuer",
           "securitiesInformation", "securitiesClassTitle",
           "brokerOrMarketmakerDetails", "name", "noOfUnitsSold",
           "aggregateMarketValue", "noOfUnitsOutstanding", "approxSaleDate",
           "securitiesExchangeName",
           "nothingToReportFlagOnSecuritiesSoldInPast3Months",
           "securitiesSoldInPast3Months", "sellerDetails", "saleDate",
           "amountOfSecuritiesSold", "grossProceeds", "edgarSubmission"):
    WORKIVA_SAMPLE = (WORKIVA_SAMPLE.replace(f"<{_t}>", f"<own:{_t}>")
                                    .replace(f"</{_t}>", f"</own:{_t}>"))


def self_test():
    """Offline test against BOTH generators' schemas. No network."""
    p = parse(SAMPLE)
    checks = [
        ("issuer cik", p["issuer_cik"], "0001769628"),
        ("issuer name", p["issuer_name"], "CoreWeave, Inc."),
        ("filer", p["filer"], "MICHAEL INTRATOR"),
        ("relationships", len(p["relationships"]), 3),
        ("noticed shares", p["noticed_shares"], 200000.0),
        ("noticed value", p["noticed_value"], 16978000.0),
        ("approx sale date", p["approx_sale_date"], "09/01/2026"),
        ("shares outstanding", p["shares_outstanding"], 458871690.0),
        ("past-3m blocks", len(p["past_3m_sales"]), 3),
        ("past-3m shares", sum(s["shares"] for s in p["past_3m_sales"]),
         320821.0),
    ]
    bad = 0
    for label, got, want in checks:
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'OK ' if ok else 'FAIL'}  {label:<22} got {got!r}"
              + ("" if ok else f"  want {want!r}"))

    # the 10b5-1 wrapper must collapse to one person, and OMNADORA must NOT
    # be merged into it -- that would be an attribution guess
    sellers = {s["seller"] for s in p["past_3m_sales"]}
    ok = sellers == {"MICHAEL INTRATOR", "OMNADORA CAPITAL LLC"}
    bad += 0 if ok else 1
    print(f"  {'OK ' if ok else 'FAIL'}  seller normalisation   {sorted(sellers)}")

    # The same document with Workiva's own: prefixes must parse identically.
    w = parse(WORKIVA_SAMPLE)
    for label, key in (("workiva noticed shares", "noticed_shares"),
                       ("workiva noticed value", "noticed_value"),
                       ("workiva approx date", "approx_sale_date"),
                       ("workiva filer", "filer")):
        ok = w.get(key) == p.get(key)
        bad += 0 if ok else 1
        print(f"  {'OK ' if ok else 'FAIL'}  {label:<22} got {w.get(key)!r}")
    ok = len(w["past_3m_sales"]) == len(p["past_3m_sales"])
    bad += 0 if ok else 1
    print(f"  {'OK ' if ok else 'FAIL'}  workiva past-3m blocks got "
          f"{len(w['past_3m_sales'])}")

    total = len(checks) + 6
    print(f"\n  {total-bad}/{total} passed")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
