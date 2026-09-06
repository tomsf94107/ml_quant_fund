#!/usr/bin/env python3
"""
build_sector_map_sic.py — sector ETF for every ticker, from SEC SIC codes.

Writes data/sector_etf_sic.csv. Touches nothing else.

WHY
    features/builder.py resolves a ticker's sector ETF in three tiers:
    SECTOR_ETF_MAP (429 hand-curated S&P names) -> tickers_metadata.csv bucket
    -> the market ETF. On the expanded universe, 1,496 of 1,920 names have no
    metadata row and are not in the hand-curated map, so sector_rel_ret falls
    through to market-relative for 78% of the universe -- it silently duplicates
    a feature the model already has.

    That matters because the leave-one-out test on 2026-09-05 put [sector] at
    -0.515pp with both seeds agreeing: dropping the sector family costs money.
    It is currently degraded on most of the names the expanded book ranks.

    SEC's submissions endpoint carries a `sic` field for every registrant, and
    SIC maps onto the eleven SPDR sector ETFs cleanly enough for a relative-
    return baseline. This builds that third tier.

WHAT IT IS NOT
    SIC is not GICS. It is coarser, it was designed in 1937 and last revised in
    1987, and it classifies by primary product rather than by how the market
    trades a name. A software firm and a semiconductor firm both land in XLK
    here; GICS would separate them. For a sector-RELATIVE return baseline that
    is adequate -- the feature asks "did this name beat its sector", not "what
    is this company".

    The hand-curated SECTOR_ETF_MAP stays authoritative and is checked first.
    This only fills names that would otherwise get the market ETF.

    Known coarse cases, accepted deliberately:
      - conglomerates (SIC 9997, 6770 blank-check) -> XLI, a guess
      - REITs (6798) -> XLRE, correct
      - biotech (2836) and pharma (2834) both -> XLV, correct
      - foreign issuers often carry a US SIC anyway; where absent they keep the
        market fallback, which is the honest outcome

    python analysis/build_sector_map_sic.py --dry-run
    python analysis/build_sector_map_sic.py
"""
import argparse
import csv
import json
import os
import time
import urllib.request

UA = {"User-Agent": "atomnguyen research atom@ifrenzy.co"}
OUT = "data/sector_etf_sic.csv"

# SIC range -> SPDR sector ETF. Ranges follow the SEC's own division
# groupings; where a division spans two ETFs the split point is the one the
# market actually uses (e.g. 5900s retail is XLY, 5400s food retail is XLP).
RANGES = [
    (100, 999, "XLB"),      # agriculture
    (1000, 1299, "XLB"),    # metal mining
    (1300, 1399, "XLE"),    # oil and gas extraction
    (1400, 1499, "XLB"),    # nonmetallic minerals
    (1500, 1799, "XLI"),    # construction
    (2000, 2199, "XLP"),    # food, tobacco
    (2200, 2399, "XLY"),    # textiles, apparel
    (2400, 2599, "XLI"),    # lumber, furniture
    (2600, 2699, "XLB"),    # paper
    (2700, 2799, "XLC"),    # publishing
    (2800, 2829, "XLB"),    # industrial chemicals
    (2830, 2836, "XLV"),    # drugs, biologics
    (2840, 2844, "XLP"),    # soap, detergents, cosmetics
    (2850, 2899, "XLB"),    # paints, industrial organics, agri chemicals --
                            # WLK (SIC 2860, Industrial Organic Chemicals) was
                            # landing in XLP under a 2840-2899 band that was too
                            # wide. Only 2840-2844 is consumer staples.
    (2900, 2999, "XLE"),    # petroleum refining
    (3000, 3299, "XLB"),    # rubber, stone, clay
    (3300, 3399, "XLB"),    # primary metals
    (3400, 3569, "XLI"),    # fabricated metal, machinery
    (3570, 3579, "XLK"),    # computers
    (3580, 3599, "XLI"),    # industrial machinery
    (3600, 3629, "XLI"),    # electrical equipment
    (3630, 3669, "XLY"),    # household appliances, audio
    (3670, 3699, "XLK"),    # semiconductors, electronic components
    (3700, 3716, "XLY"),    # motor vehicles
    (3720, 3729, "XLI"),    # aircraft
    (3730, 3799, "XLI"),    # ships, rail, transport equipment
    (3800, 3829, "XLK"),    # instruments
    (3830, 3859, "XLV"),    # medical instruments
    (3860, 3999, "XLY"),    # photographic, toys, misc
    (4000, 4299, "XLI"),    # rail, trucking
    (4400, 4599, "XLI"),    # water, air transport
    (4600, 4699, "XLE"),    # pipelines
    (4700, 4799, "XLI"),    # transport services
    (4800, 4899, "XLC"),    # communications
    (4900, 4991, "XLU"),    # utilities
    (5000, 5199, "XLI"),    # wholesale
    (5200, 5399, "XLY"),    # building materials, general merchandise
    (5400, 5499, "XLP"),    # food stores
    (5500, 5599, "XLY"),    # auto dealers
    (5600, 5799, "XLY"),    # apparel, furniture retail
    (5800, 5899, "XLY"),    # eating places
    (5900, 5912, "XLP"),    # drug stores
    (5920, 5999, "XLY"),    # misc retail
    (6000, 6199, "XLF"),    # banks, credit
    (6200, 6299, "XLF"),    # brokers
    (6300, 6499, "XLF"),    # insurance
    (6500, 6552, "XLRE"),   # real estate
    (6600, 6799, "XLF"),    # investment offices
    (6798, 6798, "XLRE"),   # REITs -- overrides the 6600-6799 band
    (7000, 7099, "XLY"),    # hotels
    (7300, 7369, "XLI"),    # business services
    (7370, 7379, "XLK"),    # software, data processing
    (7380, 7399, "XLI"),    # misc business services
    (7500, 7999, "XLY"),    # auto services, entertainment
    (8000, 8099, "XLV"),    # health services
    (8100, 8299, "XLI"),    # legal, education
    (8300, 8399, "XLV"),    # social services
    (8600, 8699, "XLI"),    # membership organisations
    (8700, 8730, "XLI"),    # engineering, accounting
    (8731, 8734, "XLV"),    # commercial biological research
    (8740, 8999, "XLI"),    # management services
]


# SIC classifies by primary PRODUCT; GICS by how the market trades a name.
# Four divergences show up in any spot-check and are corrected by hand:
#   5411 grocery retail  -> COST, KR are staples (XLP), not discretionary
#   6324 hospital plans  -> UNH, ELV, CI are healthcare (XLV), not financials
#   7990 services        -> DIS moved to Communication Services in the 2018
#                           GICS reshuffle; SIC never followed
#   3021 rubber footwear -> NKE sells apparel (XLY), it does not sell rubber
# The list is short by design. A long override list means the ranges are wrong.
SIC_OVERRIDE = {
    5411: "XLP",   # grocery stores
    5412: "XLP",
    6324: "XLV",   # hospital and medical service plans
    3021: "XLY",   # rubber and plastics footwear
}
TICKER_OVERRIDE = {
    "DIS": "XLC",  # GICS 2018 reshuffle; SIC 7990 still says services
}


def etf_for(sic):
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return None
    if s == 6798:
        return "XLRE"
    if s in SIC_OVERRIDE:
        return SIC_OVERRIDE[s]
    for lo, hi, etf in RANGES:
        if lo <= s <= hi:
            return etf
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="tickers_expanded.txt")
    ap.add_argument("--sleep", type=float, default=0.11)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.universe):
        raise SystemExit(f"{args.universe} not found -- run from the repo root")
    want = [l.strip().upper() for l in open(args.universe) if l.strip()]

    have = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            for row in csv.DictReader(f):
                have[row["ticker"]] = row["etf"]
    todo = [t for t in want if t not in have]
    print(f"{args.universe}: {len(want):,} names, {len(have):,} already mapped, "
          f"{len(todo):,} to fetch")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        print(f"  {len(RANGES)} SIC ranges cover the eleven SPDR sector ETFs")
        for sic, lab in ((3674, "semiconductors"), (2834, "pharma"),
                         (6022, "state bank"), (6798, "REIT"),
                         (1311, "crude petroleum"), (7372, "prepackaged sw"),
                         (4911, "electric services")):
            print(f"    SIC {sic:<5} {lab:<18} -> {etf_for(sic)}")
        return

    d = json.loads(urllib.request.urlopen(urllib.request.Request(
        "https://www.sec.gov/files/company_tickers.json", headers=UA),
        timeout=60).read())
    cik = {v["ticker"].upper(): f"{v['cik_str']:010d}" for v in d.values()}

    n_ok = n_nocik = n_nosic = n_nomap = 0
    rows = dict(have)
    for i, t in enumerate(todo, 1):
        if args.limit and i > args.limit:
            break
        c = cik.get(t)
        if not c:
            n_nocik += 1
            continue
        try:
            s = json.loads(urllib.request.urlopen(urllib.request.Request(
                f"https://data.sec.gov/submissions/CIK{c}.json", headers=UA),
                timeout=30).read())
            sic = s.get("sic")
            if not sic:
                n_nosic += 1
                continue
            e = etf_for(sic)
            if not e:
                n_nomap += 1
                continue
            rows[t] = TICKER_OVERRIDE.get(t, e)
            n_ok += 1
        except Exception:
            n_nosic += 1
        time.sleep(args.sleep)
        if i % 300 == 0:
            print(f"  {i}/{len(todo)}  mapped {n_ok}")

    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "etf"])
        for t in sorted(rows):
            w.writerow([t, rows[t]])

    from collections import Counter
    dist = Counter(rows.values())
    print(f"\n  mapped {n_ok} new, {n_nocik} no CIK, {n_nosic} no SIC, "
          f"{n_nomap} SIC outside the ranges")
    print(f"  wrote {OUT} with {len(rows)} tickers\n")
    print(f"  {'ETF':<6}{'names':>7}")
    for e, n in dist.most_common():
        print(f"  {e:<6}{n:>7}")
    print("\n  NEXT: patch resolve_sector_etf to read this as a THIRD tier,")
    print("  after SECTOR_ETF_MAP and the metadata bucket, before the market")
    print("  fallback. The hand-curated map stays authoritative.")


if __name__ == "__main__":
    main()
