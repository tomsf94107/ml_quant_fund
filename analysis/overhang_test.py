#!/usr/bin/env python3
"""
overhang_test.py — does REMAINING insider stake predict forward returns?

READ-ONLY. Writes nothing. Third and final test of this axis.

WHAT THE FIRST TWO TESTS GOT WRONG
    analysis/insider_construction_test.py tested five FLOW constructions --
    shares sold over a window, normalised various ways. Null.
    analysis/insider_timeseries_test.py tested five TRAJECTORY constructions at
    h=5/20/60 -- persistence, slope, breadth, cumulative. Also null, though all
    five turned positive at h=60 with magnitudes ordering by horizon.

    Both measured what was ALREADY SOLD. The literature definition of overhang
    is a STOCK variable: "a large block of securities that the market knows or
    suspects will be sold in the near future" -- what is LEFT. A holder who has
    just finished distributing and one who is 10% through produce identical flow
    measures and opposite future supply.

    This tests the stock variable, using sharesOwnedFollowingTransaction
    backfilled from EDGAR into insider_holdings (10,709 rows, 15 tickers).

WHY THIS IS A WITHIN-NAME TEST, NOT CROSS-SECTIONAL
    The backfill covers 15 tickers. A cross-sectional IC needs a wide
    cross-section per date; 15 names gives roughly 10 usable observations, below
    any sensible floor. So the question asked here is the one the data can
    answer: FOR A GIVEN NAME, did periods of high remaining overhang have lower
    forward returns than that same name's periods of low overhang?

    That is a case series, not evidence for a universe-wide feature. It is
    enough to decide whether the full 134,315-accession backfill -- which at the
    observed 0.4 accessions/second would take roughly 90 hours, not the 10.7
    originally estimated -- is worth running.

MEASUREMENT CAVEATS, both confirmed against the data
    1. PER-ACCOUNT, NOT BENEFICIAL. Magnetar's 13G reports 107,962,916 shares
       held jointly across Magnetar Financial, Capital Partners, Supernova and
       Snyderman. The Form 4 maximum here is 29,545,300 -- the largest single
       account. The series tracks direction correctly but understates the level.
    2. DOUBLE-COUNTING RISK. CW OPPORTUNITY LLC shows 29,545,300 in one row,
       identical to Magnetar's maximum -- the same block reported by a second
       entity, or a related vehicle. Summing naively across insider names would
       count it twice. This script reports both the SUM and the MAX-single-holder
       so the difference is visible rather than hidden.
    3. Excludes unvested RSUs and unexercised options.

CONSTRUCTIONS
    overhang_shares   sum of the latest shares_owned_after per insider, as of
                      each date, using only filings known by then
    overhang_max      the largest single holder's remaining stake -- immune to
                      the double-counting in (2)
    overhang_adv      overhang_shares / 20-day ADV: how many sessions of normal
                      volume the remaining supply represents
    overhang_delta    change in overhang over 60 days -- is the block shrinking
                      quickly or slowly

POINT-IN-TIME
    Keyed on filing_date. A holding is only known once the filing exists.

    python analysis/overhang_test.py
"""
import argparse
import math
import random
import sqlite3
import statistics as st
from collections import defaultdict


def spearman(pairs):
    n = len(pairs)
    if n < 8:
        return None
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1
            for m in range(i, j + 1):
                r[o[m]] = a
            i = j + 1
        return r
    rx = rank([p[0] for p in pairs]); ry = rank([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    return num / (dx * dy) if dx and dy else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insider-db", default="insider_trades.db")
    ap.add_argument("--prices-db", default="prices.db")
    ap.add_argument("--min-obs", type=int, default=60,
                    help="minimum dated observations for a ticker to be tested")
    args = ap.parse_args()
    HOR = (5, 20, 60)

    ic = sqlite3.connect(f"file:{args.insider_db}?mode=ro", uri=True)
    rows = ic.execute("""
        SELECT ticker, filing_date, insider_norm, shares_owned_after, is_ten_pct
        FROM insider_holdings
        WHERE shares_owned_after IS NOT NULL AND insider_norm IS NOT NULL
        ORDER BY ticker, filing_date, seq
    """).fetchall()
    ic.close()
    print(f"{len(rows):,} holding records, "
          f"{len({r[0] for r in rows})} tickers\n")

    # latest known holding per (ticker, insider) as of each filing_date
    by_tk = defaultdict(lambda: defaultdict(dict))   # tk -> date -> insider -> shares
    latest = defaultdict(dict)
    for tk, fd, ins, sh, ten in rows:
        fd = str(fd)[:10]
        latest[tk][ins] = sh
        by_tk[tk][fd] = dict(latest[tk])

    px = sqlite3.connect(f"file:{args.prices_db}?mode=ro", uri=True)
    close = defaultdict(dict)
    volume = defaultdict(dict)
    for t, d, c, v in px.execute("SELECT ticker, d, close, volume FROM raw_bars "
                                 "WHERE d >= '2021-01-01'"):
        if c:
            close[t][d] = c
        if v:
            volume[t][d] = v
    px.close()

    print(f"  {'ticker':<7}{'obs':>5}{'h':>4}{'IC':>9}{'null':>9}"
          f"{'hi-overhang ret':>17}{'lo-overhang ret':>17}")
    agg = defaultdict(list)
    rnd = random.Random(3)

    for tk in sorted(by_tk):
        s = close.get(tk, {})
        if not s:
            continue
        ds = sorted(s)
        adv = {}
        vs = volume.get(tk, {})
        vd = sorted(vs)
        for i in range(20, len(vd)):
            w = [vs[vd[j]] for j in range(i - 20, i)]
            m = sum(w) / len(w)
            if m > 0:
                adv[vd[i]] = m

        fdates = sorted(by_tk[tk])
        # forward-fill the holding snapshot onto every trading day
        snap = {}
        j = 0
        cur = None
        for d in ds:
            while j < len(fdates) and fdates[j] <= d:
                cur = by_tk[tk][fdates[j]]
                j += 1
            if cur:
                snap[d] = cur

        series = {}
        for d in sorted(snap):
            hold = snap[d]
            tot = sum(v for v in hold.values() if v)
            mx = max((v for v in hold.values() if v), default=0.0)
            series[d] = (tot, mx, adv.get(d))

        for h in HOR:
            obs = {"overhang_shares": [], "overhang_max": [],
                   "overhang_adv": [], "overhang_delta": []}
            sd = sorted(series)
            for i, d in enumerate(sd):
                k = ds.index(d) if d in ds else None
                if k is None or k + h >= len(ds):
                    continue
                a, b = s[ds[k]], s[ds[k + h]]
                if not a or not b or abs((b - a) / a) > 0.8:
                    continue
                r = (b - a) / a
                tot, mx, av = series[d]
                obs["overhang_shares"].append((-tot, r))
                obs["overhang_max"].append((-mx, r))
                if av:
                    obs["overhang_adv"].append((-tot / av, r))
                if i >= 60:
                    prev = series[sd[i - 60]][0]
                    obs["overhang_delta"].append((-(tot - prev), r))

            for k2, v in obs.items():
                if len(v) < args.min_obs:
                    continue
                r_ = spearman(v)
                ys = [x[1] for x in v]
                rnd.shuffle(ys)
                rn = spearman([(v[i][0], ys[i]) for i in range(len(ys))])
                if r_ is None:
                    continue
                sv = sorted(v)
                q = max(1, len(sv) // 4)
                # sorted ascending on the NEGATED value, so the FIRST quartile
                # is the HIGHEST overhang
                hi = st.mean([x[1] for x in sv[:q]])
                lo = st.mean([x[1] for x in sv[-q:]])
                agg[(k2, h)].append(r_)
                if k2 == "overhang_adv":
                    print(f"  {tk:<7}{len(v):>5}{h:>4}{r_:>+9.3f}"
                          f"{(rn or 0):>+9.3f}{100*hi:>16.2f}%{100*lo:>16.2f}%")

    print(f"\n  (rows above are overhang_adv only, one line per ticker-horizon)")
    print(f"\n  POOLED across tickers:")
    print(f"  {'construction':<18}{'h':>4}{'tickers':>9}{'mean IC':>10}"
          f"{'median':>9}{'>0':>5}")
    for k in ("overhang_shares", "overhang_max", "overhang_adv",
              "overhang_delta"):
        for h in HOR:
            v = agg.get((k, h), [])
            if len(v) < 3:
                print(f"  {k:<18}{h:>4}{len(v):>9}   too few tickers")
                continue
            pos = sum(1 for x in v if x > 0)
            print(f"  {k:<18}{h:>4}{len(v):>9}{st.mean(v):>+10.3f}"
                  f"{st.median(v):>+9.3f}{pos:>3}/{len(v)}")
        print()

    print("  Sign: negated so MORE remaining overhang is MORE NEGATIVE. A "
          "POSITIVE IC\n  means high remaining overhang predicts LOWER forward "
          "returns.\n")
    print("  'hi-overhang ret' is the mean forward return in the quartile of "
          "dates with\n  the LARGEST remaining overhang; 'lo' the smallest. A "
          "real effect shows hi\n  materially below lo, consistently across "
          "tickers.\n")
    print("  This is a WITHIN-NAME case series on 15 tickers. Consistency "
          "across names\n  is the only evidence available -- there is no "
          "cross-sectional test here, and\n  a single ticker's IC means "
          "nothing on its own.")


if __name__ == "__main__":
    main()
