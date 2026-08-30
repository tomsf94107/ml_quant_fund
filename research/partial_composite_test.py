#!/usr/bin/env python3
"""
partial_composite_test.py — does the CONJUNCTION carry information?

RESEARCH ONLY. WRITES NOTHING. DELIBERATELY BYPASSES A SAFETY RULE.

    This script computes a composite score on an UNDER-COVERED stack, which
    warning_engine refuses to do. That refusal is correct and must stay: the
    coverage floor exists so the system never publishes a number assembled from
    layers it cannot see. Nothing here writes to composite_scores, and no output
    of this script may be quoted as a system reading.

WHY IT EXISTS ANYWAY
    D23 established that every individually testable signal fails against its
    base rate -- S5, S7, S8 indistinguishable from chance or inverted, S9's fires
    an artifact of a misspecified detrend (D16). Taken alone that looks damning.

    But the report's design claim is that NO SINGLE SIGNAL WORKS. Its thesis is
    that fragility, deterioration, rupture and propagation carry information
    JOINTLY -- that the conjunction is the signal. Weak components are consistent
    with that thesis rather than a refutation of it, and the thesis itself has
    never been tested.

    With 11 signals built the conjunction is finally computable, and it is cheap
    to check. It is worth knowing whether the composite does something its parts
    do not.

WHAT THIS CANNOT ESTABLISH
    L1 sits at 25% -- S10, S13 and S15 are unbuilt -- so the thing scored here is
    NOT the specified composite. Fragility is represented by S11 alone.

    A NEGATIVE result therefore does not condemn the design: the layer the report
    puts first is almost entirely missing.
    A POSITIVE result is encouraging but not validating: it would be a partial
    composite on a 2016-2026 sample containing no credit crisis (D17), on
    components whose thresholds were specified for crises this data cannot see.

METHOD
    For each month-end where enough signals report, compute the layer scores the
    way warning_engine does -- weighted mean of state values over non-stale
    readings -- then the weighted composite across whichever layers have any
    coverage, renormalised. Compare forward SPY returns and worst drawdowns for
    high-composite dates against the unconditional distribution, exactly as
    french_fire_value.py does for individual signals.

    The comparison is the point. A composite that beats its own base rate is
    doing something; one that matches it is not.

USAGE
    python research/partial_composite_test.py --db warning.db
"""
import argparse
import os
import sqlite3
import statistics as st
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "warning"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

FWD = 63          # ~3 months
DD_WIN = 126      # worst drawdown over the following ~6 months
STATE_VALUE = {"G": 0.0, "Y": 0.33, "O": 0.66, "R": 1.0, "B": 1.0}
LAYER_WEIGHTS = {"L1": 0.25, "L2": 0.35, "L3": 0.15, "L4": 0.25}


def month_ends(lo, hi):
    out, y, m = [], int(lo[:4]), int(lo[5:7])
    while f"{y:04d}-{m:02d}" <= hi[:7]:
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append((date(ny, nm, 1) - timedelta(days=1)).isoformat())
        y, m = ny, nm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--from", dest="frm", default="2018-01-01")
    args = ap.parse_args()

    import daily_driver as DD

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    spy = con.execute("SELECT obs_date, value FROM data_vintages "
                      "WHERE series_id='SPY_CLOSE' ORDER BY obs_date").fetchall()
    px = {d: v for d, v in spy}
    pdates = [d for d, _ in spy]

    print(__doc__.split("USAGE")[0].strip()[:0] or "", end="")
    print("PARTIAL COMPOSITE TEST -- research only, writes nothing.")
    print("L1 coverage is 25%: this is NOT the specified composite.\n")

    rows = []
    for asof in month_ends(args.frm, pdates[-1]):
        readings, details = DD.build_readings(con, asof)
        by_layer = {}
        for r in readings:
            if r.state == "NA":
                continue
            by_layer.setdefault(r.layer, []).append(STATE_VALUE.get(r.state, 0.0))
        if len(by_layer) < 2:
            continue
        num = den = 0.0
        for L, vals in by_layer.items():
            w = LAYER_WEIGHTS.get(L, 0.0)
            num += w * (sum(vals) / len(vals))
            den += w
        comp = 100.0 * num / den if den else None
        if comp is None:
            continue

        # forward outcomes from the nearest trading day at or before asof
        idx = None
        for i in range(len(pdates) - 1, -1, -1):
            if pdates[i] <= asof:
                idx = i
                break
        if idx is None or idx + DD_WIN >= len(pdates):
            continue
        p0 = px[pdates[idx]]
        fwd = (px[pdates[idx + FWD]] - p0) / p0 * 100
        worst = (min(px[d] for d in pdates[idx:idx + DD_WIN]) - p0) / p0 * 100
        rows.append((asof, comp, fwd, worst, len(by_layer)))

    con.close()
    if not rows:
        print("no scoreable month-ends")
        return

    comps = sorted(r[1] for r in rows)
    n = len(rows)
    print(f"{n} month-ends scored  {rows[0][0]}..{rows[-1][0]}")
    print(f"composite: min {comps[0]:.1f}  median {st.median(comps):.1f}  "
          f"max {comps[-1]:.1f}\n")

    def summarise(label, sub):
        if not sub:
            print(f"  {label:<28} n=0")
            return
        f = [r[2] for r in sub]
        w = [r[3] for r in sub]
        print(f"  {label:<28} n={len(sub):>3}  fwd{FWD} mean {st.mean(f):+6.2f}% "
              f"median {st.median(f):+6.2f}%   worst{DD_WIN} mean {st.mean(w):+6.2f}% "
              f"median {st.median(w):+6.2f}%")

    summarise("ALL month-ends", rows)
    cut = st.quantiles(comps, n=4)[2] if n >= 8 else comps[-1]
    summarise(f"composite in TOP quartile", [r for r in rows if r[1] >= cut])
    summarise(f"composite in BOTTOM quartile",
              [r for r in rows if r[1] <= st.quantiles(comps, n=4)[0]])

    print("\nThe composite is informative only if the top-quartile row shows a "
          "WORSE\nforward return and a DEEPER drawdown than the ALL row. "
          "Matching it means\nthe conjunction adds nothing its parts did not.")
    print("\nREMINDER: L1 is 25% covered. A negative result here does not "
          "condemn the\ndesign -- the layer the report puts first is almost "
          "entirely missing.")


if __name__ == "__main__":
    main()
