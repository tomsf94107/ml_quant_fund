#!/usr/bin/env python3
"""
regime_split_test.py — did the direction model improve after 2026-05-31?

READ-ONLY. Reads logged predictions and outcomes. Trains nothing.

WHAT PROMPTED THIS
    analysis/prob_threshold_honest_test.py (2026-09-07) found the prob>=0.70
    selection's day-weighted excess over the same day's universe split sharply
    on the kill-switch date:

        h=3   before 2026-05-31   -0.094pp on 32 dates
              after               +0.391pp on 64 dates
        h=5   before              -0.096pp on 33 dates
              after               +0.788pp on 62 dates

    Negative before, clearly positive after, at both horizons. That is the
    OPPOSITE of what the kill switch implies. STEP 1 was flipped on 2026-05-31
    converting every BUY to HOLD, on the stated grounds that the direction model
    was near-coin-flip and inverted at the extremes. If the model got better
    immediately afterwards, either the switch was flipped on a temporary
    problem, or something else changed at the same time.

    Neither the split nor its t-statistic has been tested. A 32-vs-64 date split
    chosen after seeing the numbers is exactly the kind of comparison that needs
    a hurdle rather than a nod.

CANDIDATE EXPLANATIONS, ALL CHECKABLE
    The codebase records several changes near that boundary:
      2026-05-07  three-way temporal split (Sprint 1 leak fix) -- the docstring
                  in models/ensemble.py names it; the leak was worth roughly
                  0.30 AUC, so predictions BEFORE it were scored by a model
                  fitted on contaminated data
      2026-05-22  yfinance revenue removed; rev_surprise dead thereafter
      2026-05-31  STEP 1 kill switch
      2026-06-28  vix_term_structure silently pinned to the literal 1.0
      2026-08-26  short_ratio PIT rewire -- scalar broadcast to per-row join
    A jump at 2026-05-07 would point at the leak fix and mean the improvement is
    real. A jump at 2026-05-31 with nothing else changing would be a coincidence
    of dates and probably regime.

WHAT IS MEASURED
    Day-weighted excess of the prob>=0.70 selection over the same day's universe,
    split at several candidate boundaries, each with a Newey-West t on the
    per-date series and the count of dates either side.

    Also a MONTHLY series, because a single split date invites the reader to see
    a step where there is a drift. If the improvement is gradual, no boundary
    explains it and the honest reading is regime, not a code change.

THE BAR
    NW t > 3.0 per Harvey, Liu & Zhu. The post-period is ~62 dates at h=5 with
    5-day overlapping windows, so the effective independent count is closer to
    12. Treat anything under t=3 as unproven, and note that this boundary was
    chosen after seeing the split.

    python analysis/regime_split_test.py
"""
import argparse
import math
import sqlite3
import statistics as st
from collections import defaultdict

BOUNDARIES = ["2026-05-07", "2026-05-31", "2026-06-28", "2026-08-26"]


def nw_t(v, lag):
    n = len(v)
    if n < 8:
        return None
    m = sum(v) / n
    d = [x - m for x in v]
    var = sum(x * x for x in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[i] * d[i - k] for i in range(k, n)) / n
        var += 2 * (1 - k / (lag + 1.0)) * gk
    return m / math.sqrt(var / n) if var > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.70)
    ap.add_argument("--min-names", type=int, default=15)
    args = ap.parse_args()

    con = sqlite3.connect("file:accuracy.db?mode=ro", uri=True)
    rows = con.execute("""
        SELECT p.prediction_date, p.horizon, p.prob_up, o.actual_return
        FROM predictions p JOIN outcomes o
          ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
         AND p.horizon=o.horizon
        WHERE p.horizon IN (3,5)""").fetchall()
    con.close()

    per = {}
    for h in (3, 5):
        byd = defaultdict(list)
        for d, hh, pu, r in rows:
            if int(hh) == h:
                byd[d].append((pu, r))
        ex = {}
        for d, v in byd.items():
            if len(v) < args.min_names:
                continue
            sel = [r for pu, r in v if pu >= args.thresh]
            if sel:
                ex[d] = st.mean(sel) - st.mean(r for _, r in v)
        per[h] = ex
        print(f"h={h}: {len(ex)} dates where prob>={args.thresh} fires")
    print()

    print(f"  {'boundary':<14}{'h':>3}{'before':>11}{'n':>5}{'t':>7}"
          f"{'after':>11}{'n':>5}{'t':>7}")
    for b in BOUNDARIES:
        for h in (3, 5):
            ex = per[h]
            pre = [v for d, v in sorted(ex.items()) if d < b]
            post = [v for d, v in sorted(ex.items()) if d >= b]
            if len(pre) < 8 or len(post) < 8:
                continue
            tp = nw_t(pre, h) or 0.0
            ta = nw_t(post, h) or 0.0
            print(f"  {b:<14}{h:>3}{100*st.mean(pre):>+10.3f}pp{len(pre):>5}"
                  f"{tp:>+7.2f}{100*st.mean(post):>+10.3f}pp{len(post):>5}"
                  f"{ta:>+7.2f}"
                  + ("   POST t>3" if abs(ta) > 3.0 else ""))
        print()

    print("  MONTHLY — a step at one boundary means a code change; a gradual")
    print("  drift means regime, and no boundary explains it.")
    for h in (3, 5):
        ex = per[h]
        bym = defaultdict(list)
        for d, v in ex.items():
            bym[d[:7]].append(v)
        print(f"\n  h={h}")
        print(f"    {'month':<9}{'dates':>7}{'excess':>11}")
        for m in sorted(bym):
            v = bym[m]
            if len(v) < 3:
                continue
            print(f"    {m:<9}{len(v):>7}{100*st.mean(v):>+10.3f}pp")

    print("\n  Bar is NW t > 3.0. The post-period is ~62 dates at h=5 with")
    print("  5-day overlapping windows, so the effective independent count is")
    print("  nearer 12. And this boundary was chosen AFTER seeing the split,")
    print("  which is the selection effect the hurdle exists to absorb.\n")
    print("  Nearby changes on record: 2026-05-07 three-way split leak fix")
    print("  (worth ~0.30 AUC), 2026-05-22 rev_surprise died, 2026-05-31 kill")
    print("  switch, 2026-06-28 vix_term_structure pinned to 1.0, 2026-08-26")
    print("  short_ratio PIT rewire. A jump at 05-07 points at the leak fix.")


if __name__ == "__main__":
    main()
