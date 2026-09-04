#!/usr/bin/env python3
"""
pct7_status.py — score the PCT7 shadow model. Weekly.

READ-ONLY. Writes nothing.

WHY THIS RUNS AUTOMATICALLY
    PCT7 predicts P(forward 5-day return >= +7%). It was trained 2026-05-25,
    wired into signals/generator.py in shadow mode -- logged to
    predictions.prob_pct7, never traded -- and the implementation plan said
    "log alongside production for 1-2 weeks before promoting".

    Nobody scored it. It ran for FOURTEEN weeks and accumulated 24,741
    predictions unread, on a model that was never retrained. When finally
    scored on 2026-09-05 the result was the strongest measured anywhere in the
    system:

        threshold   n      dates  tickers  hit     base    lift
        0.20        3,506  65     189      22.8%   10.8%   +12.0pp
        0.30        1,105  64     116      29.0%   10.8%   +18.2pp
        0.40          299  44      75      40.1%   10.8%   +29.3pp
        0.50           66  14      37      57.6%   10.8%   +46.8pp

    Monotone in threshold across 65 dates, with the 0.20 interval [21.5, 24.2]
    nowhere near the 10.8% base. For comparison the h=5 DIRECTION model runs
    AUC 0.535 with 2-4pp of top-decile lift -- PCT7 delivers +12pp on a harder
    target, from a model fourteen weeks stale and therefore out-of-sample
    throughout.

    A first pass at threshold 0.5 gave 66 fires on 14 dates, one of which
    (2026-07-29, 25 fires) carried most of the result, and looked like
    cluster-timing rather than selection. That was an artifact of using the
    wrong cut: the monitor's own default is 0.20 "since training base rate was
    13%". At the intended threshold the clustering disappears -- 65 dates, 189
    tickers.

WHAT THIS PRINTS
    Hit rate and lift over the SAME-PERIOD base rate at five thresholds, with
    Wilson intervals and the date and ticker spread at each. Date and ticker
    counts are shown because a high hit rate concentrated in a few sessions is
    cluster timing, not stock selection, and the counts are what distinguish
    them.

    Also a per-date view of the most recent sessions, each against ITS OWN base
    rate. A +7% day for the whole market inflates any long signal; comparing
    within-date removes that.

    python analysis/pct7_status.py
    python analysis/pct7_status.py --since 2026-07-01
"""
import argparse
import math
import sqlite3
from collections import defaultdict


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 100.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, 100 * (c - s) / d), min(100.0, 100 * (c + s) / d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--since", default="2026-05-25",
                    help="PCT7 was trained on this date; default scores all")
    ap.add_argument("--target", type=float, default=0.07)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = con.execute("""
        SELECT p.prediction_date, p.ticker, p.prob_pct7, o.actual_return
        FROM predictions p
        JOIN outcomes o ON p.ticker=o.ticker
          AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
        WHERE p.horizon=5 AND p.prob_pct7 IS NOT NULL
          AND o.actual_return IS NOT NULL AND p.prediction_date >= ?
    """, (args.since,)).fetchall()
    con.close()

    if len(rows) < 200:
        print(f"only {len(rows)} scored predictions since {args.since}")
        return

    base_k = sum(1 for _, _, _, r in rows if r >= args.target)
    base = 100.0 * base_k / len(rows)
    print(f"PCT7 shadow model — target: forward 5d return >= {args.target:+.0%}")
    print(f"{len(rows):,} scored predictions since {args.since}, "
          f"{len({r[0] for r in rows})} dates, "
          f"{len({r[1] for r in rows})} tickers")
    print(f"base rate over the same period: {base:.1f}%\n")

    print(f"  {'thresh':>7}{'n':>7}{'dates':>7}{'tickers':>9}{'hit':>8}"
          f"{'95% CI':>16}{'lift':>10}")
    for cut in (0.20, 0.25, 0.30, 0.40, 0.50):
        sel = [r for r in rows if r[2] >= cut]
        if len(sel) < 30:
            print(f"  {cut:>7.2f}{len(sel):>7}   too few")
            continue
        k = sum(1 for _, _, _, r in sel if r >= args.target)
        lo, hi = wilson(k, len(sel))
        print(f"  {cut:>7.2f}{len(sel):>7}{len({s[0] for s in sel}):>7}"
              f"{len({s[1] for s in sel}):>9}{100*k/len(sel):>7.1f}%"
              f"   [{lo:>5.1f},{hi:>5.1f}]{100*k/len(sel)-base:>+9.1f}pp")

    print("\n  A hit rate that rises monotonically with the threshold is the")
    print("  signature of a real ranking. Date and ticker counts matter as much")
    print("  as n: concentration in a few sessions is cluster timing, not")
    print("  selection.\n")

    # per-date, against each date's OWN base rate
    byd = defaultdict(list)
    alld = defaultdict(list)
    for d, t, p, r in rows:
        alld[d].append(r)
        if p >= 0.20:
            byd[d].append(r)
    print("  most recent 12 sessions at threshold 0.20, "
          "each vs its OWN base rate")
    print(f"  {'date':<12}{'n':>5}{'hit':>7}{'base':>8}{'lift':>9}")
    for d in sorted(byd)[-12:]:
        v = byd[d]
        u = alld[d]
        h = 100.0 * sum(1 for r in v if r >= args.target) / len(v)
        b = 100.0 * sum(1 for r in u if r >= args.target) / len(u)
        print(f"  {d:<12}{len(v):>5}{h:>6.0f}%{b:>7.1f}%{h-b:>+8.1f}pp")

    won = sum(1 for d in byd
              if (sum(1 for r in byd[d] if r >= args.target) / len(byd[d]))
              > (sum(1 for r in alld[d] if r >= args.target) / len(alld[d])))
    print(f"\n  beat its own-date base rate on {won}/{len(byd)} sessions")
    print("  Within-date comparison removes market direction: a day when "
          "everything\n  rose inflates any long signal, and this controls for "
          "it.")


if __name__ == "__main__":
    main()
