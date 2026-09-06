#!/usr/bin/env python3
"""
si_decay_test.py — has the SI brick decayed, or was it never what the record says?

READ-ONLY. Writes nothing.

WHY
    The short-interest brick is recorded as the fund's one validated edge:
    per-date IC -0.054, NW-t -4.46, 8.3 sigma null rejection, long-leg
    Sharpe ~1.25, 71% of edge in the long leg.

    Re-run on 2026-09-05 with 126 rebalances of data, si_leg_decomp.py returns:

        LONG (low-DTC)   +0.0128/reb   +11.7%/yr   Sharpe 0.83   NW t +1.95
        SHORT (high-DTC) +0.0034/reb    +3.1%/yr   Sharpe 0.29   NW t +0.75
        BOOK (L/S)       +0.0162/reb   +14.8%/yr   Sharpe 0.73   NW t +1.83
        share: LONG 79% | SHORT 21%

    And validate_si.py independently returns WEAK/NULL on pooled IC (t=1.87),
    though beta-stripped runs t=3.86, sign-only t=6.79, and OOS IC +0.0393 at
    t=2.63.

    So the edge is not dead -- Sharpe 0.83 and +11.7%/yr on the long leg is a
    real number -- but it is not the 1.25-Sharpe, 8.3-sigma edge the record
    describes. Long-leg t=+1.95 sits below the conventional 2.0 bar.

THE QUESTION THIS ANSWERS
    Two very different explanations produce the same full-sample number:

      DECAY        the edge was real and has faded. Early years carry it, recent
                   years are flat. The live book is then running on a signal
                   that has stopped working, and that is urgent.

      OVERSTATED   the edge was always about this size and the recorded 1.25 /
                   4.46 came from a shorter window, a different construction, or
                   an optimistic pass. Then nothing has changed and the record
                   needs correcting, not the book.

    These call for opposite actions, so guessing between them is not acceptable.

METHOD
    The same leg decomposition si_leg_decomp.py performs -- long_excess and
    short_excess over the universe mean per rebalance, dollar-neutral, hold 40,
    quintile 20% -- computed separately per calendar year and per half-sample.

    Newey-West at lag 3 on the per-rebalance series, matching the original
    (overlapping 40-day windows on ~2-week rebalances induce autocorrelation of
    roughly that order).

    Reported with the count of rebalances per period, because a year with 20
    rebalances cannot support a t-statistic and should not be read as one.

    python analysis/si_decay_test.py
"""
import argparse
import datetime
import math
import os
import sqlite3
import statistics as st
from collections import defaultdict


def ro(p):
    return sqlite3.connect("file:" + os.path.abspath(p) + "?mode=ro&immutable=1",
                           uri=True, timeout=30)


def nw_t(series, lag=3):
    n = len(series)
    if n < 6:
        return None
    m = sum(series) / n
    d = [x - m for x in series]
    var = sum(x * x for x in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[i] * d[i - k] for i in range(k, n)) / n
        var += 2 * (1 - k / (lag + 1.0)) * gk
    return m / math.sqrt(var / n) if var > 0 else None


def sharpe(series, per_year):
    if len(series) < 6:
        return None
    s = st.pstdev(series)
    return (st.mean(series) / s * math.sqrt(per_year)) if s > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--hold", type=int, default=40)
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--min-names", type=int, default=20)
    ap.add_argument("--clip-dtc", type=float, default=50.0)
    args = ap.parse_args()

    si_db = os.path.join(args.root, "short_interest.db")
    px_db = os.path.join(args.root, "prices.db")

    c = ro(px_db)
    close = defaultdict(dict)
    for t, d, p in c.execute("SELECT ticker, d, close FROM raw_bars "
                             "WHERE close IS NOT NULL"):
        close[t][str(d)[:10]] = p
    c.close()
    order = {t: sorted(v) for t, v in close.items()}

    c = ro(si_db)
    rows = c.execute(
        "SELECT ticker, settlement_date, days_to_cover FROM short_interest "
        "WHERE days_to_cover IS NOT NULL AND days_to_cover <= ? "
        "ORDER BY settlement_date", (args.clip_dtc,)).fetchall()
    c.close()

    bydate = defaultdict(list)
    for t, d, v in rows:
        bydate[str(d)[:10]].append((t, v))
    print(f"{len(rows):,} SI points, {len(bydate)} settlements, "
          f"hold {args.hold}d, quintile {args.quantile:.0%}\n")

    def fwd(t, d, h):
        ds = order.get(t)
        if not ds:
            return None
        lo, hi = 0, len(ds)
        while lo < hi:
            mid = (lo + hi) // 2
            if ds[mid] <= d:
                lo = mid + 1
            else:
                hi = mid
        i = lo                      # first bar STRICTLY AFTER the settlement
        if i + h >= len(ds):
            return None
        a, b = close[t][ds[i]], close[t][ds[i + h]]
        if not a or not b:
            return None
        r = (b - a) / a
        return r if abs(r) < 3.0 else None

    per_reb = []
    for d in sorted(bydate):
        names = []
        for t, v in bydate[d]:
            r = fwd(t, d, args.hold)
            if r is not None:
                names.append((v, r, t))
        if len(names) < args.min_names:
            continue
        names.sort()
        k = max(1, int(len(names) * args.quantile))
        mkt = st.mean(r for _, r, _ in names)
        lo_leg = st.mean(r for _, r, _ in names[:k])          # LOW dtc
        hi_leg = st.mean(r for _, r, _ in names[-k:])         # HIGH dtc
        per_reb.append((d, lo_leg - mkt, mkt - hi_leg, len(names)))

    if len(per_reb) < 20:
        print(f"only {len(per_reb)} rebalances -- cannot split")
        return
    per_year = 252.0 / args.hold * (len(per_reb) /
                                    max(len(per_reb), 1))     # ~6.3 reb/yr
    py = 252.0 / args.hold

    print(f"{len(per_reb)} rebalances, "
          f"{per_reb[0][0]} .. {per_reb[-1][0]}\n")

    def report(label, sub):
        if len(sub) < 6:
            print(f"  {label:<14}{len(sub):>5}   too few rebalances")
            return
        L = [x[1] for x in sub]
        S = [x[2] for x in sub]
        B = [x[1] + x[2] for x in sub]
        lt, bt = nw_t(L), nw_t(B)
        ls, bs = sharpe(L, py), sharpe(B, py)
        share = (100 * sum(L) / (sum(L) + sum(S))
                 if (sum(L) + sum(S)) != 0 else float("nan"))
        print(f"  {label:<14}{len(sub):>5}{100*st.mean(L):>9.2f}%"
              f"{(ls if ls else 0):>8.2f}{(lt if lt else 0):>+8.2f}"
              f"{100*st.mean(B):>10.2f}%{(bs if bs else 0):>8.2f}"
              f"{(bt if bt else 0):>+8.2f}{share:>8.0f}%")

    print("  LONG leg = low days-to-cover, excess over the universe mean.")
    print(f"  {'period':<14}{'rebs':>5}{'long/reb':>9}{'Sharpe':>8}{'NW t':>8}"
          f"{'book/reb':>10}{'Sharpe':>8}{'NW t':>8}{'long%':>8}")
    report("FULL", per_reb)
    print()

    years = sorted({d[:4] for d, _, _, _ in per_reb})
    for y in years:
        report(y, [x for x in per_reb if x[0][:4] == y])
    print()

    half = len(per_reb) // 2
    report("first half", per_reb[:half])
    report("second half", per_reb[half:])

    print("\n  Recorded for this brick: long-leg Sharpe ~1.25, per-date IC")
    print("  -0.054, NW-t -4.46, 8.3 sigma. The full-sample re-run gives")
    print("  Sharpe 0.83 and t +1.95.")
    print("\n  If the early years carry the edge and recent ones are flat, this")
    print("  is DECAY and the live book is running on a faded signal. If every")
    print("  period looks alike, the edge was always this size and the RECORD")
    print("  is what needs correcting. Those call for opposite actions.")
    print("\n  A year holds only ~6 rebalances at hold=40, so per-year t-stats")
    print("  are indicative at best. The half-sample split is the more")
    print("  trustworthy comparison.")


if __name__ == "__main__":
    main()
