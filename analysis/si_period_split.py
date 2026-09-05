#!/usr/bin/env python3
"""
si_period_split.py — is the SI brick decaying? Same code path as the lag test.

READ-ONLY. Writes nothing.

WHY THIS EXISTS RATHER THAN A NEW MEASUREMENT
    Every SI number measured on 2026-09-05 came in below its recorded value:

        metric              recorded    today
        long-leg Sharpe       ~1.25      0.83     (si_leg_decomp.py)
        per-date IC          -0.054     -0.0379   (si_dissemination_lag_test.py)
        NW-t                  -4.46     -3.03
        header of the lag     -0.053     -0.0379
          test itself         -4.73      -3.03

    The lag test's own docstring cites IC -0.053 / NW-t -4.73, so the record and
    the script agree with each other and disagree with the live run. That is a
    ~28% drop in IC since July.

    An earlier attempt to answer this reimplemented the decomposition and got
    long/reb +0.57% against si_leg_decomp's +1.28% -- less than half -- because
    it entered on the first bar STRICTLY AFTER settlement while the original
    enters AT settlement (d + 0..5 days), and it read raw_bars.close rather than
    daily_prices.adj_close. A reimplementation that does not reproduce the
    baseline cannot answer a question about the baseline.

    So this script uses the SAME construction as si_dissemination_lag_test.py --
    same table, same entry rule, same lag, same Newey-West -- and only adds the
    period split.

THE QUESTION
    DECAY        early rebalances carry the edge, recent ones are flat. The live
                 book is then running on a signal that has stopped working.
    OVERSTATED   every period looks alike and the recorded -0.054 / -4.46 came
                 from a shorter or luckier window. Then nothing changed and the
                 RECORD needs correcting, not the book.

    Opposite actions, so this must be measured rather than assumed.

ENTRY LAG
    Reported at lag=0 (as the brick was published, NOT tradeable) and lag=8
    business days (the earliest a trader could act, since FINRA disseminates
    ~8 BD after settlement). The lag test found 90% of edge survives at lag=8,
    so both are meaningful; lag=8 is the one that matters for the live book.

    python analysis/si_period_split.py
"""
import argparse
import datetime
import math
import os
import sqlite3
from collections import defaultdict

import numpy as np


def ro(p):
    return sqlite3.connect("file:" + os.path.abspath(p) + "?mode=ro&immutable=1",
                           uri=True, timeout=30)


def nd(x):
    try:
        return datetime.date.fromisoformat(str(x)[:10])
    except Exception:
        return None


def nw_se(x, lag):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return None
    e = x - x.mean()
    s = float(e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return math.sqrt(s / n) if s > 0 else None


def spearman(x, y):
    if len(x) < 5:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--hold", type=int, default=40)
    ap.add_argument("--min-names", type=int, default=20)
    a = ap.parse_args()

    px_db = os.path.join(a.root, "prices.db")
    si_db = os.path.join(a.root, "short_interest.db")

    # daily_prices.adj_close -- the table si_leg_decomp and the lag test use
    c = ro(px_db)
    try:
        prows = c.execute("SELECT ticker, date, adj_close FROM daily_prices "
                          "WHERE adj_close IS NOT NULL").fetchall()
    finally:
        c.close()
    px = defaultdict(list)
    for tk, d, p in prows:
        do = nd(d)
        if do is None:
            continue
        try:
            pf = float(p)
        except Exception:
            continue
        if pf > 0:
            px[tk].append((do, pf))
    for tk in px:
        px[tk].sort()
    pos = {tk: {d: i for i, (d, _) in enumerate(v)} for tk, v in px.items()}

    def fwd(tk, d, h, lag_bd):
        lst, idx = px.get(tk), pos.get(tk)
        if not lst or not idx:
            return None
        i = None
        for off in range(0, 6):          # entry AT settlement, or next bar
            cc = d + datetime.timedelta(days=off)
            if cc in idx:
                i = idx[cc]
                break
        if i is None:
            return None
        i += lag_bd                      # push entry out by business days
        if i + h >= len(lst):
            return None
        p0 = lst[i][1]
        return (lst[i + h][1] / p0 - 1.0) if p0 > 0 else None

    c = ro(si_db)
    try:
        sirows = c.execute("SELECT ticker, settlement_date, days_to_cover "
                           "FROM short_interest").fetchall()
    finally:
        c.close()
    si = defaultdict(dict)
    for tk, d, v in sirows:
        do = nd(d)
        if do is None or v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv <= 50.0:
            si[do][tk.upper()] = fv

    lag = max(1, int(math.ceil(a.hold / 15.0)))
    print(f"{len(si)} settlements, hold {a.hold}d, Newey-West lag {lag}")
    print("entry AT settlement (d+0..5), daily_prices.adj_close, DTC<=50 "
          "-- matching si_leg_decomp\n")

    for lag_bd, note in ((0, "as published, NOT tradeable"),
                         (8, "earliest tradeable")):
        per_date = []
        for d in sorted(si):
            vals, rets = [], []
            for tk, dv in si[d].items():
                if tk not in pos:
                    continue
                r = fwd(tk, d, a.hold, lag_bd)
                if r is None:
                    continue
                vals.append(dv)
                rets.append(r)
            if len(vals) < a.min_names:
                continue
            ic = spearman(np.array(vals), np.array(rets))
            if ic is not None:
                per_date.append((d, ic))

        print(f"=== entry lag {lag_bd} business days — {note} ===")
        if len(per_date) < 20:
            print("  too few dates\n")
            continue

        def report(label, sub):
            if len(sub) < 6:
                print(f"  {label:<14}{len(sub):>6}   too few")
                return
            v = np.array([x[1] for x in sub], float)
            se = nw_se(v, lag)
            t = (v.mean() / se) if se else 0.0
            right = 100.0 * float((v < 0).mean())   # negative IC is the edge
            print(f"  {label:<14}{len(sub):>6}{v.mean():>+10.4f}{t:>+8.2f}"
                  f"{right:>10.0f}%")

        print(f"  {'period':<14}{'dates':>6}{'mean IC':>10}{'NW t':>8}"
              f"{'right sign':>11}")
        report("FULL", per_date)
        for y in sorted({d.year for d, _ in per_date}):
            report(str(y), [x for x in per_date if x[0].year == y])
        h = len(per_date) // 2
        report("first half", per_date[:h])
        report("second half", per_date[h:])
        print()

    print("  Recorded: IC -0.054, NW-t -4.46. The lag test's own header cites")
    print("  -0.053 / -4.73. Today's live run gives -0.0379 / -3.03 at lag 0.")
    print("\n  A year holds only ~6 rebalances at hold=40, so per-year figures")
    print("  are indicative. The half-sample split is the trustworthy one.")
    print("\n  If both halves look alike, the record is what needs correcting.")
    print("  If the second half is materially weaker, the live book is running")
    print("  on a faded signal and that is urgent.")


if __name__ == "__main__":
    main()
