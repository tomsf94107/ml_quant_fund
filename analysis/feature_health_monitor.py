#!/usr/bin/env python3
"""
feature_health_monitor.py — flag features that silently went dead.

READ-ONLY on the model. Reads feature_importance_history. Writes nothing.

WHY THIS EXISTS
    vix_term_structure carried importance 4.764 on 2026-06-27 and 0.000 every
    single day after. It stayed dead for TEN WEEKS. The cause was that
    builder.py asked yfinance for ^VIX3M, yfinance is XProtect-blocked on this
    machine, and the else-branch pinned the feature to the literal 1.0 -- while
    warning.db held CBOE_VIX3M with 4,261 observations the whole time.

    Nothing reported it. The evidence was sitting in accuracy.db: a daily row
    reading 0.0 where there used to be 4.764. Reading that table once a week
    would have caught it in days.

    Same for vol_x_short, which now scores 8.315 after the repair -- above
    yield_10y at 7.894 -- and was 0.0 for the same period.

WHAT IT FLAGS
    DIED         non-zero mean importance in the reference window, ~zero now.
                 The vix_term_structure case exactly. Highest priority: a
                 feature that USED to work and stopped is a broken input, not
                 an uninformative column.
    REVIVED      zero then, non-zero now. Confirms a repair landed, and catches
                 a feed that came back on its own.
    ALWAYS ZERO  zero in both windows. Not necessarily a defect -- several
                 features here are live-only and zeroed in training BY DESIGN
                 (the earnings four, sentiment_score, analyst_*). Listed
                 separately and NOT flagged as errors, because treating a
                 documented design choice as a bug wastes attention.
    FADING       dropped by more than the threshold without reaching zero.
                 Weakest signal, most likely to be noise, reported last.

    Coverage is also checked: a feature that disappears from the table entirely
    between windows means the column was removed from OUTPUT_COLUMNS, which is
    a different event and worth knowing about.

DESIGN NOTE
    A dropped feature and a dead feature look identical here -- both simply have
    no row. That is an argument for KEEPING inert columns in OUTPUT_COLUMNS
    rather than removing them: an inert column costs a little CPU and stays
    observable, while a removed one is invisible to this monitor forever.

    python analysis/feature_health_monitor.py
    python analysis/feature_health_monitor.py --recent 7 --reference 30 --gap 30
"""
import argparse
import sqlite3
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--recent", type=int, default=7,
                    help="days in the recent window")
    ap.add_argument("--gap", type=int, default=30,
                    help="days back to the START of the reference window")
    ap.add_argument("--reference", type=int, default=30,
                    help="length of the reference window in days")
    ap.add_argument("--zero", type=float, default=0.01,
                    help="mean importance at or below this counts as zero")
    ap.add_argument("--fade", type=float, default=0.60,
                    help="fractional drop that counts as FADING")
    ap.add_argument("--quiet", action="store_true",
                    help="print only flagged features; for cron")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    ref_start = args.gap + args.reference

    def window(a, b):
        return {f: v for f, v in con.execute(
            "SELECT feature, AVG(importance) FROM feature_importance_history "
            "WHERE retrain_date >= date('now', ?) "
            "AND retrain_date <  date('now', ?) GROUP BY feature",
            (f"-{a} days", f"-{b} days"))}

    recent = window(args.recent, 0)
    ref = window(ref_start, args.gap)
    con.close()

    if not recent or not ref:
        print("not enough retrain history to compare windows")
        return 0

    if not args.quiet:
        print(f"recent : last {args.recent} days           "
              f"({len(recent)} features)")
        print(f"referen: {ref_start}-{args.gap} days ago   "
              f"({len(ref)} features)")
        print(f"zero threshold {args.zero}, fade threshold "
              f"{args.fade:.0%}\n")

    died, revived, always, fading, vanished, appeared = [], [], [], [], [], []
    for f in sorted(set(recent) | set(ref)):
        r = recent.get(f)
        o = ref.get(f)
        if r is None:
            vanished.append((f, o))
            continue
        if o is None:
            appeared.append((f, r))
            continue
        if o > args.zero and r <= args.zero:
            died.append((f, o, r))
        elif o <= args.zero and r > args.zero:
            revived.append((f, o, r))
        elif o <= args.zero and r <= args.zero:
            always.append(f)
        elif o > 0 and (o - r) / o >= args.fade:
            fading.append((f, o, r))

    rc = 0
    if died:
        rc = 1
        print(f"!! DIED — non-zero before, zero now: {len(died)}")
        print(f"   {'feature':<32}{'was':>9}{'now':>9}")
        for f, o, r in sorted(died, key=lambda x: -x[1]):
            print(f"   {f:<32}{o:>9.3f}{r:>9.3f}")
        print("   A feature that USED to work and stopped is a broken input,")
        print("   not an uninformative column. Check its source first.\n")

    if revived:
        print(f"REVIVED — zero before, non-zero now: {len(revived)}")
        for f, o, r in sorted(revived, key=lambda x: -x[2]):
            print(f"   {f:<32}{o:>9.3f}{r:>9.3f}")
        print()

    if vanished:
        rc = 1
        print(f"!! VANISHED — present before, absent now: {len(vanished)}")
        for f, o in vanished:
            print(f"   {f:<32}{o:>9.3f}")
        print("   The column left OUTPUT_COLUMNS, or retraining stopped "
              "producing it.\n")

    if appeared and not args.quiet:
        print(f"NEW — absent before, present now: {len(appeared)}")
        for f, r in appeared:
            print(f"   {f:<32}{r:>9.3f}")
        print()

    if fading:
        print(f"fading — dropped {args.fade:.0%} or more but not to zero: "
              f"{len(fading)}")
        for f, o, r in sorted(fading, key=lambda x: (x[2] - x[1]))[:10]:
            print(f"   {f:<32}{o:>9.3f}{r:>9.3f}")
        print("   Weakest signal here and most likely to be noise.\n")

    if always and not args.quiet:
        print(f"always zero in both windows: {len(always)}")
        print(f"   {', '.join(always)}")
        print("   NOT flagged as errors. Several are live-only by design and")
        print("   zeroed in training_mode for PIT honesty -- the earnings four,")
        print("   sentiment_score, analyst_*. A documented design choice is not")
        print("   a defect, and treating it as one wastes attention.\n")

    if not died and not vanished:
        print("no feature died or vanished between the windows.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
