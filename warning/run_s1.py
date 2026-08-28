#!/usr/bin/env python3
"""
run_s1.py — compute S1 as of a date, or replay it across history and check the
result against the registry's OWN documented verdicts.

    python warning/run_s1.py --db warning.db                        # today
    python warning/run_s1.py --db warning.db --asof 2007-08-15      # a past date
    python warning/run_s1.py --db warning.db --validate             # the real test

--validate is the falsification run. signal_registry.csv records, ex ante:
    2000: "fired Feb+Jul 2000"
    2008: "fired Aug06-May07 + Jun07 re-steepen"
    2022: "post-peak only"
If the builder does not reproduce those from point-in-time data, the builder is
wrong (or the registry's verdict is) -- either way it is a finding, printed as
one. Nothing here writes to the database.
"""
import argparse, os, sqlite3, sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from builders import s1_term_spread as S1     # noqa: E402


def show(r):
    d = r["detail"]
    print(f"  asof {r['asof']}  state={r['state']}"
          f"  stale={r['stale']}({r['stale_days']}d)  src_asof={r['source_asof']}")
    if r["state"] == "NA":
        print(f"    NA reason: {d['reason']}")
        return
    print(f"    month {d['current_month']}  spread {d['current_spread']:+.3f}"
          f"  inverted {d['inverted_of_last3']}/3  armed={d['armed']}"
          f"  escalated={d['escalated']}")
    rs = d.get("resteepen", {})
    if rs.get("run_len"):
        print(f"    inversion run {rs.get('run_start')}..{rs.get('run_end')} "
              f"({rs['run_len']}m, ongoing={rs.get('ongoing')})  "
              f"trough {rs.get('trough')} @{rs.get('trough_month')}  "
              f"latest {rs.get('latest')}  rise {rs.get('rise_bp')}bp")
    elif rs.get("reason"):
        print(f"    resteepen: {rs['reason']}")


def _month_ends(start, end):
    y, m = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    out = []
    while (y, m) <= (ey, em):
        nm_y, nm_m = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append((date(nm_y, nm_m, 1) - __import__("datetime").timedelta(days=1)).isoformat())
        y, m = nm_y, nm_m
    return out


def _scan(con, start, end):
    from builders import s1_term_spread as _S1
    print(f"S1 state changes, {start}..{end} (point-in-time month-ends)\n")
    prev = None
    for asof in _month_ends(start, end):
        r = _S1.compute(con, asof)
        if r["state"] != prev:
            d = r["detail"]
            extra = ""
            if r["state"] != "NA":
                rs = d.get("resteepen", {})
                extra = (f"  spread {d['current_spread']:+.3f}"
                         f"  inv {d['inverted_of_last3']}/3"
                         f"  run {rs.get('run_len')}m")
                if r["state"] == "R":
                    extra += f"  rise {rs.get('rise_bp')}bp off {rs.get('trough_month')}"
            print(f"  {asof}  {prev or '-':>2} -> {r['state']:<2}{extra}")
            prev = r["state"]
    print(f"\n  (final state at {end}: {prev})")


ANCHORS = [
    ("2000-02-29", "registry: fired Feb 2000"),
    ("2000-07-31", "registry: fired Jul 2000"),
    ("2006-08-31", "registry: 2008 arm begins Aug-06"),
    ("2007-05-31", "registry: arm runs to May-07"),
    ("2007-06-29", "registry: Jun-07 RE-STEEPEN -> escalate"),
    ("2007-10-09", "SPX peak (report Part III)"),
    ("2020-02-19", "COVID peak -- designed MISS, stack was green"),
    ("2021-12-31", "2022 validation: registry says post-peak only"),
    ("2022-01-03", "SPX 2022 peak"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--validate", action="store_true",
                    help="replay the registry's documented anchor months")
    ap.add_argument("--scan", nargs=2, metavar=("START", "END"),
                    help="walk month-ends START..END and print STATE CHANGES only, "
                         "e.g. --scan 2005-01 2008-12. Answers 'when did S1 actually "
                         "arm?' rather than sampling anchor dates.")
    args = ap.parse_args()
    con = sqlite3.connect(args.db)

    if args.scan:
        _scan(con, args.scan[0], args.scan[1])
        return

    if not args.validate:
        show(S1.compute(con, args.asof))
        return

    print("S1 replay against the registry's own ex-ante verdicts")
    print("(point-in-time: each row sees only what was published by that date)\n")
    for asof, note in ANCHORS:
        print(f"[{note}]")
        show(S1.compute(con, asof))
        print()

    print("READ THIS AS:")
    print("  Feb/Jul-2000 and Aug06-May07 should be ARMED (amber).")
    print("  Jun-2007 should be ESCALATED (R) -- the re-steepen.")
    print("  Feb-2020 should NOT be red: the report calls that a designed miss.")
    print("  Any mismatch is a finding. Do not tune the builder to make it fit;")
    print("  thresholds are frozen (rule #3). Report it and we diagnose.")


if __name__ == "__main__":
    main()
