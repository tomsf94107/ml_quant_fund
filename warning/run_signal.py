#!/usr/bin/env python3
"""
run_signal.py — compute / scan / validate any signal builder. Read-only.

    python warning/run_signal.py --signal S2 --db warning.db
    python warning/run_signal.py --signal S1 --db warning.db --scan 1962-01 2026-08
    python warning/run_signal.py --signal S2 --db warning.db --validate

--validate replays the registry's OWN ex-ante verdicts. A mismatch is a finding:
report it, do not tune the builder. Thresholds are frozen (rule #3).
"""
import argparse, os, sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from builders import s1_term_spread as S1          # noqa: E402
from builders import s2_credit as S2               # noqa: E402
from builders import f2_vix_percentile as F2       # noqa: E402

BUILDERS = {"S1": S1, "S2": S2, "F2": F2}

ANCHORS = {
    "S1": [("2000-02-29", "registry: fired Feb 2000 -- see DECISIONS.md D4"),
           ("2000-07-31", "registry: fired Jul 2000 -- see DECISIONS.md D4"),
           ("2006-10-31", "arm (2-of-3 confirmation of the Aug-06 inversion)"),
           ("2007-05-31", "registry: arm runs to May-07"),
           ("2007-07-31", "registry: Jun-07 data month re-steepen -> escalate"),
           ("2007-10-09", "SPX peak (report Part III)"),
           ("2020-02-19", "COVID peak -- designed MISS"),
           ("2022-01-03", "SPX 2022 peak -- registry: post-peak only")],
    # NOTE: an as-of date reads the last FULLY PUBLISHED month (DECISIONS.md D3),
    # so anchors sit one month AFTER the data month they are meant to sample.
    "F2": [("2000-03-24", "SPX 2000 peak: registry says VXO mid-20s"),
           ("2007-01-31", "registry: VIX 9.89 Jan-07 -- the complacency low"),
           ("2007-10-09", "SPX peak: registry VIX 16.12 -- 'the trap' (report line 424)"),
           ("2020-02-19", "COVID peak: registry 14.38 exhibit"),
           ("2022-01-03", "SPX 2022 peak"),
           ("2026-08-28", "today")],
    "S2": [("2000-02-29", "reads Jan-2000 data month: registry weak fire begins"),
           ("2000-04-30", "reads Mar-2000: the +44bp peak of the weak fire (D8)"),
           ("2007-07-31", "reads Jun-2007: registry strong fire begins (D8)"),
           ("2007-11-30", "reads Oct-2007: SPX peak month, fire window ends"),
           ("1998-10-30", "registry notes: 1998 false fire, must be killed by S3/S15"),
           ("2011-10-31", "registry notes: 2011 false fire"),
           ("2015-09-30", "registry notes: 2015 false fire"),
           ("2018-12-31", "registry notes: 2018 false fire"),
           ("2021-12-31", "registry: 2022 correctly silent"),
           ("2026-08-28", "today")],
}


def show(sig, r):
    d = r["detail"]
    print(f"  asof {r['asof']}  state={r['state']}"
          f"  stale={r['stale']}({r['stale_days']}d)  src={r['source_asof']}")
    if r["state"] == "NA":
        print(f"    NA reason: {d['reason']}")
        return
    if sig == "S1":
        print(f"    month {d['current_month']}  spread {d['current_spread']:+.3f}"
              f"  inverted {d['inverted_of_last3']}/3  armed={d['armed']}"
              f"  escalated={d['escalated']}")
        rs = d.get("resteepen", {})
        if rs.get("run_len"):
            print(f"    run {rs.get('run_start')}..{rs.get('run_end')} "
                  f"({rs['run_len']}m, ongoing={rs.get('ongoing')})  "
                  f"trough {rs.get('trough')}  rise {rs.get('rise_bp')}bp")
        if rs.get("reason"):
            print(f"    {rs['reason']}")
    elif sig == "F2":
        print(f"    VIX {d['vix']:.2f}  percentile(504d) {d['percentile_504d']:.1f}"
              f"  window {d['window_low']:.2f}..{d['window_high']:.2f}")
        print(f"    <20th={d['armed_below_20th']}  <10th={d['below_10th']}"
              f"  L2={d['l2_score']}")
        if d.get("l2_note"):
            print(f"    NOTE: {d['l2_note']}")
    else:
        ma = [k for k in d if k.startswith("ma")][0]
        lo = [k for k in d if k.startswith("low")][0]
        print(f"    mode={d['mode']} ({d['series']})  period {d['last_period']}"
              f"  spread {d['spread']:+.3f}  {ma} {d[ma]:+.3f}  {lo} {d[lo]:+.3f}")
        print(f"    above_ma={d['above_ma']}  off_low {d['off_low_by_bp']}bp"
              f" (need {int(S2.WIDEN_BP*100)}) -> credit_leg={d['credit_leg']}"
              f"  equity_leg={d['equity_leg']}"
              f"{' via '+d['equity_source'] if d.get('equity_source') else ''}")
        if d.get("equity_note"):
            print(f"    NOTE: {d['equity_note']}")


def month_ends(start, end):
    y, m = int(start[:4]), int(start[5:7]); ey, em = int(end[:4]), int(end[5:7])
    out = []
    while (y, m) <= (ey, em):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append((date(ny, nm, 1) - timedelta(days=1)).isoformat())
        y, m = ny, nm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="S1", choices=sorted(BUILDERS))
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--scan", nargs=2, metavar=("START", "END"))
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    import sqlite3
    con = sqlite3.connect(args.db)
    mod = BUILDERS[args.signal]

    if args.scan:
        print(f"{args.signal} state changes, {args.scan[0]}..{args.scan[1]}\n")
        prev = None
        for asof in month_ends(*args.scan):
            r = mod.compute(con, asof)
            if r["state"] != prev:
                print(f"  {asof}  {prev or '-':>2} -> {r['state']}")
                show(args.signal, r)
                prev = r["state"]
        print(f"\n  (final state at {args.scan[1]}: {prev})")
        return

    if args.validate:
        print(f"{args.signal} replay against the registry's ex-ante verdicts\n")
        for asof, note in ANCHORS[args.signal]:
            print(f"[{note}]")
            show(args.signal, mod.compute(con, asof))
            print()
        return

    show(args.signal, mod.compute(con, args.asof))


if __name__ == "__main__":
    main()
