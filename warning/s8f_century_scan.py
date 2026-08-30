#!/usr/bin/env python3
"""
s8f_century_scan.py — how often does S8's rule fire across a century?

Runs S8F on month-ends from 1930 to 2026 and reports the firing rate plus the
dates. This is the test D19 made possible for S6 and that S8 has never had: its
SPDR version fired ONCE in ten years, on a false positive (D15), which is far too
small a sample to say whether the -15% / 200DMA / 5% thresholds are sensible.

CALIBRATION ONLY. French restates history (D20), so a fire date here is not a
claim that the signal would have fired in real time.

    python warning/s8f_century_scan.py --db warning.db
"""
import argparse, sqlite3, sys, os
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from builders import s8f_epicenter_french as S8F


def month_ends(y0, y1):
    out = []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            out.append((date(ny, nm, 1) - timedelta(days=1)).isoformat())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="warning.db")
    ap.add_argument("--from-year", type=int, default=1930)
    ap.add_argument("--to-year", type=int, default=2026)
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    counts, fires, prev = {}, [], None
    for asof in month_ends(args.from_year, args.to_year):
        r = S8F.compute(con, asof)
        st = r["state"]
        counts[st] = counts.get(st, 0) + 1
        if st == "R" and prev != "R":
            d = r["detail"]
            fires.append((asof, d["leader"], d["leader_drawdown_pct"],
                          d["market_pct_below_high"]))
        prev = st

    n = sum(counts.values())
    scored = n - counts.get("NA", 0)
    print(f"month-ends evaluated: {n}   scored: {scored}")
    for k in ("G", "Y", "R", "NA"):
        if k in counts:
            pct = 100.0 * counts[k] / max(scored, 1) if k != "NA" else \
                  100.0 * counts[k] / n
            print(f"  {k:<3} {counts[k]:>5}  {pct:>5.1f}%"
                  + ("  (of all month-ends)" if k == "NA" else "  (of scored)"))

    print(f"\n{len(fires)} distinct RED episodes:")
    print(f"  {'date':<12}{'leader':<8}{'leader DD':>10}{'mkt below high':>16}")
    for d, ldr, dd, mb in fires:
        print(f"  {d:<12}{ldr:<8}{dd:>9.1f}%{mb:>15.2f}%")
    con.close()


if __name__ == "__main__":
    main()
