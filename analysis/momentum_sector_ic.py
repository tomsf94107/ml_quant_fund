"""
Per-SECTOR momentum edge analysis — runs once momentum_shadow_outcomes resolves
(first ~late June 2026). Tells us WHERE the momentum edge actually lives on
resolved live data, which gates the universe-expansion decision: only add
decorrelated sectors if momentum is shown to work there.

Joins momentum_shadow_predictions (the BUY candidates) to momentum_shadow_outcomes
(resolved fwd returns), buckets by tickers_metadata.csv 'bucket', and reports per
sector: n resolved, hit-rate (actual_up), mean actual_return, and the spread vs
the universe mean. Also overall + per-kind.

Run: PYTHONPATH=. python3 analysis/momentum_sector_ic.py
"""
import sqlite3, csv
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "accuracy.db"

def load_buckets():
    return {r["ticker"].upper(): r.get("bucket","UNK")
            for r in csv.DictReader(open(ROOT/"tickers_metadata.csv"))}

def main():
    bmap = load_buckets()
    c = sqlite3.connect(DB)
    rows = c.execute("""
        SELECT p.prediction_date, p.ticker, p.kind,
               o.actual_return, o.actual_up
        FROM momentum_shadow_predictions p
        JOIN momentum_shadow_outcomes o
          ON p.prediction_date=o.prediction_date
         AND p.ticker=o.ticker AND p.kind=o.kind
        WHERE p.is_buy_candidate=1
    """).fetchall()
    c.close()

    if not rows:
        print("No resolved momentum outcomes yet "
              "(momentum_shadow_outcomes is empty).")
        print("First resolutions expected ~late June 2026 "
              "(20-trading-day horizon on the first shadow picks).")
        print("This script is staged — re-run once outcomes accrue.")
        return

    n = len(rows)
    allret = np.array([r[3] for r in rows])
    allhit = np.array([r[4] for r in rows])
    uni_mean = allret.mean()
    print(f"=== Resolved momentum BUY candidates: n={n} ===")
    print(f"Universe: hit-rate {100*allhit.mean():.1f}%  mean ret {100*uni_mean:+.3f}%")
    print()

    # per-kind
    print("--- by signal kind ---")
    bykind = defaultdict(list)
    for d,t,k,ret,up in rows: bykind[k].append((ret,up))
    for k in sorted(bykind):
        arr = bykind[k]; rr=np.array([x[0] for x in arr]); hh=np.array([x[1] for x in arr])
        print(f"  {k:10} n={len(arr):4}  hit {100*hh.mean():.1f}%  mean {100*rr.mean():+.3f}%")
    print()

    # per-sector
    print("--- by sector (bucket) — sorted by mean return ---")
    bysec = defaultdict(list)
    for d,t,k,ret,up in rows:
        bysec[bmap.get(t.upper(),"UNK")].append((ret,up))
    print(f"  {'bucket':<24}{'n':>5}{'hit%':>8}{'mean%':>9}{'vs uni':>9}")
    print("  " + "-"*55)
    res = []
    for b, arr in bysec.items():
        rr=np.array([x[0] for x in arr]); hh=np.array([x[1] for x in arr])
        res.append((b, len(arr), 100*hh.mean(), 100*rr.mean(), 100*(rr.mean()-uni_mean)))
    for b,nn,hit,mean,vs in sorted(res, key=lambda x:-x[3]):
        flag = "  <-- low n" if nn < 10 else ""
        print(f"  {b:<24}{nn:>5}{hit:>8.1f}{mean:>+9.3f}{vs:>+9.3f}{flag}")
    print()
    print("READING GUIDE:")
    print("  - Sectors with mean% > universe AND n>=10 = momentum edge is real there.")
    print("  - n<10 sectors: too few resolved picks to trust (wait for more).")
    print("  - Expansion: bias adds toward DECORRELATED sectors (Telecom, Industrial")
    print("    Gases, Healthcare, Cyber, Enterprise SW) THAT ALSO show positive edge here.")
    print("  - If edge is ONLY in AI-hardware: expansion into other sectors won't help")
    print("    momentum; the edge is the sector tilt and decorrelated names dilute it.")

if __name__ == "__main__":
    main()
