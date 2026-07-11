#!/usr/bin/env python3
"""
si_dissemination_lag_test.py -- does the SI brick survive FINRA's publication lag?

THE ISSUE
  short_interest.settlement_date is the date the position was MEASURED.
  FINRA DISSEMINATES it ~8 business days later. validate_si_v2 and two_brick_book
  both measure forward returns starting AT settlement_date -> entry ~8 BD before the
  number is public. Same bug class as PEAD keying off a fiscal period end.

THE TEST (dose-response, per-date IC + Newey-West -- the same standard as the brick)
  Re-measure the brick with entry pushed out by 0, 5, 8, 10, 15 BUSINESS days.
    lag=0   the brick as published  (IC -0.053, NW-t -4.73)
    lag=8   the earliest a real trader could act
    lag=10  conservative

READ
  IC holds at lag>=8  -> the brick is REAL and tradeable. Look-ahead was not the source.
  IC collapses        -> the edge lived in the pre-publication window. Not tradeable.
"""
import sqlite3, argparse, math, numpy as np
from collections import defaultdict
from datetime import datetime

def nd(x):
    try: return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()
    except Exception: return None

def nw_se(x, lag):
    x = np.asarray(x, float); n = len(x)
    if n < 5: return float("nan")
    e = x - x.mean(); s = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - k / (lag + 1.0)) * ((e[k:] @ e[:-k]) / n)
    return math.sqrt(s / n) if s > 0 else float("nan")

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--hold", type=int, default=40)
ap.add_argument("--feature", default="days_to_cover")
ap.add_argument("--min-names", type=int, default=30)
a = ap.parse_args()

pc = sqlite3.connect(f"{a.root}/prices.db")
px = defaultdict(list)
for tk, d, c in pc.execute("SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL"):
    dd = nd(d)
    if dd: px[tk].append((dd, float(c)))
pc.close()
for tk in px: px[tk].sort()

def idx_on_or_after(ser, d):
    lo, hi = 0, len(ser) - 1
    while lo < hi:
        m = (lo + hi) // 2
        if ser[m][0] < d: lo = m + 1
        else: hi = m
    return lo if ser and ser[lo][0] >= d else None

def fwd(tk, d, entry_bd, hold):
    """Return over `hold` trading days, ENTERED entry_bd trading days after settlement d."""
    ser = px.get(tk)
    if not ser: return None
    i = idx_on_or_after(ser, d)
    if i is None: return None
    s = i + entry_bd
    e = s + hold
    if e >= len(ser) or s >= len(ser): return None
    p0 = ser[s][1]
    return ser[e][1] / p0 - 1.0 if p0 > 0 else None

sc = sqlite3.connect(f"{a.root}/short_interest.db")
rows = sc.execute(f'SELECT ticker, settlement_date, "{a.feature}" FROM short_interest '
                  f'WHERE "{a.feature}" IS NOT NULL').fetchall()
sc.close()
by_date = defaultdict(list)
for tk, sd, v in rows:
    d = nd(sd)
    if d is None: continue
    try:
        fv = float(v)
    except Exception:
        continue
    if 0 <= fv <= 50: by_date[d].append((tk.upper(), fv))   # clip FINRA OTC junk (999.99)

dates = sorted(by_date)
gaps = [(dates[i+1]-dates[i]).days for i in range(len(dates)-1)]
avg_gap = int(np.median(gaps)) if gaps else 15
nwlag = max(1, int(math.ceil(a.hold / float(avg_gap))))

print("=" * 78)
print(f"SI BRICK -- does it survive FINRA's ~8 business-day dissemination lag?")
print("=" * 78)
print(f"  feature={a.feature}  hold={a.hold}d  dates={len(dates)}  "
      f"median gap={avg_gap}d  Newey-West lag={nwlag}")
print(f"  NOTE: FINRA publishes short interest ~8 BUSINESS DAYS after settlement.")
print(f"        Entry at lag=0 is NOT tradeable. lag>=8 is the honest test.")
print()
print(f"  {'entry lag':>10} {'mean IC':>9} {'NW-t':>7} {'IC IR':>7} {'dates':>6} {'right-sign':>11}   verdict")
print("  " + "-" * 74)

base = None
for lag_bd in [0, 5, 8, 10, 15]:
    ics = []
    for d in dates:
        s, r = [], []
        for tk, fv in by_date[d]:
            rr = fwd(tk, d, lag_bd, a.hold)
            if rr is None: continue
            s.append(fv); r.append(rr)
        if len(s) >= a.min_names:
            ic = spearman(s, r)
            if np.isfinite(ic): ics.append(ic)
    if len(ics) < 8:
        print(f"  {lag_bd:>8}bd  insufficient dates ({len(ics)})"); continue
    ics = np.asarray(ics)
    se  = nw_se(ics, nwlag)
    t   = ics.mean() / se if se and np.isfinite(se) else float("nan")
    ir  = ics.mean() / ics.std(ddof=1)
    if base is None: base = ics.mean()
    keep = 100 * ics.mean() / base if base else float("nan")
    v = "SURVIVES" if abs(t) >= 2.0 and np.sign(ics.mean()) == np.sign(base) else "DEAD"
    tag = "  <- as published (NOT tradeable)" if lag_bd == 0 else \
          "  <- earliest tradeable" if lag_bd == 8 else ""
    print(f"  {lag_bd:>8}bd  {ics.mean():>+9.4f} {t:>+7.2f} {ir:>+7.3f} {len(ics):>6} "
          f"{100*np.mean(ics<0):>10.0f}%   {v} ({keep:.0f}% of edge){tag}")

print()
print("=" * 78)
print("  IC holds at lag>=8  -> brick is REAL and tradeable; look-ahead was not the source.")
print("  IC collapses        -> the edge lived in the pre-publication window. Not tradeable.")
print("=" * 78)
