#!/usr/bin/env python3
"""
pead_lookahead_v2.py -- Does PEAD earn BEFORE the number was public?

v1 failed: cross-sectional IC needs many names per event-date, but events are
sparse (~2/date). v2 pools all events and tests in EVENT TIME with a
date-clustered t-stat -- the right design when breadth is thin.

report_date in earnings_surprises is a FISCAL PERIOD END (75% land on a
quarter-end; 35% on a weekend; MU's row says 2026-05-31 but MU announced ~Jun 25).
The SUE is not knowable until the announcement, ~fe+21..fe+45.

  PRE   fe+2..fe+20   signal NOT public -> a spread here can ONLY be look-ahead
  ANN   fe+21..fe+45  the announcement jump
  POST  fe+46..fe+85  genuine drift

Also prints WHY the event count is small.
"""
import sqlite3, argparse, numpy as np
from collections import defaultdict
from datetime import datetime

def nd(x):
    if x is None: return None
    try: return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()
    except Exception: return None

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
a = ap.parse_args()

# ---------- WHY SO FEW EVENTS? ----------
ec = sqlite3.connect(f"{a.root}/earnings.db")
tot, hasA, hasE, hasB = ec.execute("""
    SELECT COUNT(*),
           SUM(eps_actual   IS NOT NULL),
           SUM(eps_estimate IS NOT NULL),
           SUM(eps_actual IS NOT NULL AND eps_estimate IS NOT NULL)
    FROM earnings_surprises""").fetchone()
print("=" * 78)
print("WHY IS THE EVENT COUNT SMALL?")
print("=" * 78)
print(f"  rows total ................ {tot:,}")
print(f"  with eps_actual ........... {hasA:,}  ({100*hasA/tot:.1f}%)")
print(f"  with eps_estimate ......... {hasE:,}  ({100*hasE/tot:.1f}%)  <- the binding one")
print(f"  with BOTH (usable) ........ {hasB:,}  ({100*hasB/tot:.1f}%)")
print(f"  ...then SUE needs 4 PRIOR quarters per ticker, cutting further.")
print()

rows = ec.execute("SELECT ticker, report_date, eps_actual, eps_estimate "
                  "FROM earnings_surprises WHERE eps_actual IS NOT NULL "
                  "AND eps_estimate IS NOT NULL AND report_date IS NOT NULL").fetchall()
ec.close()

# ---------- prices ----------
pc = sqlite3.connect(f"{a.root}/prices.db")
px = defaultdict(list)
for tk, d, c in pc.execute("SELECT ticker,date,adj_close FROM daily_prices "
                           "WHERE adj_close IS NOT NULL"):
    dd = nd(d)
    if dd: px[tk].append((dd, float(c)))
pc.close()
for tk in px: px[tk].sort()

def fwd(tk, fe, i0, i1):
    ser = px.get(tk)
    if not ser: return None
    lo, hi = 0, len(ser) - 1
    while lo < hi:
        m = (lo + hi) // 2
        if ser[m][0] < fe: lo = m + 1
        else: hi = m
    if ser[lo][0] < fe: return None
    s, e = lo + i0, lo + i1
    if e >= len(ser): return None
    p0 = ser[s][1]
    return ser[e][1] / p0 - 1.0 if p0 > 0 else None

# ---------- SUE (PIT trailing std) ----------
by = defaultdict(list)
for tk, rd, act, est in rows:
    fe = nd(rd)
    if fe is None: continue
    try: by[tk].append((fe, float(act) - float(est)))
    except Exception: pass

events = []
for tk, lst in by.items():
    lst.sort(); prior = []
    for fe, raw in lst:
        if len(prior) >= 4:
            sd = np.std(prior, ddof=1)
            if sd > 1e-12: events.append((fe, tk, raw / sd))
        prior.append(raw)

print(f"  usable SUE events ......... {len(events):,}   tickers={len({t for _,t,_ in events})}")
per = defaultdict(int)
for fe, _, _ in events: per[fe] += 1
print(f"  distinct fiscal-end dates . {len(per):,}   median events/date={int(np.median(list(per.values())))}")
print()

WIN = [("PRE   fe+2..fe+20   (NOT public)", 2, 20),
       ("ANN   fe+21..fe+45  (the jump)",  21, 45),
       ("POST  fe+46..fe+85  (drift)",     46, 85)]

print("=" * 78)
print("POOLED EVENT-TIME TEST -- SUE quintiles, t clustered by fiscal-end date")
print("=" * 78)

for label, i0, i1 in WIN:
    dat = []
    for fe, tk, sue in events:
        r = fwd(tk, fe, i0, i1)
        if r is not None and abs(r) < 2.0: dat.append((fe, sue, r))
    if len(dat) < 100:
        print(f"  {label:34s}  too few usable events (n={len(dat)})"); continue
    fes  = np.array([d[0] for d in dat])
    sue  = np.array([d[1] for d in dat], float)
    ret  = np.array([d[2] for d in dat], float)
    q    = np.quantile(sue, [0.2, 0.4, 0.6, 0.8])
    b    = np.digitize(sue, q)
    means = [ret[b == k].mean() * 100 for k in range(5)]
    spread_rows = ret[b == 4] - 0  # long top
    # date-clustered t on the Q5-Q1 spread: per-date spread, then t across dates
    per_date = []
    for fe in sorted(set(fes)):
        m = fes == fe
        if m.sum() < 5: continue
        s_, r_ = sue[m], ret[m]
        hi_ = r_[s_ >= np.quantile(s_, 0.6)].mean()
        lo_ = r_[s_ <= np.quantile(s_, 0.4)].mean()
        if np.isfinite(hi_) and np.isfinite(lo_): per_date.append(hi_ - lo_)
    pd_ = np.array(per_date)
    t   = (pd_.mean() / (pd_.std(ddof=1) / np.sqrt(len(pd_)))) if len(pd_) > 4 else float("nan")
    print(f"  {label}")
    print(f"      n={len(dat):,} events   Q1..Q5 mean ret (%): "
          + "  ".join(f"{m:+.2f}" for m in means))
    print(f"      Q5-Q1 = {means[4]-means[0]:+.2f}%   "
          f"date-clustered t = {t:+.2f}   (dates={len(pd_)})")
    print()

print("=" * 78)
print("  PRE spread ~0 / t not significant -> no exploitable look-ahead. Hypothesis dies.")
print("  PRE spread large & significant    -> the book earns on a signal that did not")
print("                                       exist yet. Re-key PEAD to announcement dates.")
print("=" * 78)
