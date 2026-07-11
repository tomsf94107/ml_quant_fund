#!/usr/bin/env python3
"""POSITIVE CONTROL: does SUE predict the announcement-day jump?
If it doesn't, our SUE is broken and the null PEAD result means nothing."""
import sqlite3, math, numpy as np
from collections import defaultdict
from datetime import datetime, date

def nd(x):
    try: return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()
    except Exception: return None

def spear(a, b):
    ra = np.argsort(np.argsort(np.asarray(a,float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b,float))).astype(float)
    if ra.std()<1e-12 or rb.std()<1e-12: return np.nan
    return float(np.corrcoef(ra,rb)[0,1])

px = defaultdict(list)
pc = sqlite3.connect("prices.db")
for tk,d,c in pc.execute("SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL"):
    dd = nd(d)
    if dd: px[tk].append((dd,float(c)))
pc.close()
for tk in px: px[tk].sort()

def ret(tk,d,i0,i1):
    ser=px.get(tk)
    if not ser: return None
    lo,hi=0,len(ser)-1
    while lo<hi:
        m=(lo+hi)//2
        if ser[m][0]<d: lo=m+1
        else: hi=m
    if not ser or ser[lo][0]<d: return None
    s,e=lo+i0,lo+i1
    if s<0 or e>=len(ser): return None
    p0=ser[s][1]
    return ser[e][1]/p0-1.0 if p0>0 else None

ec=sqlite3.connect("earnings.db")
rows=ec.execute("""SELECT ticker,announce_date,eps_actual,eps_estimate FROM earnings_events
   WHERE eps_actual IS NOT NULL AND eps_estimate IS NOT NULL ORDER BY ticker,announce_date""").fetchall()
ec.close()
by=defaultdict(list)
for tk,ad,a_,e_ in rows:
    d=nd(ad)
    if d and d<=date.today(): by[tk].append((d,float(a_)-float(e_)))
ev=[]
for tk,l in by.items():
    l.sort(); pr=[]
    for d,raw in l:
        if len(pr)>=4:
            sd=np.std(pr,ddof=1)
            if sd>1e-12: ev.append((d,tk,raw/sd))
        pr.append(raw)
bd=defaultdict(list)
for d,tk,s in ev: bd[d].append((tk,s))

WIN=[("JUMP   ann-1 .. ann+1  (POSITIVE CONTROL)",-1,1),
     ("DRIFT  ann+2 .. ann+22 (~1 month)",2,22),
     ("DRIFT  ann+2 .. ann+42 (~2 months)",2,42),
     ("DRIFT  ann+2 .. ann+62 (~3 months)",2,62)]
print("="*76)
print("PEAD POSITIVE CONTROL -- does SUE predict the announcement jump?")
print("="*76)
for lbl,i0,i1 in WIN:
    ics=[]
    for d in sorted(bd):
        S,R=[],[]
        for tk,s in bd[d]:
            r=ret(tk,d,i0,i1)
            if r is None or abs(r)>2.0: continue
            S.append(s); R.append(r)
        if len(S)>=10:
            ic=spear(S,R)
            if np.isfinite(ic): ics.append(ic)
    ics=np.asarray(ics)
    if len(ics)<10: print(f"  {lbl:42s} too few dates"); continue
    t=ics.mean()/(ics.std(ddof=1)/math.sqrt(len(ics)))
    print(f"  {lbl:42s} IC={ics.mean():+.4f}  naive t={t:+6.2f}  dates={len(ics)}  +sign={100*np.mean(ics>0):.0f}%")
print("="*76)
print("  JUMP IC strongly POSITIVE -> SUE is valid. Flat drift = PEAD is genuinely dead.")
print("  JUMP IC ~0               -> SUE is BROKEN. The null result proves nothing.")
print("="*76)
