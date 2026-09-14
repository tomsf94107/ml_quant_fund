#!/usr/bin/env python3
"""nw_ttest.py — Newey-West HAC t on per-rebalance cap-3 excess + prob-gate count diagnosis.
Reads the CSV dumped by h40_yearly_test.py (ML_QUANT_DUMP=path).
Cols: seed,date,year,n,prob_n,cap3_excess,prob_excess
"""
import argparse, csv, math, statistics as st
from collections import defaultdict

def nw(x, lag):
    T=len(x)
    if T<3: return float('nan'),float('nan'),T
    m=sum(x)/T; d=[xi-m for xi in x]
    g0=sum(v*v for v in d)/T; lrv=g0
    for k in range(1,min(lag,T-1)+1):
        gk=sum(d[t]*d[t-k] for t in range(k,T))/T
        lrv+=2*(1-k/(lag+1))*gk
    if lrv<=0: lrv=g0
    se=math.sqrt(lrv/T)
    return m,(m/se if se>0 else float('nan')),T

def plain(x):
    T=len(x); m=sum(x)/T; var=sum((xi-m)**2 for xi in x)/(T-1)
    se=math.sqrt(var/T); return m/se if se>0 else float('nan')

def series_by_date(rows,col):
    bd=defaultdict(list)
    for r in rows:
        try: bd[r["date"]].append(float(r[col]))
        except (ValueError,KeyError): pass
    ds=sorted(bd); return [sum(bd[d])/len(bd[d]) for d in ds]

a=argparse.ArgumentParser()
a.add_argument("--csv",required=True); a.add_argument("--lag",type=int,default=40)
a.add_argument("--exclude-2020",action="store_true"); a.add_argument("--value-col",default="cap3_excess")
A=a.parse_args()
rows=list(csv.DictReader(open(A.csv)))
if A.exclude_2020: rows=[r for r in rows if r.get("year")!="2020"]
if not rows: raise SystemExit("no rows after filter")

ser=series_by_date(rows,A.value_col)
m,t,T=nw(ser,A.lag)
print("=== cap-3 excess significance (pooled per-date, seeds averaged) ===")
print(f"  dates={T}  mean={m*100:+.3f}pp  plain-t={plain(ser):+.2f}  NW-t(lag{A.lag})={t:+.2f}")
print("  per-seed NW-t:", end=" ")
for s in sorted({r['seed'] for r in rows}):
    ss=series_by_date([r for r in rows if r['seed']==s],A.value_col)
    print(f"S{s}={nw(ss,A.lag)[1]:+.2f}",end="  ")
print()
if rows and "prob_n" in rows[0]:
    c=[int(r["prob_n"]) for r in rows if r["prob_n"].strip().lstrip('-').isdigit()]
    if c:
        c.sort(); z=sum(1 for x in c if x==0); l3=sum(1 for x in c if x<3)
        print("\n=== prob>=0.70 gate: names/day ===")
        print(f"  days={len(c)}  mean={st.mean(c):.1f}  median={c[len(c)//2]}  "
              f"0-name={z}({100*z/len(c):.0f}%)  <3={l3}({100*l3/len(c):.0f}%)")
        print("  -> high % with <3 names => the prob>=0.70 excess is small-sample noise")
