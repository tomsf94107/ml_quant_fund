#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — LAZY PRICES VALIDATOR (monthly bucketing + measure select)
================================================================================
Tests whether YoY filing-text similarity predicts forward returns on YOUR universe,
using the SAME audited machinery as the short-interest brick (per-date Spearman IC,
Newey-West t for overlap, null control, per-year + OOS split).

KEY FIX vs the skeleton: filings scatter across ~3000 dates, so exact-date cross-
sections are far too thin (~5 names). This buckets into MONTH-END cross-sections:
for each calendar month, take every firm that filed a 10-K/10-Q THAT month, rank them
by similarity, and measure the forward return FROM MONTH-END (strictly after all those
filings were public -> PIT-clean). That yields ~20-40 names/month -> a real IC test.

MEASURE SELECT: filing_similarity.db now has similarity (=Jaccard, the measure with
real spread), plus jaccard, tfidf_cosine, raw_cosine columns. --measure picks which.
Lazy Prices direction = +1 (HIGH similarity -> HIGH return; rewriters underperform).

RULE 1: month-end forward returns strictly after the filing month; per-date IC + NW;
null control (shuffle returns within each month -> IC must vanish); READ-ONLY.

USAGE:
  python validate_lazy_prices_monthly.py --root . --measure jaccard --hold 40
  python validate_lazy_prices_monthly.py --root . --measure jaccard --hold 60
  python validate_lazy_prices_monthly.py --root . --measure tfidf_cosine --hold 40
================================================================================
"""
import argparse, os, sqlite3, math, datetime
from collections import defaultdict
import numpy as np

def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def cols_of(c,t): return [r[1] for r in Q(c,'PRAGMA table_info("'+t+'")')]
def nd(s):
    if s is None: return None
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None
def month_end(d):
    nxt=datetime.date(d.year+(1 if d.month==12 else 0),(d.month%12)+1,1)
    return nxt-datetime.timedelta(days=1)
def spearman(x,y):
    n=len(x)
    if n<5: return None
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return None
    return float(np.corrcoef(rx,ry)[0,1])
def nw_se(x,lag):
    x=np.asarray(x,float); n=len(x)
    if n<2: return None
    e=x-x.mean(); g0=float(e@e)/n; s=g0
    for k in range(1,min(lag,n-1)+1):
        gk=float(e[k:]@e[:-k])/n; w=1.0-k/(lag+1.0); s+=2.0*w*gk
    return math.sqrt(s/n) if s>0 else None
LINE="="*78

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--db",default=None)
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--measure",default="similarity",
                    help="similarity | jaccard | tfidf_cosine | raw_cosine")
    ap.add_argument("--form",default=None,help="restrict to 10-K or 10-Q")
    ap.add_argument("--direction",type=int,default=1)
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--min-names",type=int,default=20)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    sig_db=a.db or os.path.join(a.root,"filing_similarity.db")
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    print("\n"+LINE+"\nLAZY PRICES VALIDATOR — monthly cross-sections (measure=%s)\n"%a.measure+LINE)
    if not os.path.isfile(sig_db): print("  [STOP] filing_similarity.db not found"); return
    if not os.path.isfile(prices_db): print("  [STOP] prices.db not found"); return

    c=ro(sig_db)
    try:
        cols=cols_of(c,"filing_similarity")
        if a.measure not in cols:
            print("  [STOP] column '%s' not in db. Available: %s"%(a.measure,", ".join(cols))); c.close(); return
        if a.form:
            rows=Q(c,'SELECT ticker,filing_date,%s FROM filing_similarity WHERE form=? AND %s IS NOT NULL'%(a.measure,a.measure),(a.form,))
        else:
            rows=Q(c,'SELECT ticker,filing_date,%s FROM filing_similarity WHERE %s IS NOT NULL'%(a.measure,a.measure))
    finally: c.close()

    # bucket: month-end -> {ticker: similarity} (if a firm filed twice in a month, keep the latest-dated value)
    bucket=defaultdict(dict); latest=defaultdict(dict)
    for tk,fd,val in rows:
        do=nd(fd)
        if do is None or val is None: continue
        try: v=float(val)
        except Exception: continue
        me=month_end(do); tku=tk.upper()
        if tku not in latest[me] or do>latest[me][tku]:
            latest[me][tku]=do; bucket[me][tku]=v
    n_months=len(bucket)
    avg_names=np.mean([len(v) for v in bucket.values()]) if bucket else 0
    print("  %d monthly cross-sections, avg %.1f firms/month, %s..%s"
          %(n_months,avg_names,min(bucket) if bucket else "-",max(bucket) if bucket else "-"))

    # prices
    cp=ro(prices_db)
    try: prows=Q(cp,"SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL")
    finally: cp.close()
    px=defaultdict(list)
    for tk,d,p in prows:
        do=nd(d)
        if do is None: continue
        try: pf=float(p)
        except Exception: continue
        if pf>0: px[tk].append((do,pf))
    for tk in px: px[tk].sort()
    pos_of={tk:{d:i for i,(d,_) in enumerate(lst)} for tk,lst in px.items()}
    def fwd(tk,d,h):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,8):
            cc=d+datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+h
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None

    lag=max(1,int(math.ceil(a.hold/21.0)))  # monthly rebalance -> ~1 lag per ~21 trading-day month in hold
    def compute(shuffle=False,rng=None):
        ics=[]; dts=[]
        for me in sorted(bucket):
            recs=[]
            for tk,v in bucket[me].items():
                r=fwd(tk,me,a.hold)
                if r is None: continue
                recs.append((v,r))
            if len(recs)<a.min_names: continue
            sig=np.array([v for v,_ in recs])*a.direction
            ret=np.array([r for _,r in recs])
            if shuffle: ret=rng.permutation(ret)
            ic=spearman(sig,ret)
            if ic is not None: ics.append(ic); dts.append(me)
        return np.array(ics),dts

    ics,dts=compute(); N=len(dts)
    if N<8:
        print("\n  [STOP] only %d usable monthly cross-sections (need >=%d firms each)."%(N,a.min_names))
        print("  Lower --min-names or check coverage. avg firms/month was %.1f."%avg_names); return
    mean_ic=ics.mean(); se=nw_se(ics,lag); t=mean_ic/se if se else 0
    print("\n"+"-"*78+"\nPER-MONTH IC (measure=%s, direction=%+d, hold=%dd)\n"%(a.measure,a.direction,a.hold)+"-"*78)
    print("  mean IC = %+.4f | std = %.4f | IC IR = %+.3f"%(mean_ic,ics.std(),mean_ic/ics.std() if ics.std()>0 else 0))
    print("  %%-right-sign = %.0f%% | naive t = %+.2f | Newey-West t = %+.2f"
          %(100*np.mean(ics>0),mean_ic/(ics.std(ddof=1)/math.sqrt(N)) if ics.std(ddof=1)>0 else 0,t))

    # per-year
    yr=defaultdict(list)
    for ic,d in zip(ics,dts): yr[d.year].append(ic)
    print("\n  per-year mean IC:")
    for y in sorted(yr):
        a_=np.array(yr[y]); print("   %d: %+.4f  (n=%d)"%(y,a_.mean(),len(a_)))

    # OOS split
    half=N//2
    fh=ics[:half]; sh=ics[half:]
    def tof(x): 
        s=nw_se(x,lag); return x.mean()/s if s else 0
    print("\n  first half IC=%+.4f t=%+.2f (n=%d) | second half IC=%+.4f t=%+.2f (n=%d)"
          %(fh.mean(),tof(fh),len(fh),sh.mean(),tof(sh),len(sh)))

    # null control
    rng=np.random.default_rng(7); nulls=[]
    for _ in range(300):
        nc,_=compute(shuffle=True,rng=rng)
        if len(nc): nulls.append(nc.mean())
    nulls=np.array(nulls); z=(mean_ic-nulls.mean())/nulls.std() if nulls.std()>0 else 0
    print("\n  null control: real IC %.1f std's from shuffled null (need >=3)"%z)

    print("\n"+LINE+"\nVERDICT — is filing similarity (%s) a brick on your universe?\n"%a.measure+LINE)
    if abs(z)>=3 and abs(t)>=2.5 and np.mean(ics>0)>0.55:
        print("  >> CANDIDATE BRICK: IC %+.4f, NW t %+.2f, %.1f std's from null, %.0f%% right-sign."
              %(mean_ic,t,z,100*np.mean(ics>0)))
        print("     Direction %+d means %s. NEXT: decorrelation vs SI + momentum (the real point --"
              %(a.direction,"high similarity predicts high return (Lazy Prices holds)" if a.direction>0 else "low similarity predicts high return"))
        print("     is it the UNCORRELATED axis the combination thesis needed?). Then OOS + cost.")
    elif abs(z)<3:
        print("  >> NOT A BRICK (measure=%s): real IC within the shuffled null (%.1f std's). No edge here."%(a.measure,z))
        print("     Try --measure jaccard if you used cosine (cosine saturates on near-boilerplate filings),")
        print("     or --hold 60 (Lazy Prices is a slow, ~quarterly signal).")
    else:
        print("  >> SUGGESTIVE but short of bar: IC %+.4f (NW t %+.2f, null %.1f). Try --hold 60, the other"%(mean_ic,t,z))
        print("     measure, or split by form (--form 10-K). Lazy Prices is strongest in annual reports.")
    print("\n  Honest n=%d monthly cross-sections, %d firms/month avg. In-sample, survivor-tilted."%(N,avg_names))

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
