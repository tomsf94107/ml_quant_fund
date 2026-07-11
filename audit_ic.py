#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — AUDIT SCRIPT 2: VERIFY THE SECTOR-NEUTRAL / IC NUMBERS
================================================================================
Independently recomputes the short-interest IC claims, CROSS-CHECKING each number
two different ways so you can see the math agrees. Verifies:

  1. Raw per-date IC (mean, std, IR) — recomputed, with the IC series printed
  2. Newey-West t — recomputed via an EXPLICIT manual autocovariance sum, and
     cross-checked against a second implementation. If the two disagree, FAIL.
  3. Sector-neutral IC — recomputed; retention = neutral/raw shown with arithmetic
  4. A NULL control: shuffle returns within each date -> IC must collapse to ~0
     (proves the pipeline isn't manufacturing signal)
  5. The naive-vs-NW gap — shows how much the autocorrelation correction matters

The NULL CONTROL (#4) is the key honesty check: if we randomly permute the forward
returns within each date, ANY real signal must vanish (IC->0, t->0). If it doesn't,
the measurement is broken. This is the analog of the label-shuffle leak test.

RULE 1: prints the per-date IC series and intermediate sums; two independent NW
implementations must agree; null control must collapse. READ-ONLY.

USAGE:
  python audit_ic.py --root .
  python audit_ic.py --root . --hold 40
================================================================================
"""
import argparse, os, sqlite3, csv, math, datetime
from collections import defaultdict
import numpy as np

def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def nd(s):
    if s is None: return None
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None
def spearman(x,y):
    n=len(x)
    if n<5: return None
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return None
    return float(np.corrcoef(rx,ry)[0,1])
LINE="="*78

def nw_v1(x,lag):
    """Implementation A: explicit loop."""
    x=np.asarray(x,float); n=len(x); e=x-x.mean()
    g0=float(e@e)/n; s=g0
    for k in range(1,min(lag,n-1)+1):
        gk=float(e[k:]@e[:-k])/n; w=1.0-k/(lag+1.0); s+=2.0*w*gk
    return math.sqrt(s/n) if s>0 else None

def nw_v2(x,lag):
    """Implementation B: build autocovariance vector first, independent code path."""
    x=np.asarray(x,float); n=len(x); m=x.mean()
    acov=[]
    for k in range(0,lag+1):
        tot=0.0
        for t in range(k,n):
            tot+=(x[t]-m)*(x[t-k]-m)
        acov.append(tot/n)
    var=acov[0]
    for k in range(1,lag+1):
        var+=2.0*(1.0-k/(lag+1.0))*acov[k]
    return math.sqrt(var/n) if var>0 else None

def load_sectors(root):
    for p in (os.path.join(root,"tickers_metadata.csv"),):
        if os.path.isfile(p):
            with open(p,newline="") as f:
                rd=csv.DictReader(f); cols={c.lower():c for c in (rd.fieldnames or [])}
                tkey=next((cols[k] for k in ("ticker","symbol") if k in cols),None)
                skey=next((cols[k] for k in ("sector","gics_sector","industry","bucket","group") if k in cols),None)
                if tkey and skey:
                    m={}
                    for row in rd:
                        tk=(row.get(tkey) or "").strip().upper(); sec=(row.get(skey) or "").strip()
                        if tk and sec: m[tk]=sec
                    return m,skey
    return None,None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--min-names",type=int,default=15)
    ap.add_argument("--clip-dtc",type=float,default=50.0)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=os.path.join(a.root,"prices.db"); si_db=os.path.join(a.root,"short_interest.db")
    print("\n"+LINE+"\nAUDIT 2 — VERIFYING THE IC / SECTOR-NEUTRAL NUMBERS (cross-checked)\n"+LINE)

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
    def fwd(tk,d):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,6):
            cc=d+datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+a.hold
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None

    c=ro(si_db)
    try: rows=Q(c,"SELECT ticker,settlement_date,days_to_cover FROM short_interest")
    finally: c.close()
    by_date=defaultdict(list)
    for tk,d,v in rows:
        do=nd(d)
        if do is None or v is None: continue
        try: fv=float(v)
        except Exception: continue
        if a.clip_dtc and fv>a.clip_dtc: continue
        by_date[do].append((tk.upper(),fv))

    sectors,skey=load_sectors(a.root)

    # build per-date IC (raw + sector-neutral) + record (signal, return) for null test
    def per_date(neutralize, shuffle=False, rng=None):
        ic_series=[]
        for d in sorted(by_date):
            recs=[]
            for tk,v in by_date[d]:
                r=fwd(tk,d)
                if r is None: continue
                sec=sectors.get(tk,"_U") if sectors else "_A"
                recs.append((sec,v,r))
            if len(recs)<a.min_names: continue
            sig=np.array([v for _,v,_ in recs]); ret=np.array([r for _,_,r in recs])
            secs=[s for s,_,_ in recs]
            if shuffle: ret=rng.permutation(ret)
            if neutralize and sectors:
                bysec=defaultdict(list)
                for i,s in enumerate(secs): bysec[s].append(i)
                sg=np.full(len(recs),np.nan); rt=np.zeros(len(recs))
                for s,idxs in bysec.items():
                    if len(idxs)<2: continue
                    vm=np.mean([sig[i] for i in idxs]); rm=np.mean([ret[i] for i in idxs])
                    for i in idxs: sg[i]=sig[i]-vm; rt[i]=ret[i]-rm
                mask=~np.isnan(sg); sig=sg[mask]; ret=rt[mask]
            if len(sig)>=a.min_names:
                ic=spearman(sig,ret)
                if ic is not None: ic_series.append((d,ic))
        return ic_series

    raw=per_date(False); ics=np.array([ic for _,ic in raw]); N=len(ics)
    mean_ic=ics.mean(); std_ic=ics.std(ddof=1); ir=mean_ic/std_ic
    lag=max(1,int(math.ceil(a.hold/15.0)))

    print("\n"+"-"*78+"\nSTEP 1: raw per-date IC — arithmetic\n"+"-"*78)
    print("  N dates=%d"%N)
    print("  sum of ICs=%.5f  mean=sum/N=%.5f"%(ics.sum(),mean_ic))
    print("  std(ddof=1)=%.5f  IC IR=mean/std=%.4f"%(std_ic,ir))
    print("  first 8 per-date ICs:", ", ".join("%+.3f"%x for x in ics[:8]))

    print("\n"+"-"*78+"\nSTEP 2: Newey-West t — TWO independent implementations must agree\n"+"-"*78)
    se1=nw_v1(ics,lag); se2=nw_v2(ics,lag)
    print("  NW lag=%d (=ceil(%d/15))"%(lag,a.hold))
    print("  implementation A (vectorized): SE=%.6f"%se1)
    print("  implementation B (explicit loop): SE=%.6f"%se2)
    agree=abs(se1-se2)<1e-9
    print("  AGREE (diff<1e-9): %s  %s"%(agree,"PASS" if agree else "*** FAIL — bug ***"))
    t_nw=mean_ic/se1; t_naive=mean_ic/(std_ic/math.sqrt(N))
    print("  naive t = mean/(std/sqrt(N)) = %.4f/(%.5f/%.3f) = %.3f"%(mean_ic,std_ic,math.sqrt(N),t_naive))
    print("  Newey-West t = mean/NW_SE = %.4f/%.6f = %.3f"%(mean_ic,se1,t_nw))
    print("  -> NW correction changes t by %.1f%%"%(100*(t_nw/t_naive-1)))

    if sectors:
        print("\n"+"-"*78+"\nSTEP 3: sector-neutral IC + retention — arithmetic\n"+"-"*78)
        neu=per_date(True); nics=np.array([ic for _,ic in neu]); Nn=len(nics)
        nmean=nics.mean(); nstd=nics.std(ddof=1); nse=nw_v1(nics,lag); nt=nmean/nse
        print("  sector source: tickers_metadata.csv col '%s' (%d sectors)"%(skey,len(set(sectors.values()))))
        print("  neutral N dates=%d  mean IC=%.5f  NW t=%.3f"%(Nn,nmean,nt))
        print("  retention = neutral_mean / raw_mean = %.5f / %.5f = %.1f%%"%(nmean,mean_ic,100*nmean/mean_ic))
    else:
        print("\n  [STEP 3 skipped — no sector file found]")

    print("\n"+"-"*78+"\nSTEP 4: NULL CONTROL — shuffle returns within each date, IC must -> 0\n"+"-"*78)
    rng=np.random.default_rng(7)
    null_means=[]
    for _ in range(200):
        ns=per_date(False, shuffle=True, rng=rng)
        if ns: null_means.append(np.mean([ic for _,ic in ns]))
    null_means=np.array(null_means)
    print("  ran 200 within-date shuffles of the forward returns")
    print("  null mean IC: avg=%.5f  std=%.5f  (should be ~0)"%(null_means.mean(),null_means.std()))
    print("  real mean IC = %.5f"%mean_ic)
    z=(mean_ic-null_means.mean())/null_means.std() if null_means.std()>0 else 0
    print("  real IC is %.1f null-std's from 0"%z)
    if abs(z)>=3:
        print("  >> PASS: real signal is far outside the shuffled-null distribution. Not an artifact.")
    else:
        print("  >> WARNING: real signal is close to the null -> weak/possibly artifactual.")

    print("\n"+LINE+"\nAUDIT 2 CONCLUSION\n"+LINE)
    print("  STEP 2 (two NW implementations agree) verifies the t-stat math.")
    print("  STEP 4 (null control collapses) verifies the pipeline isn't manufacturing signal.")
    print("  If both pass and retention (STEP 3) is high, the sector-neutral brick is sound.")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
