#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — SHORT-INTEREST VALIDATOR (self-contained, reads ONLY short_interest.db)
================================================================================
Purpose-built to validate days_to_cover / current_short from the FINRA backfill,
with NO auto-search across other DBs (so the old accuracy.db stub can't hijack it).
Same PEAD-grade battery as validate_signal.py: pooled IC, beta-strip, sign-only,
OOS temporal split, decile monotonicity. Reads forward returns from prices.db.

RULE 1: short-side feature auto-negated (high short interest -> low return).
Forward return strictly AFTER the settlement date. READ-ONLY (mode=ro). No network.

USAGE:
  python validate_si.py --root .                       # days_to_cover, h=40 and h=20
  python validate_si.py --root . --feature days_to_cover --hold 40
  python validate_si.py --root . --feature current_short --hold 20
  python validate_si.py --root . --clip-dtc 50         # clip the 999.99 OTC placeholder
================================================================================
"""
import argparse, os, sqlite3, math, datetime
from collections import defaultdict
import numpy as np

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)
def ro(p):
    return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
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

def run_one(px,pos_of,si_db,feature,hold,cost_bps,min_names,clip_dtc):
    banner("SHORT-INTEREST VALIDATOR: %s  (h=%d)"%(feature,hold))
    c=ro(si_db)
    try:
        cols=[r[1] for r in Q(c,'PRAGMA table_info("short_interest")')]
        if feature not in cols:
            print("  [STOP] '%s' not in short_interest.db. Columns: %s"%(feature,cols)); return
        rows=Q(c,'SELECT ticker,settlement_date,"%s" FROM short_interest'%feature)
    finally:
        c.close()
    panel={}
    for tk,d,v in rows:
        do=nd(d)
        if do is None or v is None: continue
        try: fv=float(v)
        except Exception: continue
        if clip_dtc and feature in ("days_to_cover",) and fv>clip_dtc:  # drop 999.99 OTC placeholder
            continue
        panel[(do,tk.upper())]=fv
    if not panel:
        print("  [STOP] no usable rows for '%s'"%feature); return
    dates=sorted(set(d for d,_ in panel))
    print("  feature '%s' from short_interest.db: %d points, %s..%s"%(feature,len(panel),dates[0],dates[-1]))
    span_days=(dates[-1]-dates[0]).days

    auto_neg = any(k in feature.lower() for k in ["short","days_to_cover","dtc"])
    sign=-1 if auto_neg else 1
    if sign<0: print("  (NEGATED: higher %s -> lower return)"%feature)

    def fwd(tk,d,h):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,6):
            cc=d+datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+h
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None

    recs=[]
    for (d,tk),v in panel.items():
        r=fwd(tk,d,hold)
        if r is not None: recs.append((d,sign*v,r))
    print("  records with %d-day forward return: %d"%(hold,len(recs)))
    if len(recs)<min_names*3:
        print("  [STOP] too few records (%d)"%len(recs)); return

    cost=cost_bps/10000.0
    def metric(rws,beta_strip=False,sign_only=False):
        if len(rws)<min_names: return None
        ym=defaultdict(list)
        for d,v,r in rws: ym[d].append(r)
        dm={k:np.mean(v) for k,v in ym.items()}
        s=[]; rr=[]
        for d,v,r in rws:
            sv=(1.0 if v>0 else (-1.0 if v<0 else 0.0)) if sign_only else v
            ret=(r-dm[d]) if beta_strip else r
            s.append(sv); rr.append(ret)
        n=len(s); ic=spearman(s,rr)
        order=np.argsort(s); q=max(1,n//5); lo=order[:q]; hi=order[-q:]
        L=np.mean([rr[i] for i in hi]); S=np.mean([rr[i] for i in lo])
        g=L-S; net=g-2*cost
        sd=math.sqrt(np.var([rr[i] for i in hi])/q+np.var([rr[i] for i in lo])/q)
        t=g/sd if sd>0 else None
        dmn=[]
        for dd in range(10):
            idx=order[int(dd*n/10):int((dd+1)*n/10)]
            if len(idx)>0: dmn.append(np.mean([rr[i] for i in idx]))
        ups=sum(1 for i in range(1,len(dmn)) if dmn[i]>=dmn[i-1])
        return {"n":n,"ic":ic,"net":net,"t":t,"mono":[ups,len(dmn)-1]}

    def show(label,m):
        if not m: print("  %-18s n/a"%label); return
        print("  %-18s n=%-5d IC=%-8s net=%-8s t=%-6s mono=%d/%d"
              %(label,m["n"],"%+.4f"%m["ic"] if m["ic"] is not None else "NA",
                "%+.4f"%m["net"],"%.2f"%m["t"] if m["t"] else "NA",m["mono"][0],m["mono"][1]))

    sub("POOLED + robustness")
    pooled=metric(recs); beta=metric(recs,beta_strip=True); sgn=metric(recs,sign_only=True)
    show("pooled",pooled); show("beta-stripped",beta); show("sign-only",sgn)

    sub("OUT-OF-SAMPLE holdout (temporal split)")
    mid=dates[len(dates)//2]
    train=[r for r in recs if r[0]<mid]; test=[r for r in recs if r[0]>=mid]
    tr=metric(train); te=metric(test)
    show("IN-SAMPLE",tr); show("OUT-OF-SAMPLE",te)

    banner("VERDICT — is '%s' a real brick? (history: %d days, %.1f yr)"%(feature,span_days,span_days/365.0))
    short_hist = span_days<300
    if pooled and pooled["ic"] is not None:
        strong = abs(pooled["ic"])>=0.03 and abs(pooled["t"] or 0)>=2
        beta_ok = beta and abs(beta["ic"] or 0)>=0.02
        oos_ok = te and (te["ic"] is not None) and abs(te["ic"])>=0.02 and (np.sign(te["ic"])==np.sign(pooled["ic"]))
        sign_ok = sgn and (sgn["ic"] is not None) and abs(sgn["t"] or 0)>=1.5 and np.sign(sgn["ic"])==np.sign(pooled["ic"])
        print("  pooled IC=%+.4f t=%.2f | beta IC=%+.4f | OOS IC=%s | sign-only t=%s"
              %(pooled["ic"],pooled["t"] or 0,(beta["ic"] or 0) if beta else 0,
                "%+.4f"%te["ic"] if te and te["ic"] is not None else "NA",
                "%.2f"%(sgn["t"] or 0) if sgn else "NA"))
        if strong and beta_ok and oos_ok and not short_hist:
            print("  >> CONFIRMED BRICK: significant, survives beta-strip, holds OOS, %.1f-yr history."%(span_days/365.0))
            if sign_ok:
                print("     PLUS sign-only survives (t=%.2f) -> BROAD signal, not tail-driven. Strong brick."%(sgn["t"] or 0))
            else:
                print("     NOTE: sign-only weak (t=%.2f) -> edge is magnitude-concentrated (like PEAD). Real but"%(sgn["t"] or 0 if sgn else 0))
                print("     depends on the extreme-short names; size accordingly.")
        elif strong and beta_ok and oos_ok and short_hist:
            print("  >> STRONG LEAD (short history): all pass but only %.1f yr. Confirm with more."%(span_days/365.0))
        elif strong and (not oos_ok):
            print("  >> MIXED: pooled significant but OOS does NOT hold (OOS IC=%s). Possible regime/recency"
                  %("%+.4f"%te["ic"] if te and te["ic"] is not None else "NA"))
            print("     dependence. Not a clean brick; investigate before trusting.")
        elif strong:
            print("  >> PARTIAL: significant pooled but fails a robustness check (beta or OOS). Scrutinize.")
        else:
            print("  >> WEAK/NULL: not significant (pooled t=%.2f). Not a brick on this evidence."%(pooled["t"] or 0))
            print("     The 3-month blip did NOT survive on 5-yr data -> lead honestly closed.")
    print("\n  This validator reads ONLY short_interest.db (no auto-search). Same battery as PEAD.")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--si-db",default=None)
    ap.add_argument("--feature",default=None,help="default: runs days_to_cover at h=40 and h=20")
    ap.add_argument("--hold",type=int,default=None)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--min-names",type=int,default=15)
    ap.add_argument("--clip-dtc",type=float,default=50.0,help="drop days_to_cover above this (999.99 OTC placeholder)")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    si_db=a.si_db or os.path.join(a.root,"short_interest.db")
    if not os.path.isfile(prices_db): print("[STOP] prices.db not found at %s"%prices_db); return
    if not os.path.isfile(si_db): print("[STOP] short_interest.db not found at %s"%si_db); return

    cp=ro(prices_db)
    try:
        rows=Q(cp,"SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL")
    finally:
        cp.close()
    px=defaultdict(list)
    for tk,d,p in rows:
        do=nd(d)
        if do is None: continue
        try: pf=float(p)
        except Exception: continue
        if pf>0: px[tk].append((do,pf))
    for tk in px: px[tk].sort()
    pos_of={tk:{d:i for i,(d,_) in enumerate(lst)} for tk,lst in px.items()}
    print("prices loaded for %d tickers"%len(px))

    if a.feature:
        holds=[a.hold] if a.hold else [40]
        for h in holds:
            run_one(px,pos_of,si_db,a.feature,h,a.cost_bps,a.min_names,a.clip_dtc)
    else:
        for h in (40,20):
            run_one(px,pos_of,si_db,"days_to_cover",h,a.cost_bps,a.min_names,a.clip_dtc)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
