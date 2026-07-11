#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — GENERIC SIGNAL VALIDATOR  (PEAD-grade rigor for any feature)
================================================================================
The non-price hunt surfaced leads (short_ratio, pc_ratio_snap, inst_signed_flow_5d).
This applies the SAME validation battery we used to lock PEAD, to any one of them:
  * in-sample vs out-of-sample holdout (does it survive cold?)
  * early vs late (decay?)
  * beta-strip (is it just market exposure?)
  * sign-only robustness floor
  * decile monotonicity

It runs on whatever history the feature has. For prediction_features signals that's
~3 months -> UNDERPOWERED. The verdict says "lead, needs more data" rather than
pretending a 3-month result is a locked brick.

RULE 1: forward return from prices.db strictly after the feature snapshot. Feature
as-of snapshot (PIT). OOS test never tunes on the holdout. Short history flagged loud.

READ-ONLY. mode=ro&immutable=1. No network.

USAGE:
  python validate_signal.py --root . --feature short_ratio
  python validate_signal.py --root . --feature pc_ratio_snap --hold 5
  python validate_signal.py --root . --feature inst_signed_flow_5d --hold 40
  (use --negate if higher feature should mean LOWER return; auto for short/skew)
================================================================================
"""
import argparse, os, sqlite3, sys, math, json, datetime
from collections import defaultdict
try:
    import numpy as np; HAVE_NUMPY=True
except Exception: HAVE_NUMPY=False

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)
def ro(p):
    if not os.path.isfile(p): raise FileNotFoundError(p)
    return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def has_table(c,n): return bool(Q(c,"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(n,)))
def tables(c): return [r[0] for r in Q(c,"SELECT name FROM sqlite_master WHERE type='table'")]
def cols_of(c,t): return [r[1] for r in Q(c,'PRAGMA table_info("'+t+'")')]
def require(cond,msg):
    if not cond: print("  [STOP] "+msg); return False
    return True
def all_dbs(root):
    out=[]
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f.endswith((".db",".sqlite",".sqlite3")): out.append(os.path.join(dp,f))
    return sorted(out)
def nd(s):
    if s is None: return None
    s=str(s)[:10]
    try: return datetime.date.fromisoformat(s)
    except Exception: return None
def spearman(x,y):
    n=len(x)
    if n<5: return None
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return None
    return float(np.corrcoef(rx,ry)[0,1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--feature",required=True)
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--min-names",type=int,default=15)
    ap.add_argument("--negate",action="store_true",help="flip sign (higher feature -> lower return)")
    ap.add_argument("--db",default=None,help="restrict feature search to this DB (filename or path)")
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    banner("ML QUANT FUND — GENERIC SIGNAL VALIDATOR: %s"%a.feature)
    print("PEAD-grade battery on '%s' at h=%d. (offline)"%(a.feature,a.hold))
    if not require(HAVE_NUMPY,"numpy required"): return
    if not require(os.path.isfile(prices_db),"prices.db not found"): return

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
    print("  prices loaded for %d tickers"%len(px))

    # locate feature — pick the source with the LONGEST history (not just the first found),
    # unless --db restricts the search to a specific database.
    panel={}
    src=None
    candidates=[]  # (span_days, n_points, src_label, panel_dict)
    db_list=all_dbs(a.root)
    if a.db:
        want=os.path.basename(a.db)
        db_list=[d for d in db_list if os.path.basename(d)==want or d==a.db]
        if not db_list:
            print("  [STOP] --db '%s' not found among scanned DBs"%a.db); return
    for dbp in db_list:
        try: c=ro(dbp)
        except Exception: continue
        try:
            for t in tables(c):
                cl=cols_of(c,t)
                if a.feature not in cl: continue
                tcol="ticker" if "ticker" in cl else ("symbol" if "symbol" in cl else None)
                dcol=next((x for x in ("prediction_date","date","as_of","report_date","settlement_date","updated_at") if x in cl),None)
                if not (tcol and dcol): continue
                data=Q(c,"SELECT "+tcol+","+dcol+',"'+a.feature+'" FROM "'+t+'"')
                pd={}
                for tk,d,v in data:
                    do=nd(d)
                    if do is None or v is None: continue
                    try: pd[(do,tk)]=float(v)
                    except Exception: pass
                if pd:
                    ds=sorted(set(d for d,_ in pd.keys()))
                    span=(ds[-1]-ds[0]).days
                    candidates.append((span,len(pd),"%s.%s"%(os.path.basename(dbp),t),pd))
        finally:
            c.close()
    if not require(candidates,"feature '%s' not found in any DB"%a.feature): return
    # choose the longest-history source (tie-break on point count)
    candidates.sort(key=lambda x:(x[0],x[1]),reverse=True)
    span_best,n_best,src,panel=candidates[0]
    if len(candidates)>1:
        others=", ".join("%s(%dd)"%(c[2],c[0]) for c in candidates[1:])
        print("  [multiple sources for '%s'; chose longest history. others: %s]"%(a.feature,others))
    dates=sorted(set(d for d,_ in panel.keys()))
    print("  feature '%s' from %s: %d points, %s..%s"%(a.feature,src,len(panel),dates[0],dates[-1]))
    span_days=(dates[-1]-dates[0]).days
    auto_neg = any(k in a.feature.lower() for k in ["short","days_to_cover","dtc","iv_skew"])
    sign = -1 if (a.negate or auto_neg) else 1
    if sign<0: print("  (using NEGATED feature: higher %s -> lower return, per sign convention)"%a.feature)

    def fwd(tk,d,hold):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,6):
            c=d+datetime.timedelta(days=off)
            if c in idx: i=idx[c]; break
        if i is None: return None
        x=i+hold
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None

    # records: (date, value, ret)
    recs=[]
    for (d,tk),v in panel.items():
        r=fwd(tk,d,a.hold)
        if r is not None: recs.append((d,sign*v,r))
    print("  records with %d-day forward return: %d"%(a.hold,len(recs)))
    if len(recs)<a.min_names*3:
        print("  [STOP] too few records (%d) to validate"%len(recs)); return

    cost=a.cost_bps/10000.0
    def metric(rows,beta_strip=False,sign_only=False):
        if len(rows)<a.min_names: return None
        # group by date for beta-strip mean
        ym=defaultdict(list)
        for d,v,r in rows: ym[d].append(r)
        dm={k:np.mean(v) for k,v in ym.items()}
        s=[]; rr=[]
        for d,v,r in rows:
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
        for d in range(10):
            idx=order[int(d*n/10):int((d+1)*n/10)]
            if len(idx)>0: dmn.append(np.mean([rr[i] for i in idx]))
        ups=sum(1 for i in range(1,len(dmn)) if dmn[i]>=dmn[i-1])
        return {"n":n,"ic":ic,"net":net,"t":t,"mono":[ups,len(dmn)-1]}

    # pooled
    sub("POOLED + robustness")
    pooled=metric(recs); beta=metric(recs,beta_strip=True); sgn=metric(recs,sign_only=True)
    def show(label,m):
        if not m: print("  %-18s n/a"%label); return
        print("  %-18s n=%-5d IC=%-8s net=%-8s t=%-6s mono=%d/%d"
              %(label,m["n"],"%+.4f"%m["ic"] if m["ic"] is not None else "NA",
                "%+.4f"%m["net"],"%.2f"%m["t"] if m["t"] else "NA",m["mono"][0],m["mono"][1]))
    show("pooled",pooled); show("beta-stripped",beta); show("sign-only",sgn)

    # OOS split at temporal midpoint
    sub("OUT-OF-SAMPLE holdout (temporal split)")
    mid=dates[len(dates)//2]
    train=[r for r in recs if r[0]<mid]; test=[r for r in recs if r[0]>=mid]
    tr=metric(train); te=metric(test)
    show("IN-SAMPLE",tr); show("OUT-OF-SAMPLE",te)

    banner("VERDICT — is '%s' a real brick? (history: %d days)"%(a.feature,span_days))
    short_hist = span_days<300
    if pooled and pooled["ic"] is not None:
        strong = abs(pooled["ic"])>=0.03 and abs(pooled["t"] or 0)>=2
        beta_ok = beta and abs(beta["ic"] or 0)>=0.02
        oos_ok = te and (te["ic"] is not None) and abs(te["ic"])>=0.02 and (np.sign(te["ic"])==np.sign(pooled["ic"]))
        print("  pooled IC=%+.4f t=%.2f | beta-stripped IC=%+.4f | OOS IC=%s"
              %(pooled["ic"],pooled["t"] or 0,(beta["ic"] or 0) if beta else 0,
                "%+.4f"%te["ic"] if te and te["ic"] is not None else "NA"))
        if strong and beta_ok and oos_ok and not short_hist:
            print("  >> CONFIRMED BRICK: significant, survives beta-strip, holds OOS, sufficient history.")
        elif strong and beta_ok and oos_ok and short_hist:
            print("  >> STRONG LEAD (short history): all checks pass BUT only %d days of data. The"%span_days)
            print("     signal is real-shaped; CONFIRM with 2+ years before trading. Highest-priority lead.")
        elif strong and short_hist:
            print("  >> LEAD (short history): significant pooled, but %d days only and/or weak on a"%span_days)
            print("     robustness check. Promising; needs more history + scrutiny.")
        elif strong:
            print("  >> MIXED: significant pooled but fails a robustness check (beta or OOS). Caution.")
        else:
            print("  >> WEAK/NULL: not significant. Not a brick on this evidence.")
    if short_hist:
        print("\n  NOTE: %d-day history = UNDERPOWERED and OOS halves are tiny. Treat as a lead about"%span_days)
        print("  WHERE to invest in data, not a tradeable conclusion. Same caveat as the hunt.")
    if a.out:
        rep={"feature":a.feature,"pooled":pooled,"beta":beta,"sign":sgn,"oos":te,"span_days":span_days}
        with open(a.out,"a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":rep},default=str)+"\n")
        print("\n  [report appended to %s]"%a.out)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
