#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — PEAD via SUE (standardized unexpected earnings)
================================================================================
Raw eps_surprise_pct has denominator artifacts (max +41,076%!) that inflate the
rank-IC and can't be fixed by winsorizing (rank is invariant to clipping). The
academically-standard fix is SUE: scale the surprise by the stock's OWN past
surprise volatility, so tiny-EPS-base garbage becomes a bounded multiple.

    SUE_i = (eps_actual_i - eps_estimate_i) / stdev(prior surprises for that ticker)

PIT-CRITICAL: the denominator uses ONLY surprises STRICTLY BEFORE event i (a
trailing std). Using full-history std would leak. Min 4 prior surprises required.

WHAT IT DOES (offline; reads cached prices.db; NO network):
  * recompute the signal as PIT-correct trailing SUE from eps_actual/eps_estimate
  * compare RAW%-surprise vs SUE: pooled IC, recent(2022+) IC/net/t, deciles, beta-strip
  * test at BOTH horizons: h=40 (the PEAD horizon where the edge lives) AND h=5
    (your live system's horizon) — to make explicit that the edge is horizon-specific
  * sign-only robustness (beat/miss) as the magnitude-proof floor

RULE 1: SUE denominator strictly trailing (no leak). Entry day +2, window strictly
after announcement. Signal known at announcement. No return clipping.

READ-ONLY. mode=ro&immutable=1. No network.

USAGE:
  python pead_sue.py --root .
  python pead_sue.py --root . --holds 5,40 --cost-bps 10 --min-prior 4
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
def cols_of(c,t): return [r[1] for r in Q(c,'PRAGMA table_info("'+t+'")')]
def require(cond,msg):
    if not cond: print("  [STOP] "+msg); return False
    return True
def find_db(root,name):
    c=os.path.join(root,name)
    if os.path.isfile(c): return c
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        if name in fn: return os.path.join(dp,name)
    return None
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
    ap.add_argument("--holds",default="5,40")
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--min-prior",type=int,default=4,help="min prior surprises to compute SUE")
    ap.add_argument("--min-events",type=int,default=30)
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    holds=[int(x) for x in a.holds.split(",")]
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    banner("ML QUANT FUND — PEAD via SUE (standardized unexpected earnings)")
    print("PIT-trailing SUE vs raw%; tests h=40 (PEAD) AND your h=5. (offline)")
    print("Root:",os.path.abspath(a.root),"| holds:",holds,"| min prior:",a.min_prior)
    if not require(HAVE_NUMPY,"numpy required"): return
    if not require(os.path.isfile(prices_db),"prices.db not found"): return
    earnp=find_db(a.root,"earnings.db")
    if not require(earnp,"earnings.db not found"): return

    # prices
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
    pxidx={tk:{d:i for i,(d,_) in enumerate(lst)} for tk,lst in px.items()}
    print("  prices loaded for %d tickers"%len(px))

    # earnings raw components for SUE
    ce=ro(earnp)
    try:
        cols=cols_of(ce,"earnings_surprises")
        need=["ticker","report_date"]
        have_components = "eps_actual" in cols and "eps_estimate" in cols
        have_pct = "eps_surprise_pct" in cols
        sel="ticker,report_date"
        if have_components: sel+=",eps_actual,eps_estimate"
        if have_pct: sel+=",eps_surprise_pct"
        ev=Q(ce,"SELECT "+sel+" FROM earnings_surprises WHERE report_date IS NOT NULL")
        _n_ev = Q(ce, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='earnings_events'")[0][0]
        _n_ev = Q(ce, "SELECT COUNT(*) FROM earnings_events WHERE eps_surprise IS NOT NULL")[0][0] if _n_ev else 0
        if _n_ev > 1000:
            comp = False
            ev = Q(ce, "SELECT ticker, announce_date, eps_surprise FROM earnings_events "
                       "WHERE eps_surprise IS NOT NULL AND announce_date IS NOT NULL")
            print("  PEAD source: earnings_events.announce_date (%d rows) [LEAK-FIXED]" % len(ev))
        else:
            print("  PEAD source: earnings_surprises (fallback -- KNOWN LEAKED DATES)")
    finally:
        ce.close()
    if not require(have_components or have_pct,"need eps_actual/eps_estimate or eps_surprise_pct"): return
    print("  components available: eps_actual/eps_estimate=%s, eps_surprise_pct=%s"%(have_components,have_pct))

    # build per-ticker ordered event list with raw surprise (actual-estimate) and pct
    by_tkr=defaultdict(list)
    for row in ev:
        tk=row[0]; do=nd(row[1])
        if do is None: continue
        idx=2
        actual=estimate=pct=None
        if have_components:
            actual=row[idx]; estimate=row[idx+1]; idx+=2
        if have_pct:
            pct=row[idx]
        raw_surp=None
        if actual is not None and estimate is not None:
            try: raw_surp=float(actual)-float(estimate)
            except Exception: raw_surp=None
        if raw_surp is None and pct is not None:
            try: raw_surp=float(pct)  # fallback: use pct as the surprise proxy
            except Exception: raw_surp=None
        by_tkr[tk].append((do,raw_surp,(float(pct) if pct is not None else None)))
    for tk in by_tkr: by_tkr[tk].sort()

    # compute PIT-trailing SUE per event
    # SUE_i = raw_surp_i / std(raw_surp_0..i-1), min --min-prior priors
    events=[]  # (ticker, date, sue, raw_pct)
    n_sue=0; n_skip_hist=0
    for tk,lst in by_tkr.items():
        prior=[]
        for do,raw_surp,pct in lst:
            if raw_surp is None:
                continue
            if len(prior)>=a.min_prior:
                sd=np.std(prior,ddof=1)
                sue=raw_surp/sd if sd>1e-12 else None
                if sue is not None:
                    events.append((tk,do,sue,pct)); n_sue+=1
            else:
                n_skip_hist+=1
            prior.append(raw_surp)
    print("  SUE computed for %d events (%d skipped: insufficient prior history)"%(n_sue,n_skip_hist))
    if n_sue<a.min_events*5:
        print("  [WARN] few SUE events — results will be noisy")

    def fwd(tk,do,hold):
        lst=px.get(tk); idx=pxidx.get(tk)
        if not lst: return None,None
        pos=None
        for off in range(0,6):
            c=do+datetime.timedelta(days=off)
            if c in idx: pos=idx[c]; break
        if pos is None: return None,None
        e=pos+2; x=pos+2+hold
        if x>=len(lst): return None,None
        pe=lst[e][1]; pxx=lst[x][1]
        if pe<=0: return None,None
        return pxx/pe-1.0, lst[e][0].year

    cost=a.cost_bps/10000.0
    def build_records(hold):
        recs=[]  # (year, sue, raw_pct, ret)
        for tk,do,sue,pct in events:
            r,yr=fwd(tk,do,hold)
            if r is None: continue
            recs.append((yr,sue,pct,r))
        return recs

    def metrics(recs, sigsel, recent_only=False, beta_strip=False, sign=False):
        rr=[x for x in recs if (x[0]>=2022 if recent_only else True)]
        yr_mean=defaultdict(list)
        for x in rr: yr_mean[x[0]].append(x[3])
        ym={k:np.mean(v) for k,v in yr_mean.items()}
        pts=[]
        for yr,sue,pct,r in rr:
            sigval = sue if sigsel=="sue" else pct
            if sigval is None: continue
            if sign: sigval=1.0 if sigval>0 else (-1.0 if sigval<0 else 0.0)
            ret = (r-ym.get(yr,0.0)) if beta_strip else r
            pts.append((sigval,ret))
        if len(pts)<a.min_events: return None
        s=[p[0] for p in pts]; r=[p[1] for p in pts]; n=len(s)
        ic=spearman(s,r)
        order=np.argsort(s); q=max(1,n//5); lo=order[:q]; hi=order[-q:]
        L=float(np.mean([r[i] for i in hi])); S=float(np.mean([r[i] for i in lo]))
        g=L-S; net=g-2*cost
        sd=math.sqrt(np.var([r[i] for i in hi])/q+np.var([r[i] for i in lo])/q)
        t=g/sd if sd>0 else None
        return {"n":n,"ic":ic,"net":net,"t":t,"gross":g}

    def deciles(recs, sigsel):
        pts=[(x[1] if sigsel=="sue" else x[2], x[3]) for x in recs if (x[1] if sigsel=="sue" else x[2]) is not None]
        s=[p[0] for p in pts]; r=[p[1] for p in pts]; n=len(s)
        order=np.argsort(s); dm=[]; sm=[]
        for d in range(10):
            idx=order[int(d*n/10):int((d+1)*n/10)]
            if len(idx)>0:
                dm.append(np.mean([r[i] for i in idx])); sm.append(np.mean([s[i] for i in idx]))
        ups=sum(1 for i in range(1,len(dm)) if dm[i]>=dm[i-1])
        return ups,len(dm)-1,sm,dm

    report={}
    for hold in holds:
        banner("HOLD = %d days  (entry day +2)"%hold)
        recs=build_records(hold)
        if len(recs)<a.min_events*3:
            print("  too few events with %d-day prices (%d) — skip"%(hold,len(recs))); continue
        # raw% vs SUE, pooled + recent + beta + sign
        print("  %-18s | %-22s | %-22s"%("metric","RAW %-surprise","SUE (standardized)"))
        def row(label, **kw):
            mr=metrics(recs,"raw",**kw); ms=metrics(recs,"sue",**kw)
            def fmt(m):
                if not m: return "n/a"
                return "IC%+.3f net%+.4f t%.2f"%(m["ic"] or 0,m["net"],m["t"] or 0)
            print("  %-18s | %-22s | %-22s"%(label,fmt(mr),fmt(ms)))
            return mr,ms
        pooled=row("pooled")
        recent=row("recent 2022+",recent_only=True)
        recent_beta=row("recent beta-strip",recent_only=True,beta_strip=True)
        recent_sign=row("recent SIGN-only",recent_only=True,sign=True)
        # deciles for SUE
        ups_s,steps_s,sm_s,dm_s=deciles(recs,"sue")
        ups_r,steps_r,_,_=deciles(recs,"raw")
        print("  decile monotonicity: RAW %d/%d  vs  SUE %d/%d"%(ups_r,steps_r,ups_s,steps_s))
        print("  SUE deciles (sue -> ret):", ", ".join("%.1f->%+.3f"%(sm_s[i],dm_s[i]) for i in range(len(dm_s))))
        report[hold]={"pooled_sue":pooled[1],"recent_sue":recent[1],
                      "recent_beta_sue":recent_beta[1],"recent_sign_sue":recent_sign[1],
                      "mono_sue":[ups_s,steps_s],"mono_raw":[ups_r,steps_r]}

    # ---- verdict ----
    banner("VERDICT — clean (SUE) edge, and is it at YOUR horizon?")
    h_pead = 40 if 40 in report else (max(report.keys()) if report else None)
    h_short = 5 if 5 in report else (min(report.keys()) if report else None)
    if h_pead and report.get(h_pead,{}).get("recent_sue"):
        r=report[h_pead]["recent_sue"]; rb=report[h_pead]["recent_beta_sue"]; rs=report[h_pead]["recent_sign_sue"]
        print("  At h=%d (PEAD horizon), SUE recent(2022+): IC=%+.4f net=%+.4f t=%.2f"
              %(h_pead,r["ic"] or 0,r["net"],r["t"] or 0))
        if rb: print("    beta-stripped net=%+.4f  (%s)"%(rb["net"],"survives" if rb["net"]>0.005 else "weakens"))
        if rs: print("    sign-only IC=%+.4f t=%.2f  (%s)"%(rs["ic"] or 0,rs["t"] or 0,
                     "magnitude-robust" if (rs["ic"] or 0)>0.02 and (rs["t"] or 0)>=2 else "magnitude-dependent"))
        ms=report[h_pead]["mono_sue"]; mr=report[h_pead]["mono_raw"]
        print("    monotonicity SUE %d/%d vs raw%% %d/%d  (%s)"
              %(ms[0],ms[1],mr[0],mr[1],"SUE cleaner" if ms[0]>mr[0] else ("same" if ms[0]==mr[0] else "raw cleaner")))
    if h_short and report.get(h_short,{}).get("recent_sue"):
        r=report[h_short]["recent_sue"]
        print("\n  At h=%d (YOUR live horizon), SUE recent(2022+): IC=%+.4f net=%+.4f t=%.2f"
              %(h_short,r["ic"] or 0,r["net"],r["t"] or 0))
        if (r["t"] or 0)<2 or (r["net"] or 0)<=0.003:
            print("    -> NOT tradeable at your horizon. The edge is a 40-day strategy, NOT an h1/h3/h5")
            print("       signal. Trading it means a SEPARATE ~quarterly-turnover book, not a tweak")
            print("       to the current daily system.")
        else:
            print("    -> also works at the short horizon (unusual — double-check).")
    print("\n  SUE is the honest magnitude. Compare it to the contaminated raw%% (~0.138): the SUE")
    print("  number is what you'd actually size on. Sign-only is the floor (pure beat/miss).")
    if a.out:
        with open(a.out,"a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":report},default=str)+"\n")
        print("\n  [report appended to %s]"%a.out)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
