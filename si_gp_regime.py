#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — SI+GP LIFT: REGIME-CONDITIONAL AUDIT
================================================================================
Audits the hypothesis: the combination failure (GP drags SI) is driven by the
EXTREME 2025-26 regime where SI ran hot (Sharpe +2.1, +2.8 -- favorable-regime
artifact, through-cycle is ~1.2), so ANY diluter looks bad -- NOT by GP being
permanently wrong-sign. If true, "more history" is the wrong fix; the issue is
regime, not sample length.

THREE TESTS (same paired construction as si_gp_lift.py):
  1. WINDOWED lift via --start/--end: re-run on 2021-2024 (drop the extreme regime).
     If the lift goes from strongly negative -> ~0, the recent regime WAS the driver.
  2. PER-YEAR context: trailing-60d market realized vol + si_only/si+gp Sharpe + lift,
     so you can SEE whether 2025-26 is actually high-vol and where GP drags.
  3. VOL-REGIME SPLIT: classify each rebalance by trailing market vol (low vs high,
     median split, known at rebalance -> no lookahead), bootstrap the lift WITHIN each.
     If GP helps (lift>0) in low-vol and only hurts in high-vol -> regime-conditional.
     If GP hurts in BOTH -> persistent wrong-sign, regime is not the explanation.

RULE 1: identical PIT construction to si_gp_lift; vol classification strictly trailing;
window filter on rebalance dates; paired block bootstrap; READ-ONLY.

USAGE:
  python si_gp_regime.py --root .                       # full + per-year + vol split
  python si_gp_regime.py --root . --end 2024-12-31      # lift on 2021-2024 only
  python si_gp_regime.py --root . --vol-window 60
================================================================================
"""
import argparse, os, sqlite3, math, datetime
from collections import defaultdict
import numpy as np

def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def nd(s):
    if s is None: return None
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None
def ranks(v): return np.argsort(np.argsort(np.asarray(v,float))).astype(float)
def zscore(v):
    v=np.asarray(v,float); m=v.mean(); s=v.std(ddof=1)
    return (v-m)/s if s>1e-12 else np.zeros_like(v)
def sharpe_of(x,ppy):
    x=np.asarray(x,float)
    if len(x)<2: return float('nan')
    v=x.std(ddof=1)
    return (x.mean()/v*math.sqrt(ppy)) if v>0 else 0.0
def boot_ci_lift(a,b,ppy,n_boot=5000,seed=42,block=3):
    a=np.asarray(a,float); bb=np.asarray(b,float); n=len(a); rng=np.random.default_rng(seed)
    if n<4: return (float('nan'),float('nan')),float('nan'),float('nan')
    def bidx(N):
        out=[]
        while len(out)<N:
            s=rng.integers(0,N); out.extend([(s+j)%N for j in range(block)])
        return np.array(out[:N])
    diffs=[sharpe_of(a[idx],ppy)-sharpe_of(bb[idx],ppy) for idx in (bidx(n) for _ in range(n_boot))]
    diffs=np.array(diffs); point=sharpe_of(a,ppy)-sharpe_of(bb,ppy)
    return np.percentile(diffs,[2.5,97.5]), float(np.mean(diffs>0)), point
LINE="="*78

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--quantile",type=float,default=0.2)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--max-weight",type=float,default=0.05)
    ap.add_argument("--cost-bps",type=float,default=25.0)
    ap.add_argument("--max-stale-days",type=int,default=550)
    ap.add_argument("--vol-window",type=int,default=60)
    ap.add_argument("--start",default=None)
    ap.add_argument("--end",default=None)
    ap.add_argument("--n-boot",type=int,default=5000)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=os.path.join(a.root,"prices.db"); si_db=os.path.join(a.root,"short_interest.db"); fund_db=os.path.join(a.root,"fundamentals.db")
    print("\n"+LINE+"\nSI+GP LIFT — REGIME-CONDITIONAL AUDIT\n"+LINE)
    for lbl,p in (("prices.db",prices_db),("short_interest.db",si_db),("fundamentals.db",fund_db)):
        if not os.path.isfile(p): print("  [STOP] %s not found"%lbl); return
    start=nd(a.start) if a.start else None; end=nd(a.end) if a.end else None

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
        for off in range(0,6):
            cc=d+datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+h
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None

    # market daily returns + trailing vol
    rets=defaultdict(dict); alldates=set()
    for tk,lst in px.items():
        for i in range(1,len(lst)):
            d0,p0=lst[i-1]; d1,p1=lst[i]
            if p0>0: rets[tk][d1]=p1/p0-1.0; alldates.add(d1)
    caldates=sorted(alldates)
    by_date=defaultdict(list)
    for tk,dd in rets.items():
        for d,r in dd.items(): by_date[d].append(r)
    mkt={d:float(np.mean(rs)) for d,rs in by_date.items() if len(rs)>=10}
    mdates=sorted(mkt)
    def trailing_vol(d):
        wd=[x for x in mdates if x<d][-a.vol_window:]
        if len(wd)<a.vol_window//2: return None
        return float(np.std([mkt[x] for x in wd],ddof=1)*math.sqrt(252))

    c=ro(si_db)
    try: sirows=Q(c,"SELECT ticker,settlement_date,days_to_cover FROM short_interest")
    finally: c.close()
    si_by_date=defaultdict(dict)
    for tk,d,v in sirows:
        do=nd(d)
        if do is None or v is None: continue
        try: fv=float(v)
        except Exception: continue
        if fv<=50.0: si_by_date[do][tk.upper()]=fv

    cf=ro(fund_db)
    try: grows=Q(cf,"SELECT ticker,filed_date,gp FROM gross_profitability WHERE gp IS NOT NULL AND filed_date IS NOT NULL")
    finally: cf.close()
    gp_ev=defaultdict(list)
    for tk,fd,gp in grows:
        do=nd(fd)
        if do is None: continue
        try: g=float(gp)
        except Exception: continue
        gp_ev[tk.upper()].append((do,g))
    for tk in gp_ev: gp_ev[tk].sort()
    def gp_asof(tk,f):
        evs=gp_ev.get(tk)
        if not evs: return None
        best=None
        for fd,g in evs:
            if fd<=f: best=(fd,g)
            else: break
        if best is None or (f-best[0]).days>a.max_stale_days: return None
        return best[1]

    grid=sorted(si_by_date)
    if start: grid=[d for d in grid if d>=start]
    if end: grid=[d for d in grid if d<=end]
    ppy=365.25/float(a.hold)

    def run_on(dates,use_si,use_gp):
        prev_w={}; rets_=[]; rdates=[]
        for d in dates:
            names=[]; dtc=[]; gp=[]
            for tk,dv in si_by_date[d].items():
                if tk not in pos_of: continue
                g=gp_asof(tk,d)
                if g is None: continue
                r=fwd(tk,d,a.hold)
                if r is None: continue
                names.append(tk); dtc.append(dv); gp.append(g)
            if len(names)<a.min_names: continue
            comps=[]
            if use_si: comps.append(zscore(ranks(-np.array(dtc))))
            if use_gp: comps.append(zscore(ranks(np.array(gp))))
            score=np.mean(comps,axis=0)
            order=np.argsort(score); q=max(1,int(len(names)*a.quantile))
            wd={}
            for j in order[-q:]: wd[names[j]]=min(a.max_weight,1.0/(2*q))
            for j in order[:q]: wd[names[j]]=-min(a.max_weight,1.0/(2*q))
            allk=set(wd)|set(prev_w); turn=sum(abs(wd.get(k,0)-prev_w.get(k,0)) for k in allk)
            r=sum(wi*fwd(tk,d,a.hold) for tk,wi in wd.items() if fwd(tk,d,a.hold) is not None)
            r-=turn*(a.cost_bps/10000.0)
            rets_.append(r); rdates.append(d); prev_w=wd
        return np.array(rets_), rdates

    si,sd=run_on(grid,1,0); comb,cd=run_on(grid,1,1)
    if len(comb)<10: print("  [STOP] only %d paired rebalances in window."%len(comb)); return
    wlabel="%s..%s"%(cd[0],cd[-1])
    print("  window: %s | %d paired rebalances | net %.0fbps | hold %dd"%(wlabel,len(comb),a.cost_bps,a.hold))

    # overall lift
    ci,p,point=boot_ci_lift(comb,si,ppy,a.n_boot)
    print("\n"+"-"*78+"\nOVERALL LIFT in window\n"+"-"*78)
    print("  si_only Sharpe %+.2f | si+gp Sharpe %+.2f | LIFT %+.2f CI [%+.2f,%+.2f] P(>0)=%.0f%%"
          %(sharpe_of(si,ppy),sharpe_of(comb,ppy),point,ci[0],ci[1],100*p))

    # per-year: vol + sharpes + lift
    yr_si=defaultdict(list); yr_cb=defaultdict(list); yr_vol=defaultdict(list)
    for r,d in zip(si,sd): yr_si[d.year].append(r)
    for r,d in zip(comb,cd):
        yr_cb[d.year].append(r); v=trailing_vol(d)
        if v is not None: yr_vol[d.year].append(v)
    print("\n"+"-"*78+"\nPER-YEAR: market vol (annualized) + Sharpe + lift\n"+"-"*78)
    print("  %-6s %9s %10s %10s %8s"%("year","mkt vol","si_only","si+gp","lift"))
    for y in sorted(yr_cb):
        s=np.array(yr_si.get(y,[])); cb=np.array(yr_cb[y])
        vol=np.mean(yr_vol[y]) if yr_vol.get(y) else float('nan')
        if len(cb)>=3:
            ss=sharpe_of(s,ppy); cs=sharpe_of(cb,ppy)
            print("  %-6d %8.1f%% %+10.2f %+10.2f %+8.2f"%(y,100*vol,ss,cs,cs-ss))

    # vol-regime split (median trailing vol at rebalance)
    vols=[]; valid=[]
    for r,d in zip(comb,cd):
        v=trailing_vol(d)
        if v is not None: vols.append(v); valid.append((r,d))
    if len(vols)>=20:
        med=np.median(vols)
        lo_idx=[i for i,v in enumerate(vols) if v<med]; hi_idx=[i for i,v in enumerate(vols) if v>=med]
        # align si to same dates
        si_map={d:r for r,d in zip(si,sd)}
        def split_streams(idxs):
            cb=np.array([valid[i][0] for i in idxs]); ss=np.array([si_map.get(valid[i][1],np.nan) for i in idxs])
            mask=~np.isnan(ss); return cb[mask], ss[mask]
        print("\n"+"-"*78+"\nVOL-REGIME SPLIT (trailing %dd market vol, median=%.0f%%)\n"%(a.vol_window,100*med)+"-"*78)
        for label,idxs in (("LOW-VOL ",lo_idx),("HIGH-VOL",hi_idx)):
            cb,ss=split_streams(idxs)
            if len(cb)>=8:
                lci,lp,lpt=boot_ci_lift(cb,ss,ppy,a.n_boot)
                print("  %s (n=%d): si_only %+.2f | si+gp %+.2f | LIFT %+.2f CI[%+.2f,%+.2f] P(>0)=%.0f%%"
                      %(label,len(cb),sharpe_of(ss,ppy),sharpe_of(cb,ppy),lpt,lci[0],lci[1],100*lp))

    print("\n"+LINE+"\nREAD\n"+LINE)
    print("  * If the per-year LIFT is negative in CALM years too (2021-23, lower vol), GP is")
    print("    persistently wrong-sign here -> regime is NOT the explanation -> more history won't help GP.")
    print("  * If LIFT is ~0/positive in low-vol and only strongly negative in high-vol (2025-26),")
    print("    then the recent regime IS the driver -> re-test with --end 2024-12-31 to confirm.")
    print("  * Either way: GP's drag being LARGEST where SI ran hottest (2025-26) is consistent with")
    print("    dilution-of-a-hot-signal, not necessarily GP being useless in normal times.")
    print("\n  Honest n=%d paired rebalances in window. In-sample, survivor-tilted."%len(comb))

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
