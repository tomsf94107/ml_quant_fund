#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — DOES SI + GROSS-PROFITABILITY BEAT SI ALONE?
================================================================================
GP failed as a STANDALONE brick (NW-t ~1.3) but is genuinely ORTHOGONAL to SI
(rank-corr -0.05) and works in different years (positive 2019-2024, negative 2025-26).
That is exactly the profile of a DIVERSIFIER -- a signal that lifts a combined book
even when it can't stand alone, IF it's truly uncorrelated and contributes in different
periods (the Fundamental Law mechanism). This tests that directly.

The two cousins both FAILED this same lift test (SI+PEAD P=73%; full book vs SI P=73%) --
because sub-significant signals + short overlapping samples can't clear the bar. GP has
the same sub-significance and a possibly-shorter overlap (GP needs 10-K history; SI is
2021+). So the honest prior is "directionally positive, underpowered." This measures it.

CONSTRUCTION (identical to book_robustness.py so it's comparable):
  * universe each SI settlement date = names with days_to_cover AND a PIT GP (filed<=date)
    AND a forward return. si_only, gp_only, si+gp all run on this SAME universe/dates ->
    streams are PAIRED (only the score differs).
  * si signal = z(rank(-DTC)); gp signal = z(rank(GP)); combined = mean of the two.
  * dollar-neutral long/short top-vs-bottom quintile, weight cap, net-of-cost turnover, 40d hold.
  * LIFT = Sharpe(si+gp) - Sharpe(si_only), via PAIRED block bootstrap (same resampled
    blocks for both) -> CI + P(lift>0). This is gp's marginal contribution to the book.

RULE 1: GP ranked only on filed_date<=formation (PIT); forward returns strictly after;
net-of-cost; paired block bootstrap; READ-ONLY.

USAGE:
  python si_gp_lift.py --root . --cost-bps 25
  python si_gp_lift.py --root . --hold 60 --cost-bps 10
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
    x=np.asarray(x,float); v=x.std(ddof=1)
    return (x.mean()/v*math.sqrt(ppy)) if v>0 else 0.0
def boot_ci(rets,ppy,n_boot=5000,seed=42,block=3):
    rets=np.asarray(rets,float); n=len(rets); rng=np.random.default_rng(seed)
    def bidx(N):
        out=[]
        while len(out)<N:
            s=rng.integers(0,N); out.extend([(s+j)%N for j in range(block)])
        return np.array(out[:N])
    b=np.array([sharpe_of(rets[bidx(n)],ppy) for _ in range(n_boot)])
    return np.percentile(b,[2.5,97.5])
def boot_ci_lift(a,b,ppy,n_boot=5000,seed=42,block=3):
    a=np.asarray(a,float); bb=np.asarray(b,float); n=len(a); rng=np.random.default_rng(seed)
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
    ap.add_argument("--n-boot",type=int,default=5000)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=os.path.join(a.root,"prices.db"); si_db=os.path.join(a.root,"short_interest.db"); fund_db=os.path.join(a.root,"fundamentals.db")
    print("\n"+LINE+"\nSI + GROSS PROFITABILITY — does combining beat SI alone?\n"+LINE)
    for lbl,p in (("prices.db",prices_db),("short_interest.db",si_db),("fundamentals.db",fund_db)):
        if not os.path.isfile(p): print("  [STOP] %s not found"%lbl); return

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
    ppy=365.25/float(a.hold)

    def run_on(dates,use_si,use_gp):
        prev_w={}; rets=[]; rdates=[]
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
            rets.append(r); rdates.append(d); prev_w=wd
        return np.array(rets), rdates

    si,sd=run_on(grid,1,0); gp,_=run_on(grid,0,1); comb,cd=run_on(grid,1,1)
    if len(comb)<10: print("  [STOP] only %d paired rebalances on the GP-covered universe."%len(comb)); return
    print("  %d paired rebalances on GP-covered universe (%s..%s), net %.0fbps, hold %dd"
          %(len(comb),cd[0],cd[-1],a.cost_bps,a.hold))

    print("\n"+"-"*78+"\nSTANDALONE (net, on the SAME universe)\n"+"-"*78)
    for nm,stream in (("si_only",si),("gp_only",gp),("si+gp",comb)):
        ci=boot_ci(stream,ppy,a.n_boot)
        print("  %-8s Sharpe %+.2f  CI [%+.2f,%+.2f] | ann ret %+.1f%%"
              %(nm,sharpe_of(stream,ppy),ci[0],ci[1],100*stream.mean()*ppy))

    # return-stream correlation (diversification)
    if len(si)==len(gp) and si.std()>0 and gp.std()>0:
        corr=float(np.corrcoef(si,gp)[0,1])
        print("\n  si vs gp return-stream correlation = %+.2f (low = good diversification)"%corr)

    # the lift: si+gp vs si_only, paired
    ci,p,point=boot_ci_lift(comb,si,ppy,a.n_boot)
    print("\n"+"-"*78+"\nLIFT — does SI+GP beat SI alone? (paired block bootstrap)\n"+"-"*78)
    print("  si_only Sharpe %+.2f | si+gp Sharpe %+.2f"%(sharpe_of(si,ppy),sharpe_of(comb,ppy)))
    print("  LIFT = %+.2f  CI [%+.2f,%+.2f]  P(lift>0) = %.0f%%"%(point,ci[0],ci[1],100*p))

    # per-year: where does GP help vs hurt?
    by_year_si=defaultdict(list); by_year_comb=defaultdict(list)
    for r,d in zip(si,sd): by_year_si[d.year].append(r)
    for r,d in zip(comb,cd): by_year_comb[d.year].append(r)
    print("\n  per-year Sharpe (si_only -> si+gp):")
    for y in sorted(by_year_comb):
        s=np.array(by_year_si.get(y,[])); cmb=np.array(by_year_comb[y])
        if len(cmb)>=3:
            print("   %d: %+.2f -> %+.2f  (n=%d)"%(y,sharpe_of(s,ppy) if len(s)>=3 else float('nan'),sharpe_of(cmb,ppy),len(cmb)))

    print("\n"+LINE+"\nVERDICT\n"+LINE)
    if ci[0]>0:
        print("  >> GP ADDS: lift CI clears zero (P=%.0f%%). Combining SI+GP beats SI alone net-of-cost --"%(100*p))
        print("     the first candidate to do so. GP earns a weight as an uncorrelated diversifier.")
        print("     NEXT: size the combined book; confirm OOS as more GP/SI history accrues.")
    elif p>=0.85:
        print("  >> DIRECTIONALLY POSITIVE, not conclusive: P(lift>0)=%.0f%% but CI touches zero."%(100*p))
        print("     Suggestive that GP diversifies, but the overlapping sample can't confirm it. As with")
        print("     the cousins, the binding constraint is sample length, not signal quality.")
    else:
        print("  >> GP DOES NOT CLEARLY LIFT THE BOOK (P=%.0f%%, CI spans 0). Same outcome as SI+PEAD and"%(100*p))
        print("     the full book: a real, orthogonal signal that doesn't stack at this sample size.")
        print("     CONCLUSION: breadth is capped by your data HISTORY, not by signal availability --")
        print("     none of PEAD / idio-vol / GP stack with SI given the overlapping window. Trade SI alone.")
    print("\n  Honest n=%d paired rebalances. In-sample, survivor-tilted; GP universe large-cap."%len(comb))

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
