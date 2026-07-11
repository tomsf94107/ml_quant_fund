#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — IDIO-VOL AS A RISK GATE (market-timing test)
================================================================================
Idio-vol failed as a cross-sectional BRICK (orthogonal, but no steady standalone edge --
it's a regime premium: IC -0.20 in 2009/2020 risk-off, +0.12 in 2021 risk-on). This
tests the regime-dependence ITSELF as a market-timing gate.

HYPOTHESIS: the "defensive factor" = (low-idio-vol minus high-idio-vol) portfolio return.
When defensive names have been WINNING recently (flight to safety), forward market returns
are worse -> a gate that de-risks in that state should add Sharpe.

CONSTRUCTION (no lookahead):
  * at each month-end f: form idio-vol quintiles from the trailing --window; the DEFENSIVE
    FACTOR realized over the month ENDING at f (portfolios formed at the PRIOR month-end)
    is known at f.
  * GATE SIGNAL G[f] = trailing --k-month average of the realized defensive factor (all
    known at f). High G = defensive winning = risk-off tell.
  * OUTCOME = equal-weight market return over the NEXT month (known only at f_next).
  * test: does G[f] predict next-month market return? (expect NEGATIVE: defensive-winning
    -> weaker forward market.)

OUTPUTS: predictive corr + t + null control; tercile split of forward market return by
gate state; a GATED strategy (in market when gate not risk-off, else cash) vs always-in,
Sharpe compared; OOS split. Built to KILL it if it's noise.

RULE 1: portfolios formed strictly before the return window; gate strictly trailing;
outcome strictly forward; null control; OOS split; READ-ONLY. Not investment advice.

USAGE:
  python validate_vol_gate.py --root .
  python validate_vol_gate.py --root . --k 3 --window 252 --quantile 0.2
================================================================================
"""
import argparse, os, sqlite3, math, datetime
from collections import defaultdict
import numpy as np

def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def nd(s):
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None
def month_end(d):
    nxt=datetime.date(d.year+(1 if d.month==12 else 0),(d.month%12)+1,1)
    return nxt-datetime.timedelta(days=1)
def sharpe(x,ppy):
    x=np.asarray(x,float); v=x.std(ddof=1)
    return (x.mean()/v*math.sqrt(ppy)) if v>0 else 0.0
LINE="="*78

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--window",type=int,default=252)
    ap.add_argument("--k",type=int,default=3,help="trailing months for the gate signal")
    ap.add_argument("--quantile",type=float,default=0.2)
    ap.add_argument("--min-names",type=int,default=20)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    print("\n"+LINE+"\nIDIO-VOL RISK GATE — market-timing test\n"+LINE)
    if not os.path.isfile(prices_db): print("  [STOP] prices.db not found"); return

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

    def idio_asof(tk,f):
        dd=rets.get(tk)
        if not dd: return None
        wd=[d for d in caldates if d<f][-a.window:]
        if len(wd)<30: return None
        rr=[]; mm=[]
        for d in wd:
            if d in dd and d in mkt: rr.append(dd[d]); mm.append(mkt[d])
        if len(rr)<30: return None
        rr=np.array(rr); mm=np.array(mm); vm=mm.var()
        if vm<=0: return None
        beta=np.cov(rr,mm,ddof=1)[0,1]/vm; alpha=rr.mean()-beta*mm.mean()
        return float((rr-(alpha+beta*mm)).std(ddof=1))
    def ret_between(tk,d0,d1):
        idx=pos_of.get(tk); lst=px.get(tk)
        if not idx or not lst: return None
        i0=None;i1=None
        for off in range(0,8):
            if d0+datetime.timedelta(days=off) in idx: i0=idx[d0+datetime.timedelta(days=off)]; break
        for off in range(0,8):
            if d1+datetime.timedelta(days=off) in idx: i1=idx[d1+datetime.timedelta(days=off)]; break
        if i0 is None or i1 is None or i1<=i0: return None
        p0=lst[i0][1]; p1=lst[i1][1]
        return (p1/p0-1.0) if p0>0 else None

    months=sorted(set(month_end(d) for d in caldates))
    # realized defensive factor over each month [f_prev -> f], and market over each month
    Dreal={}; Mreal={}
    for j in range(1,len(months)):
        f_prev=months[j-1]; f=months[j]
        names=[]; iv=[]
        for tk in px:
            v=idio_asof(tk,f_prev)
            if v is None: continue
            r=ret_between(tk,f_prev,f)
            if r is None: continue
            names.append(tk); iv.append((v,r))
        if len(names)<a.min_names: continue
        iv_sorted=sorted(iv,key=lambda x:x[0]); q=max(1,int(len(iv_sorted)*a.quantile))
        low=[r for _,r in iv_sorted[:q]]; high=[r for _,r in iv_sorted[-q:]]
        Dreal[f]=float(np.mean(low)-np.mean(high))     # defensive factor realized this month
        Mreal[f]=float(np.mean([r for _,r in iv]))      # market realized this month
    fdates=sorted(Dreal)
    if len(fdates)<a.k+12: print("  [STOP] only %d monthly factor points."%len(fdates)); return

    # gate signal G[f] = trailing k-month avg Dreal (known at f); outcome = market NEXT month
    G=[]; Mfwd=[]; gd=[]
    for i in range(a.k-1,len(fdates)-1):
        f=fdates[i]; f_next=fdates[i+1]
        window=[Dreal[fdates[i-x]] for x in range(a.k)]
        G.append(float(np.mean(window))); Mfwd.append(Mreal[f_next]); gd.append(f_next)
    G=np.array(G); Mfwd=np.array(Mfwd); N=len(G)
    ppy=12.0

    # predictive correlation (expect negative: defensive winning -> weaker forward market)
    if G.std()>0 and Mfwd.std()>0:
        corr=float(np.corrcoef(G,Mfwd)[0,1]); tstat=corr*math.sqrt((N-2)/max(1e-9,1-corr**2))
    else: corr=0; tstat=0
    print("  %d months tested | gate k=%d | window=%dtd | quintile %.0f%%"%(N,a.k,a.window,100*a.quantile))
    print("\n"+"-"*78+"\nGATE PREDICTS FORWARD MARKET?\n"+"-"*78)
    print("  corr(gate, next-month market) = %+.3f | t = %+.2f  (negative supports the gate)"%(corr,tstat))

    # tercile split
    order=np.argsort(G); ter=N//3
    lo_state=Mfwd[order[:ter]]; mid=Mfwd[order[ter:2*ter]]; hi_state=Mfwd[order[-ter:]]
    print("\n  forward market return by gate state (gate = trailing defensive-factor strength):")
    print("   gate LOW  (risk-on, defensive lagging): mean fwd mkt %+.2f%%  (n=%d)"%(100*lo_state.mean(),len(lo_state)))
    print("   gate MID:                               mean fwd mkt %+.2f%%  (n=%d)"%(100*mid.mean(),len(mid)))
    print("   gate HIGH (risk-off, defensive winning): mean fwd mkt %+.2f%%  (n=%d)"%(100*hi_state.mean(),len(hi_state)))
    diff=lo_state.mean()-hi_state.mean()
    # t-test on the difference
    sp=math.sqrt(lo_state.var(ddof=1)/len(lo_state)+hi_state.var(ddof=1)/len(hi_state))
    tdiff=diff/sp if sp>0 else 0
    print("   spread (LOW - HIGH) = %+.2f%% | t = %+.2f"%(100*diff,tdiff))

    # gated strategy: hold market next month unless gate is in top tercile (risk-off) -> cash
    thr=np.quantile(G,2/3)
    gated=np.where(G>=thr,0.0,Mfwd)   # cash when risk-off
    always=Mfwd
    print("\n"+"-"*78+"\nGATED vs ALWAYS-INVESTED (cash when gate risk-off)\n"+"-"*78)
    print("  always-in : Sharpe %+.2f | ann ret %+.1f%% | months invested %d/%d"
          %(sharpe(always,ppy),100*always.mean()*ppy,N,N))
    inv=int((G<thr).sum())
    print("  gated     : Sharpe %+.2f | ann ret %+.1f%% | months invested %d/%d"
          %(sharpe(gated,ppy),100*gated.mean()*ppy,inv,N))

    # null control: shuffle gate vs forward -> does the LOW-HIGH spread vanish?
    rng=np.random.default_rng(7); nulls=[]
    for _ in range(2000):
        perm=rng.permutation(Mfwd); o=np.argsort(G)
        nulls.append(perm[o[:ter]].mean()-perm[o[-ter:]].mean())
    nulls=np.array(nulls); z=(diff-nulls.mean())/nulls.std() if nulls.std()>0 else 0
    print("\n  null control: real LOW-HIGH spread %.1f std's from shuffled null (need >=2.5)"%z)

    # OOS split
    half=N//2
    def corr_of(g,m):
        if g.std()>0 and m.std()>0: return float(np.corrcoef(g,m)[0,1])
        return 0
    print("\n  OOS: first-half corr %+.3f | second-half corr %+.3f"%(corr_of(G[:half],Mfwd[:half]),corr_of(G[half:],Mfwd[half:])))

    print("\n"+LINE+"\nVERDICT — does idio-vol work as a risk gate?\n"+LINE)
    gated_better = sharpe(gated,ppy) > sharpe(always,ppy)+0.15
    sig = abs(tdiff)>=2.0 and abs(z)>=2.5 and corr<0
    if sig and gated_better:
        print("  >> USABLE GATE: defensive-winning predicts weaker forward market (corr %+.2f, spread t %+.2f,"%(corr,tdiff))
        print("     %.1f std's from null), and gating lifts Sharpe %+.2f -> %+.2f. Wire as a sizing/risk overlay,"%(z,sharpe(always,ppy),sharpe(gated,ppy)))
        print("     NOT a ranking brick. Validate OOS persistence before trusting live.")
    elif sig and not gated_better:
        print("  >> PREDICTIVE BUT NOT TRADEABLE: the signal predicts (t %+.2f, null %.1f) but gating doesn't"%(tdiff,z))
        print("     beat always-in on Sharpe (the market's drift dominates). Marginal as a gate.")
    else:
        print("  >> NOT A USABLE GATE: spread t %+.2f, null %.1f std's, gated Sharpe %+.2f vs always-in %+.2f."
              %(tdiff,z,sharpe(gated,ppy),sharpe(always,ppy)))
        print("     The regime-dependence is real in-sample but doesn't time the market out-of-sample.")
        print("     (Expected -- regime timing rarely survives. Clean negative, move to fundamentals.)")
    print("\n  Honest n=%d months. In-sample, survivor-tilted (prices.db)."%N)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
