#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — GROSS PROFITABILITY VALIDATOR (brick #3, fundamentals)
================================================================================
Tests gross profitability (GP = (Revenue-COGS)/Assets, Novy-Marx 2013) on YOUR universe
with the same audited gauntlet as the SI brick. GP is one of the few documented anomalies
that has NOT decayed, and it's fundamentals-derived -> a genuinely different axis from
price/positioning. The brick-#3 question is whether it's real AND uncorrelated.

STRICT PIT (the thing that makes or breaks a fundamentals backtest): GP for fiscal year Y
only becomes public when the 10-K is FILED (months after year-end). At each monthly
formation date f, each stock is ranked by the most recent GP whose filed_date <= f. Never
the fiscal-year value before it was public. (fundamentals_fetch stored filed_date for this.)
Annual data is slow, so a GP reading stays "live" until superseded by a newer filing.

DIRECTION +1: HIGH gross profitability -> HIGH forward return (Novy-Marx). Verdict checks
significant POSITIVE IC after direction + null-clear, plus decorrelation vs momentum and SI.

RULE 1: rank only on GP with filed_date <= formation (PIT); forward returns strictly after;
per-date IC + Newey-West; null control; per-year + OOS; decorrelation; READ-ONLY.

USAGE:
  python validate_gp.py --root . --hold 40
  python validate_gp.py --root . --hold 60
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
    ap.add_argument("--fund-db",default=None)
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--si-db",default=None)
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--direction",type=int,default=1)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--max-stale-days",type=int,default=550,help="ignore a GP reading older than this (annual data ~ allow 18mo)")
    ap.add_argument("--mom-lookback",type=int,default=252)
    ap.add_argument("--mom-skip",type=int,default=21)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    fund_db=a.fund_db or os.path.join(a.root,"fundamentals.db")
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    si_db=a.si_db or os.path.join(a.root,"short_interest.db")
    print("\n"+LINE+"\nGROSS PROFITABILITY VALIDATOR (dir=%+d, hold=%dd)\n"%(a.direction,a.hold)+LINE)
    if not os.path.isfile(fund_db): print("  [STOP] fundamentals.db not found"); return
    if not os.path.isfile(prices_db): print("  [STOP] prices.db not found"); return

    # GP readings with filed_date (PIT)
    c=ro(fund_db)
    try: grows=Q(c,"SELECT ticker,filed_date,gp FROM gross_profitability WHERE gp IS NOT NULL AND filed_date IS NOT NULL")
    finally: c.close()
    gp_events=defaultdict(list)  # tk -> [(filed_date, gp)]
    for tk,fd,gp in grows:
        do=nd(fd)
        if do is None: continue
        try: g=float(gp)
        except Exception: continue
        gp_events[tk.upper()].append((do,g))
    for tk in gp_events: gp_events[tk].sort()
    def gp_asof(tk,f):
        evs=gp_events.get(tk)
        if not evs: return None
        best=None
        for fd,g in evs:
            if fd<=f: best=(fd,g)
            else: break
        if best is None: return None
        if (f-best[0]).days>a.max_stale_days: return None   # too stale
        return best[1]

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
    def momentum(tk,d):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,6):
            cc=d-datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        end=i-a.mom_skip; start=i-a.mom_lookback
        if start<0 or end<=start: return None
        p0=lst[start][1]; p1=lst[end][1]
        return (p1/p0-1.0) if p0>0 else None

    si_by_date=defaultdict(dict)
    if os.path.isfile(si_db):
        cs=ro(si_db)
        try: sirows=Q(cs,"SELECT ticker,settlement_date,days_to_cover FROM short_interest")
        finally: cs.close()
        for tk,d,v in sirows:
            do=nd(d)
            if do is None or v is None: continue
            try: fv=float(v)
            except Exception: continue
            if fv<=50.0: si_by_date[do][tk.upper()]=fv
    si_dates=sorted(si_by_date)
    def si_asof(tk,d):
        best=None
        for sd in si_dates:
            if sd<=d: best=sd
            else: break
        return si_by_date[best].get(tk.upper()) if best else None

    # formation months span the GP filed-date range
    all_filed=[fd for evs in gp_events.values() for fd,_ in evs]
    if not all_filed: print("  [STOP] no GP filed dates."); return
    start=month_end(min(all_filed)); 
    pxmax=max(d for lst in px.values() for d,_ in lst)
    months=[]; m=start
    while m<=pxmax:
        months.append(m); m=month_end(m+datetime.timedelta(days=1))

    cross={}
    for f in months:
        recs={}
        for tk in gp_events:
            g=gp_asof(tk,f)
            if g is None: continue
            r=fwd(tk,f,a.hold)
            if r is None: continue
            recs[tk]=dict(gp=g, ret=r, mom=momentum(tk,f), dtc=si_asof(tk,f))
        if len(recs)>=a.min_names: cross[f]=recs

    lag=max(1,int(math.ceil(a.hold/21.0)))
    def ic_series(shuffle=False,rng=None):
        ics=[]; dts=[]
        for f in sorted(cross):
            recs=cross[f]
            sig=np.array([v["gp"] for v in recs.values()])*a.direction
            ret=np.array([v["ret"] for v in recs.values()])
            if shuffle: ret=rng.permutation(ret)
            ic=spearman(sig,ret)
            if ic is not None: ics.append(ic); dts.append(f)
        return np.array(ics),dts
    ics,dts=ic_series(); N=len(dts)
    if N<8: print("\n  [STOP] only %d usable monthly cross-sections (need >=%d names each). GP coverage may be thin."%(N,a.min_names)); return
    mean_ic=ics.mean(); se=nw_se(ics,lag); t=mean_ic/se if se else 0
    print("  %d monthly cross-sections, avg %.0f firms/month"%(N,np.mean([len(cross[f]) for f in cross])))
    print("\n"+"-"*78+"\nPER-MONTH IC (GP * dir %+d, hold=%dd)\n"%(a.direction,a.hold)+"-"*78)
    print("  mean IC = %+.4f | std = %.4f | IC IR = %+.3f"%(mean_ic,ics.std(),mean_ic/ics.std() if ics.std()>0 else 0))
    print("  %%-right-sign = %.0f%% | naive t = %+.2f | Newey-West t = %+.2f"
          %(100*np.mean(ics>0), mean_ic/(ics.std(ddof=1)/math.sqrt(N)) if ics.std(ddof=1)>0 else 0, t))

    yr=defaultdict(list)
    for ic,d in zip(ics,dts): yr[d.year].append(ic)
    print("\n  per-year mean IC:")
    for y in sorted(yr):
        v=np.array(yr[y]); print("   %d: %+.4f  (n=%d)"%(y,v.mean(),len(v)))
    half=N//2
    def tof(x):
        s=nw_se(x,lag); return x.mean()/s if s else 0
    print("\n  first half IC=%+.4f t=%+.2f | second half IC=%+.4f t=%+.2f"
          %(ics[:half].mean(),tof(ics[:half]),ics[half:].mean(),tof(ics[half:])))

    rng=np.random.default_rng(7); nulls=[]
    for _ in range(300):
        nc,_=ic_series(shuffle=True,rng=rng)
        if len(nc): nulls.append(nc.mean())
    nulls=np.array(nulls); z=(mean_ic-nulls.mean())/nulls.std() if nulls.std()>0 else 0
    print("\n  null control: real IC %.1f std's from shuffled null (need >=3)"%z)

    # decorrelation
    print("\n"+"-"*78+"\nDECORRELATION vs momentum and short interest (the brick-#3 test)\n"+"-"*78)
    mom_corr=[]; dtc_corr=[]
    for f in sorted(cross):
        recs=cross[f]
        tks=[tk for tk,v in recs.items() if v["mom"] is not None]
        if len(tks)>=a.min_names:
            rc=spearman(np.array([recs[tk]["gp"] for tk in tks])*a.direction, np.array([recs[tk]["mom"] for tk in tks]))
            if rc is not None: mom_corr.append(rc)
        tks2=[tk for tk,v in recs.items() if v["dtc"] is not None]
        if len(tks2)>=a.min_names:
            rc=spearman(np.array([recs[tk]["gp"] for tk in tks2])*a.direction, np.array([recs[tk]["dtc"] for tk in tks2]))
            if rc is not None: dtc_corr.append(rc)
    mc=np.mean(mom_corr) if mom_corr else float('nan')
    dc=np.mean(dtc_corr) if dtc_corr else float('nan')
    print("  mean rank-corr: GP vs MOMENTUM = %+.3f (n=%d)"%(mc,len(mom_corr)))
    print("  mean rank-corr: GP vs DAYS_TO_COVER = %+.3f (n=%d)"%(dc,len(dtc_corr)))
    print("  (|corr|<~0.3 = genuinely complementary; high = redundant)")

    print("\n"+LINE+"\nVERDICT — is gross profitability a brick, and uncorrelated?\n"+LINE)
    is_brick = z>=3 and t>=2.5 and np.mean(ics>0)>0.55
    uncorr = (abs(mc)<0.35 if not math.isnan(mc) else False) and (abs(dc)<0.35 if not math.isnan(dc) else True)
    if is_brick and uncorr:
        print("  >> CANDIDATE BRICK #3 + UNCORRELATED: IC %+.4f, NW t %+.2f, %.1f std's from null;"%(mean_ic,t,z))
        print("     rank-corr vs momentum %+.2f, vs SI %+.2f. Durable (GP doesn't decay) AND orthogonal"%(mc,dc))
        print("     -> the breadth the combination needed. NEXT: add GP to the book decomposition and")
        print("     re-test the lift (does combining SI+GP finally beat SI alone?).")
    elif is_brick and not uncorr:
        print("  >> REAL BUT CORRELATED: IC significant (t %+.2f, %.1f from null) but overlaps momentum/SI"%(t,z))
        print("     (rank-corr %+.2f / %+.2f). Limited diversification value -- the PEAD+SI problem again."%(mc,dc))
    elif z<3:
        print("  >> NOT A BRICK: real IC within the shuffled null (%.1f std's). GP doesn't predict on your"%z)
        print("     universe at this horizon. Try --hold 60 (GP is slow), or the coverage (252 names,")
        print("     large-cap-tilted) may be too narrow for a fundamental cross-section.")
    else:
        print("  >> SUGGESTIVE but below bar: IC %+.4f (t %+.2f, null %.1f). Try --hold 60."%(mean_ic,t,z))
    print("\n  Honest n=%d monthly cross-sections. In-sample, survivor-tilted; large-cap GP coverage."%N)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
