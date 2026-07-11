#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — COMBINED 40d BOOK: OUT-OF-SAMPLE TEST
================================================================================
The in-sample prototype showed the combined (momentum+PEAD+short-interest) 40d book
is stable and beats singles (Sharpe 1.07, both halves positive, Calmar 1.09). This
is the necessary-but-not-sufficient follow-up: does it survive OUT-OF-SAMPLE?

WHY THE SPLIT IS CLEAN (important): this strategy has NO FITTED PARAMETERS.
  - signals are fixed a priori (momentum 12-1, SUE PIT-trailing, days_to_cover)
  - combination is EQUAL-WEIGHT (nothing tuned)
  - z-scoring + quintile breaks are PER-DATE cross-sectional (no full-sample leak)
So there's nothing that could have been overfit to the in-sample period. The honest
OOS test is therefore simply: does the SAME fixed strategy perform similarly in a
holdout period we now treat as unseen?

DESIGN:
  * Split the SI settlement dates chronologically:
      IN-SAMPLE  = first --split fraction (default 65%)
      HOLDOUT    = remaining dates (treated as unseen)
  * Run the identical combined book in each, report separately:
      Sharpe (HAC SE), bootstrap CI, annRet, maxDD, hit, Calmar
  * Decay expectation: McLean-Pontiff ~26% OOS haircut is NORMAL for a real signal.
    OOS Sharpe in the same ballpark (CI clears 0) -> edge persists, real.
    OOS Sharpe collapses to ~0 / CI spans 0 -> the IS result was regime-luck.

SURVIVORSHIP CAVEAT (cannot fully fix, quantified here): prices.db is survivor-tilted
-- delisted tickers are largely absent. This inflates the LEVEL of Sharpe in BOTH IS
and OOS, so the IS-vs-OOS PERSISTENCE comparison stays meaningful (bias is in level,
not persistence). The script reports universe coverage so the bias is explicit. The
honest claim from this test is about PERSISTENCE, not the absolute Sharpe number.

RULE 1: momentum PIT; SUE PIT-trailing; forward returns strictly after formation;
combined score same-date only; no fitted params so no split leak; HAC + bootstrap;
survivorship flagged + quantified. READ-ONLY.

USAGE:
  python combined_40d_oos.py --root .
  python combined_40d_oos.py --root . --split 0.65 --cost-bps 10
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
def find_db(root,name):
    c=os.path.join(root,name)
    if os.path.isfile(c): return c
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f==name: return os.path.join(dp,f)
    return None
def ranks(v): return np.argsort(np.argsort(np.asarray(v,float))).astype(float)
def zscore(v):
    v=np.asarray(v,float); m=v.mean(); s=v.std(ddof=1)
    return (v-m)/s if s>1e-12 else np.zeros_like(v)
def maxdd(curve):
    peak=-1e18; mdd=0
    for x in curve:
        peak=max(peak,x); mdd=min(mdd,x-peak)
    return mdd
def nw_se(x,lag):
    x=np.asarray(x,float); n=len(x)
    if n<2: return None
    e=x-x.mean(); g0=float(e@e)/n; s=g0
    for k in range(1,min(lag,n-1)+1):
        gk=float(e[k:]@e[:-k])/n; w=1.0-k/(lag+1.0); s+=2.0*w*gk
    return math.sqrt(s/n) if s>0 else None
LINE="="*78

def metrics(rets, ppy, lag):
    rets=np.asarray(rets,float); n=len(rets)
    if n<3: return None
    m=rets.mean(); v=rets.std(ddof=1)
    sr=(m/v*math.sqrt(ppy)) if v>0 else 0
    se_mean=nw_se(rets,lag); se=(se_mean/v)*math.sqrt(ppy) if (se_mean and v>0) else 0
    curve=np.cumsum(rets); mdd=maxdd(curve); ann=m*ppy
    return dict(n=n,sharpe=sr,se=se,ann=ann,vol=v*math.sqrt(ppy),hit=100*np.mean(rets>0),
                mdd=mdd,calmar=(ann/abs(mdd) if mdd<0 else float('inf')))

def boot_ci(rets, ppy, n_boot, seed=42, block=3):
    rets=np.asarray(rets,float); n=len(rets); rng=np.random.default_rng(seed)
    def bres(N):
        out=[]
        while len(out)<N:
            s=rng.integers(0,N); out.extend([(s+j)%N for j in range(block)])
        return np.array(out[:N])
    def sa(x):
        v=x.std(ddof=1); return (x.mean()/v*math.sqrt(ppy)) if v>0 else 0
    b=np.array([sa(rets[bres(n)]) for _ in range(n_boot)])
    return np.percentile(b,[2.5,97.5]), float(np.mean(b>0))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--mom-lookback",type=int,default=252)
    ap.add_argument("--mom-skip",type=int,default=21)
    ap.add_argument("--pead-window",type=int,default=45)
    ap.add_argument("--quantile",type=float,default=0.2)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--max-weight",type=float,default=0.05)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--split",type=float,default=0.65)
    ap.add_argument("--n-boot",type=int,default=5000)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=os.path.join(a.root,"prices.db"); si_db=os.path.join(a.root,"short_interest.db")
    earnp=find_db(a.root,"earnings.db")
    print("\n"+LINE+"\nCOMBINED 40d BOOK — OUT-OF-SAMPLE TEST\n"+LINE)
    for lbl,p in (("prices.db",prices_db),("short_interest.db",si_db)):
        if not p or not os.path.isfile(p): print("[STOP] %s not found"%lbl); return
    have_pead = earnp and os.path.isfile(earnp)

    cp=ro(prices_db)
    try: prows=Q(cp,"SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL")
    finally: cp.close()
    px=defaultdict(list); all_price_tk=set()
    for tk,d,p in prows:
        do=nd(d)
        if do is None: continue
        try: pf=float(p)
        except Exception: continue
        if pf>0: px[tk].append((do,pf)); all_price_tk.add(tk)
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

    sue_events=defaultdict(list)
    if have_pead:
        ce=ro(earnp)
        try:
            cl=cols_of(ce,"earnings_surprises")
            comp="eps_actual" in cl and "eps_estimate" in cl
            sel="ticker,report_date"+(",eps_actual,eps_estimate" if comp else ",eps_surprise_pct")
            ev=Q(ce,"SELECT "+sel+" FROM earnings_surprises WHERE report_date IS NOT NULL")
        finally: ce.close()
        by_tkr=defaultdict(list)
        for row in ev:
            tk=row[0]; do=nd(row[1])
            if do is None: continue
            if comp:
                try: raw=float(row[2])-float(row[3])
                except Exception: raw=None
            else:
                try: raw=float(row[2])
                except Exception: raw=None
            by_tkr[tk].append((do,raw))
        for tk in by_tkr: by_tkr[tk].sort()
        for tk,lst in by_tkr.items():
            prior=[]
            for do,raw in lst:
                if raw is None: continue
                if len(prior)>=4:
                    sd=np.std(prior,ddof=1)
                    if sd>1e-12: sue_events[tk].append((do,raw/sd))
                prior.append(raw)
    def sue_asof(tk,d):
        evs=sue_events.get(tk)
        if not evs: return None
        recent=[(ed,s) for ed,s in evs if 0<=(d-ed).days<=a.pead_window]
        if not recent: return None
        recent.sort(); return recent[-1][1]

    grid=sorted(si_by_date)
    ppy=365.25/float(a.hold); lag=max(1,int(math.ceil(a.hold/15.0)))

    def s_combined(mom,si,pead):
        zm=zscore(ranks(mom)); zs=zscore(ranks(-np.asarray(si,float)))
        comps=[zm,zs]
        if have_pead:
            arr=np.array([x if x is not None else np.nan for x in pead],float)
            zp=np.zeros(len(arr)); mask=~np.isnan(arr)
            if mask.sum()>=5: zp[mask]=zscore(ranks(arr[mask]))
            comps.append(zp)
        return np.mean(comps,axis=0)

    def run_on(dates):
        prev_w={}; rets=[]; rdates=[]
        for d in dates:
            dtc=si_by_date[d]
            names=[]; mom=[]; si=[]; pead=[]
            for tk,dv in dtc.items():
                if tk not in pos_of: continue
                m=momentum(tk,d)
                if m is None: continue
                r=fwd(tk,d,a.hold)
                if r is None: continue
                names.append(tk); mom.append(m); si.append(dv)
                pead.append(sue_asof(tk,d) if have_pead else None)
            if len(names)<a.min_names: continue
            score=s_combined(np.array(mom),np.array(si),pead)
            order=np.argsort(score); q=max(1,int(len(names)*a.quantile))
            wd={}
            for j in order[-q:]: wd[names[j]]=min(a.max_weight,1.0/(2*q))
            for j in order[:q]: wd[names[j]]=-min(a.max_weight,1.0/(2*q))
            allk=set(wd)|set(prev_w); turn=sum(abs(wd.get(k,0)-prev_w.get(k,0)) for k in allk)
            r=0.0
            for tk,wi in wd.items():
                fr=fwd(tk,d,a.hold)
                if fr is not None: r+=wi*fr
            r-=turn*(a.cost_bps/10000.0)
            rets.append(r); rdates.append(d); prev_w=wd
        return np.array(rets), rdates

    # chronological split
    split_idx=int(len(grid)*a.split)
    is_dates=grid[:split_idx]; oos_dates=grid[split_idx:]
    print("  total SI dates=%d | IN-SAMPLE=%d (%s..%s) | HOLDOUT=%d (%s..%s)"
          %(len(grid),len(is_dates),is_dates[0],is_dates[-1],len(oos_dates),oos_dates[0],oos_dates[-1]))
    print("  split=%.0f%% | NO fitted parameters -> clean split (nothing tuned to in-sample)"%(100*a.split))

    is_ret,_=run_on(is_dates); oos_ret,_=run_on(oos_dates)
    nis=len(is_ret); noos=len(oos_ret)
    if noos<8:
        print("\n  [STOP] holdout has only %d rebalances -- too few for an OOS read. Lower --split or"%noos)
        print("  accept the OOS sample is too thin (the data-sparsity wall again).")
        return

    mis=metrics(is_ret,ppy,lag); moos=metrics(oos_ret,ppy,lag)
    is_ci,is_p=boot_ci(is_ret,ppy,a.n_boot); oos_ci,oos_p=boot_ci(oos_ret,ppy,a.n_boot)

    print("\n"+"-"*78+"\nIN-SAMPLE vs HOLDOUT (same fixed strategy, net %.0f bps)\n"%a.cost_bps+"-"*78)
    print("  %-12s %7s %8s %8s %7s %7s %6s"%("period","Sharpe","CI_lo","CI_hi","annRet","maxDD","n"))
    print("  %-12s %+7.2f %+8.2f %+8.2f %+6.1f%% %6.1f%% %5d"
          %("IN-SAMPLE",mis["sharpe"],is_ci[0],is_ci[1],100*mis["ann"],100*mis["mdd"],nis))
    print("  %-12s %+7.2f %+8.2f %+8.2f %+6.1f%% %6.1f%% %5d"
          %("HOLDOUT",moos["sharpe"],oos_ci[0],oos_ci[1],100*moos["ann"],100*moos["mdd"],noos))
    decay = (1 - moos["sharpe"]/mis["sharpe"]) if mis["sharpe"]>0 else float('nan')
    print("\n  OOS Sharpe decay vs IS = %.0f%%  (McLean-Pontiff: ~26%% is normal for a real signal)"%(100*decay))

    # survivorship coverage
    si_tk=set(tk for d in grid for tk in si_by_date[d])
    covered=len(si_tk & all_price_tk)
    print("\n"+"-"*78+"\nSURVIVORSHIP CAVEAT (quantified)\n"+"-"*78)
    print("  prices.db tickers=%d | SI tickers=%d | covered by prices=%d (%.0f%%)"
          %(len(all_price_tk),len(si_tk),covered,100*covered/max(len(si_tk),1)))
    print("  prices.db is SURVIVOR-TILTED: delisted names largely absent -> inflates Sharpe LEVEL")
    print("  in BOTH periods. The IS-vs-OOS PERSISTENCE comparison stays valid; the absolute Sharpe")
    print("  is optimistic. A true survivorship-free test needs a point-in-time-constituents dataset.")

    print("\n"+LINE+"\nVERDICT — does the combined edge survive out-of-sample?\n"+LINE)
    if oos_ci[0]>0 and moos["sharpe"]>0.5:
        print("  >> SURVIVES: holdout Sharpe %+.2f with CI lower bound %+.2f>0 (decay %.0f%% — if negative, holdout EXCEEDS in-sample = favorable-regime tailwind, level optimistic; if ~26%% normal"%(moos["sharpe"],oos_ci[0],100*decay))
        print("     range). The edge PERSISTS on data it had no hand in. Combined with no fitted params,")
        print("     this is real evidence -- the strongest case yet for a 40d book. REMAINING gate before")
        print("     building: survivorship-free confirmation (need PIT-constituents data) + capacity/cost")
        print("     realism. But on this test, it held.")
    elif oos_ci[0]>0:
        print("  >> SURVIVES WEAKLY: holdout Sharpe %+.2f, CI [%+.2f,%+.2f] clears 0 but is modest."%(moos["sharpe"],oos_ci[0],oos_ci[1]))
        print("     Edge persists OOS but thin. Real but not strong; a 40d system would be a low-Sharpe")
        print("     book. Reasonable to keep researching (survivorship-free next) before committing.")
    elif moos["sharpe"]>0:
        print("  >> INCONCLUSIVE OOS: holdout Sharpe %+.2f but CI [%+.2f,%+.2f] SPANS ZERO (n=%d too thin"%(moos["sharpe"],oos_ci[0],oos_ci[1],noos))
        print("     to resolve). Positive point estimate, not significant. The OOS sample is too small")
        print("     to confirm -- same sparsity wall. Can't greenlight a build on this; needs more data.")
    else:
        print("  >> DOES NOT SURVIVE: holdout Sharpe %+.2f -- the in-sample edge DECAYED to nothing OOS."%moos["sharpe"])
        print("     The IS result (1.07, stable halves) was likely regime-luck after all. Do NOT build a")
        print("     40d system on this. The decorrelation finding stands as knowledge; the tradeable")
        print("     book does not survive honest OOS. A clean, valuable negative.")
    print("\n  Honest: IS n=%d, OOS n=%d, 40d hold, %.0f bps, HAC lag=%d. Survivor-tilted (see above)."%(nis,noos,a.cost_bps,lag))
    print("  No fitted parameters, so the split is clean -- but OOS n=%d is the binding limit on certainty."%noos)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
