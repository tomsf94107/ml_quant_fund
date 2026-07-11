#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — DECORRELATION TEST: SHORT INTEREST vs MOMENTUM (+ PEAD) @ 40d
================================================================================
Short interest is a real 40-day cross-sectional signal (IC -0.054, 8.3 sigma) but
useless at 1/3/5d (the model's horizons). The open question: does it help a ~40-day
model -- specifically, is it DECORRELATED from MOMENTUM (the obvious 40d+ signal)?

  - DECORRELATED -> adds breadth; a 40d multi-signal model has a real case
  - REDUNDANT with momentum -> "stocks people bet against" == "past losers";
    combining adds nothing; the answer is a clean "no, it doesn't help"

This measures THREE signals at h=40, all cross-sectional, all PIT:
  1. MOMENTUM  : 12-1 month return (skip most recent month to avoid 1m reversal),
                 computed from prices. High momentum = predicted high return.
  2. PEAD      : SUE (PIT-trailing), high SUE = predicted high return.
  3. SHORT-INT : days_to_cover, HIGH short = predicted LOW return (sign -1).

TWO kinds of correlation, both reported (they answer different questions):
  A. SIGNAL correlation (per-date, cross-sectional): do the signals RANK the same
     stocks the same way on a given date? High |corr| = redundant ranking.
  B. IC-SERIES correlation (across dates): are the signals "right" on the same dates?
     Low/negative = complementary timing (the diversification that matters).

Plus each signal's standalone IC at 40d (so we know they're each real on this sample),
and a null control (shuffle returns -> all ICs and correlations vanish).

DECISION:
  - SI signal-corr with momentum LOW (|r|<~0.3) AND IC-series corr low -> COMPLEMENTARY,
    real case for a 40d model. Proceed to the build decision.
  - SI signal-corr with momentum HIGH -> REDUNDANT, short interest is mostly captured
    by momentum at 40d. No new model needed; clean "no".

RULE 1: momentum computed PIT (return ending BEFORE formation date, skip-month);
SUE PIT-trailing; forward returns strictly after formation; per-date IC + Newey-West;
null control. READ-ONLY. No network.

USAGE:
  python decorrelation_test.py --root .
  python decorrelation_test.py --root . --hold 40 --mom-lookback 252 --mom-skip 21
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
def spearman(x,y):
    n=len(x)
    if n<5: return None
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return None
    return float(np.corrcoef(rx,ry)[0,1])
def ranks(v): return np.argsort(np.argsort(v)).astype(float)
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
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--mom-lookback",type=int,default=252)  # ~12 months trading days
    ap.add_argument("--mom-skip",type=int,default=21)        # skip most recent ~1 month
    ap.add_argument("--pead-window",type=int,default=45)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--clip-dtc",type=float,default=50.0)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=os.path.join(a.root,"prices.db"); si_db=os.path.join(a.root,"short_interest.db")
    earnp=find_db(a.root,"earnings.db")
    print("\n"+LINE+"\nDECORRELATION TEST — SHORT INTEREST vs MOMENTUM (+PEAD) @ h=%d\n"%a.hold+LINE)
    for lbl,p in (("prices.db",prices_db),("short_interest.db",si_db)):
        if not p or not os.path.isfile(p): print("[STOP] %s not found"%lbl); return
    have_pead = earnp and os.path.isfile(earnp)
    if not have_pead: print("  (earnings.db not found -- PEAD column skipped; SI vs momentum still runs)")

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
        for off in range(0,6):
            cc=d+datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+h
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None
    def momentum(tk,d):
        """12-1 month momentum: return from (d - lookback) to (d - skip), PIT (all before d)."""
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

    # short interest by date
    c=ro(si_db)
    try: sirows=Q(c,"SELECT ticker,settlement_date,days_to_cover FROM short_interest")
    finally: c.close()
    si_by_date=defaultdict(dict)
    for tk,d,v in sirows:
        do=nd(d)
        if do is None or v is None: continue
        try: fv=float(v)
        except Exception: continue
        if a.clip_dtc and fv>a.clip_dtc: continue
        si_by_date[do][tk.upper()]=fv

    # PEAD SUE
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

    # build per-date aligned panels on the SI grid (SI defines the dates)
    # at each date, for stocks present in ALL required signals, collect signed signals + fwd ret
    def compute(shuffle=False, rng=None):
        # returns dicts of per-date IC series and per-date signal-correlations
        ic_mom=[]; ic_si=[]; ic_pead=[]; sigcorr_mom_si=[]; sigcorr_mom_pead=[]; sigcorr_si_pead=[]
        n_per=[]
        for d in sorted(si_by_date):
            dtc=si_by_date[d]
            recs=[]
            for tk,dv in dtc.items():
                if tk not in pos_of: continue
                m=momentum(tk,d)
                if m is None: continue
                r=fwd(tk,d,a.hold)
                if r is None: continue
                s=sue_asof(tk,d) if have_pead else None
                recs.append((tk,m,dv,s,r))
            # for momentum vs SI we only need m,dtc,r
            base=[(tk,m,dv,r) for (tk,m,dv,s,r) in recs]
            if len(base)<a.min_names: continue
            mom=np.array([m for _,m,_,_ in base])
            si=np.array([dv for _,_,dv,_ in base])*(-1.0)   # high short -> low ret -> negate
            ret=np.array([r for _,_,_,r in base])
            if shuffle: ret=rng.permutation(ret)
            ic_mom.append(spearman(mom,ret)); ic_si.append(spearman(si,ret))
            # signal correlation: rank-correlation of the two signals' rankings
            sigcorr_mom_si.append(spearman(mom,si))
            n_per.append(len(base))
            # PEAD subset (stocks that ALSO have fresh SUE)
            if have_pead:
                psub=[(tk,m,dv,s,r) for (tk,m,dv,s,r) in recs if s is not None]
                if len(psub)>=max(8,a.min_names//2):
                    pm=np.array([m for _,m,_,_,_ in psub])
                    ps=np.array([dv for _,_,dv,_,_ in psub])*(-1.0)
                    pp=np.array([s for _,_,_,s,_ in psub])
                    pr=np.array([r for _,_,_,_,r in psub])
                    if shuffle: pr=rng.permutation(pr)
                    ic_pead.append(spearman(pp,pr))
                    sigcorr_mom_pead.append(spearman(pm,pp))
                    sigcorr_si_pead.append(spearman(ps,pp))
        return (np.array(ic_mom),np.array(ic_si),np.array(ic_pead),
                np.array(sigcorr_mom_si),np.array(sigcorr_mom_pead),np.array(sigcorr_si_pead),n_per)

    icm,ics,icp,scms,scmp,scsp,nper = compute()
    N=len(icm)
    if N<6:
        print("  [STOP] only %d usable dates (need momentum + SI + fwd ret)."%N); return
    lag=max(1,int(math.ceil(a.hold/15.0)))
    print("  %d dates, avg %d stocks/date (with momentum + short interest + 40d fwd return)"%(N,int(np.mean(nper))))

    def ic_line(label,arr):
        if len(arr)==0: print("  %-14s [n/a]"%label); return None
        m=arr.mean(); se=nw_se(arr,lag); t=m/se if se else 0
        print("  %-14s mean IC=%+.4f  NW t=%+.2f  (n=%d dates)"%(label,m,t,len(arr)))
        return m
    print("\n"+"-"*78+"\nSTANDALONE 40d IC (confirm each signal is real on this sample)\n"+"-"*78)
    ic_line("MOMENTUM",icm); ic_line("SHORT-INT",ics)
    if have_pead and len(icp): ic_line("PEAD",icp)

    print("\n"+"-"*78+"\nA. SIGNAL CORRELATION (per-date cross-sectional rank corr)\n"+"-"*78)
    print("   do the signals RANK stocks the same way? high |r| = redundant ranking")
    def corr_line(label,arr):
        if len(arr)==0: print("  %-22s [n/a]"%label); return None
        m=np.nanmean(arr)
        print("  %-22s mean rank-corr = %+.3f"%(label,m))
        return m
    cms=corr_line("MOMENTUM vs SHORT-INT",scms)
    if have_pead:
        corr_line("MOMENTUM vs PEAD",scmp); corr_line("SHORT-INT vs PEAD",scsp)

    print("\n"+"-"*78+"\nB. IC-SERIES CORRELATION (are they 'right' on the same dates?)\n"+"-"*78)
    print("   low/negative = complementary timing (the diversification that matters)")
    if icm.std()>0 and ics.std()>0:
        ic_corr_ms=float(np.corrcoef(icm,ics)[0,1])
        print("  MOMENTUM vs SHORT-INT IC-series corr = %+.3f"%ic_corr_ms)
    else: ic_corr_ms=0

    print("\n"+"-"*78+"\nNULL CONTROL (shuffle returns within date -> ICs vanish)\n"+"-"*78)
    rng=np.random.default_rng(11); nm=[]; ns=[]
    for _ in range(150):
        a1,a2,_,_,_,_,_=compute(shuffle=True,rng=rng)
        if len(a1): nm.append(a1.mean()); ns.append(a2.mean())
    nm=np.array(nm); ns=np.array(ns)
    zm=(icm.mean()-nm.mean())/nm.std() if nm.std()>0 else 0
    zs=(ics.mean()-ns.mean())/ns.std() if ns.std()>0 else 0
    print("  momentum real IC %.1f sd from null | short-int real IC %.1f sd from null"%(zm,zs))

    print("\n"+LINE+"\nVERDICT — is short interest COMPLEMENTARY to momentum at 40d?\n"+LINE)
    if cms is None:
        print("  >> inconclusive (no signal-corr computed)"); return
    redundant = abs(cms)>=0.4
    mild = 0.2<=abs(cms)<0.4
    if redundant:
        print("  >> REDUNDANT: momentum & short-interest rank stocks similarly (rank-corr %+.3f)."%cms)
        print("     Short interest is largely 'past losers' = what momentum already captures. Adding")
        print("     it to a momentum model buys little. Clean answer: it does NOT meaningfully help.")
    elif mild:
        print("  >> PARTIALLY OVERLAPPING: rank-corr %+.3f -- some shared ranking, some independent."%cms)
        print("     Short interest would add MODEST incremental breadth to a momentum model, not a lot.")
        print("     IC-series corr %+.3f. Worth it only if a 40d model is being built anyway."%ic_corr_ms)
    else:
        print("  >> COMPLEMENTARY: low signal rank-corr (%+.3f) -- momentum & short interest rank"%cms)
        print("     stocks differently and are 'right' on different dates (IC-series corr %+.3f)."%ic_corr_ms)
        print("     Short interest WOULD add real breadth to a 40d cross-sectional model. There's a")
        print("     genuine case -- IF a ~40d model is worth building on its own merits (separate Q).")
    print("\n  Honest n=%d dates. Signal-corr is the decision number. In-sample, survivor-tilted."%N)
    print("  NOTE: 'complementary' means a 40d model COULD use it; whether to BUILD that model")
    print("  is a separate decision from whether the signal fits.")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
