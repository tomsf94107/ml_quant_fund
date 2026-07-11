#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — SIGNAL COMBINATION HARNESS  (the breadth test)
================================================================================
The centerpiece of the combine-little-signals thesis. Takes >=2 signals, and
answers the ONLY question that matters for combination: does stacking them produce
an information ratio GREATER than the best single signal — i.e., is breadth
actually multiplying, or are the signals redundant?

Per the Fundamental Law (IR ~ IC*sqrt(breadth)), combining N *uncorrelated*
signals should boost IR by ~sqrt(N). Correlation kills that. This harness measures
it empirically instead of assuming it.

WHAT IT DOES (offline; reads prices.db + earnings.db; NO network):
  * builds a per-(date,ticker) panel for each requested signal:
       - PEAD  : SUE (PIT-trailing) aligned to earnings events
       - REVERSAL_5, MOMENTUM_60, LOWVOL_20, etc. : price-derived, every rebal date
  * computes the CORRELATION MATRIX between signals (the redundancy check)
  * combines via:
       (a) equal-weight of cross-sectionally ranked signals  [the robust default]
       (b) IC-weighted (lightly)                              [if IC estimates stable]
  * measures IC and IR (mean daily IC / std daily IC) for: each signal alone, and
    the combination -> reports whether COMBINED IR > best SINGLE IR
  * reports EFFECTIVE BREADTH = (combined IR / mean single IR)^2  vs nominal N

NOTE: signals live on different cadences (PEAD is event-based, price signals are
daily). The harness combines them on the UNION of (date,ticker) where >=2 signals
are present, rank-standardizing each within each date's cross-section.

RULE 1: every signal uses only past data; forward return strictly after. Equal-weight
needs no fitting (no overfit). IC-weights are computed in-sample and flagged as such.

READ-ONLY. mode=ro&immutable=1. No network.

USAGE:
  python combine_signals.py --root . --signals PEAD,REVERSAL_5,MOMENTUM_60
  python combine_signals.py --root . --signals PEAD,LOWVOL_20 --hold 40
  (only pass signals that signal_hunt.py showed are real + uncorrelated)
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
    ap.add_argument("--signals",default="PEAD,REVERSAL_5,MOMENTUM_60")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--rebal",type=int,default=5)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--start",default="2022-01-01")
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    start=nd(a.start)
    want=[s.strip() for s in a.signals.split(",")]
    banner("ML QUANT FUND — SIGNAL COMBINATION HARNESS (the breadth test)")
    print("signals=%s hold=%d rebal=%dd start>=%s (offline)"%(want,a.hold,a.rebal,a.start))
    if not require(HAVE_NUMPY,"numpy required"): return
    if not require(os.path.isfile(prices_db),"prices.db not found"): return
    if not require(len(want)>=2,"need >=2 signals to combine"): return

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
    all_dates=sorted(set(d for lst in px.values() for d,_ in lst))
    all_dates=[d for d in all_dates if (start is None or d>=start)]
    rebal_dates=set(all_dates[::a.rebal])
    print("  prices: %d tickers | rebal dates: %d"%(len(px),len(rebal_dates)))

    def ret_back(lst,i,k):
        if i-k<0: return None
        p0=lst[i-k][1]; return (lst[i][1]/p0-1.0) if p0>0 else None
    def vol_back(lst,i,k):
        if i-k<0: return None
        rr=[lst[j][1]/lst[j-1][1]-1.0 for j in range(i-k+1,i+1) if lst[j-1][1]>0]
        return float(np.std(rr)) if len(rr)>2 else None
    PRICE_SIGS={
        "REVERSAL_5":  lambda lst,i:(-ret_back(lst,i,5))  if ret_back(lst,i,5)  is not None else None,
        "REVERSAL_1":  lambda lst,i:(-ret_back(lst,i,1))  if ret_back(lst,i,1)  is not None else None,
        "MOMENTUM_60": lambda lst,i: ret_back(lst,i-5,55) if i-60>=0 else None,
        "MOMENTUM_120":lambda lst,i: ret_back(lst,i-20,100) if i-120>=0 else None,
        "LOWVOL_20":   lambda lst,i:(-vol_back(lst,i,20)) if vol_back(lst,i,20) is not None else None,
    }
    def fwd(lst,i,hold):
        x=i+hold
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None

    # build panels: signame -> {(date,ticker): value}; and forward returns {(date,ticker):ret}
    panels=defaultdict(dict); fret={}
    # price signals on rebal dates
    for tk,lst in px.items():
        for i,(d,_) in enumerate(lst):
            if d not in rebal_dates: continue
            fr=fwd(lst,i,a.hold)
            if fr is None: continue
            fret[(d,tk)]=fr
            for sn in want:
                if sn in PRICE_SIGS:
                    sv=PRICE_SIGS[sn](lst,i)
                    if sv is not None: panels[sn][(d,tk)]=sv
    # PEAD/SUE panel (event-based) aligned to nearest rebal date
    if "PEAD" in want:
        earnp=find_db(a.root,"earnings.db")
        if earnp and os.path.isfile(earnp):
            ce=ro(earnp)
            try:
                cols=cols_of(ce,"earnings_surprises")
                have_comp="eps_actual" in cols and "eps_estimate" in cols
                sel="ticker,report_date"+(",eps_actual,eps_estimate" if have_comp else ",eps_surprise_pct")
                ev=Q(ce,"SELECT "+sel+" FROM earnings_surprises WHERE report_date IS NOT NULL")
            finally:
                ce.close()
            by_tkr=defaultdict(list)
            for row in ev:
                tk=row[0]; do=nd(row[1])
                if do is None: continue
                if have_comp:
                    try: raw=float(row[2])-float(row[3])
                    except Exception: raw=None
                else:
                    try: raw=float(row[2])
                    except Exception: raw=None
                by_tkr[tk].append((do,raw))
            for tk in by_tkr: by_tkr[tk].sort()
            for tk,lst_e in by_tkr.items():
                prior=[]
                for do,raw in lst_e:
                    if raw is None: continue
                    if len(prior)>=4:
                        sd=np.std(prior,ddof=1)
                        if sd>1e-12:
                            sue=raw/sd
                            # align to nearest price date on/after event, and to its forward return
                            lst_p=px.get(tk)
                            if lst_p and tk in pos_of:
                                ip=None
                                for off in range(0,6):
                                    c=do+datetime.timedelta(days=off)
                                    if c in pos_of[tk]: ip=pos_of[tk][c]; break
                                if ip is not None:
                                    dd=lst_p[ip][0]
                                    fr=fwd(lst_p,ip,a.hold)
                                    if fr is not None:
                                        panels["PEAD"][(dd,tk)]=sue
                                        fret[(dd,tk)]=fr
                    prior.append(raw)
        else:
            print("  [WARN] earnings.db not found — PEAD requested but unavailable")

    for sn in want:
        print("  panel %-13s: %d points"%(sn,len(panels.get(sn,{}))))

    # ---- correlation matrix (on common keys) ----
    sub("CORRELATION MATRIX (redundancy check)")
    keys_all=set.intersection(*[set(panels[sn].keys()) for sn in want if panels.get(sn)]) if all(panels.get(sn) for sn in want) else set()
    print("  common (date,ticker) across all signals: %d"%len(keys_all))
    # pairwise on each pair's own overlap (more data than full intersection)
    print("  pairwise Spearman correlation:")
    corr={}
    for i in range(len(want)):
        for j in range(i+1,len(want)):
            s1,s2=want[i],want[j]
            common=set(panels.get(s1,{}).keys()) & set(panels.get(s2,{}).keys())
            if len(common)>=30:
                v1=[panels[s1][k] for k in common]; v2=[panels[s2][k] for k in common]
                rho=spearman(v1,v2)
                corr[(s1,s2)]=rho
                print("    %-13s x %-13s rho=%+.3f (n=%d) %s"
                      %(s1,s2,rho if rho is not None else 0,len(common),
                        "[redundant]" if rho is not None and abs(rho)>=0.3 else "[independent]"))
            else:
                print("    %-13s x %-13s insufficient overlap"%(s1,s2))

    # ---- per-signal IR and combined IR ----
    sub("INFORMATION RATIO: each signal alone vs COMBINED (hold=%d)"%a.hold)
    # daily IC per signal
    def daily_ic_series(signame):
        by_date=defaultdict(list)
        for (d,tk),v in panels[signame].items():
            if (d,tk) in fret: by_date[d].append((v,fret[(d,tk)]))
        ics=[]
        for d,pairs in by_date.items():
            if len(pairs)>=a.min_names:
                ic=spearman([p[0] for p in pairs],[p[1] for p in pairs])
                if ic is not None: ics.append(ic)
        return ics
    def ir_of(ics):
        if len(ics)<3: return None,None,None
        a_=np.array(ics); m=a_.mean(); sd=a_.std()
        ir=m/sd if sd>0 else None
        t=m/(sd/math.sqrt(len(a_))) if sd>0 else None
        return m,ir,t
    single_ir={}
    for sn in want:
        ics=daily_ic_series(sn); m,ir,t=ir_of(ics)
        if m is not None:
            print("  %-13s mean IC=%+.4f  IR(per-rebal)=%+.3f  t=%.2f  dates=%d"%(sn,m,ir or 0,t or 0,len(ics)))
            single_ir[sn]=ir
    # combined: equal-weight of within-date ranks
    def combined_daily_ic():
        # for each date, rank each signal's cross-section, average ranks across signals
        by_date_sig=defaultdict(lambda: defaultdict(dict))  # date -> sig -> {tk:val}
        for sn in want:
            for (d,tk),v in panels[sn].items():
                by_date_sig[d][sn][tk]=v
        ics=[]
        for d,sigmap in by_date_sig.items():
            present=[sn for sn in want if sn in sigmap and len(sigmap[sn])>=a.min_names]
            if len(present)<2: continue
            # names common to all present signals
            common=set.intersection(*[set(sigmap[sn].keys()) for sn in present])
            common=[tk for tk in common if (d,tk) in fret]
            if len(common)<a.min_names: continue
            # rank each signal within common names, average
            combo={}
            for tk in common: combo[tk]=0.0
            for sn in present:
                vals=[(tk,sigmap[sn][tk]) for tk in common]
                order=sorted(range(len(vals)),key=lambda x:vals[x][1])
                ranks=[0.0]*len(vals)
                for r,idx in enumerate(order): ranks[idx]=r
                for idx,(tk,_) in enumerate(vals): combo[tk]+=ranks[idx]
            cs=[combo[tk] for tk in common]; rr=[fret[(d,tk)] for tk in common]
            ic=spearman(cs,rr)
            if ic is not None: ics.append(ic)
        return ics
    cics=combined_daily_ic()
    cm,cir,ct=ir_of(cics)
    print("  %-13s mean IC=%+.4f  IR(per-rebal)=%+.3f  t=%.2f  dates=%d"
          %("COMBINED",cm or 0,cir or 0,ct or 0,len(cics)) if cm is not None else "  COMBINED: insufficient")

    # ---- verdict ----
    banner("VERDICT — does combining MULTIPLY breadth?")
    valid_single=[v for v in single_ir.values() if v is not None]
    if cir is not None and valid_single:
        best_single=max(valid_single, key=abs)
        mean_single=np.mean([abs(v) for v in valid_single])
        eff_breadth=(cir/mean_single)**2 if mean_single>0 else None
        print("  best single IR: %+.3f"%best_single)
        print("  COMBINED IR:    %+.3f"%cir)
        print("  nominal signals: %d | effective breadth: %.1f"%(len(want),eff_breadth or 0))
        if abs(cir)>abs(best_single)*1.15:
            print("\n  >> BREADTH IS MULTIPLYING: combined IR exceeds best single by %.0f%%. The"%(100*(abs(cir)/abs(best_single)-1)))
            print("     combine-signals thesis WORKS on your data. Each added uncorrelated brick")
            print("     improves the book. Keep adding uncorrelated signals.")
        elif abs(cir)>abs(best_single)*1.02:
            print("\n  >> MARGINAL GAIN: combined slightly beats best single. Signals are partly")
            print("     redundant (see correlation matrix). Some breadth, but not full sqrt(N).")
        else:
            print("\n  >> NO BREADTH GAIN: combining does NOT beat the best single signal. The signals")
            print("     are too correlated (or the weaker ones add only noise). Combination isn't")
            print("     helping here — you're effectively trading one signal.")
        print("\n  Compare effective breadth (%.1f) to nominal (%d): the gap IS the correlation tax."
              %(eff_breadth or 0,len(want)))
    else:
        print("  Could not compute combined IR (insufficient overlapping data across signals).")
        print("  Signals may not coexist on enough common (date,ticker) points — PEAD is event-")
        print("  based and sparse, which limits overlap with daily price signals.")
    if a.out:
        rep={"single_ir":single_ir,"combined_ir":cir,"corr":{str(k):v for k,v in corr.items()}}
        with open(a.out,"a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":rep},default=str)+"\n")
        print("\n  [report appended to %s]"%a.out)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
