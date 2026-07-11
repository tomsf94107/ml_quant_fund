#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — 40d BOOK: DECOMPOSITION + REGIME ROBUSTNESS
================================================================================
combined_40d_oos.py proved the *combined book* survives OOS net-of-cost, block-
bootstrapped, split-robust. It did NOT test the two things that actually decide
whether to build a COMBINED book vs just trade one brick:

  Q1 (the cousin's failed test): does combining BEAT THE BEST SINGLE BRICK,
      net-of-cost, out-of-sample? (PEAD+SI combination failed this: P(lift>0)=26%.)
  Q2: does PEAD add anything to the book, or is it momentum+SI doing the work?
  Q3 (the inverted-decay tell): how regime-dependent is the edge? Per-year Sharpe.

This builds four books on the IDENTICAL universe + construction (only the SCORE
differs, so the rebalance dates line up and the streams are paired):
   mom_only | si_only | mom+si | full(mom+si+pead)
then:
  * standalone Sharpe (net) for each, IS and holdout, block-boot CI
  * LIFT of full vs best single, and full vs (mom+si): paired block-bootstrap of
    the Sharpe DIFFERENCE on the same resampled blocks -> CI + P(lift>0). This is
    the rigorous "does combining help" test (paired = the honest way to compare).
  * per-calendar-year Sharpe of the full book -> quantifies regime-dependence.

WHY PAIRED BOOTSTRAP: comparing two strategies' Sharpes by resampling each
independently overstates the difference's uncertainty AND ignores that they share
the same months. Resampling the SAME block indices for both, then differencing,
is the correct test for "is A better than B on the same data."

SURVIVORSHIP (stated, not fixed): prices.db + the SI universe are survivor-tilted;
delisted names were never in the data, so the bias can't even be bounded here. The
short leg (high days-to-cover) is exactly where delistings cluster, so the sign of
the bias is genuinely unknown. Every Sharpe below is an optimistic LEVEL; the
COMPARISONS (lift, per-year) are what this script is for.

RULE 1: identical audited construction to combined_40d_oos.py; momentum + SUE PIT;
forward returns strictly after formation; per-date z-score; net-of-cost turnover;
paired block bootstrap; READ-ONLY.

USAGE:
  python book_robustness.py --root .
  python book_robustness.py --root . --split 0.65 --cost-bps 25
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
def sharpe_of(x,ppy):
    x=np.asarray(x,float); v=x.std(ddof=1)
    return (x.mean()/v*math.sqrt(ppy)) if v>0 else 0.0

def boot_ci(rets, ppy, n_boot=5000, seed=42, block=3):
    rets=np.asarray(rets,float); n=len(rets); rng=np.random.default_rng(seed)
    def bidx(N):
        out=[]
        while len(out)<N:
            s=rng.integers(0,N); out.extend([(s+j)%N for j in range(block)])
        return np.array(out[:N])
    b=np.array([sharpe_of(rets[bidx(n)],ppy) for _ in range(n_boot)])
    return np.percentile(b,[2.5,97.5]), float(np.mean(b>0))

def boot_ci_lift(a, b, ppy, n_boot=5000, seed=42, block=3):
    """Paired block bootstrap of Sharpe(a)-Sharpe(b) on the SAME resampled blocks.
    a,b are aligned same-date return arrays. Returns (CI, P(lift>0), point_lift)."""
    a=np.asarray(a,float); bb=np.asarray(b,float); n=len(a); rng=np.random.default_rng(seed)
    def bidx(N):
        out=[]
        while len(out)<N:
            s=rng.integers(0,N); out.extend([(s+j)%N for j in range(block)])
        return np.array(out[:N])
    diffs=[]
    for _ in range(n_boot):
        idx=bidx(n)
        diffs.append(sharpe_of(a[idx],ppy)-sharpe_of(bb[idx],ppy))
    diffs=np.array(diffs)
    point=sharpe_of(a,ppy)-sharpe_of(bb,ppy)
    return np.percentile(diffs,[2.5,97.5]), float(np.mean(diffs>0)), point

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
    print("\n"+LINE+"\n40d BOOK — DECOMPOSITION + REGIME ROBUSTNESS\n"+LINE)
    for lbl,p in (("prices.db",prices_db),("short_interest.db",si_db)):
        if not p or not os.path.isfile(p): print("[STOP] %s not found"%lbl); return
    have_pead = earnp and os.path.isfile(earnp)
    if not have_pead: print("  [WARN] earnings.db not found -> PEAD component off (mom+si only)")

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

    def score_of(mom,si,pead,use_mom,use_si,use_pead):
        comps=[]
        if use_mom: comps.append(zscore(ranks(mom)))
        if use_si:  comps.append(zscore(ranks(-np.asarray(si,float))))
        if use_pead and have_pead:
            arr=np.array([x if x is not None else np.nan for x in pead],float)
            zp=np.zeros(len(arr)); mask=~np.isnan(arr)
            if mask.sum()>=5: zp[mask]=zscore(ranks(arr[mask]))
            comps.append(zp)
        if not comps: return None
        return np.mean(comps,axis=0)

    def run_on(dates,use_mom,use_si,use_pead):
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
            score=score_of(np.array(mom),np.array(si),pead,use_mom,use_si,use_pead)
            if score is None: continue
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

    split_idx=int(len(grid)*a.split)
    is_dates=grid[:split_idx]; oos_dates=grid[split_idx:]
    print("  SI dates=%d | IS=%d (%s..%s) | HOLDOUT=%d (%s..%s) | net %.0fbps, split %.0f%%"
          %(len(grid),len(is_dates),is_dates[0],is_dates[-1],len(oos_dates),oos_dates[0],oos_dates[-1],a.cost_bps,100*a.split))

    configs=[("mom_only",1,0,0),("si_only",0,1,0),("mom+si",1,1,0),("full(m+si+pead)",1,1,1)]
    # full-sample + holdout streams for each
    oos_streams={}; is_sharpe={}; oos_sharpe={}
    print("\n"+"-"*78+"\nSTANDALONE BOOKS (net, Sharpe [CI]) — identical universe & construction\n"+"-"*78)
    print("  %-18s %18s %18s"%("book","IN-SAMPLE Sharpe","HOLDOUT Sharpe[CI]"))
    for name,um,us,up in configs:
        isr,_=run_on(is_dates,um,us,up); oos,_=run_on(oos_dates,um,us,up)
        if len(oos)<8:
            print("  %-18s  (holdout too thin: n=%d)"%(name,len(oos))); continue
        oos_streams[name]=oos
        si_s=sharpe_of(isr,ppy); oo_s=sharpe_of(oos,ppy)
        ci,_=boot_ci(oos,ppy,a.n_boot)
        is_sharpe[name]=si_s; oos_sharpe[name]=oo_s
        print("  %-18s %+18.2f   %+.2f [%+.2f,%+.2f]"%(name,si_s,oo_s,ci[0],ci[1]))

    if "full(m+si+pead)" not in oos_streams:
        print("\n  [STOP] full book holdout too thin to decompose."); return

    # ---- Q1: does FULL beat the BEST SINGLE brick (paired, on holdout)? ----
    singles={k:oos_streams[k] for k in ("mom_only","si_only") if k in oos_streams}
    best_single=max(singles, key=lambda k: oos_sharpe[k]) if singles else None
    full=oos_streams["full(m+si+pead)"]
    print("\n"+"-"*78+"\nQ1 — does COMBINING beat the BEST SINGLE brick? (paired block-boot, holdout)\n"+"-"*78)
    if best_single:
        ci,p,point=boot_ci_lift(full,oos_streams[best_single],ppy,a.n_boot)
        print("  best single brick = %s (holdout Sharpe %+.2f)"%(best_single,oos_sharpe[best_single]))
        print("  full book Sharpe %+.2f | lift = %+.2f  CI [%+.2f,%+.2f]  P(lift>0)=%.0f%%"
              %(oos_sharpe["full(m+si+pead)"],point,ci[0],ci[1],100*p))
        if ci[0]>0:
            print("  >> COMBINING ADDS VALUE: lift CI clears zero. The book beats the best single brick")
            print("     net-of-cost OOS -- unlike the PEAD+SI cousin (which had P(lift>0)=26%).")
        elif p>=0.85:
            print("  >> LIKELY but not conclusive: P(lift>0)=%.0f%% but CI touches zero. Suggestive."%(100*p))
        else:
            print("  >> COMBINING DOES NOT CLEARLY BEAT THE BEST SINGLE BRICK (P=%.0f%%, CI spans 0)."%(100*p))
            print("     Same verdict as the cousin: the book works, but you might as well trade %s alone."%best_single)

    # ---- Q2: does PEAD add anything (full vs mom+si)? ----
    print("\n"+"-"*78+"\nQ2 — does PEAD add anything to the book? (full vs mom+si, paired)\n"+"-"*78)
    if "mom+si" in oos_streams:
        ci,p,point=boot_ci_lift(full,oos_streams["mom+si"],ppy,a.n_boot)
        print("  mom+si holdout Sharpe %+.2f | full %+.2f | PEAD marginal = %+.2f  CI [%+.2f,%+.2f]  P(>0)=%.0f%%"
              %(oos_sharpe["mom+si"],oos_sharpe["full(m+si+pead)"],point,ci[0],ci[1],100*p))
        if ci[0]>0:
            print("  >> PEAD ADDS: its marginal contribution clears zero.")
        elif p<0.6:
            print("  >> PEAD ADDS ~NOTHING here: momentum+SI do the work. (PEAD is a partial overlay on")
            print("     this universe -- few names have a fresh SUE each rebalance. Consistent with the")
            print("     cousin finding that PEAD+SI didn't stack net-of-cost.)")
        else:
            print("  >> PEAD marginal is ambiguous (P=%.0f%%)."%(100*p))

    # ---- Q3: regime dependence — per-calendar-year Sharpe of the full book (full sample) ----
    allr,alld=run_on(grid,1,1,1)
    by_year=defaultdict(list)
    for r,d in zip(allr,alld): by_year[d.year].append(r)
    print("\n"+"-"*78+"\nQ3 — regime dependence: per-year Sharpe of the full book (the inverted-decay tell)\n"+"-"*78)
    print("  %-6s %8s %8s %6s"%("year","Sharpe","annRet","n"))
    yrs=sorted(by_year)
    for y in yrs:
        rr=np.array(by_year[y])
        if len(rr)>=4:
            print("  %-6d %+8.2f %+7.1f%% %5d"%(y,sharpe_of(rr,ppy),100*rr.mean()*ppy,len(rr)))
        else:
            print("  %-6d   (n=%d too few)"%(y,len(rr)))
    yr_sh=[sharpe_of(np.array(by_year[y]),ppy) for y in yrs if len(by_year[y])>=4]
    if yr_sh:
        print("\n  per-year Sharpe range: %.2f to %.2f  (spread = regime sensitivity)"%(min(yr_sh),max(yr_sh)))
        print("  through-cycle (all years pooled) Sharpe = %+.2f  <- the honest sizing number,"%sharpe_of(allr,ppy))
        print("  NOT the recent-regime holdout Sharpe. Size on this; expect the recent ~2.0 to revert.")

    print("\n"+LINE+"\nHONEST SUMMARY\n"+LINE)
    print("  * Q1 (combining vs best single) and Q2 (PEAD marginal) are the build-decision tests.")
    print("  * Q3 quantifies the regime-dependence behind the inverted decay you saw across splits.")
    print("  * ALL Sharpes are survivor-tilted LEVELS (optimistic). Survivorship is unfixable here:")
    print("    the universe itself was defined from survivors, so delisted names were never present")
    print("    and the bias can't even be bounded without point-in-time-constituents data (WRDS).")
    print("  * The COMPARISONS (lift CIs, per-year spread) are the trustworthy outputs, not levels.")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
