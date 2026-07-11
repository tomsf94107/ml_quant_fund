#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — COMBINED 40d BOOK MONITOR (self-gating OOS re-test)
================================================================================
The combined momentum+PEAD+short-interest 40d book is PROMISING but UNCONFIRMED:
in-sample Sharpe 1.07 (stable both halves), out-of-sample +1.31 but the OOS CI
[-0.22,+3.27] spanned zero on only 19 holdout rebalances. The binding constraint is
the number of short-interest settlement dates (~60 in 5yr of bi-monthly FINRA data).
The deep-history fix needs WRDS (unavailable). So the path is: WAIT for FINRA to
publish more dates, and re-run the OOS test as they accumulate, until the holdout is
large enough to confirm or kill the book.

THIS MONITOR does that automatically, on the same self-gating pattern as pead_monitor.py:
  * CHEAP weekly check: count short-interest settlement dates currently in the DB.
  * Only TRIGGER the full OOS test when enough NEW dates have accumulated since the
    last analysis (default +6 dates ~= one quarter of bi-monthly reporting). Re-running
    on the same dates would just re-measure the same inconclusive CI.
  * Logs IS/OOS Sharpe + OOS CI lower bound each analysis, so you watch the OOS CI
    lower bound climb toward zero-crossing over the coming months/years.
  * LOUDLY flags the actionable moment: when the OOS CI finally clears zero
    (CONFIRMED) or the OOS Sharpe decays to <=0 (KILLED).

>>> CRITICAL DEPENDENCY: this only works if short_interest.db ACCUMULATES dates. <<<
FINRA's free API serves a rolling 5 years -- new dates appear, old ones leave the API.
Your LOCAL db keeps whatever you pulled, so it grows past 5yr ONLY IF a refresh runs
periodically and upserts (keeps old + adds new). Wire si_refresh.py to cron alongside
this monitor. If n is not growing between checks, this monitor will say so -- that's
the signal your refresh isn't running, not that the data stopped.

The strategy has NO FITTED PARAMETERS (equal-weight blend, per-date z-score, fixed
signal defs), so the chronological IS/OOS split is clean -- nothing is tuned to leak.

RULE 1: identical audited book/OOS logic as combined_40d_oos.py; per-date z-score;
forward returns strictly after formation; momentum PIT; HAC + block bootstrap;
survivorship flagged. READ-ONLY except appending to the log CSV.

USAGE (run weekly via cron; it self-gates):
  python combined_40d_monitor.py --root .                 # check; analyze only if grown enough
  python combined_40d_monitor.py --root . --force         # force full OOS now
  python combined_40d_monitor.py --root . --growth-dates 6  # re-run when +6 new dates
  python combined_40d_monitor.py --root . --show-log      # print history
================================================================================
"""
import argparse, os, sqlite3, math, datetime, csv
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
def metrics(rets, ppy, lag):
    rets=np.asarray(rets,float); n=len(rets)
    if n<3: return None
    m=rets.mean(); v=rets.std(ddof=1)
    sr=(m/v*math.sqrt(ppy)) if v>0 else 0
    curve=np.cumsum(rets); mdd=maxdd(curve)
    return dict(n=n,sharpe=sr,ann=m*ppy,mdd=mdd)
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
LINE="="*78
LOG="combined_40d_monitor_log.csv"

def load_inputs(root, hold, mom_lb, mom_skip, pead_win):
    prices_db=os.path.join(root,"prices.db"); si_db=os.path.join(root,"short_interest.db")
    earnp=find_db(root,"earnings.db")
    if not os.path.isfile(prices_db) or not os.path.isfile(si_db): return None
    have_pead = earnp and os.path.isfile(earnp)
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
    return px,pos_of,si_by_date,sue_events,have_pead

def make_runner(px,pos_of,si_by_date,sue_events,have_pead,hold,mom_lb,mom_skip,pead_win,
                quantile,min_names,max_weight,cost_bps):
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
        end=i-mom_skip; start=i-mom_lb
        if start<0 or end<=start: return None
        p0=lst[start][1]; p1=lst[end][1]
        return (p1/p0-1.0) if p0>0 else None
    def sue_asof(tk,d):
        evs=sue_events.get(tk)
        if not evs: return None
        recent=[(ed,s) for ed,s in evs if 0<=(d-ed).days<=pead_win]
        if not recent: return None
        recent.sort(); return recent[-1][1]
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
                r=fwd(tk,d,hold)
                if r is None: continue
                names.append(tk); mom.append(m); si.append(dv)
                pead.append(sue_asof(tk,d) if have_pead else None)
            if len(names)<min_names: continue
            score=s_combined(np.array(mom),np.array(si),pead)
            order=np.argsort(score); q=max(1,int(len(names)*quantile))
            wd={}
            for j in order[-q:]: wd[names[j]]=min(max_weight,1.0/(2*q))
            for j in order[:q]: wd[names[j]]=-min(max_weight,1.0/(2*q))
            allk=set(wd)|set(prev_w); turn=sum(abs(wd.get(k,0)-prev_w.get(k,0)) for k in allk)
            r=0.0
            for tk,wi in wd.items():
                fr=fwd(tk,d,hold)
                if fr is not None: r+=wi*fr
            r-=turn*(cost_bps/10000.0)
            rets.append(r); rdates.append(d); prev_w=wd
        return np.array(rets), rdates
    return run_on

def last_logged(root):
    p=os.path.join(root,LOG)
    if not os.path.isfile(p): return 0, None
    try:
        with open(p) as f: rows=list(csv.DictReader(f))
        analyzed=[r for r in rows if r.get("triggered")=="yes"]
        if not analyzed: return 0, None
        return int(analyzed[-1]["n_dates"]), analyzed[-1].get("verdict")
    except Exception: return 0, None

def _append(path,row):
    new=not os.path.isfile(path)
    with open(path,"a",newline="") as f:
        w=csv.writer(f)
        if new: w.writerow(["date","n_dates","triggered","is_n","oos_n","is_sharpe","oos_sharpe","oos_ci_lo","oos_ci_hi","verdict"])
        w.writerow(row)

def classify(oos_sharpe, oos_ci_lo):
    if oos_ci_lo>0 and oos_sharpe>0.5: return "CONFIRMED"
    if oos_ci_lo>0: return "survives_weak"
    if oos_sharpe>0: return "inconclusive"
    return "KILLED"

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
    ap.add_argument("--growth-dates",type=int,default=6)
    ap.add_argument("--n-boot",type=int,default=5000)
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--show-log",action="store_true")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    logpath=os.path.join(a.root,LOG)

    if a.show_log:
        print(open(logpath).read() if os.path.isfile(logpath) else "  no log yet at %s"%logpath); return

    print("\n"+LINE+"\nCOMBINED 40d BOOK MONITOR (self-gating OOS re-test)\n"+LINE)
    today=datetime.date.today().isoformat()
    inp=load_inputs(a.root,a.hold,a.mom_lookback,a.mom_skip,a.pead_window)
    if inp is None: print("  [STOP] prices.db / short_interest.db not found"); return
    px,pos_of,si_by_date,sue_events,have_pead=inp
    n_dates=len(si_by_date)
    prev_n, prev_verdict = last_logged(a.root)
    grew = prev_n==0 or (n_dates-prev_n)>=a.growth_dates
    trigger = a.force or grew

    print("  short-interest settlement dates in DB: %d  (last analyzed at %d)"%(n_dates,prev_n))
    if not trigger:
        need=prev_n+a.growth_dates
        print("  only +%d new dates since last analysis (need +%d -> n>=%d). NO re-run."%(n_dates-prev_n,a.growth_dates,need))
        if n_dates<=prev_n:
            print("  >> n is NOT growing. Check that si_refresh.py is running on cron and ACCUMULATING")
            print("     dates (keeping old + adding new). Without accumulation this never resolves.")
        else:
            print("  Re-running on near-identical dates would just re-measure the same CI. Check next week.")
        _append(logpath,[today,n_dates,"no","","","","","","",""])
        return

    run_on=make_runner(px,pos_of,si_by_date,sue_events,have_pead,a.hold,a.mom_lookback,a.mom_skip,
                       a.pead_window,a.quantile,a.min_names,a.max_weight,a.cost_bps)
    grid=sorted(si_by_date)
    ppy=365.25/float(a.hold); lag=max(1,int(math.ceil(a.hold/15.0)))
    split_idx=int(len(grid)*a.split)
    is_dates=grid[:split_idx]; oos_dates=grid[split_idx:]
    is_ret,_=run_on(is_dates); oos_ret,_=run_on(oos_dates)
    nis=len(is_ret); noos=len(oos_ret)
    if noos<8:
        print("  >> holdout only %d rebalances after filtering -- still too thin. Logged, will re-check."%noos)
        _append(logpath,[today,n_dates,"yes",nis,noos,"","","","","too_thin"])
        return

    mis=metrics(is_ret,ppy,lag); moos=metrics(oos_ret,ppy,lag)
    oos_ci,oos_p=boot_ci(oos_ret,ppy,a.n_boot)
    verdict=classify(moos["sharpe"],oos_ci[0])

    print("\n  >> ANALYSIS TRIGGERED (n grew by %d, or --force)"%(n_dates-prev_n))
    print("  IN-SAMPLE  n=%d  Sharpe=%+.2f"%(nis,mis["sharpe"]))
    print("  HOLDOUT    n=%d  Sharpe=%+.2f  CI=[%+.2f,%+.2f]  P(>0)=%.0f%%"%(noos,moos["sharpe"],oos_ci[0],oos_ci[1],100*oos_p))
    print("  verdict: %s"%verdict)

    print("\n"+"-"*78+"\nACTIONABLE STATUS\n"+"-"*78)
    if verdict=="CONFIRMED" and prev_verdict!="CONFIRMED":
        print("  >> *** NEWLY CONFIRMED *** holdout CI cleared zero (lo=%+.2f) with Sharpe %+.2f."%(oos_ci[0],moos["sharpe"]))
        print("     The combined 40d book now SURVIVES out-of-sample on accumulated data. This is the")
        print("     moment to seriously consider building -- remaining gate is survivorship-free")
        print("     confirmation (still survivor-tilted) + capacity/cost realism.")
    elif verdict=="KILLED":
        print("  >> *** KILLED *** holdout Sharpe decayed to %+.2f. The in-sample edge did NOT survive as"%moos["sharpe"])
        print("     data accumulated. The earlier result was regime-luck. Do not build. Bank the")
        print("     decorrelation finding as knowledge; stop monitoring this book.")
    elif verdict in ("survives_weak","CONFIRMED"):
        print("  >> holdout CI lower bound now positive (%+.2f) but %s. Edge strengthening as data"%(oos_ci[0],"modest" if verdict=="survives_weak" else "confirmed"))
        print("     accumulates. Keep monitoring; close to a decision.")
    else:
        print("  >> STILL INCONCLUSIVE: OOS Sharpe %+.2f, CI [%+.2f,%+.2f] still spans zero (n=%d)."%(moos["sharpe"],oos_ci[0],oos_ci[1],noos))
        print("     Watch the CI lower bound (%+.2f) climb toward 0 in the log as more dates land."%oos_ci[0])

    _append(logpath,[today,n_dates,"yes",nis,noos,"%.2f"%mis["sharpe"],"%.2f"%moos["sharpe"],"%.2f"%oos_ci[0],"%.2f"%oos_ci[1],verdict])
    print("\n  logged. History: python combined_40d_monitor.py --show-log")
    print("  Honest: IS n=%d, OOS n=%d. No fitted params (clean split). In-sample/survivor-tilted level."%(nis,noos))

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
