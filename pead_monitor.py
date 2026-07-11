#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — PEAD BOOK MONITOR (data-gated, not calendar-gated)
================================================================================
PEAD status (carried): real cross-sectional signal, but as a tradeable BOOK it's
unproven and fragile -- edge concentrated in the recent ~12 months, negative in the
early half, n=24 too thin to trust. This monitors whether that picture changes as
more earnings data accumulates.

WHY NOT WEEKLY: PEAD's sample grows by EARNINGS EVENTS, not calendar weeks. Companies
report quarterly, so a weekly re-run shows ~the same n with 0-2 new points and the
"edge" jiggles on noise. Re-running a fragile n=24 statistic every week is exactly
the over-monitoring that invents signal from noise.

WHAT THIS DOES INSTEAD (the mathematically honest cadence):
  * CHEAP weekly check: count current fresh-SUE rebalances (n). Cost ~nothing.
  * Only TRIGGER the full analysis (half-split + bootstrap) when n has grown by a
    threshold (default +25%, e.g. 24 -> 30) since the last logged run. That's when
    the statistic can actually move -- roughly every 1-2 months given quarterly EPS.
  * Logs every check to pead_monitor_log.csv so you can see n and (when triggered)
    the early/recent Sharpe over time, watching whether the recent-half edge persists
    or was regime-specific.

THE KEY THING TO WATCH: the recent-half edge (+3.23 Sharpe on n=12) is the fragile
part. Either it PERSISTS as new months arrive (edge is real, was just thin) or it
DECAYS toward the early-half's negative number (it was a recent-regime artifact).
This monitor is built to surface that distinction honestly, not to manufacture
weekly excitement.

RULE 1: same verified machinery as pead_book_fullhistory.py (PIT SUE, 40d hold,
HAC, bootstrap, half-split). Cheap check is just a count -- no premature stats.
READ-ONLY except appending to the log CSV.

USAGE (run weekly via cron; it self-gates):
  python pead_monitor.py --root .                    # check; analyze only if n grew enough
  python pead_monitor.py --root . --force            # force full analysis now
  python pead_monitor.py --root . --growth-trigger 0.25   # re-run when n grows 25%
  python pead_monitor.py --root . --show-log         # print the history
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
LOG="pead_monitor_log.csv"

def build_pead_returns(root, hold, window, quantile, min_names, max_weight, cost_bps):
    prices_db=os.path.join(root,"prices.db"); earnp=find_db(root,"earnings.db")
    if not os.path.isfile(prices_db) or not earnp: return None,None
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
    alldays=sorted(set(d for lst in px.values() for d,_ in lst))
    month_last={}
    for d in alldays: month_last[(d.year,d.month)]=d
    grid=[month_last[k] for k in sorted(month_last)]
    def fwd(tk,d,h):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=idx.get(d)
        if i is None:
            for off in range(0,5):
                cc=d-datetime.timedelta(days=off)
                if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+h
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None
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
    sue_events=defaultdict(list)
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
        recent=[(ed,s) for ed,s in evs if 0<=(d-ed).days<=window]
        if not recent: return None
        recent.sort(); return recent[-1][1]
    prev_w={}; rets=[]; rdates=[]
    for d in grid:
        ranked=[(tk,sue_asof(tk,d)) for tk in sue_events if tk in pos_of]
        ranked=[(tk,s) for tk,s in ranked if s is not None]
        if len(ranked)<min_names: continue
        ranked.sort(key=lambda x:x[1])
        q=max(1,int(len(ranked)*quantile))
        low=ranked[:q]; high=ranked[-q:]
        names=[tk for tk,_ in low]+[tk for tk,_ in high]
        sgn=np.array([-1.0]*len(low)+[1.0]*len(high))
        w=np.clip(sgn/len(names),-max_weight,max_weight)
        wd={tk:wi for tk,wi in zip(names,w)}
        allk=set(wd)|set(prev_w); turn=sum(abs(wd.get(k,0)-prev_w.get(k,0)) for k in allk)
        r=0.0
        for tk,wi in wd.items():
            fr=fwd(tk,d,hold)
            if fr is not None: r+=wi*fr
        r-=turn*(cost_bps/10000.0)
        rets.append(r); rdates.append(d); prev_w=wd
    return np.array(rets), rdates

def sharpe(x,ppy):
    x=np.asarray(x,float); v=x.std(ddof=1)
    return (x.mean()/v*math.sqrt(ppy)) if v>0 else 0.0

def last_logged_n(root):
    p=os.path.join(root,LOG)
    if not os.path.isfile(p): return 0
    try:
        with open(p) as f:
            rows=list(csv.DictReader(f))
        analyzed=[r for r in rows if r.get("triggered")=="yes"]
        return int(analyzed[-1]["n"]) if analyzed else 0
    except Exception: return 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--pead-window",type=int,default=45)
    ap.add_argument("--quantile",type=float,default=0.2)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--max-weight",type=float,default=0.05)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--growth-trigger",type=float,default=0.25)
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--show-log",action="store_true")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    logpath=os.path.join(a.root,LOG)

    if a.show_log:
        if os.path.isfile(logpath):
            print(open(logpath).read())
        else: print("  no log yet at %s"%logpath)
        return

    print("\n"+LINE+"\nPEAD BOOK MONITOR (data-gated)\n"+LINE)
    today=datetime.date.today().isoformat()
    rets,rdates=build_pead_returns(a.root,a.hold,a.pead_window,a.quantile,a.min_names,a.max_weight,a.cost_bps)
    if rets is None:
        print("  [STOP] prices.db / earnings.db not found"); return
    n=len(rets)
    prev_n=last_logged_n(a.root)
    grew = prev_n==0 or (n-prev_n)/max(prev_n,1) >= a.growth_trigger
    trigger = a.force or grew

    print("  current fresh-SUE rebalances: n=%d  (last analyzed at n=%d)"%(n,prev_n))
    if not trigger:
        need=int(math.ceil(prev_n*(1+a.growth_trigger)))
        print("  n has not grown >=%.0f%% since last analysis (need n>=%d). NO re-run -- this is"%(100*a.growth_trigger,need))
        print("  correct: PEAD grows by earnings events, not weeks. Re-running now would just")
        print("  re-measure the same fragile statistic on noise. Check again next week.")
        # still log the cheap check
        _append_log(logpath,today,n,"no","","","")
        print("\n  logged check (n=%d, not analyzed). %s"%(n,LOG))
        return

    # full analysis
    ppy=365.25/float(a.hold); lag=max(1,int(math.ceil(a.hold/30.0)))
    full_sr=sharpe(rets,ppy)
    mid=n//2
    e1=sharpe(rets[:mid],ppy); e2=sharpe(rets[mid:],ppy)
    # bootstrap CI
    rng=np.random.default_rng(42); block=3
    def bres(N):
        out=[]
        while len(out)<N:
            s=rng.integers(0,N); out.extend([(s+j)%N for j in range(block)])
        return np.array(out[:N])
    boots=np.array([sharpe(rets[bres(n)],ppy) for _ in range(3000)])
    lo,hi=np.percentile(boots,[2.5,97.5])

    print("\n  >> ANALYSIS TRIGGERED (n grew enough, or --force)")
    print("  full-sample Sharpe=%+.2f  bootstrap 95%% CI=[%+.2f,%+.2f]  P(>0)=%.0f%%"%(full_sr,lo,hi,100*np.mean(boots>0)))
    print("  EARLY half Sharpe=%+.2f  |  RECENT half Sharpe=%+.2f"%(e1,e2))
    print("  maxDD=%.1f%%"%(100*maxdd(np.cumsum(rets))))

    # the watch: is the recent edge persisting or decaying?
    print("\n"+"-"*78+"\nWATCH: recent-half edge persistence\n"+"-"*78)
    if e1>0 and e2>0:
        print("  >> BOTH halves now positive — the edge is becoming STABLE across time. The")
        print("     recency-concentration is resolving as data accumulates. Upgrade confidence.")
    elif e2>e1 and e2>0:
        print("  >> Recent half (%+.2f) still stronger than early (%+.2f). Edge persists but"%(e2,e1))
        print("     remains recency-tilted. Keep monitoring — not yet stable.")
    else:
        print("  >> Recent edge has DECAYED (early %+.2f, recent %+.2f). The earlier +3.23 was"%(e1,e2))
        print("     likely a regime artifact, not a durable edge. Downgrade PEAD book confidence.")

    _append_log(logpath,today,n,"yes","%.2f"%full_sr,"%.2f"%e1,"%.2f"%e2)
    print("\n  logged analysis. Compare to prior rows: python pead_monitor.py --show-log")
    print("  Honest n=%d. In-sample, survivor-tilted. Signal is real; the BOOK remains the question."%n)

def _append_log(path,date,n,triggered,full_sr,early_sr,recent_sr):
    new = not os.path.isfile(path)
    with open(path,"a",newline="") as f:
        w=csv.writer(f)
        if new: w.writerow(["date","n","triggered","full_sharpe","early_sharpe","recent_sharpe"])
        w.writerow([date,n,triggered,full_sr,early_sr,recent_sr])

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
