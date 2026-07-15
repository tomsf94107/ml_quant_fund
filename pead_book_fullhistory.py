#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — PEAD BOOK, FULL-HISTORY LARGE SAMPLE (verify the n=13 Sharpe)
================================================================================
The two_brick_book PEAD Sharpe (+3.10) rested on n=13 -- because it rebalanced on
the 60 short-interest settlement dates, and only 13 of those had >=20 stocks with
a FRESH earnings surprise. That is the SI-overlap keyhole, not PEAD's real sample.

This measures PEAD on ITS OWN cadence across the FULL earnings history (2009-2026):
rebalance MONTHLY, and at each month hold the stocks whose most recent earnings
event is within the drift window, ranked by SUE. That yields MANY more rebalances
(~150+ months vs 13), giving an honest, well-powered read on whether the PEAD book
Sharpe holds up or collapses on a larger sample.

This is NOT inflation -- it is the opposite. n=13 was an artifact of piggybacking on
the SI grid. PEAD generates a signal every month there are recent earnings (i.e.
every month). Measuring it on its own monthly cadence is the correct, larger sample.

CONSTRUCTION (identical sizing to two_brick_book, just a denser/longer grid):
  * Monthly grid over the full prices+earnings overlap
  * Each month: stocks with an earnings event in (d-window, d], ranked by SUE
    (PIT-trailing denominator), long top / short bottom, z-score weighted,
    dollar-neutral, per-name capped
  * 40-day hold (the validated horizon), net of costs
  * Reports: Sharpe + HAC SE + BOOTSTRAP CI (so the larger-sample Sharpe comes
    with an honest confidence interval, unlike the bare n=13 number)
  * Also splits the history in HALF (early vs recent) to check stability -- a real
    edge persists across both halves; an artifact lives in one.

HONEST SCOPE: still in-sample, survivor-tilted universe (prices.db is survivor-
biased), simplified costs. The point is SAMPLE SIZE and STABILITY, not a live promise.

RULE 1: forward returns strictly after formation; SUE PIT-trailing; costs netted;
40d-overlap -> HAC SE + block bootstrap (no naive-SE inflation, the -20 lesson);
half-sample stability check. READ-ONLY.

USAGE:
  python pead_book_fullhistory.py --root .
  python pead_book_fullhistory.py --root . --hold 40 --rebal monthly
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

def stats(rets, ppy, lag):
    rets=np.asarray(rets,float); n=len(rets)
    if n<3: return None
    m=rets.mean(); v=rets.std(ddof=1)
    sr=(m/v*math.sqrt(ppy)) if v>0 else 0
    se_mean=nw_se(rets,lag); se_sr=(se_mean/v)*math.sqrt(ppy) if (se_mean and v>0) else 0
    curve=np.cumsum(rets)
    return dict(n=n,sharpe=sr,se=se_sr,ann_ret=m*ppy,ann_vol=v*math.sqrt(ppy),
                hit=100*np.mean(rets>0),mdd=maxdd(curve))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--pead-window",type=int,default=45)
    ap.add_argument("--quantile",type=float,default=0.2)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--max-weight",type=float,default=0.05)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--n-boot",type=int,default=5000)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=os.path.join(a.root,"prices.db"); earnp=find_db(a.root,"earnings.db")
    print("\n"+LINE+"\nPEAD BOOK — FULL-HISTORY LARGE SAMPLE (verify the n=13 Sharpe)\n"+LINE)
    for lbl,p in (("prices.db",prices_db),("earnings.db",earnp)):
        if not p or not os.path.isfile(p): print("[STOP] %s not found"%lbl); return

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
    alldays=sorted(set(d for lst in px.values() for d,_ in lst))
    # monthly grid = last trading day of each month
    month_last={}
    for d in alldays: month_last[(d.year,d.month)]=d
    grid=[month_last[k] for k in sorted(month_last)]
    print("  price history: %d tickers, %s to %s, %d monthly rebalance points"
          %(len(px),alldays[0],alldays[-1],len(grid)))

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

    # earnings -> SUE (PIT trailing)
    ce=ro(earnp)
    try:
        cl=cols_of(ce,"earnings_surprises")
        comp="eps_actual" in cl and "eps_estimate" in cl
        sel="ticker,report_date"+(",eps_actual,eps_estimate" if comp else ",eps_surprise_pct")
        ev=Q(ce,"SELECT "+sel+" FROM earnings_surprises WHERE report_date IS NOT NULL")
        _n_ev = Q(ce, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='earnings_events'")[0][0]
        _n_ev = Q(ce, "SELECT COUNT(*) FROM earnings_events WHERE eps_surprise IS NOT NULL")[0][0] if _n_ev else 0
        if _n_ev > 1000:
            comp = False
            ev = Q(ce, "SELECT ticker, announce_date, eps_surprise FROM earnings_events "
                       "WHERE eps_surprise IS NOT NULL AND announce_date IS NOT NULL")
            print("  PEAD source: earnings_events.announce_date (%d rows) [LEAK-FIXED]" % len(ev))
        else:
            print("  PEAD source: earnings_surprises (fallback -- KNOWN LEAKED DATES)")
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
        recent=[(ed,s) for ed,s in evs if 0<=(d-ed).days<=a.pead_window]
        if not recent: return None
        recent.sort(); return recent[-1][1]

    # build monthly PEAD long-short book on full history
    prev_w={}; rets=[]; rdates=[]; turns=[]
    for d in grid:
        ranked=[]
        for tk in sue_events:
            if tk not in pos_of: continue
            s=sue_asof(tk,d)
            if s is not None: ranked.append((tk,s))
        if len(ranked)<a.min_names:
            continue
        ranked.sort(key=lambda x:x[1])
        q=max(1,int(len(ranked)*a.quantile))
        low=ranked[:q]; high=ranked[-q:]   # low SUE = short, high SUE = long
        names=[tk for tk,_ in low]+[tk for tk,_ in high]
        sgn=np.array([-1.0]*len(low)+[1.0]*len(high))  # short low, long high
        w=sgn/len(names)
        w=np.clip(w,-a.max_weight,a.max_weight)
        wd={tk:wi for tk,wi in zip(names,w)}
        allk=set(wd)|set(prev_w)
        turn=sum(abs(wd.get(k,0)-prev_w.get(k,0)) for k in allk)
        r=0.0
        for tk,wi in wd.items():
            fr=fwd(tk,d,a.hold)
            if fr is not None: r+=wi*fr
        r-=turn*(a.cost_bps/10000.0)
        rets.append(r); rdates.append(d); turns.append(turn); prev_w=wd
    rets=np.array(rets); n=len(rets)
    print("  PEAD real rebalances (>=%d fresh-SUE names): n=%d  (vs n=13 on the SI-grid keyhole)"%(a.min_names,n))
    if n<12:
        print("  [STOP] still too few (%d). Earnings coverage thinner than expected."%n); return

    ppy=365.25/float(a.hold); lag=max(1,int(math.ceil(a.hold/30.0)))  # monthly grid, 40d hold ~ lag 2
    st=stats(rets,ppy,lag)

    print("\n"+"-"*78+"\nFULL-SAMPLE PEAD BOOK (net %.0f bps, 40d hold, n=%d)\n"%(a.cost_bps,n)+"-"*78)
    print("  Sharpe=%+.2f  HAC SE=%.2f  ~95%% CI=[%+.2f, %+.2f]"%(st["sharpe"],st["se"],
          st["sharpe"]-1.96*st["se"],st["sharpe"]+1.96*st["se"]))
    print("  annRet=%+.1f%%  annVol=%.1f%%  maxDD=%.1f%%  hit=%.0f%%  avgTurn=%.2f"
          %(100*st["ann_ret"],100*st["ann_vol"],100*st["mdd"],st["hit"],np.mean(turns)))

    # bootstrap CI on Sharpe
    rng=np.random.default_rng(42); block=3
    def bres(N):
        out=[]
        while len(out)<N:
            s=rng.integers(0,N); out.extend([(s+j)%N for j in range(block)])
        return np.array(out[:N])
    def sa(x):
        v=x.std(ddof=1); return (x.mean()/v*math.sqrt(ppy)) if v>0 else 0
    boots=np.array([sa(rets[bres(n)]) for _ in range(a.n_boot)])
    lo,hi=np.percentile(boots,[2.5,97.5])
    print("  bootstrap Sharpe: mean=%+.2f  95%% CI=[%+.2f, %+.2f]  P(Sharpe>0)=%.0f%%"
          %(boots.mean(),lo,hi,100*np.mean(boots>0)))

    # half-sample stability
    print("\n"+"-"*78+"\nSTABILITY: does the edge persist across BOTH halves of history?\n"+"-"*78)
    mid=n//2
    e1=stats(rets[:mid],ppy,lag); e2=stats(rets[mid:],ppy,lag)
    d1=rdates[0]; dmid=rdates[mid]; d2=rdates[-1]
    if e1 and e2:
        print("  EARLY half (%s..%s, n=%d): Sharpe=%+.2f  annRet=%+.1f%%  hit=%.0f%%"
              %(d1,dmid,e1["n"],e1["sharpe"],100*e1["ann_ret"],e1["hit"]))
        print("  RECENT half (%s..%s, n=%d): Sharpe=%+.2f  annRet=%+.1f%%  hit=%.0f%%"
              %(dmid,d2,e2["n"],e2["sharpe"],100*e2["ann_ret"],e2["hit"]))
        both_pos = e1["sharpe"]>0 and e2["sharpe"]>0
        print("  >> %s"%("BOTH halves positive — edge is stable across time." if both_pos
                else "Edge is NOT stable across halves — concentrated in one period (fragile)."))

    print("\n"+LINE+"\nVERDICT — does the PEAD book Sharpe survive a larger sample?\n"+LINE)
    if lo>0.5 and n>=40:
        print("  >> HOLDS UP: on n=%d rebalances the Sharpe CI [%+.2f,%+.2f] stays well above 0."%(n,lo,hi))
        print("     The n=13 number was a keyhole; on PEAD's full cadence the book Sharpe is real")
        print("     (though still in-sample/survivor-tilted — needs OOS + real costs before live).")
    elif lo>0 and n>=40:
        print("  >> POSITIVE BUT MODEST: Sharpe CI [%+.2f,%+.2f] stays >0 on n=%d, but lower than the"%(lo,hi,n))
        print("     n=13 headline (3.10). The keyhole flattered it. Real but smaller edge.")
    elif n<40:
        print("  >> STILL THIN (n=%d): even on full history, fresh-SUE rebalances are limited. The"%n)
        print("     Sharpe CI [%+.2f,%+.2f] is the honest read; treat with caution."%(lo,hi))
    else:
        print("  >> DOES NOT HOLD: on the larger sample the Sharpe CI [%+.2f,%+.2f] is weak/spans 0."%(lo,hi))
        print("     The n=13 +3.10 was a small-sample artifact. PEAD book edge is not established.")
    print("\n  Honest n=%d, %dd hold, HAC lag=%d, bootstrap %d resamples. In-sample, survivor-tilted."%(n,a.hold,lag,a.n_boot))

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
