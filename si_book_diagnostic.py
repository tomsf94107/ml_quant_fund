#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — SHORT-INTEREST BOOK DIAGNOSTIC (fix the -25.7% drawdown?)
================================================================================
The short-interest SIGNAL is real (per-date IC -0.054, 8.3 sigma, confirmed). But
as a long-short BOOK it posted Sharpe +0.82 with a -25.7% max drawdown -- a poor
risk profile. Hypothesis: the drawdown comes from the SHORT leg (shorting high-
short-interest names = standing in front of squeezes) and from positions persisting
(short interest is sticky, so the book holds losers).

This tests four variants of the SAME validated signal to see which is tradeable:
  1. LONG-SHORT (baseline)      -- long low-DTC, short high-DTC  [the -25.7% book]
  2. LONG-ONLY                  -- long low-DTC only (drop the squeeze-prone short)
  3. LONG-SHORT + drawdown stop -- flatten after the book draws down > stop%
  4. LONG-ONLY  + drawdown stop -- combine both fixes

All four: same signal (days_to_cover), per-name cap, 40d hold, net of costs, on the
full 60-date SI history. Reports Sharpe, maxDD, hit, Calmar (return/|maxDD|) for each.

THE QUESTION: does long-only (no short leg) turn the -25.7% drawdown into something
survivable while keeping a positive Sharpe? If yes, the brick is tradeable long-only.
If even long-only drawdowns hard or loses the edge, the signal is real but the BOOK
isn't viable -- a valid (if disappointing) finding: don't trade it, keep hunting.

NOTE on long-only + a negative signal: high short interest predicts LOW returns, so
the tradeable long-only expression is "AVOID/underweight high-short names" = hold the
LOW-short-interest names long. That's what variant 2/4 do (long the bottom-DTC quintile).
A long-only book also carries market beta -- so it's measured both raw and
beta-hedged (vs the equal-weight universe) to isolate the signal's contribution.

RULE 1: forward returns strictly after formation; costs netted; drawdown stop uses
only PAST cumulative P&L (no look-ahead); beta-hedge uses same-period universe return.
READ-ONLY.

USAGE:
  python si_book_diagnostic.py --root .
  python si_book_diagnostic.py --root . --dd-stop 0.10 --max-weight 0.05
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

def book_metrics(rets, ppy, lag):
    rets=np.asarray(rets,float); n=len(rets)
    if n<3: return None
    m=rets.mean(); v=rets.std(ddof=1)
    sr=(m/v*math.sqrt(ppy)) if v>0 else 0
    se_mean=nw_se(rets,lag); se=(se_mean/v)*math.sqrt(ppy) if (se_mean and v>0) else 0
    curve=np.cumsum(rets); mdd=maxdd(curve)
    ann=m*ppy; calmar=ann/abs(mdd) if mdd<0 else float('inf')
    return dict(n=n,sharpe=sr,se=se,ann_ret=ann,ann_vol=v*math.sqrt(ppy),
                hit=100*np.mean(rets>0),mdd=mdd,calmar=calmar)

def apply_dd_stop(per_period_rets, stop):
    """Flatten (zero return) for the next period after cumulative DD from peak exceeds
    stop. Uses only PAST P&L. Re-enters once a new sequence starts recovering (simple:
    re-enter next period after a flat). Returns the stopped return stream."""
    out=[]; cum=0.0; peak=0.0; flat=False
    for r in per_period_rets:
        if flat:
            out.append(0.0)        # sat out this period
            flat=False             # try re-entering next period
            continue
        out.append(r); cum+=r; peak=max(peak,cum)
        if cum-peak <= -stop:      # drawdown breached -> flatten NEXT period
            flat=True
    return np.array(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--quantile",type=float,default=0.2)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--max-weight",type=float,default=0.05)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--dd-stop",type=float,default=0.10,help="flatten after cumulative DD > this (e.g. 0.10 = 10%)")
    ap.add_argument("--clip-dtc",type=float,default=50.0)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=os.path.join(a.root,"prices.db"); si_db=os.path.join(a.root,"short_interest.db")
    print("\n"+LINE+"\nSHORT-INTEREST BOOK DIAGNOSTIC — can the -25.7%% drawdown be fixed?\n"+LINE)
    for lbl,p in (("prices.db",prices_db),("short_interest.db",si_db)):
        if not p or not os.path.isfile(p): print("[STOP] %s not found"%lbl); return

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
    def fwd(tk,d):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,6):
            cc=d+datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+a.hold
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
        if a.clip_dtc and fv>a.clip_dtc: continue
        si_by_date[do][tk.upper()]=fv
    grid=sorted(si_by_date)

    ppy=365.25/float(a.hold); lag=max(1,int(math.ceil(a.hold/15.0)))

    # build the four raw streams + universe (for beta hedge) on the SI grid
    ls=[]; lo=[]; uni=[]; prevL={}; prevLS={}
    for d in grid:
        snap=[(tk,v) for tk,v in si_by_date[d].items() if tk in pos_of]
        if len(snap)<a.min_names:
            continue
        snap.sort(key=lambda x:x[1])  # ascending DTC
        q=max(1,int(len(snap)*a.quantile))
        low=[tk for tk,_ in snap[:q]]   # low DTC -> predicted HIGH return -> LONG
        high=[tk for tk,_ in snap[-q:]] # high DTC -> predicted LOW return -> SHORT (in L/S)
        # universe equal-weight return (for beta hedge + market context)
        urs=[fwd(tk,d) for tk,_ in snap]; urs=[x for x in urs if x is not None]
        uni_ret=np.mean(urs) if urs else 0.0
        # long-short: +low / -high, dollar-neutral, per-name capped
        wls={}
        for tk in low: wls[tk]=min(a.max_weight,1.0/(2*len(low)))
        for tk in high: wls[tk]=-min(a.max_weight,1.0/(2*len(high)))
        # long-only: +low only, capped, normalized to ~1 gross
        wlo={tk:min(a.max_weight,1.0/len(low)) for tk in low}
        def book_ret(wd,prev):
            allk=set(wd)|set(prev); turn=sum(abs(wd.get(k,0)-prev.get(k,0)) for k in allk)
            r=0.0
            for tk,wi in wd.items():
                fr=fwd(tk,d)
                if fr is not None: r+=wi*fr
            return r-turn*(a.cost_bps/10000.0)
        ls.append(book_ret(wls,prevLS)); lo.append(book_ret(wlo,prevL)); uni.append(uni_ret)
        prevLS=wls; prevL=wlo
    ls=np.array(ls); lo=np.array(lo); uni=np.array(uni); n=len(ls)
    print("  rebalances: n=%d | per-name cap %.0f%% | dd-stop %.0f%% | %.0f bps"%(n,100*a.max_weight,100*a.dd_stop,a.cost_bps))
    if n<10: print("  [STOP] too few rebalances (%d)"%n); return

    # beta-hedge the long-only book: regress lo on uni, take residual
    if uni.std()>0:
        beta=np.cov(lo,uni,ddof=1)[0,1]/np.var(uni,ddof=1)
        lo_hedged=lo-beta*uni
    else:
        beta=0; lo_hedged=lo

    # variants
    ls_stop=apply_dd_stop(ls,a.dd_stop)
    lo_stop=apply_dd_stop(lo,a.dd_stop)

    rows=[
        ("1. LONG-SHORT (baseline)",   book_metrics(ls,ppy,lag)),
        ("2. LONG-ONLY (raw)",         book_metrics(lo,ppy,lag)),
        ("   LONG-ONLY (beta-hedged)", book_metrics(lo_hedged,ppy,lag)),
        ("3. LONG-SHORT + DD stop",    book_metrics(ls_stop,ppy,lag)),
        ("4. LONG-ONLY + DD stop",     book_metrics(lo_stop,ppy,lag)),
    ]
    print("\n"+"-"*78+"\nFOUR VARIANTS of the validated signal (which is tradeable?)\n"+"-"*78)
    print("  %-28s %7s %7s %8s %6s %7s"%("variant","Sharpe","maxDD","annRet","hit","Calmar"))
    for name,m in rows:
        if not m: print("  %-28s [n/a]"%name); continue
        print("  %-28s %+7.2f %6.1f%% %+7.1f%% %5.0f%% %7.2f"
              %(name,m["sharpe"],100*m["mdd"],100*m["ann_ret"],m["hit"],m["calmar"]))
    print("  (Calmar = annRet/|maxDD|; higher = better drawdown-adjusted return. beta of long-only vs universe = %.2f)"%beta)

    base=rows[0][1]; loraw=rows[1][1]; lohedge=rows[2][1]; lostop=rows[4][1]
    print("\n"+LINE+"\nVERDICT — is the short-interest signal tradeable as a book?\n"+LINE)
    def dd(m): return 100*m["mdd"] if m else float('nan')
    improved = loraw and base and (loraw["mdd"] > base["mdd"]) and loraw["sharpe"]>0   # mdd less negative
    big_improve = lostop and base and lostop["mdd"]>base["mdd"]*0.5 and lostop["sharpe"]>0
    if loraw and loraw["sharpe"]>0 and loraw["mdd"]>-0.12:
        print("  >> LONG-ONLY FIXES IT: dropping the short leg cuts the drawdown from %.1f%% to %.1f%%"%(dd(base),dd(loraw)))
        print("     while keeping Sharpe %+.2f. The squeeze-prone short leg WAS the problem. The brick"%loraw["sharpe"])
        print("     is tradeable LONG-ONLY (as 'hold low-short-interest names'). Confirm OOS + real costs.")
        if lohedge: print("     Beta-hedged long-only Sharpe %+.2f -> signal contributes beyond market beta."%lohedge["sharpe"])
    elif improved:
        print("  >> LONG-ONLY HELPS but doesn't fully fix: drawdown %.1f%% -> %.1f%%, Sharpe %+.2f."%(dd(base),dd(loraw),loraw["sharpe"]))
        print("     Better, still not clean. Adding the DD stop: drawdown %.1f%%, Sharpe %+.2f."%(dd(lostop) if lostop else float('nan'), lostop["sharpe"] if lostop else float('nan')))
        print("     Marginal tradeable book; size small if at all.")
    else:
        print("  >> BOOK NOT VIABLE: even long-only the drawdown/Sharpe don't become tradeable")
        print("     (long-only Sharpe %+.2f, maxDD %.1f%%). The SIGNAL is real (IC 8.3 sigma) but the"%(loraw["sharpe"] if loraw else float('nan'),dd(loraw)))
        print("     tradeable BOOK isn't. Valid finding: don't trade it standalone; use as a feature/")
        print("     filter inside the model, or keep hunting brick #3. Signal != book.")
    print("\n  Honest n=%d, %dd hold, %.0f bps, HAC lag=%d. In-sample, survivor-tilted, simplified costs."%(n,a.hold,a.cost_bps,lag))
    print("  Beta-hedge + DD-stop are research diagnostics, not a tuned production strategy.")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
