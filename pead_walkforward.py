#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — PEAD WALK-FORWARD / TIME-STABILITY TEST
================================================================================
The pooled 40-day PEAD (IC +0.125, t=3.62, 2251 events) is real. But pooled over
2009-2026 it could hide DECAY — strong early, dead recently — which would make it
untradeable NOW despite a great average. The research says PEAD weakened in liquid
names post-2006. This test answers: IS THE EDGE STILL ALIVE?

WHAT IT DOES (all offline, reads cached prices.db — NO network):
  1. Slices the 40-day PEAD by CALENDAR YEAR of the announcement:
       per year -> IC, quintile L/S net spread, t-stat, event count
     so you see the trajectory: stable? decaying? dead since 2020?
  2. Splits first-half vs second-half of the sample to quantify decay directly.
  3. Computes a MARKET-RELATIVE spread: subtracts each event's same-window SPY
     (or a proxy: equal-weight universe) return, to strip the bull-market beta
     that flatters the raw long-leg. This is closer to the tradeable ALPHA.
       (If no SPY in prices.db, uses the cross-sectional mean as the market proxy,
        which is what a rank-IC already neutralizes — reported for completeness.)
  4. Reports a decile (not just quintile) monotonicity check for the pooled signal:
     a real PEAD should be ~monotonic across surprise deciles, not driven by tails.

RULE 1: signal known at announcement; entry day +2; window strictly after. The
market-relative adjustment uses only same-window market return (no look-ahead).
Per-year t-stats are small-sample — read the TRAJECTORY, not any single year.

READ-ONLY. mode=ro&immutable=1. No network.

USAGE:
  python pead_walkforward.py --root .
  python pead_walkforward.py --root . --hold 40 --cost-bps 10
  python pead_walkforward.py --root . --market SPY   (name the market ticker if present)
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
    ap.add_argument("--signal",default="eps_surprise_pct")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--market",default=None,help="market ticker for beta-strip (e.g. SPY); auto if present")
    ap.add_argument("--min-year-events",type=int,default=20)
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    banner("ML QUANT FUND — PEAD WALK-FORWARD / TIME-STABILITY TEST")
    print("Is the %d-day PEAD edge STILL ALIVE, or front-loaded/decayed? (offline)"%a.hold)
    print("Root:",os.path.abspath(a.root),"| prices.db:",prices_db,"| numpy:",HAVE_NUMPY)
    if not require(HAVE_NUMPY,"numpy required"): return
    if not require(os.path.isfile(prices_db),"prices.db not found — run fetch first"): return
    earnp=find_db(a.root,"earnings.db")
    if not require(earnp,"earnings.db not found"): return

    # load prices
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
    pxidx={tk:{d:i for i,(d,_) in enumerate(lst)} for tk,lst in px.items()}
    print("  prices loaded for %d tickers"%len(px))

    # market proxy
    mkt=a.market
    if mkt is None:
        for cand in ("SPY","IVV","VOO","^GSPC"):
            if cand in px: mkt=cand; break
    if mkt and mkt in px:
        print("  market proxy for beta-strip: %s (%d days)"%(mkt,len(px[mkt])))
    else:
        print("  no market ticker in prices.db — will use cross-sectional mean as market proxy")
        mkt=None

    ce=ro(earnp)
    try:
        ev=Q(ce,"SELECT ticker,report_date,"+a.signal+" FROM earnings_surprises "
               "WHERE report_date IS NOT NULL AND "+a.signal+" IS NOT NULL")
    finally:
        ce.close()
    events=[(tk,nd(rd),sig) for tk,rd,sig in ev if nd(rd) is not None]

    def fwd(tk,do,entry_off,hold):
        lst=px.get(tk); idx=pxidx.get(tk)
        if not lst: return None,None,None
        pos=None
        for off in range(0,6):
            c=do+datetime.timedelta(days=off)
            if c in idx: pos=idx[c]; break
        if pos is None: return None,None,None
        e=pos+entry_off; x=pos+entry_off+hold
        if x>=len(lst): return None,None,None
        pe=lst[e][1]; pxx=lst[x][1]
        if pe<=0: return None,None,None
        r=pxx/pe-1.0
        # market return over same calendar window
        mr=None
        if mkt:
            ml=px.get(mkt); mi=pxidx.get(mkt)
            if ml and mi:
                de=lst[e][0]; dx=lst[x][0]
                # nearest market dates
                pe_m=None; px_m=None
                for off in range(0,6):
                    c=de+datetime.timedelta(days=off)
                    if c in mi: pe_m=ml[mi[c]][1]; break
                for off in range(0,6):
                    c=dx+datetime.timedelta(days=off)
                    if c in mi: px_m=ml[mi[c]][1]; break
                if pe_m and px_m and pe_m>0: mr=px_m/pe_m-1.0
        return r, mr, lst[e][0]  # raw return, market return, entry date

    # assemble records with year + market-relative return
    recs=[]  # (year, sig, raw_ret, rel_ret)
    cross_by_window=defaultdict(list)  # for cross-sectional mean proxy: keyed by entry month
    tmp=[]
    for tk,do,sig in events:
        r,mr,ed=fwd(tk,do,2,a.hold)
        if r is None: continue
        yr=ed.year
        tmp.append((yr,sig,r,mr,ed))
    # if no market ticker, build cross-sectional mean per entry-month as proxy
    if not mkt:
        monthly=defaultdict(list)
        for yr,sig,r,mr,ed in tmp:
            monthly[(ed.year,ed.month)].append(r)
        monthly_mean={k:np.mean(v) for k,v in monthly.items()}
        for yr,sig,r,mr,ed in tmp:
            rel=r-monthly_mean.get((ed.year,ed.month),0.0)
            recs.append((yr,sig,r,rel))
    else:
        for yr,sig,r,mr,ed in tmp:
            rel=(r-mr) if mr is not None else None
            recs.append((yr,sig,r,rel))

    cost=a.cost_bps/10000.0
    def ls_spread(rows_sr, use_rel=False):
        sig=[x[1] for x in rows_sr]
        ret=[x[3] if use_rel else x[2] for x in rows_sr]
        pair=[(s,r) for s,r in zip(sig,ret) if r is not None]
        if len(pair)<a.min_year_events: return None
        s=[p[0] for p in pair]; r=[p[1] for p in pair]; n=len(s)
        ic=spearman(s,r)
        order=np.argsort(s); q=max(1,n//5)
        lo=order[:q]; hi=order[-q:]
        L=float(np.mean([r[i] for i in hi])); S=float(np.mean([r[i] for i in lo]))
        g=L-S; net=g-2*cost
        sd=math.sqrt(np.var([r[i] for i in hi])/q+np.var([r[i] for i in lo])/q)
        t=g/sd if sd>0 else None
        return {"n":n,"ic":ic,"long":L,"short":S,"gross":g,"net":net,"t":t}

    # ---- 1. year-by-year ----
    sub("YEAR-BY-YEAR (hold=%d) — is the edge stable, decaying, or dead?"%a.hold)
    years=sorted(set(x[0] for x in recs))
    print("  %-6s %7s | %-8s %-8s %-8s | %-8s"%("year","events","IC","net(raw)","t","net(rel)"))
    report={"by_year":{}}
    for yr in years:
        yr_rows=[x for x in recs if x[0]==yr]
        raw=ls_spread(yr_rows,use_rel=False)
        rel=ls_spread(yr_rows,use_rel=True)
        if raw is None:
            print("  %-6d %7d | (below %d events)"%(yr,len(yr_rows),a.min_year_events)); continue
        print("  %-6d %7d | %-8s %-8s %-8s | %-8s"
              %(yr,raw["n"],
                "%+.3f"%raw["ic"] if raw["ic"] is not None else "NA",
                "%+.4f"%raw["net"],
                "%.2f"%raw["t"] if raw["t"] else "NA",
                "%+.4f"%rel["net"] if rel else "NA"))
        report["by_year"][yr]={"raw":raw,"rel":rel}

    # ---- 2. first half vs second half (decay) ----
    sub("DECAY CHECK — first half vs second half of sample")
    mid=years[len(years)//2]
    early=[x for x in recs if x[0]<mid]; late=[x for x in recs if x[0]>=mid]
    er=ls_spread(early,False); lr=ls_spread(late,False)
    er_rel=ls_spread(early,True); lr_rel=ls_spread(late,True)
    if er and lr:
        print("  EARLY (%d-%d): events=%d  IC=%+.4f  net_raw=%+.4f  t=%.2f  net_rel=%s"
              %(years[0],mid-1,er["n"],er["ic"],er["net"],er["t"] or 0,
                "%+.4f"%er_rel["net"] if er_rel else "NA"))
        print("  LATE  (%d-%d): events=%d  IC=%+.4f  net_raw=%+.4f  t=%.2f  net_rel=%s"
              %(mid,years[-1],lr["n"],lr["ic"],lr["net"],lr["t"] or 0,
                "%+.4f"%lr_rel["net"] if lr_rel else "NA"))
        decay = (er["ic"]-lr["ic"])
        print("  IC decay (early - late): %+.4f  -> %s"
              %(decay, "DECAYED substantially" if decay>0.04 else
                       ("mild decay" if decay>0.015 else "STABLE (no meaningful decay)")))
        report["decay"]={"early":er,"late":lr,"early_rel":er_rel,"late_rel":lr_rel}

    # ---- 3. recent-years zoom ----
    sub("RECENT YEARS (2022+) — is it alive NOW?")
    recent=[x for x in recs if x[0]>=2022]
    rr=ls_spread(recent,False); rr_rel=ls_spread(recent,True)
    if rr:
        print("  2022-%d: events=%d  IC=%+.4f  net_raw=%+.4f  t=%.2f  net_rel=%s"
              %(years[-1],rr["n"],rr["ic"],rr["net"],rr["t"] or 0,
                "%+.4f"%rr_rel["net"] if rr_rel else "NA"))
        report["recent"]={"raw":rr,"rel":rr_rel}
    else:
        print("  too few recent events for a stable estimate")

    # ---- 4. decile monotonicity (pooled) ----
    sub("DECILE MONOTONICITY (pooled) — is it broad or tail-driven?")
    allrows=[(x[1],x[2]) for x in recs]
    s=[p[0] for p in allrows]; r=[p[1] for p in allrows]; n=len(s)
    order=np.argsort(s)
    print("  decile  mean_surprise   mean_fwd_ret  n")
    dec_means=[]
    for d in range(10):
        idx=order[int(d*n/10):int((d+1)*n/10)]
        if len(idx)==0: continue
        ms=np.mean([s[i] for i in idx]); mr=np.mean([r[i] for i in idx])
        dec_means.append(mr)
        print("   D%-2d    %+.3f          %+.4f       %d"%(d+1,ms,mr,len(idx)))
    # monotonicity: count how many adjacent steps go the right way
    ups=sum(1 for i in range(1,len(dec_means)) if dec_means[i]>=dec_means[i-1])
    print("  monotonic steps: %d/%d  (higher surprise -> higher return)"%(ups,len(dec_means)-1))
    print("  D10 - D1 spread: %+.4f"%(dec_means[-1]-dec_means[0]) if dec_means else "NA")

    # ---- verdict ----
    banner("VERDICT — is the PEAD edge tradeable NOW?")
    rec_ok = report.get("recent",{}).get("raw")
    decay = report.get("decay")
    msgs=[]
    if rec_ok and rec_ok["ic"] is not None:
        if rec_ok["ic"]>=0.05 and rec_ok["net"]>0.005:
            msgs.append("Recent (2022+) IC=%+.4f, net=%+.4f -> edge appears ALIVE recently."%(rec_ok["ic"],rec_ok["net"]))
        elif rec_ok["ic"]>=0.02:
            msgs.append("Recent (2022+) IC=%+.4f -> weak-positive recently; edge may be fading."%rec_ok["ic"])
        else:
            msgs.append("Recent (2022+) IC=%+.4f -> edge looks WEAK/GONE in recent years (decayed)."%rec_ok["ic"])
    if decay:
        d=decay["early"]["ic"]-decay["late"]["ic"]
        if d>0.04: msgs.append("Substantial decay early->late (%+.4f IC) -> historical effect, weaker now."%d)
        elif d>0.015: msgs.append("Mild decay early->late (%+.4f IC) -> still present but softening."%d)
        else: msgs.append("No meaningful decay early->late -> durable across the sample.")
    # beta check
    if decay and decay.get("late_rel"):
        lr=decay["late"]; lrr=decay["late_rel"]
        msgs.append("Market-relative LATE net=%+.4f vs raw LATE net=%+.4f -> %s"
                    %(lrr["net"],lr["net"],
                      "much of late raw spread was market beta" if lrr["net"]<lr["net"]*0.5
                      else "edge survives beta-stripping"))
    for m in msgs: print("  - "+m)
    print("\n  HOW TO READ: per-year t-stats are noisy (small N). Trust the TRAJECTORY and the")
    print("  half-vs-half / recent / market-relative numbers. A signal that's strong pooled but")
    print("  dead since 2022, or that vanishes when beta-stripped, is NOT tradeable today.")
    if a.out:
        with open(a.out,"a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":report},default=str)+"\n")
        print("\n  [report appended to %s]"%a.out)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
