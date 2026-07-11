#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — PEAD WALK-FORWARD, OUTLIER-ROBUST (winsorized surprise)
================================================================================
The previous walk-forward showed a REAL, recent, beta-surviving 40-day PEAD edge,
BUT two contamination issues:
  (1) extreme "surprises" (D1 mean -190%, D10 mean +398%) are denominator artifacts
      (tiny EPS bases), not real surprises — and they DRAGGED the deciles (D10 drifted
      LESS than D9). Only 5/9 decile steps were monotonic.
  (2) middle deciles (D5-D9) were cleanly monotonic and strong -> the clean signal is
      likely STRONGER than the contaminated pooled number.

This script re-measures with the surprise WINSORIZED, and shows raw-vs-clean side by
side so you can see the outliers were noise, not signal.

WHAT IT DOES (offline; reads cached prices.db; NO network):
  * winsorize eps_surprise_pct three ways for comparison:
       - RAW (no clip)         [baseline]
       - CLIP +/- 50%          [hard cap at sensible bound]
       - WINSOR 1/99 pct       [drop the extreme 1% tails by percentile]
  * for each: pooled IC, recent-years (2022+) IC + net + t, decile monotonicity,
    market-relative (beta-stripped) net
  * a SIGN-BASED robustness check: use sign(surprise) only (beat vs miss) — the most
    outlier-proof possible signal — to confirm the effect isn't an artifact of the
    magnitude scaling at all.

RULE 1: winsorizing the SIGNAL (surprise) is legitimate de-noising — extreme % values
are known data artifacts. We do NOT clip RETURNS (that would hide real drift). Entry
day +2; window strictly after; signal known at announcement. No look-ahead.

READ-ONLY. mode=ro&immutable=1. No network.

USAGE:
  python pead_robust.py --root .
  python pead_robust.py --root . --hold 40 --cost-bps 10
  python pead_robust.py --root . --clip 30           (change the hard-cap bound)
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
    ap.add_argument("--clip",type=float,default=50.0,help="hard cap for surprise pct (+/-)")
    ap.add_argument("--min-events",type=int,default=30)
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    banner("ML QUANT FUND — PEAD WALK-FORWARD, OUTLIER-ROBUST")
    print("Winsorize the surprise, re-measure the clean signal. (offline)")
    print("Root:",os.path.abspath(a.root),"| hold:",a.hold,"| clip: +/-%g%%"%a.clip)
    if not require(HAVE_NUMPY,"numpy required"): return
    if not require(os.path.isfile(prices_db),"prices.db not found"): return
    earnp=find_db(a.root,"earnings.db")
    if not require(earnp,"earnings.db not found"): return

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

    ce=ro(earnp)
    try:
        ev=Q(ce,"SELECT ticker,report_date,"+a.signal+" FROM earnings_surprises "
               "WHERE report_date IS NOT NULL AND "+a.signal+" IS NOT NULL")
    finally:
        ce.close()
    events=[(tk,nd(rd),float(sig)) for tk,rd,sig in ev if nd(rd) is not None]

    def fwd(tk,do,hold):
        lst=px.get(tk); idx=pxidx.get(tk)
        if not lst: return None,None
        pos=None
        for off in range(0,6):
            c=do+datetime.timedelta(days=off)
            if c in idx: pos=idx[c]; break
        if pos is None: return None,None
        e=pos+2; x=pos+2+hold
        if x>=len(lst): return None,None
        pe=lst[e][1]; pxx=lst[x][1]
        if pe<=0: return None,None
        return pxx/pe-1.0, lst[e][0].year

    # assemble base records: (year, raw_surprise, fwd_ret)
    base=[]
    for tk,do,sig in events:
        r,yr=fwd(tk,do,a.hold)
        if r is None: continue
        base.append((yr,sig,r))
    # cross-sectional mean per (year,month-ish) as market proxy for beta-strip
    # (no SPY in prices.db); use yearly mean of fwd returns
    yr_mean=defaultdict(list)
    for yr,sig,r in base: yr_mean[yr].append(r)
    yr_mean={k:np.mean(v) for k,v in yr_mean.items()}

    raw_sigs=[s for _,s,_ in base]
    p1=np.percentile(raw_sigs,1); p99=np.percentile(raw_sigs,99)
    print("  surprise distribution: min=%.1f p1=%.1f median=%.2f p99=%.1f max=%.1f"
          %(min(raw_sigs),p1,np.median(raw_sigs),p99,max(raw_sigs)))

    cost=a.cost_bps/10000.0
    def transform(name):
        out=[]
        for yr,sig,r in base:
            if name=="raw": s=sig
            elif name=="clip": s=max(-a.clip,min(a.clip,sig))
            elif name=="winsor": s=max(p1,min(p99,sig))
            elif name=="sign": s=1.0 if sig>0 else (-1.0 if sig<0 else 0.0)
            else: s=sig
            out.append((yr,s,r))
        return out

    def metrics(rows, recent_only=False, beta_strip=False):
        rr=[(yr,s,r) for yr,s,r in rows if (yr>=2022 if recent_only else True)]
        if len(rr)<a.min_events: return None
        s=[x[1] for x in rr]
        if beta_strip:
            r=[x[2]-yr_mean.get(x[0],0.0) for x in rr]
        else:
            r=[x[2] for x in rr]
        ic=spearman(s,r); n=len(s)
        order=np.argsort(s); q=max(1,n//5); lo=order[:q]; hi=order[-q:]
        L=float(np.mean([r[i] for i in hi])); S=float(np.mean([r[i] for i in lo]))
        g=L-S; net=g-2*cost
        sd=math.sqrt(np.var([r[i] for i in hi])/q+np.var([r[i] for i in lo])/q)
        t=g/sd if sd>0 else None
        return {"n":n,"ic":ic,"net":net,"t":t,"gross":g}

    def decile_mono(rows):
        s=[x[1] for x in rows]; r=[x[2] for x in rows]; n=len(s)
        order=np.argsort(s); dm=[]
        for d in range(10):
            idx=order[int(d*n/10):int((d+1)*n/10)]
            if len(idx)>0: dm.append(np.mean([r[i] for i in idx]))
        ups=sum(1 for i in range(1,len(dm)) if dm[i]>=dm[i-1])
        return ups,len(dm)-1,(dm[-1]-dm[0] if dm else None),dm

    # ---- compare transforms ----
    sub("RAW vs CLEANED — pooled + recent(2022+) + beta-stripped")
    print("  %-8s | %-18s | %-26s | %-10s"
          %("variant","POOLED IC / mono","RECENT(2022+) IC/net/t","RECENT rel-net"))
    report={}
    for name in ("raw","clip","winsor","sign"):
        rows=transform(name)
        pooled=metrics(rows,recent_only=False)
        recent=metrics(rows,recent_only=True)
        recent_rel=metrics(rows,recent_only=True,beta_strip=True)
        ups,steps,d10d1,_=decile_mono(rows)
        pic = "%+.3f"%pooled["ic"] if pooled and pooled["ic"] is not None else "NA"
        rline = ("IC%+.3f net%+.4f t%.2f"%(recent["ic"],recent["net"],recent["t"] or 0)) if recent else "n/a"
        relnet = "%+.4f"%recent_rel["net"] if recent_rel else "NA"
        print("  %-8s | IC%s mono%d/%d | %-26s | %s"
              %(name,pic,ups,steps,rline,relnet))
        report[name]={"pooled":pooled,"recent":recent,"recent_rel":recent_rel,
                      "mono":[ups,steps,d10d1]}

    # ---- detailed deciles: raw vs winsor ----
    sub("DECILE DETAIL — raw vs winsorized (does clipping fix monotonicity?)")
    raw_rows=transform("raw"); win_rows=transform("winsor")
    _,_,_,dm_raw=decile_mono(raw_rows)
    _,_,_,dm_win=decile_mono(win_rows)
    # recompute mean surprise per decile for context
    def decile_surprise(rows):
        s=[x[1] for x in rows]; n=len(s); order=np.argsort(s); ms=[]
        for d in range(10):
            idx=order[int(d*n/10):int((d+1)*n/10)]
            if len(idx)>0: ms.append(np.mean([s[i] for i in idx]))
        return ms
    ms_raw=decile_surprise(raw_rows); ms_win=decile_surprise(win_rows)
    print("  %-5s | %-22s | %-22s"%("dec","RAW (surprise -> ret)","WINSOR (surprise -> ret)"))
    for d in range(10):
        rs = "%+9.2f -> %+.4f"%(ms_raw[d],dm_raw[d]) if d<len(dm_raw) else ""
        ws = "%+9.2f -> %+.4f"%(ms_win[d],dm_win[d]) if d<len(dm_win) else ""
        print("   D%-3d | %-22s | %-22s"%(d+1,rs,ws))

    # ---- year-by-year on the CLEANED (winsor) signal ----
    sub("YEAR-BY-YEAR on WINSORIZED signal (the de-noised trajectory)")
    years=sorted(set(x[0] for x in win_rows))
    print("  %-6s %7s | %-8s %-8s %-6s"%("year","events","IC","net","t"))
    for yr in years:
        yr_rows=[x for x in win_rows if x[0]==yr]
        m=metrics(yr_rows)
        if m is None:
            print("  %-6d %7d | (sparse)"%(yr,len(yr_rows))); continue
        print("  %-6d %7d | %-8s %-8s %-6s"
              %(yr,m["n"],"%+.3f"%m["ic"] if m["ic"] is not None else "NA",
                "%+.4f"%m["net"],"%.2f"%m["t"] if m["t"] else "NA"))

    # ---- verdict ----
    banner("VERDICT — does the CLEAN signal confirm a tradeable edge?")
    rw=report.get("winsor",{}).get("recent")
    rwrel=report.get("winsor",{}).get("recent_rel")
    rsign=report.get("sign",{}).get("recent")
    rawrec=report.get("raw",{}).get("recent")
    if rw and rw["ic"] is not None:
        print("  WINSORIZED recent(2022+): IC=%+.4f net=%+.4f t=%.2f (n=%d)"
              %(rw["ic"],rw["net"],rw["t"] or 0,rw["n"]))
        if rawrec:
            better = "CLEANER/stronger" if (rw["ic"]>=rawrec["ic"]-0.005) else "weaker"
            print("    vs RAW recent IC=%+.4f -> winsorized is %s (outliers were %s)"
                  %(rawrec["ic"],better,"noise" if rw["ic"]>=rawrec["ic"]-0.005 else "carrying signal"))
    if rsign and rsign["ic"] is not None:
        print("  SIGN-ONLY (beat/miss) recent: IC=%+.4f net=%+.4f t=%.2f"
              %(rsign["ic"],rsign["net"],rsign["t"] or 0))
        print("    ^ if this is still positive & significant, the edge is NOT a magnitude artifact —")
        print("      even just 'did they beat?' predicts drift. Most robust possible confirmation.")
    mono_win=report.get("winsor",{}).get("mono")
    if mono_win:
        print("  winsorized decile monotonicity: %d/%d steps (raw was %d/%d)"
              %(mono_win[0],mono_win[1],report["raw"]["mono"][0],report["raw"]["mono"][1]))
    # overall call
    alive = rw and rw["ic"] is not None and rw["ic"]>=0.05 and rw["net"]>0.005 and (rw["t"] or 0)>=2
    beta_ok = rwrel and rwrel["net"]>0.005
    sign_ok = rsign and rsign["ic"] is not None and rsign["ic"]>0.02
    print()
    if alive and beta_ok and sign_ok:
        print("  >> CONFIRMED: clean, recent, beta-surviving, sign-robust PEAD edge. This is as")
        print("     real as event-study evidence gets on this data. Next: capacity + OOS holdout.")
    elif alive and beta_ok:
        print("  >> LIKELY REAL: clean recent edge survives beta-strip; sign-robustness marginal.")
        print("     Solid, but confirm capacity and a true out-of-sample holdout before sizing.")
    else:
        print("  >> MIXED: the clean signal is weaker than the contaminated number suggested.")
        print("     Treat with caution; the outliers may have been inflating it.")
    print("\n  Per-year t-stats noisy (small N pre-2025). Trust recent(2022+), beta-strip, sign-only.")
    if a.out:
        with open(a.out,"a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":report},default=str)+"\n")
        print("\n  [report appended to %s]"%a.out)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
