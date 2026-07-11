#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — PEAD BACKTEST v2  (Option 1: corrected SHORT-horizon, h<=5)
================================================================================
Fixes the THREE code bugs the user correctly flagged in earnings_backtest.py:

  BUG 1 (gap not isolated): v1 compared horizon-h returns starting day0 vs day1 —
         overlapping windows, so "A-B" was a 1-day shift, not gap-vs-drift, giving
         nonsense gap-shares (125-242%). FIX: build NON-OVERLAPPING pieces from
         the h=1/3/5 horizons we actually have:
            announcement-day move  = h1 return anchored at/after announcement
            drift (day +2 onward)  = (h3 from day0) - (h1 from day0)  [days 2..3]
                                     and (h5 from day0) - (h1 from day0) [days 2..5]
         so the drift NEVER includes the announcement day. Entry is effectively
         Day +2, matching the canonical PEAD construction (enter day 2, skip the
         initial volatility/gap).
  BUG 2 (cost 4x): v1 charged 4*cost. A long/short = 2 legs, one round-trip each.
         FIX: net = gross - 2*cost_roundtrip.
  BUG 3 (contaminated alignment): v1 used nearest_on_or_after which could land mid
         drift. FIX: explicit anchoring, and we require the SAME anchor row to have
         both h1 and h3/h5 so the subtraction is coherent (same start date).

KNOWN LIMITATION (this is Option 1, not the full test): the canonical PEAD holds
~60 trading days. outcomes only has h=1/3/5, so this tests the SHORT-horizon
drift slice (days +2..+5), which the literature (Quantpedia) considers marginal-
to-deteriorating OOS. Option 2 (pead_backtest_60d.py) tests the real 60-day
strategy IF a price table exists. Read this result as "the short slice," not "PEAD".

LEAKAGE GUARD (RULE 1): signal = eps_surprise_pct, known at announcement. The
drift window starts day +2 (strictly after announcement), so signal and the
measured drift never overlap. The announcement-day piece is reported ONLY to show
how much sits in the (untradeable) initial move; it is never the deployable number.

READ-ONLY. mode=ro&immutable=1.

USAGE:
  python pead_backtest_v2.py --root .
  python pead_backtest_v2.py --root . --cost-bps 10 --signal eps_surprise_pct
  --cost-bps = round-trip cost PER LEG in bps (default 10). Net charges 2 legs.
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
    cand=os.path.join(root,name)
    if os.path.isfile(cand): return cand
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        if name in fn: return os.path.join(dp,name)
    return None
def nd(s):
    if s is None: return None
    s=str(s)[:10]
    try: return datetime.date.fromisoformat(s)
    except Exception: return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--signal",default="eps_surprise_pct")
    ap.add_argument("--min-events",type=int,default=30)
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    banner("ML QUANT FUND — PEAD BACKTEST v2 (Option 1: corrected short-horizon)")
    print("Fixes gap-isolation, cost (2x), alignment. Tests drift days +2..+5 only.")
    print("Root:",os.path.abspath(a.root),"| numpy:",HAVE_NUMPY,"| cost/leg:",a.cost_bps,"bps")
    if not require(HAVE_NUMPY,"numpy required"): return
    accp=find_db(a.root,"accuracy.db"); earnp=find_db(a.root,"earnings.db")
    if not require(accp,"accuracy.db not found"): return
    if not require(earnp,"earnings.db not found"): return
    ca=ro(accp); ce=ro(earnp); report={}
    try:
        if not require(has_table(ca,"outcomes"),"no outcomes"): return
        if not require(has_table(ce,"earnings_surprises"),"no earnings_surprises"): return
        if not require(a.signal in cols_of(ce,"earnings_surprises"),"no col "+a.signal): return

        # build ticker -> date -> {h:ret}; we need SAME-anchor h1/h3/h5 for clean subtraction
        rows=Q(ca,"SELECT ticker,prediction_date,horizon,actual_return FROM outcomes WHERE actual_return IS NOT NULL")
        ret=defaultdict(lambda: defaultdict(dict))
        dates=set()
        for tk,d,h,r in rows:
            do=nd(d)
            if do is None: continue
            ret[tk][do][h]=r; dates.add(do)
        dmin,dmax=(min(dates),max(dates)) if dates else (None,None)
        print("\n  outcomes: %d rows, %d tickers, %s..%s"%(len(rows),len(ret),dmin,dmax))

        ev=Q(ce,"SELECT ticker,report_date,"+a.signal+" FROM earnings_surprises "
                "WHERE report_date IS NOT NULL AND "+a.signal+" IS NOT NULL")
        events=[]
        for tk,rd,sig in ev:
            do=nd(rd)
            if do is None: continue
            if dmin and dmax and dmin<=do<=dmax: events.append((tk,do,sig))
        print("  earnings events overlapping outcomes: %d"%len(events))
        if len(events)<a.min_events*3:
            print("  [STOP] too few events"); _w(a,{"events":len(events)}); return

        def anchor(tk,do,maxgap=4):
            dd=ret[tk]
            for off in range(0,maxgap+1):
                c=do+datetime.timedelta(days=off)
                if c in dd: return c
            return None

        # assemble: for each event get same-anchor h1,h3,h5 so drift = h3-h1 (days 2..3),
        # h5-h1 (days 2..5). announcement piece = h1.
        recs=[]
        for tk,do,sig in events:
            an=anchor(tk,do,4)
            if an is None: continue
            d=ret[tk][an]
            h1=d.get(1); h3=d.get(3); h5=d.get(5)
            recs.append((sig,h1,h3,h5))
        sigs=[r[0] for r in recs]; n=len(sigs)
        order=np.argsort(sigs); tlo=order[:n//3]; thi=order[-(n//3):]
        cost=a.cost_bps/10000.0

        def spread(getter,label,note):
            lr=[getter(recs[i]) for i in thi if getter(recs[i]) is not None]
            sr=[getter(recs[i]) for i in tlo if getter(recs[i]) is not None]
            if len(lr)<a.min_events or len(sr)<a.min_events:
                print("    %-22s insufficient"%label); return None
            L=np.mean(lr); S=np.mean(sr); g=L-S; net=g-2*cost
            # t-stat of the spread (two-sample, rough)
            sd=math.sqrt(np.var(lr)/len(lr)+np.var(sr)/len(sr))
            t=g/sd if sd>0 else None
            print("    %-22s long=%+.4f short=%+.4f | GROSS=%+.4f NET=%+.4f t=%s  (nL=%d nS=%d) %s"
                  %(label,L,S,g,net,"%.2f"%t if t else "NA",len(lr),len(sr),note))
            return {"long":L,"short":S,"gross":g,"net":net,"t":t,"nL":len(lr),"nS":len(sr)}

        sub("NON-OVERLAPPING decomposition (the bug-1 fix)")
        print("  announcement piece = h1 (initial move/gap, mostly UNtradeable)")
        print("  drift +2..+3       = h3 - h1  (enter day +2, the harvestable slice)")
        print("  drift +2..+5       = h5 - h1  (enter day +2, longer slice)")
        print()
        ann   = spread(lambda r: r[1], "announcement (h1)", "<- initial move, not the strategy")
        d23   = spread(lambda r:(r[2]-r[1]) if (r[1] is not None and r[2] is not None) else None,
                       "drift +2..+3 (h3-h1)", "<- TRADEABLE")
        d25   = spread(lambda r:(r[3]-r[1]) if (r[1] is not None and r[3] is not None) else None,
                       "drift +2..+5 (h5-h1)", "<- TRADEABLE")
        report={"announcement":ann,"drift_2_3":d23,"drift_2_5":d25,"cost_bps":a.cost_bps,"events":n}

        banner("VERDICT — short-horizon (days +2..+5) drift, net of cost")
        tradeable=[x for x in [("+2..+3",d23),("+2..+5",d25)] if x[1] and x[1]["net"]>0]
        sig_tradeable=[x for x in tradeable if x[1]["t"] is not None and abs(x[1]["t"])>=2.0 and x[1]["net"]>0.001]
        if sig_tradeable:
            for lbl,s in sig_tradeable:
                print("  [TRADEABLE] drift %s: NET=%+.4f t=%.2f — survives cost AND significant"%(lbl,s["net"],s["t"]))
            print("\n  A short-horizon slice survives. But remember: this is days +2..+5 only.")
            print("  The canonical PEAD edge is at 20-60 days — run Option 2 for the real test.")
        else:
            print("  No short-horizon (+2..+5) drift slice is both positive-net AND significant.")
            print("  This matches the literature: the 1-5 day window is marginal; PEAD lives at")
            print("  20-60 days. This is NOT a verdict on PEAD — only on the short slice your")
            print("  h=1/3/5 data can see. Option 2 (60-day) is the real test.")
        if ann and (d23 or d25):
            drift_best=max([s["gross"] for _,s in [("a",d23),("b",d25)] if s], default=0)
            print("\n  For context: announcement-piece gross=%+.4f vs best drift-slice gross=%+.4f"
                  %(ann["gross"], drift_best))
            print("  (most of the earnings reaction is in the announcement move you can't trade)")
    finally:
        ca.close(); ce.close()
    _w(a,report)

def _w(a,report):
    if not a.out: return
    path=a.out
    if os.path.isdir(path) or path.endswith("/"): path=os.path.join(path,"pead_v2.json")
    with open(path,"a") as f:
        f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":report},default=str)+"\n")
    print("\n  [report appended to %s]"%path)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
