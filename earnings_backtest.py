#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — POST-EARNINGS LONG/SHORT BACKTEST  (is the edge tradeable?)
================================================================================
The IC test (earnings_signal.py) found eps_surprise carries a real event-window
edge (IC ~0.079 @+1d, t~2.9). This script answers the next question: does that
IC turn into MONEY once you account for (a) the bid-ask spread, and (b) the fact
that the announcement-gap return is NOT capturable?

WHAT IT DOES:
  * Forms a post-earnings long/short: long the top-tercile eps_surprise events,
    short the bottom tercile, each held 1/3/5 days after the announcement.
  * Reports the long-short return spread, GROSS and NET of cost.
  * Computes the spread TWO WAYS to expose the announcement-gap problem:
       (A) GAP-INCLUSIVE: return measured from the announcement reference point
           (includes the overnight earnings gap — often NOT tradeable).
       (B) TRADEABLE: return measured starting the day AFTER the announcement
           (the harvestable drift only — what you can actually capture).
    The difference (A - B) is how much of the "edge" is the un-tradeable gap.
  * RIGOROUSLY excludes any feature computed from post-announcement returns
    (post_drift_3d etc.) — only eps_surprise / eps_surprise_pct / rev_surprise
    are used as the SIGNAL, all knowable at announcement.

LEAKAGE GUARD (RULE 1): the signal (surprise) is known at announcement. The
return windows for timing (B) start strictly AFTER the announcement day, so
signal and return never overlap. Timing (A) is reported ONLY to quantify the
gap; it is labelled untradeable and never presented as the deployable number.

READ-ONLY. Never writes. SQLite mode=ro&immutable=1.

USAGE (project root, env active):
  python earnings_backtest.py --root .
  python earnings_backtest.py --root . --cost-bps 10 --holds 1,3,5 --min-events 30
  add --out earnings_backtest.json for machine-readable results.

--cost-bps is ROUND-TRIP cost in basis points charged to EACH leg per trade
(default 10bps = 0.10%; earnings names often cost MORE — try 20-30 to stress).
================================================================================
"""
import argparse, os, sqlite3, sys, math, json, datetime
from collections import defaultdict

try:
    import numpy as np; HAVE_NUMPY=True
except Exception:
    HAVE_NUMPY=False

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)
def ro(p):
    if not os.path.isfile(p): raise FileNotFoundError(p)
    return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def q(c,s,p=()): return c.execute(s,p).fetchall()
def has_table(c,n): return bool(q(c,"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(n,)))
def cols_of(c,t): return [r[1] for r in q(c,'PRAGMA table_info("'+t+'")')]
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
def norm_date(s):
    if s is None: return None
    s=str(s)[:10]
    try: return datetime.date.fromisoformat(s)
    except Exception: return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--cost-bps",type=float,default=10.0,help="round-trip cost per leg, bps")
    ap.add_argument("--holds",default="1,3,5")
    ap.add_argument("--min-events",type=int,default=30)
    ap.add_argument("--signal",default="eps_surprise_pct",
                    help="which surprise column to sort on (eps_surprise/eps_surprise_pct/rev_surprise)")
    ap.add_argument("--out",default=None)
    args=ap.parse_args(); args.root=os.path.expanduser(args.root)
    holds=[int(x) for x in args.holds.split(",")]

    banner("ML QUANT FUND — POST-EARNINGS LONG/SHORT BACKTEST")
    print("Tests whether the eps_surprise IC turns into money. Net of cost. Both timings.")
    print("Root:",os.path.abspath(args.root),"| numpy:",HAVE_NUMPY,
          "| cost/leg:",args.cost_bps,"bps | signal:",args.signal)
    if not require(HAVE_NUMPY,"numpy required"): return

    accp=find_db(args.root,"accuracy.db"); earnp=find_db(args.root,"earnings.db")
    if not require(accp,"accuracy.db not found"): return
    if not require(earnp,"earnings.db not found"): return
    conn_a=ro(accp); conn_e=ro(earnp)
    report={}
    try:
        if not require(has_table(conn_a,"outcomes"),"no outcomes"): return
        if not require(has_table(conn_e,"earnings_surprises"),"no earnings_surprises"): return
        es_cols=cols_of(conn_e,"earnings_surprises")
        if not require(args.signal in es_cols, "earnings_surprises has no column "+args.signal): return

        # ---- build per-ticker date->{h:ret} lookup ----
        sub("Loading outcomes (forward returns)")
        rows=q(conn_a,"SELECT ticker,prediction_date,horizon,actual_return FROM outcomes "
                      "WHERE actual_return IS NOT NULL")
        ret_by=defaultdict(dict)  # ticker -> {date_obj:{h:ret}}
        all_dates=set()
        for tk,d,h,r in rows:
            do=norm_date(d)
            if do is None: continue
            ret_by[tk].setdefault(do,{})[h]=r
            all_dates.add(do)
        dmin,dmax=(min(all_dates),max(all_dates)) if all_dates else (None,None)
        print("  outcomes: %d rows, %d tickers, %s..%s"%(len(rows),len(ret_by),dmin,dmax))

        # ---- load earnings events with the chosen signal, within outcomes window ----
        sub("Loading earnings events (signal = %s)" % args.signal)
        ev=q(conn_e,"SELECT ticker,report_date,"+args.signal+" FROM earnings_surprises "
                    "WHERE report_date IS NOT NULL AND "+args.signal+" IS NOT NULL")
        events=[]
        for tk,rd,sig in ev:
            do=norm_date(rd)
            if do is None: continue
            if dmin and dmax and dmin<=do<=dmax: events.append((tk,do,sig))
        print("  events with signal, overlapping outcomes: %d"%len(events))
        if len(events)<args.min_events*3:
            print("  [STOP] too few events (%d) for tercile L/S (need >=%d). "
                  %(len(events),args.min_events*3))
            _w(args,{"events":len(events)}); return

        # helpers to fetch returns under the two timings
        def nearest_on_or_after(tk,do,maxgap=4):
            dd=ret_by.get(tk,{})
            for off in range(0,maxgap+1):
                cand=do+datetime.timedelta(days=off)
                if cand in dd: return cand
            return None
        def nearest_after(tk,do,maxgap=5):
            dd=ret_by.get(tk,{})
            for off in range(1,maxgap+1):
                cand=do+datetime.timedelta(days=off)
                if cand in dd: return cand
            return None

        cost=args.cost_bps/10000.0
        for h in holds:
            sub("HOLD = %d trading day(s) after announcement"%h)
            # timing A (gap-inclusive): return at horizon h measured from the on/after-announcement
            #   reference date (this anchor includes the announcement-day move/gap)
            # timing B (tradeable): return at horizon h measured from the FIRST date strictly
            #   AFTER the announcement (the gap has already happened; we capture only the drift)
            recs=[]
            for tk,do,sig in events:
                aA=nearest_on_or_after(tk,do,4)
                aB=nearest_after(tk,do,5)
                rA=ret_by[tk][aA].get(h) if aA is not None else None
                rB=ret_by[tk][aB].get(h) if aB is not None else None
                recs.append((sig,rA,rB))
            sigs=[r[0] for r in recs]
            n=len(sigs)
            # tercile cutoffs on the signal
            order=np.argsort(sigs)
            t1=order[:n//3]; t3=order[-(n//3):]
            def spread(idx_long, idx_short, which):
                col=1 if which=="A" else 2
                lr=[recs[i][col] for i in idx_long if recs[i][col] is not None]
                sr=[recs[i][col] for i in idx_short if recs[i][col] is not None]
                if len(lr)<args.min_events or len(sr)<args.min_events: return None
                long_ret=np.mean(lr); short_ret=np.mean(sr)
                gross=long_ret-short_ret
                # cost: charged to BOTH legs (long and short), round trip => 2 legs * cost each side
                # we model entry+exit on each leg = 2*cost per leg, 2 legs = 4*cost total... but
                # conservatively charge 2*cost per leg (round trip) * 2 legs = 4*cost. Keep explicit:
                net=gross - 4*cost
                return {"long":long_ret,"short":short_ret,"gross":gross,"net":net,
                        "n_long":len(lr),"n_short":len(sr)}
            sA=spread(t3,t1,"A"); sB=spread(t3,t1,"B")
            def show(tag,s,note):
                if s is None: print("    %-12s insufficient events"%tag); return
                print("    %-12s long=%+.4f short=%+.4f | GROSS=%+.4f NET=%+.4f  (n_L=%d n_S=%d)  %s"
                      %(tag,s["long"],s["short"],s["gross"],s["net"],s["n_long"],s["n_short"],note))
            show("(A) gap-incl", sA, "<- includes announcement gap (often UNtradeable)")
            show("(B) tradeable", sB, "<- starts day AFTER announcement (harvestable)")
            if sA and sB:
                gap_share = (sA["gross"]-sB["gross"])
                print("    -> gap component (A-B gross): %+.4f  "
                      "(%.0f%% of the gap-inclusive edge is the un-tradeable gap)"
                      %(gap_share, 100*gap_share/sA["gross"] if sA["gross"] not in (0,None) else 0))
            report.setdefault("holds",{})[h]={"A":sA,"B":sB}

        # ---- verdict ----
        banner("VERDICT — is the post-earnings edge actually tradeable after costs?")
        any_trade=False
        for h in holds:
            b=report.get("holds",{}).get(h,{}).get("B")
            if b and b["net"]>0:
                # annualize roughly: each name held h days, ~quarterly events; just report per-trade
                print("  HOLD %dd: TRADEABLE net spread = %+.4f per event (after %g bps/leg)"
                      %(h,b["net"],args.cost_bps))
                if b["net"]>0.002:
                    any_trade=True
        if not any_trade:
            print("  After costs, the TRADEABLE (post-gap) long/short spread is ~0 or negative at all")
            print("  holds. The IC was real but the harvestable, after-cost edge is not — most of it")
            print("  was the un-tradeable announcement gap and/or eaten by spread. This is the honest")
            print("  answer: eps_surprise is a real signal but NOT a standalone tradeable L/S here.")
        else:
            print("\n  ^ A positive NET tradeable spread survives. This is a candidate strategy.")
            print("  Next: confirm across cost levels (--cost-bps 20, 30), check capacity in your")
            print("  liquid universe, and walk-forward it before any capital.")
        print("\n  REMINDER: gross IC != tradeable money. The (A) vs (B) gap above shows how much")
        print("  of the apparent edge you literally cannot capture. Size to the (B) NET number only.")
    finally:
        conn_a.close(); conn_e.close()
    _w(args,report)

def _w(args,report):
    if not args.out: return
    path=args.out
    if os.path.isdir(path) or path.endswith("/"): path=os.path.join(path,"earnings_backtest.json")
    with open(path,"a") as f:
        f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":report},default=str)+"\n")
    print("\n  [report appended to %s]"%path)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
