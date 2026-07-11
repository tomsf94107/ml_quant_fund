#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — SI BOOK TRACKER (mark-to-market + exit flags)
================================================================================
The live generator (si_positions_live.py) logs intended entries to si_live_ledger.csv.
This reads that ledger, marks every open position to the latest price, and flags when a
cohort has reached its ~40-trading-day hold (time to close / rebalance into the new
settlement). It is the operational counterpart to the generator -- run it whenever you
want a P&L read on the live book.

COHORT = one (generated_on, settlement) batch. Each time you run the generator on a new
settlement and trade it, that's a new cohort in the ledger; this tracks them all and
tells you which are due to exit.

WHAT IT SHOWS:
  * per cohort: trading days elapsed, days to exit, cohort return + $ P&L, EXIT DUE flag
  * per cohort: best/worst names
  * total book: open $ P&L across cohorts
P&L uses adj_close (split/div-adjusted) both at entry (ref_px in ledger) and now, so the
return is clean. Long positions profit when price rises; shorts when it falls.

RULE 1: trading-day count from the actual prices.db calendar; signed by side; READ-ONLY.
Not investment advice.

USAGE:
  python si_track.py --root .
  python si_track.py --root . --hold-days 40 --detail
================================================================================
"""
import argparse, os, csv, sqlite3, datetime
from collections import defaultdict

def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def nd(s):
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None
LINE="="*78

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--ledger",default=None)
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--hold-days",type=int,default=40)
    ap.add_argument("--detail",action="store_true")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    ledger=a.ledger or os.path.join(a.root,"si_live_ledger.csv")
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    print("\n"+LINE+"\nSI BOOK TRACKER — mark-to-market + exit flags\n"+LINE)
    if not os.path.isfile(ledger): print("  [STOP] ledger not found: %s\n  (run si_positions_live.py --log-ledger first)"%ledger); return
    if not os.path.isfile(prices_db): print("  [STOP] prices.db not found"); return

    with open(ledger) as f: rows=list(csv.DictReader(f))
    if not rows: print("  ledger is empty."); return

    cp=ro(prices_db)
    try: prows=Q(cp,"SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL")
    finally: cp.close()
    last_px={}; last_dt={}; alldates=set()
    for tk,d,p in prows:
        do=nd(d)
        if do is None: continue
        try: pf=float(p)
        except Exception: continue
        if pf<=0: continue
        alldates.add(do)
        if tk not in last_dt or do>last_dt[tk]:
            last_dt[tk]=do; last_px[tk]=pf
    caldates=sorted(alldates)
    today_px_date=caldates[-1] if caldates else None
    def tdays_between(d0,d1):
        # trading days in (d0, d1] using the price calendar
        if d0 is None or d1 is None: return 0
        return sum(1 for d in caldates if d0<d<=d1)

    # group by cohort
    cohorts=defaultdict(list)
    for r in rows:
        cohorts[(r.get("generated_on"),r.get("settlement"))].append(r)

    print("  ledger: %d positions across %d cohort(s) | latest price date: %s | hold=%dtd"
          %(len(rows),len(cohorts),today_px_date,a.hold_days))

    book_pnl=0.0; book_cost=0.0
    for (gen,settle),items in sorted(cohorts.items()):
        gd=nd(gen); elapsed=tdays_between(gd,today_px_date)
        to_exit=a.hold_days-elapsed
        coh_pnl=0.0; coh_cost=0.0; rets=[]
        for r in items:
            tk=r["ticker"].upper(); side=r.get("side","LONG")
            try:
                shares=float(r.get("shares") or 0); ref=float(r.get("ref_px") or 0); usd=float(r.get("usd") or 0)
            except Exception: continue
            cur=last_px.get(tk)
            if cur is None or ref<=0: continue
            raw=(cur-ref)*shares
            signed = raw if side.upper().startswith("LONG") else -raw
            ret=(cur/ref-1.0); ret = ret if side.upper().startswith("LONG") else -ret
            coh_pnl+=signed; coh_cost+=usd; rets.append((tk,ret,signed))
        book_pnl+=coh_pnl; book_cost+=coh_cost
        cohret=100*coh_pnl/coh_cost if coh_cost>0 else 0
        flag=" *** EXIT DUE ***" if elapsed>=a.hold_days else ""
        print("\n"+"-"*78)
        print("  COHORT  gen=%s  settlement=%s"%(gen,settle))
        print("  elapsed=%dtd | to exit=%dtd%s | positions=%d | deployed=$%.0f"
              %(elapsed, max(0,to_exit), flag, len(items), coh_cost))
        print("  cohort P&L = $%.0f  (%.2f%%)"%(coh_pnl,cohret))
        if rets:
            rets.sort(key=lambda x:x[1])
            worst=rets[0]; best=rets[-1]
            print("  best: %s %+.1f%% ($%+.0f) | worst: %s %+.1f%% ($%+.0f)"
                  %(best[0],100*best[1],best[2],worst[0],100*worst[1],worst[2]))
            if a.detail:
                print("    %-8s %6s %10s %10s"%("ticker","ret%","$P&L","side"))
                for tk,ret,signed in sorted(rets,key=lambda x:-x[1]):
                    sd=next((r.get("side") for r in items if r["ticker"].upper()==tk),"")
                    print("    %-8s %+5.1f%% %+10.0f %10s"%(tk,100*ret,signed,sd))

    print("\n"+LINE)
    bret=100*book_pnl/book_cost if book_cost>0 else 0
    print("  TOTAL OPEN BOOK: deployed $%.0f | open P&L $%+.0f (%.2f%%)"%(book_cost,book_pnl,bret))
    due=[(g,s) for (g,s),items in cohorts.items() if tdays_between(nd(g),today_px_date)>=a.hold_days]
    if due:
        print("  >> %d cohort(s) past the %dtd hold -- EXIT DUE (close/rebalance into latest settlement):"%(len(due),a.hold_days))
        for g,s in due: print("       gen=%s settlement=%s"%(g,s))
    else:
        print("  >> no cohorts due to exit yet.")
    print("\n  Marked vs adj_close @ %s. Not investment advice."%today_px_date)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
