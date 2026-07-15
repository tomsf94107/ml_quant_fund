#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — PEAD OUT-OF-SAMPLE HOLDOUT  (lock the brick)
================================================================================
Everything so far measured PEAD over the full period — in-sample. This is the
decisive test: freeze the methodology on a TRAINING period, then validate COLD on
a held-out period the analysis has never touched. If the edge survives OOS, the
brick is real. If it collapses, it was in-sample fitting.

DESIGN:
  * SIGNAL: SUE (PIT-trailing standardized surprise) — the clean signal we settled on
  * SPLIT: train = events before --split-date (default 2024-01-01)
           test  = events on/after --split-date  (the COLD holdout)
  * The training period is used ONLY to confirm the edge exists and fix the recipe
    (40-day hold, day+2 entry, quintile L/S). NOTHING is tuned on the test period.
  * Report IS (train) vs OOS (test): IC, net spread, t-stat, decile monotonicity.
  * OOS is the number that matters. IS is shown only to confirm the recipe was valid
    on the data we developed it on.

The honest test: does OOS IC hold near IS IC, or decay toward zero?

RULE 1: the test period is never used to choose parameters. SUE denominator strictly
trailing. Entry day +2, window strictly after. No look-ahead, no peeking at OOS.

READ-ONLY. mode=ro&immutable=1. No network.

USAGE:
  python pead_oos.py --root .
  python pead_oos.py --root . --split-date 2024-01-01 --hold 40
  python pead_oos.py --root . --split-date 2023-06-01   (try different splits)
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
    ap.add_argument("--split-date",default="2024-01-01")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--min-prior",type=int,default=4)
    ap.add_argument("--min-events",type=int,default=30)
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    split=nd(a.split_date)
    banner("ML QUANT FUND — PEAD OUT-OF-SAMPLE HOLDOUT (lock the brick)")
    print("Train < %s ; TEST >= %s (cold). hold=%d. SUE signal. (offline)"%(a.split_date,a.split_date,a.hold))
    if not require(HAVE_NUMPY,"numpy required"): return
    if not require(os.path.isfile(prices_db),"prices.db not found"): return
    if not require(split is not None,"bad --split-date"): return
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
        cols=cols_of(ce,"earnings_surprises")
        have_comp="eps_actual" in cols and "eps_estimate" in cols
        sel="ticker,report_date"+(",eps_actual,eps_estimate" if have_comp else ",eps_surprise_pct")
        ev=Q(ce,"SELECT "+sel+" FROM earnings_surprises WHERE report_date IS NOT NULL")
        # LEAK FIX (Jul 15 2026): earnings_surprises.report_date = FISCAL PERIOD END,
        # 14-30d BEFORE the announcement (verified AAPL/MSFT/JNJ; only 625/21064 rows
        # coincide with announce). Entry at report_date+2 traded the surprise before
        # it was public. Override with announce-dated events when available.
        _n_ev = Q(ce, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='earnings_events'")[0][0]
        _n_ev = Q(ce, "SELECT COUNT(*) FROM earnings_events WHERE eps_surprise IS NOT NULL")[0][0] if _n_ev else 0
        if _n_ev > 1000:
            have_comp = False
            ev = Q(ce, "SELECT ticker, announce_date, eps_surprise FROM earnings_events "
                       "WHERE eps_surprise IS NOT NULL AND announce_date IS NOT NULL")
            print("  PEAD source: earnings_events.announce_date (%d rows) [LEAK-FIXED]" % len(ev))
        else:
            print("  PEAD source: earnings_surprises (fallback -- KNOWN LEAKED DATES)")
    finally:
        ce.close()

    by_tkr=defaultdict(list)
    for row in ev:
        tk=row[0]; do=nd(row[1])
        if do is None: continue
        if have_comp:
            try: raw=float(row[2])-float(row[3])
            except Exception: raw=None
        else:
            try: raw=float(row[2])
            except Exception: raw=None
        by_tkr[tk].append((do,raw))
    for tk in by_tkr: by_tkr[tk].sort()

    # PIT-trailing SUE
    events=[]
    for tk,lst in by_tkr.items():
        prior=[]
        for do,raw in lst:
            if raw is None: continue
            if len(prior)>=a.min_prior:
                sd=np.std(prior,ddof=1)
                if sd>1e-12:
                    events.append((tk,do,raw/sd))
            prior.append(raw)
    print("  SUE events: %d"%len(events))

    def fwd(tk,do,hold):
        lst=px.get(tk); idx=pxidx.get(tk)
        if not lst: return None
        pos=None
        for off in range(0,6):
            c=do+datetime.timedelta(days=off)
            if c in idx: pos=idx[c]; break
        if pos is None: return None
        e=pos+2; x=pos+2+hold
        if x>=len(lst): return None
        pe=lst[e][1]; pxx=lst[x][1]
        if pe<=0: return None
        return pxx/pe-1.0

    # split into train/test by announcement date
    train=[]; test=[]
    for tk,do,sue in events:
        r=fwd(tk,do,a.hold)
        if r is None: continue
        (train if do<split else test).append((sue,r))
    print("  train events (<%s): %d"%(a.split_date,len(train)))
    print("  TEST events (>=%s): %d  [COLD HOLDOUT]"%(a.split_date,len(test)))

    cost=a.cost_bps/10000.0
    def evaluate(rows,label):
        if len(rows)<a.min_events:
            print("  %s: too few events (%d)"%(label,len(rows))); return None
        s=[x[0] for x in rows]; r=[x[1] for x in rows]; n=len(s)
        ic=spearman(s,r)
        order=np.argsort(s); q=max(1,n//5); lo=order[:q]; hi=order[-q:]
        L=float(np.mean([r[i] for i in hi])); S=float(np.mean([r[i] for i in lo]))
        g=L-S; net=g-2*cost
        sd=math.sqrt(np.var([r[i] for i in hi])/q+np.var([r[i] for i in lo])/q)
        t=g/sd if sd>0 else None
        # decile monotonicity
        dm=[]
        for d in range(10):
            idx=order[int(d*n/10):int((d+1)*n/10)]
            if len(idx)>0: dm.append(np.mean([r[i] for i in idx]))
        ups=sum(1 for i in range(1,len(dm)) if dm[i]>=dm[i-1])
        print("  %-22s n=%-5d IC=%-8s net=%-8s t=%-6s mono=%d/%d"
              %(label,n,"%+.4f"%ic if ic is not None else "NA","%+.4f"%net,
                "%.2f"%t if t else "NA",ups,len(dm)-1))
        return {"n":n,"ic":ic,"net":net,"t":t,"mono":[ups,len(dm)-1]}

    sub("IN-SAMPLE (train) vs OUT-OF-SAMPLE (test)")
    is_res=evaluate(train,"IN-SAMPLE (train)")
    oos_res=evaluate(test,"OUT-OF-SAMPLE (test)")

    banner("VERDICT — does the brick survive COLD out-of-sample?")
    if is_res and oos_res and is_res["ic"] is not None and oos_res["ic"] is not None:
        ratio = oos_res["ic"]/is_res["ic"] if is_res["ic"]!=0 else 0
        print("  IS  IC=%+.4f  (t=%.2f)"%(is_res["ic"],is_res["t"] or 0))
        print("  OOS IC=%+.4f  (t=%.2f)"%(oos_res["ic"],oos_res["t"] or 0))
        print("  OOS/IS IC ratio: %.0f%%"%(100*ratio))
        print()
        oos_significant = (oos_res["t"] or 0)>=2.0 and (oos_res["net"] or 0)>0.003
        if oos_significant and ratio>=0.6:
            print("  >> BRICK LOCKED: edge survives cold OOS with %.0f%% of IS strength, still"%(100*ratio))
            print("     significant (t=%.2f) and net-positive. This is a real, validated signal."%oos_res["t"])
            print("     Safe to build on. Proceed to hunt for brick #2.")
        elif oos_significant:
            print("  >> SURVIVES (weaker): OOS still significant (t=%.2f) but only %.0f%% of IS."%(oos_res["t"] or 0,100*ratio))
            print("     Real but decayed — usable with conservative sizing. Some IS optimism present.")
        elif (oos_res["ic"] or 0)>0.03 and ratio>0.4:
            print("  >> PARTIAL: OOS IC positive (%.4f) but not significant (t=%.2f). Likely real but"%(oos_res["ic"],oos_res["t"] or 0))
            print("     underpowered on the holdout (n=%d). Directionally confirms; needs more OOS data."%oos_res["n"])
        else:
            print("  >> FAILED OOS: edge does NOT survive the cold holdout (OOS IC=%+.4f, t=%.2f)."%(oos_res["ic"],oos_res["t"] or 0))
            print("     The in-sample edge was substantially fitting. Do NOT build on this as-is.")
        print("\n  NOTE: OOS holdout (n=%d) is smaller than full sample, so OOS t-stats are naturally"%(oos_res["n"] if oos_res else 0))
        print("  lower. Weight the IC RATIO and sign as heavily as raw significance. A split that")
        print("  puts most dense recent data in TEST is the most honest (try --split-date 2025-01-01).")
    report={"is":is_res,"oos":oos_res,"split":a.split_date,"hold":a.hold}
    if a.out:
        with open(a.out,"a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":report},default=str)+"\n")
        print("\n  [report appended to %s]"%a.out)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
