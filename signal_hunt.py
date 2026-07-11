#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — SECOND-SIGNAL HUNT  (is there an uncorrelated brick #2?)
================================================================================
PEAD is one real brick. The combine-little-signals thesis needs MORE bricks that
are UNCORRELATED with PEAD (correlated signals add no breadth). This hunts for a
second brick among NON-earnings signals, and — critically — measures each
candidate's correlation to the PEAD signal, not just its standalone IC.

CANDIDATES (all price-derived, computable from prices.db; all NON-earnings so
they have a chance of being uncorrelated with PEAD):
  * REVERSAL_5    : -1 * past 5-day return    (short-term mean reversion)
  * REVERSAL_1    : -1 * past 1-day return
  * MOMENTUM_60   : past 60-day return (skip last 5d)   (intermediate momentum)
  * MOMENTUM_120  : past 120-day return (skip last 20d)
  * VOL_20        : -1 * 20-day realized vol  (low-vol effect)
  * (if prediction_features exists) its cross-sectional columns, each tested raw

For EACH candidate, on a common rebalance schedule (we sample dates and rank the
cross-section), measure:
  * standalone cross-sectional IC vs forward h-day return
  * CORRELATION of this signal's per-date ranks with the PEAD/SUE signal's ranks
    on overlapping names/dates  -> the key "is it a NEW brick?" number

A good brick #2 = positive standalone IC AND low correlation (|rho|<0.3) to PEAD.

NOTE ON HORIZON: PEAD lives at 40d. For a fair "can these combine" test we measure
these candidates at the SAME forward horizon (default 40d) and also a short one.
But these price signals are continuous (every day), unlike PEAD (event-driven), so
the correlation is measured where they coexist.

RULE 1: all signals use ONLY past data (returns/vol computed strictly before the
prediction date). Forward return strictly after. No look-ahead.

READ-ONLY. mode=ro&immutable=1. No network.

USAGE:
  python signal_hunt.py --root .
  python signal_hunt.py --root . --hold 40 --rebal 5   (rebalance every 5 trading days)
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
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--rebal",type=int,default=5,help="rebalance every N trading days")
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--start",default="2022-01-01",help="only use dates after this (recent regime)")
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    start=nd(a.start)
    banner("ML QUANT FUND — SECOND-SIGNAL HUNT (uncorrelated brick #2?)")
    print("Tests non-earnings price signals; measures IC AND correlation to PEAD. (offline)")
    print("hold=%d rebal=%dd start>=%s"%(a.hold,a.rebal,a.start))
    if not require(HAVE_NUMPY,"numpy required"): return
    if not require(os.path.isfile(prices_db),"prices.db not found"): return

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
    # position index per ticker
    pos_of={tk:{d:i for i,(d,_) in enumerate(lst)} for tk,lst in px.items()}
    print("  prices loaded for %d tickers"%len(px))

    # build a common set of rebalance dates from a liquid reference (use union of dates)
    all_dates=sorted(set(d for lst in px.values() for d,_ in lst))
    all_dates=[d for d in all_dates if (start is None or d>=start)]
    rebal_dates=all_dates[::a.rebal]
    print("  rebalance dates: %d (every %d trading days, from %s)"%(len(rebal_dates),a.rebal,a.start))

    # signal definitions: function(ticker, date_pos, lst) -> signal value (uses only past)
    def ret_back(lst,i,k):
        if i-k<0: return None
        p0=lst[i-k][1]; p1=lst[i][1]
        return (p1/p0-1.0) if p0>0 else None
    def vol_back(lst,i,k):
        if i-k<0: return None
        rets=[]
        for j in range(i-k+1,i+1):
            if lst[j-1][1]>0: rets.append(lst[j][1]/lst[j-1][1]-1.0)
        return float(np.std(rets)) if len(rets)>2 else None

    SIGNALS={
        "REVERSAL_5":  lambda lst,i: (-ret_back(lst,i,5))  if ret_back(lst,i,5)  is not None else None,
        "REVERSAL_1":  lambda lst,i: (-ret_back(lst,i,1))  if ret_back(lst,i,1)  is not None else None,
        "MOMENTUM_60": lambda lst,i: ret_back(lst,i-5,55)  if i-60>=0 and ret_back(lst,i-5,55) is not None else None,
        "MOMENTUM_120":lambda lst,i: ret_back(lst,i-20,100) if i-120>=0 and ret_back(lst,i-20,100) is not None else None,
        "LOWVOL_20":   lambda lst,i: (-vol_back(lst,i,20)) if vol_back(lst,i,20) is not None else None,
    }

    def fwd(lst,i,hold):
        x=i+hold
        if x>=len(lst): return None
        p0=lst[i][1]; p1=lst[x][1]
        return (p1/p0-1.0) if p0>0 else None

    # For each signal: accumulate per-date cross-sections, compute daily IC, average.
    # Also store per-(date,ticker) signal rank for correlation with PEAD later.
    sub("STANDALONE IC of each non-earnings signal (hold=%d)"%a.hold)
    results={}
    signal_panel=defaultdict(dict)  # signame -> {(date,ticker): value}
    for signame,fn in SIGNALS.items():
        daily_ics=[]
        for rd in rebal_dates:
            xs=[]; ys=[]; names=[]
            for tk,lst in px.items():
                i=pos_of[tk].get(rd)
                if i is None: continue
                sv=fn(lst,i)
                fr=fwd(lst,i,a.hold)
                if sv is None or fr is None: continue
                xs.append(sv); ys.append(fr); names.append(tk)
                signal_panel[signame][(rd,tk)]=sv
            if len(xs)>=a.min_names:
                ic=spearman(xs,ys)
                if ic is not None: daily_ics.append(ic)
        if daily_ics:
            arr=np.array(daily_ics); m=arr.mean(); sd=arr.std()
            t=m/(sd/math.sqrt(len(arr))) if sd>0 else None
            print("  %-13s mean daily IC=%+.4f  std=%.4f  n_dates=%d  t=%s"
                  %(signame,m,sd,len(arr),"%.2f"%t if t else "NA"))
            results[signame]={"ic":m,"t":t,"n":len(arr)}
        else:
            print("  %-13s no usable dates"%signame)

    # ---- PEAD signal panel for correlation: SUE per (event_date, ticker) ----
    sub("CORRELATION to PEAD — is each candidate a NEW brick or a duplicate?")
    earnp=find_db(a.root,"earnings.db")
    pead_panel={}
    if earnp and os.path.isfile(earnp):
        ce=ro(earnp)
        try:
            cols=cols_of(ce,"earnings_surprises")
            have_comp="eps_actual" in cols and "eps_estimate" in cols
            sel="ticker,report_date"+(",eps_actual,eps_estimate" if have_comp else ",eps_surprise_pct")
            ev=Q(ce,"SELECT "+sel+" FROM earnings_surprises WHERE report_date IS NOT NULL")
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
        for tk,lst in by_tkr.items():
            prior=[]
            for do,raw in lst:
                if raw is None: continue
                if len(prior)>=4:
                    sd=np.std(prior,ddof=1)
                    if sd>1e-12: pead_panel[(do,tk)]=raw/sd
                prior.append(raw)
        print("  PEAD/SUE panel: %d (date,ticker) points"%len(pead_panel))

        # To correlate a CONTINUOUS daily signal with the EVENT-based PEAD, align each
        # earnings event to the candidate signal value AT the event date (nearest rebal date).
        for signame in SIGNALS:
            pairs=[]
            for (edate,tk),sue in pead_panel.items():
                # find candidate signal for this ticker near the event date
                lst=px.get(tk); i=None
                if lst and tk in pos_of:
                    # nearest available date on/before event
                    for off in range(0,6):
                        c=edate-datetime.timedelta(days=off)
                        if c in pos_of[tk]: i=pos_of[tk][c]; break
                if i is None: continue
                sv=SIGNALS[signame](lst,i)
                if sv is None: continue
                pairs.append((sue,sv))
            if len(pairs)>=30:
                rho=spearman([p[0] for p in pairs],[p[1] for p in pairs])
                tag = "NEW BRICK (low corr)" if rho is not None and abs(rho)<0.3 else \
                      ("CORRELATED (redundant)" if rho is not None and abs(rho)>=0.3 else "?")
                print("  %-13s corr-to-PEAD rho=%+.3f  (n=%d)  -> %s"
                      %(signame,rho if rho is not None else 0,len(pairs),tag))
                if signame in results: results[signame]["corr_pead"]=rho
            else:
                print("  %-13s insufficient overlap with PEAD events"%signame)
    else:
        print("  earnings.db not found — cannot compute PEAD correlation")

    # ---- verdict ----
    banner("VERDICT — did we find an uncorrelated brick #2?")
    bricks=[]
    for sn,r in results.items():
        ic=r.get("ic",0); t=r.get("t",0) or 0; corr=r.get("corr_pead")
        is_signal = abs(ic)>=0.02 and abs(t)>=2.0
        is_uncorr = corr is None or abs(corr)<0.3
        if is_signal and is_uncorr:
            bricks.append((sn,ic,t,corr))
    if bricks:
        print("  CANDIDATE BRICK(S) FOUND — significant standalone IC AND uncorrelated to PEAD:")
        for sn,ic,t,corr in bricks:
            print("    %-13s IC=%+.4f t=%.2f corr-to-PEAD=%s"
                  %(sn,ic,t,"%+.3f"%corr if corr is not None else "NA"))
        print("\n  >> The combine-signals thesis is LIVE: you have PEAD + at least one uncorrelated")
        print("     signal. Build the combination harness to stack them and measure breadth gain.")
    else:
        sig_but_corr=[(sn,r) for sn,r in results.items() if abs(r.get('ic',0))>=0.02 and abs(r.get('t',0) or 0)>=2.0]
        if sig_but_corr:
            print("  Found significant signals, but they're CORRELATED with PEAD (redundant):")
            for sn,r in sig_but_corr:
                print("    %-13s IC=%+.4f corr=%+.3f"%(sn,r['ic'],r.get('corr_pead') or 0))
            print("  These don't add breadth. Need signals from a DIFFERENT mechanism.")
        else:
            print("  No non-earnings price signal shows a significant standalone IC at hold=%d."%a.hold)
            print("  In THIS universe at THIS horizon, simple price signals (reversal/momentum/vol)")
            print("  don't carry edge. PEAD may be the lonely brick — OR these signals work at")
            print("  different horizons (try --hold 5 or --hold 20) or need different construction.")
        print("\n  Next move: try other horizons (--hold 5/20), or non-price signals (options flow,")
        print("  short interest, analyst revisions) if available in your DBs.")
    if a.out:
        with open(a.out,"a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":results},default=str)+"\n")
        print("\n  [report appended to %s]"%a.out)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
