#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — PEAD BACKTEST (Option 2: the REAL 60-day test)
================================================================================
The canonical PEAD strategy (Quantpedia / Bernard-Thomas / academic consensus):
  - sort on earnings surprise (SUE) and/or EAR, using prior-quarter data
  - ENTER on day +2 after the announcement (skip the gap & initial volatility)
  - HOLD ~20-60 trading days
  - long top-quintile surprise, short bottom-quintile
The short-horizon (1-5 day) test can't see this — the drift accelerates between
days 20 and 75. THIS script tests it properly, IF a daily price series exists.

PHASE 0 — PRICE DISCOVERY: outcomes only has h=1/3/5, so to compute 20/40/60-day
forward returns we need a daily ADJUSTED-CLOSE price table. This script scans all
DBs for one (tables with ticker + date + a price/close/adj_close column, dense
daily coverage). If found -> runs the real test. If NOT found -> reports exactly
what's missing and how to provide it (no crash, no fake result).

LEAKAGE GUARD (RULE 1): signal known at announcement; entry day +2; forward return
window [+2, +2+H] starts strictly after the signal date. Sort uses only the event's
own surprise (known at announcement). No look-ahead.

READ-ONLY. mode=ro&immutable=1.

USAGE:
  python pead_backtest_60d.py --root .
  python pead_backtest_60d.py --root . --holds 20,40,60 --cost-bps 10
  python pead_backtest_60d.py --root . --price-db prices.db --price-table daily_prices
      (force a specific price table if auto-discovery picks the wrong one)
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
def tables(c): return [r[0] for r in Q(c,"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
def has_table(c,n): return bool(Q(c,"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(n,)))
def cols_of(c,t): return [(r[1],r[2]) for r in Q(c,'PRAGMA table_info("'+t+'")')]
def require(cond,msg):
    if not cond: print("  [STOP] "+msg); return False
    return True
def nd(s):
    if s is None: return None
    s=str(s)[:10]
    try: return datetime.date.fromisoformat(s)
    except Exception: return None
def all_dbs(root):
    out=[]
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f.endswith((".db",".sqlite",".sqlite3")): out.append(os.path.join(dp,f))
    return out
def find_db(root,name):
    c=os.path.join(root,name)
    if os.path.isfile(c): return c
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        if name in fn: return os.path.join(dp,name)
    return None

PRICE_COL_HINTS=["adj_close","adjclose","adjusted_close","close","px_close","close_price","price","vwap","last"]
TICKER_HINTS=["ticker","symbol","sym"]
DATE_HINTS=["date","dt","trade_date","px_date","asof","timestamp","day"]

def discover_price_table(root, force_db=None, force_table=None):
    """Find a daily price table: ticker + date + price col, with dense daily coverage.
    Returns (dbpath, table, ticker_col, date_col, price_col, n_rows, n_tickers, dr) or None."""
    candidates=[]
    dbs=[os.path.join(root,force_db)] if force_db else all_dbs(root)
    for dbp in dbs:
        if not os.path.isfile(dbp): continue
        try: c=ro(dbp)
        except Exception: continue
        try:
            tlist=[force_table] if force_table else tables(c)
            for t in tlist:
                if not t or not has_table(c,t): continue
                cols=cols_of(c,t); names=[cn.lower() for cn,_ in cols]
                tcol=next((cn for cn,_ in cols if cn.lower() in TICKER_HINTS),None)
                dcol=next((cn for cn,_ in cols if any(h==cn.lower() or h in cn.lower() for h in DATE_HINTS)),None)
                pcol=None
                # prefer adjusted close, then close, then price
                for hint in PRICE_COL_HINTS:
                    m=next((cn for cn,ct in cols if cn.lower()==hint and (ct or "").upper() in
                            ("REAL","FLOAT","NUMERIC","DOUBLE","INTEGER","INT")),None)
                    if m: pcol=m; break
                if not pcol:
                    for hint in PRICE_COL_HINTS:
                        m=next((cn for cn,_ in cols if hint in cn.lower()),None)
                        if m: pcol=m; break
                if tcol and dcol and pcol:
                    try:
                        n=Q(c,'SELECT COUNT(*) FROM "'+t+'"')[0][0]
                        nt=Q(c,'SELECT COUNT(DISTINCT "'+tcol+'") FROM "'+t+'"')[0][0]
                        dr=Q(c,'SELECT MIN("'+dcol+'"),MAX("'+dcol+'") FROM "'+t+'"')[0]
                        nd_=Q(c,'SELECT COUNT(DISTINCT "'+dcol+'") FROM "'+t+'"')[0][0]
                        # density score: rows per ticker (want many days per ticker)
                        density=n/max(nt,1)
                        candidates.append((density,dbp,t,tcol,dcol,pcol,n,nt,dr,nd_))
                    except Exception: pass
        finally:
            c.close()
    if not candidates: return None
    # pick the densest (most rows per ticker = most daily history)
    candidates.sort(reverse=True)
    return candidates[0]

def spearman(x,y):
    n=len(x)
    if n<5: return None
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return None
    return float(np.corrcoef(rx,ry)[0,1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--holds",default="20,40,60")
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--signal",default="eps_surprise_pct")
    ap.add_argument("--price-db",default=None)
    ap.add_argument("--price-table",default=None)
    ap.add_argument("--min-events",type=int,default=30)
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    holds=[int(x) for x in a.holds.split(",")]
    banner("ML QUANT FUND — PEAD BACKTEST (Option 2: the REAL 60-day test)")
    print("Enter day +2, hold 20/40/60d, sort on surprise. Needs a daily price table.")
    print("Root:",os.path.abspath(a.root),"| numpy:",HAVE_NUMPY,"| cost/leg:",a.cost_bps,"bps")
    if not require(HAVE_NUMPY,"numpy required"): return

    banner("PHASE 0 — PRICE DISCOVERY")
    pt=discover_price_table(a.root, a.price_db, a.price_table)
    if pt is None:
        print("  No daily price table found (need: ticker + date + close/adj_close, dense daily).")
        print("\n  Scanned these DBs:")
        for d in all_dbs(a.root): print("     ",os.path.basename(d))
        print("\n  >> Option 2 CANNOT run without prices. To enable it, point me at a price table:")
        print("     python pead_backtest_60d.py --root . --price-db <file.db> --price-table <name>")
        print("  Or tell me what populates outcomes (the system computes returns from SOMETHING).")
        print("  If prices are only fetched live from an API (yfinance/polygon), say so and I'll")
        print("  build a fetch step. Until then, Option 1 (pead_backtest_v2.py) is the only test")
        print("  your stored data supports.")
        _w(a,{"price_table":None})
        return
    density,dbp,tbl,tcol,dcol,pcol,n,nt,dr,ndays=pt
    print("  FOUND price table:")
    print("     db=%s table=%s"%(os.path.basename(dbp),tbl))
    print("     ticker=%s date=%s price=%s"%(tcol,dcol,pcol))
    print("     rows=%d tickers=%d distinct_days=%d range=%s..%s (%.0f rows/ticker)"
          %(n,nt,ndays,dr[0],dr[1],density))
    if density<40:
        print("  [WARN] only ~%.0f rows/ticker — may be too sparse for 60-day holds. Proceeding,"
              " but coverage may be thin."%density)

    # load prices: ticker -> sorted list of (date, price)
    sub("Loading daily prices")
    cp=ro(dbp)
    try:
        rows=Q(cp,'SELECT "'+tcol+'","'+dcol+'","'+pcol+'" FROM "'+tbl+'" '
                  'WHERE "'+pcol+'" IS NOT NULL')
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
    # index: ticker -> {date: position} for fast forward lookup
    pxidx={tk:{d:i for i,(d,_) in enumerate(lst)} for tk,lst in px.items()}
    print("  loaded prices for %d tickers"%len(px))

    # load earnings events
    earnp=find_db(a.root,"earnings.db")
    if not require(earnp,"earnings.db not found"): return
    ce=ro(earnp)
    try:
        if not require(has_table(ce,"earnings_surprises"),"no earnings_surprises"): return
        if not require(a.signal in [c for c,_ in cols_of(ce,"earnings_surprises")],"no col "+a.signal): return
        ev=Q(ce,"SELECT ticker,report_date,"+a.signal+" FROM earnings_surprises "
                "WHERE report_date IS NOT NULL AND "+a.signal+" IS NOT NULL")
    finally:
        ce.close()
    events=[(tk,nd(rd),sig) for tk,rd,sig in ev if nd(rd) is not None]
    print("  earnings events with signal: %d"%len(events))

    def fwd_return(tk,do,entry_offset,hold):
        """Return from the (entry_offset)-th trading day on/after `do` to (entry_offset+hold).
        entry_offset=2 => enter day +2. Uses positional trading-day indexing on the price series."""
        lst=px.get(tk); idx=pxidx.get(tk)
        if not lst: return None
        # find first trading day on/after announcement
        pos=None
        for off in range(0,6):
            c=do+datetime.timedelta(days=off)
            if c in idx: pos=idx[c]; break
        if pos is None: return None
        entry=pos+entry_offset; exit_=pos+entry_offset+hold
        if exit_>=len(lst): return None
        p_entry=lst[entry][1]; p_exit=lst[exit_][1]
        if p_entry<=0: return None
        return p_exit/p_entry - 1.0

    cost=a.cost_bps/10000.0
    report={"price_table":{"db":os.path.basename(dbp),"table":tbl},"holds":{}}
    for H in holds:
        sub("HOLD = %d trading days, entry = day +2 (canonical PEAD)"%H)
        recs=[]
        for tk,do,sig in events:
            r=fwd_return(tk,do,2,H)  # enter day +2, hold H days
            if r is not None: recs.append((sig,r))
        if len(recs)<a.min_events*3:
            print("  only %d events with %d-day forward prices — insufficient (need >=%d)"
                  %(len(recs),H,a.min_events*3)); continue
        sigs=[r[0] for r in recs]; rets=[r[1] for r in recs]; n=len(sigs)
        # IC of surprise vs forward return (event-conditioned)
        ic=spearman(sigs,rets)
        # quintile L/S
        order=np.argsort(sigs); q=n//5
        lo=order[:q]; hi=order[-q:]
        L=np.mean([rets[i] for i in hi]); S=np.mean([rets[i] for i in lo])
        g=L-S; net=g-2*cost
        sd=math.sqrt(np.var([rets[i] for i in hi])/q+np.var([rets[i] for i in lo])/q)
        t=g/sd if sd>0 else None
        print("  events=%d  IC=%s"%(n,"%+.4f"%ic if ic is not None else "NA"))
        print("  quintile L/S: long=%+.4f short=%+.4f | GROSS=%+.4f NET=%+.4f t=%s"
              %(L,S,g,net,"%.2f"%t if t else "NA"))
        # annualize: hold H days, ~quarterly events -> rough per-trade -> annual
        print("  (per-trade %d-day spread; not annualized — earnings are quarterly per name)"%H)
        report["holds"][H]={"events":n,"ic":ic,"long":L,"short":S,"gross":g,"net":net,"t":t}

    banner("VERDICT — does the REAL (20-60 day) PEAD survive in your universe?")
    good=[(H,r) for H,r in report["holds"].items()
          if r.get("net",0)>0.003 and r.get("t") is not None and abs(r["t"])>=2.0]
    if good:
        for H,r in good:
            print("  [TRADEABLE] %dd hold: NET=%+.4f IC=%+.4f t=%.2f — survives cost AND significant"
                  %(H,r["net"],r["ic"] or 0,r["t"]))
        print("\n  This is the real PEAD edge at the horizon it actually lives at. Next: walk-forward")
        print("  it, check capacity in liquid names, confirm across cost levels, THEN consider sizing.")
    else:
        print("  At the canonical 20-60 day horizon, no hold is both positive-net and significant.")
        print("  Combined with the short-horizon null, this says PEAD is not a tradeable edge in")
        print("  your universe — consistent with the literature that it's arbitraged out of liquid")
        print("  large-caps post-2006. An honest, complete answer.")
    _w(a,report)

def _w(a,report):
    if not a.out: return
    path=a.out
    if os.path.isdir(path) or path.endswith("/"): path=os.path.join(path,"pead_60d.json")
    with open(path,"a") as f:
        f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":report},default=str)+"\n")
    print("\n  [report appended to %s]"%path)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
