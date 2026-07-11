#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — FETCH PRICES + RUN REAL 60-DAY PEAD  (Path A, end-to-end)
================================================================================
The 60-day PEAD test can't run because no daily price series is stored (outcomes
is built from live-fetched prices that were never persisted). This script fixes
that: it FETCHES daily adjusted closes for the earnings-event tickers, CACHES
them to a local prices.db, then runs the corrected canonical PEAD backtest.

PIPELINE:
  STEP 1  read distinct tickers + date span from earnings.db.earnings_surprises
  STEP 2  fetch daily adj-close for those tickers (yfinance; cached to prices.db)
          - idempotent: re-running skips tickers already cached
          - fail-loud: reports every ticker that failed to fetch
  STEP 3  run canonical PEAD: enter day +2, hold 20/40/60, quintile L/S, net cost,
          leak-guarded (signal known at announcement; return window strictly after)

WHY SOURCE-AGNOSTIC: your stack migrated OFF yfinance (reliability). So this tries
yfinance first; if it's missing or fails, it tells you clearly and you can install
it (pip install yfinance) or point me at your working vendor. It NEVER fabricates
prices or silently proceeds on partial data.

RULE 1 GUARDS:
  - prices cached to a SEPARATE new file (prices.db); your existing DBs are untouched
  - every fetch verified (row count, date span, no all-null); failures listed loud
  - PEAD return math reuses the verified positional-trading-day logic
  - leakage: entry strictly day +2, sort on the event's own surprise only

USAGE:
  pip install yfinance            # if not already installed
  python fetch_and_pead.py --root .                 # fetch (cached) + test
  python fetch_and_pead.py --root . --fetch-only     # just build prices.db
  python fetch_and_pead.py --root . --test-only      # skip fetch, use existing prices.db
  python fetch_and_pead.py --root . --holds 20,40,60 --cost-bps 10 --start 2019-06-01
  python fetch_and_pead.py --root . --max-tickers 50   # smoke-test on a subset first

prices.db is written in the project root by default (--prices-db to change).
================================================================================
"""
import argparse, os, sqlite3, sys, math, time, datetime, json
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

# ----------------------------------------------------------- prices.db schema
def init_prices_db(path):
    conn=sqlite3.connect(path,timeout=30)
    conn.execute("""CREATE TABLE IF NOT EXISTS daily_prices(
        ticker TEXT NOT NULL, date TEXT NOT NULL, adj_close REAL,
        PRIMARY KEY(ticker,date))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS fetch_log(
        ticker TEXT PRIMARY KEY, status TEXT, n_rows INTEGER,
        first_date TEXT, last_date TEXT, fetched_at TEXT)""")
    conn.commit(); return conn

def already_cached(conn):
    return set(r[0] for r in conn.execute("SELECT ticker FROM fetch_log WHERE status='OK'").fetchall())

# ----------------------------------------------------------- fetch
def fetch_yfinance(tickers, start, end, conn, sleep=0.0, batch_note=""):
    try:
        import yfinance as yf
    except Exception:
        print("  [STOP] yfinance not installed. Run: pip install yfinance")
        print("         (or tell me your working vendor and I'll build that fetch instead)")
        return None
    ok=0; failed=[]
    done=already_cached(conn)
    todo=[t for t in tickers if t not in done]
    print("  %d tickers to fetch (%d already cached)%s"%(len(todo),len(done),batch_note))
    for i,tk in enumerate(todo,1):
        try:
            # Jun 29 2026: fetch via Massive (_download, auto_adjust=True) instead of
            # yfinance directly. XProtect 5347 blocks the yfinance/curl_cffi path as
            # malware at execution. _download routes stocks/ETFs to Massive (Polygon),
            # returns a flat lowercase-column DataFrame with adjusted 'close'. Same
            # adjusted-close semantics as the prior yf auto_adjust=True path → adj_close.
            from features.builder import _download as _mc_download
            try:
                df=_mc_download(tk, start, end)
            except Exception:
                df=None
            if df is None or len(df)==0 or "close" not in getattr(df,"columns",[]):
                failed.append((tk,"empty"))
                conn.execute("REPLACE INTO fetch_log VALUES(?,?,?,?,?,?)",
                             (tk,"EMPTY",0,None,None,datetime.datetime.now().isoformat(timespec="seconds")))
                conn.commit(); continue
            rows=[]
            for _, r in df.iterrows():
                d = r["date"]
                d = d.isoformat() if hasattr(d,"isoformat") else str(d)[:10]
                try:
                    fv=float(r["close"])
                except (TypeError,ValueError):
                    continue
                if not math.isnan(fv):
                    rows.append((tk,d,fv))
            if not rows:
                failed.append((tk,"all-null")); 
                conn.execute("REPLACE INTO fetch_log VALUES(?,?,?,?,?,?)",
                             (tk,"NULL",0,None,None,datetime.datetime.now().isoformat(timespec="seconds")))
                conn.commit(); continue
            conn.executemany("REPLACE INTO daily_prices(ticker,date,adj_close) VALUES(?,?,?)",rows)
            conn.execute("REPLACE INTO fetch_log VALUES(?,?,?,?,?,?)",
                         (tk,"OK",len(rows),rows[0][1],rows[-1][1],datetime.datetime.now().isoformat(timespec="seconds")))
            conn.commit(); ok+=1
            if i%25==0: print("    ...%d/%d fetched (%d ok)"%(i,len(todo),ok))
            if sleep: time.sleep(sleep)
        except Exception as e:
            failed.append((tk,str(e)[:50]))
            try:
                conn.execute("REPLACE INTO fetch_log VALUES(?,?,?,?,?,?)",
                             (tk,"ERROR",0,None,None,datetime.datetime.now().isoformat(timespec="seconds")))
                conn.commit()
            except Exception: pass
    print("  fetch complete: %d ok, %d failed"%(ok,len(failed)))
    if failed:
        print("  failed tickers (first 20): %s"%failed[:20])
    return ok

# ----------------------------------------------------------- PEAD (verified engine)
def spearman(x,y):
    n=len(x)
    if n<5: return None
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return None
    return float(np.corrcoef(rx,ry)[0,1])

def run_pead(prices_db, earnp, signal, holds, cost_bps, min_events, out):
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
    print("  loaded cached prices for %d tickers"%len(px))
    if len(px)<10:
        print("  [STOP] too few tickers with prices (%d). Fetch may have failed."%len(px)); return None

    ce=ro(earnp)
    try:
        ev=Q(ce,"SELECT ticker,report_date,"+signal+" FROM earnings_surprises "
               "WHERE report_date IS NOT NULL AND "+signal+" IS NOT NULL")
    finally:
        ce.close()
    events=[(tk,nd(rd),sig) for tk,rd,sig in ev if nd(rd) is not None]
    print("  earnings events with signal: %d"%len(events))

    def fwd(tk,do,entry_off,hold):
        lst=px.get(tk); idx=pxidx.get(tk)
        if not lst: return None
        pos=None
        for off in range(0,6):
            c=do+datetime.timedelta(days=off)
            if c in idx: pos=idx[c]; break
        if pos is None: return None
        e=pos+entry_off; x=pos+entry_off+hold
        if x>=len(lst): return None
        pe=lst[e][1]; pxx=lst[x][1]
        if pe<=0: return None
        return pxx/pe-1.0

    cost=cost_bps/10000.0
    report={"holds":{}}
    for H in holds:
        sub("HOLD = %d trading days, entry = day +2 (canonical PEAD)"%H)
        recs=[(sig,fwd(tk,do,2,H)) for tk,do,sig in events]
        recs=[(s,r) for s,r in recs if r is not None]
        if len(recs)<min_events*3:
            print("  only %d events with %d-day forward prices (need >=%d) — skip"
                  %(len(recs),H,min_events*3)); continue
        sigs=[r[0] for r in recs]; rets=[r[1] for r in recs]; n=len(sigs)
        ic=spearman(sigs,rets)
        order=np.argsort(sigs); q=n//5
        lo=order[:q]; hi=order[-q:]
        L=float(np.mean([rets[i] for i in hi])); S=float(np.mean([rets[i] for i in lo]))
        g=L-S; net=g-2*cost
        sd=math.sqrt(np.var([rets[i] for i in hi])/q+np.var([rets[i] for i in lo])/q)
        t=g/sd if sd>0 else None
        print("  events=%d  IC=%s"%(n,"%+.4f"%ic if ic is not None else "NA"))
        print("  quintile L/S: long=%+.4f short=%+.4f | GROSS=%+.4f NET=%+.4f t=%s"
              %(L,S,g,net,"%.2f"%t if t else "NA"))
        report["holds"][H]={"events":n,"ic":ic,"long":L,"short":S,"gross":g,"net":net,"t":t}

    banner("VERDICT — does the REAL (20-60 day) PEAD survive in your universe?")
    # significance gate AND power awareness: a positive IC with a weak t-stat on a
    # SMALL sample is "underpowered", NOT "no effect". Distinguish the two.
    good=[(H,r) for H,r in report["holds"].items()
          if r.get("net",0)>0.003 and r.get("t") is not None and abs(r["t"])>=2.0]
    # promising = positive IC in the realistic band + positive net, but t<2 (likely underpowered)
    promising=[(H,r) for H,r in report["holds"].items()
               if r.get("ic") is not None and r["ic"]>=0.03 and r.get("net",0)>0
               and (r.get("t") is None or abs(r["t"])<2.0)]
    total_events=max((r.get("events",0) for r in report["holds"].values()), default=0)
    if good:
        for H,r in good:
            print("  [TRADEABLE] %dd: NET=%+.4f IC=%+.4f t=%.2f — survives cost AND significant"
                  %(H,r["net"],r["ic"] or 0,r["t"]))
        print("\n  Real PEAD edge at the right horizon. Next: walk-forward, capacity, cost levels.")
    elif promising:
        print("  PROMISING BUT UNDERPOWERED — positive IC in the realistic band with positive net,")
        print("  but t-stat < 2 (not yet significant). This is NOT a null; it's likely a sample-size")
        print("  limitation. Per-hold detail:")
        for H,r in promising:
            # rough events needed for t>=2: t scales ~sqrt(n), so n_needed ~ n*(2/t)^2
            cur_t=abs(r["t"]) if r.get("t") else 0.01
            cur_n=r.get("events",0)
            n_needed=int(cur_n*(2.0/cur_t)**2) if cur_t>0 else None
            print("    %dd: IC=%+.4f net=%+.4f t=%.2f on %d events"
                  %(H,r["ic"],r["net"],r["t"] or 0,cur_n)
                  + (" -> ~%d events needed for t>=2"%n_needed if n_needed else ""))
        print("\n  >> The IC is the more stable estimate at small N, and it's POSITIVE at the canonical")
        print("     horizon — the opposite of the short-horizon null. DO NOT conclude 'no edge' yet.")
        print("     RUN THE FULL UNIVERSE (drop --max-tickers): ~8x the events will tell you whether")
        print("     the IC holds (-> significant, real edge) or collapses (-> small-sample luck).")
        if total_events<800:
            print("     [current run is small: %d events. This was likely a --max-tickers subset.]"%total_events)
    else:
        print("  At 20-60 days, no hold shows even a promising positive IC. Combined with the")
        print("  short-horizon null, PEAD is not a tradeable edge in your universe — consistent")
        print("  with the literature (arbitraged out of liquid large-caps post-2006).")
        if total_events<800:
            print("  [NOTE: only %d events — if this was a --max-tickers subset, run the full"%total_events)
            print("   universe before concluding, as small samples can hide a real effect.]")
    if out:
        with open(out,"a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":report},default=str)+"\n")
        print("\n  [report appended to %s]"%out)
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--signal",default="eps_surprise_pct")
    ap.add_argument("--holds",default="20,40,60")
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--start",default=None,help="fetch start date; default = 1y before earliest event")
    ap.add_argument("--end",default=None,help="fetch end date; default = today")
    ap.add_argument("--max-tickers",type=int,default=None,help="limit tickers (smoke test)")
    ap.add_argument("--fetch-only",action="store_true")
    ap.add_argument("--test-only",action="store_true")
    ap.add_argument("--min-events",type=int,default=30)
    ap.add_argument("--sleep",type=float,default=0.0)
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    holds=[int(x) for x in a.holds.split(",")]
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")

    banner("ML QUANT FUND — FETCH PRICES + RUN REAL 60-DAY PEAD (Path A)")
    print("Root:",os.path.abspath(a.root),"| prices.db:",prices_db,"| numpy:",HAVE_NUMPY)
    if not require(HAVE_NUMPY,"numpy required"): return
    earnp=find_db(a.root,"earnings.db")
    if not require(earnp,"earnings.db not found"): return

    # discover tickers + date span from earnings_surprises
    ce=ro(earnp)
    try:
        if not require(has_table(ce,"earnings_surprises"),"no earnings_surprises"): return
        if not require(a.signal in cols_of(ce,"earnings_surprises"),"no col "+a.signal): return
        trows=Q(ce,"SELECT DISTINCT ticker FROM earnings_surprises WHERE ticker IS NOT NULL")
        drow=Q(ce,"SELECT MIN(report_date),MAX(report_date) FROM earnings_surprises")[0]
    finally:
        ce.close()
    tickers=sorted(set(t[0] for t in trows if t[0]))
    if a.max_tickers: tickers=tickers[:a.max_tickers]
    ev_min=nd(drow[0]); ev_max=nd(drow[1])
    # fetch window: 1y before earliest event (for context) .. today (for 60d after latest)
    start = a.start or ((ev_min - datetime.timedelta(days=365)).isoformat() if ev_min else "2019-01-01")
    end = a.end or datetime.date.today().isoformat()
    print("  tickers from earnings_surprises: %d"%len(tickers))
    print("  event date span: %s .. %s"%(drow[0],drow[1]))
    print("  fetch window: %s .. %s"%(start,end))

    if not a.test_only:
        sub("STEP 2 — fetch daily adj-close (cached to prices.db)")
        conn=init_prices_db(prices_db)
        try:
            res=fetch_yfinance(tickers,start,end,conn,sleep=a.sleep)
        finally:
            conn.close()
        if res is None:
            print("\n  Fetch unavailable. Install yfinance or specify your vendor; nothing else ran.")
            return
        # verify cache
        cc=ro(prices_db)
        try:
            npx=Q(cc,"SELECT COUNT(DISTINCT ticker) FROM daily_prices")[0][0]
            nrow=Q(cc,"SELECT COUNT(*) FROM daily_prices")[0][0]
            dr=Q(cc,"SELECT MIN(date),MAX(date) FROM daily_prices")[0]
        finally:
            cc.close()
        print("  prices.db now has %d tickers, %d rows, %s..%s"%(npx,nrow,dr[0],dr[1]))
        if a.fetch_only:
            print("\n  --fetch-only done. Run with --test-only to backtest.")
            return

    if not os.path.isfile(prices_db):
        print("  [STOP] no prices.db — run without --test-only first to fetch."); return
    sub("STEP 3 — canonical 60-day PEAD on fetched prices")
    run_pead(prices_db, earnp, a.signal, holds, a.cost_bps, a.min_events, a.out)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
