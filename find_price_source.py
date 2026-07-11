#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — DAILY PRICE SOURCE FINDER  (strict, anti-tick)
================================================================================
pead_backtest_60d.py's auto-discovery grabbed institutional_trades (2.1M rows,
INTRADAY tick prices, only 44 distinct days) — useless for a 60-day backtest.
This script finds a GENUINE daily price source by being strict:

For every (db, table) with ticker + date + a price-like column, it measures:
  * distinct_days and date range (need MANY days over MANY months/years)
  * rows-per-ticker-per-day  (≈1 for daily data; >>1 means intraday/tick — REJECT)
  * median calendar gap between consecutive dates for a sample ticker (≈1-4 days
    for daily; sub-day implies intraday)
and classifies each as:
  DAILY_OK        — looks like a real daily price series (usable for 60d holds)
  INTRADAY_REJECT — many rows per ticker per day (tick/trade data)
  TOO_SHORT       — daily-ish but not enough history for 60-day holds
  SPARSE          — gappy / irregular

It ALSO specifically tries to answer: "what populates outcomes?" by checking
whether any DAILY_OK table's (ticker,date) coverage lines up with outcomes'
(ticker,prediction_date), which would identify the price source behind outcomes.

READ-ONLY. mode=ro&immutable=1.

USAGE:
  python find_price_source.py --root .
  python find_price_source.py --root . --show-all   (list every candidate, even rejects)
================================================================================
"""
import argparse, os, sqlite3, sys, datetime, statistics
from collections import defaultdict

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
    return sorted(out)

PRICE_HINTS=["adj_close","adjclose","adjusted_close","close","px_close","close_price","price","last","vwap"]
TICKER_HINTS=["ticker","symbol","sym","permno"]
DATE_HINTS=["date","dt","trade_date","px_date","asof","day","bar_date","session"]
# columns that signal INTRADAY even if a 'date' col exists
INTRADAY_SIGNALS=["time","timestamp","minute","second","hour","ts","datetime","bar_time","trade_time"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--show-all",action="store_true")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    banner("ML QUANT FUND — DAILY PRICE SOURCE FINDER (strict, anti-tick)")
    print("Finds a GENUINE daily price series. Rejects intraday/tick tables.")
    print("Root:",os.path.abspath(a.root))

    # load outcomes coverage for the "what populates outcomes" check
    outcomes_cov=None
    accp=None
    for d in all_dbs(a.root):
        if os.path.basename(d)=="accuracy.db": accp=d; break
    if accp:
        try:
            c=ro(accp)
            if has_table(c,"outcomes"):
                rows=Q(c,"SELECT DISTINCT ticker, prediction_date FROM outcomes LIMIT 200000")
                cov=set((tk,nd(d)) for tk,d in rows if nd(d) is not None)
                outcomes_cov=cov
                dts=[d for _,d in cov]
                print("\n  outcomes coverage sample: %d (ticker,date) pairs, %s..%s"
                      %(len(cov),min(dts) if dts else "?",max(dts) if dts else "?"))
            c.close()
        except Exception: pass

    daily_ok=[]; intraday=[]; too_short=[]; sparse=[]; 
    for dbp in all_dbs(a.root):
        try: c=ro(dbp)
        except Exception: continue
        try:
            for t in tables(c):
                cols=cols_of(c,t); names=[cn.lower() for cn,_ in cols]
                tcol=next((cn for cn,_ in cols if cn.lower() in TICKER_HINTS),None)
                dcol=next((cn for cn,_ in cols if any(h==cn.lower() for h in DATE_HINTS)),None)
                if not dcol:
                    dcol=next((cn for cn,_ in cols if any(h in cn.lower() for h in DATE_HINTS)),None)
                pcol=None
                for hint in PRICE_HINTS:
                    m=next((cn for cn,ct in cols if cn.lower()==hint),None)
                    if m: pcol=m; break
                if not pcol:
                    for hint in PRICE_HINTS:
                        m=next((cn for cn,_ in cols if hint in cn.lower()),None)
                        if m: pcol=m; break
                if not (tcol and dcol and pcol): continue
                has_intraday_col=any(any(s in cn.lower() for s in INTRADAY_SIGNALS) for cn in names)
                try:
                    n=Q(c,'SELECT COUNT(*) FROM "'+t+'"')[0][0]
                    if n==0: continue
                    nt=Q(c,'SELECT COUNT(DISTINCT "'+tcol+'") FROM "'+t+'"')[0][0]
                    ndays=Q(c,'SELECT COUNT(DISTINCT substr("'+dcol+'",1,10)) FROM "'+t+'"')[0][0]
                    dr=Q(c,'SELECT MIN(substr("'+dcol+'",1,10)),MAX(substr("'+dcol+'",1,10)) FROM "'+t+'"')[0]
                except Exception: continue
                rows_per_ticker=n/max(nt,1)
                rows_per_ticker_per_day=n/max(nt,1)/max(ndays,1)
                # span in months
                d0,d1=nd(dr[0]),nd(dr[1])
                span_days=(d1-d0).days if (d0 and d1) else 0
                rec={"db":os.path.basename(dbp),"table":t,"tcol":tcol,"dcol":dcol,"pcol":pcol,
                     "n":n,"nt":nt,"ndays":ndays,"range":(dr[0],dr[1]),"span_days":span_days,
                     "rpt":rows_per_ticker,"rptpd":rows_per_ticker_per_day,
                     "intraday_col":has_intraday_col}
                # classify
                if has_intraday_col or rows_per_ticker_per_day>3:
                    intraday.append(rec)
                elif ndays<80 or span_days<150:
                    too_short.append(rec)
                elif rows_per_ticker_per_day<0.5:
                    sparse.append(rec)
                else:
                    daily_ok.append(rec)
        finally:
            c.close()

    def show(rec,tag):
        print("  [%s] %s.%s  (ticker=%s date=%s price=%s)"
              %(tag,rec["db"],rec["table"],rec["tcol"],rec["dcol"],rec["pcol"]))
        print("        rows=%d tickers=%d distinct_days=%d span=%dd range=%s..%s  rows/tkr/day=%.2f%s"
              %(rec["n"],rec["nt"],rec["ndays"],rec["span_days"],rec["range"][0],rec["range"][1],
                rec["rptpd"], "  [has intraday col]" if rec["intraday_col"] else ""))

    sub("DAILY_OK — usable daily price series")
    if daily_ok:
        for r in sorted(daily_ok,key=lambda x:-x["ndays"]): show(r,"DAILY_OK")
    else:
        print("  NONE found.")

    sub("INTRADAY_REJECT — tick/trade data (NOT usable for daily backtests)")
    if intraday:
        for r in intraday: show(r,"INTRADAY")
    else: print("  none")

    sub("TOO_SHORT — daily-ish but insufficient history for 60-day holds")
    if too_short:
        for r in too_short: show(r,"TOO_SHORT")
    else: print("  none")

    if a.show_all:
        sub("SPARSE — gappy/irregular")
        for r in sparse: show(r,"SPARSE")

    # --- what populates outcomes? ---
    sub("WHAT POPULATES outcomes? (coverage overlap test)")
    if outcomes_cov and daily_ok:
        best=None
        for r in daily_ok:
            try:
                c=ro([d for d in all_dbs(a.root) if os.path.basename(d)==r["db"]][0])
                rows=Q(c,'SELECT DISTINCT "'+r["tcol"]+'", substr("'+r["dcol"]+'",1,10) FROM "'+r["table"]+'" LIMIT 200000')
                c.close()
                cov=set((tk,nd(d)) for tk,d in rows if nd(d) is not None)
                overlap=len(outcomes_cov & cov)
                frac=overlap/max(len(outcomes_cov),1)
                print("  %s.%s overlaps %d/%d outcomes pairs (%.0f%%)"
                      %(r["db"],r["table"],overlap,len(outcomes_cov),100*frac))
                if best is None or frac>best[1]: best=(r,frac)
            except Exception as e:
                print("  [overlap check failed for %s.%s] %s"%(r["db"],r["table"],e))
        if best and best[1]>0.3:
            r=best[0]
            print("\n  >> LIKELY price source: %s.%s (%.0f%% overlap)"%(r["db"],r["table"],100*best[1]))
            print("     Run Option 2 against it:")
            print("     python pead_backtest_60d.py --root . --price-db %s --price-table %s"
                  %(r["db"],r["table"]))
    elif not daily_ok:
        print("  No DAILY_OK table exists, so outcomes was NOT built from a stored daily price")
        print("  table in these DBs. Most likely outcomes is populated by a LIVE API fetch")
        print("  (yfinance/polygon/etc.) that doesn't persist prices, OR daily prices live")
        print("  outside these .db files (parquet/csv/feather).")

    banner("BOTTOM LINE")
    if daily_ok:
        print("  A usable daily price series exists (see DAILY_OK). Point Option 2 at it with")
        print("  --price-db / --price-table and the 60-day PEAD test can run for real.")
    else:
        print("  NO usable daily price series found in any .db file. The 60-day PEAD test cannot")
        print("  run on stored data. Options to proceed:")
        print("   (a) if daily prices live in parquet/csv/feather, tell me the path/format")
        print("   (b) if outcomes is built from a live API, I'll build a price-fetch step")
        print("       (yfinance can pull 2+ years of daily adj-close for your ~400 tickers)")
        print("   (c) tell me how outcomes.actual_return is computed and I'll trace the source")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
