#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — FORWARD SIGNAL LOGGER  (fix the data constraint going forward)
================================================================================
The binding limitation we hit: the promising non-price bricks (short_ratio,
pc_ratio_snap, inst_signed_flow_5d, iv_skew_snap) live in prediction_features with
only ~3 months of history — too short to validate. Prices we could backfill 17
years; these live-computed features we CANNOT reconstruct from price data.

The honest fix: start LOGGING these signals daily into a dedicated, append-only
longitudinal panel NOW, so that in N months you have enough history to validate
them with the same rigor as PEAD. This script does one day's snapshot; run it daily
(cron) and the panel grows.

WHAT IT DOES:
  * reads today's values of the candidate signals from prediction_features (+ any
    short-interest / analyst tables) for all tickers
  * appends them to signal_panel.db -> signal_panel(date, ticker, signal, value)
    (idempotent per (date,ticker,signal); re-running same day overwrites, no dupes)
  * this is the ONE script here that WRITES — to its own new DB only (signal_panel.db);
    your existing DBs are read-only

Over time, signal_panel.db becomes the long-history feature store that validate_signal.py
and portfolio_combine.py need to turn leads into confirmed bricks.

SUGGESTED CRON (run after your pipeline computes daily features, e.g. post-close):
  30 16 * * 1-5  cd /path/to/ML_Quant_Fund && python signal_logger.py --root .

RULE 1: writes only to signal_panel.db (separate file). Reads source DBs read-only.
Logs what EXISTS today; never fabricates. Idempotent per day.

USAGE:
  python signal_logger.py --root .                 # snapshot today
  python signal_logger.py --root . --date 2026-06-24   # snapshot a specific date's rows
  python signal_logger.py --root . --status         # show panel growth so far
================================================================================
"""
import argparse, os, sqlite3, sys, datetime
from collections import defaultdict

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def ro(p):
    if not os.path.isfile(p): raise FileNotFoundError(p)
    return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def has_table(c,n): return bool(Q(c,"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(n,)))
def tables(c): return [r[0] for r in Q(c,"SELECT name FROM sqlite_master WHERE type='table'")]
def cols_of(c,t): return [r[1] for r in Q(c,'PRAGMA table_info("'+t+'")')]
def all_dbs(root):
    out=[]
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f.endswith((".db",".sqlite",".sqlite3")) and f!="signal_panel.db": out.append(os.path.join(dp,f))
    return sorted(out)
def nd(s):
    if s is None: return None
    s=str(s)[:10]
    try: return datetime.date.fromisoformat(s)
    except Exception: return None

# the candidate signals worth logging (the leads from the non-price hunt)
CANDIDATES=["short_ratio","pc_ratio_snap","iv_skew_snap","inst_signed_flow_5d",
            "inst_signed_flow_30d","inst_block_buy_sell_7d","inst_auction_imbal_5d",
            "days_to_cover","short_interest","si_ratio"]

def init_panel(path):
    conn=sqlite3.connect(path,timeout=30)
    conn.execute("""CREATE TABLE IF NOT EXISTS signal_panel(
        date TEXT NOT NULL, ticker TEXT NOT NULL, signal TEXT NOT NULL, value REAL,
        logged_at TEXT, PRIMARY KEY(date,ticker,signal))""")
    conn.commit(); return conn

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--panel-db",default=None)
    ap.add_argument("--date",default=None,help="snapshot rows for this date (default: latest available)")
    ap.add_argument("--status",action="store_true")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    panel_db=a.panel_db or os.path.join(a.root,"signal_panel.db")
    banner("ML QUANT FUND — FORWARD SIGNAL LOGGER")

    if a.status:
        if not os.path.isfile(panel_db):
            print("  no signal_panel.db yet — run without --status to start logging."); return
        c=ro(panel_db)
        try:
            n=Q(c,"SELECT COUNT(*) FROM signal_panel")[0][0]
            nd_=Q(c,"SELECT COUNT(DISTINCT date) FROM signal_panel")[0][0]
            dr=Q(c,"SELECT MIN(date),MAX(date) FROM signal_panel")[0]
            bysig=Q(c,"SELECT signal,COUNT(DISTINCT date),COUNT(*) FROM signal_panel GROUP BY signal ORDER BY signal")
        finally:
            c.close()
        print("  panel: %d rows, %d distinct dates, %s..%s"%(n,nd_,dr[0],dr[1]))
        print("  per-signal (distinct_dates, rows):")
        for sig,dd,rr in bysig:
            print("    %-24s %d dates, %d rows"%(sig,dd,rr))
        span=(nd(dr[1])-nd(dr[0])).days if dr[0] and dr[1] else 0
        print("\n  history span: %d days. Need ~500+ trading days (2yr) to validate a lead."%span)
        if span<700:
            print("  keep logging daily — at ~21 trading days/month you'll reach 2yr in ~%d more months."
                  %max(0,(730-span)//30))
        return

    # find candidate signals across source DBs for the target date
    target=nd(a.date) if a.date else None
    conn=init_panel(panel_db)
    total_logged=0; found_signals=defaultdict(int)
    now=datetime.datetime.now().isoformat(timespec="seconds")
    for dbp in all_dbs(a.root):
        try: c=ro(dbp)
        except Exception: continue
        try:
            for t in tables(c):
                cl=cols_of(c,t)
                tcol="ticker" if "ticker" in cl else ("symbol" if "symbol" in cl else None)
                dcol=next((x for x in ("prediction_date","date","as_of","report_date","settlement_date","updated_at") if x in cl),None)
                if not (tcol and dcol): continue
                present=[s for s in CANDIDATES if s in cl]
                # also catch analyst net
                has_analyst = "upgrades_30d" in cl and "downgrades_30d" in cl
                if not present and not has_analyst: continue
                # determine date to snapshot: target, else max date in this table
                if target is None:
                    mx=Q(c,"SELECT MAX(substr("+dcol+",1,10)) FROM "+'"'+t+'"')[0][0]
                    tdate=nd(mx)
                else:
                    tdate=target
                if tdate is None: continue
                tds=tdate.isoformat()
                # pull rows for that date
                sel_cols=[tcol]+present+(["upgrades_30d","downgrades_30d"] if has_analyst else [])
                selq=",".join('"'+x+'"' for x in sel_cols)
                rows=Q(c,"SELECT "+selq+" FROM "+'"'+t+'" WHERE substr('+dcol+',1,10)=?',(tds,))
                for row in rows:
                    tk=row[0]
                    if not tk: continue
                    idx=1
                    for s in present:
                        v=row[idx]; idx+=1
                        if v is None: continue
                        try: fv=float(v)
                        except Exception: continue
                        conn.execute("REPLACE INTO signal_panel(date,ticker,signal,value,logged_at) VALUES(?,?,?,?,?)",
                                     (tds,tk,s,fv,now))
                        total_logged+=1; found_signals[s]+=1
                    if has_analyst:
                        up=row[idx]; dn=row[idx+1]
                        try: net=float(up or 0)-float(dn or 0)
                        except Exception: net=None
                        if net is not None:
                            conn.execute("REPLACE INTO signal_panel(date,ticker,signal,value,logged_at) VALUES(?,?,?,?,?)",
                                         (tds,tk,"analyst_net",net,now))
                            total_logged+=1; found_signals["analyst_net"]+=1
            conn.commit()
        finally:
            c.close()
    conn.close()
    print("  logged %d (date,ticker,signal) values to signal_panel.db"%total_logged)
    if found_signals:
        print("  signals captured this run:")
        for s,n in sorted(found_signals.items()):
            print("    %-24s %d tickers"%(s,n))
    else:
        print("  [WARN] no candidate signals found for the target date. Check that your pipeline")
        print("  has computed today's features, or pass --date with a date that has data.")
    print("\n  Run this daily (cron) so history accumulates. Check progress: --status")
    print("  Once you have ~2yr, validate a lead: python validate_signal.py --feature short_ratio")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
