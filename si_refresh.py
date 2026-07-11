#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — SHORT-INTEREST REFRESH JOB (Next-step #3, standalone)
================================================================================
Productionizes the short-interest brick AS FAR AS CAN BE DONE WITHOUT seeing the
pipeline code. This is the STANDALONE half: fetch latest -> clip -> join to
universe -> upsert to short_interest.db. The pipeline-WIRING half (builder.py
feature, cron entry, feature-panel surface, Pipeline B importance check) is marked
with explicit TODO hooks below, NOT faked -- per RULE 1, built-but-not-wired is
worse than honestly-stubbed, and guessing your pipeline's structure would create a
module that looks wired but isn't.

WHAT THIS DOES (runnable today):
  * Fetches the most recent FINRA settlement(s) not already in short_interest.db
    (incremental refresh -- FINRA publishes on a known bi-monthly schedule)
  * Clips the 999.99 OTC days_to_cover placeholders
  * Joins to your exchange-listed universe (drops OTC/junk tickers)
  * Upserts into short_interest.db (idempotent -- safe to re-run)
  * --status shows coverage without fetching (no credentials needed)

WHAT THIS DOES NOT DO (needs your pipeline code -- see TODO hooks):
  * Register days_to_cover as a feature in features/builder.py
  * Add itself to the cron schedule
  * Surface in the feature panel / Pipeline B importance

SETTLEMENT SCHEDULE: FINRA consolidated short interest settles ~twice monthly
(mid-month and month-end), published ~8 business days after settlement. A refresh
run every few days will pick up new settlements when they post; nothing to fetch
between postings (the job no-ops, which is correct).

CREDENTIALS: same as finra_short_interest.py. Set FINRA_CLIENT_ID + FINRA_SECRET
env vars. --status and --check-universe need no credentials (local only).

USAGE:
  python si_refresh.py --status                 # coverage report, no fetch
  python si_refresh.py --root . --universe-from prices.db   # incremental refresh
  python si_refresh.py --root . --dry-run       # show what WOULD be fetched
================================================================================
"""
import argparse, os, sqlite3, datetime, base64, json, urllib.request, urllib.parse

def conn_rw(p): return sqlite3.connect(p,timeout=30)
def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def nd(s):
    if s is None: return None
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None
LINE="="*78

SI_DB_DEFAULT="short_interest.db"
DATASET="otcMarket/consolidatedShortInterest"
TOKEN_URL="https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
DATA_URL="https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
CLIP_DTC=50.0

def ensure_schema(db):
    c=conn_rw(db)
    c.execute("""CREATE TABLE IF NOT EXISTS short_interest(
        ticker TEXT, settlement_date TEXT,
        current_short REAL, avg_daily_vol REAL, days_to_cover REAL,
        PRIMARY KEY(ticker, settlement_date))""")
    c.commit(); return c

def existing_dates(db):
    if not os.path.isfile(db): return set()
    c=ro(db)
    try: rows=Q(c,"SELECT DISTINCT settlement_date FROM short_interest")
    finally: c.close()
    return set(r[0] for r in rows if r[0])

def load_universe(path):
    """Exchange-listed tickers from prices.db (or any db with a ticker column)."""
    if not path or not os.path.isfile(path): return None
    c=ro(path)
    try:
        for t in ("daily_prices","prices","fetch_log"):
            try:
                cols=[r[1].lower() for r in Q(c,'PRAGMA table_info("%s")'%t)]
                if "ticker" in cols:
                    return set(r[0].upper() for r in Q(c,'SELECT DISTINCT ticker FROM "%s"'%t) if r[0])
            except Exception: continue
    finally: c.close()
    return None

def get_token(cid,secret):
    auth=base64.b64encode(("%s:%s"%(cid,secret)).encode()).decode()
    req=urllib.request.Request(TOKEN_URL,method="POST",
        headers={"Authorization":"Basic "+auth,"Content-Type":"application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req,timeout=30) as r:
        return json.loads(r.read().decode())["access_token"]

def fetch_settlement(token, since_date):
    """Fetch rows with settlementDate >= since_date. Uses dateRangeFilters (NOT
    compareFilters -- that 400s). Returns list of dicts."""
    body={
        "limit":50000,
        "fields":["symbolCode","currentShortPositionQuantity","averageDailyVolumeQuantity",
                  "daysToCoverQuantity","settlementDate"],
        "dateRangeFilters":[{"fieldName":"settlementDate",
                             "startDate":since_date.isoformat(),
                             "endDate":datetime.date.today().isoformat()}],
    }
    req=urllib.request.Request(DATA_URL,method="POST",data=json.dumps(body).encode(),
        headers={"Authorization":"Bearer "+token,"Content-Type":"application/json","Accept":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r:
        return json.loads(r.read().decode())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--si-db",default=None)
    ap.add_argument("--universe-from",default="prices.db")
    ap.add_argument("--status",action="store_true")
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--lookback-days",type=int,default=45)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    si_db=a.si_db or os.path.join(a.root,SI_DB_DEFAULT)
    uni_path=os.path.join(a.root,a.universe_from) if a.universe_from and not os.path.isabs(a.universe_from) else a.universe_from

    print("\n"+LINE+"\nSHORT-INTEREST REFRESH JOB\n"+LINE)

    # --- status mode (no creds) ---
    have=existing_dates(si_db)
    if have:
        sd=sorted(nd(x) for x in have if nd(x))
        print("  short_interest.db: %d settlement dates, %s to %s"%(len(have),sd[0],sd[-1]))
        c=ro(si_db)
        try:
            nrows=Q(c,"SELECT COUNT(*) FROM short_interest")[0][0]
            ntk=Q(c,"SELECT COUNT(DISTINCT ticker) FROM short_interest")[0][0]
            njunk=Q(c,"SELECT COUNT(*) FROM short_interest WHERE days_to_cover>=%f"%CLIP_DTC)[0][0]
        finally: c.close()
        print("  rows=%d tickers=%d | rows with junk DTC>=%.0f still present: %d"%(nrows,ntk,CLIP_DTC,njunk))
        if sd:
            # next expected settlement (rough: ~15th and month-end)
            last=sd[-1]
            print("  most recent settlement: %s (FINRA posts ~8 business days after settlement)"%last)
    else:
        print("  short_interest.db: empty or not found at %s"%si_db)
    uni=load_universe(uni_path)
    print("  exchange-listed universe (%s): %s tickers"%(a.universe_from, len(uni) if uni else "NOT FOUND"))
    if a.status:
        print("\n  [status only] no fetch performed.")
        return

    # --- determine what to fetch ---
    today=datetime.date.today()
    since=today-datetime.timedelta(days=a.lookback_days)
    print("\n  would fetch settlements since %s (lookback %d days)"%(since,a.lookback_days))
    already=set(x for x in have)
    if a.dry_run:
        print("  [dry-run] not fetching. Existing dates in window:",
              sorted(d for d in have if nd(d) and nd(d)>=since) or "none")
        print("  [dry-run] any NEW settlement in window would be upserted; existing ones skipped.")
        return

    cid=os.environ.get("FINRA_CLIENT_ID"); secret=os.environ.get("FINRA_SECRET")
    if not cid or not secret:
        print("\n  [STOP] set FINRA_CLIENT_ID and FINRA_SECRET to fetch. (--status/--dry-run need neither.)")
        return
    try:
        token=get_token(cid,secret)
        print("  auth OK; fetching...")
        payload=fetch_settlement(token,since)
    except Exception as e:
        print("  [FETCH ERROR]",repr(e)); return
    recs = payload if isinstance(payload,list) else payload.get("data",payload.get("results",[]))
    print("  fetched %d raw rows"%len(recs))

    db=ensure_schema(si_db); ins=0; skip_uni=0; skip_junk=0; skip_dup=0
    for r in recs:
        tk=(r.get("symbolCode") or "").upper().strip()
        sdate=nd(r.get("settlementDate"))
        if not tk or sdate is None: continue
        key=(tk,sdate.isoformat())
        if key[1] in already and key[0]:  # date already loaded — still upsert (idempotent), but count
            pass
        try: dtc=float(r.get("daysToCoverQuantity"))
        except Exception: dtc=None
        if dtc is not None and dtc>=CLIP_DTC: skip_junk+=1; continue   # clip OTC 999.99 placeholders
        if uni is not None and tk not in uni: skip_uni+=1; continue    # join to exchange-listed universe
        try: cs=float(r.get("currentShortPositionQuantity"))
        except Exception: cs=None
        try: adv=float(r.get("averageDailyVolumeQuantity"))
        except Exception: adv=None
        db.execute("""INSERT INTO short_interest(ticker,settlement_date,current_short,avg_daily_vol,days_to_cover)
                      VALUES(?,?,?,?,?)
                      ON CONFLICT(ticker,settlement_date) DO UPDATE SET
                        current_short=excluded.current_short,
                        avg_daily_vol=excluded.avg_daily_vol,
                        days_to_cover=excluded.days_to_cover""",
                   (tk,key[1],cs,adv,dtc))
        ins+=1
    db.commit(); db.close()
    print("  upserted=%d | dropped(off-universe)=%d | dropped(junk DTC)=%d"%(ins,skip_uni,skip_junk))
    print("  short_interest.db refreshed.")

    print("\n"+LINE+"\nTODO HOOKS — pipeline wiring (needs your code; not faked here)\n"+LINE)
    print("""  These are the RULE-1 'not done until wired' steps. I can write each once you
  paste the relevant file, so I match your architecture instead of guessing:

  [ ] features/builder.py — add a days_to_cover feature column:
        # as-of join: for each (ticker, prediction_date), take the most recent
        # settlement_date <= prediction_date from short_interest.db (PIT — no look-ahead).
        # The validated signal is days_to_cover at h=40; expose the raw value and let
        # the model rank it. Clip>=50 already handled at ingest.
        # >>> show me features/builder.py and I'll write the exact join + column.

  [ ] cron — add this refresh to the schedule:
        # FINRA posts bi-monthly; a run every ~3 days is plenty. In your VN-anchored
        # crontab style (see crontab_VN_anchored.txt):
        #   0 9 */3 * *  cd <repo> && <python> si_refresh.py --root . >> logs/si_refresh.log 2>&1
        # >>> confirm which cron file and I'll add the line in your format.

  [ ] feature panel / Pipeline B — confirm days_to_cover shows nonzero importance:
        # after builder wiring + one Pipeline B run, check it appears with nonzero
        # importance (else it's built-but-blind). >>> I'll add the panel surface once
        # I see how existing features register.

  [ ] tests — add a PIT-correctness test (feature uses only settlement<=date) +
        a clip test (no 999.99 leaks through). >>> I'll write to your test layout.
""")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
