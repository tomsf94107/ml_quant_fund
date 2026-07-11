#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — FINRA SHORT-INTEREST FETCHER v2 (fixes the truncation bug)
================================================================================
WHY v2: the old finra_short_interest.py fetched a whole QUARTER per query and
capped at `offset>50000`. But ONE settlement date returns ~21,000+ raw rows
(every US equity with a short position), so a 6-settlement quarter is ~130k rows
and the 50k cap chopped each quarter down to its first ~2-3 settlements. FINRA
sorts ascending by date, so the LATEST settlements in each quarter were dropped --
which is why the DB stopped at 2026-05-15 and had ~60 of ~120 settlements. The old
`q_end=28` day also dropped every quarter-boundary end-of-month settlement.

THE FIX (verified approach -- the finra_probe.py run proved the query + offset
pagination work to offset>=20000):
  * Fetch in HALF-MONTH windows -> exactly ONE settlement date each (~21-25k rows,
    ~5 pages at limit=5000, offset stays <=~25k -- inside the proven-safe range).
  * Page each window FULLY (until a short page), no premature volume cap.
  * Filter to your universe CLIENT-SIDE, clip DTC>=50, upsert (idempotent).
  * No quarter boundaries -> no q_end gap.

ONE script, both jobs:
  * BACKFILL (fill all gaps incl. 05-29 + reach current):
        source ~/.finra_creds && python si_fetch_v2.py --root . --months-back 62
  * CRON incremental (just the recent settlements):
        source ~/.finra_creds && python si_fetch_v2.py --root . --months-back 2
  * Coverage report (no creds): python si_fetch_v2.py --root . --status
  * Plan only (no fetch):       python si_fetch_v2.py --root . --months-back 2 --dry-run

AUTH: FINRA_CLIENT_ID + FINRA_SECRET in env (source ~/.finra_creds). --status and
--dry-run need no creds.
================================================================================
"""
import argparse, os, sqlite3, datetime, base64, json, time, urllib.request, urllib.error

def conn_rw(p): return sqlite3.connect(p,timeout=30)
def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def nd(s):
    if s is None: return None
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None
LINE="="*78

SI_DB_DEFAULT="short_interest.db"
TOKEN_URL="https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
DATA_URL="https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
CLIP_DTC=50.0
PAGE=5000

class AuthError(Exception): pass

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

def load_universe(prices_db, si_db):
    """Exchange-listed tickers from prices.db, unioned with whatever is already in
    short_interest.db (so we never drop a previously-tracked name). Falls back to the
    existing short_interest universe if prices.db is missing."""
    uni=set()
    if prices_db and os.path.isfile(prices_db):
        c=ro(prices_db)
        try:
            for t in ("daily_prices","prices"):
                try:
                    cols=[r[1].lower() for r in Q(c,'PRAGMA table_info("%s")'%t)]
                    if "ticker" in cols:
                        uni|=set(r[0].upper() for r in Q(c,'SELECT DISTINCT ticker FROM "%s"'%t) if r[0]); break
                except Exception: continue
        finally: c.close()
    if os.path.isfile(si_db):
        c=ro(si_db)
        try: uni|=set(r[0].upper() for r in Q(c,"SELECT DISTINCT ticker FROM short_interest") if r[0])
        except Exception: pass
        finally: c.close()
    return uni or None

def get_token(cid,secret):
    auth=base64.b64encode(("%s:%s"%(cid,secret)).encode()).decode()
    req=urllib.request.Request(TOKEN_URL,method="POST",
        headers={"Authorization":"Basic "+auth,"Content-Type":"application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req,timeout=30) as r:
        return json.loads(r.read().decode())["access_token"]

def parse_records(body):
    try: j=json.loads(body)
    except Exception: return None
    rows = j if isinstance(j,list) else j.get("data", j.get("results", []))
    if not isinstance(rows,list): return []
    out=[]
    for r in rows:
        if not isinstance(r,dict): continue
        tk = r.get("symbolCode") or r.get("issueSymbolIdentifier") or r.get("symbol") or r.get("ticker")
        d  = r.get("settlementDate") or r.get("settlement_date")
        cs = r.get("currentShortPositionQuantity") or r.get("currentShortShareNumber") or r.get("currentShort")
        av = r.get("averageDailyVolumeQuantity") or r.get("averageShortShareNumber") or r.get("avgDailyVolume")
        dtc= r.get("daysToCoverQuantity") or r.get("daysToCover")
        if not tk or not d: continue
        def f(x):
            try: return float(x)
            except Exception: return None
        cs=f(cs); av=f(av); dtc=f(dtc)
        if dtc is None and cs is not None and av and av>0: dtc=cs/av
        out.append((str(tk).upper(), str(d)[:10], cs, av, dtc))
    return out

def finra_page(token, start, end, offset, limit=PAGE, timeout=120):
    body={"limit":limit,"offset":offset,
          "dateRangeFilters":[{"startDate":start,"endDate":end,"fieldName":"settlementDate"}]}
    req=urllib.request.Request(DATA_URL,method="POST",data=json.dumps(body).encode(),
        headers={"Authorization":"Bearer "+token,"Content-Type":"application/json","Accept":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            return r.getcode(), parse_records(r.read().decode("utf-8","replace"))
    except urllib.error.HTTPError as e:
        return e.code, ("__HTTP__%d__%s"%(e.code, e.read().decode("utf-8","replace")[:200]))
    except Exception as e:
        return None, ("__ERR__%s"%(str(e)[:200]))

def fetch_window(token, start, end, off_cap=80000):
    """Page through ONE half-month window fully. Returns (records, truncated_bool)."""
    rows=[]; off=0
    while True:
        code,recs=finra_page(token,start,end,off)
        if code in (401,403): raise AuthError()
        if code!=200 or not isinstance(recs,list):
            # transient/HTTP error on this window -> stop, report what we have
            return rows, False
        if not recs: break
        rows+=recs
        if len(recs)<PAGE: break
        off+=PAGE
        if off>off_cap:
            return rows, True   # window unexpectedly huge -> flag truncation
        time.sleep(0.12)        # polite throttle within a window
    return rows, False

def month_iter(months_back, today):
    y,m=today.year, today.month
    sm=m-months_back; sy=y
    while sm<=0: sm+=12; sy-=1
    cy,cm=sy,sm; out=[]
    while (cy<y) or (cy==y and cm<=m):
        out.append((cy,cm)); cm+=1
        if cm>12: cm=1; cy+=1
    return out

def half_month_windows(months_back, today):
    wins=[]
    for (yy,mm) in month_iter(months_back,today):
        nxt=datetime.date(yy+(1 if mm==12 else 0),(mm%12)+1,1)
        last=(nxt-datetime.timedelta(days=1)).day
        wins.append((datetime.date(yy,mm,1).isoformat(), datetime.date(yy,mm,16).isoformat()))
        wins.append((datetime.date(yy,mm,17).isoformat(), datetime.date(yy,mm,last).isoformat()))
    return [(s,e) for (s,e) in wins if datetime.date.fromisoformat(s)<=today]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--si-db",default=None)
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--months-back",type=int,default=2,help="62 for full backfill; 2 for cron")
    ap.add_argument("--status",action="store_true")
    ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    si_db=a.si_db or os.path.join(a.root,SI_DB_DEFAULT)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")

    print("\n"+LINE+"\nFINRA SHORT-INTEREST FETCHER v2 (half-month windows)\n"+LINE)
    have=existing_dates(si_db)
    if have:
        sd=sorted(nd(x) for x in have if nd(x))
        print("  short_interest.db: %d settlement dates, %s to %s"%(len(have),sd[0],sd[-1]))
    else:
        print("  short_interest.db: empty/not found at %s"%si_db)
    uni=load_universe(prices_db,si_db)
    print("  universe: %s tickers"%(len(uni) if uni else "NOT FOUND (will keep ALL -- set --prices-db)"))

    today=datetime.date.today()
    wins=half_month_windows(a.months_back,today)
    print("  plan: %d half-month windows over ~%d months (today=%s)"%(len(wins),a.months_back,today))
    if a.status:
        print("\n  [status only] no fetch.")
        return
    if a.dry_run:
        print("  [dry-run] first/last windows:", wins[0], "...", wins[-1])
        print("  [dry-run] each window = one settlement date, paged fully, filtered, upserted.")
        return

    cid=os.environ.get("FINRA_CLIENT_ID"); secret=os.environ.get("FINRA_SECRET")
    if not cid or not secret:
        print("\n  [STOP] set FINRA_CLIENT_ID and FINRA_SECRET (source ~/.finra_creds). --status/--dry-run need neither.")
        return
    try:
        token=get_token(cid,secret); print("  auth OK; fetching %d windows...\n"%len(wins))
    except Exception as e:
        print("  [AUTH ERROR]",repr(e)); return

    db=ensure_schema(si_db)
    tot_kept=0; new_dates=set(); trunc=[]
    try:
        for (s,e) in wins:
            recs,was_trunc=fetch_window(token,s,e)
            if was_trunc: trunc.append((s,e))
            if not recs:
                time.sleep(0.15); continue
            kept=0
            for tk,d,cs,av,dtc in recs:
                if dtc is not None and dtc>=CLIP_DTC: continue          # clip OTC 999.99 junk
                if uni is not None and tk not in uni: continue          # universe filter
                db.execute("""INSERT INTO short_interest(ticker,settlement_date,current_short,avg_daily_vol,days_to_cover)
                              VALUES(?,?,?,?,?)
                              ON CONFLICT(ticker,settlement_date) DO UPDATE SET
                                current_short=excluded.current_short,
                                avg_daily_vol=excluded.avg_daily_vol,
                                days_to_cover=excluded.days_to_cover""",(tk,d,cs,av,dtc))
                kept+=1; new_dates.add(d)
            db.commit(); tot_kept+=kept
            ds=sorted(set(d for _,d,_,_,_ in recs))
            print("  %s..%s -> %d raw, %d kept | dates: %s"%(s,e,len(recs),kept,",".join(ds) if ds else "none"))
            time.sleep(0.15)
    except AuthError:
        print("\n  [AUTH ERROR mid-fetch] token rejected (401/403)."); db.close(); return
    db.close()

    print("\n"+LINE)
    print("  upserted rows: %d | distinct settlement dates touched: %d"%(tot_kept,len(new_dates)))
    if new_dates: print("  newest date now in pull: %s"%max(new_dates))
    if trunc:
        print("  [WARN] %d windows hit the offset cap (unexpected volume): %s"%(len(trunc),trunc[:4]))
        print("         tell me -- a window with 2 settlements would need finer splitting.")
    after=existing_dates(si_db)
    if after:
        sd=sorted(nd(x) for x in after if nd(x))
        print("  short_interest.db NOW: %d settlement dates, %s to %s"%(len(after),sd[0],sd[-1]))
        print("  gained %d new dates this run."%(len(after)-len(have)))
    print("\n  Next: re-validate -> python validate_si_v2.py --root . --hold 40")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
