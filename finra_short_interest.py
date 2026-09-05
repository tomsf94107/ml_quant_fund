#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — FINRA SHORT-INTEREST BACKFILL  (free, official, 5-yr history)
================================================================================
Short interest is your strongest short-side lead, and unlike options/flow it is
PUBLIC and historically downloadable: FINRA publishes consolidated short interest
bi-monthly (Rule 4560) with ~5 rolling years available. This fetches it for your
universe, caches it, and makes it directly validatable with validate_signal.py.

This solves the data constraint for ONE real lead with ZERO vendor cost and proper
multi-year history — the RULE-1-clean path (official source, verifiable, deep).

PIPELINE:
  1. read universe tickers from earnings.db.earnings_surprises (same as prices)
  2. query FINRA's data API for short interest over --years-back, quarter by quarter
     (date-range filter on settlementDate, paginated), filtered to your universe
  3. cache to short_interest.db -> short_interest(ticker, settlement_date,
     current_short, avg_daily_vol, days_to_cover)
  4. days_to_cover = current_short / avg_daily_vol (the stronger predictor per research)

THEN validate with the same battery that locked PEAD:
  python validate_signal.py --root . --feature days_to_cover --hold 40
  python validate_signal.py --root . --feature current_short  --hold 20
(validate_signal.py auto-finds short_interest.db, recognizes settlement_date, and
 auto-negates short-side features so "high short interest -> low return".)

AUTH: FINRA's data API uses a free API account (OAuth2 client-credentials). Get a
client_id/secret from the FINRA API Developer Center, then either:
  * pass --client-id / --client-secret  (script does the token exchange), or
  * pass --token  (a bearer token you already obtained)
Some queries work without auth at low volume; if you hit 401/403 the script prints
exactly how to authenticate. Nothing is fabricated and nothing large is fetched
until auth works.

RULE 1: writes only to short_interest.db (separate file); your DBs are read-only.
Every fetch verified (counts, date span); fail-loud on auth/empty; idempotent per
(ticker, settlement_date). days_to_cover computed transparently, not guessed.

NETWORK NOTE: hits api.finra.org (+ the FINRA OAuth host). If your security tooling
flagged the price fetch, it may react again — small footprint (~20-40 calls total).

USAGE:
  python finra_short_interest.py --root . --client-id ID --client-secret SECRET
  python finra_short_interest.py --root . --token BEARER_TOKEN
  python finra_short_interest.py --root . --years-back 5
  python finra_short_interest.py --root . --status        # show what's cached
================================================================================
"""
import argparse, os, sqlite3, sys, json, base64, datetime, time
from collections import defaultdict
try:
    import urllib.request, urllib.error, urllib.parse
    HAVE_URL=True
except Exception: HAVE_URL=False

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
def all_dbs(root):
    out=[]
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f.endswith((".db",".sqlite",".sqlite3")) and f!="short_interest.db": out.append(os.path.join(dp,f))
    return sorted(out)
def find_db(root,name):
    c=os.path.join(root,name)
    if os.path.isfile(c): return c
    for d in all_dbs(root):
        if os.path.basename(d)==name: return d
    return None
def nd(s):
    if s is None: return None
    s=str(s)[:10]
    try: return datetime.date.fromisoformat(s)
    except Exception: return None

FINRA_API="https://api.finra.org/data/group/otcMarket/name/"
FINRA_OAUTH="https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
# candidate datasets (exchange-listed consolidated first, then the generic equity set)
DATASETS=["consolidatedShortInterest","EquityShortInterest"]

def init_db(path):
    conn=sqlite3.connect(path,timeout=30)
    conn.execute("""CREATE TABLE IF NOT EXISTS short_interest(
        ticker TEXT NOT NULL, settlement_date TEXT NOT NULL,
        current_short REAL, avg_daily_vol REAL, days_to_cover REAL,
        PRIMARY KEY(ticker,settlement_date))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS fetch_log(
        quarter TEXT PRIMARY KEY, status TEXT, n_rows INTEGER, fetched_at TEXT)""")
    conn.commit(); return conn

def get_token(client_id, client_secret):
    if not (client_id and client_secret): return None
    cred=base64.b64encode(("%s:%s"%(client_id,client_secret)).encode()).decode()
    url=FINRA_OAUTH+"?grant_type=client_credentials"
    req=urllib.request.Request(url, data=b"", method="POST",
        headers={"Authorization":"Basic "+cred,"Content-Type":"application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req,timeout=30) as r:
            j=json.loads(r.read().decode())
            return j.get("access_token")
    except Exception as e:
        print("  [auth] token exchange failed: %s"%str(e)[:200])
        return None

def finra_query(dataset, token, date_from, date_to, offset, limit=1000, timeout=40):
    url=FINRA_API+dataset
    # FINRA wants dateRangeFilters for date ranges (compareFilters returns 400 "Unable to parse")
    payload={
        "limit":limit, "offset":offset,
        "dateRangeFilters":[
            {"startDate":date_from,"endDate":date_to,"fieldName":"settlementDate"}
        ]
    }
    data=json.dumps(payload).encode()
    headers={"Content-Type":"application/json","Accept":"application/json"}
    if token: headers["Authorization"]="Bearer "+token
    req=urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            code=r.getcode(); body=r.read().decode("utf-8","replace")
            return code, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8","replace")[:300]
    except Exception as e:
        return None, str(e)[:200]

def parse_records(body):
    """Parse FINRA JSON response into normalized records.
    Field names confirmed via the query inspector against consolidatedShortInterest:
      symbolCode, currentShortPositionQuantity, averageDailyVolumeQuantity,
      daysToCoverQuantity, settlementDate."""
    try:
        j=json.loads(body)
    except Exception:
        return None
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
        if dtc is None and cs is not None and av and av>0:
            dtc=cs/av
        out.append((str(tk).upper(), str(d)[:10], cs, av, dtc))
    return out

def quarters_back(years):
    today=datetime.date.today()
    out=[]
    y=today.year; m=today.month
    start=today - datetime.timedelta(days=years*365)
    cur=datetime.date(start.year, ((start.month-1)//3)*3+1, 1)
    while cur<=today:
        q_end_month=cur.month+2
        q_end=datetime.date(cur.year + (1 if q_end_month>12 else 0), (q_end_month-1)%12+1, 28)
        out.append((cur.isoformat(), q_end.isoformat(), "%d-Q%d"%(cur.year,(cur.month-1)//3+1)))
        # advance one quarter
        nm=cur.month+3
        cur=datetime.date(cur.year+(1 if nm>12 else 0), (nm-1)%12+1, 1)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--db",default=None)
    ap.add_argument("--client-id",default=None)
    ap.add_argument("--client-secret",default=None)
    ap.add_argument("--token",default=None)
    ap.add_argument("--years-back",type=int,default=5)
    ap.add_argument("--tickers-file",default=None,
                    help="override tickers.txt for the keep-filter")
    ap.add_argument("--dataset",default=None,help="force a FINRA dataset name")
    ap.add_argument("--status",action="store_true")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    db=a.db or os.path.join(a.root,"short_interest.db")
    banner("ML QUANT FUND — FINRA SHORT-INTEREST BACKFILL (free, official)")

    if a.status:
        if not os.path.isfile(db): print("  no short_interest.db yet."); return
        c=ro(db)
        try:
            n=Q(c,"SELECT COUNT(*) FROM short_interest")[0][0]
            nt=Q(c,"SELECT COUNT(DISTINCT ticker) FROM short_interest")[0][0]
            nd_=Q(c,"SELECT COUNT(DISTINCT settlement_date) FROM short_interest")[0][0]
            dr=Q(c,"SELECT MIN(settlement_date),MAX(settlement_date) FROM short_interest")[0]
        finally: c.close()
        print("  short_interest.db: %d rows, %d tickers, %d settlement dates, %s..%s"%(n,nt,nd_,dr[0],dr[1]))
        span=(nd(dr[1])-nd(dr[0])).days/365.0 if dr[0] and dr[1] else 0
        print("  history span: %.1f years  (%s)"%(span,"sufficient to validate" if span>=1.8 else "keep fetching / extend --years-back"))
        print("\n  validate: python validate_signal.py --root . --feature days_to_cover --hold 40")
        return

    if not require(HAVE_URL,"urllib unavailable"): return
    earnp=find_db(a.root,"earnings.db")
    if not require(earnp,"earnings.db not found"): return
    ce=ro(earnp)
    try:
        if not require(has_table(ce,"earnings_surprises"),"no earnings_surprises"): return
        universe=set(r[0].upper() for r in Q(ce,"SELECT DISTINCT ticker FROM earnings_surprises WHERE ticker IS NOT NULL"))
        import os as _os
        # --tickers-file widens the filter without touching tickers.txt, which
        # every cron job reads. FINRA serves whole settlement files covering all
        # US equities; this set only decides which rows are KEPT, so the new
        # names had zero rows because they were never in the filter, not because
        # the data was missing.
        _tt=a.tickers_file or _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),"tickers.txt")
        if _os.path.exists(_tt):
            universe|={ln.strip().upper() for ln in open(_tt) if ln.strip() and not ln.lstrip().startswith("#")}
    finally:
        ce.close()
    print("  universe: %d tickers (from earnings_surprises + tickers.txt)"%len(universe))

    # auth
    token=a.token or get_token(a.client_id,a.client_secret)
    if token: print("  FINRA auth: token acquired")
    else: print("  FINRA auth: no token (will try anonymous; if 401/403, see instructions below)")

    datasets=[a.dataset] if a.dataset else DATASETS
    conn=init_db(db)
    done=set(r[0] for r in conn.execute("SELECT quarter FROM fetch_log WHERE status='OK'").fetchall())
    quarters=quarters_back(a.years_back)
    print("  fetching %d quarters over ~%d years (%d already cached)"%(len(quarters),a.years_back,len(done)))

    working_dataset=None
    total=0; auth_failed=False
    for date_from,date_to,qlabel in quarters:
        if qlabel in done: continue
        # find a dataset that returns data
        got=False
        for ds in ([working_dataset] if working_dataset else datasets):
            if ds is None: continue
            offset=0; qrows=0
            while True:
                code,body=finra_query(ds,token,date_from,date_to,offset)
                if code in (401,403):
                    auth_failed=True; break
                if code!=200 or not body:
                    break
                recs=parse_records(body)
                if recs is None:  # not JSON
                    break
                if not recs:
                    break
                keep=[r for r in recs if r[0] in universe]
                if keep:
                    conn.executemany("REPLACE INTO short_interest(ticker,settlement_date,current_short,avg_daily_vol,days_to_cover) VALUES(?,?,?,?,?)",keep)
                    conn.commit(); qrows+=len(keep); total+=len(keep)
                if len(recs)<1000: break
                offset+=1000
                if offset>50000: break  # safety
            if auth_failed: break
            if qrows>0:
                working_dataset=ds; got=True
                conn.execute("REPLACE INTO fetch_log VALUES(?,?,?,?)",(qlabel,"OK",qrows,datetime.datetime.now().isoformat(timespec="seconds")))
                conn.commit()
                print("    %s: %d rows (dataset=%s)"%(qlabel,qrows,ds))
                break
        if auth_failed: break
        if not got:
            conn.execute("REPLACE INTO fetch_log VALUES(?,?,?,?)",(qlabel,"EMPTY",0,datetime.datetime.now().isoformat(timespec="seconds")))
            conn.commit()
        time.sleep(0.3)

    if auth_failed:
        conn.close()
        banner("AUTH REQUIRED — how to get a free FINRA API token")
        print("  FINRA's data API rejected the request (401/403). Get free credentials:")
        print("   1. Create a FINRA API account at the FINRA API Developer Center / API Console")
        print("   2. Create an API client -> you get a client_id and client_secret")
        print("   3. Re-run:")
        print("      python finra_short_interest.py --root . --client-id YOUR_ID --client-secret YOUR_SECRET")
        print("   (the script exchanges them for a token automatically and resumes)")
        print("\n  Nothing was fabricated; no partial data trusted. This is the verify-before-act gate.")
        return

    # verify cache
    cc=ro(db)
    try:
        n=Q(cc,"SELECT COUNT(*) FROM short_interest")[0][0]
        nt=Q(cc,"SELECT COUNT(DISTINCT ticker) FROM short_interest")[0][0]
        nd_=Q(cc,"SELECT COUNT(DISTINCT settlement_date) FROM short_interest")[0][0]
        dr=Q(cc,"SELECT MIN(settlement_date),MAX(settlement_date) FROM short_interest")[0]
    finally: cc.close()
    conn.close()
    banner("DONE — short_interest.db")
    print("  %d rows, %d tickers, %d settlement dates, %s..%s"%(n,nt,nd_,dr[0],dr[1]))
    span=(nd(dr[1])-nd(dr[0])).days/365.0 if dr[0] and dr[1] else 0
    print("  history span: %.1f years"%span)
    if n==0:
        print("\n  [WARN] no rows cached. Either the dataset name differs (try --dataset consolidatedShortInterest")
        print("  or --dataset EquityShortInterest) or your universe tickers weren't in the response.")
        print("  The response format may have changed; check one raw query manually.")
    else:
        print("\n  NEXT — validate with the PEAD-grade battery (auto-finds this DB, auto-negates short side):")
        print("     python validate_signal.py --root . --feature days_to_cover --hold 40")
        print("     python validate_signal.py --root . --feature days_to_cover --hold 20")
        print("     python validate_signal.py --root . --feature current_short  --hold 40")
        print("  If days_to_cover survives OOS + beta-strip + sign on this 5-yr history -> brick #2 is REAL.")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
