#!/usr/bin/env python3
"""
FINRA PROBE — decisive test: does the consolidatedShortInterest API return the
settlements we're missing (05-29, maybe 06-15)?

FINRA's downloadable files page lists May 29, 2026 as published. Our DB stops at
05-15. This probe asks the API directly for [2026-05-20, 2026-06-25] and prints
exactly which settlement dates it returns, paging through with offset pagination.

  - If 2026-05-29 appears -> the API HAS it -> finra_short_interest.py has a fetch
    bug (we fix the fetcher).
  - If only 05-15 / nothing newer -> the API LAGS the downloadable files -> the DB
    is as current as this API allows (different problem; switch data path).

READ-ONLY: no database writes. Needs FINRA_CLIENT_ID + FINRA_SECRET in env
(source ~/.finra_creds first).

  source ~/.finra_creds && python finra_probe.py
"""
import os, json, base64, urllib.request, urllib.error, datetime

TOKEN_URL="https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
DATA_URL="https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
START="2026-05-20"; END="2026-06-25"

def main():
    cid=os.environ.get("FINRA_CLIENT_ID"); sec=os.environ.get("FINRA_SECRET")
    if not cid or not sec:
        print("[STOP] source ~/.finra_creds first (FINRA_CLIENT_ID / FINRA_SECRET not set)"); return
    auth=base64.b64encode(("%s:%s"%(cid,sec)).encode()).decode()
    try:
        req=urllib.request.Request(TOKEN_URL,method="POST",
            headers={"Authorization":"Basic "+auth,"Content-Type":"application/x-www-form-urlencoded"})
        token=json.loads(urllib.request.urlopen(req,timeout=30).read().decode())["access_token"]
        print("auth OK")
    except Exception as e:
        print("[AUTH ERROR]",repr(e)); return

    def page(offset):
        body={"limit":1000,"offset":offset,
              "dateRangeFilters":[{"startDate":START,"endDate":END,"fieldName":"settlementDate"}]}
        req=urllib.request.Request(DATA_URL,method="POST",data=json.dumps(body).encode(),
            headers={"Authorization":"Bearer "+token,"Content-Type":"application/json","Accept":"application/json"})
        try:
            with urllib.request.urlopen(req,timeout=120) as r:
                j=json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            print("  [HTTP %d] %s"%(e.code, e.read().decode()[:200])); return None
        return j if isinstance(j,list) else j.get("data",j.get("results",[]))

    print("querying settlementDate in [%s, %s] ..."%(START,END))
    allrows=[]; off=0; pages=0
    while True:
        recs=page(off)
        if recs is None: break
        pages+=1
        if not recs: break
        allrows+=recs
        if len(recs)<1000: break
        off+=1000
        if off>20000:
            print("  (hit offset safety cap)"); break

    print("\npages fetched: %d | total raw rows: %d"%(pages,len(allrows)))
    dates={}
    first_order=[]
    for r in allrows:
        d=str(r.get("settlementDate") or r.get("settlement_date"))[:10]
        dates[d]=dates.get(d,0)+1
        if len(first_order)<5: first_order.append(d)
    print("first 5 rows' settlement dates (shows sort order):", first_order)
    print("distinct settlement dates the API returned:")
    if not dates:
        print("  (none)")
    for d in sorted(dates):
        print("   %s : %d rows"%(d,dates[d]))

    print("\nVERDICT:")
    if "2026-05-29" in dates:
        print("  >> API HAS 2026-05-29. So finra_short_interest.py is DROPPING it -> fetcher bug.")
        print("     Fix the fetcher (single wide date range + pagination); backfill will pick it up.")
    elif dates:
        print("  >> API returned data but NOT 2026-05-29 (newest = %s)."%max(dates))
        print("     The API lags FINRA's downloadable files. DB is as current as this API allows.")
    else:
        print("  >> API returned nothing for this window. Either no data yet, or query/filter issue.")

if __name__=="__main__":
    main()
