#!/usr/bin/env python3
"""
FINRA QUERY INSPECTOR — shows exactly what the data API returns, so we can fix the
0-rows issue. Auth already works; this isolates dataset-name / filter / field-name.

Does 4 things:
  1. lists the datasets available to your account (so we see the REAL short-interest name)
  2. tries a couple candidate short-interest datasets with NO filter, limit 5, and
     dumps the raw JSON (so we see exact field names + a sample record)
  3. tries with a settlementDate range filter (to see if the filter is the problem)
  4. prints guidance based on what came back

USAGE:
  read -s FINRA_SECRET    (enter secret)
  python finra_query_inspect.py --client-id 02b7d1147b9e48bc9616 --client-secret $FINRA_SECRET
"""
import argparse, base64, json, sys
import urllib.request, urllib.error, urllib.parse

OAUTH="https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
API="https://api.finra.org"

def get_token(cid,sec):
    cred=base64.b64encode(("%s:%s"%(cid,sec)).encode()).decode()
    req=urllib.request.Request(OAUTH, data=b"", method="POST",
        headers={"Authorization":"Basic "+cred,"Content-Type":"application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req,timeout=30) as r:
        return json.loads(r.read().decode())["access_token"]

def call(method, path, token, payload=None, params=None):
    url=API+path
    if params: url+="?"+urllib.parse.urlencode(params)
    data=json.dumps(payload).encode() if payload is not None else None
    headers={"Authorization":"Bearer "+token,"Accept":"application/json"}
    if data is not None: headers["Content-Type"]="application/json"
    req=urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req,timeout=40) as r:
            return r.getcode(), r.read().decode("utf-8","replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8","replace")
    except Exception as e:
        return None, str(e)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--client-id",required=True)
    ap.add_argument("--client-secret",required=True)
    a=ap.parse_args()
    print("FINRA query inspector")
    try:
        token=get_token(a.client_id,a.client_secret)
        print("  token acquired OK\n")
    except Exception as e:
        print("  token failed:",str(e)[:200]); return

    # 1. list datasets
    print("="*70); print("1. DATASETS available to your account"); print("="*70)
    for path in ("/data/datasets","/datasets","/data/group/otcMarket"):
        code,body=call("GET",path,token)
        print("\n  GET %s -> HTTP %s"%(path,code))
        if code==200:
            print("  "+body[:1500])
            break
        else:
            print("  "+body[:300])

    # 2. try candidate short-interest datasets, no filter, small limit
    print("\n"+"="*70); print("2. SHORT-INTEREST datasets — raw sample (no filter, limit 5)"); print("="*70)
    candidates=[
        ("otcMarket","consolidatedShortInterest"),
        ("otcMarket","EquityShortInterest"),
        ("otcMarket","equityShortInterest"),
        ("otcmarket","consolidatedShortInterest"),
        ("OTCMarket","ConsolidatedShortInterest"),
    ]
    working=None
    for grp,name in candidates:
        path="/data/group/%s/name/%s"%(grp,name)
        code,body=call("POST",path,token,payload={"limit":5,"offset":0})
        print("\n  POST %s -> HTTP %s"%(path,code))
        if code==200 and body and body not in ("[]","{}"):
            print("  RAW RESPONSE (first 1800 chars):")
            print("  "+body[:1800])
            # show field names
            try:
                j=json.loads(body)
                rows=j if isinstance(j,list) else j.get("data",j.get("results",[]))
                if rows and isinstance(rows,list) and isinstance(rows[0],dict):
                    print("\n  >> FIELD NAMES:",list(rows[0].keys()))
                    working=(grp,name)
            except Exception as ex:
                print("  (parse note: %s)"%ex)
            break
        else:
            print("  "+str(body)[:200])

    # 3. with a date filter, to test filter syntax
    if working:
        grp,name=working
        print("\n"+"="*70); print("3. WITH settlementDate filter (test filter syntax)"); print("="*70)
        for flt in [
            {"compareFilters":[{"compareType":"GREATER_THAN_OR_EQUAL","fieldName":"settlementDate","fieldValue":"2025-01-01"}]},
            {"dateRangeFilters":[{"startDate":"2025-01-01","endDate":"2025-12-31","fieldName":"settlementDate"}]},
        ]:
            payload=dict(flt); payload["limit"]=5
            path="/data/group/%s/name/%s"%(grp,name)
            code,body=call("POST",path,token,payload=payload)
            n=0
            try:
                j=json.loads(body); rows=j if isinstance(j,list) else j.get("data",j.get("results",[])); n=len(rows)
            except Exception: pass
            print("\n  filter=%s -> HTTP %s, rows=%d"%(list(flt.keys())[0],code,n))
            if code!=200: print("  "+str(body)[:200])

    print("\n"+"="*70); print("WHAT TO PASTE BACK"); print("="*70)
    print("  Paste sections 1-3. Key things I need: the working dataset group/name, the exact")
    print("  FIELD NAMES, and which filter syntax returned rows. Then I lock the fetcher to match.")

if __name__=="__main__":
    main()
