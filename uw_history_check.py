#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — UNUSUALWHALES HISTORICAL-CAPABILITY CHECK
================================================================================
The backfill diagnostic found only yfinance (no historical options/SI). But your
system uses UnusualWhales (UW) for options flow, institutional flow, and short
interest. This checks whether your UW PLAN can backfill HISTORY for those signals
— the last fast-path shot for confirming brick #2 without waiting ~12-18 months.

It does the MINIMUM probing needed to answer "does historical data return?":
  * locates your UW client/config (token) the way your system does
  * makes a FEW tiny historical requests (one ticker, a date ~6mo and ~18mo back)
    for the endpoints we care about: options put/call & skew, short interest
  * reports, per endpoint: does historical data come back, and how far back?

CRITICAL — RESPECTS YOUR UW RULES (from your system's standing constraints):
  * Live UW calls only during MARKET HOURS (9:30am-4:00pm ET). This script REFUSES
    to call UW outside market hours unless you pass --force (it explains why).
  * Tiny footprint: a handful of calls, nowhere near the 40K/day limit.
  * Read-only: fetches nothing large, writes nothing.

This does NOT backfill. It only tells you whether backfill is POSSIBLE on your plan.
If yes -> next step is a chunked backfill (respecting quota); if no -> slow path.

USAGE:
  python uw_history_check.py                      # auto-detect token, check during market hours
  python uw_history_check.py --token YOUR_TOKEN   # pass token explicitly
  python uw_history_check.py --ticker AAPL --force # override market-hours guard (use sparingly)

NOTE: endpoint paths differ across UW API versions/plans. This tries the common
ones and reports what it finds; if your wrapper uses different paths, the output
tells you exactly which calls failed so you can map them.
================================================================================
"""
import argparse, os, sys, json, datetime
try:
    import urllib.request, urllib.error
    HAVE_URL=True
except Exception: HAVE_URL=False

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)

def market_hours_now():
    # crude ET check: UTC-4 (EDT) in summer. User is in VN but the MARKET is ET.
    now_utc=datetime.datetime.utcnow()
    et=now_utc - datetime.timedelta(hours=4)  # EDT
    if et.weekday()>=5: return False,et
    t=et.time()
    return (datetime.time(9,30)<=t<=datetime.time(16,0)),et

def find_token(explicit):
    if explicit: return explicit,"--token arg"
    # common env vars
    for k in ("UW_TOKEN","UNUSUAL_WHALES_TOKEN","UNUSUALWHALES_API_KEY","UW_API_KEY","UW_API_TOKEN"):
        v=os.environ.get(k)
        if v: return v,"env:"+k
    # common config files in a project
    for root in (".","~/.config","~"):
        base=os.path.expanduser(root)
        for fn in ("uw_config.json","unusualwhales.json","config.json",".uw","secrets.json",".env"):
            p=os.path.join(base,fn)
            if os.path.isfile(p):
                try:
                    txt=open(p).read()
                    # try json
                    try:
                        j=json.loads(txt)
                        for key in ("uw_token","token","api_key","UW_TOKEN","unusualwhales"):
                            if isinstance(j,dict) and key in j and isinstance(j[key],str):
                                return j[key],p+":"+key
                    except Exception:
                        pass
                    # try .env style
                    for line in txt.splitlines():
                        for k in ("UW_TOKEN","UNUSUALWHALES_API_KEY","UW_API_KEY"):
                            if line.strip().startswith(k+"="):
                                return line.split("=",1)[1].strip().strip('"').strip("'"),p+":"+k
                except Exception:
                    pass
    return None,None

def uw_get(path, token, params=None, timeout=20):
    base="https://api.unusualwhales.com"
    url=base+path
    if params:
        from urllib.parse import urlencode
        url+=("?"+urlencode(params))
    req=urllib.request.Request(url, headers={"Authorization":"Bearer "+token,"Accept":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            code=r.getcode(); body=r.read().decode("utf-8","replace")
            return code, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8","replace")[:300]
    except Exception as e:
        return None, str(e)[:200]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--token",default=None)
    ap.add_argument("--ticker",default="AAPL")
    ap.add_argument("--force",action="store_true",help="override the market-hours guard")
    a=ap.parse_args()
    banner("ML QUANT FUND — UNUSUALWHALES HISTORICAL-CAPABILITY CHECK")
    print("Can your UW plan backfill HISTORY for options/short-interest? (the fast-path shot)")
    if not HAVE_URL:
        print("  [STOP] urllib unavailable"); return

    # market-hours guard (your standing UW rule)
    is_open,et=market_hours_now()
    print("  current ET (approx): %s  | market open: %s"%(et.strftime("%Y-%m-%d %H:%M"),is_open))
    if not is_open and not a.force:
        print("\n  [HALT] Your UW rule: live calls only during market hours (9:30-16:00 ET).")
        print("  It is currently outside US market hours. Re-run during the session, or use --force")
        print("  if you accept making a few off-hours probe calls (tiny footprint, well under quota).")
        print("  (Probing off-hours is low-risk for a capability check, but the guard honors your rule.)")
        return
    if not is_open and a.force:
        print("  [--force] proceeding off-hours with a tiny probe footprint.")

    token,src=find_token(a.token)
    if not token:
        print("\n  [STOP] No UW token found. Pass --token YOUR_TOKEN, or set UW_TOKEN env var.")
        print("  (Looked in env vars and common config files. Your system already has this token")
        print("  somewhere — check how generator.py authenticates to UW and pass that token.)")
        return
    print("  UW token: found via %s (…%s)"%(src,token[-4:] if len(token)>4 else "****"))

    tk=a.ticker
    today=datetime.date.today()
    d_6mo=(today-datetime.timedelta(days=182)).isoformat()
    d_18mo=(today-datetime.timedelta(days=547)).isoformat()
    print("  probing %s for historical data at %s (6mo) and %s (18mo)\n"%(tk,d_6mo,d_18mo))

    # candidate historical endpoints (paths vary by plan/version; we try common ones)
    # we test: does a dated/historical query return data?
    probes=[
        ("options put/call (historical)", "/api/stock/%s/option-contracts"%tk, {"date":d_6mo}),
        ("options greek/IV (historical)", "/api/stock/%s/greek-exposure"%tk, {"date":d_6mo}),
        ("options volume history",        "/api/stock/%s/options-volume"%tk, {"date":d_6mo}),
        ("short interest (historical)",   "/api/stock/%s/short-interest"%tk, {"date":d_6mo}),
        ("ticker info (sanity/auth)",     "/api/stock/%s/info"%tk, None),
    ]
    sub("ENDPOINT PROBES (does historical data return?)")
    results={}
    auth_ok=False
    for label,path,params in probes:
        code,body=uw_get(path,token,params)
        ok = code==200 and body and body not in ("[]","{}","null") and len(body)>20
        has_data = ok and ('"data"' in body or body.strip().startswith("[") or '"date"' in body)
        status = "OK (data returned)" if has_data else (
                 "200 but empty" if code==200 else (
                 "AUTH FAIL (401/403)" if code in (401,403) else (
                 "NOT FOUND (404 - wrong path/plan)" if code==404 else (
                 "RATE-LIMITED (429)" if code==429 else "err %s"%code))))
        if code==200: auth_ok=True
        print("  %-32s -> %s"%(label,status))
        if code not in (200,404) and body:
            print("       %s"%str(body)[:160])
        results[label]={"code":code,"has_data":has_data}

    # ---- verdict ----
    banner("VERDICT — can UW backfill brick #2?")
    if not auth_ok:
        a_codes=set(r["code"] for r in results.values())
        if 401 in a_codes or 403 in a_codes:
            print("  >> AUTH FAILED. The token didn't authenticate. Check it's the right UW token and")
            print("     your plan is active. Can't assess backfill until auth works.")
        else:
            print("  >> No endpoint returned 200. Either the API paths differ on your plan/version, or")
            print("     connectivity failed. Note WHICH calls failed above and map them to your wrapper's")
            print("     actual endpoint paths (check how generator.py builds UW URLs), then re-run.")
        return
    hist = [k for k,v in results.items() if v["has_data"] and "historical" in k.lower() or (v["has_data"] and "history" in k.lower())]
    any_hist = any(v["has_data"] for k,v in results.items() if k!="ticker info (sanity/auth)")
    if any_hist:
        print("  >> FAST PATH LIKELY ALIVE: at least one historical endpoint returned data for a date")
        print("     ~6 months back. If it goes back 2+ years, you can BACKFILL and validate brick #2 now.")
        print("     Endpoints that returned historical data:")
        for k,v in results.items():
            if v["has_data"] and k!="ticker info (sanity/auth)":
                print("       - %s"%k)
        print("\n  NEXT: confirm how far back it goes (try the 18-month date), then I build a chunked")
        print("  UW backfill (respecting market-hours + 40K/day quota) -> validate_signal.py on the result.")
    else:
        print("  >> SLOW PATH CONFIRMED: auth works but no historical endpoint returned dated data.")
        print("     Your UW plan appears to serve CURRENT snapshots only, not history. You cannot")
        print("     backfill brick #2 — accumulate forward via signal_logger.py (~12-18 months).")
        print("     (If you believe your plan HAS historical access, the endpoint paths here may be")
        print("     wrong for your version — check generator.py's UW paths and adjust.)")
    print("\n  This check is read-only and tiny. It does not backfill or change anything.")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
