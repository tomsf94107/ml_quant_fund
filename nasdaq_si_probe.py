#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — NASDAQ HISTORICAL SHORT-INTEREST PROBE
================================================================================
Goal: find out HOW FAR BACK the FREE nasdaq.com short-interest endpoint actually
goes, BEFORE building a full historical pull. The question the whole pre-2021 plan
hinges on: does the free feed reach the genuinely-normal 2013-2019 regime (and 2008),
or is it just a rolling ~12-24 month window (in which case it's useless for extending
history and we're back to bank-SI-on-the-abnormal-window or pay for NYSE/Nasdaq FTP)?

HONEST UNCERTAINTY: the endpoint below is constructed from knowledge of nasdaq.com's
site API and is NOT verified here (sandbox can't reach nasdaq.com). It may 404, return
a different JSON shape, or be blocked from datacenter IPs (should work from your Mac /
residential IP). On ANY failure this prints the HTTP status + raw response head so we
can fix the parser iteratively (same way the FINRA fetcher got sorted).

WHAT IT DOES: for each --symbol, GET the short-interest JSON, parse the rows
(settlementDate, daysToCover), and report: #records, earliest date, latest date, and a
few samples. Then a verdict on whether the free depth is enough to bother with.

USAGE:
  python nasdaq_si_probe.py
  python nasdaq_si_probe.py --symbols NVDA,AMD,AMAT,MU,INTC
================================================================================
"""
import argparse, sys, json, time, datetime, urllib.request, urllib.error

# nasdaq.com site API (unverified here). Common shape:
#   https://api.nasdaq.com/api/quote/{SYMBOL}/short-interest?assetClass=stocks
ENDPOINT="https://api.nasdaq.com/api/quote/{sym}/short-interest?assetClass=stocks"
BROWSER_HEADERS={
    "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Accept":"application/json, text/plain, */*",
    "Accept-Language":"en-US,en;q=0.9",
    "Origin":"https://www.nasdaq.com",
    "Referer":"https://www.nasdaq.com/",
}
LINE="="*78

def _http_get(url, timeout=30, tries=3):
    """Returns (status, text). status=0 on network error."""
    last=None
    for attempt in range(tries):
        try:
            req=urllib.request.Request(url, headers=BROWSER_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw=r.read()
                if "gzip" in r.headers.get("Content-Encoding",""):
                    import gzip; raw=gzip.decompress(raw)
                time.sleep(0.4)
                return r.status, raw.decode("utf-8","replace")
        except urllib.error.HTTPError as e:
            last=e
            try: body=e.read().decode("utf-8","replace")
            except Exception: body=""
            if e.code in (429,403,503): time.sleep(1.5*(attempt+1)); continue
            return e.code, body
        except Exception as e:
            last=e; time.sleep(0.6*(attempt+1))
    return 0, "network error: %s"%repr(last)[:160]

def parse_rows(data):
    """Be defensive about where the rows live and what the fields are called.
    Returns list of (settlement_date:date, days_to_cover:float|None) and the raw rows."""
    rows=None
    # walk likely locations
    try: rows=data["data"]["shortInterestTable"]["rows"]
    except Exception: rows=None
    if rows is None:
        try: rows=data["data"]["rows"]
        except Exception: rows=None
    if rows is None:
        # last resort: find any list of dicts containing a date-ish + cover-ish key
        def find_rows(o):
            if isinstance(o,list) and o and isinstance(o[0],dict):
                keys=set(k.lower() for k in o[0].keys())
                if any("settl" in k or "date" in k for k in keys): return o
            if isinstance(o,dict):
                for v in o.values():
                    r=find_rows(v)
                    if r: return r
            return None
        rows=find_rows(data)
    if not rows: return [], rows
    out=[]
    for r in rows:
        if not isinstance(r,dict): continue
        # date field
        dval=None
        for k in r:
            if "settl" in k.lower() or k.lower()=="date":
                dval=r[k]; break
        # days-to-cover field
        dtc=None
        for k in r:
            if "cover" in k.lower() or "daystocover" in k.lower().replace(" ",""):
                dtc=r[k]; break
        do=None
        if dval:
            for fmt in ("%m/%d/%Y","%Y-%m-%d","%m-%d-%Y","%d-%b-%Y"):
                try: do=datetime.datetime.strptime(str(dval).strip(),fmt).date(); break
                except Exception: pass
        dc=None
        if dtc not in (None,"","--","N/A"):
            try: dc=float(str(dtc).replace(",",""))
            except Exception: dc=None
        if do: out.append((do,dc))
    out.sort()
    return out, rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--symbols",default="NVDA,AMD,AMAT,MU,INTC,PLTR")
    a=ap.parse_args()
    syms=[s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    print("\n"+LINE+"\nNASDAQ HISTORICAL SHORT-INTEREST PROBE\n"+LINE)
    print("  endpoint: %s"%ENDPOINT.format(sym="<SYM>"))
    print("  probing %d symbols. Goal: how far back does the FREE feed reach?\n"%len(syms))
    depths=[]
    for sym in syms:
        url=ENDPOINT.format(sym=sym)
        status,text=_http_get(url)
        if status!=200:
            print("  %-6s HTTP %s"%(sym,status))
            head=text[:300].replace("\n"," ")
            print("         response head: %s"%head)
            if status in (403,0):
                print("         -> blocked or unreachable. (May work from your residential IP; if not,")
                print("            the free endpoint is gated and we pivot to bank-or-pay.)")
            continue
        try: data=json.loads(text)
        except Exception:
            print("  %-6s got non-JSON (%d chars). head: %s"%(sym,len(text),text[:200].replace("\n"," "))); continue
        rows,raw=parse_rows(data)
        if not rows:
            print("  %-6s parsed 0 rows. Raw structure head (paste this back so I can fix the parser):"%sym)
            print("         %s"%json.dumps(data)[:400]); continue
        earliest=rows[0][0]; latest=rows[-1][0]; span_yr=(latest-earliest).days/365.25
        depths.append(earliest)
        print("  %-6s %d records | %s .. %s  (%.1f yrs)"%(sym,len(rows),earliest,latest,span_yr))
        # samples
        for do,dc in rows[:2]+rows[-1:]:
            print("           %s  days_to_cover=%s"%(do,dc))

    print("\n"+LINE+"\nVERDICT — is the free Nasdaq feed deep enough?\n"+LINE)
    if depths:
        oldest=min(depths)
        print("  oldest settlement reached: %s"%oldest)
        if oldest.year<=2016:
            print("  >> DEEP ENOUGH: reaches pre-COVID normal regime (2013-2019) and possibly 2008.")
            print("     Worth building the full Nasdaq-listed pull + re-validating SI on the normal window.")
        elif oldest.year<=2019:
            print("  >> PARTIAL: reaches some pre-COVID (2017-2019) but not 2008/2015. Still a useful read")
            print("     on whether SI works outside the meme/AI window. Marginal -- your call.")
        else:
            print("  >> TOO SHALLOW: only ~%d back. The free endpoint is a rolling window, not history."%oldest.year)
            print("     This path is dead -- pre-2020 exchange-listed SI needs Nasdaq/NYSE FTP (paid) or")
            print("     Compustat (WRDS). Recommend banking SI on the abnormal-window estimate instead.")
    else:
        print("  >> NO DATA RETRIEVED. Either the endpoint is wrong/blocked (paste the response heads above")
        print("     and I'll fix it) or the free feed is gated. If blocked from your Mac too, pivot to")
        print("     bank-or-pay.")
    print("\n  This is a probe -- nothing committed until we see the depth.")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
