#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — FUNDAMENTALS FETCHER: GROSS PROFITABILITY (brick #3, axis 3)
================================================================================
Gross profitability (Novy-Marx 2013): GP = (Revenue - COGS) / Total Assets. One of the
FEW documented anomalies that has NOT decayed post-publication, and derived from
financial statements -> a genuinely different axis from price/positioning (the
orthogonality idio-vol showed is achievable, but on a variable with STEADY edge).

SOURCE (FREE, no API key; SEC requires a User-Agent with your email):
  ticker -> CIK (cached company_tickers.json, likely already from the Lazy Prices crawl)
  per CIK: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
  -> all XBRL facts. We extract ANNUAL (form 10-K, fp FY) Revenue, COGS, Assets, each
     with its SEC 'filed' date (PIT = when it became public). GP computed per fiscal year.

TAG ROBUSTNESS: filers use different us-gaap tags. We try fallback lists:
  Revenue : RevenueFromContractWithCustomerExcludingAssessedTax, Revenues,
            SalesRevenueNet, RevenueFromContractWithCustomerIncludingAssessedTax
  COGS    : CostOfGoodsAndServicesSold, CostOfRevenue, CostOfGoodsSold
  Assets  : Assets
For each fiscal year we take the first tag that has a value; GP = (Rev - COGS)/Assets.

PIT: stored filing_date = the SEC 'filed' date of the Assets figure (the full 10-K is
public then). The validator will use returns strictly AFTER that date. No look-ahead.

OUTPUT: fundamentals.db, table gross_profitability(ticker, fiscal_year, period_end,
filed_date, revenue, cogs, assets, gp). Cached + resumable.

SMOKE TEST FIRST:
  python fundamentals_fetch.py --root . --email "you@example.com" --max-tickers 5
Full:
  python fundamentals_fetch.py --root . --email "you@example.com"

IMPORTANT: pass a REAL --email (SEC blocks generic/no User-Agent).
================================================================================
"""
import argparse, os, sys, json, time, sqlite3, datetime, urllib.request, urllib.error
from collections import defaultdict

UA_TMPL="ML-Quant-Research/1.0 ({email})"
TICKERS_URL="https://www.sec.gov/files/company_tickers.json"
FACTS_URL="https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
LINE="="*78

REVENUE_TAGS=["RevenueFromContractWithCustomerExcludingAssessedTax","Revenues",
              "SalesRevenueNet","RevenueFromContractWithCustomerIncludingAssessedTax"]
COGS_TAGS=["CostOfGoodsAndServicesSold","CostOfRevenue","CostOfGoodsSold"]
ASSET_TAGS=["Assets"]

def _http_get(url, email, want_json=False, sleep=0.15, timeout=60, tries=4):
    headers={"User-Agent":UA_TMPL.format(email=email),"Accept-Encoding":"gzip, deflate"}
    last=None
    for attempt in range(tries):
        try:
            req=urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw=r.read()
                if "gzip" in r.headers.get("Content-Encoding",""):
                    import gzip; raw=gzip.decompress(raw)
                time.sleep(sleep)
                txt=raw.decode("utf-8","replace")
                return json.loads(txt) if want_json else txt
        except urllib.error.HTTPError as e:
            last=e
            if e.code in (429,403,503): time.sleep(1.5*(attempt+1)); continue
            if e.code==404: return None
            time.sleep(0.5*(attempt+1))
        except Exception as e:
            last=e; time.sleep(0.5*(attempt+1))
    sys.stderr.write("  [fetch fail] %s (%s)\n"%(url,repr(last)[:100]))
    return None

def nd(s):
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None

def load_cik_map(email, cachedir, sleep):
    cf=os.path.join(cachedir,"company_tickers.json")
    data=None
    if os.path.isfile(cf):
        try: data=json.load(open(cf))
        except Exception: data=None
    if data is None:
        data=_http_get(TICKERS_URL,email,want_json=True,sleep=sleep)
        if data is None: return {}
        try: json.dump(data,open(cf,"w"))
        except Exception: pass
    out={}
    itr=data.values() if isinstance(data,dict) else data
    for row in itr:
        try: out[str(row["ticker"]).upper()]="%010d"%int(row["cik_str"])
        except Exception: continue
    return out

def annual_by_fy(facts, tags):
    """From companyfacts, return {fiscal_year: (val, end_date, filed_date)} for the first
    matching tag, ANNUAL (form 10-K, fp FY) figures only."""
    gaap=facts.get("facts",{}).get("us-gaap",{})
    for tag in tags:
        node=gaap.get(tag)
        if not node: continue
        units=node.get("units",{})
        arr=units.get("USD") or next(iter(units.values()),None)
        if not arr: continue
        out={}
        for item in arr:
            form=item.get("form",""); fp=item.get("fp","")
            if form!="10-K" or fp!="FY": continue
            fy=item.get("fy"); val=item.get("val"); end=item.get("end"); filed=item.get("filed")
            if fy is None or val is None: continue
            # keep the latest-filed value for each fiscal year (handles restatements -> use as-filed earliest? we keep original by min filed)
            if fy not in out or (filed and filed<out[fy][2]):
                out[fy]=(float(val),end,filed)
        if out: return out
    return {}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--email",default=None)
    ap.add_argument("--out-db",default=None)
    ap.add_argument("--cache-dir",default=None)
    ap.add_argument("--sleep",type=float,default=0.15)
    ap.add_argument("--max-tickers",type=int,default=None)
    ap.add_argument("--since-year",type=int,default=2010)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    out_db=a.out_db or os.path.join(a.root,"fundamentals.db")
    cachedir=a.cache_dir or os.path.join(a.root,"cache","edgar"); os.makedirs(cachedir,exist_ok=True)
    factsdir=os.path.join(cachedir,"facts"); os.makedirs(factsdir,exist_ok=True)
    print("\n"+LINE+"\nFUNDAMENTALS FETCHER — gross profitability (Novy-Marx)\n"+LINE)
    if not a.email or "@" not in a.email:
        print("  [STOP] --email REQUIRED (SEC blocks requests without a real User-Agent email)."); return

    # universe
    uni=[]
    for name,table,col in (("short_interest.db","short_interest","ticker"),
                           ("prices.db","daily_prices","ticker")):
        p=os.path.join(a.root,name)
        if os.path.isfile(p):
            try:
                c=sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro",uri=True)
                uni=sorted(set(r[0].upper() for r in c.execute('SELECT DISTINCT %s FROM %s'%(col,table)) if r[0])); c.close()
                if uni: 
                    print("  universe: %d tickers (from %s)"%(len(uni),name)); break
            except Exception: continue
    if not uni: print("  [STOP] no universe found."); return
    if a.max_tickers: uni=uni[:a.max_tickers]; print("  [SMOKE TEST] first %d"%len(uni))

    cikmap=load_cik_map(a.email,cachedir,a.sleep)
    if not cikmap: print("  [STOP] could not load ticker->CIK map."); return
    print("  ticker->CIK: %d names\n"%len(cikmap))

    db=sqlite3.connect(out_db,timeout=30)
    db.execute("""CREATE TABLE IF NOT EXISTS gross_profitability(
        ticker TEXT, fiscal_year INTEGER, period_end TEXT, filed_date TEXT,
        revenue REAL, cogs REAL, assets REAL, gp REAL,
        PRIMARY KEY(ticker, fiscal_year))""")
    db.commit()
    done=set((r[0],r[1]) for r in db.execute("SELECT ticker,fiscal_year FROM gross_profitability"))

    n_tk=0; n_rows=0; n_miss=0
    for tk in uni:
        cik10=cikmap.get(tk)
        if not cik10: n_miss+=1; continue
        ff=os.path.join(factsdir,"%s.json"%cik10); facts=None
        if os.path.isfile(ff):
            try: facts=json.load(open(ff))
            except Exception: facts=None
        if facts is None:
            facts=_http_get(FACTS_URL.format(cik10=cik10),a.email,want_json=True,sleep=a.sleep)
            if facts is not None:
                try: json.dump(facts,open(ff,"w"))
                except Exception: pass
        if facts is None:
            print("  %-6s CIK %s: no companyfacts"%(tk,cik10)); continue
        rev=annual_by_fy(facts,REVENUE_TAGS); cogs=annual_by_fy(facts,COGS_TAGS); ast=annual_by_fy(facts,ASSET_TAGS)
        wrote=0
        for fy in sorted(set(rev)&set(cogs)&set(ast)):
            if fy<a.since_year: continue
            rv,_,rfiled=rev[fy]; cg,_,_=cogs[fy]; (av,aend,afiled)=ast[fy]
            if av<=0: continue
            gp=(rv-cg)/av
            filed=afiled or rfiled
            if (tk,fy) in done: continue
            db.execute("""INSERT INTO gross_profitability(ticker,fiscal_year,period_end,filed_date,revenue,cogs,assets,gp)
                          VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(ticker,fiscal_year) DO UPDATE SET
                          period_end=excluded.period_end, filed_date=excluded.filed_date,
                          revenue=excluded.revenue, cogs=excluded.cogs, assets=excluded.assets, gp=excluded.gp""",
                       (tk,fy,aend,filed,rv,cg,av,gp))
            wrote+=1; n_rows+=1
        db.commit(); n_tk+=1
        print("  %-6s CIK %s: %d fiscal-years of GP (%d new)"%(tk,cik10,len(set(rev)&set(cogs)&set(ast)),wrote))

    tot=db.execute("SELECT COUNT(*),COUNT(DISTINCT ticker),MIN(fiscal_year),MAX(fiscal_year) FROM gross_profitability").fetchone()
    print("\n"+LINE)
    print("  processed %d tickers | %d GP rows | %d w/o CIK"%(n_tk,n_rows,n_miss))
    print("  fundamentals.db: %d rows, %d tickers, FY %s..%s"%(tot[0],tot[1],tot[2],tot[3]))
    # quick sanity: GP distribution
    gps=[r[0] for r in db.execute("SELECT gp FROM gross_profitability WHERE gp IS NOT NULL")]
    if gps:
        import statistics as st
        gps.sort(); q=lambda p: gps[min(len(gps)-1,int(p*len(gps)))]
        print("  GP distribution: p10=%.3f median=%.3f p90=%.3f (Novy-Marx typical ~0.2-0.4)"%(q(0.1),st.median(gps),q(0.9)))
    db.close()
    print("\n  Next: a GP validator (monthly cross-sections, PIT on filed_date) -- built once this lands.")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted (cached -- safe to re-run).")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
