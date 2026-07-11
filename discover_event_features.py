#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — EVENT-FEATURE DISCOVERY
================================================================================
Before we can extend the feature matrix with the signal-bearing event families
(post_earnings, eps_surprise, eightk_filings, rev_growth — the Phase-1 FDR
survivors that are NOT in prediction_features), we must find out where that data
lives and in what form. This script answers exactly that.

It scans earnings.db, sec_filings.db, fundamentals.db, earnings_calendar, and any
other DB whose tables look earnings/filing/fundamental-related, and for each:
  * prints the schema (columns + types) and row count
  * shows a few sample rows
  * reports date range and ticker coverage
  * CLASSIFIES whether the table is JOIN-READY (has ticker + date + a usable value
    column) or RAW (would need feature engineering to become a per-row signal)
  * specifically hunts for columns matching the four target families

OUTPUT TELLS YOU which of two paths the next build takes:
  PATH A (easy)  -> features stored as ticker/date/value -> a JOIN extends the matrix
  PATH B (work)  -> only raw events/expressions exist    -> reconstruct then join

READ-ONLY. Never writes. SQLite opened mode=ro&immutable=1.

USAGE (project root, env active):
  python discover_event_features.py --root .
  add --out event_discovery.json to save machine-readable results.
================================================================================
"""
import argparse, os, sqlite3, sys, json, datetime
from collections import defaultdict

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)
def ro(p):
    if not os.path.isfile(p): raise FileNotFoundError(p)
    return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=20)
def q(c,s,p=()): return c.execute(s,p).fetchall()
def tables(c): return [r[0] for r in q(c,"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
def cols_of(c,t): return [(r[1],r[2]) for r in q(c,'PRAGMA table_info("'+t+'")')]

# what we're hunting for
TARGET_FAMILIES = {
    "post_earnings": ["post_earn","postearn","pead","earnings_drift","days_since_earn","earn_drift"],
    "eps_surprise":  ["eps_surprise","eps_surp","surprise","sue","earnings_surprise","actual_eps","estimate_eps","consensus"],
    "eightk_filings":["eightk","8k","8-k","item_","filing_count","num_filings","filings_30d"],
    "rev_growth":    ["rev_growth","revenue_growth","rev_yoy","rev_qoq","sales_growth","revenue","total_revenue"],
    "earnings_date": ["earnings_date","report_date","earn_date","announce","fiscal_period","period_end","event_date"],
}

# columns that indicate join-readiness
TICKER_HINTS=["ticker","symbol","cik","permno"]
DATE_HINTS=["date","asof","as_of","report_date","filing_date","earnings_date","period","event_date","announce","fiscal"]

# DBs / table-name patterns that are earnings/filing/fundamental-related
DB_PATTERNS=["earnings","sec","filing","fundamental","financ","xbrl","estimate","analyst","calendar"]
TABLE_PATTERNS=["earn","filing","8k","eightk","fundamental","xbrl","financ","income","revenue",
                "eps","surprise","estimate","analyst","calendar","report","statement"]

def tokenize(nm):
    import re
    return set(re.split(r'[_\W]+', nm.lower()))

def find_candidate_dbs(root):
    """All .db files, prioritizing earnings/filing/fundamental-named ones, but scanning all."""
    dbs=[]
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f.endswith((".db",".sqlite",".sqlite3")):
                full=os.path.join(dp,f)
                priority = any(p in f.lower() for p in DB_PATTERNS)
                dbs.append((priority, full))
    # priority DBs first
    dbs.sort(key=lambda x: (not x[0], x[1]))
    return [full for _,full in dbs]

def classify_table(c, t):
    cols=cols_of(c,t)
    names=[cn.lower() for cn,_ in cols]
    has_ticker=any(any(h in tokenize(n) or h in n for h in TICKER_HINTS) for n in names)
    has_date=any(any(h in n for h in DATE_HINTS) for n in names)
    # find target-family column matches
    fam_hits=defaultdict(list)
    for fam,hints in TARGET_FAMILIES.items():
        for n in names:
            if any(h in n for h in hints):
                fam_hits[fam].append(n)
    # numeric value columns (potential signal values)
    val_cols=[cn for cn,ct in cols if (ct or "").upper() in ("REAL","FLOAT","NUMERIC","INTEGER","INT","DOUBLE")
              and cn.lower() not in TICKER_HINTS
              and not any(h in cn.lower() for h in DATE_HINTS)
              and cn.lower() not in ("id","rowid")]
    join_ready = has_ticker and has_date and (len(val_cols)>0 or len(fam_hits)>0)
    return {"cols":cols,"has_ticker":has_ticker,"has_date":has_date,
            "fam_hits":dict(fam_hits),"val_cols":val_cols,"join_ready":join_ready}

def looks_relevant(t):
    tl=t.lower()
    return any(p in tl for p in TABLE_PATTERNS)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--all-tables",action="store_true",help="show every table, not just earnings/filing-related")
    ap.add_argument("--out",default=None)
    args=ap.parse_args(); args.root=os.path.expanduser(args.root)

    banner("ML QUANT FUND — EVENT-FEATURE DISCOVERY")
    print("Read-only. Hunting for: post_earnings, eps_surprise, eightk_filings, rev_growth")
    print("Root:",os.path.abspath(args.root),"| Python",sys.version.split()[0])

    dbs=find_candidate_dbs(args.root)
    print("\nScanning %d databases (earnings/filing/fundamental-named first)..."%len(dbs))

    report={"databases":[],"family_locations":defaultdict(list)}
    join_ready_tables=[]; raw_tables=[]

    for dbpath in dbs:
        try:
            c=ro(dbpath)
        except Exception as e:
            continue
        try:
            ts=tables(c)
            # only dig into relevant tables unless --all-tables
            relevant=[t for t in ts if looks_relevant(t)] if not args.all_tables else ts
            if not relevant:
                c.close(); continue
            base=os.path.basename(dbpath)
            shown_db=False
            for t in relevant:
                info=classify_table(c,t)
                # skip tables with no ticker AND no family hits (not useful)
                if not info["has_ticker"] and not info["fam_hits"]:
                    continue
                if not shown_db:
                    banner("DB: %s" % base); shown_db=True
                try:
                    n=q(c,'SELECT COUNT(*) FROM "'+t+'"')[0][0]
                except Exception:
                    n=-1
                tag = "JOIN-READY" if info["join_ready"] else "RAW (needs engineering)"
                sub("%s.%s  [%s]  rows=%d" % (base,t,tag,n))
                print("  columns:")
                for cn,ct in info["cols"]:
                    mark=""
                    if cn.lower() in TICKER_HINTS: mark=" <-ticker"
                    elif any(h in cn.lower() for h in DATE_HINTS): mark=" <-date"
                    print("     %-26s %-10s%s" % (cn,ct,mark))
                if info["fam_hits"]:
                    print("  *** TARGET-FAMILY COLUMN MATCHES ***")
                    for fam,hits in info["fam_hits"].items():
                        print("      %-16s -> %s" % (fam,hits))
                        report["family_locations"][fam].append({"db":base,"table":t,"columns":hits})
                if info["val_cols"]:
                    print("  numeric value columns: %s" % info["val_cols"][:12])
                # date range + ticker count if possible
                tcol=next((cn for cn,_ in info["cols"] if cn.lower() in TICKER_HINTS),None)
                dcol=next((cn for cn,_ in info["cols"] if any(h in cn.lower() for h in DATE_HINTS)),None)
                if dcol:
                    try:
                        dr=q(c,'SELECT MIN("'+dcol+'"),MAX("'+dcol+'") FROM "'+t+'"')[0]
                        print("  date range [%s]: %s .. %s" % (dcol,dr[0],dr[1]))
                    except Exception: pass
                if tcol:
                    try:
                        nt=q(c,'SELECT COUNT(DISTINCT "'+tcol+'") FROM "'+t+'"')[0][0]
                        print("  distinct tickers [%s]: %d" % (tcol,nt))
                    except Exception: pass
                # sample rows
                try:
                    rows=q(c,'SELECT * FROM "'+t+'" LIMIT 2')
                    cnames=[cn for cn,_ in info["cols"]]
                    print("  sample:")
                    for r in rows:
                        tr=tuple((str(x)[:18]+"..") if x is not None and len(str(x))>18 else x for x in r)
                        print("     ", tr if len(cnames)>8 else dict(zip(cnames,tr)))
                except Exception as e:
                    print("  [sample failed]",e)
                (join_ready_tables if info["join_ready"] else raw_tables).append((base,t,n))
                report["databases"].append({"db":base,"table":t,"rows":n,
                    "join_ready":info["join_ready"],"fam_hits":info["fam_hits"]})
        finally:
            c.close()

    # ---- verdict ----
    banner("VERDICT — which path does the feature-extension take?")
    print("\nTARGET FAMILIES FOUND:")
    for fam in TARGET_FAMILIES:
        locs=report["family_locations"].get(fam,[])
        if locs:
            where=", ".join("%s.%s(%s)"%(l["db"],l["table"],",".join(l["columns"])) for l in locs)
            print("  [FOUND]   %-16s -> %s" % (fam,where))
        else:
            print("  [MISSING] %-16s -> not found as a named column anywhere" % fam)

    print("\nJOIN-READY tables (PATH A — extend matrix by a direct join):")
    if join_ready_tables:
        for b,t,n in join_ready_tables: print("   %s.%s (rows=%d)" % (b,t,n))
    else:
        print("   none")
    print("\nRAW tables (PATH B — reconstruct features from these first):")
    if raw_tables:
        for b,t,n in raw_tables: print("   %s.%s (rows=%d)" % (b,t,n))
    else:
        print("   none")

    found_count=sum(1 for f in TARGET_FAMILIES if report["family_locations"].get(f))
    print("\n  >> %d of %d target families found as stored columns." % (found_count,len(TARGET_FAMILIES)))
    if found_count>=3 and join_ready_tables:
        print("  >> PATH A likely: most signal families are stored + join-ready. Next build = a JOIN")
        print("     that pulls them into prediction_features format, then re-run base_model.py.")
    elif found_count>=1:
        print("  >> MIXED: some families stored, some not. Next build = join what exists +")
        print("     reconstruct the rest from the RAW tables above.")
    else:
        print("  >> PATH B: families not stored as columns — they're alpha expressions scored in")
        print("     alpha_fitness_by_ticker. Next build = reconstruct from earnings/filing raw data.")
    print("\n  Paste this whole output back and I'll write the exact extraction script for your case.")

    if args.out:
        path=args.out
        if os.path.isdir(path) or path.endswith("/"): path=os.path.join(path,"event_discovery.json")
        # convert defaultdict for json
        report["family_locations"]=dict(report["family_locations"])
        with open(path,"w") as f: json.dump(report,f,indent=2,default=str)
        print("\n  [report written to %s]"%path)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
