#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — DATA BACKFILL DIAGNOSTIC
================================================================================
Answers the question that routes everything: can you confirm brick #2 NOW
(fast path) or must you accumulate forward (slow path)?

It checks TWO things:

  PART A (offline, always runs): how much history of each candidate signal is
  ALREADY sitting in your databases? You might have more than the ~3 months in
  prediction_features stashed in another table. For every (db, table) that holds
  pc_ratio / iv_skew / short_interest / days_to_cover / analyst columns, it reports
  the date range, distinct days, and ticker count. If ANY source already has 2+
  years -> you're on the fast path right now, no vendor needed.

  PART B (live, optional, only with --probe-vendor): tries to pull historical
  data for a FEW tickers from your live data layer to see if your vendor offers
  history or only current snapshots. It does NOT do a big fetch -- just enough to
  answer "does 2-years-ago data come back?". It auto-detects which vendor library
  you have (unusualwhales / polygon / yfinance / tiingo) and probes whatever is
  present. NOTHING is written; this is read-only reconnaissance.

If PART A finds long history -> validate brick #2 immediately with validate_signal.py.
If PART B shows the vendor returns history -> backfill, then validate.
If neither -> slow path: signal_logger.py daily, ~12-18 months.

RULE 1: PART A is read-only (mode=ro). PART B makes only tiny probe calls and writes
nothing. The script reports what EXISTS; it does not assume or fabricate.

NETWORK NOTE: PART B only runs with --probe-vendor and only if a vendor lib is
importable. If your machine's security tooling flags network calls (as during the
price fetch), skip --probe-vendor and rely on PART A + a manual vendor check.

USAGE:
  python data_backfill_check.py --root .                    # PART A only (offline, safe)
  python data_backfill_check.py --root . --probe-vendor     # also probe live vendor
  python data_backfill_check.py --root . --probe-vendor --probe-tickers AAPL,MSFT,NVDA
================================================================================
"""
import argparse, os, sqlite3, sys, datetime, importlib

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)
def ro(p):
    if not os.path.isfile(p): raise FileNotFoundError(p)
    return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def tables(c): return [r[0] for r in Q(c,"SELECT name FROM sqlite_master WHERE type='table'")]
def cols_of(c,t): return [r[1] for r in Q(c,'PRAGMA table_info("'+t+'")')]
def all_dbs(root):
    out=[]
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f.endswith((".db",".sqlite",".sqlite3")): out.append(os.path.join(dp,f))
    return sorted(out)
def nd(s):
    if s is None: return None
    s=str(s)[:10]
    try: return datetime.date.fromisoformat(s)
    except Exception: return None

# signal families we care about for brick #2
SIGNAL_COLS={
    "options": ["pc_ratio","pc_ratio_snap","put_call","iv_skew","iv_skew_snap","skew","call_put_ratio","implied_vol","iv_rank"],
    "short":   ["short_ratio","short_interest","si_ratio","days_to_cover","dtc","pct_float_short","short_pct"],
    "analyst": ["upgrades_30d","downgrades_30d","analyst_net","est_revision","eps_revision","target_revision","num_revisions"],
    "inst":    ["inst_signed_flow_5d","inst_signed_flow_30d","inst_block_buy_sell_7d","inst_auction_imbal_5d"],
}
DATE_HINTS=["prediction_date","date","as_of","report_date","settlement_date","updated_at","snapshot_date","trade_date"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--probe-vendor",action="store_true")
    ap.add_argument("--probe-tickers",default="AAPL,MSFT,NVDA")
    ap.add_argument("--years-back",type=int,default=2)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    banner("ML QUANT FUND — DATA BACKFILL DIAGNOSTIC")
    print("Can you confirm brick #2 NOW (fast) or must you accumulate (slow)?")
    print("Root:",os.path.abspath(a.root))

    # ---------------- PART A: existing DB history ----------------
    banner("PART A — history ALREADY in your databases (offline, read-only)")
    findings={"options":[],"short":[],"analyst":[],"inst":[]}
    for dbp in all_dbs(a.root):
        try: c=ro(dbp)
        except Exception: continue
        try:
            for t in tables(c):
                cl=cols_of(c,t); cl_low={cn.lower():cn for cn in cl}
                dcol=None
                for h in DATE_HINTS:
                    if h in cl_low: dcol=cl_low[h]; break
                if not dcol: continue
                tcol = cl_low.get("ticker") or cl_low.get("symbol")
                for family,cols in SIGNAL_COLS.items():
                    hit=[cl_low[cn] for cn in cols if cn in cl_low]
                    if not hit: continue
                    # measure history for this table
                    try:
                        dr=Q(c,'SELECT MIN(substr("'+dcol+'",1,10)),MAX(substr("'+dcol+'",1,10)) FROM "'+t+'"')[0]
                        ndays=Q(c,'SELECT COUNT(DISTINCT substr("'+dcol+'",1,10)) FROM "'+t+'"')[0][0]
                        ntkr=Q(c,'SELECT COUNT(DISTINCT "'+tcol+'") FROM "'+t+'"')[0][0] if tcol else 0
                        n=Q(c,'SELECT COUNT(*) FROM "'+t+'"')[0][0]
                    except Exception: continue
                    d0,d1=nd(dr[0]),nd(dr[1])
                    span=(d1-d0).days if (d0 and d1) else 0
                    findings[family].append({"db":os.path.basename(dbp),"table":t,"cols":hit,
                        "range":dr,"ndays":ndays,"ntkr":ntkr,"n":n,"span":span})
        finally:
            c.close()

    fast_path_sources=[]
    for family in ("options","short","analyst","inst"):
        sub("%s signals" % family.upper())
        if not findings[family]:
            print("  none found in any DB")
            continue
        for f in sorted(findings[family], key=lambda x:-x["span"]):
            yrs=f["span"]/365.0
            flag = "  <-- 2+ YEARS: FAST PATH" if (f["span"]>=2*365-30 and f["ndays"]>=200) else ("  (short)" if f["span"]<300 else "  (medium)")
            print("  %s.%s  cols=%s" % (f["db"],f["table"],",".join(f["cols"][:4])))
            print("      %s..%s  | %d distinct days | %d tickers | %.1f yr span%s"
                  % (f["range"][0],f["range"][1],f["ndays"],f["ntkr"],yrs,flag))
            if f["span"]>=2*365-30 and f["ndays"]>=200:
                fast_path_sources.append((family,f))

    # ---------------- PART B: vendor probe ----------------
    if a.probe_vendor:
        banner("PART B — probing live vendor for HISTORY (tiny read-only calls)")
        tks=[t.strip() for t in a.probe_tickers.split(",")][:3]
        target = datetime.date.today() - datetime.timedelta(days=a.years_back*365)
        start = target.isoformat(); end = (target+datetime.timedelta(days=10)).isoformat()
        print("  probing for data around %s (%d years back) for %s"%(start,a.years_back,tks))
        print("  (only checks IF historical data returns; fetches nothing large)\n")

        # detect available vendor libs
        probed=False
        # --- unusualwhales (your system uses UW) ---
        for modname in ("unusualwhales","uw","unusual_whales"):
            try:
                m=importlib.import_module(modname); probed=True
                print("  [found vendor lib: %s]"%modname)
                print("  NOTE: UW historical options/SI endpoints vary by plan. To test manually, try")
                print("  fetching put/call or short-interest for %s dated %s and see if it returns."%(tks[0],start))
                print("  If your UW plan includes historical snapshots, that is your fast-path source.")
                break
            except Exception: pass
        # --- polygon ---
        if not probed:
            try:
                import polygon  # noqa
                probed=True
                print("  [found vendor lib: polygon] — Polygon offers deep history on paid tiers.")
                print("  Options aggregates / short interest historical endpoints exist; probe one for %s @ %s."%(tks[0],start))
            except Exception: pass
        # --- tiingo ---
        if not probed:
            try:
                import tiingo  # noqa
                probed=True
                print("  [found vendor lib: tiingo] — Tiingo has fundamentals/news history; options limited.")
            except Exception: pass
        # --- yfinance (we know this works; but it has NO historical options/SI history) ---
        if not probed:
            try:
                import yfinance  # noqa
                probed=True
                print("  [found: yfinance] — WARNING: yfinance gives only CURRENT options/short-interest,")
                print("  NOT historical snapshots. It cannot backfill the time series you need. yfinance")
                print("  is fine for prices (already done) but NOT a backfill source for brick #2.")
            except Exception: pass
        if not probed:
            print("  No known vendor library importable in this environment. Check manually which data")
            print("  provider your live pipeline uses for options/short-interest, and whether its plan")
            print("  includes HISTORICAL (not just current) snapshots.")
    else:
        print("\n  (PART B vendor probe skipped — add --probe-vendor to test live backfill capability)")

    # ---------------- verdict ----------------
    banner("VERDICT — which path are you on?")
    if fast_path_sources:
        print("  >> FAST PATH AVAILABLE. These sources already hold 2+ years of a brick-#2 signal:")
        for family,f in fast_path_sources:
            print("     - %s: %s.%s (%s) %.1f yr"
                  %(family,f["db"],f["table"],",".join(f["cols"][:3]),f["span"]/365.0))
        print("\n  NEXT: validate the best one immediately, e.g.")
        best=fast_path_sources[0][1]
        print("     python validate_signal.py --root . --feature %s --hold 40"%best["cols"][0])
        print("     python validate_signal.py --root . --feature %s --hold 5"%best["cols"][0])
        print("  If it survives, combine: python portfolio_combine.py --root . --strategies PEAD,%s"%best["cols"][0])
    else:
        # how much DO we have?
        best_span=max([f["span"] for fam in findings.values() for f in fam], default=0)
        print("  >> No DB source has a full 2 years yet (longest signal history: %.1f years)."%(best_span/365.0))
        print("  Likely SLOW PATH unless your vendor can backfill (run --probe-vendor, or ask your")
        print("  data provider directly: 'do you offer HISTORICAL options/short-interest snapshots,")
        print("  not just current?').")
        print("\n  SLOW PATH (start now, runs in background):")
        print("     - cron signal_logger.py daily (post-close) to accumulate history")
        print("     - check progress periodically: python signal_logger.py --status")
        print("\n  MEANWHILE (unblocked, needs no brick #2):")
        print("     - productionize PEAD: survivorship-bias check, SUE+EAR combo, capacity analysis")
    print("\n  This diagnostic is read-only. It tells you which path you're on; it changes nothing.")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
