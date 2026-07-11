#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — NON-PRICE SIGNAL HUNT  (research-backed brick #2 candidates)
================================================================================
Price signals (reversal/momentum/vol) are arbitraged out of your liquid universe
(confirmed: all dead). The research says the cross-sectional signals that STILL
work come from DIFFERENT mechanisms — and several map to data you have in
prediction_features / analyst_cache:

  RESEARCH-BACKED CANDIDATES (different mechanism => uncorrelated-with-PEAD potential):
    * options flow        (pc_ratio / pc_ratio_snap)        -- informed-trading signal
    * institutional flow  (inst_signed_flow_5d)             -- your monitor's flow
    * analyst revisions   (upgrades_30d - downgrades_30d)   -- ~7.6%/yr decile spread (Mill St)
    * any other numeric prediction_features columns          -- tested generically

For each candidate this measures, at the 40-day horizon (where PEAD lives, so a
combinable book):
  * standalone cross-sectional IC vs forward 40d return (from prices.db)
  * correlation to the PEAD/SUE signal  -> NEW BRICK vs duplicate

HONEST CAVEAT (baked into verdict): prediction_features / analyst_cache have only
~3 months of history (2026-03-24+). That's LOW statistical power. Results here are
EXPLORATORY (does anything look promising enough to get more data on?), NOT
confirmatory. A signal needs a much longer history before you'd trade it.

It auto-discovers prediction_features columns and the analyst_cache table, and
also tries to locate any short-interest table.

RULE 1: forward return from prices.db, strictly after the feature snapshot date.
Feature value as-of the snapshot (PIT). SUE denominator strictly trailing. The
short-history limitation is reported loudly, not hidden.

READ-ONLY. mode=ro&immutable=1. No network.

USAGE:
  python signal_hunt_nonprice.py --root .
  python signal_hunt_nonprice.py --root . --hold 40 --hold2 5
================================================================================
"""
import argparse, os, sqlite3, sys, math, json, datetime
from collections import defaultdict
try:
    import numpy as np; HAVE_NUMPY=True
except Exception: HAVE_NUMPY=False

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)
def ro(p):
    if not os.path.isfile(p): raise FileNotFoundError(p)
    return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def has_table(c,n): return bool(Q(c,"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(n,)))
def tables(c): return [r[0] for r in Q(c,"SELECT name FROM sqlite_master WHERE type='table'")]
def cols_of(c,t): return [(r[1],r[2]) for r in Q(c,'PRAGMA table_info("'+t+'")')]
def require(cond,msg):
    if not cond: print("  [STOP] "+msg); return False
    return True
def all_dbs(root):
    out=[]
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f.endswith((".db",".sqlite",".sqlite3")): out.append(os.path.join(dp,f))
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
def spearman(x,y):
    n=len(x)
    if n<5: return None
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return None
    return float(np.corrcoef(rx,ry)[0,1])

# columns to SKIP in prediction_features (these are PRICE/MACRO, already tested or not cross-sectional)
SKIP_COLS={"id","ticker","symbol","prediction_date","date","created_at","updated_at",
           "vix_close","yield_10y","fear_greed","dxy_ret","spy_ret","dxy_close","spy_close",
           "yield_2y","yield_curve","oil","gold","move_index","prob_up","prob_raw","signal",
           "tier","horizon","return_1d","return_5d","return_20d","rsi_14","macd","bb_pct","atr",
           "price","price_at_pred","close"}
# the research-backed PRIORITY signals (different mechanism)
PRIORITY={"pc_ratio","pc_ratio_snap","inst_signed_flow_5d","put_call","skew","iv_skew",
          "short_interest","si_ratio","days_to_cover","analyst_net","upgrades_30d","downgrades_30d"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--hold2",type=int,default=5)
    ap.add_argument("--min-names",type=int,default=15)
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    banner("ML QUANT FUND — NON-PRICE SIGNAL HUNT (research-backed candidates)")
    print("Hunts options/institutional/analyst signals at h=%d. EXPLORATORY (short history)."%a.hold)
    if not require(HAVE_NUMPY,"numpy required"): return
    if not require(os.path.isfile(prices_db),"prices.db not found"): return

    # prices for forward returns
    cp=ro(prices_db)
    try:
        rows=Q(cp,"SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL")
    finally:
        cp.close()
    px=defaultdict(list)
    for tk,d,p in rows:
        do=nd(d)
        if do is None: continue
        try: pf=float(p)
        except Exception: continue
        if pf>0: px[tk].append((do,pf))
    for tk in px: px[tk].sort()
    pos_of={tk:{d:i for i,(d,_) in enumerate(lst)} for tk,lst in px.items()}
    print("  prices loaded for %d tickers"%len(px))

    def fwd(tk,d,hold):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,6):
            c=d+datetime.timedelta(days=off)
            if c in idx: i=idx[c]; break
        if i is None: return None
        x=i+hold
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None

    # ---- locate prediction_features ----
    accp=find_db(a.root,"accuracy.db")
    feature_panels={}  # signame -> {(date,ticker): value}
    if accp and os.path.isfile(accp):
        c=ro(accp)
        try:
            if has_table(c,"prediction_features"):
                cols=cols_of(c,"prediction_features")
                numeric=[cn for cn,ct in cols if cn.lower() not in SKIP_COLS
                         and ("INT" in (ct or "").upper() or "REAL" in (ct or "").upper()
                              or "NUM" in (ct or "").upper() or "FLOA" in (ct or "").upper())]
                # always include priority cols if present even if typed oddly
                allcols=[cn for cn,_ in cols]
                for pc in PRIORITY:
                    if pc in allcols and pc not in numeric: numeric.append(pc)
                tcol="ticker" if "ticker" in allcols else ("symbol" if "symbol" in allcols else None)
                dcol="prediction_date" if "prediction_date" in allcols else ("date" if "date" in allcols else None)
                print("  prediction_features candidate columns: %s"%(numeric if numeric else "none"))
                if tcol and dcol and numeric:
                    sel=tcol+","+dcol+","+",".join('"'+cn+'"' for cn in numeric)
                    data=Q(c,"SELECT "+sel+" FROM prediction_features")
                    for row in data:
                        tk=row[0]; do=nd(row[1])
                        if do is None: continue
                        for k,cn in enumerate(numeric):
                            v=row[2+k]
                            if v is None: continue
                            try: fv=float(v)
                            except Exception: continue
                            feature_panels.setdefault("PF:"+cn,{})[(do,tk)]=fv
        finally:
            c.close()

    # ---- analyst_cache (revisions) ----
    for dbp in all_dbs(a.root):
        try: c=ro(dbp)
        except Exception: continue
        try:
            for t in tables(c):
                if "analyst" not in t.lower(): continue
                cl=[cn for cn,_ in cols_of(c,t)]
                has_up = "upgrades_30d" in cl; has_dn="downgrades_30d" in cl
                tcol="ticker" if "ticker" in cl else ("symbol" if "symbol" in cl else None)
                dcol=next((x for x in ("date","as_of","updated_at","snapshot_date","prediction_date") if x in cl),None)
                if has_up and has_dn and tcol and dcol:
                    data=Q(c,"SELECT "+tcol+","+dcol+",upgrades_30d,downgrades_30d FROM "+'"'+t+'"')
                    for tk,d,up,dn in data:
                        do=nd(d)
                        if do is None: continue
                        try: net=float(up or 0)-float(dn or 0)
                        except Exception: continue
                        feature_panels.setdefault("ANALYST_NET",{})[(do,tk)]=net
                    print("  analyst revisions from %s.%s: %d points"%(os.path.basename(dbp),t,len(feature_panels.get("ANALYST_NET",{}))))
        finally:
            c.close()

    # ---- locate short interest ----
    for dbp in all_dbs(a.root):
        try: c=ro(dbp)
        except Exception: continue
        try:
            for t in tables(c):
                cl=[cn for cn,_ in cols_of(c,t)]
                sicol=next((x for x in ("short_interest","si_ratio","days_to_cover","dtc","short_ratio","pct_float_short") if x in cl),None)
                tcol="ticker" if "ticker" in cl else ("symbol" if "symbol" in cl else None)
                dcol=next((x for x in ("date","as_of","report_date","settlement_date","updated_at") if x in cl),None)
                if sicol and tcol and dcol:
                    data=Q(c,"SELECT "+tcol+","+dcol+","+sicol+" FROM "+'"'+t+'"')
                    cnt=0
                    for tk,d,si in data:
                        do=nd(d)
                        if do is None or si is None: continue
                        try: fv=float(si)
                        except Exception: continue
                        # short interest predicts NEGATIVELY -> use -si so "high signal = high return"
                        feature_panels.setdefault("SHORT_INT_neg",{})[(do,tk)]=-fv; cnt+=1
                    if cnt: print("  short interest from %s.%s (%s): %d points"%(os.path.basename(dbp),t,sicol,cnt))
        finally:
            c.close()

    if not feature_panels:
        print("\n  [STOP] no non-price feature panels found. prediction_features/analyst_cache may")
        print("  use different column names. Paste: sqlite3 accuracy.db '.schema prediction_features'")
        return

    # ---- PEAD/SUE panel for correlation ----
    pead_panel={}
    earnp=find_db(a.root,"earnings.db")
    if earnp and os.path.isfile(earnp):
        ce=ro(earnp)
        try:
            cols=[cn for cn,_ in cols_of(ce,"earnings_surprises")]
            have_comp="eps_actual" in cols and "eps_estimate" in cols
            sel="ticker,report_date"+(",eps_actual,eps_estimate" if have_comp else ",eps_surprise_pct")
            ev=Q(ce,"SELECT "+sel+" FROM earnings_surprises WHERE report_date IS NOT NULL")
        finally:
            ce.close()
        by_tkr=defaultdict(list)
        for row in ev:
            tk=row[0]; do=nd(row[1])
            if do is None: continue
            if have_comp:
                try: raw=float(row[2])-float(row[3])
                except Exception: raw=None
            else:
                try: raw=float(row[2])
                except Exception: raw=None
            by_tkr[tk].append((do,raw))
        for tk in by_tkr: by_tkr[tk].sort()
        for tk,lst in by_tkr.items():
            prior=[]
            for do,raw in lst:
                if raw is None: continue
                if len(prior)>=4:
                    sd=np.std(prior,ddof=1)
                    if sd>1e-12: pead_panel[(do,tk)]=raw/sd
                prior.append(raw)

    # ---- evaluate each panel at hold and hold2 ----
    def eval_panel(panel,hold):
        by_date=defaultdict(list)
        for (d,tk),v in panel.items():
            fr=fwd(tk,d,hold)
            if fr is not None: by_date[d].append((v,fr))
        ics=[]
        for d,pairs in by_date.items():
            if len(pairs)>=a.min_names:
                ic=spearman([p[0] for p in pairs],[p[1] for p in pairs])
                if ic is not None: ics.append(ic)
        if len(ics)<3: return None
        arr=np.array(ics); m=arr.mean(); sd=arr.std()
        t=m/(sd/math.sqrt(len(arr))) if sd>0 else None
        return {"ic":m,"t":t,"n_dates":len(ics)}

    def corr_to_pead(panel):
        if not pead_panel: return None,0
        pairs=[]
        # align: for each PEAD event, find this panel's value for same ticker near same date
        bydate_tkr=defaultdict(dict)
        for (d,tk),v in panel.items(): bydate_tkr[tk][d]=v
        for (edate,tk),sue in pead_panel.items():
            cand=bydate_tkr.get(tk)
            if not cand: continue
            best=None
            for off in range(0,8):
                for c in (edate-datetime.timedelta(days=off),edate+datetime.timedelta(days=off)):
                    if c in cand: best=cand[c]; break
                if best is not None: break
            if best is not None: pairs.append((sue,best))
        if len(pairs)>=30:
            return spearman([p[0] for p in pairs],[p[1] for p in pairs]), len(pairs)
        return None,len(pairs)

    sub("STANDALONE IC at h=%d and h=%d (EXPLORATORY — short history)"%(a.hold,a.hold2))
    print("  %-22s %-22s %-22s %-12s"%("signal","h=%d IC/t/dates"%a.hold,"h=%d IC/t/dates"%a.hold2,"corr-to-PEAD"))
    results={}
    # priority signals first
    order_keys=sorted(feature_panels.keys(),
                      key=lambda k: (0 if any(p in k.lower() for p in ["pc_ratio","inst","analyst","short","skew","put_call"]) else 1, k))
    for sn in order_keys:
        panel=feature_panels[sn]
        r1=eval_panel(panel,a.hold); r2=eval_panel(panel,a.hold2)
        rho,npair=corr_to_pead(panel)
        def fmt(r):
            if not r: return "n/a"
            return "%+.3f/%.1f/%d"%(r["ic"],r["t"] or 0,r["n_dates"])
        print("  %-22s %-22s %-22s %-12s"
              %(sn[:22],fmt(r1),fmt(r2),
                ("%+.3f(n%d)"%(rho,npair)) if rho is not None else ("n/a(n%d)"%npair)))
        results[sn]={"h1":r1,"h2":r2,"corr_pead":rho,"npts":len(panel)}

    # ---- verdict ----
    banner("VERDICT — any promising non-price brick? (EXPLORATORY)")
    promising=[]
    for sn,r in results.items():
        for hk in ("h1","h2"):
            rr=r.get(hk)
            if rr and rr["ic"] is not None and abs(rr["ic"])>=0.03 and abs(rr["t"] or 0)>=2.0:
                corr=r.get("corr_pead")
                uncorr = corr is None or abs(corr)<0.3
                promising.append((sn,hk,rr["ic"],rr["t"],corr,uncorr))
    if promising:
        print("  PROMISING candidates (IC>=0.03, t>=2 — but SHORT HISTORY, treat as leads):")
        for sn,hk,ic,t,corr,uncorr in promising:
            tag="uncorrelated->potential brick" if uncorr else "correlated w/ PEAD"
            print("    %-20s @%s IC=%+.4f t=%.2f corr-PEAD=%s  [%s]"
                  %(sn,hk,ic,t or 0,"%+.3f"%corr if corr is not None else "NA",tag))
        print("\n  >> These are LEADS, not confirmed bricks. The ~3-month history means even a real")
        print("     signal is underpowered and even a fluke can clear t=2. To confirm ANY of these,")
        print("     you need a longer history of that feature (backfill pc_ratio / inst_flow / SI).")
        print("     Highest-value move: pick the best lead and get 2+ years of its history, then")
        print("     re-test with the same rigor we used on PEAD.")
    else:
        print("  No non-price feature shows even an exploratory IC>=0.03 at t>=2 in the available")
        print("  (short) history. Combined with the dead price signals, this strengthens the")
        print("  conclusion: in your CURRENT cached data, PEAD is close to the lonely brick.")
        print("  The likely-real bricks (analyst revisions, short interest, options skew) need")
        print("  LONGER history than the ~3 months in prediction_features to detect. Getting that")
        print("  history is the prerequisite to finding brick #2.")
    print("\n  REMINDER: this hunt is EXPLORATORY. prediction_features/analyst_cache ~3mo history =")
    print("  low power. Nothing here is tradeable on this evidence alone — these are directional")
    print("  leads about WHERE to invest in getting more data.")
    if a.out:
        with open(a.out,"a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":results},default=str)+"\n")
        print("\n  [report appended to %s]"%a.out)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
