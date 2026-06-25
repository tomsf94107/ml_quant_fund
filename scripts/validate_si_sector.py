#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — SHORT-INTEREST SECTOR-NEUTRAL TEST
================================================================================
Caveat #2 from the v2 validation: the per-date IC of -0.054 proves "ranking by
short interest predicts cross-sectional returns" — but NOT whether days_to_cover
ITSELF drives it vs. a correlated sector characteristic (heavily-shorted stocks
clustering in structurally low-returning sectors). This isolates that.

METHOD: within EACH date, demean BOTH the signal and the forward return WITHIN
each sector, then compute the cross-sectional IC of the residuals.
  - signal_resid  = days_to_cover - sector_mean(days_to_cover)   [within date]
  - return_resid  = fwd_return     - sector_mean(fwd_return)     [within date]
  - IC = spearman(signal_resid, return_resid) across all stocks that date
Per-date IC series, then mean + Newey-West t (same honest stats as v2).

INTERPRETATION (compare to v2 raw IC of -0.054):
  - sector-neutral IC stays ~-0.05  -> signal is WITHIN-SECTOR -> days_to_cover is
    the real driver, NOT a sector bet. Brick confirmed at the stock level.
  - sector-neutral IC collapses toward 0 -> the raw signal was mostly a SECTOR bet
    (shorted sectors underperform, not shorted stocks). Brick is really a sector tilt.
  - partial shrinkage -> mix of both; the residual is the stock-specific part.

SECTOR DATA: looks for a ticker->sector map in tickers_metadata.csv (the project's
source of truth) or common alternatives. Reports clearly if not found.

RULE 1: per-date IC + Newey-West (handles non-independence + overlap, same as v2).
Within-sector demeaning is leak-free (uses only same-date cross-section). READ-ONLY.

USAGE:
  python validate_si_sector.py --root .
  python validate_si_sector.py --root . --sector-file tickers_metadata.csv
  python validate_si_sector.py --root . --feature days_to_cover --hold 40
================================================================================
"""
import argparse, os, sqlite3, csv, math, datetime
from collections import defaultdict
import numpy as np

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)
def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def nd(s):
    if s is None: return None
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None
def spearman(x,y):
    n=len(x)
    if n<5: return None
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return None
    return float(np.corrcoef(rx,ry)[0,1])
def nw_se_mean(x,lag):
    x=np.asarray(x,dtype=float); n=len(x)
    if n<2: return None
    e=x-x.mean(); g0=float(e@e)/n; s=g0
    for k in range(1,min(lag,n-1)+1):
        gk=float(e[k:]@e[:-k])/n; w=1.0-k/(lag+1.0); s+=2.0*w*gk
    v=s/n; return math.sqrt(v) if v>0 else None

def load_sectors(root, sector_file):
    """Find ticker->sector. Try explicit file, tickers_metadata.csv, then DB tables."""
    cands=[]
    if sector_file: cands.append(sector_file if os.path.isabs(sector_file) else os.path.join(root,sector_file))
    cands += [os.path.join(root,"tickers_metadata.csv"),
              os.path.join(root,"ticker_metadata.csv"),
              os.path.join(root,"metadata.csv")]
    for p in cands:
        if p and os.path.isfile(p):
            try:
                with open(p,newline="") as f:
                    rd=csv.DictReader(f)
                    cols={c.lower():c for c in (rd.fieldnames or [])}
                    tkey=next((cols[k] for k in ("ticker","symbol","tickers") if k in cols),None)
                    skey=next((cols[k] for k in ("sector","gics_sector","industry","bucket","group") if k in cols),None)
                    if tkey and skey:
                        m={}
                        for row in rd:
                            tk=(row.get(tkey) or "").strip().upper()
                            sec=(row.get(skey) or "").strip()
                            if tk and sec: m[tk]=sec
                        if m: return m, "%s (col '%s')"%(os.path.basename(p),skey)
            except Exception:
                pass
    # DB fallback
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f.endswith((".db",".sqlite",".sqlite3")):
                try:
                    c=ro(os.path.join(dp,f))
                    for (t,) in Q(c,"SELECT name FROM sqlite_master WHERE type='table'"):
                        cl=[r[1].lower() for r in Q(c,'PRAGMA table_info("%s")'%t)]
                        if ("ticker" in cl or "symbol" in cl) and any(s in cl for s in ("sector","gics_sector","industry","bucket")):
                            tcol="ticker" if "ticker" in cl else "symbol"
                            scol=next(s for s in ("sector","gics_sector","industry","bucket") if s in cl)
                            m={}
                            for tk,sec in Q(c,'SELECT "%s","%s" FROM "%s"'%(tcol,scol,t)):
                                if tk and sec: m[str(tk).upper()]=str(sec)
                            c.close()
                            if m: return m, "%s.%s"%(f,t)
                    c.close()
                except Exception:
                    pass
    return None, None

def per_date_ic(by_date, fwd, sectors, min_names, neutralize):
    ic_series=[]
    for d in sorted(by_date):
        recs=[]  # (sector, signal, ret)
        for tk,v in by_date[d]:
            r=fwd(tk,d)
            if r is None: continue
            sec=sectors.get(tk,"_UNK") if sectors else "_ALL"
            recs.append((sec,v,r))
        if len(recs)<min_names: continue
        if neutralize:
            # demean signal and return within sector
            bysec=defaultdict(list)
            for i,(sec,v,r) in enumerate(recs): bysec[sec].append(i)
            sig=np.zeros(len(recs)); ret=np.zeros(len(recs))
            for sec,idxs in bysec.items():
                if len(idxs)<2:  # singleton sector can't be demeaned; drop
                    for i in idxs: sig[i]=np.nan
                    continue
                vmean=np.mean([recs[i][1] for i in idxs]); rmean=np.mean([recs[i][2] for i in idxs])
                for i in idxs: sig[i]=recs[i][1]-vmean; ret[i]=recs[i][2]-rmean
            mask=~np.isnan(sig)
            s=sig[mask]; rr=ret[mask]
        else:
            s=np.array([v for _,v,_ in recs]); rr=np.array([r for _,_,r in recs])
        if len(s)>=min_names:
            ic=spearman(s,rr)
            if ic is not None: ic_series.append((d,ic,len(s)))
    return ic_series

def stats(ic_series, hold, avg_gap=15):
    ics=np.array([ic for _,ic,_ in ic_series]); N=len(ics)
    mean_ic=float(ics.mean()); std_ic=float(ics.std(ddof=1))
    ir=mean_ic/std_ic if std_ic>0 else 0
    lag=max(1,int(math.ceil(hold/float(avg_gap))))
    se=nw_se_mean(ics,lag); t_nw=mean_ic/se if se else 0
    t_naive=mean_ic/(std_ic/math.sqrt(N)) if std_ic>0 else 0
    pct_neg=100.0*np.mean(ics<0)
    return dict(N=N,mean_ic=mean_ic,std_ic=std_ic,ir=ir,t_nw=t_nw,t_naive=t_naive,pct_neg=pct_neg,lag=lag)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--si-db",default=None)
    ap.add_argument("--sector-file",default=None)
    ap.add_argument("--feature",default="days_to_cover")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--min-names",type=int,default=15)
    ap.add_argument("--clip-dtc",type=float,default=50.0)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    si_db=a.si_db or os.path.join(a.root,"short_interest.db")
    banner("ML QUANT FUND — SHORT-INTEREST SECTOR-NEUTRAL TEST: %s (h=%d)"%(a.feature,a.hold))
    if not os.path.isfile(prices_db): print("[STOP] prices.db not found"); return
    if not os.path.isfile(si_db): print("[STOP] short_interest.db not found"); return

    sectors,src=load_sectors(a.root,a.sector_file)
    if not sectors:
        print("  [STOP] no ticker->sector map found. Looked for tickers_metadata.csv and DB tables")
        print("  with a sector/industry column. Pass --sector-file PATH, or tell me the column names.")
        print("  (Sector-neutralization needs a sector per ticker; the raw v2 result stands meanwhile.)")
        return
    print("  sectors from: %s  (%d tickers mapped, %d distinct sectors)"
          %(src,len(sectors),len(set(sectors.values()))))

    cp=ro(prices_db)
    try: prows=Q(cp,"SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL")
    finally: cp.close()
    px=defaultdict(list)
    for tk,d,p in prows:
        do=nd(d)
        if do is None: continue
        try: pf=float(p)
        except Exception: continue
        if pf>0: px[tk].append((do,pf))
    for tk in px: px[tk].sort()
    pos_of={tk:{d:i for i,(d,_) in enumerate(lst)} for tk,lst in px.items()}

    c=ro(si_db)
    try: rows=Q(c,'SELECT ticker,settlement_date,"%s" FROM short_interest'%a.feature)
    finally: c.close()
    by_date=defaultdict(list)
    mapped=0; total=0
    for tk,d,v in rows:
        do=nd(d)
        if do is None or v is None: continue
        try: fv=float(v)
        except Exception: continue
        if a.feature=="days_to_cover" and a.clip_dtc and fv>a.clip_dtc: continue
        tku=tk.upper(); total+=1
        if tku in sectors: mapped+=1
        by_date[do].append((tku,fv))
    print("  short-interest rows: %d (%.0f%% have a sector mapping)"%(total,100.0*mapped/max(total,1)))

    def fwd(tk,d):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,6):
            cc=d+datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+a.hold
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None

    raw=per_date_ic(by_date,fwd,sectors,a.min_names,neutralize=False)
    neu=per_date_ic(by_date,fwd,sectors,a.min_names,neutralize=True)
    if len(raw)<6 or len(neu)<6:
        print("  [STOP] too few usable dates (raw=%d neu=%d)"%(len(raw),len(neu))); return
    rs=stats(raw,a.hold); ns=stats(neu,a.hold)

    sub("RAW vs SECTOR-NEUTRAL per-date IC (h=%d)"%a.hold)
    print("  %-18s %-10s %-10s %-9s %-9s"%("","mean IC","IC IR","NW t","%neg dates"))
    print("  %-18s %+.4f    %+.3f     %+.2f     %.0f%%"%("RAW (v2)",rs["mean_ic"],rs["ir"],rs["t_nw"],rs["pct_neg"]))
    print("  %-18s %+.4f    %+.3f     %+.2f     %.0f%%"%("SECTOR-NEUTRAL",ns["mean_ic"],ns["ir"],ns["t_nw"],ns["pct_neg"]))
    retention = ns["mean_ic"]/rs["mean_ic"] if rs["mean_ic"]!=0 else 0
    print("\n  retention = sector-neutral IC / raw IC = %.0f%%"%(100*retention))

    banner("VERDICT — is the short-interest edge stock-specific or a sector bet?")
    sig_neu = abs(ns["t_nw"])>=2.0 and np.sign(ns["mean_ic"])==np.sign(rs["mean_ic"])
    if sig_neu and retention>=0.6:
        print("  >> STOCK-SPECIFIC: %.0f%% of the edge survives sector-neutralization (NW t=%.2f)."%(100*retention,ns["t_nw"]))
        print("     days_to_cover predicts returns WITHIN sectors -> it's a real stock-level signal,")
        print("     not a sector tilt. The brick is confirmed at the level that matters for combining.")
    elif sig_neu and retention>=0.3:
        print("  >> MOSTLY STOCK-SPECIFIC (partial): %.0f%% survives (NW t=%.2f). A minority of the raw"%(100*retention,ns["t_nw"]))
        print("     edge was a sector tilt; the stock-specific core is still significant. Usable, but the")
        print("     true stock-level IC is ~%.3f, not %.3f."%(ns["mean_ic"],rs["mean_ic"]))
    elif not sig_neu and abs(rs["t_nw"])>=2:
        print("  >> MOSTLY A SECTOR BET: the edge collapses after sector-neutralization (NW t %.2f -> %.2f)."%(rs["t_nw"],ns["t_nw"]))
        print("     The raw signal was largely 'shorted SECTORS underperform', not shorted STOCKS. As a")
        print("     stock-level brick it's weak; it would mostly duplicate a sector-momentum signal.")
    else:
        print("  >> INCONCLUSIVE: neutral NW t=%.2f. Edge neither clearly survives nor clearly collapses."%ns["t_nw"])
    print("\n  Honest n = %d dates (neutral). Per-date IC + Newey-West, same rigor as v2."%ns["N"])

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
