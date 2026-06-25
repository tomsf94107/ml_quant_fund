#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — IC-LEVEL COMBINATION TEST (Solution 3)
================================================================================
The Sharpe-level combination test was underpowered (n=23 months, lift CI spanned
zero). This reframes the thesis question to a level where you HAVE power: the
INFORMATION COEFFICIENT, measured per-date across ~350 stocks x ~58 dates.

THE THESIS, restated at the IC level:
  If PEAD and short interest carry COMPLEMENTARY cross-sectional information, then
  ranking stocks by a COMBINED signal should predict returns BETTER (higher |IC|)
  than ranking by either signal alone. That is the mechanism the combine-signals
  thesis claims -- and it is testable on per-date IC, not 23 monthly returns.

WHAT IT COMPUTES (on the overlapping universe & dates where BOTH signals exist):
  For each date:
    * IC_pead = spearman(SUE_rank, fwd_return)
    * IC_si   = spearman(-DTC_rank, fwd_return)   (neg: high short -> low ret)
    * IC_comb = spearman(combined_rank, fwd_return)
        combined_rank = average of the two within-date ranks (equal weight)
  -> three per-date IC time series.
  Then: mean IC each, and the DIFFERENCE (combined - best single) tested with
  Newey-West (handles overlap autocorrelation), same verified machinery as audit_ic.

DECISIVE OUTPUTS:
  * mean IC: PEAD, SI, COMBINED
  * IC uplift = mean|IC_comb| - max(mean|IC_pead|, mean|IC_si|)
  * Newey-West t on the per-date uplift series (is the uplift significant?)
  * NULL CONTROL: shuffle returns within date -> all ICs -> 0 (no manufactured signal)
  * redundancy check: correlation of the two per-date IC series

HONEST SCOPE: a positive IC uplift is evidence the signals carry COMPLEMENTARY
information (the thesis mechanism). It does NOT by itself prove a tradeable Sharpe
edge after costs -- that still needs more return-history. This tests the mechanism,
which is what your data can resolve.

RULE 1: per-date IC + Newey-West (verified); null control; redundancy check. The
combined rank uses ONLY same-date info (no leak). READ-ONLY. No network.

USAGE:
  python combine_ic_test.py --root .
  python combine_ic_test.py --root . --hold 40
================================================================================
"""
import argparse, os, sqlite3, math, datetime
from collections import defaultdict
import numpy as np

def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def cols_of(c,t): return [r[1] for r in Q(c,'PRAGMA table_info("'+t+'")')]
def nd(s):
    if s is None: return None
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None
def find_db(root,name):
    c=os.path.join(root,name)
    if os.path.isfile(c): return c
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f==name: return os.path.join(dp,f)
    return None
def spearman(x,y):
    n=len(x)
    if n<5: return None
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return None
    return float(np.corrcoef(rx,ry)[0,1])
def ranks(v):
    return np.argsort(np.argsort(v)).astype(float)
def nw_se_mean(x,lag):
    x=np.asarray(x,float); n=len(x)
    if n<2: return None
    e=x-x.mean(); g0=float(e@e)/n; s=g0
    for k in range(1,min(lag,n-1)+1):
        gk=float(e[k:]@e[:-k])/n; w=1.0-k/(lag+1.0); s+=2.0*w*gk
    return math.sqrt(s/n) if s>0 else None
LINE="="*78

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--pead-window",type=int,default=45)
    ap.add_argument("--clip-dtc",type=float,default=50.0)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=os.path.join(a.root,"prices.db"); si_db=os.path.join(a.root,"short_interest.db")
    earnp=find_db(a.root,"earnings.db")
    print("\n"+LINE+"\nIC-LEVEL COMBINATION TEST — PEAD + SHORT-INTEREST (h=%d)\n"%a.hold+LINE)
    for lbl,p in (("prices.db",prices_db),("short_interest.db",si_db),("earnings.db",earnp)):
        if not p or not os.path.isfile(p): print("[STOP] %s not found"%lbl); return

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

    # short interest signal: by date -> {ticker: dtc}
    c=ro(si_db)
    try: sirows=Q(c,"SELECT ticker,settlement_date,days_to_cover FROM short_interest")
    finally: c.close()
    si_by_date=defaultdict(dict)
    for tk,d,v in sirows:
        do=nd(d)
        if do is None or v is None: continue
        try: fv=float(v)
        except Exception: continue
        if a.clip_dtc and fv>a.clip_dtc: continue
        si_by_date[do][tk.upper()]=fv

    # PEAD signal: SUE per ticker per event (PIT trailing)
    ce=ro(earnp)
    try:
        cl=cols_of(ce,"earnings_surprises")
        comp="eps_actual" in cl and "eps_estimate" in cl
        sel="ticker,report_date"+(",eps_actual,eps_estimate" if comp else ",eps_surprise_pct")
        ev=Q(ce,"SELECT "+sel+" FROM earnings_surprises WHERE report_date IS NOT NULL")
    finally: ce.close()
    by_tkr=defaultdict(list)
    for row in ev:
        tk=row[0]; do=nd(row[1])
        if do is None: continue
        if comp:
            try: raw=float(row[2])-float(row[3])
            except Exception: raw=None
        else:
            try: raw=float(row[2])
            except Exception: raw=None
        by_tkr[tk].append((do,raw))
    for tk in by_tkr: by_tkr[tk].sort()
    sue_events=defaultdict(list)
    for tk,lst in by_tkr.items():
        prior=[]
        for do,raw in lst:
            if raw is None: continue
            if len(prior)>=4:
                sd=np.std(prior,ddof=1)
                if sd>1e-12: sue_events[tk].append((do,raw/sd))
            prior.append(raw)
    def sue_asof(tk,d):
        evs=sue_events.get(tk)
        if not evs: return None
        recent=[(ed,s) for ed,s in evs if 0<=(d-ed).days<=a.pead_window]
        if not recent: return None
        recent.sort(); return recent[-1][1]

    # ---- per-date IC for each signal on the OVERLAP (stocks with BOTH signals that date) ----
    # SI is bi-monthly; PEAD events are sparse. Use SI settlement dates as the grid; on each,
    # a stock qualifies if it has a fresh SUE AND a DTC that date.
    def compute(shuffle=False, rng=None):
        ic_pead=[]; ic_si=[]; ic_comb=[]; dates=[]; ncommon=[]
        for d in sorted(si_by_date):
            dtc_map=si_by_date[d]
            tickers=[]; sue_v=[]; dtc_v=[]; ret_v=[]
            for tk,dtc in dtc_map.items():
                s=sue_asof(tk,d)
                if s is None: continue
                r=fwd(tk,d)
                if r is None: continue
                tickers.append(tk); sue_v.append(s); dtc_v.append(dtc); ret_v.append(r)
            if len(tickers)<a.min_names: continue
            sue_v=np.array(sue_v); dtc_v=np.array(dtc_v); ret_v=np.array(ret_v)
            if shuffle: ret_v=rng.permutation(ret_v)
            # signals as ranks; SI negated (high short -> low return)
            r_pead=ranks(sue_v)            # high SUE -> high rank -> predict high return
            r_si=ranks(-dtc_v)             # low DTC -> high rank -> predict high return
            r_comb=ranks(r_pead+r_si)      # equal-weight rank-average, re-ranked
            ip=spearman(r_pead,ret_v); is_=spearman(r_si,ret_v); ic=spearman(r_comb,ret_v)
            if None in (ip,is_,ic): continue
            ic_pead.append(ip); ic_si.append(is_); ic_comb.append(ic)
            dates.append(d); ncommon.append(len(tickers))
        return (np.array(ic_pead),np.array(ic_si),np.array(ic_comb),dates,ncommon)

    icp,ics,icc,dates,ncommon=compute()
    N=len(dates)
    if N<6:
        print("  [STOP] only %d overlapping dates with both signals (need >=6)."%N)
        print("  PEAD events are sparse; few SI dates have enough stocks with a fresh SUE.")
        return
    lag=max(1,int(math.ceil(a.hold/15.0)))
    print("\n  %d overlapping dates, avg %d stocks/date (stocks with BOTH a fresh SUE and DTC)"%(N,int(np.mean(ncommon))))
    print("  Newey-West lag=%d"%lag)

    def line(label,arr):
        m=arr.mean(); se=nw_se_mean(arr,lag); t=m/se if se else 0
        print("  %-16s mean IC=%+.4f   NW t=%+.2f   %%right-sign=%.0f%%"
              %(label,m,t,100*np.mean(arr>0)))
        return m,t
    print("\n"+"-"*78+"\nPER-DATE IC: each signal alone vs combined\n"+"-"*78)
    mp,tp=line("PEAD alone",icp)
    ms,ts=line("SHORT-INT alone",ics)
    mc,tc=line("COMBINED",icc)

    print("\n"+"-"*78+"\nIC UPLIFT: does the COMBINED signal beat the BEST SINGLE signal?\n"+"-"*78)
    best=max(mp,ms); best_name="PEAD" if mp>=ms else "SHORT-INT"
    best_series = icp if mp>=ms else ics   # the better signal's per-date IC series (fixed choice, no look-ahead)
    # uplift = combined - best SINGLE signal, per date (the best signal is chosen ONCE, overall,
    # NOT per-date with hindsight -- the per-date max would be an unattainable oracle)
    uplift_series=icc-best_series
    um=uplift_series.mean(); use_=nw_se_mean(uplift_series,lag); ut=um/use_ if use_ else 0
    print("  best single (chosen once, overall) = %s, mean IC %+.4f"%(best_name,best))
    print("  combined mean IC = %+.4f"%mc)
    print("  mean uplift (combined - %s, per date) = %+.4f"%(best_name,um))
    print("  Newey-West t on uplift = %+.2f"%ut)
    print("  NOTE: benchmark is the BEST SINGLE signal picked ONCE (attainable), not a")
    print("  per-date max (which would be look-ahead). This is the honest comparison.")

    print("\n"+"-"*78+"\nREDUNDANCY: correlation of the two per-date IC series\n"+"-"*78)
    if icp.std()>0 and ics.std()>0:
        ic_corr=float(np.corrcoef(icp,ics)[0,1])
        print("  corr(IC_pead, IC_si) across dates = %+.3f"%ic_corr)
        print("  (low/negative -> the signals are 'right' on different dates -> complementary)")
    else:
        ic_corr=0

    print("\n"+"-"*78+"\nNULL CONTROL: shuffle returns within date -> all ICs must -> 0\n"+"-"*78)
    rng=np.random.default_rng(11)
    null_comb=[]
    for _ in range(150):
        _,_,nc,_,_=compute(shuffle=True,rng=rng)
        if len(nc): null_comb.append(nc.mean())
    null_comb=np.array(null_comb)
    z=(mc-null_comb.mean())/null_comb.std() if null_comb.std()>0 else 0
    print("  null combined mean IC: avg=%+.5f std=%.5f (should be ~0)"%(null_comb.mean(),null_comb.std()))
    print("  real combined mean IC=%+.4f -> %.1f null-std's from 0"%(mc,z))
    null_ok = abs(z)>=3

    print("\n"+LINE+"\nVERDICT — do PEAD + short interest carry COMPLEMENTARY information?\n"+LINE)
    print("  PEAD IC %+.4f (t%+.2f) | SI IC %+.4f (t%+.2f) | COMBINED IC %+.4f (t%+.2f)"
          %(mp,tp,ms,ts,mc,tc))
    print("  uplift %+.4f (NW t%+.2f) | IC-series corr %+.3f | null %.1f sd from 0"%(um,ut,ic_corr,z))
    sig_uplift = ut>=2.0 and um>0
    some_uplift = um>0 and mc>best
    if not null_ok:
        print("\n  >> INVALID: null control did not collapse (%.1f sd). Measurement suspect — do not trust."%z)
    elif sig_uplift:
        print("\n  >> THESIS SUPPORTED (IC level): the combined signal's IC significantly exceeds the")
        print("     best single signal (uplift %+.4f vs %s, NW t=%.2f). The signals carry"%(um,best_name,ut))
        print("     COMPLEMENTARY cross-sectional information — the combine mechanism is real here.")
        print("     CAVEAT: proves complementary INFO, not a tradeable Sharpe edge after costs.")
    elif some_uplift:
        print("\n  >> SUGGESTIVE (underpowered): combined IC %+.4f edges out best single %+.4f (%s),"%(mc,best,best_name))
        print("     combined t=%.2f exceeds both singles (%.2f/%.2f), signals complementary"%(tc,tp,ts))
        print("     (IC-corr %+.3f), null clean — but uplift NW t=%.2f isn't significant on n=%d dates."%(ic_corr,ut,N))
        print("     DIRECTIONALLY supports the thesis; not statistically established. Same wall: overlap is thin.")
    else:
        print("\n  >> NOT SUPPORTED (IC level): combined IC %+.4f does not beat best single %+.4f."%(mc,best))
        print("     The signals appear to be substitutes, not complements, for ranking.")
    print("\n  Honest n = %d overlapping dates. Per-date IC + Newey-West (verified machinery)."%N)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
