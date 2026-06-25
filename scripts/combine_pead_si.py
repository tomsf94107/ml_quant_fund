#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — COMBINATION TEST: PEAD + SHORT-INTEREST
================================================================================
The payoff. Two honestly-validated bricks (PEAD ~0.06-0.10 earnings-driven;
short interest ~0.05 positioning-driven), measured ~uncorrelated (-0.087). The
thesis: many weak UNCORRELATED signals combine to a higher information ratio than
any alone (IR scales ~ IC*sqrt(breadth) when uncorrelated). This tests it directly.

THE CADENCE PROBLEM (why portfolio_combine.py starved before): PEAD is event-time
(per earnings date, 40-day drift), short interest is calendar-time (bi-monthly).
FIX: put BOTH on a common MONTHLY grid as long-short return streams, then compare
and combine the streams.

CONSTRUCTION (both signals treated identically for a fair test):
  Monthly grid, each month-end d:
   * SHORT-INT stream: rank stocks by most-recent days_to_cover as of d; LONG bottom
     quintile / SHORT top quintile, equal-weight, dollar-neutral; earn return d -> d+~1mo.
   * PEAD stream: among stocks with an earnings event in (d-45d, d], rank by SUE;
     LONG top quintile / SHORT bottom quintile, dollar-neutral; earn return d -> d+~1mo.
   * COMBINED: 50/50 capital, and inverse-vol weighted.
  Leak-free: positions formed at d use only data <= d; returns realized strictly after d.

OUTPUTS:
   * per-stream monthly Sharpe (annualized), mean, vol, % positive months
   * correlation(PEAD, SI) monthly returns  (the diversification driver)
   * combined Sharpe (50/50 and inv-vol) vs the BEST single
   * diversification ratio = weighted-avg vol / portfolio vol  (>1 = real diversification)
   * verdict: does combining beat the best single? is the correlation low?

HONEST NOTE: monthly 1-month-hold is a COMMON-CADENCE proxy; each signal was
validated at 40d, so the monthly version may be modestly weaker per-leg. The point
is the COMBINATION effect (Sharpe lift + diversification), which is cadence-robust.

RULE 1: forward return strictly after formation. Both legs identical construction.
Sharpe annualized sqrt(12). READ-ONLY. No network.

USAGE:
  python combine_pead_si.py --root .
================================================================================
"""
import argparse, os, sqlite3, math, datetime
from collections import defaultdict
import numpy as np

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)
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

def month_key(d): return (d.year,d.month)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--si-db",default=None)
    ap.add_argument("--hold-days",type=int,default=21)  # ~1 month
    ap.add_argument("--quantile",type=float,default=0.2)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--pead-window",type=int,default=45)  # SUE fresh if earnings within N days
    ap.add_argument("--clip-dtc",type=float,default=50.0)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    si_db=a.si_db or os.path.join(a.root,"short_interest.db")
    earnp=find_db(a.root,"earnings.db")
    banner("ML QUANT FUND — COMBINATION TEST: PEAD + SHORT-INTEREST")
    for label,p in (("prices.db",prices_db),("short_interest.db",si_db),("earnings.db",earnp)):
        if not p or not os.path.isfile(p): print("[STOP] %s not found"%label); return

    # prices
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
    alltk=sorted(px.keys())
    print("  prices: %d tickers"%len(px))

    # trading-day calendar (union), for monthly grid = last trading day each month
    alldays=sorted(set(d for lst in px.values() for d,_ in lst))
    month_last={}
    for d in alldays: month_last[month_key(d)]=d  # last seen per month (sorted asc)
    grid=[month_last[k] for k in sorted(month_last)]

    def fwd_ret(tk,d,h):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=idx.get(d)
        if i is None:  # nearest on/before
            for off in range(0,5):
                cc=d-datetime.timedelta(days=off)
                if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+h
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None

    def ls_return(ranked, h, d):
        """ranked: list of (ticker, signal_val). Long bottom quantile, short top
        quantile by signal -- BUT caller sets sign so 'high signal' = short leg.
        Returns dollar-neutral long-short return, or None."""
        vals=[(tk,v) for tk,v in ranked]
        if len(vals)<a.min_names: return None
        vals.sort(key=lambda x:x[1])
        q=max(1,int(len(vals)*a.quantile))
        low=vals[:q]; high=vals[-q:]
        lr=[fwd_ret(tk,d,h) for tk,_ in low]; hr=[fwd_ret(tk,d,h) for tk,_ in high]
        lr=[x for x in lr if x is not None]; hr=[x for x in hr if x is not None]
        if len(lr)<3 or len(hr)<3: return None
        return np.mean(lr)-np.mean(hr)  # long low-signal, short high-signal

    # ---- SHORT-INTEREST monthly stream ----
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
    si_dates=sorted(si_by_date)
    def latest_si(d):
        # most recent settlement on/before d
        lo,hi=0,len(si_dates)-1; best=None
        for sd in si_dates:
            if sd<=d: best=sd
            else: break
        return si_by_date.get(best) if best else None
    si_stream={}  # month_key -> ls return
    for d in grid:
        snap=latest_si(d)
        if not snap: continue
        # signal = days_to_cover; high DTC = short leg -> long low DTC, short high DTC
        ranked=[(tk,v) for tk,v in snap.items() if tk in pos_of]
        r=ls_return(ranked,a.hold_days,d)
        if r is not None: si_stream[month_key(d)]=r

    # ---- PEAD monthly stream ----
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
    # SUE per event (PIT trailing)
    sue_events=defaultdict(list)  # ticker -> [(date, sue)]
    for tk,lst in by_tkr.items():
        prior=[]
        for do,raw in lst:
            if raw is None: continue
            if len(prior)>=4:
                sd=np.std(prior,ddof=1)
                if sd>1e-12: sue_events[tk].append((do,raw/sd))
            prior.append(raw)
    pead_stream={}
    for d in grid:
        # stocks with an earnings event in (d-window, d], ranked by most recent SUE
        ranked=[]
        for tk,evs in sue_events.items():
            if tk not in pos_of: continue
            recent=[(ed,s) for ed,s in evs if 0<=(d-ed).days<=a.pead_window]
            if recent:
                recent.sort()
                ranked.append((tk,recent[-1][1]))
        # PEAD: long HIGH sue, short LOW sue -> caller convention is long low-signal,
        # so pass NEGATED sue to reuse ls_return (long low(-sue)=high sue)
        ranked_neg=[(tk,-s) for tk,s in ranked]
        r=ls_return(ranked_neg,a.hold_days,d)
        if r is not None: pead_stream[month_key(d)]=r

    # ---- align ----
    common=sorted(set(si_stream)&set(pead_stream))
    print("  SI stream: %d months | PEAD stream: %d months | common: %d months"
          %(len(si_stream),len(pead_stream),len(common)))
    if len(common)<12:
        print("  [STOP] <12 common months — too few to compare/combine reliably."); return

    si=np.array([si_stream[m] for m in common])
    pe=np.array([pead_stream[m] for m in common])

    def sharpe(x): 
        m=x.mean(); s=x.std(ddof=1); return (m/s*math.sqrt(12)) if s>0 else 0.0
    def line(label,x):
        print("  %-16s mean=%+.4f  vol=%.4f  Sharpe(ann)=%+.2f  %%pos=%.0f%%"
              %(label,x.mean(),x.std(ddof=1),sharpe(x),100*np.mean(x>0)))

    sub("STANDALONE monthly long-short streams (hold ~%dd)"%a.hold_days)
    line("PEAD",pe); line("SHORT-INT",si)

    rho=float(np.corrcoef(pe,si)[0,1])
    sub("CORRELATION + COMBINATION")
    print("  correlation(PEAD, SHORT-INT) = %+.3f   (low |rho| -> diversification works)"%rho)

    # 50/50
    comb=0.5*pe+0.5*si
    # inverse-vol weights
    vp=pe.std(ddof=1); vs=si.std(ddof=1)
    wp=(1/vp)/((1/vp)+(1/vs)) if vp>0 and vs>0 else 0.5; ws=1-wp
    comb_iv=wp*pe+ws*si

    line("COMBINED 50/50",comb)
    line("COMBINED inv-vol",comb_iv)
    print("    (inv-vol weights: PEAD %.0f%% / SHORT-INT %.0f%%)"%(100*wp,100*ws))

    # diversification ratio for 50/50
    wavg_vol=0.5*vp+0.5*vs
    dr=wavg_vol/comb.std(ddof=1) if comb.std(ddof=1)>0 else 0
    best_single=max(sharpe(pe),sharpe(si))
    best_name="PEAD" if sharpe(pe)>=sharpe(si) else "SHORT-INT"

    banner("VERDICT — does combining two weak uncorrelated bricks beat either alone?")
    print("  Sharpe: PEAD %+.2f | SHORT-INT %+.2f | 50/50 %+.2f | inv-vol %+.2f"
          %(sharpe(pe),sharpe(si),sharpe(comb),sharpe(comb_iv)))
    print("  correlation %+.3f | diversification ratio %.2f (>1 = real diversification)"%(rho,dr))
    best_comb=max(sharpe(comb),sharpe(comb_iv))
    lift=best_comb/best_single-1 if best_single>0 else 0
    if best_comb>best_single*1.05 and abs(rho)<0.4:
        print("\n  >> THESIS CONFIRMED: combined Sharpe %.2f beats best single (%s %.2f) by %.0f%%,"
              %(best_comb,best_name,best_single,100*lift))
        print("     with low correlation (%.2f) and diversification ratio %.2f. Two weak uncorrelated"%(rho,dr))
        print("     bricks DO stack into something better than either alone. This is the payoff.")
    elif best_comb>best_single*1.05:
        print("\n  >> COMBINES, but correlation (%.2f) isn't very low. Sharpe lift %.0f%% is real but"%(rho,100*lift))
        print("     comes partly from re-weighting, not pure diversification. Still worth combining.")
    elif abs(rho)<0.4 and best_comb>=best_single*0.95:
        print("\n  >> DIVERSIFIES (vol down) but no Sharpe lift: combined Sharpe ~ best single, yet the")
        print("     low correlation (%.2f) cuts volatility (DR %.2f). Useful for risk, not raw return."%(rho,dr))
    else:
        print("\n  >> LIMITED BENEFIT: combining doesn't beat the best single (%.2f vs %.2f). On this"%(best_comb,best_single))
        print("     monthly construction the stacking effect is weak. The bricks may be individually")
        print("     too faint at monthly cadence, or less independent than the -0.087 suggested.")
    print("\n  Honest n = %d common months. Both legs identical dollar-neutral construction."%len(common))
    print("  NOTE: monthly proxy of 40d-validated signals; combination effect is the takeaway.")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
