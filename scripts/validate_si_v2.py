#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — SHORT-INTEREST VALIDATOR v2 (per-date IC, Newey-West)
================================================================================
FIXES the t-stat inflation in validate_si.py. The pooled test treated 20,882
(stock,date) rows as independent, but they're ~400 stocks x ~60 dates with
PERSISTENT short interest -> effective n ~60, not 20,000 -> t-stats inflated ~7x.
That is what produced the bogus t=-20.

CORRECT METHOD (Grinold-Kahn "IC IR" / standard cross-sectional signal test):
  1. For EACH settlement date, compute cross-sectional Spearman IC between the
     signal and forward return across that date's stocks  -> a TIME SERIES of ICs.
  2. Signal strength = MEAN of the per-date ICs.
  3. Significance = mean(IC) / SE(IC), where SE uses the number of DATES (~60),
     not stock-date rows. Reported two ways:
       - naive: std(IC)/sqrt(N_dates)
       - Newey-West: widens SE for the autocorrelation caused by OVERLAPPING
         forward windows (bi-monthly dates ~15d apart, 40d hold -> windows overlap).
  This is beta-neutral by construction (ranking is within each date's cross-section).

SIGN CONVENTION: reports RAW signed IC of days_to_cover vs forward return.
  NEGATIVE mean IC = high short interest predicts LOW return = short signal WORKS.
  (No auto-negation here; we read the sign directly to avoid confusion.)

Also reports: % of dates with correct-sign IC, IC IR, per-year mean IC (regime
check), and an OOS split (first-half vs second-half dates).

RULE 1: per-date IC removes the non-independence inflation; NW removes the overlap
inflation; per-year + OOS expose regime dependence. Forward return strictly AFTER
settlement date. READ-ONLY (mode=ro), reads ONLY short_interest.db. No network.

USAGE:
  python validate_si_v2.py --root .
  python validate_si_v2.py --root . --feature days_to_cover --hold 40
  python validate_si_v2.py --root . --feature current_short --hold 20
================================================================================
"""
import argparse, os, sqlite3, math, datetime
from collections import defaultdict
import numpy as np

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)
def ro(p):
    return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
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

def newey_west_se_mean(x, lag):
    """Newey-West HAC standard error of the MEAN of series x, accounting for
    autocorrelation up to `lag` (Bartlett weights)."""
    x=np.asarray(x,dtype=float); n=len(x)
    if n<2: return None
    e=x-x.mean()
    gamma0=float(e@e)/n
    s=gamma0
    for k in range(1,min(lag,n-1)+1):
        gk=float(e[k:]@e[:-k])/n
        w=1.0-k/(lag+1.0)
        s+=2.0*w*gk
    var_mean=s/n
    return math.sqrt(var_mean) if var_mean>0 else None

def run_one(px,pos_of,si_db,feature,hold,min_names,clip_dtc,avg_gap_days=15):
    banner("SHORT-INTEREST VALIDATOR v2: %s  (h=%d, per-date IC)"%(feature,hold))
    c=ro(si_db)
    try:
        cols=[r[1] for r in Q(c,'PRAGMA table_info("short_interest")')]
        if feature not in cols:
            print("  [STOP] '%s' not in short_interest.db. cols: %s"%(feature,cols)); return
        rows=Q(c,'SELECT ticker,settlement_date,"%s" FROM short_interest'%feature)
    finally:
        c.close()

    # group by date
    by_date=defaultdict(list)  # date -> list of (ticker, signal_value)
    for tk,d,v in rows:
        do=nd(d)
        if do is None or v is None: continue
        try: fv=float(v)
        except Exception: continue
        if clip_dtc and feature=="days_to_cover" and fv>clip_dtc: continue
        by_date[do].append((tk.upper(),fv))

    def fwd(tk,d,h):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,6):
            cc=d+datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+h
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None

    # per-date cross-sectional IC (raw signed)
    ic_series=[]  # (date, ic, n)
    for d in sorted(by_date):
        sig=[]; ret=[]
        for tk,v in by_date[d]:
            r=fwd(tk,d,hold)
            if r is not None:
                sig.append(v); ret.append(r)
        if len(sig)>=min_names:
            ic=spearman(sig,ret)
            if ic is not None: ic_series.append((d,ic,len(sig)))
    if len(ic_series)<6:
        print("  [STOP] only %d usable dates (need >=6)"%len(ic_series)); return

    dates=[d for d,_,_ in ic_series]
    ics=np.array([ic for _,ic,_ in ic_series])
    ns=[n for _,_,n in ic_series]
    N=len(ics); mean_ic=float(ics.mean()); std_ic=float(ics.std(ddof=1))
    ir=mean_ic/std_ic if std_ic>0 else 0.0
    se_naive=std_ic/math.sqrt(N)
    t_naive=mean_ic/se_naive if se_naive>0 else 0.0
    lag=max(1,int(math.ceil(hold/float(avg_gap_days))))  # overlap lag
    se_nw=newey_west_se_mean(ics,lag)
    t_nw=mean_ic/se_nw if se_nw else 0.0
    # sign: short signal works if mean_ic NEGATIVE (high DTC -> low ret)
    correct_sign = -1 if feature in ("days_to_cover","current_short","short_ratio") else 0
    pct_correct = 100.0*np.mean([1 if (ic<0)==(correct_sign<0) else 0 for ic in ics]) if correct_sign else 100.0*np.mean(ics>0)

    print("  %d settlement dates, avg %d stocks/date, %s..%s"%(N,int(np.mean(ns)),dates[0],dates[-1]))
    print("  forward window %dd vs date gap ~%dd -> Newey-West lag=%d (overlap correction)"%(hold,avg_gap_days,lag))

    sub("PER-DATE IC TIME SERIES")
    print("  mean IC      = %+.4f   (NEGATIVE = high short interest predicts LOW return = signal works)"%mean_ic)
    print("  std IC       =  %.4f"%std_ic)
    print("  IC IR        = %+.3f   (mean/std, per-date)"%ir)
    print("  %% dates 'right-sign' = %.0f%%   (>55%% = consistent)"%pct_correct)
    print("  naive t      = %+.2f   (std/sqrt(%d) — assumes dates independent)"%(t_naive,N))
    print("  Newey-West t = %+.2f   (corrects overlap autocorrelation; THE honest number)"%t_nw)

    sub("PER-YEAR mean IC (regime check)")
    yr=defaultdict(list)
    for (d,ic,_) in ic_series: yr[d.year].append(ic)
    for y in sorted(yr):
        m=np.mean(yr[y]); bar="#"*int(abs(m)*200)
        print("  %d: mean IC %+.4f  (n=%2d dates) %s"%(y,m,len(yr[y]),bar))

    sub("OUT-OF-SAMPLE (first-half vs second-half dates)")
    mid=N//2
    ic1=ics[:mid]; ic2=ics[mid:]
    for label,arr in (("first half",ic1),("second half",ic2)):
        if len(arr)>=3:
            m=arr.mean(); se=arr.std(ddof=1)/math.sqrt(len(arr)); t=m/se if se>0 else 0
            print("  %-12s mean IC=%+.4f  t=%+.2f  (n=%d dates)"%(label,m,t,len(arr)))

    banner("VERDICT — is '%s' a real brick? (per-date, NW-corrected)"%feature)
    # honest thresholds on the NW t-stat and sign consistency
    sig = abs(t_nw)>=2.0
    right_dir = (mean_ic<0) if correct_sign<0 else (abs(mean_ic)>0)
    consistent = pct_correct>=55
    oos_hold = (len(ic1)>=3 and len(ic2)>=3 and np.sign(ic1.mean())==np.sign(ic2.mean()))
    print("  mean IC=%+.4f | IC IR=%+.3f | naive t=%+.2f | Newey-West t=%+.2f | right-sign dates=%.0f%%"
          %(mean_ic,ir,t_naive,t_nw,pct_correct))
    if sig and right_dir and consistent and oos_hold:
        print("  >> REAL BRICK: per-date IC is significant after NW correction, correct direction,")
        print("     consistent across dates, and holds in both halves. This is honest evidence.")
    elif sig and right_dir and (not oos_hold):
        print("  >> REGIME-DEPENDENT: significant overall but the two halves DISAGREE in sign.")
        print("     The edge is not stable across time — not a dependable brick. Investigate regimes.")
    elif sig and right_dir and (not consistent):
        print("  >> WEAK/CONCENTRATED: significant but right-sign on only %.0f%% of dates -> driven by"%pct_correct)
        print("     a few dates, not a persistent effect. Fragile; treat with caution.")
    elif (not sig) and abs(t_naive)>=2:
        print("  >> NOT A BRICK (inflation confirmed): naive t=%.2f looked significant but the honest"%t_naive)
        print("     Newey-West t=%.2f is not. The apparent signal was non-independence/overlap inflation —"%t_nw)
        print("     EXACTLY the bug that produced the original t=-20. No real edge here.")
    else:
        print("  >> NOT A BRICK: per-date IC not significant (NW t=%.2f). No tradeable edge on this evidence."%t_nw)
    print("\n  Honest n = %d dates (not %d stock-date rows). This is the RULE-1-correct test."%(N,sum(ns)))
    return {"mean_ic":mean_ic,"ir":ir,"t_naive":t_naive,"t_nw":t_nw,"pct":pct_correct,"N":N}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--si-db",default=None)
    ap.add_argument("--feature",default=None)
    ap.add_argument("--hold",type=int,default=None)
    ap.add_argument("--min-names",type=int,default=15)
    ap.add_argument("--clip-dtc",type=float,default=50.0)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    si_db=a.si_db or os.path.join(a.root,"short_interest.db")
    if not os.path.isfile(prices_db): print("[STOP] prices.db not found"); return
    if not os.path.isfile(si_db): print("[STOP] short_interest.db not found"); return
    cp=ro(prices_db)
    try: rows=Q(cp,"SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL")
    finally: cp.close()
    px=defaultdict(list)
    for tk,d,p in rows:
        do=nd(d)
        if do is None: continue
        try: pf=float(p)
        except Exception: continue
        if pf>0: px[tk].append((do,pf))
    for tk in px: px[tk].sort()
    pos_of={tk:{d:i for i,(d,_) in enumerate(lst)} for tk,lst in px.items()}
    print("prices loaded for %d tickers"%len(px))
    if a.feature:
        for h in ([a.hold] if a.hold else [40]):
            run_one(px,pos_of,si_db,a.feature,h,a.min_names,a.clip_dtc)
    else:
        for h in (40,20):
            run_one(px,pos_of,si_db,"days_to_cover",h,a.min_names,a.clip_dtc)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
