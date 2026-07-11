#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — BRICK #3 HUNT: OPTIONS-IMPLIED SIGNAL VALIDATOR (Next-step #4)
================================================================================
The next uncorrelated brick to hunt is options-implied (put/call ratio, IV skew)
-- a DIFFERENT information axis from price (PEAD) and positioning (short interest),
so it has the best chance of being uncorrelated with the two bricks you have.

THE BLOCKER (honest, stated up front): your live options feed gives only CURRENT
snapshots -- it cannot backfill history, exactly like yfinance couldn't backfill
short interest before FINRA. A signal can't be validated without history. So this
script is a VALIDATOR SKELETON, ready to run the moment historical data is loaded,
plus the data-source research to get that history (below). It is deliberately the
SAME verified machinery as validate_si_v2.py (per-date IC + Newey-West + null
control), so brick #3 gets the identical honest treatment that confirmed brick #2.

>>> FREE HISTORICAL SOURCE FOUND (the FINRA-style unblock for options):
    Cboe publishes FREE historical EQUITY PUT/CALL RATIO archives, downloadable,
    multi-year history:
      https://www.cboe.com/us/options/market_statistics/historical_data/
    These are MARKET-WIDE/index + equity put/call ratios (a regime signal), not
    per-name skew. For PER-NAME IV skew history (the stronger cross-sectional
    signal) the realistic options are PAID: ORATS (~2007+), or FlashAlpha
    historical API (minute-level, 2018+). Start FREE with Cboe equity put/call as
    a regime feature; escalate to paid per-name skew only if the regime signal
    shows promise. Same staged approach that worked for short interest.

EXPECTED SCHEMA (load history into options_signal.db, table options_signal):
    ticker TEXT, date TEXT, signal_value REAL
  where signal_value is one of: put_call_ratio, iv_skew_25d, iv_rank, vrp, etc.
  (one signal per run; pass --signal-col to name it). For market-wide Cboe
  put/call, use ticker='_MARKET' and treat as a regime feature (different test).

WHAT IT COMPUTES (once data exists): per-date cross-sectional IC of the signal vs
forward returns, Newey-West t, % years correct sign, null control. IDENTICAL to
the short-interest validation -- so a confirmed brick #3 would be confirmed to the
same standard.

RULE 1: per-date IC + Newey-West (verified machinery copied from validate_si_v2);
null control; forward returns strictly after formation. READ-ONLY. No network in
the validation step (data must be pre-loaded, same discipline as FINRA).

USAGE (once options_signal.db exists):
  python validate_brick3.py --root . --signal-col put_call_ratio --direction -1 --hold 40
  python validate_brick3.py --root . --signal-col iv_skew_25d --direction +1 --hold 20
  python validate_brick3.py --status     # checks for data, prints source guidance
================================================================================
"""
import argparse, os, sqlite3, math, datetime
from collections import defaultdict
import numpy as np

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
    x=np.asarray(x,float); n=len(x)
    if n<2: return None
    e=x-x.mean(); g0=float(e@e)/n; s=g0
    for k in range(1,min(lag,n-1)+1):
        gk=float(e[k:]@e[:-k])/n; w=1.0-k/(lag+1.0); s+=2.0*w*gk
    return math.sqrt(s/n) if s>0 else None
LINE="="*78

SOURCE_GUIDANCE = """
  HOW TO GET HISTORICAL OPTIONS DATA (the blocker):
   * FREE, market-wide regime signal (start here):
       Cboe historical equity put/call ratio archives —
       https://www.cboe.com/us/options/market_statistics/historical_data/
       Download the equity put/call CSV/XLS, load as ticker='_MARKET'.
       This is a REGIME feature (one series), validated differently than a
       cross-sectional brick — use it as a market-state gate, not a stock ranker.
   * PAID, per-name cross-sectional skew (the stronger brick, if regime shows promise):
       - ORATS — per-name IV surface history ~2007+
       - FlashAlpha historical API — minute-level GEX/VRP/skew, 2018+
       - Cboe DataShop / LiveVol — end-of-day option prices + IV
   * Your LIVE feed (UW/Massive): current snapshots only — CANNOT backfill. Begin
     logging it daily now so history accrues going forward (same as the intraday
     plan), but that's ~months to a usable sample.
"""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--db",default=None)
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--signal-col",default="put_call_ratio")
    ap.add_argument("--direction",type=int,default=-1,
                    help="+1 if high signal predicts HIGH return, -1 if high signal predicts LOW return")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--status",action="store_true")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    sig_db=a.db or os.path.join(a.root,"options_signal.db")
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")

    print("\n"+LINE+"\nBRICK #3 HUNT — OPTIONS-IMPLIED SIGNAL VALIDATOR\n"+LINE)

    if not os.path.isfile(sig_db):
        print("  options_signal.db NOT FOUND at %s"%sig_db)
        print("  This is the expected state — the signal is blocked on historical data.")
        print(SOURCE_GUIDANCE)
        print("  Once you load history into options_signal.db (table options_signal:")
        print("  ticker, date, <signal_col>), re-run WITHOUT --status to validate.")
        return
    if a.status:
        c=ro(sig_db)
        try:
            cols=[r[1] for r in Q(c,'PRAGMA table_info("options_signal")')]
            n=Q(c,"SELECT COUNT(*) FROM options_signal")[0][0]
            dr=Q(c,"SELECT MIN(date),MAX(date) FROM options_signal")[0]
        finally: c.close()
        print("  options_signal.db present: %d rows, %s to %s"%(n,dr[0],dr[1]))
        print("  columns: %s"%", ".join(cols))
        print(SOURCE_GUIDANCE)
        return

    if not os.path.isfile(prices_db): print("  [STOP] prices.db not found"); return

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

    # signal
    c=ro(sig_db)
    try: rows=Q(c,'SELECT ticker,date,"%s" FROM options_signal WHERE "%s" IS NOT NULL'%(a.signal_col,a.signal_col))
    finally: c.close()
    by_date=defaultdict(list)
    for tk,d,v in rows:
        do=nd(d)
        if do is None or v is None: continue
        try: fv=float(v)
        except Exception: continue
        by_date[do].append((tk.upper(),fv))

    # market-wide regime case
    only_market = set(tk for d in by_date for tk,_ in by_date[d]) <= {"_MARKET"}
    if only_market:
        print("  Detected MARKET-WIDE signal (ticker='_MARKET') — this is a REGIME series,")
        print("  not a cross-sectional brick. Cross-sectional IC doesn't apply. Validate it")
        print("  instead as a state gate (e.g. does forward market return differ across")
        print("  signal terciles?). Skeleton stops here — say the word and I'll build the")
        print("  regime-gate validator (different test than the per-date IC below).")
        return

    def compute(shuffle=False,rng=None):
        ics=[]; dates=[]
        for d in sorted(by_date):
            recs=[(tk,v,fwd(tk,d)) for tk,v in by_date[d]]
            recs=[(tk,v,r) for tk,v,r in recs if r is not None]
            if len(recs)<a.min_names: continue
            sig=np.array([v for _,v,_ in recs])*a.direction
            ret=np.array([r for _,_,r in recs])
            if shuffle: ret=rng.permutation(ret)
            ic=spearman(sig,ret)
            if ic is not None: ics.append(ic); dates.append(d)
        return np.array(ics),dates

    ics,dates=compute()
    N=len(dates)
    if N<6:
        print("  [STOP] only %d usable dates. Need more history. %s"%(N,"See source guidance."))
        print(SOURCE_GUIDANCE); return
    lag=max(1,int(math.ceil(a.hold/15.0)))
    mean_ic=ics.mean(); se=nw_se_mean(ics,lag); t=mean_ic/se if se else 0
    print("\n  signal=%s  direction=%+d  hold=%dd  dates=%d"%(a.signal_col,a.direction,a.hold,N))
    print("  mean per-date IC = %+.4f | Newey-West t = %+.2f | %%right-sign = %.0f%%"
          %(mean_ic,t,100*np.mean(ics>0)))

    # null control
    rng=np.random.default_rng(11); nulls=[]
    for _ in range(200):
        nc,_=compute(shuffle=True,rng=rng)
        if len(nc): nulls.append(nc.mean())
    nulls=np.array(nulls); z=(mean_ic-nulls.mean())/nulls.std() if nulls.std()>0 else 0
    print("  null control: real IC %.1f std's from shuffled-null (need >=3)"%z)

    print("\n"+LINE+"\nVERDICT\n"+LINE)
    if abs(z)<3:
        print("  >> NOT A BRICK / measurement weak: real IC within the null. No edge here.")
    elif abs(t)>=3 and abs(z)>=3:
        print("  >> CANDIDATE BRICK #3: IC %+.4f, NW t %+.2f, %.1f std's from null. Then run the"%(mean_ic,t,z))
        print("     SAME follow-ups brick #2 got: sector-neutral (validate_si_sector pattern),")
        print("     year-by-year sign, and combination tests vs PEAD + short interest.")
    else:
        print("  >> SUGGESTIVE: IC %+.4f (t %+.2f) but below the t>=3 bar. More history needed."%(mean_ic,t))
    print("\n  Honest n = %d dates. Same verified machinery as the short-interest brick."%N)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
