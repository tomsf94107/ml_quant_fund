#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — VOLATILITY-SIGNAL VALIDATOR (idio-vol / total-vol / beta)
================================================================================
Tests price-derived risk signals as brick #3 candidates -- computed ENTIRELY from
prices.db (zero new data). The documented anomalies:
  * IDIOSYNCRATIC VOL (Ang-Hodrick-Xing-Zhang 2006): high idio-vol -> LOW future
    return. A raw-return cross-sectional anomaly -> fits this IC framework. (default)
  * TOTAL VOL (low-volatility anomaly): low vol -> higher risk-adjusted return.
  * BETA (betting-against-beta, Frazzini-Pedersen): low beta -> higher risk-ADJUSTED
    return. NOTE: BAB is a leverage-constrained / risk-adjusted story; on RAW forward
    returns high beta often earns MORE in up-markets, so raw-return IC may not show it.
    Included for completeness, flagged.

For each MONTH-END formation date, regress each stock's trailing --window daily returns
on an equal-weight market proxy (built from the universe): beta = cov(r,m)/var(m),
idio-vol = std(residual), total-vol = std(r). Rank cross-sectionally, forward return
from month-end -> per-date Spearman IC + Newey-West + null control + per-year + OOS.
(Same audited machinery as the SI and Lazy-Prices validators.)

THE POINT OF BRICK #3 = DECORRELATION. This also reports the cross-sectional rank
correlation of the vol signal vs MOMENTUM and vs DAYS_TO_COVER (short interest). A
brick that's real AND uncorrelated (|rank-corr| low) is the breadth the combination
thesis needed; a real-but-correlated one doesn't help (that's what sank PEAD+SI).

Direction auto: idiovol -1, totalvol -1, beta -1 (so the signal*direction is expected to
predict POSITIVE forward return if the anomaly holds). Verdict checks significant
positive IC after direction + null-clear.

RULE 1: trailing window strictly before formation; forward returns strictly after;
per-date IC + NW; null control; READ-ONLY.

USAGE:
  python validate_idiovol.py --root . --signal idiovol --hold 40
  python validate_idiovol.py --root . --signal idiovol --hold 21
  python validate_idiovol.py --root . --signal totalvol --hold 40
================================================================================
"""
import argparse, os, sqlite3, math, datetime
from collections import defaultdict
import numpy as np

def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def nd(s):
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None
def month_end(d):
    nxt=datetime.date(d.year+(1 if d.month==12 else 0),(d.month%12)+1,1)
    return nxt-datetime.timedelta(days=1)
def spearman(x,y):
    n=len(x)
    if n<5: return None
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    if rx.std()==0 or ry.std()==0: return None
    return float(np.corrcoef(rx,ry)[0,1])
def nw_se(x,lag):
    x=np.asarray(x,float); n=len(x)
    if n<2: return None
    e=x-x.mean(); g0=float(e@e)/n; s=g0
    for k in range(1,min(lag,n-1)+1):
        gk=float(e[k:]@e[:-k])/n; w=1.0-k/(lag+1.0); s+=2.0*w*gk
    return math.sqrt(s/n) if s>0 else None
LINE="="*78

def regress_stats(r, m):
    """r, m aligned daily returns. Returns (beta, idio_vol, total_vol) or None."""
    if len(r)<30: return None
    vm=m.var()
    if vm<=0: return None
    beta=float(np.cov(r,m,ddof=1)[0,1]/vm)
    alpha=float(r.mean()-beta*m.mean())
    resid=r-(alpha+beta*m)
    return beta, float(resid.std(ddof=1)), float(r.std(ddof=1))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--si-db",default=None)
    ap.add_argument("--signal",default="idiovol",choices=["idiovol","totalvol","beta"])
    ap.add_argument("--window",type=int,default=252,help="trailing trading days for the regression")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--direction",type=int,default=None,help="override; default -1 for all three")
    ap.add_argument("--mom-lookback",type=int,default=252)
    ap.add_argument("--mom-skip",type=int,default=21)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    si_db=a.si_db or os.path.join(a.root,"short_interest.db")
    direction=a.direction if a.direction is not None else -1
    print("\n"+LINE+"\nVOLATILITY-SIGNAL VALIDATOR (signal=%s, dir=%+d)\n"%(a.signal,direction)+LINE)
    if not os.path.isfile(prices_db): print("  [STOP] prices.db not found"); return

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

    # daily returns per ticker, aligned to a master date axis
    rets=defaultdict(dict)  # tk -> {date: ret}
    alldates=set()
    for tk,lst in px.items():
        for i in range(1,len(lst)):
            d0,p0=lst[i-1]; d1,p1=lst[i]
            if p0>0:
                rets[tk][d1]=p1/p0-1.0; alldates.add(d1)
    caldates=sorted(alldates)
    # equal-weight market return per date
    mkt={}
    by_date=defaultdict(list)
    for tk,dd in rets.items():
        for d,r in dd.items(): by_date[d].append(r)
    for d,rs in by_date.items():
        if len(rs)>=10: mkt[d]=float(np.mean(rs))

    def fwd(tk,d,h):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,8):
            cc=d+datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+h
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None
    def momentum(tk,d):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,6):
            cc=d-datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        end=i-a.mom_skip; start=i-a.mom_lookback
        if start<0 or end<=start: return None
        p0=lst[start][1]; p1=lst[end][1]
        return (p1/p0-1.0) if p0>0 else None

    # SI as-of (most recent settlement <= formation date)
    si_by_date=defaultdict(dict)
    if os.path.isfile(si_db):
        c=ro(si_db)
        try: sirows=Q(c,"SELECT ticker,settlement_date,days_to_cover FROM short_interest")
        finally: c.close()
        for tk,d,v in sirows:
            do=nd(d)
            if do is None or v is None: continue
            try: fv=float(v)
            except Exception: continue
            if fv<=50.0: si_by_date[do][tk.upper()]=fv
    si_dates=sorted(si_by_date)
    def si_asof(tk,d):
        best=None
        for sd in si_dates:
            if sd<=d: best=sd
            else: break
        if best is None: return None
        return si_by_date[best].get(tk.upper())

    # formation dates = month-ends spanning the data
    if not caldates: print("  [STOP] no returns built."); return
    months=sorted(set(month_end(d) for d in caldates))
    # signal value per ticker at formation date
    cal_idx={d:i for i,d in enumerate(caldates)}
    mkt_arr_dates=caldates
    def window_returns(tk, f):
        # trailing --window trading days strictly before f
        dd=rets.get(tk)
        if not dd: return None
        wd=[d for d in caldates if d<f][-a.window:]
        if len(wd)<30: return None
        rr=[]; mm=[]
        for d in wd:
            if d in dd and d in mkt:
                rr.append(dd[d]); mm.append(mkt[d])
        if len(rr)<30: return None
        return np.array(rr),np.array(mm)
    def sig_value(tk,f):
        wr=window_returns(tk,f)
        if wr is None: return None
        st=regress_stats(wr[0],wr[1])
        if st is None: return None
        beta,idio,tot=st
        return {"idiovol":idio,"totalvol":tot,"beta":beta}[a.signal]

    lag=max(1,int(math.ceil(a.hold/21.0)))
    # precompute per-formation cross-sections (signal, fwd, momentum, dtc)
    cross={}
    for f in months:
        recs={}
        for tk in px:
            sv=sig_value(tk,f)
            if sv is None: continue
            r=fwd(tk,f,a.hold)
            if r is None: continue
            recs[tk]=dict(sig=sv, ret=r, mom=momentum(tk,f), dtc=si_asof(tk,f))
        if len(recs)>=a.min_names: cross[f]=recs

    def ic_series(shuffle=False,rng=None):
        ics=[]; dts=[]
        for f in sorted(cross):
            recs=cross[f]
            sig=np.array([v["sig"] for v in recs.values()])*direction
            ret=np.array([v["ret"] for v in recs.values()])
            if shuffle: ret=rng.permutation(ret)
            ic=spearman(sig,ret)
            if ic is not None: ics.append(ic); dts.append(f)
        return np.array(ics),dts

    ics,dts=ic_series(); N=len(dts)
    if N<8: print("\n  [STOP] only %d usable monthly cross-sections."%N); return
    mean_ic=ics.mean(); se=nw_se(ics,lag); t=mean_ic/se if se else 0
    print("  %d monthly cross-sections, window=%dtd, hold=%dtd"%(N,a.window,a.hold))
    print("\n"+"-"*78+"\nPER-MONTH IC (signal=%s * dir %+d)\n"%(a.signal,direction)+"-"*78)
    print("  mean IC = %+.4f | std = %.4f | IC IR = %+.3f"%(mean_ic,ics.std(),mean_ic/ics.std() if ics.std()>0 else 0))
    print("  %%-right-sign = %.0f%% | naive t = %+.2f | Newey-West t = %+.2f"
          %(100*np.mean(ics>0), mean_ic/(ics.std(ddof=1)/math.sqrt(N)) if ics.std(ddof=1)>0 else 0, t))

    yr=defaultdict(list)
    for ic,d in zip(ics,dts): yr[d.year].append(ic)
    print("\n  per-year mean IC:")
    for y in sorted(yr):
        v=np.array(yr[y]); print("   %d: %+.4f  (n=%d)"%(y,v.mean(),len(v)))
    half=N//2
    def tof(x):
        s=nw_se(x,lag); return x.mean()/s if s else 0
    print("\n  first half IC=%+.4f t=%+.2f | second half IC=%+.4f t=%+.2f"
          %(ics[:half].mean(),tof(ics[:half]),ics[half:].mean(),tof(ics[half:])))

    rng=np.random.default_rng(7); nulls=[]
    for _ in range(300):
        nc,_=ic_series(shuffle=True,rng=rng)
        if len(nc): nulls.append(nc.mean())
    nulls=np.array(nulls); z=(mean_ic-nulls.mean())/nulls.std() if nulls.std()>0 else 0
    print("\n  null control: real IC %.1f std's from shuffled null (need >=3)"%z)

    # ---- DECORRELATION (the brick-#3 question) ----
    print("\n"+"-"*78+"\nDECORRELATION vs momentum and short interest (the brick-#3 test)\n"+"-"*78)
    mom_corr=[]; dtc_corr=[]
    for f in sorted(cross):
        recs=cross[f]
        tks=[tk for tk,v in recs.items() if v["mom"] is not None]
        if len(tks)>=a.min_names:
            s=np.array([recs[tk]["sig"] for tk in tks])*direction
            mo=np.array([recs[tk]["mom"] for tk in tks])
            rc=spearman(s,mo)
            if rc is not None: mom_corr.append(rc)
        tks2=[tk for tk,v in recs.items() if v["dtc"] is not None]
        if len(tks2)>=a.min_names:
            s=np.array([recs[tk]["sig"] for tk in tks2])*direction
            dc=np.array([recs[tk]["dtc"] for tk in tks2])
            rc=spearman(s,dc)
            if rc is not None: dtc_corr.append(rc)
    mc=np.mean(mom_corr) if mom_corr else float('nan')
    dc=np.mean(dtc_corr) if dtc_corr else float('nan')
    print("  mean cross-sectional rank-corr: signal vs MOMENTUM = %+.3f (n=%d)"%(mc,len(mom_corr)))
    print("  mean cross-sectional rank-corr: signal vs DAYS_TO_COVER = %+.3f (n=%d)"%(dc,len(dtc_corr)))
    print("  (|corr|<~0.3 = genuinely complementary stock selection; high = redundant)")

    print("\n"+LINE+"\nVERDICT — is %s a brick, and is it uncorrelated?\n"%a.signal+LINE)
    is_brick = abs(z)>=3 and abs(t)>=2.5 and np.mean(ics>0)>0.55
    uncorr = (abs(mc)<0.35 if not math.isnan(mc) else False) and (abs(dc)<0.35 if not math.isnan(dc) else True)
    if is_brick and uncorr:
        print("  >> CANDIDATE BRICK #3 + UNCORRELATED: IC %+.4f, NW t %+.2f, %.1f std's from null;"%(mean_ic,t,z))
        print("     rank-corr vs momentum %+.2f, vs SI %+.2f -> the breadth the combination needed."%(mc,dc))
        print("     NEXT: add it to the combined-book decomposition (does it raise the book's lift?).")
    elif is_brick and not uncorr:
        print("  >> REAL BUT CORRELATED: IC significant (t %+.2f, %.1f std's from null) BUT rank-corr"%(t,z))
        print("     vs momentum %+.2f / SI %+.2f is high -> it overlaps what you already have, so it"%(mc,dc))
        print("     won't add much breadth (the PEAD+SI problem). Real signal, limited diversification value.")
    elif abs(z)<3:
        print("  >> NOT A BRICK (%s): real IC within the shuffled null (%.1f std's). No edge here."%(a.signal,z))
        if a.signal=="beta":
            print("     (Expected for beta on RAW returns -- BAB is a risk-ADJUSTED anomaly. Try idiovol.)")
        else:
            print("     Try --hold 21 (vol anomalies are often monthly) or --signal totalvol / idiovol.")
    else:
        print("  >> SUGGESTIVE but below bar: IC %+.4f (t %+.2f, null %.1f). Try --hold 21 or --window 126."%(mean_ic,t,z))
    print("\n  Honest n=%d monthly cross-sections. In-sample, survivor-tilted (prices.db)."%N)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
