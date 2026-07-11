#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — SI STRATEGY: LONG-LEG vs SHORT-LEG (survivorship exposure)
================================================================================
The decomposition showed the only book worth anything is si_only (short interest).
Its survivorship exposure is NOT uniform: it is concentrated in the SHORT leg.
  * SHORT leg = high days-to-cover names -> these are disproportionately the names
    that later DELIST (bankruptcies, going-concern failures). Survivor-tilted
    prices.db is missing exactly these -> the short leg is the survivorship-exposed
    part of the edge.
  * LONG leg = low days-to-cover names -> these rarely delist; survivor-tilt barely
    touches them.

So the key question -- BEFORE buying/pulling any delisted-name data -- is: how much
of the SI edge comes from the (survivorship-exposed) SHORT leg vs the (safe) LONG leg?
  * edge mostly LONG-leg  -> survivor-tilt is nearly MOOT; the +1.0-1.2 Sharpe is
    largely from names that are well-covered. No delisted data needed.
  * edge mostly SHORT-leg -> survivorship is material; we genuinely need delisted
    prices (EODHD / Massive) to trust the level, AND the sign is ambiguous (delisted
    high-SI names that cratered would have HELPED the short leg -> our level may be
    CONSERVATIVE; pre-delisting squeezes would have HURT it -> could be optimistic).

DECOMPOSITION (per rebalance date d, dollar-neutral long-short):
  market(d)     = equal-weight mean forward return of the WHOLE ranked universe
  long_excess   = mean(fwd of low-DTC quintile)  - market(d)     (long-leg alpha)
  short_excess  = market(d) - mean(fwd of high-DTC quintile)     (short-leg alpha)
  book(d)       = long_excess + short_excess  (== the dollar-neutral L/S return,
                  up to the equal-weight approximation; verified in-script)
Then per-leg mean, annualized Sharpe, and the SHARE of total edge from each leg.

RULE 1: same SI construction as combined_40d_oos.py (days_to_cover, quantile,
hold, fwd strictly after formation); leg decomposition is an identity check
(long_excess+short_excess reconstructs book); per-date; READ-ONLY. Survivorship
is the very thing under test -- this script SIZES the exposure, it does not remove it.

USAGE:
  python si_leg_decomp.py --root .
  python si_leg_decomp.py --root . --hold 40 --quantile 0.2
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
def maxdd(curve):
    peak=-1e18; mdd=0
    for x in curve:
        peak=max(peak,x); mdd=min(mdd,x-peak)
    return mdd
def nw_se(x,lag):
    x=np.asarray(x,float); n=len(x)
    if n<2: return None
    e=x-x.mean(); g0=float(e@e)/n; s=g0
    for k in range(1,min(lag,n-1)+1):
        gk=float(e[k:]@e[:-k])/n; w=1.0-k/(lag+1.0); s+=2.0*w*gk
    return math.sqrt(s/n) if s>0 else None
def sharpe_of(x,ppy):
    x=np.asarray(x,float); v=x.std(ddof=1)
    return (x.mean()/v*math.sqrt(ppy)) if v>0 else 0.0
LINE="="*78

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--quantile",type=float,default=0.2)
    ap.add_argument("--min-names",type=int,default=20)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=os.path.join(a.root,"prices.db"); si_db=os.path.join(a.root,"short_interest.db")
    print("\n"+LINE+"\nSI STRATEGY — LONG-LEG vs SHORT-LEG (survivorship exposure)\n"+LINE)
    for lbl,p in (("prices.db",prices_db),("short_interest.db",si_db)):
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

    c=ro(si_db)
    try: sirows=Q(c,"SELECT ticker,settlement_date,days_to_cover FROM short_interest")
    finally: c.close()
    si_by_date=defaultdict(dict)
    for tk,d,v in sirows:
        do=nd(d)
        if do is None or v is None: continue
        try: fv=float(v)
        except Exception: continue
        if fv<=50.0: si_by_date[do][tk.upper()]=fv

    grid=sorted(si_by_date)
    ppy=365.25/float(a.hold); lag=max(1,int(math.ceil(a.hold/15.0)))

    book=[]; longx=[]; shortx=[]; recon_err=[]
    for d in grid:
        dtc=si_by_date[d]
        names=[]; vals=[]; rets=[]
        for tk,dv in dtc.items():
            if tk not in pos_of: continue
            r=fwd(tk,d,a.hold)
            if r is None: continue
            names.append(tk); vals.append(dv); rets.append(r)
        if len(names)<a.min_names: continue
        vals=np.array(vals); rets=np.array(rets)
        order=np.argsort(vals)                 # ascending DTC: low first
        q=max(1,int(len(names)*a.quantile))
        low_idx=order[:q]; high_idx=order[-q:]  # low DTC = long; high DTC = short
        market=rets.mean()
        long_ret=rets[low_idx].mean(); short_ret=rets[high_idx].mean()
        le=long_ret-market                      # long-leg excess (long low-DTC)
        se=market-short_ret                     # short-leg excess (short high-DTC)
        bk=le+se                                # dollar-neutral book (EW approx)
        # exact dollar-neutral L/S for reconstruction check:
        exact=long_ret-short_ret
        book.append(exact); longx.append(le); shortx.append(se); recon_err.append(abs((le+se)-exact))
    book=np.array(book); longx=np.array(longx); shortx=np.array(shortx)
    n=len(book)
    if n<8: print("  [STOP] only %d rebalances."%n); return

    print("  %d rebalances, hold %dd, quintile %.0f%%, HAC lag=%d"%(n,a.hold,100*a.quantile,lag))
    print("  reconstruction check: max|long_excess+short_excess - exact L/S| = %.2e (should be ~0)"%max(recon_err))

    bk_sh=sharpe_of(book,ppy); l_sh=sharpe_of(longx,ppy); s_sh=sharpe_of(shortx,ppy)
    bk_t=book.mean()/nw_se(book,lag) if nw_se(book,lag) else 0
    l_t =longx.mean()/nw_se(longx,lag) if nw_se(longx,lag) else 0
    s_t =shortx.mean()/nw_se(shortx,lag) if nw_se(shortx,lag) else 0

    print("\n"+"-"*78+"\nPER-LEG CONTRIBUTION (excess over universe mean, per rebalance)\n"+"-"*78)
    print("  %-14s %10s %9s %9s %9s"%("leg","mean/reb","ann ret","Sharpe","NW t"))
    print("  %-14s %+9.4f %+8.1f%% %+8.2f %+8.2f"%("LONG (low-DTC)",longx.mean(),100*longx.mean()*ppy,l_sh,l_t))
    print("  %-14s %+9.4f %+8.1f%% %+8.2f %+8.2f"%("SHORT (high-DTC)",shortx.mean(),100*shortx.mean()*ppy,s_sh,s_t))
    print("  %-14s %+9.4f %+8.1f%% %+8.2f %+8.2f"%("BOOK (L/S)",book.mean(),100*book.mean()*ppy,bk_sh,bk_t))

    tot=abs(longx.mean())+abs(shortx.mean())
    lshare=100*abs(longx.mean())/tot if tot>0 else 0; sshare=100-lshare
    print("\n  share of total edge:  LONG %.0f%%  |  SHORT %.0f%%"%(lshare,sshare))

    print("\n"+LINE+"\nVERDICT — how survivorship-exposed is the SI edge?\n"+LINE)
    if sshare<35:
        print("  >> MOSTLY LONG-LEG (%.0f%% long / %.0f%% short). The edge lives in LOW-DTC names, which"%(lshare,sshare))
        print("     rarely delist -> survivor-tilt is NEARLY MOOT for this strategy. You likely do NOT")
        print("     need delisted-name data; the ~1.0-1.2 through-cycle Sharpe is on well-covered names.")
        print("     Cheapest path: trade it long-tilted, treat survivorship as a minor (<~3%/yr) haircut.")
    elif sshare>65:
        print("  >> MOSTLY SHORT-LEG (%.0f%% short / %.0f%% long). The edge depends on SHORTING high-DTC"%(sshare,lshare))
        print("     names -- exactly the delisting-prone set survivor-tilt is missing. Survivorship is")
        print("     MATERIAL here and its SIGN is ambiguous (cratered delistings would have HELPED the")
        print("     short leg -> level may be CONSERVATIVE; pre-delist squeezes would have HURT it).")
        print("     -> Worth getting delisted prices: check Massive first ($0, you pay for it), else EODHD.")
    else:
        print("  >> MIXED (%.0f%% long / %.0f%% short). Both legs contribute. Survivorship matters for the"%(lshare,sshare))
        print("     short portion; a delisted-name check would firm up ~half the edge. Massive first.")
    print("\n  NOTE: this SIZES survivorship exposure using survivor data; it does not remove it. A true")
    print("  survivorship-free test still needs delisted prices. But if the edge is long-leg, that test")
    print("  is low-priority. Honest n=%d rebalances, survivor-tilted universe."%n)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
