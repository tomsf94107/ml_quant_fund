#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — INDEPENDENT TWO-BRICK BOOK (Next-step #1)
================================================================================
Trades the two validated bricks SEPARATELY (not as a proven combination), each
sized on its HAIRCUT IC. This is the honest implementation of "trade the two
bricks independently" -- it does NOT assume the combination is proven; it just
runs each book on its own merits and shows what holding both (naively, side by
side) looks like, net of costs.

SIZING (Grinold, fractional): target each name's weight proportional to its
cross-sectional signal z-score, scaled so the book's expected risk matches a
target. Position sizes derive from the HAIRCUT IC (the realistic one), not the
in-sample headline -- per the findings doc:
   PEAD haircut IC      ~0.06-0.10  (default 0.08)
   SHORT-INT haircut IC ~0.043      (sector-neutral, default 0.043)

WHAT IT DOES:
  * Builds each brick's long-short book on its NATURAL cadence/horizon (40d hold)
  * Sizes positions by signal z-score (capped), dollar-neutral
  * Applies transaction costs (per-turnover bps)
  * Reports per-book: ann return, vol, Sharpe (net), max drawdown, hit rate
  * Reports the two books held SIDE BY SIDE (capital split), with the HONEST note
    that this is two independent books, NOT a proven combined alpha
  * Caps: per-name weight, gross exposure

HONEST SCOPE: this is a research backtest (in-sample on the validation window,
survivor-tilted universe, simplified costs). It shows position sizing and relative
book behavior; it is NOT a live-tradeable P&L promise. Sharpe levels here inherit
the same imprecision flagged in the audits (short samples, no slippage/impact).

RULE 1: forward returns strictly after formation; SUE PIT-trailing; costs netted;
no look-ahead in sizing (z-scores use only same-date cross-section). READ-ONLY.

USAGE:
  python two_brick_book.py --root .
  python two_brick_book.py --root . --pead-ic 0.08 --si-ic 0.043 --cost-bps 10
  python two_brick_book.py --root . --target-vol 0.10
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
def zscore(v):
    v=np.asarray(v,float); m=v.mean(); s=v.std(ddof=1)
    return (v-m)/s if s>1e-12 else np.zeros_like(v)
LINE="="*78

def maxdd(curve):
    peak=-1e18; mdd=0
    for x in curve:
        peak=max(peak,x); mdd=min(mdd,x-peak)
    return mdd

def book_stats(rets, periods_per_year):
    rets=np.asarray(rets,float); n=len(rets)
    if n<2: return None
    m=rets.mean(); v=rets.std(ddof=1)
    ann_ret=m*periods_per_year; ann_vol=v*math.sqrt(periods_per_year)
    sharpe=ann_ret/ann_vol if ann_vol>0 else 0
    curve=np.cumsum(rets)  # additive (small returns)
    return dict(n=n,ann_ret=ann_ret,ann_vol=ann_vol,sharpe=sharpe,
                hit=100*np.mean(rets>0),mdd=maxdd(curve),total=curve[-1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--pead-ic",type=float,default=0.08)
    ap.add_argument("--si-ic",type=float,default=0.043)
    ap.add_argument("--pead-window",type=int,default=45)
    ap.add_argument("--pead-min-lag", type=int, default=0)
    ap.add_argument("--clip-dtc",type=float,default=50.0)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--max-weight",type=float,default=0.05)
    ap.add_argument("--target-vol",type=float,default=0.10)
    ap.add_argument("--min-names",type=int,default=20)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=os.path.join(a.root,"prices.db"); si_db=os.path.join(a.root,"short_interest.db")
    earnp=find_db(a.root,"earnings.db")
    print("\n"+LINE+"\nINDEPENDENT TWO-BRICK BOOK — PEAD + SHORT-INTEREST (h=%d)\n"%a.hold+LINE)
    for lbl,p in (("prices.db",prices_db),("short_interest.db",si_db),("earnings.db",earnp)):
        if not p or not os.path.isfile(p): print("[STOP] %s not found"%lbl); return
    print("  haircut ICs: PEAD %.3f | SHORT-INT %.3f | cost %.0f bps/turnover | max wt %.0f%% | target vol %.0f%%"
          %(a.pead_ic,a.si_ic,a.cost_bps,100*a.max_weight,100*a.target_vol))

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

    # short interest by date
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

    # PEAD SUE events (PIT trailing)
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
        recent=[(ed,s) for ed,s in evs if a.pead_min_lag<=(d-ed).days<=a.pead_window]
        if not recent: return None
        recent.sort(); return recent[-1][1]

    # ---- build z-score-sized, dollar-neutral book on SI settlement-date grid ----
    # (both books rebalanced on the SI grid; PEAD uses fresh SUE)
    # FIX (Bug 2): each return is a {hold}-day return. Annualize by the HOLD, not grid
    # spacing. Non-overlapping {hold}-day periods per year = 365/hold (~9.1 for 40d).
    # This avoids double-counting overlapping holds as if redeployed every ~15 days.
    grid=sorted(si_by_date)
    periods_per_year = 365.25/float(a.hold)

    def run_book(signal_fn, sign):
        """signal_fn(d) -> dict ticker->raw signal. sign=+1 if high signal predicts
        high return, -1 if high signal predicts low return.
        FIX (Bug 1): only REAL rebalances are recorded -- dates with <min_names of
        fresh signal are SKIPPED, not padded with 0.0 (padding fabricated calm periods
        that inflated Sharpe and flattened drawdown). Returns (rets, dates, turn, gross)."""
        prev_w={}; rets=[]; rdates=[]; turns=[]; grosses=[]
        for d in grid:
            sig=signal_fn(d)
            names=[tk for tk in sig if tk in pos_of]
            if len(names)<a.min_names:
                continue   # SKIP, do not pad -- no real position this date
            z=zscore([sig[tk] for tk in names])*sign
            w=np.clip(z,-3,3)
            if np.sum(np.abs(w))>0: w=w/np.sum(np.abs(w))
            w=np.clip(w,-a.max_weight,a.max_weight)
            wd={tk:wi for tk,wi in zip(names,w)}
            allk=set(wd)|set(prev_w)
            turn=sum(abs(wd.get(k,0)-prev_w.get(k,0)) for k in allk)
            gross=sum(abs(v) for v in wd.values())
            r=0.0
            for tk,wi in wd.items():
                fr=fwd(tk,d)
                if fr is not None: r+=wi*fr
            r-= turn*(a.cost_bps/10000.0)
            rets.append(r); rdates.append(d); turns.append(turn); grosses.append(gross); prev_w=wd
        return np.array(rets), rdates, (np.mean(turns) if turns else 0), (np.mean(grosses) if grosses else 0)

    pead_ret,pead_dates,pead_turn,pead_gross = run_book(lambda d:{tk:s for tk in si_by_date[d]
                                              for s in [sue_asof(tk,d)] if s is not None}, +1)
    si_ret,si_dates,si_turn,si_gross = run_book(lambda d:dict(si_by_date[d]), -1)

    ps=book_stats(pead_ret,periods_per_year)
    ss=book_stats(si_ret,periods_per_year)
    # FIX: books now have DIFFERENT real dates (PEAD skips dates without fresh SUE).
    # Align on COMMON dates before combining -- cannot average arrays positionally.
    pmap=dict(zip(pead_dates,pead_ret)); smap=dict(zip(si_dates,si_ret))
    common=sorted(set(pead_dates)&set(si_dates))
    both=np.array([0.5*pmap[d]+0.5*smap[d] for d in common]) if common else np.array([])
    bs=book_stats(both,periods_per_year) if len(both)>=2 else None

    def show(name,st,turn,n):
        if not st: print("  %-18s [insufficient data]"%name); return
        print("  %-18s netSharpe=%+.2f  annRet=%+.1f%%  annVol=%.1f%%  maxDD=%.1f%%  hit=%.0f%%  n=%d  turn/reb=%.2f"
              %(name,st["sharpe"],100*st["ann_ret"],100*st["ann_vol"],100*st["mdd"],st["hit"],st["n"],turn))

    print("\n"+"-"*78+"\nPER-BOOK PERFORMANCE (net of %.0f bps/turnover, %d-day hold, REAL rebalances only)\n"%(a.cost_bps,a.hold)+"-"*78)
    show("PEAD book",ps,pead_turn,len(pead_ret))
    show("SHORT-INT book",ss,si_turn,len(si_ret))
    print("  (n = number of REAL rebalances with >=%d names; no zero-padding)"%a.min_names)

    print("\n"+"-"*78+"\nTWO BOOKS SIDE BY SIDE (50/50 on %d common dates) — NOT a proven combined alpha\n"%len(common)+"-"*78)
    if bs:
        show("BOTH (50/50)",bs,0.5*(pead_turn+si_turn),len(both))
        pe_c=np.array([pmap[d] for d in common]); si_c=np.array([smap[d] for d in common])
        corr=float(np.corrcoef(pe_c,si_c)[0,1]) if pe_c.std()>0 and si_c.std()>0 else 0
        print("  return correlation between the two books (common dates) = %+.3f"%corr)
        print("  >> Per the audited combine_robust.py, the net-of-cost Sharpe LIFT is NEGATIVE and")
        print("     its CI spans zero. Run these as two INDEPENDENT books, each on its own haircut")
        print("     IC. Do NOT allocate to a 'combined alpha' — it does not beat the best single brick.")
    else:
        print("  [too few common dates to form a side-by-side book]")

    print("\n"+LINE+"\nSIZING REFERENCE (for live allocation)\n"+LINE)
    print("  Each book is dollar-neutral, z-score-weighted, per-name capped at %.0f%%."%(100*a.max_weight))
    print("  Expected IR per book ~ haircut_IC * sqrt(breadth_per_period). With the haircut ICs")
    print("  (PEAD %.3f, SI %.3f), keep position sizes MODEST — these are weak signals whose"%(a.pead_ic,a.si_ic))
    print("  edge shows only across many names and many periods, not on any single trade.")
    print("\n  HONEST: in-sample window, survivor-tilted universe, costs simplified (no slippage/")
    print("  impact). Treat Sharpe levels as indicative, not promises (same caveat as the audits).")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
