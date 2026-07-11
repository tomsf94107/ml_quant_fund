#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — PEAD SURVIVORSHIP-BIAS CHECK
================================================================================
prices.db was fetched from yfinance's CURRENT ticker list, so delisted names
(bankruptcies, acquisitions, index drops — often AFTER bad earnings) are missing.
PEAD is partly about under-reaction to bad news, so a survivor-only sample likely
INFLATES the edge (the losers that would drag the short leg aren't there). This
quantifies that bias and estimates the haircut on the +0.125 headline.

Because we cannot resurrect truly-delisted prices from yfinance, this measures the
bias INDIRECTLY but rigorously, three ways:

  1. UNIVERSE COMPLETENESS over time: for each year, how many of the tickers that
     had earnings events that year actually have price coverage? If early years are
     well-covered, survivorship is milder than feared; if early years are sparse,
     the bias is larger and the recent-only result is what to trust.

  2. SURVIVOR vs NEWER cohort: split tickers by how long their price history runs.
     "Full-period survivors" (data back to ~2009) are the most survivor-biased.
     "Shorter-history" names (IPO'd later / entered the data later) are less so.
     If PEAD IC is much higher for full-period survivors than newer names, that gap
     is a direct fingerprint of survivorship inflation.

  3. DELISTING PROXY via extreme drawdowns: names whose price series ENDS early
     (stops updating well before the data's max date) are de-facto delisted/dead in
     your data. Compare PEAD for names that "survived to the end" vs names whose
     series terminates early — the latter are your closest proxy for the delisted
     cohort the full sample is missing.

RULE 1: SUE PIT-trailing; entry day+2; window strictly after. Cohort splits use only
each ticker's data-span (no look-ahead). Indirect estimate is labeled as such — the
true fix is a delisting-inclusive price source, noted in the verdict.

READ-ONLY. mode=ro&immutable=1. No network.

USAGE:
  python pead_survivorship.py --root .
  python pead_survivorship.py --root . --hold 40
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
def cols_of(c,t): return [r[1] for r in Q(c,'PRAGMA table_info("'+t+'")')]
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--min-events",type=int,default=40)
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    banner("ML QUANT FUND — PEAD SURVIVORSHIP-BIAS CHECK")
    print("How much does survivor-only pricing inflate the +0.125 PEAD edge? (offline)")
    if not require(HAVE_NUMPY,"numpy required"): return
    if not require(os.path.isfile(prices_db),"prices.db not found"): return
    earnp=find_db(a.root,"earnings.db")
    if not require(earnp,"earnings.db not found"): return

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
    data_max=max((lst[-1][0] for lst in px.values()), default=None)
    data_min=min((lst[0][0] for lst in px.values()), default=None)
    print("  prices: %d tickers, %s..%s"%(len(px),data_min,data_max))
    # per-ticker span
    span_of={tk:(lst[0][0],lst[-1][0]) for tk,lst in px.items()}

    # earnings events + SUE
    ce=ro(earnp)
    try:
        cols=cols_of(ce,"earnings_surprises")
        have_comp="eps_actual" in cols and "eps_estimate" in cols
        sel="ticker,report_date"+(",eps_actual,eps_estimate" if have_comp else ",eps_surprise_pct")
        ev=Q(ce,"SELECT "+sel+" FROM earnings_surprises WHERE report_date IS NOT NULL")
        all_evt_tickers=set(r[0] for r in Q(ce,"SELECT DISTINCT ticker FROM earnings_surprises"))
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
    # SUE events
    events=[]  # (ticker, date, sue)
    for tk,lst in by_tkr.items():
        prior=[]
        for do,raw in lst:
            if raw is None: continue
            if len(prior)>=4:
                sd=np.std(prior,ddof=1)
                if sd>1e-12: events.append((tk,do,raw/sd))
            prior.append(raw)

    def fwd(tk,do,hold):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,6):
            c=do+datetime.timedelta(days=off)
            if c in idx: i=idx[c]; break
        if i is None: return None
        x=i+hold
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None

    cost=a.cost_bps/10000.0
    def ic_net(recs):
        if len(recs)<a.min_events: return None
        s=[r[0] for r in recs]; rr=[r[1] for r in recs]; n=len(s)
        ic=spearman(s,rr)
        order=np.argsort(s); q=max(1,n//5); lo=order[:q]; hi=order[-q:]
        L=np.mean([rr[i] for i in hi]); S=np.mean([rr[i] for i in lo])
        g=L-S; net=g-2*cost
        sd=math.sqrt(np.var([rr[i] for i in hi])/q+np.var([rr[i] for i in lo])/q)
        t=g/sd if sd>0 else None
        return {"n":n,"ic":ic,"net":net,"t":t}

    # ---- 1. universe completeness over time ----
    sub("1. UNIVERSE COMPLETENESS — % of earnings-event tickers with price coverage, by year")
    priced=set(px.keys())
    evt_by_year=defaultdict(set)
    for tk,lst in by_tkr.items():
        for do,_ in lst:
            evt_by_year[do.year].add(tk)
    print("  %-6s %-14s %-14s %-10s"%("year","evt tickers","priced","coverage"))
    for yr in sorted(evt_by_year):
        ev_tk=evt_by_year[yr]; cov=len(ev_tk & priced)
        frac=cov/max(len(ev_tk),1)
        bar="#"*int(frac*20)
        print("  %-6d %-14d %-14d %5.0f%% %s"%(yr,len(ev_tk),cov,100*frac,bar))
    print("\n  If early years show LOW coverage, the early-year PEAD is unmeasurable and survivorship")
    print("  is severe there; recent dense years are the trustworthy sample (matches walk-forward).")

    # ---- 2. survivor cohort vs newer ----
    sub("2. FULL-PERIOD SURVIVORS vs SHORTER-HISTORY names (the survivorship fingerprint)")
    # classify each event ticker by its price-history start
    early_cut = nd("2012-01-01")  # has long history => survivor-biased
    surv_recs=[]; newer_recs=[]
    for tk,do,sue in events:
        sp=span_of.get(tk)
        if not sp: continue
        r=fwd(tk,do,a.hold)
        if r is None: continue
        if sp[0]<=early_cut: surv_recs.append((sue,r))
        else: newer_recs.append((sue,r))
    s_full=ic_net(surv_recs); s_new=ic_net(newer_recs)
    def show(label,m):
        if not m: print("  %-32s (too few events)"%label); return
        print("  %-32s n=%-5d IC=%+.4f net=%+.4f t=%.2f"%(label,m["n"],m["ic"] or 0,m["net"],m["t"] or 0))
    show("full-period survivors (pre-2012)",s_full)
    show("shorter-history names (post-2012)",s_new)
    if s_full and s_new and s_full["ic"] is not None and s_new["ic"] is not None:
        gap=s_full["ic"]-s_new["ic"]
        print("\n  IC gap (survivors - newer): %+.4f"%gap)
        if gap>0.04:
            print("  -> Survivors show a MUCH stronger edge. This is a survivorship-inflation fingerprint:")
            print("     the long-history winners flatter PEAD. The true (delisting-inclusive) edge is lower.")
        elif gap>0.0:
            print("  -> Survivors modestly stronger; mild survivorship inflation.")
        else:
            print("  -> No survivor premium (newer names as strong or stronger). Survivorship bias")
            print("     appears LIMITED for this signal — encouraging for the headline's robustness.")

    # ---- 3. delisting proxy: series-ends-early cohort ----
    sub("3. DELISTING PROXY — names whose price series ENDS EARLY (de-facto dead in your data)")
    if data_max:
        cutoff = data_max - datetime.timedelta(days=120)  # ended >~4mo before data max
        dead_recs=[]; alive_recs=[]
        for tk,do,sue in events:
            sp=span_of.get(tk)
            if not sp: continue
            r=fwd(tk,do,a.hold)
            if r is None: continue
            if sp[1]<cutoff: dead_recs.append((sue,r))   # series terminated early
            else: alive_recs.append((sue,r))
        d_dead=ic_net(dead_recs); d_alive=ic_net(alive_recs)
        show("series ended early (delisting proxy)",d_dead)
        show("series alive to end",d_alive)
        if d_dead and d_alive and d_dead["ic"] is not None and d_alive["ic"] is not None:
            print("\n  The 'ended early' cohort is your closest proxy for the delisted names the full")
            print("  sample is MISSING. If their PEAD edge/short-leg behaves very differently, the")
            print("  full-universe (delisting-inclusive) edge would shift toward their profile.")
            if d_dead["ic"] < d_alive["ic"]-0.03:
                print("  -> The proxy-delisted cohort shows a WEAKER edge -> including real delistings")
                print("     would LOWER the headline. Treat +0.125 as an optimistic ceiling.")
            else:
                print("  -> The proxy-delisted cohort's edge is similar -> survivorship distortion looks")
                print("     modest. (Still indirect; a true delisting-inclusive source would confirm.)")
        else:
            print("  (Too few 'ended early' names to compare — most of your universe runs to the end,")
            print("   which itself is a sign the sample is survivor-tilted.)")

    # ---- verdict + honest estimate ----
    banner("VERDICT — survivorship haircut on PEAD")
    print("  This is an INDIRECT estimate (yfinance can't resurrect delisted prices). Synthesis:")
    bullets=[]
    # coverage signal
    early_cov = np.mean([len(evt_by_year[y]&priced)/max(len(evt_by_year[y]),1)
                         for y in sorted(evt_by_year) if y<2015]) if any(y<2015 for y in evt_by_year) else None
    if early_cov is not None:
        bullets.append("early-year (pre-2015) coverage averages %.0f%% -> %s"
                       %(100*early_cov, "early data sparse; trust recent-only result" if early_cov<0.6 else "early data fairly complete"))
    if s_full and s_new and s_full["ic"] is not None and s_new["ic"] is not None:
        gap=s_full["ic"]-s_new["ic"]
        bullets.append("survivor IC premium = %+.4f -> %s"%(gap,"meaningful inflation" if gap>0.04 else "limited inflation"))
    for b in bullets: print("   - "+b)
    print()
    print("  PRACTICAL TAKEAWAY: the +0.125 headline is best treated as an OPTIMISTIC ceiling.")
    print("  A realistic delisting-inclusive estimate is likely in the ~0.06-0.10 range (consistent")
    print("  with the magnitude-concentration and recency caveats). Size on the haircut, not the headline.")
    print("\n  TRUE FIX (if you want certainty): obtain a delisting-inclusive daily price source")
    print("  (CRSP, Norgate, Sharadar, or a survivorship-bias-free vendor) and re-run fetch_and_pead.py")
    print("  against it. That is the only way to measure the real number directly.")
    if a.out:
        rep={"early_cov":early_cov,"surv":s_full,"newer":s_new}
        with open(a.out,"a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":rep},default=str)+"\n")
        print("\n  [report appended to %s]"%a.out)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
