#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — AUDIT SCRIPT 1: VERIFY THE COMBINATION NUMBERS
================================================================================
Independently recomputes every number in the combination verdict FROM THE RAW
RETURN STREAMS, showing all intermediate math, so you can check it yourself
rather than trusting the assistant's audit. Recomputes:

  1. Each stream's monthly mean, vol, annualized Sharpe (from scratch)
  2. The Sharpe standard error (Lo 2002 formula) + 95% CI
  3. The combined Sharpe and the LIFT, with a PROPER significance test
     (paired bootstrap on the Sharpe difference -- the honest test)
  4. Block-bootstrap CI (respects autocorrelation) on combined-minus-best
  5. Return-stream correlation, recomputed
  6. Transaction-cost sensitivity (what the lift becomes net of costs)

It REBUILDS the streams from prices/earnings/short_interest (same construction
as combine_pead_si.py) so the inputs are real, then shows the statistics with
full arithmetic. Every formula is printed.

RULE 1: this script's PURPOSE is to check the prior result. It prints raw
intermediate values (n, sums, variances) so nothing is hidden. READ-ONLY.

USAGE:
  python audit_combination.py --root .
  python audit_combination.py --root . --n-boot 5000
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
def month_key(d): return (d.year,d.month)
LINE="="*78

def build_streams(root, hold_days, quantile, min_names, pead_window, clip_dtc):
    prices_db=os.path.join(root,"prices.db"); si_db=os.path.join(root,"short_interest.db")
    earnp=find_db(root,"earnings.db")
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
    alldays=sorted(set(d for lst in px.values() for d,_ in lst))
    month_last={}
    for d in alldays: month_last[month_key(d)]=d
    grid=[month_last[k] for k in sorted(month_last)]
    def fwd_ret(tk,d,h):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=idx.get(d)
        if i is None:
            for off in range(0,5):
                cc=d-datetime.timedelta(days=off)
                if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+h
        if x>=len(lst): return None
        p0=lst[i][1]; return (lst[x][1]/p0-1.0) if p0>0 else None
    def ls_return(ranked,h,d):
        vals=[(tk,v) for tk,v in ranked if tk in pos_of]
        if len(vals)<min_names: return None
        vals.sort(key=lambda x:x[1])
        q=max(1,int(len(vals)*quantile)); low=vals[:q]; high=vals[-q:]
        lr=[fwd_ret(tk,d,h) for tk,_ in low]; hr=[fwd_ret(tk,d,h) for tk,_ in high]
        lr=[x for x in lr if x is not None]; hr=[x for x in hr if x is not None]
        if len(lr)<3 or len(hr)<3: return None
        return np.mean(lr)-np.mean(hr)
    # SI
    c=ro(si_db)
    try: sirows=Q(c,"SELECT ticker,settlement_date,days_to_cover FROM short_interest")
    finally: c.close()
    si_by_date=defaultdict(dict)
    for tk,d,v in sirows:
        do=nd(d)
        if do is None or v is None: continue
        try: fv=float(v)
        except Exception: continue
        if clip_dtc and fv>clip_dtc: continue
        si_by_date[do][tk.upper()]=fv
    si_dates=sorted(si_by_date)
    def latest_si(d):
        best=None
        for sd in si_dates:
            if sd<=d: best=sd
            else: break
        return si_by_date.get(best) if best else None
    si_stream={}
    for d in grid:
        snap=latest_si(d)
        if not snap: continue
        r=ls_return([(tk,v) for tk,v in snap.items()],hold_days,d)
        if r is not None: si_stream[month_key(d)]=r
    # PEAD
    ce=ro(earnp)
    try:
        cl=cols_of(ce,"earnings_surprises")
        comp="eps_actual" in cl and "eps_estimate" in cl
        sel="ticker,report_date"+(",eps_actual,eps_estimate" if comp else ",eps_surprise_pct")
        ev=Q(ce,"SELECT "+sel+" FROM earnings_surprises WHERE report_date IS NOT NULL")
        _n_ev = Q(ce, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='earnings_events'")[0][0]
        _n_ev = Q(ce, "SELECT COUNT(*) FROM earnings_events WHERE eps_surprise IS NOT NULL")[0][0] if _n_ev else 0
        if _n_ev > 1000:
            comp = False
            ev = Q(ce, "SELECT ticker, announce_date, eps_surprise FROM earnings_events "
                       "WHERE eps_surprise IS NOT NULL AND announce_date IS NOT NULL")
            print("  PEAD source: earnings_events.announce_date (%d rows) [LEAK-FIXED]" % len(ev))
        else:
            print("  PEAD source: earnings_surprises (fallback -- KNOWN LEAKED DATES)")
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
    pead_stream={}
    for d in grid:
        ranked=[]
        for tk,evs in sue_events.items():
            recent=[(ed,s) for ed,s in evs if 0<=(d-ed).days<=pead_window]
            if recent:
                recent.sort(); ranked.append((tk,-recent[-1][1]))  # neg sue: long high-sue
        r=ls_return(ranked,hold_days,d)
        if r is not None: pead_stream[month_key(d)]=r
    return si_stream, pead_stream

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--hold-days",type=int,default=21)
    ap.add_argument("--quantile",type=float,default=0.2)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--pead-window",type=int,default=45)
    ap.add_argument("--clip-dtc",type=float,default=50.0)
    ap.add_argument("--n-boot",type=int,default=5000)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    print("\n"+LINE+"\nAUDIT 1 — RECOMPUTING THE COMBINATION NUMBERS FROM RAW STREAMS\n"+LINE)

    si_stream,pead_stream=build_streams(a.root,a.hold_days,a.quantile,a.min_names,a.pead_window,a.clip_dtc)
    common=sorted(set(si_stream)&set(pead_stream))
    print("\n  SI months=%d  PEAD months=%d  COMMON months=%d"%(len(si_stream),len(pead_stream),len(common)))
    if len(common)<6:
        print("  [STOP] too few common months"); return
    pe=np.array([pead_stream[m] for m in common])
    si=np.array([si_stream[m] for m in common])
    n=len(common)

    print("\n"+"-"*78+"\nSTEP 1: monthly mean / vol / Sharpe — SHOWING THE ARITHMETIC\n"+"-"*78)
    def show_sharpe(name,x):
        m=x.mean(); v=x.std(ddof=1); sr_m=m/v if v>0 else 0; sr_a=sr_m*math.sqrt(12)
        print("  %s:"%name)
        print("    n=%d  sum=%.5f  mean=sum/n=%.5f"%(len(x),x.sum(),m))
        print("    variance(ddof=1)=%.6f  std=sqrt(var)=%.5f"%(v**2,v))
        print("    monthly Sharpe = mean/std = %.5f/%.5f = %.4f"%(m,v,sr_m))
        print("    annualized = monthly_SR * sqrt(12) = %.4f * %.4f = %.4f"%(sr_m,math.sqrt(12),sr_a))
        return sr_a
    sr_pe=show_sharpe("PEAD",pe)
    sr_si=show_sharpe("SHORT-INT",si)

    print("\n"+"-"*78+"\nSTEP 2: Sharpe standard error (Lo 2002) + 95%% CI\n"+"-"*78)
    print("  Formula: SE(SR_monthly) = sqrt( (1 + 0.5*SR_monthly^2) / n )")
    def sharpe_ci(sr_a,n):
        sr_m=sr_a/math.sqrt(12)
        se_m=math.sqrt((1+0.5*sr_m**2)/n)
        se_a=se_m*math.sqrt(12)
        return se_a,(sr_a-1.96*se_a,sr_a+1.96*se_a)
    for name,sr in (("PEAD",sr_pe),("SHORT-INT",sr_si)):
        se,(lo,hi)=sharpe_ci(sr,n)
        print("  %-10s SR=%.3f  SE=%.3f  95%% CI=[%.3f, %.3f]"%(name,sr,se,lo,hi))

    print("\n"+"-"*78+"\nSTEP 3: combined Sharpe + the LIFT\n"+"-"*78)
    comb=0.5*pe+0.5*si
    sr_comb=show_sharpe("COMBINED 50/50",comb)
    best=max(sr_pe,sr_si); best_name="PEAD" if sr_pe>=sr_si else "SHORT-INT"
    lift=sr_comb-best
    print("\n  best single = %s = %.4f"%(best_name,best))
    print("  combined = %.4f"%sr_comb)
    print("  LIFT = combined - best = %.4f  (%.1f%% relative)"%(lift,100*lift/best if best else 0))

    print("\n"+"-"*78+"\nSTEP 4: is the LIFT real? PAIRED BLOCK-BOOTSTRAP (honest test)\n"+"-"*78)
    print("  Resample months in BLOCKS (preserves autocorrelation), recompute the")
    print("  Sharpe difference each time, build its distribution. If 95%% CI excludes")
    print("  0, the lift is real; if it spans 0, it's noise.")
    rng=np.random.default_rng(42)
    block=3  # ~quarter blocks
    def block_resample(idx_n):
        out=[]
        while len(out)<idx_n:
            start=rng.integers(0,idx_n)
            out.extend([(start+j)%idx_n for j in range(block)])
        return np.array(out[:idx_n])
    diffs=[]
    for _ in range(a.n_boot):
        idx=block_resample(n)
        p=pe[idx]; s=si[idx]; c=0.5*p+0.5*s
        def sa(x):
            v=x.std(ddof=1); return (x.mean()/v*math.sqrt(12)) if v>0 else 0
        diffs.append(sa(c)-max(sa(p),sa(s)))
    diffs=np.array(diffs)
    lo,hi=np.percentile(diffs,[2.5,97.5])
    frac_pos=np.mean(diffs>0)
    print("\n  bootstrap mean lift = %.4f"%diffs.mean())
    print("  95%% CI of lift = [%.4f, %.4f]"%(lo,hi))
    print("  fraction of resamples with lift>0 = %.1f%%"%(100*frac_pos))
    if lo>0:
        print("  >> CI excludes 0 -> the lift IS statistically real.")
    else:
        print("  >> CI SPANS 0 -> the lift is NOT statistically distinguishable from zero.")
        print("     The point-estimate lift is within sampling noise at n=%d. UNDERPOWERED."%n)

    print("\n"+"-"*78+"\nSTEP 5: correlation, recomputed\n"+"-"*78)
    rho=float(np.corrcoef(pe,si)[0,1])
    cov=np.mean((pe-pe.mean())*(si-si.mean()))
    print("  cov(PEAD,SI)=%.6f  std_PEAD=%.5f  std_SI=%.5f"%(cov,pe.std(),si.std()))
    print("  correlation = cov/(std*std) = %.4f"%rho)
    print("  (this is the RETURN-STREAM correlation -- the one that governs diversification)")

    print("\n"+"-"*78+"\nSTEP 6: transaction-cost sensitivity\n"+"-"*78)
    print("  Quintile LS rebalanced monthly turns over ~fully each month on the legs.")
    print("  Approximate cost = 2 * quantile_turnover * cost_bps applied to each leg.")
    cost=2*(a.cost_bps/10000.0)  # rough: long+short, ~full turnover
    pe_net=pe-cost; si_net=si-cost; comb_net=0.5*pe_net+0.5*si_net
    def sa(x):
        v=x.std(ddof=1); return (x.mean()/v*math.sqrt(12)) if v>0 else 0
    print("  at %d bps/side: PEAD %.2f->%.2f  SI %.2f->%.2f  combined %.2f->%.2f"
          %(a.cost_bps,sr_pe,sa(pe_net),sr_si,sa(si_net),sr_comb,sa(comb_net)))
    print("  net lift = %.4f  (was %.4f gross)"%(sa(comb_net)-max(sa(pe_net),sa(si_net)),lift))

    print("\n"+LINE+"\nAUDIT 1 CONCLUSION\n"+LINE)
    print("  All numbers recomputed from raw streams with arithmetic shown.")
    print("  The decisive line is STEP 4: does the bootstrap CI of the lift exclude 0?")
    print("  If it spans 0, 'THESIS CONFIRMED' was overstated and the honest verdict is")
    print("  'underpowered at n=%d common months'. Read STEP 4 above."%n)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
