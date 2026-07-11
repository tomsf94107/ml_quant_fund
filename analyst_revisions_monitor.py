#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — ANALYST REVISIONS MONITOR (self-gating; brick #3 candidate)
================================================================================
Analyst revisions are a documented signal (Chan-Jegadeesh-Lakonishok): when analysts
RAISE estimates/targets, the stock tends to outperform; cuts -> underperform. The
signal is the CHANGE in consensus over time, so it needs MANY snapshot dates to measure.

CURRENT STATE: accuracy.db.analyst_snapshots has only ~3 distinct snap_dates (the
weekly snapshot cron started recently). You CANNOT validate a change-over-time signal
on 3 dates spanning one week -- there's no "over time" yet. So this is a MONITOR, not a
validator: it self-gates exactly like pead_monitor.py and combined_40d_monitor.py --
checks cheaply, and only runs the full per-date IC validation once enough distinct
snapshot dates have accumulated (default >=24, ~6 months of weekly snapshots).

>>> DEPENDENCY: your weekly `scripts.analyst_snapshot` cron must keep running to
    accumulate snap_dates. ~1/week -> ~24 dates in ~6 months. The monitor reports how
    many more are needed each run.

THE SIGNAL (computed once data is sufficient): for each ticker, the CHANGE in consensus
between consecutive snapshots -- delta(mean_target) as a fraction, or delta(buy_pct),
or the recent_upgrade/downgrade flags. Cross-sectional IC of that change vs forward
return, per snapshot date, Newey-West t, null control -- the SAME audited machinery as
the short-interest brick.

SCHEMA (accuracy.db.analyst_snapshots): id, ticker, snap_date, payload_json
  payload_json fields seen: analyst_multiplier, n_buy, n_hold, n_sell, buy_pct,
  mean_target, target_upside, recent_upgrade, recent_downgrade.
  The revision signal uses the CHANGE in mean_target (or buy_pct) between snapshots.

RULE 1: per-date IC + Newey-West; the change uses prior snapshot < current (PIT -- no
look-ahead); forward returns strictly after snap_date; null control; self-gates on data
sufficiency; READ-ONLY except appending to the monitor log.

USAGE (run weekly via cron; self-gates):
  python analyst_revisions_monitor.py --root .                # check; validate if enough dates
  python analyst_revisions_monitor.py --root . --force        # force validation attempt now
  python analyst_revisions_monitor.py --root . --min-dates 24 # dates needed to trigger
  python analyst_revisions_monitor.py --root . --show-log
================================================================================
"""
import argparse, os, sqlite3, math, datetime, csv, json
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
LOG="analyst_revisions_monitor_log.csv"

def load_snapshots(acc_db):
    c=ro(acc_db)
    try: rows=Q(c,"SELECT ticker,snap_date,payload_json FROM analyst_snapshots")
    finally: c.close()
    # ticker -> sorted list of (date, field_dict)
    by_tk=defaultdict(list)
    for tk,d,pj in rows:
        do=nd(d)
        if do is None: continue
        try: payload=json.loads(pj) if pj else {}
        except Exception: payload={}
        by_tk[tk.upper()].append((do,payload))
    for tk in by_tk: by_tk[tk].sort()
    dates=sorted(set(do for lst in by_tk.values() for do,_ in lst))
    return by_tk, dates

def last_logged(root):
    p=os.path.join(root,LOG)
    if not os.path.isfile(p): return 0, None
    try:
        with open(p) as f: rows=list(csv.DictReader(f))
        an=[r for r in rows if r.get("triggered")=="yes"]
        return (int(an[-1]["n_dates"]), an[-1].get("verdict")) if an else (0,None)
    except Exception: return 0, None

def _append(path,row):
    new=not os.path.isfile(path)
    with open(path,"a",newline="") as f:
        w=csv.writer(f)
        if new: w.writerow(["date","n_dates","triggered","field","n_obs","mean_ic","nw_t","null_z","verdict"])
        w.writerow(row)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--acc-db",default=None)
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--field",default="mean_target",help="consensus field to diff: mean_target or buy_pct")
    ap.add_argument("--hold",type=int,default=20)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--min-dates",type=int,default=24,help="distinct snap_dates needed to validate")
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--show-log",action="store_true")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    acc_db=a.acc_db or os.path.join(a.root,"accuracy.db")
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    logpath=os.path.join(a.root,LOG)

    if a.show_log:
        print(open(logpath).read() if os.path.isfile(logpath) else "  no log yet at %s"%logpath); return

    print("\n"+LINE+"\nANALYST REVISIONS MONITOR (self-gating)\n"+LINE)
    today=datetime.date.today().isoformat()
    if not os.path.isfile(acc_db): print("  [STOP] accuracy.db not found"); return
    by_tk, dates = load_snapshots(acc_db)
    n_dates=len(dates)
    prev_n, prev_verdict = last_logged(a.root)
    enough = n_dates>=a.min_dates
    trigger = a.force or enough

    print("  analyst_snapshots: %d distinct snap_dates (%s to %s)"%(n_dates, dates[0] if dates else "-", dates[-1] if dates else "-"))
    print("  need >=%d distinct dates to validate a change-over-time signal."%a.min_dates)

    if not trigger:
        need=a.min_dates-n_dates
        weeks=need  # ~1 snapshot/week
        print("  >> NOT ENOUGH YET: %d more dates needed (~%d weeks of weekly snapshots)."%(need,weeks))
        print("     A revision signal is the CHANGE in consensus between dates; 3 dates over one")
        print("     week can't measure that. The weekly analyst_snapshot cron accumulates these.")
        print("     This monitor will auto-run the validation once the threshold is reached.")
        _append(logpath,[today,n_dates,"no","","","","","","insufficient_dates"])
        print("\n  logged. (data-gated; no premature test on thin data)")
        return

    # --- enough dates: run the per-date IC validation of consensus CHANGE ---
    if not os.path.isfile(prices_db):
        print("  [STOP] prices.db not found (needed for forward returns)"); return
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

    def change_at(tk, d):
        """consensus CHANGE: (field at d) / (field at most recent prior snapshot) - 1.
        PIT: uses only the prior snapshot strictly before d."""
        lst=by_tk.get(tk)
        if not lst: return None
        cur=None; prev=None
        for do,payload in lst:
            if do==d: cur=payload.get(a.field)
            elif do<d: prev=payload.get(a.field) if payload.get(a.field) is not None else prev
        if cur is None or prev is None: return None
        try:
            cur=float(cur); prev=float(prev)
        except Exception: return None
        if prev==0: return None
        return cur/prev-1.0

    def compute(shuffle=False,rng=None):
        ics=[]; ds=[]
        for d in dates:
            recs=[]
            for tk in by_tk:
                if tk not in pos_of: continue
                ch=change_at(tk,d)
                if ch is None: continue
                r=fwd(tk,d)
                if r is None: continue
                recs.append((tk,ch,r))
            if len(recs)<a.min_names: continue
            sig=np.array([c for _,c,_ in recs])   # +change predicts +return (upgrades good)
            ret=np.array([r for _,_,r in recs])
            if shuffle: ret=rng.permutation(ret)
            ic=spearman(sig,ret)
            if ic is not None: ics.append(ic); ds.append(d)
        return np.array(ics),ds

    ics,ds=compute()
    N=len(ds)
    if N<6:
        print("  >> dates sufficient but usable cross-sections thin (%d). Need more overlap of"%N)
        print("     {change available AND forward return} per date. Keep accumulating.")
        _append(logpath,[today,n_dates,"yes",a.field,N,"","","","too_thin"])
        return
    lag=max(1,int(math.ceil(a.hold/15.0)))
    mean_ic=ics.mean(); se=nw_se_mean(ics,lag); t=mean_ic/se if se else 0
    rng=np.random.default_rng(11); nulls=[]
    for _ in range(200):
        nc,_=compute(shuffle=True,rng=rng)
        if len(nc): nulls.append(nc.mean())
    nulls=np.array(nulls); z=(mean_ic-nulls.mean())/nulls.std() if nulls.std()>0 else 0

    print("\n  >> VALIDATION (delta %s, hold %dd, %d cross-sections)"%(a.field,a.hold,N))
    print("  mean per-date IC = %+.4f | Newey-West t = %+.2f | null %.1f std's from 0"%(mean_ic,t,z))
    if abs(z)>=3 and abs(t)>=3:
        verdict="CANDIDATE_BRICK"; msg=">> CANDIDATE BRICK #3: significant + outside null. Run sector-neutral + decorrelation next."
    elif abs(z)<3:
        verdict="not_a_brick"; msg=">> NOT A BRICK: IC within null. No edge on this evidence."
    else:
        verdict="suggestive"; msg=">> SUGGESTIVE: below t>=3 bar; more dates needed."
    print("  "+msg)
    _append(logpath,[today,n_dates,"yes",a.field,N,"%.4f"%mean_ic,"%.2f"%t,"%.1f"%z,verdict])
    print("\n  logged. Honest n=%d cross-sections, %d snap_dates. In-sample, survivor-tilted."%(N,n_dates))

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
