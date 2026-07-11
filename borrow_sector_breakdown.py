#!/usr/bin/env python3
"""
borrow_sector_breakdown.py -- WHERE does the fee_gt_5pct residual-vs-DTC signal live?

Descriptive, even-handed. Reports the per-date residual IC of the SAME signal the
battery tested (fee_gt_5pct residualized on days_to_cover, h=40 fwd return) computed
SEPARATELY WITHIN EACH SECTOR BUCKET, all buckets ranked side by side. Tech is shown
in context vs every other sector -- NOT cherry-picked.

This does NOT create a new pass/fail significance claim on a chosen subset (that would
be p-hacking). It characterizes the mechanism of an effect already shown to be ~65%
sector-driven. The "do not wire" verdict from the battery stands regardless.

Caveats printed inline: within-sector breadth is low (few names/date), so per-sector
NW-t is noisy -- read the RANKING and sign consistency, not any single t as decisive.

READ-ONLY. borrow.db + prices.db + short_interest.db + tickers_metadata.csv.

RUN
  python borrow_sector_breakdown.py
  python borrow_sector_breakdown.py --min-names 8
"""

import argparse, os, sqlite3, math, datetime, csv, sys
from collections import defaultdict
import numpy as np

def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1", uri=True, timeout=30)
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
def newey_west_se_mean(x,lag):
    x=np.asarray(x,float); n=len(x)
    if n<2: return None
    e=x-x.mean(); g0=float(e@e)/n; s=g0
    for k in range(1,min(lag,n-1)+1):
        gk=float(e[k:]@e[:-k])/n; w=1.0-k/(lag+1.0); s+=2.0*w*gk
    v=s/n
    return math.sqrt(v) if v>0 else None
def nw_t(ics,hold,gap=15):
    ics=np.asarray(ics,float)
    if len(ics)<2: return None,None
    lag=max(1,int(math.ceil(hold/float(gap))))
    se=newey_west_se_mean(ics,lag); m=float(ics.mean())
    return m,(m/se if se else None)

def load_prices(pdb):
    cp=ro(pdb)
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
    pos={tk:{d:i for i,(d,_) in enumerate(l)} for tk,l in px.items()}
    return px,pos
def mk_fwd(px,pos):
    def fwd(tk,d,h):
        l=px.get(tk); idx=pos.get(tk)
        if not l or not idx: return None
        i=None
        for off in range(0,6):
            cc=d+datetime.timedelta(days=off)
            if cc in idx: i=idx[cc]; break
        if i is None: return None
        x=i+h
        if x>=len(l): return None
        p0=l[i][1]
        return (l[x][1]/p0-1.0) if p0>0 else None
    return fwd
def load_borrow(bdb):
    c=ro(bdb)
    try: rows=Q(c,'SELECT ticker,asof_date,fee_gt_5pct FROM borrow_features')
    finally: c.close()
    bd=defaultdict(list)
    for tk,d,fl in rows:
        do=nd(d)
        if do is None or fl is None: continue
        try: v=float(fl)
        except Exception: continue
        bd[do].append((tk.upper(),v))
    return bd
def load_dtc(sdb):
    c=ro(sdb)
    try: rows=Q(c,'SELECT ticker,settlement_date,days_to_cover FROM short_interest')
    finally: c.close()
    dtc={}
    for tk,d,v in rows:
        do=nd(d)
        if do is None or v is None: continue
        try: fv=float(v)
        except Exception: continue
        if fv>50: continue
        dtc[(tk.upper(),do)]=fv
    return dtc
def load_sectors(path):
    sec={}
    if not os.path.isfile(path): return sec
    with open(path) as f:
        for row in csv.DictReader(f):
            tk=(row.get("ticker") or "").upper()
            b=row.get("bucket") or row.get("sector") or ""
            if tk: sec[tk]=b
    return sec

def resid_ic_within_sector(bd, dtc, sector, fwd, hold, target_sector, min_names):
    """Residualize fee_gt_5pct on DTC WITHIN the target sector's stocks per date,
    IC residual vs fwd return. Returns list of (date, ic, n)."""
    out=[]
    for d in sorted(bd):
        recs=[(tk,fl) for tk,fl in bd[d]
              if sector.get(tk)==target_sector and dtc.get((tk,d)) is not None]
        if len(recs)<min_names:
            continue
        feat=np.array([fl for _,fl in recs],float)
        dcov=np.array([dtc[(tk,d)] for tk,_ in recs],float)
        if feat.std()==0:  # all same flag value -> no cross-section
            continue
        if dcov.std()==0:
            resid=feat-feat.mean()
        else:
            A=np.vstack([np.ones_like(dcov),dcov]).T
            coef,*_=np.linalg.lstsq(A,feat,rcond=None)
            resid=feat-A@coef
        sig=[]; ret=[]
        for (tk,_),rv in zip(recs,resid):
            r=fwd(tk,d,hold)
            if r is not None:
                sig.append(rv); ret.append(r)
        if len(sig)<min_names:
            continue
        ic=spearman(np.array(sig),np.array(ret))
        if ic is not None:
            out.append((d,ic,len(sig)))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--borrow-db",default=None)
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--si-db",default=None)
    ap.add_argument("--meta",default=None)
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--min-names",type=int,default=8)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    bdb=a.borrow_db or os.path.join(a.root,"borrow.db")
    pdb=a.prices_db or os.path.join(a.root,"prices.db")
    sdb=a.si_db or os.path.join(a.root,"short_interest.db")
    meta=a.meta or os.path.join(a.root,"tickers_metadata.csv")

    print("loading ...")
    px,pos=load_prices(pdb); fwd=mk_fwd(px,pos)
    bd=load_borrow(bdb); dtc=load_dtc(sdb); sector=load_sectors(meta)
    hold=a.hold

    # sector sizes (distinct tickers per bucket in the borrow universe)
    tks=set(tk for d in bd for tk,_ in bd[d])
    bucket_names=defaultdict(list)
    for tk in tks:
        bucket_names[sector.get(tk,"(none)")].append(tk)

    print(f"\n{'='*78}\n  SECTOR BREAKDOWN: fee_gt_5pct residual-vs-DTC IC (h={hold}), per bucket\n{'='*78}")
    print(f"  min_names/date = {a.min_names}  (LOW within-sector breadth -> t is noisy;")
    print(f"  read the RANKING + sign, not any single t as decisive)\n")

    results=[]
    for b in sorted(bucket_names, key=lambda x: -len(bucket_names[x])):
        n_tk=len(bucket_names[b])
        ics=resid_ic_within_sector(bd,dtc,sector,fwd,hold,b,a.min_names)
        if len(ics)<6:
            results.append((b,n_tk,None,None,len(ics),None))
            continue
        arr=np.array([ic for _,ic,_ in ics])
        m,t=nw_t(arr,hold)
        pos_pct=100.0*np.mean(arr>0)
        results.append((b,n_tk,m,t,len(ics),pos_pct))

    # rank by mean IC (desc), Nones last
    ranked=sorted(results,key=lambda r:(r[2] is None, -(r[2] or -9)))
    print(f"  {'SECTOR (bucket)':<22} {'#tk':>4} {'meanIC':>9} {'NW-t':>7} {'dates':>6} {'%>0':>5}")
    print("  "+"-"*70)
    for b,n_tk,m,t,ndates,pos_pct in ranked:
        if m is None:
            print(f"  {b:<22} {n_tk:>4} {'--':>9} {'--':>7} {ndates:>6} {'(<6 dates)':>8}")
        else:
            mark=""
            if "tech" in b.lower() or "silicon" in b.lower() or "semi" in b.lower(): mark="  <- TECH"
            print(f"  {b:<22} {n_tk:>4} {m:>+9.4f} {t:>+7.2f} {ndates:>6} {pos_pct:>4.0f}%{mark}")
    print("  "+"-"*70)
    print("\n  HOW TO READ:")
    print("  - This is DESCRIPTIVE (where the sector-driven effect concentrates), NOT a")
    print("    new pass/fail test. Do-not-wire stands (battery: 35% sector-neutral, OOS t+1.47).")
    print("  - If tech buckets cluster at the TOP with consistent +sign, the effect is a")
    print("    tech/growth phenomenon. If tech is mid-pack, it is broad-but-sector-mixed.")
    print("  - Low within-sector breadth => a single sector's t is unreliable; the pattern")
    print("    across sectors is the signal, not any one row.")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
