#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — WALK-FORWARD RE-DERIVATION (raw vs overlay, out-of-sample)
================================================================================
The decision-grade test. Reconstructs a PROPER walk-forward from your existing
predictions + outcomes (no need for the Sunday job), and answers:

   "Does the overlay's AUC/IC gain over the raw model SURVIVE out-of-sample,
    or does it evaporate / flip in recent folds (= overfit)?"

It does NOT retrain anything — your prob_raw and prob_up are already the model's
out-of-sample predictions logged over time. This script simply:
  1. restricts to the window where prob_raw is populated (~May 2026 on),
  2. cuts rolling WEEKLY test folds,
  3. PURGES: a prediction made on date D for horizon h resolves at D+h, so a fold
     boundary embargoes the h days around it to avoid label overlap leakage,
  4. computes auc_raw / auc_eff / ic_raw / ic_eff per fold per horizon,
  5. reports the per-fold gain and whether it is stable, shrinking, or flipping.

READ-ONLY. Never writes. SQLite opened mode=ro&immutable=1.

USAGE (project root, env active):
  python wf_rederive.py --root .
  python wf_rederive.py --root . --fold-days 7 --min-fold-n 50
  add --out wf_rederive.jsonl for machine-readable results.

Stats: exact pure-python (Mann-Whitney AUC, Spearman). numpy if present.
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
    return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=20)
def q(c,s,p=()): return c.execute(s,p).fetchall()
def has_table(c,n): return bool(q(c,"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(n,)))
def cols_of(c,t): return [r[1] for r in q(c,'PRAGMA table_info("'+t+'")')]
def require(cond,msg):
    if not cond: print("  [STOP] "+msg); return False
    return True
def find_db(root,name):
    cand=os.path.join(root,name)
    if os.path.isfile(cand): return cand
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        if name in fn: return os.path.join(dp,name)
    return None

def auc(scores,labels):
    pairs=sorted(zip(scores,labels)); npl=sum(1 for _,l in pairs if l==1); nmi=len(pairs)-npl
    if npl==0 or nmi==0: return None
    ranks=[0.0]*len(pairs); i=0; r=1
    while i<len(pairs):
        j=i
        while j+1<len(pairs) and pairs[j+1][0]==pairs[i][0]: j+=1
        avg=(r+(r+(j-i)))/2.0
        for k in range(i,j+1): ranks[k]=avg
        r+=(j-i+1); i=j+1
    sp=sum(rk for rk,(_,l) in zip(ranks,pairs) if l==1)
    return (sp-npl*(npl+1)/2.0)/(npl*nmi)

def spearman(x,y):
    n=len(x)
    if n<3: return None
    def rk(v):
        o=sorted(range(n),key=lambda i:v[i]); rr=[0.0]*n; i=0
        while i<n:
            j=i
            while j+1<n and v[o[j+1]]==v[o[i]]: j+=1
            a=(i+j)/2.0+1
            for k in range(i,j+1): rr[o[k]]=a
            i=j+1
        return rr
    rx,ry=rk(x),rk(y); mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den=math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else None

def daterange_weeks(dmin, dmax, fold_days):
    """Yield (start,end) ISO date strings for consecutive fold windows."""
    d0=datetime.date.fromisoformat(dmin); d1=datetime.date.fromisoformat(dmax)
    cur=d0
    while cur<=d1:
        end=cur+datetime.timedelta(days=fold_days-1)
        yield cur.isoformat(), end.isoformat()
        cur=end+datetime.timedelta(days=1)

def cmd(args):
    accp=find_db(args.root,"accuracy.db")
    if not require(accp,"accuracy.db not found"): return
    conn=ro(accp); report={"folds":[]}
    try:
        for t in ("predictions","outcomes"):
            if not require(has_table(conn,t),"missing "+t): return
        pc=cols_of(conn,"predictions"); oc=cols_of(conn,"outcomes")
        for need in ("ticker","prediction_date","horizon","prob_up","prob_raw"):
            if not require(need in pc,"predictions missing "+need): return
        for need in ("ticker","prediction_date","horizon","actual_up","actual_return"):
            if not require(need in oc,"outcomes missing "+need): return

        # window where prob_raw is populated
        dr=q(conn,"SELECT MIN(prediction_date),MAX(prediction_date) FROM predictions WHERE prob_raw IS NOT NULL")[0]
        if not require(dr[0] is not None,"no rows with prob_raw"): return
        print("  prob_raw populated window: %s .. %s" % (dr[0],dr[1]))
        print("  fold size: %d days ; min rows/fold: %d ; horizons: 1,3,5"
              % (args.fold_days,args.min_fold_n))

        for h in (1,3,5):
            sub("HORIZON h=%d  (purge/embargo = %d days around fold edges)" % (h,h))
            # pull all joined rows once for this horizon
            rows=q(conn,
                "SELECT p.prediction_date, p.prob_raw, p.prob_up, o.actual_up, o.actual_return "
                "FROM predictions p JOIN outcomes o "
                "ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon "
                "WHERE p.horizon=? AND p.prob_raw IS NOT NULL AND p.prob_up IS NOT NULL "
                "AND o.actual_up IS NOT NULL", (h,))
            if len(rows)<args.min_fold_n*2:
                print("  too few joined rows (%d) — skip" % len(rows)); continue
            by_date=defaultdict(list)
            for d,pr,pu,au,ret in rows: by_date[d].append((pr,pu,au,ret))

            fold_results=[]
            print("  %-26s %6s | %-7s %-7s %-7s | %-7s %-7s %-7s"
                  % ("fold (test window)","n","auc_raw","auc_eff","dAUC","ic_raw","ic_eff","dIC"))
            for fs,fe in daterange_weeks(dr[0],dr[1],args.fold_days):
                # collect rows whose prediction_date in [fs,fe]; purge handled by gap between folds
                # (consecutive non-overlapping weekly windows; to embargo label overlap we drop
                #  the last h days of each fold so labels don't bleed into next fold's window)
                fe_dt=datetime.date.fromisoformat(fe)
                purge_cut=(fe_dt-datetime.timedelta(days=h)).isoformat()
                pr_l=[]; pu_l=[]; au_l=[]; ret_l=[]
                for d,recs in by_date.items():
                    if fs<=d<=purge_cut:  # embargo last h days of the window
                        for pr,pu,au,ret in recs:
                            pr_l.append(pr); pu_l.append(pu); au_l.append(au); ret_l.append(ret)
                n=len(pr_l)
                if n<args.min_fold_n:
                    continue
                a_raw=auc(pr_l,au_l); a_eff=auc(pu_l,au_l)
                i_raw=spearman(pr_l,ret_l); i_eff=spearman(pu_l,ret_l)
                dauc=(a_eff-a_raw) if (a_raw is not None and a_eff is not None) else None
                dic=(i_eff-i_raw) if (i_raw is not None and i_eff is not None) else None
                print("  %s..%s %6d | %-7s %-7s %-7s | %-7s %-7s %-7s"
                      % (fs,fe,n,
                         _f(a_raw),_f(a_eff),_fp(dauc),
                         _f(i_raw),_f(i_eff),_fp(dic)))
                fold_results.append((fs,dauc,dic,n))
                report["folds"].append({"h":h,"start":fs,"n":n,
                    "auc_raw":a_raw,"auc_eff":a_eff,"dauc":dauc,
                    "ic_raw":i_raw,"ic_eff":i_eff,"dic":dic})

            # summary for this horizon: is the gain stable / positive / decaying?
            if fold_results:
                gains=[g for _,g,_,_ in fold_results if g is not None]
                if gains:
                    mean_g=sum(gains)/len(gains)
                    pos=sum(1 for g in gains if g>0)
                    # trend: compare first half vs second half mean
                    half=len(gains)//2
                    early=gains[:half] or gains; late=gains[half:] or gains
                    me=sum(early)/len(early); ml=sum(late)/len(late)
                    print("  -> mean dAUC across %d folds = %+.4f ; positive in %d/%d folds"
                          % (len(gains),mean_g,pos,len(gains)))
                    print("     early-folds mean dAUC=%+.4f  late-folds mean dAUC=%+.4f  (%s)"
                          % (me,ml,
                             "DECAYING" if ml<me-0.003 else ("growing" if ml>me+0.003 else "stable")))
                    if mean_g<=0.001:
                        print("     VERDICT h=%d: overlay gain ~0 OOS — the overlay is NOT a real edge." % h)
                    elif pos>=0.7*len(gains) and ml>=0:
                        print("     VERDICT h=%d: overlay gain positive & persistent — plausibly real (small)." % h)
                    else:
                        print("     VERDICT h=%d: overlay gain inconsistent — likely partly overfit." % h)
        print("\n  BOTTOM LINE: if dAUC is ~0 or flips negative in late folds, the overlay's apparent")
        print("  edge was in-sample curve-fit. Combined with raw AUC ~0.50, that means the live system")
        print("  is at coin-flip OOS and the priority is rebuilding the base model, not tuning overlays.")
        if not HAVE_NUMPY:
            print("\n  [note] numpy absent — used exact pure-python stats.")
    finally:
        conn.close()
    _w(args,report)

def _f(x): return ("%.4f"%x) if x is not None else "NA"
def _fp(x): return ("%+.4f"%x) if x is not None else "NA"

def _w(args,report):
    if not args.out: return
    path=args.out
    if os.path.isdir(path) or path.endswith("/"): path=os.path.join(path,"wf_rederive.json")
    with open(path,"a") as f:
        f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":report})+"\n")
    print("\n  [report appended to %s]" % path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--fold-days",type=int,default=7)
    ap.add_argument("--min-fold-n",type=int,default=50)
    ap.add_argument("--out",default=None)
    args=ap.parse_args(); args.root=os.path.expanduser(args.root)
    banner("ML QUANT FUND — WALK-FORWARD RE-DERIVATION (raw vs overlay OOS)")
    print("Read-only. No writes. Root:",os.path.abspath(args.root),
          "| Python",sys.version.split()[0],"| numpy:",HAVE_NUMPY)
    cmd(args)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
