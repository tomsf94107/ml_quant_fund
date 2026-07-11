#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — BASE MODEL TRAINING HARNESS  (the rebuild)
================================================================================
Builds an HONEST base cross-sectional model from your real per-row features,
following the low-SNR research findings:
  * model = Ridge (heavy L2) AND equal-weight 1/N, head-to-head
            (trees overfit at IC~0.02 / 6mo; 1/N is hard to beat)
  * validation = PURGED + EMBARGOED walk-forward / CPCV (can't be fooled the
            way the overlay was)
  * honesty = report OOS rank-IC and AUC vs the realistic ceiling (0.52-0.55
            AUC / 0.03-0.06 IC). If it shows MORE, suspect leakage FIRST.

DATA: reconstructs the feature matrix from `prediction_features` (your real
per-row feature store, ~33k rows, 30+ cols) JOINed to `outcomes` for labels.
Auto-discovers which feature columns exist, so it adapts to your schema.

READ-ONLY on your databases. Writes ONLY its own results file (--out, optional).

USAGE (project root, env active):
  pip install scikit-learn        # if not already present
  python base_model.py --root .
  python base_model.py --root . --horizon 1 --embargo 5 --folds 6
  python base_model.py --root . --out base_results.json

If scikit-learn is absent, the script STILL runs the equal-weight 1/N baseline
and a pure-python ridge (closed-form), so you always get a result.
================================================================================
"""
import argparse, os, sqlite3, sys, math, json, datetime
from collections import defaultdict

# ---- optional deps ----
try:
    import numpy as np; HAVE_NUMPY=True
except Exception:
    HAVE_NUMPY=False
    print("[FATAL] numpy required. pip install numpy"); 
try:
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler
    HAVE_SK=True
except Exception:
    HAVE_SK=False

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)
def ro(p):
    if not os.path.isfile(p): raise FileNotFoundError(p)
    return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
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

# ---- stats ----
def spearman_ic(pred, ret):
    """Cross-sectional rank-IC: Spearman corr of prediction vs forward return."""
    n=len(pred)
    if n<5: return None
    pr=np.argsort(np.argsort(pred)); rr=np.argsort(np.argsort(ret))
    pr=pr.astype(float); rr=rr.astype(float)
    if pr.std()==0 or rr.std()==0: return None
    return float(np.corrcoef(pr,rr)[0,1])

def auc_score(pred, label):
    """Mann-Whitney AUC."""
    order=np.argsort(pred); ranks=np.empty(len(pred)); ranks[order]=np.arange(1,len(pred)+1)
    pos=(label==1); npos=pos.sum(); nneg=len(label)-npos
    if npos==0 or nneg==0: return None
    return float((ranks[pos].sum()-npos*(npos+1)/2)/(npos*nneg))

def ridge_closed_form(X, y, alpha):
    """Pure-numpy ridge if sklearn absent. Returns coef."""
    n,p=X.shape
    A=X.T@X + alpha*np.eye(p)
    return np.linalg.solve(A, X.T@y)

# ---- CPCV / purged walk-forward ----
def purged_walkforward_folds(dates_sorted_unique, n_folds, embargo_days):
    """Yield (train_date_set, test_date_set) for expanding/rolling purged folds.
    dates: sorted unique date strings. Splits into n_folds contiguous test blocks;
    training = all dates before test block start minus embargo; PLUS dates after
    test block end + embargo (so it's a proper purged scheme, not just walk-forward)."""
    import datetime as _dt
    dd=[_dt.date.fromisoformat(d) for d in dates_sorted_unique]
    nd=len(dd)
    block=max(1, nd//n_folds)
    for k in range(n_folds):
        ts=k*block; te=(k+1)*block if k<n_folds-1 else nd
        if ts>=nd: break
        test_dates=set(dates_sorted_unique[ts:te])
        test_start=dd[ts]; test_end=dd[te-1]
        emb=_dt.timedelta(days=embargo_days)
        train=set()
        for i,d in enumerate(dd):
            if d < (test_start - emb) or d > (test_end + emb):
                train.add(dates_sorted_unique[i])
        if len(train)>=block and test_dates:
            yield train, test_dates

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--horizon",type=int,default=None,help="1/3/5; default: loop all")
    ap.add_argument("--embargo",type=int,default=None,help="embargo days; default=horizon+2")
    ap.add_argument("--folds",type=int,default=6)
    ap.add_argument("--alpha",type=float,default=10.0,help="Ridge L2 strength (heavy)")
    ap.add_argument("--min-names",type=int,default=5,help="min names per day for IC")
    ap.add_argument("--out",default=None)
    args=ap.parse_args(); args.root=os.path.expanduser(args.root)

    banner("ML QUANT FUND — BASE MODEL TRAINING HARNESS (the rebuild)")
    print("Read-only on DBs. Ridge vs equal-weight 1/N. Purged+embargoed CV.")
    print("Root:",os.path.abspath(args.root),"| numpy:",HAVE_NUMPY,"| sklearn:",HAVE_SK)
    if not HAVE_NUMPY: return

    accp=find_db(args.root,"accuracy.db")
    if not require(accp,"accuracy.db not found"): return
    conn=ro(accp); report={"horizons":{}}
    try:
        if not require(has_table(conn,"prediction_features"),"no prediction_features table"): return
        if not require(has_table(conn,"outcomes"),"no outcomes table"): return
        fc=cols_of(conn,"prediction_features"); oc=cols_of(conn,"outcomes")
        # identify join keys + feature columns
        keycols=[c for c in ("ticker","prediction_date","horizon") if c in fc]
        if not require(len(keycols)==3,"prediction_features needs ticker/prediction_date/horizon"): return
        # feature columns = numeric cols that aren't keys/ids/timestamps
        exclude=set(keycols)|{"id","created_at","prediction_ts"}
        feat_cols=[c for c in fc if c not in exclude]
        print("\n  discovered %d candidate feature columns in prediction_features:" % len(feat_cols))
        print("   ", feat_cols)

        horizons=[args.horizon] if args.horizon else [1,3,5]
        for h in horizons:
            emb=args.embargo if args.embargo is not None else h+2
            sub("HORIZON h=%d   (Ridge alpha=%.1f, %d folds, embargo=%d days)"%(h,args.alpha,args.folds,emb))
            # pull joined data
            sel_feats=", ".join('p."'+c+'"' for c in feat_cols)
            rows=q(conn,
                "SELECT p.prediction_date, p.ticker, "+sel_feats+", o.actual_return, o.actual_up "
                "FROM prediction_features p JOIN outcomes o "
                "ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon "
                "WHERE p.horizon=? AND o.actual_return IS NOT NULL", (h,))
            if len(rows)<200:
                print("  only %d joined rows — skip"%len(rows)); continue
            print("  joined rows: %d" % len(rows))
            dates=[r[0] for r in rows]; tickers=[r[1] for r in rows]
            nf=len(feat_cols)
            Xraw=np.array([[ (r[2+j] if r[2+j] is not None else np.nan) for j in range(nf)] for r in rows],dtype=float)
            ret=np.array([r[2+nf] for r in rows],dtype=float)
            lab=np.array([1 if r[3+nf] in (1,True) else 0 for r in rows],dtype=int)

            # drop feature columns that are all-nan or zero-variance
            keep_mask=[]
            for j in range(nf):
                col=Xraw[:,j]; valid=col[~np.isnan(col)]
                keep_mask.append(len(valid)>0 and valid.std()>1e-12)
            kept=[feat_cols[j] for j in range(nf) if keep_mask[j]]
            Xraw=Xraw[:,[j for j in range(nf) if keep_mask[j]]]
            print("  usable features after dropping all-nan/constant: %d" % len(kept))
            # median-impute remaining nans (per column)
            for j in range(Xraw.shape[1]):
                col=Xraw[:,j]; med=np.nanmedian(col); col[np.isnan(col)]=med; Xraw[:,j]=col

            # group rows by date for cross-sectional ops
            by_date=defaultdict(list)
            for i,d in enumerate(dates): by_date[d].append(i)
            udates=sorted(by_date.keys())
            print("  date span: %s .. %s  (%d trading days)"%(udates[0],udates[-1],len(udates)))

            # --- DETECT MACRO / CONSTANT-WITHIN-DAY FEATURES ---
            # A feature identical across all names on a day has ZERO cross-sectional variance
            # -> it cannot rank stocks (market-level values: vix_close, yield_10y, fear_greed,
            # spy_ret...). Including it adds no ranking signal AND causes the divide-by-zero that
            # corrupted Ridge. Measure avg within-day std; drop features where it's ~0.
            ncols=Xraw.shape[1]
            within_day_std=np.zeros(ncols); day_count=0
            for d,idxs in by_date.items():
                idxs=np.array(idxs)
                if len(idxs)<2: continue
                day_count+=1
                for j in range(ncols):
                    within_day_std[j]+=np.std(Xraw[idxs,j])
            within_day_std/=max(day_count,1)
            # Use a RELATIVE threshold: a feature is 'macro/constant-within-day' if its
            # average within-day std is tiny relative to its overall std. Float noise on
            # real macro columns (vix/yield) gives within-day std ~1e-6 but overall std ~10,
            # so an absolute 1e-9 cutoff misses them. Relative cutoff catches them.
            overall_std=np.array([np.std(Xraw[:,j]) for j in range(ncols)])
            rel = np.array([ (within_day_std[j]/overall_std[j]) if overall_std[j]>1e-12 else 0.0
                             for j in range(ncols)])
            cross_sectional = (within_day_std > 1e-6) & (rel > 0.05)
            macro_feats=[kept[j] for j in range(ncols) if not cross_sectional[j]]
            cs_feats=[kept[j] for j in range(ncols) if cross_sectional[j]]
            if macro_feats:
                print("  [INFO] %d MACRO/constant-within-day feats dropped (no ranking power): %s"
                      % (len(macro_feats), macro_feats))
            print("  cross-sectional features used: %d -> %s" % (len(cs_feats), cs_feats))
            if len(cs_feats)<2:
                print("  [STOP] <2 cross-sectional features — cannot build CS model at h=%d"%h); continue
            cs_idx=[j for j in range(ncols) if cross_sectional[j]]
            Xraw=Xraw[:,cs_idx]; kept=cs_feats; ncols=len(cs_idx)

            # cross-sectional rank-standardize WITHIN each date (same-day only; guarded vs zero var)
            Xcs=np.zeros_like(Xraw)
            for d,idxs in by_date.items():
                idxs=np.array(idxs)
                if len(idxs)<2: continue
                for j in range(ncols):
                    v=Xraw[idxs,j]
                    if np.std(v)<1e-12: continue
                    r=np.argsort(np.argsort(v)).astype(float)
                    rs=r.std()
                    if rs<1e-12: continue
                    Xcs[idxs,j]=(r-r.mean())/rs
            Xcs=np.nan_to_num(Xcs, nan=0.0, posinf=0.0, neginf=0.0)

            # ---- run purged walk-forward ----
            folds=list(purged_walkforward_folds(udates,args.folds,emb))
            if not folds:
                print("  not enough data for %d purged folds at embargo %d — skip"%(args.folds,emb)); continue
            print("  purged folds: %d"%len(folds))

            ew_ics=[]; ridge_ics=[]; ew_aucs=[]; ridge_aucs=[]
            print("  %-3s %7s | %-8s %-8s | %-8s %-8s"%("fk","test_n","EW_IC","RIDGE_IC","EW_AUC","RIDGE_AUC"))
            for fk,(train_d,test_d) in enumerate(folds):
                tr=[i for d in train_d for i in by_date[d]]
                te=[i for d in test_d for i in by_date[d]]
                if len(tr)<50 or len(te)<args.min_names: continue
                tr=np.array(tr); te=np.array(te)
                Xtr,Xte=Xcs[tr],Xcs[te]
                ytr=ret[tr]
                # guard: ensure finite inputs (Xcs already nan_to_num'd, but be safe)
                Xtr=np.nan_to_num(Xtr); Xte=np.nan_to_num(Xte); ytr=np.nan_to_num(ytr)
                # --- equal-weight 1/N: mean of standardized signals (sign-agnostic baseline) ---
                # align each feature's sign to its in-sample IC so EW isn't cancelled out
                signs=np.ones(Xtr.shape[1])
                for j in range(Xtr.shape[1]):
                    a=np.argsort(np.argsort(Xtr[:,j])).astype(float)
                    b=np.argsort(np.argsort(ytr)).astype(float)
                    if a.std()<1e-12 or b.std()<1e-12:
                        signs[j]=1.0; continue
                    ic_j=np.corrcoef(a,b)[0,1]
                    signs[j]=1.0 if (ic_j>=0 or np.isnan(ic_j)) else -1.0
                ew_pred_te=(Xte*signs).mean(axis=1)
                # --- Ridge (alpha floored so it can't blow up on near-singular X) ---
                alpha_eff=max(args.alpha, 1.0)
                if HAVE_SK:
                    m=Ridge(alpha=alpha_eff); m.fit(Xtr,ytr); ridge_pred_te=m.predict(Xte)
                else:
                    coef=ridge_closed_form(Xtr,ytr,alpha_eff); ridge_pred_te=Xte@coef
                ridge_pred_te=np.nan_to_num(ridge_pred_te)
                # --- per-day OOS IC, averaged over test days ---
                def avg_daily_ic(pred):
                    ics=[]
                    tmp=defaultdict(list)
                    for k_,i in enumerate(te): tmp[dates[i]].append(k_)
                    for d,ks in tmp.items():
                        if len(ks)>=args.min_names:
                            ic=spearman_ic(pred[ks], ret[te][ks])
                            if ic is not None: ics.append(ic)
                    return np.mean(ics) if ics else None
                def pooled_auc(pred):
                    return auc_score(pred, lab[te])
                ew_ic=avg_daily_ic(ew_pred_te); r_ic=avg_daily_ic(ridge_pred_te)
                ew_auc=pooled_auc(ew_pred_te); r_auc=pooled_auc(ridge_pred_te)
                if ew_ic is not None: ew_ics.append(ew_ic)
                if r_ic is not None: ridge_ics.append(r_ic)
                if ew_auc is not None: ew_aucs.append(ew_auc)
                if r_auc is not None: ridge_aucs.append(r_auc)
                print("  %-3d %7d | %-8s %-8s | %-8s %-8s"%(fk,len(te),
                    _f(ew_ic),_f(r_ic),_f(ew_auc),_f(r_auc)))

            # ---- summary ----
            def summ(name,arr):
                if not arr: print("    %-10s no folds"%name); return None
                a=np.array(arr); m=a.mean(); s=a.std(); 
                # t-stat of mean IC across folds (rough)
                t=m/(s/math.sqrt(len(a))) if s>0 else float('inf')
                print("    %-10s mean=%+.4f  std=%.4f  folds=%d  t=%.2f"%(name,m,s,len(a),t))
                return {"mean":m,"std":s,"folds":len(a),"t":t}
            print("\n  OUT-OF-SAMPLE SUMMARY (averaged across purged folds):")
            r_ew_ic=summ("EW IC",ew_ics); r_rg_ic=summ("RIDGE IC",ridge_ics)
            r_ew_au=summ("EW AUC",ew_aucs); r_rg_au=summ("RIDGE AUC",ridge_aucs)

            # verdict — now requires SIGNIFICANCE (t-stat), not just IC magnitude
            print("\n  VERDICT h=%d:"%h)
            # pick the better model by mean IC, but judge by its t-stat too
            cands=[]
            if r_ew_ic: cands.append(("EW",r_ew_ic))
            if r_rg_ic: cands.append(("RIDGE",r_rg_ic))
            if not cands:
                print("    no usable OOS IC.")
            else:
                name,best=max(cands,key=lambda c:c[1]["mean"])
                m=best["mean"]; t=best["t"]; nf=best["folds"]
                # significance gate: with only ~6 folds, demand |t|>=2 to call anything real
                significant = abs(t)>=2.0
                if not significant:
                    print("    best=%s IC=%+.4f but t=%.2f (NOT significant across %d folds)."%(name,m,t,nf))
                    print("    -> CANNOT distinguish from zero. High fold-to-fold variance = noise,")
                    print("       not edge. Treat as NO confirmed edge at h=%d."%h)
                elif m<=0.01:
                    print("    OOS IC ~0 (t=%.2f) -> no edge at h=%d. Honest coin-flip."%(t,h))
                elif m<0.03:
                    print("    OOS IC %+.4f (t=%.2f, significant) -> weak but real; below 0.03 band."%(m,t))
                elif m<=0.06:
                    print("    OOS IC %+.4f (t=%.2f, significant) -> realistic 0.03-0.06 band. PLAUSIBLY REAL."%(m,t))
                else:
                    print("    OOS IC %+.4f (t=%.2f) -> ABOVE 0.06. SUSPECT LEAKAGE FIRST."%(m,t))
                if r_rg_ic and r_ew_ic:
                    if r_rg_ic["mean"] <= r_ew_ic["mean"]+0.005:
                        print("    Ridge does NOT beat equal-weight 1/N meaningfully -> ship 1/N (simpler, robust).")
                    else:
                        print("    Ridge beats 1/N by %+.4f IC -> Ridge justified IF significant, confirm on more data."%(r_rg_ic["mean"]-r_ew_ic["mean"]))
            report["horizons"][h]={"ew_ic":r_ew_ic,"ridge_ic":r_rg_ic,"ew_auc":r_ew_au,"ridge_auc":r_rg_au,
                                   "n_features":len(kept),"features":kept,"joined_rows":len(rows)}
        print("\n  REMINDER: 6 months = one regime. Even a 0.03-0.06 IC here is NOT yet")
        print("  deployable — paper-trade / size nominal until you cross a drawdown or rate regime.")
    finally:
        conn.close()
    if args.out:
        path=args.out
        if os.path.isdir(path) or path.endswith("/"): path=os.path.join(path,"base_results.json")
        with open(path,"a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":report})+"\n")
        print("\n  [report appended to %s]"%path)

def _f(x): return ("%+.4f"%x) if x is not None else "NA"

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
