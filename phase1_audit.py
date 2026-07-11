#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — PHASE 1 AUDIT TOOLKIT  (RULE 1 fixes, against real schema)
================================================================================
Single file, five subcommands. READ-ONLY by default everywhere. The only
subcommand that can ever write is `borrow --create`, and even that requires an
explicit --apply flag; without it, it prints the DDL and does nothing.

Schema facts this script is built on (verified by discovery probe 2026-06-24):
  accuracy.db
    predictions(id, ticker, prediction_date, horizon, prob_up, signal,
                confidence, model_version, created_at, prob_raw, is_watchlist,
                tier, risk_mult, ..., prob_eff_uncapped, prob_up_global,
                prob_pct7, ..., prob_up_global_ranker)   [31,017 rows]
    outcomes(id, ticker, prediction_date, horizon, outcome_date,
             actual_return, actual_up, created_at)        [696,775 rows]
    accuracy_cache(id, ticker, horizon, window_days, accuracy, roc_auc,
                   brier_score, n_predictions, computed_at)
    alpha_fitness(n_obs, n_days, rank_ic, ic_t, sharpe, turnover, fitness,
                  alpha, horizon, is_market_wide, scored_date)   [2,413 rows]
    alpha_fitness_by_ticker(ticker, alpha, horizon, n_obs, rank_ic, ic_t,
                            scored_date)                          [445,079 rows]
  data/sentiment.db
    finbert_filings(ticker, accession, filing_date, filing_type, section,
                    sentiment_score, sentiment_label, confidence, ...)  [2,846]
  sentiment.db
    sentiment_scores(..., score, ..., source, is_corrupted)            [787]

VERIFIED FACTS the probe already established (so the script confirms, not guesses):
  - signal tracks prob_up, NOT prob_raw  (BUY avg prob_up 0.699 vs prob_raw 0.641)
  - prob_raw is only 74.6% populated
  - join key: outcomes <-> predictions on (ticker, prediction_date, horizon)

USAGE (from project root, env active):
  python phase1_audit.py prob
  python phase1_audit.py ic
  python phase1_audit.py deflate
  python phase1_audit.py finbert
  python phase1_audit.py borrow            # prints DDL only
  python phase1_audit.py borrow --apply    # actually creates borrow table (opt-in)
  python phase1_audit.py all               # runs prob, ic, deflate, finbert (read-only)

Options:
  --root PATH        project root (default: .)
  --buy-threshold X  fixed BUY cutoff to test (default 0.51)
  --out PATH         write a JSON/MD report alongside console output

Dependencies: stdlib only for SQLite. `pip install duckdb` only needed if you
later extend borrow to the DuckDB store. numpy/scipy used if present for exact
stats; falls back to pure-python approximations with a loud note if absent.
================================================================================
"""

import argparse, os, sqlite3, sys, math, json, datetime
from collections import defaultdict

# ---- optional deps, loud fallback ------------------------------------------
try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_ppf(p):
    # Acklam's inverse-normal approximation (good to ~1e-9); avoids scipy dep
    if p <= 0.0: return -float('inf')
    if p >= 1.0: return float('inf')
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    plow=0.02425; phigh=1-plow
    if p<plow:
        q=math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p<=phigh:
        q=p-0.5; r=q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q=math.sqrt(-2*math.log(1-p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

# ---- db helpers (read-only) -------------------------------------------------
LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)

def ro(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("DB not found: "+path)
    uri="file:"+os.path.abspath(path)+"?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True, timeout=10)

def rw(path):
    # only used by borrow --apply
    return sqlite3.connect(path, timeout=10)

def q(conn,sql,params=()):
    return conn.execute(sql,params).fetchall()

def has_table(conn,name):
    return bool(q(conn,"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(name,)))

def cols_of(conn,t):
    return [r[1] for r in q(conn,'PRAGMA table_info("'+t+'")')]

def require(cond, msg):
    """Fail loud (RULE 1b: no silent errors)."""
    if not cond:
        print("  [STOP] "+msg)
        return False
    return True

def find_db(root, name):
    # exact path first, then walk
    cand=os.path.join(root,name)
    if os.path.isfile(cand): return cand
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        if name in fn: return os.path.join(dp,name)
    return None

# ============================================================================ PROB
def cmd_prob(args):
    banner("PROB — is the BUY signal minted on the inflated probability?")
    accp=find_db(args.root,"accuracy.db")
    if not require(accp,"accuracy.db not found under "+args.root): return
    conn=ro(accp)
    report={}
    try:
        if not require(has_table(conn,"predictions"),"no predictions table"): return
        pc=cols_of(conn,"predictions")
        for need in ("prob_up","prob_raw","signal","horizon"):
            if not require(need in pc, "predictions missing column: "+need): return

        # coverage
        n=q(conn,"SELECT COUNT(*) FROM predictions")[0][0]
        nraw=q(conn,'SELECT COUNT(prob_raw) FROM predictions')[0][0]
        cov=100.0*nraw/n if n else 0
        print("  predictions rows: %d ; prob_raw populated: %d (%.1f%%)" % (n,nraw,cov))
        report["rows"]=n; report["prob_raw_coverage_pct"]=round(cov,1)
        if cov<99:
            print("  [WARN] prob_raw has a %.1f%% logging hole — recompute below covers only populated rows." % (100-cov))

        # BUY minting gap: among BUYs, how many would NOT clear threshold on prob_raw?
        sub("BUY-minting gap — three ways (fixed / tier / percentile)")
        thr=args.buy_threshold
        # restrict to BUYs that have prob_raw
        buys=q(conn,'SELECT prob_up, prob_raw, tier, prob_pct7 FROM predictions '
                    "WHERE signal='BUY' AND prob_raw IS NOT NULL")
        nb=len(buys)
        print("  BUYs with prob_raw present: %d" % nb)
        if nb:
            # (1) fixed threshold
            fail_fixed=sum(1 for r in buys if (r[1] is not None and r[1] < thr))
            print("  (1) FIXED  cutoff %.3f on prob_raw: %d/%d BUYs (%.1f%%) would NOT be BUY"
                  % (thr, fail_fixed, nb, 100.0*fail_fixed/nb))
            # (2) tier-based: if tier present, show BUY tiers and whether prob_raw keeps order
            tiers=defaultdict(lambda:[0,0])
            for pu,pr,ti,pct in buys:
                if ti is not None:
                    tiers[ti][0]+=1
                    if pr is not None and pr<thr: tiers[ti][1]+=1
            if tiers:
                print("  (2) TIER   breakdown [tier: n_buys, n_below_%.3f_on_raw]:" % thr)
                for ti in sorted(tiers, key=lambda x:str(x)):
                    print("        %-12s %d, %d" % (str(ti), tiers[ti][0], tiers[ti][1]))
            else:
                print("  (2) TIER   tier column null for BUYs — skipped")
            # (3) percentile: prob_pct7 is the percentile column; how many BUYs are low-percentile?
            pcts=[r[3] for r in buys if r[3] is not None]
            if pcts:
                lowp=sum(1 for x in pcts if x<0.5)
                print("  (3) PCTILE prob_pct7 present for %d BUYs; %d (%.1f%%) sit below 0.5 percentile"
                      % (len(pcts), lowp, 100.0*lowp/len(pcts)))
            else:
                print("  (3) PCTILE prob_pct7 null for BUYs — skipped")
            report["buys_with_raw"]=nb
            report["buys_fail_fixed_threshold"]=fail_fixed
            report["fixed_threshold"]=thr

        # confirm signal tracks prob_up not prob_raw (gap in averages)
        sub("Average prob by signal (confirms which column the gate uses)")
        for c in ("prob_up","prob_raw"):
            rows=q(conn,'SELECT signal, COUNT(*), ROUND(AVG("'+c+'"),4), ROUND(MIN("'+c+'"),4) '
                        'FROM predictions WHERE "'+c+'" IS NOT NULL GROUP BY signal ORDER BY 2 DESC')
            print("  column=%s  [signal, n, avg, min]:" % c)
            for r in rows: print("     ", r)

        # what does accuracy_cache score against? (structure tells us)
        sub("accuracy_cache — what your LIVE numbers are computed on")
        if has_table(conn,"accuracy_cache"):
            acc_cols=cols_of(conn,"accuracy_cache")
            print("  accuracy_cache columns:", acc_cols)
            print("  [NOTE] accuracy_cache has no prob-source column; given `signal` tracks prob_up,")
            print("         your stored roc_auc/brier are almost certainly on prob_up (inflated).")
            print("         The `ic` subcommand recomputes the honest version on prob_raw.")
        print("\n  VERDICT: if (1) shows a non-trivial %% of BUYs failing on prob_raw, the gate is")
        print("  firing on inflated probability. That is Finding A. Fix the writer + re-gate on raw.")
    finally:
        conn.close()
    _maybe_write(args, "prob", report)

# ============================================================================ IC
def _auc(scores, labels):
    """ROC-AUC via rank statistic (Mann-Whitney). scores,labels aligned."""
    pairs=sorted(zip(scores,labels))
    # rank with average ties
    ranks=[0.0]*len(pairs); i=0; r=1
    while i<len(pairs):
        j=i
        while j+1<len(pairs) and pairs[j+1][0]==pairs[i][0]: j+=1
        avg=(r+(r+(j-i)))/2.0
        for k in range(i,j+1): ranks[k]=avg
        r+=(j-i+1); i=j+1
    pos=sum(1 for _,l in pairs if l==1)
    neg=len(pairs)-pos
    if pos==0 or neg==0: return None
    sum_pos=sum(rk for rk,(_,l) in zip(ranks,pairs) if l==1)
    return (sum_pos - pos*(pos+1)/2.0)/(pos*neg)

def _brier(probs, labels):
    return sum((p-l)**2 for p,l in zip(probs,labels))/len(probs)

def _spearman(x,y):
    n=len(x)
    if n<3: return None
    def rank(v):
        order=sorted(range(n), key=lambda i:v[i])
        rr=[0.0]*n; i=0
        while i<n:
            j=i
            while j+1<n and v[order[j+1]]==v[order[i]]: j+=1
            avg=(i+j)/2.0+1
            for k in range(i,j+1): rr[order[k]]=avg
            i=j+1
        return rr
    rx,ry=rank(x),rank(y)
    mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den=math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else None

def cmd_ic(args):
    banner("IC — honest accuracy / calibration on prob_raw vs prob_up (production only)")
    accp=find_db(args.root,"accuracy.db")
    if not require(accp,"accuracy.db not found"): return
    conn=ro(accp)
    report={"by_horizon":{}}
    try:
        for need_t in ("predictions","outcomes"):
            if not require(has_table(conn,need_t),"missing table: "+need_t): return
        pc=cols_of(conn,"predictions"); oc=cols_of(conn,"outcomes")
        for need in ("ticker","prediction_date","horizon","prob_up","prob_raw","signal"):
            if not require(need in pc,"predictions missing "+need): return
        for need in ("ticker","prediction_date","horizon","actual_up","actual_return"):
            if not require(need in oc,"outcomes missing "+need): return

        # join predictions to outcomes on (ticker, prediction_date, horizon)
        # NOTE: production outcomes table is huge (696k). We pull joined rows per horizon.
        for h in (1,3,5):
            rows=q(conn,
                "SELECT p.prob_up, p.prob_raw, o.actual_up, o.actual_return, p.signal "
                "FROM predictions p JOIN outcomes o "
                "ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date "
                "AND p.horizon=o.horizon "
                "WHERE p.horizon=? AND o.actual_up IS NOT NULL", (h,))
            if not rows:
                print("  h=%d: no joined rows (check join keys / outcome coverage)" % h)
                continue
            # split into has-raw and all
            up=[r[0] for r in rows if r[0] is not None]
            up_lab=[r[2] for r in rows if r[0] is not None]
            raw=[r[1] for r in rows if r[1] is not None]
            raw_lab=[r[2] for r in rows if r[1] is not None]
            up_ret=[r[3] for r in rows if r[0] is not None]
            raw_ret=[r[3] for r in rows if r[1] is not None]

            sub("h=%d  (joined rows=%d ; with prob_raw=%d)" % (h,len(rows),len(raw)))
            def block(name, p, lab, ret):
                if len(p)<10:
                    print("    %-9s n=%d too few" % (name,len(p))); return {}
                # accuracy at 0.5
                acc=sum(1 for pi,li in zip(p,lab) if (pi>=0.5)==(li==1))/len(p)
                auc=_auc(p,lab)
                brier=_brier(p,lab)
                ic=_spearman(p,ret) if ret and len(ret)==len(p) else None
                print("    %-9s n=%d  acc@0.5=%.4f  AUC=%s  Brier=%.4f  rankIC(prob,ret)=%s"
                      % (name,len(p),acc,
                         ("%.4f"%auc) if auc is not None else "NA",
                         brier,
                         ("%.4f"%ic) if ic is not None else "NA"))
                return {"n":len(p),"acc":round(acc,4),
                        "auc":round(auc,4) if auc is not None else None,
                        "brier":round(brier,4),
                        "rank_ic":round(ic,4) if ic is not None else None}
            r_up=block("prob_up", up, up_lab, up_ret)
            r_raw=block("prob_raw", raw, raw_lab, raw_ret)
            # delta
            if r_up and r_raw and r_up.get("auc") and r_raw.get("auc"):
                print("    -> AUC delta (raw - up): %+.4f   Brier delta (raw - up): %+.4f"
                      % (r_raw["auc"]-r_up["auc"], r_raw["brier"]-r_up["brier"]))
                print("       (if raw AUC < up AUC, the overlay was *adding* apparent skill the model lacks)")
            report["by_horizon"][h]={"prob_up":r_up,"prob_raw":r_raw}
        print("\n  This is the honest baseline. Compare to your stored accuracy_cache numbers;")
        print("  any gap is the overlay inflation (Finding A/B). h=1 should remain your strongest.")
        if not HAVE_NUMPY:
            print("\n  [note] numpy not present — used pure-python stats (exact AUC/Brier/Spearman, slightly slower).")
    finally:
        conn.close()
    _maybe_write(args,"ic",report)

# ============================================================================ DEFLATE
def _cluster_by_family(alphas):
    """Group alpha names by their base feature (text before first '__').
    Collapses pc_ratio_snap__ts_mean__w20 / __cs_rank / ... into one family."""
    fam=defaultdict(list)
    for a in alphas:
        base=a.split("__")[0] if "__" in a else a
        fam[base].append(a)
    return fam

def _benjamini_hochberg(pvals, alpha=0.05):
    """Return boolean keep-mask under BH-FDR. pvals: list. Order preserved."""
    m=len(pvals)
    idx=sorted(range(m), key=lambda i:pvals[i])
    keep=[False]*m
    thresh_rank=-1
    for rank,i in enumerate(idx, start=1):
        if pvals[i] <= (rank/m)*alpha:
            thresh_rank=rank
    if thresh_rank>0:
        for rank,i in enumerate(idx, start=1):
            if rank<=thresh_rank: keep[i]=True
    return keep

def _p_from_t(t, df=250):
    """two-sided p-value from t-stat using normal approx (df large)."""
    z=abs(t)
    return 2.0*(1.0-_norm_cdf(z))

def _deflated_sharpe_haircut(best_t, n_trials):
    """Closed-form: prob that the best of n independent trials exceeds observed t
    under the null. Reports the implied 'expected max t' and whether best_t beats it.
    Based on Bailey & Lopez de Prado expected-max-Sharpe logic (normal approx)."""
    if n_trials<2: return None
    e_max = (1-0.5772156649)*_norm_ppf(1-1.0/n_trials) + 0.5772156649*_norm_ppf(1-1.0/(n_trials*math.e))
    # e_max is the expected maximum of n_trials standard normals (~ expected max t under null)
    return {"expected_max_t_under_null":round(e_max,3),
            "observed_best_t":round(best_t,3),
            "beats_null_max": best_t>e_max}

def cmd_deflate(args):
    banner("DEFLATE — dedupe -> BH-FDR -> Deflated-Sharpe on the alpha search")
    accp=find_db(args.root,"accuracy.db")
    if not require(accp,"accuracy.db not found"): return
    conn=ro(accp)
    report={}
    try:
        for tbl,label in (("alpha_fitness","market-wide"),
                          ("alpha_fitness_by_ticker","by-ticker")):
            if not has_table(conn,tbl):
                print("  %s: MISSING — skipped" % tbl); continue
            c=cols_of(conn,tbl)
            if not require("alpha" in c and "ic_t" in c, tbl+" missing alpha/ic_t"): continue
            sub("%s  (table=%s)" % (label,tbl))
            rows=q(conn,'SELECT alpha, ic_t, rank_ic FROM "'+tbl+'" WHERE ic_t IS NOT NULL')
            n_rows=len(rows)
            alphas=[r[0] for r in rows]
            tvals=[r[1] for r in rows]
            distinct=len(set(alphas))
            print("  rows=%d  distinct alphas=%d" % (n_rows,distinct))

            # naive counts
            p2=sum(1 for t in tvals if abs(t)>2); p3=sum(1 for t in tvals if abs(t)>3)
            print("  naive: |t|>2 -> %d ; |t|>3 -> %d" % (p2,p3))
            # expected false positives at these thresholds across distinct trials
            exp_fp2=distinct*2*(1-_norm_cdf(2.0))
            exp_fp3=distinct*2*(1-_norm_cdf(3.0))
            print("  expected false positives by chance: |t|>2 ~%.0f ; |t|>3 ~%.0f"
                  % (exp_fp2,exp_fp3))

            # dedupe by family, keep max |t| per family
            fam=_cluster_by_family(set(alphas))
            best_per_family={}
            for a,t in zip(alphas,tvals):
                base=a.split("__")[0] if "__" in a else a
                if base not in best_per_family or abs(t)>abs(best_per_family[base][1]):
                    best_per_family[base]=(a,t)
            n_families=len(fam)
            print("  deduped families: %d  (collapsed from %d distinct alphas)"
                  % (n_families,distinct))
            # show how concentrated the top is
            fam_sizes=sorted(((len(v),k) for k,v in fam.items()), reverse=True)[:5]
            print("  largest families [size, base]:", fam_sizes)

            # BH-FDR on the deduped family-best t-stats
            fam_t=[abs(v[1]) for v in best_per_family.values()]
            fam_names=list(best_per_family.keys())
            pvals=[_p_from_t(t) for t in fam_t]
            keep=_benjamini_hochberg(pvals, alpha=0.05)
            n_keep=sum(keep)
            m=len(fam_t)
            bonf_p=0.05/m if m else 1.0
            bonf_t=abs(_norm_ppf(1-bonf_p/2)) if m else float('inf')
            surv_t=[fam_t[i] for i in range(m) if keep[i]]
            min_surv_t=min(surv_t) if surv_t else None
            print("  BH-FDR(5%%) on %d deduped families -> %d survive" % (m,n_keep))
            print("     (Bonferroni t-cutoff for context: |t|>%.2f ; smallest surviving |t|: %s)"
                  % (bonf_t, ("%.2f"%min_surv_t) if min_surv_t else "n/a"))
            if n_keep==0:
                print("     [NOTE] 0 survivors is the CORRECT strict result when nothing clears the")
                print("     multiple-testing bar. Naive |t|>3 = %d was inflated by multiplicity." % p3)
                watch=sorted([(fam_names[i],best_per_family[fam_names[i]][0],best_per_family[fam_names[i]][1])
                              for i in range(m)], key=lambda x:-abs(x[2]))[:10]
                print("     Top families to RE-TEST with a proper IC time series (WATCH, not trade):")
                for nm,full,t in watch:
                    print("       WATCH %-26s (%s)  t=%.2f" % (nm,full,t))
            survivors=sorted(
                [(fam_names[i],best_per_family[fam_names[i]][0],best_per_family[fam_names[i]][1])
                 for i in range(len(fam_names)) if keep[i]],
                key=lambda x:-abs(x[2]))
            for nm,full,t in survivors[:15]:
                print("     KEEP  %-26s (best expr: %s)  t=%.2f" % (nm,full,t))
            if len(survivors)>15: print("     ... +%d more" % (len(survivors)-15))

            # Deflated Sharpe style: best family t vs expected max under null (deduped N)
            if fam_t:
                best_t=max(fam_t)
                dsr=_deflated_sharpe_haircut(best_t, n_families)
                if dsr:
                    print("  Deflated check (N=%d families): expected max |t| under null=%.2f ; "
                          "observed best=%.2f ; beats null-max=%s"
                          % (n_families, dsr["expected_max_t_under_null"],
                             dsr["observed_best_t"], dsr["beats_null_max"]))
            report[tbl]={"rows":n_rows,"distinct":distinct,"families":n_families,
                         "naive_t2":p2,"naive_t3":p3,
                         "bh_survivors":n_keep,
                         "survivor_names":[s[0] for s in survivors[:50]]}
        print("\n  Interpretation: the gap between 'distinct alphas' and 'deduped families' is your")
        print("  duplicate inflation. The gap between naive |t|>3 and BH survivors is your")
        print("  multiple-testing inflation. Trust ONLY the BH survivors, and only as candidates")
        print("  for proper time-series Deflated Sharpe once you persist per-period IC.")
        print("\n  [CAVEAT] alpha_fitness has single-day point estimates, not an IC time series.")
        print("  This is the closed-form approximation. Real DSR needs IC-over-time — persist it.")
    finally:
        conn.close()
    _maybe_write(args,"deflate",report)

# ============================================================================ FINBERT
def cmd_finbert(args):
    banner("FINBERT — sign-fix continuity across the training window")
    sp=find_db(args.root, os.path.join("data","sentiment.db")) or find_db(args.root,"sentiment.db")
    # prefer data/sentiment.db (finbert_filings); also check top-level sentiment.db
    dpath=find_db(args.root,"sentiment.db")
    report={}
    # finbert_filings lives in data/sentiment.db per probe
    fb_db=None
    for cand in (os.path.join(args.root,"data","sentiment.db"),
                 find_db(args.root,"sentiment.db")):
        if cand and os.path.isfile(cand):
            try:
                c=ro(cand)
                if has_table(c,"finbert_filings"):
                    fb_db=cand; c.close(); break
                c.close()
            except Exception: pass
    if not require(fb_db,"could not locate a sentiment.db containing finbert_filings"):
        return
    conn=ro(fb_db)
    try:
        print("  using:", fb_db)
        c=cols_of(conn,"finbert_filings")
        if not require("sentiment_score" in c and "filing_date" in c,
                       "finbert_filings missing sentiment_score/filing_date"): return
        n=q(conn,"SELECT COUNT(*) FROM finbert_filings")[0][0]
        dr=q(conn,"SELECT MIN(filing_date),MAX(filing_date) FROM finbert_filings")[0]
        print("  rows=%d ; filing_date range %s .. %s" % (n,dr[0],dr[1]))
        # monthly mean + sign distribution to spot a discontinuity
        sub("Monthly mean sentiment_score + fraction positive (look for a sign flip)")
        rows=q(conn,
            "SELECT substr(filing_date,1,7) AS ym, COUNT(*), "
            "ROUND(AVG(sentiment_score),4), "
            "ROUND(AVG(CASE WHEN sentiment_score>0 THEN 1.0 ELSE 0.0 END),3) "
            "FROM finbert_filings WHERE filing_date IS NOT NULL "
            "GROUP BY ym ORDER BY ym")
        prev_sign=None; flips=[]
        for ym,cnt,avg,fpos in rows:
            flag=""
            if avg is not None:
                s=1 if avg>0 else (-1 if avg<0 else 0)
                if prev_sign is not None and s!=0 and prev_sign!=0 and s!=prev_sign:
                    flag="  <<< SIGN FLIP"; flips.append(ym)
                prev_sign=s if s!=0 else prev_sign
            print("    %s  n=%4d  mean=%+.4f  frac_pos=%s%s"
                  % (ym,cnt,avg if avg is not None else 0,
                     ("%.3f"%fpos) if fpos is not None else "NA", flag))
        if flips:
            print("\n  [WARN] sign flips at: %s" % ", ".join(flips))
            print("  Investigate whether these align with the 56-day FinBERT fix boundary.")
            print("  If pre-fix rows were NOT recomputed, the model trains on a discontinuity.")
        else:
            print("\n  No month-over-month mean-sign flips detected (good, but verify the fix")
            print("  boundary date directly).")
        # cross-check is_corrupted in top-level sentiment.db if present
        if dpath and os.path.isfile(dpath):
            c2=ro(dpath)
            try:
                if has_table(c2,"sentiment_scores") and "is_corrupted" in cols_of(c2,"sentiment_scores"):
                    sub("Cross-check: sentiment_scores.is_corrupted (sentiment.db)")
                    rows=q(c2,"SELECT is_corrupted, COUNT(*) FROM sentiment_scores GROUP BY is_corrupted")
                    print("    is_corrupted breakdown:", rows)
                    print("    [NOTE] any rows flagged corrupted should be excluded from training.")
            finally:
                c2.close()
        report["rows"]=n; report["range"]=[dr[0],dr[1]]; report["sign_flips"]=flips
    finally:
        conn.close()
    _maybe_write(args,"finbert",report)

# ============================================================================ BORROW
BORROW_DDL = """
-- Borrow-fee scaffold (Finding F). No data source exists yet in your 22 DBs.
-- This table is the contract every L/S backtest will join against to net costs.
CREATE TABLE IF NOT EXISTS borrow_fees (
    ticker        TEXT NOT NULL,
    asof_date     TEXT NOT NULL,          -- point-in-time date (ET)
    borrow_fee_bps REAL,                  -- annualized borrow fee, basis points
    utilization   REAL,                   -- 0..1 if available
    shares_avail  INTEGER,                -- locate availability if known
    is_htb        INTEGER,                -- 1 if hard-to-borrow
    source        TEXT,                   -- vendor / derivation
    fetched_at    TEXT,
    PRIMARY KEY (ticker, asof_date)
);
-- Index for the typical join: predictions/outcomes (ticker, prediction_date)
CREATE INDEX IF NOT EXISTS idx_borrow_ticker_date ON borrow_fees(ticker, asof_date);
"""

def cmd_borrow(args):
    banner("BORROW — scaffold the missing borrow-fee table (Finding F)")
    print("  No borrow-fee table exists in any of your 22 databases (probe-confirmed).")
    print("  Any short/options 'edge' is UNVERIFIABLE until borrow cost is netted")
    print("  (Muravyev-Pearson-Pollet 2025: options anomalies ~ -0.01%/mo net of borrow).")
    sub("Proposed DDL")
    print(BORROW_DDL)
    sub("Where to put it")
    print("  Recommend a NEW db `borrow.db` (keeps it isolated, easy to version) or add to")
    print("  accuracy.db. The join contract for backtests:")
    print("     SELECT ... FROM outcomes o")
    print("     LEFT JOIN borrow_fees b ON b.ticker=o.ticker AND b.asof_date=o.prediction_date")
    print("     -- net short P&L by (borrow_fee_bps/10000) * holding_period_fraction")
    sub("Data sources to populate it (you must license/derive — none are free+clean)")
    print("  - Interactive Brokers locate/borrow rates (if you have an IBKR acct)")
    print("  - FINRA Reg SHO / short-sale volume (proxy only — NOT a borrow fee)")
    print("  - Your short_interest_cache (accuracy.db) as a crude HTB proxy")
    print("  - Commercial: S3 Partners / Ortex / IHS Markit (paid)")
    if args.apply:
        target=os.path.join(args.root,"borrow.db")
        print("\n  --apply set: creating %s ..." % target)
        conn=rw(target)
        try:
            conn.executescript(BORROW_DDL); conn.commit()
            print("  created borrow.db with empty borrow_fees table (0 rows).")
            print("  [RULE 1] table is empty by design — fail loud when a backtest finds no")
            print("  borrow row rather than silently assuming zero cost.")
        finally:
            conn.close()
    else:
        print("\n  (dry-run) Nothing written. Re-run with --apply to create borrow.db.")
    _maybe_write(args,"borrow",{"applied":bool(args.apply)})

# ============================================================================ infra
def _maybe_write(args, name, report):
    if not args.out: return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    stamp=datetime.datetime.now().isoformat(timespec="seconds")
    payload={"subcommand":name,"timestamp":stamp,"report":report}
    path=args.out
    if os.path.isdir(path) or path.endswith("/"):
        path=os.path.join(path, "phase1_%s.json" % name)
    with open(path,"a") as f:
        f.write(json.dumps(payload)+"\n")
    print("\n  [report appended to %s]" % path)

def main():
    ap=argparse.ArgumentParser(description="Phase 1 audit toolkit (read-only by default)")
    ap.add_argument("cmd", choices=["prob","ic","deflate","finbert","borrow","all"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--buy-threshold", type=float, default=0.51)
    ap.add_argument("--apply", action="store_true", help="borrow: actually create borrow.db")
    ap.add_argument("--out", default=None, help="append JSON report to this path/dir")
    args=ap.parse_args()
    args.root=os.path.expanduser(args.root)

    banner("ML QUANT FUND — PHASE 1 AUDIT TOOLKIT")
    print("Read-only by default. Only `borrow --apply` writes (to a new borrow.db).")
    print("Root:", os.path.abspath(args.root), "| Python", sys.version.split()[0],
          "| numpy:", HAVE_NUMPY)

    if args.cmd=="all":
        for fn in (cmd_prob,cmd_ic,cmd_deflate,cmd_finbert):
            try: fn(args)
            except Exception as e:
                print("  [SUBCOMMAND FAILED] %s: %s" % (fn.__name__, e))
        print("\n(`all` skips borrow — run it explicitly with --apply when ready.)")
        return
    {"prob":cmd_prob,"ic":cmd_ic,"deflate":cmd_deflate,
     "finbert":cmd_finbert,"borrow":cmd_borrow}[args.cmd](args)

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
    except Exception:
        import traceback
        print("\n[UNEXPECTED ERROR] paste this back:")
        traceback.print_exc()
