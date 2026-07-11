#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — PHASE 2 DIAGNOSTIC TOOLKIT
================================================================================
Answers the questions Phase 1 surfaced. READ-ONLY everywhere (no --apply needed;
this script never writes).

Subcommands:
  decompose  THE BIG ONE. Separate model edge from overlay edge.
             Buckets prob_raw vs prob_up into deciles, measures forward hit-rate
             and rank-IC per bucket per horizon, and attributes the prob_up-minus-
             prob_raw gap to the multiplier columns. Tells you whether your h=1
             "edge" is the model or the overlay.
  pct7       Why do 100% of BUYs sit below the 0.5 percentile? Inspects prob_pct7:
             its distribution, what it correlates with, and whether it's
             mis-scaled / inverted / computed on a different population.
  rawgap     Which model_version / date ranges are missing prob_raw (the 25.3%
             hole)? Tells the writer-patch exactly what to backfill.
  corrupt    Characterize the 257 is_corrupted sentiment rows: dates, sources,
             what distinguishes them, and whether corruption looks ongoing.
  all        Runs decompose, pct7, rawgap, corrupt.

USAGE (project root, env active):
  python phase2_diagnose.py decompose --root .
  python phase2_diagnose.py all --root .
  add --out report.jsonl to append machine-readable results.

Stats are exact pure-python (AUC via Mann-Whitney, Spearman rank-IC); numpy used
only if present for speed. No scipy/sklearn dependency.
================================================================================
"""
import argparse, os, sqlite3, sys, math, json, datetime
from collections import defaultdict

try:
    import numpy as np
    HAVE_NUMPY=True
except Exception:
    HAVE_NUMPY=False

LINE="="*78
def banner(t): print("\n"+LINE+"\n"+t+"\n"+LINE)
def sub(t): print("\n"+"-"*78+"\n"+t+"\n"+"-"*78)

def ro(path):
    if not os.path.isfile(path): raise FileNotFoundError("DB not found: "+path)
    return sqlite3.connect("file:"+os.path.abspath(path)+"?mode=ro&immutable=1", uri=True, timeout=15)

def q(conn,sql,params=()): return conn.execute(sql,params).fetchall()
def has_table(conn,n): return bool(q(conn,"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(n,)))
def cols_of(conn,t): return [r[1] for r in q(conn,'PRAGMA table_info("'+t+'")')]
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

# ---- exact stats ----
def auc(scores,labels):
    pairs=sorted(zip(scores,labels)); nplus=sum(1 for _,l in pairs if l==1); nmin=len(pairs)-nplus
    if nplus==0 or nmin==0: return None
    ranks=[0.0]*len(pairs); i=0; r=1
    while i<len(pairs):
        j=i
        while j+1<len(pairs) and pairs[j+1][0]==pairs[i][0]: j+=1
        avg=(r+(r+(j-i)))/2.0
        for k in range(i,j+1): ranks[k]=avg
        r+=(j-i+1); i=j+1
    sp=sum(rk for rk,(_,l) in zip(ranks,pairs) if l==1)
    return (sp-nplus*(nplus+1)/2.0)/(nplus*nmin)

def spearman(x,y):
    n=len(x)
    if n<3: return None
    def rank(v):
        order=sorted(range(n),key=lambda i:v[i]); rr=[0.0]*n; i=0
        while i<n:
            j=i
            while j+1<n and v[order[j+1]]==v[order[i]]: j+=1
            avg=(i+j)/2.0+1
            for k in range(i,j+1): rr[order[k]]=avg
            i=j+1
        return rr
    rx,ry=rank(x),rank(y); mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den=math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else None

def pearson(x,y):
    n=len(x)
    if n<3: return None
    mx=sum(x)/n; my=sum(y)/n
    num=sum((a-mx)*(b-my) for a,b in zip(x,y))
    den=math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return num/den if den else None

def deciles(vals):
    """Return decile edges (10 buckets) from sorted vals."""
    s=sorted(vals); n=len(s)
    if n<10: return None
    return [s[int(n*k/10)] for k in range(1,10)]

def bucket_of(v,edges):
    for i,e in enumerate(edges):
        if v<=e: return i
    return len(edges)

# ============================================================ DECOMPOSE
def cmd_decompose(args):
    banner("DECOMPOSE — is your edge the MODEL or the OVERLAY?")
    accp=find_db(args.root,"accuracy.db")
    if not require(accp,"accuracy.db not found"): return
    conn=ro(accp); report={"by_horizon":{}}
    try:
        for t in ("predictions","outcomes"):
            if not require(has_table(conn,t),"missing "+t): return
        pc=cols_of(conn,"predictions")
        mults=[c for c in ("risk_mult","sent_mult","regime_mult","options_mult",
                           "squeeze_mult","intraday_mult","fg_mult") if c in pc]
        print("  multiplier columns present:", mults if mults else "NONE FOUND")
        # coverage of each multiplier
        n=q(conn,"SELECT COUNT(*) FROM predictions")[0][0]
        for m in mults:
            nn=q(conn,'SELECT COUNT("'+m+'") FROM predictions')[0][0]
            avg=q(conn,'SELECT ROUND(AVG("'+m+'"),4) FROM predictions WHERE "'+m+'" IS NOT NULL')[0][0]
            print("     %-16s coverage %5.1f%%  avg=%s" % (m,100.0*nn/n,avg))

        for h in (1,3,5):
            rows=q(conn,
                "SELECT p.prob_raw, p.prob_up, o.actual_up, o.actual_return "
                "FROM predictions p JOIN outcomes o "
                "ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon "
                "WHERE p.horizon=? AND o.actual_up IS NOT NULL AND p.prob_raw IS NOT NULL "
                "AND p.prob_up IS NOT NULL", (h,))
            if len(rows)<100:
                print("  h=%d: only %d joined rows with both probs — skip" % (h,len(rows))); continue
            raw=[r[0] for r in rows]; up=[r[1] for r in rows]
            au=[r[2] for r in rows]; ret=[r[3] for r in rows]
            sub("h=%d  (rows with prob_raw AND prob_up AND outcome = %d)" % (h,len(rows)))

            # headline metrics both ways
            def metrics(p):
                a=auc(p,au); ic=spearman(p,ret); acc=sum(1 for pi,li in zip(p,au) if (pi>=0.5)==(li==1))/len(p)
                return acc,a,ic
            racc,rauc,ric=metrics(raw); uacc,uauc,uic=metrics(up)
            print("    MODEL (prob_raw):   acc@0.5=%.4f  AUC=%s  rankIC=%s"
                  % (racc, "%.4f"%rauc if rauc else "NA", "%.4f"%ric if ric else "NA"))
            print("    OVERLAY (prob_up):  acc@0.5=%.4f  AUC=%s  rankIC=%s"
                  % (uacc, "%.4f"%uauc if uauc else "NA", "%.4f"%uic if uic else "NA"))
            gain_auc=(uauc-rauc) if (uauc and rauc) else None
            gain_ic=(uic-ric) if (uic and ric) else None
            print("    OVERLAY GAIN: dAUC=%s  dRankIC=%s"
                  % ("%+.4f"%gain_auc if gain_auc is not None else "NA",
                     "%+.4f"%gain_ic if gain_ic is not None else "NA"))
            if gain_auc is not None:
                if gain_auc>0.005:
                    print("    -> overlay ADDS skill. But is it real or curve-fit? (needs walk-forward)")
                elif gain_auc<-0.005:
                    print("    -> overlay DESTROYS skill. The multipliers are hurting you.")
                else:
                    print("    -> overlay is ~neutral on ranking; it mainly shifts the threshold.")

            # decile lift table on prob_raw (does the MODEL monotonically sort returns?)
            edges=deciles(raw)
            if edges:
                buckets=defaultdict(list)
                for pr,rr_ in zip(raw,ret): buckets[bucket_of(pr,edges)].append(rr_)
                bup=defaultdict(list)
                for pu,rr_ in zip(up,ret): bup[bucket_of(pu,deciles(up))].append(rr_)
                print("    MODEL decile lift (prob_raw decile -> mean fwd return):")
                prev=None; mono_raw=True
                for d in range(10):
                    if buckets[d]:
                        mr=sum(buckets[d])/len(buckets[d])
                        if prev is not None and mr<prev-1e-9: mono_raw=False
                        prev=mr
                        print("       D%d n=%4d  mean_ret=%+.4f" % (d,len(buckets[d]),mr))
                print("       monotonic (model sorts returns)? %s" % mono_raw)
                # top-minus-bottom spread, both
                def tmb(b):
                    if b[0] and b[9]: return sum(b[9])/len(b[9]) - sum(b[0])/len(b[0])
                    return None
                tr=tmb(buckets); tu=tmb(bup)
                print("    top-decile minus bottom-decile mean return:")
                print("       MODEL (prob_raw): %s   OVERLAY (prob_up): %s"
                      % ("%+.4f"%tr if tr is not None else "NA",
                         "%+.4f"%tu if tu is not None else "NA"))

            # how much does each multiplier explain the gap (prob_up/prob_raw ratio)?
            if mults:
                grows=q(conn,
                    "SELECT "+",".join('p."'+m+'"' for m in mults)+", p.prob_raw, p.prob_up "
                    "FROM predictions p JOIN outcomes o "
                    "ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon "
                    "WHERE p.horizon=? AND o.actual_up IS NOT NULL AND p.prob_raw IS NOT NULL "
                    "AND p.prob_up IS NOT NULL AND p.prob_raw>0", (h,))
                if grows:
                    logratio=[math.log(r[-1]/r[-2]) for r in grows if r[-2] and r[-1] and r[-2]>0 and r[-1]>0]
                    print("    multiplier correlation with log(prob_up/prob_raw) gap:")
                    for mi,m in enumerate(mults):
                        mv=[(math.log(r[mi]) if r[mi] and r[mi]>0 else 0.0) for r in grows
                            if r[-2] and r[-1] and r[-2]>0 and r[-1]>0]
                        if len(mv)==len(logratio) and len(mv)>10:
                            c=pearson(mv,logratio)
                            print("       %-16s corr=%s" % (m, "%+.3f"%c if c is not None else "NA"))
            report["by_horizon"][h]={"model_auc":rauc,"overlay_auc":uauc,
                                     "model_ic":ric,"overlay_ic":uic,
                                     "overlay_gain_auc":gain_auc}
        print("\n  READ THIS: if MODEL AUC ~0.50 and OVERLAY GAIN is large+positive, your edge is")
        print("  the overlay, not the model — and overlays are easy to overfit. The next step is a")
        print("  WALK-FORWARD test: does the overlay gain survive out-of-sample, or is it fit to the past?")
    finally:
        conn.close()
    _w(args,"decompose",report)

# ============================================================ PCT7
def cmd_pct7(args):
    banner("PCT7 — why do 100% of BUYs sit below the 0.5 percentile?")
    accp=find_db(args.root,"accuracy.db")
    if not require(accp,"accuracy.db not found"): return
    conn=ro(accp); report={}
    try:
        pc=cols_of(conn,"predictions")
        if not require("prob_pct7" in pc,"no prob_pct7 column"): return
        n=q(conn,"SELECT COUNT(*) FROM predictions")[0][0]
        nn=q(conn,"SELECT COUNT(prob_pct7) FROM predictions")[0][0]
        mn,med,mx=q(conn,"SELECT MIN(prob_pct7),"
                    "(SELECT prob_pct7 FROM predictions WHERE prob_pct7 IS NOT NULL ORDER BY prob_pct7 LIMIT 1 OFFSET "
                    +str(nn//2)+"), MAX(prob_pct7) FROM predictions")[0]
        print("  coverage %.1f%% (%d/%d) ; min=%s median=%s max=%s" % (100.0*nn/n,nn,n,mn,med,mx))
        # distribution histogram
        sub("Distribution of prob_pct7 (deciles of its own values)")
        vals=[r[0] for r in q(conn,"SELECT prob_pct7 FROM predictions WHERE prob_pct7 IS NOT NULL")]
        if vals:
            vals_sorted=sorted(vals)
            for k in range(0,11,1):
                idx=min(len(vals_sorted)-1,int(len(vals_sorted)*k/10))
                print("       %3d%%-ile value = %.4f" % (k*10, vals_sorted[idx]))
            frac_below_half=sum(1 for v in vals if v<0.5)/len(vals)
            print("    fraction of ALL prob_pct7 below 0.5: %.3f" % frac_below_half)
            print("    [if this is ~1.0, prob_pct7 is NOT a 0..1 percentile — it's mis-scaled or")
            print("     it's a percentile of something most rows score low on]")
        # relationship to prob_up / prob_raw
        sub("Does prob_pct7 track prob_up or prob_raw? (correlation)")
        rows=q(conn,"SELECT prob_pct7, prob_up, prob_raw FROM predictions "
                    "WHERE prob_pct7 IS NOT NULL AND prob_up IS NOT NULL")
        if len(rows)>10:
            p7=[r[0] for r in rows]; pu=[r[1] for r in rows]
            c_up=pearson(p7,pu)
            print("    corr(prob_pct7, prob_up) = %s" % ("%+.3f"%c_up if c_up else "NA"))
            rows2=[r for r in rows if r[2] is not None]
            if len(rows2)>10:
                c_raw=pearson([r[0] for r in rows2],[r[2] for r in rows2])
                print("    corr(prob_pct7, prob_raw) = %s" % ("%+.3f"%c_raw if c_raw else "NA"))
        # by signal
        sub("prob_pct7 by signal")
        for r in q(conn,"SELECT signal, COUNT(prob_pct7), ROUND(AVG(prob_pct7),4), "
                        "ROUND(MIN(prob_pct7),4), ROUND(MAX(prob_pct7),4) FROM predictions "
                        "WHERE prob_pct7 IS NOT NULL GROUP BY signal"):
            print("      ", r)
        print("\n  DIAGNOSIS: prob_pct7 with BUYs all <0.5 means it is NOT measuring 'this BUY is")
        print("  high-percentile'. Likely it's a 7-day percentile of raw prob across a window where")
        print("  most days are low, or it's inverted. Either way it should not gate BUYs as-is.")
    finally:
        conn.close()
    _w(args,"pct7",report)

# ============================================================ RAWGAP
def cmd_rawgap(args):
    banner("RAWGAP — which rows are missing prob_raw (the 25.3% hole)?")
    accp=find_db(args.root,"accuracy.db")
    if not require(accp,"accuracy.db not found"): return
    conn=ro(accp); report={}
    try:
        pc=cols_of(conn,"predictions")
        if not require("prob_raw" in pc,"no prob_raw column"): return
        n=q(conn,"SELECT COUNT(*) FROM predictions")[0][0]
        miss=q(conn,"SELECT COUNT(*) FROM predictions WHERE prob_raw IS NULL")[0][0]
        print("  total=%d  missing prob_raw=%d (%.1f%%)" % (n,miss,100.0*miss/n))
        # by model_version
        if "model_version" in pc:
            sub("Missing prob_raw by model_version")
            for r in q(conn,"SELECT model_version, COUNT(*) total, "
                            "SUM(CASE WHEN prob_raw IS NULL THEN 1 ELSE 0 END) missing "
                            "FROM predictions GROUP BY model_version ORDER BY missing DESC LIMIT 20"):
                mv,tot,ms=r; pct=100.0*ms/tot if tot else 0
                flag="  <<< fully missing" if ms==tot else ("  <-- partial" if ms>0 else "")
                print("       %-22s total=%6d missing=%6d (%5.1f%%)%s" % (str(mv),tot,ms,pct,flag))
        # by date range (month)
        if "prediction_date" in pc:
            sub("Missing prob_raw by month (find the boundary where logging started)")
            for r in q(conn,"SELECT substr(prediction_date,1,7) ym, COUNT(*) total, "
                            "SUM(CASE WHEN prob_raw IS NULL THEN 1 ELSE 0 END) missing "
                            "FROM predictions GROUP BY ym ORDER BY ym"):
                ym,tot,ms=r; pct=100.0*ms/tot if tot else 0
                bar="#"*int(pct/5)
                print("       %s total=%5d missing=%5d (%5.1f%%) %s" % (ym,tot,ms,pct,bar))
        print("\n  ACTION: the writer patch must backfill/duplicate prob_raw for the fully-missing")
        print("  model_versions or pre-boundary months. Rows where prob_raw IS NULL cannot enter")
        print("  the honest IC recompute — that's why 'ic' covered only ~74.7%.")
    finally:
        conn.close()
    _w(args,"rawgap",report)

# ============================================================ CORRUPT
def cmd_corrupt(args):
    banner("CORRUPT — characterize the flagged sentiment rows")
    sp=find_db(args.root,"sentiment.db")
    if not require(sp,"sentiment.db not found"): return
    conn=ro(sp); report={}
    try:
        if not require(has_table(conn,"sentiment_scores"),"no sentiment_scores"): return
        sc=cols_of(conn,"sentiment_scores")
        if not require("is_corrupted" in sc,"no is_corrupted column"): return
        tot=q(conn,"SELECT COUNT(*) FROM sentiment_scores")[0][0]
        bad=q(conn,"SELECT COUNT(*) FROM sentiment_scores WHERE is_corrupted=1")[0][0]
        print("  total=%d  corrupted=%d (%.1f%%)" % (tot,bad,100.0*bad/tot))
        # by date
        if "date" in sc:
            sub("Corrupted by month (is it ongoing or a one-time window?)")
            for r in q(conn,"SELECT substr(date,1,7) ym, COUNT(*) total, "
                            "SUM(is_corrupted) bad FROM sentiment_scores GROUP BY ym ORDER BY ym"):
                ym,t,b=r; pct=100.0*b/t if t else 0
                flag="  <<< all corrupt" if b==t else ""
                print("       %s total=%4d corrupt=%4d (%5.1f%%)%s" % (ym,t,b,pct,flag))
        # by source
        if "source" in sc:
            sub("Corrupted by source")
            for r in q(conn,"SELECT source, COUNT(*) total, SUM(is_corrupted) bad "
                            "FROM sentiment_scores GROUP BY source ORDER BY bad DESC"):
                s,t,b=r; print("       %-20s total=%5d corrupt=%5d (%5.1f%%)" % (str(s),t,b,100.0*b/t if t else 0))
        # what distinguishes corrupt rows? compare score distributions
        sub("Score distribution: corrupt vs clean")
        for flag,lab in ((1,"corrupt"),(0,"clean")):
            r=q(conn,"SELECT COUNT(*), ROUND(AVG(score),4), ROUND(MIN(score),4), ROUND(MAX(score),4) "
                     "FROM sentiment_scores WHERE is_corrupted=? AND score IS NOT NULL",(flag,))
            print("       %-8s n=%s avg=%s min=%s max=%s" % (lab,r[0][0],r[0][1],r[0][2],r[0][3]))
        # is the most recent row corrupt? (ongoing check)
        if "date" in sc:
            recent=q(conn,"SELECT date,is_corrupted FROM sentiment_scores WHERE date IS NOT NULL "
                          "ORDER BY date DESC LIMIT 10")
            print("\n    most recent 10 rows [date, is_corrupted]:")
            for r in recent: print("      ",r)
            last_bad=any(r[1]==1 for r in recent)
            print("    corruption in last 10 rows? %s %s" %
                  (last_bad, "<-- ONGOING, fix the writer" if last_bad else "(looks historical)"))
        print("\n  ACTION: exclude is_corrupted=1 from ALL training joins. If recent rows are flagged,")
        print("  the corruption process is live — trace it before trusting any new sentiment feature.")
    finally:
        conn.close()
    _w(args,"corrupt",report)

# ---- infra ----
def _w(args,name,report):
    if not args.out: return
    path=args.out
    if os.path.isdir(path) or path.endswith("/"): path=os.path.join(path,"phase2_%s.json"%name)
    with open(path,"a") as f:
        f.write(json.dumps({"subcommand":name,"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":report})+"\n")
    print("\n  [report appended to %s]" % path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cmd",choices=["decompose","pct7","rawgap","corrupt","all"])
    ap.add_argument("--root",default=".")
    ap.add_argument("--out",default=None)
    args=ap.parse_args(); args.root=os.path.expanduser(args.root)
    banner("ML QUANT FUND — PHASE 2 DIAGNOSTIC TOOLKIT")
    print("Read-only. No writes ever. Root:",os.path.abspath(args.root),
          "| Python",sys.version.split()[0],"| numpy:",HAVE_NUMPY)
    if args.cmd=="all":
        for fn in (cmd_decompose,cmd_pct7,cmd_rawgap,cmd_corrupt):
            try: fn(args)
            except Exception as e:
                import traceback; print("  [SUBCOMMAND FAILED] %s: %s"%(fn.__name__,e)); traceback.print_exc()
        return
    {"decompose":cmd_decompose,"pct7":cmd_pct7,"rawgap":cmd_rawgap,"corrupt":cmd_corrupt}[args.cmd](args)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
