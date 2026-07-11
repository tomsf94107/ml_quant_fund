#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — WALK-FORWARD INSPECTOR & ANALYZER  (self-configuring)
================================================================================
Goal: answer ONE question without you having to know anything in advance —
  "Does your existing Sunday walk-forward already isolate the OVERLAY GAIN
   (raw model vs effective/prob_up) out-of-sample, or has it been validating
   only the inflated number this whole time?"

It does this in three automatic phases:
  PHASE 1  DISCOVER: find walk_forward_history (and any sibling WF tables) across
           every .db under --root; print full schema + sample rows + date range.
  PHASE 2  CLASSIFY: inspect the columns and decide what the table actually tracks
           — does it carry raw AND effective metrics per fold? just one? AUC/IC,
           or only raw predictions you'd re-aggregate? It tells you which case
           you're in, in plain language.
  PHASE 3  ANALYZE: run whatever the data supports —
             * if per-fold raw & effective metrics exist -> report overlay gain
               per horizon out-of-sample (the decision-grade number).
             * if only predictions-with-dates exist -> aggregate OOS AUC/IC itself
               from the folds (joining to outcomes if needed).
             * if only one probability is tracked -> say so loudly: the Sunday job
               has been blind to the raw-vs-overlay question.

READ-ONLY. Never writes. SQLite opened mode=ro&immutable=1.

USAGE (project root, env active):
  python wf_inspect.py --root .
  add --out wf_report.jsonl to append machine-readable results.

Stats: exact pure-python (Mann-Whitney AUC, Spearman). numpy used if present.
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
    return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=15)
def q(c,s,p=()): return c.execute(s,p).fetchall()
def has_table(c,n): return bool(q(c,"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(n,)))
def tables(c): return [r[0] for r in q(c,"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
def cols_of(c,t): return [(r[1],r[2]) for r in q(c,'PRAGMA table_info("'+t+'")')]

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

# column-intent buckets
RAW_HINTS=["raw","base","model_prob","prob_raw","unadj"]
EFF_HINTS=["eff","prob_up","effective","final","adj","overlay","capped"]
METRIC_HINTS=["auc","roc","ic","rank_ic","rankic","acc","accuracy","sharpe","brier","hit","precision","recall","f1","logloss","auroc"]
PROB_HINTS=["prob","proba","score","pred","p_up","y_prob"]
RET_HINTS=["ret","return","fwd","actual","label","target","outcome","y_true","pnl","up"]
DATE_HINTS=["date","asof","as_of","ts","timestamp","fold","period","train_end","test_start","test_end","oos","window"]
HZN_HINTS=["horizon","hzn","h","period_days"]

def _tokenize(nm):
    # split on underscores and digits boundaries to get tokens
    import re
    return set(re.split(r'[_\W]+', nm.lower()))

def classify_columns(cols):
    names=[c[0].lower() for c in cols]
    found={"raw":[], "eff":[], "metric":[], "prob":[], "ret":[], "date":[], "hzn":[]}
    for nm in names:
        toks=_tokenize(nm)
        # metric: require a hint to appear as a whole token OR as a clear suffix like _auc
        if any(h in toks for h in METRIC_HINTS) or any(nm.endswith("_"+h) for h in METRIC_HINTS):
            found["metric"].append(nm)
        if any(h in toks for h in DATE_HINTS) or any(h in nm for h in ("date","asof","timestamp","train_end","test_start","test_end")):
            found["date"].append(nm)
        if nm in ("horizon","hzn","h","period_days"): found["hzn"].append(nm)
    for nm in names:
        toks=_tokenize(nm)
        if any(h in toks for h in RAW_HINTS) or nm.startswith("raw") or "_raw" in nm: found["raw"].append(nm)
        if any(h in toks for h in EFF_HINTS) or "_eff" in nm or "prob_up" in nm: found["eff"].append(nm)
        # prob: token match, but exclude pure metric/return columns
        if (any(h in toks for h in PROB_HINTS) or "prob" in nm) and nm not in found["metric"]:
            found["prob"].append(nm)
        if (any(h in toks for h in RET_HINTS) or "return" in nm) and nm not in found["metric"]:
            found["ret"].append(nm)
    for k in found: found[k]=sorted(set(found[k]))
    return found

def find_wf_tables(root):
    """Return list of (dbpath, tablename) for any table that looks like walk-forward."""
    hits=[]
    seen=set()
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f.endswith((".db",".sqlite",".sqlite3")):
                full=os.path.join(dp,f)
                if full in seen: continue
                seen.add(full)
                try:
                    c=ro(full)
                except Exception: continue
                try:
                    for t in tables(c):
                        tl=t.lower()
                        if any(k in tl for k in ("walk_forward","walkforward","wf_","_wf","walk")):
                            hits.append((full,t))
                finally:
                    c.close()
    return hits

def analyze_table(dbpath, tname, args, report):
    c=ro(dbpath)
    try:
        cols=cols_of(c,tname)
        n=q(c,'SELECT COUNT(*) FROM "'+tname+'"')[0][0]
        sub("TABLE %s  in  %s   (rows=%d)" % (tname, os.path.basename(dbpath), n))
        print("  columns:")
        for cn,ct in cols: print("     %-28s %s" % (cn,ct))
        intent=classify_columns(cols)
        print("\n  column intent classification:")
        for k in ("raw","eff","metric","prob","ret","date","hzn"):
            if intent[k]: print("     %-8s -> %s" % (k,intent[k]))
        # sample rows
        print("\n  sample rows (latest 3 by first date-ish col if any):")
        order=""
        if intent["date"]:
            order=' ORDER BY "'+intent["date"][-1]+'" DESC'
        try:
            rows=q(c,'SELECT * FROM "'+tname+'"'+order+' LIMIT 3')
            cnames=[cn for cn,_ in cols]
            for r in rows:
                trunc=tuple((str(x)[:22]+"…") if x is not None and len(str(x))>22 else x for x in r)
                print("     ",dict(zip(cnames,trunc)) if len(cnames)<=10 else trunc)
        except Exception as e:
            print("     [sample failed]",e)
        # date range
        if intent["date"]:
            dc=intent["date"][-1]
            try:
                dr=q(c,'SELECT MIN("'+dc+'"),MAX("'+dc+'") FROM "'+tname+'"')[0]
                print("\n  date range [%s]: %s .. %s" % (dc,dr[0],dr[1]))
            except Exception: pass

        # ---- CLASSIFY what this table can tell us ----
        sub("VERDICT for %s" % tname)
        has_raw_metric = any(any(rh in m for rh in RAW_HINTS) for m in intent["metric"]) or \
                         (intent["raw"] and intent["metric"])
        has_eff_metric = any(any(eh in m for eh in EFF_HINTS) for m in intent["metric"]) or \
                         (intent["eff"] and intent["metric"])
        # case A: explicit raw AND eff metric columns
        raw_metric_cols=[m for m in intent["metric"] if any(rh in m for rh in RAW_HINTS)]
        eff_metric_cols=[m for m in intent["metric"] if any(eh in m for eh in EFF_HINTS)]
        plain_metric_cols=[m for m in intent["metric"]
                           if m not in raw_metric_cols and m not in eff_metric_cols]

        verdict=None
        if raw_metric_cols and eff_metric_cols:
            verdict="A_BOTH_METRICS"
            print("  CASE A: table stores BOTH raw and effective metrics per fold.")
            print("  -> We can read the overlay gain DIRECTLY. raw cols=%s eff cols=%s"
                  % (raw_metric_cols,eff_metric_cols))
            _report_direct_gain(c,tname,intent,raw_metric_cols,eff_metric_cols,report)
        elif intent["prob"] and (intent["ret"] or _has_outcomes_db(args)):
            verdict="B_PREDICTIONS"
            print("  CASE B: table stores per-fold predictions (prob) %s."
                  % ("with returns/labels" if intent["ret"] else "— will join to outcomes"))
            print("  -> We can COMPUTE out-of-sample AUC/IC from the folds ourselves.")
            _aggregate_from_predictions(c,tname,intent,args,report)
        elif plain_metric_cols:
            verdict="C_ONE_METRIC"
            print("  CASE C: table stores fold metrics but NOT separated into raw vs effective.")
            print("  metric cols present: %s" % plain_metric_cols)
            print("  -> Cannot isolate overlay gain from THIS table. Reports the single tracked")
            print("     metric over time; you likely need raw-vs-eff added to the Sunday job.")
            _report_single_metric(c,tname,intent,plain_metric_cols,report)
        else:
            verdict="D_UNKNOWN"
            print("  CASE D: columns don't clearly map to metrics or predictions.")
            print("  -> Inspect the schema above; tell me which columns are the OOS metric/prob")
            print("     and I'll wire the exact analysis. (No guess made — RULE 1: fail loud.)")
        report.setdefault("tables",[]).append(
            {"db":os.path.basename(dbpath),"table":tname,"rows":n,
             "verdict":verdict,"intent":intent})
        print("\n  >> CASE %s" % verdict)
    finally:
        c.close()

def _has_outcomes_db(args):
    for dp,dn,fn in os.walk(args.root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f=="accuracy.db":
                try:
                    c=ro(os.path.join(dp,f))
                    ok=has_table(c,"outcomes"); c.close()
                    if ok: return True
                except Exception: pass
    return False

def _report_direct_gain(c,tname,intent,raw_cols,eff_cols,report):
    sub("OVERLAY GAIN out-of-sample (read directly from fold metrics)")
    hzn=intent["hzn"][0] if intent["hzn"] else None
    # pair raw/eff by matching the metric suffix (e.g. auc_raw vs auc_eff)
    def base_metric(m):
        for h in RAW_HINTS+EFF_HINTS: m=m.replace(h,"")
        return m.strip("_")
    pairs=[]
    for rc in raw_cols:
        bm=base_metric(rc)
        for ec in eff_cols:
            if base_metric(ec)==bm: pairs.append((bm,rc,ec))
    if not pairs:
        print("  could not auto-pair raw/eff metric columns; showing averages of each.")
        for col in raw_cols+eff_cols:
            v=q(c,'SELECT AVG("'+col+'") FROM "'+tname+'" WHERE "'+col+'" IS NOT NULL')[0][0]
            print("     avg %s = %s" % (col,v))
        return
    if hzn:
        hzs=[r[0] for r in q(c,'SELECT DISTINCT "'+hzn+'" FROM "'+tname+'" ORDER BY 1')]
    else:
        hzs=[None]
    for bm,rc,ec in pairs:
        print("  metric: %s" % bm)
        for h in hzs:
            where=' WHERE "'+rc+'" IS NOT NULL AND "'+ec+'" IS NOT NULL'
            params=()
            if h is not None and hzn:
                where+=' AND "'+hzn+'"=?'; params=(h,)
            r=q(c,'SELECT AVG("'+rc+'"), AVG("'+ec+'"), COUNT(*) FROM "'+tname+'"'+where,params)[0]
            if r[2]:
                gain=(r[1]-r[0]) if (r[0] is not None and r[1] is not None) else None
                print("     h=%-4s folds=%-4d  raw=%.4f  eff=%.4f  GAIN=%s"
                      % (str(h),r[2],r[0] or 0,r[1] or 0,
                         "%+.4f"%gain if gain is not None else "NA"))
    print("\n  -> Positive, STABLE gain across folds = overlay edge is real OOS.")
    print("     Gain that shrinks/flips in recent folds = overlay decaying or overfit.")

def _aggregate_from_predictions(c,tname,intent,args,report):
    sub("Computing OOS AUC/IC from per-fold predictions")
    probc=intent["prob"][0]
    retc=intent["ret"][0] if intent["ret"] else None
    datec=intent["date"][-1] if intent["date"] else None
    hznc=intent["hzn"][0] if intent["hzn"] else None
    # if no return col in WF table, try join to accuracy.db outcomes
    if retc is None:
        print("  no return/label column in WF table — attempting join to accuracy.db outcomes")
        print("  (needs ticker+date+horizon in the WF table; if absent, can't join)")
        names=[x[0].lower() for x in cols_of(c,tname)]
        if not ({"ticker"} <= set(names)):
            print("  [STOP] no ticker column to join on — cannot aggregate. Add a label col to WF.")
            return
    # pull and aggregate
    hzs=[None]
    if hznc:
        hzs=[r[0] for r in q(c,'SELECT DISTINCT "'+hznc+'" FROM "'+tname+'" ORDER BY 1')]
    for h in hzs:
        where=' WHERE "'+probc+'" IS NOT NULL'
        params=()
        if h is not None and hznc: where+=' AND "'+hznc+'"=?'; params=(h,)
        if retc:
            rows=q(c,'SELECT "'+probc+'","'+retc+'" FROM "'+tname+'"'+where,params)
            probs=[r[0] for r in rows]
            rets=[r[1] for r in rows]
            labs=[1 if (r[1] is not None and r[1]>0) else 0 for r in rows]
            a=auc(probs,labs); ic=spearman(probs,rets)
            print("     h=%-4s n=%-6d  AUC=%s  rankIC=%s"
                  % (str(h),len(rows),"%.4f"%a if a else "NA","%.4f"%ic if ic else "NA"))
        else:
            print("     h=%s: join path not implemented for this schema; paste schema and I'll wire it." % h)

def _report_single_metric(c,tname,intent,metric_cols,report):
    sub("Single tracked metric over time (cannot isolate overlay gain)")
    datec=intent["date"][-1] if intent["date"] else None
    hznc=intent["hzn"][0] if intent["hzn"] else None
    for mc in metric_cols[:4]:
        print("  metric: %s" % mc)
        if datec:
            rows=q(c,'SELECT substr("'+datec+'",1,7) ym'+(',"'+hznc+'"' if hznc else '')+
                     ', AVG("'+mc+'"), COUNT(*) FROM "'+tname+'" '
                     'WHERE "'+mc+'" IS NOT NULL GROUP BY ym'+(',"'+hznc+'"' if hznc else '')+
                     ' ORDER BY ym LIMIT 30')
            for r in rows: print("     ",r)
        else:
            v=q(c,'SELECT AVG("'+mc+'"),MIN("'+mc+'"),MAX("'+mc+'") FROM "'+tname+'"')[0]
            print("     avg=%s min=%s max=%s" % (v[0],v[1],v[2]))

def _w(args,report):
    if not args.out: return
    path=args.out
    if os.path.isdir(path) or path.endswith("/"): path=os.path.join(path,"wf_report.json")
    with open(path,"a") as f:
        f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":report})+"\n")
    print("\n  [report appended to %s]" % path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--out",default=None)
    args=ap.parse_args(); args.root=os.path.expanduser(args.root)
    banner("ML QUANT FUND — WALK-FORWARD INSPECTOR & ANALYZER")
    print("Read-only. Self-configuring. Root:",os.path.abspath(args.root),
          "| Python",sys.version.split()[0],"| numpy:",HAVE_NUMPY)

    banner("PHASE 1 — DISCOVER walk-forward tables")
    hits=find_wf_tables(args.root)
    if not hits:
        print("  No table matching walk_forward/walkforward/wf found under",args.root)
        print("  If the Sunday job writes elsewhere (a log, another dir), point --root there.")
        # also list ALL tables in accuracy.db to help
        acc=None
        for dp,dn,fn in os.walk(args.root):
            if "accuracy.db" in fn: acc=os.path.join(dp,"accuracy.db"); break
        if acc:
            c=ro(acc)
            print("\n  (for reference, tables in accuracy.db:)")
            for t in tables(c): print("     ",t)
            c.close()
        return
    print("  Found %d candidate walk-forward table(s):" % len(hits))
    for db,t in hits: print("     %s : %s" % (os.path.basename(db),t))

    report={}
    banner("PHASE 2/3 — CLASSIFY & ANALYZE each")
    for db,t in hits:
        try:
            analyze_table(db,t,args,report)
        except Exception as e:
            import traceback; print("  [FAILED %s] %s"%(t,e)); traceback.print_exc()

    banner("SUMMARY")
    for entry in report.get("tables",[]):
        print("  %-26s (%s)  rows=%d  -> %s"
              % (entry["table"],entry["db"],entry["rows"],entry["verdict"]))
    print("\n  CASE A = overlay gain readable directly (best).")
    print("  CASE B = we compute OOS AUC/IC from fold predictions.")
    print("  CASE C = only one metric tracked -> Sunday job is BLIND to raw-vs-overlay; needs patching.")
    print("  CASE D = schema unclear -> paste output, I wire the exact analysis.")
    _w(args,report)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
