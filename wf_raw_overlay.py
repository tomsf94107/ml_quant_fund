#!/usr/bin/env python3
"""
================================================================================
wf_raw_overlay.py — DROP-IN patch for the Sunday walk-forward job
================================================================================
Your walk_forward_history table logs ONE blended metric (auc/accuracy/buy_hit_55)
per fold, so it cannot tell whether the RAW model or the OVERLAY is responsible.
This module adds raw-vs-effective logging WITHOUT you editing the Sunday logic.

You add ONE import and ONE function call to wherever the Sunday job computes a
fold. Everything else (schema creation, both-metric computation, the write) is
handled here.

--------------------------------------------------------------------------------
WHAT IT WRITES
--------------------------------------------------------------------------------
A NEW table `walk_forward_raw_overlay` (leaves walk_forward_history untouched):

  run_date, ticker, horizon,
  auc_raw, auc_eff, ic_raw, ic_eff,
  acc_raw, acc_eff, n, created_at

So every Sunday you get the overlay gain (eff - raw) per ticker per horizon,
and can finally see whether your edge is the model or the multipliers.

--------------------------------------------------------------------------------
HOW TO INTEGRATE (3 lines)
--------------------------------------------------------------------------------
In your Sunday walk-forward script, wherever you already have, for one fold/ticker:
    - the per-row RAW model probabilities      (prob_raw)
    - the per-row EFFECTIVE/overlay probs       (prob_up)
    - the realized labels (actual_up, 0/1)      and forward returns (actual_return)

add at top:
    from wf_raw_overlay import log_raw_overlay_fold, ensure_schema

once at job start (optional — log_raw_overlay_fold also ensures it):
    ensure_schema("accuracy.db")

then per (ticker, horizon) fold, right after you compute the existing metrics:
    log_raw_overlay_fold(
        db_path="accuracy.db",
        run_date=run_date,            # e.g. "2026-06-29"
        ticker=ticker,
        horizon=h,
        prob_raw=list_of_raw_probs,   # aligned lists
        prob_eff=list_of_prob_up,
        actual_up=list_of_labels_0_1,
        actual_return=list_of_returns,
    )

That's it. The function computes auc_raw/auc_eff/ic_raw/ic_eff/acc_raw/acc_eff
and inserts one row. Safe to call repeatedly (idempotent per run_date+ticker+horizon
via REPLACE).

--------------------------------------------------------------------------------
IF YOU DON'T WANT A NEW TABLE — alter the existing one instead
--------------------------------------------------------------------------------
Call ensure_schema(db, mode="alter") to ADD columns auc_raw, auc_eff, ic_raw,
ic_eff to walk_forward_history itself. Then use update_existing_row(...) instead
of log_raw_overlay_fold(...). (See functions below.) The new-table approach is
recommended — it can't break your current Sunday writes.

This module is import-safe: importing it does nothing. It only acts when called.
================================================================================
"""
import sqlite3, math, datetime

# ----------------------------------------------------------------- exact stats
def _auc(scores, labels):
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

def _spearman(x,y):
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

def _acc(probs, labels, thr=0.5):
    if not probs: return None
    return sum(1 for p,l in zip(probs,labels) if (p>=thr)==(l==1))/len(probs)

# ----------------------------------------------------------------- schema
NEW_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS walk_forward_raw_overlay (
    run_date    TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    horizon     INTEGER NOT NULL,
    auc_raw     REAL,
    auc_eff     REAL,
    ic_raw      REAL,
    ic_eff      REAL,
    acc_raw     REAL,
    acc_eff     REAL,
    n           INTEGER,
    created_at  TEXT,
    PRIMARY KEY (run_date, ticker, horizon)
);
"""

def ensure_schema(db_path, mode="new"):
    """mode='new' creates walk_forward_raw_overlay (recommended).
       mode='alter' adds auc_raw/auc_eff/ic_raw/ic_eff to walk_forward_history."""
    conn=sqlite3.connect(db_path, timeout=20)
    try:
        if mode=="new":
            conn.executescript(NEW_TABLE_DDL); conn.commit()
            return "created/verified walk_forward_raw_overlay"
        elif mode=="alter":
            existing=[r[1] for r in conn.execute('PRAGMA table_info("walk_forward_history")')]
            added=[]
            for col in ("auc_raw","auc_eff","ic_raw","ic_eff"):
                if col not in existing:
                    conn.execute('ALTER TABLE walk_forward_history ADD COLUMN %s REAL' % col)
                    added.append(col)
            conn.commit()
            return "added columns to walk_forward_history: "+(", ".join(added) if added else "(none, already present)")
        else:
            raise ValueError("mode must be 'new' or 'alter'")
    finally:
        conn.close()

# ----------------------------------------------------------------- main entry
def log_raw_overlay_fold(db_path, run_date, ticker, horizon,
                          prob_raw, prob_eff, actual_up, actual_return,
                          ensure=True):
    """Compute raw & effective metrics for one fold and write one row.
    All four lists must be aligned (same length, same order). Returns the dict written.
    Idempotent: REPLACE on (run_date, ticker, horizon)."""
    # align + drop any None
    rows=[(pr,pe,au,rt) for pr,pe,au,rt in zip(prob_raw,prob_eff,actual_up,actual_return)
          if pr is not None and pe is not None and au is not None]
    if not rows:
        return None
    pr=[r[0] for r in rows]; pe=[r[1] for r in rows]
    au=[1 if r[2] in (1,True) else 0 for r in rows]
    rt=[r[3] for r in rows if r[3] is not None]
    rt_aligned = [r[3] for r in rows]  # may contain None; spearman needs pairs
    # for IC use only rows with a return
    pr_ic=[r[0] for r in rows if r[3] is not None]
    pe_ic=[r[1] for r in rows if r[3] is not None]
    rec=dict(
        run_date=run_date, ticker=ticker, horizon=int(horizon),
        auc_raw=_auc(pr,au), auc_eff=_auc(pe,au),
        ic_raw=_spearman(pr_ic,rt) if len(pr_ic)==len(rt) and rt else None,
        ic_eff=_spearman(pe_ic,rt) if len(pe_ic)==len(rt) and rt else None,
        acc_raw=_acc(pr,au), acc_eff=_acc(pe,au),
        n=len(rows), created_at=datetime.datetime.now().isoformat(timespec="seconds"))
    if ensure:
        ensure_schema(db_path, mode="new")
    conn=sqlite3.connect(db_path, timeout=20)
    try:
        conn.execute(
            "REPLACE INTO walk_forward_raw_overlay "
            "(run_date,ticker,horizon,auc_raw,auc_eff,ic_raw,ic_eff,acc_raw,acc_eff,n,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (rec["run_date"],rec["ticker"],rec["horizon"],rec["auc_raw"],rec["auc_eff"],
             rec["ic_raw"],rec["ic_eff"],rec["acc_raw"],rec["acc_eff"],rec["n"],rec["created_at"]))
        conn.commit()
    finally:
        conn.close()
    return rec

def update_existing_row(db_path, run_date, ticker, horizon,
                         prob_raw, prob_eff, actual_up, actual_return):
    """Alternative: write raw/eff metrics back into walk_forward_history
    (requires ensure_schema(mode='alter') first). Updates the matching row."""
    rows=[(pr,pe,au,rt) for pr,pe,au,rt in zip(prob_raw,prob_eff,actual_up,actual_return)
          if pr is not None and pe is not None and au is not None]
    if not rows: return None
    pr=[r[0] for r in rows]; pe=[r[1] for r in rows]
    au=[1 if r[2] in (1,True) else 0 for r in rows]
    rt=[r[3] for r in rows if r[3] is not None]
    pr_ic=[r[0] for r in rows if r[3] is not None]; pe_ic=[r[1] for r in rows if r[3] is not None]
    conn=sqlite3.connect(db_path, timeout=20)
    try:
        conn.execute(
            "UPDATE walk_forward_history SET auc_raw=?, auc_eff=?, ic_raw=?, ic_eff=? "
            "WHERE run_date=? AND ticker=? AND horizon=?",
            (_auc(pr,au),_auc(pe,au),
             _spearman(pr_ic,rt) if len(pr_ic)==len(rt) and rt else None,
             _spearman(pe_ic,rt) if len(pe_ic)==len(rt) and rt else None,
             run_date,ticker,int(horizon)))
        conn.commit()
    finally:
        conn.close()

# ----------------------------------------------------------------- self-test
if __name__=="__main__":
    # tiny self-test on a temp db (proves the module works; safe, uses /tmp)
    import tempfile, os, random
    random.seed(0)
    d=tempfile.mkdtemp(); db=os.path.join(d,"selftest.db")
    print("self-test db:",db)
    print(ensure_schema(db,"new"))
    pr=[random.uniform(0.3,0.7) for _ in range(200)]
    pe=[min(0.95,p*random.uniform(0.9,1.3)) for p in pr]
    au=[1 if random.random()<p else 0 for p in pr]
    rt=[random.uniform(-0.05,0.05)+(p-0.5)*0.04 for p in pr]
    rec=log_raw_overlay_fold(db,"2026-06-29","TEST",1,pr,pe,au,rt)
    print("wrote:",rec)
    conn=sqlite3.connect(db)
    print("rows in table:",conn.execute("SELECT COUNT(*) FROM walk_forward_raw_overlay").fetchone()[0])
    print("readback:",conn.execute("SELECT run_date,ticker,horizon,auc_raw,auc_eff,ic_raw,ic_eff,n FROM walk_forward_raw_overlay").fetchall())
    conn.close()
    print("OK — module works. Integrate with the 3 lines documented at top.")
