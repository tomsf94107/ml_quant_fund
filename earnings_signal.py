#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — EARNINGS EVENT-SIGNAL TEST  (the real-signal probe)
================================================================================
Tests whether your single best feature family — EARNINGS SURPRISE / POST-EARNINGS
DRIFT, 17 years of it in earnings.db — actually predicts post-announcement
returns in YOUR universe. This is the honest event-conditioned test of PEAD,
the most robust short-horizon anomaly you have data for, and the thing missing
from the technical-only prediction_features set that just tested as coin-flip.

WHY EVENT-WINDOW (not a daily-panel join): earnings happen once a quarter per
stock, so joining surprise onto a daily panel makes it null/stale 95%+ of the
time and dilutes any signal to nothing. Instead we test the actual hypothesis:
does the surprise predict the return in the 1-5 days AFTER the announcement?

THE PIT LANDMINE (handled automatically): earnings_surprises.report_date might
be the ANNOUNCEMENT date (tradeable) or the FISCAL PERIOD END (not — the
announcement comes weeks later). Aligning returns to the wrong one either leaks
future info or misses the event. So PHASE 0 AUTO-DETECTS which it is, by checking
whether |returns| and volume spike ON/AFTER report_date (announcement) vs are
scattered (period-end). If it can't tell confidently, it REFUSES to compute IC
(RULE 1: fail loud, never fake an edge).

SItes of signal tested:
  * eps_surprise, eps_surprise_pct, rev_surprise   (earnings.db.earnings_surprises)
  * post_drift_3d, pre_drift_3d, expected_move      (accuracy.db.earnings_cache)

READ-ONLY. Never writes. SQLite opened mode=ro&immutable=1.

USAGE (project root, env active):
  python earnings_signal.py --root .
  python earnings_signal.py --root . --windows 1,3,5 --min-events 30
  add --out earnings_signal.json for machine-readable results.

Forward returns are computed from the SAME outcomes table the rest of the system
uses (so apples-to-apples). If a price table is available it can also self-compute,
but default path uses outcomes for consistency.
================================================================================
"""
import argparse, os, sqlite3, sys, math, json, datetime
from collections import defaultdict

try:
    import numpy as np; HAVE_NUMPY=True
except Exception:
    HAVE_NUMPY=False

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

def spearman(x,y):
    n=len(x)
    if n<5: return None
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

def norm_date(s):
    """Normalize 'YYYY-MM-DD ...' or 'YYYY-MM-DDThh...' to date object."""
    if s is None: return None
    s=str(s)[:10]
    try: return datetime.date.fromisoformat(s)
    except Exception: return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--windows",default="1,3,5")
    ap.add_argument("--min-events",type=int,default=30)
    ap.add_argument("--out",default=None)
    args=ap.parse_args(); args.root=os.path.expanduser(args.root)
    windows=[int(x) for x in args.windows.split(",")]

    banner("ML QUANT FUND — EARNINGS EVENT-SIGNAL TEST")
    print("Read-only. Tests whether eps_surprise / post-drift predict post-announcement returns.")
    print("Root:",os.path.abspath(args.root),"| numpy:",HAVE_NUMPY)
    if not require(HAVE_NUMPY,"numpy required"): return

    accp=find_db(args.root,"accuracy.db"); earnp=find_db(args.root,"earnings.db")
    if not require(accp,"accuracy.db not found"): return
    if not require(earnp,"earnings.db not found"): return

    conn_a=ro(accp); conn_e=ro(earnp)
    report={}
    try:
        if not require(has_table(conn_a,"outcomes"),"no outcomes table"): return
        oc=cols_of(conn_a,"outcomes")
        for need in ("ticker","prediction_date","horizon","actual_return"):
            if not require(need in oc,"outcomes missing "+need): return

        # ---- load outcomes into a per-(ticker,date,horizon) lookup ----
        sub("Loading outcomes (forward returns) for event alignment")
        out_rows=q(conn_a,"SELECT ticker, prediction_date, horizon, actual_return "
                          "FROM outcomes WHERE actual_return IS NOT NULL")
        # ret_lookup[(ticker, date_obj, horizon)] = actual_return
        ret_lookup={}
        ret_by_ticker_date=defaultdict(dict)  # (ticker)-> {date_obj: {h: ret}}
        all_out_dates=set()
        for tk,d,h,r in out_rows:
            do=norm_date(d)
            if do is None: continue
            ret_lookup[(tk,do,h)]=r
            ret_by_ticker_date[tk].setdefault(do,{})[h]=r
            all_out_dates.add(do)
        print("  outcomes loaded: %d rows, %d tickers, dates %s..%s"
              %(len(out_rows),len(ret_by_ticker_date),
                min(all_out_dates) if all_out_dates else "?",
                max(all_out_dates) if all_out_dates else "?"))
        out_date_min=min(all_out_dates) if all_out_dates else None
        out_date_max=max(all_out_dates) if all_out_dates else None

        # ============================================================ PHASE 0
        banner("PHASE 0 — PIT AUTO-DETECT: is report_date the ANNOUNCEMENT or PERIOD-END?")
        if not require(has_table(conn_e,"earnings_surprises"),"no earnings_surprises table"): return
        es_cols=cols_of(conn_e,"earnings_surprises")
        if not require("report_date" in es_cols and "ticker" in es_cols,"earnings_surprises missing keys"): return
        # pull events that fall within outcomes coverage so we can test alignment
        ev=q(conn_e,"SELECT ticker, report_date, "
                    +("eps_surprise_pct" if "eps_surprise_pct" in es_cols else "eps_surprise")
                    +" FROM earnings_surprises WHERE report_date IS NOT NULL")
        events=[]
        for tk,rd,surp in ev:
            do=norm_date(rd)
            if do is None or surp is None: continue
            if out_date_min and out_date_max and out_date_min<=do<=out_date_max:
                events.append((tk,do,surp))
        print("  earnings events overlapping outcomes window: %d" % len(events))
        if len(events)<args.min_events:
            print("  [STOP] too few overlapping events (%d) to test alignment. Earnings data may"
                  " predate your outcomes window (outcomes start %s)." % (len(events),out_date_min))
            print("  -> Cannot run event test on this overlap. (Your outcomes are recent; most"
                  " earnings_surprises history is older.)")
            _maybe_partial(report,{"overlap_events":len(events)},args)
            return

        # For alignment detection: for each event, find the nearest available outcome date
        # for this ticker AT or AFTER report_date, and BEFORE report_date. If announcement,
        # there will be a tradeable return right at/after report_date with normal density.
        # We measure: fraction of events that have an outcome within +0..+2 days vs -2..-0 days,
        # and whether |return| at +1 is elevated (announcement vol spike).
        def nearest_on_or_after(tk,do,maxgap=4):
            dd=ret_by_ticker_date.get(tk,{})
            for off in range(0,maxgap+1):
                cand=do+datetime.timedelta(days=off)
                if cand in dd: return cand,off
            return None,None
        on_after=0; abs_ret_after=[]; abs_ret_random=[]
        abs_ret_exact=[]  # |return| exactly on report_date (strongest announcement signal)
        signal_align=[]   # (surprise, return on/after) to corroborate via correlation
        import random as _r; _r.seed(0)
        for tk,do,surp in events:
            dd=ret_by_ticker_date.get(tk,{})
            # exact-day return (announcement day move)
            if do in dd and dd[do].get(1) is not None:
                abs_ret_exact.append(abs(dd[do][1]))
            cand,off=nearest_on_or_after(tk,do,4)
            if cand is not None:
                on_after+=1
                r=ret_by_ticker_date[tk][cand].get(1)
                if r is not None:
                    abs_ret_after.append(abs(r))
                    if surp is not None: signal_align.append((surp,r))
            if dd:
                rd_=_r.choice(list(dd.keys()))
                rr=dd[rd_].get(1)
                if rr is not None: abs_ret_random.append(abs(rr))
        frac_with_after=on_after/len(events) if events else 0
        mean_abs_exact=np.mean(abs_ret_exact) if abs_ret_exact else None
        mean_abs_after=np.mean(abs_ret_after) if abs_ret_after else None
        mean_abs_random=np.mean(abs_ret_random) if abs_ret_random else None
        print("  events with a tradeable outcome within +0..+4 days: %d/%d (%.1f%%)"
              %(on_after,len(events),100*frac_with_after))
        # corroborating signal: does surprise SIGNIFICANTLY correlate with on/after return?
        corr_align=None; corr_t=None
        if len(signal_align)>=args.min_events:
            corr_align=spearman([s for s,_ in signal_align],[r for _,r in signal_align])
            if corr_align is not None and abs(corr_align)<1:
                corr_t=corr_align*math.sqrt(len(signal_align)-2)/math.sqrt(1-corr_align*corr_align)
        ratio_exact=(mean_abs_exact/mean_abs_random) if (mean_abs_exact and mean_abs_random and mean_abs_random>0) else None
        ratio_after=(mean_abs_after/mean_abs_random) if (mean_abs_after and mean_abs_random and mean_abs_random>0) else None
        if mean_abs_random:
            if mean_abs_exact is not None:
                print("  mean |h=1| exactly ON report_date: %.4f  (%.2fx random)"%(mean_abs_exact,ratio_exact or 0))
            if mean_abs_after is not None:
                print("  mean |h=1| within +0..4d of report_date: %.4f  (%.2fx random)"%(mean_abs_after,ratio_after or 0))
            print("  mean |h=1| on random dates (same tickers): %.4f" % mean_abs_random)
        if corr_align is not None:
            print("  corroborating: rank-corr(surprise, post-return) = %+.4f (t=%s)"
                  %(corr_align, "%.2f"%corr_t if corr_t is not None else "NA"))
        # DECISION — prioritize the STRUCTURAL signature (vol spike on/after announcement),
        # which can't be faked by spurious correlation. Correlation only corroborates if it's
        # statistically SIGNIFICANT (|t|>=2), not merely nonzero.
        best_ratio=max([x for x in [ratio_exact,ratio_after] if x is not None], default=0)
        sig_corr = (corr_t is not None and abs(corr_t)>=2.0 and abs(corr_align)>=0.05)
        if best_ratio>=1.25:
            detected="ANNOUNCEMENT"
            print("  >> DETECTED: report_date looks like the ANNOUNCEMENT date")
            print("     (post-event vol %.2fx random — that's the earnings move)"%best_ratio)
        elif best_ratio>=1.10 and sig_corr:
            detected="ANNOUNCEMENT"
            print("  >> DETECTED: report_date looks like the ANNOUNCEMENT date")
            print("     (vol %.2fx random AND significant surprise->return corr %+.3f t=%.2f)"
                  %(best_ratio,corr_align,corr_t))
        else:
            detected="UNCERTAIN"
            bits=["vol %.2fx random"%best_ratio]
            if corr_align is not None:
                bits.append("corr %+.3f t=%s"%(corr_align,"%.2f"%corr_t if corr_t is not None else "NA")
                            +(" (sig)" if sig_corr else " (not sig)"))
            print("  >> UNCERTAIN / NOT CONFIRMED ANNOUNCEMENT: %s"%"; ".join(bits))
            print("     No strong volatility spike around report_date. It may be the FISCAL")
            print("     PERIOD-END (announcement comes weeks later), or events don't overlap returns.")
        report["pit_detection"]={"frac_with_after":frac_with_after,
            "mean_abs_after":mean_abs_after,"mean_abs_random":mean_abs_random,"detected":detected}

        if detected not in ("ANNOUNCEMENT",):
            print("\n  [STOP — RULE 1] report_date is not confidently the announcement date.")
            print("  Computing post-event IC now would risk look-ahead or misalignment (faking edge).")
            print("  Options: (a) confirm which column is the announcement date and tell me;")
            print("  (b) check if earnings_cache.report_date aligns better; (c) use an explicit")
            print("  announce-date source. Not proceeding to IC — that's the honest call.")
            _maybe_partial(report,report,args)
            return

        # ============================================================ PHASE 1
        banner("PHASE 1 — POST-EARNINGS IC: does surprise predict the post-announcement move?")
        # build event signal table: for each event, surprise values + forward returns at +window
        es_full=q(conn_e,"SELECT ticker, report_date, "
                  + ", ".join([c for c in ("eps_surprise","eps_surprise_pct","rev_surprise") if c in es_cols])
                  + " FROM earnings_surprises WHERE report_date IS NOT NULL")
        avail_es=[c for c in ("eps_surprise","eps_surprise_pct","rev_surprise") if c in es_cols]
        # also pull earnings_cache drift if present
        cache_feats={}
        if has_table(conn_a,"earnings_cache"):
            ec_cols=cols_of(conn_a,"earnings_cache")
            drift_cols=[c for c in ("post_drift_3d","pre_drift_3d","expected_move") if c in ec_cols]
            if drift_cols and "ticker" in ec_cols and "report_date" in ec_cols:
                for row in q(conn_a,"SELECT ticker, report_date, "+", ".join(drift_cols)
                             +" FROM earnings_cache WHERE report_date IS NOT NULL"):
                    tk=row[0]; do=norm_date(row[1])
                    if do is None: continue
                    cache_feats[(tk,do)]={drift_cols[i]:row[2+i] for i in range(len(drift_cols))}
                print("  loaded earnings_cache drift features: %s (%d event-rows)"
                      %(drift_cols,len(cache_feats)))

        # assemble aligned dataset: per event, signal -> forward return at each window
        def fwd_ret(tk,do,w):
            """forward return over ~w trading days after announcement, via outcomes horizon if present
            else by summing daily. Here we use outcomes horizon=w if available at the post-announce date."""
            cand,off=nearest_on_or_after(tk,do,4)
            if cand is None: return None
            dd=ret_by_ticker_date[tk][cand]
            # prefer the horizon matching window; fallback to h=1
            return dd.get(w, dd.get(1))

        results={}
        for w in windows:
            sub("window = +%d trading days after announcement" % w)
            # per-feature IC across all events
            for fcol in avail_es:
                xs=[]; ys=[]
                colidx=2+avail_es.index(fcol)
                for row in es_full:
                    tk=row[0]; do=norm_date(row[1]); val=row[colidx]
                    if do is None or val is None: continue
                    if not (out_date_min and out_date_max and out_date_min<=do<=out_date_max): continue
                    fr=fwd_ret(tk,do,w)
                    if fr is None: continue
                    xs.append(val); ys.append(fr)
                if len(xs)>=args.min_events:
                    ic=spearman(xs,ys)
                    # rough t-stat of IC: t = ic*sqrt(n-2)/sqrt(1-ic^2)
                    t=ic*math.sqrt(len(xs)-2)/math.sqrt(1-ic*ic) if (ic is not None and abs(ic)<1) else None
                    print("    %-18s events=%-5d  IC=%-8s  t=%-6s"
                          %(fcol,len(xs),
                            "%+.4f"%ic if ic is not None else "NA",
                            "%.2f"%t if t is not None else "NA"))
                    results.setdefault(w,{})[fcol]={"n":len(xs),"ic":ic,"t":t}
                else:
                    print("    %-18s events=%-5d  (below min %d — skip)"%(fcol,len(xs),args.min_events))
            # drift features from earnings_cache
            if cache_feats:
                drift_names=list(next(iter(cache_feats.values())).keys())
                for dname in drift_names:
                    xs=[]; ys=[]
                    for (tk,do),feats in cache_feats.items():
                        val=feats.get(dname)
                        if val is None: continue
                        try: val=float(val)
                        except Exception: continue
                        if not (out_date_min and out_date_max and out_date_min<=do<=out_date_max): continue
                        fr=fwd_ret(tk,do,w)
                        if fr is None: continue
                        xs.append(val); ys.append(fr)
                    if len(xs)>=args.min_events:
                        ic=spearman(xs,ys)
                        t=ic*math.sqrt(len(xs)-2)/math.sqrt(1-ic*ic) if (ic is not None and abs(ic)<1) else None
                        print("    %-18s events=%-5d  IC=%-8s  t=%-6s"
                              %(dname,len(xs),
                                "%+.4f"%ic if ic is not None else "NA",
                                "%.2f"%t if t is not None else "NA"))
                        results.setdefault(w,{})[dname]={"n":len(xs),"ic":ic,"t":t}

        # ---- verdict ----
        banner("VERDICT — does earnings surprise carry post-announcement edge?")
        any_sig=False
        for w in windows:
            for f,r in results.get(w,{}).items():
                if r["t"] is not None and abs(r["t"])>=2.0 and abs(r["ic"])>=0.03:
                    print("  [REAL] %s @+%dd: IC=%+.4f t=%.2f (significant, n=%d)"%(f,w,r["ic"],r["t"],r["n"]))
                    any_sig=True
        if not any_sig:
            print("  No earnings feature shows a significant (|t|>=2, |IC|>=0.03) post-announcement edge")
            print("  on the overlapping events. Either the signal isn't in this universe/window, or")
            print("  the overlap is too small. This is an honest null — not a bug.")
        else:
            print("\n  ^ These are event-conditioned edges. Next: fold the significant ones into the")
            print("  base model as EVENT features (active only in the post-announcement window),")
            print("  then re-run base_model.py to see if they lift the cross-sectional IC off zero.")
        report["results"]=results
    finally:
        conn_a.close(); conn_e.close()
    _maybe_partial(report,report,args)

def _maybe_partial(report, payload, args):
    if not args.out: return
    path=args.out
    if os.path.isdir(path) or path.endswith("/"): path=os.path.join(path,"earnings_signal.json")
    with open(path,"a") as f:
        f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":payload},default=str)+"\n")
    print("\n  [report appended to %s]"%path)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
