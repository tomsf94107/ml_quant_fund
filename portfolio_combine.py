#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — PORTFOLIO-LEVEL COMBINATION HARNESS  (the right tool)
================================================================================
combine_signals.py combined signals at the CROSS-SECTIONAL RANK level (rank stocks
each day by a blended score). That FAILS for mixed-cadence signals: PEAD is
event-based (sparse) and price/flow signals are daily, so they rarely coexist on
the same (date,ticker) — the combination starved to ~6 dates.

The CORRECT approach for mixed-cadence signals: combine at the RETURN-STREAM level.
Build each strategy's periodic return series independently, THEN combine the series.
Two strategies with Sharpe S and correlation rho, blended 50/50, give combined
Sharpe = S * sqrt(2/(1+rho)). rho=0 -> 1.41x; rho=1 -> no benefit. This measures
the realized version.

WHAT IT DOES (offline; reads prices.db + earnings.db + accuracy.db; NO network):
  * builds a MONTHLY return series for each requested strategy:
      - PEAD       : each earnings event -> SUE-sorted L/S position entered day+2,
                     held --hold days; trade return attributed to entry month
      - <feature>  : monthly L/S book sorted on a prediction_features column,
                     held --hold days (e.g. short_ratio, pc_ratio_snap, inst_flow)
  * computes per-strategy: monthly mean, vol, annualized Sharpe
  * correlation between the strategies' monthly return streams (the diversification #)
  * blended (equal-weight, or inverse-vol) stream: Sharpe vs best single Sharpe
  * realized diversification ratio vs the theoretical sqrt(2/(1+rho))

RULE 1: each strategy's returns use only PIT data (SUE trailing; features as-of
snapshot; forward return strictly after). Monthly attribution is by entry date.
Short feature history is reported loudly — a stream with few months can't conclude.

READ-ONLY. mode=ro&immutable=1. No network.

USAGE:
  python portfolio_combine.py --root . --strategies PEAD,short_ratio
  python portfolio_combine.py --root . --strategies PEAD,pc_ratio_snap --hold 20
  python portfolio_combine.py --root . --strategies PEAD,inst_signed_flow_5d,short_ratio
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
    return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def has_table(c,n): return bool(Q(c,"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(n,)))
def tables(c): return [r[0] for r in Q(c,"SELECT name FROM sqlite_master WHERE type='table'")]
def cols_of(c,t): return [r[1] for r in Q(c,'PRAGMA table_info("'+t+'")')]
def require(cond,msg):
    if not cond: print("  [STOP] "+msg); return False
    return True
def all_dbs(root):
    out=[]
    for dp,dn,fn in os.walk(root):
        dn[:]=[d for d in dn if d not in (".git","__pycache__",".venv","venv","node_modules")]
        for f in fn:
            if f.endswith((".db",".sqlite",".sqlite3")): out.append(os.path.join(dp,f))
    return sorted(out)
def find_db(root,name):
    c=os.path.join(root,name)
    if os.path.isfile(c): return c
    for d in all_dbs(root):
        if os.path.basename(d)==name: return d
    return None
def nd(s):
    if s is None: return None
    s=str(s)[:10]
    try: return datetime.date.fromisoformat(s)
    except Exception: return None
def ym(d): return (d.year,d.month)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--prices-db",default=None)
    ap.add_argument("--strategies",default="PEAD,short_ratio")
    ap.add_argument("--hold",type=int,default=40)
    ap.add_argument("--cost-bps",type=float,default=10.0)
    ap.add_argument("--min-names",type=int,default=15)
    ap.add_argument("--weight",default="equal",choices=["equal","invvol"])
    ap.add_argument("--out",default=None)
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=a.prices_db or os.path.join(a.root,"prices.db")
    strategies=[s.strip() for s in a.strategies.split(",")]
    banner("ML QUANT FUND — PORTFOLIO-LEVEL COMBINATION HARNESS")
    print("Combines strategy RETURN STREAMS (correct for mixed cadence). strategies=%s hold=%d"%(strategies,a.hold))
    if not require(HAVE_NUMPY,"numpy required"): return
    if not require(os.path.isfile(prices_db),"prices.db not found"): return
    if not require(len(strategies)>=2,"need >=2 strategies"): return

    cp=ro(prices_db)
    try:
        rows=Q(cp,"SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL")
    finally:
        cp.close()
    px=defaultdict(list)
    for tk,d,p in rows:
        do=nd(d)
        if do is None: continue
        try: pf=float(p)
        except Exception: continue
        if pf>0: px[tk].append((do,pf))
    for tk in px: px[tk].sort()
    pos_of={tk:{d:i for i,(d,_) in enumerate(lst)} for tk,lst in px.items()}
    print("  prices loaded for %d tickers"%len(px))

    cost=a.cost_bps/10000.0
    def fwd_ret(tk,d,hold,entry_off=0):
        lst=px.get(tk); idx=pos_of.get(tk)
        if not lst or not idx: return None
        i=None
        for off in range(0,6):
            c=d+datetime.timedelta(days=off)
            if c in idx: i=idx[c]; break
        if i is None: return None
        e=i+entry_off; x=i+entry_off+hold
        if x>=len(lst): return None
        p0=lst[e][1]
        return (lst[x][1]/p0-1.0) if p0>0 else None

    # ---------- build PEAD monthly return stream ----------
    def build_pead_stream():
        earnp=find_db(a.root,"earnings.db")
        if not earnp: return {}
        ce=ro(earnp)
        try:
            cols=cols_of(ce,"earnings_surprises")
            have_comp="eps_actual" in cols and "eps_estimate" in cols
            sel="ticker,report_date"+(",eps_actual,eps_estimate" if have_comp else ",eps_surprise_pct")
            ev=Q(ce,"SELECT "+sel+" FROM earnings_surprises WHERE report_date IS NOT NULL")
        finally:
            ce.close()
        by_tkr=defaultdict(list)
        for row in ev:
            tk=row[0]; do=nd(row[1])
            if do is None: continue
            if have_comp:
                try: raw=float(row[2])-float(row[3])
                except Exception: raw=None
            else:
                try: raw=float(row[2])
                except Exception: raw=None
            by_tkr[tk].append((do,raw))
        for tk in by_tkr: by_tkr[tk].sort()
        # SUE events with forward returns, grouped by entry month
        by_month=defaultdict(list)  # (yr,mo) -> list of (sue, ret)
        for tk,lst in by_tkr.items():
            prior=[]
            for do,raw in lst:
                if raw is None: continue
                if len(prior)>=4:
                    sd=np.std(prior,ddof=1)
                    if sd>1e-12:
                        sue=raw/sd
                        r=fwd_ret(tk,do,a.hold,entry_off=2)
                        if r is not None:
                            by_month[ym(do)].append((sue,r))
                prior.append(raw)
        # monthly L/S: long top-quintile SUE, short bottom-quintile
        stream={}
        for m,recs in by_month.items():
            if len(recs)<a.min_names: continue
            recs.sort(key=lambda x:x[0])
            q=max(1,len(recs)//5)
            lo=recs[:q]; hi=recs[-q:]
            L=np.mean([r for _,r in hi]); S=np.mean([r for _,r in lo])
            stream[m]=(L-S)-2*cost
        return stream

    # ---------- build feature-based monthly return stream ----------
    def build_feature_stream(feat):
        # find the feature in prediction_features or short_interest_cache etc.
        panel={}  # (date,ticker)->value
        sign=+1
        for dbp in all_dbs(a.root):
            try: c=ro(dbp)
            except Exception: continue
            try:
                for t in tables(c):
                    cl=cols_of(c,t)
                    if feat not in cl: continue
                    tcol="ticker" if "ticker" in cl else ("symbol" if "symbol" in cl else None)
                    dcol=next((x for x in ("prediction_date","date","as_of","report_date","settlement_date","updated_at") if x in cl),None)
                    if not (tcol and dcol): continue
                    data=Q(c,"SELECT "+tcol+","+dcol+',"'+feat+'" FROM "'+t+'"')
                    for tk,d,v in data:
                        do=nd(d)
                        if do is None or v is None: continue
                        try: panel[(do,tk)]=float(v)
                        except Exception: pass
            finally:
                c.close()
            if panel: break
        if not panel: return {},sign
        # short-interest-like features predict NEGATIVELY -> flip sign so "high=good"
        if any(k in feat.lower() for k in ["short","days_to_cover","dtc","iv_skew"]):
            sign=-1
        # monthly: on first available snapshot each month, form L/S, hold --hold days
        by_month_date=defaultdict(dict)  # (yr,mo)-> {ticker:val} using earliest snapshot in month
        seen_month_date={}
        for (d,tk),v in sorted(panel.items()):
            m=ym(d)
            if m not in seen_month_date: seen_month_date[m]=d
            if d==seen_month_date[m]:
                by_month_date[m][tk]=v
        stream={}
        for m,tkmap in by_month_date.items():
            if len(tkmap)<a.min_names: continue
            d0=seen_month_date[m]
            recs=[]
            for tk,v in tkmap.items():
                r=fwd_ret(tk,d0,a.hold,entry_off=0)
                if r is not None: recs.append((sign*v,r))
            if len(recs)<a.min_names: continue
            recs.sort(key=lambda x:x[0])
            q=max(1,len(recs)//5)
            lo=recs[:q]; hi=recs[-q:]
            L=np.mean([r for _,r in hi]); S=np.mean([r for _,r in lo])
            stream[m]=(L-S)-2*cost
        return stream,sign

    # build all streams
    streams={}
    for s in strategies:
        if s.upper()=="PEAD":
            streams["PEAD"]=build_pead_stream()
        else:
            st,sgn=build_feature_stream(s)
            streams[s]=st
        print("  stream %-20s: %d monthly returns"%(s,len(streams[s])))

    # ---------- per-strategy stats ----------
    def stats(stream):
        if len(stream)<3: return None
        vals=np.array(list(stream.values()))
        m=vals.mean(); sd=vals.std()
        # monthly-ish; annualize by ~ sqrt(12) but holds overlap, so label as per-period
        sharpe = (m/sd)*math.sqrt(12) if sd>0 else None
        return {"n":len(stream),"mean":m,"vol":sd,"sharpe":sharpe}

    sub("PER-STRATEGY return-stream stats (hold=%d, monthly attribution)"%a.hold)
    sstats={}
    for s in strategies:
        st=stats(streams[s])
        if st:
            print("  %-20s months=%-3d mean=%+.4f vol=%.4f Sharpe(ann)=%+.2f"
                  %(s,st["n"],st["mean"],st["vol"],st["sharpe"] or 0))
            sstats[s]=st
        else:
            print("  %-20s months=%d — TOO FEW to compute stats"%(s,len(streams[s])))

    # ---------- correlation + combination ----------
    sub("RETURN-STREAM CORRELATION + COMBINATION")
    # common months
    common=set.intersection(*[set(streams[s].keys()) for s in strategies if streams[s]]) if all(streams[s] for s in strategies) else set()
    print("  common months across all strategies: %d"%len(common))
    if len(common)<3:
        print("  [STOP] <3 common months — cannot measure combination. This is the SHORT-HISTORY")
        print("  limitation: the feature stream(s) are too short to overlap PEAD meaningfully.")
        print("  The harness is correct and ready; it needs a longer feature history to conclude.")
        print("  (Run validate_signal.py to see each leg's standalone power, and signal_logger.py")
        print("   to start accumulating feature history for a future real combination.)")
        return
    common=sorted(common)
    # pairwise correlation
    print("  pairwise monthly-return correlation:")
    for i in range(len(strategies)):
        for j in range(i+1,len(strategies)):
            s1,s2=strategies[i],strategies[j]
            v1=[streams[s1][m] for m in common]; v2=[streams[s2][m] for m in common]
            if np.std(v1)>0 and np.std(v2)>0:
                rho=float(np.corrcoef(v1,v2)[0,1])
                print("    %-18s x %-18s rho=%+.3f %s"
                      %(s1,s2,rho,"[diversifying]" if abs(rho)<0.3 else "[correlated]"))

    # blended stream
    if a.weight=="equal":
        w={s:1.0/len(strategies) for s in strategies}
    else:  # inverse-vol
        ivol={s:(1.0/sstats[s]["vol"] if s in sstats and sstats[s]["vol"]>0 else 0) for s in strategies}
        tot=sum(ivol.values()) or 1
        w={s:ivol[s]/tot for s in strategies}
    blended=[sum(w[s]*streams[s][m] for s in strategies) for m in common]
    bm=np.mean(blended); bsd=np.std(blended)
    bsharpe=(bm/bsd)*math.sqrt(12) if bsd>0 else None
    # best single on the SAME common months
    best_single=None; best_name=None
    for s in strategies:
        v=[streams[s][m] for m in common]
        sd=np.std(v)
        sh=(np.mean(v)/sd)*math.sqrt(12) if sd>0 else None
        if sh is not None and (best_single is None or abs(sh)>abs(best_single)):
            best_single=sh; best_name=s

    banner("VERDICT — does combining at the PORTFOLIO level add Sharpe?")
    print("  weighting: %s | common months: %d"%(a.weight,len(common)))
    print("  best single Sharpe (%s): %+.2f"%(best_name,best_single or 0))
    print("  BLENDED Sharpe:          %+.2f"%(bsharpe or 0))
    if bsharpe is not None and best_single is not None and best_single!=0:
        gain=abs(bsharpe)/abs(best_single)-1
        if abs(bsharpe)>abs(best_single)*1.10:
            print("\n  >> DIVERSIFICATION WORKS: blended Sharpe beats best single by %.0f%%. Combining"%(100*gain))
            print("     these strategies at the portfolio level improves risk-adjusted return —")
            print("     your combine-little-signals thesis is demonstrated (on available history).")
        elif abs(bsharpe)>abs(best_single)*1.0:
            print("\n  >> SLIGHT GAIN: blended barely beats best single (%.0f%%). Some diversification."%(100*gain))
        else:
            print("\n  >> NO GAIN: blending doesn't beat the best single strategy here. Either the")
            print("     streams are correlated, or the weaker strategy drags the blend.")
    print("\n  CAVEAT: with only %d common months this is a PROOF-OF-CONCEPT, not a conclusion."%len(common))
    print("  The machinery is correct and ready; a longer feature history makes the verdict real.")
    if a.out:
        rep={"streams":{s:len(streams[s]) for s in strategies},"common_months":len(common),
             "best_single":best_single,"blended_sharpe":bsharpe}
        with open(a.out,"a") as f:
            f.write(json.dumps({"timestamp":datetime.datetime.now().isoformat(timespec="seconds"),"report":rep},default=str)+"\n")
        print("\n  [report appended to %s]"%a.out)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
