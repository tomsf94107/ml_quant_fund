#!/usr/bin/env python3
"""
================================================================================
ML QUANT FUND — SI STRATEGY: LIVE POSITION GENERATOR
================================================================================
Emits TODAY's tradeable book for the validated single-brick short-interest strategy.
This is an EXECUTION tool (real capital), not a backtest -- so it is built defensively.

WHAT IT DOES: rank the universe by days_to_cover from the MOST RECENT FINRA
settlement, LONG the low-DTC quintile (the survivorship-safe, borrow-free, high-
Sharpe leg -- 71% of the edge, leg Sharpe +1.25), and OPTIONALLY short a REDUCED
fraction of the high-DTC quintile. Output: weights, $ allocation, and share counts.

WHY LONG-TILTED (not dollar-neutral): the leg decomposition showed the short leg is
only 29% of the edge, not individually significant (NW t +1.71), AND it shorts high-
days-to-cover names -- exactly the hardest/most expensive to borrow (high SI = high
borrow cost). Backtest costs (10-40bps) did NOT include borrow fees, which would
erode the short leg in live trading. So default is long-tilted; --short-frac 0 =
long-only, --short-frac 1.0 = dollar-neutral (not recommended live).

LIVE SAFEGUARDS:
  * STALE GUARD: FINRA settles ~bi-monthly (publish ~8 business days later). If the
    most recent settlement in the DB is older than --max-stale-days (default 30), the
    signal is stale -> REFUSES to emit positions unless --force. (Run si_fetch_v2.py
    first to pull any newer settlement.)
  * PRICE-AGE CHECK: reports how old the latest price is; warns if > 5 days.
  * PIT: ranking uses ONLY the latest settlement; entry is at the next real price. No
    look-ahead -- you are acting today on already-published data.
  * per-name weight cap (--max-weight).

HONEST EXPECTATION (do not oversize): through-cycle Sharpe ~1.0-1.2 NET (NOT the
recent ~2.0 -- that's a favorable-regime reading that will revert). Apply a ~3%/yr
survivorship haircut to the (small) short-leg portion. Sizing vs your capital is YOUR
decision; this tool does the arithmetic, not the risk call.

RULE 1: latest-settlement ranking only (PIT); clip DTC>=50; weight cap; stale guard;
READ-ONLY except an optional --log-ledger. Not investment advice.

USAGE:
  python si_positions_live.py --root .                          # weights only, with guards
  python si_positions_live.py --root . --capital 100000         # + $ and share counts
  python si_positions_live.py --root . --capital 100000 --short-frac 0   # long-only
  python si_positions_live.py --root . --capital 100000 --gross 0.5 --log-ledger
================================================================================
"""
import argparse, os, sqlite3, math, datetime, csv
from collections import defaultdict

def ro(p): return sqlite3.connect("file:"+os.path.abspath(p)+"?mode=ro&immutable=1",uri=True,timeout=30)
def conn_rw(p): return sqlite3.connect(p,timeout=30)
def Q(c,s,p=()): return c.execute(s,p).fetchall()
def nd(s):
    if s is None: return None
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None
LINE="="*78

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--capital",type=float,default=None,help="account capital for $ + share sizing")
    ap.add_argument("--gross",type=float,default=1.0,help="fraction of capital to deploy as LONG gross (default 1.0)")
    ap.add_argument("--short-frac",type=float,default=0.3,help="short gross as fraction of long gross (0=long-only, 1=dollar-neutral)")
    ap.add_argument("--quantile",type=float,default=0.2)
    ap.add_argument("--min-names",type=int,default=20)
    ap.add_argument("--max-weight",type=float,default=0.05)
    ap.add_argument("--vol-target",action="store_true",
        help="size by 1/vol (equal RISK per name) instead of equal DOLLARS. "
             "Equal-dollar means a 90%%-vol name contributes ~7x the portfolio "
             "variance of a 12%%-vol name: the book's risk is then dominated by "
             "whichever handful of names happen to be most volatile, which is NOT "
             "a decision anyone made. Predicting vol is the one thing this system "
             "does well (OOS Spearman +0.71), so use it where it pays: sizing.")
    ap.add_argument("--vol-window",type=int,default=20,
        help="trailing days for the realized-vol estimate used by --vol-target")
    ap.add_argument("--max-stale-days",type=int,default=30)
    ap.add_argument("--force",action="store_true",help="emit positions even if the settlement is stale")
    ap.add_argument("--log-ledger",action="store_true")
    a=ap.parse_args(); a.root=os.path.expanduser(a.root)
    prices_db=os.path.join(a.root,"prices.db"); si_db=os.path.join(a.root,"short_interest.db")
    print("\n"+LINE+"\nSI STRATEGY — LIVE POSITION GENERATOR\n"+LINE)
    for lbl,p in (("prices.db",prices_db),("short_interest.db",si_db)):
        if not p or not os.path.isfile(p): print("[STOP] %s not found"%lbl); return
    today=datetime.date.today()

    # latest settlement + freshness
    c=ro(si_db)
    try:
        latest=Q(c,"SELECT MAX(settlement_date) FROM short_interest")[0][0]
        rows=Q(c,"SELECT ticker,days_to_cover FROM short_interest WHERE settlement_date=? AND days_to_cover IS NOT NULL",(latest,))
    finally: c.close()
    d0=nd(latest); age=(today-d0).days if d0 else 9999
    print("  latest settlement: %s  (%d days old)"%(latest,age))
    if age>a.max_stale_days and not a.force:
        print("\n  [STOP — STALE] the most recent settlement is %d days old (> %d)."%(age,a.max_stale_days))
        print("  FINRA settles ~bi-monthly; a newer one may have published. Run first:")
        print("     source ~/.finra_creds && python si_fetch_v2.py --root . --months-back 2")
        print("  Then re-run. (Override with --force only if you knowingly want this settlement.)")
        return
    if age>a.max_stale_days and a.force:
        print("  [FORCED] emitting on a %d-day-old settlement at your instruction."%age)

    # latest price per ticker (for sizing)
    cp=ro(prices_db)
    try: prows=Q(cp,"SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL")
    finally: cp.close()
    last_px={}; last_dt={}
    for tk,d,p in prows:
        do=nd(d)
        if do is None: continue
        try: pf=float(p)
        except Exception: continue
        if pf<=0: continue
        if tk not in last_dt or do>last_dt[tk]:
            last_dt[tk]=do; last_px[tk]=pf
    if last_dt:
        newest_px=max(last_dt.values()); pxage=(today-newest_px).days
        print("  latest price date: %s  (%d days old)%s"%(newest_px,pxage," — WARNING: prices look stale" if pxage>5 else ""))

    # rankable universe: has DTC at latest settlement (clip junk) AND a price
    uni=[]
    for tk,dtc in rows:
        try: v=float(dtc)
        except Exception: continue
        if v>50.0: continue
        tku=tk.upper()
        if tku in last_px: uni.append((tku,v))
    if len(uni)<a.min_names:
        print("\n  [STOP] only %d rankable names (need %d)."%(len(uni),a.min_names)); return
    uni.sort(key=lambda x:x[1])  # ascending DTC
    q=max(1,int(len(uni)*a.quantile))
    longs=uni[:q]            # low DTC
    shorts=uni[-q:]          # high DTC
    print("  rankable universe: %d names | quintile size: %d"%(len(uni),q))

    # ── SIZING ───────────────────────────────────────────────────────────────
    # DEFAULT (equal weight): every name gets the same DOLLARS. That means a
    # 90%-vol name contributes ~7x the portfolio variance of a 12%-vol name, so
    # the book's realized risk is dominated by whichever handful of names happen
    # to be most volatile. Nobody decided that; it is an accident of the sizing.
    #
    # --vol-target: every name gets the same RISK. w_i proportional to 1/vol_i.
    # Justification (measured 2026-07-12, not assumed):
    #   log(trailing_vol_20d) -> log(forward_vol_5d), OOS Spearman +0.623
    #   the full 118-feature XGB model                              +0.706
    # Forward vol is the ONE thing this system predicts well -- direction sits at
    # 0.51 in and out of sample. Use the good forecast where it actually pays.
    #
    # WHY TRAILING VOL AND NOT THE XGB MODEL: this script moves real money. The
    # model is only +0.08 Spearman better, and it drags in xgboost, the feature
    # builder, and ~400 API calls -- any of which can fail at 3am and silently
    # emit a broken book. Trailing realized vol is one line of SQL, cannot fail,
    # and captures ~90% of the benefit. Defensive by design (see RULE 1 above).
    long_gross=a.gross; short_gross=a.gross*a.short_frac

    def _vols(names):
        """Trailing realized vol per ticker from daily_prices. Returns {} on any
        problem -- the caller then falls back to equal weight rather than sizing
        on a number it could not compute."""
        if not names: return {}
        c2=ro(prices_db)
        try:
            out={}
            for tk in names:
                r=Q(c2,"SELECT adj_close FROM daily_prices WHERE ticker=? "
                       "ORDER BY date DESC LIMIT ?",(tk,a.vol_window+1))
                if len(r)<a.vol_window//2: continue
                px=[float(x[0]) for x in r][::-1]
                rets=[(px[i]/px[i-1]-1.0) for i in range(1,len(px)) if px[i-1]>0]
                if len(rets)<5: continue
                m=sum(rets)/len(rets)
                sd=(sum((x-m)**2 for x in rets)/(len(rets)-1))**0.5
                if sd>1e-6: out[tk]=sd*(252**0.5)          # annualised
            return out
        finally: c2.close()

    def _weights(names, gross):
        """Equal weight, or 1/vol with the cap applied and the excess REDISTRIBUTED.

        Capping without redistributing silently under-deploys: if 5 names hit a 5%
        cap you lose that gross entirely and the book is smaller than you asked for.
        Iterate until nothing else binds.
        """
        if not names: return {}
        tks=[t for t,_ in names]
        if not a.vol_target:
            w=min(a.max_weight, gross/len(tks))
            return {t:w for t in tks}
        vols=_vols(tks)
        if len(vols)<len(tks)*0.8:
            print("  [WARN] vol available for only %d/%d names -> EQUAL WEIGHT fallback"
                  %(len(vols),len(tks)))
            w=min(a.max_weight, gross/len(tks))
            return {t:w for t in tks}
        med=sorted(vols.values())[len(vols)//2]
        inv={t: 1.0/max(vols.get(t,med),0.05) for t in tks}      # floor vol at 5%
        raw={t: gross*inv[t]/sum(inv.values()) for t in tks}
        # iterative cap + redistribute
        capped={}; free=dict(raw)
        for _ in range(20):
            over=[t for t,w in free.items() if w>a.max_weight]
            if not over: break
            for t in over:
                capped[t]=a.max_weight; free.pop(t)
            rem=gross-a.max_weight*len(capped)
            if rem<=0 or not free: break
            fi=sum(inv[t] for t in free)
            free={t: rem*inv[t]/fi for t in free}
        out={**capped, **free}
        lo=min(out.values()); hi=max(out.values())
        print("  [VOL-TARGET] vols %.0f%%-%.0f%% (median %.0f%%) -> weights %.2f%%-%.2f%%"
              %(100*min(vols.values()),100*max(vols.values()),100*med,100*lo,100*hi))
        return out

    W_long  = _weights(longs,  long_gross)
    W_short = _weights(shorts, short_gross)
    wl = (sum(W_long.values())/len(W_long))   if W_long  else 0
    ws = (sum(W_short.values())/len(W_short)) if W_short else 0
    # renormalize if cap binds
    actual_long=wl*len(longs); actual_short=ws*len(shorts)

    print("\n"+"-"*78)
    print("  CONSTRUCTION: long-tilted | long gross %.0f%% of capital | short %.0f%% of long (%s)"
          %(100*actual_long,100*a.short_frac,"long-only" if a.short_frac==0 else ("dollar-neutral" if abs(a.short_frac-1)<1e-9 else "tilted")))
    print("-"*78)

    def emit(side, items, W):
        """W is a PER-NAME dict {ticker: weight}. It used to be a scalar, and the
        first cut of --vol-target computed the dict and then averaged it back into
        a scalar before emitting -- so the vol weights were calculated, printed in
        the header, and then thrown away. Every name still got equal dollars. The
        book would have been identical with and without the flag.
        (A float is still accepted so nothing else that calls emit() breaks.)
        """
        if not isinstance(W, dict):
            W = {tk: float(W) for tk, _ in items}
        gross = sum(W.get(tk, 0.0) for tk, _ in items)
        mode = "1/vol RISK-parity" if a.vol_target else "equal DOLLARS"
        print("\n  %s — %d names, %.0f%% gross (%s):"%(side, len(items), 100*gross, mode))
        hdr="    %-8s %10s"%("ticker","weight")
        if a.capital: hdr+=" %12s %10s"%("$ alloc","shares")
        if a.vol_target: hdr+=" %8s"%"vol"
        print(hdr)
        vmap = _vols([t for t,_ in items]) if a.vol_target else {}
        rowlog=[]
        for tk,v in items:
            w = W.get(tk, 0.0)
            line="    %-8s %9.2f%%"%(tk,100*w)
            if a.capital:
                dollars=w*a.capital; px=last_px[tk]; sh=int(round(dollars/px)) if px>0 else 0
                line+=" %12s %10d"%("$%.0f"%dollars, sh)
                rowlog.append((tk,side,round(w,5),round(dollars,2),sh,round(px,2)))
            else:
                rowlog.append((tk,side,round(w,5),"","",round(last_px[tk],2)))
            if a.vol_target and tk in vmap:
                line+=" %7.0f%%"%(100*vmap[tk])
            print(line+"   (DTC %.1f)"%v)
        return rowlog

    log=[]
    log+=emit("LONG  (low DTC)", longs, W_long)
    if a.short_frac>0 and shorts:
        log+=emit("SHORT (high DTC)", shorts, W_short)
    else:
        print("\n  SHORT leg: OFF (long-only).")

    print("\n"+LINE+"\nNOTES THAT TRAVEL WITH THESE POSITIONS\n"+LINE)
    print("  * AS-OF: ranked on the %s settlement (days_to_cover). Hold horizon ~40 trading days."%latest)
    print("  * Through-cycle Sharpe ~1.0-1.2 NET is the planning basis. The recent ~2.0 is a")
    print("    favorable-regime reading and will revert -- do NOT size as if 2.0 is the expectation.")
    if a.short_frac>0:
        print("  * SHORT LEG borrow cost: high-DTC names are hard/expensive to borrow; the backtest did")
        print("    NOT include borrow fees. Your realized short-leg return will be LOWER than tested.")
        print("    The short leg is also only ~29%% of edge and not individually significant.")
    print("  * Survivorship: the long leg is survivorship-safe (low-DTC names rarely delist); apply a")
    print("    ~3%%/yr haircut to the short-leg portion only.")
    print("  * Share counts use the latest adj_close (sizing approximation); actual fills at market.")
    print("  * Not investment advice. Sizing and the decision to deploy are yours.")

    if a.log_ledger and a.capital:
        led=os.path.join(a.root,"si_live_ledger.csv"); new=not os.path.isfile(led)
        with open(led,"a",newline="") as f:
            w=csv.writer(f)
            if new: w.writerow(["generated_on","settlement","ticker","side","weight","usd","shares","ref_px","hold_days"])
            for tk,side,wt,usd,sh,px in log:
                w.writerow([today.isoformat(),latest,tk,side,wt,usd,sh,px,40])
        print("\n  ledger appended: %s (record of what you intended to enter)"%led)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("\ninterrupted.")
    except Exception:
        import traceback; print("\n[UNEXPECTED ERROR] paste back:"); traceback.print_exc()
