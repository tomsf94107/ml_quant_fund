#!/usr/bin/env python3
"""
analysis/sector_neutral_combine.py -- is the momentum/SI correlation a SECTOR bet?

WHERE THIS CAME FROM
  Momentum and SI, both beta-hedged, correlate +0.610 -- just over the 0.60 gate,
  so the combiner was rejected. But the overlap is not random: 13 of the top 40 in
  each book are the SAME NAMES, and every one is a semi / AI-infra stock (AMD, ARM,
  ASML, CIEN, COHR, DELL, FLEX, GFS, INTC, MRVL, MU, NOK).

  So the correlation is not beta (already stripped). It is a shared SECTOR bet.

WHAT IS AND IS NOT ALREADY KNOWN (RULE 1 -- checked the docs first)
  A8              : sector-neutral TESTED + REJECTED May 28. "DO NOT REVISIT." (:520)
  Direction model : sector-neutral destroys the edge (:487). That model is dead anyway.
  SHORT INTEREST  : ALREADY PASSES -- 80% of the IC survives 47-bucket
                    sector-neutralisation, NW-t -4.07. Stock-specific, proven.
  MOMENTUM        : NEVER TESTED. This is the open question.

  The docs' own hint (:98): momentum is "semi/memory CONCENTRATED so sector caps
  required." That is a warning, not a result.

THE TEST
  Demean each signal WITHIN its sector bucket, so the book picks the best names
  INSIDE each sector rather than loading up on whichever sector is hot. Rebuild both
  books, beta-hedge both, re-measure Sharpe and the correlation.

PRE-REGISTERED (written before the numbers exist)
  1. Does MOMENTUM survive?  sector-neutral Sharpe >= 60% of raw -> survives.
                             < 60% -> momentum's edge IS the semi tilt.
  2. Do they decorrelate?    hedged corr < 0.30 -> COMBINER UNBLOCKS.
                             0.30-0.60 -> marginal.  > 0.60 -> same bet, stop.
  3. Does the 50/50 beat both singles? Grinold: 2 uncorrelated -> ~1.41x IR.

  BOTH conditions must hold. A momentum book that survives but stays correlated is
  no use; a decorrelated pair where one leg is dead is no use either.
"""
import sys, sqlite3, math, csv
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
COST_BPS, LAG_BD = 10.0, 8
MOM_HOLD, MOM_DEC = 20, 0.10
SI_HOLD,  SI_Q    = 40, 0.20

sec = {r["ticker"].upper(): r.get("bucket","?")
       for r in csv.DictReader(open(ROOT/"tickers_metadata.csv"))}

px = pd.read_sql("SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL",
                 sqlite3.connect(ROOT/"prices.db"))
px["date"]=pd.to_datetime(px["date"]); px["ticker"]=px["ticker"].str.upper()
close = px.pivot_table(index="date",columns="ticker",values="adj_close").sort_index()
close = close[close.index >= "2021-01-01"]
ret, tdays = close.pct_change(fill_method=None), close.index
mkt = ret.mean(axis=1)

def sector_demean(row):
    """Rank WITHIN sector, not across the universe. A name is 'good' if it beats its
    OWN sector's average -- so the book cannot express a view on semis vs utilities,
    only on which semi beats which semi."""
    s = pd.Series({t: sec.get(t,"?") for t in row.index})
    return row - row.groupby(s).transform("mean")

mom_raw = close.pct_change(126,fill_method=None) - close.pct_change(21,fill_method=None)

def book(pick, hold, dates):
    daily=pd.Series(0.0,index=tdays); active=pd.Series(0.0,index=tdays)
    for d in dates:
        i=tdays.searchsorted(d)
        if i>=len(tdays)-hold-1: continue
        d=tdays[i]; names=pick(d)
        if names is None or len(names)<5: continue
        w=pd.Series(1.0/len(names),index=names)
        win=tdays[i+1:i+1+hold]
        r=(ret.loc[win,names]*w.values).sum(axis=1)
        r.iloc[0]-=(COST_BPS/1e4)*2.0
        daily.loc[win]+=r.values; active.loc[win]+=1.0
    live=active>0
    return (daily[live]/active[live]).dropna()

def mom_pick(d, neutral):
    row = mom_raw.loc[d].dropna()
    if len(row)<30: return None
    if neutral: row = sector_demean(row)
    k=max(1,int(len(row)*MOM_DEC))
    return list(row.sort_values(ascending=False).head(k).index)

si = pd.read_sql("SELECT ticker,settlement_date,days_to_cover FROM short_interest "
                 "WHERE days_to_cover IS NOT NULL", sqlite3.connect(ROOT/"short_interest.db"))
si["settlement_date"]=pd.to_datetime(si["settlement_date"])
si=si[si["days_to_cover"]<=50.0]
si["entry"]=si["settlement_date"]+pd.tseries.offsets.BDay(LAG_BD)
si["ticker"]=si["ticker"].str.upper()
si_map={d:g for d,g in si.groupby("entry")}

def si_pick(d, neutral):
    cand=[k for k in si_map if abs((k-d).days)<=5]
    if not cand: return None
    g=si_map[cand[0]]; g=g[g.ticker.isin(close.columns)]
    if len(g)<30: return None
    s=pd.Series(-g["days_to_cover"].values, index=g["ticker"].values)   # low DTC = good
    if neutral: s = sector_demean(s)
    k=max(1,int(len(s)*SI_Q))
    return list(s.sort_values(ascending=False).head(k).index)

def hedge(r,m):
    b=np.polyfit(m,r,1)[0]; return r-b*m, b
def stats(r):
    sd=r.std(); sh=(r.mean()/sd)*math.sqrt(252) if sd>0 else np.nan
    c=(1+r).cumprod(); return sh, float((c/c.cummax()-1).min())

out={}
for neutral in (False, True):
    rm = book(lambda d: mom_pick(d, neutral), MOM_HOLD, tdays[::MOM_HOLD])
    rs = book(lambda d: si_pick(d, neutral),  SI_HOLD,  sorted(si_map.keys()))
    df = pd.DataFrame({"mom":rm,"si":rs,"mkt":mkt}).dropna()
    hm,_ = hedge(df["mom"], df["mkt"]); hs,_ = hedge(df["si"], df["mkt"])
    out[neutral] = dict(hm=hm, hs=hs, c=hm.corr(hs), n=len(df))

print("="*78)
print("  IS THE MOMENTUM/SI CORRELATION A SECTOR BET?")
print("="*78)
print(f"  {'':30}{'RAW':>16}{'SECTOR-NEUTRAL':>18}")
for lbl, key in [("MOMENTUM Sharpe (hedged)","hm"), ("SHORT INT Sharpe (hedged)","hs")]:
    a,_ = stats(out[False][key]); b,_ = stats(out[True][key])
    print(f"  {lbl:30}{a:>+16.2f}{b:>+18.2f}")
for lbl, key in [("MOMENTUM maxDD","hm"), ("SHORT INT maxDD","hs")]:
    _,a = stats(out[False][key]); _,b = stats(out[True][key])
    print(f"  {lbl:30}{100*a:>15.1f}%{100*b:>17.1f}%")
print(f"  {'CORRELATION (hedged)':30}{out[False]['c']:>+16.3f}{out[True]['c']:>+18.3f}")

nc = out[True]["c"]
m_raw,_  = stats(out[False]["hm"]); m_neu,_ = stats(out[True]["hm"])
s_neu,_  = stats(out[True]["hs"])
comb = 0.5*out[True]["hm"] + 0.5*out[True]["hs"]
sh_c,dd_c = stats(comb); best = max(m_neu, s_neu)

print()
print(f"  50/50 COMBINED (sector-neutral, hedged): Sharpe {sh_c:+.2f}  maxDD {100*dd_c:.1f}%")
print(f"  best single: {best:+.2f}   lift: {sh_c-best:+.2f}")
print()
print("="*78)
surv = (m_neu >= 0.6*m_raw) if m_raw > 0 else False
if not surv:
    print(f"  >> MOMENTUM DIES under sector-neutralisation ({m_raw:+.2f} -> {m_neu:+.2f}).")
    print("     Its edge IS the semi/memory tilt -- it is a SECTOR BET, not stock")
    print("     selection. That is a real finding about what you own: a levered")
    print("     semiconductor position. No combiner. SI (which DOES survive, 80% of")
    print("     IC retained) is the only stock-specific signal in the system.")
elif nc < 0.30 and sh_c > best + 0.10:
    print(f"  >> COMBINER UNBLOCKS. Sector-neutral, both survive, correlation {nc:+.3f}")
    print(f"     < 0.30, and the 50/50 beats the best single by {sh_c-best:+.2f}.")
    print("     The C1 gate -- blocked since June 1 for lack of a 2nd return alpha --")
    print("     is MET. Build the combiner.")
else:
    print(f"  >> STILL NOT SEPARABLE. Momentum survives ({m_neu:+.2f}) but correlation")
    print(f"     is {nc:+.3f} and the lift is {sh_c-best:+.2f}. Sector was not the whole")
    print("     story. No combiner.")
print("="*78)
print("  In-sample, one bull regime, survivor-tilted. Sector buckets are your own")
print("  47-way labels, not GICS.")
