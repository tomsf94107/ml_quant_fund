#!/usr/bin/env python3
"""
validate_gex_regime.py -- GEX as a MARKET REGIME signal, not a stock picker.

WHY THIS TEST AND NOT THE LAST ONE
  validate_gex.py tested GEX cross-sectionally ("does high-gamma stock A move less
  than low-gamma stock B today?") because that is how the SI brick was validated.
  Nothing cleared |t|=2 -- but the beta strip retained 94% of the IC, so whatever
  it measures is NOT beta, and every sign was in the theoretically expected
  direction. That is the fingerprint of a real mechanism tested the wrong way.

  The GEX mechanism is not cross-sectional. It is AGGREGATE:
    market GEX < 0  -> dealers are net short gamma -> they BUY rallies and SELL
                       dips -> they AMPLIFY -> HIGH realized vol, momentum regime
    market GEX > 0  -> dealers are net long gamma  -> they SELL rallies and BUY
                       dips -> they DAMPEN  -> LOW realized vol, mean-revert regime

  Testing that per-date across stocks throws the mechanism away -- like asking
  whether VIX predicts which stock outperforms. The object is a TIME SERIES.

WHAT THIS TESTS
  1. Does aggregate GEX predict FORWARD REALIZED VOLATILITY of the market?
     (The core claim. Theory says yes, negatively.)
  2. Does the GEX regime flip the sign of MOMENTUM vs MEAN-REVERSION?
     (The tradeable claim: in negative gamma, momentum works; in positive gamma,
      reversal works. If true, this is a REGIME GATE for the whole system.)
  3. Null control: shuffle the GEX series; both effects must vanish.

  This is a small-n test (250 days) and it will say so. A t-stat on 250 daily
  observations of an autocorrelated series is not 250 independent observations --
  Newey-West is mandatory, and even then treat |t| just over 2 with suspicion.
"""
import sqlite3, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")

def nw_t(x, lag):
    x = np.asarray(x, float); n = len(x)
    if n < 3: return np.nan
    e = x - x.mean(); var = (e @ e) / n
    for k in range(1, min(lag, n-1)+1):
        var += 2.0 * (1.0 - k/(lag+1.0)) * ((e[k:] @ e[:-k]) / n)
    return x.mean()/np.sqrt(var/n) if var > 0 else np.nan

con = sqlite3.connect("accuracy.db")
# BUG THIS FIXES: SUM(net_gamma) across 393 tickers is dominated by mega-caps.
# NVDA's gamma is ~1e9; SENS's is ~1e3. The "market aggregate" was therefore just
# NVDA plus noise -- which is why it came back 0% negative-gamma days while 23% of
# individual ticker-days ARE negative. A size-weighted sum is not a market regime.
#
# The fix: z-score each ticker against ITS OWN history first, THEN average. That
# makes each name contribute equally and the object becomes "how unusual is dealer
# gamma across the market today" rather than "what is NVDA's gamma".
gx = pd.read_sql("SELECT ticker, date, net_gamma FROM options_greeks "
                 "WHERE net_gamma IS NOT NULL ORDER BY ticker, date", con)
gx["date"] = pd.to_datetime(gx["date"])
g = gx.groupby("ticker")["net_gamma"]
gx["z"] = ((gx["net_gamma"] - g.transform(lambda s: s.rolling(60, min_periods=20).mean()))
           / g.transform(lambda s: s.rolling(60, min_periods=20).std()))
gx = gx.replace([np.inf, -np.inf], np.nan).dropna(subset=["z"])
# equal-weight breadth measures, not a size-weighted sum
gx = (gx.groupby("date")
        .agg(gex=("z", "mean"),
             pct_neg=("net_gamma", lambda s: (s < 0).mean()),
             n=("z", "size"))
        .reset_index())
con.close()
con = sqlite3.connect("prices.db")
px = pd.read_sql("SELECT ticker, date, adj_close c FROM daily_prices "
                 "WHERE date >= '2025-06-01'", con)
con.close()

px["date"] = pd.to_datetime(px["date"])
px = px.sort_values(["ticker","date"])
px["ret"] = px.groupby("ticker")["c"].pct_change()
mkt = px.groupby("date")["ret"].mean().rename("mkt_ret").reset_index()

d = gx.merge(mkt, on="date", how="inner").sort_values("date").reset_index(drop=True)
d["gex_z"] = (d["gex"] - d["gex"].rolling(60, min_periods=20).mean()) \
             / d["gex"].rolling(60, min_periods=20).std()
d = d.dropna(subset=["gex_z"]).reset_index(drop=True)

print(f"  aggregate GEX: {len(d)} trading days, {d.date.min().date()} .. {d.date.max().date()}")
print(f"  mean % of names in negative gamma per day: {100*d.pct_neg.mean():.0f}%")
print(f"  range: {100*d.pct_neg.min():.0f}% .. {100*d.pct_neg.max():.0f}%\n")

print("="*80)
print("  TEST 1 -- does aggregate GEX predict FORWARD REALIZED VOL?")
print("="*80)
print(f"  {'fwd window':>12}{'corr(gex_z, vol)':>20}{'NW-t':>9}{'n':>6}   read")
for w in (1, 3, 5, 10, 20):
    d[f"fv{w}"] = d["mkt_ret"].rolling(w).std().shift(-w) * np.sqrt(252)
    s = d.dropna(subset=[f"fv{w}", "gex_z"])
    if len(s) < 40: continue
    r = s["gex_z"].corr(s[f"fv{w}"], method="spearman")
    prod = (s["gex_z"] - s["gex_z"].mean()) * (s[f"fv{w}"] - s[f"fv{w}"].mean())
    t = nw_t(prod.values, max(1, w))
    read = "NEG (theory: dampening)" if r < -0.15 and abs(t) > 2 else \
           ("weak/none" if abs(t) < 2 else "POS (against theory)")
    print(f"  {w:>10}d{r:>20.4f}{t:>+9.2f}{len(s):>6}   {read}")

print()
print("="*80)
print("  TEST 2 -- does the GEX regime FLIP momentum vs mean-reversion?")
print("="*80)
print("  In NEGATIVE gamma dealers amplify -> yesterday's move should CONTINUE.")
print("  In POSITIVE gamma dealers dampen  -> yesterday's move should REVERSE.\n")
d["prev"] = d["mkt_ret"].shift(1)
# regime = BREADTH of negative gamma across the universe, split at the median.
# The sign of a size-weighted sum is meaningless (see the bug note above).
d["neg_gamma"] = d["pct_neg"] > d["pct_neg"].median()
print(f"  {'regime':>16}{'days':>7}{'corr(prev, today)':>20}{'NW-t':>9}   read")
for lbl, mask in [("HIGH neg-gamma", d.neg_gamma), ("LOW neg-gamma", ~d.neg_gamma)]:
    s = d[mask].dropna(subset=["prev","mkt_ret"])
    if len(s) < 30:
        print(f"  {lbl:>16}{len(s):>7}   too few days"); continue
    r = s["prev"].corr(s["mkt_ret"])
    prod = (s["prev"]-s["prev"].mean())*(s["mkt_ret"]-s["mkt_ret"].mean())
    t = nw_t(prod.values, 3)
    read = ("MOMENTUM" if r > 0.10 else "REVERSAL" if r < -0.10 else "neither") \
           + ("" if abs(t) > 2 else " (not sig)")
    print(f"  {lbl:>16}{len(s):>7}{r:>20.4f}{t:>+9.2f}   {read}")

print()
print("="*80)
print("  NULL CONTROL -- shuffle GEX; every effect above must vanish")
print("="*80)
rng = np.random.default_rng(42)
dn = d.copy(); dn["gex_z"] = rng.permutation(dn["gex_z"].values)
s = dn.dropna(subset=["fv5","gex_z"])
print(f"  shuffled corr(gex_z, fwd 5d vol) = {s['gex_z'].corr(s['fv5'], method='spearman'):+.4f}"
      f"   (should be ~0)")
print()
print("  CAVEAT: 250 daily observations of an autocorrelated series is NOT 250")
print("  independent samples. Newey-West is applied. Treat |t| barely over 2 with")
print("  suspicion -- that is the bar three fake findings cleared in this fund.")
