#!/usr/bin/env python3
"""validate_pead_v2.py -- PEAD on REAL announcement dates. First honest test."""
import sqlite3, argparse, math, numpy as np
from collections import defaultdict
from datetime import datetime, date

def nd(x):
    try: return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()
    except Exception: return None

def nw_se(x, lag):
    x = np.asarray(x, float); n = len(x)
    if n < 5: return float("nan")
    e = x - x.mean(); s = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - k / (lag + 1.0)) * ((e[k:] @ e[:-k]) / n)
    return math.sqrt(s / n) if s > 0 else float("nan")

def spear(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--hold", type=int, default=40)
ap.add_argument("--entry", type=int, default=2)
ap.add_argument("--min-names", type=int, default=10)
ap.add_argument("--nulls", type=int, default=200)
a = ap.parse_args()
TODAY = date.today()

pc = sqlite3.connect(f"{a.root}/prices.db")
px = defaultdict(list)
for tk, d, c in pc.execute("SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL"):
    dd = nd(d)
    if dd: px[tk].append((dd, float(c)))
pc.close()
for tk in px: px[tk].sort()

def fwd(tk, d, entry, hold):
    ser = px.get(tk)
    if not ser: return None
    lo, hi = 0, len(ser) - 1
    while lo < hi:
        m = (lo + hi) // 2
        if ser[m][0] < d: lo = m + 1
        else: hi = m
    if not ser or ser[lo][0] < d: return None
    s, e = lo + entry, lo + entry + hold
    if e >= len(ser): return None
    p0 = ser[s][1]
    return ser[e][1] / p0 - 1.0 if p0 > 0 else None

ec = sqlite3.connect(f"{a.root}/earnings.db")
rows = ec.execute("""SELECT ticker, announce_date, eps_actual, eps_estimate
                     FROM earnings_events WHERE eps_actual IS NOT NULL
                     AND eps_estimate IS NOT NULL ORDER BY ticker, announce_date""").fetchall()
ec.close()

by = defaultdict(list)
for tk, ad, act, est in rows:
    d = nd(ad)
    if d and d <= TODAY: by[tk].append((d, float(act) - float(est)))

events = []
for tk, lst in by.items():
    lst.sort(); prior = []
    for d, raw in lst:
        if len(prior) >= 4:
            sd = np.std(prior, ddof=1)
            if sd > 1e-12: events.append((d, tk, raw / sd))
        prior.append(raw)

by_date = defaultdict(list)
for d, tk, s in events: by_date[d].append((tk, s))

ics, dts, ns = [], [], 0
rng = np.random.default_rng(7)
null_means = np.zeros(a.nulls)
for d in sorted(by_date):
    S, R = [], []
    for tk, s in by_date[d]:
        r = fwd(tk, d, a.entry, a.hold)
        if r is None or abs(r) > 2.0: continue
        S.append(s); R.append(r)
    if len(S) < a.min_names: continue
    ic = spear(S, R)
    if not np.isfinite(ic): continue
    ics.append(ic); dts.append(d); ns += len(S)
    Sa = np.asarray(S)
    for k in range(a.nulls):
        null_means[k] += spear(rng.permutation(Sa), R)

ics = np.asarray(ics)
if len(ics) < 10:
    print(f"  insufficient dates ({len(ics)})"); raise SystemExit
null_means /= len(ics)
gaps = [(dts[i+1]-dts[i]).days for i in range(len(dts)-1)]
gap = max(1, int(np.median(gaps))) if gaps else 1
lag = max(1, int(math.ceil(a.hold * 1.4 / gap)))
se = nw_se(ics, lag); t = ics.mean() / se

print("=" * 78)
print(f"PEAD VALIDATOR v2 -- entry = announcement + {a.entry}d, hold {a.hold}d")
print("=" * 78)
print(f"  events={len(events):,}  tickers={len(by)}  dates={len(ics)}  stock-events={ns:,}")
print(f"  span {min(dts)} .. {max(dts)}   median gap={gap}d   NW lag={lag}")
print()
print(f"  mean IC       = {ics.mean():+.4f}")
print(f"  IC IR         = {ics.mean()/ics.std(ddof=1):+.3f}")
print(f"  naive t       = {ics.mean()/(ics.std(ddof=1)/math.sqrt(len(ics))):+.2f}")
print(f"  Newey-West t  = {t:+.2f}   <- THE honest number")
print(f"  right-sign    = {100*np.mean(ics>0):.0f}% of dates")
print()
print("  NULL CONTROL (shuffle SUE within each date):")
print(f"    real mean IC = {ics.mean():+.4f}")
print(f"    null mean IC = {null_means.mean():+.4f}  (sd {null_means.std():.4f})")
z = (ics.mean() - null_means.mean()) / null_means.std() if null_means.std() > 0 else float("nan")
p = float(np.mean(np.abs(null_means - null_means.mean()) >= abs(ics.mean() - null_means.mean())))
print(f"    z vs null    = {z:+.2f}   permutation p = {p:.3f}")
print()
print("  PER-YEAR mean IC:")
yr = defaultdict(list)
for d, ic in zip(dts, ics): yr[d.year].append(ic)
for y in sorted(yr):
    v = np.mean(yr[y])
    print(f"    {y}: {v:+.4f}  (n={len(yr[y]):3d})  {'#'*int(abs(v)*200)}")
print()
h = len(ics)//2
print("  OUT-OF-SAMPLE (first vs second half):")
for lbl, seg in [("first ", ics[:h]), ("second", ics[h:])]:
    print(f"    {lbl} half  mean IC={seg.mean():+.4f}  t={seg.mean()/nw_se(seg,lag):+.2f}  (n={len(seg)})")
print()
print("=" * 78)
ok = abs(t) >= 2.0 and abs(z) >= 2.0 and p < 0.05
print(f"  VERDICT: {'REAL BRICK' if ok else 'NOT ESTABLISHED'} (NW-t {t:+.2f}, null z {z:+.2f}, p {p:.3f})")
print("  Literature PEAD IC ~0.02-0.05. The old +0.20-0.24 was measured on FISCAL")
print("  PERIOD ENDS -- 15-27 days before the number was public.")
print("=" * 78)
