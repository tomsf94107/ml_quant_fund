#!/usr/bin/env python3
"""
pead_lookahead_test.py -- Is the PEAD brick trading on information that was not public?

THE PROBLEM
  earnings_surprises.report_date is a FISCAL PERIOD END, not an announcement date.
  Proof: 75% of rows land exactly on 03-31/06-30/09-30/12-31; 35% fall on a weekend;
  MU's row reads 2026-05-31 but MU announced ~2026-06-25.
  The SUE is NOT KNOWABLE until the announcement (typically fiscal_end + 21..45d).

THE TEST (event study, per-date IC + Newey-West -- the RULE-1 standard)
  For each event, split the forward path into three windows measured from fiscal end:
    PRE   fe+2  .. fe+20   -> signal is NOT public. IC here can ONLY be look-ahead.
    ANN   fe+21 .. fe+45   -> the announcement itself lands here (the jump).
    POST  fe+46 .. fe+85   -> genuine post-earnings drift.

READ
  PRE IC ~ 0 (not significant)  -> no exploitable look-ahead. Hypothesis dies.
  PRE IC strongly positive      -> the backtest earns from a signal that did not exist.
                                   PEAD must be re-keyed to real announcement dates.

Honest n = number of EVENT DATES (not stock-event rows). Overlap -> Newey-West.
"""
import sqlite3, argparse, numpy as np
from collections import defaultdict
from datetime import datetime

def nd(x):
    if x is None: return None
    s = str(x)[:10]
    try: return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception: return None

def nw_t(x, lag):
    x = np.asarray(x, float); n = len(x)
    if n < 5: return float("nan")
    mu = x.mean(); e = x - mu
    g0 = (e @ e) / n; s = g0
    for L in range(1, min(lag, n - 1) + 1):
        gL = (e[L:] @ e[:-L]) / n
        s += 2.0 * (1.0 - L / (lag + 1.0)) * gL
    if s <= 0: return float("nan")
    return mu / np.sqrt(s / n)

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--min-names", type=int, default=8)
a = ap.parse_args()

# ---- prices: ticker -> sorted [(date, px)] + index for positional offsets
pc = sqlite3.connect(f"{a.root}/prices.db")
px = defaultdict(list)
for tk, d, c in pc.execute(
        "SELECT ticker, date, adj_close FROM daily_prices WHERE adj_close IS NOT NULL"):
    dd = nd(d)
    if dd: px[tk].append((dd, float(c)))
pc.close()
for tk in px: px[tk].sort()
pos = {tk: {d: i for i, (d, _) in enumerate(v)} for tk, v in px.items()}

def fwd_ret(tk, fe, i0, i1):
    """Return over TRADING-DAY offsets [i0, i1] measured from the first session on/after fe."""
    ser = px.get(tk)
    if not ser: return None
    lo, hi = 0, len(ser) - 1
    while lo < hi:                       # first index with date >= fe
        m = (lo + hi) // 2
        if ser[m][0] < fe: lo = m + 1
        else: hi = m
    if ser[lo][0] < fe: return None
    s, e = lo + i0, lo + i1
    if e >= len(ser) or s >= len(ser): return None
    p0, p1 = ser[s][1], ser[e][1]
    if p0 <= 0: return None
    return p1 / p0 - 1.0

# ---- SUE, keyed to fiscal end, PIT trailing std (mirrors two_brick_book)
ec = sqlite3.connect(f"{a.root}/earnings.db")
rows = ec.execute("SELECT ticker, report_date, eps_actual, eps_estimate "
                  "FROM earnings_surprises WHERE report_date IS NOT NULL "
                  "AND eps_actual IS NOT NULL AND eps_estimate IS NOT NULL").fetchall()
ec.close()

by = defaultdict(list)
for tk, rd, act, est in rows:
    fe = nd(rd)
    if fe is None: continue
    try: raw = float(act) - float(est)
    except Exception: continue
    by[tk].append((fe, raw))

events = []          # (fiscal_end, ticker, sue)
for tk, lst in by.items():
    lst.sort(); prior = []
    for fe, raw in lst:
        if len(prior) >= 4:
            sd = np.std(prior, ddof=1)
            if sd > 1e-12: events.append((fe, tk, raw / sd))
        prior.append(raw)

WINDOWS = [("PRE   fe+2..fe+20  (NOT public -- look-ahead only)", 2, 20),
           ("ANN   fe+21..fe+45 (the announcement jump)",         21, 45),
           ("POST  fe+46..fe+85 (genuine drift)",                 46, 85)]

by_date = defaultdict(list)
for fe, tk, sue in events: by_date[fe].append((tk, sue))

print("=" * 78)
print("PEAD LOOK-AHEAD TEST -- is the signal earning before it was public?")
print("=" * 78)
print(f"  events={len(events)}  event-dates={len(by_date)}  tickers={len(by)}")
print(f"  NOTE: report_date is a FISCAL PERIOD END. Announcements land ~fe+21..fe+45.")
print()

for label, i0, i1 in WINDOWS:
    ics, ns = [], 0
    for fe in sorted(by_date):
        s, r = [], []
        for tk, sue in by_date[fe]:
            rr = fwd_ret(tk, fe, i0, i1)
            if rr is None: continue
            s.append(sue); r.append(rr)
        if len(s) >= a.min_names:
            ic = spearman(s, r)
            if np.isfinite(ic): ics.append(ic); ns += len(s)
    if len(ics) < 5:
        print(f"  {label:52s}  insufficient dates (n={len(ics)})"); continue
    ics = np.asarray(ics)
    t   = nw_t(ics, lag=4)
    print(f"  {label:52s}")
    print(f"      mean IC = {ics.mean():+.4f}   NW-t = {t:+.2f}   "
          f"dates = {len(ics)}   rows = {ns}   right-sign = {100*np.mean(ics>0):.0f}%")
    print()

print("=" * 78)
print("  READ:")
print("   PRE IC not significant  -> no exploitable look-ahead. Hypothesis dies.")
print("   PRE IC strongly positive-> the book earns from a signal that did not yet exist;")
print("                              PEAD must be re-keyed to real announcement dates.")
print("=" * 78)
