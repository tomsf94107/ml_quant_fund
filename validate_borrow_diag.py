#!/usr/bin/env python3
"""
validate_borrow_diag.py -- Diagnostics 2 & 3 for the borrow-fee IC test.

This does NOT change the verdict (all raw ICs failed |t|>=2 -> scanner-only).
It closes two loose ends the main test surfaced:

  DIAG 2 -- Is the fee_gt_5pct residual-vs-DTC IC (+2.69 NW-t at h=40) REAL or a
            residualization ARTIFACT?
    Method: PERMUTATION NULL ON THE RESIDUAL PATH. Shuffle forward returns
    (within date) BEFORE residualizing against DTC, run the full
    residualize->IC pipeline, repeat K times. Build the null distribution of the
    residual mean-IC. If the real +0.0318 sits far outside the null -> real.
    If the null is ALSO wide/shifted (residualization manufactures IC even on
    shuffled returns) -> artifact.

  DIAG 3 -- Were the h=20 NULL WARNs a real leak or just OVERLAP autocorrelation
            my within-date shuffle didn't break?
    Method: two nulls compared -- (a) within-date shuffle (what the main test
    did), (b) GLOBAL shuffle (permute the entire pooled return vector across all
    stock-dates, breaking BOTH within-date and cross-date structure). If (b)
    collapses to ~0 while (a) didn't, the WARN was overlap, not a pipeline leak.

READ-ONLY. borrow.db + prices.db + short_interest.db. No writes. No network.

RUN
  python validate_borrow_diag.py                       # both diags, defaults
  python validate_borrow_diag.py --k 500               # more permutations
"""

import argparse, os, sqlite3, math, datetime, sys
from collections import defaultdict
import numpy as np

LINE = "=" * 78
def banner(t): print("\n" + LINE + "\n" + t + "\n" + LINE)
def sub(t): print("\n" + "-" * 78 + "\n" + t + "\n" + "-" * 78)

def ro(p):
    return sqlite3.connect("file:" + os.path.abspath(p) + "?mode=ro&immutable=1", uri=True, timeout=30)
def Q(c, s, p=()): return c.execute(s, p).fetchall()

def nd(s):
    if s is None: return None
    try: return datetime.date.fromisoformat(str(s)[:10])
    except Exception: return None

def spearman(x, y):
    n = len(x)
    if n < 5: return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0: return None
    return float(np.corrcoef(rx, ry)[0, 1])

def newey_west_se_mean(x, lag):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 2: return None
    e = x - x.mean()
    gamma0 = float(e @ e) / n
    s = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * gk
    var_mean = s / n
    return math.sqrt(var_mean) if var_mean > 0 else None


def load_prices(prices_db):
    cp = ro(prices_db)
    try:
        rows = Q(cp, "SELECT ticker,date,adj_close FROM daily_prices WHERE adj_close IS NOT NULL")
    finally:
        cp.close()
    px = defaultdict(list)
    for tk, d, p in rows:
        do = nd(d)
        if do is None: continue
        try: pf = float(p)
        except Exception: continue
        if pf > 0: px[tk].append((do, pf))
    for tk in px: px[tk].sort()
    pos_of = {tk: {d: i for i, (d, _) in enumerate(lst)} for tk, lst in px.items()}
    return px, pos_of


def make_fwd(px, pos_of):
    def fwd(tk, d, h):
        lst = px.get(tk); idx = pos_of.get(tk)
        if not lst or not idx: return None
        i = None
        for off in range(0, 6):
            cc = d + datetime.timedelta(days=off)
            if cc in idx: i = idx[cc]; break
        if i is None: return None
        x = i + h
        if x >= len(lst): return None
        p0 = lst[i][1]
        return (lst[x][1] / p0 - 1.0) if p0 > 0 else None
    return fwd


def load_feature(borrow_db, feature):
    c = ro(borrow_db)
    try:
        rows = Q(c, f'SELECT ticker,asof_date,"{feature}" FROM borrow_features')
    finally:
        c.close()
    by_date = defaultdict(list)
    for tk, d, v in rows:
        do = nd(d)
        if do is None or v is None: continue
        try: fv = float(v)
        except Exception: continue
        by_date[do].append((tk.upper(), fv))
    return by_date


def load_dtc(si_db):
    c = ro(si_db)
    try:
        rows = Q(c, 'SELECT ticker,settlement_date,days_to_cover FROM short_interest')
    finally:
        c.close()
    dtc = {}
    for tk, d, v in rows:
        do = nd(d)
        if do is None or v is None: continue
        try: fv = float(v)
        except Exception: continue
        if fv > 50: continue
        dtc[(tk.upper(), do)] = fv
    return dtc


def build_panel(by_date, fwd, dtc, hold, min_names):
    """Per date: aligned arrays of (feature, dtc, fwd_return) for stocks that have all three."""
    panel = {}  # date -> (feat[], dcov[], ret[])
    for d in sorted(by_date):
        feat = []; dcov = []; ret = []
        for tk, v in by_date[d]:
            dd = dtc.get((tk, d))
            if dd is None:
                continue
            r = fwd(tk, d, hold)
            if r is None:
                continue
            feat.append(v); dcov.append(dd); ret.append(r)
        if len(feat) >= min_names:
            panel[d] = (np.array(feat, float), np.array(dcov, float), np.array(ret, float))
    return panel


def resid_ic_meanic(panel, ret_override=None):
    """Mean per-date residual IC: residualize feature ~ 1+dtc, IC residual vs return.
    If ret_override is a dict date->ret array (shuffled), use that instead."""
    ics = []
    for d, (feat, dcov, ret) in panel.items():
        r = ret if ret_override is None else ret_override[d]
        if dcov.std() == 0:
            resid = feat - feat.mean()
        else:
            A = np.vstack([np.ones_like(dcov), dcov]).T
            coef, *_ = np.linalg.lstsq(A, feat, rcond=None)
            resid = feat - A @ coef
        ic = spearman(resid, r)
        if ic is not None:
            ics.append(ic)
    return float(np.mean(ics)) if ics else None, len(ics)


def diag2_residual_permutation(panel, real_resid_meanic, hold, k, seed=42):
    sub(f"DIAG 2: permutation null on the RESIDUAL path (fee_gt_5pct, h={hold}, K={k})")
    rng = np.random.default_rng(seed)
    null_means = []
    for _ in range(k):
        ret_shuf = {}
        for d, (feat, dcov, ret) in panel.items():
            rr = ret.copy()
            rng.shuffle(rr)   # within-date shuffle of returns BEFORE residualizing
            ret_shuf[d] = rr
        m, _ = resid_ic_meanic(panel, ret_override=ret_shuf)
        if m is not None:
            null_means.append(m)
    null_means = np.array(null_means)
    mu = null_means.mean(); sd = null_means.std(ddof=1)
    # where does the REAL residual mean-IC fall in this null?
    z = (real_resid_meanic - mu) / sd if sd > 0 else float("nan")
    p_two = float(np.mean(np.abs(null_means - mu) >= abs(real_resid_meanic - mu)))
    print(f"  real residual mean-IC      = {real_resid_meanic:+.4f}")
    print(f"  null residual mean-IC dist = mean {mu:+.4f}, std {sd:.4f}  (K={len(null_means)})")
    print(f"  null 2.5/97.5 pct          = [{np.percentile(null_means,2.5):+.4f}, {np.percentile(null_means,97.5):+.4f}]")
    print(f"  real vs null: z = {z:+.2f},  permutation p(two-sided) = {p_two:.3f}")
    if abs(z) >= 2.5 and p_two < 0.05:
        print("  -> REAL: the residual IC sits far outside its own shuffled null.")
        print("     (Not a residualization artifact -- worth a closer look, though raw IC still failed.)")
    else:
        print("  -> ARTIFACT / NOT ROBUST: the residual IC is within (or near) its shuffled null.")
        print("     Residualizing against DTC manufactures comparable IC on RANDOM returns,")
        print("     so the +2.69 is a residualization artifact, not tradeable signal.")


def diag3_null_comparison(by_date, fwd, hold, min_names, k, seed=7):
    sub(f"DIAG 3: within-date vs GLOBAL shuffle null (h={hold})")
    # Build simple panel: date -> (feat[], ret[])
    P = {}
    all_ret = []
    for d in sorted(by_date):
        feat = []; ret = []
        for tk, v in by_date[d]:
            r = fwd(tk, d, hold)
            if r is not None:
                feat.append(v); ret.append(r)
        if len(feat) >= min_names:
            P[d] = (np.array(feat, float), np.array(ret, float))
            all_ret.extend(ret)
    if len(P) < 6:
        print("  [SKIP] insufficient dates"); return
    all_ret = np.array(all_ret)

    # real IC
    def mean_ic(ret_map):
        ics = []
        for d, (feat, ret) in P.items():
            r = ret if ret_map is None else ret_map[d]
            ic = spearman(feat, r)
            if ic is not None: ics.append(ic)
        return float(np.mean(ics)) if ics else None

    real = mean_ic(None)

    rng = np.random.default_rng(seed)
    # (a) within-date shuffle
    a_means = []
    for _ in range(k):
        rm = {}
        for d, (feat, ret) in P.items():
            rr = ret.copy(); rng.shuffle(rr); rm[d] = rr
        m = mean_ic(rm)
        if m is not None: a_means.append(m)
    a_means = np.array(a_means)
    # (b) GLOBAL shuffle: permute the entire pooled return pool, redeal to dates
    b_means = []
    sizes = [(d, len(P[d][1])) for d in P]
    for _ in range(k):
        pool = all_ret.copy(); rng.shuffle(pool)
        rm = {}; off = 0
        for d, n in sizes:
            rm[d] = pool[off:off+n]; off += n
        m = mean_ic(rm)
        if m is not None: b_means.append(m)
    b_means = np.array(b_means)

    print(f"  real mean-IC                 = {real:+.4f}")
    print(f"  (a) within-date null  mean   = {a_means.mean():+.4f}  std {a_means.std(ddof=1):.4f}  "
          f"95% [{np.percentile(a_means,2.5):+.4f},{np.percentile(a_means,97.5):+.4f}]")
    print(f"  (b) GLOBAL shuffle null mean = {b_means.mean():+.4f}  std {b_means.std(ddof=1):.4f}  "
          f"95% [{np.percentile(b_means,2.5):+.4f},{np.percentile(b_means,97.5):+.4f}]")
    if abs(b_means.mean()) < abs(a_means.mean()) * 0.5 or abs(b_means.mean()) < 0.002:
        print("  -> The GLOBAL null is tighter/centered on 0 => the h=20 WARN was OVERLAP")
        print("     autocorrelation (within-date shuffle can't break cross-date structure),")
        print("     NOT a pipeline leak. Methodology caveat, not a data problem.")
    else:
        print("  -> Both nulls behave similarly; WARN not explained by overlap alone -- inspect.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--borrow-db", default=None)
    ap.add_argument("--prices-db", default=None)
    ap.add_argument("--si-db", default=None)
    ap.add_argument("--k", type=int, default=300, help="permutations (default 300)")
    ap.add_argument("--min-names", type=int, default=15)
    a = ap.parse_args(); a.root = os.path.expanduser(a.root)

    borrow_db = a.borrow_db or os.path.join(a.root, "borrow.db")
    prices_db = a.prices_db or os.path.join(a.root, "prices.db")
    si_db     = a.si_db or os.path.join(a.root, "short_interest.db")
    for p, nm in [(borrow_db, "borrow.db"), (prices_db, "prices.db"), (si_db, "short_interest.db")]:
        if not os.path.isfile(p):
            print(f"[STOP] {nm} not found at {p}"); return

    print("loading prices ...")
    px, pos_of = load_prices(prices_db)
    fwd = make_fwd(px, pos_of)
    dtc = load_dtc(si_db)
    print(f"prices: {len(px)} tickers | DTC: {len(dtc)} (ticker,date) points")

    # ---- DIAG 2: fee_gt_5pct residual anomaly at h=40 ----
    banner("DIAG 2 -- fee_gt_5pct residual anomaly (is +2.69 NW-t real or artifact?)")
    bd = load_feature(borrow_db, "fee_gt_5pct")
    panel40 = build_panel(bd, fwd, dtc, 40, a.min_names)
    real_m, ndates = resid_ic_meanic(panel40)
    print(f"  rebuilt real residual mean-IC (h=40) = {real_m:+.4f} over {ndates} dates")
    diag2_residual_permutation(panel40, real_m, 40, a.k)

    # ---- DIAG 3: h=20 null WARN cases ----
    banner("DIAG 3 -- h=20 NULL WARN: overlap autocorrelation or real leak?")
    for feat in ["fee_change_1s", "fee_change_3s", "fee_gt_5pct"]:
        bd = load_feature(borrow_db, feat)
        print(f"\n>>> {feat} (h=20)")
        diag3_null_comparison(bd, fwd, 20, a.min_names, a.k)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
    except Exception:
        import traceback
        print("\n[UNEXPECTED ERROR] paste back:")
        traceback.print_exc()
