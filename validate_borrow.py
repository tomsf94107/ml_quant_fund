#!/usr/bin/env python3
"""
validate_borrow.py -- Phase 2: IC-test the transformed borrow-fee features.

MIRRORS validate_si_v2.py EXACTLY (same honest methodology):
  - per-date cross-sectional Spearman IC  (n = DATES, not stock-date rows)
  - Newey-West HAC t-stat (overlap correction for the 40d window vs ~15d gap)
  - forward return via adj_close from prices.db, 0-5d forward-search on each date

ADDS TWO controls the base validator does not have:
  (1) NULL CONTROL  -- shuffle forward returns within each date; real IC must vanish.
  (2) SI-BRICK CONTROL -- the decisive test. Does each borrow feature add IC BEYOND
      days_to_cover (your validated SI brick)? Computed two honest ways:
        a) residual IC: on each date, regress the borrow feature on DTC across stocks,
           take the RESIDUAL (the part of the feature DTC does NOT explain), and IC
           that residual vs forward return. If this collapses to ~0, the feature is
           just re-expressing short interest -> no incremental value.
        b) correlation of the feature with DTC (how redundant they are).

READ-ONLY. Reads borrow.db (borrow_features) + prices.db (adj_close) +
short_interest.db (days_to_cover). No network. No writes.

RUN
  python validate_borrow.py                          # all features, h=40 and h=20
  python validate_borrow.py --feature fee_change_1s  # one feature
  python validate_borrow.py --hold 40                # one horizon
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

# features to test (all in borrow_features)
BORROW_FEATURES = ["fee_change_1s", "fee_change_3s", "avail_change_1s",
                   "fee_zscore_xsec", "fee_gt_5pct"]

# expected sign hint (for readability only; the test reports raw signed IC):
#  rising fee / rising DTC-like crowding -> LOWER forward return (like the SI brick) => NEGATIVE IC "works"
SIGN_HINT = {
    "fee_change_1s":   -1,   # rising borrow cost -> lower fwd return (hypothesis)
    "fee_change_3s":   -1,
    "avail_change_1s": +1,   # availability RISING (less scarce) -> higher fwd return (hypothesis)
    "fee_zscore_xsec": -1,   # unusually high fee -> lower fwd return
    "fee_gt_5pct":     -1,   # hard-to-borrow -> lower fwd return
}


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


def load_borrow_feature(borrow_db, feature):
    """date -> list of (ticker, value) for the given borrow feature."""
    c = ro(borrow_db)
    try:
        cols = [r[1] for r in Q(c, 'PRAGMA table_info("borrow_features")')]
        if feature not in cols:
            print(f"  [STOP] '{feature}' not in borrow_features. cols: {cols}")
            return None
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
    """(ticker, date) -> days_to_cover, for the SI-brick control."""
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
        if fv > 50: continue  # clip FINRA OTC junk (your rule)
        dtc[(tk.upper(), do)] = fv
    return dtc


def ic_series_for(by_date, fwd, hold, min_names, shuffle=False, rng=None):
    """Per-date IC time series. If shuffle=True, permute returns within each date (null)."""
    ic_series = []
    for d in sorted(by_date):
        sig = []; ret = []
        for tk, v in by_date[d]:
            r = fwd(tk, d, hold)
            if r is not None:
                sig.append(v); ret.append(r)
        if len(sig) >= min_names:
            if shuffle:
                ret = list(ret)
                rng.shuffle(ret)
            ic = spearman(sig, ret)
            if ic is not None:
                ic_series.append((d, ic, len(sig)))
    return ic_series


def residualize_against_dtc(by_date, dtc):
    """For each date, regress feature on DTC across stocks; return date -> list of
    (ticker, RESIDUAL). Only stocks with a DTC reading on that date are kept.
    The residual is the part of the borrow feature that DTC does NOT explain."""
    resid_by_date = defaultdict(list)
    for d in sorted(by_date):
        pairs = [(tk, v, dtc.get((tk, d))) for tk, v in by_date[d]]
        pairs = [(tk, v, dv) for tk, v, dv in pairs if dv is not None]
        if len(pairs) < 5:
            continue
        feat = np.array([v for _, v, _ in pairs], dtype=float)
        dcov = np.array([dv for _, _, dv in pairs], dtype=float)
        # OLS residual of feat ~ 1 + dcov
        if dcov.std() == 0:
            resid = feat - feat.mean()
        else:
            A = np.vstack([np.ones_like(dcov), dcov]).T
            coef, *_ = np.linalg.lstsq(A, feat, rcond=None)
            resid = feat - A @ coef
        for (tk, _, _), rv in zip(pairs, resid):
            resid_by_date[d].append((tk, float(rv)))
    return resid_by_date


def summarize(ic_series, hold, avg_gap_days, label):
    if len(ic_series) < 6:
        print(f"  [{label}] only {len(ic_series)} usable dates (need >=6) -- SKIP")
        return None
    ics = np.array([ic for _, ic, _ in ic_series])
    ns = [n for _, _, n in ic_series]
    dates = [d for d, _, _ in ic_series]
    N = len(ics); mean_ic = float(ics.mean()); std_ic = float(ics.std(ddof=1))
    ir = mean_ic / std_ic if std_ic > 0 else 0.0
    se_naive = std_ic / math.sqrt(N)
    t_naive = mean_ic / se_naive if se_naive > 0 else 0.0
    lag = max(1, int(math.ceil(hold / float(avg_gap_days))))
    se_nw = newey_west_se_mean(ics, lag)
    t_nw = mean_ic / se_nw if se_nw else 0.0
    print(f"  [{label}]  N={N} dates, avg {int(np.mean(ns))} stocks/date, {dates[0]}..{dates[-1]}")
    print(f"       mean IC={mean_ic:+.4f}  IR={ir:+.3f}  naive t={t_naive:+.2f}  Newey-West t={t_nw:+.2f}")
    return {"mean_ic": mean_ic, "t_nw": t_nw, "N": N, "ir": ir}


def run_feature(feature, by_date, fwd, dtc, hold, min_names, avg_gap_days=15, seed=42):
    banner(f"BORROW FEATURE: {feature}  (h={hold}, per-date IC)  sign-hint={SIGN_HINT.get(feature,0):+d}")

    # 1) RAW IC
    sub("1) RAW per-date IC (feature vs forward return)")
    raw = ic_series_for(by_date, fwd, hold, min_names)
    raw_s = summarize(raw, hold, avg_gap_days, "RAW")

    # 2) NULL CONTROL (shuffle returns within date)
    sub("2) NULL CONTROL (returns shuffled within each date -> IC must vanish)")
    rng = np.random.default_rng(seed)
    null = ic_series_for(by_date, fwd, hold, min_names, shuffle=True, rng=rng)
    null_s = summarize(null, hold, avg_gap_days, "NULL")
    if raw_s and null_s:
        if abs(null_s["mean_ic"]) < abs(raw_s["mean_ic"]) * 0.3 or abs(null_s["t_nw"]) < 1.0:
            print("       -> PASS: null IC collapses toward zero (raw IC is not a pipeline artifact)")
        else:
            print("       -> WARN: null IC not near zero -- possible leak/artifact, investigate")

    # 3) SI-BRICK CONTROL (residual IC beyond days_to_cover)
    sub("3) SI-BRICK CONTROL: residual IC after removing days_to_cover")
    resid_by_date = residualize_against_dtc(by_date, dtc)
    resid = ic_series_for(resid_by_date, fwd, hold, min_names)
    resid_s = summarize(resid, hold, avg_gap_days, "RESID vs DTC")

    # redundancy: correlation of feature with DTC across all overlapping (tk,date)
    fv = []; dv = []
    for d in by_date:
        for tk, v in by_date[d]:
            dd = dtc.get((tk, d))
            if dd is not None:
                fv.append(v); dv.append(dd)
    if len(fv) >= 5:
        corr = spearman(fv, dv)
        print(f"       feature vs DTC rank-corr (redundancy) = {corr:+.3f}"
              if corr is not None else "       feature vs DTC corr: n/a")

    # VERDICT
    sub("VERDICT")
    if not raw_s:
        print("  INSUFFICIENT DATA -- cannot judge")
        return
    raw_sig = abs(raw_s["t_nw"]) >= 2.0
    resid_sig = bool(resid_s) and abs(resid_s["t_nw"]) >= 2.0
    resid_keeps = bool(resid_s) and raw_s["mean_ic"] != 0 and \
                  abs(resid_s["mean_ic"]) >= 0.5 * abs(raw_s["mean_ic"])
    if raw_sig and resid_sig and resid_keeps:
        print(f"  ** INCREMENTAL EDGE ** raw NW-t={raw_s['t_nw']:+.2f}, residual-vs-DTC NW-t={resid_s['t_nw']:+.2f}")
        print("     The feature predicts returns AND keeps most of its IC after removing")
        print("     short interest -> it adds something DTC does not. Candidate to wire.")
    elif raw_sig and not resid_keeps:
        print(f"  REDUNDANT WITH SI BRICK: raw NW-t={raw_s['t_nw']:+.2f} but residual IC collapses")
        print(f"     (residual mean IC {resid_s['mean_ic']:+.4f} vs raw {raw_s['mean_ic']:+.4f}).")
        print("     The 'signal' is mostly short interest re-expressed -> scanner-only, do NOT wire.")
    elif not raw_sig:
        print(f"  NO STANDALONE EDGE: raw Newey-West t={raw_s['t_nw']:+.2f} (|t|<2). Scanner-only.")
    else:
        print("  MARGINAL / MIXED -- inspect the three blocks above before deciding.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--borrow-db", default=None)
    ap.add_argument("--prices-db", default=None)
    ap.add_argument("--si-db", default=None)
    ap.add_argument("--feature", default=None, help="one feature (default: all)")
    ap.add_argument("--hold", type=int, default=None, help="one horizon (default: 40 and 20)")
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
    print(f"prices loaded for {len(px)} tickers")
    dtc = load_dtc(si_db)
    print(f"days_to_cover loaded for {len(dtc)} (ticker,date) points")

    feats = [a.feature] if a.feature else BORROW_FEATURES
    holds = [a.hold] if a.hold else [40, 20]

    for feature in feats:
        by_date = load_borrow_feature(borrow_db, feature)
        if by_date is None:
            continue
        for h in holds:
            run_feature(feature, by_date, fwd, dtc, h, a.min_names)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
    except Exception:
        import traceback
        print("\n[UNEXPECTED ERROR] paste back:")
        traceback.print_exc()
