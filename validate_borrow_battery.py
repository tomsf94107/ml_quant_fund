#!/usr/bin/env python3
"""
validate_borrow_battery.py -- Full validation battery for the fee_gt_5pct residual signal.

CONTEXT: DIAG 2 showed the DTC-residualized fee_gt_5pct flag has a REAL (permutation
p=0.000, z=+6.26) positive residual IC at h=40 -- but the RAW IC failed standalone,
so it is a CONDITIONAL/interaction signal (HTB given short interest). This battery
subjects it to the SAME rigor the SI brick passed, to decide: real 3rd brick, or
residualization mirage?

THE SIGNAL UNDER TEST (fixed throughout):
  On each settlement date, residualize fee_gt_5pct on days_to_cover across stocks
  (OLS, keep residual), then IC that residual vs h=40 forward return. The residual
  is "hard-to-borrow beyond what short interest explains."

FIVE TESTS (each reports per-date IC mean, Newey-West t, and where applicable a
permutation p vs shuffled-return null):

  1. OOS COLD SPLIT   -- pre-2024 vs 2024+; must hold sign+significance in the
                         held-out half (SI brick survived cold holdout).
  2. SECTOR-NEUTRAL   -- also demean feature & return within `bucket` per date
                         (mirrors validate_si_sector.py). Survives => stock-level,
                         not a sector bet.
  3. YEAR-BY-YEAR     -- residual IC each year 2021-2026; sign stability
                         (SI brick: same sign every year).
  4. FLOAT-BUCKET     -- per-date float terciles; where does the signal concentrate?
                         (mechanism: is it only small-float scarcity?)
  5. INDEX-EVENT      -- residual IC in vs out of Russell/S&P reconstitution windows
                         (mechanism: mechanical-flow contamination?)

READ-ONLY. borrow.db + prices.db + short_interest.db + tickers_metadata.csv. No writes.

RUN
  python validate_borrow_battery.py
  python validate_borrow_battery.py --k 500      # more permutations
  python validate_borrow_battery.py --hold 40    # (default 40)
"""

import argparse, os, sqlite3, math, datetime, csv, sys
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

def nw_t(ics, hold, gap=15):
    ics = np.asarray(ics, float)
    if len(ics) < 2: return None, None
    lag = max(1, int(math.ceil(hold / float(gap))))
    se = newey_west_se_mean(ics, lag)
    m = float(ics.mean())
    return m, (m / se if se else None)


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

def load_borrow_rows(borrow_db):
    """date -> list of (ticker, fee_gt_5pct, total_float)."""
    c = ro(borrow_db)
    try:
        rows = Q(c, 'SELECT ticker,asof_date,fee_gt_5pct,total_float FROM borrow_features bf '
                    'JOIN borrow_fees USING(ticker,asof_date)') if False else \
               Q(c, 'SELECT f.ticker,f.asof_date,f.fee_gt_5pct,b.total_float '
                    'FROM borrow_features f JOIN borrow_fees b '
                    'ON f.ticker=b.ticker AND f.asof_date=b.asof_date')
    finally:
        c.close()
    by_date = defaultdict(list)
    for tk, d, flag, tf in rows:
        do = nd(d)
        if do is None or flag is None: continue
        try: fl = float(flag)
        except Exception: continue
        tfv = None
        if tf is not None:
            try: tfv = float(tf)
            except Exception: tfv = None
        by_date[do].append((tk.upper(), fl, tfv))
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

def load_sectors(path):
    sec = {}
    if not os.path.isfile(path):
        return sec
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            tk = (row.get("ticker") or "").upper()
            b = row.get("bucket") or row.get("sector") or ""
            if tk: sec[tk] = b
    return sec


# ---- Russell/S&P reconstitution windows (approx): +/- 5 calendar days ----
def in_index_window(d):
    y = d.year
    # Russell: last Friday of June
    june = [datetime.date(y, 6, day) for day in range(24, 31) if datetime.date(y, 6, day).weekday() == 4]
    russell = june[-1] if june else datetime.date(y, 6, 30)
    # S&P quarterly: 3rd Friday of Mar/Jun/Sep/Dec
    sp = []
    for m in (3, 6, 9, 12):
        fris = [datetime.date(y, m, day) for day in range(15, 22) if datetime.date(y, m, day).weekday() == 4]
        if fris: sp.append(fris[0])
    for ev in [russell] + sp:
        if abs((d - ev).days) <= 5:
            return True
    return False


def residual_feature_by_date(by_date, dtc, sector=None, sector_neutral=False,
                             float_filter=None):
    """Per date: residualize fee_gt_5pct on DTC across stocks (optionally also
    sector-demean feature+return); return date -> list of (ticker, resid_value).
    float_filter: None, or ('tercile', which) where which in {'small','mid','large'}.
    Returns resid map; the caller pairs with returns."""
    out = defaultdict(list)
    for d in sorted(by_date):
        recs = [(tk, fl, tf) for tk, fl, tf in by_date[d] if dtc.get((tk, d)) is not None]
        if float_filter is not None:
            recs = [(tk, fl, tf) for tk, fl, tf in recs if tf is not None]
            if len(recs) >= 6:
                floats = np.array([tf for _, _, tf in recs])
                q1, q2 = np.percentile(floats, [33.33, 66.67])
                which = float_filter[1]
                if which == "small":
                    recs = [(tk, fl, tf) for tk, fl, tf in recs if tf <= q1]
                elif which == "mid":
                    recs = [(tk, fl, tf) for tk, fl, tf in recs if q1 < tf <= q2]
                else:
                    recs = [(tk, fl, tf) for tk, fl, tf in recs if tf > q2]
        if len(recs) < 6:
            continue
        feat = np.array([fl for _, fl, _ in recs], float)
        dcov = np.array([dtc[(tk, d)] for tk, _, _ in recs], float)
        # residualize feature on DTC
        if dcov.std() == 0:
            resid = feat - feat.mean()
        else:
            A = np.vstack([np.ones_like(dcov), dcov]).T
            coef, *_ = np.linalg.lstsq(A, feat, rcond=None)
            resid = feat - A @ coef
        # optional sector-demeaning of the residual
        if sector_neutral and sector:
            secs = [sector.get(tk, "") for tk, _, _ in recs]
            smean = defaultdict(list)
            for s, rv in zip(secs, resid):
                smean[s].append(rv)
            savg = {s: np.mean(v) for s, v in smean.items()}
            resid = np.array([rv - savg[s] for s, rv in zip(secs, resid)])
        for (tk, _, _), rv in zip(recs, resid):
            out[d].append((tk, float(rv)))
    return out

def ic_from_resid(resid_map, fwd, hold, sector=None, sector_neutral=False,
                  min_names=15, date_filter=None):
    ics = []
    for d in sorted(resid_map):
        if date_filter and not date_filter(d):
            continue
        sig = []; ret = []; secs = []
        for tk, rv in resid_map[d]:
            r = fwd(tk, d, hold)
            if r is not None:
                sig.append(rv); ret.append(r); secs.append(sector.get(tk, "") if sector else "")
        if len(sig) < min_names:
            continue
        ret = np.array(ret, float)
        if sector_neutral and sector:
            smean = defaultdict(list)
            for s, rr in zip(secs, ret): smean[s].append(rr)
            savg = {s: np.mean(v) for s, v in smean.items()}
            ret = np.array([rr - savg[s] for s, rr in zip(secs, ret)])
        ic = spearman(np.array(sig), ret)
        if ic is not None:
            ics.append((d, ic, len(sig)))
    return ics

def summ(ics, hold, label, gap=15):
    if len(ics) < 6:
        print(f"  [{label}] {len(ics)} dates (<6) -- SKIP"); return None
    arr = np.array([ic for _, ic, _ in ics])
    ns = [n for _, _, n in ics]
    dts = [d for d, _, _ in ics]
    m, t = nw_t(arr, hold, gap)
    pos = 100.0 * np.mean(arr > 0)
    print(f"  [{label}] N={len(arr)} dates, avg {int(np.mean(ns))} stk, {dts[0]}..{dts[-1]}")
    print(f"       mean resid-IC={m:+.4f}  NW-t={t:+.2f}  %dates>0={pos:.0f}%")
    return {"mean": m, "t": t, "N": len(arr), "pos": pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--borrow-db", default=None)
    ap.add_argument("--prices-db", default=None)
    ap.add_argument("--si-db", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--hold", type=int, default=40)
    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--min-names", type=int, default=15)
    ap.add_argument("--split-date", default="2024-01-01")
    a = ap.parse_args(); a.root = os.path.expanduser(a.root)

    borrow_db = a.borrow_db or os.path.join(a.root, "borrow.db")
    prices_db = a.prices_db or os.path.join(a.root, "prices.db")
    si_db     = a.si_db or os.path.join(a.root, "short_interest.db")
    meta      = a.meta or os.path.join(a.root, "tickers_metadata.csv")
    for p, nm in [(borrow_db,"borrow.db"),(prices_db,"prices.db"),(si_db,"short_interest.db")]:
        if not os.path.isfile(p): print(f"[STOP] {nm} not found"); return

    print("loading ...")
    px, pos_of = load_prices(prices_db)
    fwd = make_fwd(px, pos_of)
    by_date = load_borrow_rows(borrow_db)
    dtc = load_dtc(si_db)
    sector = load_sectors(meta)
    hold = a.hold
    print(f"prices {len(px)} tk | borrow dates {len(by_date)} | DTC {len(dtc)} pts | sectors {len(sector)}")

    banner(f"BATTERY: fee_gt_5pct residual-vs-DTC signal  (h={hold})")

    # base residual map (no sector-neutral, all floats)
    resid = residual_feature_by_date(by_date, dtc)

    # ---- baseline (full sample) ----
    sub("BASELINE (full sample, residual-vs-DTC)")
    base = summ(ic_from_resid(resid, fwd, hold, min_names=a.min_names), hold, "FULL")
    split = nd(a.split_date)

    # ---- TEST 1: OOS cold split ----
    sub(f"TEST 1 -- OOS COLD SPLIT at {a.split_date}")
    pre = summ(ic_from_resid(resid, fwd, hold, min_names=a.min_names,
                             date_filter=lambda d: d < split), hold, "PRE  (train era)")
    post = summ(ic_from_resid(resid, fwd, hold, min_names=a.min_names,
                              date_filter=lambda d: d >= split), hold, "POST (held-out)")
    if pre and post:
        same_sign = (pre["mean"] > 0) == (post["mean"] > 0)
        print(f"  -> {'SAME sign' if same_sign else 'SIGN FLIP'} across split; "
              f"held-out NW-t={post['t']:+.2f} "
              f"({'holds' if (same_sign and abs(post['t'])>=1.5) else 'weak/breaks'} OOS)")

    # ---- TEST 2: sector-neutral ----
    sub("TEST 2 -- SECTOR-NEUTRAL (demean feature+return within bucket)")
    if sector:
        resid_sn = residual_feature_by_date(by_date, dtc, sector=sector, sector_neutral=True)
        sn = summ(ic_from_resid(resid_sn, fwd, hold, sector=sector, sector_neutral=True,
                                min_names=a.min_names), hold, "SECTOR-NEUTRAL")
        if base and sn and base["mean"] != 0:
            keep = 100.0 * sn["mean"] / base["mean"]
            print(f"  -> sector-neutral retains {keep:.0f}% of full IC "
                  f"({'stock-level signal' if abs(sn.get('t',0))>=1.5 and keep>40 else 'likely sector bet'})")
    else:
        print("  [SKIP] no sector metadata loaded")

    # ---- TEST 3: year-by-year ----
    sub("TEST 3 -- YEAR-BY-YEAR sign stability")
    years = sorted(set(d.year for d in resid))
    ybk = {}
    for y in years:
        ics_y = ic_from_resid(resid, fwd, hold, min_names=a.min_names,
                              date_filter=lambda d, yy=y: d.year == yy)
        if len(ics_y) >= 3:
            arr = np.array([ic for _, ic, _ in ics_y])
            ybk[y] = float(arr.mean())
            print(f"  {y}: mean resid-IC={arr.mean():+.4f}  ({len(ics_y)} dates)")
    if ybk:
        signs = [1 if v > 0 else -1 for v in ybk.values()]
        agree = 100.0 * max(signs.count(1), signs.count(-1)) / len(signs)
        print(f"  -> {agree:.0f}% of years share the dominant sign "
              f"({'STABLE' if agree >= 80 else 'UNSTABLE -- driven by subset of years'})")

    # ---- TEST 4: float terciles ----
    sub("TEST 4 -- FLOAT-BUCKET concentration (per-date terciles)")
    for which in ("small", "mid", "large"):
        rf = residual_feature_by_date(by_date, dtc, float_filter=("tercile", which))
        summ(ic_from_resid(rf, fwd, hold, min_names=max(6, a.min_names // 2)), hold, f"{which.upper():5s} float")
    print("  -> if signal concentrates ONLY in 'small' -> scarcity/small-float effect, narrow")

    # ---- TEST 5: index-event windows ----
    sub("TEST 5 -- INDEX-EVENT windows (Russell late-Jun, S&P quarterly 3rd-Fri)")
    inw = summ(ic_from_resid(resid, fwd, hold, min_names=max(6, a.min_names // 2),
                             date_filter=lambda d: in_index_window(d)), hold, "IN  window")
    outw = summ(ic_from_resid(resid, fwd, hold, min_names=a.min_names,
                              date_filter=lambda d: not in_index_window(d)), hold, "OUT window")
    if inw and outw:
        print(f"  -> OUT-of-window NW-t={outw['t']:+.2f} "
              f"({'signal survives outside index events (not mechanical)' if abs(outw['t'])>=1.5 else 'weak outside events -- possible mechanical flow'})")

    # ---- overall read ----
    banner("BATTERY READ")
    print("  A real 3rd-brick candidate needs: OOS held-out holds sign+|t|>=1.5,")
    print("  sector-neutral retains >40% IC, year signs >=80% stable, and survives")
    print("  OUTSIDE index-event windows. Concentration only in small-float or only")
    print("  in-window => narrow artifact, not a broad brick.")
    print("\n  (raw standalone IC already FAILED |t|>=2 -- this is a CONDITIONAL signal;")
    print("   even if it passes, wiring it means the DTC-residualized/interaction form,")
    print("   with its own integration + live validation, not a plain feature add.)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
    except Exception:
        import traceback
        print("\n[UNEXPECTED ERROR] paste back:")
        traceback.print_exc()
