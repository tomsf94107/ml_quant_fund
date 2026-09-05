#!/usr/bin/env python3
"""
feature_break_audit.py — structural break scan on every feature series.

READ-ONLY. Writes nothing.

WHY
    On 2026-09-05 vix_term_structure was found pinned to the literal 1.0 for TEN
    WEEKS: builder.py asked yfinance for ^VIX3M, yfinance is XProtect-blocked on
    this machine, and the else-branch defaulted -- while warning.db held
    CBOE_VIX3M the whole time. Importance went 4.764 on 2026-06-27 to 0.000
    every day after and nothing reported it.

    feature_health_monitor.py now catches that, but only AFTER the model stops
    splitting on the feature. This catches the VALUE series changing character,
    which is earlier, and also catches a feature that jumps to a new level while
    remaining useful -- something importance monitoring never will.

WHAT THE FIRST VERSION GOT WRONG (2026-09-05, same day)
    A first pass graded every break by the ratio of segment medians on log|x|.
    It flagged 39 of 57 features, and nearly all the level-shift flags were
    artifacts:

      macd 2028x, obv 713x, macd_signal 330x   series that OSCILLATE AROUND
                                               ZERO. A segment median near zero
                                               makes the ratio explode.
      risk_today / risk_next_1d / 3d  inf      BINARY 0/1 series. The median of
                                               a mostly-zero binary is 0, so any
                                               shift divides by zero.
      close, ma_5/10/20, bb_upper  9-10x       TRENDING series. A stock going
                                               $50 to $500 is a real 10x, and it
                                               is price appreciation, not a
                                               defect.

    The literature is explicit that detrending is a required preprocessing step,
    and distinguishes a LEVEL SHIFT (intercept only) from a LEVEL SHIFT WITH
    TREND and a REGIME SHIFT (intercept and slope). Grading a trending series by
    a level ratio cannot tell them apart. The first version skipped that step.

WHAT THIS VERSION DOES
    1. CLASSIFY each series first, because the right test differs by type:
         CONSTANT      one unique value
         BINARY        two unique values, typically 0/1 flags
         ZERO_CROSSING sign changes in at least 5% of observations -- macd, obv,
                       returns. Ratios are meaningless here.
         TRENDING      |Spearman(value, time)| >= 0.7 -- close, ma_*. Detrended
                       by first differencing before detection, so a break is a
                       change in the DRIFT, not in the level.
         BOUNDED       stays inside a fixed range -- rsi_14, bb_pct
         LEVEL         everything else; a ratio is meaningful

    2. DETECT on the appropriate transform: raw for LEVEL and BOUNDED, first
       differences for TRENDING, raw for ZERO_CROSSING.

    3. GRADE by STANDARDISED SHIFT -- |median_after - median_before| divided by
       the pooled MAD -- rather than a ratio. This is scale-free, defined when a
       segment median is zero, and comparable across features. A ratio is kept
       alongside only where it is meaningful (LEVEL series that never cross
       zero).

    4. CONFIRM with CUSUM for mean shifts and CUSUM-SQ for variance shifts. The
       literature notes CUSUM is sensitive to mean changes and CUSUM-SQ to
       variance changes; the vix_term_structure signature is a variance collapse,
       so both are needed.

WHAT IT DOES NOT DO
    It reports; it does not repair. A genuine regime shift and a vendor
    recalculation are identical in the data. The date and magnitude let a human
    check the commit log -- which is what distinguished the deliberate feature
    cull in commit 8da49533 from vix_term_structure dying on 2026-06-28.

    The signature of a CONSTRUCTION or VENDOR change is a flag on nearly EVERY
    ticker on the SAME DATE. A flag on one ticker is usually that company.

    python analysis/feature_break_audit.py
    python analysis/feature_break_audit.py --tickers 40 --min-shift 4.0
"""
import argparse
import math
import statistics as st
import sys
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")


def mad(v):
    """Median absolute deviation, scaled to be comparable to a std dev."""
    if len(v) < 2:
        return 0.0
    m = st.median(v)
    return 1.4826 * st.median([abs(x - m) for x in v])


def classify(v):
    """Series type decides which transform and which grading rule apply."""
    u = set(v)
    if len(u) <= 1:
        return "CONSTANT"
    if len(u) == 2:
        return "BINARY"
    n = len(v)
    neg = sum(1 for x in v if x < 0)
    pos = sum(1 for x in v if x > 0)
    if min(neg, pos) / n >= 0.05:
        return "ZERO_CROSSING"
    # Spearman against time, to catch monotone drift without assuming linearity
    idx = list(range(n))
    rv = _rank(v)
    ri = _rank(idx)
    mv, mi = sum(rv) / n, sum(ri) / n
    num = sum((rv[i] - mv) * (ri[i] - mi) for i in range(n))
    dv = math.sqrt(sum((x - mv) ** 2 for x in rv))
    di = math.sqrt(sum((x - mi) ** 2 for x in ri))
    rho = (num / (dv * di)) if dv and di else 0.0
    # A STEP also scores high Spearman against time, and differencing a step
    # destroys the very shift being looked for. A genuine trend accumulates
    # gradually: its largest single jump is small next to the total change. A
    # step puts nearly all of the change into one observation.
    if abs(rho) >= 0.7:
        d = [abs(v[k] - v[k - 1]) for k in range(1, n)]
        total = abs(v[-1] - v[0]) or 1e-12
        if d and max(d) / total < 0.5:
            return "TRENDING"
        return "LEVEL"
    lo, hi = min(v), max(v)
    if lo >= -1.001 and hi <= 1.001:
        return "BOUNDED"
    return "LEVEL"


def _rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        a = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[o[k]] = a
        i = j + 1
    return r


def _cost(seg):
    n = len(seg)
    if n < 2:
        return 0.0
    m = sum(seg) / n
    return sum((x - m) ** 2 for x in seg)


def binseg(x, min_seg=20, max_breaks=5):
    """Binary segmentation with a Normal (L2) cost. The PELT literature finds
    the Normal cost best for mean shifts and notes L1 does markedly worse on
    variance shifts, so L2 is the right default here."""
    n = len(x)
    if n < 2 * min_seg + 2:
        return []
    var = st.pvariance(x) or 1e-12
    pen = 3.0 * math.log(n) * var
    found = []

    def split(lo, hi):
        if len(found) >= max_breaks or hi - lo < 2 * min_seg:
            return
        base = _cost(x[lo:hi])
        best, bi = 0.0, None
        for i in range(lo + min_seg, hi - min_seg):
            g = base - _cost(x[lo:i]) - _cost(x[i:hi])
            if g > best:
                best, bi = g, i
        if bi is not None and best > pen:
            found.append(bi)
            split(lo, bi)
            split(bi, hi)

    split(0, n)
    return sorted(found)


def cusum_mean(x, idx, window=40, k=1.5):
    n = len(x)
    m, s = sum(x) / n, (st.pstdev(x) or 1e-12)
    c, peak, pi = 0.0, 0.0, 0
    for i, v in enumerate(x):
        c += (v - m) / s
        if abs(c) > peak:
            peak, pi = abs(c), i
    return peak > k * math.sqrt(n) and abs(pi - idx) <= window


def cusum_sq(x, idx, window=40):
    """CUSUM of squares -- sensitive to VARIANCE change, which is the
    vix_term_structure signature (a live feature collapsing to a constant)."""
    n = len(x)
    m = sum(x) / n
    sq = [(v - m) ** 2 for v in x]
    tot = sum(sq) or 1e-12
    run, worst, wi = 0.0, 0.0, 0
    for i, v in enumerate(sq):
        run += v
        dev = abs(run / tot - (i + 1) / n)
        if dev > worst:
            worst, wi = dev, i
    return worst > 1.36 / math.sqrt(n) and abs(wi - idx) <= window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=25)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--min-seg", type=int, default=20)
    ap.add_argument("--min-shift", type=float, default=3.0,
                    help="standardised shift (MAD units) to flag")
    ap.add_argument("--hard-shift", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe
    import random

    uni = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    random.Random(args.seed).shuffle(uni)
    uni = uni[:args.tickers]
    print(f"break audit — {len(uni)} tickers from {args.start}")
    print(f"  graded by STANDARDISED SHIFT in MAD units, not by ratio;")
    print(f"  SOFT >= {args.min_shift}, HARD >= {args.hard_shift}\n")

    findings = defaultdict(list)
    kinds_of = defaultdict(Counter)
    seen = Counter()
    built = 0

    for i, t in enumerate(uni, 1):
        try:
            df = build_feature_dataframe(t, start_date=args.start,
                                         training_mode=True)
            if df is None or len(df) < 120:
                continue
            built += 1
            dates = [str(d)[:10] for d in df["date"]]
            num = df.select_dtypes("number")
            for c in num.columns:
                if c.startswith("target_"):
                    continue
                seen[c] += 1
                raw = [float(v) for v in num[c]]
                nn = [v for v in raw if v == v]
                if not nn:
                    findings[c].append((t, "ALL_NULL", "-", 0.0))
                    kinds_of[c]["ALL_NULL"] += 1
                    continue
                kind = classify(nn)
                kinds_of[c][kind] += 1
                if kind in ("CONSTANT", "BINARY"):
                    if kind == "CONSTANT":
                        findings[c].append((t, "CONSTANT", "-", 0.0))
                    continue
                tail = raw[-40:]
                if all(v != v for v in tail) and any(v == v for v in raw[:-40]):
                    findings[c].append((t, "NEWLY_NULL", dates[-40], 0.0))
                    continue

                # detrend by first differencing when the series trends, so a
                # break means a change in DRIFT rather than a change in level
                if kind == "TRENDING":
                    sig = [nn[k] - nn[k - 1] for k in range(1, len(nn))]
                    off = 1
                else:
                    sig = nn
                    off = 0
                if len(sig) < 2 * args.min_seg + 2:
                    continue

                # Two detection passes. An L2 cost measures deviation from a
                # segment MEAN, so a variance-only change with an unchanged mean
                # produces almost no cost reduction and is missed entirely --
                # which is the vix_term_structure signature. Squared deviations
                # turn a variance change into a mean change, which L2 does see.
                # The literature makes the same split: CUSUM for mean shifts,
                # CUSUM-SQ for variance shifts.
                brks = binseg(sig, min_seg=args.min_seg)
                _m = sum(sig) / len(sig)
                brks_v = binseg([(z - _m) ** 2 for z in sig],
                                min_seg=args.min_seg)
                brks = sorted(set(brks) | set(brks_v))
                best_shift, best_i, best_var = 0.0, None, 1.0
                for b in brks:
                    a_, b_ = sig[:b], sig[b:]
                    if len(a_) < 10 or len(b_) < 10:
                        continue
                    pooled = (mad(a_) + mad(b_)) / 2.0
                    if pooled < 1e-12:
                        pooled = (st.pstdev(sig) or 1e-12)
                    shift = abs(st.median(b_) - st.median(a_)) / pooled
                    va = st.pvariance(a_) or 1e-18
                    vb = st.pvariance(b_) or 1e-18
                    vr = va / vb
                    ok = cusum_mean(sig, b) or cusum_sq(sig, b)
                    if not ok:
                        continue
                    if vr >= 100 or shift > best_shift:
                        if shift > best_shift or vr >= 100:
                            best_shift, best_i, best_var = shift, b, vr
                if best_i is None:
                    continue
                di = min(best_i + off, len(dates) - 1)
                if best_var >= 100:
                    lab = "COLLAPSED"
                elif best_shift >= args.hard_shift:
                    lab = "HARD"
                elif best_shift >= args.min_shift:
                    lab = "SOFT"
                else:
                    continue
                findings[c].append((t, lab, dates[di], best_shift))
            if i % 10 == 0:
                print(f"  ...{i}/{len(uni)}")
        except Exception:
            continue

    print(f"\nscanned {built} tickers, {len(seen)} feature columns\n")
    order = {"ALL_NULL": 0, "CONSTANT": 1, "COLLAPSED": 2, "NEWLY_NULL": 3,
             "HARD": 4, "SOFT": 5}
    rows = []
    for c, f in findings.items():
        cnt = Counter(k for _, k, _, _ in f)
        worst = min(cnt, key=lambda k: order.get(k, 9))
        ex = [x for x in f if x[1] == worst]
        dt = Counter(x[2] for x in ex).most_common(1)[0]
        shift = max((x[3] for x in ex), default=0.0)
        typ = kinds_of[c].most_common(1)[0][0] if kinds_of[c] else "?"
        rows.append((order.get(worst, 9), -cnt[worst], c, worst, cnt[worst],
                     seen[c], dt[0], dt[1], shift, typ))
    rows.sort()

    print(f"  {'feature':<30}{'type':<14}{'flag':<11}{'tickers':>9}"
          f"{'same date':>11}{'shift':>8}")
    for _, _, c, k, n, tot, dt, dn, sh, typ in rows:
        same = f"{dn}/{n}" if k in ("HARD", "SOFT", "COLLAPSED") else "—"
        shs = f"{sh:.1f}" if sh else "—"
        print(f"  {c:<30}{typ:<14}{k:<11}{n:>4}/{tot:<4}{same:>11}{shs:>8}")

    print(f"\n  {len(rows)} of {len(seen)} features flagged, "
          f"{len(seen)-len(rows)} clean\n")
    print("  'same date' counts how many of the flagged tickers break on the")
    print("  SAME day. High share AND one shared date is the signature of a")
    print("  construction or vendor change -- the case worth investigating.")
    print("  Scattered dates are per-company events.\n")
    print("  'shift' is in MAD units, so it is comparable across features and")
    print("  defined even when a segment median is zero. TRENDING series are")
    print("  first-differenced before detection, so a flag there means the DRIFT")
    print("  changed, not that the price went up.")


if __name__ == "__main__":
    main()
