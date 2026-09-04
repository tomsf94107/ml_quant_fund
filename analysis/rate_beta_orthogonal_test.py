#!/usr/bin/env python3
"""
rate_beta_orthogonal_test.py — does rate_beta add anything the model already has?

READ-ONLY. Writes nothing.

THE QUESTION
    rate_beta survived the regime test: the tercile spread kept its sign when
    yields FELL (+4.7pp at h=5) as well as when they rose (+0.5pp), which rules
    out pure duration exposure. ICs ran t=+2.19 and t=+2.20 with clean nulls.

    But surviving is not the same as adding. rate_beta almost certainly proxies
    LEVERAGE -- the CRWV reports say it outright: ~$35B of debt where interest
    expense swamps operating income. Highly levered firms fall when rates rise;
    that is mechanical. "Low rate-beta outperforms" may simply be the
    quality/low-leverage factor, documented for decades.

    The model already carries ~100 features including beta_60d (importance
    6.91), realised volatility, and a range of price-based characteristics that
    a levered firm's behaviour would show up in. If rate_beta is spanned by
    those, adding it changes nothing.

WHAT IS TESTED
    Each date, cross-sectionally:

      RAW           IC of rate_beta against forward returns
      ORTHOGONAL    IC of the RESIDUAL of rate_beta after regressing it on the
                    existing features, against forward returns

    If the orthogonal IC collapses toward zero, rate_beta is spanned by what
    the model already sees and adds nothing. If it survives at close to the raw
    magnitude, it carries independent information.

    Controls used, all already in prediction_features:
      beta_60d          market beta -- the closest existing analogue
      short_ratio       positioning
      vix_close         market volatility level (constant per date, so it drops
                        out of a cross-sectional regression -- included only to
                        confirm the code handles it)
      return_20d        recent momentum
      rsi_14            oscillator
      atr               realised range
      sector_rel_ret    sector relative strength

    Cross-sectional demeaning happens implicitly: each date is regressed
    separately, so anything constant across names that day cannot explain
    cross-sectional variation.

METHOD
    Per date: multivariate OLS of rate_beta on the controls, take residuals,
    Spearman the residuals against forward returns. Newey-West on the IC series
    at the horizon lag. Shuffle null on both raw and orthogonal.

    OLS is solved by Gaussian elimination on the normal equations with a small
    ridge term for numerical stability; with 7 controls and 100+ names per date
    the system is well over-determined, so the ridge is negligible.

INTERPRETATION BAR
    This is roughly the 75th cell in this sequence. A raw IC that survives
    orthogonalisation at t >= 3.0 with a clean null would be a genuine finding.
    Anything less -- and especially a collapse toward zero -- means the
    characteristic is already in the model and no feature should be added.

    python analysis/rate_beta_orthogonal_test.py
"""
import argparse
import math
import random
import sqlite3
import statistics as st
from collections import defaultdict

CONTROLS = ["beta_60d", "short_ratio", "return_20d", "rsi_14", "atr",
            "sector_rel_ret", "vix_close"]


def spearman(pairs):
    n = len(pairs)
    if n < 10:
        return None
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1
            for m in range(i, j + 1):
                r[o[m]] = a
            i = j + 1
        return r
    rx = rank([p[0] for p in pairs]); ry = rank([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    return num / (dx * dy) if dx and dy else None


def nw_t(s, lag):
    n = len(s)
    if n < 10:
        return None
    m = sum(s) / n
    d = [x - m for x in s]
    var = sum(x * x for x in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[i] * d[i - k] for i in range(k, n)) / n
        var += 2 * (1 - k / (lag + 1.0)) * gk
    return m / math.sqrt(var / n) if var > 0 else None


def ols_residuals(Y, X, ridge=1e-8):
    """Residuals of Y on X (list of rows, each a list of regressors) + intercept."""
    n = len(Y)
    k = len(X[0]) + 1
    A = [[1.0] + list(row) for row in X]
    # normal equations
    XtX = [[sum(A[i][a] * A[i][b] for i in range(n)) for b in range(k)]
           for a in range(k)]
    for a in range(k):
        XtX[a][a] += ridge
    Xty = [sum(A[i][a] * Y[i] for i in range(n)) for a in range(k)]
    # gaussian elimination
    M = [XtX[a][:] + [Xty[a]] for a in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for j in range(c, k + 1):
            M[c][j] /= pv
        for r in range(k):
            if r == c:
                continue
            f = M[r][c]
            if f:
                for j in range(c, k + 1):
                    M[r][j] -= f * M[c][j]
    beta = [M[a][k] for a in range(k)]
    return [Y[i] - sum(beta[a] * A[i][a] for a in range(k)) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices-db", default="prices.db")
    ap.add_argument("--warning-db", default="warning.db")
    ap.add_argument("--accuracy-db", default="accuracy.db")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--min-names", type=int, default=40)
    args = ap.parse_args()
    HOR = (5, 20)

    wc = sqlite3.connect(f"file:{args.warning_db}?mode=ro", uri=True)
    y = {str(d)[:10]: v for d, v in wc.execute(
        "SELECT obs_date, value FROM data_vintages WHERE series_id='DGS10' "
        "AND obs_date >= ?", (args.start,))}
    wc.close()
    yd = sorted(y)
    dy = {yd[i]: y[yd[i]] - y[yd[i - 1]] for i in range(1, len(yd))}

    px = sqlite3.connect(f"file:{args.prices_db}?mode=ro", uri=True)
    close = defaultdict(dict)
    for t, d, c in px.execute(
            "SELECT ticker, d, close FROM raw_bars WHERE d >= ? AND close>0",
            (args.start,)):
        close[t][d] = c
    px.close()

    fwd = {h: {} for h in HOR}
    beta = {}
    for t, s in close.items():
        ds = sorted(s)
        rets = {}
        for i in range(1, len(ds)):
            a, b = s[ds[i - 1]], s[ds[i]]
            if a and abs((b - a) / a) < 0.5:
                rets[ds[i]] = (b - a) / a
        for h in HOR:
            for i in range(len(ds) - h):
                a, b = s[ds[i]], s[ds[i + h]]
                if a and b and abs((b - a) / a) < 0.8:
                    fwd[h][(t, ds[i])] = (b - a) / a
        rd = sorted(rets)
        for i in range(args.window, len(rd)):
            w = rd[i - args.window:i]
            pair = [(dy[d], rets[d]) for d in w if d in dy]
            if len(pair) < args.window * 0.6:
                continue
            mx = sum(p[0] for p in pair) / len(pair)
            my = sum(p[1] for p in pair) / len(pair)
            sxx = sum((p[0] - mx) ** 2 for p in pair)
            if sxx > 0:
                beta[(t, rd[i])] = sum((p[0] - mx) * (p[1] - my)
                                       for p in pair) / sxx

    ac = sqlite3.connect(f"file:{args.accuracy_db}?mode=ro", uri=True)
    have = [r[1] for r in ac.execute("PRAGMA table_info(prediction_features)")]
    ctrl = [c for c in CONTROLS if c in have]
    missing = [c for c in CONTROLS if c not in have]
    print(f"controls available: {', '.join(ctrl)}")
    if missing:
        print(f"controls MISSING from prediction_features: {', '.join(missing)}")
    feats = defaultdict(dict)
    for row in ac.execute(
            f"SELECT ticker, prediction_date, {', '.join(ctrl)} "
            f"FROM prediction_features WHERE horizon=5"):
        t, d = row[0], row[1]
        vals = row[2:]
        if all(v is not None for v in vals):
            feats[d][t] = list(vals)
    ac.close()
    print(f"{sum(len(v) for v in feats.values()):,} feature rows over "
          f"{len(feats)} dates\n")

    rnd = random.Random(29)
    res = {}
    for h in HOR:
        raw_ics, orth_ics, raw_null, orth_null = [], [], [], []
        for d in sorted(feats):
            names = [t for t in feats[d]
                     if (t, d) in beta and (t, d) in fwd[h]]
            if len(names) < args.min_names:
                continue
            Y = [beta[(t, d)] for t in names]
            X = [feats[d][t] for t in names]
            R = [fwd[h][(t, d)] for t in names]
            r_raw = spearman(list(zip(Y, R)))
            resid = ols_residuals(Y, X)
            if r_raw is not None:
                raw_ics.append(r_raw)
            if resid is not None:
                r_o = spearman(list(zip(resid, R)))
                if r_o is not None:
                    orth_ics.append(r_o)
            sh = R[:]
            rnd.shuffle(sh)
            rn = spearman(list(zip(Y, sh)))
            if rn is not None:
                raw_null.append(rn)
            if resid is not None:
                rn2 = spearman(list(zip(resid, sh)))
                if rn2 is not None:
                    orth_null.append(rn2)
        res[h] = (raw_ics, orth_ics, raw_null, orth_null)

    print(f"  {'variant':<14}{'h':>4}{'dates':>7}{'mean IC':>10}{'NW t':>8}"
          f"{'null t':>9}")
    for h in HOR:
        raw_ics, orth_ics, raw_null, orth_null = res[h]
        for label, ics, nl in (("raw", raw_ics, raw_null),
                               ("orthogonal", orth_ics, orth_null)):
            if len(ics) < 20:
                print(f"  {label:<14}{h:>4}{len(ics):>7}   too few dates")
                continue
            t_ = nw_t(ics, h) or 0.0
            nt = nw_t(nl, h) or 0.0
            flag = "  <<<" if abs(t_) >= 3.0 and abs(nt) < 1.5 else ""
            print(f"  {label:<14}{h:>4}{len(ics):>7}{st.mean(ics):>+10.4f}"
                  f"{t_:>+8.2f}{nt:>+9.2f}{flag}")
        if len(raw_ics) >= 20 and len(orth_ics) >= 20:
            keep = (st.mean(orth_ics) / st.mean(raw_ics)
                    if st.mean(raw_ics) else 0)
            print(f"  {'':<14}{h:>4}   orthogonal retains "
                  f"{100*keep:.0f}% of the raw IC")
        print()

    print("  A collapse toward zero means rate_beta is SPANNED by features the "
          "model\n  already has -- adding it would be redundant. Retention near "
          "100% with a\n  clean null and t >= 3.0 would mean independent "
          "information.\n")
    print("  Note beta_60d is the closest existing analogue: market beta and "
          "rate beta\n  are mechanically related for a levered firm, so heavy "
          "shrinkage is the\n  expected outcome and would be an honest negative "
          "result.")


if __name__ == "__main__":
    main()
