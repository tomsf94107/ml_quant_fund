#!/usr/bin/env python3
"""
fund_feature_test.py — do the five fundamental features add anything?

READ-ONLY. Writes nothing. Tests before wiring.

THE SITUATION
    features/builder.py builds five fundamental features and has since at least
    May 2026:

        fund_gp_assets   (revenue - cogs) / total_assets   Novy-Marx quality
        fund_op_equity   operating_income / equity
        fund_ni_margin   net_income / revenue
        fund_bm          equity / market cap               value
        fund_ep          net_income / market cap           earnings yield

    They are populated -- 0% NaN on AAPL, KO and NVDA -- PIT-correct via
    load_fundamental_features_pit reading xbrl_facts.filed_date, and
    economically sensible: NVDA gross profitability 0.306 against KO 0.126,
    AAPL book-to-market 0.022 against KO 0.100.

    They are NOT in models/classifier.py FEATURE_COLUMNS, and
    feature_importance_history contains ZERO rows for any of them. They have
    never been in a trained model. Built, verified, classified in
    analysis/detect_mw.py -- and never wired. That is the built-but-not-wired
    pattern RULE-1 names.

WHY THE CLOSED AXIS DOES NOT SETTLE THIS
    validate_gp.py closed gross profitability, but read its own header: "brick
    #3", monthly formation dates, --hold 40, decorrelation against momentum and
    SI. That tested GP as a STANDALONE ALPHA BRICK at a 40-day hold, against the
    SI-brick gauntlet.

    One feature among 96 in a daily h=3/5 gradient-boosted classifier is a
    different object. A characteristic can fail as standalone monthly alpha and
    still contribute in combination at a five-day horizon. And fund_bm and
    fund_ep are value and earnings yield -- different factors that
    validate_gp.py never tested at all.

THE TEST
    Per date, cross-sectionally:
      RAW          Spearman IC of each feature against forward returns
      ORTHOGONAL   IC of the residual after regressing the feature on the
                   controls the model already has

    rate_beta failed exactly this test: raw t=+3.80 at h=20, but orthogonalised
    it retained 37-41% with residual t=+1.06 to +1.52, because it is derived
    from returns and the model carries ~96 return-based features.

    Fundamentals are NOT derived from returns. fund_gp_assets comes from an
    income statement and a balance sheet. It cannot be a linear combination of
    momentum, RSI, ATR and beta. So high retention here would be structurally
    meaningful -- and low retention would be a genuinely surprising negative.

    Note fund_bm and fund_ep DIVIDE BY MARKET CAP, so they contain price. They
    are partly price-derived and should be expected to shrink more than the
    three pure-accounting ratios. That difference is itself informative and is
    reported per feature rather than pooled.

BAR
    |t| >= 3.0 on the ORTHOGONAL IC with a clean shuffle null (Harvey, Liu &
    Zhu, RFS 2016). Wiring a feature triggers a full retrain of ~2,600 model
    files, so the bar should be met before the cost is paid -- not after.

    python analysis/fund_feature_test.py
"""
import argparse
import math
import random
import sqlite3
import statistics as st
import sys
from collections import defaultdict

FUND = ["fund_gp_assets", "fund_op_equity", "fund_ni_margin",
        "fund_bm", "fund_ep"]
CONTROLS = ["beta_60d", "short_ratio", "return_20d", "rsi_14", "atr",
            "sector_rel_ret"]


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
    n = len(Y); k = len(X[0]) + 1
    A = [[1.0] + list(r) for r in X]
    M = [[sum(A[i][a] * A[i][b] for i in range(n)) for b in range(k)]
         + [sum(A[i][a] * Y[i] for i in range(n))] for a in range(k)]
    for a in range(k):
        M[a][a] += ridge
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for j in range(c, k + 1):
            M[c][j] /= pv
        for r in range(k):
            if r != c and M[r][c]:
                f = M[r][c]
                for j in range(c, k + 1):
                    M[r][j] -= f * M[c][j]
    b = [M[a][k] for a in range(k)]
    return [Y[i] - sum(b[a] * A[i][a] for a in range(k)) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=120,
                    help="tickers to build features for; the binding cost")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--min-names", type=int, default=30)
    args = ap.parse_args()
    HOR = (5, 20)

    sys.path.insert(0, ".")
    from features.builder import build_feature_dataframe

    universe = [l.strip().upper() for l in open("tickers.txt") if l.strip()]
    random.Random(3).shuffle(universe)
    universe = universe[:args.sample]
    print(f"building features for {len(universe)} tickers "
          f"from {args.start} -- this is slow\n")

    # date -> ticker -> {feature: value}
    panel = defaultdict(dict)
    fwd = {h: {} for h in HOR}
    built = 0
    for i, t in enumerate(universe, 1):
        try:
            df = build_feature_dataframe(t, start_date=args.start,
                                         training_mode=True)
            if df is None or df.empty:
                continue
            need = FUND + CONTROLS + ["close", "date"]
            if any(c not in df.columns for c in need):
                continue
            built += 1
            ds = [str(d)[:10] for d in df["date"]]
            cl = list(df["close"])
            for h in HOR:
                for j in range(len(ds) - h):
                    a, b = cl[j], cl[j + h]
                    if a and b and abs((b - a) / a) < 0.8:
                        fwd[h][(t, ds[j])] = (b - a) / a
            for j, d in enumerate(ds):
                row = {}
                bad = False
                for c in FUND + CONTROLS:
                    v = df[c].iloc[j]
                    if v != v:
                        bad = True
                        break
                    row[c] = float(v)
                if not bad:
                    panel[d][t] = row
            if i % 30 == 0:
                print(f"  ...{i} tickers, {built} usable")
        except Exception:
            continue
    print(f"\nbuilt {built} tickers, {len(panel)} dates with complete rows\n")

    dates = sorted(panel)[::5]
    rnd = random.Random(19)
    print(f"  {'feature':<18}{'h':>4}{'dates':>7}{'raw IC':>9}{'raw t':>8}"
          f"{'orth IC':>10}{'orth t':>8}{'null t':>8}{'kept':>7}")
    for f in FUND:
        for h in HOR:
            raws, orths, nulls = [], [], []
            for d in dates:
                names = [t for t in panel[d] if (t, d) in fwd[h]]
                if len(names) < args.min_names:
                    continue
                Y = [panel[d][t][f] for t in names]
                X = [[panel[d][t][c] for c in CONTROLS] for t in names]
                R = [fwd[h][(t, d)] for t in names]
                rr = spearman(list(zip(Y, R)))
                if rr is not None:
                    raws.append(rr)
                res = ols_residuals(Y, X)
                if res is None:
                    continue
                ro_ = spearman(list(zip(res, R)))
                if ro_ is not None:
                    orths.append(ro_)
                sh = R[:]
                rnd.shuffle(sh)
                rn = spearman(list(zip(res, sh)))
                if rn is not None:
                    nulls.append(rn)
            if len(orths) < 20:
                print(f"  {f:<18}{h:>4}{len(orths):>7}   too few dates")
                continue
            rm = st.mean(raws) if raws else 0.0
            om = st.mean(orths)
            kept = 100 * om / rm if abs(rm) > 1e-6 else float("nan")
            ot = nw_t(orths, h) or 0.0
            nt = nw_t(nulls, h) or 0.0
            flag = "  <<<" if abs(ot) >= 3.0 and abs(nt) < 1.5 else ""
            print(f"  {f:<18}{h:>4}{len(orths):>7}{rm:>+9.4f}"
                  f"{(nw_t(raws,h) or 0):>+8.2f}{om:>+10.4f}{ot:>+8.2f}"
                  f"{nt:>+8.2f}{kept:>6.0f}%{flag}")
        print()

    print("  Sign: values are NOT negated. A POSITIVE IC means a HIGHER value "
          "of the\n  feature predicts HIGHER forward returns -- the Novy-Marx "
          "direction for\n  gp_assets, and the value direction for bm/ep.\n")
    print("  fund_bm and fund_ep divide by market cap, so they CONTAIN PRICE "
          "and should\n  shrink more under orthogonalisation than the three "
          "pure-accounting ratios.\n  That difference is expected and is why "
          "retention is reported per feature.\n")
    print("  Bar: |orth t| >= 3.0 with a clean null. Wiring triggers a full "
          "retrain of\n  ~2,600 model files, so the bar is met BEFORE the cost "
          "is paid.")


if __name__ == "__main__":
    main()
