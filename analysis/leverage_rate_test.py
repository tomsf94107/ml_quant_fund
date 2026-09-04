#!/usr/bin/env python3
"""
leverage_rate_test.py — does BALANCE-SHEET leverage add what rate_beta could not?

READ-ONLY. Writes nothing.

WHY THIS IS THE RIGHT TEST AFTER EIGHT NULLS
    rate_beta -- the rolling beta of a stock's returns to yield changes -- looked
    promising: raw IC t=+3.80 at h=20 with a clean null, and the tercile spread
    survived a falling-yield regime, ruling out pure duration exposure.

    It then failed orthogonalisation. Regressed on beta_60d, momentum, RSI, ATR
    and sector strength, it retained only 37-41% of its IC and the residual ran
    t=+1.06 to +1.52. The reason is structural: rate_beta is DERIVED FROM
    RETURNS, and the model already carries ~100 return-based features. Anything
    built from prices is likely already in there in some form.

    Assets / StockholdersEquity is NOT a price. It comes from a balance sheet,
    changes only when a filing lands, and cannot be a linear combination of
    momentum, volatility and beta. That makes it the one construction in this
    sequence that CAN survive orthogonalisation on structural grounds -- and if
    it does not, that is a much stronger negative than the others, because it
    would mean price features encode even the balance sheet.

THE MECHANISM BEING TESTED
    Twelve CRWV research reports name it: ~$35B of debt where interest expense
    swamps operating income, beta ~3.2, and a -12.1% session on the day the
    30-year hit a two-decade high. Meanwhile a multibillion-dollar customer win
    (8/20) and an NVDA blowout (8/27) both failed to lift the stock.
    "Demand keeps arriving, rates keep pricing it."

    CRWV's equity went from -$414M at end-2024 to $5,024M by mid-2026 against
    ~$35B of debt -- roughly 8x assets-to-equity.

    Leverage alone is a STANDING characteristic: CRWV was always levered. The
    signal should be leverage TIMES an actual rate move, which is why 8/18 was
    -12.1% and most days were not.

CONSTRUCTIONS
    leverage        Assets / StockholdersEquity, from the latest filing whose
                    filed_date is on or before the evaluation date
    lev_x_dy        leverage * the trailing 20-day change in the 10-year yield
    lev_x_dy5       leverage * the trailing 5-day change

    Each tested raw AND orthogonalised against the same controls that killed
    rate_beta.

POINT-IN-TIME
    xbrl_facts stores filed_date and its own schema comment calls it "THE date
    that matters". Facts are used only from filed_date forward. A single filing
    carries comparatives for several prior periods, all sharing one filed_date;
    the most recent period_end within the latest available filing is used.

    This is the discipline whose absence voided this project's PEAD work, where
    report_date turned out to be fiscal-period-end rather than announcement.

KNOWN TRAPS, HANDLED EXPLICITLY
    1. NEGATIVE EQUITY makes the ratio meaningless and explosive -- CRWV was
       -$414M pre-IPO. Rows with equity <= 0 are DROPPED, not floored, and the
       count is reported.
    2. FINANCIALS AND REITS carry structurally high leverage that is not
       distress. Without a sector map this cannot be neutralised here, so
       sector_rel_ret is included among the controls as a partial proxy and the
       limitation is stated rather than hidden.
    3. Leverage is highly persistent, so consecutive dates are close to the same
       cross-section. Per-date IC with Newey-West handles the autocorrelation.

    python analysis/leverage_rate_test.py
"""
import argparse
import math
import random
import sqlite3
import statistics as st
from collections import defaultdict

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
    n = len(Y)
    k = len(X[0]) + 1
    A = [[1.0] + list(r) for r in X]
    XtX = [[sum(A[i][a] * A[i][b] for i in range(n)) for b in range(k)]
           for a in range(k)]
    for a in range(k):
        XtX[a][a] += ridge
    Xty = [sum(A[i][a] * Y[i] for i in range(n)) for a in range(k)]
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
            if r != c and M[r][c]:
                f = M[r][c]
                for j in range(c, k + 1):
                    M[r][j] -= f * M[c][j]
    b = [M[a][k] for a in range(k)]
    return [Y[i] - sum(b[a] * A[i][a] for a in range(k)) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fundamentals-db", default="fundamentals.db")
    ap.add_argument("--prices-db", default="prices.db")
    ap.add_argument("--warning-db", default="warning.db")
    ap.add_argument("--accuracy-db", default="accuracy.db")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--min-names", type=int, default=40)
    args = ap.parse_args()
    HOR = (5, 20)

    # ---- leverage, PIT ----
    fc = sqlite3.connect(f"file:{args.fundamentals_db}?mode=ro", uri=True)
    facts = defaultdict(lambda: defaultdict(dict))   # tkr -> filed -> tag -> (pe,val)
    for tk, tag, pe, fd, v in fc.execute(
            "SELECT ticker, tag, period_end, filed_date, value FROM xbrl_facts "
            "WHERE tag IN ('Assets','StockholdersEquity') AND value IS NOT NULL"):
        fd = str(fd)[:10]
        cur = facts[tk][fd].get(tag)
        if cur is None or str(pe) > cur[0]:
            facts[tk][fd][tag] = (str(pe), v)
    fc.close()

    lev_hist = defaultdict(list)     # tkr -> [(filed_date, leverage)]
    neg_eq = 0
    for tk, byfd in facts.items():
        for fd in sorted(byfd):
            a = byfd[fd].get("Assets")
            e = byfd[fd].get("StockholdersEquity")
            if not a or not e:
                continue
            if e[1] <= 0:
                neg_eq += 1
                continue
            lev_hist[tk].append((fd, a[1] / e[1]))
    print(f"leverage: {len(lev_hist)} tickers with at least one filing")
    print(f"  {neg_eq} filings dropped for equity <= 0 "
          f"(CRWV was -$414M pre-IPO)")
    if "CRWV" in lev_hist:
        print(f"  CRWV: " + ", ".join(f"{d}={v:.1f}x"
                                      for d, v in lev_hist["CRWV"][-4:]))

    def lev_asof(tk, d):
        h = lev_hist.get(tk)
        if not h:
            return None
        best = None
        for fd, v in h:
            if fd <= d:
                best = v
            else:
                break
        return best

    # ---- yields ----
    wc = sqlite3.connect(f"file:{args.warning_db}?mode=ro", uri=True)
    y = {str(d)[:10]: v for d, v in wc.execute(
        "SELECT obs_date, value FROM data_vintages WHERE series_id='DGS10' "
        "AND obs_date >= ?", (args.start,))}
    wc.close()
    yd = sorted(y)
    dy20 = {yd[i]: y[yd[i]] - y[yd[i - 20]] for i in range(20, len(yd))}
    dy5 = {yd[i]: y[yd[i]] - y[yd[i - 5]] for i in range(5, len(yd))}

    px = sqlite3.connect(f"file:{args.prices_db}?mode=ro", uri=True)
    close = defaultdict(dict)
    for t, d, c in px.execute(
            "SELECT ticker, d, close FROM raw_bars WHERE d >= ? AND close>0",
            (args.start,)):
        close[t][d] = c
    px.close()
    fwd = {h: {} for h in HOR}
    for t, s in close.items():
        ds = sorted(s)
        for h in HOR:
            for i in range(len(ds) - h):
                a, b = s[ds[i]], s[ds[i + h]]
                if a and b and abs((b - a) / a) < 0.8:
                    fwd[h][(t, ds[i])] = (b - a) / a

    ac = sqlite3.connect(f"file:{args.accuracy_db}?mode=ro", uri=True)
    have = [r[1] for r in ac.execute("PRAGMA table_info(prediction_features)")]
    ctrl = [c for c in CONTROLS if c in have]
    feats = defaultdict(dict)
    for row in ac.execute(
            f"SELECT ticker, prediction_date, {', '.join(ctrl)} "
            f"FROM prediction_features WHERE horizon=5"):
        if all(v is not None for v in row[2:]):
            feats[row[1]][row[0]] = list(row[2:])
    ac.close()
    print(f"controls: {', '.join(ctrl)}   ({len(feats)} dates with features)\n")

    dates = sorted({d for t in close for d in close[t] if d >= args.start})[::5]
    names = ["leverage", "lev_x_dy", "lev_x_dy5"]
    rnd = random.Random(31)

    print(f"RAW IC — all {len(dates)} dates\n")
    print(f"  {'construction':<14}{'h':>4}{'dates':>7}{'mean IC':>10}"
          f"{'NW t':>8}{'null t':>9}")
    for k in names:
        for h in HOR:
            ics, nl = [], []
            for d in dates:
                obs = []
                for t in close:
                    L = lev_asof(t, d)
                    if L is None or (t, d) not in fwd[h]:
                        continue
                    if k == "leverage":
                        v = -L
                    elif k == "lev_x_dy":
                        c = dy20.get(d)
                        if c is None:
                            continue
                        v = -L * c
                    else:
                        c = dy5.get(d)
                        if c is None:
                            continue
                        v = -L * c
                    obs.append((v, fwd[h][(t, d)]))
                if len(obs) < args.min_names:
                    continue
                r = spearman(obs)
                if r is not None:
                    ics.append(r)
                ys = [o[1] for o in obs]
                rnd.shuffle(ys)
                rn = spearman([(obs[i][0], ys[i]) for i in range(len(ys))])
                if rn is not None:
                    nl.append(rn)
            if len(ics) < 20:
                print(f"  {k:<14}{h:>4}{len(ics):>7}   too few dates")
                continue
            t_ = nw_t(ics, h) or 0.0
            nt = nw_t(nl, h) or 0.0
            flag = "  <<<" if abs(t_) >= 3.0 and abs(nt) < 1.5 else ""
            print(f"  {k:<14}{h:>4}{len(ics):>7}{st.mean(ics):>+10.4f}"
                  f"{t_:>+8.2f}{nt:>+9.2f}{flag}")
        print()

    print("ORTHOGONALISED against the controls that killed rate_beta\n")
    print(f"  {'construction':<14}{'h':>4}{'dates':>7}{'raw IC':>9}"
          f"{'orth IC':>10}{'orth t':>8}{'retained':>10}")
    for k in names:
        for h in HOR:
            raws, orths, nl = [], [], []
            for d in sorted(feats):
                cand = []
                for t in feats[d]:
                    L = lev_asof(t, d)
                    if L is None or (t, d) not in fwd[h]:
                        continue
                    if k == "leverage":
                        v = -L
                    elif k == "lev_x_dy":
                        c = dy20.get(d)
                        if c is None:
                            continue
                        v = -L * c
                    else:
                        c = dy5.get(d)
                        if c is None:
                            continue
                        v = -L * c
                    cand.append((t, v))
                if len(cand) < args.min_names:
                    continue
                Y = [v for _, v in cand]
                X = [feats[d][t] for t, _ in cand]
                R = [fwd[h][(t, d)] for t, _ in cand]
                rr = spearman(list(zip(Y, R)))
                if rr is not None:
                    raws.append(rr)
                res = ols_residuals(Y, X)
                if res is None:
                    continue
                ro = spearman(list(zip(res, R)))
                if ro is not None:
                    orths.append(ro)
                sh = R[:]
                rnd.shuffle(sh)
                rn = spearman(list(zip(res, sh)))
                if rn is not None:
                    nl.append(rn)
            if len(orths) < 20:
                print(f"  {k:<14}{h:>4}{len(orths):>7}   too few dates")
                continue
            keep = (st.mean(orths) / st.mean(raws)) if raws and st.mean(raws) else 0
            print(f"  {k:<14}{h:>4}{len(orths):>7}{st.mean(raws):>+9.4f}"
                  f"{st.mean(orths):>+10.4f}{(nw_t(orths,h) or 0):>+8.2f}"
                  f"{100*keep:>9.0f}%")
        print()

    print("  Sign: negated so MORE leverage is MORE NEGATIVE. A POSITIVE IC "
          "means\n  higher leverage predicts LOWER forward returns.\n")
    print("  rate_beta retained 37-41% and its residual ran t=+1.06 to +1.52. "
          "Leverage\n  is a balance-sheet fact and CANNOT be a linear "
          "combination of returns-based\n  controls, so high retention here "
          "would be structurally meaningful.\n")
    print("  LIMITATION: financials and REITs carry structurally high leverage "
          "that is\n  not distress. Without a sector map this is not "
          "neutralised; sector_rel_ret\n  is a partial proxy only. A positive "
          "result would need a sector-neutral rerun\n  before being believed.")


if __name__ == "__main__":
    main()
