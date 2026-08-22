#!/usr/bin/env python3
"""
c1_holdout.py -- do the four decorrelated candidates combine? COLD HOLDOUT.

WHY (2026-08-22)
  c1_combiner.py returned NO-SHIP, and the reason was the inputs, not the method:
        mom    gp    op    ep
  mom  1.00  0.59  0.61  0.57
  gp   0.59  1.00  0.81  0.70
  op   0.61  0.81  1.00  0.86
  ep   0.57  0.70  0.86  1.00
  At rho ~ 0.8 you hold one bet in three costumes and IR = IC x sqrt(breadth)
  cannot help.

  After fixing three filter bugs in the alpha gate (mono plumbing, tie guard,
  market-wide threshold + sampling), the survivors reduce to four families whose
  pairwise rank correlation is 0.10-0.29:
        inst_signed_flow_30d  Sharpe 0.539   (institutional flow, ~47 names/date)
        volatility_10d        Sharpe 0.312   (5 redundant bases collapsed to 1)
        rev_x_low52w          Sharpe 0.177   (fundamental)
        short_pct_float       Sharpe 0.117   (the SI feature, off-horizon here)
  First decorrelated inputs this system has had.

  EVERY ONE OF THOSE NUMBERS IS IN-SAMPLE on 636 dates, and the candidates were
  chosen BECAUSE they looked best on those dates. The holdout is the only part
  not yet fit.

WHY NOT REUSE c1_combiner.py DIRECTLY
  It reads data/qv_books.csv: MONTHLY book returns from 2017-02 (113 months).
  These alphas were validated at h=5. Monthly books would test a different
  horizon, and 636 daily dates is only ~30 months -- a 516/120 split leaves 6
  test months. So books are built at h=5 with 5-day rebalancing (~127
  rebalances), reusing c1_combiner's LOGIC (equal / inv-vol / HRP,
  walk-forward), not its file.

PRE-REGISTERED (fixed before running, per SYSTEM_AUDIT rev2 section 7)
  train  : first 516 dates          test: last 120 dates, untouched
  book   : top-decile long, inverse trailing-20d-vol weighted, net 10bps on
           measured turnover, 5-day hold
  weights: equal, inverse-vol, HRP -- all fit on TRAIN ONLY
  PASS   : combined TEST Sharpe > best single-stream TEST Sharpe
  FAIL   : anything else, including "close". No re-picking inputs on failure.

USAGE
  python scripts/c1_holdout.py
  python scripts/c1_holdout.py --test-dates 120 --cost-bps 10
  python scripts/c1_holdout.py --alphas "a__x,b__y" --csv books.csv
"""
import argparse
import csv as _csv
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))

DEFAULT_ALPHAS = {
    "inst_flow": "inst_signed_flow_30d__ts_argmax__w20",
    "vol":       "volatility_10d__ts_delta__w10",
    "rev":       "rev_x_low52w__cs_rank",
    "short":     "short_pct_float__cs_rank",
}


def sharpe(xs, per_year):
    n = len(xs)
    if n < 3:
        return float("nan")
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    return (mu / sd) * math.sqrt(per_year) if sd > 0 else float("nan")


def maxdd(xs):
    eq = 1.0
    peak = 1.0
    dd = 0.0
    for x in xs:
        eq *= (1.0 + x)
        peak = max(peak, eq)
        dd = min(dd, eq / peak - 1.0)
    return dd


def hrp_weights(cols, rets):
    """Hierarchical risk parity; falls back to inverse-vol if scipy is absent."""
    import numpy as np
    X = np.array([rets[c] for c in cols], dtype=float)
    cov = np.cov(X)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
    except Exception:
        v = np.sqrt(np.diag(cov))
        w = 1.0 / np.where(v > 0, v, np.inf)
        return dict(zip(cols, w / w.sum()))
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    dist = np.sqrt(0.5 * (1 - np.clip(corr, -1, 1)))
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))
    w = {i: 1.0 for i in order}
    clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) < 2:
                continue
            h = len(cl) // 2
            a, b = cl[:h], cl[h:]
            va = 1.0 / np.sum(1.0 / np.diag(cov)[a])
            vb = 1.0 / np.sum(1.0 / np.diag(cov)[b])
            alpha = 1 - va / (va + vb)
            for i in a:
                w[i] *= alpha
            for i in b:
                w[i] *= (1 - alpha)
            nxt += [a, b]
        clusters = nxt
    tot = sum(w.values())
    return {cols[i]: w[i] / tot for i in range(len(cols))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default=None, help="comma list name=alpha_column")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--test-dates", type=int, default=120)
    ap.add_argument("--decile", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--min-names", type=int, default=30)
    ap.add_argument("--vol-window", type=int, default=20)
    ap.add_argument("--csv")
    ap.add_argument("--root")
    args = ap.parse_args()

    global ROOT
    if args.root:
        ROOT = os.path.expanduser(args.root)
    sys.path.insert(0, ROOT)
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from analysis.alpha_fitness import _load_panel, _merge_outcomes

    cand = DEFAULT_ALPHAS
    if args.alphas:
        cand = {}
        for tok in args.alphas.split(","):
            k, _, v = tok.partition("=")
            cand[k.strip() or v.strip()] = (v or k).strip()

    m = _merge_outcomes(_load_panel(Path(ROOT) / "data" / "alpha_panel"),
                        Path(ROOT) / "accuracy.db", args.horizon)
    missing = [v for v in cand.values() if v not in m.columns]
    if missing:
        sys.exit(f"FATAL: alpha column(s) not in panel: {missing}")

    # trailing vol per ticker-date, TRAILING ONLY (no look-ahead)
    import sqlite3
    con = sqlite3.connect(os.path.join(ROOT, "prices.db"), timeout=30)
    px = defaultdict(list)
    for t, d, c in con.execute("SELECT ticker,d,close FROM raw_bars WHERE close>0 "
                               "ORDER BY ticker,d"):
        px[t.upper()].append((d, float(c)))
    con.close()
    vol = {}
    W = args.vol_window
    for t, rows in px.items():
        cs = [c for _, c in rows]
        ds = [d for d, _ in rows]
        rets = [0.0] + [cs[i] / cs[i - 1] - 1.0 for i in range(1, len(cs))]
        for i in range(W, len(cs)):
            w = rets[i - W:i]            # strictly prior to date i
            mu = sum(w) / W
            sd = math.sqrt(sum((x - mu) ** 2 for x in w) / (W - 1))
            if sd > 0:
                vol[(t, ds[i])] = sd

    dates = sorted(m["date"].unique())
    print(f"# c1_holdout  horizon={args.horizon}  panel dates={len(dates)}  "
          f"({str(dates[0])[:10]} .. {str(dates[-1])[:10]})")
    if len(dates) <= args.test_dates + 60:
        sys.exit(f"FATAL: only {len(dates)} dates; need > {args.test_dates+60}")

    # 5-day rebalances -> non-overlapping holds
    reb = dates[::args.horizon]
    split_date = dates[len(dates) - args.test_dates]
    print(f"# rebalances={len(reb)}  test starts {str(split_date)[:10]} "
          f"(last {args.test_dates} dates held out)")

    from analysis.book_build import build_books
    books, rdates, diag = build_books(m, cand, vol, reb,
                                      decile=args.decile,
                                      cost_bps=args.cost_bps,
                                      min_names=args.min_names)
    print(f"# book diag: zero-vol names {dict(diag['zero_wt'])} of "
          f"{dict(diag['names'])} | eq-wt fallback {dict(diag['eq_fallback'])} "
          f"| skipped thin-date {diag['skip_thin_date']} "
          f"thin-alpha {diag['skip_thin_alpha']}")

    n_reb = len(rdates)
    per_year = 252.0 / args.horizon
    tr = [i for i, d in enumerate(rdates) if d < split_date]
    te = [i for i, d in enumerate(rdates) if d >= split_date]
    print(f"# usable rebalances={n_reb}  train={len(tr)}  TEST={len(te)}\n")
    if len(te) < 8 or len(tr) < 20:
        sys.exit(f"FATAL: split too thin (train {len(tr)}, test {len(te)})")

    cols = list(cand.keys())
    print("TRAIN correlation (the C1 premise):")
    import numpy as np
    A = np.array([[books[c][i] for i in tr] for c in cols])
    C = np.corrcoef(A)
    print("        " + "".join(f"{c:>11}" for c in cols))
    for i, c in enumerate(cols):
        print(f"  {c:<6}" + "".join(f"{C[i][j]:>11.3f}" for j in range(len(cols))))
    print()

    tr_rets = {c: [books[c][i] for i in tr] for c in cols}
    te_rets = {c: [books[c][i] for i in te] for c in cols}

    schemes = {}
    schemes["equal"] = {c: 1.0 / len(cols) for c in cols}
    iv = {c: 1.0 / (np.std(tr_rets[c], ddof=1) or 1e9) for c in cols}
    s = sum(iv.values())
    schemes["invvol"] = {c: v / s for c, v in iv.items()}
    schemes["hrp"] = hrp_weights(cols, tr_rets)

    print(f"{'stream':<12}{'TRAIN Sh':>10}{'TEST Sh':>10}{'TEST maxDD':>12}"
          f"{'TEST ann%':>11}")
    print("-" * 55)
    best_single_te = -9e9
    best_single = None
    for c in cols:
        s_tr = sharpe(tr_rets[c], per_year)
        s_te = sharpe(te_rets[c], per_year)
        ann = (sum(te_rets[c]) / len(te_rets[c])) * per_year * 100
        print(f"{c:<12}{s_tr:>10.2f}{s_te:>10.2f}{maxdd(te_rets[c])*100:>11.1f}%{ann:>10.1f}%")
        if s_te > best_single_te:
            best_single_te, best_single = s_te, c
    print("-" * 55)
    results = {}
    for name, w in schemes.items():
        ctr = [sum(w[c] * tr_rets[c][i] for c in cols) for i in range(len(tr))]
        cte = [sum(w[c] * te_rets[c][i] for c in cols) for i in range(len(te))]
        s_tr, s_te = sharpe(ctr, per_year), sharpe(cte, per_year)
        ann = (sum(cte) / len(cte)) * per_year * 100
        results[name] = s_te
        print(f"{name:<12}{s_tr:>10.2f}{s_te:>10.2f}{maxdd(cte)*100:>11.1f}%{ann:>10.1f}%")
        print(f"{'':<12}weights " + " ".join(f"{c}={w[c]:.2f}" for c in cols))

    print()
    print("=" * 66)
    print("PRE-REGISTERED GATE: combined TEST Sharpe > best single TEST Sharpe")
    print(f"  best single stream : {best_single} @ {best_single_te:+.2f}")
    win = [(k, v) for k, v in results.items() if v > best_single_te]
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {k:<10} {v:+.2f}   {'PASS' if v > best_single_te else 'fail'}")
    print()
    if win:
        print(f"VERDICT: PASS — {', '.join(k for k, _ in win)} beat the best single "
              f"stream out of sample.")
        print(f"  {len(te)} test rebalances is a SMALL sample. Before sizing:")
        print("  re-run with a different split, check regime dependence, and")
        print("  confirm the correlations hold in the test window too.")
    else:
        print("VERDICT: FAIL — no weighting beat the best single stream out of "
              "sample.")
        print("  Decorrelation was necessary, not sufficient: four weak books")
        print("  that do not individually survive cannot be rescued by mixing.")
        print("  Do NOT re-pick inputs and re-run; that is fitting the holdout.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w_ = _csv.writer(f)
            w_.writerow(["date", "split"] + cols)
            for i, d in enumerate(rdates):
                w_.writerow([str(d)[:10], "test" if d >= split_date else "train"]
                            + [round(books[c][i], 6) for c in cols])
        print(f"\n# wrote {args.csv}")
    return 0 if win else 1


if __name__ == "__main__":
    sys.exit(main())
