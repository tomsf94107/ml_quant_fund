#!/usr/bin/env python3
"""
book_build.py -- shared top-decile book construction.

Extracted from scripts/c1_holdout.py main() on 2026-08-22 so the beta sweep and
the holdout build IDENTICAL books. Two copies of decile logic drift; one copy
plus a positive control does not.

Behaviour preserved deliberately, warts included, so the control is exact:
  D. turnover is Jaccard on NAME SETS, not weights. Inv-vol weights shift on
     retained names every rebalance, so realised turnover exceeds this.
     Directionally optimistic on cost.
  B. a min_names break discards the rebalance for EVERY alpha. Correct for a
     common-sample comparison; WRONG for a sweep -- call once per alpha there
     and read len(rdates) per alpha.

ADDED, non-behavioural: zero-weight accounting (C). A name with missing
trailing vol gets weight 0.0 and silently leaves the book; the equal-weight
fallback fires only if EVERY name is missing. Counted, not acted on.

DROPPED: `ok = ok and True` (E). Dead -- ok is True on every path that reaches
it, since the only False assignment is followed by break.
"""
from collections import defaultdict


def _leg(sl, vol, d, diag, k):
    tks = list(sl["ticker"]); rr = list(sl["actual_return"])
    wv = []
    for t in tks:
        v = vol.get((t.upper(), d)) or vol.get((t.upper(), str(d)[:10]))
        wv.append(1.0 / v if v and v > 0 else 0.0)
    diag["zero_wt"][k] += sum(1 for w in wv if w == 0.0)
    diag["names"][k] += len(tks)
    if sum(wv) <= 0:
        wv = [1.0] * len(tks); diag["eq_fallback"][k] += 1
    sw = sum(wv)
    return sum(w * r for w, r in zip(wv, rr)) / sw, frozenset(tks)


def build_books(m, cand, vol, reb, decile=10, cost_bps=10.0, min_names=30,
                legs="long"):
    """m: outcome-merged panel. cand: {display_key: alpha_column}.
    vol: {(TICKER, date): trailing_vol}. reb: rebalance dates.
    Returns (books, rdates, diag)."""
    books = defaultdict(list)
    rdates = []
    diag = {"zero_wt": defaultdict(int), "names": defaultdict(int),
            "eq_fallback": defaultdict(int),
            "break_by": defaultdict(int),
            "skip_thin_date": 0, "skip_thin_alpha": 0}
    prev = {k: frozenset() for k in cand}
    for d in reb:
        g = m[m["date"] == d]
        if len(g) < min_names:
            diag["skip_thin_date"] += 1
            continue
        row = {}
        staged = {}
        ok = True
        for k, colname in cand.items():
            sub = g[[colname, "actual_return", "ticker"]].dropna()
            if len(sub) < min_names:
                diag["break_by"][k] += 1
                ok = False
                break
            sub = sub.sort_values(colname)
            n = len(sub)
            cut = max(1, n // decile)
            gross, cur = _leg(sub.iloc[n - cut:], vol, d, diag, k)
            if legs == "ls":
                gs, cs_ = _leg(sub.iloc[:cut], vol, d, diag, k)
                gross -= gs
                cur = cur | cs_
            to = 1.0 if not prev[k] else len(cur ^ prev[k]) / max(len(cur | prev[k]), 1)
            staged[k] = cur
            row[k] = gross - to * cost_bps / 10000.0
        if not ok or len(row) != len(cand):
            if not ok:
                diag["skip_thin_alpha"] += 1
            continue
        prev.update(staged)
        for k, v in row.items():
            books[k].append(v)
        rdates.append(d)
    return dict(books), rdates, diag
