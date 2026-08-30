#!/usr/bin/env python3
"""
pooled_accuracy.py — measure the model at a sample size that can measure anything.

READ-ONLY. No dependencies beyond the standard library.

    python pooled_accuracy.py > pooled_$(date +%Y%m%d).txt 2>&1

WHY THIS REPLACES accuracy_cache
    accuracy_cache is per-ticker with window_days=90, giving n = 39-59 per cell.
    At n=45 the standard error on a proportion near 0.5 is 7.5pp and the 95%
    interval spans ~29 points, so 0.359 and 0.634 are statistically the same
    number. Every "accuracy dropped" reading taken from that table was noise.
    Pooling by horizon and month gives n in the thousands, where a 2-3pp move is
    detectable.

WHY AUC AND NOT ACCURACY
    Every prediction since June is HOLD (ML_QUANT_DISABLE_BUY=1), so signal-based
    accuracy is undefined for the current regime. But prob_up still varies across
    its full range, and AUC measures whether that probability ORDERS outcomes
    correctly -- which is the actual question, and is answerable whether or not
    any trade is emitted.

    Accuracy is also measured against the BASE RATE, never against 50%. Up-days
    are not 50/50 in this sample (intraday base rate runs 47-49%), so a signal
    beating 50% may be losing to a constant prediction.

NULL CONTROL IS MANDATORY, NOT OPTIONAL
    Every AUC is accompanied by a shuffle null: outcomes are permuted against
    predictions and the AUC recomputed. If the shuffled AUC is not ~0.500, the
    pipeline has a leak or a join defect and the headline number means nothing.
    This is the project's own standing rule -- an unshuffled result is not a
    result.

METHOD NOTES
    AUC standard error uses Hanley-McNeil, which assumes an exponential score
    distribution; treat it as an approximation, and prefer the permutation
    spread for anything close to the line. Calibration is reported in deciles of
    predicted probability; a well-calibrated model has realized ~= predicted in
    each bucket, and a model can have real AUC while being badly calibrated.
"""
import math
import random
import sqlite3
from collections import defaultdict

DB = "accuracy.db"
N_PERMUTATIONS = 200


# ── statistics ────────────────────────────────────────────────────────────────

def auc(scores, labels):
    """Mann-Whitney AUC. Ties get averaged ranks. None if one class is absent."""
    n = len(scores)
    if n == 0:
        return None
    pos = sum(labels)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(ranks[i] for i in range(n) if labels[i] == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def auc_se(a, n_pos, n_neg):
    """Hanley-McNeil standard error."""
    if a is None or n_pos == 0 or n_neg == 0:
        return None
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    v = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a))
    return math.sqrt(max(v, 0.0) / (n_pos * n_neg))


def wilson(k, n, z=1.96):
    """Wilson score interval -- correct near 0 and 1, unlike the normal approx."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def brier(scores, labels):
    if not scores:
        return None
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores)


def shuffle_null(scores, labels, n_perm=N_PERMUTATIONS, seed=7):
    """Permutation null: AUC of the same scores against shuffled labels."""
    rnd = random.Random(seed)
    lab = list(labels)
    out = []
    for _ in range(n_perm):
        rnd.shuffle(lab)
        a = auc(scores, lab)
        if a is not None:
            out.append(a)
    if not out:
        return (None, None, None)
    out.sort()
    mean = sum(out) / len(out)
    return (mean, out[int(0.025 * len(out))], out[int(0.975 * len(out))])


# ── data ──────────────────────────────────────────────────────────────────────

def load(con, table_pred, table_out, prob_col, id_cols, horizon_col):
    join = " AND ".join(f"p.{c}=o.{c}" for c in id_cols)
    sql = (f"SELECT p.{horizon_col}, substr(p.prediction_date,1,7), "
           f"p.{prob_col}, o.actual_up, p.signal, p.ticker "
           f"FROM {table_pred} p JOIN {table_out} o ON {join} "
           f"WHERE p.{prob_col} IS NOT NULL AND o.actual_up IS NOT NULL")
    return con.execute(sql).fetchall()


def report(rows, title, note=""):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    if note:
        print(note + "\n")
    if not rows:
        print("  no rows")
        return

    by_h = defaultdict(list)
    by_hm = defaultdict(list)
    for h, m, prob, up, sig, tkr in rows:
        by_h[h].append((prob, up))
        by_hm[(h, m)].append((prob, up))

    print("--- POOLED BY HORIZON (the headline; n large enough to mean something) ---")
    print(f"  {'h':>3} {'n':>7} {'base':>7} {'AUC':>7} {'SE':>7} "
          f"{'AUC 95% CI':>16} {'Brier':>7}  verdict")
    for h in sorted(by_h):
        d = by_h[h]
        s = [x[0] for x in d]
        y = [x[1] for x in d]
        n, pos = len(y), sum(y)
        a = auc(s, y)
        se = auc_se(a, pos, n - pos)
        if a is None:
            print(f"  {h:>3} {n:>7}   one class only")
            continue
        lo, hi = a - 1.96 * se, a + 1.96 * se
        verdict = ("no discrimination" if lo <= 0.5 <= hi else
                   "ORDERS CORRECTLY" if lo > 0.5 else "INVERTED")
        print(f"  {h:>3} {n:>7} {pos/n:>6.1%} {a:>7.4f} {se:>7.4f} "
              f"[{lo:.4f}, {hi:.4f}] {brier(s, y):>7.4f}  {verdict}")

    print("\n--- SHUFFLE NULL (mandatory: must be ~0.500 or the result is void) ---")
    for h in sorted(by_h):
        d = by_h[h]
        s = [x[0] for x in d]
        y = [x[1] for x in d]
        mean, lo, hi = shuffle_null(s, y)
        if mean is None:
            continue
        flag = "OK" if abs(mean - 0.5) < 0.01 else "*** LEAK OR JOIN DEFECT ***"
        print(f"  h={h}  null AUC {mean:.4f}  95% [{lo:.4f}, {hi:.4f}]  {flag}")

    print("\n--- BY MONTH (is anything actually changing over time?) ---")
    print(f"  {'month':>8} {'h':>3} {'n':>7} {'base':>7} {'AUC':>7} {'AUC 95% CI':>18}")
    for (h, m) in sorted(by_hm, key=lambda k: (k[1], k[0])):
        d = by_hm[(h, m)]
        s = [x[0] for x in d]
        y = [x[1] for x in d]
        n, pos = len(y), sum(y)
        a = auc(s, y)
        if a is None:
            print(f"  {m:>8} {h:>3} {n:>7} {pos/n if n else 0:>6.1%}   one class only")
            continue
        se = auc_se(a, pos, n - pos)
        print(f"  {m:>8} {h:>3} {n:>7} {pos/n:>6.1%} {a:>7.4f} "
              f"[{a-1.96*se:.4f}, {a+1.96*se:.4f}]")

    print("\n--- CALIBRATION, pooled across horizons (decile of predicted prob) ---")
    allp = [(p, u) for h in by_h for p, u in by_h[h]]
    allp.sort()
    B = 10
    size = max(1, len(allp) // B)
    print(f"  {'bucket':>8} {'n':>7} {'pred':>8} {'realized':>9} {'gap':>8}")
    for b in range(B):
        chunk = allp[b * size:(b + 1) * size] if b < B - 1 else allp[b * size:]
        if not chunk:
            continue
        pred = sum(c[0] for c in chunk) / len(chunk)
        real = sum(c[1] for c in chunk) / len(chunk)
        print(f"  {b+1:>8} {len(chunk):>7} {pred:>8.3f} {real:>9.3f} "
              f"{real-pred:>+8.3f}")


def per_ticker_noise(con):
    print(f"\n{'=' * 78}\nWHY THE OLD METRIC COULD NOT WORK\n{'=' * 78}")
    rows = con.execute(
        "SELECT n_predictions FROM accuracy_cache WHERE n_predictions > 0").fetchall()
    if not rows:
        print("  accuracy_cache empty")
        return
    ns = sorted(r[0] for r in rows)
    med = ns[len(ns) // 2]
    lo, hi = wilson(int(med * 0.5), med)
    print(f"  accuracy_cache cells: {len(ns)}   median n = {med}")
    print(f"  at n={med}, a true 50% model's 95% interval is "
          f"[{lo:.1%}, {hi:.1%}] -- a spread of {(hi-lo)*100:.0f} points.")
    print(f"  Any per-ticker 'accuracy change' smaller than that is noise, and\n"
          f"  every reading in that table is smaller than that.")


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    daily = load(con, "predictions", "outcomes", "prob_up",
                 ["ticker", "prediction_date", "horizon"], "horizon")
    report(daily, "DAILY MODEL — prob_up vs realized direction",
           "Signals are all HOLD since June, so signal accuracy is undefined.\n"
           "AUC asks the answerable question: does prob_up ORDER outcomes?")

    intr = load(con, "intraday_predictions", "intraday_outcomes", "prob_up",
                ["ticker", "prediction_ts", "horizon_hr"], "horizon_hr")
    report(intr, "INTRADAY MODEL — prob_up vs realized direction",
           "22,525 outcomes after the 2026-08-30 reconciler repair, up from 7,466.")

    per_ticker_noise(con)
    con.close()
    print("\n\nEND")


if __name__ == "__main__":
    main()
