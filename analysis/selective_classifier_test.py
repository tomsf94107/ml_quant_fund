#!/usr/bin/env python3
"""
selective_classifier_test.py — can a SELECTION MODEL beat prob_up at knowing
when the model is right?

READ-ONLY. Writes nothing.

THE FRAMEWORK, AND WHY IT IS NOT ANOTHER FEATURE TEST
    Everything tested so far asked "does X predict returns". Seven nulls and one
    largely-spanned result. This asks a different question, from a different
    literature: selective classification (El-Yaniv & Wiener 2010; Geifman &
    El-Yaniv 2017; Fisch et al. 2022).

    A selection model sits ALONGSIDE the classifier and rejects a portion of its
    predictions to improve accuracy on the accepted subset. It does not need to
    beat the market. It needs to beat the classifier's OWN confidence at ranking
    which of its predictions will be correct.

    That distinction matters here because of an incidental finding:
    high-confidence lift was +11.2pp in the top tercile of iv_skew_snap against
    +2.7pp in the bottom, on non-overlapping intervals -- and iv_skew_snap is
    ALREADY a model feature. The model has the information and still does not
    use it to modulate its own confidence. A selection model is the standard way
    to exploit exactly that gap.

THE BASELINE THAT MUST BE BEATEN
    The naive selector is prob_up itself: accept the predictions the model is
    most confident about. Any selection model must beat that, not merely beat
    random. Reporting accuracy at low coverage without the prob_up baseline
    would make a useless model look good, because ANY sensible selector
    improves accuracy as coverage falls.

    So the output is a RISK-COVERAGE CURVE with both selectors on it.

METHOD
    Target: was the base model's directional call correct?
        correct = (prob_up >= 0.5) == (actual_up == 1)
    Features: the state variables in prediction_features -- volatility, skew,
    put/call, positioning, sector strength, plus |prob_up - 0.5| as the model's
    own confidence signal.

    Selection model: logistic regression, fitted by gradient descent on
    standardised features. Deliberately simple. A gradient-boosted selector on
    34 features and 27k rows would overfit in ways this sample cannot detect,
    and the question is whether ANY state information helps, not how much a
    flexible model can extract.

    TEMPORAL SPLIT: fitted on the earliest 60% of dates, evaluated on the last
    40%. Never on the same dates. Random splits would leak because adjacent
    dates share market conditions.

WHAT WOULD BE A REAL RESULT
    At a given coverage -- say accepting 20% of predictions -- the selection
    model's accuracy materially exceeds the prob_up baseline's, out of sample,
    with non-overlapping Wilson intervals.

    If the curves overlap, the model's own confidence is already the best
    available selector and no second layer is warranted. That is the likely
    outcome and would be an honest negative.

    python analysis/selective_classifier_test.py
"""
import argparse
import math
import random
import sqlite3
from collections import defaultdict


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 100.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, 100 * (c - s) / d), min(100.0, 100 * (c + s) / d))


def fit_logistic(X, y, epochs=400, lr=0.5, l2=1e-3):
    """Plain logistic regression by gradient descent on standardised inputs."""
    n, k = len(X), len(X[0])
    mu = [sum(r[j] for r in X) / n for j in range(k)]
    sd = [max(math.sqrt(sum((r[j] - mu[j]) ** 2 for r in X) / n), 1e-9)
          for j in range(k)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(k)] for r in X]
    w = [0.0] * k
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * k
        gb = 0.0
        for i in range(n):
            z = b + sum(w[j] * Z[i][j] for j in range(k))
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            e = p - y[i]
            gb += e
            for j in range(k):
                gw[j] += e * Z[i][j]
        b -= lr * gb / n
        for j in range(k):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
    return (w, b, mu, sd)


def predict(model, row):
    w, b, mu, sd = model
    z = b + sum(w[j] * ((row[j] - mu[j]) / sd[j]) for j in range(len(w)))
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    cols = [r[1] for r in con.execute("PRAGMA table_info(prediction_features)")]
    skip = {"id", "ticker", "prediction_date", "horizon", "created_at"}
    use = [c for c in cols if c not in skip]
    print(f"{len(use)} state features available")

    rows = con.execute(f"""
        SELECT p.prediction_date, p.prob_up, o.actual_up, {', '.join('f.'+c for c in use)}
        FROM predictions p
        JOIN outcomes o ON p.ticker=o.ticker
          AND p.prediction_date=o.prediction_date AND p.horizon=o.horizon
        JOIN prediction_features f ON f.ticker=p.ticker
          AND f.prediction_date=p.prediction_date AND f.horizon=p.horizon
        WHERE p.horizon=? AND o.actual_up IS NOT NULL AND p.prob_up IS NOT NULL
        ORDER BY p.prediction_date
    """, (args.horizon,)).fetchall()
    con.close()

    data = []
    for r in rows:
        d, prob, actual = r[0], r[1], r[2]
        feats = r[3:]
        if any(v is None for v in feats):
            continue
        correct = 1 if ((prob >= 0.5) == (actual == 1)) else 0
        # the model's own confidence, as a feature the selector may use
        data.append((d, prob, correct, list(feats) + [abs(prob - 0.5)]))
    print(f"{len(data):,} complete rows, "
          f"{data[0][0]}..{data[-1][0]}")
    if len(data) < 2000:
        raise SystemExit("too few complete rows to split and fit")

    dates = sorted({r[0] for r in data})
    cut = dates[int(len(dates) * 0.6)]
    train = [r for r in data if r[0] < cut]
    test = [r for r in data if r[0] >= cut]
    print(f"train {len(train):,} rows before {cut}; "
          f"test {len(test):,} rows from {cut}")
    base_acc = 100.0 * sum(r[2] for r in test) / len(test)
    print(f"base model accuracy on the test period: {base_acc:.1f}%\n")

    model = fit_logistic([r[3] for r in train], [r[2] for r in train])
    w, b, mu, sd = model
    ranked = sorted(zip(use + ["own_confidence"], w),
                    key=lambda x: -abs(x[1]))[:8]
    print("  selection-model weights, largest magnitude first:")
    for nm, wt in ranked:
        print(f"    {nm:<24}{wt:+.3f}")

    scored = [(predict(model, r[3]), r[1], r[2]) for r in test]
    print(f"\n  RISK-COVERAGE, out of sample ({len(scored):,} predictions)\n")
    print(f"  {'coverage':<10}{'n':>7}"
          f"{'SELECTOR acc':>15}{'95% CI':>16}"
          f"{'prob_up acc':>14}{'95% CI':>16}{'diff':>8}")
    for cov in (0.05, 0.10, 0.20, 0.30, 0.50, 1.00):
        k = max(30, int(len(scored) * cov))
        by_sel = sorted(scored, key=lambda x: -x[0])[:k]
        by_prob = sorted(scored, key=lambda x: -abs(x[1] - 0.5))[:k]
        a1 = sum(x[2] for x in by_sel)
        a2 = sum(x[2] for x in by_prob)
        l1, h1 = wilson(a1, k)
        l2, h2 = wilson(a2, k)
        print(f"  {cov:<10.0%}{k:>7}{100*a1/k:>14.1f}%"
              f"   [{l1:>5.1f},{h1:>5.1f}]{100*a2/k:>13.1f}%"
              f"   [{l2:>5.1f},{h2:>5.1f}]{100*(a1-a2)/k:>+7.1f}pp")

    print("\n  SELECTOR is the fitted selection model; prob_up is the naive "
          "baseline of\n  accepting the predictions the model is already most "
          "confident about.\n")
    print("  A real result: SELECTOR materially above prob_up at low coverage "
          "with\n  non-overlapping intervals. Overlapping curves mean the "
          "model's own\n  confidence is already the best available selector and "
          "no second layer\n  is warranted -- which is the likely outcome and "
          "an honest negative.\n")
    print("  Note the selector is given own_confidence as a feature, so it can "
          "only\n  beat the baseline by ADDING state information to it.")


if __name__ == "__main__":
    main()
