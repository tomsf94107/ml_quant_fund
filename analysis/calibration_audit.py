#!/usr/bin/env python3
"""
calibration_audit.py — would rolling recalibration actually help? Measure first.

READ-ONLY. Writes nothing. This does NOT install a fix; it tests whether the fix
is worth installing, out-of-sample, before any pipeline change.

THE DEFECT BEING TESTED
    Measured 2026-09-01 on 23,383 h=5 predictions since May: the model's
    calibrated output spans 0.275 to 0.705 across deciles while realised
    outcomes span only 0.478 to 0.571. A 4.6x overconfidence, monotone from
    +0.203 in the bottom decile to -0.134 in the top.

    The overlay is NOT the cause -- prob_raw shows 4.6x and prob_up 4.3x, so the
    seven multipliers are nearly inert. The miscalibration is in the model's own
    isotonic layer.

    Likely cause: CalibratedClassifierCV is fit on X_train, which with
    TRAIN_START=2018 and a 60/20/20 split is roughly 2018-2023. Predictions are
    made in 2026. The mapping was learned in a different regime.

WHAT RECALIBRATION CAN AND CANNOT DO
    Calibration maps are MONOTONE, so they cannot reorder predictions. AUC and
    ranking are unchanged BY CONSTRUCTION. Accuracy at a fixed 0.5 threshold is
    essentially unchanged too.

    What it fixes is what the numbers MEAN: today a "HIGH confidence" 0.70 is a
    57% event, and every threshold in the pipeline (BUY at 0.55, confidence tiers
    at 0.70/0.55) is calibrated against a scale that does not hold. That is worth
    fixing on its own terms -- but it will NOT move the accuracy figure that
    prompted this audit.

METHOD -- WALK-FORWARD, NEVER IN-SAMPLE
    For each evaluation month M: fit a calibrator on pairs whose outcomes had
    RESOLVED before M began, apply it to M's predictions, and score. The
    calibrator never sees the month it is judged on. Reported per horizon:

        Brier   mean squared error of the probability. Lower is better.
        ECE     expected calibration error -- mean |predicted - realised| across
                bins, weighted by bin size. This is the number that should fall.
        AUC     included ONLY to demonstrate it does not move. If it changes by
                more than rounding, something is wrong with the implementation.

    Three candidates plus a control:
        raw         no recalibration (the current system)
        isotonic    non-parametric, flexible, needs more data
        platt       two-parameter sigmoid, data-efficient, cannot express the
                    identity map -- so it can make things WORSE if the
                    distortion is not sigmoid-shaped
        shrink      pull probabilities toward the base rate by a fitted factor;
                    the simplest thing that could work, included so a complex
                    method has to beat a trivial one

MIN_FIT guards the instability the literature warns about: refitting on too few
samples makes the map oscillate. A month with fewer resolved pairs is skipped
rather than fitted badly.

OVERLAPPING WINDOWS
    h=3 and h=5 predictions on consecutive days share outcome days, so the
    effective sample is smaller than n suggests. That understates the standard
    error on these metrics. It does not bias the RANKING of the four methods,
    which is what this script is for.

    python analysis/calibration_audit.py --db accuracy.db
"""
import argparse
import sqlite3
import sys
from collections import defaultdict

MIN_FIT = 500            # refuse to fit a calibrator on fewer resolved pairs
K_FLOOR = 0.02           # shrink never collapses to a constant; see fit_shrink
N_BINS = 10


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def ece(pairs, n_bins=N_BINS):
    """Expected calibration error over equal-count bins."""
    d = sorted(pairs)
    size = max(1, len(d) // n_bins)
    tot, n = 0.0, len(d)
    for b in range(n_bins):
        chunk = d[b * size:(b + 1) * size] if b < n_bins - 1 else d[b * size:]
        if not chunk:
            continue
        pred = sum(x[0] for x in chunk) / len(chunk)
        real = sum(x[1] for x in chunk) / len(chunk)
        tot += len(chunk) * abs(pred - real)
    return tot / n


def auc(pairs):
    n = len(pairs)
    pos = sum(y for _, y in pairs)
    neg = n - pos
    if not pos or not neg:
        return None
    order = sorted(range(n), key=lambda i: pairs[i][0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rs = sum(ranks[i] for i in range(n) if pairs[i][1] == 1)
    return (rs - pos * (pos + 1) / 2.0) / (pos * neg)


def fit_isotonic(pairs):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit([p for p, _ in pairs], [y for _, y in pairs])
    return lambda xs: list(ir.predict(xs))


def fit_platt(pairs):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit([[p] for p, _ in pairs], [y for _, y in pairs])
    return lambda xs: [r[1] for r in lr.predict_proba([[x] for x in xs])]


def fit_shrink(pairs):
    """Pull toward the base rate by the factor that best matches realised spread.

    The simplest correction that could work: if predictions span 43 points and
    outcomes span 9, shrink by roughly 9/43. Included so isotonic and Platt have
    to beat something trivial before earning their complexity.
    """
    base = sum(y for _, y in pairs) / len(pairs)
    d = sorted(pairs)
    size = max(1, len(d) // N_BINS)
    lo_p = sum(x[0] for x in d[:size]) / size
    hi_p = sum(x[0] for x in d[-size:]) / size
    lo_r = sum(x[1] for x in d[:size]) / size
    hi_r = sum(x[1] for x in d[-size:]) / size
    k = (hi_r - lo_r) / (hi_p - lo_p) if (hi_p - lo_p) else 1.0

    # FLOOR k STRICTLY ABOVE ZERO.
    #
    # In a month where the model was INVERTED, the realised spread is negative,
    # so k goes negative. Clamping it to 0 collapses the map to a constant:
    # every prediction ties, that month's AUC becomes exactly 0.5, and the
    # method stops being strictly monotone. That is what dragged h=3's shrink
    # AUC to 0.5168 against raw's 0.5248 and tripped the correctness check.
    #
    # A tiny positive floor keeps the map strictly monotone -- ranking fully
    # preserved -- while still shrinking almost to the base rate, which is the
    # right response to a fit window showing no usable signal.
    k = max(K_FLOOR, min(1.0, k))
    return lambda xs: [base + k * (x - base) for x in xs]


METHODS = {"isotonic": fit_isotonic, "platt": fit_platt, "shrink": fit_shrink}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--from", dest="frm", default="2026-01-01")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = con.execute("""
        SELECT p.horizon, p.prediction_date, o.outcome_date, p.prob_raw, o.actual_up
        FROM predictions p JOIN outcomes o
          ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
         AND p.horizon=o.horizon
        WHERE p.prob_raw IS NOT NULL AND o.actual_up IS NOT NULL
          AND p.prediction_date >= ?
        ORDER BY p.prediction_date
    """, (args.frm,)).fetchall()
    con.close()

    by_h = defaultdict(list)
    for h, pd_, od, prob, y in rows:
        by_h[h].append((pd_, od or pd_, float(prob), int(y)))
    if not by_h:
        print("no rows")
        return

    for h in sorted(by_h):
        recs = by_h[h]
        months = sorted({r[0][:7] for r in recs})
        print(f"\n{'=' * 74}\nHORIZON {h}d   n={len(recs)}   "
              f"{months[0]}..{months[-1]}\n{'=' * 74}")

        # SCORE EACH MONTH SEPARATELY, THEN AVERAGE. Never pool.
        #
        # An earlier version pooled every month's calibrated outputs and scored
        # the pile. That is wrong: each month gets its OWN monotone map, so a
        # 0.6 in April and a 0.6 in June mean different things once calibrated.
        # Pooling scrambles the joint ranking even though each month's ranking
        # is perfectly preserved -- which showed up as AUC drifting 0.5278 ->
        # 0.5169 across methods and a nonsensical +0.530 "realised spread".
        # ECE was corrupted the same way, because its bins spanned incompatible
        # scales. Brier survived, being pointwise.
        per_month = {k: [] for k in ["raw"] + list(METHODS)}
        skipped = 0
        n_eval = 0
        for m in months[1:]:
            # fit only on pairs whose OUTCOME resolved before this month began
            fit = [(r[2], r[3]) for r in recs if r[1][:7] < m]
            test = [(r[2], r[3]) for r in recs if r[0][:7] == m]
            if len(fit) < MIN_FIT or not test:
                skipped += 1
                continue
            n_eval += len(test)
            xs = [p for p, _ in test]
            ys = [y for _, y in test]
            per_month["raw"].append(test)
            for name, fitter in METHODS.items():
                try:
                    f = fitter(fit)
                    per_month[name].append(list(zip(f(xs), ys)))
                except Exception as e:
                    print(f"  {name} failed on {m}: {type(e).__name__}: {e}")
                    per_month[name].append([])

        if not n_eval:
            print(f"  no evaluable months (skipped {skipped})")
            continue
        print(f"  evaluated on {n_eval} out-of-sample predictions "
              f"({len(months)-1-skipped} months, {skipped} skipped)")
        print(f"  metrics computed PER MONTH and size-weighted -- never pooled\n")
        print(f"  {'method':<10}{'Brier':>9}{'ECE':>9}{'AUC':>9}"
              f"   spread(pred/real)")

        def wavg(vals_weights):
            vw = [(v, w) for v, w in vals_weights if v is not None and w]
            if not vw:
                return None
            return sum(v * w for v, w in vw) / sum(w for _, w in vw)

        base_e = None
        for name in ["raw"] + list(METHODS):
            months_data = [mm for mm in per_month[name] if mm]
            if not months_data:
                continue
            b = wavg([(brier(mm), len(mm)) for mm in months_data])
            e = wavg([(ece(mm), len(mm)) for mm in months_data])
            a = wavg([(auc(mm), len(mm)) for mm in months_data])
            sps, srs = [], []
            for mm in months_data:
                d = sorted(mm)
                size = max(1, len(d) // N_BINS)
                sps.append((sum(x[0] for x in d[-size:]) / size
                            - sum(x[0] for x in d[:size]) / size, len(mm)))
                srs.append((sum(x[1] for x in d[-size:]) / size
                            - sum(x[1] for x in d[:size]) / size, len(mm)))
            sp, sr = wavg(sps), wavg(srs)
            if name == "raw":
                base_e = e
            mark = ""
            if name != "raw" and base_e:
                mark = "  ECE {:+.1f}%".format(100 * (e - base_e) / base_e)
            print(f"  {name:<10}{b:>9.4f}{e:>9.4f}"
                  f"{(a if a is not None else float('nan')):>9.4f}"
                  f"   {sp:+.3f} / {sr:+.3f}{mark}")

        print("\n  CORRECTNESS CHECK, not a metric: platt and shrink are "
              "STRICTLY monotone,\n  so their AUC must match raw EXACTLY. "
              "isotonic creates ties and may differ\n  slightly -- that is "
              "legitimate. Any other movement means the code is wrong.")


if __name__ == "__main__":
    main()
