"""
recession/validation/threshold_analysis.py

Warning-threshold analysis — per horizon, on the hit/false-alarm curve.

WHY THIS EXISTS
---------------
The recession report card and alert system used a 0.5 warning threshold.
0.5 is borrowed from balanced-class classification and is WRONG for a
rare event. The economy is in expansion ~85% of the time, so a recession
model's probabilities are structurally compressed toward the low end — a
probit almost never emits 0.8+ because 0.8+ of history was not
pre-recession. The recession-forecasting literature reflects this: the
NY Fed yield-curve model has had recessions follow readings in the
20-40% range, and ~30% is a widely-cited "take it seriously" level (the
~50% level is the higher-specificity "every time it got here, a
recession followed" bar).

So the warning threshold should be LOWER than 0.5, and chosen per
horizon. But it must be chosen HONESTLY — on the hit-rate vs false-alarm
trade-off across ALL history — NOT tuned until a particular recession
(e.g. 2008) flips to "called". Fitting the threshold to a known recession
on a ~2-recession out-of-sample window is exactly the overfitting trap
the project's discipline exists to prevent.

WHAT IT DOES
------------
For each horizon, with the correct model (M1 at h=12; M2 at h=3/h=6, per
B-track), it sweeps a fine grid of thresholds and reports, at each:

  HIT RATE         — fraction of recession onsets that got a standing
                     warning (the benefit of a lower threshold).
  FALSE-ALARM RATE — fraction of warning months NOT followed by a
                     recession (the cost of a lower threshold).
  MEAN LEAD        — average months of warning before an onset.

It then RECOMMENDS a threshold per horizon by the literature-standard
criterion: maximise the KUIPER SCORE (hit rate minus false-alarm rate),
also known as Youden's J. This is what the recession-forecasting
literature uses to pick an optimal threshold (Berge-Jorda 2011; the
ROC / Kuiper-score approach). Ties are broken toward the lower threshold
(earlier, more sensitive warning).

Maximising hit-rate-minus-false-alarm-rate structurally cannot recommend
a threshold that catches nothing — a zero-hit-rate threshold scores <= 0
and never wins. (An earlier draft used a false-alarm ceiling with no
hit-rate floor and could recommend a zero-hit threshold; that was wrong
and is replaced by the Kuiper rule.)

The criterion is fixed BEFORE seeing results and uses the whole-history
trade-off — not any single recession's outcome.

HONEST LIMITS
-------------
- Hit rate is computed over very few out-of-sample recessions (~2-3).
  A hit rate of "1 of 2" is not a stable statistic. The false-alarm
  rate, computed over many calm months, is far better estimated. The
  Kuiper score depends on BOTH, so the recommended threshold inherits
  the hit rate's noise — treat the recommendation as indicative, not
  precise, and read the whole sweep, not just the picked row.
- This tool RECOMMENDS thresholds; adopting them in the report card /
  alert system is a separate, explicit change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from recession.validation.lead_time import run_lead_time_analysis


# the fine threshold grid to sweep
THRESHOLD_GRID = [round(0.05 * i, 2) for i in range(1, 19)]   # 0.05..0.90

# the horizon ladder, with the correct model per horizon (per B-track)
THRESHOLD_LADDER = [
    ("h=12 (yield curve / M1)", "T1", "h=12", "M1"),
    ("h=6 (macro / M2)", "T1", "h=6", "M2"),
    ("h=3 (macro / M2)", "T1", "h=3", "M2"),
]


# the recommendation criterion: the Kuiper score (a.k.a. Youden's J) —
# hit rate minus false-alarm rate. This is the standard the recession-
# forecasting literature uses to pick an optimal threshold (Berge-Jorda
# 2011; the ROC/Kuiper approach). Maximising it cannot recommend a
# zero-hit-rate threshold (such a threshold scores <= 0 and never wins),
# which is the failure mode of a false-alarm-ceiling-only rule.


def _recommend(sweep_rows: list[dict]) -> dict:
    """Recommend a threshold by maximising the Kuiper score
    (TPR - FPR) — the literature-standard ROC criterion.

    TPR and FPR are the proper ROC pair (both per-month, expansion
    months as the FPR denominator). Ties break toward the lower
    threshold (earlier / more sensitive warning).

    sweep_rows: list of {threshold, tpr, fpr, hit_rate, mean_lead,
                warning_reliability_false_rate, n_warning_months}.
    """
    # rows where BOTH ROC rates are defined
    usable = [r for r in sweep_rows
              if r.get("tpr") is not None
              and r.get("fpr") is not None]
    if not usable:
        return {"threshold": None, "kuiper": None,
                "reason": "no threshold has both a TPR and an FPR "
                          "defined — cannot score"}

    # Kuiper score (Youden's J) per row, from the ROC pair
    for r in usable:
        r["_kuiper"] = r["tpr"] - r["fpr"]

    best = max(usable, key=lambda r: r["_kuiper"])
    # tie-break: lowest threshold (earliest / most sensitive warning)
    top = best["_kuiper"]
    tied = sorted((r for r in usable
                   if abs(r["_kuiper"] - top) < 1e-9),
                  key=lambda r: r["threshold"])
    pick = tied[0]

    positive = pick["_kuiper"] > 0
    return {
        "threshold": pick["threshold"],
        "kuiper": pick["_kuiper"],
        "tpr": pick["tpr"],
        "fpr": pick["fpr"],
        "hit_rate": pick["hit_rate"],
        "mean_lead": pick["mean_lead"],
        "positive_skill": positive,
        "reason": ("maximises the Kuiper score TPR - FPR (the ROC "
                   "criterion); ties broken toward the lower threshold"
                   if positive else
                   "maximises the Kuiper score, but the best score is "
                   "<= 0 — at no threshold does TPR exceed FPR (no "
                   "usable skill)"),
    }


def analyze_one_horizon(
    label: str, target: str, horizon: str, model: str,
    *, db_path, min_history_year, grid: list[float],
    **wf_kwargs,
) -> dict:
    """Sweep the threshold grid for one horizon and recommend a threshold."""
    res = run_lead_time_analysis(
        target=target, horizon=horizon, model=model,
        min_history_year=min_history_year, db_path=db_path,
        thresholds=grid, **wf_kwargs,
    )
    if res.get("error") or "sweep" not in res or not res["sweep"]:
        return {"label": label, "horizon": horizon, "model": model,
                "error": res.get("error", "no sweep produced")}

    rows = []
    for thr in sorted(res["sweep"].keys()):
        s = res["sweep"][thr]
        rows.append({
            "threshold": thr,
            # per-onset hit rate (operational: did it catch the recession)
            "hit_rate": s.get("hit_rate"),
            "mean_lead": s.get("mean_lead"),
            # the ROC pair — these define the Kuiper score
            "tpr": s.get("tpr"),
            "fpr": s.get("fpr"),
            # precision-style warning reliability (NOT the FPR)
            "warning_reliability_false_rate":
                s.get("warning_reliability_false_rate"),
            "n_warning_months": s.get("n_warning_months"),
        })

    recommendation = _recommend(rows)
    return {"label": label, "horizon": horizon, "model": model,
            "n_onsets": len(res.get("onsets", [])),
            "rows": rows, "recommendation": recommendation,
            "error": None}


def run_threshold_analysis(
    *,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
    grid: Optional[list[float]] = None,
    **wf_kwargs,
) -> dict:
    """Run the threshold analysis across the horizon ladder.

    Returns {'horizons': [per-horizon dict...]}.
    """
    if grid is None:
        grid = THRESHOLD_GRID
    horizons = []
    for label, target, horizon, model in THRESHOLD_LADDER:
        horizons.append(analyze_one_horizon(
            label, target, horizon, model,
            db_path=db_path, min_history_year=min_history_year,
            grid=grid, **wf_kwargs))
    return {"horizons": horizons}


def print_threshold_analysis(result: dict) -> None:
    """Print the per-horizon threshold sweep + recommendation."""
    print("=" * 76)
    print("WARNING-THRESHOLD ANALYSIS — hit rate vs false alarms per horizon")
    print("=" * 76)
    print(f"  Recommendation: the threshold that maximises the Kuiper")
    print(f"  score (hit rate - false-alarm rate) — the literature-")
    print(f"  standard criterion. Chosen on the whole-history trade-off,")
    print(f"  NOT fitted to any single recession.")
    print()

    for h in result["horizons"]:
        print("-" * 76)
        if h.get("error"):
            print(f"  {h['label']}: ERROR — {h['error']}")
            continue
        print(f"  HORIZON: {h['label']}   "
              f"(recession onsets in OOS window: {h['n_onsets']})")
        print()
        print(f"  {'threshold':>10} {'TPR':>6} {'FPR':>6} {'Kuiper':>8} "
              f"{'hit rate':>9} {'mean lead':>10} {'warn mo':>9}")
        rec_thr = h["recommendation"].get("threshold")
        for r in h["rows"]:
            tpr = "—" if r["tpr"] is None else f"{r['tpr']:.2f}"
            fpr = "—" if r["fpr"] is None else f"{r['fpr']:.2f}"
            hr = ("—" if r["hit_rate"] is None
                  else f"{r['hit_rate']:.2f}")
            ml = ("—" if r["mean_lead"] is None
                  else f"{r['mean_lead']:.1f}")
            if r["tpr"] is not None and r["fpr"] is not None:
                ks = f"{r['tpr'] - r['fpr']:+.2f}"
            else:
                ks = "—"
            mark = "  <== recommended" if r["threshold"] == rec_thr else ""
            print(f"  {r['threshold']:>10.2f} {tpr:>6} {fpr:>6} "
                  f"{ks:>8} {hr:>9} {ml:>10} "
                  f"{r['n_warning_months']:>9}{mark}")
        print()
        rec = h["recommendation"]
        if rec.get("threshold") is None:
            print(f"  -> RECOMMENDATION: none — {rec['reason']}")
        else:
            tag = ("" if rec.get("positive_skill")
                   else "  [WARNING: best Kuiper score <= 0 — no usable "
                        "skill at any threshold]")
            print(f"  -> RECOMMENDED THRESHOLD: {rec['threshold']:.2f}{tag}")
            print(f"     {rec['reason']}")
            print(f"     at this threshold: Kuiper score "
                  f"{rec.get('kuiper'):+.2f} (TPR "
                  f"{rec.get('tpr'):.2f}, FPR {rec.get('fpr'):.2f}); "
                  f"per-onset hit rate {rec.get('hit_rate')}, mean lead "
                  f"{rec.get('mean_lead')} months")
        print()

    print("=" * 76)
    print("  READING THIS:")
    print("  - TPR (true-positive rate): of eligible pre-recession")
    print("    months, the fraction the model flagged. FPR (false-")
    print("    positive rate): of eligible expansion months, the")
    print("    fraction wrongly flagged — this is 1 - specificity, the")
    print("    literature-standard rate (denominator = all calm months,")
    print("    so a low threshold does not inflate it).")
    print("  - Kuiper score = TPR - FPR. A LOWER threshold raises both")
    print("    TPR and FPR; the recommended threshold maximises their")
    print("    difference. In-recession and post-recession recovery")
    print("    months are excluded from both denominators.")
    print("  - TPR rests on very few OOS recessions (~2-3) — noisy; the")
    print("    Kuiper score inherits that. Read the whole sweep, not")
    print("    just the picked row. The per-onset hit rate is shown as")
    print("    operational context (did each recession get caught).")
    print("  - These are RECOMMENDATIONS. Adopting them in the report")
    print("    card / alert system is a separate, explicit change.")
    print("=" * 76)
