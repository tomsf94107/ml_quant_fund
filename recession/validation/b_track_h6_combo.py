"""
recession/validation/b_track_h6_combo.py

B-track follow-up — the h=6 combination experiment.

PRE-REGISTRATION (written before the run)
-----------------------------------------
B-track's forecast encompassing test reported, at h=6, for every macro
challenger vs M1: "both forecasts carry independent information — neither
encompasses the other; a combination could help." That is a concrete,
testable claim. This module tests it.

THE QUESTION
  At h=6, does an equal-weight combination of M1 (yield curve) and M2
  (macro logit) beat the better of M1-alone and M2-alone, out-of-sample?

PRE-REGISTERED DESIGN
  - Combination method: EQUAL-WEIGHT average of the two models'
    recession probabilities. Fixed in advance. Equal weighting is chosen
    over a fitted weighting because the forecast-combination literature
    (e.g. Smith & Wallis 2009; the Cleveland Fed regional-sentiment study)
    repeatedly finds equal-weighted averages beat estimated-weight
    combinations out-of-sample — the "forecast combination puzzle". A
    fitted weight would also add a tuned parameter and a leakage surface.
  - Horizon: h=6 only. h=3 is already a clear M2 win (M2 encompasses the
    curve); h=12 is the established yield-curve cell. h=6 is the only
    horizon where the encompassing test said a combination could help.
  - Success criterion: the SAME pre-registered bar as B-track — the
    combination "wins" only if its walk-forward mean fold AUC exceeds the
    BETTER of M1 and M2 by more than the seed-noise band (0.03).
  - Confirmatory: the encompassing test, combination vs the better single.

PRE-REGISTERED HYPOTHESIS
  HC1: the M1+M2 equal-weight combination beats the better single model
       at h=6 on the primary criterion. (The encompassing test predicts
       this; this module tests it OOS.)

HONEST NULL OUTCOME
  If the combination does not clear the band, that is a valid finding:
  it would mean the encompassing test's "independent information" did not
  translate into a better OOS forecast — the same in-sample-vs-OOS gap
  the nested test already showed at h=12. Reported as-is.

No deviation from this pre-registration without a dated amendment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from recession.validation.walk_forward import walk_forward
from recession.validation.b_track import (
    encompassing_test, _pooled_oos, _common_axis,
    SEED_NOISE_BAND, B_TRACK_FEATURES,
)
from recession.models.m1_probit import M1Probit, M1_FEATURES
from recession.models.m2_logit import M2Logit, M2_FEATURES


# pre-registered: h=6 only
H6_COMBO_HORIZON = "h=6"


class M1M2EqualWeight:
    """Equal-weight combination of M1 and M2, behind the RecessionModel
    protocol. predict_proba averages the two models' recession
    probabilities. Each sub-model is fit on the columns it is designed
    for (M1 on T10Y3M only; M2 on the 4-feature set)."""

    def __init__(self) -> None:
        self._m1: Optional[M1Probit] = None
        self._m2: Optional[M2Logit] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "M1M2EqualWeight":
        self._m1 = M1Probit()
        self._m2 = M2Logit(C=1.0)
        m1_cols = [c for c in M1_FEATURES if c in X.columns]
        m2_cols = [c for c in M2_FEATURES if c in X.columns]
        self._m1.fit(X[m1_cols], y)
        self._m2.fit(X[m2_cols], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        m1_cols = [c for c in M1_FEATURES if c in X.columns]
        m2_cols = [c for c in M2_FEATURES if c in X.columns]
        p1 = np.asarray(self._m1.predict_proba(X[m1_cols]), dtype=float)
        p2 = np.asarray(self._m2.predict_proba(X[m2_cols]), dtype=float)
        return 0.5 * (p1 + p2)


def run_h6_combination(
    *,
    target: str = "T1",
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
    **walk_forward_kwargs,
) -> dict:
    """Run the pre-registered h=6 combination experiment.

    Validates M1, M2, and the equal-weight combination on one shared fold
    axis, applies the pre-registered seed-noise band against the better
    single model, and runs the confirmatory encompassing test.

    Returns {'m1','m2','combo' : WalkForwardResult, 'better_single',
             'combo_edge', 'beats', 'encompassing', 'verdict'}.
    """
    horizon = H6_COMBO_HORIZON
    axis = _common_axis(target, horizon, db_path, min_history_year)
    common = dict(target=target, horizon=horizon,
                  min_history_year=min_history_year, db_path=db_path,
                  restrict_to_months=axis, **walk_forward_kwargs)

    m1 = walk_forward(
        model_factory=M1Probit, feature_subset=M1_FEATURES,
        model_columns=M1_FEATURES, model_name="M1 (h=6)", **common,
    )
    m2 = walk_forward(
        model_factory=lambda: M2Logit(C=1.0), feature_subset=M2_FEATURES,
        model_columns=M2_FEATURES, model_name="M2 (h=6)", **common,
    )
    combo = walk_forward(
        model_factory=M1M2EqualWeight, feature_subset=B_TRACK_FEATURES,
        model_columns=B_TRACK_FEATURES,
        model_name="M1+M2 equal-weight (h=6)", **common,
    )

    a1 = m1.mean_fold_auc
    a2 = m2.mean_fold_auc
    ac = combo.mean_fold_auc

    if a1 is None or a2 is None or ac is None:
        return {"m1": m1, "m2": m2, "combo": combo,
                "error": "insufficient scoreable folds for a verdict"}

    # the better of the two single models
    better_name, better_auc = ("M2", a2) if a2 >= a1 else ("M1", a1)
    combo_edge = ac - better_auc
    beats = combo_edge > SEED_NOISE_BAND

    # confirmatory: encompassing test, combination vs the better single
    combo_pooled = _pooled_oos(combo)
    better_pooled = _pooled_oos(m2 if better_name == "M2" else m1)
    encompassing = None
    if combo_pooled is not None and better_pooled is not None:
        cp, ca = combo_pooled
        bp, ba = better_pooled
        idx = cp.index.intersection(bp.index)
        if len(idx) >= 30:
            # treat the better single as the "baseline", combo as challenger
            encompassing = encompassing_test(
                ba.reindex(idx).to_numpy(),
                bp.reindex(idx).to_numpy(),
                cp.reindex(idx).to_numpy(),
            )

    enc_confirms = bool(encompassing and not encompassing.get("error")
                        and encompassing.get("challenger_adds_info"))

    if beats and enc_confirms:
        verdict = (f"The M1+M2 equal-weight combination BEATS the better "
                   f"single model ({better_name} {better_auc:.4f}) at h=6 "
                   f"by {combo_edge:+.4f}, and the encompassing test "
                   f"confirms it carries information the better single "
                   f"does not. HC1 supported — the combination is a "
                   f"genuine improvement at h=6.")
    elif beats and not enc_confirms:
        verdict = (f"The combination beats {better_name} on AUC "
                   f"({combo_edge:+.4f}) but the encompassing test does "
                   f"not confirm — not counted as a genuine improvement "
                   f"under the pre-registered both-criteria rule.")
    else:
        verdict = (f"The M1+M2 combination does NOT beat the better "
                   f"single model ({better_name} {better_auc:.4f}) at h=6 "
                   f"(edge {combo_edge:+.4f}, within the {SEED_NOISE_BAND} "
                   f"seed-noise band). HC1 not supported: the encompassing "
                   f"test's 'independent information' did not translate "
                   f"into a better OOS forecast — ship the single model. "
                   f"A valid pre-registered null outcome.")

    return {
        "m1": m1, "m2": m2, "combo": combo,
        "m1_auc": a1, "m2_auc": a2, "combo_auc": ac,
        "better_single": better_name, "better_auc": better_auc,
        "combo_edge": combo_edge, "beats": beats,
        "encompassing": encompassing, "enc_confirms": enc_confirms,
        "verdict": verdict,
    }


def print_h6_combination_report(result: dict) -> None:
    """Print the h=6 combination experiment report."""
    print("=" * 70)
    print("B-TRACK FOLLOW-UP — h=6 M1+M2 COMBINATION EXPERIMENT")
    print("=" * 70)
    if result.get("error"):
        print(f"  {result['error']}")
        print("=" * 70)
        return

    print(f"  pre-registered: equal-weight average of M1 and M2 at h=6;")
    print(f"  wins only if it beats the better single by > "
          f"{SEED_NOISE_BAND} AND the encompassing test confirms.")
    print()
    print(f"  {'model':>26} {'mean fold AUC':>15}")
    print(f"  {'M1 (yield curve)':>26} {result['m1_auc']:>15.4f}")
    print(f"  {'M2 (macro logit)':>26} {result['m2_auc']:>15.4f}")
    print(f"  {'M1+M2 equal-weight':>26} {result['combo_auc']:>15.4f}")
    print()
    print(f"  better single model: {result['better_single']} "
          f"({result['better_auc']:.4f})")
    print(f"  combination edge over better single: "
          f"{result['combo_edge']:+.4f}")
    print(f"  primary criterion (seed-noise band {SEED_NOISE_BAND}): "
          f"{'BEATS' if result['beats'] else 'does NOT beat'}")

    enc = result.get("encompassing")
    if enc and not enc.get("error"):
        print()
        print(f"  encompassing test (combination vs better single):")
        print(f"    combination p={enc['challenger_p']:.4f}, "
              f"better-single p={enc['baseline_p']:.4f}")
        print(f"    confirms combination adds information: "
              f"{result['enc_confirms']}")
    elif enc and enc.get("error"):
        print(f"  encompassing test: {enc['error']}")

    print()
    print("-" * 70)
    words = result["verdict"].split()
    line = "  "
    for w in words:
        if len(line) + len(w) + 1 > 68:
            print(line); line = "  " + w
        else:
            line += (" " if line.strip() else "") + w
    if line.strip():
        print(line)
    print("=" * 70)
