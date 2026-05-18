"""
recession/validation/b_track.py

B-track driver — the short-horizon recession model series.

Implements the design locked in B_TRACK_PREREGISTRATION.md. Nothing here
deviates from that document; if a change is needed it must be a dated
amendment to the pre-registration, not a quiet edit here.

WHAT IT RUNS
------------
Per the pre-registered model matrix:
    h=3   : M1, M2, M3, M2-binary   (the full short-horizon comparison)
    h=6   : M1, M2, M3, M2-binary
    h=12  : M2-binary vs M1 only    (A-track verdict stands; M2-binary is
            a single targeted test of the binarization at 12 months)

All models at a given horizon run on ONE common fold axis (months where
every feature in the 4-feature set is present), so the comparison at that
horizon is exact.

THE TWO PRE-REGISTERED CRITERIA
-------------------------------
1. PRIMARY — OOS AUC, seed-stable. A model "beats the baseline" only if
   its walk-forward mean fold AUC exceeds M1's by more than the
   pre-registered SEED_NOISE_BAND (0.03). A numeric edge inside the band
   is not a win.
2. CONFIRMATORY — forecast encompassing test. A probit regression of the
   realized recession indicator on the log-odds of two competing OOS
   forecasts (M1 and the challenger). If the challenger's coefficient is
   significant and M1's is not, the challenger carries information M1
   does not. Reported alongside the AUC verdict.

A model is a genuine improvement only if it passes BOTH.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm

from recession.validation.walk_forward import walk_forward, WalkForwardResult
from recession.features.builder import build_feature_dataframe
from recession.models.m1_probit import M1Probit, M1_FEATURES
from recession.models.m2_logit import M2Logit, M2_FEATURES
from recession.models.m3_forest import M3Forest, M3_CORE_FEATURES
from recession.models.m2_binary import M2Binary, M2_BINARY_FEATURES


# pre-registered constant — the seed-noise band (B_TRACK_PREREGISTRATION
# section F3). A numeric AUC edge over M1 smaller than this is not a win.
SEED_NOISE_BAND = 0.03

# the 4-feature set every B-track model's axis is keyed to
B_TRACK_FEATURES = ["T10Y3M", "NFCI", "INDPRO", "REAL_FFR_GAP"]


# =============================================================================
# forecast encompassing test
# =============================================================================

def encompassing_test(
    realized: np.ndarray,
    proba_baseline: np.ndarray,
    proba_challenger: np.ndarray,
) -> dict:
    """Forecast encompassing test (FRB Philadelphia 2025 style).

    Regress the realized 0/1 recession indicator (probit) on the LOG-ODDS
    of the two competing OOS forecasts. If the challenger's coefficient is
    significant, it carries information not in the baseline.

    Returns {'baseline_coef', 'baseline_p', 'challenger_coef',
             'challenger_p', 'challenger_adds_info', 'verdict'}.
    """
    y = np.asarray(realized, dtype=int)
    if len(np.unique(y)) < 2:
        return {"error": "realized outcome is single-class — test "
                          "undefined"}

    def _logodds(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    lo_base = _logodds(proba_baseline)
    lo_chal = _logodds(proba_challenger)
    X = np.column_stack([lo_base, lo_chal])
    Xd = sm.add_constant(X, has_constant="add")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Probit(y, Xd).fit(disp=0, maxiter=200)
    except Exception as e:
        return {"error": f"encompassing probit failed: {e}"}

    # params: [const, baseline_logodds, challenger_logodds]
    base_coef, base_p = float(res.params[1]), float(res.pvalues[1])
    chal_coef, chal_p = float(res.params[2]), float(res.pvalues[2])
    challenger_adds = chal_p < 0.05

    if challenger_adds and base_p >= 0.05:
        verdict = ("Challenger encompasses the baseline — it carries the "
                   "predictive information and the baseline adds nothing "
                   "beyond it.")
    elif challenger_adds and base_p < 0.05:
        verdict = ("Both forecasts carry independent information — neither "
                   "encompasses the other; a combination could help.")
    elif not challenger_adds and base_p < 0.05:
        verdict = ("Baseline encompasses the challenger — the challenger "
                   "adds no information beyond the yield curve.")
    else:
        verdict = ("Neither forecast is individually significant in the "
                   "encompassing regression — inconclusive.")

    return {
        "baseline_coef": base_coef, "baseline_p": base_p,
        "challenger_coef": chal_coef, "challenger_p": chal_p,
        "challenger_adds_info": challenger_adds,
        "verdict": verdict,
    }


# =============================================================================
# helpers
# =============================================================================

def _common_axis(target, horizon, db_path, min_history_year):
    """Months where every B-track feature is present — the shared fold
    axis for one horizon."""
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year
    probe = build_feature_dataframe(
        target=target, horizon=horizon,
        as_of="today", train_cutoff="today",
        feature_subset=B_TRACK_FEATURES, **build_kwargs,
    )
    cols = [c for c in B_TRACK_FEATURES if c in probe.X.columns]
    return probe.X.index[probe.X[cols].notna().all(axis=1)]


def _pooled_oos(result: WalkForwardResult):
    """Pool a model's OOS fold predictions into aligned (date, proba,
    actual) arrays."""
    dates, proba, actual = [], [], []
    for fold in result.folds:
        dates.extend(pd.to_datetime(fold.test_dates))
        proba.extend(fold.test_proba)
        actual.extend(fold.test_actual)
    if not dates:
        return None
    idx = pd.DatetimeIndex(dates)
    p = pd.Series(proba, index=idx).groupby(level=0).mean()
    a = pd.Series(actual, index=idx).groupby(level=0).max()
    return p, a


# =============================================================================
# the driver
# =============================================================================

def run_b_track_horizon(
    horizon: str,
    *,
    models: list[str],
    target: str = "T1",
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
    **walk_forward_kwargs,
) -> dict:
    """Run the requested models at one horizon on a shared fold axis.

    models: subset of ['M1','M2','M3','M2-binary'].
    Returns {'horizon', 'results': {name: WalkForwardResult},
             'baseline_auc', 'verdicts': {name: {...}},
             'encompassing': {name: {...}}}.
    """
    axis = _common_axis(target, horizon, db_path, min_history_year)
    common = dict(target=target, horizon=horizon,
                  min_history_year=min_history_year, db_path=db_path,
                  restrict_to_months=axis, **walk_forward_kwargs)

    spec = {
        "M1": (M1Probit, M1_FEATURES),
        "M2": (lambda: M2Logit(C=1.0), M2_FEATURES),
        "M3": (M3Forest, M3_CORE_FEATURES),
        "M2-binary": (M2Binary, M2_BINARY_FEATURES),
    }

    results = {}
    for name in models:
        factory, feats = spec[name]
        results[name] = walk_forward(
            model_factory=factory, feature_subset=feats,
            model_columns=feats, model_name=f"{name} ({horizon})",
            **common,
        )

    baseline_auc = (results["M1"].mean_fold_auc
                    if "M1" in results else None)

    # primary criterion: seed-noise band vs M1
    verdicts = {}
    for name, res in results.items():
        if name == "M1":
            continue
        auc = res.mean_fold_auc
        if auc is None or baseline_auc is None:
            verdicts[name] = {"auc": auc, "edge": None,
                              "beats_baseline": False,
                              "note": "insufficient folds"}
            continue
        edge = auc - baseline_auc
        beats = edge > SEED_NOISE_BAND
        verdicts[name] = {
            "auc": auc, "edge": edge, "beats_baseline": beats,
            "note": (f"edge {edge:+.4f} exceeds the {SEED_NOISE_BAND} "
                     f"seed-noise band — BEATS baseline" if beats
                     else f"edge {edge:+.4f} within the {SEED_NOISE_BAND} "
                          f"seed-noise band — does NOT beat baseline"),
        }

    # confirmatory criterion: encompassing test, each challenger vs M1
    encompassing = {}
    if "M1" in results:
        m1_pooled = _pooled_oos(results["M1"])
        for name, res in results.items():
            if name == "M1":
                continue
            chal_pooled = _pooled_oos(res)
            if m1_pooled is None or chal_pooled is None:
                encompassing[name] = {"error": "no OOS predictions"}
                continue
            # align on common months
            m1_p, m1_a = m1_pooled
            ch_p, ch_a = chal_pooled
            common_idx = m1_p.index.intersection(ch_p.index)
            if len(common_idx) < 30:
                encompassing[name] = {"error": "too few common months"}
                continue
            encompassing[name] = encompassing_test(
                m1_a.reindex(common_idx).to_numpy(),
                m1_p.reindex(common_idx).to_numpy(),
                ch_p.reindex(common_idx).to_numpy(),
            )

    return {"horizon": horizon, "results": results,
            "baseline_auc": baseline_auc, "verdicts": verdicts,
            "encompassing": encompassing}


def run_b_track(
    *,
    target: str = "T1",
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
    **walk_forward_kwargs,
) -> dict:
    """Run the full pre-registered B-track matrix: h=3, h=6, h=12.

    Returns {'h=3': {...}, 'h=6': {...}, 'h=12': {...}}.
    """
    matrix = {
        "h=3": ["M1", "M2", "M3", "M2-binary"],
        "h=6": ["M1", "M2", "M3", "M2-binary"],
        "h=12": ["M1", "M2-binary"],   # A-track verdict stands; targeted test
    }
    out = {}
    for horizon, models in matrix.items():
        out[horizon] = run_b_track_horizon(
            horizon, models=models, target=target, db_path=db_path,
            min_history_year=min_history_year, **walk_forward_kwargs,
        )
    return out


def print_b_track_report(results: dict) -> None:
    """Print the B-track report against the pre-registered criteria."""
    print("=" * 72)
    print("B-TRACK — SHORT-HORIZON RECESSION MODEL SERIES")
    print("=" * 72)
    print(f"  pre-registered seed-noise band: {SEED_NOISE_BAND}")
    print("  a challenger 'beats baseline' only if its OOS-AUC edge over")
    print("  M1 exceeds that band AND the encompassing test agrees.")

    for horizon in ("h=3", "h=6", "h=12"):
        if horizon not in results:
            continue
        h = results[horizon]
        print()
        print("-" * 72)
        print(f"  HORIZON {horizon}")
        ba = h["baseline_auc"]
        print(f"  M1 baseline mean fold AUC: "
              + (f"{ba:.4f}" if ba is not None else "n/a"))
        print()
        print(f"  {'model':>12} {'OOS AUC':>9} {'edge vs M1':>11} "
              f"{'primary':>16}")
        for name, res in h["results"].items():
            if name == "M1":
                continue
            v = h["verdicts"].get(name, {})
            auc = v.get("auc")
            edge = v.get("edge")
            prim = ("BEATS" if v.get("beats_baseline") else "no")
            auc_s = f"{auc:.4f}" if auc is not None else "n/a"
            edge_s = f"{edge:+.4f}" if edge is not None else "n/a"
            print(f"  {name:>12} {auc_s:>9} {edge_s:>11} {prim:>16}")

        # encompassing
        print()
        print("  forecast encompassing test (challenger vs M1):")
        for name, enc in h["encompassing"].items():
            if enc.get("error"):
                print(f"    {name}: {enc['error']}")
                continue
            print(f"    {name}: challenger p={enc['challenger_p']:.4f}, "
                  f"baseline p={enc['baseline_p']:.4f}")
            # wrap verdict
            words = enc["verdict"].split()
            line = "      "
            for w in words:
                if len(line) + len(w) + 1 > 70:
                    print(line); line = "      " + w
                else:
                    line += (" " if line.strip() else "") + w
            if line.strip():
                print(line)

    # overall verdict
    print()
    print("=" * 72)
    print("  OVERALL B-TRACK VERDICT")
    any_beat = False
    for horizon in ("h=3", "h=6", "h=12"):
        if horizon not in results:
            continue
        h = results[horizon]
        for name, v in h["verdicts"].items():
            if not v.get("beats_baseline"):
                continue
            enc = h["encompassing"].get(name, {})
            enc_ok = enc.get("challenger_adds_info", False)
            if enc_ok:
                any_beat = True
                print(f"  {name} at {horizon}: passes BOTH criteria — a "
                      f"genuine improvement over the yield curve.")
            else:
                print(f"  {name} at {horizon}: beats on AUC but the "
                      f"encompassing test does not confirm — not a "
                      f"genuine improvement.")
    if not any_beat:
        print("  No model passes both pre-registered criteria at any")
        print("  horizon. The yield curve (M1) is not beaten — the macro")
        print("  signal, though statistically real (nested test), does not")
        print("  become generalizable OOS skill even at short horizons.")
        print("  This is the pre-registered null outcome — a valid finding.")
    print("=" * 72)
