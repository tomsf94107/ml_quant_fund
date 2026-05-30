"""
recession/validation/calibration_diagnostic.py

Calibration diagnostic — the reliability diagram for the recession models.

WHY THIS EXISTS
---------------
The recession report card showed the short-horizon models produce
compressed probabilities: for the 2008 GFC the peak out-of-sample
probability fell 0.77 (h=12) -> 0.34 (h=6) -> 0.18 (h=3). A model can
RANK recession-months above calm-months well (a good AUC) yet never emit
a confident absolute probability — that is a CALIBRATION failure, and it
is separate from discrimination (AUC).

This module is the DIAGNOSIS, not the fix. The recession-forecasting and
ML literature is consistent: diagnose calibration with a reliability
diagram BEFORE applying any correction. Only once the diagram confirms a
model is miscalibrated — and shows HOW — does a fix (e.g. Platt scaling)
make sense.

WHAT IT COMPUTES
----------------
For each horizon, from the pooled walk-forward out-of-sample
predictions:

  RELIABILITY DIAGRAM. Predictions are binned (default 10 bins on
    [0, 1]). For each bin: the mean predicted probability vs the actual
    fraction of recession-positive months in that bin. A perfectly
    calibrated model lies on the diagonal (mean predicted == actual
    frequency). Points below the diagonal = overconfident; points above
    = underconfident.

  ECE — Expected Calibration Error. The bin-count-weighted average of
    |mean predicted - actual frequency| across bins. 0 = perfect; larger
    = worse calibration.

  MAX PREDICTED PROBABILITY. The single largest probability the model
    ever emitted out-of-sample. If this is far below 1.0 (e.g. 0.34),
    the model is COMPRESSED — it structurally cannot express confidence,
    regardless of how well it ranks.

  COMPRESSION / DIRECTION VERDICT. A plain-language read: is the model
    compressed, and is it over- or under-confident on average.

HONEST LIMITS
-------------
- Recessions are rare (~10-12 in the modern sample). After a train/test
  split the number of OOS recession-MONTHS is small, so high-probability
  bins may hold very few observations — the actual-frequency estimate in
  those bins is noisy. The diagram reports per-bin counts so a thin bin
  is visible and not over-read.
- This module DIAGNOSES. It does not modify any model. A calibration
  FIX is a separate, pre-registered change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from recession.validation.lead_time import run_lead_time_analysis


# default number of reliability-diagram bins on [0, 1]
N_BINS = 10

# the model ladder to diagnose: (label, target, horizon, model).
# M1 (yield curve) at h=12; M2 (macro logit) at the short horizons, per
# B-track. Diagnosing M1 at h=3 would diagnose the yield curve where it
# is known to be weak — not the model that actually runs there.
DIAGNOSTIC_LADDER = [
    ("h=12 (yield curve / M1)", "T1", "h=12", "M1"),
    ("h=6 (macro / M2)", "T1", "h=6", "M2"),
    ("h=3 (macro / M2)", "T1", "h=3", "M2"),
]

# a model whose largest OOS probability never exceeds this is flagged
# COMPRESSED — it structurally cannot reach a 0.5 decision threshold
# with any margin.
COMPRESSION_MAX_PROBA = 0.5


def reliability_diagram(
    proba: pd.Series, actual: pd.Series, *, n_bins: int = N_BINS,
) -> dict:
    """Bin predictions and compare mean predicted probability vs actual
    recession frequency per bin.

    proba, actual: aligned month-indexed Series (predicted probability;
    realized 0/1 label the model was predicting).

    Returns {'bins': [{bin_lo, bin_hi, n, mean_pred, actual_freq,
             gap}...], 'ece', 'max_proba', 'mean_pred', 'mean_actual',
             'n_obs'}.
    """
    df = pd.DataFrame({"p": proba, "y": actual}).dropna()
    if len(df) == 0:
        return {"bins": [], "ece": None, "max_proba": None,
                "mean_pred": None, "mean_actual": None, "n_obs": 0}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece_num = 0.0
    n_total = len(df)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is closed on the right so p == 1.0 is included
        if i == n_bins - 1:
            mask = (df["p"] >= lo) & (df["p"] <= hi)
        else:
            mask = (df["p"] >= lo) & (df["p"] < hi)
        sub = df[mask]
        n = len(sub)
        if n == 0:
            bins.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                         "mean_pred": None, "actual_freq": None,
                         "gap": None})
            continue
        mean_pred = float(sub["p"].mean())
        actual_freq = float(sub["y"].mean())
        gap = mean_pred - actual_freq          # +ve = overconfident
        bins.append({"bin_lo": lo, "bin_hi": hi, "n": n,
                     "mean_pred": mean_pred, "actual_freq": actual_freq,
                     "gap": gap})
        ece_num += n * abs(gap)

    return {
        "bins": bins,
        "ece": ece_num / n_total,
        "max_proba": float(df["p"].max()),
        "mean_pred": float(df["p"].mean()),
        "mean_actual": float(df["y"].mean()),
        "n_obs": n_total,
    }


def _diagnose(diagram: dict) -> dict:
    """Turn a reliability diagram into a plain-language verdict."""
    if diagram["n_obs"] == 0 or diagram["ece"] is None:
        return {"compressed": None, "direction": None,
                "verdict": "no out-of-sample predictions to diagnose"}

    max_p = diagram["max_proba"]
    compressed = max_p < COMPRESSION_MAX_PROBA

    # average signed gap (count-weighted): +ve overconfident overall
    num = sum(b["n"] * b["gap"] for b in diagram["bins"]
              if b["gap"] is not None)
    signed = num / diagram["n_obs"]
    if signed > 0.05:
        direction = "overconfident"
    elif signed < -0.05:
        direction = "underconfident"
    else:
        direction = "roughly centered"

    parts = []
    if compressed:
        parts.append(
            f"COMPRESSED — the largest probability the model ever emits "
            f"out-of-sample is {max_p:.2f}, below the {COMPRESSION_MAX_PROBA} "
            f"decision threshold. It structurally cannot express a "
            f"confident recession call, however well it ranks.")
    else:
        parts.append(
            f"not compressed — the model does reach high probabilities "
            f"(max {max_p:.2f}).")
    parts.append(f"ECE = {diagram['ece']:.3f} "
                 f"({'well' if diagram['ece'] < 0.1 else 'poorly'} "
                 f"calibrated overall).")
    parts.append(f"On average the model is {direction} "
                 f"(mean predicted {diagram['mean_pred']:.3f} vs "
                 f"actual recession frequency {diagram['mean_actual']:.3f}).")

    return {"compressed": compressed, "direction": direction,
            "signed_gap": signed, "verdict": " ".join(parts)}


def run_calibration_diagnostic(
    *,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
    n_bins: int = N_BINS,
    **wf_kwargs,
) -> dict:
    """Run the calibration diagnostic across the model ladder.

    Returns {'horizons': [{label, horizon, diagram, diagnosis}...]}.
    """
    horizons = []
    for label, target, horizon, model in DIAGNOSTIC_LADDER:
        res = run_lead_time_analysis(
            target=target, horizon=horizon, model=model,
            min_history_year=min_history_year, db_path=db_path,
            **wf_kwargs)
        if res.get("error") or "proba" not in res:
            horizons.append({"label": label, "horizon": horizon,
                             "error": res.get("error",
                                              "no OOS predictions")})
            continue
        diagram = reliability_diagram(res["proba"], res["actual"],
                                      n_bins=n_bins)
        diagnosis = _diagnose(diagram)
        horizons.append({"label": label, "horizon": horizon,
                         "diagram": diagram, "diagnosis": diagnosis,
                         "error": None})
    return {"horizons": horizons}


def print_calibration_diagnostic(result: dict) -> None:
    """Print the reliability diagram + diagnosis for each horizon."""
    print("=" * 74)
    print("CALIBRATION DIAGNOSTIC — reliability diagram per horizon")
    print("=" * 74)
    print("  Does each model's predicted probability match the actual")
    print("  recession frequency? This DIAGNOSES calibration; it does")
    print("  not change any model.")
    print()

    for h in result["horizons"]:
        print("-" * 74)
        if h.get("error"):
            print(f"  {h['label']}: ERROR — {h['error']}")
            continue
        d = h["diagram"]
        print(f"  HORIZON: {h['label']}   "
              f"(n_obs={d['n_obs']}, ECE={d['ece']:.3f}, "
              f"max prob={d['max_proba']:.2f})")
        print()
        print(f"  {'bin':>13} {'count':>7} {'mean pred':>11} "
              f"{'actual freq':>13} {'gap':>9}")
        for b in d["bins"]:
            label = f"[{b['bin_lo']:.1f},{b['bin_hi']:.1f}]"
            if b["n"] == 0:
                print(f"  {label:>13} {0:>7} {'—':>11} {'—':>13} {'—':>9}")
            else:
                print(f"  {label:>13} {b['n']:>7} {b['mean_pred']:>11.3f} "
                      f"{b['actual_freq']:>13.3f} {b['gap']:>+9.3f}")
        print()
        # the diagnosis, wrapped
        words = h["diagnosis"]["verdict"].split()
        line = "  -> "
        for w in words:
            if len(line) + len(w) + 1 > 72:
                print(line); line = "     " + w
            else:
                line += (" " if line.strip() != "->" else "") + w
        if line.strip():
            print(line)
        print()

    print("=" * 74)
    print("  READING THIS:")
    print("  - gap = mean predicted - actual frequency. Positive gap =")
    print("    overconfident in that bin; negative = underconfident.")
    print("  - A COMPRESSED model (low max prob) ranks but won't commit;")
    print("    the fix for that is probability calibration, NOT a new")
    print("    model. Calibration cannot create signal that is not there")
    print("    — if a horizon also ranks poorly (low AUC), calibration")
    print("    will not rescue it.")
    print("  - High-probability bins may hold few observations (rare")
    print("    recessions) — read thin bins (low count) with caution.")
    print("=" * 74)
