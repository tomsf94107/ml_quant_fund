"""
recession/validation/lead_time.py

Step 10 — lead-time validation. An AUC of 0.80 says the model RANKS
recession and calm months well. It does NOT say the warning arrives early
enough to act on. Step 10 answers the question a user of a recession model
actually has: WHEN M1 FIRES, HOW MANY MONTHS OF WARNING DO YOU GET?

WHAT IS MEASURED
----------------
Using the out-of-sample walk-forward predictions of M1 (so lead time is
honest, not in-sample):

  1. LEAD TIME per recession. For each recession ONSET in the OOS period,
     find the most recent month before the onset at which M1's predicted
     probability first rose above a warning threshold and STAYED above it
     up to the onset. The gap (onset - first-crossing) is the lead time:
     how many months of standing warning preceded the recession.

  2. HIT RATE. Of the recession onsets in the OOS period, how many got
     any standing warning at all before they began.

  3. FALSE-ALARM RATE. How often the probability crossed the threshold
     during calm periods that were NOT followed by a recession — the cost
     side of an early warning system.

  4. THRESHOLD SWEEP. All of the above across a range of warning
     thresholds, because lead time and false alarms trade off: a low
     threshold warns earlier but cries wolf more.

WHY THIS MATTERS FOR THE PROJECT
--------------------------------
The M1-M5 ladder established M1 (the yield-curve probit) as the
production recession model. Step 10 characterises its OPERATIONAL
behaviour — the numbers a dashboard user needs: typical warning lead,
hit rate, false-alarm rate. This is what makes M1 a usable product
rather than just an AUC.

NOTE ON THE TARGET. M1 here is validated on T1/h=12. The h=12 target is
itself "recession within 12 months", so M1's probability already encodes
a 12-month-ahead view. Lead time is measured against the actual recession
ONSET dates (from the target's transition from 0 to 1 in the realised
labels), independent of the horizon used to train.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from recession.models.m1_probit import run_m1
from recession.models.m2_logit import run_m2
from recession.features.pit_loader import load_targets


# default warning thresholds for the sweep
LEAD_TIME_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


# =============================================================================
# Recession-onset detection
# =============================================================================

def find_recession_onsets(labels: pd.Series) -> list[pd.Timestamp]:
    """Months where the recession label transitions 0 -> 1 — the onsets.

    labels: a 0/1 Series indexed by month (the realised recession state,
    NOT the h=12 target — onset means the recession actually began).
    """
    s = labels.dropna().astype(int).sort_index()
    onsets = []
    prev = None                       # None until the first observed month
    for month, val in s.items():
        # An onset requires an OBSERVED 0 -> 1 transition. If the series
        # starts already in recession (prev is None and val == 1) there is
        # no observed onset — the recession began before our data.
        if prev == 0 and val == 1:
            onsets.append(month)
        prev = val
    return onsets


# =============================================================================
# Lead-time measurement
# =============================================================================

def measure_lead_times(
    proba: pd.Series,
    onsets: list[pd.Timestamp],
    threshold: float,
    *,
    max_lookback_months: int = 24,
) -> dict:
    """For each onset, the lead time at a warning threshold.

    proba: M1's OOS predicted recession probability, indexed by month.
    onsets: recession-onset months.
    threshold: warning fires when proba >= threshold.

    Lead time for an onset = the number of consecutive months immediately
    before the onset during which proba stayed >= threshold (a STANDING
    warning). 0 means no warning was in place at the onset. Capped at
    max_lookback_months.

    Returns {'lead_times': {onset: months}, 'hit_rate': float,
             'mean_lead': float, 'median_lead': float}.
    """
    proba = proba.dropna().sort_index()
    lead_times: dict = {}

    for onset in onsets:
        # months strictly before the onset, most-recent first
        before = proba[proba.index < onset]
        if len(before) == 0:
            lead_times[onset] = None        # onset outside the OOS coverage
            continue
        before = before.iloc[::-1]          # reverse: nearest month first
        lead = 0
        for _, p in before.items():
            if lead >= max_lookback_months:
                break
            if p >= threshold:
                lead += 1
            else:
                break                        # warning must be CONTINUOUS
        lead_times[onset] = lead

    measured = [v for v in lead_times.values() if v is not None]
    hits = [v for v in measured if v > 0]
    return {
        "lead_times": lead_times,
        "n_onsets_covered": len(measured),
        "hit_rate": (len(hits) / len(measured)) if measured else None,
        "mean_lead": float(np.mean(hits)) if hits else 0.0,
        "median_lead": float(np.median(hits)) if hits else 0.0,
    }


def false_alarm_rate(
    proba: pd.Series,
    labels: pd.Series,
    threshold: float,
    *,
    horizon_months: int = 12,
    recovery_months: int = 6,
) -> dict:
    """Classifier-quality metrics for a recession warning at one threshold.

    Two different, both-meaningful rates are returned — they answer
    different questions and have different denominators:

    ROC FALSE-POSITIVE RATE (`fpr`) — the literature standard.
      false-positive months / ALL eligible expansion months.
      "Of all the calm months, how often did the model wrongly flag?"
      This is 1 - specificity. Its denominator is the (large, stable)
      count of expansion months, so it does NOT blow up just because a
      low threshold flags many months. This is the rate the recession-
      forecasting literature (Berge-Jorda, the Fed ROC studies) uses,
      and the rate that makes the Kuiper score TPR - FPR well-defined.

    WARNING RELIABILITY (`warning_reliability_false_rate`) — precision-
      style. false-alarm months / WARNING months. "When the model
      flags, how often is it wrong?" Meaningful, but its denominator is
      the flagged months, so a low threshold inflates it. Reported, but
      it is NOT the FPR and must not be used as one.

    TRUE-POSITIVE RATE (`tpr`) — true-positive months / all eligible
      pre-recession months. The ROC partner of `fpr`.

    A month is classed as a warning if proba >= threshold. Eligibility:
    months that are IN a recession, or within `recovery_months` after a
    recession ended, are EXCLUDED from all denominators — per Richmond/
    Fed practice, the in-recession and choppy-recovery months are
    ambiguous and pollute the rates. A warning month is a TRUE positive
    if a recession onset falls within the next `horizon_months`.

    Returns {'n_warning_months', 'n_false_alarm_months',
             'false_alarm_rate' (== warning_reliability_false_rate, kept
             for backward compatibility), 'warning_reliability_false_rate',
             'fpr', 'tpr', 'n_expansion_months', 'n_pre_recession_months'}.
    """
    proba = proba.dropna().sort_index()
    labels = labels.dropna().astype(int).sort_index()
    onsets = set(find_recession_onsets(labels))

    # months where a recession ENDED (1 -> 0 transition) — for the
    # post-recession recovery exclusion window
    ends = []
    prev_val = None
    for month, val in labels.items():
        if prev_val == 1 and val == 0:
            ends.append(month)
        prev_val = val

    def _in_recession(m):
        return m in labels.index and labels.loc[m] == 1

    def _in_recovery(m):
        return any(e < m <= e + pd.DateOffset(months=recovery_months)
                   for e in ends)

    def _onset_within(m):
        window_end = m + pd.DateOffset(months=horizon_months)
        return any(m < o <= window_end for o in onsets)

    # classify every month with a prediction
    n_warning = 0
    n_false = 0                      # warning, eligible, no onset following
    n_tp = 0                         # warning, eligible, onset following
    n_expansion = 0                  # eligible months with NO onset following
    n_pre_recession = 0              # eligible months WITH an onset following
    for m in proba.index:
        # eligibility: drop in-recession and recovery-window months
        if _in_recession(m) or _in_recovery(m):
            continue
        is_warning = proba.loc[m] >= threshold
        has_onset = _onset_within(m)
        if has_onset:
            n_pre_recession += 1
            if is_warning:
                n_tp += 1
        else:
            n_expansion += 1
            if is_warning:
                n_false += 1
        if is_warning:
            n_warning += 1

    warning_reliability_false = (n_false / n_warning) if n_warning else None
    fpr = (n_false / n_expansion) if n_expansion else None
    tpr = (n_tp / n_pre_recession) if n_pre_recession else None

    return {
        "n_warning_months": n_warning,
        "n_false_alarm_months": n_false,
        # backward-compatible key — equals the warning-reliability rate
        "false_alarm_rate": warning_reliability_false,
        "warning_reliability_false_rate": warning_reliability_false,
        "fpr": fpr,
        "tpr": tpr,
        "n_expansion_months": n_expansion,
        "n_pre_recession_months": n_pre_recession,
    }


# =============================================================================
# The driver
def _horizon_to_months(horizon: str) -> int:
    """Parse a horizon string ('h=3', 'h=12') to a month count.
    Used to size the false-alarm window to the model's horizon."""
    try:
        return int(str(horizon).split("=")[1])
    except (IndexError, ValueError):
        return 12          # safe default — the conventional 12m window


# =============================================================================

def run_lead_time_analysis(
    target: str = "T1",
    horizon: str = "h=12",
    *,
    model: str = "M1",
    min_history_year: Optional[int] = 1986,
    db_path: Optional[Path] = None,
    thresholds: Optional[list[float]] = None,
    **walk_forward_kwargs,
) -> dict:
    """Run a model through the walk-forward harness, pool its OOS
    predictions, and characterise lead time + false alarms across a
    threshold sweep.

    `model` selects which model's OOS predictions to analyse:
      'M1' (default) — the yield-curve probit. Correct for h=12.
      'M2'           — the 4-feature macro logit. B-track established M2
                       as the short-horizon model; use 'M2' for h=3/h=6.
    Defaulting to 'M1' keeps every existing caller unchanged.

    Returns {'sweep': {threshold: {...}}, 'onsets': [...],
             'n_oos_months': int, 'proba': Series, 'actual': Series,
             'model': str}.
    """
    if thresholds is None:
        thresholds = LEAD_TIME_THRESHOLDS

    # route to the requested model's runner; both return a dict with a
    # WalkForwardResult under a model-specific key.
    if model == "M1":
        run_out = run_m1(target=target, horizon=horizon,
                         min_history_year=min_history_year,
                         db_path=db_path, **walk_forward_kwargs)
        wf = run_out["m1"]
    elif model == "M2":
        run_out = run_m2(target=target, horizon=horizon,
                         min_history_year=min_history_year,
                         db_path=db_path, **walk_forward_kwargs)
        wf = run_out["m2"]
    else:
        raise ValueError(f"unknown model {model!r} — expected 'M1' or 'M2'")

    # pool OOS fold predictions into one month-indexed Series
    dates, probs, actuals = [], [], []
    for fold in wf.folds:
        dates.extend(pd.to_datetime(fold.test_dates))
        probs.extend(fold.test_proba)
        actuals.extend(fold.test_actual)
    if not dates:
        return {"sweep": {}, "onsets": [], "n_oos_months": 0,
                "error": "no OOS predictions", "model": model}

    proba = pd.Series(probs, index=pd.DatetimeIndex(dates)).sort_index()
    # overlapping folds can repeat a month — average duplicates
    proba = proba.groupby(proba.index).mean()

    # `actual` here is the model's TARGET — at h>0 it is the forward-
    # shifted "recession within h months" series. It is NOT the realized
    # recession state, so onsets read off it land h months early. For
    # onset dates and false-alarm scoring we need the REALIZED label
    # (h=0). Load it and restrict to the OOS span.
    load_kwargs = {}
    if db_path is not None:
        load_kwargs["db_path"] = Path(db_path)
    realized = load_targets(target, "h=0", **load_kwargs)
    realized = realized.dropna().astype(int).sort_index()
    realized = realized[(realized.index >= proba.index.min())
                        & (realized.index <= proba.index.max())]

    onsets = find_recession_onsets(realized)

    # the false-alarm window is matched to the horizon: a warning is
    # legitimate if a recession follows within ~the horizon length (a
    # warning from an h=3 model need only be right about the next ~3
    # months, not 12).
    fa_horizon_months = _horizon_to_months(horizon)

    sweep = {}
    for thr in thresholds:
        lt = measure_lead_times(proba, onsets, thr)
        # false alarms scored against the REALIZED label, horizon-matched
        fa = false_alarm_rate(proba, realized, thr,
                              horizon_months=fa_horizon_months)
        sweep[thr] = {**lt, **fa}

    return {"sweep": sweep, "onsets": onsets,
            "n_oos_months": len(proba),
            "proba": proba, "actual": realized, "model": model,
            "fa_horizon_months": fa_horizon_months}


def print_lead_time_report(results: dict) -> None:
    """Print the Step-10 lead-time report."""
    print("=" * 70)
    print("STEP 10 — LEAD-TIME VALIDATION (M1 yield-curve probit, OOS)")
    print("=" * 70)
    if results.get("error"):
        print(f"  {results['error']}")
        print("=" * 70)
        return

    onsets = results["onsets"]
    print(f"  OOS coverage: {results['n_oos_months']} months, "
          f"{len(onsets)} recession onset(s)")
    if onsets:
        print("  onset months: "
              + ", ".join(o.strftime("%Y-%m") for o in onsets))
    print()
    print("  THRESHOLD SWEEP — lead time (months of standing warning before")
    print("  onset) vs false alarms:")
    print(f"  {'thresh':>7} {'hit rate':>9} {'mean lead':>10} "
          f"{'med lead':>9} {'warn mo':>8} {'false-alarm':>12}")
    print("  " + "-" * 60)
    for thr, s in results["sweep"].items():
        hr = (f"{s['hit_rate']*100:.0f}%"
              if s["hit_rate"] is not None else "n/a")
        far = (f"{s['false_alarm_rate']*100:.0f}%"
               if s["false_alarm_rate"] is not None else "n/a")
        print(f"  {thr:>7.2f} {hr:>9} {s['mean_lead']:>10.1f} "
              f"{s['median_lead']:>9.1f} {s['n_warning_months']:>8} "
              f"{far:>12}")

    print()
    print("-" * 70)
    print("  READING: lower thresholds warn earlier (longer lead) but cry")
    print("  wolf more (higher false-alarm rate). The right operating point")
    print("  depends on the cost of a missed recession vs a false alarm —")
    print("  a choice for the dashboard user, not the model.")
    print("=" * 70)
