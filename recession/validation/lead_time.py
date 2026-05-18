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
) -> dict:
    """How often the warning fires WITHOUT a recession following.

    A warning month is a month with proba >= threshold. It is a FALSE
    alarm if no recession onset occurs within the next horizon_months.

    Returns {'n_warning_months', 'n_false_alarm_months', 'false_alarm_rate'}.
    """
    proba = proba.dropna().sort_index()
    labels = labels.dropna().astype(int).sort_index()
    onsets = set(find_recession_onsets(labels))

    warning_months = proba[proba >= threshold].index
    n_warning = len(warning_months)
    n_false = 0
    for m in warning_months:
        # is there an onset within the next horizon_months?
        window_end = m + pd.DateOffset(months=horizon_months)
        has_onset = any(m < o <= window_end for o in onsets)
        # also not a false alarm if we are already IN a recession
        in_recession = (m in labels.index and labels.loc[m] == 1)
        if not has_onset and not in_recession:
            n_false += 1
    return {
        "n_warning_months": n_warning,
        "n_false_alarm_months": n_false,
        "false_alarm_rate": (n_false / n_warning) if n_warning else None,
    }


# =============================================================================
# The driver
# =============================================================================

def run_lead_time_analysis(
    target: str = "T1",
    horizon: str = "h=12",
    *,
    min_history_year: Optional[int] = 1986,
    db_path: Optional[Path] = None,
    thresholds: Optional[list[float]] = None,
    **walk_forward_kwargs,
) -> dict:
    """Run M1 through the walk-forward harness, pool its OOS predictions,
    and characterise lead time + false alarms across a threshold sweep.

    Returns {'sweep': {threshold: {...}}, 'onsets': [...],
             'n_oos_months': int}.
    """
    if thresholds is None:
        thresholds = LEAD_TIME_THRESHOLDS

    # M1 OOS predictions via the harness
    m1_results = run_m1(target=target, horizon=horizon,
                        min_history_year=min_history_year, db_path=db_path,
                        **walk_forward_kwargs)
    m1 = m1_results["m1"]

    # pool OOS fold predictions into one month-indexed Series
    dates, probs, actuals = [], [], []
    for fold in m1.folds:
        dates.extend(pd.to_datetime(fold.test_dates))
        probs.extend(fold.test_proba)
        actuals.extend(fold.test_actual)
    if not dates:
        return {"sweep": {}, "onsets": [], "n_oos_months": 0,
                "error": "no OOS predictions"}

    proba = pd.Series(probs, index=pd.DatetimeIndex(dates)).sort_index()
    # overlapping folds can repeat a month — average duplicates
    proba = proba.groupby(proba.index).mean()
    actual = pd.Series(actuals, index=pd.DatetimeIndex(dates)).sort_index()
    actual = actual.groupby(actual.index).max()

    onsets = find_recession_onsets(actual)

    sweep = {}
    for thr in thresholds:
        lt = measure_lead_times(proba, onsets, thr)
        fa = false_alarm_rate(proba, actual, thr)
        sweep[thr] = {**lt, **fa}

    return {"sweep": sweep, "onsets": onsets,
            "n_oos_months": len(proba),
            "proba": proba, "actual": actual}


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
