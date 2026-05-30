"""
recession/validation/recession_report_card.py

Item #3 — the historical recession report card.

WHAT IT ANSWERS
---------------
"For each past US recession, what would the model have said in the
run-up — and did it miss any?" A per-recession scorecard: warning lead
time, hit or miss, for each recession the data covers.

THE THREE-TIER DESIGN — why it is built this way
------------------------------------------------
A model can only be honestly judged on data it did NOT train on. The
walk-forward trains on an early span of history and tests forward; so
past recessions split into two kinds, and a report card that blurs them
would mislead. The recession-forecasting literature is explicit about
this (it distinguishes the genuine out-of-sample period from the
training period). So the report card has three tiers:

  TIER 1 — THE REAL TRACK RECORD. Recessions whose onset falls in the
    out-of-sample period (after the first walk-forward test fold begins).
    The model never trained on these. This is the honest headline.

  TIER 2 — IN-SAMPLE CONTEXT. Recessions inside the initial training
    span. The model was FIT on these — "predicting" them is not a real
    test. Shown for context, explicitly STAMPED in-sample, never counted
    in the track record.

  TIER 3 — THE METHODOLOGY CAVEAT. A standing note on every report: the
    evaluation is genuine out-of-sample and vintage-aware (it uses the
    project's point-in-time data), but it is NOT a full real-time vintage
    simulation. The report states what kind of test it is, so it cannot
    be over-read.

HONEST EXPECTATIONS
-------------------
The recession-forecasting literature finds the 2008 financial crisis was
broadly predictable, while the 2020 COVID recession was NOT — it was an
exogenous pandemic shock with no economic early signal. A report card
that shows "2008 caught, 2020 missed" is therefore CORRECT and honest,
not a model failure. Missing 2020 is the expected, literature-consistent
result.

IT REUSES, DOES NOT REBUILD
---------------------------
The lead-time engine (recession/validation/lead_time.py) already finds
recession onsets and measures per-onset warning lead time from pooled
walk-forward OOS predictions. The report card calls that engine — for M1
at h=12 AND M2 at h=3/h=6 — and reframes the output as the three-tier
scorecard. No new prediction logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from recession.validation.lead_time import (
    run_lead_time_analysis, find_recession_onsets, measure_lead_times,
)
from recession.features.pit_loader import load_targets

# named US recessions, by NBER onset month — for labelling the report.
# (The report does not depend on this list; it derives onsets from the
# realised labels. This only attaches human-readable names.)
KNOWN_RECESSION_NAMES = {
    "1980-01": "1980 recession",
    "1981-07": "1981-82 double-dip (Volcker)",
    "1990-07": "1990-91 recession",
    "2001-03": "2001 dot-com recession",
    "2007-12": "2008 Great Recession (GFC)",
    "2020-02": "2020 COVID recession",
}

# the warning threshold the report card uses for the headline hit/miss
# call. 0.5 is the natural midpoint; the full threshold sweep remains
# available via lead_time.run_lead_time_analysis for those who want it.
REPORT_CARD_THRESHOLD = 0.5

# the model ladder the report card scores: (label, target, horizon, model).
# M1 (yield-curve probit) is the h=12 model. B-track established M2 (the
# 4-feature macro logit) as the SHORT-horizon model — so h=6 and h=3 are
# scored with M2, not M1. Scoring M1 at h=3 would test the yield curve
# where it is known to be weak; that is not what those horizons run.
REPORT_CARD_LADDER = [
    ("h=12 (yield curve / M1)", "T1", "h=12", "M1"),
    ("h=6 (macro / M2)", "T1", "h=6", "M2"),
    ("h=3 (macro / M2)", "T1", "h=3", "M2"),
]


# the lead window per horizon: how many months before an onset count as
# a valid early warning. This MUST match what each model predicts. The
# h=12 model forecasts "recession within 12 months" — the literature's
# yield-curve standard is an ~18-month window (the curve inverts ~12
# months ahead, 18 gives margin). The h=6 / h=3 models forecast a
# recession within 6 / 3 months — they are near-coincident detectors,
# NOT long-lead models. Scoring h=3 on an 18-month window asks it for a
# warning it was never built to give. Each horizon is scored on a window
# matched to its own forecast horizon (the horizon + a few months margin).
LEAD_WINDOW_BY_HORIZON = {
    "h=12": 18,
    "h=6": 9,
    "h=3": 6,
}
# fallback if a horizon is not in the map
DEFAULT_LEAD_WINDOW_MONTHS = 18


def _lead_window_for(horizon: str) -> int:
    """The lead window (months) for a horizon — matched to what the
    model at that horizon actually forecasts."""
    return LEAD_WINDOW_BY_HORIZON.get(horizon, DEFAULT_LEAD_WINDOW_MONTHS)


def _name_for(onset: pd.Timestamp) -> str:
    """Human-readable name for a recession onset, or a generic label."""
    key = onset.strftime("%Y-%m")
    if key in KNOWN_RECESSION_NAMES:
        return KNOWN_RECESSION_NAMES[key]
    # tolerate a 1-2 month dating difference vs the NBER table
    for offset in (-2, -1, 1, 2):
        k2 = (onset + pd.DateOffset(months=offset)).strftime("%Y-%m")
        if k2 in KNOWN_RECESSION_NAMES:
            return KNOWN_RECESSION_NAMES[k2]
    return f"recession onset {key}"


def _verdict_for_onset(
    proba: pd.Series, onset: pd.Timestamp, threshold: float,
    *, standing_lead: Optional[int], lead_window_months: int,
) -> dict:
    """The recession-by-recession verdict, using a lead window matched to
    the model's own forecast horizon.

    A recession is CALLED if the predicted probability crossed `threshold`
    AT ANY POINT in the lead window [onset - lead_window_months, onset) —
    not only if a warning was still standing AT the onset. The yield
    curve characteristically inverts ~12 months ahead and then re-steepens
    (the Fed starts cutting) so the warning often fades before the onset;
    scoring only "standing at onset" would fail the model on its own
    normal mechanism.

    `lead_window_months` MUST match what the model forecasts: an h=12
    model gets ~18 months; an h=3 model is near-coincident and gets ~6.
    Scoring a short-horizon model on a long window asks it for a warning
    it was never built to give.

    `standing_lead` is the continuous-warning-into-onset figure from
    measure_lead_times — used here only as a secondary annotation (did
    the warning fade, and if so when), NOT as the headline verdict.

    Returns {'verdict', 'first_cross_lead', 'peak_proba', 'peak_lead',
             'standing_lead', 'faded', 'lead_window_months'}.
    """
    window_start = onset - pd.DateOffset(months=lead_window_months)
    # months in [window_start, onset) that actually exist in proba
    # (the OOS series can have gaps — recession-free spans drop folds;
    #  a missing month is simply absent, never treated as below-threshold)
    window = proba[(proba.index >= window_start) & (proba.index < onset)]

    if len(window) == 0:
        return {"verdict": "NO COVERAGE", "first_cross_lead": None,
                "peak_proba": None, "peak_lead": None,
                "standing_lead": standing_lead, "faded": None,
                "lead_window_months": lead_window_months}

    crossings = window[window >= threshold]
    peak_proba = float(window.max())
    peak_month = window.idxmax()
    peak_lead = _months_between(peak_month, onset)

    if len(crossings) == 0:
        # never crossed the threshold anywhere in the lead window
        return {"verdict": "MISSED", "first_cross_lead": None,
                "peak_proba": peak_proba, "peak_lead": peak_lead,
                "standing_lead": standing_lead, "faded": None,
                "lead_window_months": lead_window_months}

    # CALLED — earliest crossing in the window
    first_cross = crossings.index.min()
    first_cross_lead = _months_between(first_cross, onset)
    # faded if the warning was NOT still standing at the onset
    faded = (standing_lead is None) or (standing_lead == 0)
    return {"verdict": "CALLED", "first_cross_lead": first_cross_lead,
            "peak_proba": peak_proba, "peak_lead": peak_lead,
            "standing_lead": standing_lead, "faded": faded,
            "lead_window_months": lead_window_months}


def _months_between(earlier: pd.Timestamp, later: pd.Timestamp) -> int:
    """Whole months from `earlier` to `later` (>= 0 when earlier <= later)."""
    return ((later.year - earlier.year) * 12
            + (later.month - earlier.month))


def _score_one_horizon(
    label: str, target: str, horizon: str, model: str,
    *, db_path, min_history_year, true_onsets, **wf_kwargs,
) -> dict:
    """Run the lead-time engine for one horizon and split its onsets into
    out-of-sample (Tier 1) and in-sample (Tier 2).

    `model` selects which model to score at this horizon — 'M1' (yield
    curve) for h=12, 'M2' (macro logit) for the short horizons, per
    B-track. Scoring the wrong model at a horizon tests it where it is
    not designed to work.

    `true_onsets` are the REALIZED recession onset dates (from the h=0
    target — the month a recession actually began). They are passed in so
    every horizon is scored against the SAME true dates. This is critical:
    each horizon's own target is shifted forward by the horizon length, so
    onsets read off a shifted target land h months early. Lead time and
    hit/miss must be measured against the true onset, not the shifted one.
    """
    res = run_lead_time_analysis(
        target=target, horizon=horizon, model=model,
        min_history_year=min_history_year, db_path=db_path,
        thresholds=[REPORT_CARD_THRESHOLD], **wf_kwargs,
    )
    if res.get("error") or "proba" not in res:
        return {"label": label, "horizon": horizon,
                "error": res.get("error", "no OOS predictions")}

    proba = res["proba"]

    # the OOS period is the span the pooled walk-forward predictions
    # cover. A realized recession onset INSIDE that span is genuinely
    # out-of-sample; an onset before it is in-sample.
    oos_start = proba.index.min()
    oos_end = proba.index.max()

    # continuous-warning-into-onset figure (used as a fade annotation)
    lt = measure_lead_times(proba, true_onsets, REPORT_CARD_THRESHOLD)
    standing_leads = lt["lead_times"]

    # the lead window is matched to THIS horizon — an h=3 model is scored
    # on a short window, an h=12 model on a long one.
    lead_window = _lead_window_for(horizon)

    tier1, tier2 = [], []
    for onset in true_onsets:
        in_oos = oos_start <= onset <= oos_end
        entry = {
            "onset": onset, "name": _name_for(onset),
            "in_oos": in_oos,
        }
        if in_oos:
            v = _verdict_for_onset(
                proba, onset, REPORT_CARD_THRESHOLD,
                standing_lead=standing_leads.get(onset),
                lead_window_months=lead_window)
            entry.update(v)
            entry["called"] = (v["verdict"] == "CALLED")
            tier1.append(entry)
        else:
            # in-sample: the model trained on this period, no honest
            # OOS prediction exists. Recorded for context only.
            tier2.append(entry)

    return {
        "label": label, "horizon": horizon,
        "oos_start": oos_start, "oos_end": oos_end,
        "tier1_oos": tier1, "tier2_in_sample": tier2,
        "n_oos_months": res["n_oos_months"],
        "error": None,
    }


def build_recession_report_card(
    *,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
    **wf_kwargs,
) -> dict:
    """Build the three-tier recession report card across the model ladder
    (M1 h=12, plus h=6 and h=3).

    Returns {'horizons': [per-horizon dict...], 'threshold': float,
             'true_onsets': [...]}.
    """
    # the REALIZED recession label (h=0 — no horizon shift) gives the
    # TRUE onset dates. Every horizon is scored against these same dates,
    # so a horizon's forward-shifted target cannot move the onset.
    target_name = REPORT_CARD_LADDER[0][1]   # 'T1'
    load_kwargs = {}
    if db_path is not None:
        load_kwargs["db_path"] = Path(db_path)
    realized = load_targets(target_name, "h=0", **load_kwargs)
    true_onsets = find_recession_onsets(realized)

    horizons = []
    for label, target, horizon, model in REPORT_CARD_LADDER:
        horizons.append(_score_one_horizon(
            label, target, horizon, model,
            db_path=db_path, min_history_year=min_history_year,
            true_onsets=true_onsets, **wf_kwargs))
    return {"horizons": horizons, "threshold": REPORT_CARD_THRESHOLD,
            "true_onsets": true_onsets}


def print_recession_report_card(card: dict) -> None:
    """Print the three-tier recession report card."""
    print("=" * 74)
    print("RECESSION REPORT CARD — per-recession warning track record")
    print("=" * 74)
    print(f"  warning threshold: probability >= {card['threshold']}")
    print()

    for h in card["horizons"]:
        print("-" * 74)
        if h.get("error"):
            print(f"  {h['label']}: ERROR — {h['error']}")
            continue
        print(f"  HORIZON: {h['label']}")
        print(f"  out-of-sample period: "
              f"{h['oos_start']:%Y-%m} .. {h['oos_end']:%Y-%m}  "
              f"({h['n_oos_months']} months)")
        print(f"  lead window for this horizon: "
              f"{_lead_window_for(h['horizon'])} months "
              f"(matched to what an {h['horizon']} model forecasts)")
        print()

        # TIER 1 — the real track record
        print("  TIER 1 — REAL TRACK RECORD (genuine out-of-sample):")
        if not h["tier1_oos"]:
            print("    (no recession onsets fall in the out-of-sample "
                  "period)")
        else:
            for e in h["tier1_oos"]:
                print(f"    {e['onset']:%Y-%m}  {e['name']}")
                v = e.get("verdict")
                if v == "CALLED":
                    line = (f"CALLED — first warning {e['first_cross_lead']} "
                            f"months before onset; peak probability "
                            f"{e['peak_proba']:.2f} at {e['peak_lead']} "
                            f"months out")
                    print(f"        -> {line}")
                    if e.get("faded"):
                        print(f"           (warning had FADED below the "
                              f"threshold by the onset — the curve "
                              f"re-steered; an early call, not a "
                              f"standing one)")
                    else:
                        sl = e.get("standing_lead")
                        print(f"           (warning still standing at "
                              f"onset — {sl} months continuous)")
                elif v == "MISSED":
                    pk = e.get("peak_proba")
                    pk_s = f"{pk:.2f}" if pk is not None else "n/a"
                    lw = e.get("lead_window_months", "?")
                    print(f"        -> MISSED — probability never crossed "
                          f"{card['threshold']} in the {lw}-month lead "
                          f"window before onset (peak {pk_s})")
                elif v == "NO COVERAGE":
                    print(f"        -> not covered — onset at the edge of "
                          f"the out-of-sample span, no lead window")
                else:
                    print(f"        -> {v}")

        # TIER 2 — in-sample context
        print()
        print("  TIER 2 — IN-SAMPLE CONTEXT (model trained on these —")
        print("           NOT a real test, shown for context only):")
        if not h["tier2_in_sample"]:
            print("    (none — all recessions in the data are "
                  "out-of-sample)")
        else:
            for e in h["tier2_in_sample"]:
                print(f"    {e['onset']:%Y-%m}  {e['name']}  "
                      f"[in-sample — not scored]")
        print()

    # TIER 3 — the methodology caveat, on every report
    print("=" * 74)
    print("  TIER 3 — WHAT THIS TEST IS (read before citing the above)")
    print("  - Tier 1 is GENUINE out-of-sample: the model did not train")
    print("    on those recessions. It is a real track record.")
    print("  - The evaluation is vintage-aware (point-in-time data), but")
    print("    it is NOT a full real-time vintage simulation — treat it")
    print("    as strong pseudo-out-of-sample, not live-forecast proof.")
    print("  - Tier 2 recessions are in-sample: the model was fit on")
    print("    them. They are context, never a track record.")
    print("  - Expect 2008 to be caught and 2020 to be missed: the")
    print("    recession-forecasting literature finds the COVID")
    print("    recession was an exogenous shock, unpredictable by any")
    print("    economic model. A 'missed 2020' is the honest, expected")
    print("    result — not a model failure.")
    print("  - CALLED means the probability crossed the threshold at")
    print("    some point in the lead window before the onset — the")
    print("    standard the literature applies to the yield curve, which")
    print("    characteristically inverts ~12 months ahead then fades.")
    print("    A faded warning is still a real early call.")
    print("  - This card scores HITS and MISSES (did each recession get")
    print("    a warning). It does NOT score false alarms — warnings")
    print("    that fired with no recession after. For false-alarm rate,")
    print("    use lead_time.run_lead_time_analysis's threshold sweep.")
    print("=" * 74)


# =============================================================================
# Onset-window inspection — "why did the report card say MISSED?"
# =============================================================================

def inspect_onset_window(
    onset: str,
    horizon: str = "h=3",
    *,
    model: str = "M2",
    target: str = "T1",
    context_months: int = 30,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
    **wf_kwargs,
) -> dict:
    """Print a model's month-by-month OOS probability around one recession
    onset — the diagnostic behind a report-card verdict.

    The report card's verdict for an onset depends on whether the
    probability crossed the threshold INSIDE that horizon's lead window.
    When a model ranks well overall (good reliability diagram) yet the
    report card says MISSED, the question is WHEN the model actually
    fired — inside the window, or just outside it. This prints exactly
    that: each month's probability for `context_months` months around the
    onset, marking the lead window and the threshold crossings.

    Returns {'onset', 'horizon', 'model', 'lead_window', 'window_rows',
             'crossed_in_window', 'crossings_outside'}.
    """
    onset_ts = pd.Timestamp(onset)
    lead_window = _lead_window_for(horizon)

    res = run_lead_time_analysis(
        target=target, horizon=horizon, model=model,
        min_history_year=min_history_year, db_path=db_path,
        thresholds=[REPORT_CARD_THRESHOLD], **wf_kwargs,
    )
    if res.get("error") or "proba" not in res:
        return {"onset": onset, "horizon": horizon, "model": model,
                "error": res.get("error", "no OOS predictions")}

    proba = res["proba"]
    lo = onset_ts - pd.DateOffset(months=context_months)
    hi = onset_ts + pd.DateOffset(months=6)
    ctx = proba[(proba.index >= lo) & (proba.index <= hi)]

    window_start = onset_ts - pd.DateOffset(months=lead_window)
    rows = []
    crossed_in_window = False
    crossings_outside = []
    for d, p in ctx.items():
        in_window = window_start <= d < onset_ts
        crossed = p >= REPORT_CARD_THRESHOLD
        if crossed and in_window:
            crossed_in_window = True
        if crossed and not in_window:
            crossings_outside.append((d, float(p)))
        rows.append({"month": d, "proba": float(p),
                     "in_window": in_window, "crossed": crossed})

    return {"onset": onset, "horizon": horizon, "model": model,
            "lead_window": lead_window,
            "window_start": window_start, "onset_ts": onset_ts,
            "window_rows": rows,
            "crossed_in_window": crossed_in_window,
            "crossings_outside": crossings_outside,
            "error": None}


def print_onset_window(inspection: dict) -> None:
    """Print the onset-window inspection."""
    print("=" * 70)
    print(f"ONSET-WINDOW INSPECTION — {inspection.get('model')} at "
          f"{inspection.get('horizon')}, onset {inspection.get('onset')}")
    print("=" * 70)
    if inspection.get("error"):
        print(f"  ERROR: {inspection['error']}")
        print("=" * 70)
        return

    lw = inspection["lead_window"]
    print(f"  lead window: {lw} months before onset "
          f"({inspection['window_start']:%Y-%m} .. "
          f"{inspection['onset_ts']:%Y-%m})")
    print(f"  threshold: probability >= {REPORT_CARD_THRESHOLD}")
    print()
    print(f"  {'month':>9} {'proba':>8}  {'in window':>10}  flag")
    for r in inspection["window_rows"]:
        win = "yes" if r["in_window"] else ""
        flag = ""
        if r["crossed"] and r["in_window"]:
            flag = "<-- CROSSED, in window (would be CALLED)"
        elif r["crossed"]:
            flag = "<-- crossed, OUTSIDE window"
        print(f"  {r['month']:%Y-%m} {r['proba']:>8.3f}  {win:>10}  {flag}")
    print()
    if inspection["crossed_in_window"]:
        print("  VERDICT: the probability DID cross the threshold inside")
        print("  the lead window — this onset should read CALLED.")
    else:
        print("  VERDICT: the probability never crossed the threshold")
        print("  inside the lead window — the report card's MISSED is")
        print("  correct for this window.")
        if inspection["crossings_outside"]:
            cs = inspection["crossings_outside"]
            print(f"  BUT the model DID cross the threshold "
                  f"{len(cs)} time(s) OUTSIDE the window:")
            for d, p in cs:
                lead = _months_between(d, inspection["onset_ts"])
                where = (f"{lead} months before onset" if lead > 0
                         else f"{-lead} months after onset")
                print(f"    {d:%Y-%m}  proba {p:.3f}  ({where})")
            print("  -> the model fired for this recession, but not")
            print("     within the window matched to this horizon.")
            print("     Either the window is too narrow for how this")
            print("     model leads, or the signal genuinely came at a")
            print("     different lead than the horizon implies.")
    print("=" * 70)
