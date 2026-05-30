"""
recession/dashboard.py

The recession-model research dashboard — full model ladder.

WHAT THIS DASHBOARD SHOWS
-------------------------
The recession model is a LADDER of horizon-specific models, and the
dashboard presents them BY ROLE, because they do different jobs:

  EARLY-WARNING GAUGE — M1, the yield-curve probit, at the 12-month
    horizon. This is the leading indicator: it called the 2008 GFC ~17
    months ahead. It is the headline "is a recession coming" gauge.

  NEAR-COINCIDENT GAUGES — M2, the macro logit, at the 6- and 3-month
    horizons. These are NOWCASTING gauges: they detect recession
    conditions as they arrive, not far in advance. A HIGH reading is a
    real alarm; a LOW reading is NOT an all-clear (it only means the
    macro data shows nothing yet — rely on the 12-month gauge for
    advance warning).

  REPORT CARD — the per-recession out-of-sample track record.

  CREDIT WATCH — a credit-stress monitoring panel (EBP / BAA10Y /
    private-credit), separate from the models.

WARNING THRESHOLDS
------------------
Each gauge uses a per-horizon warning threshold chosen by the ROC /
Kuiper-score analysis (recession/validation/threshold_analysis.py), NOT
the naive 0.5. Recessions are rare, so model probabilities are
compressed; the honest thresholds are well below 0.5. The thresholds
here are the analysis's recommended values and are INDICATIVE — they
rest on few out-of-sample recessions.

This is a RESEARCH dashboard, not a trading product. It states its own
caveats throughout.

Run (on a machine with streamlit installed):
    streamlit run recession/dashboard.py

The dashboard reads recession.db and calls the same validated code paths
(run_m1, run_m2, run_lead_time_analysis, the report card) used
throughout the project — no separate, unvalidated logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- import path shim --------------------------------------------------
# Streamlit runs this script with recession/ (the script's own directory)
# on sys.path, NOT the project root. The dashboard imports the `recession`
# PACKAGE, which lives one level up. Add the project root before those
# imports run, so `streamlit run recession/dashboard.py` works anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

# Streamlit is imported lazily inside main() so this module can be
# imported (and syntax-checked / partially tested) without streamlit.


# --- per-horizon warning thresholds ------------------------------------
# From the ROC / Kuiper-score threshold analysis. NOT 0.5 — recessions
# are rare so probabilities are compressed; these are the Kuiper-optimal
# operating points. Indicative (few OOS recessions), not precise.
WARNING_THRESHOLDS = {
    "h=12": 0.20,
    "h=6": 0.10,
    "h=3": 0.15,
}

# the ladder: (horizon, model, role label, model description)
LADDER = [
    ("h=12", "M1", "EARLY-WARNING", "yield-curve probit"),
    ("h=6", "M2", "NEAR-COINCIDENT", "macro logit"),
    ("h=3", "M2", "NEAR-COINCIDENT", "macro logit"),
]


def _default_db_path() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "recession.db", here / "recession.db",
                 Path.cwd() / "recession.db"):
        if cand.exists():
            return cand
    return here.parent / "recession.db"


# =============================================================================
# Data assembly — pure functions, testable without streamlit
# =============================================================================

def horizon_probability(db_path: Path, horizon: str, model: str) -> dict:
    """Fit the horizon's model on all available history and return the
    latest recession probability plus the full historical series.

    `model` is 'M1' (yield-curve probit) or 'M2' (macro logit).

    Returns {'horizon', 'model', 'latest_month', 'latest_proba',
             'history': Series, 'actual': Series, 'threshold': float}.
    """
    from recession.features.builder import build_feature_dataframe

    if model == "M1":
        from recession.models.m1_probit import M1Probit, M1_FEATURES
        feats = M1_FEATURES
        estimator = M1Probit()
    elif model == "M2":
        from recession.models.m2_logit import M2Logit, M2_FEATURES
        feats = M2_FEATURES
        estimator = M2Logit(C=1.0)
    else:
        raise ValueError(f"unknown model {model!r}")

    fr = build_feature_dataframe(
        target="T1", horizon=horizon,
        as_of="today", train_cutoff="today",
        feature_subset=feats, db_path=db_path,
    )
    X = fr.X[[c for c in feats if c in fr.X.columns]]
    y = fr.y
    mask = X.notna().all(axis=1)
    X_ok = X.loc[mask]
    train_mask = mask & y.notna()
    model_fitted = estimator.fit(X.loc[train_mask],
                                 y.loc[train_mask].astype(int))
    proba = pd.Series(model_fitted.predict_proba(X_ok), index=X_ok.index)

    latest_month = proba.index.max()
    return {
        "horizon": horizon, "model": model,
        "latest_month": latest_month,
        "latest_proba": float(proba.loc[latest_month]),
        "history": proba,
        "actual": y.reindex(proba.index),
        "threshold": WARNING_THRESHOLDS.get(horizon, 0.5),
    }


def gauge_status(proba: float, threshold: float) -> tuple[str, str]:
    """Map a probability + threshold to a (level, message) pair.

    Levels: 'elevated' (>= threshold), 'watch' (>= half threshold),
    'low' (below). The message is plain-language and honest about what a
    low reading does and does not mean."""
    if proba >= threshold:
        return ("elevated",
                f"Elevated — above the {threshold:.2f} warning threshold.")
    if proba >= threshold / 2:
        return ("watch",
                f"Watch — climbing, but below the {threshold:.2f} "
                f"threshold.")
    return ("low",
            f"Low — below the {threshold:.2f} threshold. Note: a low "
            f"reading here is not an all-clear, only the absence of a "
            f"signal so far.")


def ladder_summary() -> list[dict]:
    """The model ladder as a plain table for display."""
    return [
        {"horizon": "h=12", "model": "M1 — yield-curve probit",
         "role": "early-warning",
         "note": "leads ~12 months; called the 2008 GFC ~17 months out"},
        {"horizon": "h=6", "model": "M2 — macro logit",
         "role": "near-coincident",
         "note": "detects recession conditions as they arrive"},
        {"horizon": "h=3", "model": "M2 — macro logit",
         "role": "near-coincident",
         "note": "shortest horizon; a confirmation gauge, not a forecast"},
    ]


# =============================================================================
# The Streamlit app
# =============================================================================

def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Recession Model", layout="wide")
    db_path = _default_db_path()

    st.title("US Recession Model — Research Dashboard")
    st.caption("A ladder of horizon-specific models. M1 (yield curve) is "
               "the 12-month early-warning gauge; M2 (macro) gives the "
               "3- and 6-month near-coincident gauges. Research tool, "
               "not a market-timing product.")

    if not db_path.exists():
        st.error(f"recession.db not found (looked at {db_path}).")
        return

    # ---- the three gauges ----------------------------------------------
    st.subheader("Current readings")
    st.markdown(
        "Each gauge uses a warning threshold chosen by the ROC / "
        "Kuiper-score analysis — **not** 0.5. Recessions are rare, so "
        "model probabilities are compressed; the honest thresholds are "
        "lower (and indicative — they rest on few out-of-sample "
        "recessions).")

    gauges = []
    for horizon, model, role, _desc in LADDER:
        try:
            gauges.append(horizon_probability(db_path, horizon, model))
        except Exception as e:
            gauges.append({"horizon": horizon, "model": model,
                           "error": str(e)})

    cols = st.columns(3)
    for col, (horizon, model, role, desc), g in zip(
            cols, LADDER, gauges):
        with col:
            st.markdown(f"**{horizon} — {role}**")
            st.caption(f"{model} ({desc})")
            if g.get("error"):
                st.error(f"unavailable: {g['error']}")
                continue
            prob = g["latest_proba"]
            thr = g["threshold"]
            level, msg = gauge_status(prob, thr)
            st.metric(
                label=f"P(recession) — as of {g['latest_month']:%b %Y}",
                value=f"{prob*100:.0f}%",
                delta=f"threshold {thr*100:.0f}%", delta_color="off")
            if level == "elevated":
                st.warning(msg)
            elif level == "watch":
                st.info(msg)
            else:
                st.success(msg)

    # the honest framing note — central to the dashboard
    st.info(
        "**Reading the ladder.** The **h=12 early-warning gauge** is the "
        "one to watch for *whether a recession is coming* — it leads by "
        "about a year. The **h=3 / h=6 near-coincident gauges** detect a "
        "recession *as it arrives*; a high reading there is a real "
        "alarm, but a low reading is **not** an all-clear — it only "
        "means the macro data shows nothing yet. For advance warning, "
        "rely on the 12-month gauge.")

    # ---- history charts -------------------------------------------------
    st.subheader("Historical probability — by horizon")
    for (horizon, model, role, desc), g in zip(LADDER, gauges):
        if g.get("error"):
            continue
        st.markdown(f"**{horizon} ({role}) — {model}**")
        st.line_chart(pd.DataFrame(
            {"recession probability": g["history"]}))
    st.caption("Each model's estimated recession probability over the "
               "full sample. The early-warning model (h=12) peaks well "
               "before recessions; the near-coincident models (h=3/h=6) "
               "peak close to them.")

    # ---- per-recession report card -------------------------------------
    st.subheader("Track record — per-recession report card")
    with st.spinner("Building the out-of-sample report card..."):
        try:
            from recession.validation.recession_report_card import (
                build_recession_report_card)
            card = build_recession_report_card(db_path=db_path)
            for h in card["horizons"]:
                if h.get("error"):
                    st.info(f"{h['label']}: {h['error']}")
                    continue
                st.markdown(f"**{h['label']}**")
                rows = []
                for e in h["tier1_oos"]:
                    verdict = e.get("verdict", "?")
                    detail = ""
                    if verdict == "CALLED":
                        detail = (f"first warning "
                                  f"{e.get('first_cross_lead')} months "
                                  f"out; peak {e.get('peak_proba'):.2f}")
                        if e.get("faded"):
                            detail += " (faded before onset)"
                    elif verdict == "MISSED":
                        pk = e.get("peak_proba")
                        detail = (f"peak {pk:.2f} — never crossed"
                                  if pk is not None else "")
                    rows.append({
                        "recession": e["name"],
                        "verdict": verdict,
                        "detail": detail})
                if rows:
                    st.table(pd.DataFrame(rows))
                else:
                    st.caption("no out-of-sample recessions for this "
                               "horizon.")
            st.caption("Tier 1 (genuine out-of-sample) only. Expect 2008 "
                       "caught by the early-warning model and 2020 "
                       "missed — COVID was an exogenous shock the "
                       "literature agrees was unpredictable.")
        except Exception as e:
            st.info(f"Report card unavailable: {e}")

    # ---- lead time / threshold sweep -----------------------------------
    st.subheader("Lead time and false alarms — h=12 early-warning model")
    with st.spinner("Running out-of-sample lead-time analysis..."):
        try:
            from recession.validation.lead_time import (
                run_lead_time_analysis)
            lt = run_lead_time_analysis(db_path=db_path, model="M1",
                                        horizon="h=12")
            rows = []
            for thr, s in sorted(lt["sweep"].items()):
                rows.append({
                    "threshold": f"{thr:.2f}",
                    "TPR": (f"{s['tpr']:.2f}"
                            if s.get("tpr") is not None else "n/a"),
                    "FPR": (f"{s['fpr']:.2f}"
                            if s.get("fpr") is not None else "n/a"),
                    "mean lead (mo)": f"{s['mean_lead']:.1f}",
                })
            st.table(pd.DataFrame(rows))
            st.caption("TPR = true-positive rate; FPR = false-positive "
                       "rate (1 - specificity, the literature-standard "
                       "rate). Lower thresholds warn earlier but raise "
                       "the FPR. The operating point is the user's "
                       "choice; the gauge above uses the Kuiper-optimal "
                       "0.20.")
        except Exception as e:
            st.info(f"Lead-time analysis unavailable: {e}")

    # ---- credit watch ---------------------------------------------------
    st.subheader("Credit watch — credit-market stress monitor")
    try:
        from recession.credit_watch import credit_watch
        panel = credit_watch(db_path=db_path)
        cw_cols = st.columns(3)
        with cw_cols[0]:
            ebp = panel.get("ebp")
            if ebp:
                st.metric("EBP (excess bond premium)",
                          f"{ebp['value']:.2f}", delta=ebp["band"],
                          delta_color="off")
            else:
                st.caption("EBP: no data")
        with cw_cols[1]:
            baa = panel.get("baa10y")
            if baa:
                st.metric("BAA10Y spread",
                          f"{baa['value']:.2f}pp", delta=baa["band"],
                          delta_color="off")
            else:
                st.caption("BAA10Y: no data")
        with cw_cols[2]:
            pc = panel.get("private_credit")
            if pc:
                st.metric("Private-credit default rate",
                          f"{pc['default_rate_pct']:.1f}%",
                          delta=pc["band"], delta_color="off")
                if panel.get("private_credit_stale"):
                    st.warning("private-credit figure is stale — "
                               "update it")
            else:
                st.caption("private credit: no manual figure set")
        st.caption("A monitoring panel, not a predictor. EBP and BAA10Y "
                   "are public-market gauges and can miss private-credit "
                   "stress until it reaches public markets.")
    except Exception as e:
        st.info(f"Credit watch unavailable: {e}")

    # ---- the model ladder explainer ------------------------------------
    st.subheader("The model ladder")
    st.table(pd.DataFrame(ladder_summary()))
    st.markdown(
        "**The finding.** At the 12-month horizon the Treasury "
        "yield-curve spread dominates — no regularized linear model, "
        "tree ensemble, or boosting model robustly beats it (the M1-M5 "
        "result). The macro features carry signal only at SHORT "
        "horizons, where M2 serves as a near-coincident detector. The "
        "two are complementary: the yield curve is the leading "
        "indicator, the macro model is the confirmation.")

    st.divider()
    st.caption("Recession model research project. Probabilities from "
               "models fitted on all history; track record and "
               "threshold figures from walk-forward out-of-sample "
               "validation. Warning thresholds are Kuiper-optimal and "
               "indicative, not precise.")


if __name__ == "__main__":
    main()
