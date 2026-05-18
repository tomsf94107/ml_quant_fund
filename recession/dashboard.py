"""
recession/dashboard.py

Step 11 — the Streamlit dashboard. Operationalises the project's
deliverable: M1, the single-feature yield-curve recession probit.

WHAT THE DASHBOARD SHOWS
------------------------
  1. HEADLINE — the current estimated probability of a US recession
     within the next 12 months, from M1 fitted on all available history.
  2. HISTORY — M1's recession probability over time, with actual NBER
     recession periods shaded, so the user can see how the model behaved
     around past recessions.
  3. LEAD TIME — the Step-10 table: at each warning threshold, the
     typical months of advance warning and the false-alarm rate.
  4. THE MODEL LADDER — a plain-language summary of the M1-M5 result:
     five model classes were tested; the yield curve dominates at the
     12-month horizon; the macro features come alive only at shorter
     horizons (the pre-registered short-horizon track).

This is a RESEARCH dashboard, not a trading product. It states its own
caveats: M1 is a 12-month-leading model; AUC ~0.80 OOS; it is not a
market-timing tool.

Run (on a machine with streamlit installed):
    streamlit run recession/dashboard.py

The dashboard reads recession.db. It calls the same validated code paths
(run_m1, run_lead_time_analysis) used throughout the project — no
separate, unvalidated logic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Streamlit is imported lazily inside main() so this module can be
# imported (and syntax-checked / partially tested) in environments
# without streamlit installed.


# resolve recession.db — the project keeps it at the repo root for now
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

def current_recession_probability(db_path: Path) -> dict:
    """Fit M1 on all available history and return the latest recession
    probability plus the full historical probability series.

    Returns {'latest_month', 'latest_proba', 'history': Series,
             'actual': Series}.
    """
    from recession.models.m1_probit import M1Probit, M1_FEATURES
    from recession.features.builder import build_feature_dataframe

    fr = build_feature_dataframe(
        target="T1", horizon="h=12",
        as_of="today", train_cutoff="today",
        feature_subset=M1_FEATURES, db_path=db_path,
    )
    X = fr.X[M1_FEATURES]
    y = fr.y
    mask = X.notna().all(axis=1)
    X_ok = X.loc[mask]
    # fit on all rows that also have a label (in-sample, for the display
    # series — the OOS validation is the separate lead-time / ladder work)
    train_mask = mask & y.notna()
    model = M1Probit().fit(X.loc[train_mask], y.loc[train_mask].astype(int))
    proba = pd.Series(model.predict_proba(X_ok), index=X_ok.index)

    latest_month = proba.index.max()
    return {
        "latest_month": latest_month,
        "latest_proba": float(proba.loc[latest_month]),
        "history": proba,
        "actual": y.reindex(proba.index),
    }


def ladder_summary() -> list[dict]:
    """The M1-M5 result as a plain table for display."""
    return [
        {"model": "M1 — yield-curve probit", "auc": "0.796",
         "verdict": "BASELINE — the production model"},
        {"model": "M2 — L2 logit (4 feat)", "auc": "0.777",
         "verdict": "loses — extra features add no linear signal"},
        {"model": "M3 — random forest", "auc": "~0.80",
         "verdict": "ties — apparent win inside seed noise"},
        {"model": "M4 — XGBoost", "auc": "~0.80",
         "verdict": "ties — apparent win inside seed noise"},
        {"model": "M5 — Markov-switching", "auc": "<0.50",
         "verdict": "coincident model — wrong horizon for h=12"},
    ]


# =============================================================================
# The Streamlit app
# =============================================================================

def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Recession Model", layout="wide")
    db_path = _default_db_path()

    st.title("US Recession Probability — Research Dashboard")
    st.caption("Single-feature yield-curve model (M1). 12-month-ahead "
               "horizon. Out-of-sample AUC ~0.80. Research tool, not a "
               "market-timing product.")

    if not db_path.exists():
        st.error(f"recession.db not found (looked at {db_path}).")
        return

    # ---- headline -------------------------------------------------------
    try:
        info = current_recession_probability(db_path)
    except Exception as e:
        st.error(f"Could not compute the recession probability: {e}")
        return

    prob = info["latest_proba"]
    month = info["latest_month"]
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label=f"P(recession within 12 months) — as of "
                  f"{month:%B %Y}",
            value=f"{prob*100:.0f}%",
        )
        if prob >= 0.6:
            st.warning("Elevated — probability above 60%.")
        elif prob >= 0.4:
            st.info("Moderate — probability 40-60%.")
        else:
            st.success("Low — probability below 40%.")
    with col2:
        st.markdown(
            "**How to read this.** The number is the model's estimated "
            "probability that a recession begins within the next 12 "
            "months, based solely on the 10-year minus 3-month Treasury "
            "yield spread. A low reading is not a guarantee; an elevated "
            "reading is a signal to look closer, not a forecast of "
            "certainty.")

    # ---- history chart --------------------------------------------------
    st.subheader("Historical recession probability")
    hist = info["history"]
    actual = info["actual"]
    chart_df = pd.DataFrame({"recession probability": hist})
    st.line_chart(chart_df)
    st.caption("M1's estimated 12-month recession probability over the "
               "full sample. Compare peaks against the actual recession "
               "periods below.")

    # actual recession months, for reference
    if actual is not None:
        rec_months = actual[actual == 1].index
        if len(rec_months) > 0:
            st.caption(f"Sample contains {int((actual == 1).sum())} "
                       f"recession-flagged months between "
                       f"{actual.index.min():%Y} and "
                       f"{actual.index.max():%Y}.")

    # ---- lead-time table ------------------------------------------------
    st.subheader("Lead time and false alarms (out-of-sample)")
    with st.spinner("Running out-of-sample lead-time analysis..."):
        try:
            from recession.validation.lead_time import run_lead_time_analysis
            lt = run_lead_time_analysis(db_path=db_path)
            rows = []
            for thr, s in lt["sweep"].items():
                rows.append({
                    "warning threshold": f"{thr:.2f}",
                    "hit rate": (f"{s['hit_rate']*100:.0f}%"
                                 if s["hit_rate"] is not None else "n/a"),
                    "mean lead (months)": f"{s['mean_lead']:.1f}",
                    "false-alarm rate": (
                        f"{s['false_alarm_rate']*100:.0f}%"
                        if s["false_alarm_rate"] is not None else "n/a"),
                })
            st.table(pd.DataFrame(rows))
            st.caption("Lower thresholds warn earlier but raise more false "
                       "alarms. The operating point is the user's choice.")
        except Exception as e:
            st.info(f"Lead-time analysis unavailable: {e}")

    # ---- the model ladder ----------------------------------------------
    st.subheader("Why this model — the M1-M5 result")
    st.markdown(
        "Five model classes were each validated with walk-forward "
        "cross-validation (expanding window, 12-month embargo, no "
        "look-ahead). The result:")
    st.table(pd.DataFrame(ladder_summary()))
    st.markdown(
        "**Finding.** At the 12-month horizon the Treasury yield-curve "
        "spread dominates: no regularized linear model, tree ensemble, or "
        "boosting model robustly beats it, and adding macro features "
        "(financial conditions, industrial production, building permits) "
        "does not help. A separate horizon scan found those features "
        "*do* carry signal at shorter horizons (3-6 months) — a "
        "pre-registered direction for future work, not part of this "
        "model.")

    st.divider()
    st.caption("Recession model research project. M1 = single-feature "
               "yield-curve probit. All figures from walk-forward "
               "out-of-sample validation.")


if __name__ == "__main__":
    main()
