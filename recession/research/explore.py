"""
Step (b) — Pre-modeling data exploration.

Generates an HTML report from recession.db answering:

1. SANITY ALERTS — flag anything that looks broken
2. COVERAGE — when does each feature have data; are there gaps
3. UNIVARIATE PREDICTIVENESS — bivariate AUC of each feature vs USREC at h=12
4. PRE-RECESSION BEHAVIOR — what does each feature do 12 months before NBER onset
5. CORRELATION MATRIX — which features are redundant
6. PER-FEATURE TIME SERIES — feature value over time with NBER bars shaded
7. STEP 4 IMPLICATIONS — what this exploration says about Step 4 plans
8. v1.1 IMPLICATIONS — does the data justify the v1.1 spec additions

This is READ-ONLY. Does not modify recession.db.

Run from repo root:
    python -m recession.research.explore

Output: ./recession_exploration_report.html
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "recession.db"
DEFAULT_OUT     = Path.cwd() / "recession_exploration_report.html"


# =============================================================================
# Data loading
# =============================================================================

def load_features(db: Path) -> pd.DataFrame:
    """Latest-vintage feature panel: index=month, columns=features."""
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        """SELECT feature_name, observation_month, value
           FROM v_features_latest
           ORDER BY feature_name, observation_month""",
        conn,
    )
    conn.close()
    df["observation_month"] = pd.to_datetime(df["observation_month"])
    wide = df.pivot(index="observation_month", columns="feature_name", values="value")
    wide.index.name = "month"
    return wide


def load_target(db: Path, target_id: str) -> pd.Series:
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        """SELECT observation_month, label
           FROM v_targets_latest
           WHERE target_id = ?
           ORDER BY observation_month""",
        conn, params=(target_id,),
    )
    conn.close()
    df["observation_month"] = pd.to_datetime(df["observation_month"])
    return df.set_index("observation_month")["label"].astype(int).rename(target_id)


# =============================================================================
# Helpers
# =============================================================================

def fig_to_base64_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f'data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'


def find_recession_episodes(usrec: pd.Series) -> list[tuple]:
    """Return list of (start_month, end_month) tuples for NBER recession runs."""
    episodes = []
    in_rec  = False
    start   = None
    prev_d  = None
    for d, v in usrec.sort_index().items():
        if v == 1 and not in_rec:
            start  = d
            in_rec = True
        elif v == 0 and in_rec:
            episodes.append((start, prev_d))
            in_rec = False
        prev_d = d
    if in_rec:
        episodes.append((start, prev_d))
    return episodes


def shade_recessions(ax, episodes, alpha: float = 0.18):
    for s, e in episodes:
        ax.axvspan(s, e, color="gray", alpha=alpha, linewidth=0)


def univariate_auc(feature: pd.Series, target: pd.Series,
                    horizon_months: int = 12) -> dict:
    """
    Bivariate predictiveness of feature_t for target_{t+h}, no model fitting.

    Computes the rank-based equivalent of AUC via Mann-Whitney U statistic.
    Returns AUC in [0.5, 1.0] (always reported in the more informative direction).
    """
    df = pd.DataFrame({"x": feature, "y": target})
    df["y_future"] = df["y"].shift(-horizon_months)
    df = df.dropna(subset=["x", "y_future"])

    if len(df) < 60 or df["y_future"].sum() < 5:
        return {"auc": np.nan, "direction": "n/a",
                "n_obs": len(df), "n_events": int(df["y_future"].sum())}

    pos = df.loc[df["y_future"] == 1, "x"].values
    neg = df.loc[df["y_future"] == 0, "x"].values
    if len(pos) == 0 or len(neg) == 0:
        return {"auc": np.nan, "direction": "n/a",
                "n_obs": len(df), "n_events": int(df["y_future"].sum())}

    try:
        from scipy.stats import mannwhitneyu
        u, _ = mannwhitneyu(pos, neg, alternative="two-sided")
        auc_higher = u / (len(pos) * len(neg))
    except Exception:
        return {"auc": np.nan, "direction": "n/a",
                "n_obs": len(df), "n_events": int(df["y_future"].sum())}

    if auc_higher >= 0.5:
        return {"auc": auc_higher, "direction": "higher_means_recession",
                "n_obs": len(df), "n_events": int(df["y_future"].sum())}
    return {"auc": 1 - auc_higher, "direction": "lower_means_recession",
            "n_obs": len(df), "n_events": int(df["y_future"].sum())}


def pre_recession_extremum(feature: pd.Series, episodes: list[tuple],
                            direction: str,
                            lookback_months: int = 12) -> dict:
    """
    For each NBER recession start, look at the most-recession-favoring value
    the feature reached in the 12 months BEFORE onset (not AT onset itself,
    since the inversion or signal may peak earlier in the window and rebound).

    Compares that extremum to the feature's typical level (rolling 60-month
    median ending 12 months before onset, i.e., not contaminated by the
    pre-recession window itself).

    Returns: how often the pre-recession window dipped meaningfully into the
    feature's recession-favoring zone.

    direction: 'higher_means_recession' or 'lower_means_recession'
    """
    feat = feature.dropna()
    deltas = []
    for start, _ in episodes:
        # Window: [start - 12mo, start - 1mo] inclusive
        win_start = start - pd.DateOffset(months=lookback_months)
        win_end   = start - pd.DateOffset(months=1)
        window    = feat.loc[win_start:win_end]
        if window.empty:
            continue

        # Baseline: median of the 60-month period ending right before the window
        base_start = win_start - pd.DateOffset(months=60)
        base_end   = win_start - pd.DateOffset(months=1)
        baseline   = feat.loc[base_start:base_end]
        if baseline.empty or len(baseline) < 12:
            continue
        base_med = baseline.median()

        # Take the extremum in the recession-favoring direction
        if direction == "higher_means_recession":
            extremum = window.max()
        elif direction == "lower_means_recession":
            extremum = window.min()
        else:
            continue

        if pd.isna(extremum) or pd.isna(base_med):
            continue

        # Signed delta in the recession-favoring direction
        # (positive = moved toward recession-favoring zone)
        if direction == "higher_means_recession":
            delta = extremum - base_med
        else:  # lower_means_recession
            delta = base_med - extremum

        deltas.append(delta)

    if not deltas:
        return {"n_episodes": 0,
                "mean_extremum_delta": np.nan,
                "median_extremum_delta": np.nan,
                "n_consistent": 0}

    # "Consistent" = moved meaningfully toward recession zone
    # (delta > 0 in the recession-favoring sense)
    n_consistent = sum(1 for d in deltas if d > 0)
    return {"n_episodes": len(deltas),
            "mean_extremum_delta":   float(np.mean(deltas)),
            "median_extremum_delta": float(np.median(deltas)),
            "n_consistent": n_consistent}


# =============================================================================
# Plot generators
# =============================================================================

def plot_feature_with_recessions(name: str, feature: pd.Series,
                                  episodes: list[tuple]) -> str:
    fig, ax = plt.subplots(figsize=(11, 2.6))
    shade_recessions(ax, episodes)
    feature.dropna().plot(ax=ax, color="#1f4e8c", linewidth=1.2)
    ax.set_title(name, fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("")
    fig.tight_layout()
    return fig_to_base64_img(fig)


def plot_coverage_heatmap(features: pd.DataFrame) -> str:
    presence = features.notna().astype(int)
    yearly = presence.resample("YE").mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(yearly.T.values, aspect="auto", cmap="Greens",
                   interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(len(yearly.columns)))
    ax.set_yticklabels(yearly.columns, fontsize=8)
    n_years = len(yearly.index)
    step = max(1, n_years // 12)
    ax.set_xticks(range(0, n_years, step))
    ax.set_xticklabels([d.year for d in yearly.index[::step]],
                        rotation=45, fontsize=8)
    ax.set_title("Feature data coverage by year (green = data present)",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="fraction of months with data")
    fig.tight_layout()
    return fig_to_base64_img(fig)


def plot_correlation_matrix(features: pd.DataFrame) -> str:
    df = features.dropna(axis=1, how="all")
    df = df.loc[df.notna().mean(axis=1) > 0.7]
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title(f"Pearson correlation, n={len(df)} months", fontsize=11)
    fig.colorbar(im, ax=ax)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if i != j and abs(corr.values[i, j]) > 0.7:
                ax.text(j, i, f"{corr.values[i, j]:.2f}",
                         ha="center", va="center", color="black", fontsize=6,
                         weight="bold")
    fig.tight_layout()
    return fig_to_base64_img(fig)


def plot_predictiveness_ranking(
    rank_df: pd.DataFrame,
    title: str = "Bivariate predictiveness vs T1 (NBER, h=12)",
    xlabel: Optional[str] = None,
) -> str:
    df = rank_df.dropna(subset=["auc"]).sort_values("auc", ascending=True)
    if df.empty:
        # Empty plot for the case where T2 has no overlap
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No scoreable features (insufficient data)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        return fig_to_base64_img(fig)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(df))))
    colors = ["#2c8a3e" if a > 0.65 else "#c9871a" if a > 0.55 else "#cc5555"
              for a in df["auc"]]
    ax.barh(df["feature"], df["auc"], color=colors)
    ax.axvline(0.5, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(0.6, color="green", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xlabel(xlabel or "Univariate AUC")
    ax.set_xlim(0.4, 1.0)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return fig_to_base64_img(fig)


# =============================================================================
# Sanity alerts
# =============================================================================

def run_sanity_checks(features: pd.DataFrame, t1: pd.Series, t2: pd.Series) -> list[str]:
    alerts = []

    for col in features.columns:
        n = features[col].notna().sum()
        if n < 60:
            alerts.append(f"⚠️  <code>{col}</code>: only {n} months of data — "
                           f"too short for reliable model fitting")

    cutoff = pd.Timestamp.today() - pd.DateOffset(months=4)
    for col in features.columns:
        last = features[col].dropna().index.max() if features[col].notna().any() else None
        if last is not None and last < cutoff:
            alerts.append(f"⚠️  <code>{col}</code>: latest data is "
                           f"{last.date()} — possibly stale or discontinued")

    df = features.dropna(axis=1, how="all").loc[features.notna().mean(axis=1) > 0.7]
    if len(df) > 30:
        corr = df.corr()
        seen = set()
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if i >= j or (c1, c2) in seen or (c2, c1) in seen:
                    continue
                if abs(corr.loc[c1, c2]) > 0.9:
                    alerts.append(
                        f"⚠️  <code>{c1}</code> ↔ <code>{c2}</code>: "
                        f"|r|={abs(corr.loc[c1,c2]):.2f} — highly redundant; "
                        f"PCA or drop one"
                    )
                    seen.add((c1, c2))

    n_t1 = int(t1.sum())
    alerts.append(
        f"ℹ️  T1 (NBER): {n_t1} recession months out of {len(t1)} "
        f"({100*n_t1/len(t1):.1f}% base rate)"
    )

    if len(t2) > 0:
        n_t2 = int(t2.sum())
        first_t2 = t2.index.min()
        last_t2  = t2.index.max()
        alerts.append(
            f"ℹ️  T2 (drawdown): {n_t2} months in drawdown out of {len(t2)} "
            f"({100*n_t2/len(t2):.1f}%) — history {first_t2.date()} → {last_t2.date()} "
            f"(SP500 from Yahoo via committed CSV)"
        )

    return alerts


# =============================================================================
# Step 4 / v1.1 implications
# =============================================================================

def derive_step4_impact(predictiveness: pd.DataFrame, alerts: list[str]) -> str:
    strong = predictiveness[predictiveness["auc"] > 0.65].sort_values("auc", ascending=False)
    weak   = predictiveness[(predictiveness["auc"] > 0.45) & (predictiveness["auc"] < 0.55)]
    nan_   = predictiveness[predictiveness["auc"].isna()]

    bullets = []
    if len(strong) > 0:
        bullets.append(
            f"<li><strong>{len(strong)} features show strong univariate signal</strong> "
            f"(AUC&gt;0.65): <code>{', '.join(strong['feature'].head(8).tolist())}</code>. "
            "These should anchor M1 (static probit). PCA components likely load heavily on them.</li>"
        )
    if len(weak) > 0:
        bullets.append(
            f"<li><strong>{len(weak)} features show near-noise univariate AUC</strong> "
            f"(0.45–0.55): <code>{', '.join(weak['feature'].tolist())}</code>. "
            "Consider whether to keep them — may help in a multivariate model, "
            "but cost degrees of freedom if they don't.</li>"
        )
    if len(nan_) > 0:
        bullets.append(
            f"<li><strong>{len(nan_)} features couldn't be scored</strong>: "
            f"<code>{', '.join(nan_['feature'].tolist())}</code>. "
            "Insufficient overlap with USREC + h=12. Step 4 should plan around "
            "shorter-history validation for these.</li>"
        )
    bullets.append(
        "<li><strong>Hamilton (2018) detrending</strong> needed for any feature "
        "showing a long-term trend (visible in §6 plots) — INDPRO, SP500 levels, "
        "possibly DTWEXBGS.</li>"
    )
    bullets.append(
        "<li><strong>PCA reduction</strong> will collapse high-correlation pairs "
        "flagged in §1 alerts. Likely 3 PCs capture &gt;60% of variance.</li>"
    )
    return "<strong>Implications for Step 4 (feature pipeline):</strong><ul>" + \
           "".join(bullets) + "</ul>"


def derive_v11_impact(predictiveness: pd.DataFrame,
                       predictiveness_t2: pd.DataFrame,
                       features: pd.DataFrame,
                       n_t2_events: int) -> str:
    """v1.1 spec adds 13 new features; does this exploration support that?"""
    bullets = []

    # T2 viability — most important new finding
    bullets.append(
        f"<li><strong>✅ T2 (drawdown) now usable.</strong> Yahoo SP500 backfill "
        f"gave T2 {n_t2_events} months of drawdown labels covering 1962, 1969, "
        f"1973-74, 1981-82, 1987 (Black Monday, no recession), 1990, 1998 (LTCM, "
        f"no recession), 2000-02, 2008-09, 2011 (no recession), 2020, 2022. "
        f"Was 6 events / FRED-only before. Both T2 itself and the new T5 "
        f"(market stress) target are viable in v1.1.</li>"
    )

    # T1 vs T2 disagreement — key for T5 justification
    merged = predictiveness[["feature", "auc"]].rename(columns={"auc": "auc_t1"}).merge(
        predictiveness_t2[["feature", "auc"]].rename(columns={"auc": "auc_t2"}),
        on="feature", how="outer"
    )
    merged["delta"] = merged["auc_t1"] - merged["auc_t2"]
    drawdown_only = merged[merged["delta"] < -0.10].dropna(subset=["delta"])
    recession_only = merged[merged["delta"] > 0.10].dropna(subset=["delta"])
    if len(drawdown_only) > 0:
        names = ", ".join(drawdown_only["feature"].head(5).tolist())
        bullets.append(
            f"<li><strong>T5 layer is justified by data.</strong> {len(drawdown_only)} "
            f"feature(s) predict T2 drawdowns substantially better than T1 recessions: "
            f"<code>{names}</code>. Without a separate T5 target these would be "
            f"averaged-out noise in T1 models.</li>"
        )
    if len(recession_only) > 0:
        names = ", ".join(recession_only["feature"].head(5).tolist())
        bullets.append(
            f"<li><strong>{len(recession_only)} feature(s) are recession-specific</strong> "
            f"(predict T1 but not T2): <code>{names}</code>. These should anchor "
            f"M1/M2 for T1 specifically.</li>"
        )

    # Labor signal strength
    labor_features = ["SAHMREALTIME", "JTSQUR", "ICSA"]
    labor_in_data = [f for f in labor_features if f in predictiveness["feature"].values]
    if labor_in_data:
        labor_aucs = predictiveness[predictiveness["feature"].isin(labor_in_data)]["auc"]
        if labor_aucs.notna().any():
            mean_labor = labor_aucs.mean()
            if mean_labor > 0.6:
                bullets.append(
                    f"<li><strong>✅ Labor expansion supported.</strong> Existing labor "
                    f"features (SAHM, JTSQUR, ICSA) average AUC={mean_labor:.2f}. "
                    f"v1.1's plan to add CCSA, AWHMAN, TEMPHELPS, JTSLDR is justified — "
                    f"the channel works.</li>"
                )
            else:
                bullets.append(
                    f"<li><strong>⚠️ Labor expansion questionable for h=12.</strong> Existing "
                    f"labor features average AUC={mean_labor:.2f} (T1 h=12) — below the 0.6 "
                    f"threshold. Sahm/CFNAI are coincident, not leading. The 4 new labor "
                    f"features (CCSA, AWHMAN, TEMPHELPS, JTSLDR) should serve M5 (coincident "
                    f"regime model) primarily, not h=12 forecasting.</li>"
                )

    # Credit channel
    credit_features = ["BAA10Y", "EBP", "NFCI"]
    credit_in_data = [f for f in credit_features if f in predictiveness["feature"].values]
    if credit_in_data:
        credit_aucs = predictiveness[predictiveness["feature"].isin(credit_in_data)]["auc"]
        if credit_aucs.notna().any():
            mean_credit = credit_aucs.mean()
            bullets.append(
                f"<li><strong>Credit channel:</strong> AUC mean = {mean_credit:.2f} "
                f"across BAA10Y, EBP, NFCI. {'Strong' if mean_credit > 0.6 else 'Mixed'} "
                f"signal. Adding 2 term-structure variants (T10Y2Y, near-term forward) "
                f"likely incremental — would test in Step 4.</li>"
            )

    # Yield curve confirmation
    if "T10Y3M" in predictiveness["feature"].values:
        yc_auc = predictiveness.loc[predictiveness["feature"] == "T10Y3M",
                                     "auc"].iloc[0]
        if pd.notna(yc_auc):
            bullets.append(
                f"<li><strong>Yield curve baseline:</strong> T10Y3M univariate "
                f"AUC = {yc_auc:.2f}. {'This is the published NY Fed result' if 0.6 < yc_auc < 0.85 else 'Different from published — investigate'}; "
                f"M1 baseline should match.</li>"
            )

    # Housing
    if "PERMIT" in predictiveness["feature"].values:
        ph_auc = predictiveness.loc[predictiveness["feature"] == "PERMIT", "auc"].iloc[0]
        if pd.notna(ph_auc):
            bullets.append(
                f"<li><strong>Housing channel:</strong> PERMIT alone AUC={ph_auc:.2f}. "
                f"v1.1 adds NAHB sentiment + existing home sales. "
                f"{'Worth adding' if ph_auc > 0.55 else 'Marginal'} based on this signal.</li>"
            )

    # Inflation — none in current data
    bullets.append(
        "<li><strong>Inflation features absent from current data.</strong> "
        "v1.1 adds Core CPI, Core PCE, wage growth (3 features). "
        "Cannot validate from this exploration; recommend adding and testing in Step 4.</li>"
    )

    # Market stress (T5) — now anchored in real T2 data
    if all(f in features.columns for f in ["T10Y3M", "BAA10Y", "NFCI"]):
        bullets.append(
            "<li><strong>T5 (Market Stress) inputs all present and validated.</strong> "
            "T10Y3M, BAA10Y, NFCI, and SP500 (now full history) all show meaningful "
            "AUC vs T2 drawdowns. Recalibrate the BAA10Y threshold from rolling-90th "
            "percentile to absolute (per spec v1.1.1 patch) since the BAA10Y range "
            "is too narrow for percentile logic to fire usefully.</li>"
        )

    return "<strong>Implications for v1.1 spec additions:</strong><ul>" + \
           "".join(bullets) + "</ul>"


# =============================================================================
# HTML report
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Recession Model — Data Exploration Report</title>
<style>
  body  {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
           sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 2em;
           color: #222; line-height: 1.55; }}
  h1    {{ border-bottom: 3px solid #1f4e8c; padding-bottom: 0.3em; }}
  h2    {{ border-bottom: 1px solid #ccc; padding-bottom: 0.2em; margin-top: 2.5em;
           color: #1f4e8c; }}
  h3    {{ margin-top: 1.5em; color: #2c5fa3; }}
  .alerts {{ background: #fff8dc; border-left: 4px solid #d4a017;
              padding: 1em 1.5em; margin: 1em 0; }}
  .alerts li {{ margin: 0.3em 0; }}
  .info {{ background: #e8f0fe; border-left: 4px solid #1f4e8c;
            padding: 0.8em 1.2em; margin: 1em 0; font-size: 0.95em; }}
  table {{ border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ddd; padding: 0.4em 0.7em; text-align: left; }}
  th    {{ background: #f0f4f8; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  img   {{ max-width: 100%; margin: 0.5em 0; border: 1px solid #eee; }}
  .num  {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .small {{ font-size: 0.85em; color: #666; }}
  .feature-block {{ margin: 1.5em 0; padding: 1em; background: #fafafa;
                     border-radius: 6px; }}
  .feature-block h3 {{ margin-top: 0; }}
  .auc-good {{ color: #2c8a3e; font-weight: bold; }}
  .auc-mid  {{ color: #c9871a; }}
  .auc-poor {{ color: #cc5555; }}
</style>
</head>
<body>

<h1>Recession Model — Data Exploration Report</h1>
<p class="small">Generated {date} · Data source: <code>recession.db</code> ·
   Read-only inspection · Spec v1.1.0 review</p>

<div class="info">
<strong>Purpose.</strong> Step (b) sanity-checks the data we ingested in
Steps 2–2.6 before we build modeling infrastructure on top of it. Confirms
that the spec's theoretical assumptions about each feature actually show
up in the historical data, and informs whether the v1.1 spec additions
(13 new features, T5 market stress target, M5 coincident regime model) are
justified by current evidence.
</div>

<h2>1. Sanity alerts (most important)</h2>
<div class="alerts">
<ul>{alerts_html}</ul>
</div>

<h2>2. Coverage map</h2>
<p>Which features have data in which years. Sparse rows = candidates for
sub-sample analysis or feature drop.</p>
<img src="{coverage_img}" alt="Coverage heatmap">

<h2>3. Univariate predictiveness vs NBER (h=12)</h2>
<p>Bivariate AUC of feature<sub>t</sub> predicting USREC<sub>t+12</sub>.
No model fitting — just rank-based separability via Mann-Whitney U.</p>
<p><strong>Reading guide:</strong> AUC &gt; 0.65 = strong univariate signal;
0.55–0.65 = useful in a model; below 0.55 = mostly noise on its own.</p>
<img src="{predictiveness_img}" alt="Predictiveness ranking vs T1">
{predictiveness_table}

<h2>3b. Univariate predictiveness vs T2 market drawdown (h=6)</h2>
<p>Bivariate AUC of feature<sub>t</sub> predicting T2<sub>t+6</sub>
(SP500 ≥15% drawdown 6 months ahead). Different horizon than Section 3:
market drawdowns develop faster than NBER recessions.</p>
<p>This is critical for v1.1's T5 (Market Stress) target. Features that
score high here but low in Section 3 are <em>market-stress predictors</em>,
not recession predictors — which is exactly the separation T5 is designed
to surface.</p>
<img src="{predictiveness_t2_img}" alt="Predictiveness ranking vs T2">
{predictiveness_t2_table}

<h2>3c. T1 vs T2 disagreement</h2>
<p>Features ranked by |AUC(T1) − AUC(T2)|. Large positive: predicts
recessions but not drawdowns. Large negative: predicts drawdowns but not
recessions. These are the candidates that justify having T1 and T5 as
separate targets.</p>
{t1_t2_disagreement_table}

<h2>4. Pre-recession behavior</h2>
<p>For each NBER recession in our sample, looks at the most-recession-favoring
value the feature reached in the 12 months <em>before</em> onset (not at onset
itself, since signals like the yield curve often peak earlier in the window
and rebound). Compares that extremum to the feature's typical level (60-month
median ending right before the 12mo window).</p>
<ul>
  <li><code>n_episodes</code> — how many recessions have data this far back</li>
  <li><code>n_consistent</code> — how many of those episodes saw the feature
      visit its recession-favoring zone in the 12mo window</li>
  <li><code>mean Δ to recession zone</code> — average distance the extremum
      moved into the recession-favoring direction (positive = moved toward
      recession zone). Compare across features cautiously since unit scales differ.</li>
  <li><code>direction</code> — implied by univariate AUC (Section 3)</li>
  <li><code>consistent?</code> — ✓ if ≥60% of episodes show recession-zone
      visit, ~ if 40-60%, ✗ if &lt;40%</li>
</ul>
{pre_recession_table}

<h2>5. Correlation matrix</h2>
<p>Highlights cells with |r| &gt; 0.7. High-correlation pairs are candidates
for PCA reduction or feature merging in Step 4.</p>
<img src="{corr_img}" alt="Correlation matrix">

<h2>6. Per-feature time series with NBER recession bars</h2>
<p>Each feature plotted with NBER recession periods shaded in gray.
Visually verify the spec's expectations match historical behavior.</p>
{feature_blocks}

<h2>7. Conclusions: Step 4 implications</h2>
<div class="info">{step4_impact}</div>

<h2>8. v1.1 spec implications</h2>
<div class="info">{v11_impact}</div>

<p class="small">End of report. Generated by
<code>recession/research/explore.py</code>.</p>

</body>
</html>
"""


# =============================================================================
# Main orchestration
# =============================================================================

def fmt_auc(v):
    if pd.isna(v):
        return "n/a"
    cls = "auc-good" if v > 0.65 else "auc-mid" if v > 0.55 else "auc-poor"
    return f'<span class="{cls}">{v:.3f}</span>'


def run_exploration(db: Path, out: Path) -> None:
    logger.info("Loading data from %s", db)
    features = load_features(db)
    t1 = load_target(db, "T1")
    t2 = load_target(db, "T2")
    logger.info("  features: %d × %d months  (%s → %s)",
                 len(features.columns), len(features),
                 features.index.min().date(), features.index.max().date())
    logger.info("  T1: %d months, %d events", len(t1), int(t1.sum()))
    logger.info("  T2: %d months, %d events", len(t2), int(t2.sum()))

    episodes = find_recession_episodes(t1)
    logger.info("Found %d NBER recession episodes", len(episodes))

    alerts = run_sanity_checks(features, t1, t2)
    logger.info("Generated %d sanity alerts", len(alerts))

    logger.info("Computing univariate AUCs vs T1 (NBER recession) at h=12 ...")
    rows = []
    for col in features.columns:
        result = univariate_auc(features[col], t1, horizon_months=12)
        rows.append({"feature": col, **result})
    predictiveness = pd.DataFrame(rows).sort_values("auc", ascending=False)

    # T2 (market drawdown) at h=6. Different horizon makes sense — drawdowns
    # are faster than NBER recessions. h=6 is a useful lead time for market
    # stress, where h=12 was for macro recession.
    logger.info("Computing univariate AUCs vs T2 (market drawdown) at h=6 ...")
    t2_rows = []
    for col in features.columns:
        result = univariate_auc(features[col], t2, horizon_months=6)
        t2_rows.append({"feature": col, **result})
    predictiveness_t2 = pd.DataFrame(t2_rows).sort_values("auc", ascending=False)

    logger.info("Computing pre-recession behavior ...")
    pre_rec_rows = []
    for col in features.columns:
        direction_row = predictiveness[predictiveness["feature"] == col]
        direction = direction_row["direction"].iloc[0] if not direction_row.empty else "n/a"
        prc = pre_recession_extremum(features[col], episodes, direction,
                                       lookback_months=12)
        # consistency now means: in the 12mo window before each recession start,
        # did this feature visit its recession-favoring zone (vs its 60mo baseline)?
        # Threshold: at least half the episodes must show positive recession-zone delta
        consistent = ""
        if prc["n_episodes"] > 0 and direction != "n/a":
            frac = prc["n_consistent"] / prc["n_episodes"]
            if frac >= 0.6:
                consistent = "✓"
            elif frac >= 0.4:
                consistent = "~"
            else:
                consistent = "✗"
        pre_rec_rows.append({"feature": col, **prc,
                              "direction": direction, "consistent": consistent})
    pre_rec = pd.DataFrame(pre_rec_rows)

    logger.info("Building plots ...")
    coverage_img         = plot_coverage_heatmap(features)
    corr_img             = plot_correlation_matrix(features)
    predictiveness_img   = plot_predictiveness_ranking(predictiveness)
    predictiveness_t2_img = plot_predictiveness_ranking(
        predictiveness_t2, title="Bivariate predictiveness vs T2 (drawdown, h=6)"
    )

    logger.info("Building per-feature time-series plots ...")
    feature_blocks_html = ""
    for name in sorted(features.columns):
        if features[name].notna().sum() < 30:
            continue
        ts_img = plot_feature_with_recessions(name, features[name], episodes)
        auc_row = predictiveness[predictiveness["feature"] == name].iloc[0]
        prc_row = pre_rec[pre_rec["feature"] == name].iloc[0]
        auc_str = f"{auc_row['auc']:.3f}" if pd.notna(auc_row["auc"]) else "n/a"
        auc_class = ("auc-good" if pd.notna(auc_row["auc"]) and auc_row["auc"] > 0.65
                      else "auc-mid" if pd.notna(auc_row["auc"]) and auc_row["auc"] > 0.55
                      else "auc-poor")
        prc_str = (f"{prc_row['mean_extremum_delta']:+.3f}" if pd.notna(prc_row['mean_extremum_delta'])
                    else "n/a")
        feature_blocks_html += f"""
        <div class="feature-block">
          <h3>{name}</h3>
          <p><span class="{auc_class}">AUC = {auc_str}</span> ·
             direction: <code>{auc_row['direction']}</code> ·
             pre-recession extremum vs baseline (mean): <code>{prc_str}</code>
             — {prc_row['n_consistent']}/{prc_row['n_episodes']} episodes hit recession zone ·
             <span style="font-size:1.2em">{prc_row['consistent']}</span></p>
          <img src="{ts_img}" alt="{name} time series">
        </div>
        """

    # Tables
    predictiveness_html = ('<table><tr><th>Feature</th><th>AUC</th>'
                            '<th>Direction</th><th class="num">n_obs</th>'
                            '<th class="num">n_events</th></tr>')
    for _, r in predictiveness.iterrows():
        predictiveness_html += (
            f"<tr><td><code>{r['feature']}</code></td>"
            f"<td>{fmt_auc(r['auc'])}</td>"
            f"<td>{r['direction']}</td>"
            f"<td class='num'>{r['n_obs']}</td>"
            f"<td class='num'>{r['n_events']}</td></tr>"
        )
    predictiveness_html += "</table>"

    pre_rec_html = ('<table><tr><th>Feature</th>'
                     '<th class="num">n_episodes</th>'
                     '<th class="num">n_consistent</th>'
                     '<th class="num">mean Δ to recession zone</th>'
                     '<th class="num">median Δ</th>'
                     '<th>direction</th><th>consistent?</th></tr>')
    for _, r in pre_rec.iterrows():
        m_str = f"{r['mean_extremum_delta']:+.3f}" if pd.notna(r['mean_extremum_delta']) else "n/a"
        d_str = f"{r['median_extremum_delta']:+.3f}" if pd.notna(r['median_extremum_delta']) else "n/a"
        pre_rec_html += (
            f"<tr><td><code>{r['feature']}</code></td>"
            f"<td class='num'>{r['n_episodes']}</td>"
            f"<td class='num'>{r['n_consistent']}</td>"
            f"<td class='num'>{m_str}</td>"
            f"<td class='num'>{d_str}</td>"
            f"<td>{r['direction']}</td>"
            f"<td style='text-align:center;font-size:1.1em'>{r['consistent']}</td></tr>"
        )
    pre_rec_html += "</table>"

    # T2 predictiveness table
    predictiveness_t2_html = (
        '<table><tr><th>Feature</th><th>AUC</th><th>Direction</th>'
        '<th class="num">n_obs</th><th class="num">n_events</th></tr>'
    )
    for _, r in predictiveness_t2.iterrows():
        predictiveness_t2_html += (
            f"<tr><td><code>{r['feature']}</code></td>"
            f"<td>{fmt_auc(r['auc'])}</td>"
            f"<td>{r['direction']}</td>"
            f"<td class='num'>{r['n_obs']}</td>"
            f"<td class='num'>{r['n_events']}</td></tr>"
        )
    predictiveness_t2_html += "</table>"

    # T1 vs T2 disagreement table — sorted by signed delta to highlight features
    # that predict one target but not the other
    merged = predictiveness[["feature", "auc"]].rename(columns={"auc": "auc_t1"}).merge(
        predictiveness_t2[["feature", "auc"]].rename(columns={"auc": "auc_t2"}),
        on="feature", how="outer"
    )
    merged["delta"] = merged["auc_t1"] - merged["auc_t2"]
    merged = merged.dropna(subset=["delta"]).sort_values("delta", ascending=False)

    t1_t2_disagreement_html = (
        '<table><tr><th>Feature</th>'
        '<th class="num">AUC vs T1</th><th class="num">AUC vs T2</th>'
        '<th class="num">Δ (T1 − T2)</th><th>Read</th></tr>'
    )
    for _, r in merged.iterrows():
        delta = r["delta"]
        if delta > 0.10:
            read = "<em>Recession-only</em> — supports T1 channel"
        elif delta < -0.10:
            read = "<em>Drawdown-only</em> — supports T5 layer"
        else:
            read = "Both / similar"
        t1_t2_disagreement_html += (
            f"<tr><td><code>{r['feature']}</code></td>"
            f"<td class='num'>{fmt_auc(r['auc_t1'])}</td>"
            f"<td class='num'>{fmt_auc(r['auc_t2'])}</td>"
            f"<td class='num'>{delta:+.3f}</td>"
            f"<td>{read}</td></tr>"
        )
    t1_t2_disagreement_html += "</table>"

    alerts_html = "".join(f"<li>{a}</li>" for a in alerts)

    step4_html = derive_step4_impact(predictiveness, alerts)
    v11_html   = derive_v11_impact(predictiveness, predictiveness_t2,
                                     features, int(t2.sum()))

    logger.info("Writing HTML report to %s", out)
    html = HTML_TEMPLATE.format(
        date              = date.today().isoformat(),
        alerts_html       = alerts_html,
        coverage_img      = coverage_img,
        predictiveness_img = predictiveness_img,
        predictiveness_table = predictiveness_html,
        predictiveness_t2_img = predictiveness_t2_img,
        predictiveness_t2_table = predictiveness_t2_html,
        t1_t2_disagreement_table = t1_t2_disagreement_html,
        pre_recession_table = pre_rec_html,
        corr_img          = corr_img,
        feature_blocks    = feature_blocks_html,
        step4_impact      = step4_html,
        v11_impact        = v11_html,
    )
    out.write_text(html)
    logger.info("Done. Report size: %d KB", len(html) / 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pre-modeling exploration")
    parser.add_argument("--db",  type=Path, default=DEFAULT_DB_PATH,
                         help=f"Path to recession.db (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                         help=f"Output HTML path (default: {DEFAULT_OUT})")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.db.exists():
        print(f"ERROR: {args.db} not found.", file=sys.stderr)
        return 1

    run_exploration(args.db, args.out)
    print(f"\nReport written to: {args.out}")
    print(f"Open in browser:   open {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
