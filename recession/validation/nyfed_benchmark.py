"""
recession/validation/nyfed_benchmark.py

Research point A (system audit) — external validation against the
New York Fed's published recession-probability series.

WHY
---
M1 — the single-feature yield-curve probit — is essentially the same
model the New York Fed publishes as its "Probability of US Recession
Predicted by Treasury Spread" series. The strongest external sanity
check available is therefore: does M1's recession probability TRACK the
NY Fed's published series? If it does, that is independent third-party
validation that M1 is implemented correctly. If it diverges materially,
something in the data pipeline or the model is off and should be found.

This module is the comparison harness. It does NOT ship the NY Fed data
— that series is published by the NY Fed and must be supplied by the
user as a CSV. The harness:
  - loads M1's in-sample recession probability over history,
  - loads the NY Fed series from a user-supplied CSV,
  - aligns them on common months,
  - reports correlation, mean absolute difference, and the largest
    divergences.

HOW TO GET THE NY FED SERIES
----------------------------
The NY Fed publishes the series on its website ("The Yield Curve as a
Leading Indicator"). Download it as a CSV with two columns: a date column
and the recession-probability column (values in 0..1 or 0..100). Pass the
path and the column names to compare_to_nyfed().

NOTE ON EXPECTED DIFFERENCES
----------------------------
M1 and the NY Fed model are the same FAMILY but not identical: the NY Fed
uses a specific spread definition, estimation window, and horizon
convention. Some level difference is expected and not alarming. What
matters is that the two series are HIGHLY CORRELATED and move together
around recessions. High correlation + small typical gap = validated.
Low correlation = investigate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_m1_probability(db_path: Path,
                         min_history_year: Optional[int] = 1986
                         ) -> pd.Series:
    """M1's in-sample recession probability over all available history,
    month-indexed. (In-sample is correct here: we are checking the model
    is IMPLEMENTED right by comparison to the NY Fed, not measuring OOS
    skill — that is the walk-forward work.)"""
    from recession.models.m1_probit import M1Probit, M1_FEATURES
    from recession.features.builder import build_feature_dataframe

    fr = build_feature_dataframe(
        target="T1", horizon="h=12",
        as_of="today", train_cutoff="today",
        feature_subset=M1_FEATURES, db_path=db_path,
        min_history_year=min_history_year,
    )
    X = fr.X[M1_FEATURES]
    y = fr.y
    mask = X.notna().all(axis=1)
    train_mask = mask & y.notna()
    model = M1Probit().fit(X.loc[train_mask],
                           y.loc[train_mask].astype(int))
    return pd.Series(model.predict_proba(X.loc[mask]),
                     index=X.loc[mask].index, name="M1")


def load_nyfed_series(csv_path: Path, date_col: str,
                      prob_col: str) -> pd.Series:
    """Load the NY Fed recession-probability series from a user CSV.

    Values may be 0..1 or 0..100; this normalises to 0..1.
    """
    df = pd.read_csv(csv_path)
    if date_col not in df.columns or prob_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns {date_col!r} and {prob_col!r}; "
            f"found {list(df.columns)}")
    s = pd.Series(
        df[prob_col].to_numpy(dtype=float),
        index=pd.to_datetime(df[date_col]),
        name="NYFed",
    ).sort_index()
    # normalise 0..100 -> 0..1 if needed
    if s.max() > 1.5:
        s = s / 100.0
    # snap to month start so it aligns with M1's monthly index
    s.index = s.index.to_period("M").to_timestamp()
    return s


# The NY Fed publishes Rec_prob dated to the PREDICTION month: the value
# at month M is "P(recession in the 12 months after M)". This project's
# M1 probability is dated to the OBSERVATION month. The two series are
# therefore 12 months out of phase, and must be aligned before comparison.
# Confirmed empirically: a -12-month shift of the NY Fed series lifts the
# M1-vs-NYFed correlation from 0.42 (raw) to 0.99. Shifting the NY Fed
# series back 12 months puts its prediction-dated value onto the same
# observation month M1 uses.
NYFED_DATE_CONVENTION_SHIFT_MONTHS = -12


def compare_to_nyfed(
    db_path: Path,
    nyfed_csv: Path,
    *,
    date_col: str = "date",
    prob_col: str = "probability",
    min_history_year: Optional[int] = 1986,
    align_shift_months: int = NYFED_DATE_CONVENTION_SHIFT_MONTHS,
) -> dict:
    """Compare M1's recession probability to the NY Fed published series.

    The NY Fed series is dated to the PREDICTION month while M1 is dated
    to the OBSERVATION month — a 12-month phase difference. The NY Fed
    series is shifted by `align_shift_months` (default -12) to put both
    on M1's date convention before comparison. Pass align_shift_months=0
    to compare raw (diagnostic only — this will show the ~0.42 misaligned
    correlation).

    Returns {'n_common_months', 'correlation', 'mean_abs_diff',
             'max_divergences': DataFrame, 'm1', 'nyfed', 'align_shift'}.
    """
    m1 = load_m1_probability(db_path, min_history_year)
    nyfed = load_nyfed_series(nyfed_csv, date_col, prob_col)

    # align the NY Fed series onto M1's observation-month date convention
    if align_shift_months != 0:
        nyfed = nyfed.copy()
        nyfed.index = nyfed.index + pd.DateOffset(months=align_shift_months)

    common = m1.index.intersection(nyfed.index)
    if len(common) == 0:
        return {"error": "no overlapping months between M1 and the "
                          "NY Fed series — check the CSV date range/format"}

    m1c = m1.reindex(common)
    nfc = nyfed.reindex(common)
    diff = (m1c - nfc).abs()


    corr = float(np.corrcoef(m1c.to_numpy(), nfc.to_numpy())[0, 1])
    mad = float(diff.mean())

    # the months where the two disagree most
    worst = diff.sort_values(ascending=False).head(6)
    max_div = pd.DataFrame({
        "month": [d.strftime("%Y-%m") for d in worst.index],
        "M1": [round(float(m1c.loc[d]), 3) for d in worst.index],
        "NYFed": [round(float(nfc.loc[d]), 3) for d in worst.index],
        "abs_diff": [round(float(v), 3) for v in worst.values],
    })

    return {
        "n_common_months": len(common),
        "correlation": corr,
        "mean_abs_diff": mad,
        "max_divergences": max_div,
        "align_shift": align_shift_months,
        "m1": m1c, "nyfed": nfc,
    }


def print_nyfed_report(result: dict) -> None:
    """Print the NY Fed benchmark report."""
    print("=" * 70)
    print("RESEARCH A — M1 vs NY FED PUBLISHED RECESSION-PROBABILITY SERIES")
    print("=" * 70)
    if result.get("error"):
        print(f"  {result['error']}")
        print("=" * 70)
        return

    print(f"  common months compared: {result['n_common_months']}")
    if result.get("align_shift") is not None:
        print(f"  NY Fed series shifted {result['align_shift']:+d} months "
              f"onto M1's observation-month date convention")
    print(f"  correlation:            {result['correlation']:.4f}")
    print(f"  mean absolute diff:     {result['mean_abs_diff']:.4f}")
    print()
    print("  largest divergences:")
    print("  " + result["max_divergences"].to_string(index=False)
          .replace("\n", "\n  "))
    print()
    print("-" * 70)
    corr = result["correlation"]
    if corr >= 0.90:
        print("  VALIDATED: M1 tracks the NY Fed series very closely "
              f"(corr {corr:.2f}).")
        print("  M1 is implemented consistently with the published model.")
    elif corr >= 0.75:
        print(f"  BROADLY CONSISTENT: correlation {corr:.2f}. The two move")
        print("  together; some level difference is expected (different")
        print("  spread definition / estimation window). No red flag.")
    else:
        print(f"  INVESTIGATE: correlation {corr:.2f} is low. M1 and the")
        print("  NY Fed series should move closely together — a weak")
        print("  correlation suggests a data or model discrepancy worth")
        print("  tracing before relying on M1.")
    print("=" * 70)
