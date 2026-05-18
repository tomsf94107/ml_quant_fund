"""
recession/validation/axis_check.py

Finding 1 (system audit) — the fold-axis consistency guard.

THE ISSUE
---------
Each model driver (run_m2, run_m3, run_m4, run_m5) computes its own
common fold axis from its own feature set: run_m2 keys off the 4-feature
set, run_m3/run_m4 off the 6-feature set, run_m5 off INDPRO+T10Y3M.
Different feature sets have different start dates, so in principle each
report's embedded "M1 baseline" could be a DIFFERENT number on a
DIFFERENT fold set. On the current data they happen to coincide (every
report reads M1 = 0.7959 on 11 folds) — but that is a data coincidence,
not a guarantee.

WHY A GUARD, NOT A REFACTOR
---------------------------
Forcing every driver onto one global axis is NOT a small change: M3-wide
genuinely NEEDS PERMIT/ISRATIO present, so its axis must be where those
exist; collapsing everyone onto the most-restrictive window would discard
data for M1/M2. A proper unification is a design change across five
drivers — risky to bolt on to a 236/236-green system.

So Finding 1 is addressed the safe way: a GUARD that makes the
inconsistency impossible to MISS rather than impossible to occur. This
module re-derives each driver's fold axis and checks them against the
canonical baseline axis (M1's). Any divergence is reported loudly. It is
purely additive — it changes no driver, breaks nothing, and turns a
silent latent bug into a visible, asserted check.

USAGE
-----
    from recession.validation.axis_check import check_axis_consistency
    report = check_axis_consistency(db_path="recession.db")
    print_axis_check(report)

Run it after any change to the feature registry or feature start dates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from recession.features.builder import build_feature_dataframe
from recession.models.m1_probit import M1_FEATURES, M1_EXTENDED_FEATURES
from recession.models.m2_logit import M2_FEATURES
from recession.models.m3_forest import M3_WIDE_FEATURES
from recession.models.m4_xgboost import M4_WIDE_FEATURES
from recession.models.m5_markov import M5_FEATURES


def _axis_for(features: list[str], target: str, horizon: str,
              db_path: Optional[Path], min_history_year: Optional[int]
              ) -> pd.DatetimeIndex:
    """Re-derive the common fold axis a driver would use for a feature set
    — months where every feature in the set is present."""
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year
    probe = build_feature_dataframe(
        target=target, horizon=horizon,
        as_of="today", train_cutoff="today",
        feature_subset=features, **build_kwargs,
    )
    cols = [c for c in features if c in probe.X.columns]
    return probe.X.index[probe.X[cols].notna().all(axis=1)].sort_values()


def check_axis_consistency(
    target: str = "T1",
    horizon: str = "h=12",
    *,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
) -> dict:
    """Re-derive every driver's fold axis and compare.

    The CANONICAL axis is M1's (the project baseline). Each other
    driver's axis is checked against it: same start, same end, same
    length => consistent. Any difference is flagged.

    Returns {'canonical': {...}, 'drivers': {name: {...}},
             'all_consistent': bool}.
    """
    # each driver's axis is keyed off the feature set named below —
    # mirroring exactly what run_mX does internally
    driver_features = {
        "M1 (baseline)": M1_EXTENDED_FEATURES,   # run_m1 keys off extended
        "M2": M2_FEATURES,
        "M3": M3_WIDE_FEATURES,
        "M4": M4_WIDE_FEATURES,
        "M5": sorted(set(M5_FEATURES) | set(M1_FEATURES)),
        "combination": M2_FEATURES,              # combination uses 4-feat
    }

    axes = {}
    for name, feats in driver_features.items():
        ax = _axis_for(feats, target, horizon, db_path, min_history_year)
        axes[name] = ax

    canonical = axes["M1 (baseline)"]
    can_info = {
        "n_months": len(canonical),
        "start": (canonical.min().strftime("%Y-%m")
                  if len(canonical) else None),
        "end": (canonical.max().strftime("%Y-%m")
                if len(canonical) else None),
    }

    drivers = {}
    all_consistent = True
    for name, ax in axes.items():
        same = (
            len(ax) == len(canonical)
            and len(ax) > 0
            and ax.min() == canonical.min()
            and ax.max() == canonical.max()
        )
        if not same:
            all_consistent = False
        drivers[name] = {
            "n_months": len(ax),
            "start": ax.min().strftime("%Y-%m") if len(ax) else None,
            "end": ax.max().strftime("%Y-%m") if len(ax) else None,
            "consistent_with_canonical": same,
        }

    return {"canonical": can_info, "drivers": drivers,
            "all_consistent": all_consistent,
            "target": target, "horizon": horizon}


def print_axis_check(report: dict) -> None:
    """Print the axis-consistency report."""
    print("=" * 70)
    print(f"FOLD-AXIS CONSISTENCY CHECK — {report['target']} "
          f"{report['horizon']}")
    print("=" * 70)
    can = report["canonical"]
    print(f"  Canonical axis (M1 baseline): {can['n_months']} months, "
          f"{can['start']} .. {can['end']}")
    print()
    print(f"  {'driver':>16} {'months':>8} {'start':>9} {'end':>9} "
          f"{'consistent':>12}")
    print("  " + "-" * 60)
    for name, d in report["drivers"].items():
        flag = "OK" if d["consistent_with_canonical"] else "*** DIVERGES"
        print(f"  {name:>16} {d['n_months']:>8} {str(d['start']):>9} "
              f"{str(d['end']):>9} {flag:>12}")
    print()
    print("-" * 70)
    if report["all_consistent"]:
        print("  ALL DRIVERS CONSISTENT — every model validated on the same")
        print("  fold axis; the 'M1 baseline' is one number across all")
        print("  reports. The cross-model comparisons are sound.")
    else:
        print("  *** AXIS DIVERGENCE DETECTED ***")
        print("  At least one driver validated on a different fold axis.")
        print("  The 'M1 baseline' is NOT the same number across reports —")
        print("  cross-model AUC comparisons between a diverging driver and")
        print("  the others are NOT valid. This happens when a feature's")
        print("  history changes. Re-derive the affected cross-comparisons,")
        print("  or unify the drivers onto one explicit axis.")
    print("=" * 70)
