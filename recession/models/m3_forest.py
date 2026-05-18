"""
recession/models/m3_forest.py

M3 — the random forest. The third rung of the model ladder, and the first
model that CAN beat the yield curve, because it is the first that can
represent nonlinearity and feature interactions.

WHAT M1 AND M2 ESTABLISHED
--------------------------
M1 (unregularized probit): the single yield-curve feature beats the
4-feature set OOS (0.796 vs 0.771). M2 (L2-regularized logit): the
4-feature set still loses to the yield-curve baseline, robustly across
every C in {0.1..10} (0.777 vs 0.796). Conclusion: LINEAR models — penalized
or not — cannot use NFCI / INDPRO / REAL_FFR_GAP beyond what the yield
curve already contains at the T1/h=12 cell.

M3'S MANDATE
------------
If the extra features have any value, it must be NONLINEAR or
INTERACTION-based — e.g. "NFCI matters only once INDPRO is already
weakening" — structure a logit cannot represent but a tree ensemble can.
M3 tests exactly that. If M3 also fails to beat 0.796, the finding
hardens: those features carry no extra signal at this cell, by any model.

DESIGN — decision (1c+), confirmed May 2026
-------------------------------------------
TWO walk-forward runs, B+-style, on one common fold axis:
  - M3-core: the 4 features M1/M2 used. M3-core vs M2 isolates the
    question "does nonlinearity ALONE beat linear?" (same features).
  - M3-wide: the 4 plus the two §4.16 features deliberately held back for
    the tree models — PERMIT (building permits) and ISRATIO (inventory/
    sales ratio). M3-wide vs M3-core isolates "do the held-back features
    pay off once the model can exploit them?"
Plus a DISPLAY-ONLY diagnostic the two-run split cannot give on its own:
  - OOS permutation importance on M3-wide — which of the 6 features
    actually carried out-of-sample signal. Computed AFTER fitting, on the
    already-held-out test folds; nothing is selected on it (the feature
    set is pre-committed to §4.16's six). A readout, not a decision.

HYPERPARAMETERS — decision (2), fixed and principled (no search)
----------------------------------------------------------------
Random forest is far less tuning-sensitive than boosting, and the usable
recession sample is too small for a stable nested-CV inner loop (the M2
reasoning). So hyperparameters are fixed a priori:
  - n_estimators = 500. More trees only reduces variance, never overfits.
  - max_depth = None (unbounded). Depth is NOT the regularizer here.
  - min_samples_leaf = 20. THIS is the regularizer: with ~11 folds of rare
    recessions a leaf must hold enough months to be meaningful. Fixed at a
    firm value; a min_samples_leaf SENSITIVITY STRIP is reported (display
    only) to show the M3-vs-baseline conclusion is not leaf-size-dependent.

OVERFITTING GUARDRAILS — decision (3), upgraded under Rule 1
-----------------------------------------------------------
A tree ensemble can memorize rare events. Guards:
  - The walk-forward harness judges M3 OOS-only — a memorizing forest
    simply scores badly OOS; it cannot fake the headline.
  - min_samples_leaf is the in-model defense (see above).
  - Per-fold TRAIN-vs-OOS AUC gap is reported as the memorization
    signature — AND shown against M2's gap on the same folds as the
    "healthy low-variance model" reference, so the gap number is readable.
  - random_state is FIXED for the headline run (reproducibility), and a
    SEED-STABILITY STRIP is reported: M3's AUC across several seeds, so a
    marginal M3 win cannot be RNG luck. Display only.

No feature scaling — trees are scale-invariant (a simplification over
M1/M2's z-scoring). M3Forest implements the RecessionModel protocol, so
the Step-5 harness validates it with zero new harness code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from recession.validation.walk_forward import (
    walk_forward, WalkForwardResult, roc_auc,
)
from recession.features.builder import build_feature_dataframe
from recession.models.m1_probit import M1Probit, M1_FEATURES
from recession.models.m2_logit import M2Logit, M2_FEATURES


# M3 feature sets — decision (1c+).
M3_CORE_FEATURES = ["T10Y3M", "NFCI", "INDPRO", "REAL_FFR_GAP"]
# the §4.16 features held back for the tree models
M3_WIDE_FEATURES = ["T10Y3M", "NFCI", "INDPRO", "REAL_FFR_GAP",
                    "PERMIT", "ISRATIO"]

# Hyperparameters — decision (2), fixed.
M3_N_ESTIMATORS = 500
M3_MIN_SAMPLES_LEAF = 20
M3_HEADLINE_SEED = 0

# Diagnostic strips — display only.
M3_LEAF_GRID = [5, 10, 20, 40, 80]            # min_samples_leaf sensitivity
M3_SEED_GRID = [0, 1, 2, 3, 4]                # seed stability


# =============================================================================
# The model
# =============================================================================

class M3Forest:
    """Random-forest recession model implementing the RecessionModel
    protocol. No feature scaling — trees are scale-invariant.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    min_samples_leaf : int
        Minimum months per leaf — the regularizer.
    random_state : int
        Fixed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = M3_N_ESTIMATORS,
        min_samples_leaf: int = M3_MIN_SAMPLES_LEAF,
        random_state: int = M3_HEADLINE_SEED,
    ) -> None:
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._model: Optional[RandomForestClassifier] = None
        self._feature_names: list[str] = []
        self._fallback_rate: Optional[float] = None
        # train-set predictions kept for the train-vs-OOS gap diagnostic
        self._train_proba: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None

    # -- RecessionModel interface ------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "M3Forest":
        """Fit the random forest on training features X and target y."""
        self._feature_names = list(X.columns)
        y_arr = np.asarray(y, dtype=int)

        # Degenerate target — fall back to base rate, consistent with
        # M1Probit / M2Logit.
        if len(np.unique(y_arr)) < 2:
            self._fallback_rate = float(y_arr.mean())
            self._model = None
            self._train_proba = np.full(len(X), self._fallback_rate)
            self._train_y = y_arr
            return self

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=None,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X.to_numpy(), y_arr)
        self._fallback_rate = None
        # store train predictions for the memorization-gap diagnostic
        self._train_proba = self._model.predict_proba(X.to_numpy())[:, 1]
        self._train_y = y_arr
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(recession) for each row of X."""
        if self._model is None:
            rate = self._fallback_rate if self._fallback_rate is not None else 0.5
            return np.full(len(X), rate, dtype=float)
        return np.asarray(
            self._model.predict_proba(X[self._feature_names].to_numpy())[:, 1],
            dtype=float,
        )

    # -- diagnostics -------------------------------------------------------

    def train_auc(self) -> Optional[float]:
        """In-sample AUC on the training fold — for the train-vs-OOS gap."""
        if self._train_proba is None or self._train_y is None:
            return None
        return roc_auc(self._train_y, self._train_proba)

    def impurity_importances(self) -> Optional[dict]:
        """RF built-in (impurity) feature importances. A quick readout;
        the report uses OOS permutation importance as the headline
        attribution, which is more honest."""
        if self._model is None:
            return None
        return {n: float(i) for n, i
                in zip(self._feature_names, self._model.feature_importances_)}


def _m3_factory(min_samples_leaf: int = M3_MIN_SAMPLES_LEAF,
                seed: int = M3_HEADLINE_SEED):
    """Zero-arg model factory for the harness at given leaf-size / seed."""
    return lambda: M3Forest(min_samples_leaf=min_samples_leaf,
                            random_state=seed)


# =============================================================================
# OOS permutation importance
# =============================================================================

def oos_permutation_importance(
    result: WalkForwardResult,
    features: list[str],
    target: str,
    horizon: str,
    *,
    common_axis: pd.DatetimeIndex,
    min_samples_leaf: int = M3_MIN_SAMPLES_LEAF,
    seed: int = M3_HEADLINE_SEED,
    n_repeats: int = 10,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
) -> dict[str, float]:
    """Permutation importance computed on the HELD-OUT test folds.

    For each fold: refit M3 on the fold's training rows, then for each
    feature, shuffle that feature's column in the TEST rows and measure the
    drop in test AUC. A feature that carries real OOS signal shows a large
    AUC drop when shuffled. Averaged over folds and repeats.

    This is a DISPLAY-ONLY diagnostic — nothing is selected on it. The
    feature set is pre-committed (§4.16). It only reveals which features
    earned their place.

    Returns {feature_name: mean_auc_drop}.
    """
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year

    rng = np.random.default_rng(seed)
    drops: dict[str, list[float]] = {f: [] for f in features}

    for fold in result.folds:
        # rebuild features as-of this fold (PIT honest)
        fr = build_feature_dataframe(
            target=target, horizon=horizon,
            as_of=fold.test_end, train_cutoff=fold.train_end,
            feature_subset=features, **build_kwargs,
        )
        X, y = fr.X[features], fr.y
        train_mask = (X.index <= pd.Timestamp(fold.train_end)) & y.notna()
        test_mask = (
            (X.index >= pd.Timestamp(fold.test_start))
            & (X.index <= pd.Timestamp(fold.test_end))
            & y.notna()
        )
        Xtr = X.loc[train_mask].dropna()
        ytr = y.loc[Xtr.index].astype(int)
        Xte = X.loc[test_mask].dropna()
        yte = y.loc[Xte.index].astype(int)
        if len(Xtr) < 24 or len(Xte) == 0 or yte.nunique() < 2:
            continue

        model = M3Forest(min_samples_leaf=min_samples_leaf, random_state=seed)
        model.fit(Xtr, ytr)
        base_auc = roc_auc(yte.to_numpy(), model.predict_proba(Xte))
        if base_auc is None:
            continue

        for feat in features:
            for _ in range(n_repeats):
                Xte_perm = Xte.copy()
                Xte_perm[feat] = rng.permutation(Xte_perm[feat].to_numpy())
                perm_auc = roc_auc(yte.to_numpy(),
                                   model.predict_proba(Xte_perm))
                if perm_auc is not None:
                    drops[feat].append(base_auc - perm_auc)

    return {f: (float(np.mean(v)) if v else 0.0) for f, v in drops.items()}


# =============================================================================
# The driver
# =============================================================================

def _common_axis(target, horizon, features, db_path, min_history_year):
    """Months where every feature in `features` is present — the binding
    fold axis, so M3-core / M3-wide / M2 / M1 are all comparable."""
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
    present = probe.X[cols].notna().all(axis=1)
    return probe.X.index[present]


def run_m3(
    target: str = "T1",
    horizon: str = "h=12",
    *,
    min_history_year: Optional[int] = 1986,
    db_path: Optional[Path] = None,
    leaf_grid: Optional[list[int]] = None,
    seed_grid: Optional[list[int]] = None,
    **walk_forward_kwargs,
) -> dict:
    """Run M3 validation — decision (1c+).

    Returns a dict with:
        'm3_core'     : WalkForwardResult — M3 on the 4 core features
        'm3_wide'     : WalkForwardResult — M3 on the 6 §4.16 features
        'm2'          : WalkForwardResult — M2 baseline, same folds
        'm1'          : WalkForwardResult — M1 baseline, same folds
        'leaf_strip'  : dict[int, WalkForwardResult] — M3-wide by leaf size
        'seed_strip'  : dict[int, WalkForwardResult] — M3-wide by seed
        'perm_importance' : dict[str, float] — OOS permutation importance
                            on M3-wide (display-only attribution)
        'common_axis' : the shared fold axis
    """
    if leaf_grid is None:
        leaf_grid = M3_LEAF_GRID
    if seed_grid is None:
        seed_grid = M3_SEED_GRID

    common = dict(
        target=target, horizon=horizon,
        min_history_year=min_history_year,
        db_path=db_path,
        **walk_forward_kwargs,
    )

    # One common fold axis for everything — keyed off the WIDEST feature
    # set (M3-wide's 6), the binding constraint.
    axis = _common_axis(target, horizon, M3_WIDE_FEATURES,
                         db_path, min_history_year)

    # M3-core (4 features) and M3-wide (6 features).
    m3_core = walk_forward(
        model_factory=_m3_factory(),
        feature_subset=M3_CORE_FEATURES,
        model_columns=M3_CORE_FEATURES,
        model_name=f"M3-core RF (4 features, leaf={M3_MIN_SAMPLES_LEAF})",
        restrict_to_months=axis,
        **common,
    )
    m3_wide = walk_forward(
        model_factory=_m3_factory(),
        feature_subset=M3_WIDE_FEATURES,
        model_columns=M3_WIDE_FEATURES,
        model_name=f"M3-wide RF (6 features, leaf={M3_MIN_SAMPLES_LEAF})",
        restrict_to_months=axis,
        **common,
    )

    # Baselines on identical folds.
    m2 = walk_forward(
        model_factory=lambda: M2Logit(C=1.0),
        feature_subset=M2_FEATURES, model_columns=M2_FEATURES,
        model_name="M2 baseline (L2 logit, 4 features)",
        restrict_to_months=axis, **common,
    )
    m1 = walk_forward(
        model_factory=M1Probit,
        feature_subset=M1_FEATURES, model_columns=M1_FEATURES,
        model_name="M1 baseline (T10Y3M probit)",
        restrict_to_months=axis, **common,
    )

    # min_samples_leaf sensitivity strip — display only, on M3-wide.
    leaf_strip = {}
    for leaf in leaf_grid:
        leaf_strip[leaf] = walk_forward(
            model_factory=_m3_factory(min_samples_leaf=leaf),
            feature_subset=M3_WIDE_FEATURES, model_columns=M3_WIDE_FEATURES,
            model_name=f"M3-wide RF (leaf={leaf})",
            restrict_to_months=axis, **common,
        )

    # seed-stability strip — display only, on M3-wide.
    seed_strip = {}
    for sd in seed_grid:
        seed_strip[sd] = walk_forward(
            model_factory=_m3_factory(seed=sd),
            feature_subset=M3_WIDE_FEATURES, model_columns=M3_WIDE_FEATURES,
            model_name=f"M3-wide RF (seed={sd})",
            restrict_to_months=axis, **common,
        )

    # OOS permutation importance on M3-wide — display-only attribution.
    perm = oos_permutation_importance(
        m3_wide, M3_WIDE_FEATURES, target, horizon,
        common_axis=axis, db_path=db_path,
        min_history_year=min_history_year,
    )

    # train-vs-OOS gap — M3-wide (memorization signature) against M2 (the
    # low-variance reference, so the gap is readable).
    gap_m3 = train_oos_gap(
        m3_wide, M3_WIDE_FEATURES, target, horizon,
        model_kind="m3", common_axis=axis, db_path=db_path,
        min_history_year=min_history_year,
    )
    gap_m2 = train_oos_gap(
        m2, M2_FEATURES, target, horizon,
        model_kind="m2", common_axis=axis, db_path=db_path,
        min_history_year=min_history_year,
    )

    return {
        "m3_core": m3_core, "m3_wide": m3_wide,
        "m2": m2, "m1": m1,
        "leaf_strip": leaf_strip, "seed_strip": seed_strip,
        "perm_importance": perm,
        "gap_m3": gap_m3, "gap_m2": gap_m2,
        "common_axis": axis,
    }


def train_oos_gap(
    fold_specs: WalkForwardResult,
    features: list[str],
    target: str,
    horizon: str,
    *,
    model_kind: str,
    common_axis: pd.DatetimeIndex,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
) -> dict:
    """Per-fold (train AUC - OOS AUC) — the memorization signature.

    A large train-vs-OOS gap means the model fits the training fold far
    better than it generalizes — the overfitting fingerprint. The harness
    does not retain fitted models, so this does a lightweight refit per
    fold.

    model_kind: 'm3' (random forest) or 'm2' (L2 logit, the low-variance
    reference). Reported for both so M3's gap is readable against M2's.

    Returns {'gaps': [per-fold gaps], 'mean_gap': float,
             'mean_train_auc': float, 'mean_oos_auc': float}.
    """
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year

    gaps, train_aucs, oos_aucs = [], [], []

    for fold in fold_specs.folds:
        fr = build_feature_dataframe(
            target=target, horizon=horizon,
            as_of=fold.test_end, train_cutoff=fold.train_end,
            feature_subset=features, **build_kwargs,
        )
        X, y = fr.X[features], fr.y
        train_mask = (X.index <= pd.Timestamp(fold.train_end)) & y.notna()
        test_mask = (
            (X.index >= pd.Timestamp(fold.test_start))
            & (X.index <= pd.Timestamp(fold.test_end))
            & y.notna()
        )
        Xtr = X.loc[train_mask].dropna()
        ytr = y.loc[Xtr.index].astype(int)
        Xte = X.loc[test_mask].dropna()
        yte = y.loc[Xte.index].astype(int)
        if len(Xtr) < 24 or len(Xte) == 0 or yte.nunique() < 2:
            continue
        if ytr.nunique() < 2:
            continue

        if model_kind == "m3":
            model = M3Forest()
        elif model_kind == "m2":
            model = M2Logit(C=1.0)
        else:
            raise ValueError(f"unknown model_kind {model_kind!r}")
        model.fit(Xtr, ytr)

        train_auc = roc_auc(ytr.to_numpy(), model.predict_proba(Xtr))
        oos_auc = roc_auc(yte.to_numpy(), model.predict_proba(Xte))
        if train_auc is None or oos_auc is None:
            continue
        gaps.append(train_auc - oos_auc)
        train_aucs.append(train_auc)
        oos_aucs.append(oos_auc)

    return {
        "gaps": gaps,
        "mean_gap": float(np.mean(gaps)) if gaps else None,
        "mean_train_auc": float(np.mean(train_aucs)) if train_aucs else None,
        "mean_oos_auc": float(np.mean(oos_aucs)) if oos_aucs else None,
    }


def print_m3_report(results: dict) -> None:
    """Pretty-print the M3 (1c+) report."""
    m3c, m3w = results["m3_core"], results["m3_wide"]
    m2, m1 = results["m2"], results["m1"]
    leaf_strip = results["leaf_strip"]
    seed_strip = results["seed_strip"]
    perm = results["perm_importance"]

    def auc(r):
        return r.mean_fold_auc

    print("=" * 70)
    print(f"M3 VALIDATION (random forest, decision 1c+) — "
          f"{m3c.target} {m3c.horizon}")
    print("=" * 70)
    for r in (m3c, m3w, m2, m1):
        print()
        print(r.summary())

    print()
    print("-" * 70)
    b1 = auc(m1)     # project baseline
    print("  HEAD-TO-HEAD (mean fold AUC):")
    print(f"    M1 baseline (yield curve)  : {auc(m1):.4f}   <- PROJECT BASELINE")
    print(f"    M2 (L2 logit, 4 feat)      : {auc(m2):.4f}")
    print(f"    M3-core (RF, 4 feat)       : {auc(m3c):.4f}")
    print(f"    M3-wide (RF, 6 feat)       : {auc(m3w):.4f}")
    print()

    # (a) does nonlinearity alone beat linear?  M3-core vs M2
    if auc(m3c) is not None and auc(m2) is not None:
        d = auc(m3c) - auc(m2)
        print(f"  (a) nonlinearity alone:  M3-core - M2 = {d:+.4f}  "
              + ("nonlinearity helps" if d > 0
                 else "nonlinearity does not help"))
    # (b) do the held-back features pay off under nonlinearity?
    if auc(m3w) is not None and auc(m3c) is not None:
        d = auc(m3w) - auc(m3c)
        print(f"  (b) held-back features:  M3-wide - M3-core = {d:+.4f}  "
              + ("PERMIT/ISRATIO add signal" if d > 0
                 else "PERMIT/ISRATIO add nothing"))
    # overall vs baseline
    if auc(m3w) is not None and b1 is not None:
        d = auc(m3w) - b1
        print(f"  overall:  best M3 - baseline = {d:+.4f}  "
              + ("M3 BEATS the baseline" if d > 0
                 else "M3 does NOT beat the baseline"))

    # min_samples_leaf sensitivity strip
    print()
    print("-" * 70)
    print("  min_samples_leaf sensitivity (M3-wide; DISPLAY ONLY, "
          f"headline leaf={M3_MIN_SAMPLES_LEAF}):")
    print(f"  {'leaf':>6} {'mean fold AUC':>15} {'vs baseline':>13}")
    for leaf, r in leaf_strip.items():
        a = auc(r)
        if a is None:
            print(f"  {leaf:>6} {'n/a':>15}")
            continue
        vs = f"{a - b1:+.4f}" if b1 is not None else "n/a"
        flag = " *headline" if leaf == M3_MIN_SAMPLES_LEAF else ""
        print(f"  {leaf:>6} {a:>15.4f} {vs:>13}{flag}")

    # seed-stability strip
    print()
    print("  seed stability (M3-wide; DISPLAY ONLY, "
          f"headline seed={M3_HEADLINE_SEED}):")
    print(f"  {'seed':>6} {'mean fold AUC':>15}")
    seed_aucs = []
    for sd, r in seed_strip.items():
        a = auc(r)
        seed_aucs.append(a)
        flag = " *headline" if sd == M3_HEADLINE_SEED else ""
        print(f"  {sd:>6} {a:>15.4f}{flag}" if a is not None
              else f"  {sd:>6} {'n/a':>15}")
    valid = [a for a in seed_aucs if a is not None]
    if len(valid) > 1:
        spread = max(valid) - min(valid)
        print(f"  seed spread: {spread:.4f}  "
              + ("(stable)" if spread < 0.02
                 else "(UNSTABLE — a marginal win may be RNG luck)"))

    # OOS permutation importance
    print()
    print("-" * 70)
    print("  OOS permutation importance (M3-wide; DISPLAY ONLY — which "
          "features carried")
    print("  out-of-sample signal. Mean AUC drop when the feature is "
          "shuffled in test rows):")
    for feat, drop in sorted(perm.items(), key=lambda kv: -kv[1]):
        bar = "#" * max(0, int(drop * 200))
        print(f"  {feat:>14} {drop:+.4f}  {bar}")

    # train-vs-OOS gap — memorization signature
    gap_m3 = results.get("gap_m3")
    gap_m2 = results.get("gap_m2")
    if gap_m3 is not None and gap_m2 is not None:
        print()
        print("-" * 70)
        print("  TRAIN-vs-OOS AUC gap (memorization signature; M2 is the "
              "low-variance reference):")
        if gap_m3["mean_gap"] is not None:
            print(f"    M3-wide:  train {gap_m3['mean_train_auc']:.4f}  "
                  f"OOS {gap_m3['mean_oos_auc']:.4f}  "
                  f"gap {gap_m3['mean_gap']:+.4f}")
        if gap_m2["mean_gap"] is not None:
            print(f"    M2 (ref): train {gap_m2['mean_train_auc']:.4f}  "
                  f"OOS {gap_m2['mean_oos_auc']:.4f}  "
                  f"gap {gap_m2['mean_gap']:+.4f}")
        if (gap_m3["mean_gap"] is not None
                and gap_m2["mean_gap"] is not None):
            excess = gap_m3["mean_gap"] - gap_m2["mean_gap"]
            print(f"    M3 excess gap over M2 reference: {excess:+.4f}  "
                  + ("(M3 memorizes more — expected for a forest; "
                     "OOS number is what counts)" if excess > 0.05
                     else "(comparable to the linear reference)"))
    print("=" * 70)
