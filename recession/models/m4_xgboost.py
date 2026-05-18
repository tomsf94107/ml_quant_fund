"""
recession/models/m4_xgboost.py

M4 — gradient-boosted trees (XGBoost). The fourth and final model class on
the ladder, and a CONFIRMATORY CAPSTONE.

WHERE THE LADDER STANDS BEFORE M4 (all at T1/h=12)
--------------------------------------------------
  M1 single-feature yield-curve probit : mean fold AUC 0.796  <- BASELINE
  M2 L2-regularized logit (4 feat)     : 0.777  (loses)
  M3 random forest (4 / 6 feat)        : ~0.80  (ties — within seed noise)

M2 (regularized linear) and M3 (bagged nonlinear trees) both failed to
beat the yield curve at h=12. M4 adds the last untested model class —
gradient boosting. Its role is confirmatory: if boosting ALSO ties the
baseline, the conclusion "no model class beats T10Y3M at h=12" is
airtight. (The separate horizon scan already showed the macro features
come alive at SHORTER horizons — but per the D+ no-target-selection rule
that is a pre-registered hypothesis for a future model series; M4 stays
at the pre-committed T1/h=12 cell.)

DESIGN — decision (1): 1c+, same as M3
--------------------------------------
TWO walk-forward runs on one common fold axis:
  - M4-core: the 4 features M1/M2/M3 used.
  - M4-wide: the 4 plus PERMIT + ISRATIO (the §4.16 tree-routed features).
Plus OOS permutation importance on M4-wide. Same feature sets and fold
axis as M3, so M3-vs-M4 is a clean read.

HYPERPARAMETERS — decision (2): sharpened 2c+, capacity-constrained
------------------------------------------------------------------
XGBoost is a HIGH-CAPACITY learner and genuinely tuning-sensitive (unlike
the random forest). On ~11 folds of rare recessions a generic config
(depth 6, hundreds of trees) memorises the training data almost perfectly
and the OOS number becomes pure variance. Boosting makes this WORSE than
bagging: each tree corrects the last, so the ensemble chases the training
signal harder rather than averaging it away.

So M4's config is deliberately CAPACITY-CONSTRAINED, and the choice is
PRE-REGISTERED with an explicit written justification keyed to the
recession-event count (NOT a bare guess, and NOT a formula with a hidden
tunable constant — that would just move the arbitrariness somewhere less
visible):

  JUSTIFICATION. The smallest training fold in this walk-forward holds on
  the order of a few dozen recession months (recessions are ~15% of
  months; an early fold's training window is ~240 months => ~30-40
  recession months). A model that can memorise that many events will. An
  ensemble of ~80 depth-3 trees has limited capacity: depth-3 trees are
  shallow (a true pairwise interaction needs a split conditioned on a
  prior split — depth-3 barely affords one), 80 is few, and a 0.05
  learning rate plus subsampling further damps it. This config can
  represent simple interactions but cannot fit ~35 events to noise.

  Headline config (pre-committed):
    max_depth        = 3      shallow — enough for one interaction, no more
    n_estimators     = 80     few rounds
    learning_rate    = 0.05   slow — each tree contributes little
    min_child_weight = 5      a leaf must cover several months
    subsample        = 0.8    stochastic regularisation
    colsample_bytree = 0.8    stochastic regularisation
    reg_lambda       = 5.0    L2 on leaf weights

  This mirrors the lesson from the equity ML Quant Fund, where the
  production XGBoost is deliberately tiny (depth 2, 30 trees, reg_lambda
  10) BECAUSE the loose default config overfit and gave a misleading AUC.

SENSITIVITY STRIPS — decisions (2) and (3)
------------------------------------------
  - max_depth strip {1,2,3,4,6}: shows whether the M4-vs-baseline verdict
    is stable across capacity. DISPLAY ONLY — headline is depth 3.
  - n_estimators strip {20,50,80,150,300}: serves DOUBLE DUTY (the
    "3-hybrid"). (i) capacity sensitivity. (ii) a study-level stand-in for
    EARLY STOPPING — per-fold early stopping is impossible here (it needs
    a validation carve-out, and an 11-fold rare-event window cannot spare
    one). If mean-fold AUC rises to a peak then DECLINES at high tree
    count, that decline IS "boosting into noise" — exactly what early
    stopping prevents — observed at the study level instead of per fold.

  PRE-REGISTERED reading rule for the early-stopping half of the strip
  (committed before the run): the model is "boosting into noise" if the
  mean-fold AUC at the highest tree count (300) is more than
  EARLY_STOP_DECLINE (=0.02) below the strip's peak. Reading the strip is
  interpretation; SELECTING the peak tree count would be leakage and is
  NOT done — the headline stays the pre-committed n_estimators=80.

OVERFITTING GUARDRAILS — decision (3)
-------------------------------------
  - The capacity constraint above is the primary defence (it replaces
    early stopping, which cannot be done cleanly here).
  - Per-fold TRAIN-vs-OOS AUC gap, reported against M2 as the low-variance
    reference — the memorisation signature. M4's gap is expected to be the
    widest of the four models.
  - SEED strip — XGBoost is stochastic (subsample/colsample); on tiny data
    the seed swing can be large (M3's real h=12 seed spread was 0.050,
    enough to void its apparent win). The seed strip is the DECISIVE
    robustness gate: a marginal M4 win that does not survive reseeding is
    not a win.

M4Xgb implements the RecessionModel protocol (fit / predict_proba), so the
Step-5 harness validates it with zero new harness code. No feature scaling
— trees are scale-invariant.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from recession.validation.walk_forward import (
    walk_forward, WalkForwardResult, roc_auc,
)
from recession.features.builder import build_feature_dataframe
from recession.models.m1_probit import M1Probit, M1_FEATURES
from recession.models.m2_logit import M2Logit, M2_FEATURES


# M4 feature sets — decision (1c+), identical to M3's.
M4_CORE_FEATURES = ["T10Y3M", "NFCI", "INDPRO", "REAL_FFR_GAP"]
M4_WIDE_FEATURES = ["T10Y3M", "NFCI", "INDPRO", "REAL_FFR_GAP",
                    "PERMIT", "ISRATIO"]

# Headline hyperparameters — decision (2), sharpened 2c+, capacity-constrained.
M4_MAX_DEPTH = 3
M4_N_ESTIMATORS = 80
M4_LEARNING_RATE = 0.05
M4_MIN_CHILD_WEIGHT = 5
M4_SUBSAMPLE = 0.8
M4_COLSAMPLE = 0.8
M4_REG_LAMBDA = 5.0
M4_HEADLINE_SEED = 0

# Diagnostic strips — display only.
M4_DEPTH_GRID = [1, 2, 3, 4, 6]
M4_NEST_GRID = [20, 50, 80, 150, 300]
M4_SEED_GRID = [0, 1, 2, 3, 4]

# Pre-registered reading rule for the early-stopping half of the n_est strip.
EARLY_STOP_DECLINE = 0.02   # AUC drop from peak to 300 trees => boosting into noise


# =============================================================================
# The model
# =============================================================================

class M4Xgb:
    """Gradient-boosted-tree recession model implementing the RecessionModel
    protocol. No feature scaling — trees are scale-invariant.

    The constructor exposes the capacity knobs so the sensitivity strips
    can vary them; defaults are the pre-committed headline config.
    """

    def __init__(
        self,
        max_depth: int = M4_MAX_DEPTH,
        n_estimators: int = M4_N_ESTIMATORS,
        learning_rate: float = M4_LEARNING_RATE,
        min_child_weight: int = M4_MIN_CHILD_WEIGHT,
        subsample: float = M4_SUBSAMPLE,
        colsample_bytree: float = M4_COLSAMPLE,
        reg_lambda: float = M4_REG_LAMBDA,
        random_state: int = M4_HEADLINE_SEED,
    ) -> None:
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self._model: Optional[XGBClassifier] = None
        self._feature_names: list[str] = []
        self._fallback_rate: Optional[float] = None
        self._train_proba: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None

    # -- RecessionModel interface ------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "M4Xgb":
        """Fit the boosted ensemble on training features X and target y."""
        self._feature_names = list(X.columns)
        y_arr = np.asarray(y, dtype=int)

        # Degenerate target — fall back to base rate, consistent with the
        # other models.
        if len(np.unique(y_arr)) < 2:
            self._fallback_rate = float(y_arr.mean())
            self._model = None
            self._train_proba = np.full(len(X), self._fallback_rate)
            self._train_y = y_arr
            return self

        self._model = XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        )
        self._model.fit(X.to_numpy(), y_arr)
        self._fallback_rate = None
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


def _m4_factory(max_depth: int = M4_MAX_DEPTH,
                n_estimators: int = M4_N_ESTIMATORS,
                seed: int = M4_HEADLINE_SEED):
    """Zero-arg model factory for the harness at given capacity / seed."""
    return lambda: M4Xgb(max_depth=max_depth, n_estimators=n_estimators,
                         random_state=seed)


# =============================================================================
# OOS permutation importance  (held-out folds; display-only attribution)
# =============================================================================

def oos_permutation_importance(
    result: WalkForwardResult,
    features: list[str],
    target: str,
    horizon: str,
    *,
    seed: int = M4_HEADLINE_SEED,
    n_repeats: int = 10,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
) -> dict[str, float]:
    """Permutation importance computed on the HELD-OUT test folds.

    For each fold: refit M4 on the fold's training rows; for each feature,
    shuffle that feature in the TEST rows and measure the drop in test AUC.
    A feature carrying real OOS signal shows a large drop. Averaged over
    folds and repeats. DISPLAY ONLY — nothing is selected on it.
    """
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year

    rng = np.random.default_rng(seed)
    drops: dict[str, list[float]] = {f: [] for f in features}

    for fold in result.folds:
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
        if len(Xtr) < 24 or len(Xte) == 0 or yte.nunique() < 2 \
                or ytr.nunique() < 2:
            continue

        model = M4Xgb(random_state=seed).fit(Xtr, ytr)
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
# train-vs-OOS gap  (memorisation signature)
# =============================================================================

def train_oos_gap(
    fold_specs: WalkForwardResult,
    features: list[str],
    target: str,
    horizon: str,
    *,
    model_kind: str,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
) -> dict:
    """Per-fold (train AUC - OOS AUC). model_kind: 'm4' or 'm2' (the
    low-variance linear reference). Lightweight refit per fold."""
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
        if len(Xtr) < 24 or len(Xte) == 0 or yte.nunique() < 2 \
                or ytr.nunique() < 2:
            continue

        if model_kind == "m4":
            model = M4Xgb()
        elif model_kind == "m2":
            model = M2Logit(C=1.0)
        else:
            raise ValueError(f"unknown model_kind {model_kind!r}")
        model.fit(Xtr, ytr)
        ta = roc_auc(ytr.to_numpy(), model.predict_proba(Xtr))
        oa = roc_auc(yte.to_numpy(), model.predict_proba(Xte))
        if ta is None or oa is None:
            continue
        gaps.append(ta - oa)
        train_aucs.append(ta)
        oos_aucs.append(oa)

    return {
        "gaps": gaps,
        "mean_gap": float(np.mean(gaps)) if gaps else None,
        "mean_train_auc": float(np.mean(train_aucs)) if train_aucs else None,
        "mean_oos_auc": float(np.mean(oos_aucs)) if oos_aucs else None,
    }


# =============================================================================
# The driver
# =============================================================================

def _common_axis(target, horizon, features, db_path, min_history_year):
    """Months where every feature in `features` is present."""
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


def run_m4(
    target: str = "T1",
    horizon: str = "h=12",
    *,
    min_history_year: Optional[int] = 1986,
    db_path: Optional[Path] = None,
    depth_grid: Optional[list[int]] = None,
    nest_grid: Optional[list[int]] = None,
    seed_grid: Optional[list[int]] = None,
    **walk_forward_kwargs,
) -> dict:
    """Run M4 validation — decision (1c+) with the 2c+ / 3-hybrid diagnostics.

    Returns a dict with:
        'm4_core'      : WalkForwardResult — M4 on the 4 core features
        'm4_wide'      : WalkForwardResult — M4 on the 6 §4.16 features
        'm2', 'm1'     : WalkForwardResult — baselines on the same folds
        'depth_strip'  : dict[int, WalkForwardResult] — M4-wide by max_depth
        'nest_strip'   : dict[int, WalkForwardResult] — M4-wide by n_estimators
                         (DOUBLE DUTY: capacity + early-stopping diagnostic)
        'seed_strip'   : dict[int, WalkForwardResult] — M4-wide by seed
        'perm_importance' : dict[str, float] — OOS permutation importance
        'gap_m4', 'gap_m2' : train-vs-OOS gap dicts
        'common_axis'  : the shared fold axis
    """
    if depth_grid is None:
        depth_grid = M4_DEPTH_GRID
    if nest_grid is None:
        nest_grid = M4_NEST_GRID
    if seed_grid is None:
        seed_grid = M4_SEED_GRID

    common = dict(
        target=target, horizon=horizon,
        min_history_year=min_history_year,
        db_path=db_path,
        **walk_forward_kwargs,
    )
    axis = _common_axis(target, horizon, M4_WIDE_FEATURES,
                        db_path, min_history_year)

    # M4-core / M4-wide.
    m4_core = walk_forward(
        model_factory=_m4_factory(),
        feature_subset=M4_CORE_FEATURES, model_columns=M4_CORE_FEATURES,
        model_name=f"M4-core XGB (4 feat, depth={M4_MAX_DEPTH}, "
                   f"n_est={M4_N_ESTIMATORS})",
        restrict_to_months=axis, **common,
    )
    m4_wide = walk_forward(
        model_factory=_m4_factory(),
        feature_subset=M4_WIDE_FEATURES, model_columns=M4_WIDE_FEATURES,
        model_name=f"M4-wide XGB (6 feat, depth={M4_MAX_DEPTH}, "
                   f"n_est={M4_N_ESTIMATORS})",
        restrict_to_months=axis, **common,
    )
    # Baselines on identical folds.
    m2 = walk_forward(
        model_factory=lambda: M2Logit(C=1.0),
        feature_subset=M2_FEATURES, model_columns=M2_FEATURES,
        model_name="M2 baseline (L2 logit, 4 feat)",
        restrict_to_months=axis, **common,
    )
    m1 = walk_forward(
        model_factory=M1Probit,
        feature_subset=M1_FEATURES, model_columns=M1_FEATURES,
        model_name="M1 baseline (T10Y3M probit)",
        restrict_to_months=axis, **common,
    )

    # max_depth sensitivity strip — display only, M4-wide.
    depth_strip = {}
    for d in depth_grid:
        depth_strip[d] = walk_forward(
            model_factory=_m4_factory(max_depth=d),
            feature_subset=M4_WIDE_FEATURES, model_columns=M4_WIDE_FEATURES,
            model_name=f"M4-wide XGB (depth={d})",
            restrict_to_months=axis, **common,
        )

    # n_estimators strip — DOUBLE DUTY (capacity + early-stopping), M4-wide.
    nest_strip = {}
    for ne in nest_grid:
        nest_strip[ne] = walk_forward(
            model_factory=_m4_factory(n_estimators=ne),
            feature_subset=M4_WIDE_FEATURES, model_columns=M4_WIDE_FEATURES,
            model_name=f"M4-wide XGB (n_est={ne})",
            restrict_to_months=axis, **common,
        )

    # seed-stability strip — display only, M4-wide.
    seed_strip = {}
    for sd in seed_grid:
        seed_strip[sd] = walk_forward(
            model_factory=_m4_factory(seed=sd),
            feature_subset=M4_WIDE_FEATURES, model_columns=M4_WIDE_FEATURES,
            model_name=f"M4-wide XGB (seed={sd})",
            restrict_to_months=axis, **common,
        )

    perm = oos_permutation_importance(
        m4_wide, M4_WIDE_FEATURES, target, horizon,
        db_path=db_path, min_history_year=min_history_year,
    )
    gap_m4 = train_oos_gap(
        m4_wide, M4_WIDE_FEATURES, target, horizon,
        model_kind="m4", db_path=db_path, min_history_year=min_history_year,
    )
    gap_m2 = train_oos_gap(
        m2, M2_FEATURES, target, horizon,
        model_kind="m2", db_path=db_path, min_history_year=min_history_year,
    )

    return {
        "m4_core": m4_core, "m4_wide": m4_wide,
        "m2": m2, "m1": m1,
        "depth_strip": depth_strip, "nest_strip": nest_strip,
        "seed_strip": seed_strip,
        "perm_importance": perm,
        "gap_m4": gap_m4, "gap_m2": gap_m2,
        "common_axis": axis,
    }


def _early_stopping_read(nest_strip: dict) -> Optional[dict]:
    """Apply the pre-registered early-stopping reading rule to the
    n_estimators strip. Returns {'peak_n', 'peak_auc', 'auc_at_max',
    'decline', 'boosting_into_noise'} or None if insufficient data."""
    pts = [(ne, r.mean_fold_auc) for ne, r in sorted(nest_strip.items())
           if r.mean_fold_auc is not None]
    if len(pts) < 2:
        return None
    peak_n, peak_auc = max(pts, key=lambda kv: kv[1])
    max_n, auc_at_max = pts[-1]            # highest tree count
    decline = peak_auc - auc_at_max
    return {
        "peak_n": peak_n, "peak_auc": peak_auc,
        "max_n": max_n, "auc_at_max": auc_at_max,
        "decline": decline,
        "boosting_into_noise": decline > EARLY_STOP_DECLINE,
    }


def print_m4_report(results: dict) -> None:
    """Pretty-print the M4 (1c+ / 2c+ / 3-hybrid) report."""
    m4c, m4w = results["m4_core"], results["m4_wide"]
    m2, m1 = results["m2"], results["m1"]
    depth_strip = results["depth_strip"]
    nest_strip = results["nest_strip"]
    seed_strip = results["seed_strip"]
    perm = results["perm_importance"]

    def auc(r):
        return r.mean_fold_auc

    print("=" * 72)
    print(f"M4 VALIDATION (XGBoost, decisions 1c+/2c+/3-hybrid) — "
          f"{m4c.target} {m4c.horizon}")
    print("=" * 72)
    for r in (m4c, m4w, m2, m1):
        print()
        print(r.summary())

    # head-to-head
    print()
    print("-" * 72)
    b1 = auc(m1)
    print("  HEAD-TO-HEAD (mean fold AUC):")
    print(f"    M1 baseline (yield curve) : {auc(m1):.4f}   <- PROJECT BASELINE")
    print(f"    M2 (L2 logit, 4 feat)     : {auc(m2):.4f}")
    print(f"    M4-core (XGB, 4 feat)     : {auc(m4c):.4f}")
    print(f"    M4-wide (XGB, 6 feat)     : {auc(m4w):.4f}")
    print()
    best_m4 = max(a for a in (auc(m4c), auc(m4w)) if a is not None)
    if b1 is not None:
        d = best_m4 - b1
        print(f"  best M4 - baseline = {d:+.4f}  "
              + ("M4 BEATS the baseline" if d > 0
                 else "M4 does NOT beat the baseline"))

    # max_depth sensitivity strip
    print()
    print("-" * 72)
    print(f"  max_depth sensitivity (M4-wide; DISPLAY ONLY, headline "
          f"depth={M4_MAX_DEPTH}):")
    print(f"  {'depth':>6} {'mean fold AUC':>15} {'vs baseline':>13}")
    for d, r in depth_strip.items():
        a = auc(r)
        if a is None:
            print(f"  {d:>6} {'n/a':>15}")
            continue
        vs = f"{a - b1:+.4f}" if b1 is not None else "n/a"
        flag = " *headline" if d == M4_MAX_DEPTH else ""
        print(f"  {d:>6} {a:>15.4f} {vs:>13}{flag}")

    # n_estimators strip — DOUBLE DUTY
    print()
    print("  n_estimators strip (M4-wide; DISPLAY ONLY, headline "
          f"n_est={M4_N_ESTIMATORS}) — DOUBLE DUTY:")
    print("    role 1: capacity sensitivity   role 2: early-stopping "
          "diagnostic")
    print(f"  {'n_est':>6} {'mean fold AUC':>15} {'vs baseline':>13}")
    for ne, r in nest_strip.items():
        a = auc(r)
        if a is None:
            print(f"  {ne:>6} {'n/a':>15}")
            continue
        vs = f"{a - b1:+.4f}" if b1 is not None else "n/a"
        flag = " *headline" if ne == M4_N_ESTIMATORS else ""
        print(f"  {ne:>6} {a:>15.4f} {vs:>13}{flag}")
    es = _early_stopping_read(nest_strip)
    if es is not None:
        print(f"    early-stopping read (pre-registered rule, decline "
              f"threshold {EARLY_STOP_DECLINE}):")
        print(f"      peak AUC {es['peak_auc']:.4f} at n_est={es['peak_n']}; "
              f"at n_est={es['max_n']} AUC {es['auc_at_max']:.4f}; "
              f"decline {es['decline']:+.4f}")
        if es["boosting_into_noise"]:
            print("      => BOOSTING INTO NOISE at high tree count "
                  "(decline exceeds threshold) — capacity must stay low.")
        else:
            print("      => no boosting-into-noise signature "
                  "(decline within threshold).")

    # seed-stability strip — decisive robustness gate
    print()
    print(f"  seed stability (M4-wide; DISPLAY ONLY, headline "
          f"seed={M4_HEADLINE_SEED}) — DECISIVE robustness gate:")
    print(f"  {'seed':>6} {'mean fold AUC':>15}")
    seed_aucs = []
    for sd, r in seed_strip.items():
        a = auc(r)
        seed_aucs.append(a)
        flag = " *headline" if sd == M4_HEADLINE_SEED else ""
        print(f"  {sd:>6} {a:>15.4f}{flag}" if a is not None
              else f"  {sd:>6} {'n/a':>15}")
    valid = [a for a in seed_aucs if a is not None]
    if len(valid) > 1:
        spread = max(valid) - min(valid)
        verdict = ("(stable — a win here would be real)" if spread < 0.02
                   else "(UNSTABLE — any marginal win is RNG luck, not real)")
        print(f"  seed spread: {spread:.4f}  {verdict}")

    # OOS permutation importance
    print()
    print("-" * 72)
    print("  OOS permutation importance (M4-wide; DISPLAY ONLY — which "
          "features carried")
    print("  out-of-sample signal):")
    for feat, drop in sorted(perm.items(), key=lambda kv: -kv[1]):
        bar = "#" * max(0, int(drop * 200))
        print(f"  {feat:>14} {drop:+.4f}  {bar}")

    # train-vs-OOS gap
    gap_m4 = results.get("gap_m4")
    gap_m2 = results.get("gap_m2")
    if gap_m4 is not None and gap_m2 is not None:
        print()
        print("-" * 72)
        print("  TRAIN-vs-OOS AUC gap (memorisation signature; M2 is the "
              "low-variance reference):")
        if gap_m4["mean_gap"] is not None:
            print(f"    M4-wide:  train {gap_m4['mean_train_auc']:.4f}  "
                  f"OOS {gap_m4['mean_oos_auc']:.4f}  "
                  f"gap {gap_m4['mean_gap']:+.4f}")
        if gap_m2["mean_gap"] is not None:
            print(f"    M2 (ref): train {gap_m2['mean_train_auc']:.4f}  "
                  f"OOS {gap_m2['mean_oos_auc']:.4f}  "
                  f"gap {gap_m2['mean_gap']:+.4f}")
        if (gap_m4["mean_gap"] is not None
                and gap_m2["mean_gap"] is not None):
            excess = gap_m4["mean_gap"] - gap_m2["mean_gap"]
            print(f"    M4 excess gap over M2 reference: {excess:+.4f}  "
                  + ("(M4 memorises more — expected for boosting; OOS is "
                     "what counts)" if excess > 0.05
                     else "(comparable to the linear reference)"))
    print("=" * 72)
