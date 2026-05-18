"""
recession/validation/walk_forward.py

Walk-forward validation harness — the "validation spine" of the recession
model project. Every model (M1-M8) is validated through this harness, so it
is deliberately model-agnostic.

DESIGN (locked in Step 5 planning)
----------------------------------
- Expanding window. The training window grows each fold: fold 1 trains on
  1960..T1, fold 2 on 1960..T2, etc. Rolling windows were rejected because
  recessions are rare (~95 recession-months in 65 years) — a fixed-width
  rolling window could leave a fold with zero recessions to learn from.
  Expanding guarantees every fold trains on multiple recessions.

- Embargo = the forecast horizon h. This is the critical leakage guard.
  Target T1 at h=12 means "recession within 12 months", so a training row
  dated M has a LABEL that depends on data through M+h. If the test fold
  starts at S, training rows from S-h onward have labels that peek into the
  test period. The harness drops the final h months of every training
  window so no training label overlaps the test fold.

- Metrics per fold: ROC-AUC (threshold-free discrimination), Brier score
  (calibration), and a confusion matrix at a fixed probability threshold.
  Aggregated across folds at the end. Lead-time analysis is NOT computed
  here — the harness exposes raw per-fold predictions so Step 10 can do it.

- Model-agnostic via a tiny interface. The harness only ever calls
  model.fit(X, y) and model.predict_proba(X). Anything implementing those
  two methods (the RecessionModel protocol) can be validated — statsmodels
  Probit, sklearn LogisticRegression, XGBoost, etc.

LEAKAGE DISCIPLINE
------------------
The harness obtains features by calling build_feature_dataframe with
train_cutoff set to each fold's (embargoed) training boundary. The feature
pipeline then fits its learned transforms (PCA, empirical thresholds) on
training rows only. The harness adds the embargo on top. Two layers:
  - builder's train_cutoff: feature transforms fit on <= cutoff
  - harness embargo: training LABELS dropped within h of the test fold
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from recession.features.builder import build_feature_dataframe


# =============================================================================
# Model interface
# =============================================================================

@runtime_checkable
class RecessionModel(Protocol):
    """The interface every model must implement to be validated.

    Deliberately minimal — two methods. This is what lets the harness
    validate statsmodels, sklearn, and XGBoost models identically.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RecessionModel":
        """Fit the model on training features X and binary target y."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(recession) for each row of X — a 1-D array in [0, 1]."""
        ...


# =============================================================================
# Result containers
# =============================================================================

@dataclass
class FoldResult:
    """Metrics and predictions for one walk-forward fold."""
    fold_index: int
    train_start: str
    train_end: str            # embargoed training boundary
    test_start: str
    test_end: str
    n_train: int
    n_train_positive: int     # recession months in training
    n_test: int
    n_test_positive: int
    auc: Optional[float]      # None if test fold has only one class
    brier: Optional[float]
    # confusion at threshold (TP, FP, TN, FN)
    tp: int
    fp: int
    tn: int
    fn: int
    # raw predictions for downstream (e.g. Step 10 lead-time analysis)
    test_dates: list[str] = field(default_factory=list, repr=False)
    test_proba: list[float] = field(default_factory=list, repr=False)
    test_actual: list[int] = field(default_factory=list, repr=False)


@dataclass
class WalkForwardResult:
    """Aggregate result across all folds."""
    model_name: str
    target: str
    horizon: str
    n_folds: int
    folds: list[FoldResult]
    # pooled metrics (computed over all test predictions concatenated).
    # NOTE: pooled AUC is a SECONDARY diagnostic only. With expanding folds
    # the test windows overlap, and each fold refits the model, so per-fold
    # probability scales drift — pooling raw probabilities then ranking
    # globally is not a valid headline metric. Use mean_fold_auc.
    pooled_auc: Optional[float]
    pooled_brier: Optional[float]
    # HEADLINE METRIC: mean of per-fold AUCs over scoreable (two-class) folds.
    mean_fold_auc: Optional[float]
    n_scoreable_folds: int          # folds with a defined (two-class) AUC
    threshold: float

    def summary(self) -> str:
        """One-line-per-metric human summary. Headline = mean fold AUC."""
        def fmt(v):
            return f"{v:.4f}" if v is not None else "n/a"

        return "\n".join([
            f"Walk-forward: {self.model_name} | {self.target} {self.horizon}",
            f"  folds: {self.n_folds}  "
            f"(scoreable / two-class: {self.n_scoreable_folds})",
            f"  HEADLINE mean fold AUC: {fmt(self.mean_fold_auc)}",
            f"  mean fold Brier:        {fmt(self._mean_fold_brier())}",
            f"  pooled AUC (secondary): {fmt(self.pooled_auc)}",
        ])

    def _mean_fold_brier(self) -> Optional[float]:
        """Mean Brier over folds (Brier is defined on single-class folds
        too, unlike AUC)."""
        briers = [f.brier for f in self.folds if f.brier is not None]
        return float(np.mean(briers)) if briers else None


# =============================================================================
# Metric helpers (no sklearn dependency — keep the harness light)
# =============================================================================

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """ROC-AUC via the rank-sum (Mann-Whitney U) identity.

    Returns None if y_true is single-class (AUC undefined).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    # rank scores; ties get average rank
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    _assign_tie_ranks(y_score, ranks)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _assign_tie_ranks(scores: np.ndarray, ranks: np.ndarray) -> None:
    """In-place: replace ranks of tied scores with their average rank."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and outcome."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def confusion_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN) at a probability threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


# =============================================================================
# Fold generation
# =============================================================================

def _horizon_months(horizon: str) -> int:
    """Parse 'h=12' -> 12."""
    return int(horizon.split("=")[1])


def make_expanding_folds(
    all_months: pd.DatetimeIndex,
    horizon: str,
    min_train_months: int,
    test_window_months: int,
    step_months: int,
    labels: Optional[pd.Series] = None,
    require_two_class_test: bool = True,
) -> list[dict]:
    """Generate expanding-window folds with an embargo.

    Recession-aware fold generation. Recessions are rare and clustered, so a
    naive fixed-width test window very often lands entirely inside one regime
    (all-recession or all-calm) — a single-class fold, on which AUC is
    undefined. Such folds pollute any pooled metric. This generator can skip
    single-class test folds at generation time, given the label series.

    Args:
        all_months: every observation month available, sorted ascending.
        horizon: 'h=12' etc. — the embargo length in months.
        min_train_months: the first fold trains on at least this many months
            (before the embargo is subtracted).
        test_window_months: months per test fold. For rare-event targets
            this should be wide enough that a window can straddle a regime
            change (e.g. 60). Narrow windows mostly produce single-class
            folds.
        step_months: months to advance the test window each fold.
        labels: optional label series (the target y) indexed by month. If
            given and require_two_class_test is True, folds whose test
            window is single-class are skipped.
        require_two_class_test: when True (and labels given), only emit
            folds whose test window contains BOTH classes.

    Returns:
        list of fold dicts with keys train_start, train_end_embargoed,
        test_start, test_end. train_end_embargoed already has the h-month
        embargo subtracted.
    """
    h = _horizon_months(horizon)
    months = list(all_months)
    n = len(months)
    folds = []

    # The first test fold starts after min_train_months + embargo.
    test_start_idx = min_train_months + h
    fold_idx = 0
    while test_start_idx + test_window_months <= n:
        test_lo = test_start_idx
        test_hi = min(test_start_idx + test_window_months, n)
        # training window: everything from the start up to (test_start - h - 1)
        train_hi_embargoed = test_lo - h          # exclusive upper bound
        if train_hi_embargoed < min_train_months:
            test_start_idx += step_months
            continue

        # Skip single-class test folds — AUC is undefined on them and they
        # corrupt any aggregate metric.
        if require_two_class_test and labels is not None:
            test_months = months[test_lo:test_hi]
            test_labels = labels.reindex(test_months).dropna()
            n_pos = int((test_labels == 1).sum())
            n_neg = int((test_labels == 0).sum())
            if n_pos == 0 or n_neg == 0:
                test_start_idx += step_months
                continue

        folds.append({
            "fold_index": fold_idx,
            "train_start": months[0],
            "train_end_embargoed": months[train_hi_embargoed - 1],
            "test_start": months[test_lo],
            "test_end": months[test_hi - 1],
            "_train_hi_idx": train_hi_embargoed,    # exclusive
            "_test_lo_idx": test_lo,
            "_test_hi_idx": test_hi,                 # exclusive
        })
        fold_idx += 1
        test_start_idx += step_months

    return folds


# =============================================================================
# The harness
# =============================================================================

def walk_forward(
    model_factory,
    target: str,
    horizon: str,
    *,
    feature_subset: Optional[list[str]] = None,
    model_columns: Optional[list[str]] = None,
    model_name: str = "model",
    min_train_months: int = 240,        # 20 years before the first test fold
    test_window_months: int = 60,       # 5y — wide enough to straddle regimes
    step_months: int = 12,
    threshold: float = 0.5,
    min_history_year: Optional[int] = None,
    db_path: Optional[Path] = None,
    restrict_to_months: Optional[pd.DatetimeIndex] = None,
) -> WalkForwardResult:
    """Run expanding-window walk-forward validation for one model.

    Args:
        model_factory: a zero-arg callable returning a FRESH model each call
            (the harness fits one model per fold — it must not reuse a
            fitted model across folds).
        target: 'T1' / 'T2' / 'T5'.
        horizon: 'h=12' etc. Also the embargo length.
        feature_subset: explicit RAW feature list passed to the builder
            (controls which raw features the feature pipeline loads). If
            None, the builder loads all eligible features.
        model_columns: the EXACT columns the model is fit on, selected from
            the built dataframe. This is distinct from feature_subset: the
            builder always appends engineered columns (at-risk dummies,
            breadth, freshness, PCs) on top of the raw features, so a model
            that should see only its named raw features (e.g. M1's static
            probit) MUST pass model_columns to exclude the engineered ones.
            If None, the model is fit on every column the builder produced.
        model_name: label for the result.
        min_train_months: minimum training months before the first fold.
        test_window_months: months per test fold.
        step_months: months to advance each fold.
        threshold: probability threshold for the confusion matrix.
        min_history_year: passed through to the builder.
        db_path: passed through to the builder.

    Returns:
        WalkForwardResult.
    """
    # First, get the full panel once (live mode) just to discover the month
    # axis and target availability. We use as_of='today' for axis discovery;
    # each fold re-builds features with its own train_cutoff.
    build_kwargs = {}
    if db_path is not None:
        build_kwargs["db_path"] = db_path
    if min_history_year is not None:
        build_kwargs["min_history_year"] = min_history_year

    probe = build_feature_dataframe(
        target=target, horizon=horizon,
        as_of="today", train_cutoff="today",
        feature_subset=feature_subset,
        **build_kwargs,
    )

    # Months where the target is known (non-NaN).
    labelled = probe.y.dropna()

    # Restrict the fold axis to months where the MODEL'S FEATURES exist.
    # The target spans 1960-present, but a model's features may start much
    # later (e.g. T10Y3M ~1982). Folds before the features exist would test
    # on all-NaN rows. We intersect the labelled months with the months
    # where every model column is present.
    #
    # restrict_to_months: when given, it overrides the auto-computed axis.
    # B+ uses this so the full model and its sub-baseline (which have
    # DIFFERENT feature-availability windows) are validated on one common
    # axis — otherwise their fold sets diverge and the comparison is void.
    probe_X = probe.X
    if restrict_to_months is not None:
        usable_months = labelled.index.intersection(restrict_to_months)
    else:
        if model_columns is not None:
            feat_cols = [c for c in model_columns if c in probe_X.columns]
        else:
            feat_cols = list(probe_X.columns)
        if feat_cols:
            feats_present = probe_X[feat_cols].notna().all(axis=1)
            feature_months = probe_X.index[feats_present]
            usable_months = labelled.index.intersection(feature_months)
        else:
            usable_months = labelled.index
    all_months = usable_months.sort_values()

    # Labels on the usable axis — used to skip single-class test folds.
    fold_labels = probe.y.reindex(all_months)

    folds_spec = make_expanding_folds(
        all_months, horizon,
        min_train_months=min_train_months,
        test_window_months=test_window_months,
        step_months=step_months,
        labels=fold_labels,
        require_two_class_test=True,
    )

    fold_results: list[FoldResult] = []
    pooled_proba: list[float] = []
    pooled_actual: list[int] = []

    for spec in folds_spec:
        # Build features as-of the test fold start (PIT honest), with the
        # feature pipeline's learned transforms fit on the embargoed
        # training boundary.
        as_of = spec["test_end"].strftime("%Y-%m-%d")
        train_cutoff = spec["train_end_embargoed"].strftime("%Y-%m-%d")

        fr = build_feature_dataframe(
            target=target, horizon=horizon,
            as_of=as_of, train_cutoff=train_cutoff,
            feature_subset=feature_subset,
            **build_kwargs,
        )
        X, y = fr.X, fr.y

        # Restrict to the model's intended columns. The builder appends
        # engineered columns (at-risk dummies, breadth, freshness, PCs) on
        # top of the raw features; a model that should only see its named
        # features (e.g. M1's static probit) passes model_columns.
        if model_columns is not None:
            missing = [c for c in model_columns if c not in X.columns]
            if missing:
                raise KeyError(
                    f"model_columns not found in built features: {missing}. "
                    f"Available: {sorted(X.columns)}"
                )
            X = X[model_columns]

        # Training rows: index <= embargoed train boundary, target not NaN.
        train_mask = (X.index <= spec["train_end_embargoed"]) & y.notna()
        # Test rows: within the test window, target not NaN.
        test_mask = (
            (X.index >= spec["test_start"])
            & (X.index <= spec["test_end"])
            & y.notna()
        )

        X_train, y_train = X.loc[train_mask], y.loc[train_mask].astype(int)
        X_test, y_test = X.loc[test_mask], y.loc[test_mask].astype(int)

        # Drop feature columns that are all-NaN in training (e.g. a feature
        # with no data yet this early). Models can't use them.
        usable_cols = [c for c in X_train.columns
                       if X_train[c].notna().any()]
        X_train = X_train[usable_cols]
        X_test = X_test[usable_cols]

        # Rows with any NaN feature can't be fit/predicted — drop them.
        tr_ok = X_train.notna().all(axis=1)
        te_ok = X_test.notna().all(axis=1)
        X_train, y_train = X_train.loc[tr_ok], y_train.loc[tr_ok]
        X_test, y_test = X_test.loc[te_ok], y_test.loc[te_ok]

        if len(X_train) < 24 or len(X_test) == 0:
            # not enough to fit or nothing to test — skip this fold
            continue

        # Fit a FRESH model and predict.
        model = model_factory()
        model.fit(X_train, y_train)
        proba = np.asarray(model.predict_proba(X_test), dtype=float)

        # Metrics
        yt = y_test.to_numpy()
        auc = roc_auc(yt, proba)
        brier = brier_score(yt, proba)
        tp, fp, tn, fn = confusion_at_threshold(yt, proba, threshold)

        fold_results.append(FoldResult(
            fold_index=spec["fold_index"],
            train_start=spec["train_start"].strftime("%Y-%m-%d"),
            train_end=spec["train_end_embargoed"].strftime("%Y-%m-%d"),
            test_start=spec["test_start"].strftime("%Y-%m-%d"),
            test_end=spec["test_end"].strftime("%Y-%m-%d"),
            n_train=len(X_train),
            n_train_positive=int(y_train.sum()),
            n_test=len(X_test),
            n_test_positive=int(y_test.sum()),
            auc=auc, brier=brier,
            tp=tp, fp=fp, tn=tn, fn=fn,
            test_dates=[d.strftime("%Y-%m-%d") for d in X_test.index],
            test_proba=proba.tolist(),
            test_actual=yt.tolist(),
        ))
        pooled_proba.extend(proba.tolist())
        pooled_actual.extend(yt.tolist())

    # Aggregate
    pooled_auc = (roc_auc(np.array(pooled_actual), np.array(pooled_proba))
                  if pooled_actual else None)
    pooled_brier = (brier_score(np.array(pooled_actual), np.array(pooled_proba))
                    if pooled_actual else None)
    fold_aucs = [f.auc for f in fold_results if f.auc is not None]
    mean_fold_auc = float(np.mean(fold_aucs)) if fold_aucs else None
    n_scoreable = len(fold_aucs)

    return WalkForwardResult(
        model_name=model_name,
        target=target,
        horizon=horizon,
        n_folds=len(fold_results),
        folds=fold_results,
        pooled_auc=pooled_auc,
        pooled_brier=pooled_brier,
        mean_fold_auc=mean_fold_auc,
        n_scoreable_folds=n_scoreable,
        threshold=threshold,
    )
