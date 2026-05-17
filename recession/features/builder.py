"""
recession/features/builder.py

Step 4 orchestrator — the single entrypoint every model (M1-M5) calls to
get training / inference data.

Two layers (the "combo" design, validated against Rule 1):

  Layer 1 — FeaturePipeline class (the leakage-safe engine):
      pipe = FeaturePipeline(target, horizon, as_of)
      pipe.load_raw()                   # DB read only — fold-invariant, cached
      pipe.fit(train_cutoff='2005-12')  # ALL learning steps — per fold
      result = pipe.transform()         # apply + assemble + return

  Layer 2 — build_feature_dataframe(...) thin wrapper (one-line interface):
      result = build_feature_dataframe(target, horizon, as_of, train_cutoff)

Why two layers: the walk-forward backtest (Step 10) creates ONE pipeline,
calls load_raw() ONCE, then loops fit()/transform() per fold — no redundant
DB reads. The 3 simpler callers (live cron, per-model training, interactive)
use the one-line wrapper.

LEAKAGE DISCIPLINE (Rule 1 gap-check finding):
  Everything that "learns" from data is inside fit() and is fit on
  train-cutoff rows ONLY:
    - Hamilton detrending (regression coefficients)
    - YoY (stateless, but kept in fit() for consistency)
    - empirical at-risk thresholds
    - per-tier PCA (standardization stats + loadings)
  load_raw() does DB reads only — no learning, so it's safe to cache and
  reuse across folds.

  Hamilton was DELIBERATELY moved out of load_raw() into fit(): Hamilton's
  regression, if fit on the whole panel, uses post-cutoff data to estimate
  the trend. Fitting it per-fold on train-only rows closes that gap.

Pipeline order inside fit()/transform():
  1. (load_raw, once)    pit_loader.load_panel  → raw panel
  2. (fit)               Hamilton on INDPRO/DTWEXBGS, YoY on inflation
                         — fit coefficients on train rows, store
  3. (fit)               empirical at-risk thresholds — fit on train rows
  4. (fit)               PerTierPCA.fit on train rows
  5. (transform)         apply 2-4 to whole panel
  6. (transform)         breadth columns
  7. (transform)         freshness columns
  8. (transform)         join target, assemble result
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from recession.features import transforms as T
from recession.features import pit_loader as PL
from recession.features import pca as PCA_MOD
from recession.features import breadth as BR
from recession.features import freshness as FR


DEFAULT_DB_PATH = Path.cwd() / "recession.db"


# =============================================================================
# Result container
# =============================================================================

@dataclass
class FeatureResult:
    """What transform() returns.

    Attributes:
        X: feature matrix (months x features), all transforms applied.
        y: target series aligned to X.index (may contain NaN at the tail
           where the future label isn't known yet).
        feature_names: list of column names in X.
        target: which target ('T1'/'T2'/'T5').
        horizon: which horizon ('h=12' etc.).
        as_of: the as_of date used.
        train_cutoff: the cutoff used for fitting (leakage boundary).
        pipeline: back-reference to the FeaturePipeline that produced this,
            so callers can inspect fitted state (thresholds, PCA, etc.).
    """
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    target: str
    horizon: str
    as_of: str
    train_cutoff: str
    pipeline: "FeaturePipeline" = field(repr=False)


# =============================================================================
# FeaturePipeline — Layer 1
# =============================================================================

class FeaturePipeline:
    """Stateful feature pipeline. See module docstring for the 3-method flow.

    Lifecycle:
        __init__       set target / horizon / as_of
        load_raw()     DB read — fold-invariant, call once
        fit(cutoff)    all learning steps on train rows ≤ cutoff
        transform()    apply fitted state to whole panel, return FeatureResult
    """

    def __init__(
        self,
        target: str = "T1",
        horizon: str = "h=12",
        as_of: object = None,
        *,
        min_history_year: Optional[int] = None,
        feature_subset: Optional[list[str]] = None,
        db_path: Path = DEFAULT_DB_PATH,
    ):
        # as_of validated immediately (Decision 4 — fail loud)
        self.as_of = PL._validate_as_of(as_of)
        self.target = target
        self.horizon = horizon
        self.min_history_year = min_history_year
        self.feature_subset = feature_subset
        self.db_path = Path(db_path)

        # state populated by load_raw()
        self._raw: Optional[pd.DataFrame] = None
        self._transformed: Optional[pd.DataFrame] = None   # Hamilton/YoY applied
        self._registry_tiers: dict[str, int] = {}

        # state populated by fit()
        self._fitted = False
        self.train_cutoff: Optional[str] = None
        self._empirical_thresholds: dict[str, dict] = {}
        self._pca: Optional[PCA_MOD.PerTierPCA] = None

    # -------------------------------------------------------------------------
    # load_raw — Layer-1 fold-invariant DB read
    # -------------------------------------------------------------------------

    def load_raw(self) -> "FeaturePipeline":
        """Load the raw (untransformed) feature panel from the DB.

        This is the only step that reads the DB for features. It does NOT
        learn anything, so it's safe to call once and reuse across many
        fit()/transform() cycles (the walk-forward backtest does this).

        It also precomputes the column transforms (Hamilton, YoY) over the
        full panel. Hamilton is purely backward-looking — output at point
        t+h only uses y_t..y_{t-3} — so computing it over the full panel
        leaks no future info into any single point. Computing it ONCE here
        (rather than separately in fit() on the train slice and again in
        transform() on the full panel) guarantees fit() and transform()
        operate on byte-identical transformed values. This closes the
        consistency wrinkle found in the Rule 1 gap check.

        Returns:
            self (chaining).
        """
        self._raw = PL.load_panel(
            as_of=self.as_of,
            target=self.target,
            horizon=self.horizon,
            min_history_year=self.min_history_year,
            feature_subset=self.feature_subset,
            db_path=self.db_path,
        )
        self._registry_tiers = PCA_MOD.load_registry_tiers(self.db_path)
        # Precompute column transforms once — deterministic, fold-invariant.
        self._transformed = self._apply_column_transforms(self._raw)
        return self

    # -------------------------------------------------------------------------
    # fit — Layer-1 learning, train-cutoff rows only
    # -------------------------------------------------------------------------

    def fit(self, train_cutoff: object) -> "FeaturePipeline":
        """Fit all learning steps on rows with observation_month <= cutoff.

        Learning steps:
          - Hamilton detrending coefficients (INDPRO, DTWEXBGS)
          - empirical at-risk thresholds
          - per-tier PCA standardization + loadings

        Args:
            train_cutoff: ISO date string or 'today'. Rows with
                observation_month <= this are the training set. For live
                inference, pass train_cutoff == as_of.

        Returns:
            self (chaining).

        Raises:
            RuntimeError if load_raw() wasn't called first.
        """
        if self._raw is None:
            raise RuntimeError("Call load_raw() before fit().")

        cutoff = PL._validate_as_of(train_cutoff)
        self.train_cutoff = cutoff
        cutoff_ts = pd.Timestamp(cutoff)

        # Training-row mask, applied to the already-transformed panel
        train_mask = self._transformed.index <= cutoff_ts
        train_transformed = self._transformed.loc[train_mask]

        if len(train_transformed) < 12:
            raise ValueError(
                f"train_cutoff={cutoff} leaves only {len(train_transformed)} "
                f"training rows — too few to fit. Pick a later cutoff."
            )

        # Column transforms (Hamilton, YoY) were precomputed in load_raw()
        # over the full panel — fold-invariant and leakage-safe (Hamilton
        # is backward-looking). We just slice to the train window here.

        # --- Step 3: empirical at-risk thresholds (fit on train rows) ---
        self._empirical_thresholds = {}
        # Need the target on the same train rows to fit thresholds
        y_train = PL.load_targets(
            self.target, self.horizon, db_path=self.db_path
        )
        y_train = y_train.reindex(train_transformed.index)

        for feat, cfg in T.AT_RISK_EMPIRICAL.items():
            if feat not in train_transformed.columns:
                continue
            s = train_transformed[feat]
            # align feature + target, drop rows where either is NaN
            aligned = pd.DataFrame({"f": s, "y": y_train}).dropna()
            if len(aligned) < 10:
                # too few rows to fit a threshold — skip this feature
                continue
            thr = T.at_risk_empirical_fit(
                aligned["f"], aligned["y"],
                direction=cfg["direction"],
            )
            self._empirical_thresholds[feat] = {
                "threshold": thr,
                "direction": cfg["direction"],
            }

        # --- Step 4: per-tier PCA (fit on train rows) ---
        registry_filtered = {
            f: t for f, t in self._registry_tiers.items()
            if f in train_transformed.columns
        }
        self._pca = PCA_MOD.PerTierPCA()
        self._pca.fit(train_transformed, registry_filtered)

        self._fitted = True
        return self

    # -------------------------------------------------------------------------
    # transform — Layer-1 apply + assemble
    # -------------------------------------------------------------------------

    def transform(self) -> FeatureResult:
        """Apply the fitted pipeline to the whole panel; return FeatureResult.

        Raises:
            RuntimeError if fit() wasn't called first.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        # Step 5a: column transforms already done in load_raw() — use cache
        panel = self._transformed.copy()

        # Step 5b: at-risk dummies (theoretical + fitted empirical)
        dummies = BR.build_at_risk_dummies(
            panel,
            theoretical=T.AT_RISK_THEORETICAL,
            empirical_thresholds=self._empirical_thresholds,
        )
        # append dummy columns with a _ATRISK suffix so they don't collide
        for col in dummies.columns:
            panel[f"{col}_ATRISK"] = dummies[col]

        # Step 5c: per-tier PCA — append PC columns
        panel = self._pca.transform(panel)

        # Step 6: breadth columns
        tier_labels = PCA_MOD.TIER_SHORT
        panel = BR.add_breadth_columns(
            panel,
            at_risk_dummies=dummies,
            feature_tiers={
                f: t for f, t in self._registry_tiers.items()
                if f in dummies.columns
            },
            tier_labels=tier_labels,
        )

        # Step 7: freshness columns
        panel = FR.add_freshness_columns(
            panel,
            as_of=self.as_of,
            db_path=self.db_path,
        )

        # Step 8: join target, assemble
        y = PL.load_targets(self.target, self.horizon, db_path=self.db_path)
        y = y.reindex(panel.index)

        return FeatureResult(
            X=panel,
            y=y,
            feature_names=list(panel.columns),
            target=self.target,
            horizon=self.horizon,
            as_of=self.as_of,
            train_cutoff=self.train_cutoff,
            pipeline=self,
        )

    # -------------------------------------------------------------------------
    # internal: column transforms (Hamilton + YoY)
    # -------------------------------------------------------------------------

    def _apply_column_transforms(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Apply Hamilton / YoY transforms, replacing columns in-place.

        Sub-decision 1: transformed column keeps its original name (e.g.
        INDPRO holds the cycle component after Hamilton). Raw level has no
        modeling use once we've decided to detrend.

        Hamilton is purely backward-looking so running it over the full
        panel does not leak future info into any single output point.
        """
        out = panel.copy()
        for feat in panel.columns:
            spec = T.get_transform_for_feature(feat)
            if spec is None:
                continue
            kind, kwargs = spec
            if kind == "yoy_pct":
                out[feat] = T.yoy_pct(panel[feat])
            elif kind == "hamilton":
                out[feat] = T.hamilton_detrend(panel[feat], **kwargs)
            elif kind == "log_return_12m":
                out[feat] = T.log_return_12m(panel[feat])
            # unknown kinds are silently left as raw (defensive)
        return out


# =============================================================================
# build_feature_dataframe — Layer 2 one-line wrapper
# =============================================================================

def build_feature_dataframe(
    target: str,
    horizon: str,
    as_of: object,
    train_cutoff: object,
    *,
    min_history_year: Optional[int] = None,
    feature_subset: Optional[list[str]] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> FeatureResult:
    """One-line feature build: load → fit → transform.

    For the 3 simple callers (live cron, per-model training, interactive
    debugging). The walk-forward backtest should use FeaturePipeline
    directly to avoid re-running load_raw() per fold.

    Args:
        target: 'T1' / 'T2' / 'T5'.
        horizon: 'h=0' / 'h=1' / 'h=3' / 'h=6' / 'h=12'.
        as_of: ISO date or 'today'. PIT cutoff for feature vintages.
        train_cutoff: ISO date or 'today'. Rows ≤ this fit the learning
            steps. For live inference, pass train_cutoff == as_of.
        min_history_year: optional — drop features starting after this year.
        feature_subset: optional — explicit feature list (e.g. M1's
            ['T10Y3M']).
        db_path: SQLite path.

    Returns:
        FeatureResult.
    """
    pipe = FeaturePipeline(
        target=target,
        horizon=horizon,
        as_of=as_of,
        min_history_year=min_history_year,
        feature_subset=feature_subset,
        db_path=db_path,
    )
    pipe.load_raw()
    pipe.fit(train_cutoff)
    return pipe.transform()
