"""
recession/features/pca.py

Per-tier PCA for collinearity cleanup and interpretability.

Decision 3 + sub-decisions A-E recap:
  A. One PC per tier (composite "factor")
  B. Min tier size 3+ to apply PCA
  C. Z-score standardization on training fold
  D. Fit on complete-rows-only; transform is NaN-tolerant
  E. Append PCs to raw panel (both available)

Key class: PerTierPCA
  - fit(panel, registry):  learn standardization stats and PCA loadings per tier
  - transform(panel):       apply learned transformations to any panel
  - fit_transform(...):     convenience

The class enforces no-leakage: fit on train fold, then transform is pure
(no peek at the data). Train mean/std and PCA loadings are stored on the
fitted instance.

Design rule: this is a stateful class because PCA needs to remember:
  - per-feature train mean and std (for z-scoring)
  - per-tier PCA loadings (eigenvectors)
  - per-tier sign convention (eigenvectors are sign-ambiguous; we fix the
    sign so that the PC correlates positively with the most-correlated-
    with-recession feature in each tier)

Using sklearn under the hood for the PCA math because it's well-tested.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

# sklearn is part of the standard scientific Python stack we already use
# (the equity-side ML Quant Fund uses it heavily; it's in ml_quant_310 env)
from sklearn.decomposition import PCA


# =============================================================================
# Configuration
# =============================================================================

MIN_TIER_SIZE = 3   # Sub-decision B: only apply PCA to tiers with 3+ features

# Map tier number -> short label used in PC column name.
# Matches features_registry.tier_label values exactly (verified against
# live DB on May 3 2026 — see Step 4 module 3 review).
TIER_SHORT: dict[int, str] = {
    1:  "yield_credit",
    2:  "labor",
    3:  "real_activity",       # (was wrongly "real_economy" in earlier draft)
    4:  "financial_conditions",
    5:  "monetary_stance",
    6:  "credit_supply",
    7:  "global",
    8:  "ai_cycle",
    9:  "housing",
    10: "inflation",
    11: "engineered",
}

# Default sign anchor per tier. Convention: PC value should be POSITIVE
# in recession-favoring conditions. We pick a feature in each tier whose
# direction makes "positive PC = stress / recession-favoring" intuitive.
#
# Anchor must be a feature that's actually in the tier per the live DB.
# Direction comes from the v2 exploration's univariate AUC analysis.
# Where multiple features have the right direction, we pick the one with
# the highest AUC vs T1.
DEFAULT_SIGN_ANCHORS: dict[int, str] = {
    1:  "EBP",            # tier 1 yield_credit. higher_means_recession (T1 AUC 0.65)
    2:  "SAHMREALTIME",   # tier 2 labor. higher_means_recession
    3:  "INDPRO",         # tier 3 real_activity. higher_means_recession (post-Hamilton cycle)
    4:  "NFCI",           # tier 4 financial_conditions. higher_means_recession (T1 AUC 0.80)
    7:  "DCOILWTICO",     # tier 7 global. higher_means_recession (oil spikes precede recessions)
    10: "CPILFESL",       # tier 10 inflation. higher_means_recession (inflation peaks → recession)
}


# =============================================================================
# PerTierPCA class
# =============================================================================

class PerTierPCA:
    """Fit one PCA per tier; apply with NaN-tolerant transform.

    Usage:
        ptp = PerTierPCA()
        ptp.fit(train_panel, registry)
        transformed_train = ptp.transform(train_panel)
        transformed_test  = ptp.transform(test_panel)

        # Or:
        transformed_train = ptp.fit_transform(train_panel, registry)

    State stored after fit():
        self.feature_tiers: {feature_name -> tier_int} from registry
        self.tier_features: {tier_int -> list of feature_names with 3+ count}
        self.train_mean:    {feature_name -> float}  z-score numerator subtractor
        self.train_std:     {feature_name -> float}  z-score denominator
        self.pca_models:    {tier_int -> sklearn PCA object}
        self.pc_signs:      {tier_int -> +1 or -1}   sign convention
    """

    def __init__(self):
        self.feature_tiers: dict[str, int] = {}
        self.tier_features: dict[int, list[str]] = {}
        self.train_mean:    dict[str, float] = {}
        self.train_std:     dict[str, float] = {}
        self.pca_models:    dict[int, PCA] = {}
        self.pc_signs:      dict[int, int] = {}
        self._fitted = False

    # -------------------------------------------------------------------------
    # fit
    # -------------------------------------------------------------------------

    def fit(
        self,
        panel: pd.DataFrame,
        registry: dict[str, int],
        sign_anchor: Optional[dict[int, str]] = None,
    ) -> "PerTierPCA":
        """Fit z-score stats and PCA loadings per tier.

        Args:
            panel: training feature panel (index = months, cols = features).
                Use the panel returned by pit_loader.load_panel(as_of=fold_cutoff).
            registry: {feature_name -> tier_int}. Get from features_registry table.
            sign_anchor: optional {tier_int -> feature_name} to fix the sign
                convention. PC will be flipped (if necessary) so that the PC
                correlates POSITIVELY with the anchor feature. If a tier
                isn't in this dict, sign anchored to first feature in tier
                alphabetically. This makes PCs comparable across folds.

        Returns:
            self (for chaining).
        """
        # 1. Map every feature to its tier (only features actually in panel)
        self.feature_tiers = {
            f: registry[f] for f in panel.columns if f in registry
        }

        # 2. Group features by tier; only keep tiers with >= MIN_TIER_SIZE
        tier_to_feats: dict[int, list[str]] = defaultdict(list)
        for f, t in self.feature_tiers.items():
            tier_to_feats[t].append(f)
        self.tier_features = {
            t: sorted(feats)
            for t, feats in tier_to_feats.items()
            if len(feats) >= MIN_TIER_SIZE
        }

        # 3. Compute z-score stats per feature (on training panel)
        # Use only features in tiers we'll PCA; standardize each feature.
        for tier, feats in self.tier_features.items():
            for f in feats:
                col = panel[f].dropna()
                if len(col) < 2:
                    # Pathological: feature has 0-1 observations in training.
                    # Set mean=0, std=1 to act as identity.
                    self.train_mean[f] = 0.0
                    self.train_std[f]  = 1.0
                    continue
                mu, sd = float(col.mean()), float(col.std(ddof=0))
                self.train_mean[f] = mu
                # Guard against zero variance (constant feature in this fold)
                self.train_std[f]  = sd if sd > 1e-12 else 1.0

        # 4. Fit PCA per tier on z-scored, complete-rows-only data
        for tier, feats in self.tier_features.items():
            sub = panel[feats].copy()
            # z-score using train mean/std
            for f in feats:
                sub[f] = (sub[f] - self.train_mean[f]) / self.train_std[f]
            # Complete rows only (Sub-decision D)
            complete = sub.dropna(axis=0, how="any")
            if len(complete) < MIN_TIER_SIZE:
                # Not enough complete rows to fit PCA reliably.
                # Skip this tier — caller will see no PC for this tier.
                # Log via a printed message (caller can capture if needed).
                continue
            pca = PCA(n_components=1)   # Sub-decision A: 1 PC per tier
            pca.fit(complete.values)
            self.pca_models[tier] = pca

            # Determine sign convention. PCA loadings are unique up to sign;
            # without anchoring, two folds could produce PCs with flipped
            # signs, which makes time-series comparison and model coefficient
            # interpretation impossible.
            #
            # Convention: positive PC = recession-favoring conditions.
            # Use DEFAULT_SIGN_ANCHORS unless caller passes a custom mapping.
            anchor_feat = (
                (sign_anchor or {}).get(tier)
                or DEFAULT_SIGN_ANCHORS.get(tier)
                or feats[0]    # last-resort fallback: alphabetically first
            )
            if anchor_feat not in feats:
                anchor_feat = feats[0]
            anchor_idx = feats.index(anchor_feat)
            # Loading of anchor feature in PC1
            anchor_loading = pca.components_[0, anchor_idx]
            self.pc_signs[tier] = -1 if anchor_loading < 0 else 1

        self._fitted = True
        return self

    # -------------------------------------------------------------------------
    # transform
    # -------------------------------------------------------------------------

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted standardization + PCA to a panel.

        Returns the input panel with new PC columns appended:
            tier1_PC1, tier2_PC1, etc.

        Rows where any input feature for a tier is NaN will have NaN for
        that tier's PC. Rows where ALL inputs are present get a real PC value.

        Decision 3 / Sub-decision E: input columns are preserved. PCs are
        appended.

        Args:
            panel: feature panel to transform.

        Returns:
            Same panel + new PC columns. Index unchanged.
        """
        if not self._fitted:
            raise RuntimeError(
                "PerTierPCA must be fit() before transform(). Call fit() "
                "on training-fold data first."
            )

        out = panel.copy()
        for tier, feats in self.tier_features.items():
            if tier not in self.pca_models:
                continue   # tier didn't get a PCA (e.g. too few complete rows)
            # Subset to tier features that exist in this panel
            avail = [f for f in feats if f in panel.columns]
            if len(avail) != len(feats):
                # Panel is missing some features the PCA expects — can't
                # transform this tier. (Could happen if test panel uses a
                # different feature_subset than train.) Skip with NaN PC.
                tier_label = TIER_SHORT.get(tier, f"tier{tier}")
                out[f"{tier_label}_PC1"] = float("nan")
                continue

            # Identify rows where all tier features are present.
            # Doing this BEFORE z-scoring avoids overflow warnings on rows
            # we'd discard anyway (e.g. NaN values in z-score arithmetic
            # propagating through PCA's matmul).
            raw_sub = panel[feats]
            mask_complete = raw_sub.notna().all(axis=1)

            tier_label = TIER_SHORT.get(tier, f"tier{tier}")
            pc_col = f"{tier_label}_PC1"
            out[pc_col] = float("nan")

            if not mask_complete.any():
                continue

            # Z-score only the complete rows (using train mean/std)
            sub_complete = raw_sub.loc[mask_complete].copy()
            for f in feats:
                sub_complete[f] = (sub_complete[f] - self.train_mean[f]) / self.train_std[f]

            # Suppress non-fatal numerical warnings during PCA.transform.
            # These can occur when train fold has very few complete rows
            # (e.g., 31 rows × 4 features) and z-score values become extreme
            # for some test rows. Output values are still correct; the
            # warnings are noise.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                pcs = self.pca_models[tier].transform(sub_complete.values)
            # Apply sign convention
            pcs = pcs * self.pc_signs[tier]
            out.loc[mask_complete, pc_col] = pcs[:, 0]

        return out

    # -------------------------------------------------------------------------
    # convenience
    # -------------------------------------------------------------------------

    def fit_transform(
        self,
        panel: pd.DataFrame,
        registry: dict[str, int],
        sign_anchor: Optional[dict[int, str]] = None,
    ) -> pd.DataFrame:
        """Fit on panel, then transform panel. Use only when there's no
        train/test split (e.g., final live inference)."""
        self.fit(panel, registry, sign_anchor=sign_anchor)
        return self.transform(panel)

    # -------------------------------------------------------------------------
    # diagnostics
    # -------------------------------------------------------------------------

    def explained_variance_summary(self) -> pd.DataFrame:
        """Return a DataFrame of variance explained by PC1 per tier.

        Useful for debugging — if a tier's PC1 captures only 30%, that's
        a flag that one PC isn't enough (or that the tier features aren't
        actually correlated, in which case PCA isn't helpful at all).
        """
        if not self._fitted:
            raise RuntimeError("Must fit() before requesting diagnostics.")
        rows = []
        for tier, feats in self.tier_features.items():
            label = TIER_SHORT.get(tier, f"tier{tier}")
            if tier not in self.pca_models:
                rows.append({
                    "tier":    tier,
                    "label":   label,
                    "n_feats": len(feats),
                    "var_pc1": None,
                    "note":    "skipped: too few complete rows",
                })
                continue
            ev = float(self.pca_models[tier].explained_variance_ratio_[0])
            rows.append({
                "tier":    tier,
                "label":   label,
                "n_feats": len(feats),
                "var_pc1": ev,
                "note":    "ok",
            })
        return pd.DataFrame(rows)


# =============================================================================
# Convenience: load tier mapping from DB
# =============================================================================

def load_registry_tiers(db_path) -> dict[str, int]:
    """Load {feature_name -> tier_int} from features_registry.

    Args:
        db_path: path to recession.db.

    Returns:
        Dict for use with PerTierPCA.fit().
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            """SELECT feature_name, tier
               FROM features_registry
               WHERE is_active = 1"""
        )
        return {name: int(tier) for name, tier in cur.fetchall()}
    finally:
        conn.close()
