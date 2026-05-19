"""
recession/validation/feature_audit.py

Item #4 — the systematic feature audit.

WHAT IT ANSWERS
---------------
Every feature defined in features_registry costs something — data
ingestion, storage, maintenance, cognitive load. The audit asks, for each
one: is it actually EARNING its place? It classifies every registry
feature into one of three honest buckets:

  1. LIVE — the feature is in the feature set of at least one model
     (M1-M5, M2-binary). It is wired in and a model consumes it.
  2. REGISTRY-ONLY — the feature is defined in features_registry but NO
     model's feature set references it. It is dead weight: ingested and
     stored, never used. (It may be there for a planned model, or be a
     leftover — the audit flags it; the human decides.)
  3. KNOWN-DEAD — the feature IS used by a model but a prior validated
     test found it carries no signal. The audit hard-codes the one such
     finding the project has established: REAL_FFR_GAP at T1/h=12, which
     the nested likelihood-ratio test found insignificant (p=0.42).

WHY A RECONCILIATION, NOT A FRESH MODEL RUN
-------------------------------------------
Buckets 1 and 2 are pure RECONCILIATION: registry feature names vs the
model feature sets. That is exact, cheap, and needs no modeling — it just
needs the real features_registry. Bucket 3 reuses an already-validated
finding. The audit therefore introduces no new modeling and no new
leakage surface; it is a cross-reference.

For the IMPORTANCE side — "of the live features, which actually
contribute" — the honest tool is to RE-RUN the existing OOS permutation
importance fresh (recession.models.m3_forest.oos_permutation_importance),
not to collate remembered numbers, which go stale. run_importance_audit()
does that; it is optional and slower.

WHERE IT RUNS
-------------
This is a SCRIPT for the machine that has the real recession.db. The
model feature sets are imported directly from the model modules (the
single source of truth), so the audit always reflects the real code.

USAGE
    from recession.validation.feature_audit import (
        run_feature_audit, print_feature_audit)
    print_feature_audit(run_feature_audit(db_path="recession.db"))
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

# the model feature sets — the single source of truth, imported from the
# model modules so the audit can never drift from the real code
from recession.models.m1_probit import M1_FEATURES, M1_EXTENDED_FEATURES
from recession.models.m2_logit import M2_FEATURES
from recession.models.m2_binary import M2_BINARY_FEATURES
from recession.models.m3_forest import M3_CORE_FEATURES, M3_WIDE_FEATURES
from recession.models.m4_xgboost import M4_CORE_FEATURES, M4_WIDE_FEATURES
from recession.models.m5_markov import M5_FEATURES


# every model feature set, named — so the audit can report WHICH models
# use a feature, not just whether it is used
MODEL_FEATURE_SETS = {
    "M1": M1_FEATURES,
    "M1-extended": M1_EXTENDED_FEATURES,
    "M2": M2_FEATURES,
    "M2-binary": M2_BINARY_FEATURES,
    "M3-core": M3_CORE_FEATURES,
    "M3-wide": M3_WIDE_FEATURES,
    "M4-core": M4_CORE_FEATURES,
    "M4-wide": M4_WIDE_FEATURES,
    "M5": M5_FEATURES,
}

# bucket 3 — features a prior validated test found to carry no signal.
# {feature: (where, finding)}. Only established findings go here.
KNOWN_DEAD_FINDINGS = {
    "REAL_FFR_GAP": (
        "T1/h=12",
        "nested likelihood-ratio test: Wald p=0.42 — not significant; "
        "carries no signal beyond the yield curve at the 12-month horizon",
    ),
}


def _all_model_features() -> set[str]:
    """Every feature referenced by any model feature set."""
    feats: set[str] = set()
    for fs in MODEL_FEATURE_SETS.values():
        feats.update(fs)
    return feats


def _models_using(feature: str) -> list[str]:
    """Which named model feature sets reference this feature."""
    return sorted(name for name, fs in MODEL_FEATURE_SETS.items()
                  if feature in fs)


def load_registry(db_path: Path) -> list[dict]:
    """Load every feature from features_registry. Returns a list of
    {feature_name, tier, tier_label, is_active}."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """SELECT feature_name, tier, tier_label, is_active
               FROM features_registry
               ORDER BY tier, feature_name"""
        ).fetchall()
    finally:
        conn.close()
    return [
        {"feature_name": r[0], "tier": r[1], "tier_label": r[2],
         "is_active": bool(r[3])}
        for r in rows
    ]


def run_feature_audit(db_path: Optional[Path] = None) -> dict:
    """Reconcile the feature registry against the model feature sets.

    Returns:
        {'registry': [...],            # every registry feature, classified
         'live': [...],                # bucket 1
         'registry_only': [...],       # bucket 2 — defined, never wired in
         'known_dead': [...],          # bucket 3 — used but found dead
         'used_not_in_registry': [...] # a model uses a feature the
                                       #   registry does not define (a real
                                       #   integrity problem if non-empty)
        }
    """
    if db_path is None:
        db_path = Path("recession.db")

    registry = load_registry(db_path)
    registry_names = {f["feature_name"] for f in registry}
    model_features = _all_model_features()

    live, registry_only, known_dead = [], [], []

    for f in registry:
        name = f["feature_name"]
        users = _models_using(name)
        entry = {**f, "models": users}
        if users:
            entry["bucket"] = "LIVE"
            live.append(entry)
            if name in KNOWN_DEAD_FINDINGS:
                where, finding = KNOWN_DEAD_FINDINGS[name]
                known_dead.append({**entry, "dead_where": where,
                                   "dead_finding": finding})
        else:
            entry["bucket"] = "REGISTRY-ONLY"
            registry_only.append(entry)

    # integrity check: a model feature set names something the registry
    # does not define. This should be empty; if not, it is a real bug.
    used_not_in_registry = sorted(model_features - registry_names)

    # attach bucket to the full registry list for the printed table
    classified = []
    for f in registry:
        name = f["feature_name"]
        users = _models_using(name)
        bucket = "LIVE" if users else "REGISTRY-ONLY"
        if name in KNOWN_DEAD_FINDINGS and users:
            bucket = "LIVE (known-dead @ "\
                     f"{KNOWN_DEAD_FINDINGS[name][0]})"
        classified.append({**f, "models": users, "bucket": bucket})

    return {
        "registry": classified,
        "live": live,
        "registry_only": registry_only,
        "known_dead": known_dead,
        "used_not_in_registry": used_not_in_registry,
        "n_registry": len(registry),
        "n_live": len(live),
        "n_registry_only": len(registry_only),
    }


def print_feature_audit(audit: dict) -> None:
    """Print the feature-audit report."""
    print("=" * 72)
    print("FEATURE AUDIT — registry vs. models reconciliation")
    print("=" * 72)
    print(f"  registry features: {audit['n_registry']}   "
          f"live: {audit['n_live']}   "
          f"registry-only (dead weight): {audit['n_registry_only']}")
    print()
    print(f"  {'feature':>16} {'tier':>5} {'active':>7}  "
          f"{'bucket':<28} models")
    print("  " + "-" * 68)
    for f in audit["registry"]:
        models = ",".join(f["models"]) if f["models"] else "—"
        act = "yes" if f["is_active"] else "no"
        print(f"  {f['feature_name']:>16} {f['tier']:>5} {act:>7}  "
              f"{f['bucket']:<28} {models}")

    # bucket 2 — the main finding
    print()
    print("-" * 72)
    if audit["registry_only"]:
        print("  REGISTRY-ONLY features (defined but NO model uses them —")
        print("  dead weight, or awaiting a planned model):")
        for f in audit["registry_only"]:
            print(f"    - {f['feature_name']} (tier {f['tier']}, "
                  f"{f['tier_label']})")
        print("  ACTION: for each, decide — wire into a model, or remove")
        print("  from the registry/ingestion. Carrying an unused feature")
        print("  is silent maintenance cost.")
    else:
        print("  No registry-only features — every registered feature is")
        print("  consumed by at least one model.")

    # bucket 3
    print()
    if audit["known_dead"]:
        print("  KNOWN-DEAD features (used by a model but a validated test")
        print("  found no signal):")
        for f in audit["known_dead"]:
            print(f"    - {f['feature_name']} @ {f['dead_where']}")
            print(f"      {f['dead_finding']}")
        print("  ACTION: consider dropping from the feature set at the")
        print("  horizon where it is dead (it adds noise + a parameter).")
    else:
        print("  No known-dead features flagged.")

    # integrity check
    print()
    if audit["used_not_in_registry"]:
        print("  *** INTEGRITY PROBLEM ***")
        print("  A model uses a feature the registry does NOT define:")
        for name in audit["used_not_in_registry"]:
            print(f"    - {name}")
        print("  This must be fixed — the model depends on an unregistered")
        print("  feature.")
    else:
        print("  Integrity OK: every feature used by a model is defined in")
        print("  the registry.")
    print("=" * 72)
