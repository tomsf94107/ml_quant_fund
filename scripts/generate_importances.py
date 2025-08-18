#!/usr/bin/env python3
# scripts/generate_importances.py

import argparse
import glob
import os
import sys
import pickle
from collections import defaultdict

# Headless plotting for CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None


def _safe_load(path):
    """Load a pickled model via joblib if available, else pickle."""
    try:
        if joblib is not None:
            return joblib.load(path)
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[skip] failed to load {path}: {e}")
        return None


def _unwrap_estimator(model):
    """
    If model is a Pipeline or has a .best_estimator_, unwrap to the final estimator.
    Otherwise return model as-is.
    """
    try:
        from sklearn.pipeline import Pipeline  # noqa
        if hasattr(model, "best_estimator_") and model.best_estimator_ is not None:
            model = model.best_estimator_
        if hasattr(model, "steps"):  # pipeline-like
            steps = getattr(model, "steps", []) or []
            if steps:
                model = steps[-1][1]
    except Exception:
        # If sklearn not present or other issue, just return original
        pass
    return model


def _get_feature_importances(model):
    """
    Try to extract a dict {feature_name: importance} from various model types.
    Returns {} if not available.
    """
    if model is None:
        return {}

    model = _unwrap_estimator(model)

    # 1) Native sklearn/xgboost/sklearn-like models exposing feature_importances_
    fi = getattr(model, "feature_importances_", None)
    if fi is not None:
        fi = np.asarray(fi, dtype=float)

        # Avoid ambiguous truthiness on numpy arrays:
        names = getattr(model, "feature_names_in_", None)
        try:
            names = list(names) if names is not None else None
        except Exception:
            names = None

        if names is None or len(names) != len(fi):
            names = [f"f{i}" for i in range(len(fi))]

        return {str(n): float(v) for n, v in zip(names, fi)}

    # 2) XGBoost Booster (gain-based importances)
    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None:
            score = booster.get_score(importance_type="gain") or {}
            return {str(k): float(v) for k, v in score.items()}
    except Exception as e:
        print(f"[warn] xgboost get_score failed: {e}")

    # 3) LightGBM (feature_importances_ with feature_name)
    try:
        if hasattr(model, "feature_importances_"):
            fi = np.asarray(model.feature_importances_, dtype=float)
            try:
                names = list(model.feature_name())
            except Exception:
                names = None
            if names is None or len(names) != len(fi):
                names = [f"f{i}" for i in range(len(fi))]
            return {str(n): float(v) for n, v in zip(names, fi)}
    except Exception:
        pass

    return {}


def main():
    ap = argparse.ArgumentParser(description="Aggregate and plot feature importances from pickled models.")
    ap.add_argument("--models_glob", default="models/*.pkl", help="Glob pattern for model files")
    ap.add_argument("--out", default="charts/feature_importances.png", help="Output chart path (PNG)")
    ap.add_argument("--csv", default="charts/feature_importances.csv", help="Output CSV summary path")
    ap.add_argument("--top", type=int, default=25, help="Top-N features to plot")
    args = ap.parse_args()

    files = sorted(glob.glob(args.models_glob))
    if not files:
        print(f"[info] No models matched {args.models_glob}. Exiting 0.")
        return

    agg = defaultdict(float)
    nmodels = 0

    for p in files:
        model = _safe_load(p)
        if model is None:
            continue

        imp = _get_feature_importances(model)
        if not imp:
            print(f"[info] {os.path.basename(p)} has no usable importances; skipping.")
            continue

        nmodels += 1
        total = float(sum(imp.values())) or 1.0  # normalize per model
        for k, v in imp.items():
            agg[k] += v / total

    if nmodels == 0:
        print("[info] No usable importances found across models. Exiting 0.")
        return

    # Average across models
    for k in list(agg.keys()):
        agg[k] /= nmodels

    # Ensure output dirs
    for path in (args.out, args.csv):
        out_dir = os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)

    # Build DataFrame
    df = (
        pd.DataFrame({"feature": list(agg.keys()), "importance": list(agg.values())})
        .sort_values("importance", ascending=False)
    )
    top = df.head(max(1, int(args.top)))

    # Write CSV
    top.to_csv(args.csv, index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.title(f"Average Feature Importance (across {nmodels} models)")
    plt.xlabel("Normalized importance")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    print(f"[done] wrote: {args.out} and {args.csv} (models: {nmodels}, features: {len(df)})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
