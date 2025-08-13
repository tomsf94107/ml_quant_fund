# scripts/generate_importances.py
import argparse, glob, os, pickle, math
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_importances_from_model(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"[skip] failed to load {path}: {e}")
        return {}

    # Try scikit-learn style
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        # Try to get names; fall back to f0..fn
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            names = [f"f{i}" for i in range(len(imp))]
        return {str(n): float(v) for n, v in zip(names, imp)}

    # Try XGBoost style
    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster:
            # gain-based importances (more stable)
            score = booster.get_score(importance_type="gain")
            # keys can be "f0", "f1", ...; convert values to float
            return {k: float(v) for k, v in score.items()}
    except Exception as e:
        print(f"[warn] xgboost get_score failed for {path}: {e}")

    print(f"[skip] no importances found in {path}")
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_glob", default="models/*.pkl", help="Glob for model files")
    ap.add_argument("--out", default="charts/feature_importances.png", help="Output chart path")
    ap.add_argument("--csv", default="charts/feature_importances.csv", help="Also write CSV")
    ap.add_argument("--top", type=int, default=25, help="Top-N features")
    args = ap.parse_args()

    files = sorted(glob.glob(args.models_glob))
    if not files:
        print(f"No model files matched {args.models_glob}. Nothing to do.")
        # Exit 0 so CI doesn’t fail when models aren’t present
        return

    agg = defaultdict(float)
    n_models = 0
    for p in files:
        imp = load_importances_from_model(p)
        if imp:
            n_models += 1
            # normalize per model to avoid bias from model scale
            total = sum(imp.values()) or 1.0
            for k, v in imp.items():
                agg[k] += (v / total)

    if n_models == 0:
        print("Loaded 0 models with usable importances. Exiting.")
        return

    # average across models
    for k in list(agg.keys()):
        agg[k] /= n_models

    df = (
        pd.DataFrame({"feature": list(agg.keys()), "importance": list(agg.values())})
        .sort_values("importance", ascending=False)
        .head(args.top)
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df.to_csv(args.csv, index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.title(f"Average Feature Importance (across {n_models} models)")
    plt.xlabel("Normalized importance")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Wrote: {args.out} and {args.csv}")

if __name__ == "__main__":
    main()
