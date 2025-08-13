#!/usr/bin/env python3
import argparse, glob, os, pickle
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless CI
import matplotlib.pyplot as plt

def _to_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    if hasattr(x, "tolist"):
        return list(x.tolist())
    return [str(x)]

def load_importances(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"[skip] failed to load {path}: {e}")
        return {}

    # scikit-learn / xgboost-sklearn style
    if hasattr(model, "feature_importances_"):
        names = _to_list(getattr(model, "feature_names_in_", None))
        if not names:
            names = [f"f{i}" for i in range(len(model.feature_importances_))]
        return {str(n): float(v) for n, v in zip(names, model.feature_importances_)}

    # native xgboost booster
    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster:
            score = booster.get_score(importance_type="gain")
            return {str(k): float(v) for k, v in score.items()}
    except Exception as e:
        print(f"[warn] xgboost get_score failed for {path}: {e}")

    print(f"[skip] no importances found in {path}")
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_glob", default="models/*.pkl")
    ap.add_argument("--out", default="charts/feature_importances.png")
    ap.add_argument("--csv", default="charts/feature_importances.csv")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    files = sorted(glob.glob(args.models_glob))
    if not files:
        print(f"No model files matched {args.models_glob}. Nothing to do.")
        return

    agg = defaultdict(float)
    n_models = 0
    for p in files:
        imp = load_importances(p)
        if not imp:
            continue
        n_models += 1
        total = sum(imp.values()) or 1.0
        for k, v in imp.items():
            agg[k] += v / total  # normalize per model

    if n_models == 0:
        print("Loaded 0 models with usable importances. Exiting.")
        return

    # average across models
    for k in list(agg.keys()):
        agg[k] /= n_models

    df = (pd.DataFrame({"feature": list(agg.keys()), "importance": list(agg.values())})
            .sort_values("importance", ascending=False)
            .head(args.top))

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
