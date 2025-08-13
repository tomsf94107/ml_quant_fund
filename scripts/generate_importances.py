#!/usr/bin/env python3
import argparse, glob, os, pickle
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

def load_importances(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"[skip] {path}: {e}")
        return {}
    if hasattr(model, "feature_importances_"):
        names = getattr(model, "feature_names_in_", None) or [f"f{i}" for i in range(len(model.feature_importances_))]
        return {str(n): float(v) for n, v in zip(names, model.feature_importances_)}
    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster:
            score = booster.get_score(importance_type="gain")
            return {str(k): float(v) for k, v in score.items()}
    except Exception as e:
        print(f"[warn] xgboost get_score failed for {path}: {e}")
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
        print(f"No models matched {args.models_glob}. Exiting 0.")
        return

    agg = defaultdict(float); nmodels = 0
    for p in files:
        imp = load_importances(p)
        if not imp: continue
        nmodels += 1
        total = sum(imp.values()) or 1.0
        for k, v in imp.items():
            agg[k] += v / total  # normalize per model

    if nmodels == 0:
        print("No usable importances found. Exiting 0.")
        return

    for k in list(agg): agg[k] /= nmodels

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    df = (pd.DataFrame({"feature": list(agg.keys()), "importance": list(agg.values())})
          .sort_values("importance", ascending=False)
          .head(args.top))
    df.to_csv(args.csv, index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.title(f"Average Feature Importance (across {nmodels} models)")
    plt.xlabel("Normalized importance")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Wrote: {args.out} and {args.csv}")

if __name__ == "__main__":
    main()

