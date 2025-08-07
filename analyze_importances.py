# analyze_importances.py
# Updated to include pandemic dummy feature in feature importance analysis

import os
import glob
import pickle
import pandas as pd
import numpy as np

# -------------------- Settings --------------------
MODEL_DIR = "models"
FEATURE_IMPORTANCES_CSV = "feature_importances_summary.csv"

# -------------------- Feature Names --------------------
# Must match order in training pipeline, including new pandemic dummy
FEATURE_NAMES = [
    "close", "volume",
    "return_1d", "return_3d", "return_5d",
    "ma_5", "ma_10", "ma_20",
    "volatility_5d", "volatility_10d",
    "rsi_14", "macd", "macd_signal",
    "bollinger_upper", "bollinger_lower", "bollinger_width",
    "volume_zscore", "volume_spike",
    "sentiment_score",
    "is_pandemic",              # new regime indicator
    "insider_net_shares", "insider_7d", "insider_21d"
]

# -------------------- Load Models & Extract Importances --------------------
def load_feature_importances(model_dir):
    model_paths = sorted(glob.glob(os.path.join(model_dir, "*.pkl")))
    if not model_paths:
        raise SystemExit(f"No model files found in '{model_dir}'")

    records = []
    expected = len(FEATURE_NAMES)

    for path in model_paths:
        model_name = os.path.basename(path).replace(".pkl", "")
        with open(path, "rb") as f:
            model = pickle.load(f)
        imp = list(model.feature_importances_)  # cast to list for easier insertion

        # if this model is missing the pandemic feature, pad a zero
        if len(imp) == expected - 1:
            idx = FEATURE_NAMES.index("is_pandemic")
            imp.insert(idx, 0.0)

        imp = np.array(imp)  # use numpy array here
        if len(imp) != expected:
            raise ValueError(
                f"Feature length mismatch for {model_name}: expected {expected}, got {len(imp)}"
            )

        records.append(pd.Series(imp, index=FEATURE_NAMES, name=model_name))

    return pd.DataFrame(records)

# -------------------- Main --------------------
if __name__ == "__main__":
    # 1. Load importances
    df_importances = load_feature_importances(MODEL_DIR)

    # 2. Display to console
    pd.set_option("display.max_columns", None)
    print("\n=== Feature Importances Summary ===")
    print(df_importances)

    # 3. Save to CSV
    df_importances.to_csv(FEATURE_IMPORTANCES_CSV)
    print(f"\n✅ Saved feature importances to {FEATURE_IMPORTANCES_CSV}")
