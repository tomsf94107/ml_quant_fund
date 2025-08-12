# evaluate_importances_over_time.py

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Where we look
MODELS_DIR        = "models"
TIMESTAMPED_GLOB  = os.path.join(MODELS_DIR, "feature_importances_summary_*.csv")
ROOT_SUMMARY      = "feature_importances_summary.csv"

# 1) Gather files
paths = sorted(glob.glob(TIMESTAMPED_GLOB))

# 2) Fallback to root summary if no timestamped files
if not paths and os.path.exists(ROOT_SUMMARY):
    print(f"No timestamped files under {MODELS_DIR}; using root summary {ROOT_SUMMARY}")
    paths = [ROOT_SUMMARY]

if not paths:
    raise SystemExit("❌ No feature-importances CSVs found. Run analyze_importances.py first.")

# 3) Load each into a long form with an associated date
records = []
for path in paths:
    fn = os.path.basename(path)
    if fn.startswith("feature_importances_summary_"):
        date_str = fn.replace("feature_importances_summary_", "").replace(".csv", "")
    else:
        # root summary: use file mtime as date
        mtime    = os.path.getmtime(path)
        date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    date = datetime.strptime(date_str, "%Y-%m-%d")
    df   = pd.read_csv(path, index_col=0)

    long = (
        df.reset_index()
          .melt(id_vars=["index"], var_name="feature", value_name="importance")
          .rename(columns={"index": "model"})
    )
    long["date"] = date
    records.append(long)

# 4) Concatenate and pivot
all_imp = pd.concat(records, ignore_index=True)
pivot   = all_imp.pivot_table(index="date", columns="feature", values="importance", aggfunc="mean")

# 5) Rolling-average & Plot
features_to_plot = [
    "insider_net_shares", "insider_7d", "insider_21d",
    "return_1d", "ma_10", "volatility_5d"
]
rolling = pivot[features_to_plot].rolling(window=7, min_periods=1).mean()

plt.figure(figsize=(12, 6))
rolling.plot()
plt.title("7-Day Rolling Feature Importances")
plt.ylabel("Avg. Importance")
plt.xlabel("Retrain Date")
plt.legend(loc="upper right")
plt.tight_layout()

# 6) Save to PNG for dashboard
out_dir = "charts"
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "importances_over_time.png")
plt.savefig(png_path, bbox_inches="tight")
print(f"✅ Saved chart to {png_path}")

# 7) Display (optional)
plt.show()
