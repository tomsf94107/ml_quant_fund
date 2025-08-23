# evaluate_importances_over_time.py

import glob
import os
import sys
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────
# Locations
MODELS_DIR       = "models"
TIMESTAMPED_GLOB = os.path.join(MODELS_DIR, "feature_importances_summary_*.csv")
ROOT_SUMMARY     = "feature_importances_summary.csv"
OUT_DIR          = "charts"
OUT_PNG          = os.path.join(OUT_DIR, "importances_over_time.png")

# Pretty labels (covers new + legacy names)
PRETTY = {
    # Insider (calculated)
    "ins_net_shares_7d":        "Insider Net (7d, calc)",
    "ins_net_shares_30d":       "Insider Net (30d, calc)",
    "ins_pressure_30d_z":       "Insider Pressure z (30d)",
    "ins_large_or_exec_7d":     "Exec/Large Days (7d)",
    # Insider (DB rollups from insider_flows)
    "ins_net_shares_7d_db":     "Insider Net (7d, DB)",
    "ins_net_shares_21d_db":    "Insider Net (21d, DB)",
    # Legacy aliases that might appear in older CSVs
    "insider_net_shares":       "Insider Net (legacy)",
    "insider_7d":               "Insider Net (7d, legacy)",
    "insider_21d":              "Insider Net (21d, legacy)",
    # Common market features
    "return_1d":                "Return (1d)",
    "ma_10":                    "Moving Avg (10)",
    "volatility_5d":            "Volatility (5d)",
}

# Priority list to try to plot (we'll filter to those present)
PRIORITY_FEATURES = [
    # Insider first
    "ins_pressure_30d_z",
    "ins_large_or_exec_7d",
    "ins_net_shares_7d",
    "ins_net_shares_30d",
    "ins_net_shares_7d_db",
    "ins_net_shares_21d_db",
    # Legacy fallbacks (if older runs)
    "insider_net_shares",
    "insider_7d",
    "insider_21d",
    # A few common technicals (optional)
    "return_1d",
    "ma_10",
    "volatility_5d",
]

TOP_N_FALLBACK = 6   # how many features to plot if none of the priority ones are present
ROLL_WIN       = 7   # rolling window (days across retrain snapshots)

# ────────────────────────────────────────────────────────────────────────────
# 1) Find files
paths = sorted(glob.glob(TIMESTAMPED_GLOB))

# 2) Fallback to root summary if no timestamped files
if not paths and os.path.exists(ROOT_SUMMARY):
    print(f"No timestamped files under {MODELS_DIR}; using root summary {ROOT_SUMMARY}")
    paths = [ROOT_SUMMARY]

if not paths:
    sys.exit("❌ No feature-importances CSVs found. Run analyze_importances.py first.")

# ────────────────────────────────────────────────────────────────────────────
# 3) Load each summary and convert to long form with a date
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

    df = pd.read_csv(path, index_col=0)  # rows=models, cols=features
    if df.empty:
        continue

    # melt to long: columns -> 'feature', values -> 'importance'
    long = (
        df.reset_index()
          .melt(id_vars=["index"], var_name="feature", value_name="importance")
          .rename(columns={"index": "model"})
    )
    long["date"] = date
    records.append(long)

if not records:
    sys.exit("❌ No valid data in feature importance CSVs.")

# ────────────────────────────────────────────────────────────────────────────
# 4) Concatenate and pivot to date x feature (mean across models)
all_imp = pd.concat(records, ignore_index=True)
pivot   = all_imp.pivot_table(index="date", columns="feature", values="importance", aggfunc="mean")
pivot   = pivot.sort_index()

# ────────────────────────────────────────────────────────────────────────────
# 5) Choose features to plot (only ones that exist)
available = [f for f in PRIORITY_FEATURES if f in pivot.columns]

if not available:
    # Fallback: pick the TOP_N_FALLBACK features with highest latest importance
    last_row = pivot.iloc[-1].dropna()
    if last_row.empty:
        sys.exit("❌ No non-NaN importances to plot.")
    available = last_row.sort_values(ascending=False).head(TOP_N_FALLBACK).index.tolist()
    print(f"ℹ️ Using fallback top-{TOP_N_FALLBACK} features:", available)
else:
    print("ℹ️ Plotting features:", available)

# Restrict pivot to selected features
sub = pivot[available].copy()

# 6) Rolling average
rolling = sub.rolling(window=ROLL_WIN, min_periods=1).mean()

# 7) Plot
plt.close("all")
fig, ax = plt.subplots(figsize=(12, 6))
# Rename columns to pretty labels for the legend
rolling_pretty = rolling.rename(columns=lambda c: PRETTY.get(c, c))
rolling_pretty.plot(ax=ax)

ax.set_title(f"{ROLL_WIN}-Day Rolling Feature Importances")
ax.set_ylabel("Avg. Importance")
ax.set_xlabel("Retrain Date")
ax.legend(loc="upper right", ncol=1, frameon=True)
fig.tight_layout()

# 8) Save to PNG for dashboard
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(OUT_PNG, bbox_inches="tight")
print(f"✅ Saved chart to {OUT_PNG}")

# 9) (Optional) Show interactively
# plt.show()
