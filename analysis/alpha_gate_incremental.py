"""
analysis/alpha_gate_incremental.py
Final P3.2 step: of gated survivors, which TRANSFORMS beat the RAW base the
model already uses? Compares each survivor's |IC| to its base cs_rank |IC|.
WARNING: judge by ABSOLUTE uplift, not uplift_% (tiny raw_IC denominators
produce absurd percentages, e.g. 31000%).
"""
import sys
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.classifier import FEATURE_COLUMNS

res = pd.read_csv(ROOT / "analysis/alpha_gate_results_h5.csv")
res["base"] = res["feature"].str.split("__").str[0]
model_feats = set(FEATURE_COLUMNS)
surv = res[res["SURVIVOR"]].copy()
raw_ic = {}
for base in surv["base"].unique():
    rows = res[(res["base"] == base) & (res["feature"].str.endswith("__cs_rank"))]
    if len(rows):
        raw_ic[base] = rows["mean_IC"].abs().iloc[0]
out = []
for _, r in surv.iterrows():
    base = r["base"]; raw = raw_ic.get(base, np.nan)
    tf = abs(r["mean_IC"]); up = tf - raw if not np.isnan(raw) else np.nan
    pct = (up / raw * 100) if (raw and not np.isnan(raw) and raw > 0) else np.nan
    out.append((r["feature"], base, base in model_feats, tf, raw, up, pct, r["t_stat"]))
df = pd.DataFrame(out, columns=["feature","base","base_in_model","tf_IC","raw_IC",
                                "uplift","uplift_pct","t_stat"]).sort_values("uplift", ascending=False)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(df.to_string(index=False))
real = df[(df["raw_IC"] >= 0.005) & (df["uplift"] >= 0.012)]
print("\n--- DEFENSIBLE additions (raw_IC>=0.005 AND absolute uplift>=0.012) ---")
print(real[["feature","base_in_model","tf_IC","raw_IC","uplift"]].to_string(index=False) if len(real) else "  none")
print("\nSUMMARY: %d survivors | %d base-not-in-model | %d defensible additions"
      % (len(surv), (~df["base_in_model"]).sum(), len(real)))
