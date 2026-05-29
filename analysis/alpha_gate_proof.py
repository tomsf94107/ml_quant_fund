"""
analysis/alpha_gate_proof.py  (v2 — Massive-only)
PROOF-OF-CONCEPT for P3.2 gating: validate rank-IC math on 50 features
(h=5) before scaling to all 3,390.

RULE #1 NOTE: fwd returns come from mc.download (Massive, stocks) ONLY.
Do NOT call build_feature_dataframe here — it triggers live UW API calls
(stock-state, options-volume, short-interest, insider) which violate the
no-UW-outside-market-hours rule (cf. Apr 24 2026 9k-error incident). For
forward returns we need CLOSE prices only; Massive is unlimited + UW-free.

Rank-IC = per-date Spearman corr(feature across tickers, fwd_ret across
tickers), averaged. t = IC-IR * sqrt(n_dates). NOT yet DSR-corrected.
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features import massive_client as mc

H = 5
N_DATES = 60
N_FEATURES = 50

# 1. recent panel files
files = sorted(glob.glob(str(ROOT / "data/alpha_panel/*.parquet")))[-N_DATES:]
print(f"Panel dates: {Path(files[0]).stem} -> {Path(files[-1]).stem} ({len(files)})")

sample = pd.read_parquet(files[-1])
all_cols = list(sample.columns)
picks = []
for key in ("return_5d__cs_rank","return_1d__cs_rank","rsi_14__cs_rank",
            "day_of_week__cs_rank","is_month_end__cs_rank","macd__cs_zscore",
            "volume_zscore__cs_rank","short_ratio__cs_rank","obv_trend__ts_min__w5"):
    if key in all_cols: picks.append(key)
step = max(1, len(all_cols)//(N_FEATURES-len(picks)))
for c in all_cols[::step]:
    if c not in picks: picks.append(c)
    if len(picks) >= N_FEATURES: break
picks = picks[:N_FEATURES]

# 2. stack panel for picks
panel_by_date = {}
tickers_seen = set()
for f in files:
    d = pd.read_parquet(f, columns=picks)
    panel_by_date[pd.Timestamp(Path(f).stem)] = d
    tickers_seen |= set(d.index)
tickers = sorted(tickers_seen)
print(f"{len(tickers)} tickers, {len(picks)} features")

# 3. fwd_ret_5d via MASSIVE batch (UW-free, vectorized)
print("Fetching close via mc.download (Massive, no UW)...")
start = (min(panel_by_date) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
end   = (max(panel_by_date) + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
px = mc.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
if isinstance(px.columns, pd.MultiIndex):
    close = px["Close"]
else:  # single-ticker safety
    close = px[["Close"]]; close.columns = tickers[:1]
close.index = pd.to_datetime(close.index)
fwd = close.shift(-H) / close - 1.0     # date x ticker forward returns
print(f"close shape {close.shape}, fwd shape {fwd.shape}")

# 4. per-date rank-IC
records = []
for feat in picks:
    ics = []
    for dt, dframe in panel_by_date.items():
        if feat not in dframe.columns or dt not in fwd.index: continue
        fvals = dframe[feat]
        rets = fwd.loc[dt].reindex(fvals.index)
        j = pd.concat([fvals.rename("f"), rets.rename("r")], axis=1).dropna()
        if len(j) >= 10 and j["f"].std() > 0 and j["r"].std() > 0:
            ic = j["f"].corr(j["r"], method="spearman")
            if not np.isnan(ic): ics.append(ic)
    if len(ics) >= 20:
        a = np.array(ics); m, s = a.mean(), a.std()
        ir = m/s if s > 0 else np.nan
        t = ir*np.sqrt(len(a)) if not np.isnan(ir) else np.nan
        records.append((feat, len(a), m, s, ir, t))

res = pd.DataFrame(records, columns=["feature","n_dates","mean_IC","std_IC","IC_IR","t_stat"])
res = res.sort_values("t_stat", key=abs, ascending=False)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("\n"+"="*78)
print(f"RANK-IC PROOF v2 (Massive) — h={H}, {N_FEATURES} feats, {len(panel_by_date)} dates")
print("="*78)
print(res.to_string(index=False))
print("\n--- SANITY ---")
print(f"max |mean_IC|: {res['mean_IC'].abs().max():.4f} (suspicious if >0.3)")
print(f"median |mean_IC|: {res['mean_IC'].abs().median():.4f}")
ob = res[res['feature'].str.startswith('obv_trend')]
if len(ob): print(f"obv_trend IC (should be ~0): {ob['mean_IC'].values}")
dw = res[res['feature'].str.startswith('day_of_week')]
if len(dw): print(f"day_of_week IC (should be ~0): {dw['mean_IC'].values}")
print(f"raw |t|>3.0 (NOT DSR-corrected): {(res['t_stat'].abs()>3.0).sum()}")
