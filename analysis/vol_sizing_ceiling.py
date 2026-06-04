import pandas as pd
import numpy as np
import csv
from pathlib import Path
from analysis.walk_forward import purged_kfold_indices, EMBARGO_DAYS, N_FOLDS
from analysis.momentum_purged_wf import build_close_panel, momentum, COST_BPS

ROOT = Path("/Users/atomnguyen/Desktop/ML_Quant_Fund")
DECILE, FWD, STEP, VOL_LB = 0.10, 20, 20, 40

def fold_sized(sig, ret_fwd, daily_ret, fwd_vol, fold_dates, step_dates, mode):
    use = [d for d in step_dates if d in fold_dates]
    rets = []
    for d in use:
        row = sig.loc[d].dropna()
        if len(row) < 10:
            continue
        k = max(1, int(len(row) * DECILE))
        picks = list(row.sort_values(ascending=False).head(k).index)
        fr = ret_fwd.loc[d]
        w = {}
        if mode == "equal":
            w = {t: 1.0 for t in picks}
        elif mode == "naive":
            for t in picks:
                tr = daily_ret[t].loc[:d].tail(VOL_LB) if t in daily_ret.columns else None
                v = tr.std() if (tr is not None and tr.notna().sum() >= 10) else np.nan
                w[t] = (1.0 / v) if (v and not np.isnan(v) and v > 0) else np.nan
        else:
            for t in picks:
                fv = fwd_vol.loc[d, t] if (d in fwd_vol.index and t in fwd_vol.columns) else np.nan
                w[t] = (1.0 / fv) if (fv and not np.isnan(fv) and fv > 0) else np.nan
        ws = pd.Series(w, dtype="float64")
        if ws.notna().any():
            ws = ws.fillna(ws.median())
            ws = ws / ws.sum()
        else:
            continue
        port = np.nansum([ws.get(t, 0) * fr.get(t, np.nan) for t in picks])
        if not np.isnan(port):
            rets.append(port)
    if len(rets) < 2:
        return None
    sa = np.array(rets)
    sd = sa.std()
    cost = 1.0 * (COST_BPS / 1e4) * 2.0
    net = ((sa.mean() - cost) / sd * np.sqrt(252 / 20)) if sd > 0 else float("nan")
    return round(float(net), 3)

def run(panel, kind):
    sig = momentum(panel, kind)
    ret_fwd = (panel.shift(-FWD) / panel - 1.0).clip(-0.40, 0.40)
    daily_ret = panel.pct_change()
    fwd_vol = daily_ret[::-1].rolling(FWD).std()[::-1].shift(-1)
    valid = panel.index[252:len(panel) - FWD]
    step_dates = list(valid[::STEP])
    daily = pd.Series(list(valid))
    res = {}
    for mode in ("equal", "naive", "oracle"):
        nets = []
        for fi, (tr, te) in enumerate(purged_kfold_indices(daily, n_folds=N_FOLDS, embargo=EMBARGO_DAYS)):
            fd = set(daily.iloc[te].tolist())
            r = fold_sized(sig, ret_fwd, daily_ret, fwd_vol, fd, step_dates, mode)
            if r is not None:
                nets.append(r)
        res[mode] = (round(float(np.mean(nets)), 3) if nets else None, nets)
    return res

if __name__ == "__main__":
    tk = [r["ticker"] for r in csv.DictReader(open(ROOT / "tickers_metadata.csv"))]
    panel = build_close_panel(tk, "2018-01-01")
    print("panel: %d x %d" % (panel.shape[1], panel.shape[0]))
    for kind in ("mom_6_1", "mom_12_1"):
        res = run(panel, kind)
        eq, nv, orc = res["equal"][0], res["naive"][0], res["oracle"][0]
        print("=== %s : sizing net Sharpe (purged WF) ===" % kind)
        print("  equal-weight        %+.3f" % eq)
        print("  naive trailing-vol  %+.3f  (CURRENT LIVE)" % nv)
        print("  PERFECT future-vol  %+.3f  (ORACLE ceiling)" % orc)
        if nv is not None and orc is not None:
            print("  --> room a forecaster could fill: %+.3f Sharpe" % (orc - nv))
            if (orc - eq) != 0:
                print("  --> naive captures %d%% of equal->oracle gain" % (100*(nv-eq)/(orc-eq)))
    print("READ: room < ~0.1 -> DONT build. large room AND naive <60%% -> MAY build.")
