"""
analysis/momentum_purged_wf.py — STEP 2b VALIDATION: momentum through the strict
purged walk-forward harness (the production gate).

Momentum survived the clean non-overlap + regime gate (mom_6_1 +1.05, mom_12_1
+0.83, positive every regime). This is the STRICT test: reuse walk_forward's
purged_kfold_indices (true walk-forward, train-on-past-only, 5d embargo) to
evaluate the momentum signal on sequential embargoed out-of-sample folds, and
report PER-FOLD consistency — not just the average. A robust signal is positive
in most folds; a fragile one is carried by one.

Builds the date x ticker close panel via features.builder._download (full hist),
forms the momentum signal, then for each purged test fold computes the net-of-cost
quintile spread over NON-overlapping 20d-rebalance dates within that fold.
GATE: pooled net Sh > 0.3 AND positive in >=4 of 5 folds.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features.builder import _download
from analysis.walk_forward import purged_kfold_indices, EMBARGO_DAYS, N_FOLDS

COST_BPS = 10.0


def build_close_panel(tickers, start):
    closes = {}
    for i, tk in enumerate(tickers, 1):
        if i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] {tk}", flush=True)
        try:
            closes[tk] = _download(tk, start, None).set_index("date")["close"]
        except Exception as e:
            print(f"  skip {tk}: {type(e).__name__}", flush=True)
    panel = pd.DataFrame(closes).sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel


def momentum(panel, kind):
    if kind == "mom_6_1":
        return panel.pct_change(126, fill_method=None) - panel.pct_change(21, fill_method=None)
    return panel.pct_change(252, fill_method=None) - panel.pct_change(21, fill_method=None)


def fold_spread(sig, ret_fwd, dates_in_fold, step_dates, winsor=0.40):
    """Net-of-cost quintile spread over the rebalance dates that fall in this fold."""
    use = [d for d in step_dates if d in dates_in_fold]
    spreads = []
    for d in use:
        row = sig.loc[d].dropna()
        if len(row) < 10:
            continue
        fr = ret_fwd.loc[d]
        order = row.sort_values()
        k = max(1, len(order)//5)
        lr = np.nanmean([fr.get(t, np.nan) for t in order.tail(k).index])
        sr = np.nanmean([fr.get(t, np.nan) for t in order.head(k).index])
        if not (np.isnan(lr) or np.isnan(sr)):
            spreads.append(lr - sr)
    if len(spreads) < 2:
        return None
    sa = np.array(spreads); sd = sa.std()
    cost = 1.0*(COST_BPS/1e4)*2.0
    pers = 252/20
    net = ((sa.mean()-cost)/sd*np.sqrt(pers)) if sd>0 else float("nan")
    return round(float(net),3), len(spreads), round(float(sa.mean())*100,3)


def run(panel, kind, fwd=20, step=20):
    sig = momentum(panel, kind)
    ret_fwd = (panel.shift(-fwd)/panel - 1.0).clip(-0.40, 0.40)
    # rebalance dates: every 20th trading day where signal is warm
    valid = panel.index[252 : len(panel)-fwd]
    step_dates = list(valid[::step])
    # purged folds over the FULL DAILY timeline (1862 days >> classifier row mins),
    # then evaluate the signal on the rebalance dates that fall in each OOS test fold.
    # Momentum is a fixed-rule signal (no model fit) so daily-granularity embargo is
    # the correct purged-WF here; rebalance-date granularity (80) fails MIN_TEST_ROWS=50.
    daily = pd.Series(list(valid))
    fold_nets = []
    print(f"\n  {kind}: purged WF ({N_FOLDS} folds, {EMBARGO_DAYS}d embargo, daily-granularity)")
    print(f"  {'fold':<6}{'test dates':<26}{'net Sh':>9}{'n':>5}{'mean%':>9}")
    print("  " + "-"*56)
    for fi, (tr, te) in enumerate(purged_kfold_indices(daily, n_folds=N_FOLDS, embargo=EMBARGO_DAYS)):
        fold_dates = set(daily.iloc[te].tolist())
        r = fold_spread(sig, ret_fwd, fold_dates, step_dates)
        if r:
            net, n, mp = r
            lo, hi = daily.iloc[te].min(), daily.iloc[te].max()
            print(f"  {fi:<6}{str(lo.date())+'..'+str(hi.date()):<26}{net:>+9.3f}{n:>5d}{mp:>+9.3f}")
            fold_nets.append(net)
    if fold_nets:
        pos = sum(1 for x in fold_nets if x > 0)
        print(f"  POOLED: mean fold net Sh {np.mean(fold_nets):+.3f}, "
              f"positive {pos}/{len(fold_nets)} folds")
        gate = np.mean(fold_nets) > 0.3 and pos >= max(1, int(0.8*len(fold_nets)))
        print(f"  GATE (mean>0.3 AND >=80% folds positive): {'PASS' if gate else 'FAIL'}")
    return fold_nets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", default="tickers.txt")
    ap.add_argument("--start", default="2019-01-01")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in (ROOT/args.tickers_file).read_text().splitlines()
               if t.strip() and not t.startswith("#")]
    print(f"=== MOMENTUM PURGED-WF VALIDATION: {len(tickers)} names ===")
    panel = build_close_panel(tickers, args.start)
    print(f"  close panel: {panel.shape[0]} days x {panel.shape[1]} names")
    for kind in ["mom_6_1", "mom_12_1"]:
        run(panel, kind)
    print("\n  Per-fold consistency is the real test: an edge in 4-5/5 embargoed")
    print("  out-of-sample folds is robust; one carrying fold = fragile.")


if __name__ == "__main__":
    main()
