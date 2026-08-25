#!/usr/bin/env python3
"""
alpha_select.py -- score and gate alphas on a DATE RANGE, for nested holdout.

WHY (2026-08-22)
  c1_holdout.py held out the last 120 dates and fit combination WEIGHTS on train.
  But the four candidate alphas were selected by alpha_fitness scored across ALL
  637 dates -- including those 120. So:

      held out : the combination weights
      NOT held out : the alpha SELECTION

  Picking 4 from 2,403 is where nearly all the overfitting risk lives, and it saw
  the test window. The reported test Sharpes (vol 2.56, rev 2.64) are therefore
  contaminated. Only the FAIL verdict on COMBINATION is clean, because both sides
  of that comparison share the same leak.

  alpha_fitness has no --start/--end, so selection could not be restricted. This
  reuses its internals (_load_panel, _merge_outcomes, _score_one -- so the IC,
  t-stat, Sharpe, turnover and monotonicity are computed by the SAME code) and
  adds only a date filter.

NESTED DESIGN
    train dates  -> score all alphas, apply the gate, pick survivors
    test  dates  -> NEVER SEEN by selection; used once, by c1_holdout

USAGE
  # select on train only (everything before the holdout boundary)
  python scripts/alpha_select.py --horizon 5 --end 2026-02-22 --top 10

  # then feed the printed --alphas string straight into c1_holdout
  python scripts/c1_holdout.py --alphas "<paste>" --test-dates 120

GATE (same as the production gate, applied to the train window only)
  is_market_wide = 0  AND  |ic_t| > 3  AND  mono >= 0.30  AND  rank_ic > 0
  AND sharpe > 0, then one representative per BASE feature (transform-copies
  inflate the count ~8x and are near-duplicates of each other).
"""
import argparse
import os
import sys

ROOT = os.path.expanduser(os.environ.get("ML_QUANT_ROOT", "~/ML_Quant_Fund"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", default=None,
                    help="YYYY-MM-DD inclusive. Set this to the day BEFORE the "
                         "holdout starts so selection never sees the test window.")
    ap.add_argument("--min-t", type=float, default=3.0)
    ap.add_argument("--min-mono", type=float, default=0.30)
    ap.add_argument("--top", type=int, default=10, help="max distinct bases to keep")
    ap.add_argument("--max-corr", type=float, default=None,
                    help="greedy correlation cull: skip a survivor whose |rho| against an\n"
                         "ALREADY-SELECTED alpha exceeds this. Default None = base-name\n"
                         "dedup only (unchanged). 0.7 is the Finding Alphas Ch.10 bar.")
    ap.add_argument("--positive-only", action="store_true", default=True)
    ap.add_argument("--panel-dir", default="data/alpha_panel",
                    help="data/alpha_panel_deep for the 2017-start panel")
    ap.add_argument("--outcomes-table", default="outcomes",
                    help="deep_outcomes for the recomputed labels")
    ap.add_argument("--csv")
    ap.add_argument("--root")
    args = ap.parse_args()

    global ROOT
    if args.root:
        ROOT = os.path.expanduser(args.root)
    sys.path.insert(0, ROOT)
    from pathlib import Path
    import pandas as pd
    from analysis.alpha_fitness import _load_panel, _merge_outcomes, _score_one
    from analysis.detect_mw import classify_bases, classify_alpha

    m = _merge_outcomes(_load_panel(Path(ROOT) / args.panel_dir),
                        Path(ROOT) / "accuracy.db", args.horizon,
                        table=args.outcomes_table)
    all_lo, all_hi = str(m["date"].min())[:10], str(m["date"].max())[:10]
    if args.start:
        m = m[m["date"] >= pd.Timestamp(args.start)]
    if args.end:
        m = m[m["date"] <= pd.Timestamp(args.end)]
    if m.empty:
        sys.exit("FATAL: no rows in the requested date range")
    nd = m["date"].nunique()
    print(f"# alpha_select  h={args.horizon}  panel {all_lo}..{all_hi}")
    print(f"# SELECTION WINDOW {str(m['date'].min())[:10]}..{str(m['date'].max())[:10]}"
          f"  ({nd} dates, {len(m)} rows)")
    if not args.end:
        print("# WARNING: no --end given. Selection sees every date, so anything")
        print("#          downstream is NOT a nested holdout.")

    print("# classifying market-wide bases (this rebuilds base panels, ~1 min)...")
    bc = classify_bases()
    cols = [c for c in m.columns if c not in ("ticker", "date", "actual_return")]
    print(f"# scoring {len(cols)} alphas on the selection window...\n")

    rows = []
    r = m["actual_return"]
    d = m["date"]
    for i, c in enumerate(cols):
        if classify_alpha(c, bc) != "per_ticker":
            continue
        try:
            s = _score_one(m[c], r, d)
        except Exception:
            continue
        if not s:
            continue
        s["alpha"] = c
        s["base"] = c.split("__")[0]
        rows.append(s)
    if not rows:
        sys.exit("FATAL: nothing scored")
    df = pd.DataFrame(rows)

    keep = df[(df["ic_t"].abs() > args.min_t)
              & (df["mono"].notna()) & (df["mono"] >= args.min_mono)
              & (df["rank_ic"] > 0) & (df["sharpe"] > 0)]
    print(f"scored {len(df)} stock-picking alphas -> {len(keep)} pass the gate")
    if keep.empty:
        print("\nNo alpha passes on the selection window. That is the result:")
        print("the gate finds nothing when it cannot see the test data.")
        return 1

    # DEDUP BY BASE NAME IS NOT DEDUP BY INFORMATION (2026-08-25).
    # base = alpha.split("__")[0], so five DIFFERENT bases reaching the gate through
    # the SAME operator count as five picks. Measured on the deep panel: the h=5
    # survivors were low_52w_ratio / high_52w_ratio / vwap_dev_eod / rsi_14 / bb_pct,
    # every one via ts_argmax__w5 -- which returns WHICH of the last 5 days held the
    # max, i.e. a coarse 5-day momentum sign, identical whether the underlying series
    # is RSI, %B or distance-from-52w-low. Pairwise |rho| 0.598-0.863 (mean ~0.72);
    # on 2026-03-24 low_52w_ratio and rsi_14 both put exactly 144 names at value 0.0.
    # Effective independent count ~1.3, not 5.
    # Separately, five rank-equivalent rsi_14 transforms produced books identical to
    # three decimals -- same defect, different route (monotone transforms do not
    # change a sort, so they do not change a top decile).
    # --max-corr culls on the REALIZED alpha series over the SELECTION WINDOW ONLY.
    best = (keep.sort_values("sharpe", ascending=False)
                .groupby("base", as_index=False).first()
                .sort_values("sharpe", ascending=False))
    if args.max_corr is not None:
        cand = list(best["alpha"])
        sub = m[["date", "ticker"] + [c for c in cand if c in m.columns]].dropna(how="all")
        chosen, dropped = [], []
        for a in cand:
            if a not in sub.columns:
                continue
            worst = 0.0
            for b in chosen:
                pair = sub[[a, b]].dropna()
                if len(pair) < 100:
                    continue
                r = abs(pair[a].corr(pair[b], method="spearman"))
                worst = max(worst, 0.0 if pd.isna(r) else r)
            if worst > args.max_corr:
                dropped.append((a, round(worst, 3)))
            else:
                chosen.append(a)
            if len(chosen) >= args.top:
                break
        if dropped:
            print(f"\n# correlation cull at |rho| > {args.max_corr} "
                  f"dropped {len(dropped)}:")
            for a, r in dropped:
                print(f"#   {a:<44} rho={r}")
        best = best[best["alpha"].isin(chosen)].sort_values("sharpe", ascending=False)
    best = best.head(args.top)
    print(f"\n{'base':<24}{'alpha':<42}{'ic':>8}{'t':>7}{'sh':>7}{'mono':>7}")
    print("-" * 95)
    for _, x in best.iterrows():
        print(f"{x['base']:<24}{x['alpha']:<42}{x['rank_ic']:>8.4f}"
              f"{x['ic_t']:>7.2f}{x['sharpe']:>7.3f}{x['mono']:>7.2f}")

    # [:12] truncated display keys ("low_52w_rati="); keys are free-form, use the base.
    spec = ",".join(f"{x['base']}={x['alpha']}" for _, x in best.iterrows())
    print(f"\n# feed this to c1_holdout (selection never saw the test window):")
    print(f"python scripts/c1_holdout.py --alphas \"{spec}\"")

    if args.csv:
        keep.to_csv(args.csv, index=False)
        print(f"\n# wrote {args.csv} ({len(keep)} gated alphas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
