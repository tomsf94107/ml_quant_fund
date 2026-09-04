#!/usr/bin/env python3
"""
fund_ep_verdict.py — did wiring fund_ep and fund_ni_margin help?

READ-ONLY. Prints a verdict against a pre-registered baseline.

THE EXPERIMENT
    fund_ep and fund_ni_margin were added to models/classifier.py
    FEATURE_COLUMNS on 2026-09-04, taking it from 96 to 98. Both had been BUILT
    since May 2026 and never once in a trained model -- feature_importance_history
    held zero rows for any fund_* column.

    The pre-registered baseline, walk_forward_history on 2026-08-30 across 400
    tickers, recorded BEFORE the change:

        h=1  AUC 0.5110
        h=3  AUC 0.5243
        h=5  AUC 0.5362

    First single-ticker retrain after wiring (AAPL, 2026-09-05):
        fund_ep         9.647 / 14.777 / 9.415  at h=1/3/5
        fund_ni_margin  0.000 on all three

    Read that cautiously. On ONE ticker a new column often absorbs importance
    previously spread across correlated features -- high importance is not the
    same as new information. AUC on that same run was 0.515 at h=5 against
    0.5214 an hour earlier without the features. One fit, noise-dominated.

THE PRE-REGISTERED DECISION RULE, written before the result was seen
    KEEP if, after at least five full nightly retrains:
       - fund_ep holds non-zero mean importance across the universe, AND
       - walk-forward AUC at h=3 and h=5 is not materially BELOW baseline
         (a fall of more than 0.005 counts as material)
    DROP fund_ni_margin if its mean importance stays at ~0.000, which is what
       its 24-35% orthogonalisation retention predicted.
    DROP BOTH if AUC falls materially at two horizons.

    Writing the rule down first is the point. A threshold chosen after seeing
    the outcome is not a test.

    python analysis/fund_ep_verdict.py
"""
import argparse
import sqlite3
import statistics as st

BASELINE = {1: 0.5110, 3: 0.5243, 5: 0.5362}
BASE_DATE = "2026-08-30"
WIRED_ON = "2026-09-04"
MATERIAL = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    print(f"baseline {BASE_DATE} (400 tickers, BEFORE wiring): "
          + "  ".join(f"h={h} {v:.4f}" for h, v in BASELINE.items()))
    print(f"features wired {WIRED_ON}\n")

    print("WALK-FORWARD AUC")
    print(f"  {'run_date':<13}{'h':>3}{'tickers':>9}{'AUC':>9}{'vs base':>10}")
    seen = {}
    for d, h, n, a in con.execute(
            "SELECT run_date, horizon, COUNT(*), AVG(auc) "
            "FROM walk_forward_history WHERE run_date >= ? "
            "GROUP BY run_date, horizon ORDER BY run_date, horizon",
            (BASE_DATE,)):
        b = BASELINE.get(h)
        mark = ""
        if d > WIRED_ON and b is not None:
            if a - b <= -MATERIAL:
                mark = "  <-- MATERIALLY BELOW"
            seen.setdefault(h, []).append(a)
        print(f"  {str(d)[:10]:<13}{h:>3}{n:>9}{a:>9.4f}"
              f"{(a - b if b else 0):>+10.4f}{mark}")

    print("\nfund_* IMPORTANCE, universe mean")
    print(f"  {'retrain_date':<14}{'feature':<18}{'tickers':>9}{'mean imp':>10}")
    imp = {}
    for d, f, n, v in con.execute(
            "SELECT retrain_date, feature, COUNT(DISTINCT ticker), "
            "AVG(importance) FROM feature_importance_history "
            "WHERE feature LIKE 'fund\\_%' ESCAPE '\\' "
            "GROUP BY retrain_date, feature ORDER BY retrain_date DESC, feature "
            "LIMIT 20"):
        print(f"  {str(d)[:10]:<14}{f:<18}{n:>9}{v:>10.3f}")
        imp.setdefault(f, []).append((d, n, v))
    con.close()

    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    runs = max((len(v) for v in seen.values()), default=0)
    if runs < 5:
        print(f"  {runs} post-wiring walk-forward run(s). The rule needs at "
              f"least 5.")
        print("  Too early. Re-run after more nightly retrains.")
    else:
        bad = [h for h, v in seen.items()
               if st.mean(v) - BASELINE[h] <= -MATERIAL]
        print(f"  horizons materially below baseline: "
              f"{bad if bad else 'none'}")
        if len(bad) >= 2:
            print("  -> DROP BOTH. AUC fell at two horizons.")
        else:
            print("  -> AUC holds.")

    for f in ("fund_ep", "fund_ni_margin"):
        rows = [r for r in imp.get(f, []) if r[1] >= 50]   # universe-wide only
        if not rows:
            print(f"  {f}: no universe-wide retrain yet "
                  f"(single-ticker runs are excluded)")
            continue
        m = st.mean(r[2] for r in rows)
        if m <= 0.01:
            print(f"  {f}: mean importance {m:.3f} -> DROP, spanned in practice")
        else:
            print(f"  {f}: mean importance {m:.3f} -> KEEP")

    print("\n  Reminder: importance is not predictive value. A new column can")
    print("  absorb importance previously spread across correlated features")
    print("  while adding nothing. AUC against the baseline is the real test.")


if __name__ == "__main__":
    main()
