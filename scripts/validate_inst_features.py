#!/usr/bin/env python3
"""
Task B validation — institutional features, post-Pipeline-B (run Friday May 22).

Outputs the 3 criteria pre-registered in the May 21 calendar event:
  (A) NaN-coverage of inst_ features in the OOS window  — the VALIDITY GATE
  (B) Per-horizon feature importance from feature_importance_history
  (C) Pooled OOS AUC delta vs baseline (walk_forward_history)

Then prints a SUCCESS / PARTIAL / FAIL / INVALID verdict per the locked
decision rule.

Run:  python scripts/validate_inst_features.py
"""
import sqlite3
import sys
from datetime import date

DB = "accuracy.db"
INST_FEATS = [
    "inst_block_buy_sell_7d",
    "inst_signed_flow_30d",
    "inst_auction_imbal_5d",
    "inst_signed_flow_5d",
]
RETRAIN_CUTOFF = "2026-05-21"   # Pipeline B retrain date to inspect
AUC_BASELINE = 0.486            # verified pooled OOS baseline (memory #23)
INST_DATA_START = "2026-03-19"  # UW Basic historical floor

# Institutional data only exists from Mar 19. The OOS window for NaN coverage:
# everything from the start of inst data forward is "could-have-data"; before
# is structurally NaN and not the model's fault.
OOS_WINDOW_START = "2026-04-01"  # conservative recent-OOS slice


def section(title):
    print("\n" + "=" * 68)
    print(f" {title}")
    print("=" * 68)


def main():
    con = sqlite3.connect(DB)

    # ── (A) NaN COVERAGE — THE VALIDITY GATE ──────────────────────────────
    section("(A) NaN COVERAGE — validity gate")
    print(f"Checking inst_ feature NULL rate in prediction_features")
    print(f"window: {OOS_WINDOW_START} -> today\n")

    # Confirm the columns exist (Phase 3 migration must have run)
    cols = {r[1] for r in con.execute("PRAGMA table_info(prediction_features)")}
    missing = [f for f in INST_FEATS if f not in cols]
    if missing:
        print(f"  ERROR: columns missing from prediction_features: {missing}")
        print("  -> Phase 3 schema migration did not run. Investigate before")
        print("     trusting any other section.")
        con.close()
        sys.exit(1)

    total = con.execute(
        "SELECT COUNT(*) FROM prediction_features WHERE prediction_date >= ?",
        (OOS_WINDOW_START,)).fetchone()[0]

    if total == 0:
        print(f"  No prediction_features rows since {OOS_WINDOW_START}.")
        print("  Cannot assess coverage. Verdict: INVALID — defer.")
        con.close()
        sys.exit(0)

    worst_nan_pct = 0.0
    for f in INST_FEATS:
        n_null = con.execute(
            f"SELECT COUNT(*) FROM prediction_features "
            f"WHERE prediction_date >= ? AND {f} IS NULL",
            (OOS_WINDOW_START,)).fetchone()[0]
        pct = 100.0 * n_null / total
        worst_nan_pct = max(worst_nan_pct, pct)
        print(f"  {f:28s} {pct:5.1f}% NULL  ({n_null}/{total})")

    if worst_nan_pct < 40:
        coverage_verdict = "VALID"
    elif worst_nan_pct <= 70:
        coverage_verdict = "WEAK"
    else:
        coverage_verdict = "INVALID"
    print(f"\n  Worst NaN rate: {worst_nan_pct:.1f}%  ->  coverage = {coverage_verdict}")
    if coverage_verdict == "INVALID":
        print("  Low importance below is a DATA-SHORTAGE ARTIFACT, not a dud.")
        print("  Do NOT revert the flag. Real verdict = mid-June re-run.")
    elif coverage_verdict == "WEAK":
        print("  Treat (B)/(C) below as PRELIMINARY only.")

    # ── (B) PER-HORIZON IMPORTANCE ────────────────────────────────────────
    section("(B) PER-HORIZON FEATURE IMPORTANCE")
    print(f"feature_importance_history rows with retrain_date >= {RETRAIN_CUTOFF}\n")

    retrains = [r[0] for r in con.execute(
        "SELECT DISTINCT retrain_date FROM feature_importance_history "
        "WHERE retrain_date >= ? ORDER BY retrain_date", (RETRAIN_CUTOFF,))]
    if not retrains:
        print(f"  No retrain rows since {RETRAIN_CUTOFF}.")
        print("  -> Pipeline B has not logged importance yet. Re-run this")
        print("     script after Pipeline B completes.")
        importance_verdict = "NO_DATA"
    else:
        print(f"  Retrain dates found: {retrains}\n")
        # Aggregate: mean importance + mean rank across all tickers, per
        # (feature, horizon). A feature is "registering" if mean importance
        # is meaningfully > 0.
        n_features_registering = set()
        best_rank = {}
        for h in (1, 3, 5):
            print(f"  --- horizon {h} ---")
            rows = con.execute("""
                SELECT feature,
                       COUNT(*)            AS n_models,
                       AVG(importance)     AS mean_imp,
                       AVG(rank)           AS mean_rank,
                       MIN(rank)           AS best_rank
                FROM feature_importance_history
                WHERE retrain_date >= ? AND horizon = ?
                  AND feature LIKE 'inst_%'
                GROUP BY feature
                ORDER BY mean_imp DESC
            """, (RETRAIN_CUTOFF, h)).fetchall()
            if not rows:
                print(f"    (no inst_ features logged for horizon {h})")
                continue
            for feat, n_models, mean_imp, mean_rank, brank in rows:
                flag = ""
                if mean_imp and mean_imp > 0.0:
                    n_features_registering.add(feat)
                    best_rank[feat] = min(best_rank.get(feat, 999), brank)
                print(f"    {feat:26s} n={n_models:3d}  "
                      f"mean_imp={mean_imp or 0:.5f}  "
                      f"mean_rank={mean_rank or 0:.1f}  best_rank={brank}")

        n_reg = len(n_features_registering)
        any_top20 = any(r <= 20 for r in best_rank.values())
        print(f"\n  Features registering nonzero importance: {n_reg}/4")
        print(f"  At least one inst_ feature in top 20: {any_top20}")
        if n_reg >= 2 and any_top20:
            importance_verdict = "SUCCESS"
        elif n_reg >= 1:
            importance_verdict = "PARTIAL"
        else:
            importance_verdict = "FAIL"
        print(f"  -> importance = {importance_verdict}")

    # ── (C) OOS AUC DELTA ─────────────────────────────────────────────────
    section("(C) POOLED OOS AUC DELTA")
    runs = [r[0] for r in con.execute(
        "SELECT DISTINCT run_date FROM walk_forward_history "
        "ORDER BY run_date DESC")]
    print(f"  walk_forward_history run_dates: {runs[:5]}")

    post = [r for r in runs if r >= RETRAIN_CUTOFF]
    if not post:
        print(f"\n  No walk_forward run since {RETRAIN_CUTOFF}.")
        print("  walk_forward is a SEPARATE job from Pipeline B — it may not")
        print("  have run automatically. To evaluate (C), run the walk-forward")
        print("  harness manually, then re-run this script.")
        auc_verdict = "NO_DATA"
    else:
        latest = post[0]
        pooled = con.execute(
            "SELECT AVG(auc), COUNT(*) FROM walk_forward_history "
            "WHERE run_date = ?", (latest,)).fetchone()
        new_auc, n = pooled
        delta = new_auc - AUC_BASELINE
        print(f"\n  Latest run {latest}: pooled OOS AUC = {new_auc:.4f} "
              f"(n={n} ticker-horizons)")
        print(f"  Baseline: {AUC_BASELINE:.4f}   delta = {delta:+.4f}")
        if delta >= 0.005:
            auc_verdict = "SUCCESS"
        elif delta >= -0.005:
            auc_verdict = "NEUTRAL"
        else:
            auc_verdict = "FAIL"
        print(f"  -> AUC = {auc_verdict}")

    # ── FINAL VERDICT ─────────────────────────────────────────────────────
    section("FINAL VERDICT (per locked decision rule)")
    print(f"  (A) coverage   = {coverage_verdict}")
    print(f"  (B) importance = {importance_verdict}")
    print(f"  (C) AUC        = {auc_verdict}")
    print()

    if coverage_verdict == "INVALID":
        print("  VERDICT: INVALID — keep flag ON, do nothing.")
        print("  Real verdict is the mid-June re-run when inst data has")
        print("  accumulated enough OOS coverage.")
    elif importance_verdict in ("NO_DATA",) or auc_verdict == "NO_DATA":
        print("  VERDICT: INCOMPLETE — Pipeline B and/or walk_forward have")
        print("  not produced data yet. Keep flag ON, re-run this script")
        print("  once both have completed.")
    elif importance_verdict == "FAIL" and auc_verdict == "FAIL":
        print("  VERDICT: TASK B FAILED — revert the flag.")
        print("  unset ML_QUANT_INST_FEATURES (or remove from crontab/.env).")
        print("  Pipeline C will run clean without features. Investigate why")
        print("  the audit-validated features did not survive a real retrain.")
    elif importance_verdict in ("SUCCESS", "PARTIAL") and \
         auc_verdict in ("SUCCESS", "NEUTRAL"):
        print("  VERDICT: TASK B SUCCESS — keep flag ON.")
        print("  Institutional features add (or at least do not harm) signal.")
        print("  Tick the final Rule #1 box: Pipeline B nonzero importance.")
        if coverage_verdict == "WEAK":
            print("  NOTE: coverage was WEAK — confirm again at mid-June re-run.")
    else:
        print("  VERDICT: MIXED — keep flag ON, re-evaluate at mid-June.")
        print("  Friday can confirm a win; only mid-June can confirm a non-win.")

    con.close()


if __name__ == "__main__":
    main()
