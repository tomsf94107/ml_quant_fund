"""
recession/validation/horizon_scan.py

D+ Part 1 — the horizon scan. A DIAGNOSTIC-ONLY sweep of the existing
models M1 / M2 / M3 across forecast horizons {h=3, h=6, h=12} at target T1.

WHY THIS EXISTS
---------------
M1, M2, M3 all agree: at T1/h=12 the yield curve dominates and the other
macro features (NFCI, INDPRO, REAL_FFR_GAP, PERMIT, ISRATIO) add nothing —
linearly or nonlinearly. But h=12 is a LONG horizon. NFCI (financial
conditions) and INDPRO (industrial production) are not 12-month-leading by
economic nature; they move CLOSE to the recession. They may carry real
signal at h=3 or h=6 even though they are dead at h=12. The project has
only ever tested h=12. This scan looks at the shorter horizons.

WHAT THIS IS — AND IS NOT (the D+ discipline)
---------------------------------------------
This scan is PURELY DIAGNOSTIC. It informs INTERPRETATION. It does NOT
select the target cell for M4. M4 is pre-committed to T1/h=12 (the
project's primary cell) regardless of what this scan finds. This is a
deliberate anti-leakage decision: choosing M4's cell by "whichever cell
scored highest in the scan" would be selection on the validation metric —
the same leak as tuning a hyperparameter on the test set, one level up.

If the scan reveals signal at a shorter horizon, that does NOT retarget
M4. It becomes a PRE-REGISTERED HYPOTHESIS for a future, separately-
validated model series at that horizon — used honestly as a new
experiment, not as a post-hoc pivot.

PRE-REGISTERED READING RULE  (written BEFORE the scan is run)
-------------------------------------------------------------
The scan's verdict is mechanical, not a post-hoc narrative. Fixed in
advance:

  R1. FOLD FLOOR. A (target, horizon) cell yields a VERDICT only if it has
      >= MIN_SCOREABLE_FOLDS (=6) two-class scoreable folds. Cells below
      the floor are reported but marked NO-VERDICT — their numbers are
      indicative only, never compared on equal footing.

  R2. "BASELINE" PER CELL. Within each cell, the baseline is M1 (the
      single-feature yield-curve probit) at that horizon — exactly as at
      h=12. Each cell is judged against its OWN M1.

  R3. "FEATURES ALIVE" CRITERION. The non-yield-curve macro features are
      declared ALIVE at a horizon if BOTH:
        (a) the best of {M2, M3-core, M3-wide} at that horizon beats that
            horizon's M1 by >= MIN_MEANINGFUL_DELTA (=0.02 mean fold AUC),
            AND
        (b) that win is ROBUST — for M3, the seed strip spread is
            < SEED_STABLE_SPREAD (=0.02), so the win is not RNG luck.
      A win that fails (b) is reported as NOT ROBUST and does not count.

  R4. CROSS-HORIZON READING. The macro features are said to "come alive at
      shorter horizons" only if R3 holds at h=3 or h=6 while NOT holding
      at h=12. If R3 fails at every horizon, the conclusion is
      "yield-curve-dominated at all tested horizons" and the M-ladder's
      h=12 negative result generalises.

  R5. NO TARGET SELECTION. Whatever the scan finds, M4's cell stays
      T1/h=12. The scan output explicitly restates this.

These thresholds (6 folds, 0.02 AUC, 0.02 seed spread) are committed here,
in code, before the scan runs. They are not tuned afterward.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from recession.models.m1_probit import run_m1
from recession.models.m2_logit import run_m2
from recession.models.m3_forest import run_m3


# ---- pre-registered thresholds (committed before running) -------------------
MIN_SCOREABLE_FOLDS = 6        # R1 — fold floor for a verdict
MIN_MEANINGFUL_DELTA = 0.02    # R3a — AUC delta that counts as a real win
SEED_STABLE_SPREAD = 0.02      # R3b — M3 seed spread below this = robust

SCAN_HORIZONS = ["h=3", "h=6", "h=12"]
SCAN_TARGET = "T1"


# =============================================================================
# Per-cell evaluation
# =============================================================================

def _seed_spread(m3_results: dict) -> Optional[float]:
    """Max-min of M3-wide mean fold AUC across the seed strip."""
    strip = m3_results.get("seed_strip", {})
    aucs = [r.mean_fold_auc for r in strip.values()
            if r.mean_fold_auc is not None]
    if len(aucs) < 2:
        return None
    return max(aucs) - min(aucs)


def scan_cell(
    horizon: str,
    *,
    target: str = SCAN_TARGET,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
    **kwargs,
) -> dict:
    """Run M1, M2, M3 at one (target, horizon) cell and apply the
    pre-registered reading rule. Returns a dict of metrics + verdict.

    kwargs may mix genuine walk_forward parameters (min_train_months,
    test_window_months, step_months, threshold) with M3-only parameters
    (leaf_grid, seed_grid). They are routed: M3-only params go to run_m3
    alone; walk_forward params go to all three model drivers. Sending an
    M3-only param to run_m1/run_m2 would reach walk_forward and raise.
    """
    # split M3-only kwargs from shared walk_forward kwargs
    m3_only_keys = ("leaf_grid", "seed_grid")
    m3_kwargs = {k: v for k, v in kwargs.items() if k in m3_only_keys}
    wf_kwargs = {k: v for k, v in kwargs.items() if k not in m3_only_keys}

    # M3 also produces M1 and M2 on its common axis, but we run each
    # model's own driver so every model uses its own validated path.
    m1r = run_m1(target=target, horizon=horizon,
                 min_history_year=min_history_year, db_path=db_path,
                 **wf_kwargs)
    m2r = run_m2(target=target, horizon=horizon,
                 min_history_year=min_history_year, db_path=db_path,
                 **wf_kwargs)
    m3r = run_m3(target=target, horizon=horizon,
                 min_history_year=min_history_year, db_path=db_path,
                 **wf_kwargs, **m3_kwargs)

    m1 = m1r["m1"]
    m2 = m2r["m2"]
    m3c = m3r["m3_core"]
    m3w = m3r["m3_wide"]

    n_folds = m1.n_scoreable_folds
    baseline = m1.mean_fold_auc

    # candidate challengers
    challengers = {
        "M2": m2.mean_fold_auc,
        "M3-core": m3c.mean_fold_auc,
        "M3-wide": m3w.mean_fold_auc,
    }
    best_name, best_auc = None, None
    for name, auc in challengers.items():
        if auc is None:
            continue
        if best_auc is None or auc > best_auc:
            best_name, best_auc = name, auc

    seed_spread = _seed_spread(m3r)

    # ---- apply the pre-registered reading rule --------------------------
    # R1 — fold floor
    has_verdict = n_folds >= MIN_SCOREABLE_FOLDS

    delta = (best_auc - baseline
             if (best_auc is not None and baseline is not None) else None)

    # R3a — meaningful win?
    meaningful = (delta is not None and delta >= MIN_MEANINGFUL_DELTA)
    # R3b — robust? (only relevant if the winner is an M3 variant)
    if best_name in ("M3-core", "M3-wide"):
        robust = (seed_spread is not None and seed_spread < SEED_STABLE_SPREAD)
    else:
        # M2 is deterministic — no seed instability to worry about
        robust = True

    features_alive = bool(has_verdict and meaningful and robust)

    if not has_verdict:
        verdict = "NO-VERDICT (below fold floor)"
    elif features_alive:
        verdict = f"FEATURES ALIVE ({best_name} beats M1 by {delta:+.4f}, robust)"
    elif meaningful and not robust:
        verdict = (f"NOT ROBUST ({best_name} beats M1 by {delta:+.4f} but "
                   f"seed spread {seed_spread:.4f} >= {SEED_STABLE_SPREAD})")
    else:
        verdict = "YIELD-CURVE-DOMINATED (no challenger beats M1 meaningfully)"

    return {
        "target": target, "horizon": horizon,
        "n_scoreable_folds": n_folds,
        "m1_auc": baseline,
        "m2_auc": m2.mean_fold_auc,
        "m3_core_auc": m3c.mean_fold_auc,
        "m3_wide_auc": m3w.mean_fold_auc,
        "best_challenger": best_name,
        "best_challenger_auc": best_auc,
        "delta_vs_m1": delta,
        "m3_seed_spread": seed_spread,
        "perm_importance": m3r.get("perm_importance", {}),
        "has_verdict": has_verdict,
        "features_alive": features_alive,
        "verdict": verdict,
    }


# =============================================================================
# The scan
# =============================================================================

def run_horizon_scan(
    *,
    target: str = SCAN_TARGET,
    horizons: Optional[list[str]] = None,
    db_path: Optional[Path] = None,
    min_history_year: Optional[int] = 1986,
    **model_kwargs,
) -> dict:
    """Run the full diagnostic scan across horizons. One call, one process.

    model_kwargs may mix walk_forward params and M3-only params
    (leaf_grid, seed_grid); scan_cell routes them to the right drivers.

    Returns {'cells': {horizon: cell_dict}, 'cross_horizon': str}.
    """
    if horizons is None:
        horizons = SCAN_HORIZONS

    cells = {}
    for h in horizons:
        cells[h] = scan_cell(
            h, target=target, db_path=db_path,
            min_history_year=min_history_year, **model_kwargs,
        )

    # R4 — cross-horizon reading
    alive_short = any(
        cells[h]["features_alive"]
        for h in horizons if h in ("h=3", "h=6")
    )
    alive_long = (cells["h=12"]["features_alive"]
                  if "h=12" in cells else False)
    if alive_short and not alive_long:
        cross = ("Macro features COME ALIVE at shorter horizons — "
                 "R3 holds at h=3/h=6 but not h=12.")
    elif alive_short and alive_long:
        cross = "Macro features show signal across multiple horizons."
    elif not alive_short and not alive_long:
        cross = ("YIELD-CURVE-DOMINATED at every tested horizon — "
                 "the h=12 negative result generalises.")
    else:
        cross = ("Signal at h=12 only — unusual; inspect per-cell detail.")

    return {"cells": cells, "cross_horizon": cross,
            "target": target, "horizons": horizons}


def print_scan_report(scan: dict) -> None:
    """Print the horizon-scan map and the mechanical cross-horizon verdict."""
    cells = scan["cells"]
    horizons = scan["horizons"]

    print("=" * 74)
    print(f"HORIZON SCAN (D+ — DIAGNOSTIC ONLY) — target {scan['target']}")
    print("=" * 74)
    print("  Pre-registered thresholds: "
          f"fold floor {MIN_SCOREABLE_FOLDS}, "
          f"meaningful delta {MIN_MEANINGFUL_DELTA}, "
          f"seed-stable spread {SEED_STABLE_SPREAD}")
    print()

    # the map
    hdr = (f"  {'horizon':>8} {'folds':>6} {'M1':>8} {'M2':>8} "
           f"{'M3-core':>8} {'M3-wide':>8} {'best Δ':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for h in horizons:
        c = cells[h]
        def f(x):
            return f"{x:.4f}" if x is not None else "  n/a"
        d = (f"{c['delta_vs_m1']:+.4f}"
             if c["delta_vs_m1"] is not None else "  n/a")
        floor_mark = "" if c["has_verdict"] else "  <- below fold floor"
        print(f"  {h:>8} {c['n_scoreable_folds']:>6} "
              f"{f(c['m1_auc']):>8} {f(c['m2_auc']):>8} "
              f"{f(c['m3_core_auc']):>8} {f(c['m3_wide_auc']):>8} "
              f"{d:>9}{floor_mark}")

    # per-cell verdicts
    print()
    print("  PER-CELL VERDICT (pre-registered reading rule):")
    for h in horizons:
        c = cells[h]
        print(f"    {h:>6}: {c['verdict']}")
        # permutation importance — which features carried OOS signal
        perm = c.get("perm_importance", {})
        if perm and c["has_verdict"]:
            ranked = sorted(perm.items(), key=lambda kv: -kv[1])
            top = ", ".join(f"{name} {val:+.3f}" for name, val in ranked[:3])
            print(f"            OOS perm-importance (top 3): {top}")

    # cross-horizon
    print()
    print("-" * 74)
    print(f"  CROSS-HORIZON VERDICT (R4): {scan['cross_horizon']}")
    print()
    print("  R5 — NO TARGET SELECTION: this scan is diagnostic only. "
          "M4 remains")
    print("  pre-committed to T1/h=12. Any shorter-horizon signal found "
          "here becomes")
    print("  a pre-registered hypothesis for a future separately-validated "
          "model series,")
    print("  NOT a retargeting of M4.")
    print("=" * 74)
