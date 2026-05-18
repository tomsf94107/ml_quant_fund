"""
fix_indpro_rebasing.py — Option G: re-chain INDPRO across FRED rebasing breaks.

THE PROBLEM
-----------
INDPRO in features_monthly has 4 spurious ~30-37% one-month cliffs:
    ~1971-07, ~1985-06, ~1990-03, ~2002-11
These are NOT real economic events. They are artifacts of FRED rebasing the
Industrial Production index (re-referencing which year = 100). The DB stored
pre-rebase months on old bases and post-rebase months on new bases, so
adjacent months across a rebase sit on different bases and line up as a cliff.

A recession model fed this would learn 4 fake 37% IP crashes. Must fix the
stored data before any model sees it.

THE METHOD (Option G — standard index re-chaining)
--------------------------------------------------
This is the method national statistical agencies (incl. the Federal Reserve,
which produces INDPRO) use to join rebased index segments:

  1. Compute month-over-month growth rates. Growth is BASE-INVARIANT: a 0.4%
     rise is 0.4% regardless of which year = 100. Growth rates are valid
     WITHIN each base segment.
  2. The 4 cross-cliff growth rates are fake (they encode the rebase, not
     real growth). Replace each with a local estimate from neighbouring
     months — IP is near-flat at all 4 seams, so this is well-determined.
  3. Re-accumulate the (now clean) growth rates into a single continuous
     LEVEL index, anchored so the most recent segment keeps its actual
     published values. Older history is reconstructed onto that one base.

Result: one continuous INDPRO level series, no cliffs, on a single base.
INDPRO stays a level feature -> Hamilton detrending still applies ->
no downstream config changes needed (AT_RISK_EMPIRICAL, pca sign anchors
all unchanged).

KNOWN, ACCEPTED LIMITATION
--------------------------
Re-accumulation reconstructs old-history levels using growth rates and seam
repairs from the whole series, including post-date corrections. A backtest
standing at e.g. 1990 reads a 1965 level reconstructed with knowledge of the
2002 seam. This is a mild look-ahead. It is ACCEPTED because INDPRO is
Hamilton-detrended downstream: a smooth multiplicative re-levelling of history
leaves the cycle (deviations from trend) — which is what the model sees —
unchanged. Documented here so future readers know it was a conscious choice,
not an oversight.

SAFEGUARDS (Rule 1: check, gap-check, test every path)
------------------------------------------------------
  1. Backs up the INDPRO rows before any write.
  2. Idempotency guard: if INDPRO already has 0 cliffs, no-op.
  3. In-script gap check: re-runs the cliff audit AFTER rescaling and
     ABORTS (rolls back) if any cliff remains or the level range is
     economically insane.
  4. Prints before/after at all 4 seams for human eyeballing.
  5. --dry-run mode writes nothing.

USAGE
-----
    python fix_indpro_rebasing.py --dry-run     # show what would happen
    python fix_indpro_rebasing.py               # apply (after backup)
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DB = Path.cwd() / "recession.db"

# A "cliff" = a month-over-month change whose robust z-score (via MAD) exceeds
# this. Real INDPRO monthly moves are ~1% (MAD-scaled); a rebasing cliff is
# 30+ sigma. 15 is a safe threshold well above any real move.
CLIFF_Z_THRESHOLD = 15.0

# Gap check (post-fix): real INDPRO never moves more than this fraction
# month-over-month. The four rebasing cliffs were 20-37% moves; a correctly
# re-chained series has no move this large. The ONE legitimate exception in
# 1960-2026 is the April 2020 COVID collapse (~-12.7% raw; can read slightly
# differently post-rechain), so we allow up to COVID_OUTLIER_ALLOWANCE
# months to exceed the bound.
MAX_REAL_MOM_MOVE = 0.15          # 15% month-over-month
COVID_OUTLIER_ALLOWANCE = 1       # April 2020 is allowed to exceed

# Number of neighbouring months averaged to repair a seam growth rate.
SEAM_REPAIR_WINDOW = 3


# =============================================================================
# Core helpers
# =============================================================================

def load_indpro(conn: sqlite3.Connection) -> tuple[list[str], np.ndarray]:
    """Load INDPRO as (months, values), one value per observation_month
    (latest vintage per month)."""
    rows = conn.execute(
        "SELECT observation_month, value, vintage_date "
        "FROM features_monthly WHERE feature_name='INDPRO' "
        "ORDER BY observation_month, vintage_date"
    ).fetchall()
    by_month: dict[str, float] = {}
    for m, v, _vd in rows:
        if v is not None:
            by_month[m] = v          # later vintage overwrites -> latest wins
    months = sorted(by_month)
    vals = np.array([by_month[m] for m in months], dtype=float)
    return months, vals


def find_cliffs(vals: np.ndarray) -> list[int]:
    """Return indices i where vals[i]->vals[i+1] is a REBASING cliff.

    A rebasing cliff must satisfy BOTH:
      - relative: a robust MAD z-score outlier among month-over-month moves
      - absolute: the move exceeds MAX_REAL_MOM_MOVE (15%)

    The absolute floor is essential. The MAD z-score alone false-positives:
    on a clean (already-fixed) series the typical step is tiny, so a normal
    8% recession move scores as a >15-sigma "cliff". Requiring the move to
    also be >15% in absolute terms means find_cliffs only ever flags the
    genuine ~20-37% rebasing breaks — never a real economic move. This also
    makes the function a correct idempotency check: a re-chained series has
    no >15% move, so find_cliffs returns [] and the migration no-ops.
    """
    if len(vals) < 5:
        return []
    d = np.diff(vals)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    if mad < 1e-12:
        return []
    rz = np.abs(d - med) / (1.4826 * mad)
    mom = np.abs(d / vals[:-1])
    is_cliff = (rz > CLIFF_Z_THRESHOLD) & (mom > MAX_REAL_MOM_MOVE)
    return sorted(int(i) for i in np.where(is_cliff)[0])


def _seam_growth(growth: np.ndarray, ci: int, cliffs: list[int]) -> float:
    """Estimate the TRUE month-over-month growth across a rebasing seam.

    The raw growth[ci] is the fake cliff. We estimate the real growth from
    neighbouring months (IP is near-flat across all 4 seams, so the
    neighbour mean is a good estimate). Other cliff months are excluded
    from the neighbour set.
    """
    lo = max(0, ci - SEAM_REPAIR_WINDOW)
    hi = min(len(growth), ci + SEAM_REPAIR_WINDOW + 1)
    neighbours = [growth[j] for j in range(lo, hi)
                  if j != ci and j not in cliffs]
    if not neighbours:
        return float(np.median(growth))
    return float(np.mean(neighbours))


def rechain_indpro(
    months: list[str],
    vals: np.ndarray,
    cliffs: list[int],
    verbose: bool = True,
) -> np.ndarray:
    """Re-chain INDPRO across rebasing breaks via segment rescaling (Option G).

    Method (standard index chain-linking):
      The 4 cliffs split INDPRO into 5 segments, each on its own base.
      The LAST segment is the anchor — it's on the current FRED base, so
      its real published values are kept untouched (factor = 1.0).
      Walking backwards, each older segment k is multiplied by a single
      factor chosen so that segment k chains continuously onto segment k+1:

          last_value(seg k) * factor[k] * exp(true_seam_growth)
              == first_value(seg k+1, already rescaled)

      where true_seam_growth is estimated from neighbouring months (the
      raw cross-cliff growth is the fake rebasing jump and is discarded).

    This keeps modern data literally as FRED published it, removes all 4
    cliffs, and produces one continuous series. Because INDPRO is
    Hamilton-detrended downstream, the absolute scale of rescaled older
    history is immaterial — only the (preserved) cycle structure matters.

    Returns a new value array, same length, no cliffs.
    """
    n = len(vals)
    if (vals <= 0).any():
        raise ValueError("INDPRO has non-positive values — cannot log-chain.")
    if not cliffs:
        return vals.astype(float).copy()

    # log month-over-month growth; growth[i] = change from month i to i+1
    growth = np.diff(np.log(vals))

    # Segment boundaries: segment k spans indices (bounds[k]+1 .. bounds[k+1])
    bounds = [-1] + list(cliffs) + [n - 1]
    n_seg = len(bounds) - 1

    out = vals.astype(float).copy()
    factor = [1.0] * n_seg            # last segment anchored at 1.0

    # Walk backwards from the second-last segment to the first.
    for k in range(n_seg - 2, -1, -1):
        ci = cliffs[k]                       # seam between seg k and seg k+1
        last_old = vals[ci]                  # last RAW value of segment k
        first_new = out[ci + 1]              # first (already-scaled) value, seg k+1
        g = _seam_growth(growth, ci, cliffs)  # true growth across the seam
        # choose factor[k] so seg k chains continuously onto seg k+1
        factor[k] = first_new / (last_old * np.exp(g))
        seg_lo, seg_hi = bounds[k] + 1, bounds[k + 1] + 1
        out[seg_lo:seg_hi] = vals[seg_lo:seg_hi] * factor[k]
        if verbose:
            print(f"    seam at {months[ci]}->{months[ci+1]}: "
                  f"fake jump {(vals[ci+1]/vals[ci]-1)*100:+.1f}% "
                  f"-> repaired {(out[ci+1]/out[ci]-1)*100:+.2f}% "
                  f"(segment factor {factor[k]:.4f})")

    return out


# =============================================================================
# Main migration
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Re-chain INDPRO across FRED rebasings.")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report, write nothing")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"ERROR: DB not found: {args.db}")
        return 2

    print("=" * 70)
    print("INDPRO re-chaining migration (Option G — growth-rate splice)")
    print("=" * 70)

    conn = sqlite3.connect(args.db)
    try:
        months, vals = load_indpro(conn)
        print(f"Loaded INDPRO: {len(vals)} months, "
              f"{months[0]} .. {months[-1]}, "
              f"range {vals.min():.1f}-{vals.max():.1f}")

        cliffs = find_cliffs(vals)
        print(f"Cliffs detected: {len(cliffs)}")
        for i in cliffs:
            print(f"  {months[i]} {vals[i]:.1f} -> {months[i+1]} {vals[i+1]:.1f} "
                  f"({(vals[i+1]/vals[i]-1)*100:+.1f}%)")

        # --- Idempotency guard ---
        if not cliffs:
            print("\nNo cliffs found — INDPRO already clean. Nothing to do.")
            return 0

        # --- Re-chain ---
        print("\nRe-chaining via growth-rate splice:")
        rechained = rechain_indpro(months, vals, cliffs, verbose=True)

        # --- In-script GAP CHECK (Rule 1) ---
        # The post-fix series is smooth, so a RELATIVE cliff detector would
        # false-positive on real recessions (its threshold shrinks with the
        # cleaned series' low volatility). The correct check is ABSOLUTE:
        # real INDPRO never moves more than ~15% month-over-month, with the
        # single legitimate exception of the April 2020 COVID collapse.
        print("\nGap check on re-chained series (absolute MoM bound):")
        mom = np.abs(np.diff(rechained) / rechained[:-1])
        worst = float(mom.max())
        n_excessive = int((mom > MAX_REAL_MOM_MOVE).sum())
        # all 4 original seams must now be small moves
        seam_moves = [abs(rechained[ci + 1] / rechained[ci] - 1)
                      for ci in cliffs]
        worst_seam = max(seam_moves)

        ok_mom = (n_excessive <= COVID_OUTLIER_ALLOWANCE)
        ok_seams = (worst_seam < MAX_REAL_MOM_MOVE)

        print(f"  max |MoM move| anywhere: {worst*100:.2f}%")
        print(f"  months exceeding {MAX_REAL_MOM_MOVE*100:.0f}%: "
              f"{n_excessive}  (<= {COVID_OUTLIER_ALLOWANCE} allowed for COVID)  "
              f"{'OK' if ok_mom else 'FAIL'}")
        print(f"  worst of the 4 repaired seams: {worst_seam*100:.2f}%  "
              f"{'OK' if ok_seams else 'FAIL'}")

        if not (ok_mom and ok_seams):
            print("\nABORT: re-chained series failed the gap check. "
                  "No changes written.")
            return 1

        # --- Before/after at each seam (human eyeball) ---
        print("\nBefore / after at each seam:")
        for i in cliffs:
            lo, hi = max(0, i - 1), min(len(months), i + 3)
            print(f"  seam {months[i]}->{months[i+1]}:")
            for j in range(lo, hi):
                print(f"    {months[j]}  raw={vals[j]:8.2f}   "
                      f"rechained={rechained[j]:8.2f}")

        if args.dry_run:
            print("\n--dry-run: no changes written.")
            return 0

        # --- Backup ---
        backup = args.db.with_suffix(
            ".db.bak.before_indpro_rechain"
        )
        shutil.copy2(args.db, backup)
        print(f"\nBacked up DB to: {backup}")

        # --- Write rechained values back ---
        # Update every (INDPRO, observation_month) row's value. Each month
        # has one value (latest vintage); we update all vintage rows for
        # that month to the rechained value so the series is consistent
        # regardless of which vintage pit_loader picks.
        month_to_val = {m: float(v) for m, v in zip(months, rechained)}
        n_updated = 0
        for m, v in month_to_val.items():
            cur = conn.execute(
                "UPDATE features_monthly SET value=? "
                "WHERE feature_name='INDPRO' AND observation_month=?",
                (v, m),
            )
            n_updated += cur.rowcount
        conn.commit()
        print(f"Updated {n_updated} INDPRO rows.")

        # --- Post-write verification (same absolute MoM bound) ---
        months2, vals2 = load_indpro(conn)
        mom2 = np.abs(np.diff(vals2) / vals2[:-1])
        n_excessive2 = int((mom2 > MAX_REAL_MOM_MOVE).sum())
        verify_ok = (n_excessive2 <= COVID_OUTLIER_ALLOWANCE)
        print(f"\nPost-write verification: max |MoM| {mom2.max()*100:.2f}%, "
              f"{n_excessive2} excessive months  "
              f"{'OK' if verify_ok else 'FAIL'}")
        if not verify_ok:
            print("ERROR: excessive moves remain after write. Restore from backup:")
            print(f"  cp {backup} {args.db}")
            return 1

        print("\n" + "=" * 70)
        print("INDPRO re-chaining complete. Series is now base-consistent.")
        print("=" * 70)
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
