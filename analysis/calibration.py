#!/usr/bin/env python3
"""
calibration.py — rolling recalibration of prediction probabilities.

    python analysis/calibration.py --fit          # weekly: fit and store a map
    python analysis/calibration.py --apply        # daily: fill prob_cal
    python analysis/calibration.py --status       # show the stored maps

WHAT THIS FIXES
    Measured 2026-09-01: the model's probabilities are ~5x overconfident. At h=5
    the predicted decile spread is +0.413 against a realised +0.071 -- a "HIGH
    confidence" 0.70 is a 57% event. The overlay is not responsible (4.6x on
    prob_raw, 4.3x on prob_up); the isotonic layer inside CalibratedClassifierCV
    is fitted on X_train, which with TRAIN_START=2018 and a 60/20/20 split is
    roughly 2018-2023, then applied to 2026 predictions.

    A walk-forward test (analysis/calibration_audit.py) compared four options on
    21k-23k out-of-sample predictions. Shrink-to-base-rate won: h=5 ECE 0.0871 ->
    0.0229 (-74%), Brier 0.2587 -> 0.2487, predicted spread landing on the
    realised one (+0.075 vs +0.071). Platt was close (-70%). Isotonic was much
    WORSE (+245%) -- it collapses to a step function on a weak-signal model.

WHAT IT DOES NOT FIX
    Calibration maps are MONOTONE. AUC and accuracy at a 0.5 threshold are
    unchanged BY CONSTRUCTION. This does not improve the model's discrimination
    and will not move any accuracy figure. It fixes what the numbers MEAN, which
    matters because every threshold in the pipeline -- BUY at 0.55, confidence
    tiers at 0.70/0.55 -- is set against a scale that does not currently hold.

NON-BREAKING BY DESIGN
    Writes a NEW column, prob_cal. prob_raw and prob_up are untouched, so no
    existing threshold, signal or downstream consumer changes behaviour. The two
    scales can be compared side by side before anything is switched over. If the
    calibration turns out to be wrong, dropping the column reverts it entirely.

POINT-IN-TIME
    Two rules, both enforced here:
      1. A map fitted on date F may only use pairs whose OUTCOME resolved before
         F. Otherwise the map would encode outcomes it should not know.
      2. A prediction dated D is calibrated with the latest map where
         fitted_on <= D. Applying today's map to an old prediction would leak
         the future into the historical record and make any backtest of prob_cal
         meaningless.
    Rule 2 is why prob_cal for old rows may use an older, worse map. That is
    correct: it reproduces what the system would have said at the time.
"""
import argparse
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIN_FIT = 500            # matches calibration_audit; below this the map wobbles
K_FLOOR = 0.02           # never collapse to a constant -- see calibration_audit
K_CEIL = 1.0             # never EXPAND the spread; this only shrinks
N_BINS = 10
DEFAULT_WINDOW_DAYS = 180

DDL = """
CREATE TABLE IF NOT EXISTS calibration_map (
    horizon      INTEGER NOT NULL,
    fitted_on    TEXT    NOT NULL,
    base_rate    REAL    NOT NULL,
    k            REAL    NOT NULL,
    n_fit        INTEGER NOT NULL,
    window_start TEXT,
    window_end   TEXT,
    pred_spread  REAL,
    real_spread  REAL,
    k_floored    INTEGER DEFAULT 0,
    method       TEXT    DEFAULT 'shrink',
    created_at   TEXT    NOT NULL,
    PRIMARY KEY (horizon, fitted_on)
)
"""


def ensure_schema(con):
    con.execute(DDL)
    cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)")]
    if "prob_cal" not in cols:
        con.execute("ALTER TABLE predictions ADD COLUMN prob_cal REAL")
        print("  added column predictions.prob_cal")
    con.commit()


def fit_one(pairs):
    """-> (base_rate, k, pred_spread, real_spread, floored)

    k is the ratio of realised to predicted decile spread. A month where the
    model was INVERTED gives a negative realised spread and hence a negative k;
    flooring it at K_FLOOR keeps the map strictly monotone so ranking is fully
    preserved, while shrinking almost to the base rate -- the right response to
    a window showing no usable signal.
    """
    base = sum(y for _, y in pairs) / len(pairs)
    d = sorted(pairs)
    size = max(1, len(d) // N_BINS)
    lo_p = sum(x[0] for x in d[:size]) / size
    hi_p = sum(x[0] for x in d[-size:]) / size
    lo_r = sum(x[1] for x in d[:size]) / size
    hi_r = sum(x[1] for x in d[-size:]) / size
    raw_k = (hi_r - lo_r) / (hi_p - lo_p) if (hi_p - lo_p) else 1.0
    k = max(K_FLOOR, min(K_CEIL, raw_k))
    return base, k, hi_p - lo_p, hi_r - lo_r, int(k != raw_k)


def cmd_fit(con, args):
    ensure_schema(con)
    today = datetime.now().strftime("%Y-%m-%d")
    cutoff = (date.fromisoformat(today)
              - timedelta(days=args.window)).isoformat()
    print(f"fitting on outcomes RESOLVED before {today}, "
          f"window back to {cutoff}\n")

    for h in (1, 3, 5):
        rows = con.execute("""
            SELECT p.prob_raw, o.actual_up
            FROM predictions p JOIN outcomes o
              ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
             AND p.horizon=o.horizon
            WHERE p.horizon=? AND p.prob_raw IS NOT NULL
              AND o.actual_up IS NOT NULL
              AND o.outcome_date < ?          -- rule 1: resolved, not pending
              AND p.prediction_date >= ?
        """, (h, today, cutoff)).fetchall()
        if len(rows) < MIN_FIT:
            print(f"  h={h}: only {len(rows)} resolved pairs "
                  f"(need {MIN_FIT}) -- NOT fitted, previous map stands")
            continue
        pairs = [(float(p), int(y)) for p, y in rows]
        base, k, sp, sr, floored = fit_one(pairs)
        rng = con.execute("""
            SELECT MIN(p.prediction_date), MAX(p.prediction_date)
            FROM predictions p JOIN outcomes o
              ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
             AND p.horizon=o.horizon
            WHERE p.horizon=? AND o.outcome_date < ? AND p.prediction_date >= ?
        """, (h, today, cutoff)).fetchone()
        if not args.dry_run:
            con.execute("""INSERT OR REPLACE INTO calibration_map
                (horizon, fitted_on, base_rate, k, n_fit, window_start,
                 window_end, pred_spread, real_spread, k_floored, method,
                 created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,'shrink',?)""",
                (h, today, base, k, len(pairs), rng[0], rng[1], sp, sr,
                 floored, datetime.now().isoformat(timespec="seconds")))
        flag = "  [k FLOORED -- window showed no usable signal]" if floored else ""
        print(f"  h={h}: n={len(pairs):>6}  base={base:.4f}  k={k:.4f}  "
              f"spread pred {sp:+.3f} -> real {sr:+.3f}{flag}")
    if args.dry_run:
        print("\nDRY RUN -- no map written.")
    else:
        con.commit()
        print("\nmaps written.")


def cmd_apply(con, args):
    ensure_schema(con)
    maps = {}
    for h, f, base, k in con.execute(
            "SELECT horizon, fitted_on, base_rate, k FROM calibration_map "
            "ORDER BY horizon, fitted_on"):
        maps.setdefault(h, []).append((f, base, k))
    if not maps:
        print("no calibration maps stored -- run --fit first")
        return

    todo = con.execute(
        "SELECT id, horizon, prediction_date, prob_raw FROM predictions "
        "WHERE prob_cal IS NULL AND prob_raw IS NOT NULL"
        + ("" if args.all else " AND prediction_date >= date('now','-400 days')")
    ).fetchall()
    print(f"{len(todo)} rows without prob_cal")

    n, skipped = 0, 0
    for rid, h, pdate, praw in todo:
        # rule 2: the latest map fitted ON OR BEFORE the prediction date
        cands = [m for m in maps.get(h, []) if m[0] <= pdate]
        if not cands:
            skipped += 1
            continue
        _f, base, k = cands[-1]
        val = base + k * (float(praw) - base)
        if not args.dry_run:
            con.execute("UPDATE predictions SET prob_cal=? WHERE id=?",
                        (val, rid))
        n += 1
    if not args.dry_run:
        con.commit()
    print(f"  calibrated {n} rows; {skipped} skipped (no map fitted on or "
          f"before their prediction date -- correct, not an error)")
    if args.dry_run:
        print("DRY RUN -- nothing written.")


def cmd_backfill(con, args):
    """Fit a map at each month boundary from prior data only, then apply.

    WHY THIS IS NEEDED. PIT rule 2 says a prediction is calibrated with the
    latest map fitted ON OR BEFORE its date. On a fresh install the only map is
    today's, so every historical row is correctly skipped -- and prob_cal can
    never be evaluated against past outcomes.

    This walks history the way calibration_audit.py does: at the start of each
    month, fit on pairs whose outcomes had RESOLVED before that month began, and
    store the map dated to the month start. --apply then fills each month with
    the map that would have been in force. The result is a genuine point-in-time
    series, not today's map painted over the past.
    """
    ensure_schema(con)
    months = [r[0] for r in con.execute(
        "SELECT DISTINCT substr(prediction_date,1,7) FROM predictions "
        "WHERE prob_raw IS NOT NULL ORDER BY 1")]
    print(f"{len(months)} months of predictions: "
          f"{months[0] if months else '-'}..{months[-1] if months else '-'}\n")

    written = 0
    for h in (1, 3, 5):
        for m in months:
            start = f"{m}-01"
            cutoff = (date.fromisoformat(start)
                      - timedelta(days=args.window)).isoformat()
            rows = con.execute("""
                SELECT p.prob_raw, o.actual_up
                FROM predictions p JOIN outcomes o
                  ON p.ticker=o.ticker AND p.prediction_date=o.prediction_date
                 AND p.horizon=o.horizon
                WHERE p.horizon=? AND p.prob_raw IS NOT NULL
                  AND o.actual_up IS NOT NULL
                  AND o.outcome_date < ?      -- resolved before the month began
                  AND p.prediction_date >= ?
            """, (h, start, cutoff)).fetchall()
            if len(rows) < MIN_FIT:
                continue
            pairs = [(float(p), int(y)) for p, y in rows]
            base, k, sp, sr, floored = fit_one(pairs)
            if not args.dry_run:
                con.execute("""INSERT OR REPLACE INTO calibration_map
                    (horizon, fitted_on, base_rate, k, n_fit, window_start,
                     window_end, pred_spread, real_spread, k_floored, method,
                     created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,'shrink',?)""",
                    (h, start, base, k, len(pairs), cutoff, start, sp, sr,
                     floored, datetime.now().isoformat(timespec="seconds")))
            written += 1
            print(f"  h={h} {start}: n={len(pairs):>6} base={base:.4f} "
                  f"k={k:.4f}  spread {sp:+.3f} -> {sr:+.3f}"
                  + ("  [k FLOORED]" if floored else ""))
    if args.dry_run:
        print(f"\nDRY RUN -- {written} maps would be written.")
    else:
        con.commit()
        print(f"\n{written} historical maps written. Now run --apply --all.")


def cmd_status(con, args):
    ensure_schema(con)
    rows = con.execute(
        "SELECT horizon, fitted_on, base_rate, k, n_fit, pred_spread, "
        "real_spread, k_floored FROM calibration_map "
        "ORDER BY fitted_on DESC, horizon").fetchall()
    if not rows:
        print("no maps stored")
        return
    print(f"{'h':>3}{'fitted_on':>13}{'base':>8}{'k':>8}{'n':>8}"
          f"{'pred sp':>10}{'real sp':>10}  floored")
    for h, f, b, k, n, sp, sr, fl in rows:
        print(f"{h:>3}{f:>13}{b:>8.4f}{k:>8.4f}{n:>8}{sp:>10.3f}{sr:>10.3f}"
              f"  {'YES' if fl else ''}")
    cov = con.execute(
        "SELECT COUNT(*), SUM(prob_cal IS NOT NULL) FROM predictions "
        "WHERE prob_raw IS NOT NULL").fetchone()
    print(f"\npredictions with prob_raw: {cov[0]}   "
          f"with prob_cal: {cov[1] or 0}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="accuracy.db")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--backfill", action="store_true",
                    help="fit a PIT map at each month boundary so "
                         "history can be calibrated and evaluated")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW_DAYS)
    ap.add_argument("--all", action="store_true",
                    help="with --apply, backfill the entire history")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    if args.fit:
        cmd_fit(con, args)
    elif args.backfill:
        cmd_backfill(con, args)
    elif args.apply:
        cmd_apply(con, args)
    else:
        cmd_status(con, args)
    con.close()


if __name__ == "__main__":
    main()
