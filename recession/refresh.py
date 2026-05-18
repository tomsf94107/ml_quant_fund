"""
recession/refresh.py

Step 12 — the refresh script. The final step of the build; running it on a
schedule is what makes the recession model a LIVE research tool rather
than a one-off analysis.

WHAT IT DOES
------------
  1. Fits M1 (the yield-curve probit) on all history currently in
     recession.db.
  2. Computes the latest 12-month recession probability.
  3. Appends a timestamped line to recession/refresh_log.txt — a running
     journal of the model's reading over time.
  4. Exits non-zero on any failure, so a cron wrapper can detect it.

It deliberately does NOT pull new data from FRED itself — data ingestion
is a separate concern owned by the Phase-1 ingestion scripts. refresh.py
re-reads whatever is in recession.db and re-evaluates M1. Point the data
ingestion cron before this one if fresh FRED data is wanted.

CADENCE
-------
The recession project runs on a weekly (Sunday) research cadence — recession
data is monthly and slow-moving, so daily refreshes add nothing. The
crontab line (Vietnam-anchored, consistent with the project convention):

    # recession model — weekly refresh, Sunday 08:00 Vietnam time
    0 8 * * 0  cd ~/Desktop/ML_Quant_Fund && \\
        /path/to/ml_quant_310/bin/python -m recession.refresh \\
        >> ~/Desktop/ML_Quant_Fund/recession/cron.log 2>&1

(macOS BSD cron uses system localtime; the box is on Vietnam time, so
08:00 there. Adjust the python path to the ml_quant_310 environment.)

This script ships recession model v1.1.1 — the final piece of the
12-step build.
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path


def _default_db_path() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "recession.db", here / "recession.db",
                 Path.cwd() / "recession.db"):
        if cand.exists():
            return cand
    return here.parent / "recession.db"


def _log_path() -> Path:
    return Path(__file__).resolve().parent / "refresh_log.txt"


def refresh(db_path: Path | None = None,
            log_path: Path | None = None) -> dict:
    """Re-evaluate M1 on current data and log the latest recession
    probability.

    Returns {'ok': bool, 'latest_month', 'latest_proba', 'message'}.
    Raises nothing — failures are returned in the dict.
    """
    db_path = db_path or _default_db_path()
    log_path = log_path or _log_path()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not db_path.exists():
        msg = f"recession.db not found at {db_path}"
        _append_log(log_path, f"[{stamp}] ERROR  {msg}")
        return {"ok": False, "message": msg}

    try:
        from recession.models.m1_probit import M1Probit, M1_FEATURES
        from recession.features.builder import build_feature_dataframe

        fr = build_feature_dataframe(
            target="T1", horizon="h=12",
            as_of="today", train_cutoff="today",
            feature_subset=M1_FEATURES, db_path=db_path,
        )
        X = fr.X[M1_FEATURES]
        y = fr.y
        mask = X.notna().all(axis=1)
        train_mask = mask & y.notna()
        if train_mask.sum() < 24:
            msg = f"only {int(train_mask.sum())} usable training rows"
            _append_log(log_path, f"[{stamp}] ERROR  {msg}")
            return {"ok": False, "message": msg}

        model = M1Probit().fit(X.loc[train_mask],
                               y.loc[train_mask].astype(int))
        X_ok = X.loc[mask]
        proba = model.predict_proba(X_ok)
        latest_month = X_ok.index.max()
        latest_proba = float(proba[-1])

        line = (f"[{stamp}] OK     "
                f"recession_prob({latest_month:%Y-%m})="
                f"{latest_proba:.4f}")
        _append_log(log_path, line)
        return {
            "ok": True,
            "latest_month": latest_month,
            "latest_proba": latest_proba,
            "message": line,
        }
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        _append_log(log_path,
                    f"[{stamp}] ERROR  {type(e).__name__}: {e}\n{tb}")
        return {"ok": False, "message": f"{type(e).__name__}: {e}"}


def _append_log(log_path: Path, line: str) -> None:
    """Append one line to the refresh journal, creating it if needed."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(line.rstrip("\n") + "\n")


def main() -> int:
    """Entry point for `python -m recession.refresh`. Returns an exit
    code: 0 on success, 1 on failure (for cron to detect)."""
    result = refresh()
    if result["ok"]:
        print(result["message"])
        return 0
    print(f"refresh failed: {result['message']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
