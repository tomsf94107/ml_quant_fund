"""
recession/features/derive_copper_gold.py

Derives the COPPER_GOLD feature — the copper-to-gold price ratio.

STATUS: v2-ready, not yet active. COPPER_GOLD is marked `skip_v1` in
series_specs.py because both FRED London gold series
(GOLDAMGBD228NLBM, GOLDPMGBD228NLBM) were discontinued in 2025 and return
HTTP 400 — there is currently no working gold series to divide by. This
module is the derivation, written and tested ahead of time; it becomes
active once a v2 gold series (Yahoo GC=F futures, LBMA, or a vendor) is
ingested. Until then, run against the real DB it will report an honest
"gold series not present" failure and write nothing — by design.

WHY COPPER/GOLD
---------------
The copper-to-gold ratio is a market-based growth/risk gauge: copper
("Dr. Copper") rises with industrial demand and global growth; gold rises
with risk aversion. A falling ratio signals the market pricing slower
growth / higher risk — a recession-relevant signal. It is engineered
rather than downloaded: there is no single "copper/gold" FRED series, so
it must be computed from two inputs.

WHAT THIS SCRIPT NEEDS
----------------------
Two source price series must already be present in features_monthly. The
script auto-detects them by trying common feature names; if neither input
is present it reports exactly what is missing and does nothing
destructive.

  copper  : tried as PCOPPUSDM (FRED global copper price) or COPPER
  gold    : tried as one of GOLDAMGBD228NLBM / GOLDPMGBD228NLBM (FRED
            London gold fix) or GOLD

If your DB uses different names, pass them explicitly to
derive_copper_gold(copper_feature=..., gold_feature=...).

POINT-IN-TIME SAFETY
--------------------
The ratio for observation month M is computed from copper(M) / gold(M) —
same-month inputs only, no look-ahead. The derived row's vintage_date is
set to the LATER of the two inputs' vintage dates (the ratio could not
have been known before both inputs were known). This keeps COPPER_GOLD
consistent with the project's PIT discipline.

USAGE
    from recession.features.derive_copper_gold import derive_copper_gold
    result = derive_copper_gold(db_path="recession.db")
    print(result["message"])
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional


# candidate source-series names, tried in order
COPPER_CANDIDATES = ["PCOPPUSDM", "COPPER", "COPPER_PRICE"]
GOLD_CANDIDATES = ["GOLDAMGBD228NLBM", "GOLDPMGBD228NLBM", "GOLD",
                   "GOLD_PRICE"]

DERIVED_FEATURE_NAME = "COPPER_GOLD"


def _present_features(conn: sqlite3.Connection) -> set[str]:
    """Feature names that actually have rows in features_monthly."""
    rows = conn.execute(
        "SELECT DISTINCT feature_name FROM features_monthly").fetchall()
    return {r[0] for r in rows}


def _pick(candidates: list[str], present: set[str]) -> Optional[str]:
    """First candidate name that is present, or None."""
    for c in candidates:
        if c in present:
            return c
    return None


def _load_series(conn: sqlite3.Connection, feature: str) -> dict:
    """Latest-vintage value per observation_month for one feature.
    Returns {observation_month: (value, vintage_date)}."""
    rows = conn.execute(
        """SELECT f.observation_month, f.value, f.vintage_date
           FROM features_monthly f
           INNER JOIN (
               SELECT observation_month, MAX(vintage_date) AS mv
               FROM features_monthly WHERE feature_name = ?
               GROUP BY observation_month
           ) latest
             ON f.observation_month = latest.observation_month
             AND f.vintage_date = latest.mv
           WHERE f.feature_name = ?""",
        (feature, feature),
    ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def derive_copper_gold(
    db_path: Optional[Path] = None,
    *,
    copper_feature: Optional[str] = None,
    gold_feature: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Compute COPPER_GOLD = copper / gold and write it to features_monthly.

    copper_feature / gold_feature: override the source series names. If
    None, the script auto-detects from COPPER_CANDIDATES / GOLD_CANDIDATES.

    dry_run: if True, compute and report but do NOT write.

    Returns {'ok', 'message', 'n_written', 'copper_feature',
             'gold_feature', 'months'}.
    """
    if db_path is None:
        db_path = Path("recession.db")

    conn = sqlite3.connect(db_path)
    try:
        present = _present_features(conn)

        cu = copper_feature or _pick(COPPER_CANDIDATES, present)
        au = gold_feature or _pick(GOLD_CANDIDATES, present)

        # honest failure: say exactly what is missing
        if cu is None and au is None:
            return {"ok": False, "n_written": 0,
                    "message": (
                        "Neither a copper nor a gold source series is in "
                        "features_monthly. COPPER_GOLD cannot be derived "
                        "until both are ingested. Tried copper: "
                        f"{COPPER_CANDIDATES}; gold: {GOLD_CANDIDATES}. "
                        "Ingest a copper price series and a gold price "
                        "series first, or pass copper_feature/gold_feature "
                        "with the names your DB uses.")}
        if cu is None:
            return {"ok": False, "n_written": 0,
                    "message": (
                        f"Gold series '{au}' is present but no copper "
                        f"series found (tried {COPPER_CANDIDATES}). Ingest "
                        f"a copper price series first.")}
        if au is None:
            return {"ok": False, "n_written": 0,
                    "message": (
                        f"Copper series '{cu}' is present but no gold "
                        f"series found (tried {GOLD_CANDIDATES}). Ingest a "
                        f"gold price series first.")}

        copper = _load_series(conn, cu)
        gold = _load_series(conn, au)

        # ratio on the common months where gold != 0
        common = sorted(set(copper) & set(gold))
        rows_to_write = []
        skipped_zero = 0
        pull_date = date.today().isoformat()
        for month in common:
            c_val, c_vin = copper[month]
            g_val, g_vin = gold[month]
            if c_val is None or g_val is None:
                continue
            if g_val == 0:
                skipped_zero += 1
                continue
            ratio = float(c_val) / float(g_val)
            # PIT: the ratio is known only once BOTH inputs are known
            vintage = max(c_vin, g_vin)
            rows_to_write.append(
                (DERIVED_FEATURE_NAME, month, vintage, ratio, pull_date))

        if not rows_to_write:
            return {"ok": False, "n_written": 0,
                    "copper_feature": cu, "gold_feature": au,
                    "message": (
                        f"Found copper ('{cu}') and gold ('{au}') but no "
                        f"overlapping months with usable values "
                        f"({skipped_zero} months skipped for zero gold).")}

        if dry_run:
            return {"ok": True, "n_written": 0,
                    "copper_feature": cu, "gold_feature": au,
                    "months": (rows_to_write[0][1], rows_to_write[-1][1]),
                    "message": (
                        f"DRY RUN: would write {len(rows_to_write)} "
                        f"COPPER_GOLD rows from {cu}/{au}, "
                        f"{rows_to_write[0][1]}..{rows_to_write[-1][1]}. "
                        f"No data written.")}

        # write — INSERT OR REPLACE so re-running is idempotent
        conn.executemany(
            """INSERT OR REPLACE INTO features_monthly
               (feature_name, observation_month, vintage_date, value,
                source_pull_date)
               VALUES (?, ?, ?, ?, ?)""",
            rows_to_write,
        )
        conn.commit()

        return {"ok": True, "n_written": len(rows_to_write),
                "copper_feature": cu, "gold_feature": au,
                "months": (rows_to_write[0][1], rows_to_write[-1][1]),
                "message": (
                    f"Wrote {len(rows_to_write)} COPPER_GOLD rows "
                    f"({rows_to_write[0][1]}..{rows_to_write[-1][1]}) "
                    f"from {cu} / {au}."
                    + (f" Skipped {skipped_zero} months (zero gold value)."
                       if skipped_zero else ""))}
    finally:
        conn.close()
