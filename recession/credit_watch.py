"""
recession/credit_watch.py

Credit-watch — a monitoring panel for credit-market stress.

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
Experiment A (see A_PREREGISTRATION.md, Amendment 1) found the excess
bond premium (EBP) is regime-conditional: it predicts credit-driven
recessions (2008) and mispredicts non-credit / Fed-suppressed ones
(2020). The robustness check therefore correctly kept EBP OUT of the
recession models (M1-M5).

But EBP is still informative as a *monitored indicator* — a rising EBP is
a genuine real-time signal that credit-market stress is building, even
though it cannot be trusted as a model input. This is the same predict-
vs-monitor distinction the recession alert system embodies: some signals
belong on a dashboard, not in a model.

`credit_watch` is that dashboard. It is NOT a predictor, NOT wired into
any model, and produces NO recession probability. It reports the current
state of three credit-stress gauges so a human can see credit conditions
at a glance.

THE THREE GAUGES
----------------
1. EBP — the excess bond premium. Public-market credit risk appetite.
   Read from features_monthly (already ingested, manual Fed-CSV source).
2. BAA10Y — Moody's Baa corporate yield minus the 10-year Treasury.
   Public credit-risk spread. Read from features_monthly.
3. PRIVATE_CREDIT_DEFAULT_RATE — the direct-lending default rate. This
   is the gauge for the *private* credit risk that EBP and BAA10Y, being
   public-market measures, are structurally BLIND to (private credit is
   opaque, illiquid, not marked to market). There is no free public feed
   for it; it is maintained MANUALLY, updated ~quarterly from sources
   such as the Financial Stability Board's private-credit vulnerability
   reports and direct-lending default-rate estimates published by major
   banks' credit-strategy teams.

THE STALENESS FLAG — the self-reminder
--------------------------------------
Because the private-credit figure is manual, it can silently go stale.
`credit_watch` checks the age of that entry every time it runs: if it is
older than PRIVATE_CREDIT_STALE_DAYS, the panel prints a STALE warning
telling the user to update it. This replaces an external calendar
reminder — the tool reminds you itself, and it cannot be forgotten.

HONEST LIMITS
-------------
- EBP and BAA10Y are PUBLIC-market gauges. They can miss private-credit
  stress entirely until it spills into public markets. The panel states
  this; it is the reason gauge 3 exists.
- The thresholds below are monitoring bands for human attention, NOT
  validated prediction thresholds. Crossing one means "look", not
  "a recession is coming".
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional


# -- monitoring bands ------------------------------------------------------
# Attention thresholds for a human reader. NOT prediction thresholds.
# EBP: its historical mean is ~0; sustained positive readings indicate
# credit-market stress. ~0.5 and ~1.0 are widely-watched attention levels.
EBP_BANDS = [(0.5, "ELEVATED"), (1.0, "STRESSED")]
# BAA10Y: the Baa-Treasury spread. ~2.5pp normal-ish, ~3.5pp+ stressed.
BAA10Y_BANDS = [(3.0, "ELEVATED"), (3.5, "STRESSED")]
# Private-credit direct-lending default rate (percent). ~3% has been
# "normal"; bank credit-strategy desks flag ~8% as approaching COVID-era
# peaks. These are the human-attention bands.
PRIVATE_CREDIT_BANDS = [(5.0, "ELEVATED"), (8.0, "STRESSED")]

# the manual private-credit figure is considered stale after this many
# days — it should be refreshed ~quarterly from FSB / bank reports
PRIVATE_CREDIT_STALE_DAYS = 100

EBP_FEATURE = "EBP"
BAA10Y_FEATURE = "BAA10Y"


def _private_credit_path() -> Path:
    """The manually-maintained private-credit figure (JSON)."""
    return Path(__file__).resolve().parent / "private_credit_manual.json"


def _band_for(value: Optional[float], bands: list[tuple]) -> str:
    """The highest attention band `value` falls into, or 'normal'."""
    if value is None:
        return "n/a"
    label = "normal"
    for thr, lab in bands:
        if value >= thr:
            label = lab
    return label


def _latest_feature_value(conn: sqlite3.Connection,
                          feature: str) -> Optional[dict]:
    """The latest-vintage value for the latest observation_month of a
    feature in features_monthly. Returns {value, observation_month,
    vintage_date} or None."""
    row = conn.execute(
        """SELECT f.value, f.observation_month, f.vintage_date
           FROM features_monthly f
           INNER JOIN (
               SELECT observation_month, MAX(vintage_date) AS mv
               FROM features_monthly WHERE feature_name = ?
               GROUP BY observation_month
           ) latest
             ON f.observation_month = latest.observation_month
             AND f.vintage_date = latest.mv
           WHERE f.feature_name = ?
           ORDER BY f.observation_month DESC
           LIMIT 1""",
        (feature, feature),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return {"value": float(row[0]), "observation_month": row[1],
            "vintage_date": row[2]}


def _read_private_credit(path: Path) -> Optional[dict]:
    """The manually-maintained private-credit default-rate entry.

    Expected JSON: {"default_rate_pct": <float>, "as_of": "YYYY-MM-DD",
                    "source": "<text>", "notes": "<text>"}.
    Returns the dict (with an added 'age_days') or None if absent/invalid.
    """
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        rate = float(data["default_rate_pct"])
        as_of = datetime.strptime(data["as_of"], "%Y-%m-%d").date()
        age_days = (date.today() - as_of).days
        return {"default_rate_pct": rate, "as_of": data["as_of"],
                "source": data.get("source", ""),
                "notes": data.get("notes", ""),
                "age_days": age_days}
    except Exception:
        return None


def write_private_credit_figure(
    default_rate_pct: float,
    as_of: str,
    source: str,
    notes: str = "",
    *,
    path: Optional[Path] = None,
) -> dict:
    """Update the manually-maintained private-credit default-rate figure.

    Call this ~quarterly with the latest direct-lending default rate from
    an FSB report or a bank credit-strategy publication.

      default_rate_pct : the direct-lending default rate, in percent.
      as_of            : 'YYYY-MM-DD' — the date the figure refers to.
      source           : where the number came from (report name/date).
    """
    path = path or _private_credit_path()
    # validate the date
    datetime.strptime(as_of, "%Y-%m-%d")
    payload = {"default_rate_pct": float(default_rate_pct),
               "as_of": as_of, "source": source, "notes": notes,
               "updated": datetime.now().isoformat(timespec="seconds")}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return {"ok": True, "message": f"private-credit figure set to "
            f"{default_rate_pct}% as of {as_of}", "path": str(path)}


def credit_watch(
    db_path: Optional[Path] = None,
    *,
    private_credit_path: Optional[Path] = None,
) -> dict:
    """Assemble the credit-watch panel: the three gauges + staleness.

    Returns {'ebp', 'baa10y', 'private_credit', 'private_credit_stale',
             'overall', 'generated'} — a monitoring snapshot, NOT a
    prediction.
    """
    db_path = db_path or (Path(__file__).resolve().parent.parent
                          / "recession.db")
    private_credit_path = private_credit_path or _private_credit_path()

    conn = sqlite3.connect(db_path)
    try:
        ebp_raw = _latest_feature_value(conn, EBP_FEATURE)
        baa_raw = _latest_feature_value(conn, BAA10Y_FEATURE)
    finally:
        conn.close()

    ebp = None
    if ebp_raw is not None:
        ebp = {**ebp_raw,
               "band": _band_for(ebp_raw["value"], EBP_BANDS)}
    baa = None
    if baa_raw is not None:
        baa = {**baa_raw,
               "band": _band_for(baa_raw["value"], BAA10Y_BANDS)}

    pc = _read_private_credit(private_credit_path)
    pc_stale = False
    if pc is not None:
        pc["band"] = _band_for(pc["default_rate_pct"],
                               PRIVATE_CREDIT_BANDS)
        pc_stale = pc["age_days"] > PRIVATE_CREDIT_STALE_DAYS

    # overall: the highest band across whichever gauges are available
    rank = {"normal": 0, "ELEVATED": 1, "STRESSED": 2, "n/a": -1}
    bands_present = [g["band"] for g in (ebp, baa, pc) if g is not None]
    if not bands_present:
        overall = "n/a"
    else:
        overall = max(bands_present, key=lambda b: rank.get(b, -1))

    return {
        "ebp": ebp, "baa10y": baa,
        "private_credit": pc,
        "private_credit_stale": pc_stale,
        "overall": overall,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def print_credit_watch(panel: dict) -> None:
    """Print the credit-watch panel."""
    print("=" * 70)
    print("CREDIT WATCH — credit-market stress monitor")
    print("=" * 70)
    print("  A MONITORING panel, not a predictor. No recession")
    print("  probability is produced here. Bands flag human attention.")
    print()

    def _line(name, gauge, unit):
        if gauge is None:
            print(f"  {name:>26}: no data")
            return
        print(f"  {name:>26}: {gauge['value']:.3f}{unit}  "
              f"[{gauge['band']}]  (obs {gauge['observation_month']})")

    _line("EBP (excess bond premium)", panel["ebp"], "")
    _line("BAA10Y (Baa-Treasury sprd)", panel["baa10y"], " pp")

    pc = panel["private_credit"]
    if pc is None:
        print(f"  {'private-credit default':>26}: no manual figure set — "
              f"use write_private_credit_figure()")
    else:
        print(f"  {'private-credit default':>26}: "
              f"{pc['default_rate_pct']:.1f}%  [{pc['band']}]  "
              f"(as of {pc['as_of']}, {pc['age_days']}d ago)")
        if panel["private_credit_stale"]:
            print()
            print(f"  *** STALE: the private-credit figure is "
                  f"{pc['age_days']} days old (> {PRIVATE_CREDIT_STALE_DAYS}).")
            print(f"      Update it from the latest FSB / bank credit-")
            print(f"      strategy report via write_private_credit_figure().")

    print()
    print(f"  OVERALL credit-stress band: {panel['overall']}")
    print()
    print("  Note: EBP and BAA10Y are PUBLIC-market gauges — they can")
    print("  miss private-credit stress until it reaches public markets.")
    print("  The private-credit gauge exists to cover that blind spot.")
    print("=" * 70)
