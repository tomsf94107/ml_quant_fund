"""
Ingestion orchestrator.

Pulls all FRED-available series, transforms them (frequency conversion,
derivations), stamps vintage_dates appropriately, and writes to
features_monthly / targets_monthly in recession.db.

Each ingestion run creates a `runs` row of type 'backfill' (or 'monthly_predict'
for incremental updates). All writes are inside a single transaction per run.

CLI:
    python -m recession.data.ingest --validate          # check series IDs exist
    python -m recession.data.ingest --backfill          # full historical pull
    python -m recession.data.ingest --update            # incremental (latest data only)
    python -m recession.data.ingest --series T10Y3M     # one feature only
    python -m recession.data.ingest --refresh           # bypass cache
    python -m recession.data.ingest --dry-run           # don't write to DB
    python -m recession.data.ingest --cache-stats       # report cache size
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:                                      # py < 3.9
    from backports.zoneinfo import ZoneInfo              # type: ignore

# Make package importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from recession.data.fred_client import (                  # noqa: E402
    FredClient, FredSeriesNotFoundError, FredApiError,
)
from recession.data.series_specs import (                 # noqa: E402
    SERIES_SPECS, TARGET_SPECS, SeriesSpec, TargetSpec,
    fred_series_ids_to_fetch, get_spec,
)
from recession.data.manual_sources import MANUAL_FETCHERS  # noqa: E402

logger = logging.getLogger(__name__)

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "recession.db"


# =============================================================================
# Helpers
# =============================================================================

def vn_now_iso() -> str:
    return datetime.now(VN_TZ).isoformat(timespec="seconds")


def today_iso() -> str:
    return date.today().isoformat()


def month_floor(d: str) -> str:
    """'2008-05-12' -> '2008-05-01'."""
    return d[:7] + "-01"


def add_days_iso(d: str, days: int) -> str:
    return (date.fromisoformat(d) + timedelta(days=days)).isoformat()


# =============================================================================
# Frequency conversion
# =============================================================================

def aggregate_to_monthly(
    obs: list[dict],
    method: str,
) -> list[dict]:
    """
    Convert daily/weekly observations to monthly using one of:
      - 'eop' = end-of-period: take last non-null value of each month
      - 'avg' = mean of non-null values in the month
      - 'sum' = sum of non-null values in the month

    Input:  [{'date': 'YYYY-MM-DD', 'value': float|None}, ...]
            Optionally with 'vintage_date' (preserved through eop/avg).
    Output: [{'date': 'YYYY-MM-01', 'value': float|None, 'vintage_date': str}, ...]

    For monthly data, this is essentially a passthrough that normalizes dates.
    """
    if not obs:
        return []

    # Bucket by YYYY-MM
    buckets: dict[str, list[dict]] = {}
    for o in obs:
        if o.get("value") is None:
            continue
        ym = o["date"][:7]
        buckets.setdefault(ym, []).append(o)

    out = []
    for ym in sorted(buckets):
        rows = buckets[ym]
        if method == "eop":
            # Last (chronologically) non-null
            latest = max(rows, key=lambda r: r["date"])
            out.append({
                "date":         f"{ym}-01",
                "value":        latest["value"],
                "vintage_date": latest.get("vintage_date", latest["date"]),
            })
        elif method == "avg":
            avg_val = sum(r["value"] for r in rows) / len(rows)
            # Use the latest vintage in the month (most conservative)
            latest_vintage = max(
                (r.get("vintage_date", r["date"]) for r in rows),
                default=f"{ym}-01",
            )
            out.append({
                "date":         f"{ym}-01",
                "value":        avg_val,
                "vintage_date": latest_vintage,
            })
        elif method == "sum":
            sum_val = sum(r["value"] for r in rows)
            latest_vintage = max(
                (r.get("vintage_date", r["date"]) for r in rows),
                default=f"{ym}-01",
            )
            out.append({
                "date":         f"{ym}-01",
                "value":        sum_val,
                "vintage_date": latest_vintage,
            })
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    return out


def forward_fill_monthly(obs: list[dict], end: str) -> list[dict]:
    """
    For quarterly series (e.g., SLOOS): repeat the most recent quarterly value
    forward through every monthly slot until the next quarter.

    `end` = ISO date of the last month to fill (inclusive).
    """
    if not obs:
        return []
    obs = sorted(obs, key=lambda r: r["date"])

    # End boundary should be the month AFTER end_ym so that the last
    # quarterly value propagates through end_ym inclusive.
    end_d = date.fromisoformat(end)
    if end_d.month == 12:
        end_boundary = date(end_d.year + 1, 1, 1)
    else:
        end_boundary = date(end_d.year, end_d.month + 1, 1)

    out = []
    for i, row in enumerate(obs):
        next_date = (
            obs[i + 1]["date"] if i + 1 < len(obs)
            else end_boundary.isoformat()
        )
        # Generate monthly fills from row['date'] up to next_date (exclusive)
        cur = date.fromisoformat(row["date"])
        nxt = date.fromisoformat(next_date)
        while cur < nxt:
            out.append({
                "date":         cur.isoformat(),
                "value":        row["value"],
                "vintage_date": row.get("vintage_date", row["date"]),
            })
            # advance one month
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)
    return out


# =============================================================================
# DB writers
# =============================================================================

def insert_features(
    conn: sqlite3.Connection,
    feature_name: str,
    rows: list[dict],
    pull_date: str,
) -> int:
    """
    Insert vintage rows into features_monthly. Uses INSERT OR IGNORE so
    re-running ingest doesn't duplicate existing (feature, month, vintage) rows.

    Returns count actually inserted.
    """
    if not rows:
        return 0
    sql = """
        INSERT OR IGNORE INTO features_monthly
            (feature_name, observation_month, vintage_date, value, source_pull_date)
        VALUES (?, ?, ?, ?, ?)
    """
    payload = [
        (feature_name, month_floor(r["date"]), r["vintage_date"], r["value"], pull_date)
        for r in rows
    ]
    cur = conn.executemany(sql, payload)
    return cur.rowcount


def insert_target(
    conn: sqlite3.Connection,
    target_id: str,
    rows: list[dict],
) -> int:
    """
    Insert into targets_monthly.

    rows expected: [{'date', 'announcement_date', 'label', 'notes'?}]
    """
    if not rows:
        return 0
    sql = """
        INSERT OR IGNORE INTO targets_monthly
            (target_id, observation_month, announcement_date, label, notes)
        VALUES (?, ?, ?, ?, ?)
    """
    payload = [
        (target_id, month_floor(r["date"]), r["announcement_date"],
         int(r["label"]), r.get("notes"))
        for r in rows
    ]
    cur = conn.executemany(sql, payload)
    return cur.rowcount


def start_run(conn: sqlite3.Connection, run_type: str, notes: str = "") -> str:
    run_id = f"{vn_now_iso()}_{run_type}_{uuid.uuid4().hex[:6]}"
    conn.execute(
        """INSERT INTO runs (run_id, run_timestamp, run_type, status, spec_version, notes)
           VALUES (?, ?, ?, 'in_progress', 'v1.0', ?)""",
        (run_id, vn_now_iso(), run_type, notes),
    )
    return run_id


def finalize_run(
    conn: sqlite3.Connection,
    run_id: str,
    status: str,
    n_features: int,
    n_obs: int,
    error: Optional[str] = None,
) -> None:
    conn.execute(
        """UPDATE runs SET status=?, n_features=?, n_observations=?, error_message=?
           WHERE run_id=?""",
        (status, n_features, n_obs, error, run_id),
    )


# =============================================================================
# Per-feature ingest
# =============================================================================

def ingest_feature(
    client: FredClient,
    conn: sqlite3.Connection,
    spec: SeriesSpec,
    start: str,
    end: str,
    pull_date: str,
    force_refresh: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Ingest a single feature: pull from FRED, transform, write to DB.

    Returns summary dict: {feature_name, fetch_method, n_raw, n_monthly, n_inserted}
    """
    summary = {
        "feature_name": spec.feature_name,
        "fetch_method": spec.fetch_method,
        "n_raw":       0,
        "n_monthly":   0,
        "n_inserted":  0,
        "status":      "ok",
        "error":       None,
    }

    try:
        # ----- 1. Pull raw observations from FRED -----
        if spec.fetch_method == "fred_latest":
            raw = client.observations(
                spec.fred_series_id, start=start, end=end,
                force_refresh=force_refresh,
            )
            # Stamp vintage_date = observation_date + publication_lag_days
            for r in raw:
                r["vintage_date"] = add_days_iso(r["date"], spec.publication_lag_days)

        elif spec.fetch_method == "fred_alfred":
            raw = client.observations_all_vintages(
                spec.fred_series_id, start=start, end=end,
                force_refresh=force_refresh,
            )
            # vintage_date is already set by the API response

        elif spec.fetch_method == "derived":
            raw = ingest_derived(client, spec, start, end, force_refresh)

        elif spec.fetch_method == "manual":
            fetcher = MANUAL_FETCHERS.get(spec.feature_name)
            if fetcher is None:
                summary["status"] = "skipped"
                summary["error"]  = f"No manual fetcher registered for {spec.feature_name}"
                logger.info("SKIP %-22s — no manual fetcher", spec.feature_name)
                return summary
            raw = fetcher(force_refresh=force_refresh)
            # Filter by date window (manual sources may return full history)
            raw = [r for r in raw if start <= r["date"] <= end]

        elif spec.fetch_method == "skip_v1":
            summary["status"] = "skipped"
            summary["error"]  = f"fetch_method={spec.fetch_method}; not handled in v1"
            logger.info("SKIP %-22s — %s", spec.feature_name, summary["error"])
            return summary

        else:
            raise ValueError(f"Unknown fetch_method: {spec.fetch_method}")

        summary["n_raw"] = len(raw)

        # ----- 2. Aggregate to monthly -----
        if spec.native_frequency == "quarterly":
            # First aggregate within quarter (eop), then forward-fill to monthly
            monthly = aggregate_to_monthly(raw, "eop")
            monthly = forward_fill_monthly(monthly, end=end)
        elif spec.native_frequency == "monthly":
            monthly = aggregate_to_monthly(raw, "eop")     # passthrough/normalize
        else:
            monthly = aggregate_to_monthly(raw, spec.aggregation)

        summary["n_monthly"] = len(monthly)

        # ----- 3. Write to DB -----
        if not dry_run:
            n_inserted = insert_features(conn, spec.feature_name, monthly, pull_date)
            summary["n_inserted"] = n_inserted

        logger.info(
            "OK   %-22s %-12s raw=%4d monthly=%4d inserted=%4d",
            spec.feature_name, spec.fetch_method,
            summary["n_raw"], summary["n_monthly"], summary["n_inserted"],
        )

    except FredSeriesNotFoundError as e:
        summary["status"] = "not_found"
        summary["error"]  = str(e)
        logger.error("MISS %-22s — %s", spec.feature_name, e)
    except FredApiError as e:
        summary["status"] = "api_error"
        summary["error"]  = str(e)
        logger.error("FAIL %-22s — %s", spec.feature_name, e)
    except Exception as e:
        summary["status"] = "error"
        summary["error"]  = f"{type(e).__name__}: {e}"
        logger.exception("FAIL %-22s — unexpected error", spec.feature_name)

    return summary


# =============================================================================
# Derived series
# =============================================================================

def ingest_derived(
    client: FredClient,
    spec: SeriesSpec,
    start: str,
    end: str,
    force_refresh: bool,
) -> list[dict]:
    """
    Compute a derived series from its inputs. Each derived series gets a
    bespoke implementation here.

    Returns list of {date, value, vintage_date}.
    """
    if spec.feature_name == "REAL_FFR_GAP":
        # v1 simple version: DFF - MICH (Michigan inflation expectations).
        # MICH is monthly; DFF is daily — average DFF within the month.
        dff  = client.observations("DFF",  start=start, end=end,
                                   force_refresh=force_refresh)
        mich = client.observations("MICH", start=start, end=end,
                                   force_refresh=force_refresh)
        dff_monthly  = aggregate_to_monthly(dff,  "avg")
        mich_monthly = aggregate_to_monthly(mich, "eop")

        mich_by_month = {r["date"]: r["value"] for r in mich_monthly}
        out = []
        for d in dff_monthly:
            mv = mich_by_month.get(d["date"])
            if mv is None or d["value"] is None:
                continue
            out.append({
                "date":         d["date"],
                "value":        d["value"] - mv,
                "vintage_date": add_days_iso(d["date"], spec.publication_lag_days),
            })
        return out

    if spec.feature_name == "COPPER_GOLD":
        copper = client.observations("PCOPPUSDM", start=start, end=end,
                                     force_refresh=force_refresh)
        gold   = client.observations("GOLDPMGBD228NLBM", start=start, end=end,
                                     force_refresh=force_refresh)
        copper_m = aggregate_to_monthly(copper, "eop")
        gold_m   = aggregate_to_monthly(gold,   "eop")

        gold_by_month = {r["date"]: r["value"] for r in gold_m}
        out = []
        for c in copper_m:
            g = gold_by_month.get(c["date"])
            if g is None or c["value"] is None or g == 0:
                continue
            out.append({
                "date":         c["date"],
                "value":        c["value"] / g,
                "vintage_date": add_days_iso(c["date"], spec.publication_lag_days),
            })
        return out

    if spec.feature_name == "NAPMPI":
        # 3-region Fed manufacturing composite (substitute for ISM PMI).
        # Pull all 3 regional series and average whichever are available
        # at each month. Pre-2001 = Philly only; 2001-2003 = +Empire;
        # 2004+ = +Dallas. Auto-degrades cleanly.
        philly  = client.observations("GACDFSA066MSFRBPHI", start=start, end=end,
                                       force_refresh=force_refresh)
        empire  = client.observations("GACDISA066MSFRBNY",       start=start, end=end,
                                       force_refresh=force_refresh)
        dallas  = client.observations("BACTSAMFRBDAL",    start=start, end=end,
                                       force_refresh=force_refresh)

        philly_m = aggregate_to_monthly(philly, "eop")
        empire_m = aggregate_to_monthly(empire, "eop")
        dallas_m = aggregate_to_monthly(dallas, "eop")

        philly_by = {r["date"]: r["value"] for r in philly_m if r["value"] is not None}
        empire_by = {r["date"]: r["value"] for r in empire_m if r["value"] is not None}
        dallas_by = {r["date"]: r["value"] for r in dallas_m if r["value"] is not None}

        all_months = sorted(set(philly_by) | set(empire_by) | set(dallas_by))
        out = []
        for m in all_months:
            vals = [v for v in (philly_by.get(m), empire_by.get(m), dallas_by.get(m))
                    if v is not None]
            if not vals:
                continue
            composite = sum(vals) / len(vals)
            out.append({
                "date":         m,
                "value":        composite,
                "vintage_date": add_days_iso(m, spec.publication_lag_days),
            })
        return out

    raise NotImplementedError(f"No derivation for {spec.feature_name}")


# =============================================================================
# Targets
# =============================================================================

def ingest_targets(
    client: FredClient,
    conn: sqlite3.Connection,
    start: str,
    end: str,
    pull_date: str,
    force_refresh: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Ingest T1 (USREC) and T2 (computed drawdown). T3 deferred to Step 3."""
    summaries = []

    # ---- T1: NBER recession ----
    summary_t1 = {"target_id": "T1", "n_inserted": 0, "status": "ok"}
    try:
        usrec = client.observations("USREC", start=start, end=end,
                                    force_refresh=force_refresh)
        rows = [
            {
                "date":              month_floor(o["date"]),
                "announcement_date": add_days_iso(o["date"], 180),
                "label":             1 if (o["value"] or 0) > 0.5 else 0,
                "notes":             "USREC vintage; announcement_date proxied by 180d lag",
            }
            for o in usrec if o["value"] is not None
        ]
        if not dry_run:
            summary_t1["n_inserted"] = insert_target(conn, "T1", rows)
        logger.info("OK   T1 NBER          rows=%4d inserted=%4d",
                    len(rows), summary_t1["n_inserted"])
    except Exception as e:
        summary_t1["status"] = "error"
        summary_t1["error"]  = str(e)
        logger.error("FAIL T1 NBER          — %s", e)
    summaries.append(summary_t1)

    # ---- T2: SPX 15% drawdown ----
    summary_t2 = {"target_id": "T2", "n_inserted": 0, "status": "ok"}
    try:
        sp = client.observations("SP500", start=start, end=end,
                                 force_refresh=force_refresh)
        sp_monthly = aggregate_to_monthly(sp, "eop")
        # Compute 12-month rolling max and check drawdown
        rows = []
        values = [r["value"] for r in sp_monthly]
        for i, r in enumerate(sp_monthly):
            window = values[max(0, i - 11): i + 1]      # trailing 12 months
            if not window or r["value"] is None:
                continue
            rolling_max = max(v for v in window if v is not None)
            drawdown = r["value"] / rolling_max - 1.0
            label = 1 if drawdown <= -0.15 else 0
            rows.append({
                "date":              r["date"],
                "announcement_date": add_days_iso(r["date"], 1),
                "label":             label,
                "notes":             f"drawdown_pct={drawdown:.4f}",
            })
        if not dry_run:
            summary_t2["n_inserted"] = insert_target(conn, "T2", rows)
        n_events = sum(r["label"] for r in rows)
        logger.info("OK   T2 Drawdown      rows=%4d inserted=%4d events=%d",
                    len(rows), summary_t2["n_inserted"], n_events)
    except Exception as e:
        summary_t2["status"] = "error"
        summary_t2["error"]  = str(e)
        logger.error("FAIL T2 Drawdown      — %s", e)
    summaries.append(summary_t2)

    # ---- T3: deferred to Step 3 ----
    logger.info("SKIP T3 AI Kill-Switch — deferred to Step 3 (triggers ingestion)")

    return summaries


# =============================================================================
# Validation
# =============================================================================

def validate_series(client: FredClient) -> dict:
    """Check every FRED series ID we depend on actually exists."""
    ids = fred_series_ids_to_fetch()
    # Also include target series (USREC, SP500 already in features but included for safety)
    target_ids = [t.fred_series_id for t in TARGET_SPECS if t.fred_series_id]
    all_ids = sorted(set(ids) | set(target_ids))

    logger.info("Validating %d FRED series IDs ...", len(all_ids))
    results = client.validate_series_ids(all_ids)
    missing = [sid for sid, ok in results.items() if not ok]
    found   = [sid for sid, ok in results.items() if ok]

    logger.info("Found:   %d / %d", len(found), len(all_ids))
    logger.info("Missing: %s", missing if missing else "(none)")
    return {"found": found, "missing": missing}


# =============================================================================
# Main orchestrator
# =============================================================================

def run_ingest(
    db_path: Path,
    mode: str,
    feature_filter: Optional[list[str]] = None,
    start_override: Optional[str] = None,
    end_override:   Optional[str] = None,
    force_refresh:  bool = False,
    dry_run:        bool = False,
    skip_targets:   bool = False,
) -> dict:
    """
    Top-level ingest orchestration. Returns summary stats.

    mode: 'backfill' (full history), 'update' (last 6 months), 'single' (one feature)
    """
    client = FredClient.from_env()

    # Default date ranges
    if mode == "backfill":
        start = start_override or "1960-01-01"
        end   = end_override   or today_iso()
    elif mode == "update":
        start = start_override or add_days_iso(today_iso(), -180)
        end   = end_override   or today_iso()
    else:  # single / explicit
        start = start_override or "1960-01-01"
        end   = end_override   or today_iso()

    pull_date = today_iso()

    # Pick specs to run
    specs_to_run = SERIES_SPECS
    if feature_filter:
        specs_to_run = [s for s in SERIES_SPECS if s.feature_name in feature_filter]
        if not specs_to_run:
            raise ValueError(f"No specs match filter: {feature_filter}")

    logger.info("=" * 70)
    logger.info("INGEST  mode=%s  start=%s  end=%s  features=%d  dry_run=%s",
                mode, start, end, len(specs_to_run), dry_run)
    logger.info("=" * 70)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row

    run_id = start_run(conn, "backfill" if mode == "backfill" else "monthly_predict",
                       notes=f"mode={mode} filter={feature_filter}")
    conn.commit()

    summaries = []
    try:
        # Features
        for spec in specs_to_run:
            s = ingest_feature(
                client, conn, spec, start, end, pull_date,
                force_refresh=force_refresh, dry_run=dry_run,
            )
            summaries.append(s)
            if not dry_run:
                conn.commit()

        # Targets (skip if filtering specific features)
        target_summaries = []
        if not feature_filter and not skip_targets:
            target_summaries = ingest_targets(
                client, conn, start, end, pull_date,
                force_refresh=force_refresh, dry_run=dry_run,
            )
            if not dry_run:
                conn.commit()

        # Run summary
        n_ok      = sum(1 for s in summaries if s["status"] == "ok")
        n_skipped = sum(1 for s in summaries if s["status"] == "skipped")
        n_failed  = sum(1 for s in summaries if s["status"] not in ("ok", "skipped"))
        n_obs     = sum(s["n_inserted"] for s in summaries)
        n_obs    += sum(s["n_inserted"] for s in target_summaries)

        finalize_run(conn, run_id, "success" if n_failed == 0 else "partial",
                     n_features=n_ok, n_obs=n_obs)
        conn.commit()

        logger.info("=" * 70)
        logger.info("DONE  ok=%d  skipped=%d  failed=%d  rows_written=%d",
                    n_ok, n_skipped, n_failed, n_obs)
        logger.info("=" * 70)

        return {
            "run_id":     run_id,
            "summaries":  summaries,
            "targets":    target_summaries,
            "n_ok":       n_ok,
            "n_skipped":  n_skipped,
            "n_failed":   n_failed,
            "n_obs":      n_obs,
        }
    except Exception as e:
        finalize_run(conn, run_id, "failed", 0, 0, error=str(e))
        conn.commit()
        raise
    finally:
        conn.close()


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest FRED data into recession.db")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--backfill",   action="store_true", help="Full historical pull (default)")
    mode.add_argument("--update",     action="store_true", help="Incremental: last 180 days")
    mode.add_argument("--validate",   action="store_true", help="Check all series IDs exist")
    mode.add_argument("--cache-stats", action="store_true", help="Report cache size")

    parser.add_argument("--series",   nargs="+", help="Specific feature names to ingest")
    parser.add_argument("--start",    help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end",      help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--refresh",  action="store_true", help="Bypass cache")
    parser.add_argument("--dry-run",  action="store_true", help="Don't write to DB")
    parser.add_argument("--skip-targets", action="store_true",
                        help="Skip T1/T2 ingestion (features only)")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.cache_stats:
        client = FredClient.from_env()
        stats = client.cache_stats()
        print(f"Cache: {stats['n_files']} files, "
              f"{stats['total_bytes'] / 1024:.1f} KB at {stats.get('cache_dir')}")
        return 0

    if args.validate:
        client = FredClient.from_env()
        results = validate_series(client)
        return 0 if not results["missing"] else 1

    mode_str = "update" if args.update else ("single" if args.series else "backfill")
    try:
        run_ingest(
            db_path=args.db,
            mode=mode_str,
            feature_filter=args.series,
            start_override=args.start,
            end_override=args.end,
            force_refresh=args.refresh,
            dry_run=args.dry_run,
            skip_targets=args.skip_targets,
        )
        return 0
    except Exception as e:
        logger.exception("Ingest failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
