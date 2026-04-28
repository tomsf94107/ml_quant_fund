"""
Tests for the FRED client and ingestion logic.

Two modes:
  - Default (offline): tests transformations, cache, rate limiter, schema integration.
    Uses synthetic / mocked data — no network required.
  - --live: also runs end-to-end tests that hit the real FRED API.
    Requires FRED_API_KEY in env or .env.

Run:
    python -m recession.tests.test_data                   # offline only
    python -m recession.tests.test_data --live            # offline + live
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

# Make package importable when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from recession.db.migrate import run_migration                                   # noqa: E402
from recession.data.fred_client import (                                         # noqa: E402
    FredClient, JsonFileCache, TokenBucket,
    FredApiError, FredSeriesNotFoundError,
)
from recession.data.ingest import (                                              # noqa: E402
    aggregate_to_monthly, forward_fill_monthly, ingest_targets, run_ingest,
    add_days_iso, month_floor,
)
from recession.data.series_specs import (                                        # noqa: E402
    SERIES_SPECS, SPECS_BY_NAME, fred_series_ids_to_fetch,
    specs_by_method,
)


# -----------------------------------------------------------------------------
# Test infrastructure
# -----------------------------------------------------------------------------

_passed: list[str] = []
_failed: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        _passed.append(name)
        print(f"  PASS  {name}")
    else:
        _failed.append((name, detail))
        print(f"  FAIL  {name}  ({detail})")


# =============================================================================
# Offline tests
# =============================================================================

def test_series_specs() -> None:
    print("\n[1] Series specs sanity")

    check("21_specs_total", len(SERIES_SPECS) == 21, f"got {len(SERIES_SPECS)}")
    check("specs_match_registry_names",
          {s.feature_name for s in SERIES_SPECS} == set(SPECS_BY_NAME.keys()))

    # Every spec with fetch_method=fred_* must have fred_series_id
    for spec in SERIES_SPECS:
        if spec.fetch_method in ("fred_latest", "fred_alfred"):
            check(f"{spec.feature_name}_has_fred_id",
                  spec.fred_series_id is not None,
                  f"{spec.feature_name} fetch={spec.fetch_method} but no fred_series_id")

    # Every derived spec must have derived_from
    for spec in SERIES_SPECS:
        if spec.fetch_method == "derived":
            check(f"{spec.feature_name}_has_derived_from",
                  len(spec.derived_from) > 0)

    # FRED IDs to fetch must include derived dependencies
    ids = fred_series_ids_to_fetch()
    check("fetch_list_includes_DFF",   "DFF" in ids,   "REAL_FFR_GAP needs DFF")
    check("fetch_list_includes_MICH",  "MICH" in ids,  "REAL_FFR_GAP needs MICH")
    # COPPER_GOLD dropped to skip_v1 (both AM and PM London gold series gone from FRED).
    # Defer to v2 with Yahoo Finance gold source. Hence neither copper nor gold IDs
    # should appear in the FRED fetch list.
    check("fetch_list_excludes_PCOPPUSDM", "PCOPPUSDM" not in ids,
          "COPPER_GOLD now skip_v1; PCOPPUSDM should not be in FRED fetch list")
    check("fetch_list_excludes_AM_gold", "GOLDAMGBD228NLBM" not in ids,
          "AM gold fix is discontinued")
    check("fetch_list_excludes_PM_gold", "GOLDPMGBD228NLBM" not in ids,
          "PM gold fix is also discontinued from FRED")
    check("fetch_list_includes_3_regional_PMIs",
          all(s in ids for s in ("GACDFSA066MSFRBPHI", "GACDISA066MSFRBNY", "BACTSAMFRBDAL")),
          "NAPMPI composite needs 3 regional Fed mfg surveys")
    check("fetch_list_excludes_NAPMPI",  "NAPMPI" not in ids,
          "ISM NAPMPI no longer available; should not be in fetch list")
    check("fetch_list_excludes_EBP",     "EBP" not in ids,
          "EBP is manual (Fed Board CSV); should not be in FRED fetch list")

    # Skip count: 4 in v1 (3 original + COPPER_GOLD after gold series discontinued)
    skipped = specs_by_method("skip_v1")
    check("4_specs_marked_skip_v1", len(skipped) == 4, f"got {len(skipped)}")

    # Manual count: EBP only in v1
    manual = specs_by_method("manual")
    check("1_spec_marked_manual", len(manual) == 1, f"got {len(manual)}")
    check("manual_spec_is_EBP",
          manual[0].feature_name == "EBP" if manual else False)


def test_aggregate_to_monthly() -> None:
    print("\n[2] Frequency conversion: aggregate_to_monthly")

    daily = [
        {"date": "2020-01-15", "value": 1.0},
        {"date": "2020-01-31", "value": 2.0},      # last of January
        {"date": "2020-02-10", "value": 3.0},
        {"date": "2020-02-28", "value": 4.0},      # last of February
        {"date": "2020-03-15", "value": None},     # nulls dropped
    ]

    # eop = end-of-period (last value)
    monthly_eop = aggregate_to_monthly(daily, "eop")
    check("eop_returns_2_months", len(monthly_eop) == 2)
    check("eop_jan_value_is_2",  monthly_eop[0]["value"] == 2.0,
          f"got {monthly_eop[0]['value']}")
    check("eop_feb_value_is_4",  monthly_eop[1]["value"] == 4.0,
          f"got {monthly_eop[1]['value']}")
    check("eop_dates_normalized_to_first",
          monthly_eop[0]["date"] == "2020-01-01" and monthly_eop[1]["date"] == "2020-02-01")

    # avg = mean
    monthly_avg = aggregate_to_monthly(daily, "avg")
    check("avg_jan_value_is_1.5", abs(monthly_avg[0]["value"] - 1.5) < 1e-9)
    check("avg_feb_value_is_3.5", abs(monthly_avg[1]["value"] - 3.5) < 1e-9)

    # sum
    monthly_sum = aggregate_to_monthly(daily, "sum")
    check("sum_jan_value_is_3", monthly_sum[0]["value"] == 3.0)

    # All-null month is dropped
    all_null = [{"date": "2020-01-15", "value": None}]
    check("all_null_month_dropped", aggregate_to_monthly(all_null, "eop") == [])


def test_forward_fill() -> None:
    print("\n[3] Quarterly forward-fill")

    quarterly = [
        {"date": "2020-01-01", "value": 10.0, "vintage_date": "2020-04-15"},
        {"date": "2020-04-01", "value": 12.0, "vintage_date": "2020-07-15"},
    ]
    monthly = forward_fill_monthly(quarterly, end="2020-08-31")

    # Should fill: Jan, Feb, Mar with 10.0; Apr, May, Jun with 12.0; Jul, Aug with 12.0
    by_month = {r["date"]: r["value"] for r in monthly}
    check("forward_fill_jan_is_10", by_month.get("2020-01-01") == 10.0)
    check("forward_fill_mar_is_10", by_month.get("2020-03-01") == 10.0)
    check("forward_fill_apr_is_12", by_month.get("2020-04-01") == 12.0)
    check("forward_fill_jul_is_12", by_month.get("2020-07-01") == 12.0)
    check("forward_fill_n_months",  len(monthly) == 8, f"got {len(monthly)}")


def test_token_bucket() -> None:
    print("\n[4] Rate limiter (TokenBucket)")

    # 600 per minute = 10 per second; should allow ~10 acquires fast
    bucket = TokenBucket(rate_per_minute=600)
    t0 = time.monotonic()
    for _ in range(10):
        bucket.acquire()
    elapsed = time.monotonic() - t0
    check("10_acquires_under_1s_at_600rpm", elapsed < 1.0, f"took {elapsed:.2f}s")

    # 60 per minute = 1 per second; 3 acquires should take ~ 0–2s (first is free)
    bucket = TokenBucket(rate_per_minute=60)
    bucket.tokens = 1                                       # only one available
    t0 = time.monotonic()
    bucket.acquire()                                        # first — instant
    bucket.acquire()                                        # second — must wait ~1s
    elapsed = time.monotonic() - t0
    check("rate_limiter_blocks_when_empty",
          0.5 < elapsed < 2.5, f"took {elapsed:.2f}s, expected ~1s")


def test_json_cache() -> None:
    print("\n[5] JSON file cache")

    with tempfile.TemporaryDirectory() as tmp:
        cache = JsonFileCache(Path(tmp))

        # Miss on empty cache
        check("cache_miss_returns_none",
              cache.get("https://x.com/a", {"q": 1}) is None)

        # Put then get
        cache.put("https://x.com/a", {"q": 1, "api_key": "secret"}, {"value": 42})
        got = cache.get("https://x.com/a", {"q": 1, "api_key": "secret"})
        check("cache_roundtrip", got == {"value": 42})

        # api_key not in key (should hit even if api_key changes)
        got2 = cache.get("https://x.com/a", {"q": 1, "api_key": "different"})
        check("cache_ignores_api_key", got2 == {"value": 42})

        # Different params → different key
        got3 = cache.get("https://x.com/a", {"q": 2})
        check("cache_misses_on_different_params", got3 is None)

        # Stats
        stats = cache.stats()
        check("cache_stats_reports_files", stats["n_files"] >= 1)


def test_fred_client_construction() -> None:
    print("\n[6] FredClient construction")

    # Bad key length → raise
    try:
        FredClient(api_key="too_short")
        check("bad_key_length_rejected", False)
    except ValueError:
        check("bad_key_length_rejected", True)

    # Empty key → raise
    try:
        FredClient(api_key="")
        check("empty_key_rejected", False)
    except ValueError:
        check("empty_key_rejected", True)

    # Valid 32-char hex string
    fake_key = "a" * 32
    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(api_key=fake_key, cache_dir=Path(tmp))
        check("valid_client_constructs",
              client.api_key == fake_key and client.cache.cache_dir == Path(tmp))


def test_mocked_observations() -> None:
    """Test that a mocked FRED response goes through the full pipeline."""
    print("\n[7] Mocked end-to-end: observations -> aggregation -> DB")

    fake_response = {
        "observations": [
            {"date": "2020-01-15", "value": "1.0", "realtime_start": "2020-01-16",
             "realtime_end": "9999-12-31"},
            {"date": "2020-01-31", "value": "2.0", "realtime_start": "2020-02-01",
             "realtime_end": "9999-12-31"},
            {"date": "2020-02-15", "value": "3.0", "realtime_start": "2020-02-16",
             "realtime_end": "9999-12-31"},
            {"date": "2020-02-28", "value": ".",   "realtime_start": "2020-03-01",
             "realtime_end": "9999-12-31"},  # missing
        ]
    }

    with tempfile.TemporaryDirectory() as tmp:
        client = FredClient(api_key="a" * 32, cache_dir=Path(tmp))
        with patch.object(client, "_request", return_value=fake_response):
            obs = client.observations("FAKE_SERIES")
        check("missing_value_converted_to_None",
              obs[3]["value"] is None, f"got {obs[3]['value']}")
        check("valid_value_converted_to_float",
              obs[0]["value"] == 1.0)
        check("4_observations_returned", len(obs) == 4)


def test_ingest_to_real_db() -> None:
    """End-to-end with mocked FRED + real SQLite."""
    print("\n[8] End-to-end ingest: mocked FRED -> real DB schema")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        cache_dir = Path(tmp) / "cache"
        run_migration(db_path)

        # Spoof env so FredClient.from_env() works
        with patch.dict("os.environ", {"FRED_API_KEY": "a" * 32}):
            client = FredClient.from_env(cache_dir=cache_dir)

            fake_t10y3m = {
                "observations": [
                    {"date": "2024-01-15", "value": "0.5"},
                    {"date": "2024-01-31", "value": "0.4"},
                    {"date": "2024-02-15", "value": "0.3"},
                    {"date": "2024-02-29", "value": "0.2"},
                ]
            }

            with patch.object(client, "_request", return_value=fake_t10y3m):
                obs = client.observations("T10Y3M",
                                          start="2024-01-01", end="2024-02-29")

            check("mock_returns_4_obs", len(obs) == 4)

            # Aggregate and write directly through the schema path
            from recession.data.ingest import aggregate_to_monthly, insert_features
            for o in obs:
                o["vintage_date"] = add_days_iso(o["date"], 1)
            monthly = aggregate_to_monthly(obs, "eop")
            check("aggregation_returns_2_months", len(monthly) == 2)

            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            n = insert_features(conn, "T10Y3M", monthly, "2024-03-01")
            conn.commit()
            check("inserted_2_rows", n == 2)

            # Read back
            rows = list(conn.execute(
                """SELECT observation_month, value, vintage_date
                   FROM features_monthly
                   WHERE feature_name='T10Y3M' ORDER BY observation_month"""))
            check("readback_2_rows", len(rows) == 2)
            check("jan_eop_value_is_0.4", rows[0][1] == 0.4)
            check("feb_eop_value_is_0.2", rows[1][1] == 0.2)
            check("vintages_stamped",
                  rows[0][2] is not None and rows[1][2] is not None)
            conn.close()


def test_manual_sources_ebp_parser() -> None:
    """Test the EBP CSV parser without hitting the live URL."""
    print("\n[10] Manual sources: EBP CSV parser")

    from recession.data.manual_sources import fetch_ebp
    from unittest.mock import patch

    # Realistic EBP CSV format
    fake_csv = """date,gz_spread,ebp,est_prob
1973-01-01,1.86,0.42,0.0167
1973-02-01,1.92,0.45,0.0175
2008-09-01,5.62,3.84,0.0521
2008-12-01,8.37,6.12,0.0987
2020-04-01,4.51,2.15,0.0341
2026-03-01,1.28,0.31,0.0142
"""

    with patch("recession.data.manual_sources._cached_download",
               return_value=fake_csv):
        rows = fetch_ebp()

    check("ebp_returns_6_rows", len(rows) == 6, f"got {len(rows)}")
    check("ebp_dates_normalized_to_first",
          all(r["date"].endswith("-01") for r in rows))
    check("ebp_jan1973_value", rows[0]["value"] == 0.42)
    check("ebp_sep2008_crisis_value", rows[2]["value"] == 3.84)
    check("ebp_dec2008_peak_value", rows[3]["value"] == 6.12)
    check("ebp_apr2020_covid_value", abs(rows[4]["value"] - 2.15) < 1e-9)
    check("ebp_vintage_dates_set",
          all(r["vintage_date"] is not None and r["vintage_date"] > r["date"]
              for r in rows))

    # Bad column → returns empty
    bad_csv = """date,wrong_column,whatever
1973-01-01,1.86,0.42"""
    with patch("recession.data.manual_sources._cached_download",
               return_value=bad_csv):
        rows = fetch_ebp()
    check("ebp_missing_column_returns_empty", rows == [])

    # Skip dotted/empty values
    sparse_csv = """date,gz_spread,ebp,est_prob
1973-01-01,1.86,.,0.0167
1973-02-01,1.92,0.45,0.0175
1973-03-01,1.99,,0.0181
"""
    with patch("recession.data.manual_sources._cached_download",
               return_value=sparse_csv):
        rows = fetch_ebp()
    check("ebp_skips_missing_values", len(rows) == 1, f"got {len(rows)}")
    check("ebp_skips_correctly_to_feb", rows[0]["value"] == 0.45 if rows else False)


def test_napmpi_composite_logic() -> None:
    """Test that NAPMPI composite averages whichever regional series are available."""
    print("\n[11] NAPMPI 3-region composite logic")

    from unittest.mock import patch
    from recession.data.ingest import ingest_derived

    fake_responses = {
        # Pre-2001: Philly only
        ("GACDFSA066MSFRBPHI", "1995-01-01"): {"observations":
            [{"date": "1995-01-15", "value": "10.0"}]},
        ("GACDISA066MSFRBNY",       "1995-01-01"): {"observations": []},  # not yet started
        ("BACTSAMFRBDAL",    "1995-01-01"): {"observations": []},  # not yet started

        # Modern: all 3 available
        ("GACDFSA066MSFRBPHI", "2024-01-01"): {"observations":
            [{"date": "2024-01-15", "value": "5.0"}]},
        ("GACDISA066MSFRBNY",       "2024-01-01"): {"observations":
            [{"date": "2024-01-15", "value": "11.0"}]},
        ("BACTSAMFRBDAL",    "2024-01-01"): {"observations":
            [{"date": "2024-01-15", "value": "8.0"}]},
    }

    def fake_request(endpoint, params, force_refresh=False):
        sid = params.get("series_id")
        start = params.get("observation_start", "1995-01-01")
        return fake_responses.get((sid, start),
                                  fake_responses.get((sid, "1995-01-01"),
                                                     {"observations": []}))

    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict("os.environ", {"FRED_API_KEY": "a" * 32}):
            client = FredClient.from_env(cache_dir=Path(tmp))
            spec = SPECS_BY_NAME["NAPMPI"]

            # Test pre-2001 case: only Philly available
            with patch.object(client, "_request", side_effect=fake_request):
                result = ingest_derived(client, spec,
                                        start="1995-01-01", end="1995-01-31",
                                        force_refresh=False)
            check("composite_pre2001_uses_philly_only",
                  len(result) == 1 and result[0]["value"] == 10.0,
                  f"got {result}")

            # Test modern case: all 3 averaged
            with patch.object(client, "_request", side_effect=fake_request):
                result = ingest_derived(client, spec,
                                        start="2024-01-01", end="2024-01-31",
                                        force_refresh=False)
            expected_avg = (5.0 + 11.0 + 8.0) / 3
            check("composite_modern_averages_all_3",
                  len(result) == 1 and abs(result[0]["value"] - expected_avg) < 1e-9,
                  f"got {result}")


def test_drawdown_target_logic() -> None:
    """T2 drawdown computation: rolling 12m max → 15% threshold."""
    print("\n[9] T2 drawdown target logic")

    # SP500 monthly: peak at 100, drawdown 20% to 80
    sp_data = (
        [{"date": f"2023-{m:02d}-15", "value": 100.0 - m} for m in range(1, 13)]   # 12 obs descending
        + [{"date": "2024-01-15", "value": 80.0}]                                  # drawdown ~20%
    )

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        run_migration(db_path)

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")

        with patch.dict("os.environ", {"FRED_API_KEY": "a" * 32}):
            client = FredClient.from_env(cache_dir=Path(tmp) / "cache")
            with patch.object(client, "_request",
                              return_value={"observations":
                                            [{"date": d["date"],
                                              "value": str(d["value"])} for d in sp_data]}):
                # Mock USREC too — return empty for simplicity
                def fake_request(endpoint, params, force_refresh=False):
                    sid = params.get("series_id")
                    if sid == "USREC":
                        return {"observations": []}
                    if sid == "SP500":
                        return {"observations":
                                [{"date": d["date"], "value": str(d["value"])} for d in sp_data]}
                    return {"observations": []}

                with patch.object(client, "_request", side_effect=fake_request):
                    summaries = ingest_targets(client, conn,
                                               start="2023-01-01", end="2024-01-31",
                                               pull_date="2024-02-01")

        t2_summary = next(s for s in summaries if s["target_id"] == "T2")
        check("t2_status_ok", t2_summary["status"] == "ok",
              f"got {t2_summary.get('error')}")
        check("t2_inserted_some_rows", t2_summary["n_inserted"] > 0)

        # Last month should have label=1 (20% drawdown > 15% threshold)
        last_label = conn.execute(
            """SELECT label FROM targets_monthly
               WHERE target_id='T2' AND observation_month='2024-01-01'""",
        ).fetchone()
        check("t2_jan2024_drawdown_label_is_1",
              last_label is not None and last_label[0] == 1,
              f"got {last_label}")

        conn.close()


# =============================================================================
# Live tests (--live flag required)
# =============================================================================

def test_live_validate_series() -> None:
    print("\n[L1] Live: validate all FRED series IDs")
    client = FredClient.from_env()
    ids = fred_series_ids_to_fetch()
    print(f"   Validating {len(ids)} series IDs against live FRED ...")
    results = client.validate_series_ids(ids)
    missing = [sid for sid, ok in results.items() if not ok]
    check("all_fred_ids_exist", len(missing) == 0,
          f"missing: {missing}" if missing else "")


def test_live_pull_short_series() -> None:
    print("\n[L2] Live: pull T10Y3M last 60 days")
    client = FredClient.from_env()
    obs = client.observations("T10Y3M",
                              start=add_days_iso("2026-04-01", -60),
                              end="2026-04-27")
    check("live_t10y3m_returned_data", len(obs) > 10, f"got {len(obs)}")

    valid = [o for o in obs if o["value"] is not None]
    check("live_t10y3m_has_finite_values", len(valid) > 5)
    if valid:
        check("live_t10y3m_value_in_plausible_range",
              -3.0 < valid[-1]["value"] < 5.0,
              f"got {valid[-1]['value']}")


def test_live_pull_alfred_vintages() -> None:
    print("\n[L3] Live: pull CFNAI with full vintages")
    client = FredClient.from_env()
    obs = client.observations_all_vintages("CFNAI",
                                           start="2008-01-01", end="2008-12-31")
    check("live_usslind_returned_data", len(obs) > 0, f"got {len(obs)}")

    # Should see multiple vintages for some observations (LEI gets revised)
    vintage_counts: dict[str, int] = {}
    for o in obs:
        vintage_counts[o["date"]] = vintage_counts.get(o["date"], 0) + 1
    multi_vintage = [d for d, n in vintage_counts.items() if n > 1]
    check("live_usslind_has_multi_vintage_months",
          len(multi_vintage) > 0,
          f"vintages per month: {vintage_counts}")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live FRED API tests")
    args = parser.parse_args()

    print("=" * 70)
    print("OFFLINE TESTS")
    print("=" * 70)

    test_series_specs()
    test_aggregate_to_monthly()
    test_forward_fill()
    test_token_bucket()
    test_json_cache()
    test_fred_client_construction()
    test_mocked_observations()
    test_ingest_to_real_db()
    test_manual_sources_ebp_parser()
    test_napmpi_composite_logic()
    test_drawdown_target_logic()

    if args.live:
        print("\n" + "=" * 70)
        print("LIVE TESTS (hits real FRED API)")
        print("=" * 70)
        try:
            test_live_validate_series()
            test_live_pull_short_series()
            test_live_pull_alfred_vintages()
        except Exception as e:
            print(f"\n  Live tests aborted: {e}")
            _failed.append(("live_tests_setup", str(e)))

    print("\n" + "=" * 70)
    print(f"RESULTS: {len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        print("\nFAILURES:")
        for name, detail in _failed:
            print(f"  - {name}: {detail}")
        return 1
    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
