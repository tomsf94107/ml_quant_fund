"""
recession.db migration & seeder.

Applies schema.sql, then seeds the features_registry, targets_registry, and
triggers_registry tables with the v1 spec content.

Idempotent: running twice does nothing destructive. Each migration version is
logged in schema_migrations.

Usage:
    python -m recession.db.migrate                  # apply migrations to default path
    python -m recession.db.migrate --db PATH        # custom path
    python -m recession.db.migrate --reset          # DROP and recreate (dev only!)
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Following the project convention: timezone helpers in utils/timezone.py.
# For this standalone module we inline minimal logic; later we can switch to
# the shared utility once we confirm the import path won't create coupling.
try:
    from zoneinfo import ZoneInfo
except ImportError:                                       # py < 3.9 fallback
    from backports.zoneinfo import ZoneInfo               # type: ignore

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

logger = logging.getLogger(__name__)

# Schema version this migrate.py knows how to produce. Bump when adding
# migration steps. Must match the PRAGMA user_version in schema.sql.
TARGET_SCHEMA_VERSION = 1

# Default DB path: <repo_root>/recession.db. The recession module lives at
# <repo_root>/recession/, so we go up one level.
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "recession.db"
SCHEMA_FILE     = Path(__file__).resolve().parent / "schema.sql"


# =============================================================================
# Registry seed data — ground truth from spec v1.0 §4 and §2
# =============================================================================

FEATURES_REGISTRY_SEED = [
    # (feature_name, tier, tier_label, description, source, fred_id,
    #  revisable, pub_lag_days, detrend_method, available_from, notes)

    # Tier 1: Yield curve & credit
    ("T10Y3M",        1, "yield_credit", "10y minus 3m Treasury spread",
     "FRED", "T10Y3M",         0, 1, "none", "1982-01", "NY Fed probit input"),
    ("BAMLH0A0HYM2",  1, "yield_credit", "ICE BofA US High Yield OAS",
     "FRED", "BAMLH0A0HYM2",   0, 1, "none", "1996-12", None),
    ("EBP",           1, "yield_credit", "Excess Bond Premium (Gilchrist-Zakrajsek)",
     "FRED", "EBP",            0, 7, "none", "1973-01",
     "Strongest single non-yield-curve recession predictor in published lit"),

    # Tier 2: Labor market
    ("SAHMREALTIME",  2, "labor", "Sahm Rule recession indicator (real-time)",
     "FRED", "SAHMREALTIME",   0, 7, "none", "1959-12", "Real-time vintage; no further revision"),
    ("JTSQUR",        2, "labor", "Quits rate (JOLTS)",
     "ALFRED", "JTSQUR",       1, 35, "none", "2000-12",
     "Beveridge-curve / labor tightness proxy"),
    ("ICSA",          2, "labor", "Initial unemployment claims, 4-week MA",
     "FRED", "ICSA",           1, 7, "none", "1967-01", None),

    # Tier 3: Real activity
    ("USSLIND",       3, "real_activity", "Conference Board LEI 6-month change",
     "ALFRED", "USSLIND",      1, 21, "none", "1982-01", None),
    ("ISRATIO",       3, "real_activity", "Manufacturing & trade inventory-to-sales ratio",
     "ALFRED", "ISRATIO",      1, 45, "none", "1992-01",
     "Reliable lead; available_from limits training"),
    ("PERMIT",        3, "real_activity", "Building permits, YoY % change",
     "ALFRED", "PERMIT",       1, 18, "yoy_pct", "1960-01", "LEI component"),
    ("INDPRO",        3, "real_activity", "Industrial production, 6-month change",
     "ALFRED", "INDPRO",       1, 16, "first_diff", "1919-01",
     "Long history; primary coincident indicator"),
    ("NAPMPI",        3, "real_activity", "ISM Manufacturing PMI (level + <50 dummy)",
     "FRED", "NAPMPI",         0, 1, "none", "1948-01",
     "Reported same-day; <50 = contraction signal"),

    # Tier 4: Financial conditions
    ("NFCI",          4, "financial_conditions", "Chicago Fed National Financial Conditions Index",
     "FRED", "NFCI",           0, 5, "none", "1971-01",
     "Composite of 105 measures; weekly→monthly aggregation"),
    ("SP500",         4, "financial_conditions", "S&P 500 close, 6-month return",
     "FRED", "SP500",          0, 0, "yoy_pct", "1957-03", "Market data; no revision"),
    ("DTWEXBGS",      4, "financial_conditions", "Broad trade-weighted USD index, 12m change",
     "FRED", "DTWEXBGS",       0, 1, "yoy_pct", "2006-01", None),

    # Tier 5: Monetary stance
    ("REAL_FFR_GAP",  5, "monetary_stance",
     "Real Fed funds gap = DFF - inflation_expectations - r*_HLW",
     "DERIVED", None,          1, 7, "none", "1985-01",
     "Composite: needs DFF, breakevens or Michigan, HLW r* estimate"),

    # Tier 6: Credit supply
    ("DRTSCILM",      6, "credit_supply", "SLOOS net % banks tightening C&I lending standards",
     "FRED", "DRTSCILM",       1, 45, "none", "1990-04", "Quarterly; forward-fill to monthly"),

    # Tier 7: Global
    ("CHINA_CREDIT_IMPULSE", 7, "global",
     "China credit impulse (BIS/Bloomberg derivation)",
     "BIS", None,              0, 60, "none", "2002-01",
     "Leads global cycle ~9 months; needs separate ingestion"),
    ("COPPER_GOLD",   7, "global", "Copper-to-gold price ratio",
     "DERIVED", None,          0, 1, "none", "1971-01",
     "Risk-on/risk-off proxy; computed from PCOPPUSDM and GOLDAMGBD228NLBM"),
    ("DCOILWTICO",    7, "global", "WTI crude oil, 12-month change",
     "FRED", "DCOILWTICO",     0, 1, "yoy_pct", "1986-01", None),

    # Tier 8: Sector — AI cycle (post-2010)
    ("HYPERSCALER_CAPEX_YOY", 8, "ai_cycle",
     "Top-7 hyperscaler quarterly capex, YoY % change",
     "MANUAL", None,           0, 45, "none", "2010-01",
     "Aggregated from MSFT/GOOGL/META/AMZN/ORCL/AAPL/NVDA filings; quarterly"),
    ("MEMORY_CONTRACT_PX",    8, "ai_cycle",
     "Memory contract price index (NAND + DRAM blended)",
     "TRENDFORCE", None,       0, 30, "none", "2010-01",
     "Monthly; composite of NAND and DRAM contract prices"),

    # ─────────────────────────────────────────────────────────────────────
    # v1.1 / v1.1.1 additions (Step 3.5, May 2026)
    # ─────────────────────────────────────────────────────────────────────

    # Tier 1: Yield curve additions
    ("T10Y2Y",       1, "yield_credit", "10y - 2y Treasury spread (Engstrom-Sharpe)",
     "FRED",   "T10Y2Y",          0, 1, "none", "1976-06",
     "Daily; complements T10Y3M"),
    ("NEAR_TERM_FORWARD", 1, "yield_credit",
     "Near-term forward rate spread (Engstrom-Sharpe 2018)",
     "DERIVED", None,             0, 1, "none", "1982-01",
     "v1 proxy from DGS3MO; v2 should use full forward curve"),

    # Tier 2: Labor expansion
    ("CCSA",         2, "labor", "Continued unemployment claims, weekly",
     "ALFRED", "CCSA",            1, 7, "none", "1967-01",
     "Coincident; restricted to h=0/1/3 per v1.1.1"),
    ("AWHMAN",       2, "labor", "Avg weekly hours, manufacturing",
     "ALFRED", "AWHMAN",          1, 10, "none", "1939-01",
     "LEI component; restricted to h=0/1/3"),
    ("TEMPHELPS",    2, "labor", "Temporary help services employment",
     "ALFRED", "TEMPHELPS",       1, 10, "none", "1990-01",
     "Layoffs in temp help lead broader layoffs by 2-3 months"),
    ("JTSLDR",       2, "labor", "JOLTS layoffs and discharges rate",
     "ALFRED", "JTSLDR",          1, 35, "none", "2000-12",
     "Pairs with JTSQUR for Beveridge-curve coverage"),

    # Tier 9: Housing
    ("UMCSENT",      9, "housing", "Univ. Michigan Consumer Sentiment",
     "FRED",   "UMCSENT",         0, 30, "none", "1952-11",
     "Substitute for HOMENSA (not on FRED) and HPSI (discontinued). "
     "1952+ history; FRED Blog uses it as HPSI comparable."),
    ("EXHOSLUSM495S", 9, "housing", "Existing home sales (NAR via FRED) — DEFERRED v2",
     "FRED", "EXHOSLUSM495S",     0, 21, "none", "1999-01",
     "DEFERRED: NAR rolling-window license; only 13 months available. "
     "Replaced by HSN1F. Row kept for documentation."),
    ("HSN1F",         9, "housing", "New One Family Houses Sold (Census/HUD)",
     "ALFRED", "HSN1F",            1, 25, "none", "1963-01",
     "Public domain Census/HUD; full vintage history on ALFRED. "
     "Replaces EXHOSLUSM495S which is NAR-licensed."),

    # Tier 10: Inflation
    ("CPILFESL",     10, "inflation", "Core CPI (ex food and energy)",
     "ALFRED", "CPILFESL",        1, 14, "yoy_pct", "1957-01",
     "YoY transformation"),
    ("PCEPILFE",     10, "inflation", "Core PCE (Fed preferred gauge)",
     "ALFRED", "PCEPILFE",        1, 30, "yoy_pct", "1959-01",
     "YoY transformation"),
    ("CES0500000003", 10, "inflation", "Avg hourly earnings, total private",
     "ALFRED", "CES0500000003",   1, 10, "yoy_pct", "2006-03",
     "Wage growth proxy; YoY transformation"),

    # Tier 11: Engineered (1 shippable now; 3 more in Steps 3.7/4)
    ("COVID_DUMMY",  11, "engineered", "Binary 1 if 2020-03..2021-12 else 0",
     "DERIVED", None,             0, 0, "none", "1960-01",
     "Per brief §G15; absorbs COVID structural break"),
]


TARGETS_REGISTRY_SEED = [
    # (target_id, target_name, target_type, description, is_exploratory,
    #  exploratory_caveat, available_from, n_events, notes)

    ("T1", "NBER Recession", "binary_atomic",
     "Official NBER Business Cycle Dating Committee recession indicator",
     0, None, "1960-01", 10,
     "Primary academic target. Vintage handling critical: label only after announcement."),

    ("T2", "SPX 15% Drawdown", "binary_atomic",
     "S&P 500 closing price >=15% below 12m rolling maximum",
     0, None, "1960-01", 12,
     "Equity-market regime target. Computable directly from prices."),

    ("T3", "AI Kill-Switch Composite", "binary_composite",
     "Binary 1 if 2 or more of 5 AI thesis macro triggers fire in a given month",
     1,
     "EXPLORATORY: Limited history (post-2010 with proxies; post-2015 for full triggers). "
     "Triggers were defined as a thesis-monitoring heuristic, not validated as a "
     "statistical target. Treat T3 outputs as directional only, not as production-grade "
     "probabilities. v2 will reassess after 2-3 years of out-of-sample trigger data.",
     "2010-01", None,
     "5 underlying triggers per playbook §12; see triggers_registry table."),

    ("T4a", "Composite — Equal Weighted", "combination",
     "(P(T1) + P(T2) + P(T3)) / 3. Headline at-a-glance number.",
     0, None, "2010-01", None,
     "Inherits T3's exploratory limitation but is the least overfit composite."),

    ("T4b", "Composite — Constrained Data-Driven", "combination",
     "Granger-Ramanathan weights with floor 0.1 / ceiling 0.5 on each component",
     0, None, "2010-01", None,
     "Reported alongside T4a as overfit-detector: divergence from T4a signals concern."),

    ("T4c", "Composite — Maximum (warning indicator only)", "combination",
     "max(P(T1), P(T2), P(T3)). Not a calibrated probability.",
     1,
     "WARNING INDICATOR ONLY: T4c is not a calibrated probability. It fires when ANY "
     "of T1/T2/T3 fires, so it is biased upward. Use only as 'any-target-firing' alarm.",
     "2010-01", None,
     "Useful for risk-management view; do not use for sizing or calibration tests."),
]


TRIGGERS_REGISTRY_SEED = [
    # (trigger_id, name, description, threshold_rule, threshold_value,
    #  available_from, proxy_used_before, is_proxied_in_v1)

    (1, "Hyperscaler Capex Plateau",
     "Top-4 hyperscalers (MSFT/GOOGL/META/AMZN) guide forward-year capex flat or below current year",
     "Forward capex YoY <= 0%", 0.0,
     "2015-01", "Tech-sector aggregate capex YoY <= 0% (FRED + manual)", 1),

    (2, "Memory Contract Price Decline",
     "NAND or DRAM contract prices declined QoQ for 2 consecutive quarters",
     "QoQ change < 0 for 2+ consecutive quarters", 0.0,
     "2010-01", "Semi industry sales YoY <= 0% (SOX/SIA)", 1),

    (3, "AI Productivity Disappointment",
     ">=2 enterprise AI productivity studies in 90d window report null/negative results",
     ">=2 negative studies / 90d", 2.0,
     "2023-01", None, 1),

    (4, "NVDA Customer Concentration",
     "NVDA top-4 customer revenue concentration > 70%",
     "Top-4 concentration > 70%", 70.0,
     "2015-01", None, 1),

    (5, "Data Center Cancellations",
     ">10 GW announced data center capacity cancelled in trailing 90 days",
     "Cancelled GW > 10 in 90d", 10.0,
     "2018-01", None, 1),
]


# =============================================================================
# Migration logic
# =============================================================================

def vn_now_iso() -> str:
    """Current time in Vietnam tz, ISO 8601 — project convention."""
    return datetime.now(VN_TZ).isoformat(timespec="seconds")


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return PRAGMA user_version (0 if never set)."""
    return conn.execute("PRAGMA user_version").fetchone()[0]


def apply_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    """Execute the full schema.sql. Uses CREATE IF NOT EXISTS so it's idempotent."""
    sql = schema_path.read_text()
    conn.executescript(sql)


def seed_features_registry(conn: sqlite3.Connection) -> int:
    """Insert features_registry rows. Returns count inserted."""
    sql = """
        INSERT OR IGNORE INTO features_registry
            (feature_name, tier, tier_label, description, source, fred_series_id,
             revisable, publication_lag_days, detrend_method, available_from, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cur = conn.executemany(sql, FEATURES_REGISTRY_SEED)
    return cur.rowcount


def seed_targets_registry(conn: sqlite3.Connection) -> int:
    """Insert targets_registry rows. Returns count inserted."""
    sql = """
        INSERT OR IGNORE INTO targets_registry
            (target_id, target_name, target_type, description,
             is_exploratory, exploratory_caveat, available_from, n_events_full_sample, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cur = conn.executemany(sql, TARGETS_REGISTRY_SEED)
    return cur.rowcount


def seed_triggers_registry(conn: sqlite3.Connection) -> int:
    """Insert triggers_registry rows. Returns count inserted."""
    sql = """
        INSERT OR IGNORE INTO triggers_registry
            (trigger_id, trigger_name, description, threshold_rule, threshold_value,
             available_from, proxy_used_before, is_proxied_in_v1)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    cur = conn.executemany(sql, TRIGGERS_REGISTRY_SEED)
    return cur.rowcount


def record_migration(
    conn: sqlite3.Connection,
    version: int,
    description: str,
    code_sha: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO schema_migrations (version, applied_at, description, code_sha)
        VALUES (?, ?, ?, ?)
        """,
        (version, vn_now_iso(), description, code_sha),
    )


def run_migration(db_path: Path, reset: bool = False) -> None:
    """Apply all pending migrations. Idempotent."""
    if reset:
        if db_path.exists():
            logger.warning("--reset specified; removing existing %s", db_path)
            db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        current = get_schema_version(conn)
        logger.info("Connected to %s (current schema version: %d)", db_path, current)

        if current >= TARGET_SCHEMA_VERSION:
            logger.info("Schema already at v%d; no migration needed.", current)
        else:
            logger.info("Applying schema v%d → v%d", current, TARGET_SCHEMA_VERSION)
            apply_schema(conn, SCHEMA_FILE)
            record_migration(conn, 1, "Initial schema v1 per spec v1.0")
            conn.commit()
            logger.info("Schema applied.")

        # Seed registries (idempotent via INSERT OR IGNORE)
        n_feat = seed_features_registry(conn)
        n_targ = seed_targets_registry(conn)
        n_trig = seed_triggers_registry(conn)
        conn.commit()

        logger.info(
            "Registries seeded: %d features, %d targets, %d triggers (new rows only).",
            n_feat, n_targ, n_trig,
        )

        # Quick sanity counts
        feat_total = conn.execute("SELECT COUNT(*) FROM features_registry").fetchone()[0]
        targ_total = conn.execute("SELECT COUNT(*) FROM targets_registry").fetchone()[0]
        trig_total = conn.execute("SELECT COUNT(*) FROM triggers_registry").fetchone()[0]
        explor      = conn.execute(
            "SELECT COUNT(*) FROM targets_registry WHERE is_exploratory = 1"
        ).fetchone()[0]
        logger.info(
            "Registry totals: %d features, %d targets (%d exploratory), %d triggers",
            feat_total, targ_total, explor, trig_total,
        )

    finally:
        conn.close()


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Apply recession.db migrations")
    parser.add_argument("--db",    type=Path, default=DEFAULT_DB_PATH,
                        help=f"DB path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate (DEV ONLY — destroys data)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.reset:
        confirm = input(f"DROP and recreate {args.db}? Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            logger.info("Aborted.")
            return 1

    run_migration(args.db, reset=args.reset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
