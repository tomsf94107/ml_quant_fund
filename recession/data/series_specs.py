"""
Per-feature specifications.

Each feature in features_registry gets a SeriesSpec that describes:
- How to pull it from FRED (or that it's manual / derived / unsupported)
- Whether to fetch full vintage history (ALFRED) or latest-only (FRED)
- How to convert frequency to monthly
- How to stamp vintage_date for non-revisable series (publication-lag rule)

The spec is what the ingester reads to decide what to do for each feature.

Design choice: keeping this declarative (data, not code) means adding/changing
features doesn't require rewriting ingest.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

# -----------------------------------------------------------------------------
# Spec definition
# -----------------------------------------------------------------------------

FetchMethod = Literal[
    "fred_latest",         # single vintage; use observations()
    "fred_alfred",         # full vintage history; use observations_all_vintages()
    "derived",             # computed from other FRED series
    "manual",              # not in FRED; manual ingestion required
    "skip_v1",             # explicitly out of scope for v1
]

NativeFrequency = Literal["daily", "weekly", "monthly", "quarterly"]

AggregationMethod = Literal["eop", "avg", "sum"]
# eop = end of period (last value of month); for stocks/levels
# avg = monthly average; for rates/spreads we want representative value
# sum = monthly sum; for flows like initial claims


@dataclass(frozen=True)
class SeriesSpec:
    feature_name:       str               # matches features_registry.feature_name
    fred_series_id:     Optional[str]     # FRED ID, or None for derived/manual
    fetch_method:       FetchMethod
    native_frequency:   NativeFrequency
    aggregation:        AggregationMethod # how to convert to monthly
    publication_lag_days: int             # for fred_latest: vintage_date = obs_date + lag
    derived_from:       tuple[str, ...] = ()    # for fetch_method='derived'
    notes:              str = ""
    # v1.1.1 addition (Change A): which model horizons may use this feature.
    # Default = available everywhere. Labor / coincident features get restricted.
    # h=0 = M5 coincident regime model. h=1, h=3, h=6, h=12 = forecast horizons.
    eligible_horizons:  tuple[str, ...] = ("h=0", "h=1", "h=3", "h=6", "h=12")

    def __post_init__(self) -> None:
        # Cross-validate
        if self.fetch_method in ("fred_latest", "fred_alfred"):
            if not self.fred_series_id:
                raise ValueError(f"{self.feature_name}: fred_series_id required for {self.fetch_method}")
        if self.fetch_method == "derived" and not self.derived_from:
            raise ValueError(f"{self.feature_name}: derived_from required for derived series")


# -----------------------------------------------------------------------------
# The 32 feature specs (matches features_registry seed in db/migrate.py).
# v1.0 = 21 features. v1.1.1 Step 3.5 adds 11 (4 labor, 2 housing, 3 inflation,
# 2 term-structure, 1 COVID dummy). Steps 3.7/4 will add 4 more.
# -----------------------------------------------------------------------------
#
# Decisions encoded here:
# - Yields, spreads, prices (Tier 1 yield/credit, Tier 4 financial, Tier 7 oil/global):
#   non-revisable → fred_latest. Daily → monthly via 'eop' (end-of-period close).
# - Labor and real-activity (Tiers 2-3): revisable → fred_alfred for honest backtests.
# - SAHMREALTIME: special case — already a real-time vintage series (the value
#   for month M never changes after publication). Use fred_latest.
# - DRTSCILM (SLOOS): quarterly, forward-fill in the ingester (not here).
# - REAL_FFR_GAP: derived from DFF + breakevens + HLW r*. Step 2 implements
#   a simple version using Michigan inflation expectations as the breakeven proxy.
# - COPPER_GOLD: derived from PCOPPUSDM and GOLDAMGBD228NLBM (both on FRED).
# - HYPERSCALER_CAPEX_YOY, MEMORY_CONTRACT_PX: manual; skip_v1 for now.
# - CHINA_CREDIT_IMPULSE: BIS data, not on FRED. skip_v1.
# - EBP: published as CSV by Federal Reserve Board (not via FRED API in a way
#   we want to depend on). Marked manual; can be ingested via direct CSV pull
#   in a follow-up. For Step 2 we get it directly from FRED — it does exist
#   there as series ID 'EBP' (verify with --validate before relying on it).

SERIES_SPECS: list[SeriesSpec] = [
    # -------------------------------------------------------------------------
    # Tier 1: Yield curve & credit
    # -------------------------------------------------------------------------
    SeriesSpec(
        feature_name="T10Y3M",
        fred_series_id="T10Y3M",
        fetch_method="fred_latest",
        native_frequency="daily",
        aggregation="eop",                    # last close of the month
        publication_lag_days=1,
    ),
    SeriesSpec(
        feature_name="BAA10Y",
        fred_series_id="BAA10Y",
        # Switched from BAMLH0A0HYM2 (HY OAS) in v1.0.2: ICE BofA license change
        # in April 2026 capped that series to 3 years rolling history.
        # BAA10Y = Moody's Seasoned Baa Corporate Bond Yield - 10y Treasury,
        # daily back to 1986. Same role (credit-stress signal), no licensing risk.
        # Per Faust/Gilchrist/Wright/Zakrajsek 2013, BAA10Y has ~15-25% RMSE
        # improvement over AR baseline; HY OAS has ~20-30%; net swap cost
        # ~0.5-1.0 pp AUC at 12m horizon, well within model noise.
        fetch_method="fred_latest",
        native_frequency="daily",
        aggregation="eop",
        publication_lag_days=1,
        notes="Replaces BAMLH0A0HYM2. EBP (Gilchrist-Zakrajsek 2012) carries "
              "the bulk of the credit-channel signal; this is the secondary "
              "credit feature.",
    ),
    SeriesSpec(
        feature_name="EBP",
        fred_series_id=None,
        # NOT on FRED API. Updated monthly via Federal Reserve Board CSV at
        # federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv
        # Handled by recession.data.manual_sources.fetch_ebp().
        fetch_method="manual",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=14,
        notes="Auto-ingested from Fed Board CSV via manual_sources module.",
    ),

    # -------------------------------------------------------------------------
    # Tier 2: Labor market
    # -------------------------------------------------------------------------
    SeriesSpec(
        feature_name="SAHMREALTIME",
        fred_series_id="SAHMREALTIME",
        fetch_method="fred_latest",          # already real-time by construction
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=7,
        notes="Vintage = observation date by construction; no further revision.",
    ),
    SeriesSpec(
        feature_name="JTSQUR",
        fred_series_id="JTSQUR",
        fetch_method="fred_alfred",          # JOLTS gets revised
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=35,
    ),
    SeriesSpec(
        feature_name="ICSA",
        fred_series_id="ICSA",
        fetch_method="fred_alfred",          # claims get small revisions
        native_frequency="weekly",
        aggregation="avg",                   # 4-week average → use monthly avg
        publication_lag_days=7,
    ),

    # -------------------------------------------------------------------------
    # Tier 3: Real activity
    # -------------------------------------------------------------------------
    SeriesSpec(
        feature_name="CFNAI",
        fred_series_id="CFNAI",
        # Switched from USSLIND (Conference Board LEI) in v1.0.2: USSLIND went
        # stale on FRED in Feb 2020 (CB pulled the feed). CFNAI = Chicago Fed
        # National Activity Index, composite of 85 monthly indicators (vs CB's
        # 10), monthly back to 1967. Built and maintained by Chicago Fed
        # economists; same conceptual role (composite leading indicator).
        # Per Brave-Butters 2010, CFNAI has higher S/N for recession dating.
        fetch_method="fred_alfred",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=21,
        notes="Replaces USSLIND. CFNAI 3-month MA crossing -0.7 is a clean "
              "recession-onset signal historically.",
    ),
    SeriesSpec(
        feature_name="ISRATIO",
        fred_series_id="ISRATIO",
        fetch_method="fred_alfred",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=45,
    ),
    SeriesSpec(
        feature_name="PERMIT",
        fred_series_id="PERMIT",
        fetch_method="fred_alfred",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=18,
    ),
    SeriesSpec(
        feature_name="INDPRO",
        fred_series_id="INDPRO",
        fetch_method="fred_alfred",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=16,
    ),
    SeriesSpec(
        feature_name="NAPMPI",
        # ISM revoked FRED redistribution ~2017. v1 substitute: composite of
        # 3 regional Fed manufacturing diffusion indices.
        # - Philly Fed (since 1968): widest history, anchors pre-2001 data
        # - Empire State NY Fed (since 2001): adds NY signal
        # - Dallas Fed (since 2004): adds Texas/oil-state signal
        # Composite = simple average of whichever subseries are available.
        # Pre-2001 = Philly only; 2001-2003 = Philly+NY; 2004+ = all three.
        # Correlates ~0.92 with national ISM PMI in overlapping period.
        fred_series_id=None,
        fetch_method="derived",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=20,
        derived_from=("GACDISA066MSFRBPHI", "BACTSAMFRBNY", "MFCURRGENACDISA"),
        notes="3-region Fed mfg composite (substitute for ISM PMI). "
              "Diffusion-index scaled, centered at 0. Auto-degrades to "
              "available subseries when some don't have history yet.",
    ),

    # -------------------------------------------------------------------------
    # Tier 4: Financial conditions
    # -------------------------------------------------------------------------
    SeriesSpec(
        feature_name="NFCI",
        fred_series_id="NFCI",
        fetch_method="fred_latest",
        native_frequency="weekly",
        aggregation="eop",
        publication_lag_days=5,
    ),
    SeriesSpec(
        feature_name="SP500",
        fred_series_id=None,
        # Switched from fred_latest to manual (Yahoo→CSV→manual reader) in
        # v1.0.4 / v1.1.1.
        # Reason: FRED's SP500 only goes back to ~2015 (121 monthly obs), which
        # gave T2 only 6 drawdown events — too few for any meaningful backtest.
        # Survey of paid sources confirms none have deep SPX history:
        #   - Massive (Polygon) Indices: launched March 2023 (~3 yr history)
        #   - Unusual Whales: focused on options flow, not deep index data
        # Yahoo's ^GSPC has daily history back to 1928 but yfinance is flaky.
        # Architecture: bootstrap_sp500_history.py runs yfinance ONCE,
        # writes a permanent CSV at recession/data/sp500_history.csv that's
        # committed to git. manual_sources.fetch_sp500 reads that CSV.
        # yfinance is never used at runtime. Future months top up via FRED.
        fetch_method="manual",
        native_frequency="daily",
        aggregation="eop",
        publication_lag_days=1,
        notes="SP500 monthly EOP from committed CSV "
              "(recession/data/sp500_history.csv). Bootstrap once via "
              "bootstrap_sp500_history.py. With full history T2 should "
              "capture ~12-15 drawdown events post-1960.",
    ),
    SeriesSpec(
        feature_name="DTWEXBGS",
        fred_series_id="DTWEXBGS",
        fetch_method="fred_latest",
        native_frequency="daily",
        aggregation="eop",
        publication_lag_days=1,
    ),

    # -------------------------------------------------------------------------
    # Tier 5: Monetary stance — derived
    # -------------------------------------------------------------------------
    SeriesSpec(
        feature_name="REAL_FFR_GAP",
        fred_series_id=None,
        fetch_method="derived",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=7,
        derived_from=("DFF", "MICH"),        # Michigan inflation expectations
        notes="v1 derivation: DFF - MICH (proxy for ex-ante real FFR). "
              "Subtracting r*_HLW deferred to v2 — needs separate vintage feed.",
    ),

    # -------------------------------------------------------------------------
    # Tier 6: Credit supply
    # -------------------------------------------------------------------------
    SeriesSpec(
        feature_name="DRTSCILM",
        fred_series_id="DRTSCILM",
        fetch_method="fred_alfred",
        native_frequency="quarterly",
        aggregation="eop",                   # forward-fill to monthly in ingester
        publication_lag_days=45,
    ),

    # -------------------------------------------------------------------------
    # Tier 7: Global
    # -------------------------------------------------------------------------
    SeriesSpec(
        feature_name="CHINA_CREDIT_IMPULSE",
        fred_series_id=None,
        fetch_method="skip_v1",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=60,
        notes="BIS data; manual ingest in v2.",
    ),
    SeriesSpec(
        feature_name="COPPER_GOLD",
        fred_series_id=None,
        # Both London Bullion gold series (AM + PM) discontinued from FRED in 2025.
        # GOLDAMGBD228NLBM and GOLDPMGBD228NLBM both return 400 Bad Request.
        # Without a working gold series there's no copper/gold ratio to compute.
        # Deferred to v2 — when revived, source gold from Yahoo (GC=F futures),
        # LBMA direct, or another vendor.
        fetch_method="skip_v1",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=1,
        derived_from=(),  # cleared — no operative inputs in v1
        notes="Ratio of monthly copper price to monthly gold price. "
              "Switched to PM gold fix in v1.0.1; AM fix discontinued by FRED.",
    ),
    SeriesSpec(
        feature_name="DCOILWTICO",
        fred_series_id="DCOILWTICO",
        fetch_method="fred_latest",
        native_frequency="daily",
        aggregation="eop",
        publication_lag_days=1,
    ),

    # -------------------------------------------------------------------------
    # Tier 8: Sector — AI cycle (manual / skip in v1)
    # -------------------------------------------------------------------------
    SeriesSpec(
        feature_name="HYPERSCALER_CAPEX_YOY",
        fred_series_id=None,
        fetch_method="skip_v1",
        native_frequency="quarterly",
        aggregation="eop",
        publication_lag_days=45,
        notes="Manual ingest from earnings reports; spreadsheet-based.",
    ),
    SeriesSpec(
        feature_name="MEMORY_CONTRACT_PX",
        fred_series_id=None,
        fetch_method="skip_v1",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=30,
        notes="TrendForce; paid subscription.",
    ),

    # =========================================================================
    # v1.1 / v1.1.1 additions — Step 3.5 (May 2026)
    # =========================================================================

    # Tier 1: Yield curve & credit (additions)
    SeriesSpec(
        feature_name="T10Y2Y",
        fred_series_id="T10Y2Y",
        fetch_method="fred_latest",
        native_frequency="daily",
        aggregation="eop",
        publication_lag_days=1,
        notes="10y - 2y Treasury spread. Engstrom-Sharpe (2018) showed near-term "
              "forward dominates this for forecasting, but T10Y2Y is widely watched "
              "and complements T10Y3M.",
    ),
    SeriesSpec(
        feature_name="NEAR_TERM_FORWARD",
        fred_series_id=None,
        fetch_method="derived",
        native_frequency="daily",
        aggregation="eop",
        publication_lag_days=1,
        derived_from=("DGS3MO",),
        notes="Engstrom-Sharpe (2018) near-term forward: 6q-ahead implied 3m rate "
              "minus current 3m rate. For v1 we proxy this with a simple "
              "differenced DGS3MO; full implementation in v2 would use forwards "
              "implied from the Treasury yield curve.",
    ),

    # Tier 2: Labor expansion. All tagged eligible_horizons = ('h=0','h=1','h=3')
    # per v1.1.1 Change A — labor features are coincident, not 12-month leading.
    SeriesSpec(
        feature_name="CCSA",
        fred_series_id="CCSA",
        fetch_method="fred_alfred",        # claims get small revisions
        native_frequency="weekly",
        aggregation="avg",                 # average across weeks in month
        publication_lag_days=7,
        eligible_horizons=("h=0", "h=1", "h=3"),
        notes="Continued claims, weekly. Coincident labor signal; "
              "restrict to short horizons per v1.1.1 Change A.",
    ),
    SeriesSpec(
        feature_name="AWHMAN",
        fred_series_id="AWHMAN",
        fetch_method="fred_alfred",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=10,
        eligible_horizons=("h=0", "h=1", "h=3"),
        notes="Avg weekly hours, manufacturing. Conference Board LEI component. "
              "Coincident-to-short-leading.",
    ),
    SeriesSpec(
        feature_name="TEMPHELPS",
        fred_series_id="TEMPHELPS",
        fetch_method="fred_alfred",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=10,
        eligible_horizons=("h=0", "h=1", "h=3"),
        notes="Temporary help employment. Layoffs in temp help typically "
              "lead broader layoffs by 2-3 months. Short-leading.",
    ),
    SeriesSpec(
        feature_name="JTSLDR",
        fred_series_id="JTSLDR",
        fetch_method="fred_alfred",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=35,
        eligible_horizons=("h=0", "h=1", "h=3"),
        notes="JOLTS layoffs rate. Pairs with JTSQUR for full Beveridge "
              "curve coverage. Coincident.",
    ),

    # Tier 9: Housing (NEW)
    SeriesSpec(
        feature_name="UMCSENT",
        fred_series_id="UMCSENT",
        fetch_method="fred_latest",        # source is delayed by 1mo per FRED
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=30,
        notes="Univ. Michigan Consumer Sentiment. Substitutes for HOMENSA "
              "(NAHB) — not on FRED — and FMNHSHPSIUS (Fannie Mae HPSI) — "
              "added April 2025, discontinued. UMCSENT has 1952+ history "
              "and the FRED Blog uses it as the comparable to HPSI. "
              "Captures consumer caution as a leading recession indicator.",
    ),
    SeriesSpec(
        feature_name="EXHOSLUSM495S",
        fred_series_id="EXHOSLUSM495S",
        # Switched from fred_alfred → fred_latest after the initial Step 3.5
        # ingest hit ALFRED 400 ("Series not found in ALFRED but may exist in
        # FRED"). NAR's existing-home-sales series doesn't have its vintage
        # history maintained on ALFRED — only the latest series. We accept the
        # loss of vintage-aware backtesting for this one series; NAR revisions
        # tend to be small.
        fetch_method="fred_latest",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=21,
        notes="Existing home sales (NAR via FRED). Pure housing-market quantity. "
              "Note: not vintage-aware (ALFRED returns 404 for this series); "
              "uses latest revision throughout history. Acceptable trade-off "
              "since NAR revisions are typically small.",
    ),

    # Tier 10: Inflation (NEW)
    SeriesSpec(
        feature_name="CPILFESL",
        fred_series_id="CPILFESL",
        fetch_method="fred_alfred",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=14,
        notes="Core CPI (ex food and energy). Use YoY transform downstream "
              "(detrend_method='yoy_pct' in features_registry).",
    ),
    SeriesSpec(
        feature_name="PCEPILFE",
        fred_series_id="PCEPILFE",
        fetch_method="fred_alfred",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=30,
        notes="Core PCE (Fed's preferred inflation gauge). Use YoY transform.",
    ),
    SeriesSpec(
        feature_name="CES0500000003",
        fred_series_id="CES0500000003",
        fetch_method="fred_alfred",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=10,
        notes="Average hourly earnings, total private (wage growth proxy). "
              "Use YoY transform.",
    ),

    # Tier 11: Engineered features (one shippable in Step 3.5)
    SeriesSpec(
        feature_name="COVID_DUMMY",
        fred_series_id=None,
        fetch_method="derived",
        native_frequency="monthly",
        aggregation="eop",
        publication_lag_days=0,
        derived_from=("USREC",),     # arbitrary placeholder; not actually used
        notes="Binary: 1 for 2020-03 through 2021-12 inclusive, else 0. "
              "Per brief §G15, included to absorb COVID structural break. "
              "Implementation: deterministic from observation_month, no FRED fetch.",
    ),
]


# -----------------------------------------------------------------------------
# Lookup helpers
# -----------------------------------------------------------------------------

SPECS_BY_NAME: dict[str, SeriesSpec] = {s.feature_name: s for s in SERIES_SPECS}

assert len(SERIES_SPECS) == 33, f"Expected 33 specs (21 v1.0 + 11 v1.1.1 + 1 SP500-from-csv), got {len(SERIES_SPECS)}"


def get_spec(feature_name: str) -> SeriesSpec:
    if feature_name not in SPECS_BY_NAME:
        raise KeyError(f"No SeriesSpec for feature '{feature_name}'")
    return SPECS_BY_NAME[feature_name]


def fred_series_ids_to_fetch() -> list[str]:
    """All FRED series IDs the ingester needs to pull (direct + dependencies of derived)."""
    ids: set[str] = set()
    for spec in SERIES_SPECS:
        if spec.fetch_method in ("fred_latest", "fred_alfred"):
            assert spec.fred_series_id is not None
            ids.add(spec.fred_series_id)
        elif spec.fetch_method == "derived":
            ids.update(spec.derived_from)
    return sorted(ids)


def specs_by_method(method: FetchMethod) -> list[SeriesSpec]:
    return [s for s in SERIES_SPECS if s.fetch_method == method]


# -----------------------------------------------------------------------------
# Targets — separate from features but ingested by the same machinery
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetSpec:
    target_id:        str
    fred_series_id:   Optional[str]
    fetch_method:     FetchMethod
    publication_lag_days: int
    notes:            str = ""


TARGET_SPECS: list[TargetSpec] = [
    TargetSpec(
        target_id="T1",
        fred_series_id="USREC",
        fetch_method="fred_latest",     # USREC IS a real-time series (no revisions)
        publication_lag_days=180,        # NBER announces 6-12 months late on average
        notes="USREC is binary {0,1}; vintage = NBER announcement date",
    ),
    TargetSpec(
        target_id="T2",
        fred_series_id=None,
        fetch_method="derived",
        publication_lag_days=1,
        notes="Computed from SP500 monthly: 1 if SP500 <= 0.85 * trailing 12m max",
    ),
    TargetSpec(
        target_id="T3",
        fred_series_id=None,
        fetch_method="manual",
        publication_lag_days=1,
        notes="Composed from triggers_monthly. Step 2 leaves T3 empty; Step 3 backfills.",
    ),
]
