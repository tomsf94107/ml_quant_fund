# Feature Backlog — Empty Registry Features

*The feature audit (recession/validation/feature_audit.py) found 7
features registered in `features_registry` with NO data in
`features_monthly`. They are kept (not deleted) — each is a deliberate,
recorded intent. This document is the sourcing plan: why each is empty,
exactly where the data comes from, how hard it is, and the concrete next
step.*

The 7 split into three difficulty tiers.

---

## TIER 1 — Easy: standard FRED series, just not pulled yet

These have no obstacle at all. They are ordinary FRED series; the
ingestion pipeline simply was never pointed at them. Each is one FRED
series ID. Closing these is a short job — run the project's existing FRED
ingestion for these three series IDs.

| Feature | FRED series ID | What it is | Recession relevance |
|---|---|---|---|
| `BAMLH0A0HYM2` | BAMLH0A0HYM2 | ICE BofA US High Yield Option-Adjusted Spread | Credit-market stress. Widening HY spreads are a classic recession lead — arguably the strongest of the three. |
| `USSLIND` | USSLIND | Leading Index for the United States (Philadelphia Fed, state-level aggregate) | A composite leading indicator — purpose-built to lead the cycle. |
| `EXHOSLUSM495S` | EXHOSLUSM495S | Existing Home Sales | Housing turns before the broad economy; a real-activity lead. |

**Next step (Tier 1):** run the project's FRED ingestion script for these
three series IDs. They are daily/monthly FRED series with long history —
ingestion should be routine. After ingestion, re-run the feature audit:
they move from "empty" to "populated", and become candidates for the
feature-testing experiments (A / B).

---

## TIER 2 — Engineered: must be computed, not downloaded

| Feature | Inputs | What it is | Status |
|---|---|---|---|
| `COPPER_GOLD` | a copper price series + a gold price series | The copper-to-gold ratio — a market-based growth/risk gauge ("Dr. Copper" up with growth; gold up with risk aversion). | **Build DONE** — `recession/features/derive_copper_gold.py`. |

There is no single FRED "copper/gold" series, so `COPPER_GOLD` must be
derived. `derive_copper_gold.py` does this: it reads a copper price
series and a gold price series from `features_monthly`, computes the
ratio per month (PIT-safe — derived vintage = the later of the two input
vintages), and writes `COPPER_GOLD`.

**Next step (Tier 2):** the derivation needs its two INPUTS in the DB
first. Ingest a copper price series (FRED `PCOPPUSDM` — Global Price of
Copper) and a gold price series (FRED `GOLDAMGBD228NLBM` — Gold Fixing
Price, London) via the FRED ingestion. Then run:

    python -c "from recession.features.derive_copper_gold import derive_copper_gold; \
        r = derive_copper_gold(db_path='recession.db'); print(r['message'])"

`derive_copper_gold` auto-detects the input series names; if the DB uses
other names, pass `copper_feature=` / `gold_feature=` explicitly. Run with
`dry_run=True` first to preview.

---

## TIER 3 — Hard: data is not cleanly or freely available

These are genuine data-engineering / data-procurement projects. They are
NOT quick fixes, and this document does not pretend otherwise. Each entry
records the honest options so the decision is informed.

### `CHINA_CREDIT_IMPULSE` (tier: global)
- **What it is:** the change in new credit flow as a share of GDP in
  China — a widely-watched global growth lead.
- **Why empty:** there is no clean, free, single public series. It is
  constructed from China's Total Social Financing (TSF) and GDP.
- **Sourcing options:**
  1. **Construct it.** PBoC publishes TSF (aggregate financing) and GDP.
     The credit impulse = 12-month change in (new credit / GDP). This is
     buildable from public PBoC/NBS data — but the data is awkward to
     pull (no clean API like FRED), and methodology choices (flow vs
     stock, seasonal adjustment) materially affect the result.
  2. **Buy it.** Bloomberg and some research providers publish a
     ready-made China credit impulse series — paid.
  3. **Proxy it.** Use a cleaner adjacent series as a stand-in (e.g. a
     China activity proxy already on FRED). Lower fidelity.
- **Recommended:** defer. Revisit only if the feature-testing experiments
  show the `global` tier matters; if so, start with option 1
  (construct from public TSF data) and validate before trusting it.

### `HYPERSCALER_CAPEX_YOY` (tier: ai_cycle)
- **What it is:** year-over-year growth in hyperscaler (cloud) capital
  expenditure — an AI-cycle demand signal.
- **Why empty:** this is not a macro series at all. It comes from
  individual company earnings reports (MSFT, GOOGL, AMZN, META).
- **Sourcing options:**
  1. **Manual / scraped quarterly update** from the four hyperscalers'
     earnings releases. Low frequency (quarterly), small effort per
     update, but ongoing.
  2. **Buy** an aggregated capex dataset from a financial-data vendor.
- **Honest scope note:** this feature originated in the *equity* project's
  AI investment thesis. Whether it belongs in a *recession* model is a
  real question — an AI-capex slowdown is a sector signal, not obviously
  a US-recession lead. Recommended: defer, and decide if it is in scope
  for the recession model at all before investing in sourcing it.

### `MEMORY_CONTRACT_PX` (tier: ai_cycle)
- **What it is:** memory (DRAM/NAND) contract prices — a semiconductor /
  AI-cycle indicator.
- **Why empty:** industry data, not macro. Published by industry sources
  such as TrendForce / DRAMeXchange — largely paid.
- **Sourcing options:**
  1. **Buy** the industry data feed.
  2. **Proxy** with a public semiconductor-related series.
- **Honest scope note:** same as `HYPERSCALER_CAPEX_YOY` — this is an
  equity-thesis feature. Recommended: defer; decide scope first.

---

## SUMMARY — recommended order

1. **Tier 1 (3 features)** — `BAMLH0A0HYM2`, `USSLIND`, `EXHOSLUSM495S`.
   Routine FRED ingestion. Do these first; `BAMLH0A0HYM2` (high-yield
   spread) is the highest-value of the three.
2. **Tier 2 (`COPPER_GOLD`)** — ingest its two FRED inputs (`PCOPPUSDM`,
   `GOLDAMGBD228NLBM`), then run the derivation script. Code is done.
3. **Tier 3 (3 features)** — `CHINA_CREDIT_IMPULSE`,
   `HYPERSCALER_CAPEX_YOY`, `MEMORY_CONTRACT_PX`. Genuine projects.
   Defer until the feature-testing experiments show whether the `global`
   and `ai_cycle` tiers carry signal — and, for the two `ai_cycle`
   features, decide first whether they belong in a recession model at
   all rather than the equity project.

**Discipline note.** Ingesting a feature does not make it useful — it
makes it *testable*. After any of these is populated, it must go through
the same pre-registered, walk-forward validation as every other feature
before any model uses it. The constraint on the model is not missing
data; it is *untested* data — there are already 25 populated-but-untested
features. Sourcing new data and testing existing data are separate jobs;
do not let the first outrun the second.
