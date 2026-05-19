# Feature Backlog — Empty Registry Features

*The feature audit (recession/validation/feature_audit.py) found 7
features registered in `features_registry` with NO data in
`features_monthly`. This document records the truth about each — and the
truth, established by reading `recession/data/series_specs.py`, is that
**none of the 7 is an accidental gap.** They fall into two groups, both
the result of deliberate, documented decisions.*

---

## GROUP 1 — Deprecated: replaced, the replacement is already live (3)

These three were each evaluated, found to have a data-source problem, and
**replaced with a better series that is already wired in and populated.**
The empty registry rows are kept only as foreign-key anchors and as a
record of the decision — `features_monthly.feature_name` has a foreign
key to `features_registry`, and `recession/tests/test_schema.py` uses one
of these names as a vintage-test fixture, so the rows are NOT deleted.
They are instead **marked DEPRECATED in their registry description**, so
the feature audit reports them as a resolved decision, not an open gap.

| Empty feature | Replaced by (live, populated) | Why replaced |
|---|---|---|
| `BAMLH0A0HYM2` | **`BAA10Y`** | ICE BofA license change (Apr 2026) capped BAMLH0A0HYM2 to 3 years rolling history. BAA10Y (Moody's Baa - 10y Treasury) covers the same credit-stress role, daily back to 1986, no licensing risk. |
| `USSLIND` | **`CFNAI`** | USSLIND (Conference Board LEI) was discontinued / went stale. CFNAI (Chicago Fed National Activity Index) covers the same real-activity-lead role. |
| `EXHOSLUSM495S` | **`HSN1F`** | EXHOSLUSM495S (existing home sales, NAR-licensed) has no vintage history on ALFRED. HSN1F (new one-family house sales) has full ALFRED vintage history - required for honest point-in-time backtests. |

**Action: none beyond marking.** The replacements are live. The audit, once
the descriptions are tagged `DEPRECATED`, reports these correctly and
does not re-flag them. Do NOT re-ingest these series - `BAMLH0A0HYM2` in
particular would fail (the license change is why it is empty), and all
three would duplicate a role already filled.

---

## GROUP 2 — Deliberately deferred to v2: `skip_v1` (4)

These four carry `fetch_method="skip_v1"` in `series_specs.py` - the
project's explicit "out of scope for v1, on purpose" marker. They are
genuine future work, not oversights. Each entry below records the honest
sourcing options.

### `COPPER_GOLD` (tier 7, global) — engineered feature, blocked
- **What it is:** the copper-to-gold price ratio - a market-based
  growth/risk gauge.
- **Why deferred:** both London Bullion gold series on FRED
  (`GOLDAMGBD228NLBM`, `GOLDPMGBD228NLBM`) were discontinued in 2025 and
  now return HTTP 400. Without a working gold series there is no ratio to
  compute. `series_specs.py` sets `derived_from=()` (cleared) and
  `fetch_method="skip_v1"`.
- **v2 fix:** source gold from an alternative - Yahoo Finance (`GC=F`
  futures), LBMA direct, or a data vendor. The derivation code is already
  written and tested: `recession/features/derive_copper_gold.py` (it
  reads a copper series and a gold series from `features_monthly`,
  computes the PIT-safe ratio, and writes `COPPER_GOLD`). It is
  **v2-ready** - it only needs a working gold series ingested first.

### `CHINA_CREDIT_IMPULSE` (tier 7, global)
- **What it is:** the change in new credit flow as a share of GDP in
  China - a global growth lead.
- **Why deferred:** no clean free public series. Constructed from China's
  Total Social Financing and GDP; methodology choices matter.
- **v2 options:** (1) construct from public PBoC/NBS TSF data; (2) buy a
  ready-made series (Bloomberg / research vendor); (3) proxy with a
  cleaner adjacent China-activity series.

### `HYPERSCALER_CAPEX_YOY` (tier 8, ai_cycle)
- **What it is:** YoY growth in hyperscaler (cloud) capital expenditure.
- **Why deferred:** not a macro series - it comes from individual company
  earnings reports (MSFT, GOOGL, AMZN, META).
- **Scope question first:** this feature originated in the *equity*
  project's AI investment thesis. Before sourcing it, decide whether an
  AI-capex signal belongs in a *recession* model at all.

### `MEMORY_CONTRACT_PX` (tier 8, ai_cycle)
- **What it is:** memory (DRAM/NAND) contract prices - a semiconductor /
  AI-cycle indicator.
- **Why deferred:** industry data (TrendForce / DRAMeXchange), largely
  paid; not a macro series.
- **Scope question first:** same as `HYPERSCALER_CAPEX_YOY` - an
  equity-thesis feature; decide recession-model scope before sourcing.

---

## SUMMARY

- **Group 1 (3 features)** - already solved. Replacements `BAA10Y`,
  `CFNAI`, `HSN1F` are live and populated. The registry rows are kept as
  FK anchors and marked `DEPRECATED`. No further action.
- **Group 2 (4 features)** - deliberate v2 work. `COPPER_GOLD` is the
  closest (code done, needs a gold series). The other three are genuine
  data-procurement decisions, and the two `ai_cycle` features need a
  scope decision first.

**Discipline note.** The project's real opportunity is not these 7 empty
rows - it is the **25 populated-but-untested features** already in the
database (EBP, NEAR_TERM_FORWARD, ICSA, SAHMREALTIME, T10Y2Y, and more).
Testing existing data outranks sourcing new data. Sourcing a feature
makes it *testable*, not *useful*; only pre-registered, walk-forward
validation makes it useful.

---

*Corrected after reading `series_specs.py`: an earlier version of this
document mistakenly framed the 7 as a single ingestion to-do list. The
specs show 3 are deprecated-with-live-replacements and 4 are deliberate
`skip_v1` deferrals. This version reflects the code.*
