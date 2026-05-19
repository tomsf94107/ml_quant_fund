# Experiment C — International Regime-Dependence Test (Scoping Document)

*This is a SCOPING document, not a finished experiment. It exists because
Experiment A found EBP's recession signal is regime-conditional (helps in
credit-driven recessions, fails otherwise) but could not PROVE it — the
US has only 2 recessions in the EBP-testable window (2008, 2020). One
credit-driven, one not: one data point per regime. No method proves a
two-regime theory from one example of each.*

*Experiment C is the genuine path to more evidence: other countries had
credit-driven and non-credit recessions on different timelines, which
multiplies the regime-events available. This document makes that project
**bounded and staged** so it can actually be started.*

---

## 1. The Question C Tests

Pre-stated, to be locked properly before Stage 1 runs:

> Is the recession-prediction value of a credit-spread / excess-bond-
> premium-type feature **regime-conditional** — strong for credit-driven
> downturns, weak or negative for non-credit ones — when tested across
> multiple countries and therefore multiple recessions per regime?

Experiment A is the US n=2 observation. C is the multi-country test of
whether that observation generalises.

---

## 2. Why C Is Staged (and not one big project)

"Test 8 countries' credit data" is unbounded — different data sources,
different recession dating, different vintage conventions, 8× the work
before any result. That is why it has never been done in this project.

The fix: **C is staged. Stage 1 is one country, end to end.** If the
existing walk-forward apparatus runs correctly on one foreign country,
it runs on all of them — every later stage is repetition, not new
design. Stage 1 is small enough to actually start; later stages are
mechanical copies.

---

## 3. Stage 1 — One Country, End to End

### 3.1 Country choice — recommended: the United Kingdom

Criteria for the Stage-1 country: long clean data, clear recession
dating, and — critically — a recession history that includes BOTH a
credit-driven and a non-credit recession, so one country already gives a
regime contrast.

**Recommended: the United Kingdom.** Rationale:
- Recession dating is clear and published (ONS / well-documented).
- It has a credit-driven recession (2008, the GFC) AND non-credit ones
  (the 2020 COVID recession; the early-1990s recession is arguably
  policy/ERM-driven, not a credit crisis). Regime contrast within one
  country.
- Credit-spread data: UK corporate bond spreads / the Bank of England's
  corporate-bond-spread datasets, and OECD/BIS series, are obtainable.
- English-language sources; Bank of England research has itself
  examined credit spreads and UK activity — prior art to calibrate
  against.

Alternative if UK data proves awkward: **Canada** (clean StatCan
recession dating, long bond-spread history, GFC + COVID contrast).

### 3.2 Stage-1 data to source

| Item | What | Candidate source |
|---|---|---|
| Recession dates | UK recession periods (the prediction target) | ONS GDP-based dating; or the OECD-based recession indicator for the UK on FRED (series like `GBRRECDM`) |
| Credit spread | A UK corporate-bond-spread / credit-risk-premium series — the EBP-analogue | Bank of England datasets; OECD; BIS credit statistics; ICE/Bloomberg UK corporate spread indices if licensing allows |
| Baseline | A UK term-spread (long minus short government yield) — the M1-analogue baseline to test the credit feature AGAINST | OECD / BoE government yield data; FRED has UK yield series |

Honest data-risk note: the US EBP is a *specific* Gilchrist-Zakrajšek
construction not trivially reproduced abroad. Stage 1 will likely use a
simpler credit-spread measure (a corporate-minus-government spread) as
the EBP-analogue, not a full EBP reconstruction. That is acceptable — the
question is about the credit channel, and a corporate spread captures it;
a full EBP reconstruction is a later refinement, not a Stage-1 blocker.

### 3.3 Stage-1 method — reuse, do not rebuild

The walk-forward harness, the nested LR test, and the robustness check
are country-agnostic — they take a feature panel and a target. Stage 1:
1. Ingest the UK recession target + UK term-spread + UK credit-spread
   into a parallel table (or a separate small DB) in the same shape the
   builder expects.
2. Run the SAME `candidate_eval` two-gate test: does the UK credit
   spread beat the UK term-spread baseline?
3. Run the SAME `robustness_check` fold-by-fold: does any UK edge hold
   across the UK's credit (2008) vs non-credit (2020, 1990s) recessions?
4. Report. One country, one regime-contrast, using validated machinery.

### 3.4 Stage-1 effort estimate (honest)

- Data sourcing + cleaning: the real cost — UK series, vintage handling,
  aligning recession dates. Realistically the bulk of the work.
- Wiring into the builder's expected shape: moderate — a new ingestion
  path, but the schema is known.
- Running the experiment: small — the harness already exists.
- **Stage 1 is a real piece of work, not a one-session task — but it is
  BOUNDED: one country, known sources, existing apparatus.**

---

## 4. Stages 2+ — Mechanical Repetition

Each later stage adds one country: Canada, Germany, France, Australia,
Japan, etc. — each a copy of Stage 1 with new data. The Bank-of-England /
8-European-economies credit-spread literature (Bleaney, Mizen, Veleanu)
is prior art identifying which countries have usable series.

After ~5-8 countries the project has dozens of recession-events spanning
credit and non-credit regimes — enough to actually test the regime-
dependence hypothesis with more than one example per regime.

**Honest limit:** even all stages give maybe 30-50 recession-events
globally. Better than n=2; still not "proof" in a strong sense.
Recessions are rare events — this is the project's founding constraint
and C reduces it, it does not remove it.

---

## 5. Pre-Registration Requirement

Before Stage 1 RUNS (not before sourcing data), a proper pre-registration
must be written — like A_PREREGISTRATION.md and B_TRACK_PREREGISTRATION.md
— locking: the country, the exact series, the two gates, the robustness
criterion, and the hypotheses. Sourcing data is allowed before that;
running the test is not.

---

## 6. Status

- **C is scoped, not started.** This document is the bounded plan.
- **Stage 1 (UK) is the concrete next task** whenever the international
  work is picked up — it has a country, named data sources, a method
  (reuse the existing harness), and a pre-registration requirement.
- C is independent of Experiment B (the broad feature sweep). They can
  proceed in either order or in parallel.

---

*Scoping document. Stage 1 requires its own pre-registration before any
test is run.*
