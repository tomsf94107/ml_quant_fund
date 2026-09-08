# CEWS — Reconciled Current State & Action List
**2026-09-08 · successor to `warning/HANDOFF_CRASH_WARNING_2026-08-30.md`.**
**Supersedes `warning/CEWS_DEFECT_LEDGER_2026-09-07.md` §1–§11 as the *actionable* document.**

The source ledger is a chronological record of five diagnostic pastes. Later pastes overturned
earlier hypotheses without deleting them. This document keeps only what survived, in one
ordered list. Retain the original as the evidence log — do **not** action it top-down.

Tags: `[fact]` = in the source ledger's evidence column · `[inf]` = reasoning ·
`[unconfirmed]` = needs a check on the Mac.

---

## 0. READ FIRST — do this before anything else

`warning/HANDOFF_CRASH_WARNING_2026-08-28.md`, `…08-30.md`, and `warning/DECISIONS.md`
post-date the 25 Aug brief the ledger was written against. `[fact — source ledger §11]`
Items below may already be decided there. **Reconcile, then act.** A fix that contradicts a
recorded decision is worse than no fix.

---

## 1. Current state (2026-09-07 re-step, post manual backfill) `[fact]`

| Layer | Coverage | Gate | Status | Blocked by |
|---|---|---|---|---|
| L1 | 25% | 0.70 | **NA** | S10, S13, S15 unbuilt |
| L2 | 78% | 0.70 | passes, score 0.000 | — (S1 R, S6 R, neither persisted) |
| L3 | 100% | 0.70 | passes, score 0.000 | — |
| L4 | 40% | 0.70 | **NA** | L4C stale (Cboe COR); L4D ← S10; L4E ← F9 |

Gate: `STALE_COVERAGE_MIN = 0.70`, coverage = usable **weight** ÷ total weight
(`warning_engine.py:44,135-150`). No builder passes `weight`, so every reading uses the dataclass
default of 1.0 and coverage is a plain fraction of signals. `[fact — grep of warning/builders/]`

Composite is **NULL / frozen**. That freeze is correct behaviour, not a defect. `[fact]`

Remaining stale/NA on 7 Sep: `S9` (stale), `L4C` (stale), `S10/S12/S13/S15/L4D/L4E` (NA).
Everything else fresh at 3–4 Sep. `[fact]`

---

## 2. Superseded — do NOT action these

| Retired claim | Where it was overturned | Current truth |
|---|---|---|
| A1: one stalled Massive feed | §5, §8 | **Three feeds, three causes** — see §3 |
| §5: daily `--only cboe` fixes F2/F3 | §7 | F2/F3 are **FRED**-fed (`VIXCLS`, `VXVCLS`, `f3_vix_term_slope.py:56-57`). The FRED job is the fix. Cboe leg only serves series FRED lacks → **L4C** |
| §6/§8: FRED timeout = payload size | §9 | Interactive FRED = 200, 177 KB, 1.83 s. `IORB` (1.8k rows) also timed out → **connection stall, not payload**. Retry/backoff is the fix; daily-vs-weekly split is a freshness win, not the timeout fix |
| A2: F3 uses `ORDER BY … LIMIT 1` | §7 | F3 has a **primary→fallback chain** with no freshness check on the fallback: `elif fut_slopes:` takes 2018 VX futures unconditionally |
| §10: bump S9 staleness limit to ≥26 d | §11 | S9's builder stamps the **settlement** date instead of `pub_date`. Fix the builder. **Bumping the limit would mask the bug** |
| §8 fixed-clock cron lines (05:15/20/30) | §9 | Sequenced wrapper gated on the ADB chain. **Install one scheme only** |

---

## 3. The three broken feeds `[fact, per §5/§7/§8/§9]`

| Feed | Symptom | Cause | Fix |
|---|---|---|---|
| **FRED weekly** | All 18 series `read operation timed out` | `urlopen(req, timeout=60)`, no retry, no per-series commit. Sunday 16:00 VN slot | Retry ×3 backoff 10/30/90 s + **per-series commit**; move off the Sunday slot; split daily vs weekly series |
| **Massive → warning.db** | `SPY_CLOSE`/`BREADTH_*` last 29 Aug | `warning/ingest_spx.py` + `ingest_breadth.py` are **not scheduled at all**. Last rows = last manual run | Sequence after `pipeline_chain_ADB.sh` (launchd, 04:00 VN) — §5 wrapper |
| **Cboe** | COR1M/3M, SKEW, VVIX, VIX9D last 28 Aug | Weekly cron runs `--only fred`; the cboe leg ran once | Add `cboe` to the daily leg list. Lifts **L4C** only |

Legs `--only` accepts: `fred, alfred, cboe, cfe, french, ritter`. `[fact]`

---

## 4. Two parallel P0 tracks — do not conflate

- **Track A — unfreeze the composite.** Feeds + the S13/S10 builders.
- **Track B — stop corrupting history.** Every scheduled run without B8/B1/B2 writes a
  permanent point-in-time violation into `signal_values`. Track B is cheap **now** and
  expensive later, and it does not wait on Track A.

**Fix B8 before the next scheduled 06:00 VN run.** `[inf]`

---

## 5. Ordered action list

### P0-B — integrity (do first; blocks nothing, corrupts everything)

**B8 — `asof_date` must be a completed trading day.** `[fact]`
28 Aug rows written 13:31–17:47 UTC (mid-session) labelled `2026-08-28`; a Sun 30 Aug run
wrote `asof_date = 2026-08-30`.
Fix: `asof_date` = last completed US trading day from an exchange calendar
(`pandas_market_calendars` XNYS); driver refuses to run before close + 60 min; never labels a
non-trading day.

> **Ordering constraint (not in the source ledger):** B8 must land **before** the B1 rebuild.
> The B1 gaps-and-islands query counts consecutive `asof_date`s — running it while a Sunday row
> and a mislabelled mid-session row are still present bakes both into the persistence history. `[inf]`

**B2 — `runs` table.**
`runs(run_id PK, asof_date, started_at, finished_at, status, registry_version, engine_version,
git_sha, is_latest)`. Every `signal_values` / `composite_scores` / `alerts` row carries
`run_id`. A re-run for the same `asof_date` sets the prior run `is_latest = 0`; views select
`is_latest = 1`. Then mark the 28 Aug and 30 Aug rows `is_latest = 0`.

**B1 — persistence becomes derived, not a mutable counter.** `[fact]`

> **Two counters, not one.** `apply_persistence` (`warning_engine.py:114-127`) counts per-signal
> days toward `min_persistence`; `hysteresis_step` (`:228-237`) counts `candidate_days` toward
> `PERSIST_DAYS_DEFAULT=10` / `PERSIST_DAYS_DEFENSIVE=21` for band changes. Both live in the same
> `engine_state` blob and both increment per *invocation*. A fix that addresses only the first
> leaves half the defect. `candidate_days` is 0 today, so nothing is currently corrupted. `[fact]`
>
> **NA does not reset a run** — `apply_persistence:119` returns before touching `st.persistence`,
> so the counter freezes rather than resetting. This is shipped, tested behaviour: reproduce it,
> do not "fix" it. `[fact]`
>
> **Left-censoring is live.** After the rebuild, 6 real `asof_dates` exist. S1 (`persistence_days`
> = 21, currently R on all 6) needs 21 consecutive sessions, so no 21-day signal can reach
> effective state before **2026-09-28** at the earliest, assuming daily runs resume with no gaps.
> Until then S1 contributes G to L2 while reading R — L2's 0.000 score must not be read as
> "nothing happening". `[inf]`
`schema_meta.engine_state` holds counts up to 10 against only **7 distinct asof_dates**;
28 Aug ran six times. The counter increments per *run*, not per *asof_date*. This is the
engine's core hysteresis input, so `effective_state` will fire early for whatever was re-run most.

Fix: compute consecutive same-state `asof_date`s from `signal_values` at step time; store in
`signal_values.effective_state`; delete the mutable counter.

> **Verify the idempotency premise first** — the whole rebuild rests on it:
> ```sql
> SELECT signal_id, asof_date, COUNT(*) c FROM signal_values
> GROUP BY 1,2 HAVING c > 1;
> ```
> Must return zero rows. `[unconfirmed]`
>
> **Two semantics the ledger's SQL sketch does not define — decide before running:**
> 1. Does an `NA` day **break** a run, or is it transparent?
> 2. Do missing trading days (no run at all) break a run?
> Both change every persistence count. `[inf]`

**C2 — `WARNING_DB` env var; dev runs → `warning_dev.db`.** Six dev re-runs reached production. `[fact]`

Acceptance for P0-B: `pytest` green · re-stepping any `asof_date` twice produces byte-identical
`signal_values`/`composite_scores` and no new alerts · `count(distinct asof_date)` in
`signal_values` == `count(*) from runs where is_latest=1` · no `asof_date` on a non-trading day.

---

### P0-A — feeds

**A-1. Fix the FRED job.** Retry ×3 with backoff, per-series commit, off the Sunday slot.
Until it works, S2 / L4B / F10 inputs age out weekly. `[fact]`

**A-2. Schedule the Massive bridge and the Cboe leg** via **one sequenced wrapper**
(`scripts/warning_chain.sh`, draft supplied) — not fixed clock minutes. `pipeline_chain_ADB.sh`
has variable duration; fixed minutes race it. `[inf]`

**A-3. Replace the wrapper's log-grep gate with a done-file.** Add as the chain's last line:
`touch logs/pipeline_ADB.$(date +%F).done`. Grepping a log for today's date is fragile —
a date string can appear in a *failure* line. `[inf]`

**A-4. Cron ordering:** `daily_driver.py` (06:00 VN) currently runs **before** `uw_archiver.py`
(06:30 VN). Harmless today, wrong once Phase 5 UW signals exist. The wrapper fixes this by
sequencing. `[fact]`

---

### P1 — builders (the actual unlock)

Feeds alone cannot unfreeze the composite: **L1 caps at 25% and L4 at ≤60% on unbuilt signals
even with every feed green.** `[fact]`

| Order | Signal | Why | Effort |
|---|---|---|---|
| 1 | **S13** valuation gate | Shiller `ie_data.xls`, one file, one running percentile. Simplest | low |
| 2 | **S10** margin debt | **Keystone — gates BOTH L1 and L4D.** With S13: L1 = 75% **passes** | med, manual |
| 3 | L4C Cboe leg | Falls out of A-2 | trivial |
| 4 | S15 credit-boom | FRED Z.1 + Case-Shiller, two legs, quarterly. Hardest | high |

**Result after 1–3:** L1 = 75% pass · L4 = 80% (L4A+L4B+L4C+L4D) pass.
L4E stays NA until F9 (needs months of UW archive) — record it as `unbuilt`, not `NA`.

**S10 work:** (1) download the FINRA margin-statistics xlsx (VPN), (2)
`warning/ingest_finra_margin.py` → `data_vintages` (monthly, pub = release date, +3 w lag),
(3) `builders/s10_*.py`, (4) unblock L4D (`l4_propagation.py:68` hard-codes the NA reason). `[fact]`

> **Fragility flag:** a monthly manual VPN xlsx download now sits on the critical path for two
> layers. It will rot. Needs a runbook entry with an explicit owner, or an automated fetch. `[inf]`

---

### P1 — the stale-source bug class (generalise, don't patch one builder)

The F3 patch in source-ledger §7 is correct and should land. But F3 is one instance.
Same class, already in `data_vintages`:

| Series | Dead since | Note |
|---|---|---|
| `VX_FRONT` / `VX_SECOND` | 2018-02-24 | Scraper points at a URL Cboe retired ~2018 — fixable |
| `CBOE_PC_*` | 2019-10-05 | Cboe stopped the free daily P/C files; F1 needs the new site or UW |
| `CBOE_VXO` / `VXOCLS` | 2021-09-24 | Covered by the brief's backfill guard |
| `TEDRATE` | 2022-01-22 | **GUARDED — not a defect.** `series_meta.py:56` flags it False; `s4_funding.py:19,79-80` runs historic/modern modes and reads its staleness; `test_builders.py:706-720` tests that auto mode stops using it. Corrects the source ledger. |

Live exposure is **F3 only** — S4 and S14 both handle their dead legs correctly. The invariant is still worth having as a guard against future builders: *no source past its staleness limit may emit a
state — the builder returns NA.* Applies to every leg of every fallback chain. Enforce centrally,
then the per-builder patches are belt-and-braces. `[inf]`

`s14_vol_structure.py:146-154` already drops the dead futures leg correctly — use it as the
reference implementation. `[fact]`

**A2/F3 note:** F3 is dashboard tier, so coverage was unaffected. The same pattern in a
**counted** builder would poison layer state silently. `[fact]`

---

### P1 — schema (all endorsed as written in the source ledger, no changes)

| ID | Fix |
|---|---|
| B3 | Split `events(run_id, asof_date, kind, layer, detail JSON)` from `alerts` (transitions only, real `from_state`/`to_state`). Currently 52 rows, 0 transitions |
| B4 / B9 | `registry_versions` + `signal_registry` tables; `signal_values.registry_version NOT NULL REFERENCES`; loader refuses an unversioned CSV |
| B5 | `layer_scores(run_id, asof_date, layer, n_counted, n_live, n_stale, n_na, n_unbuilt, gate, passed, score)` |
| B6 | `age_days` + `limit_days` on `signal_values` — two integers, makes every staleness decision reproducible |
| A3 | `build_status` column (`unbuilt \| built \| retired`) so NA means *data missing*, not *code missing* |

**Interim registry_version fix (10 lines, do now):** `daily_driver.py:134,144` write the literal
`"unversioned"`. Read `warning/signal_registry.csv` once per run, stamp `sha256[:12]` into every
row, refuse to run if the file is missing. Closes B4's unenforceable half ahead of the full
table. `[fact]`

---

### P1 — ELEVATED from the source ledger's P2/unconfirmed

**B10 — NOT YET WIRED.** `grep -rn composite_score features/ signals/` returns nothing: the emit
does not exist, so there is no live contamination today. This is a build-it-right requirement, not
an active leak — my earlier present-tense framing was wrong. When it is built: the consumer must
**mask** a NULL composite, never impute 0 or `NORMAL`, or a frozen crash-warning becomes a
"market is normal" feature in the fund's models. `[fact — grep returned empty]`

Verify before touching anything else in the emit path:
```
grep -rn "composite_score" features/ signals/ | head
```

---

### P2

- **S9 staleness** — fix the builder's stamped date (`pub_date`, not settlement) in
  `s9_short_interest.py`. **Do not also bump `max_staleness_days`** — that masks it. `[fact/inf]`
- **A4 general** — audit `staleness_limit` vs `cadence + publication_lag + 2 bd` per registry row,
  at the next version bump, not as mid-year single-row edits.
- **A5 / OAS** — pull the two OAS series daily (cheap); weekly cadence stays for everything else.
- **B7 — `zscore`** needs the Phase 1 backfill. Until then: split `raw_sub_score` /
  `effective_sub_score`, and write **NULL, never 0**, with the dashboard saying so.
- **C3 — ALREADY AUTOMATED.** `exports/cews_engine_json.py` runs 06:05 VN Tue–Sat (crontab line 359), after the driver. Corrects the source ledger. Fold into the wrapper and delete the standalone line, or it fires on its own clock.
- **French / Ritter files** last pulled 1 Jul / 31 Mar — not covered by the `--only fred` cron.
  Monthly job.

---

## 6. Open decisions — owner, not code

1. **Does `unbuilt` count against coverage denominators?**
   For a crash-warning system the conservative default is *yes, it counts as missing* — the
   composite stays NULL until the signal exists. Excluding `unbuilt` would let a composite fire
   on 2-of-4 L1 signals and call itself full coverage. Recommend counting it. **Decide, don't
   code around it.** `[inf]`

2. **Source lag vs run time.** 06:00 VN Tue = 19:00 ET **Mon** `[fact: VN = ET+11]`. Cboe/FRED
   post the prior session's values on the following morning ET, so free sources are structurally
   ~1 session behind the driver's `asof_date`. B8 fixes *run labelling*; it does not fix
   *per-source lag*. Either accept T-1 for slow sources explicitly in the registry, or move the
   driver. `[inf]` `[unconfirmed against real posting times]`

3. **B1 semantics** — does NA break a persistence run? Do missing days? (see P0-B above)

---

## 7. Verify-first command block

Nothing below changes state. Run before implementing; several fixes above depend on the answers.

```bash
cd ~/Desktop/ML_Quant_Fund

# 0. Reconcile with the newer decision docs FIRST
ls -la warning/HANDOFF_CRASH_WARNING_*.md warning/DECISIONS.md

# 1. B1's premise: is signal_values really 1 row per (asof_date, signal_id)?
sqlite3 warning.db "SELECT signal_id, asof_date, COUNT(*) c FROM signal_values
                    GROUP BY 1,2 HAVING c > 1 LIMIT 20;"

# 2. B10 — the only item touching the live trading stack
grep -rn "composite_score" features/ signals/ | head

# 3. Dead series reaching live builders (the F3 class)
grep -rn "TEDRATE\|VX_FRONT\|CBOE_PC" warning/ features/ signals/ 2>/dev/null | grep -v __pycache__

# 4. Confirm the CLI flags before writing any cron line
python warning/ingest_spx.py --help | head -30
python warning/ingest_breadth.py --help | head -30

# 5. What is currently scheduled (avoid double-install)
crontab -l | grep -nE 'warning|fred|cboe|ingest|driver|archiver'
launchctl list | grep -i -E 'quant|pipeline'

# 6. Does pipeline_chain_ADB.sh write a completion marker?
tail -5 scripts/pipeline_chain_ADB.sh
```

---

## ASSUMPTIONS

- The source ledger's `[fact]` items are accurate to the Mac. This document **re-reads and
  reconciles the ledger; it does not re-verify the machine.** `[unconfirmed]`
- `pandas_market_calendars` is available in `ml_quant_310`, or an equivalent XNYS calendar is.
  `[unconfirmed]`
- The wrapper draft's CLI flags are placeholders — they come from the source ledger, not from
  `--help`. Verify before installing. `[unconfirmed]`
- VN = UTC+7, ET offset 11 h at this date. `[unconfirmed — DST-sensitive; the fund's crontab
  already requires manual DST edits in Mar/Nov]`
- Coverage = usable weight ÷ total weight, F-tier excluded. `[unconfirmed]`
- Date stamped from session context, not from a clock call. `[unconfirmed]`
