# CEWS — Defect Ledger & Remediation Plan
Source: `cews_export.json` (warning.db, 7 runs 2026-08-28 → 2026-09-04) + `HANDOFF_BUILD_BRIEF.md` v1.0.
Written 2026-09-07. Tags: [fact] = visible in the export · [inf] = inference from it · [unconfirmed] = needs a check on the Mac.
Hand this to the build session as-is. Nothing here changes a threshold or a signal choice (brief rule 3).

---

## 0. What is NOT a problem
- The freeze itself. `INSUFFICIENT_DATA` with `composite NULL`, `action_gross NULL`, `action_hedge = freeze` on every run is the specified behaviour when a layer is below gate. [fact]
- `signal_values` is idempotent per (asof_date, signal_id): 22 rows per run even on the day with six re-runs. [fact]
- Staleness flagging works: it caught F2, F3, S5, S9, S14, L4C within the limits. [fact]
- OAS archive is accumulating (791 rows) and the UW archiver is writing daily. [fact]

---

## 1. Data-flow defects

### A1 — Market-data feed stalled at 2026-08-27  · P0 · hard block
- Evidence [fact]: F2, F3, S4, S5, S6, S7, S8, S14, L4A, L4C carry `source_asof = 2026-08-27` in all seven runs. Staleness flags flipped from 2 Sep. Coverage fell L2 89%→67%, L3 100%→50%, L4 60%→40%.
- Likely cause [inf]: a one-shot Massive pull on 27 Aug and no daily ingest job; or a failing job. The engine cron (23:00 UTC) is running; the ingest is not.
- Verify:
  ```
  sqlite3 warning.db "select series_id, max(pub_date), max(ref_date), count(*) from data_vintages group by 1 order by 2;"
  crontab -l | grep -iE 'massive|ingest|fetch|warning|kill'
  ls -la scripts/ | grep -iE 'ingest|massive|fetch|universe'
  ```
- Fix: daily Massive ingest in the same cron slot, ordered **before** the engine step. Add a feed-level freshness check in the daily driver: if `max(ref_date)` for the universe < last trading day, write a `FEED_STALE` event and skip the builders that depend on it (they will read NA, not a stale row).

### A2 — F3 builder falls back to an 8-year-old row  · P1 · defect
- Evidence [fact]: from 2 Sep, F3 `source_asof = 2018-02-23`, raw 0.0075 (was 0.2102 on 27 Aug). State still "G", flagged stale.
- Cause [inf]: lookup returns the oldest row when the current window is empty (`ORDER BY date LIMIT 1` without `DESC`, or `.iloc[0]` on an unsorted frame, or an explicit fallback).
- Fix: builders return NA when no row falls inside `[asof − staleness_limit, asof]`; never search outside the window. Add to `test_warning_engine.py` (or a new `test_builders.py`): "current row missing → NA, never an older row". Add a driver invariant: `source_asof < asof − limit` ⇒ `state = NA`, not a state + stale flag.
- Scope note: F3 is dashboard tier so coverage was unaffected; the same pattern in a counted builder would poison state silently.

### A3 — Six signals have never produced a row  · P1 · hard block
- Evidence [fact]: S10, S13, S15 (L1), S12 (L2), L4D, L4E (L4) — `NA`, `source_asof NULL`, every run.
- Consequence [fact]: L1 caps at 25% (1/4). L4 caps at 60% (3/5) even after A1 is fixed — below the gate. Both layers are hard-blocked until code lands.
- Fix (build order): S10, S13, S15 and **one of** L4D/L4E are the critical path to a non-NULL composite. S12 is not on the critical path (L2 reaches 89% without it).
- Structural fix: the registry needs a `build_status` column (`unbuilt | built | retired`) so "NA" means *data missing*, not *code missing*. Coverage denominators should exclude `unbuilt` only if the brief's owner decides so — this is a policy question, flag it, don't decide it in code.

### A4 — Staleness limits vs publication cadence  · P2 · [unconfirmed]
- Evidence [fact]: S9 (monthly source, `source_asof` 14 Aug) went stale on 4 Sep. S11 (annual, 31 Dec 2025) is not stale. L4B updated 26 Aug → 1 Sep and was stale for two runs in between.
- Inference [inf]: S9's limit is shorter than its source's monthly cadence + 3-week lag, so it will read stale for most of every month by construction.
- Verify: `signal_registry.csv` rows S9, L4B — compare `staleness_limit` with `publication_lag + cadence`.
- Fix: limit ≥ cadence + lag + 2 business days, per row, at the next registry version bump (not mid-year edits to individual rows).

### A5 — S2 (HY OAS) fed by a weekly FRED append  · P2
- Evidence [fact]: OAS vintages end 2 Sep on 7 Sep; S2 `source_asof` 1 Sep on the 4 Sep run.
- Risk [inf]: a weekly append leaves a daily series up to 7 days behind; if S2's limit is < 7 days it will trip stale every week-end.
- Fix: pull the two OAS series daily (cheap), keep the weekly cadence for everything else. Append-only stays.

---

## 2. Schema / data-structure defects

### B1 — Persistence counters are not point-in-time and are corrupted by re-runs  · P0
- Evidence [fact]: `schema_meta.engine_state` holds `{"S1": ["R", 10, "G"], "S2": ["G", 10, "G"], "L4B": ["G", 9, "G"], "S4": ["G", 8, "G"], ...}` while only **7 distinct asof_dates** exist. 28 Aug was run six times.
- Inference [inf]: the counter increments per *run*, not per *asof_date*, so re-running a day double-counts. The counts can't be reproduced from `signal_values`.
- Consequence: `effective_state` will fire early for whichever signal was re-run most. This is the engine's core hysteresis input.
- Fix: make persistence a **derived** quantity — consecutive `asof_date`s with the same state, computed from `signal_values` at step time — and store the result in `signal_values.effective_state`. Delete the mutable counter from `schema_meta` (keep `engine_state` only for band/candidate, which are also recomputable). Rebuild now:
  ```sql
  -- consecutive-days-in-state per signal, from history (sketch)
  WITH s AS (
    SELECT signal_id, asof_date, state,
           ROW_NUMBER() OVER (PARTITION BY signal_id ORDER BY asof_date) rn,
           ROW_NUMBER() OVER (PARTITION BY signal_id, state ORDER BY asof_date) rs
    FROM signal_values)
  SELECT signal_id, state, COUNT(*) AS run_len, MAX(asof_date) AS last
  FROM s GROUP BY signal_id, state, rn - rs ORDER BY signal_id, last;
  ```
  Add a test: "same asof_date stepped twice → identical persistence".

### B2 — No `runs` table; re-runs accumulate instead of supersede  · P0
- Evidence [fact]: alerts ids 1–30 are six repeats of the same five alerts on 28 Aug (13:31–17:47 UTC).
- Fix: `runs(run_id PK, asof_date, started_at, finished_at, status, registry_version, engine_version, git_sha, is_latest)`. Every row in `signal_values`, `composite_scores`, `alerts` carries `run_id`. A re-run for the same `asof_date` marks the previous run `is_latest = 0`; views select `is_latest = 1`. Dev iterations then stop polluting the production log, and the brief's "what ran / what it wrote" reporting has a home.

### B3 — `alerts` is an event log wearing an alert schema  · P1
- Evidence [fact]: 52 rows, 0 band transitions. `LAYER_NA` rows put a layer name in `from_state`; `INSUFFICIENT_DATA` rows have `from_state = to_state = NORMAL`.
- Fix: split. `events(run_id, asof_date, kind, layer, detail JSON)` for per-run facts (layer NA, feed stale, freeze). `alerts` only for transitions (`band`, `path`, `effective_state` changes, L4 override, re-entry ladder steps) with proper `from_state`/`to_state`. The dashboard's alert log then means something.

### B4 — Registry version is a free-text string with no anchor  · P1 · brief rule 3
- Evidence [fact]: `registry_version = "unversioned"` on all 161 signal/composite rows; `schema_meta.registry_version` = "UNSET — set to signal_registry.version on first load".
- Fix: `registry_versions(version PK, loaded_at, sha256, row_count)` + `signal_registry(version, signal_id, layer, tier, formula, source, series_ids, history_start, pub_lag_days, thresholds JSON, persistence_days, staleness_limit_days, build_status)` loaded from the CSV. `signal_values.registry_version` becomes `NOT NULL REFERENCES registry_versions`. The loader refuses to run with an unversioned CSV. Thresholds actually used are then auditable per run — that is what rule 3 needs to be enforceable.

### B5 — Coverage is stored as a ratio only  · P1
- Evidence [fact]: `l1_cov = 0.25` etc. The numerators/denominators, the gate, and which signals were counted are not stored; the dashboard had to infer them.
- Fix: `layer_scores(run_id, asof_date, layer, n_counted, n_live, n_stale, n_na, n_unbuilt, gate, passed, score)`. Keep the ratio columns on `composite_scores` for convenience, derived.

### B6 — `stale` is a flag with no audit trail  · P1
- Evidence [fact]: `stale = 1` with no age or limit stored.
- Fix: add `age_days` and `limit_days` to `signal_values`. Two integers; makes every staleness decision reproducible and gives the dashboard its staleness column for free.

### B7 — `zscore` NULL everywhere, `sub_score` 0.0 for a red signal  · P2 · brief rule 9
- Evidence [fact]: S1 is R in every run with `sub_score 0.0` and `zscore NULL`.
- Reading [inf]: sub_score is post-persistence (effective state NULL ⇒ 0). Fine — but then rule 9 ("every probability decomposes into coefficients × z") has no z to decompose.
- Fix: two columns — `raw_sub_score` (from state/threshold) and `effective_sub_score` (after persistence). `zscore` needs signal history (Phase 1 backfill); until then write NULL and say so in the dashboard, never 0.

### B8 — `asof_date` is not a trading-day, post-close date  · P0 · brief rule 1
- Evidence [fact]: 28 Aug rows written 13:31–17:47 UTC (09:31–13:47 ET, mid-session) with `asof_date = 2026-08-28`. A run on Sun 30 Aug wrote `asof_date = 2026-08-30`; Sat 29 Aug has none.
- Consequence: mid-session data labelled as the day's close is a point-in-time violation that stays in the history.
- Fix: `asof_date` = last completed US trading day from an exchange calendar (`pandas_market_calendars` XNYS); the driver refuses to run before close + 60 min and never labels a non-trading day. Mark the 28 Aug and 30 Aug rows `is_latest = 0` once `runs` exists, and re-step 28 Aug from archived vintages if they exist.

### B9 — No signal dimension table  · P1
- Evidence [fact]: layer membership, tier, names, limits exist only in the CSV and in code. The dashboard could not name a single signal beyond what the brief mentions.
- Fix: covered by B4 — `signal_registry` in the DB.

### B10 — `prediction_features` receives NULL composite  · [unconfirmed]
- Brief: emit `composite_score, path_label, funding_z, breadth_div_days, epicenter_flag` to the ML stack.
- Check that the consumer treats NULL as missing (mask/skip), not as 0 or as "NORMAL". If it imputes, that is a rule-2 violation downstream.
- Verify: `grep -rn "composite_score" features/ signals/ | head`.

---

## 3. Process defects

### C1 — Phase order diverged from the brief  · P1
- Evidence [fact]: Phase 3 signals S3, S5–S9, S11, S14 have builders; Phase 2 core S10, S13, S15 do not; F5, F10 are absent from the DB entirely.
- Consequence: no amount of further Phase 3 work can unfreeze the composite.
- Fix: stop Phase 3 work until S10, S13, S15 and one L4D/L4E builder exist. F10's series IDs are resolved (`DCPF3M` daily / `CPF3M` monthly, both live on FRED, minus `DTB3` or `DGS3MO` per registry).

### C2 — Development runs hit the production DB  · P1
- Evidence [fact]: six re-runs on 28 Aug persisted alerts and inflated persistence counters.
- Fix: `WARNING_DB` env var; dev runs go to `warning_dev.db`; `runs.is_latest` supersession for legitimate re-steps.

### C3 — Dashboard refresh is manual  · P2
- Fix: `scripts/export_dashboard_json.py` runs after the engine step in the same cron and writes `exports/cews_latest.json` (composite + layer_scores + latest signal state + last 60 events/alerts + schema_meta). Attach it, or link the Mac and I read it from the connected folder.

---

## 4. Order of work

| Priority | Item | Why first |
|---|---|---|
| P0 | A1 feed ingest | Every day without it adds stale history |
| P0 | B8 trading-calendar asof_date | Each run without it writes a point-in-time violation |
| P0 | B1 + B2 persistence from history, `runs` table | Counters are already wrong; the fix is a rebuild, cheap now, expensive later |
| P1 | A3 S10/S13/S15 + one of L4D/L4E | Critical path to a non-NULL composite |
| P1 | A2 F3 fix + builder test | Same bug class will recur in counted builders |
| P1 | B4/B9 registry in DB, versioned | Rule 3 enforcement; also unblocks naming on the dashboard |
| P1 | B3 events vs alerts, B5 layer_scores, B6 age/limit | Makes the daily log readable and auditable |
| P2 | A4/A5 staleness limits, daily OAS pull | Reduces false-stale noise |
| P2 | B7 z-scores (needs Phase 1 backfill) | Rule 9 |
| P2 | C3 export automation | Dashboard refresh |

Acceptance for the P0 block: `pytest` green; re-stepping any asof_date twice produces byte-identical `signal_values`/`composite_scores` rows and no new alerts; `select count(distinct asof_date) from signal_values` equals `select count(*) from runs where is_latest=1`; no `asof_date` falls on a non-trading day.

---

## ASSUMPTIONS
- Coverage = live counted ÷ counted, F-tier excluded — inferred from ratios [unconfirmed]
- `engine_state` tuple = [state, count, effective] [unconfirmed]
- Massive is the source for the 27-Aug-stuck signals — inferred from the brief's data map, not from code [unconfirmed]
- `data_vintages` has `ref_date`/`pub_date` columns as the brief implies [unconfirmed — adjust the verify queries to the real schema]

---

## 5. Root-cause update (2026-09-07, after `data_vintages` + crontab check)

Supersedes the A1 cause hypothesis above. Three separate feeds stopped, for three different reasons.

| Feed | Last pub_date | Signals it drives [inf] | Cause | Fix |
|---|---|---|---|---|
| Cboe free CSVs (CBOE_VIX, VIX3M, VIX9D, VVIX, SKEW, COR1M/3M) | 28 Aug | F2 (= CBOE_VIX, raw 14.51), F3 (= VIX3M/VIX − 1, 0.2102), probably L4C | Only fetched by `fetch_free_history.py`; the weekly cron runs it with `--only fred`, so the Cboe leg ran once (28 Aug) and never again. [fact: cron line] | Daily job `fetch_free_history.py --only cboe` at ~05:30 VN Tue–Sat (Cboe posts VIX_History next morning ET), before the driver. |
| Massive-derived closes + breadth (SPY/RSP/XL*_CLOSE, BREADTH_*) | 29 Aug | S4, S5–S8, S14, L4A | Pipeline A → warning.db bridge stopped after the 29 Aug run. No Pipeline A cron line surfaced in the grep; crontab carries `MASSIVE_API_KEY=YOUR_MASSIVE_KEY_HERE` (placeholder) and a comment about a job that "stopped silently after the Massive..." with a bare `except Exception: pass`. [fact: crontab text] Which of these applies is [unconfirmed]. | Find the Pipeline A entry (`crontab -l \| grep -n pipeline`), read `logs/` for 30 Aug onward, replace the bare except with a logged failure + non-zero exit, remove the placeholder key from crontab (jobs that don't source `.env` inherit it). |
| FRED weekly (`--only fred`, Sun 16:00 VN) | mixed: DTB3/DGS10 27 Aug, DGS2/BAML*/WTI 2 Sep, T10Y2Y 3 Sep, AAA/BAA/BAA10YM 2 Jul | S2 (= BAMLH0A0HYM2, 2.65), L4B, F10 inputs | Partial. A Sunday-6-Sep run would have carried DTB3 to 4 Sep; it is at 27 Aug. Either the 30 Aug/6 Sep runs failed part-way or the series list is split across jobs. [inf] | `tail -50 logs/fred_weekly.log`; then move the daily-frequency FRED series to a daily pull, keep weekly for monthly/quarterly. |

### F3 fallback — resolved cause
F3's 2 Sep value (raw 0.0075, source 2018-02-23) is the VIX-futures term slope from `VX_FRONT`/`VX_SECOND`, whose last row is 2018-02-24. So F3 has a primary (VIX3M/VIX) → fallback (VX futures) chain, and when the primary aged out the builder took the fallback without checking *its* age. [inf, strong — the dates match exactly] Two fixes, not one: (1) the CFE settles scraper points at a URL Cboe retired around 2018 — re-point to the current CFE historical endpoint so the futures leg is live again; (2) every source in a fallback chain is subject to the same staleness limit before it is used.

### Dead or discontinued series in `data_vintages` (history only — must never feed a live builder)
- `VX_FRONT`/`VX_SECOND` → 2018-02-24 (scraper URL, fixable)
- `CBOE_PC_*` → 2019-10-05 (Cboe stopped the free daily P/C files; F1 needs the new site or UW)
- `CBOE_VXO`/`VXOCLS` → 2021-09-24 (VXO discontinued — brief's backfill guard covers it)
- `TEDRATE` → 2022-01-22 (discontinued with LIBOR — **not** in the brief's backfill guard; if any builder reads it live, replace with F10's CP − T-bill or SOFR − T-bill at a registry version bump)

### Also observed
- `RIFSPPFAAD90NB` (90-day AA financial CP, daily) is already in the DB to 25 Aug — F10's numerator series is present; F10 itself has no builder. `DTB3` is the denominator.
- Cron ordering: `daily_driver.py` 06:00 VN runs *before* `uw_archiver.py` 06:30 VN. Harmless today (no UW-fed signal is live); wrong once Phase 5 signals exist. Move the driver to 07:00 or the archiver to 05:30.
- Ken French files last pulled 1 Jul; Ritter 31 Mar. Not refreshed by the `--only fred` cron either. Monthly job.
- `SI:<ticker>` short-interest vintages are current (26 Aug = 15 Aug settlement + lag) — the every-3-days `si_fetch_v2.py` job is working.

### Next commands (paste output)
```
crontab -l | grep -nE 'pipeline|accuracy' 
ls -lt logs/ | head -25
tail -40 logs/fred_weekly.log
tail -40 logs/warning_daily.log
grep -c "MASSIVE_API_KEY" .env
grep -n "only" scripts/fetch_free_history.py | head -20
grep -rn "VX_FRONT\|CBOE_VIX3M" warning/ features/ signals/ 2>/dev/null | head -20
```

---

## 6. Confirmations from logs (2026-09-07, second paste)

| Item | Status | Evidence |
|---|---|---|
| FRED weekly leg | **Failing outright**, not partial | `fred_weekly.log`: all 18 series `The read operation timed out` (DGS10, DTB3, BAA, AAA, BAA10YM, ABCOMP, DRTSCILM, SOFR, CSUSHPINSA, HOUST, BAMLH0A0HYM2, BAMLC0A0CM, VIXCLS, VXVCLS, VXOCLS, RIFSPPFAAD90NB, TEDRATE, IORB). [fact] The 2–3 Sep rows for DGS2/T10Y2Y/WTI/DTWEXBGS are not in this list → a second writer feeds `data_vintages` [inf]. Likely cause: the same ISP SNI filtering the crontab notes for FINRA, or a too-short socket timeout [unconfirmed]. |
| S10 | Not just "unbuilt" — needs a **manual FINRA margin-statistics xlsx ingest** (VPN) | `warning_daily.log`: "L4D … needs S10 margin data (FINRA xlsx not ingested)". [fact] |
| L4D / L4E | L4D ← S10. L4E ← F3 **and** F9 (Phase 5, needs months of UW archive) | log lines. [fact] L4E is off the table for now; L4 can reach 4/5 = 80% only via L4A + L4B + L4C (Cboe fix) + L4D (S10 ingest). |
| S12, S13, S15 | "builder not implemented" | log. [fact] |
| F3 mechanism | Confirmed: spot VIX/VIX3M primary, CFE futures fallback; futures end 2018-02-23 **by the code's own comment** (`f3_vix_term_slope.py:59`). 4 Sep log prints VIX 14.51 / VIX3M 17.56 (27 Aug spot) but slope 0.75% = the 2018 futures value. | [fact] |
| S14 | Same exposure: `s14_vol_structure.py` reads `VX_FRONT/VX_SECOND`; line 147 comment says a MIN-across-series fix was added after "fresh SPY_CLOSE masked a stale VX_FRONT". Whether S14's G state for 28 Aug–4 Sep used 2018 futures rows is **[unconfirmed]** — check what it does when the futures series is 8 years old. | |
| F2 | = VIX, with percentile (6.0 on 4 Sep — low-vol complacency read) | log. [fact] |
| Pipeline A | Runs outside crontab — `launchd_c_out.log`, `pipeline_C_20260907/` exist; the main fund already has `feed_freshness`, `repair_stale_feeds`, `crontab_drift`, `heartbeat` jobs. None of them cover `warning.db`. | [fact: file list] |
| `MASSIVE_API_KEY` | Present in `.env` (1 line). Crontab placeholder is overridden for jobs that source `.env`; launchd jobs — check. | [fact / unconfirmed] |

### Revised critical path to a non-NULL composite
1. Network: FRED fetch must succeed (VPN or timeout/retry). Until it does, S2/L4B/F10 inputs age out weekly.
2. Cboe leg daily (`--only cboe`) → F2, F3, L4C.
3. Massive→warning.db bridge back on a schedule (find the writer of `SPY_CLOSE`/`BREADTH_*`) → S4–S8, S14, L4A.
4. S10: FINRA margin xlsx ingest (manual, VPN) → L1 and L4D both move.
5. S13, S15 builders → L1 to 4/4. (Whether 3/4 = 75% clears the gate depends on the exact threshold — read it from `warning_engine.py` before deciding S13/S15 order.)
6. Then L4 = 4/5 = 80% with L4D; L4E stays NA until F9 exists — document it as `unbuilt`, not `NA`.

### Next commands (paste output)
```
sed -n 185,200p scripts/fetch_free_history.py                      # the leg names --only accepts
grep -rln "BREADTH_AD_CUM\|SPY_CLOSE" --include=*.py . | head       # who writes the Massive-derived rows
launchctl list | grep -i -E 'quant|pipeline|massive'                 # launchd jobs
grep -n -i "gate\|coverage" warning/warning_engine.py | head -20       # exact coverage threshold
sed -n 50,120p warning/builders/f3_vix_term_slope.py                # source selection logic
grep -n "VX_FRONT\|source_asof\|stale" warning/builders/s14_vol_structure.py | head -30
curl -m 15 -s -o /dev/null -w "%{http_code} %{time_total}s\n" https://api.stlouisfed.org/   # with VPN off, then on
```

---

## 7. Code-level confirmations (2026-09-07, third paste) — supersedes §5's Cboe row

| Item | Now known | Consequence |
|---|---|---|
| Gate | `STALE_COVERAGE_MIN = 0.70`; coverage = usable **weight** ÷ total weight (`warning_engine.py:44,135-150`) [fact] | L1 clears with any 2 of {S10, S13, S15} (3/4 = 75%). L2 needs 7/9 — one of S5/S9 un-stale. L3 needs both S4 and S14. L4 needs 4/5 = L4A+L4B+L4C+L4D. Count ≈ weight only if registry weights are equal [unconfirmed]. |
| F2 / F3 live inputs are **FRED** (`VIXCLS`, `VXVCLS`), not the Cboe files | `f3_vix_term_slope.py:56-57` [fact] | The Cboe leg is *not* the F2/F3 fix — the failing FRED job is. Cboe leg still needed for series FRED doesn't carry (COR1M/3M, SKEW, VVIX, VIX9D) → L4C. Correction to §5. |
| F3 fallback has no freshness check | `csv_fresh = … and _fresh(...)` but `elif fut_slopes:` takes the futures leg unconditionally; `stale` is then computed and flagged, but state is still emitted from 2018 rows [fact] | Patch below. |
| S14 handles the dead futures leg correctly | `s14_vol_structure.py:146-154` drops leg (b) when `VX_FRONT` is stale and continues on leg (a) [fact] | No 2018 contamination in S14. Its stale flag since 2 Sep comes from leg (a) inputs (SPY_CLOSE 29 Aug / VIXCLS 28 Aug) [inf]. |
| FRED host reachable | `curl https://api.stlouisfed.org/` → 301 in 1.09 s [fact] | Not SNI blocking of the host. The observations call itself times out — full ALFRED vintage pulls are large; socket timeout in the script is the suspect [inf]. |
| Pipeline A under launchd | `com.atom.pipeline-c`, `com.atom.pipeline-chain` [fact] | The warning.db writer for `SPY_CLOSE`/`BREADTH_*` is still unidentified (grep failed on an unquoted zsh glob — rerun below). |
| `--only` legs | `fred, alfred, cboe, cfe, french, ritter` [fact] | Weekly = fred only by design ("rude to the source") — fine, once fred works; add a light **daily** job for the ~6 daily FRED series + Cboe COR/SKEW/VVIX. |

### F3 patch (one hunk + one test)
```python
# warning/builders/f3_vix_term_slope.py — compute()
csv_fresh = bool(csv_slopes) and _fresh(con, asof, VIX3M_SERIES, VIX_SERIES)
fut_fresh = bool(fut_slopes) and _fresh(con, asof, FUT_SECOND, FUT_FRONT)
if csv_fresh:
    slopes, leg = csv_slopes, "vix3m"
elif fut_fresh:
    slopes, leg = fut_slopes, "futures"
else:
    return _na(asof, f"no fresh leg within {MAX_STALENESS_DAYS} bdays: "
                     f"{VIX3M_SERIES}/{VIX_SERIES} ({len(csv_slopes)} obs), "
                     f"{FUT_SECOND}/{FUT_FRONT} ({len(fut_slopes)} obs)")
```
```python
# warning/test_builders.py
def test_f3_never_uses_stale_futures_leg(tmp_db):
    # spot leg ends 10 bdays before asof; futures leg ends 2018 -> NA, not "futures"
    ...
    assert r["state"] == "NA" and r["source_asof"] is None
```
Same rule applies to every builder with a fallback chain (brief §7: "never substitute a different dataset silently").

### FRED job — what to check, in order
1. `grep -n "timeout" scripts/fetch_free_history.py` — the per-request timeout; ALFRED vintage pulls for 18 series can run minutes.
2. Retry with backoff (3 tries, 10/30/90 s) and **per-series** commit so one timeout doesn't lose the batch.
3. Split: daily job for the daily series (DTB3, DGS10, VIXCLS, VXVCLS, BAMLH0A0HYM2, BAMLC0A0CM, SOFR, RIFSPPFAAD90NB, IORB) with `--only fred` and a series filter; weekly for monthly/quarterly.
4. Only if 1–3 fail: VPN on for the job, as with FINRA.

### Next commands (paste output)
```
grep -rln --include='*.py' "BREADTH_AD_CUM\|SPY_CLOSE" . | grep -v __pycache__
grep -n "timeout\|retry\|sleep" scripts/fetch_free_history.py | head
sed -n 1,60p scripts/fetch_free_history.py | grep -n "def \|import"
launchctl print gui/$(id -u)/com.atom.pipeline-chain | grep -iE 'program|argument|StartCalendar|Hour|Minute' | head
```

---

## 8. Massive bridge and FRED timeout — resolved (2026-09-07, fourth paste)

| Item | Fact | Fix |
|---|---|---|
| warning.db price/breadth writers | `warning/ingest_spx.py`, `warning/ingest_breadth.py`. Neither is in crontab; Pipeline A/D/B runs via launchd `scripts/pipeline_chain_ADB.sh` at 04:00 VN. Last rows 29 Aug = last manual run. [fact] | Schedule both Tue–Sat at 05:15 VN (after the 04:00 chain, before the 06:00 driver), or append them to the end of `pipeline_chain_ADB.sh`. Exact CLI flags: run `--help` on each first. |
| FRED job | `urllib.request.urlopen(req, timeout=60)`, no retry, no per-series commit. All 18 series time out — including small ones (IORB, 1.8k rows) — so it is connection stalling, not payload size. [fact / inf] | (1) Test the real endpoint with the key from `.env` (command below), VPN off then on. (2) Retry ×3 with backoff, commit per series. (3) Daily light job for the daily series, weekly for the rest. (4) VPN for the job only if (1) shows the API stalls without it. |
| Builder names | s2_credit, s5_breadth, s6_concentration, s7_defensive_rotation, s8_epicenter_fracture, s14_vol_structure, f3_vix_term_slope; L4A funding seizure, L4B spread blowout velocity, L4C correlation spike, L4D forced deleveraging, L4E hedging feedback. [fact] | Dashboards updated. |

### Proposed crontab additions (VN local; adjust flags after `--help`)
```
# Crash early-warning: Massive-derived series into warning.db — after pipeline_chain_ADB (04:00), before daily_driver (06:00)
15 5 * * 2-6 cd /Users/atomnguyen/Desktop/ML_Quant_Fund && set -a && . ./.env && set +a && /opt/homebrew/bin/timeout 900 /Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python warning/ingest_spx.py --db warning.db >> logs/warning_ingest.log 2>&1
20 5 * * 2-6 cd /Users/atomnguyen/Desktop/ML_Quant_Fund && set -a && . ./.env && set +a && /opt/homebrew/bin/timeout 900 /Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python warning/ingest_breadth.py --db warning.db >> logs/warning_ingest.log 2>&1
# Crash early-warning: daily FRED + Cboe (COR/SKEW/VVIX) — needs a --series filter or a light wrapper; until then run the fred leg daily and accept the cost
30 5 * * 2-6 cd /Users/atomnguyen/Desktop/ML_Quant_Fund && set -a && . ./.env && set +a && /opt/homebrew/bin/timeout 1500 /Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python scripts/fetch_free_history.py --db warning.db --out data/raw --only fred,cboe >> logs/fred_daily.log 2>&1
```
Move `uw_archiver.py` from 06:30 to 05:45 so the driver at 06:00 sees the day's snapshot.

### Next commands (paste output; the curl prints size and time, never the key)
```
python warning/ingest_spx.py --help | head -30
python warning/ingest_breadth.py --help | head -30
set -a; . ./.env; set +a; curl -m 90 -s -o /dev/null -w "%{http_code} %{size_download}B %{time_total}s\n" "https://api.stlouisfed.org/fred/series/observations?series_id=IORB&api_key=$FRED_API_KEY&file_type=json"
tail -5 logs/warning_daily.log; ls -la logs/ | grep -i warning
```

---

## 9. FRED reachable; ingest CLIs known (2026-09-07, fifth paste)

| Item | Fact | Reading |
|---|---|---|
| FRED observations endpoint | 200, 177 KB, 1.83 s (VPN on) / 0.56 s (VPN off) [fact] | Not blocked, not slow, VPN not required. The two Sunday failures (30 Aug, 6 Sep — all 18 series, identical message) were environmental at 16:00 VN Sunday or request-shape related (full-vintage pulls), not connectivity now. [inf] |
| `ingest_spx.py` | Reads `prices.db` (Pipeline A output) by default; `--from-massive` only for tickers prices.db lacks (RSP); `--table raw_bars` preferred [fact] | It is a prices.db → warning.db copy, so it must run after `pipeline_chain_ADB.sh` finishes, not at a fixed clock time. |
| `ingest_breadth.py` | `--prices`, `--db`, `--dry-run` [fact] | Same dependency. |

### Do now (manual, in this order; paste the tail of each)
```
python warning/ingest_spx.py --db warning.db                       # defaults: check they cover SPY + XL* (grep -n "default" warning/ingest_spx.py)
python warning/ingest_spx.py --db warning.db --ticker RSP --from-massive
python warning/ingest_breadth.py --db warning.db
time python scripts/fetch_free_history.py --db warning.db --out data/raw --only fred 2>&1 | tail -25
sqlite3 warning.db "select series_id, max(pub_date) from data_vintages where series_id in ('SPY_CLOSE','RSP_CLOSE','BREADTH_AD_CUM','DTB3','VIXCLS','BAMLH0A0HYM2') group by 1;"
python warning/daily_driver.py --db warning.db 2>&1 | tail -40     # re-step today; watch coverage move
```
If the interactive FRED run succeeds, the Sunday failures were environmental → add retry ×3 with backoff and per-series commit anyway, and keep VPN **off** for that job (faster).
If it fails interactively with the same message, paste the full log — the request shape is the problem.

### Scheduling — one sequenced wrapper, not four clock times
`prices.db` is written by `pipeline_chain_ADB.sh` (launchd, 04:00 VN) whose duration is not fixed. Fixed cron minutes race it. Replace the 06:00 driver line with one wrapper run at 05:45 that checks the chain finished, then runs in order:
```bash
#!/bin/zsh
# scripts/warning_chain.sh — run after pipeline_chain_ADB; exits non-zero on the first failure
set -euo pipefail
cd /Users/atomnguyen/Desktop/ML_Quant_Fund
set -a; . ./.env; set +a
PY=/Users/atomnguyen/.pyenv/versions/ml_quant_310/bin/python
LOG=logs/warning_chain.log
# 1. wait (max 60 min) for today's Pipeline A marker — replace with the chain's real completion marker/log line
for i in {1..60}; do grep -q "$(date +%F)" logs/launchd_c_out.log 2>/dev/null && break; sleep 60; done
$PY warning/ingest_spx.py --db warning.db                               >> $LOG 2>&1
$PY warning/ingest_spx.py --db warning.db --ticker RSP --from-massive   >> $LOG 2>&1
$PY warning/ingest_breadth.py --db warning.db                           >> $LOG 2>&1
$PY scripts/uw_archiver.py --db warning.db                              >> $LOG 2>&1
$PY warning/daily_driver.py --db warning.db                             >> logs/warning_daily.log 2>&1
```
```
45 5 * * 2-6 /opt/homebrew/bin/timeout 3600 /Users/atomnguyen/Desktop/ML_Quant_Fund/scripts/warning_chain.sh >> /Users/atomnguyen/Desktop/ML_Quant_Fund/logs/warning_chain_cron.log 2>&1
```
Then delete the separate 06:00 driver and 06:30 archiver lines. The completion check on line "1." is a placeholder — use whatever `pipeline_chain_ADB.sh` writes when it finishes (a done-file is cleanest: `touch logs/pipeline_ADB.$(date +%F).done` as its last line).

FRED: keep the Sunday weekly for full vintages; add `--series` (comma list) to `fetch_free_history.py` and run a daily light pull of the ~9 daily series inside the wrapper, before the driver. Until `--series` exists, the weekly job is the only FRED source — fix its retry first.

---

## 10. After the manual backfill (2026-09-07 re-step)

| Layer | Coverage | Status |
|---|---|---|
| L1 | 25% | NA — S10, S13, S15 unbuilt |
| L2 | 78% | **passes**, score 0.000 (S1 R, S6 R — neither persisted yet) |
| L3 | 100% | **passes**, score 0.000 |
| L4 | 40% | NA — L4C stale (Cboe COR, last 27 Aug), L4D ← S10, L4E ← F9 |

Remaining stale/NA on 7 Sep: `S9` (stale, source 14 Aug), `L4C` (stale, 27 Aug), `S10/S12/S13/S15/L4D/L4E` (NA). Everything else fresh at 3–4 Sep. [fact]

- **S9 = short interest** (`s9_short_interest.py`, FINRA semi-monthly). Source 14 Aug is *correct* on 7 Sep: the 31 Aug settlement publishes ~11 Sep. Its staleness limit is shorter than the series' natural maximum age (~24 days) — A4 confirmed; set limit ≥ 26 days at the next registry version. [fact / inf]
- **L4C** needs the Cboe leg (`--only cboe`) — COR1M/3M have no FRED copy. Daily inside the wrapper. Lifts L4 to 60%; still short of 70% without L4D.
- **S10 has no ingest path at all.** `fetch_free_history.py` only prints the FINRA margin-statistics URL; no parser writes a margin series into `data_vintages`; `l4_propagation.py:68` hard-codes the L4D NA reason. Work = (1) download the xlsx (VPN), (2) `warning/ingest_finra_margin.py` → `data_vintages` (monthly, pub = release date, +3 w lag per brief rule 1), (3) `builders/s10_*.py`, (4) unblock L4D. This is the single item that moves both L1 and L4.
- The 7 Sep row is labelled on a US market holiday (Labor Day) with 4 Sep data — B8. The 8 Sep 06:00 VN scheduled run will write the same label again.

### Next
- Attach the new `cews_export.json` (8 runs) → dashboards republished.
- Decide S13 vs S15 order (either one + S10 clears L1 at 75%); paste `grep -n "S13\|S15" signal_registry.csv` so the registry rows are in the ledger.

---

## 11. Registry rows and the version literal (2026-09-07)

Registry lives at `warning/signal_registry.csv` (not repo root). Header: `id,name,family,layer,role,formula,data_source,series_or_endpoint,history_start,frequency,publication_lag,threshold_arm,threshold_red,persistence_days,direction,tier,…,max_staleness_days,notes`. No version column. [fact]

| Signal | Registry | What it means for the build |
|---|---|---|
| S9 short interest | semi-monthly, lag ~8 bd, `max_staleness_days = 20`, `source_asof` written as the **settlement** date (14 Aug) | Schema says `source_asof` = max **pub_date** of vintages used. With pub_date (26 Aug) the age on 7 Sep is 12 d and 20 is fine; with settlement date it is 24 d and trips. **Builder bug (s9_short_interest.py), not a registry change** — fix the date the builder stamps; no version bump needed. [fact / inf] |
| S10 margin debt | FINRA margin-statistics xlsx, monthly, lag ~3 w, arm YoY>+30%, red YoY>+40% then first 3-m decline from peak, staleness 45, role confirmer, "L1+trigger" | No ingest exists. Needs xlsx download (VPN) → parser → builder. Also gates L4D. |
| S13 valuation gate | Shiller `ie_data.xls`, monthly, lag ~1 m, CAPE percentile vs own history-to-date (amber 85–95th, red >95th), ERP check 1/CAPE − 10y real, staleness 45 | **Simplest of the three**: one file, one series, a running percentile. Build first. "Scales exposure ceilings only." |
| S15 credit-boom R-zone | FRED Z.1 + Case-Shiller, quarterly, lag ~10 w, two legs (business, household), staleness 120 | Hardest; build last. |

**L1 order:** S13 → S10 → (L1 = 75%, passes) → S15 later. S10 also lifts L4 to 60% (with L4C's Cboe leg) and, with L4D live, to 80%.

**registry_version:** `daily_driver.py:134,144` write the literal `"unversioned"`. Fix: read `warning/signal_registry.csv` once per run, stamp `sha256[:12]` (or a `version` row in a sidecar `signal_registry.version`) into every row; refuse to run if the file is missing. Ten lines; closes B4's "unenforceable" half. The `signal_registry` table in the DB (B4/B9) remains the full fix.

**Newer docs exist** — `warning/HANDOFF_CRASH_WARNING_2026-08-28.md`, `…08-30.md`, `warning/DECISIONS.md` — post-dating the 25 Aug brief this ledger was written against. Reconcile before acting on §2–§3: items may already be decided there.
