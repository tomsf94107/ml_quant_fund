# HANDOFF — Crash Early-Warning System

**Session: 2026-08-28 (ET) / 2026-08-29 (VN).** Repo `ml_quant_fund`, branch
`research-track`, HEAD `691cedde`, pushed. Suite: **87 tests green**.
Read `warning/DECISIONS.md` alongside this — it holds the fourteen recorded
decisions and is the authority on every spec gap.

---

## 0. What exists now

Phase 0 is complete except the second UW snapshot (cron fires 06:30 ICT Tue–Sat;
first snapshot 2026-08-28, 13 endpoints, 1.23 MB).

| Component | State |
|---|---|
| `warning.db` | schema applied: 10 tables + 5 views |
| Composite inputs | **3 / 15** — S1 curve, S2 credit, S4 funding |
| Dashboard features | **F2** VIX percentile, **F3** VIX term slope |
| L4 propagation | **2 / 5** — L4B spread blowout, L4C correlation spike |
| Daily driver | `warning/daily_driver.py`, persists to `composite_scores`/`signal_values`/`alerts` |
| Band today | **INSUFFICIENT_DATA** — correct at 3/15 coverage |

**Live read, 2026-08-28:** S1=R (2025-01 escalation, still inside the 24-month
window), S2=G, S4=G (composite z −0.47 across three funding legs), F2=Y (VIX
14.51, **6.0th percentile**), F3=G (+21.0% contango). Complacent vol, quiet
funding, and the one red is a curve signal 19 months stale.

---

## 1. Data loaded

| Series | Coverage | Note |
|---|---|---|
| DGS10 / DTB3 | 1962-01 / 1954-01 → | curve |
| BAA / AAA / BAA10YM | 1919 / 1953 → | credit, monthly |
| BAMLH0A0HYM2 / BAMLC0A0CM | **2023-08-28** → | FRED rolling 3y window; the archive IS the history |
| TEDRATE | 1986-01 → **2022-01** | discontinued with LIBOR |
| RIFSPPFAAD90NB / IORB / SOFR | 1997 / 2021-07 / 2018-04 → | funding legs (OPEN ITEM 1 resolved) |
| VIXCLS / VXVCLS / VXOCLS | 1990-01 / 2007-12 / 1986-01→2021-09 | via FRED |
| CBOE_* (15 series) | SKEW 1990+, VVIX/COR 2006+, P/C 1995–**2019-10** | archive files |
| VX_FRONT / VX_SECOND | **2004-03-26 → 2018-02-23** | CFE, D13-normalized |
| SPY_CLOSE | **2016-07-18** → | proxy for SPX |
| ABCOMP / DRTSCILM / HOUST / CSUSHPINSA | revisable — **pull-date stamped** | historical reads return NA until ALFRED |

---

## 2. Findings that changed the build

These came out of running builders against real data. Each is a real property,
not a bug that was patched away.

**Vintage stamping was wrong.** `pub_date` was the pull date, so every historical
point-in-time read returned NA. Corrected for the 8 never-revised series (41,435
rows restamped in place). The 4 revisable series still stamp the pull date and
correctly return NA in history — that is the honest answer, not a gap.

**Three signals were benign at the 2007-10-09 peak.** Independently reproduced:
F2 at the 85.9th percentile, F3 in +7.20% contango, S2 green, S4 amber-not-red.
Only S1 was red, and it had fired four months earlier. This is report line 41
("at the October 2007 peak every real-time options indicator was benign")
reproduced from point-in-time data by separate builders.

**F3's registry verdict confirmed exactly** — "Aug-07 inversion; contango at
top": −9.80% inverted 4d on 2007-08-15, +7.20% contango on 2007-10-09. Only
testable because the CFE futures leg reaches back before VIX3M exists.

**The two F3 legs disagree materially and must never be averaged (D14).** On
2020-03-20 the vix3m leg read −3.51% (inverted) while futures read +0.75%
(contango) — opposite directions. An average would match neither tenor and would
flip the state.

**Persistence beats magnitude, as designed.** 2018 volmageddon had the deepest
inversion in the sample (−24.62%) and reached only amber at 2 days; COVID had a
shallow −3.51% and hit red at 19 days.

**Registry verdicts sometimes cite data the stack no longer has.** S1's 2000 cell
(D4) and S2's 2007 cell (D8) both fail for this reason — one is sub-threshold on
the registry's own numbers, the other was established on HY OAS that FRED has
since truncated. Assume more of this in the remaining rows.

---

## 3. Open rulings — the two that matter before live

**D10 — INSUFFICIENT_DATA suppresses the L4 crisis override.** `step()` tests
`insufficient` before `hysteresis_step`, so a fired L4 override is computed as
True and then discarded: the band freezes and the action is `freeze`. With 3 of
15 builders live, **a 150bp HY blowout today would print "freeze" rather than
"CRISIS"**. Report line 601 says L4 "overrides composite"; the honesty rule says
do-nothing binds. Both cannot hold. Engine unchanged, behaviour pinned by test.

**D12 — z-scored signals self-neutralize in a sustained crisis.** S4 reads G in
Nov-2008 with TED at **221bp** (z +0.83) because its own 252-day window absorbed
the crisis; F2 read G at the Oct-2007 peak for the same reason. Both are faithful
to the frozen formulas — a trailing-window normalization measures change, not
level. **Direct consequence: L4A ("S4 red + breadth") cannot fire through a
crisis either.** A level floor would fix it but would be a new threshold (rule
#3), so nothing was applied.

Also open: **D4** (S1's 2000 registry cell contradicts its own formula) and
**D8** (S2's pre-2023 validation is UNDECIDABLE).

---

## 4. Next steps, in order

Everything unblocked has been built. Each remaining item needs a file only the
operator can fetch.

1. **`FRED_API_KEY`** — free signup. Turns on the ALFRED leg for the four
   revisable series, which currently return NA in every historical read.
   Unblocks **S3** (SLOOS) and makes S15's inputs honest. Cheapest unlock.
2. **Shiller `ie_data`** from shillerdata.com (manual; the link target moves).
   Unblocks **S13** (CAPE, L1 at 0% coverage) *and* carries monthly S&P back to
   1871, which would give **S2's equity leg** its pre-2016 history and let the
   2000/2007 replays run with both legs.
3. **FINRA margin xlsx** (manual). Unblocks **S10** → and with it **L4D**.
4. **Ritter PDFs** — fetchable: `--only ritter`, VPN on. Unblocks **S11**.
5. **S5–S8** breadth/concentration/rotation/epicenter — Massive universe,
   survivorship-safe forward only, so they accumulate rather than backfill.

---

## 5. Operational notes

- **cboe.com is ISP-filtered on this connection.** Not DNS: `--resolve` to
  Cloudflare's real IP with correct SNI still gets RST, with the VPN
  disconnected. **Route is the VPN.** The archives are static, so the fetch is
  one-time — but any re-pull needs the VPN up.
- **Browsers append `-1` on re-download.** A stale file that still imports and
  runs is the hardest failure to spot; one delivery this session shipped an
  unmoved `run_signal.py` and the commit looked clean. Verify with
  `grep -c <new-token> <file>` after moving.
- **zsh has `interactive_comments` unset** — `#` lines in a pasted block execute
  as commands. Send comment-free blocks.
- **Cron:** UW archiver 06:30 ICT Tue–Sat; FRED weekly append Sun 16:00 VN.
  Both installed append-only from a fresh `crontab -l`; never install from a
  stored file.
- **`data/raw/` is committed** (3.4 MB Cboe + CFE) since re-fetching is VPN-gated.

---

## 6. How to verify state from cold

```
cd ~/Desktop/ML_Quant_Fund
cd warning && python -m pytest -q ; cd ..          # expect 87 passed
python warning/daily_driver.py --db warning.db --dry-run
python warning/run_signal.py --signal S1 --db warning.db --validate
python warning/run_signal.py --signal F3 --db warning.db --validate
sqlite3 warning.db "SELECT series_id, COUNT(*), MIN(obs_date), MAX(obs_date) FROM data_vintages GROUP BY series_id ORDER BY series_id;"
```

Every builder has a `--validate` mode that replays the registry's own ex-ante
verdicts. **A mismatch is a finding: report it, do not tune the builder.**
Thresholds are frozen (rule #3); every deviation this session was recorded in
`DECISIONS.md` rather than absorbed into code.
