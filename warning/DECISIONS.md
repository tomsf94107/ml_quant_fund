# DECISIONS — crash early-warning system

Spec gaps found during implementation where the report and `signal_registry.csv`
are silent or inconsistent. Each has a **default in force** so the build can
proceed, each is a single named constant, and each is reversible in one line.

These are NOT threshold changes (rule #3). No frozen threshold has been altered.
They are choices the frozen documents do not make.

Status: **all four awaiting Atom's ratification** as of 2026-08-28.

---

## D1 — amber maps to `Y` (0.33), not `O` (0.66)

**Where:** `warning/builders/s1_term_spread.py::AMBER_STATE` (will apply to every builder)

**Gap.** The report and registry use three states — green / amber / red
(S3: ">+10 = amber, >+20 = red"; S13: ">95th = red, 85-95th = amber"). The
engine has five (G/Y/O/R/B, scores 0 / 0.33 / 0.66 / 1.0 / 1.0). Neither
document says which engine state "amber" is.

**Default and why.** `Y` (0.33). Report line 469: *"Layer-1 amber can persist for
years — exposure gates only."* At 0.66 a multi-year amber would hold the
composite elevated indefinitely and erode the do-nothing discipline the system
exists to enforce.

**If ratified otherwise:** change one constant per builder. No test depends on
the value, only on the mapping being consistent.

---

## D2 — `ESCALATE_WINDOW_MONTHS = 24`

**Where:** `warning/builders/s1_term_spread.py::ESCALATE_WINDOW_MONTHS`

**Gap.** S1's registry formula says *"ESCALATE if re-steepens +50bp from trough
after >=6m inversion"* with no limit on how old the inversion may be.

**Found on real data.** Without a limit, S1 fired **R in Feb-2000** off the
**1980-11..1981-08** inversion (trough −2.65, "rise 363bp"). A 19-year-old trough
generated a live escalation.

**Default and why.** 24 months. **Not a fitted parameter:** 2007-10 requires
>= 6 months (inversion ended 2007-04) and excluding the 1981 artifact at 2000-02
requires < 222 months. Every historical anchor is identical across `[6, 222)`,
so 24 sits in the middle of a flat region. Escalation also clears when a new
inversion run begins.

**Regression test:** `test_stale_inversion_cannot_escalate_decades_later`.

---

## D3 — months must be FULLY PUBLISHED before they count

**Where:** `warning/pit.py::monthly_mean_complete`, used by S1 via `PUB_LAG_DAYS`

**Gap.** The registry says `frequency: daily->monthly` but does not say when a
month becomes usable.

**Found on real data.** The 1999–2001 scan produced `Y -> R -> G` across two
reads:

```
2001-01-31   Y -> R   spread -0.001   run 6m   rise 52.8bp
2001-02-28   R -> G   spread +0.219   run 5m
```

At the 2001-01-31 read January's final observation was not yet published, so the
month averaged −0.001 — fractionally inverted. It counted toward the run, hit the
6-month floor, and fired R. Once fully published January turned positive, the run
shrank to 5 months, and the escalation vanished. **A signal must not enter and
leave a state because a month was half-counted.**

**Default and why.** A month counts only when `asof >= month_end + publication_lag`.
Costs one extra month of lag — immaterial for a signal whose documented lead is
14–16 months. The registry's verdicts name **data months**, not read dates, so
"Jun07 re-steepen" is still satisfied: the June-2007 data month escalates, observed
at the July read.

**Regression test:** `test_incomplete_month_is_excluded_until_published`.

---

## D4 — registry `historical_verdict_2000` for S1 is inconsistent with S1's own formula

**Status: NOT patched. Flagged only. No code was tuned to fit it.**

**The conflict.** The registry cell reads *"fired Feb+Jul 2000"*. But S1's own
formula is `m_avg(DGS10-DTB3)`, and the live data shows that spread was
**+0.981 in Feb-2000** and **+0.104 in Jul-2000** — never inverted. S1 as
specified cannot fire on those dates.

**What the report body says instead.** Line 41 describes the 2000 sequence as
*"curve inversion from July 2000 (post-peak)"*. The `--scan` shows S1 arming
**2000-09** — a July daily inversion plus the 2-of-3 monthly confirmation lag.
SPX peaked 2000-03-24, so S1 was **post-peak in 2000**, which is exactly what the
report body claims and exactly what the builder produces.

**Reading.** Feb-2000 was the 10y−2y / 10y−30y inversion, a different spread from
S1's 10y−3m. The registry cell appears to describe a different measure, or is
loose. The report body and the builder agree with each other.

**Ruling needed.** Either
(a) the registry cell is corrected to match the formula and the report body, or
(b) S1's `series_or_endpoint` is wrong and should be a different spread — in
which case the whole 2000 leg of the validation is re-run.

Until ruled, S1 stands as built and the 2000 anchor in `run_s1.py --validate` is
expected to read G, not amber.
