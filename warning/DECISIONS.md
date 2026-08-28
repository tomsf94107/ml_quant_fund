# DECISIONS — crash early-warning system

Spec gaps found during implementation where the report and `signal_registry.csv`
are silent or inconsistent. Each has a **default in force** so the build can
proceed, each is a single named constant, and each is reversible in one line.

These are NOT threshold changes (rule #3). No frozen threshold has been altered.
They are choices the frozen documents do not make.

Status as of 2026-08-28: **D1 RATIFIED** (evidence-backed by the 64-year scan).
D2, D3 in force. D4 open. D5, D6, D7 added below.

---

## D1 — amber maps to `Y` (0.33), not `O` (0.66)   [RATIFIED 2026-08-28]

**Where:** `warning/builders/s1_term_spread.py::AMBER_STATE` (will apply to every builder)

**Gap.** The report and registry use three states — green / amber / red
(S3: ">+10 = amber, >+20 = red"; S13: ">95th = red, 85-95th = amber"). The
engine has five (G/Y/O/R/B, scores 0 / 0.33 / 0.66 / 1.0 / 1.0). Neither
document says which engine state "amber" is.

**Default and why.** `Y` (0.33). Report line 469: *"Layer-1 amber can persist for
years — exposure gates only."* At 0.66 a multi-year amber would hold the
composite elevated indefinitely and erode the do-nothing discipline the system
exists to enforce.

**Evidence that settled it.** The full 1962-2026 scan shows S1 amber in 1966-67,
1969, 1970, 1979, 2000-01, 2019 and 2023-25 -- long stretches, most without a bear
following. That is the report's *"L1/L2 amber can persist for years"* observed
directly on 64 years of data. At 0.66 those stretches would have held the composite
elevated for most of the sample. RATIFIED.

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


---

## D5 — R does not decay; it clears only on the window or a new inversion

**Where:** consequence of D2. No new code.

**The property.** Once S1 escalates, R holds until the inversion is
`ESCALATE_WINDOW_MONTHS` old or a new inversion run starts. Observed:

| Fired | Cleared | Held |
|---|---|---|
| 2007-07 | 2009-06 | 23m -- **correct**, spanned the entire bear |
| 1981-10 | 1983-10 | 24m -- ~14m past the Aug-1982 trough, into a bull market |
| 2025-01 | ~2027-01 projected | **19m so far, unresolved** |

**Decision: add NO decay rule. Deferred to Phase 4.**

**Why not fix it now.** Any decay rule needs to know whether a bear actually
arrived, and `warning.db` has no SPX price series yet. The alternatives are all
invention: a second time constant, or a "spread decisively steep" threshold that
appears in neither the report nor the registry. D2 is already one addition to a
frozen spec; compounding it with a second, unmeasurable one is how a spec becomes
folklore.

**Falsifiable question for Phase 4**, to be answered with data rather than
argued: does persistent-R after the bear ends degrade the composite's PR-AUC or
the Part VIII policy's Sharpe/MaxDD versus a decayed variant? If yes, the rule
earns its place with a version bump. If no, the current behaviour stands.

**Mitigation already in place.** S1 is one of 13 L2 signals and L2 saturates at
`SATURATION_K = 3` weight-units, so a lone signal at R contributes about a third
of the layer, not the whole of it. The dilution is by design.

---

## D6 — S2 "half-condition" = the credit leg alone

**Where:** `warning/builders/s2_credit.py`

**Gap.** S2's registry row gives `threshold_arm: half-condition` and
`threshold_red: full condition` for a formula with two legs:
*spread > 200d MA AND spread >= 126d_low + 75bp* (credit) **WHILE** *SPX within
3% of its 52w high for 21d* (equity). Neither document says which half is "half".

**Default and why.** The credit leg alone arms; both legs together fire. The
signal's name is "credit trend & credit-equity **divergence**" -- the divergence
IS the conjunction, so it cannot be the half. And the equity leg alone (index near
its high) is an ordinary bull market, not a warning.

---

## D7 — S2 runs in two modes; the monthly mode converts trading days to months

**Where:** `warning/builders/s2_credit.py`

**Gap.** The registry gives S2 daily windows (200d MA, 126d low) but names
`BAA10YM` as "primary history" -- and `BAA10YM` is **monthly** (880 observations
from 1953-04). The daily series `BAMLH0A0HYM2` starts 2023-08 on FRED's rolling
3-year window, so a 200-day MA is only computable from roughly 2024-05.

**Default and why.** Two implementations, both declared in the output:
- **daily mode** (OAS, 2024-05 onward): 200-day MA, 126-day low, as written.
- **monthly mode** (`BAA10YM`, 1953 onward): 200 trading days -> **10 months**,
  126 trading days -> **6 months**. Straight calendar conversion at ~21 trading
  days per month, not a fitted choice.

The `+75bp` threshold is unchanged in both modes. Every reading reports which
mode produced it; they are never silently mixed.


---

## D8 — S2's pre-2023 historical validation is UNDECIDABLE. Do not tune.

**Status: finding, not a defect. No threshold changed.**

**What the replay showed.** S2 on `BAA10YM` returns G at both of its documented
fires and produces **no pre-peak fire anywhere in the sample**:

| Registry verdict | Replay | Why |
|---|---|---|
| 2000 "weak fire (+44bp Jan-Mar00)" | **G** | Report line 209 sources the move as `1.67 -> 2.11pp` = **+44bp**. S2's threshold is **+75bp**. 44 < 75: the frozen threshold cannot fire on this move. The verdict and the threshold contradict each other. |
| 2008 "strong fire (Jun-Oct07)" | **G** | Report line 344 sources this as **HY OAS** (trough 2.41% Jun-1-07), tagged `[SOURCED-2y]`. Part 0 row 7: pre-2023 ICE OAS is **"NO for history"** since FRED truncated to a rolling 3-year window in Apr 2026. On `BAA10YM` the same window moved 1.640 -> 2.070 = **+43bp**, also under threshold. Investment grade barely moved; 2007 was a high-yield event. |

**The substitution is a category error, and the report says so.** Part 0 row 6
names Baa-10y "the only free full-history credit spread" and row 7 rules OAS out
for history. Those two facts together mean S2's long-history leg measures IG
credit while its verdicts were established on HY credit. They are different
instruments with different amplitudes in the same episode.

**Decision.** S2's pre-2023 verdicts are marked **UNDECIDABLE**, the report's own
category for a question the data cannot answer. Specifically:
- Do **not** lower the +75bp threshold to make 2000 or 2007 fire. That is fitting
  a frozen threshold to a verdict derived from a different series (rule #3).
- Do **not** relabel Baa-10y results as if they validated the HY formula.
- S2's **daily/HY leg is forward-only**, exactly like S5's cross-sectional legs:
  its history begins with the OAS archive started 2026-08-28. This is why the
  weekly FRED append matters -- those rows are the only OAS history that will
  ever exist for these dates.

**What the replay DID establish** (all reproducible):
- 1998-11 fires amber at +102bp -- the registry's documented false positive,
  which is supposed to fire and be killed by S3/S15 pairing. Correct behaviour.
- 2011 (+73bp), 2015 (+52bp), 2018 (+22bp) do not fire at all -- better than the
  registry's "false fire" expectation for those years.
- 2021-12 G: "2022 correctly silent" reproduced.
- 2026-08-26 G in daily mode (spread 2.670, ma200 2.855, +4bp off the 126d low):
  credit is currently calm, and this reading IS on the correct HY instrument.

**Open option, not yet taken.** Report line 27 lists `HYG (2007+)` / `JNK` price
proxies as a mitigation. A price-based HY proxy could cover 2007-08 but not 2000,
and would be a proxy -- it must be labeled as one and validated separately before
any verdict rests on it. Requires checking whether HYG history exists in
`prices.db`.
