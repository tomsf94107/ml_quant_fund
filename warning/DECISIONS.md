# DECISIONS — crash early-warning system

Spec gaps found during implementation where the report and `signal_registry.csv`
are silent or inconsistent. Each has a **default in force** so the build can
proceed, each is a single named constant, and each is reversible in one line.

These are NOT threshold changes (rule #3). No frozen threshold has been altered.
They are choices the frozen documents do not make.

Status as of 2026-08-28: **D1 RATIFIED** (evidence-backed by the 64-year scan).
D2, D3 in force. D4, D8, D12 OPEN. D5-D7, D9-D11, D13, D14 in force.
D10 RESOLVED 2026-08-30.

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


---

## D9 — L4C's "top-decile" lookback is 504 trading days

**Where:** `warning/builders/l4_propagation.py::CORR_WINDOW`

**Gap.** Report line 601 specifies "correlation spike (avg pairwise 63d corr
top-decile jump)" but gives no lookback over which "top-decile" is measured.

**Default and why.** 504 trading days, matching F2's percentile window, so the
two percentile-based conditions in this system share one convention rather than
each inventing its own. Also: Cboe's 3-month implied correlation index is used as
the measure, since 3 months is approximately the stated 63 trading days, and the
report's Part VII F8 already names the Cboe COR files as the documented proxy for
average pairwise correlation. Both the window and the proxy are declared in every
reading.

**Also encoded:** "spike" requires a JUMP, not merely a high level. A plateau at
a high correlation does not fire. Pinned by
`test_l4c_needs_both_top_decile_and_a_jump`.

---

## D10 — INSUFFICIENT_DATA suppressed the L4 crisis override   [RESOLVED 2026-08-30]

**Status: RESOLVED. The override now reaches the band; engine patched.**

**The conflict.** Two rules in Part VI cannot both hold:
- Line 601: L4 is "stress underway -> **overrides composite** to B", and
  `warning_engine.l4_propagation_red` is written to bypass persistence entirely.
- The honesty rule: when a layer's coverage is inadequate "the layer reports NA,
  the composite prints a coverage %, and Part VIII's **do nothing** rule binds."

`step()` evaluates `insufficient` before `hysteresis_step`, so today the honesty
rule wins: `l4_override` is computed as True, the band freezes at
INSUFFICIENT_DATA, and the action is `freeze`.

**Why it matters NOW rather than eventually.** With 2 of 15 builders live, every
layer is NA. If HY OAS widened 150bp over 21 days tomorrow -- an observable,
unambiguous crisis condition needing no composite -- the system would print
"INSUFFICIENT_DATA / freeze" instead of "CRISIS / gross 0.40". The do-nothing
rule exists to prevent acting on a number you cannot compute; an L4 condition is
not a computed number, it is a direct observation of stress underway.

**The argument each way.**
- *Override should win:* L4 needs no composite. Freezing at NORMAL while credit
  gaps 150bp is the worst available failure mode, and it is the one the whole
  four-layer design exists to avoid.
- *Honesty should win:* acting on a system that cannot see 13 of its 15 inputs is
  precisely what the insufficient-data state was built to stop.

**Not decided unilaterally.** `warning_engine.py` is shipped, tested code and the
change would alter crisis behaviour. A candidate fix, if ratified: let an L4
override set the band even when insufficient, while still reporting coverage and
flagging that the composite is unavailable -- CRISIS on observation, with the
data caveat printed alongside.


---

## D11 — S4 modern red drops the historic "& level" conjunct

**Where:** `warning/builders/s4_funding.py`

**Gap.** The historic rule is explicit: `TED z>2 AND >100bp for >=5d`. The modern
composite is a mean of z-scores over CP-Tbill, SOFR-IORB and the ABCP 4-week
delta; it has no natural basis-point level, and the registry supplies none.

**Default.** Modern red = `z>2` alone. This DROPS a conjunct rather than adding
one, so **modern red is easier to reach than historic red**. Any evaluation
spanning the 2022 changeover must carry that asymmetry rather than treating S4 as
one homogeneous series.

**Also encoded:** `MIN_MODERN_LEGS = 2` -- one funding market is not "funding
stress", so fewer than two available legs returns NA. And the ABCP leg's sign is
flipped: a CONTRACTING ABCP market is funding withdrawal, so contraction reads as
stress. Without the flip that leg would point the wrong way and partly cancel the
others. Pinned by `test_s4_abcp_contraction_counts_as_stress_not_relief`.

---

## D12 — z-scored signals self-neutralize during a sustained crisis

**Status: OPEN. Property of the frozen thresholds, not a defect. No fix applied.**

**Observed on real data, twice, in two unrelated signals:**

| Signal | Date | Absolute level | Relative reading | State |
|---|---|---|---|---|
| S4 | 2007-08-31 | TED 1.84pp | z **+4.35** | **R** |
| S4 | 2007-12-31 | TED 1.63pp | z +1.20 | G |
| S4 | **2008-11-30** | **TED 2.21pp** | z **+0.83** | **G** |
| F2 | 2007-01-31 | VIX 10.96 | 12.7th pctile | Y |
| F2 | **2007-10-09** | VIX 17.46 | **85.9th pctile** | G |

S4 reads GREEN in November 2008 with TED at 221bp, because its own 252-day window
had absorbed the crisis. F2 reads GREEN at the October 2007 peak because the
August 2007 vol shock lifted its 504-day window.

**Why this is faithful and still dangerous.** Both formulas are implemented as
written (`z>2 (1y)`, `vs 504d percentile`). A trailing-window normalization
measures CHANGE, not LEVEL, so a regime that persists becomes the new normal.
The historic S4 rule has a level guard -- `AND >100bp` -- but it is a conjunct
for RED only; it cannot stop the signal decaying to G when z falls.

**Direct consequence for L4A.** L4A is defined as "S4 red + breadth-of-stress
across >=2 funding markets". If S4 cannot hold red through a crisis, L4A cannot
fire through one either -- precisely when crash propagation is the thing being
measured.

**Not fixed here.** A level floor (e.g. S4 cannot fall below amber while TED
exceeds 100bp) would fix it, but that is a new threshold, and rule #3 forbids
inventing one. Raised for a ruling, and it belongs in the Phase 4 evaluation as a
falsifiable question: does a level-floored variant beat the pure-z form on
PR-AUC over the 2007-09 and 2020 windows?


---

## D13 — CFE VIX futures are scale-normalized per row, not by an asserted date

**Where:** `warning/parse_cfe.py::normalize`

**The problem.** CFE's original VIX futures were quoted on a multiplied index and
were later de-multiplied. Measured against same-day VIXCLS on the real files:

| Year | n | median VX_FRONT / VIX |
|---|---|---|
| 2004 | 194 | **10.685** |
| 2005 | 252 | **10.275** |
| 2006 | 251 | **10.222** |
| 2007 | 251 | 1.020 |
| 2008–2018 | ~250/yr | 1.00–1.14 |

Feeding both eras into F3's `front - second` unnormalized would produce a term
structure that is pure artifact.

**Decision.** Classify EACH ROW by its own ratio to same-day VIXCLS: above 5.0 the
row is multiplied-era and is divided by 10, otherwise it is left alone. The two
regimes sit at ~1 and ~10, so a split at 5 separates them with enormous margin.

**Why not a changeover date.** Hardcoding one would be an assertion about CFE
contract history that no ingested source states. Per-row classification decides
on evidence that is in the data, and the observed switch date is printed for the
record rather than assumed in advance. Rows with no same-day VIXCLS inherit the
previous classification; if none exists yet they are DROPPED, because an
unclassifiable settle is worse than a missing one.

**Coverage finding, recorded not repaired.** The CFE archive URL pattern resolved
for 152 of 276 attempted contracts, giving VX_FRONT/VX_SECOND from
**2004-03-26 to 2018-02-23** only; CFE appears to have reorganized the archive
after that. This is sufficient for the leg's purpose -- the futures leg exists to
cover 2004-2007, before VIX3M starts on 2007-12-04 -- and 2007 is fully covered
(n=251), which is what makes F3's registry verdict testable at all. For dates
after 2018-02 F3 runs on the VIX3M leg alone, as it already does.


### D10 resolution (2026-08-30)

**Ruling: the override wins, and the reporting stays honest.**

The two rules are not symmetric. The do-nothing rule exists to stop the system
ACTING ON A NUMBER IT CANNOT COMPUTE -- a composite assembled from layers that
are mostly NA. An L4 propagation condition is not a computed number: L4B is
"HY OAS widened >=150bp over 21 sessions", true or false on its own, needing no
other layer to be valid. Suppressing it was not caution; it discarded a
measurement that had been taken successfully.

**Change:** `step()` now tests `insufficient and l4_red` before the plain
`insufficient` branch. That path sets the band to CRISIS, advances EngineState,
and emits a new alert type `L4_OVERRIDE_LOW_COVERAGE`.

**What did NOT change**, verified by running the old and new engines side by side
across five scenarios -- exactly one differs:

| scenario | before | after |
|---|---|---|
| all layers NA + L4B fires B | INSUFFICIENT_DATA / freeze | **CRISIS / gross 0.40** |
| all layers NA, no L4 fire | INSUFFICIENT_DATA / freeze | unchanged |
| all layers NA, L4 also NA | INSUFFICIENT_DATA / freeze | unchanged |
| covered + L4B fires B | CRISIS | unchanged |
| covered, no L4 fire | NORMAL | unchanged |

The composite still returns None, coverage is still reported per layer, and the
LAYER_NA alerts still fire. Only the SUPPRESSION was removed -- a reader can
always tell that CRISIS was reached on an observation rather than a score.

**Tests:** `test_l4_override_survives_low_coverage` and
`test_low_coverage_without_l4_still_freezes` (the second pins that the fix is
surgical). The previous test, which pinned the old suppressing behaviour, was
replaced.

---

## D12 — still OPEN, and currently UNTESTABLE

D12 asks whether a level-floored variant beats the pure-z form on PR-AUC over
the 2007-09 and 2020 windows. That question needs an SPX series to define the
target, and `warning.db` holds equity data only from 2016-07-18 (SPY_CLOSE).

**D12 cannot be resolved until Shiller `ie_data` is loaded** (monthly S&P back
to 1871). Recorded here so the blocker is explicit rather than rediscovered.


---

## D15 — S8's leader definition cannot distinguish an epicenter from a rotation

**Status: OPEN. Property of the frozen formula. No code changed.**

**Observed on real data, 2019-01 to 2026-08.** S8 fired exactly once:

| Date | Leader | Drawdown | Index | State |
|---|---|---|---|---|
| 2023-05-31 | XLE | -17.13%, below 200DMA | 2.22% below high | **R** |

**That is a false positive.** May 2023 was the start of a large rally, not a
peak. XLE held top trailing-2y RS from the 2021-22 energy cycle, then fell
through 2023 while the index rose on an unrelated theme. The formula saw
"the leader broke while the index held"; what happened was sector rotation.

The registry defines the leader as "sector/theme with top trailing-2y RS". That
selects the best PAST performer, which is not the same as the epicenter of the
CURRENT cycle. When leadership changes hands between cycles, the outgoing leader
decays while the market advances -- and S8 reads that as a fracture.

**Second property, same root.** At 2026-08-28 the leader is XLI with a 2-year RS
of **+1.1%**. When no sector meaningfully leads, "the leader" is whichever is
marginally ahead, and the signal is measuring noise. S8 assumes an epicenter
exists; it has no way to report that none does.

**Not patched.** A minimum-leadership threshold (e.g. "leader RS must exceed
X%") would be a new number, which rule #3 forbids. So would a rule requiring the
leader to have led recently rather than over the full two years.

**For the ruling.** Two candidate amendments, both requiring ratification:
1. Require a minimum RS gap between the leader and the second-place sector, so
   "no clear leadership" reports NA instead of picking a marginal winner.
2. Require the leader to also be within some distance of its own high at the
   START of the drawdown window, which would exclude a sector already in
   multi-year decline.

Until ruled, S8 stands as written and its single historical fire is recorded as
a false positive rather than a hit. **Do not count it as validation.**

**Also pinned by test:** a sector that collapses far enough loses its 2-year
leadership, so S8 goes quiet once a fracture completes. It has a window --
deep enough to breach -15% and the 200DMA, shallow enough to remain the leader.
See `test_s8_loses_sight_of_an_epicenter_that_has_fully_collapsed`.


---

## D16 — S9's linear detrend is misspecified, and its fires are all false so far

**Status: OPEN. Property of the frozen formula plus a data limitation. No code
changed.**

**Firing record, 2022-06 to 2026-08 (the whole computable sample):**

| Date | z | What followed |
|---|---|---|
| 2022-10-31 | +1.66 | **the bear-market low** |
| 2024-06-30 | +3.34 | market rose |
| 2024-09-30 | +1.60 | market rose |
| 2024-11-30 | +1.55 | market rose |
| 2025-12-31 | +2.08 | no decline |
| 2026-05-31 | +2.15 | no decline |

Six reds, zero preceding declines. The 2022-10 fire is the worst case a
`rising_SI_bearish` signal can produce: maximum bearishness at the bottom. It is
also the only era the registry considers checkable -- 2000 and 2008 are already
marked UNDECIDABLE -- so S9's single testable verdict is a miss.

**THE SPECIFICATION PROBLEM.** The fitted trend slope rises monotonically across
the scan: 0.00262 -> 0.00283 -> 0.00311 -> 0.00349 -> 0.00471 -> 0.00589 per
observation. Aggregate short interest is not growing at a constant exponential
rate; it is ACCELERATING. An expanding linear fit of log(aggregate) on time
therefore leaves systematically POSITIVE residuals at the endpoint, because the
line is always catching up to a curve that keeps steepening.

That biases z upward and toward firing. The reds above are substantially an
artifact of fitting a straight line to a convex series, not a measurement of
positioning stress.

**Why it is not patched.** The registry says "detrended log aggregate SI (linear
trend to date)". A quadratic term, a rolling-window trend, or first-differencing
would each fix the bias and each is a new specification -- rule #3 forbids
choosing one unilaterally.

**Compounding data limitation.** short_interest.db starts 2021-04-15, not the
registry's 2014. The whole computable sample is one regime of rising short
interest, so there is no period of flat or falling aggregate SI against which to
calibrate. Even a correctly specified trend would be estimated on a monotone
sample.

**For the ruling.** Candidates, in rough order of how little they invent:
1. First-difference the log aggregate and z-score the CHANGE. Removes any trend
   without choosing a functional form.
2. Rolling-window linear trend rather than expanding, so the fit tracks
   acceleration instead of lagging it.
3. Leave as written and treat S9 as descriptive-only until FINRA history back to
   2014 is loaded, which would at least span more than one regime.

Until ruled, **S9's reds must not be counted as validation of anything.** The
signal is computable and its plumbing is correct -- point-in-time panel,
publication lag, expanding fit -- but its output in this sample is dominated by
a fitting artifact.


---

## D17 — no signal built from local data has a demonstrated hit rate

**Status: FINDING, recorded 2026-08-30. Not a defect. No code changed.**

Eight composite inputs are now built. The four whose fires can be evaluated on
the available sample have produced nineteen fires and no validated hit.

| Signal | Fires | Outcome |
|---|---|---|
| S5 breadth | 11 red (8 episodes) | mean 126d drawdown -8.94% vs -8.05% for ALL new-high days, n=11. Indistinguishable. AMBER (-6.95%) shows LESS drawdown than GREEN (-8.52%) -- backwards for a signal with real ordering. |
| S7 defensive rotation | 1 (2024-09-30) | ~5 months before a >9% drawdown. Plausible, n=1. |
| S8 epicenter fracture | 1 (2023-05-31) | False positive: XLE rotation, not a fracture. See D15. |
| S9 aggregate SI | 6 | All false; the 2022-10-31 fire landed on the bear-market low. See D16. |

**Why this is expected rather than alarming.**
- The computable sample is 2016-2026 for price-derived signals and 2021-2026 for
  short interest. It contains no major credit crisis -- exactly the regime these
  thresholds were specified against.
- The registry's own verdicts concede this: S5's 1998 and 2007 anchors predate
  the data, S9's 2000 and 2008 are UNDECIDABLE, and S2's 2007 fire was
  established on HY OAS the stack no longer has (D8).
- S5's survivorship bias runs AGAINST firing, so its record is an undercount.

**Why it still matters.** When L2 completes, the composite will be assembled from
components with no demonstrated hit rate on any data this project can see. The
band will start producing numbers that LOOK like a measurement. Before that
happens, the honest position must be recorded: these signals are correctly
implemented and point-in-time clean, and that is a different claim from
validated.

**What would change this.** Shiller `ie_data` (extends the equity record to
1871), FINRA short-interest history to 2014, and a delisting-inclusive universe
for S5. Until then, Part V's false-positive accounting has a numerator and no
usable denominator.

**Explicitly NOT concluded:** that these signals do not work. A sample with no
credit crisis cannot disprove a crisis detector. The finding is that the sample
cannot support a verdict either way, and that no verdict should be implied by
the composite once it turns on.


---

## D18 — L4A measures "breadth of stress" with S4's own arm threshold

**Where:** `warning/builders/l4_propagation.py::funding_seizure`

**Gap.** Report line 601 defines L4A as "funding seizure (S4 red +
breadth-of-stress across >=2 funding markets)". It quantifies neither what makes
an individual market "stressed" nor which markets count.

**Default and why.** A market counts as stressed when its own z exceeds S4's
`threshold_arm` (z>1.5) -- a value already frozen in the registry for exactly
this data, reused rather than replaced. Inventing a fresh number would breach
rule #3; borrowing the one the registry already applies to these same series
does not.

The markets are S4's modern legs: CP-Tbill, SOFR-IORB, and the ABCP 4-week
delta. Those are the three funding markets the registry itself names for S4.

**Historic mode returns NA, deliberately.** S4's historic mode is TED alone.
One spread cannot supply breadth across two markets, and treating it as its own
confirmation would be circular. L4A therefore reports NA whenever S4 is in
historic mode rather than degrading to "S4 red" with the breadth condition
quietly dropped -- the same discipline applied to S4's own D11.

**Operational significance.** Under D10 an L4 condition reaches the band even
when coverage is inadequate. With L1 at 0% and the composite frozen at
INSUFFICIENT_DATA, L4A is one of the few paths by which this system can say
anything at all today. Building it materially improves live usefulness without
waiting on the blocked L1 downloads.

**Coverage after this:** L4 goes 2/5 to 3/5 (60%). Still below the 70% floor, so
L4 remains an NA layer -- L4D needs S10 (FINRA margin) and L4E needs F9
(negative-gamma, Phase 5). The composite still cannot turn on.


---

## D19 — S6's thresholds assume a distribution the 2016-2026 sample does not have

**Status: OPEN. Property of the frozen thresholds. No code changed.**

**Measured on 449 sampled readings, 2017-2026:**

| statistic | value |
|---|---|
| mean EW-CW 126d | **-1.67%** |
| median | -1.48% |
| below -2% (the ARM level) | **41% of the time** |
| below -4% (the RED level) | **23% of the time** |
| above zero | 29% |

S6 fires red on roughly a quarter of all readings and sits at or past its arm
level on 41%. A signal meant to flag anomalous narrowing instead describes the
baseline of the mega-cap era.

**The mechanism.** The registry's -2%/-4% thresholds are absolute levels, which
presume EW-CW oscillates around zero. Across 2016-2026 equal-weight has
persistently lagged cap-weight, so the distribution is shifted roughly -1.7%
below where the thresholds assume it sits. What the signal reads as narrowing is
mostly the era's constant.

**Distinct from D16, though the consequence is the same.** S9's defect is a
misspecified TREND -- a linear fit chasing accelerating growth. S6's is a
misspecified LOCATION -- a stationary threshold against a shifted mean. Neither
is a coding error and neither can be fixed without changing a frozen number.

**Not patched.** Detrending, demeaning, or z-scoring EW-CW would each fix it and
each replaces the registry's specification, which rule #3 forbids. The registry
also anticipates a longer history (RSP from 2003, French size portfolios before
that) in which the pre-2016 era may well be better centred -- so the thresholds
may be right for the full sample and wrong for the decade available here.

**One reading that may be real.** 2021-10-31 fired R at -4.05%, roughly two
months before the January 2022 peak. That is the second genuine pre-peak fire in
the build after S11's 2000-03. It should not be counted as validation while the
signal also fires 23% of the time.

**For the ruling.** Extending the history from Massive is NOT available --
tested 2026-08-30 and recorded below. So the choice is between accepting S6 as
descriptive-only, or sourcing equity history elsewhere.

### Massive's history is capped at 2016-07-18 (tested 2026-08-30)

Requesting RSP from 2003-01-08, 2008-01-01 and 2012-01-01 each returned the
SAME 2,511 rows beginning 2016-09-01. SPY, XLK, XLP and AAPL each returned 2,544
rows beginning 2016-07-18 for a 2003 request. The `start` parameter is ignored
below the plan's floor.

**This is a plan limit, not a collection gap.** prices.db's 2016 start is
therefore the data available, not an unfinished backfill -- and no amount of
re-ingesting will extend it.

**Consequences, which reach well beyond S6.** Every price-derived signal in this
build is confined to 2016-2026 by the same limit: S2's equity leg, S5, S6, S7,
S8 and S14's leg (a). None of them can see 2008, and the sample they share
contains no credit crisis. That is the mechanical cause of D17, and it cannot be
fixed inside this data source.

**What would actually fix it.** Shiller `ie_data` supplies monthly S&P back to
1871, which addresses the index-level signals (S2's gate, S13, D12) but not the
sector or breadth signals, which need per-constituent daily bars. Those would
need a different vendor or the Ken French portfolios as a coarse substitute.
