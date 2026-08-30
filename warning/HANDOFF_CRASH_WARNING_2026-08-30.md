# HANDOFF — Crash Early-Warning System

**Regenerated 2026-08-30 (ET).** Supersedes `HANDOFF_CRASH_WARNING_2026-08-28.md`,
which described 3 builders and 4 decisions. Branch `research-track`, HEAD
`bf223568`, pushed. **151 tests green. 23 decisions recorded** — read
`warning/DECISIONS.md` alongside this; it is the authority on every spec gap.

---

## 1. State

| | |
|---|---|
| Composite inputs built | **11 of 15** — S1 S2 S3 S4 S5 S6 S7 S8 S9 S11 S14 |
| Not built | S10 (FINRA), S13 (Shiller), S15 (Shiller + Z.1), S12 (EDGAR Form 4 to 2003) |
| Dashboard | F2 VIX percentile, F3 VIX term slope |
| L4 propagation | 3 of 5 — L4A funding seizure, L4B spread blowout, L4C correlation spike |
| Calibration companions | S7F, S8F — S7/S8 thresholds on French data 1926-2026 |
| Layer coverage | L1 25%, L2 89%, L3 100%, L4 60% |
| Band today | **INSUFFICIENT_DATA** — two NA layers, correct |

**Live read 2026-08-28:** S1=R (2025-01 escalation, inside the 24-month window),
everything else G, F2=Y (VIX 14.51, 6th percentile), F3=G (+21% contango).
Complacent vol, quiet credit, quiet funding.

---

## 2. What the data can and cannot do — the hard ceilings

Each of these was tested, not assumed, and each bounds what any signal built on
it can claim.

| Source | Reach | Consequence |
|---|---|---|
| **Massive prices** | **2016-07-18, hard** | `start` is ignored below the floor. S2's equity leg, S5, S6, S7, S8, S14(a) are all confined to a decade with no credit crisis. This is the mechanical cause of D17. |
| ALFRED vintages | HOUST **1960**, DRTSCILM 2010, ABCOMP/CSUSHPINSA 2014 | only HOUST gains real depth; S3 is blind before 2010 |
| FRED daily | 1954-1962 for curve/credit | S1, S2, S4 reach the real crises |
| Cboe / CFE | VIX 1990, futures 2004-**2018-02** | CFE archive reorganised after Feb 2018 |
| Ritter | 1980, annual | S11 spans 2000 and 2008 |
| Short interest | **2021-04** | S9's 12m z has n=24 |
| French portfolios | **1926**, survivorship-safe | calibration-grade only (D20) |
| Form 4 | 2025-07 only, 13 code-P rows | S12 not buildable |

---

## 3. The method that mattered

Four decisions were settled by running **frozen thresholds against a century**
of French data rather than arguing from the 2016-2026 decade. The answers went
different ways, which is the point:

- **D19** — S6's thresholds *vindicated*. EW−CW averaged +0.96% for ninety
  years; the registry's −2%/−4% levels fire 6% of the time across the century.
  The 2016-2026 decade is the outlier (mean −1.11%, red 18%). I had called the
  thresholds misspecified; the century said the decade was.
- **D15** — S8's thresholds *cleared* (0.6% firing rate) but its leader
  definition *indicted*. Four of six century episodes are the previous cycle's
  leader decaying while the market advances. It never fired before a major peak
  in 96 years.
- **D21 → D23** — S7's thresholds cleared, then its apparent record destroyed by
  the base rate.

**A verdict from the recent decade alone would have been wrong in at least two
of these.**

---

## 4. D23 — the finding that shapes what comes next

Against the unconditional base rate over 1,167 month-ends, 1926-2026:

| | fwd 126d median | worst 252d mean |
|---|---|---|
| ALL month-ends | +6.48% | −9.47% |
| S7F RED (n=13) | **+7.18%** | −9.52% |
| S8F RED (n=6) | +2.98% | **−7.82%** |

S7F is indistinguishable from chance — its median forward return is *higher*
than the base rate. S8F's fires precede *shallower* drawdowns than average.

**Every outcome-testable signal has now failed**: S5 (n=11, indistinguishable),
S7 (n=13, indistinguishable), S8 (inverted), S9 (6 fires, one at the
bear-market low). Only S11's 2000-03 fire is uncontradicted, at n=1.

**This is not necessarily fatal to the report's thesis.** Its design claim is
that no *single* signal works — that the four-layer conjunction carries the
information. Weak components are consistent with that. But **no component behind
the composite has demonstrated standalone value**, and the band must not be read
as though one had.

---

## 5. Open rulings

| | |
|---|---|
| **D15** | S8's leader definition — amendment needed, target the leader not the thresholds |
| **D16** | S9's linear detrend is misspecified on accelerating growth. **Untested** — exactly where D15 was before the century test |
| **D12** | z-scored signals self-neutralize in a crisis (S4 green at TED 221bp in Nov-2008). Blocked on Shiller |
| D4, D8 | registry cells that contradict their own formulas |

Resolved today: D10 (L4 override now reaches the band), D19, D21→D23.

---

## 6. Next steps, in order

1. **Shiller `ie_data`** (shillerdata.com, manual) → S13, S2's pre-2016 equity
   leg, S15's equity leg, **and D12's test**. L1 → 50%.
2. **FINRA margin xlsx** (finra.org, manual) → S10 **and L4D**. L1 → 75%,
   L4 → 80%. **With both, at most one NA layer remains and the band turns on.**
3. **Part V false-positive accounting on the COMPOSITE — before any band is read
   as advice.** Individual components have been tested and found weak; the
   conjunction never has.
4. D16's century test, by analogy with D15. S9 is short interest so French
   cannot serve; FINRA history to 2014 would.

---

## 7. Operational notes

- **cboe.com is ISP-filtered.** Not DNS — `--resolve` to the real IP with correct
  SNI still gets RST. Route is the VPN. Archives are static, so one-time.
- **`warning.db` is 225 MB and gitignored** (`*.db`, line 9). Every row is
  derived; the ingest chain reproduces it.
- **UW cron** 06:30 ICT Tue–Sat = 19:30 ET Mon–Fri. Phase 0's second session
  closes **Tuesday 2026-09-01**; Saturday's run collided pre-migration and lost
  Friday.
- Browsers append `-1` on re-download; verify with `grep -c` after moving a file.
  One delivery this session shipped an unmoved `run_signal.py` and the commit
  looked clean.
- Long scans: `french_century_scan.py` loads once and walks forward. Its
  predecessor re-compounded thirteen century-long series per month-end and took
  four minutes.

---

## 8. Verify from cold

```
cd ~/Desktop/ML_Quant_Fund
cd warning && python -m pytest -q ; cd ..           # expect 151 passed
python warning/daily_driver.py --db warning.db --dry-run
python warning/french_century_scan.py --db warning.db
python warning/french_fire_value.py --db warning.db
```

Every builder has `--validate`, which replays the registry's own ex-ante
verdicts. **A mismatch is a finding: report it, do not tune the builder.**
Thresholds are frozen (rule #3); every deviation is recorded in `DECISIONS.md`
rather than absorbed into code.
