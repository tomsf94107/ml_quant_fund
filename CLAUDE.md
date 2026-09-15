# CLAUDE.md

Solo systematic ML quant fund, US equities. Real capital is not currently
deployed. **One** signal has ever cleared this fund's bar: the SI / days-to-cover
brick. Six axes have been closed against it. Assume any new idea is more likely
to join the closed list than the validated one, and test accordingly.

---

## RULES OF EVIDENCE

Each of these was violated in a single session on 2026-09-14/15, in a repo whose
own documentation already warned against it. They are not general advice.

**1. No number without its benchmark, in the same output.**
A raw hit rate is uninterpretable. 58% is excellent against a 50% base and
negative against a 60% one. `docs/MASTER_TODO_LIST.md` §1.1b: *"The 58% is the
market's rising tide (57-60% of EVERYTHING went up this window), not skill."*
A model's h=5 hit rate falling 61.9% → 37.4% was read as decay three times. The
same months, benchmarked against the rest of each day's universe, ran **+7.8,
+6.8, +8.8, +8.1, +9.3pp** — the strongest month was the one that looked worst,
because the field fell faster than the selection.

**2. Portfolio performance is DAY-WEIGHTED.**
Pooled weights every name equally regardless of which date it fired, which
describes a book you could only size knowing the future distribution of daily
counts. PCT7: **+1.46% pooled, −1.97% day-weighted, identical data** — see
`RETRACTED_BRICK_2_PCT7_2026-09-05.md`. A long/short book read "positive in six
of seven months" pooled and +0.436% with two clearly negative months when
day-weighted.

**3. State the timezone on every timestamp comparison.**
Cron logs are **VN-dated** (BSD cron ignores TZ). Data is **ET-dated**. Two
non-existent bugs were chased from reading ET as VN — a "missing DB write" that
was a 16:37 VN write stamped 05:37 ET, and a "second repo".

**4. One seed or a reduced universe is a PROBE.**
A long/short book read **+2.796pp at NW t +3.34** on 120 tickers and
**+1.502pp at t +1.38** on 400. Thin cross-sections have more extreme tails.
See `FINDING_h40_dispersion_is_draw_size_2026-09-13.md`: 80-name draws spread
3.32pp across seeds, 400-name draws 0.58pp, same universe.

**5. Absence is not evidence.**
Three wrong conclusions in one session came from "I looked and didn't find it":
separate repos, a missing DB write, dead features that were live. Read what the
code opens. A failed grep means the grep was wrong.

**6. When the basis is untrustworthy, SUPPRESS the derived figure — don't
annotate it.** 13F ownership of 273.8% was printed with the numbers beside it;
a stale health verdict was printed under a staleness warning. Each time the
annotation was there and got read past.

---

## ENVIRONMENT

**yfinance is dead.** XProtect 5347 SIGKILLs the process on exec, uncatchable by
try/except. Index symbols route to FRED via `_FRED_FOR_INDEX` in
`features/massive_client.py` (both the single-ticker and list branches). Futures
and international indices have no FRED series and correctly return empty.

**`prices.db` allows exactly ONE writer.** `analysis/universe_fetch.py` records
four lock incidents on 2026-09-05; one corrupted a seed's panel silently —
`except Exception: continue` swallows the failure and the seed just has fewer
names. Check `ps aux | grep -E "[h]40_|[t]rain_all|[u]niverse_fetch"` returns
nothing before starting anything that builds features or fetches bars.

**`/Users/atomnguyen/ML_Quant_Fund` and `~/Desktop/ML_Quant_Fund` are
HARDLINKED** — same inodes, one set of files. Not two repos.

**Silent-constant failures are this repo's characteristic bug.** `vix_ret` and
`dxy_ret` were identically 0.0 in 100% of stored rows for three months because
FRED publishes a day late, the builder forward-fills, `pct_change()` on a
duplicate is zero by arithmetic, and `daily_runner` logs `df.iloc[-1]`. The
regime classifier's VIX sat at its dataclass default of 20. `alerts/scanner.py`
returned zero alerts for ten weeks. None raised an error. Two `health_check.py`
assertions now catch the class: a feature constant over 30 days, and a daily
macro scalar varying by ticker within a date.

---

## STANDING BARS

- **Deployment needs NW t > 3.0** on the net figure at realistic cost, not gross.
- **Pre-register the bar before running the test.** `validate_borrow_battery.py`
  is the model: criteria written into the docstring, then run.
- **Tag decision-relevant claims** `[fact]` / `[inf]` / `[fact?]`.
- **No fabrication.** Missing data is a finding, not a gap to fill.
- **Overlapping forward windows need Newey-West**, lag = horizon. A raw t on
  40-day windows sampled daily is inflated by roughly √(H).
- **A check that cries wolf is worse than no check.** `health_check.py` records
  77 consecutive false failures that trained everyone to ignore the log, which
  is how four dead feeds went unnoticed for six days. Thresholds must sit in an
  empty gap between the normal and broken regimes, and be shown to be
  insensitive across a wide range.

---

## CLOSED — do not re-test without a new reason

| axis | why |
|---|---|
| Insider selling | `CLOSED_AXIS_insider_selling_2026-09-03.md` |
| Insider breadth | 7 constructions, 365,910 filings, nothing cleared t=3.0 |
| Macro announcements | Effect died ~2015; post-publication decay to zero |
| PTH selloff | Within-beta IC −0.0487, max \|t\| 2.43 against a 3.0 bar |
| Borrow-fee residual | Sector-neutral retained 34% against a 40% bar; OOS t +1.25 |
| PCT7 | Retracted: pooled/day-weighted error |
| Universe expansion | Arm B is the illiquidity slice; arm D showed the wider training panel *destabilises* — 4.17pp spread on a fixed scored set |
| Long/short at h=5 | +0.436pp gross, zero at 10bps/leg; tighter gates halved the t-stat |

Features from closed axes stay wired inside models. No standalone signal exists
for any of them.

---

## THE KILL SWITCH

`signals/generator.py:996` forces every BUY to HOLD when
`ML_QUANT_DISABLE_BUY != "0"`, and the default is ON. Set 2026-05-31 on
*"decile spread t=−2.29 at h=5"*, which `analysis/decile_spread_test.py` now
reproduces **exactly**. Post-switch it reads +1.59 on 66 dates — positive, not
significant. **It lifts when post-switch h=5 clears t > 3.0, not before.**
Re-measured monthly on the 1st.

---

## SCHEDULED SELF-CHECKS

`health_check.py` 13:00 VN · `decile_spread_test.py` 1st 09:00 VN ·
`recovery_tracker.py` 1st 09:05 VN · crontab snapshot Sundays 11:00 VN ·
`crontab_drift.sh` daily.

The tracked crontab drifted 261 lines behind the live one because nothing
regenerated it. Any record maintained by hand goes stale silently.
