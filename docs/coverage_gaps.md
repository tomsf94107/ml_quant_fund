# Monitoring coverage — what is watched, and what still is not

**2026-09-06.** Written after a session in which four separate feeds were found
frozen or dead, none by the monitors that existed.

## The failure shape, in the codebase's own words

`scripts/feed_freshness_check.py`, header:

> *"prediction_features / portfolio_returns_ab had date_col='date'. Neither
> table HAS a 'date' column ... so both ERRORed every run and were effectively
> UNWATCHED. An entry that ERRORs every run is not 'watched'; it is unwatched
> with extra steps."*

The monitor built to catch stale feeds had gone blind on two of its own entries.
That is the shape to design against: **it runs, writes nothing useful, exits 0,
and nobody is told.**

Four more instances surfaced on 2026-09-05:

| what | how long unnoticed |
|---|---|
| `vix_term_structure` pinned to the literal 1.0 | 10 weeks |
| `eightk_items` frozen at 2026-05-27 | 3 months |
| `sentiment_scores` frozen at 2026-05-28 | 3 months |
| 24,741 PCT7 shadow predictions written, never scored | 14 weeks |

## Now covered

| monitor | cadence | catches |
|---|---|---|
| `feature_health_monitor` | Mon 10:00 | features that DIED, REVIVED or VANISHED from importance |
| `feature_break_audit` | Mon 11:30 | value-series breaks, constants, collapses, newly-null |
| `pipeline_audit` | Mon 11:45 | source freshness and end-to-end feature health |
| `fund_ep_verdict` | Mon 10:30 | whether two wired features earned their place |
| `si_period_split` | monthly | SI brick IC by year and half-sample |
| `si_staleness_check` | Mon 10:45 | a missed FINRA publication, which `[AUTH ERROR]` cannot distinguish from a quiet day |
| `feed_freshness_check` | daily | per-table `MAX(date)` against a per-feed budget |
| `h40_shadow` | Tue–Sat 09:15 | logs the frozen book's daily picks |
| `monitor_heartbeat` | daily 13:00 | that the monitors themselves are still writing |

## Still NOT covered — ordered by how much it would cost to miss

**1. The frozen shadow book's model_sha.**
`h40_shadow.py --status` reports the sha and flags if more than one ever
appears. It is not scheduled. The entire out-of-sample claim rests on the model
never being retrained, and nothing checks that. A retrain would void every
observation silently.

**2. Features computed but not in `OUTPUT_COLUMNS`.**
`features/builder.py` line ~1982 runs `df = df[OUTPUT_COLUMNS]`, which drops any
column not in that list with no error and no warning. This is how `macd_signal`
and `es_overnight` vanished in April, and it cost six wrong diagnoses on
2026-09-05 before being found. The break audit cannot see it either, because it
reads the builder's OUTPUT. A check comparing assigned column names in the
source against `OUTPUT_COLUMNS` would catch it; none exists.

**3. `boulton_cell` will be flagged ALL_NULL and will be fine.**
It is NaN on roughly 92% of rows because dark-pool history starts 2026-03-19 and
it needs 60 observations of a 252-day rolling quantile. The break audit will
report it correctly by its own rules and wrongly as a defect. Expect this until
roughly March 2027.

**4. The audit samples 25 of 415 tickers.**
A break affecting a subset can be missed. Raising the sample costs runtime
linearly; 25 was chosen because a construction or vendor change shows on every
ticker, which is the case it is built for.

**5. Universe drift.**
The audit reads `tickers.txt` (415 names). `tickers_expanded.txt` holds 1,920
with full price history, short interest on 1,942 and XBRL on 1,871. If the
expanded set is adopted, every monitor pointing at `tickers.txt` watches a
quarter of the data.

**6. Verdict correctness.**
`monitor_heartbeat` checks that logs are recent and non-empty. It cannot check
that a verdict is right. `feed_freshness_check` ran, wrote, and was blind on two
entries for months while looking healthy. No automation closes this; it needs
someone reading output occasionally.

## The honest limit

Every monitor here detects a KNOWN failure shape. The four feeds found on
2026-09-05 were found by a human running an audit that had never been run
before, prompted by reading someone else's research methodology.

**Monitors catch repeats. They do not catch the first instance of anything.**
That is an argument for periodic manual audit, not against automation.
