# MONITOR DEFECT LEDGER — single source of truth
*Both threads (pipeline-engineering AND report-writing) read this before writing any §4 / §16.4
and append here when shipping or discovering. Report §4 renders FROM this file; a monitor pull
alone is NOT sufficient evidence a defect is open — several fixes are invisible in stdout.*
*Format: one row per defect, never deleted. Status ∈ OPEN / FIXED / SUPERSEDED / DEAD / GATED.*
*Last sync: 2026-07-26 (session: report-audit + fix batch, commits c6b6e636..c32394db).*

| Item | Status | Commit / evidence | Notes |
|---|---|---|---|
| UW /ohlc/1d 3-rows-per-session (returns, spot) | FIXED | f4b6ce6d | `_daily_returns` from prices.db; IWM→SPY |
| VN date off-by-one (`date.today()`) | FIXED | c6b6e636 | `_today_et()` at 11 sites |
| Stale earnings_date anchors front weekly | FIXED | b1eb33b2 + guard-routed 7e8ae039-era | `section_implied_move` routed through guard |
| Form 4 other-issuer (GV) misattribution | FIXED | cb294e5a | **CIK filter — supersedes any issuer-price heuristic. STRIKE issuer-price BUILD** |
| form4_transactions dupes / case / no unique index | FIXED | 7e8ae039 | 914 rows deduped, `tsla`→TSLA, `uq_form4_tx` permanent |
| EPS contamination GOOG 07-22/04-29 (GAAP marks) | FIXED | (EPS batch) | DB NULLed + writer + display quarantine + >100% tripwire |
| EPS basis mismatch DDOG (GAAP vs adj, systematic) | FIXED | 7e8ae039 | ticker-level quarantine tier |
| Missing `street_mean_est` key (Est all "?") | FIXED | (EPS batch) | also revived surprise + tripwire |
| Options aggressor side dropped | FIXED | (tilt commit) | ask/bid tilt + A% + vendor OPENING tag |
| Dark-pool partial days (13-ticker universe) | FIXED | repair runs Jul 24-26 | 0 partials all tickers; pre-2026-05-29 permanently partial (vendor cap) |
| Top-10 prints rank AH/mechanical as organic | FIXED | 379749f7 | AH/MECH tags |
| Skew universe silent (57% specimen) | FIXED | 379749f7 | signed-universe line |
| Closing crosses inside signed skew (PLTR claim) | DEAD | probe Jul 24 | **REFUTED** — ext_hours excluded them all along |
| clean_skew wiring into 7d aggregate | DEAD | — | justifying defect refuted; not shipped |
| 1d-move column all "?" | FIXED | (batch Jul 26) | prices.db backfill, `*` marker, normalize-gate bugfix |
| Insider rolling-window denominator artifact | FIXED | (batch Jul 26) | fixed 90d/365d totals printed |
| Put-call parity flags cost-of-carry | FIXED | (batch Jul 26) | C−P = S−K·e^(−rT), r=4.5% |
| PARTIAL label misleads on healed windows | FIXED | (batch Jul 26) | label states live-walk-stops-at-cache |
| Generic-prior historical-move category error | FIXED | c32394db | RICH/CHEAP suppressed under generic prior |
| GOOG cohort undefined | FIXED | c32394db | MSFT/AMZN/META; first read −13.44pp/20d |
| Live-book membership not printed (§16.2 blocker) | FIXED | c32394db | LIVE BOOK line in every ticker header |
| Dark-pool skew predictive validation | GATED | 23849ec6 + cron aa777f86 | v1.0 live; 2026-07-26 run: UNDERPOWERED (8 dates, h=1 IC +0.138, t 1.54, z 1.2σ); monthly on the 12th; bar \|z\|≥3 ∧ \|NW-t\|≥3 both horizons |
| DB lock crashes (Jul-17, Jul-25) | FIXED | timeout=30 (parallel session) + iCloud relocation | WAL deliberately withheld until post-migration verify |
| 10-Q extraction | OPEN | — | sole BUILD-PRIORITY; test accession 0001652044-26-000066 |
| Intraday session H/L | OPEN | — | blocks range analysis; external substitutes class-mismatched |
| GOOG next-earnings date 3-way conflict | OPEN | — | monitor 2026-11-04 (UW placeholder [inf]) vs external Oct 27–28; update on Alphabet announcement |
| Peer-relative DORMANT root-cause | OPEN | — | 5 clean runs; clean ≠ fixed |
| UW news 404 / Massive session_reset | COSMETIC | — | vendor-plan / log-only |
| Cohort peer-residual factor | RESEARCH | — | REV3 candidate; goes through the standard gate (null control, HLZ t>3) before any sink |
| Aggressor-tilt validation | BLOCKED | — | needs flow-alert persistence build first; then reuses skew gate via --signal |
| Block desynchronization (3 prices/run) | FIXED | (this commit) | staleness fall-through + bar-date print; universal via _daily_returns |
| NVDA/DDOG/BYND earnings dates | FIXED | (this commit) | all company-PR/WSH confirmed; OKLO QUBT CRWD SNOW NVMI remain stale-guarded, batch-source before windows |
| RZLV dark-pool 7-session gap (13-22 Jul) | FIXED | repair Jul 29 (347 rows, 0 partials) | RZLV not in DEFAULT_TICKERS -- monthly repair skips it, heal manually or add |
| si_live_ledger staleness (all reports) | DATA-REFRESH | hand-maintained CSV, no generator; stamped Jun-26 | NOT a code defect -- Atom re-dates/refreshes; report threads must stop flagging as monitor bug |
| Implied-move published on broken chains | FIXED | (this commit) | parity before print; number suppressed, not caveated |
| Spot/chain clock mismatch post-event | FIXED | (this commit) | re-anchor to live quote past 2pct divergence |
| earnings_date coverage (13 hand names) | FIXED | (this commit) | earnings_calendar fallback = 312 tickers; hand config wins when fresh |
| AMD + MRVL HELD in si_live_ledger | NOTED | grep Jul-31 | 7sh AMD / 14sh MRVL, SI long leg -- those reports ARE position-relevant |
| DB locks (Jul-17, Jul-25) | FIXED | dfa50620 | 108 bare sqlite3.connect sites, 65 files -- default 5s timeout meant collisions crashed instead of waiting; ui/ concentration matches the dashboard long-read note |
| Scheduler duplicates | FIXED | 245c4e5e + inventory | 3 jobs double-owned Jul 26-Aug 1 (my install resurrected them); chain now holds an atomic lock |
| Destroyed crontab jobs (Jul-26) | FIXED | 800b6f8a | 11 jobs restored incl. feed_freshness_check; install_crontab.sh refuses destructive installs, crontab_drift_check.sh on launchd |
| health_check false alarms | FIXED | 9f26efcd + 02ddf15b | 77 consecutive false failures, 0 passes ever; maturity-aware now, status file + pipecheck age display |
| cron_canary | REMOVED | dfa50620 | echo-only stub, 2880 runs/day |
| 24-vs-50 form4 count | UNEXPLAINED | -- | overwrite hypothesis REFUTED (5 quarantine markers durable, 3 new catches). No replacement story |
