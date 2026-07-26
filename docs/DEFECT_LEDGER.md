# MONITOR DEFECT LEDGER — single source of truth
*Both threads (pipeline-engineering AND report-writing) read this before writing any §4 / §16.4
and append here when shipping or discovering. Report §4 renders FROM this file; a monitor pull
alone is NOT sufficient evidence a defect is open — several fixes are invisible in stdout.*
*Format: one row per defect, never deleted. Status: OPEN / FIXED / SUPERSEDED / DEAD / GATED.*
*Last sync: 2026-07-26 (report-audit + fix batch, commits c6b6e636..c32394db).*

| Item | Status | Commit / evidence | Notes |
|---|---|---|---|
| UW /ohlc/1d 3-rows-per-session (returns, spot) | FIXED | f4b6ce6d | _daily_returns from prices.db; IWM to SPY |
| VN date off-by-one (date.today) | FIXED | c6b6e636 | _today_et at 11 sites |
| Stale earnings_date anchors front weekly | FIXED | b1eb33b2 | plus section_implied_move routed through guard |
| Form 4 other-issuer (GV) misattribution | FIXED | cb294e5a | CIK filter supersedes issuer-price heuristic. STRIKE the issuer-price BUILD |
| form4_transactions dupes / case / no unique index | FIXED | 7e8ae039 | 914 deduped, tsla to TSLA, uq_form4_tx permanent |
| EPS contamination GOOG 07-22 and 04-29 (GAAP marks) | FIXED | EPS batch | DB NULLed + writer + display quarantine + 100pct tripwire |
| EPS basis mismatch DDOG (GAAP vs adj, systematic) | FIXED | 7e8ae039 | ticker-level quarantine tier |
| Missing street_mean_est key (Est all ?) | FIXED | EPS batch | also revived surprise + tripwire |
| Options aggressor side dropped | FIXED | tilt commit | ask/bid tilt + A pct + vendor OPENING tag |
| Dark-pool partial days (13-ticker universe) | FIXED | repairs Jul 24-26 | 0 partials all tickers; pre-2026-05-29 permanently partial (vendor cap) |
| Top-10 prints rank AH/mechanical as organic | FIXED | 379749f7 | AH/MECH tags |
| Skew universe silent (57 pct specimen) | FIXED | 379749f7 | signed-universe line |
| Closing crosses inside signed skew (PLTR claim) | DEAD | probe Jul 24 | REFUTED — ext_hours excluded them all along |
| clean_skew wiring into 7d aggregate | DEAD | never shipped | justifying defect refuted |
| 1d-move column all ? | FIXED | batch Jul 26 | prices.db backfill, star marker, normalize-gate bugfix |
| Insider rolling-window denominator artifact | FIXED | batch Jul 26 | fixed 90d/365d totals printed |
| Put-call parity flags cost-of-carry | FIXED | batch Jul 26 | C-P = S-K*exp(-rT), r=4.5 pct |
| PARTIAL label misleads on healed windows | FIXED | batch Jul 26 | live walk stops at cache by design |
| Generic-prior historical-move category error | FIXED | c32394db | RICH/CHEAP suppressed under generic prior |
| GOOG cohort undefined | FIXED | c32394db | MSFT/AMZN/META; first read -13.44pp/20d |
| Live-book membership not printed (16.2 blocker) | FIXED | c32394db | LIVE BOOK line in every ticker header; GOOG not held |
| Dark-pool skew predictive validation | GATED | 23849ec6 + cron aa777f86 | first run 2026-07-26 UNDERPOWERED (8 dates, h=1 IC +0.138, t 1.54, z 1.2 sigma); monthly on the 12th; bar abs(z)>=3 AND abs(NW-t)>=3 both horizons |
| DB lock crashes (Jul-17, Jul-25) | FIXED | timeout=30 + iCloud relocation | WAL deliberately withheld until post-migration verify |
| 10-Q extraction | OPEN | none | sole BUILD-PRIORITY; test accession 0001652044-26-000066 |
| Intraday session H/L | OPEN | none | blocks range analysis; external substitutes class-mismatched |
| GOOG next-earnings date 3-way conflict | OPEN | none | monitor 2026-11-04 (UW placeholder, inf) vs external Oct 27-28; update on announcement |
| Peer-relative DORMANT root-cause | OPEN | none | 5 clean runs; clean does not equal fixed |
| UW news 404 / Massive session_reset | COSMETIC | none | vendor-plan / log-only |
| Cohort peer-residual factor | RESEARCH | none | REV3 candidate; standard gate (null control, HLZ t>3) before any sink |
| Aggressor-tilt validation | BLOCKED | none | needs flow-alert persistence build first; then reuses skew gate via --signal |
