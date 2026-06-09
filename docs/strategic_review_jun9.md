# Strategic Review — Everything Built, Everything Killed, What’s Actually Next

*June 9 2026. Full audit + research synthesis. Companion to MASTER_TODO_LIST, QUANT_BUILD_MENU, Research_Report.*

-----

## PART 1 — AUDIT: the verified state (Rule #1 inventory)

### 1.1 VALIDATED + ALIVE (the entire list)

|Asset                                                                     |Status                                                                                                                                |Evidence                                 |
|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
|**Momentum** (mom_6_1 cross-sectional, 20d hold, inverse-vol)             |ONLY validated return-alpha. Shadow, first live verdict ~Jun 29                                                                       |Purged-WF net Sharpe +1.24               |
|**Vol regime classifier** (vol_prediction.py)                             |Survivor but REGIME signal only — NOT return-alpha, NOT sizing (oracle test killed vol-sizing)                                        |rank-IC +0.10, decorr from momentum +0.11|
|**Accuracy infra**                                                        |Honest after this week: PSR/n-trades Sharpe guards, zero-outcome artifact fix (97c4161), registry, calibration                        |This session                             |
|**Pre-earnings monitoring layer** (monitor_ticker.py, earnings_monitor.db)|Active discretionary layer, separate from systematic                                                                                  |In daily use                             |
|**Data assets**                                                           |UW dark pool ~1.9M rows; options flow; insider ETL (EDGAR); 8-K FinBERT; econ calendar; 10-K sections (1,960, from killed Lazy Prices)|In DBs                                   |

### 1.2 BUILT BUT NOT WIRED (Rule #1 flags — audit before any new build)

- **features/institutional_features.py — 374 lines, 8 features, NOT imported in builder.py.** Explicitly flagged as next-session work. This is 13F/institutional signal work ALREADY WRITTEN. Audit first: what do the 8 features do, what data do they read, why unwired?
- **data/etl_insider.py + etl_insider_raw.py** — insider Form 4 ETL exists (Pipeline A, cron 16:00 ET). Depth/coverage unknown → audit below.
- 10-K section data (sec_filings.db) — orphaned by Lazy Prices kill but the corpus exists.

### 1.3 KILLED — do not re-attempt (the frozen list)

Per-ticker direction (.487/.493) · global direction (.496) · lambdarank @5d (leak) · cs-demean reversion (-0.43) · PCA-residual reversion (COVID artifact, -0.92 current regime) · pairs/cointegration (-0.82, corr +0.40) · PEAD (unprofitable) · linear/ridge (same gap as XGB) · A8 overlays ×5 · vol-sizing (oracle: naive +1.44 > perfect +1.10) · **Lazy Prices (Jun 3: negative spread, ~zero IC)** · price-complement hunt (any 2nd price signal = momentum in disguise, +0.82 corr)

### 1.4 DATA-GATED (scheduled, do not build early)

- **VRP / IV-skew-change** — ~Aug 2026 (IV history accruing since ~May)
- Quality (Novy-Marx) + Value — blocked on fundamentals feed (no COGS/balance sheet) **← but see Part 3: the feed is FREE**
- Orthogonal recall axes (iv_skew/short_ratio/inst_flow) — ~Apr 2027
- h=1 salvage — ~Sep 2026
- Intraday rebuild — Jun–Jul plan, currently broken/deferred

### 1.5 THE META-FINDING (drives everything below)

**FIVE return-signal hunts killed at 149-159 tickers, all the same way: no edge at this breadth.** Fundamental Law: IR = IC × √breadth × TC. Documented anomalies (CMN thousands of firms, CMP thousands, Jegadeesh thousands) do not replicate at ~150 names. **The bottleneck is BREADTH, not signal choice or model architecture.** Every recommendation below is ranked with this as the constraint.

-----

## PART 2 — WHAT GLOBAL FUNDS RUN vs WHAT’S FEASIBLE SOLO (honest mapping)

From the Build Menu + this week’s research, what the institutional stack looks like and the solo-feasible subset:

|Institutional practice                                   |Feasible at your scale?                                      |Verdict                                 |
|---------------------------------------------------------|-------------------------------------------------------------|----------------------------------------|
|Many weak decorrelated alphas + combiner (WorldQuant)    |YES in principle — but you have 1 survivor; C1 gated on a 2nd|The plan of record                      |
|Cross-sectional ranking ML, monthly (GKX/Two Sigma)      |Marginal at 149; viable at 400-1000 names                    |After universe expansion                |
|Conditional autoencoder / IPCA (state of the art)        |NO at 149 (your IPCA overfit); viable ~400+                  |After expansion                         |
|Event-driven: insider, 13F, filings (Point72/multistrats)|YES — free EDGAR data, infra partly built                    |**Part 3, the new track**               |
|Options/vol surface signals                              |YES — UW paid, data accruing                                 |Aug (already planned)                   |
|Stat-arb / pairs / microstructure HFT                    |NO — killed (pairs) / no infra (HFT)                         |Closed                                  |
|Alt data (satellite, credit cards)                       |NO — institutional budgets                                   |Skip                                    |
|LLM alpha generation behind a strict gate                |YES — you have the gate (HLZ t>3, PBO 0.15)                  |Methods layer, after a 2nd signal exists|

-----

## PART 3 — NEW RESEARCH: the strongest unbuilt candidates (ranked)

### ★ TRACK A — UNIVERSE EXPANSION (the structural fix; highest value)

Nothing else changes the math. 149 → 400-1,000 names:

- Momentum breadth: √(400/150) ≈ +63% IR uplift on the ONE validated signal, mechanically
- Makes IPCA/autoencoder viable (your IPCA failed *because* 149×T is too small)
- Allows honest RE-TESTS of killed signals (reversal/Lazy Prices may work at 500+ names — they were killed at YOUR scale, not in general)
- **The Add-Ticker Tool is already specced (spec v2 locked, SANDBOX_SETUP.md done)** — this was already identified as the path; the strategic review confirms it’s priority #1
- Constraints to engineer around: UW 40K daily call cap, Massive plan limits, Pipeline B retrain time per ticker. The expansion is an infra project, not a research project.

### ★ TRACK B — INSIDER ALPHA: opportunistic vs routine (the strongest NEW buildable signal)

**Cohen-Malloy-Pomorski 2012 (JoF), “Decoding Inside Information”:** classify each insider as ROUTINE (trades same calendar month ≥3 consecutive years — diversification/liquidity, zero information) vs OPPORTUNISTIC (no pattern). Strip routine; what remains is information-rich: opportunistic-buys-minus-sells earned ~82bp/month value-weighted (t=2.15) and ~180bp/month equal-weighted (t=6.07). Opportunistic trades also predict firm news/events.
**Replication status (researched Jun 9):** holds up. Recent replication: “Only opportunistic insider trades lead to consistent abnormal returns, while routine trades mostly do not”; 2025 J. Banking & Finance work confirms non-routine insider signal still forecasts anomaly returns; the framework keeps extending (politically-connected insiders 2020, herding 2025).
**Why it fits you specifically:**

1. **Data is FREE + FULL-HISTORY** — SEC EDGAR Form 4, decades deep. No subscription, no accrual wait (unlike VRP).
1. **Infra partly EXISTS** — etl_insider.py / etl_insider_raw.py already on Pipeline A cron. The build is classification + signal, not ingestion from zero.
1. **Orthogonal axis** — insider behavior, not price history. The price-complement hunt proved any 2nd PRICE signal is momentum in disguise; this is structurally different information. Decorrelation from momentum is plausible (must still be tested).
1. **Monthly/event horizon** — sidesteps the 1-5d cost-drag death zone where everything else died.
   **Honest caveats (pre-registered):** CMP used thousands of firms; at 149 names insider events are sparse (maybe 30-80 opportunistic events/month across the universe — audit will tell). Same small-universe risk that killed the others. McLean-Pontiff decay applies (published 2012 → expect the 82bp haircut substantially). Gates: purged-WF, per-regime, net-of-cost, |corr vs momentum| < 0.3, and a minimum-events threshold before trusting anything.
   **Verdict: the single best NEW signal to test now. Free data, partial infra, different axis, right horizon.**

### ★ TRACK C — FUNDAMENTALS INGESTION unlocks TWO durable anomalies at once

Quality (#3) and Value (#5) are data-gated, NOT killed — blocked only on missing income-statement/balance-sheet data. **SEC EDGAR XBRL company facts API is free** (the same EDGAR you already hit for filings) and serves point-in-time fundamentals: revenues, COGS, total assets, book equity. One ingestion project (`data/etl_fundamentals_xbrl.py`) unblocks:

- **Gross profitability** (Novy-Marx 2013 — gross profit/assets, among the most durable anomalies in the literature)
- **Value composite** (B/M, E/P, FCF yield)
- Both monthly-horizon, both different-axis from momentum (quality-momentum correlation historically low/negative)
  Honest caveats: XBRL tag mapping is messy (the real work, ~60% of effort); PIT discipline essential (use filing dates not period dates — the publication-lag staircase lesson from the recession project applies directly); same breadth limits.

### TRACK D — 13F / INSTITUTIONAL (audit FIRST — code already exists)

institutional_features.py (8 features, 374 lines) is built and unwired. Before any new institutional work: read it, wire it or consciously kill it. 13F data is free (EDGAR), quarterly, with the known 45-day lag. Literature: breadth-of-ownership changes, institutional demand shifts predict returns (Chen-Hong-Stein). But 45-day-stale quarterly data on 149 names = weak prior; the existing 8 features may already cover the useful part. **Action: audit, don’t build.**

### TRACK E — already scheduled (no action now)

- **VRP ~Aug** — the planned 2nd-alpha test; data accruing correctly (verify logging monthly)
- **h=1 salvage ~Sep**; **orthogonal recall ~Apr 2027**

### TRACK F — methods layer (only AFTER a 2nd validated signal exists)

- **Momentum composite** (residual momentum, 52-wk-high, industry momentum): same family (+0.82 corr — NOT new signals) but can make the ONE momentum signal more robust. Low priority, real but small.
- **C1 combiner (HRP/equal-weight)**: still gated on ≥2 decorrelated return-alphas. If insider OR VRP OR quality validates → build immediately. This remains the highest-value gated item in the system.
- **C2 sizing**: meta-labeling / fractional Kelly on momentum book (vol-targeting specifically is killed; Kelly off win-rate is a different, untested idea).
- **LLM alpha loop** behind the HLZ/PBO gate: generate candidate formulaic alphas → strict gate → survivors. Cheap to run once the gate harness is scripted; expect ~1-in-20 survival, that’s the design.

### NOT recommended (relitigating settled kills)

More 1-5d direction work · more reversion variants · pairs · a 6th price-derived signal · intraday before the Jun-Jul rebuild · buying alt-data subscriptions before free axes are exhausted.

-----

## PART 4 — THE ORDERED PLAN (what to actually do, in sequence)

1. **AUDIT (this week, 30 min):** run the audit block below — insider data depth, institutional_features content, XBRL feasibility. Results determine whether Track B starts now or needs ingestion first.
1. **Jun 29 (fixed):** momentum live verdict + promotion gate. The system’s first live-validated signal decision.
1. **Track B — insider opportunistic/routine** (next research build, ~2-3 sessions): classify from existing Form 4 data → purged-WF validate → decorrelation gate. THE candidate 2nd alpha with no data wait.
1. **Track A — universe expansion** (parallel infra project, Add-Ticker Tool spec already locked): start with the AI-thesis additions (non-US semis via ADRs, industrials, DC REITs) toward 300-400, engineering around API caps.
1. **Track C — XBRL fundamentals ingestion** (after or parallel to B): unlocks quality+value for testing at the expanded universe.
1. **Aug:** VRP test as scheduled.
1. **On any 2nd validated decorrelated signal → C1 combiner immediately.** This is the entire point of the hunt.
1. **Re-tests at expanded universe** (only after A lands): reversal, Lazy Prices, cross-sectional ML — killed at 149, possibly alive at 500.

## PART 5 — AUDIT COMMANDS (run before starting anything)

```bash
# 1. Insider data: depth, coverage, opportunistic-classifiability
sqlite3 institutional_trades.db ".tables" 2>/dev/null
sqlite3 accuracy.db ".tables" 2>/dev/null | tr ' ' '\n' | grep -i "insider"
grep -rn "CREATE TABLE" data/etl_insider.py | head -5
# (then: row count, date range, distinct insiders per ticker — need ~3yr history per insider to classify routine)

# 2. institutional_features.py: what are the 8 features, what data do they read
head -60 features/institutional_features.py | grep -v "^import\|^from"
grep -n "def \|SELECT\|FROM " features/institutional_features.py | head -20

# 3. XBRL feasibility probe (free EDGAR companyfacts)
curl -s "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json" -H "User-Agent: research test" | head -c 500
```

*Verdict in one line: the hunt at 149 names is over — momentum won, everything else died. The next edge comes from NEW AXES (insider now, fundamentals next, VRP in Aug) and from BREADTH (universe expansion), with the combiner waiting for the first decorrelated survivor.*