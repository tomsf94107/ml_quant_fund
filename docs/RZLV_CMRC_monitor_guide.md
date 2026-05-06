# RZLV / CMRC Vote Monitor — Operations Guide

Companion to `monitor_rzlv_cmrc.py`. Read this once; refer back when a daily
run flags something you don't immediately know how to interpret.

---

## The framing (read this first)

**May 14 is not a vote on the merger.** CMRC's board has already rejected
Rezolve's offer twice and adopted a poison pill (effective Apr 14 2026,
expiring Apr 12 2027; triggers at 10% beneficial ownership / 20% for passive
institutional). May 14 is the **2026 annual meeting / director election**.
Rezolve is using the proxy fight to seat directors who would actually engage
with the deal. So the question this monitor is helping you answer is:

> Is institutional/insider/dark-pool activity tilting toward "the board's
> position is weakening" or "the board's position is holding"?

Both readings are tradeable. The activity tells you which.

---

## How far back to look

| Section | Default `--since` | Why |
|---|---|---|
| Insiders | `2026-03-01` | Captures the ~2 weeks before the Apr 8 public bid plus all post-bid Form 4 activity. Earlier than that is normal-state baseline noise. |
| Institutional 13F | n/a (latest filing only) | 13Fs are quarterly. The Q1 2026 filings deadline was May 15; expect filings Apr 30 – May 14 with positions as of Mar 31. Watch `report_date` field. |
| Dark pool | `2026-04-01` | The deal-active window starts here. Pre-April prints are ambient flow, not deal-related. |
| Options flow | `2026-04-01` | Same reasoning. CMRC's options market is thin; RZLV's is meaningful. |
| Short interest | n/a (snapshot) | FINRA reports semi-monthly; latest 6 snapshots cover ~3 months. |

If you want a deeper baseline for Form 4 activity (to compare normal-state
sell pressure vs deal-window sell pressure), run once with `--since 2025-10-01`
and store the output for comparison.

---

## Reading each section

### Insider trades (Form 4)

**Bullish for "vote pushes board" / Rezolve wins:**
- CMRC officers/directors **selling** in the deal window. Confidence-eroding.
- Wagner-affiliated entities (e.g. DBLP Sea Cow Limited) **adding to RZLV**.
  You already know the Apr 2 add. New ones reinforce the signal.
- Cluster of CMRC C-suite Form 4 sells (more than one within 30 days).

**Bearish for Rezolve / status quo holds:**
- CMRC directors or officers **buying** in the deal window. Real money on
  the standalone case.
- Insider buying clustered across multiple insiders is the single strongest
  bullish-for-defendant signal in the academic literature
  (Lakonishok–Lee 2001, Cohen et al. 2012).

**Transaction code reference:**
- `P` = open-market purchase (high-conviction signal)
- `S` = open-market sale
- `A` = grant/award (mostly compensation, low signal)
- `M` = exercise of derivative (low signal alone; pair with `S` after for net
  read)
- `F` = tax-withholding (effectively forced; low signal)

When the script prints aggregates, **strip out `A`, `M`, and `F` mentally**
before drawing conclusions. The `P` vs `S` ratio is what you care about.

### Institutional ownership (13F)

**Watch for in the table:**
- `NEW` flag on a holder you don't recognize → look up the 13F filer name.
  Activist-style names (Elliott, Starboard, Engaged Capital, Land & Buildings,
  etc.) appearing fresh on CMRC's roster is a major bullish-for-Rezolve
  signal even if no public statement.
- `CHG` flag with a large positive delta on a known activist or hedge fund
  → accumulation under-the-radar pre-vote.
- Large negative delta from a long-only mutual fund (Vanguard, BlackRock,
  Fidelity passive funds) is **not** a signal; it's mechanical
  index/passive flow.
- Active managers (Wellington, T. Rowe, Capital Group active funds) trimming
  in size **is** a signal — they don't like the standalone case either.

**Important caveat on 13Fs for this timeline:** Q1 2026 13Fs (positions as of
Mar 31) are filed by May 15. So most of the deal-window accumulation (Apr 8
public bid → May 14 vote) **will not show up in 13F until the Q2 filing
deadline of Aug 14**. By that point the vote is over. Treat 13F as
*establishing the baseline* of who was already large; treat dark pool +
short interest changes + 13D filings as the real-time signal.

**Critical addition (not in this script, by design):** the most predictive
single-source signal for this kind of fight is **Schedule 13D filings**.
Anyone crossing 5% with intent to influence files within 10 days. UW exposes
13F but not (cleanly) 13D. Set up a separate SEC EDGAR watch on:
- `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=CMRC&type=SC+13&dateb=&owner=include&count=40`
- `https://efts.sec.gov/LATEST/search-index?q=%22Commerce.com%22&forms=SC%2013D,SC%2013G,SC%2013D%2FA,SC%2013G%2FA`

A new SC 13D filer at CMRC during the deal window is the single biggest
signal you can get. The script doesn't fetch this because EDGAR scraping is
its own thing — but adding a `urllib` daily check on those URLs is a
20-minute job.

### Dark pool prints

CMRC daily volume is thin (20-day average ~570k pre-Apr-14 per the public
filings). Anything over a few hundred thousand shares is real.

**Bullish for Rezolve / accumulation in progress:**
- `BLOCK` flags (single prints ≥ $250k) clustered into specific days
- `HEAVY` flag (daily aggregate ≥ $1M) on multiple consecutive days
- Prints clustered late in the trading day (institutional footprint)
- Disproportionately above VWAP (real money lifting the offer)

**Lower-signal:**
- Single isolated prints with no clustering — likely just liquidity
  rebalancing
- Prints near the close of the rights record date (Apr 27) — partly
  mechanical re-positioning

For RZLV: dark pool flow is more diffuse; less informative for the vote
specifically, but worth watching because heavy RZLV accumulation pre-vote
implies arbs are betting on deal completion.

### Options flow

CMRC has a thin options market. Don't read too much into low-premium prints.
Real signals only appear at premium ≥ $100k per trade and require
volume-greater-than-OI confirmation.

**Bullish for Rezolve / vote wins:**
- Heavy call premium with short-dated expiry (May / June 2026)
- Out-of-the-money call buying with vol-greater-than-OI
- Put/call premium ratio < 0.5

**Bearish for Rezolve / status quo:**
- Heavy put premium near vote
- Put/call premium ratio > 1.5
- Risk reversals (large call selling against put buying)

For RZLV (more liquid options): the same logic applies, but a put-heavy RZLV
also signals merger-completion concern (RZLV dilutes if deal goes through;
some shareholders hedge against that even if they support the strategic
case). Don't over-read RZLV puts as a vote signal; do read RZLV calls as
deal-completion bets.

### Short interest

The FINRA short interest snapshot is semi-monthly, lagged ~7 trading days.

**Bullish for Rezolve / squeeze setup:**
- CMRC short % of float **rising** going into the vote → hedge fund
  arb-pair pressure that unwinds violently if vote passes
- Days-to-cover rising

**Bearish:**
- CMRC short interest **falling** → shorts covering, less squeeze fuel,
  market less convinced of vote outcome

For the merger arb pair specifically (long CMRC / short RZLV at the 2:1
ratio), watch the *spread* between CMRC short interest and RZLV short
interest. CMRC short interest rising while RZLV short interest stable =
classic "hedge funds long CMRC short RZLV" arb. That's a bet on completion.

### Price / volume

**Volume ≥ 2x 20d average** flags. Cross-reference with the dark pool
aggregate from the same day. If both are heavy, real institutional turnover
is happening. If only volume is heavy with no dark pool, retail flow.

---

## Daily routine

```bash
# Once: set keys (don't paste in terminal history)
export UW_API_KEY="..."
export MASSIVE_API_KEY="..."   # optional

# Daily, after Vietnam morning pipeline
cd ~/Desktop/ML_Quant_Fund
python scripts/monitor_rzlv_cmrc.py 2>&1 | tee -a logs/merger_monitor_$(date +%F).log

# Just CMRC, deeper insider lookback
python scripts/monitor_rzlv_cmrc.py --tickers CMRC --since 2025-10-01

# Just check today's dark pool, skip everything else
python scripts/monitor_rzlv_cmrc.py --skip insiders institutional options shorts ohlc
```

Cron entry (Vietnam time, after main 7AM pipeline finishes ~7:25):

```cron
30 7 * * 1-5 cd ~/Desktop/ML_Quant_Fund && /Users/atom/miniconda/envs/ml_quant_310/bin/python scripts/monitor_rzlv_cmrc.py >> logs/merger_monitor.log 2>&1
```

(Adjust the python path to wherever `ml_quant_310` lives.)

---

## What's not in here, and why

- **Schedule 13D / 13G filings.** Most predictive signal for this fight.
  UW doesn't expose these cleanly. Add an EDGAR scraper as a follow-up; in
  the meantime, set a daily browser bookmark refresh on the EDGAR search
  URLs above.
- **Proxy advisor recommendations** (ISS, Glass Lewis). These typically come
  out 7–14 days before the meeting. Their recommendations move ~10% of
  institutional votes mechanically. Set a calendar reminder for May 1–7 to
  manually check ISS/GL coverage.
- **Wagner-affiliated entity buys at RZLV** beyond what's in the standard
  insider feed. DBLP Sea Cow Limited may file under multiple CIKs depending
  on structure. If you see a new "10% owner" in the RZLV insider feed, look
  up the CIK in EDGAR to confirm affiliation.
- **Litigation docket.** Activist proxy fights almost always include a
  Delaware Chancery suit. If CMRC sues Rezolve or vice versa, that's a
  major catalyst not visible in market-data APIs. Use a free alert on
  Court Listener (`https://www.courtlistener.com/`) with party name
  "Commerce.com" or "Rezolve".

---

## Sanity checks before sizing this trade

This script gives you better information; it does not give you edge by
itself. Before you commit capital to either side of the trade:

1. The **realistic AUC of your existing ML system on small-cap event-driven
   names is unknown** — your `prediction_features` table is mostly large-cap
   AI names. Don't pretend a 0.510 generic-ticker model gives you any read
   on RZLV/CMRC. This is a discretionary trade informed by signals, not a
   model trade.
2. RZLV is a 🔴-tier name in your AI thesis playbook (high beta, binary
   event, small-cap). Position size accordingly: 0.3–1.0% of AI sleeve max,
   per your own rulebook. Don't break the rule because the trade is exciting.
3. CMRC longs are the higher-conviction side if you read the activity as
   pro-merger, because the offer (1 RZLV per 2 CMRC) plus any board
   replacement implies a takeout bid that the market is currently pricing
   at a 47% discount. Downside if vote fails is roughly the standalone
   trajectory (which is bad — 1.5% growth guide). Asymmetry is the case;
   path to realization is the question.
4. **Pre-commit your exit.** Section 9.1 of your AI playbook applies:
   tactical 🟡 tier, –25% drawdown = mandatory thesis review. If you size
   this 1% and it goes –25%, you've lost 25 bps. If you size it 5% and the
   vote breaks against you on May 14, the gap-down could be 30%+ in a
   single session. Asymmetric trades demand small sizing, not large sizing.
