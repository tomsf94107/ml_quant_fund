# FINDING — Form 4 open-market purchases underperform
**2026-09-20.** 7,924 events, 322 tickers, 2019-01 to 2026-09. Benchmarked
against SPY over identical bars, entry at the close AFTER `filing_date`.

## Result

| cut | n | h=20 | h=60 | h=120 |
|---|---|---|---|---|
| All `P` buys | 7,915 | −0.10 | −1.01 | **−2.92** |
| C-suite only | 1,604 | −1.32 | −2.53 | **−4.14** |
| Cluster, 2+ insiders in 30d | 3,768 | −2.07 | −2.88 | **−7.07** |
| Notional ≥ $250k | 2,877 | +1.69 | +1.25 | +0.67 |

Nothing cleared the registered bar (excess > 1pp, ≥55% positive, sign holding
in both halves).

**The cluster cut is the striking one.** Two or more different insiders buying
the same name inside 30 days was the cut most likely to work -- independent
people reaching the same conclusion, with their own money, knowing they will be
scrutinised. It is the WORST: −7.07pp at h=120 on 3,541 events, t = −8.76, and
negative in both halves (−3.63 early, −10.48 late).

**The one positive cut fails the halves test.** Notional ≥ $250k reads +1.69pp
at h=20, but +0.69 early against +3.46 late; at h=60 it reverses entirely, from
+4.10 early to −3.80 late. Same-sign is the criterion that killed the
earnings-season hypothesis the same day (+1.60pp in 2016-2020, +0.63pp after)
and it kills this too.

## What the data was, so this is not re-litigated

`insider_filings_raw`, 383,355 rows, 379 tickers, transaction-level with named
insider, title, code, shares, actual price, trade date and filing date. The
filing lag on `P` rows is about one day -- TPL filed 2026-07-17 for a 2026-07-16
trade. This is the FASTEST and most granular institutional source in the repo,
and the only one where the actor spends their own money.

`transaction_code` matters and pooling it would have inverted the test:

| code | n | what it is |
|---|---|---|
| S | 124,678 | sales, mostly scheduled 10b5-1 |
| A | 106,390 | grants -- compensation, not a purchase |
| M | 60,700 | option exercises |
| F | 59,412 | shares withheld for tax |
| G | 8,060 | gifts |
| **P** | **8,009** | **open-market purchases** |

Only `P` is someone choosing to spend their own money. 1,606 are C-suite.
Pre-2019 counts (9, 16, 38 per year) are coverage starting, not insiders not
buying, so the window begins 2019.

## The confound, stated but NOT used to rescue the result

Insiders buy falling stocks. A director buying after a drawdown is saying it is
cheap, and cheap frequently gets cheaper. SPY does not control for the fact
that these names were already declining, so a momentum- or beta-matched
benchmark could produce a different number.

**That is a legitimate separate pre-registered test, not a revision of this
one.** Adding a control after seeing a negative result, and keeping it if the
sign flips, is how a null becomes a finding through iteration. The test failed
as written and is recorded as failed.

If the momentum-adjusted version is run, it gets its own bar written before the
data is touched, and this result stands regardless of what it finds.

## Where this leaves the three institutional sources

| source | lag | attributed | tested | result |
|---|---|---|---|---|
| Form 4 `P` | ~1 day | named person | **yes** | **negative** |
| Dark pool | same day | anonymous | yes, 2026-05-17 | 4 of 8 survived, in the model |
| 13F | 45 days | manager | T1-T4 pending | -- |

The fastest and most attributable source is the one that failed. The 13F prior
was already poor -- Fama-French find aggregate active managers have zero gross
alpha and that the top 3% can expect zero -- and it runs at a 45-day lag on
quarter-end snapshots with no transaction data at all.

## Files
- `analysis/form4_test.py`
- `insider_trades.db` :: `insider_filings_raw`
