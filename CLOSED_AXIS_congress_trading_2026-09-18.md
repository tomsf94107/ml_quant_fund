# CLOSED AXIS — congressional trading disclosures
**2026-09-18.** 27,223 disclosures, 306 members, 1,182 tickers, filings 2013-04-16
to 2026-09-16. Benchmarked against SPY over identical bars at h=60.

Eight angles tested against bars registered before the numbers were seen. Eight
nulls. This reproduces the post-2012 literature independently on the fund's own
data and closes congressional trading as a standalone signal.

---

## The headline

| | n | raw | SPY | excess |
|---|---|---|---|---|
| Buys | 11,514 | +4.16% | +4.18% | **-0.03pp** |
| Sells | 11,807 | +4.37% | +4.57% | -0.20pp |

**Congressional buys match the index to three basis points.** The +4.16% is the
number that circulates as "Congress returns 4% a trade". It is the market.

Members' own entry timing was WORSE than the filing-date entry by 0.27pp, so the
45-day disclosure lag hides no edge -- it is not that the information decays
before you see it, it is that there is nothing to decay.

## Eight angles

All at the RETURN grain, buys, matured only.

| angle | result | verdict |
|---|---|---|
| All buys | -0.02pp, n=11,528 | null |
| **Leadership** | **-0.04pp vs rank-and-file +0.18pp** | **reverses** |
| Party, ex-2020 | R +0.34pp vs D -0.40pp | 0.74pp, thin |
| Senate | +1.78pp, n=809 | positive in 4 of 7 years |
| Trade size | +0.46 / +0.20 / +0.56 / **-1.79** by band | no pattern, largest is worst |
| Filing lag | 0-30d -0.03, 31-45d -0.04, 46-90d **+1.88**, 90+d **-1.00** | adjacent bands contradict |
| Sells | -0.20pp overall, -0.07pp on lag 0-30 | not in the early filings |
| Clusters | 1-2 -0.03, 3-4 **-0.08**, 5+ +0.57 | not monotone |

**Leadership is the important one.** It is the only slice with post-2012
evidence in the literature -- alphas for the small group at the top of the
hierarchy survive even on disclosure dates (CEPR 2025) -- and here it is
NEGATIVE and below rank-and-file. A clean non-replication on 4,229 observations.

**Clusters were the most plausible survivor** and failed in the most telling
way. Every other angle asks whether a MEMBER has skill; clusters ask whether
AGREEMENT carries information, which needs nobody to trade on inside knowledge.
If it worked the bands would rise monotonically. Instead 3-4 members is worse
than 1-2, and the only positive is 219 rows across 30 tickers.

**Trade size kills the conviction hypothesis.** A member putting $500k in does
no better than one putting $5k; the largest band is the worst, on 71 rows.

## The grain trap, which cost a factor of 2.6

`congress_returns` is keyed per (ticker, filed_date, member, direction).
`congress_trades` keeps `amounts` in its primary key, so one filing split across
several line items is several rows there and ONE row in returns. A naive join
fans out about 1.4x, and unevenly -- members who file many line items per
disclosure count many times.

Measured cost: the first party split read **+2.34pp vs -0.30pp**, a 2.64pp gap.
At the return grain with DISTINCT on the join key it is **+0.74pp vs -0.28pp**,
1.02pp. Two thirds of the apparent gap was line-item weighting, and most of the
rest was 2020.

**Any query joining these two tables needs DISTINCT on (ticker, filed_at_date,
name, txn_type) or an explicit choice of grain.** Per-line-item is correct for
trade-size analysis, where each line has its own amount. Per-return is correct
for everything else.

This is the fourth appearance of the pooled-versus-weighted error in one week:
PCT7 (+1.46% pooled, -1.97% day-weighted), the h=5 long/short book, the recovery
tracker's first pass, and this.

## 2020 is the confound in every party comparison

Republican excess by year: 2017 -0.20, 2018 +2.25, 2019 -4.30, **2020 +14.89**,
2021 +0.32, 2022 +3.00, 2023 -1.94, 2024 -3.31, 2025 +3.27, 2026 -0.49.

2020 is 460 buys at +14.89pp and carries roughly two thirds of the raw gap.
Buying anything in March-April 2020 and holding 60 days produced enormous excess
returns; that is a violently mean-reverting tape, not stock selection. The other
nine years disagree on direction.

## Coverage, stated rather than implied

| | rows |
|---|---|
| Stored disclosures | 27,223 |
| No priceable ticker -- crypto, LPs, mutual funds | 787 |
| Ticker present but absent from `raw_bars` | 444 (2%) |
| Filed before SPY history begins 2016-07-18 | 398 |
| Inside the 60-bar window, cannot be scored | 1,523 |
| **Returnable** | **~25,200 (95%)** |

Congress trades bonds, private funds, crypto and foreign listings. `$BTC`,
`3G FUND VI LP` and `ABNFX` are in the ticker field and have no equity price.
Those rows stay in counts and dollar volume with NULL returns rather than being
deleted -- a $5M private-fund purchase is information even without a return.

## Data notes for anyone returning to this

- **UW has no pagination.** `page` and `offset` are accepted and SILENTLY
  IGNORED, as are `filed_at_date`, `start_date`/`end_date` and `older_than`. The
  first backfill "succeeded" 200 times and stored one window. Only `limit`
  (caps at 200), `date` (a window AROUND it) and `ticker` (that name's history,
  reaching 2013) do anything.
- **`name` is canonical, `reporter` is not.** 306 distinct names against 482
  reporter variants -- `Ro Khanna` and `Rohit Khanna` are one row under `name`
  and two under `reporter`. Group by `name`.
- **Roster** from unitedstates/congress-legislators GitHub Pages JSON; the
  `raw...main/*.json` paths 404.
- **Leadership means FULL committees only.** The membership file covers 230
  committees of which 49 are defined; the rest are subcommittees whose chairs
  carry identical title strings. Counting those flagged 360 of 539 members --
  two thirds of Congress. THOMAS ids are 4 characters for a full committee, so
  the length test separates them: 95 flagged after narrowing.
- **`oversees_sectors` is inference**, a committee-to-sector map written from
  published jurisdictions. Nobody publishes it and it is not validated.

## What the data is still good for

**Not** a source of tickers to buy. Nothing measured here supports that.

- **As a model feature.** `congress_net_shares` exists in `features/builder.py`
  but is not in `FEATURE_COLUMNS`. A -0.02pp standalone can still carry
  conditional information, and the feature path is untested.
- **Political exposure mapping** of the 422-name book -- which holdings are
  politically wired, which is a risk question rather than an alpha one.
- **Crowding.** Retail follows these filings and NANC/KRUZ trade them. The flow
  is real even where the alpha is not.

## Files
- `analysis/ingest_congress.py` -- UW ingest, daily cron 06:00 VN
- `analysis/enrich_congress.py` -- roster, committees, `congress_360` view
- `analysis/congress_returns.py` -- SPY-benchmarked returns, both date bases
- `congress_trades.db` -- `congress_trades`, `congress_members`,
  `congress_returns`, views `congress_360` and `congress_flows`
