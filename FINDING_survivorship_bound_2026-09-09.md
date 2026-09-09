# FINDING — the h=40 edge is monotone in illiquidity
**2026-09-09.** `analysis/survivorship_bound_test.py`, 3 seeds, 70 names sampled
per slice, h=40, universe `tickers_expanded.txt` (1,919 names).

This does not close an axis or open one. It reframes the universe-expansion
question, which had been treated as "add names for estimation breadth".

---

## Result

| slice | median ADV | names | cap-3 | seeds + | prob>=0.70 | NW t |
|---|---|---|---|---|---|---|
| mega | $2,139M | 100 | **+1.721pp** | 3/3 | +0.848pp | +2.86 |
| large | $500M | 300 | +1.972pp | 3/3 | +2.062pp | +3.60 |
| mid | $190M | 500 | +2.998pp | 3/3 | +2.144pp | +4.87 |
| small | $53M | 1,019 | **+4.932pp** | 3/3 | +6.961pp | +6.99 |

Monotone in both directions and in the t-stats. Small minus mega is
**+3.211pp per 40-day period, +20.2pp annualised** at ~6.3 periods a year.

## Why this matters more than it looks

**The current book IS the mega/large slice.** `tickers.txt` carries 423 names
and the S&P-500 route that built it; the expansion's 1,512 new names are mostly
mid and small. So adopting `tickers_expanded.txt` does not add breadth to the
existing return distribution -- it **moves the book into a different one**,
where the measured edge is roughly three times larger.

That is a materially bigger decision than the one that was being made, and this
test does not settle it.

## Three readings, all consistent with the data

**1. Survivorship.** Small names that failed between 2016 and 2026 are absent
from `prices.db`, and small names fail far more often than mega caps. Part of
+4.93pp is names that survived where their peers did not.

Against published estimates this gap is enormous: CRSP index 1.6pp/year,
Elton-Gruber-Blake 0.9-1.5%/year, both peer-reviewed. **+20.2pp annualised is
roughly thirteen times the upper peer-reviewed figure.** The script's own
decision rule -- "if this gap is near or below the peer-reviewed range,
survivorship is not the main worry" -- fails by an order of magnitude.

**2. A genuine small-cap ML premium.** This matches the literature closely
enough to be the leading explanation. Cakici et al. document that ML return
prediction works best for small stocks; a review of the same work finds that
confined to large caps, ML excess returns fell to a range of -3.66% to -0.36%
over the recent twenty years. Avramov, Cheng and Metzker find ML predictability
attenuates once microcaps are excluded. The shape of this table is what that
literature predicts.

**3. Both**, in unknown proportion. Most likely.

## The trap in reading 2

If the edge is real rather than survivorship, that is *not* straightforwardly
good news. The same literature finds ML predictability concentrates in
hard-to-trade segments and that profitability drops sharply once trading costs
and constraints are imposed.

At $53M median ADV the current book's sizing is not capacity-constrained -- the
h=40 deployment check found median position at 0.001% of dollar ADV -- but
spreads, impact and borrow are not what they are at $2.1B. The small slice's
+4.93pp is a gross number measured without costs.

**So both readings argue against adopting the expanded universe for TRADING**,
by different routes: reading 1 says part of the edge is not real, reading 2 says
part of it will not survive execution.

## What this does not say

The slices differ in liquidity, analyst coverage, volatility, investor base and
index membership as well as attrition. The bound is loose and confounded by
construction, and the script says so. **It cannot separate the three readings.**
Doing that needs delisted price history, which the fund does not have -- and
Shumway (1997) shows even CRSP does not fully close the hole, since correct
delisting returns are unavailable for most negatively-delisted stocks and the
omitted returns are large.

## Consequence for the expansion decision

The original case, from `analysis/universe_expand.py`, was estimation breadth:
h=40 ranged +2.14pp to +5.46pp across three 80-name draws, twenty model
configurations on 2026-09-05 all landed in the same place, and the sector
feature falls through to market-relative for 78% of the expanded universe with
`[sector]` measured at -0.515pp in leave-one-out. **That case is untouched by
this result and still stands -- for TRAINING.**

What this result removes is the assumption that the expanded universe is the
same return regime with more names in it. It is not.

**Revised position:**

- **Train on 1,920.** The estimation argument holds, the sector map already
  covers 1,885 names via `data/sector_etf_sic.csv`, and training cost is the
  cheap half. Consistent with the group model's own result: train on all, buy
  top-400 ADV, +1.601 against +1.373 pooled.
- **Do not extend the traded universe on this evidence.** The measured edge in
  the names that would be added is inflated by an unknown mix of survivorship
  and cost-sensitivity, and separating them is not possible with the data here.
- **If the traded universe is ever widened**, it needs a cost model, not just a
  liquidity screen. The literature's current position is to retain the full
  universe during training while incorporating transaction costs directly into
  the objective, reweighting toward implementable securities -- rather than
  filtering the universe up front.

## Open

Nobody has measured what the h=40 book's costs actually are at $53M ADV versus
$2.1B. Until that exists, the small slice's +4.93pp cannot be compared to the
mega slice's +1.72pp on equal terms, and the monotone pattern above is a gross
comparison across segments with materially different execution.

## Files
- `analysis/survivorship_bound_test.py` -- the test, 28 minutes on 3 seeds
- `analysis/universe_expand.py` -- the screen and the original expansion case
- `data/sector_etf_sic.csv` -- sector ETF for 1,885 names, built 2026-09-06
