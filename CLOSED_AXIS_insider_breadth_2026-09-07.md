# CLOSED AXIS — insider breadth (construction #7)

**2026-09-07.** Six insider constructions were closed on 2026-09-03. This is the
seventh, testing the one thing none of them measured, at the horizon none of
them used. It is also null.

---

## Why it was reopened

Two reasons, both legitimate at the time.

**Every prior test ran at h=5.** The horizon sweep on 2026-09-05 found this
feature set carries 40-day information and essentially none at 5 — XGBoost is
*below random* at h=3/5 (AUC 0.478) and 0.577 at h=40. Six nulls at the wrong
horizon are not six nulls.

**Every prior test measured DOLLARS.** Two equity-research reports on this same
data treat **breadth** as the operative signal:

> *"~$417M, unmarked 10b5-1, no new filers — a yellow flag, not red: two
> insiders, below spot (not top-ticking), no escalation"*

> *"Director-sale follow-through — a spread to MORE INSIDERS or continuation
> escalates the yellow flag; a first BUY would be the real (bullish) tell"*

Three filings from one director is a liquidity event. Three from three directors
is a view. Dollar flow cannot separate them.

## What was measured

`analysis/insider_breadth_test.py`. 365,910 Form-4 filings, 377 tickers with
both price and filing data, 2019-06 onward, ~346 sampled dates. Per-date
cross-sectional Spearman against forward returns, Newey-West, within-date
shuffle null.

| metric | h=40 IC | NW t | null |
|---|---|---|---|
| n_buyers | −0.0188 | −2.16 | −0.0025 |
| px_vs_spot | +0.0357 | +2.05 | −0.0070 |
| csuite_frac | −0.0135 | −1.95 | −0.0042 |
| seller_breadth | −0.0106 | −0.85 | +0.0013 |
| new_filers | −0.0077 | −0.83 | +0.0011 |
| n_sellers | (see log) | — | — |

**Nothing clears NW t = 3.0.** Every null is under 0.007, so the pipeline is
sound and these are real measurements of nothing.

**The two metrics the reports single out are the WEAKEST in the table.**
`new_filers` at t = −0.83 and `seller_breadth` at t = −0.85. The escalation
framing reads well in prose and does not survive as a cross-sectional feature.

**Two cells sit between 2 and 3 and should be read as noise.** This is a seventh
look at an axis closed six times, which is exactly the multiple-testing
situation Harvey, Liu & Zhu's t > 3.0 hurdle exists for. And `n_buyers` at −2.16
carries the *wrong sign* — more insider buyers predicting lower returns
contradicts the literature, which marks it as noise rather than a finding.

## PIT discipline

Joined on **`filing_date`, never `trade_date`**. The market learns when the
Form 4 is filed, not when the trade happened, and the gap is real: an NVDA row
shows trade 2026-03-20, filed 2026-03-24. Using `trade_date` would repeat the
error class that voided the PEAD work, where `report_date` was the fiscal period
end and admitted the figure 14–30 days early — measured at IC +0.2612, t = +30.

## The one thing left unexplained

Tonight's leave-one-out put the **`[insider]` group at −0.433pp with both seeds
agreeing** — dropping those five features costs money at h=40. Yet no individual
construction shows a standalone IC, across seven attempts.

That is the shape of features useful **in interaction** rather than alone: the
model finds something when insider flow is combined with other columns, which a
univariate IC test cannot detect and the leave-one-out can.

**So the honest status is: insider features stay wired, and no standalone
insider signal exists.** Those are compatible statements, and the distinction is
worth preserving — a future attempt should test interactions, not another
univariate construction.

## Constructions now closed

1. Flow across five Form-4 types
2. Trajectory — persistence, slope, dollar breadth, cumulative
3. Remaining overhang from a `sharesOwnedFollowingTransaction` backfill
4. Buying conditioned on short-interest terciles
5. Cohen-Malloy-Pomorski routine/opportunistic split
6. Model-accuracy gate
7. **Breadth — distinct sellers, distinct buyers, new filers, C-suite share,
   sale price vs spot — at h=5, h=20 and h=40**

The literature offers an explanation for the family being weak here: insider
selling is less predictive in tech, and the sell-side signal weakened after 2023
under the amended 10b5-1 rules. This is a tech-heavy universe on a post-2019
sample.
