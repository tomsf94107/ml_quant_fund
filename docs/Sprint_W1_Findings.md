# Sprint W1 — Findings Note

*May 17 2026. ML Quant Fund. Sprint W1 was: survivorship audit + cost model + h=3/h=5 cap validation.*

All three items are done. Two produced staged-but-uncommitted code; one was a clean audit with no code change.

---

## 1. Survivorship bias — CLEAN, no code change

**Question:** does the backtest universe silently exclude delisted/removed tickers, inflating historical accuracy?

**Finding:** no. `analysis/walk_forward.py` (`load_panel_pit`) builds its universe from the `outcomes` table — `SELECT ticker, prediction_date, horizon FROM outcomes` — not from `tickers.txt`. It backtests on whatever ticker had an outcome row on a given date, regardless of whether the ticker is still in the live file today.

**Evidence:**
- All 6 Gap-doc suspect tickers (BNED, BYND, CRCL, RZLV, SENS, WCC) have outcome rows. Delisted names like BNED (Mar 16-20 only), CRCL (Mar 5-13), SENS (Mar 16-20) retain their historical rows — they are NOT purged.
- Watchlist names BYND (32 outcome rows) and RZLV (39) ARE scored in `outcomes`. The "excluded from accuracy scoring" label is a UI/accuracy-page concern only; walk-forward still trains/tests on them.
- `outcomes` has 131 distinct tickers vs 125 in `tickers.txt` — the backtest universe is broader than the live file. Dead names are retained.

**Verdict:** the verified OOS AUC 0.486 baseline is NOT survivorship-inflated on this axis. No fix needed.

---

## 2. Cost model — SHIPPED to `analysis/fitness_scorer.py`

**Problem (per Gap doc):** `fitness_scorer` used turnover only as a divisor penalty in the fitness formula; `annualized_return` was gross — no actual transaction cost was subtracted.

**Built:** a flat per-trade basis-points cost model.
- New constant `COST_PER_UNIT_TURNOVER_BPS` (default 10 bps, env override `ML_QUANT_COST_BPS`).
- `compute_group_fitness` now deducts a **per-bar** cost (`pos_change * _COST_RATE`) from `strat_ret` before computing any metric — so win_rate, vol, sharpe and annualized return are all net-of-cost, not just the headline number.
- `gross_annualized_return` is stored alongside net on `FitnessRow` and in the `fitness_scores` table — so when the cost rate is later re-tuned (or replaced by a spread-aware v2), net is re-derivable without re-running the scorer.
- 8 unit tests (`tests/test_cost_model.py`), 180 full regression — all green.

**Result — cost drag from 10 bps, per horizon:**

| Horizon | Gross ann. ret | Net ann. ret | Drag | Drag % of gross |
|---|---|---|---|---|
| h=1 | 0.607 | 0.541 | 0.067 | ~11% |
| h=3 | 0.783 | 0.765 | 0.018 | ~2% |
| h=5 | 0.725 | 0.713 | 0.011 | ~1.5% |

**Read:** the edge survives realistic friction. h=1 wears more cost (it trades more often → more turnover); h=3/h=5 are nearly cost-free. The fitness ranking is preserved (MU/INTC/HUM stay top). The Gap-doc worry that "any tiny edge gets eaten by costs" is not what the data shows for these low-turnover models.

**Caveat:** 10 bps is a liquid-equity assumption, on in-sample fitness in a bull-ish window. It is "costs don't kill THESE signals at THIS rate" — not "costs never matter." Spread-aware per-ticker cost is a deliberate v2; the gross-storage seam is in place for it.

**Status:** committed-ready. This is a `fitness_scorer.py` change, not a generator change — it does NOT affect Pipeline B/C signals, so it is safe to commit independently of the Friday constraint.

---

## 3. h=3/h=5 confidence cap — VALIDATED, verdict: REMOVE

**The cap:** `signals/generator.py` lines ~798-806 cap effective confidence at `CONFIDENCE_CAP = 0.65` for `INVERSION_HORIZONS = {3, 5}`. The code comment justifies it: *"h=3 and h=5 measured INVERTED at prob_up >= 0.70 per May 7 SHAP analysis."*

**Problem:** the May 7 analysis predates the May 4 outcomes-reconciliation fix and the May 12 validator fix. It ran on bugged labels — the same bug that made AUC look like 0.520 before correction to 0.486.

**Re-test on clean labels** (`validate_confidence_cap.py`, window Apr 15 - May 14, win rate by prob_up bucket):

| Bucket | h=1 (control) | h=3 (capped) | h=5 (capped) |
|---|---|---|---|
| mid 0.40-0.60 | 51.8% | 50.1% | 49.9% |
| vhigh >=0.70 | 64.6% | 62.0% | 55.6% |

**Finding:** there is no inversion. At every horizon, high confidence wins MORE than mid — the normal, correctly-ordered pattern. At h=3 the gap is statistically real: vhigh 62.0% (n=166) vs mid 50.1%, **non-overlapping Wilson CIs**. h=5 points the same way (55.6% vs 49.9%) but CIs overlap — directionally clear, not airtight.

**Verdict:** the May 7 inversion was a bugged-label artifact (same failure mode as the retracted OKLO blind-spot). The 0.65 cap is now actively harmful — it throttles the model's best-performing region (h=3 vhigh, 62%) down to 0.65, weakening position sizing and flattening signal ranking. Recommended change: delete the cap block, or raise `CONFIDENCE_CAP` to a value the data never reaches (~0.95) so it becomes a dead safety rail.

**Status:** NOT changed yet. This is a `signals/generator.py` change and must not ship before Friday's Pipeline B institutional-features baseline.

---

## Generator changes now queued for one post-Friday commit

There are TWO staged/decided `signals/generator.py` changes, both held to avoid confounding Friday's Pipeline B read:

1. **Task D fitness gate** — per-horizon thresholds `{1:1.0, 3:3.0, 5:3.0}`, code applied + 12 tests passing, backup at `signals/generator.py.taskd.bak`.
2. **Cap removal** (this note, item 3) — decided, not yet coded.

Recommendation: ship both together as a single "generator cleanup" commit after Friday's baseline is captured. Atom's calendar event (~Sat May 23) covers the Task D commit; fold the cap removal into the same commit.

The cost model (item 2) is a `fitness_scorer.py` change with no Pipeline B/C effect — it can be committed independently, any time.

---

## W1 in one line

Two old "findings" died this sprint — survivorship bias (never existed) and the h3/h5 inversion (bugged-label artifact). One real tool was built — a transaction cost model showing the weak edge survives 10 bps friction. The honest baseline is intact and slightly better understood.
