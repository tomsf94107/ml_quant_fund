# HANDOFF — System Audit & Repair

**Session: 2026-08-30.** Repo `ml_quant_fund`, branch `research-track`, HEAD
`b5edd340`, pushed. Warning-system suite: **90 tests green**.

Companion document: `warning/HANDOFF_CRASH_WARNING_2026-08-28.md` covers the
crash early-warning build. This one covers the audit of the retraining model and
the repairs that came out of it.

**Starting question:** "accuracy has dropped significantly — are we collecting
and using all the data a quant system should?"

**Answer:** the accuracy question was unanswerable as posed, because the metric
being read could not measure anything. Five defects were found along the way,
two of them serious. Data coverage was never the binding problem.

---

## 1. The metric could not measure anything

`accuracy_cache` is per-ticker with `window_days=90`, giving **median n = 21**.

At n=21 a true 50% model's 95% interval is **[28.3%, 67.6%]** — a spread of 39
points. Every reading in that table was inside its own noise band. The observed
"drop" (0.359 to 0.634 across three horizons of the same ticker on the same day)
was random variation, not degradation.

Architecture behind it: **416 per-ticker models per horizon**, ~100 features,
n≈50 samples each, refit **daily**. p exceeds n by roughly 2×.

---

## 2. What the model actually does, measured properly

`pooled_accuracy.py` and `economic_value.py` (both read-only, stdlib only, both
unit-tested against known-truth cases before use).

**Daily model — statistically real ordering, no economic value.**

| h | n | AUC | 95% CI | shuffle null |
|---|---|---|---|---|
| 1 | 28,277 | 0.5155 | [0.5088, 0.5222] | 0.5000 |
| 3 | 26,417 | 0.5261 | [0.5191, 0.5330] | 0.5001 |
| 5 | 25,584 | 0.5247 | [0.5176, 0.5318] | 0.4997 |

Nulls clean, so the joins are sound. Calibration deciles are **monotone** —
realized rises 0.493 → 0.563 across the range — but predicted spans 0.313 →
0.685. The model is roughly **5× overdispersed**: far more confident than the
data supports. That matters if probabilities ever gate position size.

Converted to basis points (winsorized 1/99 within date, Newey-West t):

| h | spread gross | NW t | net 10bps | net 20bps |
|---|---|---|---|---|
| 1 | +0.0123% | **+0.08** | negative | negative |
| 3 | +0.3889% | **+1.12** | +0.19% | negative |
| 5 | +0.0946% | **+0.17** | negative | negative |

h=3 is the only cell with anything and it does not clear any reasonable bar. For
scale, the project's own momentum gauntlet cleared **t = +3.19 at 10bps**.

**Intraday model — significantly inverted, with sign instability.**

AUC 0.4668 / 0.4539 / 0.4605, all CIs below 0.5, calibration cleanly inverted.
Against base rates (48.7% / 47.4% / 47.3%), the DOWN leg is **−6.1 SE** at h=1.

**Do not invert it.** June reads 0.578 / 0.549 / 0.546 — above 0.5 while every
other month is below. That is the same sign instability that closed the daily
SELL signal (n=1,234, 51.7%, flipped). A leg reliably wrong except when it isn't
is not an edge.

Also: **no directional output since June.** Daily is 100% HOLD
(`ML_QUANT_DISABLE_BUY=1`); intraday is 100% NEUTRAL (the 0.35/0.65 gate). These
are post-mortem measurements, not live signals.

---

## 3. Defects found and fixed

**T0.2 — intraday reconciler dead for months.** `hist.index.tz_convert(ET)`
needs a tz-aware index; yfinance returned aware, **Massive returns naive UTC**.
Since the migration it raised `TypeError` on every row with data, and
`except Exception: pass` ate all of it. Timezone verified empirically rather than
assumed: AAPL 1-minute bars span 08:00–23:59 with volume peaks at 13:30 and
19:59–20:00, exactly the 09:30 and 15:59–16:00 ET boundaries.

Also fixed in the same function: no download cache (3 identical API calls per
ticker-day), oldest-first ordering (the run spent its time on unrecoverable
rows), commit only after the loop (a Ctrl-C discarded everything), no progress
output. Now on cron twice per session.

**Result: 14,459 of 15,092 rows recovered. `intraday_outcomes` 7,466 → 22,525.**
Only 633 lost to aged-out bars — far better than the "unrecoverable backlog" this
was first diagnosed as.

**T0.1 — a failed generator wrote `prob_up = 0.0`.** CBRS and SPCX sit in
`tickers_watchlist.txt` and are predicted daily but have **zero rows** in
`feature_importance_history` — never trained. `generate_signals` raised "No saved
model for CBRS horizon=1d", returned a placeholder 0.0 with `error=str(e)`
attached, and `log_prediction_to_db` discarded the error. 60 rows recorded a
missing model as a maximum-confidence DOWN call. Error now threaded through; the
row is skipped and logged. 60 rows deleted.

**Not defects, correctly identified as such:**
- `fear_greed` frozen at 0.5 — a *deliberate* constant, documented in-code
  2026-05-21 ("dropped from model, no historical source").
- `ff_factors_daily` 93 days stale — orphaned table, one consumer
  (`momentum_18yr_test.py`, a completed study), no live writer.
- Momentum shadow outcomes 30 days stale — 20-day forward horizon, August
  simply not due.

---

## 4. Data-quality repairs

**BYND 1-for-30 reverse split unadjusted.** `auto_adjust=True` was already being
passed and Massive did not apply it. 9 rows read +2950% to +3458%.
`prices.db.splits_cache` had the record all along; the writer never consulted it.
Fixed with an explicit lookup — arithmetic verified against the real case
(+2908% unadjusted → +0.28% adjusted).

**1,128 ticker-reuse rows deleted.** Predictions dated before the instrument
existed under that symbol: AI (C3.ai listed 2020-12-09), META ("META" was Meta
Materials until mid-2022), S ("S" was Sprint), FIG (Figma IPO'd 2025-07-31). Not
repairable — there is no correct price for a company that had not listed.

**Guard added, then corrected.** The first bound (300%) would have discarded
GME's January 2021 squeeze (+788%), AMC (+570%), BNED (+495%), SMMT, QURE, QUBT
— all real. For a fund whose one validated brick is a short-interest signal,
dropping squeeze outcomes biases the record against precisely the events the
strategy targets. Raised to 1000% and paired with a listing-date check, which
catches reuse that no return bound can.

Two smaller fixes in the same writer: flat closes were being discarded
(`actual_ret == 0.0: continue`), biasing the base rate upward; and per-row
failures were swallowed.

**Effect on the measurement: none.** After all cleaning, h=3 moved from +0.3879%
to +0.3889% and NW-t stayed at 1.12. The data was dirty and the answer is the
same clean — which removes "the contamination was hiding it" as a hypothesis.

---

## 5. Decision rule D, agreed and live

Monitoring with a graduation path and a circuit breaker, **no auto-retire**.

**Primary metric:** winsorized long-short decile spread at h=3, Newey-West t,
lag 2. **Baseline locked 2026-08-30:** NW-t 1.12, AUC 0.5261, spread +0.3889%
gross / +0.1889% net of 10bps, 103 rebalance dates, positive on 58.3% of dates.

| Trigger | Threshold |
|---|---|
| **Graduate** | NW-t ≥ 2.5 on ≥250 dates, sign stable, positive net of 10bps → walk-forward test (*not* live) |
| **Circuit break** | sign flips negative with \|t\| > 1.5 → immediate flag, requires a ruling |
| **Review** | ≥250 dates without graduating → numbers laid out, operator decides |
| **Auto-retire** | none |

`track_model_eval.py` appends to `model_eval_history` monthly (cron: 1st, 08:00
VN). Never overwrites — the path of the metric is the point, and an overwritten
metric is how `accuracy_cache` became meaningless.

**Honest arithmetic, recorded so nobody is surprised later:** at the current
+0.39% effect size, 250 dates gives t ≈ 1.8 and 400 gives t ≈ 2.2. Neither
reaches 2.5. **Graduation requires the effect to grow, not just the sample.**
Threshold is 2.5 rather than 2.0 because a metric checked monthly is a
multiple-comparisons problem.

---

## 6. The pattern worth remembering

Five defects, one shape: **a failure rendered as a plausible value.**

| Where | Failure rendered as |
|---|---|
| `reconcile_intraday_outcomes` | `except Exception: pass` |
| `daily_driver.save_state` | missing `con.commit()` — state silently discarded |
| `predict_today` | error path returns `signal: "HOLD"` |
| `generate_signals` → DB | failed model writes `prob_up = 0.0` |
| outcomes writer | `except Exception: continue` |

Each hid a real defect for months. None produced an error, a log line, or a
visibly wrong number. They were found only by checking the thing *downstream* of
the code, never by reading the code itself.

Every fix in this session includes making the failure visible, not merely
correcting it. That is worth a review convention: **a handler that neither
raises nor logs is a defect regardless of what it is handling.**

---

## 7. Open items

**Blocked on a file only the operator can fetch:**

| Needs | Unlocks |
|---|---|
| `FRED_API_KEY` (free signup) | ALFRED vintages for 4 revisable series → S3, honest S15 |
| Shiller `ie_data` (shillerdata.com) | S13, S2's pre-2016 equity leg, **and D12's test** |
| FINRA margin xlsx | S10 → L4D |
| Ritter PDFs (`--only ritter`, VPN on) | S11 |

**Decisions still open:** D4 (S1's 2000 registry cell contradicts its own
formula), D8 (S2 pre-2023 UNDECIDABLE), **D12** (z-scored signals self-neutralize
in a sustained crisis — S4 reads G in Nov-2008 with TED at 221bp). D12 is
blocked on Shiller: its falsifiable question needs SPX history to define a target.

**Unresolved, low priority:** momentum shadow book was killed 2026-08-25 but
wrote 14,876 August predictions. Either it is being kept for the k≥10 re-open
condition in 2027, or its cron should be disabled — a question of intent, not a
defect.

**Not investigated:** `outcomes` holds ~90–118k rows per year back to 2020 while
`predictions` starts 2026-03, so most of that table is backfill of unknown
provenance. The ticker-reuse rows were found in it; nothing rules out other
systematic errors that are simply too small to be visible as outliers.

---

## 8. Verification from cold

```
cd ~/Desktop/ML_Quant_Fund
cd warning && python -m pytest -q ; cd ..          # expect 90 passed
python pooled_accuracy.py | head -30               # AUC + mandatory shuffle null
python track_model_eval.py --history               # the recorded metric path
sqlite3 accuracy.db "SELECT COUNT(*), MAX(prediction_ts) FROM intraday_outcomes;"
```

Every evaluation script carries a shuffle null and refuses to be read without it.
An unshuffled result is not a result.
