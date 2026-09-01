# CORRECTION — Audit Handoff, 2026-08-30

**Written 2026-09-01.** Read alongside `HANDOFF_SYSTEM_AUDIT_2026-08-30.md`.
Two claims in that document are wrong. One was load-bearing.

---

## 1. The architecture claim was wrong by 25x

**What the handoff says:**

> "416 per-ticker models, n≈50 samples each, ~100 features, retrained DAILY.
> At n≈50 the 95% CI spans ~28pp... p exceeds n by roughly 2x."

**What is actually true:**

| | claimed | measured |
|---|---|---|
| training rows | ~50 | **1,291** (AAPL, NVDA) |
| feature rows | — | 2,158 from `TRAIN_START = 2018-01-01` |
| split | — | 60/20/20 → 1,291 train / 430 val / 430 test |
| p/n ratio | ~2:1 (pathological) | **~1:13 (ordinary)** |

**Where the error came from.** The n≈39–59 figures were read from
`accuracy_cache.n_predictions` — the count of SCORED PREDICTIONS in a rolling
90-day window — and reported as the training sample size. They measure entirely
different things.

**Why it mattered.** It was the main argument for Tier 2 option A: *"adding
features to per-ticker n≈50 models worsens p/n."* At 1,291 rows against ~100
features that argument does not hold. The recommendation may still be right;
this reason for it is not.

**What survives**, because it was measured on pooled prediction data rather than
inferred from architecture: AUC 0.5155–0.5261 with clean shuffle nulls, no
economic value at 10bps (h=3 NW-t 1.12), and the calibration defect. The
critique of `accuracy_cache` also stands — median n=21 with a ±39-point interval
is a real property OF THE METRIC, which is where those numbers belong.

**Corollary worth keeping.** Tickers below the `len(X) < 60` floor genuinely
cannot train: CBRS has 74 bars → 30 training rows → raises. That is the same
"No saved model for CBRS" error diagnosed as T0.1, so the two findings connect.

---

## 2. "No evidence of accuracy decline" was too broad

The handoff concluded pooled accuracy was flat and treated the matter as closed.
Pooled accuracy IS flat — 50.7 / 53.2 / 51.4 / 50.9 for h=5 across May–August at
n in the thousands. But the operator's question was about HIGH-CONFIDENCE
accuracy, and there the picture differs.

**Top-decile h=5 edge over each month's own base rate, fixed May-149 universe
(so universe growth cannot explain it):**

| month | top-dec | 95% interval | base | beats base? |
|---|---|---|---|---|
| 2026-05 | 64.2% | [58.1, 69.8] | 52.4% | **yes** |
| 2026-06 | 56.0% | [50.4, 61.6] | 49.1% | **yes** |
| 2026-07 | 51.8% | [46.4, 57.2] | 49.9% | no |
| 2026-08 | 51.7% | [45.4, 58.0] | 50.8% | no |

**The decline is real**: +11.7 → +6.9 → +1.9 → +0.8pp. Four months, monotone,
and July/August are indistinguishable from random selection.

**But it is regression, not breakage.** `walk_forward_history` — an independent
path, refit weekly across ~400 tickers — shows h=5 AUC of **0.5347 in May and
0.5362 on 30 August**. Flat all year. An AUC of 0.535 supports a top-decile edge
of roughly 2–4pp. It cannot support +11.7pp.

**So May was the anomaly, not August.** July and August are the model performing
at its measured capability.

**Caveat on walk-forward**, which the 08-30 handoff also over-read: it averages
per-ticker performance across years of history, so a four-month change barely
moves it. Flat walk-forward is evidence the long-run average is intact, not
proof that nothing changed recently.

---

## 3. Defects found on 2026-09-01, all fixed

| | |
|---|---|
| **`accuracy_cache` frozen 13 days** | Its ONLY writer was `ui/pages/2_Accuracy.py:86`, behind a Streamlit button — plus a cron that sent a NOTIFICATION reminding the operator to click it. Frozen 08-19 to 09-01, so the numbers under investigation could not move. Third instance of UI-only invocation, after the intraday reconciler and `daily_driver`. **Now croned.** |
| **Probabilities ~5x overconfident** | h=5 predicted decile spread +0.429 against a realised +0.097. A "HIGH confidence" 0.70 was a 57% event. Cause: the isotonic layer in `CalibratedClassifierCV` is fitted on `X_train` (≈2018–2023) and applied to 2026. **Fixed** — rolling shrink-to-base-rate in `analysis/calibration.py`, writing a new `prob_cal` column, PIT and non-breaking. Walk-forward ECE 0.0871 → 0.0229. |
| **Thin-history tickers anti-predictive** | h=5 Jul+Aug top-decile edge by bar count: 250–999 → **−6.6pp**, 1000–1999 → +0.5pp, 2000+ → +1.1pp. **Fixed** — names under 1,000 bars can reach MEDIUM but never HIGH. Recovers the ~4.7pp gap between the full universe (−3.9pp) and the fixed 149 (+0.8pp). |
| **Borrow rows stamped with probe time** | `log_borrow_live` discarded UW's `timestamp`, so three probes of one unchanged 20-hour-old snapshot wrote three "observations". **Fixed** — `ts_utc` is UW's clock, `probed_at` is ours. |
| **UW borrow is once-daily** | Updates 09:00–15:23 UTC = **05:00–11:23 ET**, ~13 observations/day, then nothing. Freshest PRE-MARKET, contrary to expectation. Vendor limit, not fixable — IBKR FTP blocked, iborrowdesk 403. **Now labelled** with an age warning. |

---

## 4. What has NOT been fixed

- **The Accuracy dashboard still shows per-ticker cells** with n≈40 and a
  ±30-point interval, no interval displayed. That is the mechanism that
  generated this investigation and it will do so again. Pointing the page at
  pooled figures, or printing a Wilson interval beside each cell, is the durable
  fix.
- **CYBR and EA have feed gaps** — CYBR failed on both Massive and yfinance for
  2026-05-27..06-24; EA has an empty 2026-08-05..08-17 window with
  `repair_stale_feeds.py` suggested in the log.
- **`prob_cal` is written but nothing reads it.** Deliberate: switching
  thresholds to the calibrated scale is a behaviour change and should be a
  separate decision.

---

## 5. Process note

Two patches today parsed cleanly and failed at runtime — a docstring insert
placed above the docstring, and a block referencing a variable assigned further
down. `ast.parse` proves syntax, not scope or execution order. Grep shows that a
line exists; it does not show what is in scope there or what type a variable
holds.

In the same session, `stale_h` was found never to reach the DataFrame at all
because only `fee` and `avail` were copied — caught by reading the code rather
than assuming a third time.

**Patch scripts now compile the target and revert on failure.** That still would
not catch an `UnboundLocalError`. Only running the tool does.
