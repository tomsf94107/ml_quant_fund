# Older Files — Reconciliation Note

A short audit of the five earlier files against the sector-accuracy work you've now done. TL;DR: most are fine as-is. Only `fitness_scorer.py` and `BRAIN_replication.md` could use small touch-ups, and only `fitness_scorer.py` is actually included in this drop because the others are markdown.

| File | Status | Why |
|---|---|---|
| `alpha_transformations.py` | ✅ **No changes needed** | Operator library is independent of sector work. The `cs_demean_group()` function already accepts a `group_map` dict — feed it from `tickers_metadata.csv` when you use it. |
| `Finding_Alphas_Summary.md` | ✅ **No changes needed** | Documentation. Still maps cleanly to your stack. |
| `neutralizer.py` | ✅ **Already updated** (v2) | The version you have already reads from `tickers_metadata.csv` via the `bucket` column. No action. |
| `fitness_scorer.py` | 🔧 **Optional update included here** | Now supports `--by-sector` flag for per-sector fitness leaderboard. |
| `BRAIN_replication.md` | 📝 **Cosmetic only** | A few places mention `data/sectors.csv` — this is now unified to `tickers_metadata.csv`. Functionally identical. |

---

## What the `--by-sector` flag adds to `fitness_scorer.py`

The existing scorer ranks 303 (ticker, horizon) models by fitness. The new flag adds a parallel mode that rolls those up to (sector, horizon) — weighted by `n_obs` so high-volume tickers don't get drowned out by low-sample ones.

```bash
# Per-ticker leaderboard (existing behaviour, unchanged)
python -m analysis.fitness_scorer --db accuracy.db

# Per-sector leaderboard (NEW)
python -m analysis.fitness_scorer --db accuracy.db --by-sector --csv fit_sec.csv

# Persist to accuracy.db for the dashboard to read
python -m analysis.fitness_scorer --db accuracy.db --by-sector --write-table
# → creates table `fitness_scores_sector` with columns
#   bucket, horizon, n_tickers, total_n_obs, win_rate, annualized_return,
#   annualized_vol, sharpe, turnover, fitness
```

The sector fitness leaderboard is a **different lens** than the sector accuracy view in the new Explorer tab:
- Accuracy explorer answers "which sectors have BUYs that go up?"
- Fitness scorer answers "which sectors have *trade-able* edge after turnover and Sharpe penalties?"

A sector can have 60% accuracy but terrible fitness if the BUYs are flipping daily (high turnover) or the wins are small while the losses are large. Both views matter; they tell you different things.

---

## How everything ties together now

```
                  ┌─────────────────────────────────────┐
                  │       tickers_metadata.csv          │
                  │  (single source of truth: bucket)   │
                  └────────────┬────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────────┐
       │                       │                           │
       ▼                       ▼                           ▼
┌─────────────┐      ┌──────────────────┐      ┌──────────────────────┐
│ neutralizer │      │ sector_accuracy  │      │ fitness_scorer       │
│  (portfolio │      │      _v2         │      │  --by-sector         │
│  weights)   │      │ (Wilson + shrink)│      │ (per-sector fitness) │
└─────────────┘      └────────┬─────────┘      └──────────┬───────────┘
                              │                           │
                              ▼                           ▼
                    ┌──────────────────────────────────────────┐
                    │  ui/2_Accuracy.py — unified, 4 tabs:     │
                    │  Overview | By ticker | Calibration |    │
                    │  Explorer (sector + Wilson + shrinkage)  │
                    └──────────────────────────────────────────┘
```

`tickers_metadata.csv` is now read by 4 places: neutralizer, sector_accuracy_v2, fitness_scorer (via `--by-sector`), and the Explorer tab in 2_Accuracy.py. Don't fork it. One file, one schema:

```csv
ticker,bucket
NVDA,Core Silicon
MU,Memory
GLD,Commodities
...
```

Optional column `industry` for sub-industry neutralization in `neutralizer.py mode='subindustry'`. Optional column `active` (0/1) if you ever want to filter dead tickers in one place.

---

## What's in this drop

1. **`2_Accuracy.py`** — the unified dashboard. Replaces your existing 574-line page. Adds a 4th tab (Explorer) without touching any of your existing UI. ~1040 lines total.
2. **`fitness_scorer.py`** — same as before plus the `--by-sector` flag. Existing behaviour unchanged.

Files NOT in this drop because they don't need changes: `alpha_transformations.py`, `neutralizer.py` (v2 already), `Finding_Alphas_Summary.md`. Keep what you have.

---

## What I'd actually do this Saturday

1. Drop in the new `2_Accuracy.py`. Smoke-test all 4 tabs work. Existing 3 tabs should look identical to today.
2. Open the Explorer tab. **Run the AUC sanity check first** (5 clicks): set `prob_up range` to [0.80, 1.00], group by Horizon. If high-confidence BUYs aren't 65%+ accurate, the AUC 0.967 is overfit and everything else is downstream of fixing that.
3. If high-confidence BUYs *are* 65%+ accurate: group by Sector, sort by `Edge?` checked → those are the sectors where a multiplier might help. If 0 sectors flip `Edge?` to true, you don't have enough data yet — keep collecting.
4. Optional: `python -m analysis.fitness_scorer --by-sector --write-table`. Compare the fitness ranking to the accuracy ranking. They will disagree on at least a few sectors. The disagreements are the most informative rows in either table.

Skip the YAML blocklist (Option A) entirely. The Explorer tab gives you everything Option C would, with proper statistics. If after 30 days the pattern holds out-of-sample, ship a shrunk-accuracy multiplier (refined Option B), not a hard blocklist.
