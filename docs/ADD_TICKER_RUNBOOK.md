# Add Ticker — Runbook

*ML Quant Fund · `docs/ADD_TICKER_RUNBOOK.md` · v1.0 · established 2026-08-14*

How to add a new ticker to the universe and backfill every data source, aligned
to the current pipeline setup. Worked example uses **WDAY**; substitute any symbol.

---

## 0. Rules that matter

| Rule | Why |
|---|---|
| **Data BEFORE enrollment** | A ticker in `tickers.txt` with no price data trips the stale-panel guard and logs `REFUSING` on every run. |
| **Dark pool is perishable** | UW serves ~44 days and nothing older. Unfetched days are **permanently lost**. Backfill it first, same day. |
| **`tickers.txt` is the universe** | `daily_runner.load_tickers()` reads it. `tickers_metadata.csv` enrols nothing BUT is load-bearing: its `bucket` drives `resolve_sector_etf` tier-2 AND `ETF_EXCLUDE` (since 2026-08-31). An unmapped bucket silently falls back to XLK -- not the market. |
| **Always `--dry-run` first** | Prints the exact commands and the classification before anything is written. |

### The three universe files

| File | Read by | Effect |
|---|---|---|
| `tickers.txt` | `daily_runner.load_tickers()` | **Generates predictions + accuracy scoring** |
| `tickers_watchlist.txt` | `daily_runner.load_watchlist()` | Predictions only, **excluded from accuracy** |
| `tickers_metadata.csv` | Accuracy Explorer, cohort fallback | Sector/cohort metadata only |

---

## 1. Pre-check

```bash
cd ~/ML_Quant_Fund
grep -c . tickers.txt
grep -inE "^WDAY" tickers.txt tickers_metadata.csv
head -1 tickers_metadata.csv
cut -d, -f2,3 tickers_metadata.csv | sort -u | head -20
```

- No grep hits → genuinely new.
- Last two commands show the CSV column names and the sector/cohort labels
  already in use. **Reuse an existing label** — inventing one creates a
  one-member bucket in the Accuracy Explorer grouping.

---

## 2. Dry-run

```bash
python scripts/add_ticker.py WDAY --dry-run
```

Confirm:
- `type = equity` (or pass `--type etf` / `--type adr`)
- `universe = ADD` (`EXISTS(skip)` means it's already in the CSV — add `--force`)
- The echoed commands look right

---

## 3. Seed price history **first**

`backfill_raw_bars.py` only *deepens* tickers already present in `raw_bars`; it
reports "already deep — nothing to do" for a symbol with zero rows. So a brand-new
ticker must be seeded directly. This is the exact path verified on JPM / IBM / ORCL.

```bash
PYTHONPATH=. python3 -c "
from features import massive_client as mc
from features import price_cache as pc
t = 'WDAY'
raw = mc.download(t, start='2016-01-01', end='2026-08-13', auto_adjust=False)
n = 0 if raw is None else len(raw)
if n:
    con = pc._conn(); pc._write_raw(con, t, raw); con.commit(); con.close()
print(t, n, 'rows')
"
sqlite3 prices.db "SELECT COUNT(*), MIN(d), MAX(d) FROM raw_bars WHERE ticker='WDAY';"
```

Set `end` to the last completed trading day.

### 3b. Sync raw_bars into daily_prices -- REQUIRED, easy to miss

`raw_bars` is NOT what the book reads. `si_positions_live.py` ranks on
`daily_prices.adj_close`; a ticker absent there is dropped SILENTLY (this cost a
full session on 2026-08-31 -- BE/CRDO were invisible with 271 rows each).

`sync_prices_from_rawbars.py` is incremental on a GLOBAL date watermark, so a new
ticker whose history sits below it is never synced. **NEVER** run
`--rebuild-from` with an early date: it DELETEs the range across the whole table,
and `daily_prices` (1.59M rows, back to 2008) is LARGER than `raw_bars` (1.02M,
back to 2016). A deep rebuild destroys history raw cannot reproduce. A
`--rebuild-from 2025-08-01` on 2026-08-31 wiped dividend adjustment from 52,991
rows across 300 tickers; recovered only because a pre-rebuild backup existed.

```bash
python backfill_daily_prices_tickers.py --tickers WDAY --dry-run
python backfill_daily_prices_tickers.py --tickers WDAY
```

Do NOT backfill DOW, FOX, FOXA, IR -- `raw_bars` carries predecessor entities.

**Basis warning.** Rows written before the 2026-06-29 yfinance->Polygon migration
are split AND dividend adjusted. Everything written since is SPLIT ONLY:
`massive_client.py:481` maps `auto_adjust` to Polygon's `adjusted=` param, and
Polygon's `/v2/aggs` adjustment is split-only. The comment at
`fetch_and_pead.py:100` claiming equivalent `auto_adjust=True` semantics is
WRONG. No dividend-adjusted source exists in the stack today.

**Expected:** ~2,500 rows for a mature large-cap, back to 2016-08.

**Stop if:**
- `0 rows` → symbol not served by the vendor. Verify it's live and correctly
  spelled before going further (see §8).
- `MAX(d)` is not the last trading day → stale on arrival; investigate first.

---

## 4. Add + backfill

```bash
python scripts/add_ticker.py WDAY \
  --sector "<existing sector>" \
  --cohort "<existing cohort>" \
  --dp-budget 5000
```

What runs, in order:

| Step | Script | Notes |
|---|---|---|
| 1. Dark pool | `initiate_darkpool_universe.py` | **First — perishable.** Walks all unwalked tickers within budget. |
| 2. OHLCV | `backfill_raw_bars.py` | No-op if §3 already seeded |
| 3. Short interest | `finra_short_interest.py` | **NOT automatic.** Universe = `earnings_surprises` union `tickers.txt` (union added 2026-08-31). `fetch_log` caches per QUARTER, so cached quarters skip new tickers: `sqlite3 short_interest.db "DELETE FROM fetch_log;"` then re-fetch. **VPN REQUIRED** -- FINRA is ISP-filtered like cboe.com; a blocked fetch logs EMPTY and still prints DONE with unchanged row counts. |
| 4. Monitor | `monitor_ticker.py` | Form 4 insider + peer panel + institutional |
| Options / greeks | — | **Not wired.** Shows `VERIFY-CMD` |
| Earnings | — | **Not wired.** Shows `NEEDS-BUILD` |

Also writes `tickers.txt` and `tickers_metadata.csv`.

### Variants

```bash
python scripts/add_ticker.py WDAY --type etf              # skips SI / insider / earnings
python scripts/add_ticker.py WDAY --type adr              # skips insider (20-F filers)
python scripts/add_ticker.py WDAY --watchlist --no-runner # watchlist only, no accuracy scoring
python scripts/add_ticker.py --from-file new_tickers.txt  # batch
python scripts/add_ticker.py WDAY --skip short_interest   # exclude a data type
python scripts/add_ticker.py WDAY --force                 # re-run for a ticker already in the CSV
```

Use `--watchlist --no-runner` for anything not yet fit for systematic inclusion:
recent IPOs, thin history, unresolved corporate actions, pending lockups.

---

## 5. Verify

```bash
python scripts/repair_stale_feeds.py --dry-run
sqlite3 earnings_monitor.db "SELECT COUNT(*), MAX(date(executed_at)) FROM darkpool_prints WHERE ticker='WDAY';"
sqlite3 short_interest.db  "SELECT COUNT(*), MAX(settlement_date) FROM short_interest WHERE ticker='WDAY';"
sqlite3 earnings_monitor.db "SELECT COUNT(*) FROM form4_transactions WHERE ticker='WDAY';"
python scripts/ticker_lifecycle.py --status WDAY
```

**Pass criteria**

| Check | Expected |
|---|---|
| `repair_stale_feeds --dry-run` | `stale=0` |
| `--status` | `tickers.txt (RUNNER)  PRESENT` |
| Dark pool `MAX` | last trading day |
| Short interest | **must be non-zero.** `0` does NOT self-heal -- see step 3. A `0` means the ticker cannot enter the SI book at all. |
| Form 4 | non-zero for a US domestic filer; `0` is normal for an ADR |

---

## 6. Manual leftovers

**Earnings config** — not automated:

```bash
python scripts/monitor_ticker.py WDAY | head -40
```

If it prints `No earnings date configured for WDAY`, add it to the earnings
config by hand. Note fiscal-year quirks (e.g. WDAY's FY ends January).

**Options / greeks** — `backfill_greeks.py` has no confirmed CLI; run manually
if options features are needed.

---

## 7. Confirm the next pipeline run

```bash
grep -i wday ~/Desktop/ML_Quant_Fund/logs/pipeline_B_$(TZ=America/New_York date +%Y%m%d)/03_daily_runner.log
```

Looking for predictions generated and **no** `STALE PANEL … REFUSING`.

**Thin-history caveat:** a ticker with under ~1 year of bars will have undefined
252-day features (annual vol, 12-1 momentum, seasonality). It still gets
predictions — on partially-null feature vectors. Treat those signals as
unreliable until history builds. Watchlist tier is the safer placement.

---

## 8. Instrument types

| Type | Flag | Gets | Skips |
|---|---|---|---|
| US equity | *(default)* | all | — |
| ETF | `--type etf` | price, dark pool, options | SI, insider, earnings |
| ADR | `--type adr` | price, dark pool, options, SI, earnings | insider (20-F filers) |
| Future `=F` | auto | — | excluded, off equity tape |
| Crypto `-USD` | auto | — | excluded |
| Foreign `.PA` etc. | auto | — | excluded, not on US tape |

Warrants (`…W`) and pre-revenue micro-caps: technically addable, but distorted
by dilution/issuance. Prefer watchlist.

**If the vendor returns 0 rows**, check for a corporate action before assuming a
bad symbol — this session produced all three cases:

| Case | Example | Action |
|---|---|---|
| Delisted / acquired | CYBR (PANW merger 2026-02-11) | `ticker_lifecycle.py --retire` |
| Taken private | EA (PIF buyout 2026-08-04) | `ticker_lifecycle.py --retire` |
| **Ticker renamed** | SATS → ECHO (2026-06-24) | `ticker_lifecycle.py --rename SATS:ECHO` |

Renaming preserves continuous history (same CUSIP = same security). Retiring a
renamed ticker silently drops a live constituent.

---

## 9. Quick reference

```bash
# full add
python scripts/add_ticker.py TICKER --sector "X" --cohort "Y" --dry-run
# ...seed price per §3...
python scripts/add_ticker.py TICKER --sector "X" --cohort "Y"
python scripts/repair_stale_feeds.py --dry-run

# lifecycle
python scripts/ticker_lifecycle.py --status TICKER
python scripts/ticker_lifecycle.py --rename OLD:NEW
python scripts/ticker_lifecycle.py --retire TICKER --reason "delisted: ..."

# health
python scripts/repair_stale_feeds.py --dry-run
python scripts/repair_stale_feeds.py
python scripts/dump_universe.py
```

---

## 10. Known gaps

| Gap | Impact | Status |
|---|---|---|
| Price seed not in `add_ticker` | §3 is manual | Open — fold into v2.3 |
| `backfill_greeks` interface unknown | No options data on add | Open |
| `backfill_earnings_uw_new_tickers.py` hardcoded to 30 names | No earnings on add | Open — needs `--tickers` |
| Earnings date config manual | `No earnings date configured` | Open |
| `ticker_lifecycle --retire` only logs if ticker was in the CSV | Incomplete retirement record | Open |
| No `--move-to-watchlist` mode | Moves are logged as retirements | Open |
| SI universe keyed to a data table, not the runner | New tickers silently get no SI | PARTIAL -- union patched 2026-08-31; `fetch_log` clear still manual |
| `sync_prices_from_rawbars` global watermark | New tickers never reach `daily_prices` | Open -- use `backfill_daily_prices_tickers.py` |
| `ETF_EXCLUDE` hand-maintained | Untagged funds rank into the book (AAAU, IGV, RSP found 2026-08-31) | PARTIAL -- now unions metadata-tagged ETFs |
| Ticker reuse (4th corporate-action class) | SPCX, BETR splice two entities on one symbol | Open -- see HANDOFF_2026-08-22:238 |
| No dividend-adjusted price source | `adj_close` is split-only since 2026-06-29 | Open -- needs Polygon `/v3/reference/dividends` build |

---

*Update this runbook when a gap closes. Related: `docs/DEFECT_LEDGER.md`,
`docs/SCHEDULER_INVENTORY.md`.*
