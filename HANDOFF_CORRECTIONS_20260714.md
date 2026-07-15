## CORRECTIONS & DARKPOOL CLUSTER CLOSE-OUT (2026-07-14, session 2)

**Handoff errors corrected:**
- darkpool_prints lives in earnings_monitor.db — NOT accuracy.db. DB_PATH (monitor_ticker.py:61) defaults to a RELATIVE path; running monitor from outside the repo silently creates a fresh empty DB. Absolute-path fix still pending.
- mechanical_block_filter is NOT imported by monitor_ticker.py. The exclusion there is an inline 3x-window-median rule (~:3190). Two coexisting exclusion rules: module (print-level, dark-pool skew) vs inline (daily volume baseline). squeeze_scan now uses the inline rule (transplanted this session).
- The prescribed "DELETE today's rows then re-insert" fix: RESCINDED. tracking_id is PRIMARY KEY; INSERT OR IGNORE dedupes. The real fix was pagination.

**UW darkpool API semantics [measured, not guessed]:**
- limit hard-caps at 500 (422 above). older_than paginates, EXCLUSIVE at second granularity — step cursor +1s, PK absorbs re-serves.
- No-date walk serves only ~2 sessions deep. Per-day date= walks BLEED into prior days once the cursor exits the day (no empty page at boundary) — day-start stop mandatory.
- date param is ET-anchored. Feed contains ~2.7% duplicate rows. Timestamps are UTC Z-strings; bucket on et_date (ET-converted), never substr(executed_at,1,10).

**Data caveats, permanent:**
- Rows with et_date < 2026-05-29 (~2,956): partial slices, outside UW's ~44-day serving window, unhealable. Never trust their skew.
- ALL dark-pool skew produced before 2026-07-14 is unreliable: frozen time-of-day slices (9% of tape), VWAP self-benchmark, UTC buckets. MSFT 7d aggregate went +22.9% -> +0.3% at full tape/NBBO-midpoint; 2 of 7 days flipped sign. Prior report claims (e.g. MSFT -29.4%) need this caveat.

**Fixed & pushed:** ad14ca25 (cursor-walk fetch, et_date, NBBO-midpoint signing, canceled/ext_hours), 7b4c49f7 (footer + etl_earnings PIT cutoff UTC->ET; NOTE: monitor_ticker.py is BROKEN at this commit — skip when bisecting), e78ea4b8 (repair). scripts/repair_darkpool_days.py = gap healer.

**monitor_ticker.py importers (all break if it doesn't compile):** squeeze_live.py, squeeze_scan.py, borrow_fetch.py, monitor shell fn (.zshrc), squeezeselect.

## MOMENTUM 18-YEAR GAUNTLET (2026-07-15) — PASSED, PROVISIONAL
Same gate that killed direction: beta-strip -> net of cost -> year-clustered t -> null control.
- Momentum-specific edge (EW strip, vs own investable universe):
    mom_12_1: +1.212%/20d net, t=+3.19, 13/17 yrs+ @10bps; survives 20bps (t=+3.13)
    mom_6_1:  +1.130%/20d net, t=+3.09, 14/18 yrs+ @10bps; FAILS 20bps by 0.01
- FF Mkt-RF strip confirms (t=3.5+ both) but INCLUDES a measured survivor-universe
  premium: null-shuffle vs FF market = +0.25%/20d (t=2.78) -- a random basket of the
  418 survivors beats the market by ~3%/yr. FF-minus-null reconciles to EW numbers.
- Null under EW: resid ~0, net = -cost x turnover to the third decimal. Harness clean.
- 2009 signature correct: crash lives in the L/S short leg (-4.15/-6.93%); long-only
  top decile held. CAVEAT: mom_12_1's net line starts 2010 (BETA_MIN warm-up eats
  2009); mom_6_1's 2009 IS included (+2.23 net).
- PASS is PROVISIONAL: survivor bias beyond the measured universe premium cannot be
  bounded on this panel. Shadow book (clock from 06-11) remains confirmatory.
- Candidate: mom_12_1 (22% turnover, robust at 20bps on both factors).
- Data: ff_factors_daily table in prices.db (Ken French daily, 1926 -> 2026-05-29,
  monthly publication lag). Flags: ML_QUANT_FACTOR=ff|ew, ML_QUANT_NULL=1.
Contrast: direction model = -0.122%/trade, t=-3.28, killed 07-14 on the same gate.
