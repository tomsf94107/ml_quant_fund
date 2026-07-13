
## DIRECTION MODEL -- KILLED 2026-07-13, ALL HORIZONS, ON RETURNS
Clean panel (post-1b9b2d35 PIT fix), purged WF, 382 tickers, 213,507 OOS rows/horizon,
OOS 2024-02-02 -> 2026-05-07 (one regime).
- h=1: hit edge t=1.85, fails monotonicity. Dead.
- h=3: hit edge +0.95pp t=+4.04 -> gross +0.053% -> beta-stripped +0.019%
       -> NET 10bps: -0.081% t=-3.50, 1/8 folds. Dead.
- h=5: hit edge +1.16pp t=+3.44 -> gross +0.099% -> beta-stripped +0.044%
       -> NET 10bps: -0.056% t=-1.24, 3/8 folds. Dead.
Sector cut: tech-only WORSE (beta-stripped -0.046%). Recency cut: best-ever 90d window
still net negative. A hit-rate edge with no payoff magnitude.
LESSON: walk_forward_history stores hit rates; a hit rate cannot tell you if a signal
makes money. EVERY future signal goes through analysis/wf_returns_test.py first.
TOOLING: analysis/wf_returns_dump.py + analysis/wf_returns_test.py (parity-checked).

## MOMENTUM -- PASSED THE 18-YEAR MONEY GATE 2026-07-13 (PROVISIONAL: survivor-only data)
Same validated definition (signals/momentum_signal.py), daily_prices 2008->2026,
20td hold, top decile EW, beta-stripped vs EW universe, 10bps x measured turnover,
t clustered by year.
- mom_6_1 : NET +1.130%/rebal, t=+3.09, 14/18 yrs+. 2022+ sub: +1.917% t=+4.17.
- mom_12_1: NET +1.212%/rebal, t=+3.19, 13/17 yrs+. 2022+ sub: +2.472% t=+3.33.
- 2009 crash visible in L/S (-4.2%/-6.9%); long-only top-decile dodges it. L/S fails.
- Weak years = textbook momentum-hostile (2011/2016/2021). 2009-10 resid = beta
  artifact (12-obs expanding window), trust L/S there.
- CAVEATS: survivor-only universe (pass = provisional); 2022+ splice div-adjustment.
- DECISION: 12-week shadow CONTINUES as the promotion gate, per pre-committed table.
  Direction died at +0.04% net; momentum clears at +1.2% net. 30x scale difference.
Evidence: reports/momentum_18yr_mom_6_1.csv, reports/momentum_18yr_mom_12_1.csv
