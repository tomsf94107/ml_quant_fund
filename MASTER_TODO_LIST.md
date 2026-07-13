
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
