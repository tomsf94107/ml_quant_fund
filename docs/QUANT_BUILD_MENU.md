# The Complete Quant Build Menu — Every Model & Alpha Family

Full landscape of what top quant funds use. Each item: what it is, who uses it,
evidence, and what it takes to build. Nothing filtered for "feasibility" — where
it needs new infrastructure, that's flagged as a build task, not a reason to skip.

Current state for reference: XGBoost classification, 1-5d direction, 149-578 US
equities, daily bars. Tested OOS at rank-IC ~0.011 (t=1.66 on curated 149) = a
cross-sector tilt, not stock-picking. This menu is everything NOT yet tried.

================================================================================
PART A — THE MODEL LANDSCAPE  (simple -> complex; ★ = highest-priority untested)
================================================================================

A1. LINEAR & REGULARIZED  ★
  OLS / Ridge / Lasso / Elastic Net / Bayesian linear regression.
  Who: everyone, as baseline + production backbone of factor investing (AQR, DFA).
  Evidence: GKX(2020) found regularized linear COMPETITIVE on monthly equity;
    gains from complex models modest. On low-SNR data linear often generalizes
    BETTER than trees (can't memorize noise) — directly relevant to your 0.25 gap.
  ★ Why: your tree overfits (train 0.66-0.79, test ~0.50). Ridge/elastic-net on
    same features may show smaller gap + more honest OOS IC. Cheapest test, hours.
  Build: trivial, scikit-learn.

A2. FACTOR MODELS (institutional core)
  Fama-French 3/5 + Carhart momentum; statistical (PCA/ICA); Barra risk models.
  ★ IPCA — Instrumented PCA (Kelly-Pruitt-Su 2019): latent factors w/ loadings
    that are functions of firm characteristics + vary over time. Strictly stronger
    than static PCA/FF; explains cross-section far better.
  Who: AQR, DFA, BlackRock (factor); IPCA academic->practitioner.
  ★ Why: IPCA is the "right" model for cross-sectional equity; clean upgrade from
    generic classifier; targets the ranking you actually trade.
  Build: FF downloadable (Ken French). IPCA: open-source `ipca` pkg + your
    characteristic panel. ~1-2 weeks.

A3. CLASSICAL ML
  Random Forests (lower variance than boosting); GBM (XGB/LightGBM/CatBoost —
    your tool; CatBoost better w/ categorical/regime, LightGBM has native ranking);
    SVM/kNN (rarely production equity).
  Who: Two Sigma, WorldQuant use GBM heavily — but as ONE of hundreds, never sole.
  Evidence: GKX trees+NN best, but OOS R² tiny (0.33-0.40% monthly). Model isn't
    the bottleneck; target + combination are.
  Build: have it. Upgrade = LightGBM lambdarank (A4).

A4. ★★ LEARNING-TO-RANK (most important untested model for you)
  LambdaMART/LambdaRank/RankNet/listwise: directly learn to ORDER stocks
    best-to-worst; loss optimizes ranking (NDCG/rank-IC), not per-name accuracy.
  Who: increasingly standard for cross-sectional equity; natural for L/S or
    top-decile books.
  Evidence: Poh-Lim-Zohren-Roberts(2021) ~3x SHARPE vs classification/regression.
  ★ Why TOP priority: you HAVE the file (train_global_ranker.py, lambdarank) and
    NEVER evaluated it honestly (only in-sample predict). This is Stage B, skipped.
    Directly addresses "signal is in ranking not direction." Fastest high-value test.
  Build: minimal — model exists, needs honest purged-WF eval reporting rank-IC. Days.

A5. NEURAL NETS for cross-sectional returns
  MLP (GKX: shallow 3-5 layer NN was their single best performer).
  ★ Conditional Autoencoder (GKX 2021): NN generalization of IPCA — nonlinear
    factor exposures, latent factors, no-arbitrage. DOMINATES FF/PCA/IPCA on OOS
    pricing errors. State-of-the-art academic equity model.
  Neural factor / deep SDF (Chen-Pelger-Zhu): estimate SDF directly w/ deep nets.
  Who: Two Sigma, DE Shaw, academic-adjacent.
  ★ Why: CA model is the most powerful documented model for YOUR exact problem.
    CAVEAT (Avramov 2023): returns halve when microcaps/no-credit-rating excluded
    -> honest edge smaller than headline but real.
  Build: custom PyTorch/TF, reference code exists. ~3-4 wks. Needs clean char panel.

A6. SEQUENCE MODELS (temporal structure)
  RNN/LSTM/GRU; TCN (often beat LSTM, parallelizable); Transformers/Temporal Fusion
    Transformer; attention across stocks per date.
  Who: newer ML shops; mixed production record. Shine in intraday/microstructure.
  Evidence: MIXED. On daily equity direction rarely beat features+boosting/ranking.
  Build: moderate-high. RISK: more params = more overfit on low-SNR. Heavy reg only.

A7. REGIME / STATE-SPACE
  HMM (regime detection); Kalman/particle filters (dynamic beta, pairs spreads);
    Markov regime-switching (Hamilton).
  Who: macro/stat-arb use Kalman for spreads; HMM as risk router.
  Evidence: better as RISK ROUTER/GATE than return predictor. Regime-conditioning
    fragments scarce data (your plan's caution).
  Build: low-mod (hmmlearn, pykalman). Use as overlay/gate, not standalone.

A8. BAYESIAN  ★(for per-ticker problem)
  Gaussian Processes (uncertainty, expensive); hierarchical Bayesian (pool across
    stocks/sectors w/ shrinkage — handles small per-name samples); Black-Litterman.
  Who: AQR uses Bayesian shrinkage extensively.
  ★ Why: hierarchical shrinkage is the PRINCIPLED answer to "per-ticker too little
    data" — pools each stock toward sector/market, allows deviation. Could rescue
    per-ticker signal XGBoost couldn't.
  Build: moderate (PyMC, numpyro).

A9. ENSEMBLES / STACKING / META-LEARNING  ★(cheap)
  Bagging/model-averaging (reduces variance = your core problem); stacking;
    LdP sequential bootstrap (handles overlapping-label correlation — your 5d overlap).
  Who: universal. No serious shop runs a single model.
  ★ Why: variance reduction attacks overfit directly. Cheap, layers on existing.
  Build: low.

A10. GENETIC PROGRAMMING / SYMBOLIC REGRESSION
  Evolve formulaic alphas (WorldQuant 101 Alphas style) via GP.
  Who: WorldQuant explicitly.
  Evidence: real but decay-prone; overfits backtests w/o multiple-testing control
    (you HAVE the gate: HLZ t>3, PBO).
  Build: moderate (gplearn/custom). Gate is the guardrail.

A11. LLM ALPHA GENERATION (2023-2026 frontier)  ★
  Alpha-GPT, AlphaForge(AAAI'25), AlphaAgent(KDD'25), RD-Agent, QuantaAlpha('26):
    LLM generates formulaic alpha hypotheses, backtests, refines — automates the
    quant hypothesis loop. AlphaAgent adds regularization vs decay/crowding.
  Who: frontier research, some China shops; deployment unverified.
  Evidence — SKEPTICAL: self-reported, short OOS windows, prone to backtest-overfit
    they cite. QuantaAlpha('26) notes "fragile controllability" — noisy backtest
    feedback drives toward spurious correlations. Durable lesson = automated
    generation of MANY candidates fed through a STRICT GATE, not the headline returns.
  ★ Why: you have an LLM + a strict gate (PBO 0.15). LLM proposes formula -> gate
    validates -> keep survivors -> combine = modern Alpha Factory. Gate makes it honest.
  Build: moderate. Generation loop scriptable; gate exists.

================================================================================
PART B — THE ALPHA TAXONOMY  (✅=have data, 🔨=needs build; IC monthly rank-IC,
  good≈0.05 vgood≈0.10; ALL decay post-pub ~26-58% McLean-Pontiff)
================================================================================

B1. PRICE/MOMENTUM ✅
  X-sectional momentum (Jegadeesh-Titman 12-1; canonical, robust, "momentum crashes"
    Daniel-Moskowitz); TS/absolute momentum (Moskowitz-Ooi-Pedersen); 52wk-high
    (George-Hwang); residual momentum (Blitz, lower vol).
  Build: all from OHLCV; likely have variants.

B2. REVERSAL  ★(fits your data)
  Short-term reversal (Lehmann/Jegadeesh; last wk/mo losers bounce; high turnover) ✅;
    long-term reversal (DeBondt-Thaler 3-5yr) ✅; overnight/intraday reversal
    (Lou-Polk-Skouras; opposite-sign) 🔨.
  ★ Your NEGATIVE within-sector decile spread (XLE -5.43) IS a reversal signature —
    model's "winners" mean-revert. Short-term-reversal framing may fit better than momentum.

B3. VALUE 🔨
  B/M, E/P, FCF yield, EV/EBITDA, sales/price. Cheap outperform long-run; weak 2010s,
    partial revival. Needs PIT fundamentals (partial via your filings/earnings).

B4. QUALITY/PROFITABILITY 🔨
  Gross profitability (Novy-Marx 2013; among MOST robust); ROE/ROA; accruals (Sloan
    1996); F-score (Piotroski). Needs fundamentals. High-value family.

B5. SIZE/LIQUIDITY ✅
  Size (weak post-pub); Amihud illiquidity (robust); turnover; bid-ask. From OHLCV+vol.

B6. VOLATILITY/RISK ✅  ★(easier target)
  Low-vol anomaly; betting-against-beta (Frazzini-Pedersen); idio-vol (Ang).
  ★ Predicting VOLATILITY (not direction) is much easier (AUC 0.6+ achievable),
    feeds sizing/options. Different, more tractable objective.

B7. EARNINGS/EVENT  ★(durable, partial data)
  ★ PEAD (Bernard-Thomas; prices drift w/ earnings surprise for WEEKS; among most
    durable anomalies) ✅partial; earnings surprise SUE; analyst revisions
    (Chan-Jegadeesh-Lakonishok ~7.5% 6mo spread; Womack); guidance. ✅ have
    eps_surprise/days_to_earnings.
  ★ Why: event-driven, fits 1-5d, partly have data, high-Sharpe durable.

B8. OPTIONS-DERIVED ✅(UW, partly built)  ★(different axis, less crowded)
  IV level/rank ✅, IV skew level ✅, IV skew CHANGE 🔨(research: change is signal),
    put-call ✅, VRP=IV-realized 🔨(logging fixed, accruing), GEX 🔨, O-S ratio 🔨,
    IV term-structure slope 🔨.
  Forward-looking (what market PRICES IN) — orthogonal to price history. Top of
    Stage-1 new-data list.

B9. MICROSTRUCTURE/FLOW ✅(have UW, underused)
  Order-flow imbalance; dark pool ✅; short interest/days-to-cover ✅(robust);
    institutional 13F; FTDs.

B10. SENTIMENT/TEXT ✅partial(FinBERT)  ★(Lazy Prices)
  News sentiment ✅; social 🔨; ★ "Lazy Prices" (Cohen-Malloy-Nguyen 2020: firms
    that CHANGE 10-K/10-Q language YoY underperform; strong/durable) 🔨 (you have
    8-K FinBERT, NOT 10-K similarity); earnings-call tone 🔨.
  ★ Lazy Prices: strong anomaly, FREE EDGAR data, different axis. Needs 10-K
    ingestion + YoY similarity.

B11. ALTERNATIVE DATA 🔨(mostly new)
  Satellite, credit-card, web traffic, app downloads, supply-chain, patents, ESG.
  Who: Point72, Two Sigma w/ data teams + 7-fig budgets.
  Honest: lowest $/signal for personal book except free sources (patents USPTO,
    some web traffic). Mostly skip.

B12. CROSS-SECTIONAL/NETWORK 🔨
  Lead-lag/industry spillover (Hou); customer-supplier momentum (Cohen-Frazzini
    ~150bps/mo but decayed); co-mention networks (GDELT — uncrowded but multi-day
    throttled crawl, you tested).
  Uncrowded (don't scale -> funds underweight) but operationally heavy.

B13. MACRO/CROSS-ASSET ✅partial
  Term-structure slope, credit spreads (HYG/LQD; strong regime indicator), FX carry,
    commodity-as-equity-predictor. Have VIX/oil/DXY/yields.

B14. ★★ STATISTICAL ARBITRAGE (different paradigm — best reframe given your data)
  Pairs trading (Gatev-Goetzmann-Rouwenhorst; trade cointegrated spread, market-
    neutral); cointegration baskets; PCA residual reversal (Avellaneda-Lee; trade
    deviations from factor-implied price).
  Who: original Renaissance/Morgan-Stanley-Tartaglia paradigm; core at many stat-arb.
  ★ Why TOP reframe: a DIFFERENT QUESTION — convergence, not direction. Your negative
    decile spread = mean-reversion signature. PCA residual reversal is the natural
    model for "stocks deviating from factor-implied value snap back" — and it's
    MARKET-NEUTRAL, sidestepping the long-only breadth ceiling. Possibly the single
    most promising direction given tonight's findings.
  Build: moderate. statsmodels has cointegration; PCA residuals reuse your machinery.

================================================================================
PART C — HOW WEAK SIGNALS BECOME MONEY (the part retail skips — and you skipped)
================================================================================

C1. ★★ ALPHA COMBINATION (your missing Stage C)
  HRP (LdP 2016; cluster by corr, recursive bisection, no cov inversion, robust OOS);
    equal-weight (hard to beat); mean-variance (optimal-but-unstable); Bayesian
    shrinkage/Black-Litterman.
  CORE MATH — Fundamental Law: IR = IC × √Breadth × TransferCoef. A 0.011-IC signal
    combined with 20 DECORRELATED 0.011 signals has portfolio IC FAR above any single
    — noise cancels, signals stack. You proved (PBO 0.15) the gate finds real signals
    + have unused survivors + NEVER built the combiner.
  ★★ HIGHEST-VALUE UNTESTED THING IN YOUR ENTIRE SYSTEM.

C2. BET SIZING
  Fractional Kelly (1/4-1/2; full Kelly overbets mis-estimated edge); meta-labeling
    (LdP; secondary model predicts if primary is right, sizes accordingly — turns
    low-precision signal tradeable); vol-targeting.
  A 51% edge IS tradeable if sized right. Meta-labeling = Stage D, untested.

C3. PORTFOLIO CONSTRUCTION & RISK
  Sector/factor neutralization (you found it destroys YOUR edge — because your edge
    IS the sector tilt; itself diagnostic); risk-model limits; drawdown control;
    turnover penalties.

C4. TRANSACTION COSTS (non-optional)
  At 1-5d holding, spread+impact can eat the entire thin edge. Every backtest nets
    costs. You have 10bps/turnover in fitness_scorer — must be in LIVE path, not
    just research.

C5. WHAT SEPARATES WINNERS FROM LOSERS
  NOT a magic signal/model. LdP "10 Reasons ML Funds Fail": winners use industrialized
    pipeline (many signals/specialists, not lone model), proper labels (triple-barrier/
    meta-label), purged/combinatorial CV, multiple-testing control (you have), and
    COMBINE many weak decorrelated bets + size well. Losers p-hack one strategy (~20
    backtest tries "finds" false strategy at 5%).

================================================================================
BUILD ORDER (synthesis — value×evidence ÷ cost, given your data's findings)
================================================================================
1. ★ Learning-to-rank eval (A4) — model exists, never honestly tested, research #1
   remedy, days. STAGE B.
2. ★ Alpha combiner/HRP (C1) — mechanism that makes weak signals tradeable; have
   survivors, no combiner. STAGE C — highest structural value.
3. ★ Linear/ridge baseline (A1) — tests whether overfit is the problem; hours.
4. ★ Stat-arb: PCA residual reversal & pairs (B14) — DIFFERENT question fitting your
   mean-reversion signature, market-neutral, sidesteps long-only breadth ceiling.
   Possibly best reframe.
5. Meta-labeling + fractional Kelly (C2) — sizing. STAGE D.
6. Conditional Autoencoder / IPCA (A2/A5) — principled SOTA cross-sectional; bigger
   build, highest model-side ceiling.
7. PEAD + analyst revisions (B7), VRP + options surface (B8), Lazy Prices (B10) —
   new INFORMATION axes, durable evidence, partly have data.
8. Hierarchical Bayesian (A8) — principled fix for per-ticker small-sample.
9. LLM alpha-generation loop (A11) — modern Alpha Factory; have LLM + gate.
10. Ensembling/bagging (A9) — cheap variance reduction on any of the above.

META-POINT: you tested ~6 configs of ONE model on ONE target with NO combination.
This menu has dozens of untested directions. Strongest near-term: (1) rank not
classify, (2) COMBINE signals not run one, (3) reframe to stat-arb/mean-reversion
(your data points here). None need new data — only new method.

Sources: Gu-Kelly-Xiu (2020 RFS; 2021 autoencoder), Kelly-Pruitt-Su (2019 IPCA),
Lopez de Prado (Adv Financial ML; 10 Reasons ML Funds Fail), Grinold-Kahn, Kakushadze
(101 Alphas), Poh et al (2021 L2R), McLean-Pontiff (2016), Novy-Marx (2013),
Cohen-Malloy-Nguyen (2020 Lazy Prices), Frazzini-Pedersen (BAB), Avellaneda-Lee
(stat-arb), AlphaForge (AAAI'25), AlphaAgent (KDD'25), QuantaAlpha ('26). All
published anomalies decay post-publication.
