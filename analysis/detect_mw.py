# Data-driven market-wide detector — classifies at the BASE-FEATURE level.
#
# Why base-level: transforms (ts_std, ts_argmax) destroy the signal needed to
# classify from the alpha panel. ts_argmax returns 0..w (a few values) whether
# or not the base is per-ticker, so nunique/ratio on the TRANSFORMED column
# fails (verified: fund_op_equity__ts_argmax__w5 looked market-wide). The
# base feature panel is where oil_ret is genuinely 1-value-per-date and
# fund_op_equity is genuinely per-ticker. Classify there, map transforms by base.
#
# market_wide: median share of names holding the modal value >= MW_SHARE
# per_ticker:  modal value held by a minority of names
# unknown:     panel too thin/empty to classify -- EXCLUDED, never admitted
# Each alpha 'base__op__w' maps to its base's class via prefix.

from analysis.build_alpha_panel import build_panels_from_tickers, load_tickers

MW_SHARE  = 0.90   # modal share above which a base cannot meaningfully rank a cross-section
MW_CONST  = 0.999  # effectively constant -> market-wide regardless of modal behaviour
MW_MOVES  = 0.50   # share of dates with a DISTINCT modal value; separates a moving
                   # market series (0.72-0.93) from a fixed sentinel (0.014)


def classify_bases(tickers=None, start_date="2024-01-01", n_sample=120):
    # DEFAULTS (2026-08-23): start_date must precede the classified window by
    # >= 252 trading days or every 52-week feature is all-NaN and reads ABSENT.
    # n_sample 40 -> 120: an 11%-coverage feature (inst_signed_flow_30d, ~47 of
    # 411 names) expects ~4.6 covered names at 40, below the floor of 5; ~13.7 at 120.
    """Build base panels for a ticker sample, classify each base feature by
    median distinct-values-across-tickers-per-date. Returns dict[base -> class]."""
    if tickers is None:
        # SAMPLE FIX (2026-08-22): was load_tickers()[:n_sample] -- the first N
        # ALPHABETICALLY. For a sparse feature that is a biased sample:
        # inst_signed_flow_30d covers ~47 of 411 names per date, so most of the
        # first 40 are NaN, distinct-count collapses, and it was classified
        # market_wide despite 46.6 distinct values/date on the full panel.
        # It carried IC +0.034, t 3.24, Sharpe +0.55, mono 0.697 and was
        # excluded from every survivor list.
        import random as _rnd
        _all = load_tickers()
        tickers = (_rnd.Random(17).sample(_all, min(n_sample, len(_all)))
                   if len(_all) > n_sample else _all)
    # training_mode=True (2026-08-23): default False made LIVE UW calls per ticker,
    # out of market hours, burning quota -- and classified panels containing
    # UW features that a training_mode build will not have.
    panels = build_panels_from_tickers(tickers, start_date, None, verbose=False,
                                       training_mode=True, include_sentiment=False)
    out = {}
    for base, p in panels.items():
        if p.shape[1] < 5:
            out[base] = "unknown"; continue
        # COVERAGE GUARD, 3rd pass (2026-08-23). History:
        #   "per_ticker"  ADMITTED absent bases -- alpha_select keeps only
        #                 == per_ticker, so a panel that failed to BUILD was
        #                 scored as a stock-picking signal (^VIX3M / ES=F).
        #   "unknown"     then EXCLUDED merely-sparse bases: the 4 inst_* family
        #                 (~47 of 411 names -> ~4.6 covered in a 40-name sample)
        #                 and low/high_52w_ratio (all-NaN when start_date leaves
        #                 no room for their 252-day warm-up).
        # DISCRIMINATOR: a market-wide feature is ONE series broadcast across the
        # cross-section, so it is non-null for EVERY ticker. Low coverage
        # therefore IMPLIES per-ticker. Only a wholly empty panel is "absent".
        _tot = p.notna().sum().sum()
        if _tot == 0:
            out[base] = "unknown"; continue      # ABSENT: nothing built
        _cov = p.notna().sum(axis=1).median()
        if _cov < 5:
            out[base] = "per_ticker"; continue   # SPARSE: broadcast is never sparse
        # METRIC FIX (2026-08-23): distinct-count replaced by MAX VALUE SHARE.
        # `med <= 2` wrongly excluded binary per-ticker flags; `med <= 1` then
        # wrongly ADMITTED market-wide bases carrying a single odd ticker.
        # Measured 2026-07-01, share of names holding the modal value:
        #   semi_etf_momentum_60d  0.998 (409/410, odd name BNY)  market-wide
        #   igv_vs_sp500_ret_30d   0.998 (409/410, odd name BNY)  market-wide
        #   ma5_above_ma20         0.593 (243/167)                per-ticker flag
        #   rsi_14                 0.002 (all distinct)           per-ticker
        # Distinct-count reads the first two as 2 and cannot separate a 243/167
        # split from a 409/1 one. Share separates them by 0.4. It is also
        # SAMPLE-ROBUST: a 40-name sample gives 1.00 or 0.975, both >= the bar.
        # Why it matters: build_books does sort_values(col).iloc[n-cut:]. On a
        # fully-tied column the "top decile" is an arbitrary fixed basket with a
        # real-looking return series.
        # SENTINEL FIX (2026-08-23): share alone flipped volume_spike -- a RARE-EVENT
        # per-ticker flag whose modal value is the sentinel 0 on ~94% of names --
        # into market_wide. A market-wide base's modal value IS the series and moves
        # every date; a sentinel-dominated flag's modal value never moves.
        # Measured on 40-ticker base panels, 141 dates from 2026-01-02:
        #   base                   share  modal_uniq/dates
        #   vix_close              1.000  0.94   market-wide (clause 1)
        #   spy_ret                1.000  1.00   market-wide (clause 1)
        #   day_of_week            1.000  0.035  market-wide (clause 1, constant)
        #   igv_vs_sp500_ret_30d   0.995  0.93   market-wide (clause 2)
        #   semi_etf_momentum_60d  0.989  0.72   market-wide (clause 2)
        #   volume_spike           0.943  0.014  PER-TICKER  -- sentinel mode
        #   ma5_above_ma20         0.618  0.014  per-ticker
        # 0.72/0.93 vs 0.014 is a 50x gap; 0.5 sits in neither cluster.
        # NOTE: must be computed on the BASE panel. `scale` divides by sum-of-abs,
        # so an identical cross-section becomes exactly 1/n on every date and the
        # modal value stops moving -- classifying on transforms inverts this test.
        _sh, _md = [], []
        for _, _r in p.iterrows():
            _vc = _r.value_counts(dropna=True)
            if not len(_vc):
                continue
            _sh.append(_vc.iloc[0] / _r.notna().sum())
            _md.append(round(float(_vc.index[0]), 10))
        if not _sh:
            out[base] = "unknown"; continue
        share = sorted(_sh)[len(_sh) // 2]
        moves = len(set(_md)) / len(_md)
        out[base] = ("market_wide"
                     if share >= MW_CONST or (share >= MW_SHARE and moves >= MW_MOVES)
                     else "per_ticker")
    return out


def alpha_base(alpha_name: str) -> str:
    """base feature = substring before first '__'."""
    return alpha_name.split("__")[0]


def classify_alpha(alpha_name: str, base_classes: dict) -> str:
    """Map a transformed alpha to its base's class. Unknown base -> "unknown".

    FIX 2026-08-23: defaulted to "per_ticker", which ADMITTED any base missing
    from the map. Paired with the guard fix above -- both halves are needed, or
    a base that never classified still reaches the gate."""
    return base_classes.get(alpha_base(alpha_name), "unknown")


def _selftest():
    bc = classify_bases(n_sample=30)
    expect = {"oil_ret": "market_wide", "dxy_ret": "market_wide",
              "fund_op_equity": "per_ticker", "fund_ni_margin": "per_ticker",
              "vix_close": "market_wide", "return_1d": "per_ticker",
              # 2026-08-23 regression guards for the share metric
              "semi_etf_momentum_60d": "market_wide",
              "igv_vs_sp500_ret_30d": "market_wide",
              "ma5_above_ma20": "per_ticker",
              "volume_spike": "per_ticker",   # sentinel-mode regression guard
              "spy_ret": "market_wide"}
    ok = True
    for base, want in expect.items():
        got = bc.get(base, "MISSING")
        flag = "OK" if got == want else "*** MISMATCH ***"
        if got != want:
            ok = False
        print(f"  {base:20s} got={got:12s} want={want:12s} {flag}")
    from collections import Counter
    print("\nbase class counts:", dict(Counter(bc.values())))
    print("total bases:", len(bc))
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _selftest()
