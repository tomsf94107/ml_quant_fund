# REPORT CHECKLIST (run before delivery, every report)
1. PRICE BASE: every return computed close/close; quote appears only as labeled context.
2. BENCHMARK: named explicitly (SPY/SMH/IGV/...); SOX vs SOXX never interchanged --
   one canonical instrument per report, its return verified against prices.db.
3. SKEW DISCIPLINE: dark-pool skew appears under context, never as directional evidence,
   until a null-controlled test passes (MSFT Jul-21 report sec 8.2 is the standing finding).
4. P/C: report aggregate AND top-alert ratios, labeled; note the 100-alert window.
5. PROBABILITIES: qualitative only (most likely / secondary / tail). No numerics. House rule.
6. RANGES: scenario bands must tile each horizon -- no gaps, no overlaps.
7. STALENESS: data-as-of date in the header; if more than 1 session old, say so in the lede.
8. SCORECARD: every report scores its predecessor's calls. No exceptions.
9. LEVELS: any N-rejections claim lists the dates; each must satisfy the level's definition.
10. TAGS: fact/inf tags in chat; documents clean (unless styled a personal note).
11. TABLES: no blank cells shipped.
12. CORRECTIONS: factual errors in prior reports corrected explicitly, not silently
    (current queue: NVDA SOXX-vs-SOX lede; AMZN retrain-contamination claim -- retracted,
    display-only; PLTR five-rejections -> four).
