# RULE #1 — the discipline (revised 2026-05-31)

Origin: this project has repeatedly shipped models that looked validated but
weren't — the inverted direction model, the "+1.56pp" ranker claim, the +0.55
reversion artifact, a +5.6 in-sample ranker score. Every one was the same root
cause: trusting a flattering number, building validation last, and reaching for
the first idea without checking what already exists or what the research says.

## TWO THINGS COME BEFORE ANY FIX, CODE, OR RECOMMENDATION — ALWAYS, UNPROMPTED:

A. RESEARCH EXTERNAL SOURCES FIRST. Before committing to ANY approach, search
   ALL available sources — papers, books, libraries, articles, everything in the
   universe — for a potentially better solution. Never anchor on the first idea
   or only on what's in the repo. The literature has usually already solved it
   better AND already named the failure mode. This is mandatory every time, not
   when convenient.

B. AUDIT THE CODEBASE. grep, schema, gap-check, test the real path before
   touching it. Compiled != verified. Memory != codebase truth.

## THEN, ON ANY RESULT:

1. ATTACK BEFORE BELIEVE. Write the test that tries to KILL a signal before
   reporting what it scores. My own good results are the prime suspects.

2. A NUMBER IS FAKE UNTIL I CAN NAME THE TRAIN/TEST BOUNDARY. State what the
   model trained on, what it was tested on, and that they don't overlap. No
   nameable split -> the number does not exist.

3. TOO-GOOD = LEAK, AUTOMATICALLY. Cross-sectional 1-5d equity Sharpe > ~1.5 is
   a leak until proven otherwise. Hunt the leak; do not report the number.

4. VALIDATION IS BUILT FIRST, NOT BOLTED ON. The gate (purged walk-forward +
   embargo + retrain-per-fold + net-of-cost + per-regime) exists before the
   model is trusted. Not validated = not done = a draft.

## ALWAYS:

5. ONE CONTINUOUS ME. No "past session," no "predates me." Every line with my
   fingerprints is mine to own and fix. No blame-shifting across sessions.

6. UNPROMPTED. I run this entire gauntlet without being told. If I have to be
   reminded to research or audit, I already failed the rule.
