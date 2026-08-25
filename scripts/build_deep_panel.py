#!/usr/bin/env python3
"""Deep alpha panel build, 2017-01-03 -> 2026-08-14.

MUST be a FILE, not a heredoc: explode_panels_parallel uses multiprocessing,
macOS defaults to spawn, and each worker re-imports __main__. From stdin that
is '<stdin>' and every worker dies with FileNotFoundError.

PRE-REGISTERED (see HANDOFF section 4.1):
  whitelist   deep_whitelist.txt -- bases non-constant for every sampled ticker
              in every year, plus 4 warm-up cases; market-wide bases KEPT in the
              panel and filtered at SCORING by alpha_select's != per_ticker.
  mode        training_mode=True, include_sentiment=False (no live UW calls)
  output      data/alpha_panel_deep/ -- NEVER data/alpha_panel/
  verdicts    KILL ONLY. The 442-name universe was selected 2024-26 and
              backfilled, so the 2017-23 panel is names that would become
              interesting by 2026. A signal that fails here is dead; one that
              passes has shown nothing.
  construction long-short only. Long-only decile books are inflated by
              survivorship; LS is deflated and reads as a floor.
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from analysis.build_alpha_panel import build_alpha_panel
    wl = [l.strip() for l in open(ROOT / "deep_whitelist.txt") if l.strip()]
    out = ROOT / "data" / "alpha_panel_deep"
    assert out.name != "alpha_panel", "ABORT: would overwrite the live panel"
    print(f"# whitelist {len(wl)} bases -> ~{len(wl)*30} alphas")
    print(f"# output {out}")
    t0 = time.time()
    s = build_alpha_panel(start_date="2017-01-03", end_date="2026-08-14",
                          output_dir=out, feature_whitelist=wl,
                          training_mode=True, include_sentiment=False,
                          parallel=True, verbose=True)
    dt = time.time() - t0
    print(f"\n# {s['dates_written']} dates, {s['alphas_written']} alphas, {dt:.0f}s")
    n = len(list(out.glob("*.parquet")))
    print(f"# parquet files on disk: {n}")
    assert n == s["dates_written"], f"ABORT: {n} files vs {s['dates_written']} reported"
    return 0


if __name__ == "__main__":
    sys.exit(main())
