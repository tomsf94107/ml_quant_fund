#!/usr/bin/env python3
"""
scripts/ab_polygon_test.py
─────────────────────────────────────────────────────────────────────────────
Standalone A/B test: current system vs Polygon-enhanced system.

Run A = current model predictions (no Polygon)
Run B = same predictions + dark pool + 25-delta IV skew + absorption multipliers

Usage:
    python scripts/ab_polygon_test.py --save A    # save current predictions as Run A
    python scripts/ab_polygon_test.py --save B    # save Polygon-enhanced as Run B
    python scripts/ab_polygon_test.py --compare   # compare accuracy after market close
    python scripts/ab_polygon_test.py --status    # show current snapshot status

Zero changes to existing pipeline. Standalone script only.
─────────────────────────────────────────────────────────────────────────────
"""
import json
import sys
import sqlite3
import shutil
from pathlib import Path
from datetime import date, datetime

ROOT      = Path("/Users/atomnguyen/Desktop/ML_Quant_Fund")
CACHE     = ROOT / "data" / "signals_cache.json"
CACHE_A   = ROOT / "data" / "ab_polygon_A.json"
CACHE_B   = ROOT / "data" / "ab_polygon_B.json"
DB_PATH   = ROOT / "accuracy.db"
META_PATH = ROOT / "tickers_metadata.csv"


def _load_cache(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_metadata() -> dict:
    try:
        import pandas as pd
        if META_PATH.exists():
            df = pd.read_csv(META_PATH)
            return df.set_index("ticker").to_dict("index")
    except Exception:
        pass
    return {}


def save_snapshot_a():
    """Save current cache as Run A (no Polygon)."""
    if not CACHE.exists():
        print("No cache found — run runfund first")
        return
    shutil.copy(CACHE, CACHE_A)
    d = _load_cache(CACHE_A)
    print(f"Saved Run A (no Polygon): {d.get('generated_at')} — {len(d.get('signals', []))} signals")


def save_snapshot_b():
    """
    Save Polygon-enhanced predictions as Run B.
    Loads current cache and applies Polygon multipliers to prob_eff.
    Falls back gracefully if Polygon returns 403.
    """
    if not CACHE.exists():
        print("No cache found — run runfund first")
        return

    sys.path.insert(0, str(ROOT))

    d = _load_cache(CACHE)
    signals = d.get("signals", [])

    print(f"Enhancing {len(signals)} signals with Polygon data...")
    print("(If Polygon returns 403, multipliers default to 1.0 — upgrade to Starter+ to activate)\n")

    enhanced = []
    polygon_active = {"dark_pool": False, "options": False, "absorption": False}

    for s in signals:
        ticker = s.get("ticker", "")
        s_new  = dict(s)
        prob_eff = s.get("prob_eff", s.get("prob", 0.5))

        # ── Dark pool multiplier ─────────────────────────────────────────────
        dp_mult = 1.0
        try:
            from features.dark_pool import get_dark_pool_ratio, dark_pool_to_multiplier
            dp = get_dark_pool_ratio(ticker)
            if dp.get("error") is None:
                dp_mult = dark_pool_to_multiplier(dp["dp_ratio"])
                polygon_active["dark_pool"] = True
                s_new["dp_ratio"] = dp["dp_ratio"]
                s_new["dp_signal"] = dp["dp_signal"]
        except Exception:
            pass

        # ── Options skew multiplier ──────────────────────────────────────────
        opts_mult = 1.0
        try:
            from features.options_flow import get_25delta_skew
            opts = get_25delta_skew(ticker)
            if opts.get("error") is None and opts.get("skew_25d") is not None:
                skew = opts["skew_25d"]
                opts_mult = 0.96 if skew > 0.03 else 1.04 if skew < -0.02 else 1.0
                polygon_active["options"] = True
                s_new["skew_25d"]        = opts.get("skew_25d")
                s_new["iv_rank"]         = opts.get("iv_rank")
                s_new["skew_25d_signal"] = opts.get("skew_signal")
        except Exception:
            pass

        # ── Absorption multiplier ────────────────────────────────────────────
        abs_mult = 1.0
        try:
            from features.absorption import get_absorption_signal, absorption_to_multiplier
            abs_sig = get_absorption_signal(ticker)
            if abs_sig.get("error") is None:
                abs_mult = absorption_to_multiplier(
                    abs_sig["abs_signal"],
                    abs_sig.get("win_rate_30m")
                )
                polygon_active["absorption"] = True
                s_new["abs_signal"]    = abs_sig["abs_signal"]
                s_new["abs_count"]     = abs_sig["abs_count"]
                s_new["win_rate_30m"]  = abs_sig.get("win_rate_30m")
        except Exception:
            pass

        # ── Apply combined multiplier ────────────────────────────────────────
        combined_mult = dp_mult * opts_mult * abs_mult
        new_prob_eff  = round(min(max(prob_eff * combined_mult, 0.0), 0.95), 4)

        s_new["prob_eff"]          = new_prob_eff
        s_new["polygon_dp_mult"]   = round(dp_mult, 4)
        s_new["polygon_opts_mult"] = round(opts_mult, 4)
        s_new["polygon_abs_mult"]  = round(abs_mult, 4)
        s_new["polygon_combined"]  = round(combined_mult, 4)
        s_new["signal"]            = "BUY" if new_prob_eff >= 0.55 else "HOLD"

        enhanced.append(s_new)

    # Save Run B
    b_data = {
        "generated_at": datetime.now().isoformat(),
        "date":         d.get("date"),
        "run":          "B_polygon_enhanced",
        "polygon_active": polygon_active,
        "signals":      enhanced,
    }
    with open(CACHE_B, "w") as f:
        json.dump(b_data, f, indent=2)

    print(f"Saved Run B (Polygon-enhanced): {len(enhanced)} signals")
    print(f"Polygon active: dark_pool={polygon_active['dark_pool']} options={polygon_active['options']} absorption={polygon_active['absorption']}")
    if not any(polygon_active.values()):
        print("\n⚠️  All Polygon endpoints returned 403 — upgrade to Stocks Starter + Options add-on to activate")
        print("   Run B saved with multipliers=1.0 (identical to Run A until upgraded)")


def compare():
    """Compare Run A vs Run B accuracy after market close."""
    if not CACHE_A.exists() or not CACHE_B.exists():
        print("Missing snapshots — run --save A and --save B first")
        return

    da = _load_cache(CACHE_A)
    db = _load_cache(CACHE_B)

    sigs_a = {s["ticker"]: s for s in da.get("signals", []) if s.get("horizon") == 1}
    sigs_b = {s["ticker"]: s for s in db.get("signals", []) if s.get("horizon") == 1}

    date_str = da.get("date")

    print(f"\n{'='*60}")
    print(f"  A/B Polygon Enhancement Test")
    print(f"  Run A: {da.get('generated_at')} (no Polygon)")
    print(f"  Run B: {db.get('generated_at')} (Polygon-enhanced)")
    print(f"  Date:  {date_str}")
    print(f"{'='*60}")

    # Signal changes A → B
    changed = []
    for t in sigs_a:
        if t in sigs_b:
            pa = sigs_a[t].get("prob_eff", 0.5)
            pb = sigs_b[t].get("prob_eff", 0.5)
            if abs(pa - pb) > 0.005:
                changed.append((t, pa, pb, sigs_b[t].get("polygon_combined", 1.0)))

    print(f"\n  Signals changed by Polygon: {len(changed)}/{len(sigs_a)}")
    if changed:
        print(f"\n  Top changes:")
        for t, pa, pb, mult in sorted(changed, key=lambda x: abs(x[2]-x[1]), reverse=True)[:15]:
            direction = "↑" if pb > pa else "↓"
            print(f"    {t:6s}: {pa:.3f} → {pb:.3f} {direction} (mult={mult:.3f})")

    # Score both vs outcomes
    try:
        conn = sqlite3.connect(DB_PATH)

        def score(sigs):
            correct = total = 0
            for t, s in sigs.items():
                row = conn.execute(
                    "SELECT actual_return FROM outcomes WHERE ticker=? AND prediction_date=? AND horizon=1",
                    (t, date_str)
                ).fetchone()
                if row:
                    pred_up   = s.get("prob_eff", 0.5) > 0.5
                    actual_up = row[0] > 0
                    correct  += int(pred_up == actual_up)
                    total    += 1
            return correct, total

        def score_buy_only(sigs):
            correct = total = 0
            for t, s in sigs.items():
                if s.get("signal") != "BUY":
                    continue
                row = conn.execute(
                    "SELECT actual_return FROM outcomes WHERE ticker=? AND prediction_date=? AND horizon=1",
                    (t, date_str)
                ).fetchone()
                if row:
                    actual_up = row[0] > 0
                    correct  += int(actual_up)
                    total    += 1
            return correct, total

        ca, ta = score(sigs_a)
        cb, tb = score(sigs_b)
        ca_buy, ta_buy = score_buy_only(sigs_a)
        cb_buy, tb_buy = score_buy_only(sigs_b)
        conn.close()

        if ta > 0 and tb > 0:
            print(f"\n  Overall directional accuracy:")
            print(f"  Run A (no Polygon):     {ca}/{ta} = {100*ca/ta:.1f}%")
            print(f"  Run B (Polygon):        {cb}/{tb} = {100*cb/tb:.1f}%")
            diff = (cb/tb - ca/ta) * 100
            winner = "B better ✅" if diff > 0.5 else "A better" if diff < -0.5 else "No meaningful difference"
            print(f"  Polygon impact:         {diff:+.1f}% — {winner}")

        if ta_buy > 0 and tb_buy > 0:
            print(f"\n  BUY signal accuracy:")
            print(f"  Run A BUY accuracy:     {ca_buy}/{ta_buy} = {100*ca_buy/ta_buy:.1f}%")
            print(f"  Run B BUY accuracy:     {cb_buy}/{tb_buy} = {100*cb_buy/tb_buy:.1f}%")
            diff_buy = (cb_buy/tb_buy - ca_buy/ta_buy) * 100
            print(f"  Polygon BUY impact:     {diff_buy:+.1f}%")

        print(f"\n  {'='*58}")
        if not any(db.get("polygon_active", {}).values()):
            print("  ⚠️  Polygon was NOT active (all 403) — A and B are identical")
            print("  Upgrade to Polygon Stocks Starter + Options to see real impact")

    except Exception as e:
        print(f"\n  Outcomes not yet available for {date_str} — run after market close")
        print(f"  Error: {e}")

    print()


def status():
    """Show current snapshot status."""
    meta = _load_metadata()
    print(f"\n{'='*50}")
    print(f"  A/B Polygon Test Status")
    print(f"{'='*50}")
    print(f"  Cache A exists:  {CACHE_A.exists()}")
    print(f"  Cache B exists:  {CACHE_B.exists()}")

    if CACHE_A.exists():
        da = _load_cache(CACHE_A)
        print(f"  Run A date:      {da.get('date')} @ {da.get('generated_at', '')[:19]}")
        print(f"  Run A signals:   {len(da.get('signals', []))}")

    if CACHE_B.exists():
        db_data = _load_cache(CACHE_B)
        print(f"  Run B date:      {db_data.get('date')} @ {db_data.get('generated_at', '')[:19]}")
        print(f"  Run B signals:   {len(db_data.get('signals', []))}")
        pa = db_data.get("polygon_active", {})
        print(f"  Polygon active:  dark_pool={pa.get('dark_pool')} options={pa.get('options')} absorption={pa.get('absorption')}")

    if not CACHE_A.exists() or not CACHE_B.exists():
        print(f"\n  Next steps:")
        print(f"  1. python scripts/ab_polygon_test.py --save A")
        print(f"  2. [upgrade Polygon if not done]")
        print(f"  3. python scripts/ab_polygon_test.py --save B")
        print(f"  4. [after market close] python scripts/ab_polygon_test.py --compare")
    print()


if __name__ == "__main__":
    if "--save" in sys.argv:
        label = sys.argv[sys.argv.index("--save") + 1].upper()
        if label == "A":
            save_snapshot_a()
        elif label == "B":
            save_snapshot_b()
        else:
            print("Usage: --save A or --save B")
    elif "--compare" in sys.argv:
        compare()
    elif "--status" in sys.argv:
        status()
    else:
        print("Usage:")
        print("  python scripts/ab_polygon_test.py --save A      (before Polygon)")
        print("  python scripts/ab_polygon_test.py --save B      (after Polygon upgrade)")
        print("  python scripts/ab_polygon_test.py --compare     (after market close)")
        print("  python scripts/ab_polygon_test.py --status      (check snapshot status)")
