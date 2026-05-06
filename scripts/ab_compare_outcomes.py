#!/usr/bin/env python3
"""
scripts/ab_compare_outcomes.py

Compare Run A (sentiment OFF) vs Run B (sentiment ON) using actual outcomes
from accuracy.db. Run AFTER outcomes have been reconciled (~3 AM VN next day).

Usage:
  python scripts/ab_compare_outcomes.py                # use yesterday's date
  python scripts/ab_compare_outcomes.py --date 2026-04-30  # specific date
  python scripts/ab_compare_outcomes.py --horizon 1    # single horizon
"""
from __future__ import annotations
import sys, os, json, sqlite3, argparse
from pathlib import Path
from datetime import datetime, date, timedelta

ROOT = Path("/Users/atomnguyen/Desktop/ML_Quant_Fund")
ACCURACY_DB = ROOT / "accuracy.db"


def load_snapshot(label: str, date_str: str) -> dict | None:
    """Load Run A or B snapshot for given date."""
    date_compact = date_str.replace("-", "")
    path = ROOT / "data" / f"ab_cache_{label}_{date_compact}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_outcomes(date_str: str, horizon: int) -> dict:
    """Get actual outcomes for predictions made on date_str at given horizon."""
    if not ACCURACY_DB.exists():
        return {}

    conn = sqlite3.connect(ACCURACY_DB)
    try:
        rows = conn.execute("""
            SELECT ticker, prediction_date, horizon_days, signal_pred, prob,
                   actual_return, hit_target, evaluated
            FROM predictions
            WHERE prediction_date = ?
              AND horizon_days = ?
              AND evaluated = 1
        """, (date_str, horizon)).fetchall()
    except sqlite3.Error as e:
        print(f"DB error: {e}")
        return {}
    finally:
        conn.close()

    return {row[0]: {
        'signal': row[3],
        'prob': row[4],
        'return': row[5],
        'hit': row[6],
    } for row in rows}


def evaluate(snapshot: dict, outcomes: dict, horizon: int) -> dict:
    """For a given snapshot's predictions at horizon, compare to outcomes."""
    if not snapshot:
        return {'n': 0}
    sigs = [s for s in snapshot.get('signals', []) if s.get('horizon') == horizon]

    n_total = 0
    n_buy = 0
    n_buy_correct = 0
    sum_return_buy = 0.0
    sum_return_all = 0.0

    for s in sigs:
        ticker = s['ticker']
        signal = s.get('signal', 'HOLD')
        if ticker not in outcomes:
            continue

        out = outcomes[ticker]
        n_total += 1
        sum_return_all += out['return'] or 0

        if signal == 'BUY':
            n_buy += 1
            if out['hit'] == 1 or (out['return'] or 0) > 0:
                n_buy_correct += 1
            sum_return_buy += out['return'] or 0

    return {
        'n_total': n_total,
        'n_buy': n_buy,
        'buy_accuracy': n_buy_correct / n_buy if n_buy else None,
        'avg_return_all': sum_return_all / n_total if n_total else None,
        'avg_return_buy': sum_return_buy / n_buy if n_buy else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--date', default=None, help='YYYY-MM-DD (default: yesterday)')
    p.add_argument('--horizon', type=int, default=None, help='1, 3, or 5')
    args = p.parse_args()

    target_date = args.date or str(date.today() - timedelta(days=1))
    horizons = [args.horizon] if args.horizon else [1, 3, 5]

    print(f"\n{'='*70}")
    print(f"  AB Sentiment Comparison — {target_date}")
    print(f"{'='*70}")

    a = load_snapshot('A', target_date)
    b = load_snapshot('B', target_date)

    if not a or not b:
        print(f"\n  Missing snapshots for {target_date}:")
        print(f"    A (off) exists: {a is not None}")
        print(f"    B (on)  exists: {b is not None}")
        print(f"\n  Run scripts/ab_daily_workflow.py after Pipeline C completes")
        return

    print(f"\n  Run A (sentiment OFF): {len(a.get('signals',[]))} signals at {a.get('generated_at','?')[:19]}")
    print(f"  Run B (sentiment ON ): {len(b.get('signals',[]))} signals at {b.get('generated_at','?')[:19]}")

    for h in horizons:
        outcomes = get_outcomes(target_date, h)
        if not outcomes:
            print(f"\n  Horizon {h}d — no outcomes yet (or DB empty)")
            continue

        print(f"\n  ── Horizon {h}d ──  ({len(outcomes)} reconciled outcomes)")

        ea = evaluate(a, outcomes, h)
        eb = evaluate(b, outcomes, h)

        print(f"  {'metric':<22s} {'A (OFF)':<14s} {'B (ON)':<14s} {'Δ':<10s}")
        print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*10}")

        for metric, label, fmt in [
            ('n_buy',           'BUY signals',     '{:>10d}'),
            ('buy_accuracy',    'BUY accuracy',    '{:>9.1%}'),
            ('avg_return_buy',  'BUY avg return',  '{:>+9.2%}'),
            ('avg_return_all',  'All avg return',  '{:>+9.2%}'),
        ]:
            va = ea.get(metric)
            vb = eb.get(metric)
            sa = fmt.format(va) if va is not None else '   N/A   '
            sb = fmt.format(vb) if vb is not None else '   N/A   '

            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                diff = vb - va
                if metric == 'buy_accuracy':
                    sd = f'{diff*100:+.1f}pp'
                elif metric == 'n_buy':
                    sd = f'{diff:+d}'
                else:
                    sd = f'{diff*100:+.2f}pp'
            else:
                sd = '?'

            print(f"  {label:<22s} {sa:<14s} {sb:<14s} {sd:<10s}")

        print(f"\n  Verdict (h={h}d):")
        if eb.get('buy_accuracy') is not None and ea.get('buy_accuracy') is not None:
            d = eb['buy_accuracy'] - ea['buy_accuracy']
            if d > 0.03:    print(f"    ✓ Sentiment HELPS by {d*100:.1f}pp on BUY accuracy")
            elif d < -0.03: print(f"    ✗ Sentiment HURTS by {-d*100:.1f}pp on BUY accuracy")
            else:           print(f"    ~ Sentiment NEUTRAL ({d*100:+.1f}pp)")
        if eb.get('avg_return_buy') is not None and ea.get('avg_return_buy') is not None:
            d = eb['avg_return_buy'] - ea['avg_return_buy']
            if d > 0.005:   print(f"    ✓ Sentiment HELPS by {d*100:+.2f}pp on BUY avg return")
            elif d < -0.005:print(f"    ✗ Sentiment HURTS by {d*100:+.2f}pp on BUY avg return")
            else:           print(f"    ~ Sentiment NEUTRAL on BUY return ({d*100:+.2f}pp)")

    print(f"\n  Note: Single-day verdict is noisy. Track 7-10 days minimum.")
    print()


if __name__ == "__main__":
    main()
