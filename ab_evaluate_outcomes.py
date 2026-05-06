#!/usr/bin/env python3
"""
ab_evaluate_outcomes.py

Compare AB test runs (A vs B) against ACTUAL realized outcomes from accuracy.db.

Tells you whether the enhancement (sentiment / polygon / etc.) actually
HELPED or HURT prediction accuracy.

Usage:
  python3 ab_evaluate_outcomes.py --test sentiment
  python3 ab_evaluate_outcomes.py --test polygon
  python3 ab_evaluate_outcomes.py --test sentiment --horizon 1
  python3 ab_evaluate_outcomes.py --test sentiment --threshold 0.55
"""
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path("/Users/atomnguyen/Desktop/ML_Quant_Fund")
DB_PATH = ROOT / "accuracy.db"


def load_run(test: str, label: str) -> dict:
    """Load Run A or Run B JSON snapshot."""
    if test == "sentiment":
        path = ROOT / "data" / f"ab_cache_{label}.json"
    elif test == "polygon":
        path = ROOT / "data" / f"ab_polygon_{label}.json"
    else:
        raise ValueError(f"Unknown test: {test}")
    
    if not path.exists():
        raise FileNotFoundError(f"Snapshot missing: {path}")
    
    with open(path) as f:
        return json.load(f)


def get_signal_prob(sig: dict) -> float:
    """Extract probability from signal dict (handles different formats)."""
    return sig.get("prob_eff") or sig.get("prob") or sig.get("prob_up", 0.5)


def evaluate_run(run_data: dict, horizon: int = None, threshold: float = 0.55) -> dict:
    """Evaluate a run against actual outcomes from DB."""
    signals = run_data.get("signals", [])
    if horizon is not None:
        signals = [s for s in signals if s.get("horizon") == horizon]
    
    conn = sqlite3.connect(DB_PATH)
    
    results = {
        "total": len(signals),
        "with_outcome": 0,
        "predicted_up_correct": 0,
        "predicted_up_wrong": 0,
        "predicted_down_correct": 0,
        "predicted_down_wrong": 0,
        "buy_signals": 0,
        "buy_correct": 0,
        "buy_wrong": 0,
        "buy_avg_return": [],
        "no_outcome_yet": 0,
    }
    
    pred_date = run_data.get("generated_at", "")[:10]  # YYYY-MM-DD
    
    for sig in signals:
        ticker = sig.get("ticker")
        h = sig.get("horizon")
        prob = get_signal_prob(sig)
        is_buy = sig.get("signal") == "BUY"
        
        # Look up actual outcome
        row = conn.execute("""
            SELECT actual_up, actual_return FROM outcomes
            WHERE ticker = ? AND horizon = ? AND prediction_date = ?
        """, (ticker, h, pred_date)).fetchone()
        
        if row is None:
            results["no_outcome_yet"] += 1
            continue
        
        results["with_outcome"] += 1
        actual_up, actual_return = row
        
        # General accuracy: predicted up if prob >= 0.5
        predicted_up = prob >= 0.5
        if predicted_up and actual_up:
            results["predicted_up_correct"] += 1
        elif predicted_up and not actual_up:
            results["predicted_up_wrong"] += 1
        elif not predicted_up and not actual_up:
            results["predicted_down_correct"] += 1
        else:
            results["predicted_down_wrong"] += 1
        
        # BUY-specific accuracy (high-conviction signals)
        if is_buy:
            results["buy_signals"] += 1
            if actual_up:
                results["buy_correct"] += 1
            else:
                results["buy_wrong"] += 1
            results["buy_avg_return"].append(actual_return)
    
    conn.close()
    
    # Compute summary metrics
    if results["with_outcome"] > 0:
        results["overall_accuracy"] = (
            results["predicted_up_correct"] + results["predicted_down_correct"]
        ) / results["with_outcome"]
    else:
        results["overall_accuracy"] = None
    
    if results["buy_signals"] > 0:
        results["buy_accuracy"] = results["buy_correct"] / results["buy_signals"]
        results["buy_avg_return_pct"] = sum(results["buy_avg_return"]) / len(results["buy_avg_return"]) * 100
    else:
        results["buy_accuracy"] = None
        results["buy_avg_return_pct"] = None
    
    return results


def print_results(label: str, results: dict, horizon: int = None):
    """Pretty print evaluation results."""
    h_str = f" (h={horizon}d)" if horizon else ""
    print(f"\nRun {label}{h_str}")
    print(f"  Total signals: {results['total']}")
    print(f"  With outcomes: {results['with_outcome']}")
    print(f"  Awaiting outcome: {results['no_outcome_yet']}")
    
    if results["with_outcome"] == 0:
        print("  (No outcomes available yet — run after market close)")
        return
    
    print(f"  Overall accuracy: {results['overall_accuracy']:.1%}")
    
    if results["buy_signals"] > 0:
        print(f"  BUY signals: {results['buy_signals']}")
        print(f"  BUY accuracy: {results['buy_accuracy']:.1%}")
        print(f"  BUY avg return: {results['buy_avg_return_pct']:+.2f}%")
    else:
        print(f"  BUY signals: 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["sentiment", "polygon"], required=True,
                        help="Which AB test to evaluate")
    parser.add_argument("--horizon", type=int, choices=[1, 3, 5], default=None,
                        help="Filter to a specific horizon (default: all)")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  AB Test Evaluation: {args.test.upper()}")
    print(f"{'='*60}")
    
    try:
        run_a = load_run(args.test, "A")
        run_b = load_run(args.test, "B")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    results_a = evaluate_run(run_a, horizon=args.horizon)
    results_b = evaluate_run(run_b, horizon=args.horizon)
    
    print_results("A", results_a, args.horizon)
    print_results("B", results_b, args.horizon)
    
    # Verdict
    if results_a["with_outcome"] > 0 and results_b["with_outcome"] > 0:
        print(f"\n{'='*60}")
        print("VERDICT")
        print(f"{'='*60}")
        
        a_acc = results_a["overall_accuracy"]
        b_acc = results_b["overall_accuracy"]
        diff = (b_acc - a_acc) * 100
        
        if abs(diff) < 1:
            verdict = "NEUTRAL — no meaningful difference"
        elif diff > 0:
            verdict = f"B BETTER by {diff:+.1f} percentage points"
        else:
            verdict = f"B WORSE by {diff:+.1f} percentage points"
        
        print(f"Overall accuracy: {verdict}")
        
        if results_a["buy_signals"] > 0 and results_b["buy_signals"] > 0:
            a_buy = results_a["buy_accuracy"]
            b_buy = results_b["buy_accuracy"]
            buy_diff = (b_buy - a_buy) * 100
            
            a_ret = results_a["buy_avg_return_pct"]
            b_ret = results_b["buy_avg_return_pct"]
            ret_diff = b_ret - a_ret
            
            print(f"\nBUY-only:")
            print(f"  Run A: {a_buy:.1%} accuracy, {a_ret:+.2f}% avg return")
            print(f"  Run B: {b_buy:.1%} accuracy, {b_ret:+.2f}% avg return")
            print(f"  Diff:  accuracy {buy_diff:+.1f}pp, return {ret_diff:+.2f}pp")
            
            if abs(buy_diff) < 2:
                print("  Verdict: NEUTRAL on BUY signals")
            elif buy_diff > 0:
                print(f"  Verdict: B WINS on BUY accuracy")
            else:
                print(f"  Verdict: A WINS on BUY accuracy")
    
    print()


if __name__ == "__main__":
    main()
