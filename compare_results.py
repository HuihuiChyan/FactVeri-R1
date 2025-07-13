#!/usr/bin/env python3
"""
Script to compare results between search-enabled and baseline fact-checking modes.
"""

import json
import argparse
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score


def load_results(file_path: str) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics for a set of results."""
    y_true = []
    y_pred = []
    
    label_map = {"supported": 1, "unsupported": 0}
    
    for item in results:
        true_label = item.get("label", "").lower()
        pred_label = item.get("final_verdict", "").lower()
        
        if true_label in label_map and pred_label in label_map:
            y_true.append(label_map[true_label])
            y_pred.append(label_map[pred_label])
    
    if not y_true:
        return {"error": "No valid labels found"}
    
    return {
        "total_items": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4)
    }


def compare_individual_cases(search_results: List[Dict], baseline_results: List[Dict]) -> List[Dict]:
    """Compare individual cases between search and baseline results."""
    comparisons = []
    
    # Create lookup for baseline results
    baseline_lookup = {}
    for item in baseline_results:
        # Use response as key since that's the claim being verified
        claim = item.get("response", "")
        baseline_lookup[claim] = item
    
    for search_item in search_results:
        claim = search_item.get("response", "")
        baseline_item = baseline_lookup.get(claim)
        
        if baseline_item:
            search_verdict = search_item.get("final_verdict", "")
            baseline_verdict = baseline_item.get("final_verdict", "")
            true_label = search_item.get("label", "")
            
            comparison = {
                "claim": claim[:100] + "..." if len(claim) > 100 else claim,
                "true_label": true_label,
                "search_verdict": search_verdict,
                "baseline_verdict": baseline_verdict,
                "search_correct": search_verdict.lower() == true_label.lower(),
                "baseline_correct": baseline_verdict.lower() == true_label.lower(),
                "verdicts_match": search_verdict.lower() == baseline_verdict.lower(),
                "search_queries": len(search_item.get("search_queries_made", [])),
            }
            comparisons.append(comparison)
    
    return comparisons


def print_comparison_report(search_metrics: Dict, baseline_metrics: Dict, comparisons: List[Dict]):
    """Print a detailed comparison report."""
    print("\n" + "="*80)
    print("FACTVERI-R1 COMPARISON REPORT: SEARCH vs BASELINE")
    print("="*80)
    
    # Overall metrics comparison
    print("\n📊 OVERALL PERFORMANCE METRICS")
    print("-" * 50)
    print(f"{'Metric':<20} {'Search':<12} {'Baseline':<12} {'Difference':<12}")
    print("-" * 50)
    
    metrics_to_compare = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']
    for metric in metrics_to_compare:
        search_val = search_metrics.get(metric, 0)
        baseline_val = baseline_metrics.get(metric, 0)
        diff = search_val - baseline_val
        diff_str = f"{diff:+.4f}"
        print(f"{metric.replace('_', ' ').title():<20} {search_val:<12.4f} {baseline_val:<12.4f} {diff_str:<12}")
    
    print(f"\nTotal items: {search_metrics.get('total_items', 0)}")
    
    # Agreement analysis
    total_cases = len(comparisons)
    agreement_cases = sum(1 for c in comparisons if c['verdicts_match'])
    disagreement_cases = total_cases - agreement_cases
    
    print(f"\n🤝 AGREEMENT ANALYSIS")
    print("-" * 30)
    print(f"Total cases: {total_cases}")
    print(f"Agreement: {agreement_cases} ({agreement_cases/total_cases*100:.1f}%)")
    print(f"Disagreement: {disagreement_cases} ({disagreement_cases/total_cases*100:.1f}%)")
    
    # Performance in different scenarios
    search_only_correct = sum(1 for c in comparisons if c['search_correct'] and not c['baseline_correct'])
    baseline_only_correct = sum(1 for c in comparisons if c['baseline_correct'] and not c['search_correct'])
    both_correct = sum(1 for c in comparisons if c['search_correct'] and c['baseline_correct'])
    both_wrong = sum(1 for c in comparisons if not c['search_correct'] and not c['baseline_correct'])
    
    print(f"\n🎯 CORRECTNESS ANALYSIS")
    print("-" * 30)
    print(f"Both correct: {both_correct} ({both_correct/total_cases*100:.1f}%)")
    print(f"Both wrong: {both_wrong} ({both_wrong/total_cases*100:.1f}%)")
    print(f"Search only correct: {search_only_correct} ({search_only_correct/total_cases*100:.1f}%)")
    print(f"Baseline only correct: {baseline_only_correct} ({baseline_only_correct/total_cases*100:.1f}%)")
    
    # Search usage statistics
    search_queries_used = [c['search_queries'] for c in comparisons]
    avg_searches = sum(search_queries_used) / len(search_queries_used) if search_queries_used else 0
    max_searches = max(search_queries_used) if search_queries_used else 0
    
    print(f"\n🔍 SEARCH USAGE STATISTICS")
    print("-" * 30)
    print(f"Average searches per claim: {avg_searches:.2f}")
    print(f"Maximum searches used: {max_searches}")
    
    # Show some disagreement cases
    disagreement_examples = [c for c in comparisons if not c['verdicts_match']][:5]
    if disagreement_examples:
        print(f"\n❌ SAMPLE DISAGREEMENT CASES")
        print("-" * 50)
        for i, case in enumerate(disagreement_examples, 1):
            print(f"\n{i}. Claim: {case['claim']}")
            print(f"   True Label: {case['true_label']}")
            print(f"   Search: {case['search_verdict']} ({'✓' if case['search_correct'] else '✗'})")
            print(f"   Baseline: {case['baseline_verdict']} ({'✓' if case['baseline_correct'] else '✗'})")
            print(f"   Searches used: {case['search_queries']}")


def main():
    parser = argparse.ArgumentParser(description="Compare search-enabled vs baseline fact-checking results")
    parser.add_argument("--search_results", required=True, help="Path to search-enabled results JSONL file")
    parser.add_argument("--baseline_results", required=True, help="Path to baseline results JSONL file")
    parser.add_argument("--output", help="Path to save detailed comparison JSON (optional)")
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    search_results = load_results(args.search_results)
    baseline_results = load_results(args.baseline_results)
    
    print(f"Loaded {len(search_results)} search results and {len(baseline_results)} baseline results")
    
    # Calculate metrics
    search_metrics = calculate_metrics(search_results)
    baseline_metrics = calculate_metrics(baseline_results)
    
    # Compare individual cases
    comparisons = compare_individual_cases(search_results, baseline_results)
    
    # Print report
    print_comparison_report(search_metrics, baseline_metrics, comparisons)
    
    # Save detailed comparison if requested
    if args.output:
        detailed_comparison = {
            "search_metrics": search_metrics,
            "baseline_metrics": baseline_metrics,
            "individual_comparisons": comparisons,
            "summary": {
                "total_cases": len(comparisons),
                "agreement_rate": sum(1 for c in comparisons if c['verdicts_match']) / len(comparisons),
                "search_advantage": sum(1 for c in comparisons if c['search_correct'] and not c['baseline_correct']),
                "baseline_advantage": sum(1 for c in comparisons if c['baseline_correct'] and not c['search_correct'])
            }
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(detailed_comparison, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
