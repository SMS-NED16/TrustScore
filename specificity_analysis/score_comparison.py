"""
Compare scores between baseline and perturbed datasets
"""

import json
import os
from typing import List, Dict, Any, Optional
import statistics


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def compare_scores(
    baseline_results_path: str,
    perturbed_results_path: str,
    error_type: str  # "T", "B", or "E"
) -> Dict[str, Any]:
    """
    Compare baseline and perturbed scores to measure specificity.
    
    Args:
        baseline_results_path: Path to baseline results
        perturbed_results_path: Path to perturbed results
        error_type: Type of error injected ("T", "B", or "E")
        
    Returns:
        Dictionary with comparison statistics
    """
    print("=" * 70)
    print(f"Step 4: Comparing Scores for {error_type}_perturbed")
    print("=" * 70)
    
    baseline_results = load_results(baseline_results_path)
    perturbed_results = load_results(perturbed_results_path)
    
    # Create mapping by sample_id
    baseline_map = {r["sample_id"]: r for r in baseline_results if "error" not in r}
    perturbed_map = {r["sample_id"]: r for r in perturbed_results if "error" not in r}
    
    # Match samples
    matched_pairs = []
    for sample_id in baseline_map:
        if sample_id in perturbed_map:
            matched_pairs.append({
                "sample_id": sample_id,
                "baseline": baseline_map[sample_id],
                "perturbed": perturbed_map[sample_id]
            })
    
    print(f"Matched {len(matched_pairs)} samples for comparison")
    
    if len(matched_pairs) == 0:
        print("Warning: No matched pairs found for comparison!")
        return {
            "error_type": error_type,
            "num_matched_samples": 0,
            "error": "No matched pairs found"
        }
    
    # Calculate score drops
    t_drops = []
    e_drops = []
    b_drops = []
    trust_score_drops = []
    
    for pair in matched_pairs:
        baseline = pair["baseline"]
        perturbed = pair["perturbed"]
        
        t_drop = baseline["agg_score_T"] - perturbed["agg_score_T"]
        e_drop = baseline["agg_score_E"] - perturbed["agg_score_E"]
        b_drop = baseline["agg_score_B"] - perturbed["agg_score_B"]
        trust_drop = baseline["trust_score"] - perturbed["trust_score"]
        
        t_drops.append(t_drop)
        e_drops.append(e_drop)
        b_drops.append(b_drop)
        trust_score_drops.append(trust_drop)
    
    # Calculate statistics
    def calc_stats(drops: List[float], name: str) -> Dict[str, float]:
        if not drops:
            return {}
        
        return {
            f"{name}_mean_drop": statistics.mean(drops),
            f"{name}_median_drop": statistics.median(drops),
            f"{name}_std_drop": statistics.stdev(drops) if len(drops) > 1 else 0.0,
            f"{name}_min_drop": min(drops),
            f"{name}_max_drop": max(drops),
            f"{name}_positive_drops": sum(1 for d in drops if d > 0),
            f"{name}_total_samples": len(drops)
        }
    
    comparison = {
        "error_type": error_type,
        "num_matched_samples": len(matched_pairs),
        "trustworthiness_score": calc_stats(t_drops, "T"),
        "explainability_score": calc_stats(e_drops, "E"),
        "bias_score": calc_stats(b_drops, "B"),
        "trust_score": calc_stats(trust_score_drops, "trust"),
        "specificity_analysis": {
            # Check if the injected error type shows the largest drop
            f"T_drop_when_{error_type}_injected": statistics.mean(t_drops) if t_drops else 0,
            f"E_drop_when_{error_type}_injected": statistics.mean(e_drops) if e_drops else 0,
            f"B_drop_when_{error_type}_injected": statistics.mean(b_drops) if b_drops else 0,
        }
    }
    
    # Add specificity check
    if error_type == "T":
        target_drop = statistics.mean(t_drops) if t_drops else 0
        other_drops = [
            statistics.mean(e_drops) if e_drops else 0,
            statistics.mean(b_drops) if b_drops else 0
        ]
        comparison["specificity_analysis"]["target_dimension_drop"] = target_drop
        comparison["specificity_analysis"]["other_dimensions_drop"] = other_drops
        comparison["specificity_analysis"]["is_specific"] = target_drop > max(other_drops) if other_drops else False
    
    elif error_type == "B":
        target_drop = statistics.mean(b_drops) if b_drops else 0
        other_drops = [
            statistics.mean(t_drops) if t_drops else 0,
            statistics.mean(e_drops) if e_drops else 0
        ]
        comparison["specificity_analysis"]["target_dimension_drop"] = target_drop
        comparison["specificity_analysis"]["other_dimensions_drop"] = other_drops
        comparison["specificity_analysis"]["is_specific"] = target_drop > max(other_drops) if other_drops else False
    
    elif error_type == "E":
        target_drop = statistics.mean(e_drops) if e_drops else 0
        other_drops = [
            statistics.mean(t_drops) if t_drops else 0,
            statistics.mean(b_drops) if b_drops else 0
        ]
        comparison["specificity_analysis"]["target_dimension_drop"] = target_drop
        comparison["specificity_analysis"]["other_dimensions_drop"] = other_drops
        comparison["specificity_analysis"]["is_specific"] = target_drop > max(other_drops) if other_drops else False
    
    elif error_type == "PLACEBO":
        # For placebo, we expect minimal/no drops
        max_drop = max(
            statistics.mean(t_drops) if t_drops else 0,
            statistics.mean(e_drops) if e_drops else 0,
            statistics.mean(b_drops) if b_drops else 0,
            abs(statistics.mean(trust_score_drops)) if trust_score_drops else 0
        )
        comparison["specificity_analysis"]["max_dimension_drop"] = max_drop
        comparison["specificity_analysis"]["is_placebo_effective"] = max_drop < 0.1  # Threshold for minimal change
    
    # Print summary
    print(f"\nScore Drops (Mean):")
    print(f"  Trustworthiness (T): {comparison['specificity_analysis'][f'T_drop_when_{error_type}_injected']:.3f}")
    print(f"  Explainability (E): {comparison['specificity_analysis'][f'E_drop_when_{error_type}_injected']:.3f}")
    print(f"  Bias (B): {comparison['specificity_analysis'][f'B_drop_when_{error_type}_injected']:.3f}")
    
    if error_type == "PLACEBO":
        max_drop = comparison['specificity_analysis'].get('max_dimension_drop', 0)
        is_effective = comparison['specificity_analysis'].get('is_placebo_effective', False)
        print(f"\nPlacebo Effect: Max drop = {max_drop:.3f}")
        print(f"Placebo Effective (minimal change): {'✓ YES' if is_effective else '✗ NO'}")
    else:
        print(f"\nSpecificity: {'✓ YES' if comparison['specificity_analysis'].get('is_specific', False) else '✗ NO'}")
    
    return comparison


def generate_report(
    comparisons: Dict[str, Dict[str, Any]],  # {error_type: comparison}
    output_path: str
) -> None:
    """Generate a comprehensive report from all comparisons."""
    
    report = {
        "summary": {
            "total_error_types_tested": len(comparisons),
            "error_types": list(comparisons.keys())
        },
        "detailed_results": comparisons,
        "overall_assessment": {}
    }
    
    # Overall specificity assessment
    for error_type, comparison in comparisons.items():
        if "error" in comparison:
            continue
            
        spec_analysis = comparison.get("specificity_analysis", {})
        
        if error_type == "PLACEBO":
            report["overall_assessment"][error_type] = {
                "is_placebo_effective": spec_analysis.get("is_placebo_effective", False),
                "max_dimension_drop": spec_analysis.get("max_dimension_drop", 0)
            }
        else:
            report["overall_assessment"][error_type] = {
                "is_specific": spec_analysis.get("is_specific", False),
                "target_drop": spec_analysis.get("target_dimension_drop", 0),
                "other_drops": spec_analysis.get("other_dimensions_drop", [])
            }
    
    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SPECIFICITY ANALYSIS REPORT")
    print("=" * 70)
    
    for error_type, comp in comparisons.items():
        if "error" in comp:
            continue
            
        spec_analysis = comp.get("specificity_analysis", {})
        
        if error_type == "PLACEBO":
            max_drop = spec_analysis.get("max_dimension_drop", 0)
            is_effective = spec_analysis.get("is_placebo_effective", False)
            print(f"\n{error_type} Injection:")
            print(f"  Max dimension drop: {max_drop:.3f}")
            print(f"  Placebo effective (minimal change): {'✓ YES' if is_effective else '✗ NO'}")
        else:
            is_specific = spec_analysis.get("is_specific", False)
            target_drop = spec_analysis.get("target_dimension_drop", 0)
            print(f"\n{error_type} Error Injection:")
            print(f"  Target dimension drop: {target_drop:.3f}")
            print(f"  Is specific: {'✓ YES' if is_specific else '✗ NO'}")
    
    print(f"\n✓ Report saved to {output_path}")


if __name__ == "__main__":
    # Test comparison
    baseline_path = "results/specificity_analysis/baseline_results.jsonl"
    t_perturbed_path = "results/specificity_analysis/T_perturbed_results.jsonl"
    
    if os.path.exists(baseline_path) and os.path.exists(t_perturbed_path):
        comparison = compare_scores(baseline_path, t_perturbed_path, "T")
        print(json.dumps(comparison, indent=2))
    else:
        print("Test files not found. Run full analysis first.")
