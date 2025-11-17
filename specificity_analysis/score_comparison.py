"""
Compare scores between baseline and perturbed datasets
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
import statistics
from config.settings import TrustScoreConfig


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def merge_baseline_errors_into_perturbed(
    baseline_result: Dict[str, Any],
    perturbed_result: Dict[str, Any],
    config: TrustScoreConfig
) -> Dict[str, Any]:
    """
    Merge baseline errors into perturbed results and recalculate scores.
    
    This ensures perturbed results contain all baseline errors plus newly injected errors.
    This should improve specificity analysis by ensuring we're comparing:
    - Baseline: original errors
    - Perturbed: original errors + newly injected error
    
    Args:
        baseline_result: Baseline result with original errors
        perturbed_result: Perturbed result with newly detected errors
        config: TrustScoreConfig for recalculating scores
        
    Returns:
        Updated perturbed result with merged errors and recalculated scores
    """
    baseline_spans = baseline_result.get("spans", {})
    perturbed_spans = perturbed_result.get("spans", {})
    
    # Merge: start with baseline spans, then add perturbed spans
    # Keep both even if duplicates (as requested)
    merged_spans = baseline_spans.copy()
    
    # Add perturbed spans with new IDs to avoid conflicts
    for span_id, span in perturbed_spans.items():
        new_id = f"perturbed_{len(merged_spans)}"
        merged_spans[new_id] = span
    
    # Recalculate scores from merged spans
    return recalculate_scores_from_spans(
        perturbed_result, merged_spans, config
    )


def recalculate_scores_from_spans(
    original_result: Dict[str, Any],
    merged_spans: Dict[str, Any],
    config: TrustScoreConfig
) -> Dict[str, Any]:
    """
    Recalculate TrustScore from merged spans without re-running inference.
    
    Args:
        original_result: Original result to update
        merged_spans: Dictionary of merged spans
        config: TrustScoreConfig for weights and aggregation
        
    Returns:
        Updated result with recalculated scores
    """
    from modules.aggregator import Aggregator
    
    # Group spans by type
    t_spans = [s for s in merged_spans.values() if s.get("type") == "T"]
    e_spans = [s for s in merged_spans.values() if s.get("type") == "E"]
    b_spans = [s for s in merged_spans.values() if s.get("type") == "B"]
    
    # Calculate category severity scores (sum of weighted severity)
    t_score = sum(
        s.get("severity_score", 0) * config.get_error_subtype_weight("T", s.get("subtype", ""))
        for s in t_spans
    )
    e_score = sum(
        s.get("severity_score", 0) * config.get_error_subtype_weight("E", s.get("subtype", ""))
        for s in e_spans
    )
    b_score = sum(
        s.get("severity_score", 0) * config.get_error_subtype_weight("B", s.get("subtype", ""))
        for s in b_spans
    )
    
    # Calculate overall TrustScore (using aggregation weights)
    weights = config.aggregation_weights
    trust_score = (
        weights.trustworthiness * t_score +
        weights.explainability * e_score +
        weights.bias * b_score
    )
    
    # Convert to quality scores using aggregator
    aggregator = Aggregator(config)
    t_quality = aggregator._severity_to_quality(t_score)
    e_quality = aggregator._severity_to_quality(e_score)
    b_quality = aggregator._severity_to_quality(b_score)
    trust_quality = aggregator._severity_to_quality(trust_score)
    
    # Create updated result
    updated_result = original_result.copy()
    updated_result.update({
        "agg_score_T": t_score,
        "agg_quality_T": t_quality,
        "agg_score_E": e_score,
        "agg_quality_E": e_quality,
        "agg_score_B": b_score,
        "agg_quality_B": b_quality,
        "trust_score": trust_score,
        "trust_quality": trust_quality,
        "num_errors": len(merged_spans),
        "spans": merged_spans
    })
    
    return updated_result


def compare_scores(
    baseline_results_path: str,
    perturbed_results_path: str,
    error_type: str,  # "T", "B", or "E"
    config: Optional[TrustScoreConfig] = None  # Add config parameter
) -> Dict[str, Any]:
    """
    Compare baseline and perturbed scores to measure specificity.
    
    Args:
        baseline_results_path: Path to baseline results
        perturbed_results_path: Path to perturbed results
        error_type: Type of error injected ("T", "B", or "E")
        config: Optional TrustScoreConfig for recalculating scores (required for merging)
        
    Returns:
        Dictionary with comparison statistics
    """
    print("=" * 70)
    print(f"Step 4: Comparing Scores for {error_type}_perturbed")
    print("=" * 70)
    
    # Load config if not provided
    if config is None:
        config = TrustScoreConfig()  # Use default config
    
    baseline_results = load_results(baseline_results_path)
    perturbed_results = load_results(perturbed_results_path)
    
    # Create mapping by unique_dataset_id (preferred) or sample_id (fallback)
    baseline_map = {}
    perturbed_map = {}
    
    for r in baseline_results:
        if "error" not in r:
            # Prefer unique_dataset_id if available, fallback to sample_id
            key = r.get("unique_dataset_id") or r.get("sample_id", "")
            if key:
                baseline_map[key] = r
    
    for r in perturbed_results:
        if "error" not in r:
            # Prefer unique_dataset_id if available, fallback to sample_id
            key = r.get("unique_dataset_id") or r.get("sample_id", "")
            if key:
                perturbed_map[key] = r
    
    # Match samples and merge baseline errors into perturbed results
    matched_pairs = []
    # Use union of both sets to find all possible matches
    all_keys = set(baseline_map.keys()) | set(perturbed_map.keys())
    for key in all_keys:
        if key in baseline_map and key in perturbed_map:
            baseline = baseline_map[key]
            perturbed = perturbed_map[key]
            
            # Merge baseline errors into perturbed results
            perturbed_merged = merge_baseline_errors_into_perturbed(
                baseline, perturbed, config
            )
            
            matched_pairs.append({
                "unique_dataset_id": key,  # Use unique_dataset_id as primary identifier
                "sample_id": baseline.get("sample_id", ""),  # Also include sample_id for reference
                "baseline": baseline,  # Keep original baseline
                "perturbed": perturbed_merged  # Use merged perturbed
            })
    
    print(f"Matched {len(matched_pairs)} samples for comparison")
    print(f"  (Merged baseline errors into perturbed results before comparison)")
    
    if len(matched_pairs) == 0:
        print("Warning: No matched pairs found for comparison!")
        return {
            "error_type": error_type,
            "num_matched_samples": 0,
            "error": "No matched pairs found"
        }
    
    # Calculate quality decreases (for quality scores: higher = better, so decrease = baseline - perturbed)
    # For raw severity scores: higher = worse, so drop = baseline - perturbed (positive when severity increases)
    t_quality_decreases = []  # Absolute point differences
    e_quality_decreases = []
    b_quality_decreases = []
    trust_quality_decreases = []
    t_quality_pct_changes = []  # Percentage changes
    e_quality_pct_changes = []
    b_quality_pct_changes = []
    trust_quality_pct_changes = []
    # Also keep raw severity drops for backward compatibility
    t_severity_drops = []
    e_severity_drops = []
    b_severity_drops = []
    trust_severity_drops = []
    
    # Debug: Track baseline and perturbed values for first few samples
    debug_samples = []
    
    for i, pair in enumerate(matched_pairs):
        baseline = pair["baseline"]
        perturbed = pair["perturbed"]
        
        # Get quality scores (prefer quality scores, fallback to raw scores)
        t_baseline = baseline.get("agg_quality_T", baseline.get("agg_score_T", 0))
        t_perturbed = perturbed.get("agg_quality_T", perturbed.get("agg_score_T", 0))
        e_baseline = baseline.get("agg_quality_E", baseline.get("agg_score_E", 0))
        e_perturbed = perturbed.get("agg_quality_E", perturbed.get("agg_score_E", 0))
        b_baseline = baseline.get("agg_quality_B", baseline.get("agg_score_B", 0))
        b_perturbed = perturbed.get("agg_quality_B", perturbed.get("agg_score_B", 0))
        trust_baseline = baseline.get("trust_quality", baseline.get("trust_score", 0))
        trust_perturbed = perturbed.get("trust_quality", perturbed.get("trust_score", 0))
        
        # Quality scores: higher = better, so decrease = baseline - perturbed (positive when quality decreases)
        t_quality_decrease = t_baseline - t_perturbed
        e_quality_decrease = e_baseline - e_perturbed
        b_quality_decrease = b_baseline - b_perturbed
        trust_quality_decrease = trust_baseline - trust_perturbed
        
        # Calculate percentage changes: ((baseline - perturbed) / baseline) * 100
        # Handle division by zero and very small baselines (which cause huge percentages)
        def calc_pct_change(baseline_val, perturbed_val):
            if baseline_val == 0:
                # If baseline is 0, use absolute change as percentage of scale (100)
                return ((baseline_val - perturbed_val) / 100.0) * 100.0 if baseline_val != perturbed_val else 0.0
            elif abs(baseline_val) < 1.0:
                # For very small baselines (< 1.0), percentage changes are misleading
                # Instead, report as percentage of scale (100) to avoid huge numbers
                return ((baseline_val - perturbed_val) / 100.0) * 100.0
            else:
                return ((baseline_val - perturbed_val) / baseline_val) * 100.0
        
        t_pct_change = calc_pct_change(t_baseline, t_perturbed)
        e_pct_change = calc_pct_change(e_baseline, e_perturbed)
        b_pct_change = calc_pct_change(b_baseline, b_perturbed)
        trust_pct_change = calc_pct_change(trust_baseline, trust_perturbed)
        
        # Raw severity scores: higher = worse, so drop = baseline - perturbed (positive when severity increases)
        t_severity_drop = baseline["agg_score_T"] - perturbed["agg_score_T"]
        e_severity_drop = baseline["agg_score_E"] - perturbed["agg_score_E"]
        b_severity_drop = baseline["agg_score_B"] - perturbed["agg_score_B"]
        trust_severity_drop = baseline["trust_score"] - perturbed["trust_score"]
        
        t_quality_decreases.append(t_quality_decrease)
        e_quality_decreases.append(e_quality_decrease)
        b_quality_decreases.append(b_quality_decrease)
        trust_quality_decreases.append(trust_quality_decrease)
        
        t_quality_pct_changes.append(t_pct_change)
        e_quality_pct_changes.append(e_pct_change)
        b_quality_pct_changes.append(b_pct_change)
        trust_quality_pct_changes.append(trust_pct_change)
        
        t_severity_drops.append(t_severity_drop)
        e_severity_drops.append(e_severity_drop)
        b_severity_drops.append(b_severity_drop)
        trust_severity_drops.append(trust_severity_drop)
        
        # Store debug info for first 3 samples
        if i < 3:
            debug_samples.append({
                "sample_id": pair.get("sample_id", pair.get("unique_dataset_id", "unknown")),
                "unique_dataset_id": pair.get("unique_dataset_id", "unknown"),
                "T": {"baseline": t_baseline, "perturbed": t_perturbed, "decrease": t_quality_decrease, "pct_change": t_pct_change},
                "E": {"baseline": e_baseline, "perturbed": e_perturbed, "decrease": e_quality_decrease, "pct_change": e_pct_change},
                "B": {"baseline": b_baseline, "perturbed": b_perturbed, "decrease": b_quality_decrease, "pct_change": b_pct_change},
            })
    
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
        # Quality scores (higher = better, so positive decrease means quality went down)
        "trustworthiness_quality": calc_stats(t_quality_decreases, "T_quality"),
        "explainability_quality": calc_stats(e_quality_decreases, "E_quality"),
        "bias_quality": calc_stats(b_quality_decreases, "B_quality"),
        "trust_quality": calc_stats(trust_quality_decreases, "trust_quality"),
        # Raw severity scores (higher = worse, so positive drop means severity increased) - for backward compatibility
        "trustworthiness_severity": calc_stats(t_severity_drops, "T_severity"),
        "explainability_severity": calc_stats(e_severity_drops, "E_severity"),
        "bias_severity": calc_stats(b_severity_drops, "B_severity"),
        "trust_severity": calc_stats(trust_severity_drops, "trust_severity"),
        "specificity_analysis": {
            # Check if the injected error type shows the largest quality decrease
            # Using quality scores (higher = better, so positive decrease means quality went down)
            # Store both absolute point differences and percentage changes
            f"T_quality_decrease_when_{error_type}_injected": statistics.mean(t_quality_decreases) if t_quality_decreases else 0,
            f"E_quality_decrease_when_{error_type}_injected": statistics.mean(e_quality_decreases) if e_quality_decreases else 0,
            f"B_quality_decrease_when_{error_type}_injected": statistics.mean(b_quality_decreases) if b_quality_decreases else 0,
            f"T_quality_pct_change_when_{error_type}_injected": statistics.mean(t_quality_pct_changes) if t_quality_pct_changes else 0,
            f"E_quality_pct_change_when_{error_type}_injected": statistics.mean(e_quality_pct_changes) if e_quality_pct_changes else 0,
            f"B_quality_pct_change_when_{error_type}_injected": statistics.mean(b_quality_pct_changes) if b_quality_pct_changes else 0,
        },
        "debug_samples": debug_samples
    }
    
    # Add specificity check using quality decreases (higher = better, so positive decrease means quality went down)
    if error_type == "T":
        target_decrease = statistics.mean(t_quality_decreases) if t_quality_decreases else 0
        target_pct_change = statistics.mean(t_quality_pct_changes) if t_quality_pct_changes else 0
        other_decreases = [
            statistics.mean(e_quality_decreases) if e_quality_decreases else 0,
            statistics.mean(b_quality_decreases) if b_quality_decreases else 0
        ]
        comparison["specificity_analysis"]["target_dimension_quality_decrease"] = target_decrease
        comparison["specificity_analysis"]["target_dimension_drop"] = target_decrease  # For backward compatibility with report
        comparison["specificity_analysis"]["target_dimension_pct_change"] = target_pct_change
        comparison["specificity_analysis"]["other_dimensions_quality_decrease"] = other_decreases
        comparison["specificity_analysis"]["other_dimensions_drop"] = other_decreases  # For backward compatibility
        comparison["specificity_analysis"]["is_specific"] = target_decrease > max(other_decreases) if other_decreases else False
    
    elif error_type == "B":
        target_decrease = statistics.mean(b_quality_decreases) if b_quality_decreases else 0
        target_pct_change = statistics.mean(b_quality_pct_changes) if b_quality_pct_changes else 0
        other_decreases = [
            statistics.mean(t_quality_decreases) if t_quality_decreases else 0,
            statistics.mean(e_quality_decreases) if e_quality_decreases else 0
        ]
        comparison["specificity_analysis"]["target_dimension_quality_decrease"] = target_decrease
        comparison["specificity_analysis"]["target_dimension_drop"] = target_decrease  # For backward compatibility with report
        comparison["specificity_analysis"]["target_dimension_pct_change"] = target_pct_change
        comparison["specificity_analysis"]["other_dimensions_quality_decrease"] = other_decreases
        comparison["specificity_analysis"]["other_dimensions_drop"] = other_decreases  # For backward compatibility
        comparison["specificity_analysis"]["is_specific"] = target_decrease > max(other_decreases) if other_decreases else False
    
    elif error_type == "E":
        target_decrease = statistics.mean(e_quality_decreases) if e_quality_decreases else 0
        target_pct_change = statistics.mean(e_quality_pct_changes) if e_quality_pct_changes else 0
        other_decreases = [
            statistics.mean(t_quality_decreases) if t_quality_decreases else 0,
            statistics.mean(b_quality_decreases) if b_quality_decreases else 0
        ]
        comparison["specificity_analysis"]["target_dimension_quality_decrease"] = target_decrease
        comparison["specificity_analysis"]["target_dimension_drop"] = target_decrease  # For backward compatibility with report
        comparison["specificity_analysis"]["target_dimension_pct_change"] = target_pct_change
        comparison["specificity_analysis"]["other_dimensions_quality_decrease"] = other_decreases
        comparison["specificity_analysis"]["other_dimensions_drop"] = other_decreases  # For backward compatibility
        comparison["specificity_analysis"]["is_specific"] = target_decrease > max(other_decreases) if other_decreases else False
    
    elif error_type == "PLACEBO":
        # For placebo, we expect minimal/no quality decreases
        max_decrease = max(
            abs(statistics.mean(t_quality_decreases)) if t_quality_decreases else 0,
            abs(statistics.mean(e_quality_decreases)) if e_quality_decreases else 0,
            abs(statistics.mean(b_quality_decreases)) if b_quality_decreases else 0,
            abs(statistics.mean(trust_quality_decreases)) if trust_quality_decreases else 0
        )
        comparison["specificity_analysis"]["max_quality_decrease"] = max_decrease
        comparison["specificity_analysis"]["max_dimension_drop"] = max_decrease  # For backward compatibility with report
        comparison["specificity_analysis"]["is_placebo_effective"] = max_decrease < 1.0  # Threshold for minimal quality change (< 1 point drop)
    
    # Print summary with both absolute and percentage changes
    print(f"\nQuality Changes (Mean) - Higher = Better, so positive decrease means quality went down:")
    t_abs = comparison['specificity_analysis'][f'T_quality_decrease_when_{error_type}_injected']
    e_abs = comparison['specificity_analysis'][f'E_quality_decrease_when_{error_type}_injected']
    b_abs = comparison['specificity_analysis'][f'B_quality_decrease_when_{error_type}_injected']
    t_pct = comparison['specificity_analysis'][f'T_quality_pct_change_when_{error_type}_injected']
    e_pct = comparison['specificity_analysis'][f'E_quality_pct_change_when_{error_type}_injected']
    b_pct = comparison['specificity_analysis'][f'B_quality_pct_change_when_{error_type}_injected']
    
    print(f"  Trustworthiness (T): {t_abs:+.2f} points ({t_pct:+.2f}%)")
    print(f"  Explainability (E): {e_abs:+.2f} points ({e_pct:+.2f}%)")
    print(f"  Bias (B): {b_abs:+.2f} points ({b_pct:+.2f}%)")
    
    # Print debug info for first few samples
    if debug_samples:
        print(f"\nDebug: First {len(debug_samples)} sample(s) comparison:")
        for i, sample_debug in enumerate(debug_samples, 1):
            print(f"  Sample {i} ({sample_debug['sample_id']}):")
            for dim in ["T", "E", "B"]:
                dim_data = sample_debug[dim]
                print(f"    {dim}: baseline={dim_data['baseline']:.2f}, perturbed={dim_data['perturbed']:.2f}, "
                      f"decrease={dim_data['decrease']:+.2f} points ({dim_data['pct_change']:+.2f}%)")
    
    if error_type == "PLACEBO":
        max_decrease = comparison['specificity_analysis'].get('max_quality_decrease', 0)
        is_effective = comparison['specificity_analysis'].get('is_placebo_effective', False)
        print(f"\nPlacebo Effect: Max quality decrease = {max_decrease:.2f} points")
        print(f"Placebo Effective (minimal change): {'✓ YES' if is_effective else '✗ NO'}")
    else:
        target_decrease = comparison['specificity_analysis'].get('target_dimension_quality_decrease', 0)
        other_decreases = comparison['specificity_analysis'].get('other_dimensions_quality_decrease', [])
        is_specific = comparison['specificity_analysis'].get('is_specific', False)
        print(f"\nSpecificity Analysis:")
        print(f"  Target dimension ({error_type}) decrease: {target_decrease:+.2f} points")
        print(f"  Other dimensions decreases: {[f'{d:+.2f}' for d in other_decreases]}")
        print(f"  Is specific: {'✓ YES' if is_specific else '✗ NO'}")
    
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
