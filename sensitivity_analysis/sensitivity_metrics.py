"""
Sensitivity Analysis Metrics

Calculate Kendall's tau and Spearman's rho correlations to verify monotonic
decrease in TrustScore as error count increases.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
import statistics
from scipy.stats import kendalltau, spearmanr


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def calculate_monotonicity_metrics(
    results_by_k: Dict[int, List[Dict[str, Any]]],
    target_dimension: str  # "T", "B", or "E"
) -> Dict[str, Any]:
    """
    Calculate monotonicity metrics (Kendall's tau and Spearman's rho) for target dimension
    and off-target dimensions.
    
    Args:
        results_by_k: Dictionary mapping k (0,1,2,...,K_max) to list of result dictionaries
        target_dimension: The dimension being tested (e.g., "T" when injecting T errors)
        
    Returns:
        Dictionary with correlation metrics
    """
    # Get all k values sorted
    k_values = sorted([k for k in results_by_k.keys() if results_by_k[k]])
    
    if len(k_values) < 2:
        raise ValueError("Need at least 2 different k values to calculate correlation")
    
    # Create mapping by unique_dataset_id
    sample_map: Dict[str, Dict[int, Dict[str, Any]]] = {}
    
    for k in k_values:
        for result in results_by_k[k]:
            sample_id = result.get("unique_dataset_id") or result.get("sample_id", "unknown")
            if sample_id not in sample_map:
                sample_map[sample_id] = {}
            sample_map[sample_id][k] = result
    
    # Extract scores for each sample across k values
    target_scores_by_sample: Dict[str, List[Tuple[int, float]]] = {}  # sample_id -> [(k, score), ...]
    off_target_scores: Dict[str, Dict[str, List[Tuple[int, float]]]] = {  # dimension -> sample_id -> [(k, score), ...]
        "T": {},
        "E": {},
        "B": {}
    }
    
    for sample_id, k_results in sample_map.items():
        # Sort by k for this sample
        sorted_k_results = sorted(k_results.items())
        
        # Extract target dimension scores
        # Use quality scores (higher = better) instead of severity scores (higher = worse)
        target_scores = []
        for k, result in sorted_k_results:
            if "error" not in result:  # Skip failed results
                # Prefer quality scores, fallback to severity scores for backward compatibility
                score = result.get(f"agg_quality_{target_dimension}", result.get(f"agg_score_{target_dimension}", 0.0))
                target_scores.append((k, score))
        
        if len(target_scores) >= 2:
            target_scores_by_sample[sample_id] = target_scores
        
        # Extract off-target dimension scores
        off_target_dims = ["T", "E", "B"]
        off_target_dims.remove(target_dimension)
        
        for dim in off_target_dims:
            if sample_id not in off_target_scores[dim]:
                off_target_scores[dim][sample_id] = []
            for k, result in sorted_k_results:
                if "error" not in result:
                    # Prefer quality scores, fallback to severity scores for backward compatibility
                    score = result.get(f"agg_quality_{dim}", result.get(f"agg_score_{dim}", 0.0))
                    off_target_scores[dim][sample_id].append((k, score))
    
    # Calculate correlations for target dimension
    # Aggregate across all samples: for each k, get mean score
    k_to_scores: Dict[int, List[float]] = {}
    for sample_id, scores in target_scores_by_sample.items():
        for k, score in scores:
            if k not in k_to_scores:
                k_to_scores[k] = []
            k_to_scores[k].append(score)
    
    # Calculate mean score for each k
    k_values_sorted = sorted([k for k in k_to_scores.keys() if k_to_scores[k]])  # Only k with scores
    mean_scores = [statistics.mean(k_to_scores[k]) for k in k_values_sorted]
    
    # Calculate Kendall's tau and Spearman's rho
    # Note: We expect NEGATIVE correlation (quality decreases as k increases)
    # This is correct for quality scores (higher = better, so more errors = lower quality)
    if len(k_values_sorted) >= 2:
        kendall_result = kendalltau(k_values_sorted, mean_scores)
        spearman_result = spearmanr(k_values_sorted, mean_scores)
        
        # Convert numpy types to native Python types for JSON serialization
        kendall_tau = float(kendall_result.correlation) if kendall_result.correlation is not None else None
        kendall_pvalue = float(kendall_result.pvalue) if kendall_result.pvalue is not None else None
        spearman_rho = float(spearman_result.correlation) if spearman_result.correlation is not None else None
        spearman_pvalue = float(spearman_result.pvalue) if spearman_result.pvalue is not None else None
        
        # Monotonic if correlation is significantly negative
        # Convert to native Python bool
        monotonic = bool(kendall_tau < -0.5 and kendall_pvalue < 0.05) if kendall_tau is not None and kendall_pvalue is not None else False
    else:
        kendall_tau = None
        kendall_pvalue = None
        spearman_rho = None
        spearman_pvalue = None
        monotonic = False
    
    # Calculate off-target correlations (should be near 0)
    off_target_metrics = {}
    off_target_mean_scores_by_k = {}  # Store mean scores by k for each off-target dimension
    for dim in off_target_dims:
        dim_k_to_scores: Dict[int, List[float]] = {}
        for sample_id, scores in off_target_scores[dim].items():
            for k, score in scores:
                if k not in dim_k_to_scores:
                    dim_k_to_scores[k] = []
                dim_k_to_scores[k].append(score)
        
        # Store mean scores by k for this off-target dimension
        dim_mean_scores_by_k = {}
        for k, scores_list in dim_k_to_scores.items():
            if scores_list:  # Only if there are scores for this k
                dim_mean_scores_by_k[str(k)] = float(statistics.mean(scores_list))
        off_target_mean_scores_by_k[dim] = dim_mean_scores_by_k
        
        if len(dim_k_to_scores) >= 2:
            dim_k_values = sorted([k for k in dim_k_to_scores.keys() if dim_k_to_scores[k]])  # Only k with scores
            dim_mean_scores = [statistics.mean(dim_k_to_scores[k]) for k in dim_k_values]
            
            dim_kendall = kendalltau(dim_k_values, dim_mean_scores)
            dim_spearman = spearmanr(dim_k_values, dim_mean_scores)
            
            # Convert numpy types to native Python types for JSON serialization
            off_target_metrics[dim] = {
                "kendall_tau": float(dim_kendall.correlation) if dim_kendall.correlation is not None else None,
                "kendall_pvalue": float(dim_kendall.pvalue) if dim_kendall.pvalue is not None else None,
                "spearman_rho": float(dim_spearman.correlation) if dim_spearman.correlation is not None else None,
                "spearman_pvalue": float(dim_spearman.pvalue) if dim_spearman.pvalue is not None else None
            }
        else:
            off_target_metrics[dim] = {
                "kendall_tau": None,
                "kendall_pvalue": None,
                "spearman_rho": None,
                "spearman_pvalue": None
            }
    
    # Create mean scores by k dictionary
    # Convert to native Python float for JSON serialization
    mean_scores_by_k = {}
    for k, scores in k_to_scores.items():
        if scores:  # Only if there are scores for this k
            mean_scores_by_k[str(k)] = float(statistics.mean(scores))
    
    return {
        "target_gauge": {
            "kendall_tau": kendall_tau,
            "kendall_pvalue": kendall_pvalue,
            "spearman_rho": spearman_rho,
            "spearman_pvalue": spearman_pvalue,
            "monotonic": monotonic
        },
        "off_target_gauges": off_target_metrics,
        "mean_scores_by_k": mean_scores_by_k,
        "off_target_mean_scores_by_k": off_target_mean_scores_by_k,  # Mean scores by k for each off-target dimension
        "num_samples": len(target_scores_by_sample)
    }


def calculate_sensitivity_report(
    results_dir: str,
    error_type: str,
    subtype: str,
    k_max: int
) -> Dict[str, Any]:
    """
    Calculate sensitivity report for a specific dimension/subtype combination.
    
    Args:
        results_dir: Directory containing result files
        error_type: Error type ("T", "B", or "E")
        subtype: Error subtype (e.g., "factual_error")
        k_max: Maximum k value
        
    Returns:
        Dictionary with sensitivity analysis report
    """
    # Load results for each k
    results_by_k: Dict[int, List[Dict[str, Any]]] = {}
    
    for k in range(0, k_max + 1):
        result_file = os.path.join(results_dir, f"{error_type}_{subtype}_k{k}_results.jsonl")
        if os.path.exists(result_file):
            try:
                results_by_k[k] = load_results(result_file)
            except Exception as e:
                print(f"[WARNING] Failed to load results for k={k}: {str(e)}")
                results_by_k[k] = []
        else:
            print(f"[WARNING] Results file not found for k={k}: {result_file}")
            results_by_k[k] = []
    
    # Calculate monotonicity metrics
    metrics = calculate_monotonicity_metrics(results_by_k, error_type)
    
    return {
        "dimension": error_type,
        "subtype": subtype,
        "k_max": k_max,
        **metrics
    }

