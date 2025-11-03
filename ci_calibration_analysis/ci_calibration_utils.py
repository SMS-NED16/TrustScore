"""
CI Calibration Analysis Utilities

This module provides utilities for analyzing CI calibration results.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats
from collections import defaultdict


def load_calibration_results(results_path: str) -> List[Dict[str, Any]]:
    """
    Load calibration results from JSONL file.
    
    Args:
        results_path: Path to calibration_results.jsonl
        
    Returns:
        List of result dictionaries
    """
    results = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    result = json.loads(line.strip())
                    results.append(result)
                except json.JSONDecodeError as e:
                    print(f"⚠ Warning: Could not parse line: {e}")
                    continue
    return results


def compute_ci_width(ci: Optional[Dict[str, float]]) -> Optional[float]:
    """
    Compute CI width (upper - lower).
    
    Args:
        ci: Confidence interval dict with 'lower' and 'upper' keys
        
    Returns:
        CI width or None if CI is None
    """
    if ci is None or ci.get("lower") is None or ci.get("upper") is None:
        return None
    return ci["upper"] - ci["lower"]


def compute_variance(values: List[float]) -> float:
    """
    Compute variance of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Variance (or 0 if insufficient data)
    """
    if len(values) < 2:
        return 0.0
    return np.var(values, ddof=1)  # Sample variance


def compute_std(values: List[float]) -> float:
    """
    Compute standard deviation of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Standard deviation (or 0 if insufficient data)
    """
    if len(values) < 2:
        return 0.0
    return np.std(values, ddof=1)  # Sample standard deviation


def check_coverage(
    ci_list: List[Optional[Dict[str, float]]],
    point_estimates: List[float],
    true_value: Optional[float] = None
) -> Tuple[float, int, int]:
    """
    Check coverage: fraction of CIs that contain the true value.
    
    Args:
        ci_list: List of CI dictionaries
        point_estimates: List of point estimates corresponding to CIs
        true_value: True value to check against. If None, uses mean of point estimates.
        
    Returns:
        Tuple of (coverage_fraction, num_covered, total_valid)
    """
    if true_value is None:
        # Use mean-as-truth method
        valid_points = [p for p in point_estimates if p is not None]
        if len(valid_points) == 0:
            return 0.0, 0, 0
        true_value = np.mean(valid_points)
    
    covered = 0
    total_valid = 0
    
    for ci, point in zip(ci_list, point_estimates):
        if ci is None or point is None:
            continue
        
        lower = ci.get("lower")
        upper = ci.get("upper")
        
        if lower is None or upper is None:
            continue
        
        total_valid += 1
        if lower <= true_value <= upper:
            covered += 1
    
    coverage = covered / total_valid if total_valid > 0 else 0.0
    return coverage, covered, total_valid


def compute_correlation(
    ci_widths: List[float],
    observed_stds: List[float]
) -> Tuple[float, float, float]:
    """
    Compute Pearson and Spearman correlations between CI width and observed std.
    
    Args:
        ci_widths: List of CI widths
        observed_stds: List of observed standard deviations
        
    Returns:
        Tuple of (pearson_r, spearman_r, p_value)
    """
    if len(ci_widths) < 2 or len(observed_stds) < 2:
        return 0.0, 0.0, 1.0
    
    # Filter out None values
    pairs = [(w, s) for w, s in zip(ci_widths, observed_stds) if w is not None and s is not None]
    
    if len(pairs) < 2:
        return 0.0, 0.0, 1.0
    
    widths, stds = zip(*pairs)
    widths = np.array(widths)
    stds = np.array(stds)
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(widths, stds)
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(widths, stds)
    
    return pearson_r, spearman_r, pearson_p


def analyze_ci_level(
    results: List[Dict[str, Any]],
    ci_field: str,
    point_field: str,
    item_id_field: str = "item_id",
    judge_count_field: str = "num_judges",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Analyze CI calibration for a specific CI level.
    
    Args:
        results: List of result dictionaries
        ci_field: Field name for CI (e.g., "trust_score_ci")
        point_field: Field name for point estimate (e.g., "trust_score")
        item_id_field: Field name for item identifier
        judge_count_field: Field name for judge count
        confidence_level: Expected confidence level (e.g., 0.95)
        
    Returns:
        Analysis dictionary
    """
    # Group by (item_id, num_judges)
    grouped = defaultdict(lambda: {"ci_list": [], "points": [], "judge_count": None})
    
    for result in results:
        if "error" in result:  # Skip error records
            continue
        
        item_id = result.get(item_id_field)
        num_judges = result.get(judge_count_field)
        ci = result.get(ci_field)
        point = result.get(point_field)
        
        if item_id is None or num_judges is None:
            continue
        
        key = (item_id, num_judges)
        grouped[key]["ci_list"].append(ci)
        grouped[key]["points"].append(point)
        grouped[key]["judge_count"] = num_judges
    
    # Analyze each group
    analysis = {
        "ci_field": ci_field,
        "point_field": point_field,
        "confidence_level": confidence_level,
        "by_item_judge": {},
        "by_judge_count": defaultdict(lambda: {
            "ci_widths": [],
            "observed_stds": [],
            "mean_ci_width": None,
            "mean_observed_std": None,
            "coverage": None,
            "num_items": 0
        })
    }
    
    # Analyze each (item, judge_count) combination
    for (item_id, num_judges), group_data in grouped.items():
        ci_list = group_data["ci_list"]
        points = group_data["points"]
        judge_count = group_data["judge_count"]
        
        # Filter out None values
        valid_pairs = [(ci, p) for ci, p in zip(ci_list, points) if p is not None]
        if len(valid_pairs) < 2:
            continue
        
        valid_ci_list, valid_points = zip(*valid_pairs)
        valid_ci_list = list(valid_ci_list)
        valid_points = list(valid_points)
        
        # Compute CI widths
        ci_widths = [compute_ci_width(ci) for ci in valid_ci_list]
        ci_widths = [w for w in ci_widths if w is not None]
        
        # Compute observed variance
        observed_std = compute_std(valid_points)
        mean_ci_width = np.mean(ci_widths) if len(ci_widths) > 0 else None
        
        # Coverage check
        coverage, num_covered, total_valid = check_coverage(valid_ci_list, valid_points)
        
        # Store analysis for this (item, judge_count)
        item_judge_key = f"{item_id}_J{judge_count}"
        analysis["by_item_judge"][item_judge_key] = {
            "item_id": item_id,
            "num_judges": judge_count,
            "num_runs": len(valid_points),
            "mean_ci_width": float(mean_ci_width) if mean_ci_width is not None else None,
            "observed_std": float(observed_std),
            "observed_variance": float(compute_variance(valid_points)),
            "coverage": float(coverage),
            "num_covered": num_covered,
            "total_valid": total_valid,
            "point_estimates": [float(p) for p in valid_points]
        }
        
        # Aggregate by judge count
        if mean_ci_width is not None and observed_std is not None:
            analysis["by_judge_count"][judge_count]["ci_widths"].append(mean_ci_width)
            analysis["by_judge_count"][judge_count]["observed_stds"].append(observed_std)
            analysis["by_judge_count"][judge_count]["num_items"] += 1
    
    # Finalize aggregation by judge count
    for judge_count, judge_data in analysis["by_judge_count"].items():
        if len(judge_data["ci_widths"]) > 0:
            judge_data["mean_ci_width"] = float(np.mean(judge_data["ci_widths"]))
            judge_data["mean_observed_std"] = float(np.mean(judge_data["observed_stds"]))
            
            # Compute overall coverage for this judge count
            all_coverage = [
                analysis["by_item_judge"][key]["coverage"]
                for key in analysis["by_item_judge"]
                if analysis["by_item_judge"][key]["num_judges"] == judge_count
            ]
            if len(all_coverage) > 0:
                judge_data["mean_coverage"] = float(np.mean(all_coverage))
            
            # Compute correlation between CI width and observed std
            if len(judge_data["ci_widths"]) >= 2:
                pearson_r, spearman_r, p_value = compute_correlation(
                    judge_data["ci_widths"],
                    judge_data["observed_stds"]
                )
                judge_data["pearson_r"] = float(pearson_r)
                judge_data["spearman_r"] = float(spearman_r)
                judge_data["correlation_p_value"] = float(p_value)
    
    # Convert defaultdict to dict
    analysis["by_judge_count"] = dict(analysis["by_judge_count"])
    
    return analysis


def make_json_serializable(obj):
    """
    Convert numpy types and other non-JSON-serializable types to native Python types.
    
    Args:
        obj: Object to make JSON-serializable
        
    Returns:
        JSON-serializable version of obj
    """
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif obj is None:
        return None
    else:
        return obj

