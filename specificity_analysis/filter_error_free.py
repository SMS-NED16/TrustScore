"""
Filter baseline results to find error-free samples for error injection
"""

import json
import os
from typing import List, Dict, Any, Set, Optional


def filter_error_free_samples(
    baseline_results_path: str,
    max_errors: int = 0,
    error_type_filter: Optional[str] = None,
    max_samples: Optional[int] = None
) -> List[str]:
    """
    Filter baseline results to find samples with no errors (or very few).
    
    Args:
        baseline_results_path: Path to baseline results JSONL file
        max_errors: Maximum number of errors allowed (default: 0 for error-free)
        error_type_filter: If provided, only count errors of this type ("T", "B", "E")
        max_samples: Maximum number of samples to return (None for all)
        
    Returns:
        List of unique_dataset_id values for error-free samples
    """
    if not os.path.exists(baseline_results_path):
        raise FileNotFoundError(f"Baseline results file not found: {baseline_results_path}")
    
    error_free_ids = []
    error_counts = []
    
    with open(baseline_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                result = json.loads(line)
                
                # Skip error results
                if "error" in result:
                    continue
                
                unique_id = result.get("unique_dataset_id") or result.get("sample_id", "")
                if not unique_id:
                    continue
                
                # Count errors
                num_errors = result.get("num_errors", 0)
                
                # If filtering by error type, count only that type
                if error_type_filter and num_errors > 0:
                    errors = result.get("errors", {})
                    type_count = sum(
                        1 for error in errors.values()
                        if error.get("type") == error_type_filter
                    )
                    num_errors = type_count
                
                error_counts.append({
                    "unique_dataset_id": unique_id,
                    "num_errors": num_errors,
                    "total_errors": result.get("num_errors", 0)
                })
                
                if num_errors <= max_errors:
                    error_free_ids.append(unique_id)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {e}")
                continue
    
    # Sort by error count (ascending) to prioritize samples with fewer errors
    error_counts.sort(key=lambda x: x["num_errors"])
    
    # If max_samples is specified, take the best ones
    if max_samples and len(error_free_ids) > max_samples:
        # Get the IDs of samples with the fewest errors
        selected_ids = [ec["unique_dataset_id"] for ec in error_counts[:max_samples]]
        error_free_ids = selected_ids
    
    print(f"Found {len(error_free_ids)} samples with <= {max_errors} errors")
    if error_type_filter:
        print(f"  (filtered by {error_type_filter} error type)")
    if error_counts:
        error_distribution = {}
        for ec in error_counts:
            err_count = ec["num_errors"]
            error_distribution[err_count] = error_distribution.get(err_count, 0) + 1
        print(f"  Error distribution: {dict(sorted(error_distribution.items()))}")
    
    return error_free_ids


def filter_samples_by_ids(
    samples: List[Dict[str, Any]],
    valid_ids: Set[str]
) -> List[Dict[str, Any]]:
    """
    Filter samples to only include those with IDs in valid_ids.
    
    Args:
        samples: List of samples
        valid_ids: Set of unique_dataset_id values to keep
        
    Returns:
        Filtered list of samples
    """
    filtered = []
    for sample in samples:
        unique_id = sample.get("unique_dataset_id") or f"{sample.get('sample_id', 'unknown')}-{sample.get('model', 'unknown')}"
        if unique_id in valid_ids:
            filtered.append(sample)
    
    return filtered


if __name__ == "__main__":
    # Test filtering
    import sys
    if len(sys.argv) > 1:
        baseline_path = sys.argv[1]
        error_free_ids = filter_error_free_samples(baseline_path, max_errors=0)
        print(f"\nError-free sample IDs ({len(error_free_ids)}):")
        for i, sample_id in enumerate(error_free_ids[:10]):
            print(f"  {i+1}. {sample_id}")
        if len(error_free_ids) > 10:
            print(f"  ... and {len(error_free_ids) - 10} more")
    else:
        print("Usage: python filter_error_free.py <baseline_results.jsonl>")

