"""
Generate CI Calibration Report

This script generates a comprehensive calibration report from existing results.
"""

import json
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ci_calibration_analysis.ci_calibration_utils import (
    load_calibration_results,
    analyze_ci_level,
    make_json_serializable
)
from specificity_analysis.dual_logger import DualLogger, initialize_logging, cleanup_logging


# Configuration
CONFIDENCE_LEVEL = 0.95  # 95% CI
DRIVE_RESULTS_PATH = "/content/drive/MyDrive/TrustScore_Results"
IN_COLAB = os.path.exists("/content") and os.path.exists("/root")


def save_to_drive(local_path: str, drive_path: Optional[str] = None):
    """Save file to Google Drive if in Colab and Drive is mounted."""
    if IN_COLAB and os.path.exists("/content/drive"):
        try:
            if drive_path is None:
                # Auto-generate drive path from local path
                rel_path = os.path.relpath(local_path, "results")
                drive_path = os.path.join(DRIVE_RESULTS_PATH, rel_path)
            
            drive_dir = os.path.dirname(drive_path)
            os.makedirs(drive_dir, exist_ok=True)
            
            # Copy file to drive
            import shutil
            shutil.copy2(local_path, drive_path)
            print(f"  ✓ Saved to Google Drive: {drive_path}")
            return drive_path
        except Exception as e:
            print(f"  ⚠ Could not save to Google Drive: {e}")
            return None
    elif IN_COLAB:
        print(f"  ⚠ Google Drive not mounted - skipping Drive save")
    return None


def find_result_files(results_dir: str) -> Optional[str]:
    """
    Find calibration_results.jsonl in results directory or subdirectories.
    
    Args:
        results_dir: Root results directory
        
    Returns:
        Path to calibration_results.jsonl or None if not found
    """
    # Try direct path first
    direct_path = os.path.join(results_dir, "calibration_results.jsonl")
    if os.path.exists(direct_path):
        return direct_path
    
    # Try subdirectories
    for root, dirs, files in os.walk(results_dir):
        if "calibration_results.jsonl" in files:
            return os.path.join(root, "calibration_results.jsonl")
    
    # Try Google Drive path
    if IN_COLAB and os.path.exists("/content/drive"):
        drive_path = os.path.join(DRIVE_RESULTS_PATH, "calibration_results.jsonl")
        if os.path.exists(drive_path):
            return drive_path
        
        # Try subdirectories in Drive
        for root, dirs, files in os.walk(DRIVE_RESULTS_PATH):
            if "calibration_results.jsonl" in files:
                return os.path.join(root, "calibration_results.jsonl")
    
    return None


def generate_ci_calibration_report(
    results_path: Optional[str] = None,
    results_dir: str = "results",
    output_path: Optional[str] = None,
    confidence_level: float = CONFIDENCE_LEVEL
) -> str:
    """
    Generate CI calibration report from existing results.
    
    Args:
        results_path: Path to calibration_results.jsonl (if None, searches)
        results_dir: Directory to search for results
        output_path: Path to save report (if None, saves in results_dir)
        confidence_level: Expected confidence level (default: 0.95)
        
    Returns:
        Path to generated report
    """
    print("=" * 70)
    print("GENERATING CI CALIBRATION REPORT")
    print("=" * 70)
    
    # Find results file
    if results_path is None:
        results_path = find_result_files(results_dir)
    
    if results_path is None or not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Could not find calibration_results.jsonl. "
            f"Searched in: {results_dir}"
        )
    
    print(f"Loading results from: {results_path}")
    
    # Load results
    results = load_calibration_results(results_path)
    print(f"Loaded {len(results)} results")
    
    if len(results) == 0:
        raise ValueError("No results found in file")
    
    # Extract metadata
    metadata = {
        "results_file": results_path,
        "num_runs": len(results),
        "confidence_level": confidence_level,
        "generated_at": datetime.now().isoformat()
    }
    
    # Extract unique configurations
    unique_items = set()
    unique_judge_counts = set()
    error_count = 0
    missing_item_id = 0
    missing_num_judges = 0
    
    for result in results:
        if "error" in result:
            error_count += 1
            continue
        
        item_id = result.get("item_id")
        num_judges = result.get("num_judges")
        
        if item_id is None:
            missing_item_id += 1
            continue
            
        if num_judges is None:
            missing_num_judges += 1
            continue
        
        unique_items.add(item_id)
        unique_judge_counts.add(num_judges)
    
    metadata["num_unique_items"] = len(unique_items)
    metadata["judge_counts"] = sorted([x for x in unique_judge_counts if x is not None])
    
    # Debug information
    if error_count > 0:
        print(f"  ⚠ Warning: {error_count} results had errors and were skipped")
    if missing_item_id > 0:
        print(f"  ⚠ Warning: {missing_item_id} results missing 'item_id' field")
    if missing_num_judges > 0:
        print(f"  ⚠ Warning: {missing_num_judges} results missing 'num_judges' field")
    
    print(f"\nAnalysis configuration:")
    print(f"  - Number of runs: {metadata['num_runs']}")
    print(f"  - Number of unique items: {metadata['num_unique_items']}")
    print(f"  - Judge counts: {metadata['judge_counts']}")
    print(f"  - Confidence level: {confidence_level}")
    print("=" * 70)
    
    # Define CI levels to analyze
    ci_levels = [
        # Final TrustScore CIs (Quality space - primary analysis)
        {
            "name": "trust_score_quality",
            "ci_field": "trust_quality_ci",
            "point_field": "trust_quality",
            "description": "Final TrustScore quality CI (quality space [0-100])"
        },
        # Final TrustScore CIs (Severity space - for backward compatibility)
        {
            "name": "trust_score_severity",
            "ci_field": "trust_score_ci",
            "point_field": "trust_score",
            "description": "Final TrustScore severity CI (severity space)"
        },
        {
            "name": "trust_score_confidence",
            "ci_field": "trust_confidence_ci",
            "point_field": "trust_confidence",
            "description": "Final TrustScore confidence CI (probability space)"
        },
        # Category-level CIs (T only, since E and B will be 0)
        # Quality space (primary analysis)
        {
            "name": "category_T_quality",
            "ci_field": "agg_quality_T_ci",
            "point_field": "agg_quality_T",
            "description": "Category-level T quality CI (quality space [0-100])"
        },
        # Severity space (for backward compatibility)
        {
            "name": "category_T_severity",
            "ci_field": "agg_score_T_ci",
            "point_field": "agg_score_T",
            "description": "Category-level T severity CI (severity space)"
        },
        {
            "name": "category_T_confidence",
            "ci_field": "agg_confidence_T_ci",
            "point_field": "agg_confidence_T",
            "description": "Category-level T confidence CI (probability space)"
        },
    ]
    
    print("\nAnalyzing CI levels...")
    
    # Analyze each CI level
    analyses = {}
    for ci_level_config in tqdm(ci_levels, desc="CI levels", unit="level"):
        name = ci_level_config["name"]
        print(f"\n  Analyzing {name}...")
        
        analysis = analyze_ci_level(
            results=results,
            ci_field=ci_level_config["ci_field"],
            point_field=ci_level_config["point_field"],
            confidence_level=confidence_level
        )
        analysis["description"] = ci_level_config["description"]
        analyses[name] = analysis
        
        # Print summary
        print(f"    - Items analyzed: {len(analysis['by_item_judge'])}")
        print(f"    - Judge counts: {list(analysis['by_judge_count'].keys())}")
    
    # Also analyze span-level CIs (aggregate across all spans)
    print("\n  Analyzing span-level CIs...")
    
    span_analyses = {
        "span_severity": {
            "description": "Span-level severity CI (severity space, aggregated)",
            "by_item_judge": {},
            "by_judge_count": {}
        },
        "span_confidence": {
            "description": "Span-level confidence CI (probability space, aggregated)",
            "by_item_judge": {},
            "by_judge_count": {}
        }
    }
    
    # Group results by (item_id, num_judges, repeat)
    span_grouped = {}
    for result in results:
        if "error" in result:
            continue
        
        item_id = result.get("item_id")
        num_judges = result.get("num_judges")
        repeat = result.get("repeat")
        
        if item_id is None or num_judges is None:
            continue
        
        key = (item_id, num_judges, repeat)
        if key not in span_grouped:
            span_grouped[key] = {
                "item_id": item_id,
                "num_judges": num_judges,
                "span_severity_cis": [],
                "span_severity_points": [],
                "span_confidence_cis": [],
                "span_confidence_points": []
            }
        
        # Extract span-level CIs
        span_level_ci = result.get("span_level_ci", [])
        for span_ci_data in span_level_ci:
            # Don't store with 3-tuple keys here - aggregation happens later with proper 2-tuple keys
            # The span_grouped dict will be used to create properly keyed entries in the analysis section
            
            if span_ci_data.get("severity_score_ci"):
                span_grouped[key]["span_severity_cis"].append(span_ci_data["severity_score_ci"])
                span_grouped[key]["span_severity_points"].append(span_ci_data["severity_score"])
            
            if span_ci_data.get("confidence_ci"):
                span_grouped[key]["span_confidence_cis"].append(span_ci_data["confidence_ci"])
                span_grouped[key]["span_confidence_points"].append(span_ci_data["confidence_level"])
    
    # Analyze span-level (aggregate per run, then across runs)
    for span_type in ["span_severity", "span_confidence"]:
        ci_field = f"{span_type.replace('span_', '')}_ci"  # severity_score_ci or confidence_ci
        point_field = f"{span_type.replace('span_', '')}_points"  # severity_score or confidence_level
        
        # Re-aggregate by (item_id, num_judges)
        item_judge_grouped = {}
        for (item_id, num_judges, repeat), group_data in span_grouped.items():
            item_judge_key = (item_id, num_judges)
            if item_judge_key not in item_judge_grouped:
                item_judge_grouped[item_judge_key] = {
                    "item_id": item_id,
                    "num_judges": num_judges,
                    "all_cis": [],
                    "all_points": []
                }
            
            if span_type == "span_severity":
                item_judge_grouped[item_judge_key]["all_cis"].extend(group_data["span_severity_cis"])
                item_judge_grouped[item_judge_key]["all_points"].extend(group_data["span_severity_points"])
            else:
                item_judge_grouped[item_judge_key]["all_cis"].extend(group_data["span_confidence_cis"])
                item_judge_grouped[item_judge_key]["all_points"].extend(group_data["span_confidence_points"])
        
        # Analyze each (item, judge_count)
        for (item_id, num_judges), group_data in item_judge_grouped.items():
            ci_list = group_data["all_cis"]
            points = group_data["all_points"]
            
            if len(points) < 2:
                continue
            
            # Compute statistics
            from ci_calibration_analysis.ci_calibration_utils import (
                compute_ci_width, compute_std, check_coverage
            )
            import numpy as np
            
            ci_widths = [compute_ci_width(ci) for ci in ci_list]
            ci_widths = [w for w in ci_widths if w is not None]
            mean_ci_width = np.mean(ci_widths) if len(ci_widths) > 0 else None
            observed_std = compute_std(points)
            coverage, num_covered, total_valid = check_coverage(ci_list, points)
            
            item_judge_key = f"{item_id}_J{num_judges}"
            span_analyses[span_type]["by_item_judge"][item_judge_key] = {
                "item_id": item_id,
                "num_judges": num_judges,
                "num_spans": len(points),
                "mean_ci_width": float(mean_ci_width) if mean_ci_width is not None else None,
                "observed_std": float(observed_std),
                "coverage": float(coverage),
                "num_covered": num_covered,
                "total_valid": total_valid
            }
            
            # Aggregate by judge count
            if num_judges not in span_analyses[span_type]["by_judge_count"]:
                span_analyses[span_type]["by_judge_count"][num_judges] = {
                    "ci_widths": [],
                    "observed_stds": [],
                    "num_items": 0
                }
            
            if mean_ci_width is not None:
                span_analyses[span_type]["by_judge_count"][num_judges]["ci_widths"].append(mean_ci_width)
                span_analyses[span_type]["by_judge_count"][num_judges]["observed_stds"].append(observed_std)
                span_analyses[span_type]["by_judge_count"][num_judges]["num_items"] += 1
    
    # Finalize span-level aggregation
    for span_type in ["span_severity", "span_confidence"]:
        for judge_count, judge_data in span_analyses[span_type]["by_judge_count"].items():
            if len(judge_data["ci_widths"]) > 0:
                import numpy as np
                judge_data["mean_ci_width"] = float(np.mean(judge_data["ci_widths"]))
                judge_data["mean_observed_std"] = float(np.mean(judge_data["observed_stds"]))
                
                # Compute overall coverage
                # Only include keys that have the expected structure (string keys with proper fields)
                all_coverage = [
                    span_analyses[span_type]["by_item_judge"][key]["coverage"]
                    for key in span_analyses[span_type]["by_item_judge"]
                    if (isinstance(key, str) and  # Only process string keys (not 3-tuple keys from old code)
                        "coverage" in span_analyses[span_type]["by_item_judge"][key] and
                        span_analyses[span_type]["by_item_judge"][key].get("num_judges") == judge_count)
                ]
                if len(all_coverage) > 0:
                    judge_data["mean_coverage"] = float(np.mean(all_coverage))
    
    analyses.update(span_analyses)
    
    # Compile report
    report = {
        "metadata": metadata,
        "analyses": make_json_serializable(analyses),
        "summary": {}
    }
    
    # Generate summary
    print("\nGenerating summary...")
    
    summary = {
        "confidence_level": confidence_level,
        "num_runs": metadata["num_runs"],
        "num_unique_items": metadata["num_unique_items"],
        "judge_counts": metadata["judge_counts"],
        "by_judge_count": {}
    }
    
    # Aggregate summary by judge count
    for judge_count in metadata["judge_counts"]:
        judge_summary = {
            "num_judges": judge_count,
            "ci_levels": {}
        }
        
        for ci_level_name, analysis in analyses.items():
            if judge_count in analysis["by_judge_count"]:
                judge_data = analysis["by_judge_count"][judge_count]
                judge_summary["ci_levels"][ci_level_name] = {
                    "mean_ci_width": judge_data.get("mean_ci_width"),
                    "mean_observed_std": judge_data.get("mean_observed_std"),
                    "mean_coverage": judge_data.get("mean_coverage"),
                    "pearson_r": judge_data.get("pearson_r"),
                    "spearman_r": judge_data.get("spearman_r"),
                    "num_items": judge_data.get("num_items", 0)
                }
        
        summary["by_judge_count"][judge_count] = judge_summary
    
    report["summary"] = make_json_serializable(summary)
    
    # Save report
    if output_path is None:
        results_dir = os.path.dirname(results_path) if results_path else "results"
        output_path = os.path.join(results_dir, "ci_calibration_report.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Report saved to: {output_path}")
    
    # Save to Drive
    save_to_drive(output_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("REPORT SUMMARY")
    print("=" * 70)
    print(f"\nConfidence Level: {confidence_level}")
    print(f"Total Runs: {metadata['num_runs']}")
    print(f"Unique Items: {metadata['num_unique_items']}")
    print(f"Judge Counts: {metadata['judge_counts']}")
    
    print("\nBy Judge Count:")
    for judge_count in sorted(metadata["judge_counts"]):
        print(f"\n  J={judge_count}:")
        if judge_count in summary["by_judge_count"]:
            judge_summary = summary["by_judge_count"][judge_count]
            for ci_level, level_data in judge_summary["ci_levels"].items():
                print(f"    {ci_level}:")
                print(f"      Mean CI Width: {level_data.get('mean_ci_width', 'N/A')}")
                print(f"      Mean Observed Std: {level_data.get('mean_observed_std', 'N/A')}")
                print(f"      Mean Coverage: {level_data.get('mean_coverage', 'N/A')}")
                print(f"      Correlation (Pearson): {level_data.get('pearson_r', 'N/A')}")
                print(f"      Correlation (Spearman): {level_data.get('spearman_r', 'N/A')}")
    
    print("\n" + "=" * 70)
    
    return output_path


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CI Calibration Report")
    parser.add_argument("--results-path", type=str, default=None,
                        help="Path to calibration_results.jsonl (if None, searches)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to search for results (default: results)")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path to save report (if None, saves in results_dir)")
    parser.add_argument("--confidence-level", type=float, default=CONFIDENCE_LEVEL,
                        help=f"Expected confidence level (default: {CONFIDENCE_LEVEL})")
    
    args = parser.parse_args()
    
    # Generate report
    generate_ci_calibration_report(
        results_path=args.results_path,
        results_dir=args.results_dir,
        output_path=args.output_path,
        confidence_level=args.confidence_level
    )


if __name__ == "__main__":
    main()

