"""
Standalone script to generate sensitivity report without running full pipeline.
This can be imported without triggering the main analysis.
"""

import json
import os
from typing import Optional, Dict, Any
from sensitivity_analysis.sensitivity_metrics import calculate_sensitivity_report

# Configuration (should match your run)
K_MAX = 5
ERROR_SUBTYPES = {
    "T": ["factual_error", "hallucination"],
    "B": ["gender_bias", "sycophancy_bias"],
    "E": ["unclear_explanation", "missing_context"]
}


def find_results_file(base_dir: str, error_type: str, subtype: str, k: int) -> Optional[str]:
    """
    Find result file in multiple possible locations.
    
    Checks:
    1. Local results directory
    2. Google Drive (if mounted) - including timestamped directories
    3. Alternative paths
    """
    filename = f"{error_type}_{subtype}_k{k}_results.jsonl"
    
    # Try local path first
    local_path = os.path.join(base_dir, filename)
    if os.path.exists(local_path):
        return local_path
    
    # Try Google Drive if in Colab
    try:
        from google.colab import drive
        drive_base = "/content/drive/MyDrive/TrustScore_Results"
        
        # First, try timestamped directories (most recent first)
        if os.path.exists(drive_base):
            # Look for timestamped directories
            timestamped_dirs = []
            for item in os.listdir(drive_base):
                item_path = os.path.join(drive_base, item)
                if os.path.isdir(item_path) and item.startswith("sensitivity_analysis_"):
                    timestamped_dirs.append(item)
            
            # Sort by timestamp (newest first)
            timestamped_dirs.sort(reverse=True)
            
            # Try timestamped directories first
            for dir_name in timestamped_dirs:
                drive_path = os.path.join(drive_base, dir_name, filename)
                if os.path.exists(drive_path):
                    return drive_path
        
        # Fallback to non-timestamped directory
        drive_paths = [
            os.path.join(drive_base, "sensitivity_analysis", filename),
            os.path.join(drive_base, filename),
        ]
        for drive_path in drive_paths:
            if os.path.exists(drive_path):
                return drive_path
    except ImportError:
        pass
    
    return None


def generate_sensitivity_report_only(
    results_dir: Optional[str] = None,
    drive_results_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate sensitivity report from existing result files.
    
    Args:
        results_dir: Local results directory (default: "results/sensitivity_analysis")
        drive_results_dir: Google Drive results directory (optional)
        
    Returns:
        Dictionary with all sensitivity reports
    """
    # Determine results directory
    if results_dir is None:
        results_dir = "results/sensitivity_analysis"
    
    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"⚠ Local results directory not found: {results_dir}")
        print("Checking Google Drive...")
        
        # Try Google Drive
        try:
            from google.colab import drive
            if drive_results_dir is None:
                drive_results_dir = "/content/drive/MyDrive/TrustScore_Results/sensitivity_analysis"
            
            if os.path.exists(drive_results_dir):
                print(f"✓ Using Google Drive directory: {drive_results_dir}")
                results_dir = drive_results_dir
            else:
                print(f"⚠ Google Drive directory not found: {drive_results_dir}")
                print("\nPlease specify the correct path to your results directory.")
                return {}
        except ImportError:
            print("Not running in Colab. Please specify the correct results_dir path.")
            return {}
    
    print(f"Using results directory: {results_dir}")
    
    # List available files
    print("\nChecking for result files...")
    available_files = []
    for error_type, subtypes in ERROR_SUBTYPES.items():
        for subtype in subtypes:
            for k in range(0, K_MAX + 1):
                file_path = find_results_file(results_dir, error_type, subtype, k)
                if file_path:
                    available_files.append((error_type, subtype, k, file_path))
    
    if not available_files:
        print("⚠ No result files found!")
        print(f"\nExpected files in format: {{error_type}}_{{subtype}}_k{{k}}_results.jsonl")
        print(f"Example: T_factual_error_k0_results.jsonl")
        print(f"\nSearched in: {results_dir}")
        return {}
    
    print(f"✓ Found {len(available_files)} result files")
    
    # Generate reports
    print("\n" + "=" * 70)
    print("STEP 5: Calculating Sensitivity Metrics")
    print("=" * 70)
    
    sensitivity_reports = {}
    
    for error_type, subtypes in ERROR_SUBTYPES.items():
        for subtype in subtypes:
            try:
                print(f"\nCalculating sensitivity metrics for {error_type}_{subtype}...")
                
                # Use the find_results_file logic but pass the base directory
                # We'll modify calculate_sensitivity_report to use our finder
                report = calculate_sensitivity_report_with_finder(
                    results_dir=results_dir,
                    error_type=error_type,
                    subtype=subtype,
                    k_max=K_MAX,
                    find_results_file=find_results_file
                )
                
                if report:
                    sensitivity_reports[f"{error_type}_{subtype}"] = report
                    
                    # Print summary
                    target_gauge = report.get("target_gauge", {})
                    print(f"  Target dimension ({error_type}):")
                    print(f"    Kendall's tau: {target_gauge.get('kendall_tau', 'N/A'):.3f} (p={target_gauge.get('kendall_pvalue', 'N/A'):.4f})")
                    print(f"    Spearman's rho: {target_gauge.get('spearman_rho', 'N/A'):.3f} (p={target_gauge.get('spearman_pvalue', 'N/A'):.4f})")
                    print(f"    Monotonic: {target_gauge.get('monotonic', False)}")
                    
                    mean_scores = report.get("mean_scores_by_k", {})
                    print(f"  Mean scores by k: {mean_scores}")
                else:
                    print(f"  ⚠ Skipped {error_type}_{subtype} - insufficient data")
                
            except ValueError as e:
                if "Need at least 2 different k values" in str(e):
                    print(f"  ⚠ Skipped {error_type}_{subtype} - need at least 2 k values with results")
                else:
                    print(f"✗ Error calculating metrics for {error_type}_{subtype}: {str(e)}")
            except Exception as e:
                print(f"✗ Error calculating metrics for {error_type}_{subtype}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Save sensitivity report
    if sensitivity_reports:
        output_dir = results_dir  # Save report in same directory as results
        report_path = os.path.join(output_dir, "sensitivity_report.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(sensitivity_reports, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Sensitivity report saved to {report_path}")
        
        # Try to save to Google Drive if in Colab
        try:
            from google.colab import drive
            if os.path.exists("/content/drive"):
                drive_report_path = "/content/drive/MyDrive/TrustScore_Results/sensitivity_analysis/sensitivity_report.json"
                os.makedirs(os.path.dirname(drive_report_path), exist_ok=True)
                import shutil
                shutil.copy2(report_path, drive_report_path)
                print(f"✓ Also saved to Google Drive: {drive_report_path}")
        except ImportError:
            pass
        
        return sensitivity_reports
    else:
        print("\n⚠ No reports generated - check that result files exist")
        return {}


def calculate_sensitivity_report_with_finder(
    results_dir: str,
    error_type: str,
    subtype: str,
    k_max: int,
    find_results_file
) -> Optional[Dict[str, Any]]:
    """
    Calculate sensitivity report using custom file finder.
    Returns None if insufficient data.
    """
    from sensitivity_analysis.sensitivity_metrics import load_results, calculate_monotonicity_metrics
    
    # Load results for each k
    results_by_k: Dict[int, list] = {}
    
    for k in range(0, k_max + 1):
        result_file = find_results_file(results_dir, error_type, subtype, k)
        if result_file:
            try:
                results_by_k[k] = load_results(result_file)
            except Exception as e:
                print(f"  [WARNING] Failed to load results for k={k}: {str(e)}")
                results_by_k[k] = []
        else:
            results_by_k[k] = []
    
    # Check if we have enough data
    k_values_with_data = [k for k, results in results_by_k.items() if results]
    if len(k_values_with_data) < 2:
        return None
    
    # Calculate monotonicity metrics
    try:
        metrics = calculate_monotonicity_metrics(results_by_k, error_type)
    except ValueError as e:
        if "Need at least 2 different k values" in str(e):
            return None
        raise
    
    return {
        "dimension": error_type,
        "subtype": subtype,
        "k_max": k_max,
        **metrics
    }


if __name__ == "__main__":
    # Run report generation
    print("=" * 70)
    print("SENSITIVITY ANALYSIS - REPORT GENERATION ONLY")
    print("=" * 70)
    
    reports = generate_sensitivity_report_only()
    
    if reports:
        print("\n" + "=" * 70)
        print("REPORT GENERATION COMPLETE!")
        print("=" * 70)
        print(f"\nGenerated reports for {len(reports)} subtype(s):")
        for key in reports.keys():
            print(f"  - {key}")
    else:
        print("\n" + "=" * 70)
        print("REPORT GENERATION FAILED")
        print("=" * 70)
        print("\nNo reports were generated. Please check:")
        print("  1. Result files exist in the expected location")
        print("  2. Files are named correctly: {error_type}_{subtype}_k{k}_results.jsonl")
        print("  3. At least 2 k values have result files for each subtype")

