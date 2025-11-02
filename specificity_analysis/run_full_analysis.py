"""
Main orchestrator script for full specificity analysis
"""

import os
import argparse
from typing import Optional
from load_dataset import load_and_sample_dataset, save_samples
from error_injector import ErrorInjector
from baseline_inference import run_baseline_inference
from perturbed_inference import run_perturbed_inference
from score_comparison import compare_scores, generate_report


def run_full_analysis(
    dataset_name: str = "summeval",
    num_samples: int = 50,
    use_mock: bool = False,
    api_key: Optional[str] = None,
    output_dir: str = "results/specificity_analysis",
    skip_baseline: bool = False,
    skip_perturbation: bool = False
):
    """
    Run complete specificity analysis pipeline.
    
    Args:
        dataset_name: Dataset to use ("summeval" or "cnn_dailymail")
        num_samples: Number of samples to analyze
        use_mock: Use mock components (for testing)
        api_key: API key for LLM providers
        output_dir: Directory to save all results
        skip_baseline: Skip baseline inference if already done
        skip_perturbation: Skip error injection if already done
    """
    print("=" * 70)
    print("TRUSTSCORE SPECIFICITY ANALYSIS")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 0: Load and sample dataset
    print("\n[Step 0] Loading and sampling dataset...")
    samples = load_and_sample_dataset(
        dataset_name=dataset_name,
        max_samples=num_samples
    )
    print(f"✓ Loaded {len(samples)} samples")
    
    samples_path = os.path.join(output_dir, "sampled_dataset.jsonl")
    save_samples(samples, samples_path)
    
    # Step 1: Baseline inference
    baseline_path = os.path.join(output_dir, "baseline_results.jsonl")
    
    if not skip_baseline or not os.path.exists(baseline_path):
        print("\n[Step 1] Running baseline TrustScore inference...")
        baseline_results = run_baseline_inference(
            samples=samples,
            output_path=baseline_path,
            use_mock=use_mock,
            api_key=api_key
        )
    else:
        print("\n[Step 1] Skipping baseline inference (results already exist)")
    
    # Step 2: Create perturbed datasets
    print("\n[Step 2] Creating perturbed datasets...")
    
    injector = None
    if not use_mock:
        try:
            injector = ErrorInjector(
                api_key=api_key
            )
        except Exception as e:
            print(f"Warning: Could not initialize error injector: {e}")
            print("Falling back to mock mode for error injection")
            use_mock = True
    
    error_types = ["T", "B", "E", "PLACEBO"]
    perturbed_datasets = {}
    
    for error_type in error_types:
        perturbed_path = os.path.join(output_dir, f"{error_type}_perturbed.jsonl")
        
        if not skip_perturbation or not os.path.exists(perturbed_path):
            print(f"\n  Creating {error_type}_perturbed dataset...")
            
            if use_mock or injector is None:
                # For mock mode, just copy samples (no actual injection)
                import json
                perturbed_samples = []
                for sample in samples:
                    perturbed = sample.copy()
                    perturbed["error_type_injected"] = error_type
                    # Ensure unique_dataset_id is preserved
                    if "unique_dataset_id" not in perturbed:
                        perturbed["unique_dataset_id"] = f"{perturbed.get('sample_id', 'unknown')}-{perturbed.get('model', 'unknown')}"
                    # In mock mode, we can add a simple marker to the response
                    if "response" in sample:
                        if error_type == "PLACEBO":
                            # For placebo in mock mode, just add whitespace
                            perturbed["response"] = sample["response"] + " \n"
                        else:
                            perturbed["response"] = sample["response"] + f" [MOCK_{error_type}_ERROR]"
                    perturbed_samples.append(perturbed)
                
                with open(perturbed_path, 'w', encoding='utf-8') as f:
                    for sample in perturbed_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
                perturbed_datasets[error_type] = perturbed_samples
            else:
                perturbed_samples = injector.create_perturbed_dataset(
                    samples=samples,
                    error_type=error_type
                )
                
                import json
                with open(perturbed_path, 'w', encoding='utf-8') as f:
                    for sample in perturbed_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
                perturbed_datasets[error_type] = perturbed_samples
            
            print(f"  ✓ Created {error_type}_perturbed dataset ({len(perturbed_samples)} samples)")
        else:
            print(f"\n  Skipping {error_type}_perturbed creation (already exists)")
            # Load existing
            import json
            perturbed_samples = []
            with open(perturbed_path, 'r', encoding='utf-8') as f:
                for line in f:
                    perturbed_samples.append(json.loads(line.strip()))
            perturbed_datasets[error_type] = perturbed_samples
    
    # Step 3: Run TrustScore on perturbed datasets
    print("\n[Step 3] Running TrustScore on perturbed datasets...")
    
    perturbed_results_paths = {}
    for error_type in error_types:
        perturbed_path = os.path.join(output_dir, f"{error_type}_perturbed_results.jsonl")
        perturbed_results_paths[error_type] = perturbed_path
        
        if not os.path.exists(perturbed_path):
            run_perturbed_inference(
                perturbed_samples=perturbed_datasets[error_type],
                output_path=perturbed_path,
                error_type=error_type,
                use_mock=use_mock,
                api_key=api_key
            )
        else:
            print(f"  Skipping {error_type}_perturbed inference (results already exist)")
    
    # Step 4: Compare scores
    print("\n[Step 4] Comparing scores...")
    
    comparisons = {}
    for error_type in error_types:
        try:
            comparison = compare_scores(
                baseline_results_path=baseline_path,
                perturbed_results_path=perturbed_results_paths[error_type],
                error_type=error_type
            )
            comparisons[error_type] = comparison
        except Exception as e:
            print(f"  Error comparing {error_type}: {str(e)}")
            comparisons[error_type] = {"error": str(e)}
    
    # Generate final report
    report_path = os.path.join(output_dir, "specificity_report.json")
    generate_report(comparisons, report_path)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TrustScore specificity analysis")
    parser.add_argument("--dataset", default="summeval", choices=["summeval", "cnn_dailymail"])
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to analyze")
    parser.add_argument("--use-mock", action="store_true", help="Use mock components (for testing)")
    parser.add_argument("--api-key", type=str, default=None, help="API key for LLM providers")
    parser.add_argument("--output-dir", default="results/specificity_analysis", help="Output directory")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline inference if results exist")
    parser.add_argument("--skip-perturbation", action="store_true", help="Skip error injection if datasets exist")
    
    args = parser.parse_args()
    
    run_full_analysis(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        use_mock=args.use_mock,
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        output_dir=args.output_dir,
        skip_baseline=args.skip_baseline,
        skip_perturbation=args.skip_perturbation
    )
