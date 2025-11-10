"""
Run CI Calibration Analysis

This script runs TrustScore pipeline with varying numbers of judges
to evaluate confidence interval calibration.
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
import random
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.orchestrator import TrustScorePipeline
from config.settings import TrustScoreConfig, JudgeConfig, SpanTaggerConfig, LLMProvider, AggregationWeights
from scripts.load_summeval import load_summeval_with_sources
from specificity_analysis.dual_logger import DualLogger, initialize_logging, cleanup_logging


# Configuration
NUM_EXAMPLES = 5  # 3-5 examples
JUDGE_COUNTS = [1, 3, 5]  # Number of trustworthiness judges
NUM_REPEATS = 5  # Repeats per (example, J) combination
SPAN_TAGGER_TEMPERATURE = 0.0  # Deterministic for consistent spans
JUDGE_TEMPERATURE = 0.7  # Stochastic for variability
CONFIDENCE_LEVEL = 0.95  # 95% CI

# Model configuration
VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
QWEN_MODEL = "Qwen/Qwen2-7B-Instruct"
MAX_TOKENS = 2000
DEVICE = "cuda"

# Google Drive configuration
IN_COLAB = os.path.exists("/content") and os.path.exists("/root")
DRIVE_MOUNT_PATH = "/content/drive"
DRIVE_RESULTS_PATH = "/content/drive/MyDrive/TrustScore_Results"


def mount_google_drive():
    """Mount Google Drive in Colab."""
    if IN_COLAB:
        try:
            from google.colab import drive
            print("Mounting Google Drive...")
            drive.mount(DRIVE_MOUNT_PATH)
            os.makedirs(DRIVE_RESULTS_PATH, exist_ok=True)
            print(f"✓ Google Drive mounted at {DRIVE_MOUNT_PATH}")
            return True
        except Exception as e:
            print(f"⚠ Could not mount Google Drive: {e}")
            return False
    return False


def save_to_drive(local_path: str, drive_path: Optional[str] = None):
    """Save file to Google Drive if in Colab and Drive is mounted."""
    if IN_COLAB and os.path.exists(DRIVE_MOUNT_PATH):
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


def create_config_for_judge_count(num_judges: int) -> TrustScoreConfig:
    """
    Create TrustScoreConfig with specified number of trustworthiness judges.
    Uses different models based on judge count:
    - 1 judge: LLaMA 3.1 8B
    - 3 judges: LLaMA 3.1 8B, Mistral 7B, Qwen 7B
    - 5 judges: 2 LLaMAs, 2 Mistrals, 1 Qwen
    
    Args:
        num_judges: Number of trustworthiness judges (1, 3, or 5)
        
    Returns:
        TrustScoreConfig instance
    """
    # Span Tagger Config with VLLM (temperature = 0.0 for consistency)
    span_tagger_config = SpanTaggerConfig(
        provider=LLMProvider.VLLM,
        model=VLLM_MODEL,
        model_path=VLLM_MODEL,
        temperature=SPAN_TAGGER_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        device=DEVICE,
        torch_dtype="float16"
    )
    
    # Judge Configs with VLLM - Use same model for all judges (VLLM limitation)
    # NOTE: VLLM can only load one model at a time in a single process
    # Using the same model with temperature=0.7 provides randomness via different seeds
    judge_configs = {}
    
    # Use same model for all judges (temperature variability provides randomness)
    model_list = [VLLM_MODEL] * num_judges
    
    # Create judge configs with assigned models
    for i in range(1, num_judges + 1):
        model = model_list[i - 1]  # 0-indexed
        judge_configs[f"trust_judge_{i}"] = JudgeConfig(
            name=f"trust_judge_{i}",
            enabled=True,
            provider=LLMProvider.VLLM,
            model=model,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    
    # No bias or explainability judges (as per experiment design)
    # But we still need to create disabled judges for the pipeline
    judge_configs["bias_judge_1"] = JudgeConfig(
        name="bias_judge_1",
        enabled=False,
        provider=LLMProvider.VLLM,
        model=VLLM_MODEL,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    judge_configs["explainability_judge"] = JudgeConfig(
        name="explainability_judge",
        enabled=False,
        provider=LLMProvider.VLLM,
        model=VLLM_MODEL,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    # Create full config
    config = TrustScoreConfig(
        span_tagger=span_tagger_config,
        judges=judge_configs,
        aggregation_weights=AggregationWeights(
            trustworthiness=0.6,
            explainability=0.3,
            bias=0.1
        ),
        confidence_level=CONFIDENCE_LEVEL
    )
    
    return config


def format_result_for_storage(result, sample: Dict[str, Any], run_id: str, num_judges: int, repeat: int) -> Dict[str, Any]:
    """
    Format pipeline result for storage.
    
    Args:
        result: AggregatedOutput from pipeline
        sample: Original sample data
        run_id: Unique run identifier
        num_judges: Number of judges used
        repeat: Repeat number (1-indexed)
        
    Returns:
        Formatted result dictionary
    """
    # Helper to format CI
    def format_ci(ci):
        if ci is None:
            return None
        return {
            "lower": ci.lower,
            "upper": ci.upper
        }
    
    # Extract all CI data at different levels
    result_data = {
        "item_id": sample.get("unique_dataset_id", f"{sample.get('sample_id', 'unknown')}-{sample.get('model', 'unknown')}"),
        "sample_id": sample.get("sample_id", "unknown"),
        "model_id": sample.get("model", "unknown"),
        "run_id": run_id,
        "num_judges": num_judges,
        "repeat": repeat,
        "confidence_level": CONFIDENCE_LEVEL,
        
        # Final TrustScore CIs
        "trust_score": result.summary.trust_score,
        "trust_score_ci": format_ci(result.summary.trust_score_ci),
        "trust_confidence": result.summary.trust_confidence,
        "trust_confidence_ci": format_ci(result.summary.trust_confidence_ci),
        
        # Category-level CIs (T only, since E and B will be 0)
        "agg_score_T": result.summary.agg_score_T,
        "agg_score_T_ci": format_ci(result.summary.agg_score_T_ci),
        "agg_confidence_T": result.summary.agg_confidence_T,
        "agg_confidence_T_ci": format_ci(result.summary.agg_confidence_T_ci),
        
        "agg_score_E": result.summary.agg_score_E,
        "agg_score_E_ci": format_ci(result.summary.agg_score_E_ci),
        "agg_confidence_E": result.summary.agg_confidence_E,
        "agg_confidence_E_ci": format_ci(result.summary.agg_confidence_E_ci),
        
        "agg_score_B": result.summary.agg_score_B,
        "agg_score_B_ci": format_ci(result.summary.agg_score_B_ci),
        "agg_confidence_B": result.summary.agg_confidence_B,
        "agg_confidence_B_ci": format_ci(result.summary.agg_confidence_B_ci),
        
        # Span-level CIs (for each error)
        "span_level_ci": []
    }
    
    # Add span-level CIs
    for error_id, error in result.errors.items():
        span_ci_data = {
            "error_id": error_id,
            "type": error.type.value,
            "subtype": error.subtype,
            "severity_score": error.severity_score,
            "severity_score_ci": format_ci(error.severity_score_ci),
            "confidence_level": error.confidence_level,
            "confidence_ci": format_ci(error.confidence_ci)
        }
        result_data["span_level_ci"].append(span_ci_data)
    
    # Metadata
    result_data["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "span_tagger_temperature": SPAN_TAGGER_TEMPERATURE,
        "judge_temperature": JUDGE_TEMPERATURE,
        "model": VLLM_MODEL,
        "num_spans_detected": len(result.errors)
    }
    
    return result_data


def run_ci_calibration_analysis(
    summeval_path: str,
    output_dir: str,
    num_examples: int = NUM_EXAMPLES,
    judge_counts: List[int] = JUDGE_COUNTS,
    num_repeats: int = NUM_REPEATS,
    use_mock: bool = False,
    api_key: Optional[str] = None,
    random_seed: int = 42
) -> str:
    """
    Run CI calibration analysis.
    
    Args:
        summeval_path: Path to SummEval JSONL file
        output_dir: Directory to save results
        num_examples: Number of examples to use (3-5)
        judge_counts: List of judge counts to test [1, 3, 5]
        num_repeats: Number of repeats per (example, J) combination
        use_mock: Whether to use mock components (for testing)
        api_key: API key for LLM providers
        random_seed: Random seed for reproducibility
        
    Returns:
        Path to results directory
    """
    # Set random seeds for reproducibility (sample selection, etc.)
    # Note: We do NOT set cudnn.deterministic=True to allow natural randomness for judges
    # The span tagger uses temperature=0.0 with seed=42 for determinism
    # Judges use temperature=0.7 with random seeds for variability
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        # DO NOT set cudnn.deterministic = True - this would force all operations to be deterministic
        # and prevent natural randomness in judge outputs
        torch.backends.cudnn.benchmark = False
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_subdir = f"ci_calibration_results_{timestamp}"
    full_output_dir = os.path.join(output_dir, results_subdir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Initialize logging
    logger = initialize_logging(full_output_dir, "execution.log")
    
    try:
        print("=" * 70)
        print("CI CALIBRATION ANALYSIS")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  - Number of examples: {num_examples}")
        print(f"  - Judge counts: {judge_counts}")
        print(f"  - Repeats per (example, J): {num_repeats}")
        print(f"  - Total runs: {num_examples * len(judge_counts) * num_repeats}")
        print(f"  - Span tagger temperature: {SPAN_TAGGER_TEMPERATURE}")
        print(f"  - Judge temperature: {JUDGE_TEMPERATURE}")
        print(f"  - Confidence level: {CONFIDENCE_LEVEL}")
        print(f"  - Model: {VLLM_MODEL}")
        print("=" * 70)
        
        # Load SummEval data
        print("\nLoading SummEval data...")
        all_samples = load_summeval_with_sources(summeval_path, max_samples=None)
        print(f"Loaded {len(all_samples)} samples")
        
        # Sort samples by unique_dataset_id for deterministic order
        all_samples.sort(key=lambda x: x.get("unique_dataset_id", ""))
        print("Sorted samples by unique_dataset_id for reproducibility")
        
        # Select random subset (now deterministic due to sorted order)
        selected_samples = random.sample(all_samples, min(num_examples, len(all_samples)))
        print(f"Selected {len(selected_samples)} samples for analysis")
        
        # Transform samples to TrustScore format
        # load_summeval_with_sources returns samples with 'summary' and 'source_article'
        # but pipeline expects 'prompt' and 'response'
        for sample in selected_samples:
            source_article = sample.get("source_article", "")
            summary = sample.get("summary", "")
            
            # Create prompt from source article
            if source_article:
                sample["prompt"] = f"Summarize the following article:\n\n{source_article}"
            else:
                sample["prompt"] = "Generate a summary of the article."
            
            # Map summary to response
            sample["response"] = summary
            
            # Map model_id to model for compatibility
            sample["model"] = sample.get("model_id", "unknown")
            sample["sample_id"] = sample.get("id", "unknown")
        
        print("Transformed samples to TrustScore format (prompt/response)")
        
        # Verify transformation worked
        for i, sample in enumerate(selected_samples, 1):
            if "prompt" not in sample or "response" not in sample:
                print(f"  ⚠ Warning: Sample {i} missing prompt/response after transformation")
                print(f"     Available keys: {list(sample.keys())}")
                print(f"     Has 'summary': {'summary' in sample}")
                print(f"     Has 'source_article': {'source_article' in sample}")
        
        # Save selected samples metadata
        samples_metadata = {
            "num_examples": len(selected_samples),
            "samples": [
                {
                    "item_id": s.get("unique_dataset_id", f"{s.get('sample_id', 'unknown')}-{s.get('model', 'unknown')}"),
                    "sample_id": s.get("sample_id", "unknown"),
                    "model_id": s.get("model", "unknown")
                }
                for s in selected_samples
            ]
        }
        metadata_path = os.path.join(full_output_dir, "samples_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(samples_metadata, f, indent=2)
        save_to_drive(metadata_path)
        
        # Prepare results file
        results_file = os.path.join(full_output_dir, "calibration_results.jsonl")
        
        # Run experiments
        total_runs = len(selected_samples) * len(judge_counts) * num_repeats
        run_counter = 0
        
        with open(results_file, 'w', encoding='utf-8') as f:
            for sample_idx, sample in enumerate(selected_samples, 1):
                print(f"\n{'='*70}")
                print(f"Processing Example {sample_idx}/{len(selected_samples)}")
                print(f"Item ID: {sample.get('unique_dataset_id', 'unknown')}")
                print(f"{'='*70}")
                
                for num_judges in judge_counts:
                    print(f"\n--- Config: {num_judges} judge(s) ---")
                    
                    # Determine models being used (all same model due to VLLM limitation)
                    models_str = f"{num_judges}× LLaMA 3.1 8B (same model, different seeds for randomness)"
                    print(f"  Models: {models_str}")
                    
                    # Create config for this judge count
                    config = create_config_for_judge_count(num_judges)
                    
                    # Initialize pipeline
                    pipeline = TrustScorePipeline(config=config, api_key=api_key, use_mock=use_mock)
                    
                    for repeat in tqdm(range(1, num_repeats + 1), desc=f"  {num_judges} judge(s), repeats", unit="repeat"):
                        run_counter += 1
                        
                        # Generate run ID
                        run_id = f"run_{run_counter}::judges_{num_judges}::repeat_{repeat}::sample_{sample_idx}::{sample.get('unique_dataset_id', 'unknown')}"
                        
                        try:
                            # Verify required fields exist before processing
                            if "prompt" not in sample:
                                raise KeyError(f"Sample missing 'prompt' field. Available keys: {list(sample.keys())}")
                            if "response" not in sample:
                                raise KeyError(f"Sample missing 'response' field. Available keys: {list(sample.keys())}")
                            
                            # Note: Judges use natural randomness (no seeds) for variability
                            # Span tagger uses temperature=0.0 with seed=42 for deterministic results
                            # This ensures:
                            # - Same spans detected across repeats (deterministic span tagger)
                            # - Different judge outputs across repeats (natural randomness with temperature=0.7)
                            
                            # Run pipeline (generation_seed only affects span tagger if needed)
                            result = pipeline.process(
                                prompt=sample["prompt"],
                                response=sample["response"],
                                model=sample.get("model", "unknown"),
                                generated_on=datetime.now(),
                                generation_seed=None  # Judges use natural randomness
                            )
                            
                            # Format and save result
                            result_data = format_result_for_storage(
                                result, sample, run_id, num_judges, repeat
                            )
                            
                            f.write(json.dumps(result_data) + '\n')
                            f.flush()
                            
                        except Exception as e:
                            print(f"\n  ✗ Error in run {run_id}: {str(e)}")
                            # Save error record
                            error_data = {
                                "item_id": sample.get("unique_dataset_id", "unknown"),
                                "run_id": run_id,
                                "num_judges": num_judges,
                                "repeat": repeat,
                                "error": str(e)
                            }
                            f.write(json.dumps(error_data) + '\n')
                            f.flush()
        
        # Save results to Drive
        save_to_drive(results_file)
        print(f"\n✓ Results saved to: {results_file}")
        
        print("\n" + "=" * 70)
        print("CI CALIBRATION ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"\nResults saved in: {full_output_dir}")
        if IN_COLAB:
            print(f"Results also saved to Google Drive: {DRIVE_RESULTS_PATH}")
        print(f"\nFiles created:")
        print(f"  - samples_metadata.json")
        print(f"  - calibration_results.jsonl")
        print(f"  - execution.log")
        
        return full_output_dir
        
    finally:
        # Cleanup logging
        if logger is not None:
            cleanup_logging(logger)
            # Save log file to Google Drive
            log_file_path = os.path.join(full_output_dir, "execution.log")
            if os.path.exists(log_file_path):
                save_to_drive(log_file_path)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CI Calibration Analysis")
    parser.add_argument("--summeval-path", type=str, required=True,
                        help="Path to SummEval JSONL file")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES,
                        help=f"Number of examples (default: {NUM_EXAMPLES})")
    parser.add_argument("--judge-counts", type=int, nargs="+", default=JUDGE_COUNTS,
                        help=f"Judge counts to test (default: {JUDGE_COUNTS})")
    parser.add_argument("--num-repeats", type=int, default=NUM_REPEATS,
                        help=f"Number of repeats per (example, J) (default: {NUM_REPEATS})")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--use-mock", action="store_true",
                        help="Use mock components (for testing)")
    
    args = parser.parse_args()
    
    # Mount Google Drive if in Colab
    mount_google_drive()
    
    # Run analysis
    run_ci_calibration_analysis(
        summeval_path=args.summeval_path,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        judge_counts=args.judge_counts,
        num_repeats=args.num_repeats,
        use_mock=args.use_mock,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()

