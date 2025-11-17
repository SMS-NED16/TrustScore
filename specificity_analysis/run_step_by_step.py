"""
Step-by-step specificity analysis pipeline
Run each step manually and inspect results
Supports VLLM/LLaMA from HuggingFace with minimized randomness and Google Drive integration
"""

import os
import json
import random
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Set seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    # Enable deterministic CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from specificity_analysis.load_dataset import load_and_sample_dataset, save_samples
from specificity_analysis.error_injector import ErrorInjector
from specificity_analysis.baseline_inference import run_baseline_inference
from specificity_analysis.perturbed_inference import run_perturbed_inference
from specificity_analysis.score_comparison import compare_scores, generate_report
from specificity_analysis.dual_logger import initialize_logging, cleanup_logging
from specificity_analysis.filter_error_free import filter_error_free_samples, filter_samples_by_ids

from config.settings import (
    TrustScoreConfig, SpanTaggerConfig, JudgeConfig, 
    LLMProvider, AggregationWeights
)

# Google Drive integration
try:
    from google.colab import drive
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    # Don't print - might not be running in Colab intentionally

# VLLM Configuration (for Google Colab)
VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # Or your preferred model
TEMPERATURE = 0.0  # Deterministic (0.0 = greedy decoding)
MAX_TOKENS = 4096  # Increased to ensure complete JSON responses
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Google Drive paths
DRIVE_MOUNT_PATH = "/content/drive"
DRIVE_RESULTS_PATH = "/content/drive/MyDrive/TrustScore_Results"

# Configuration
MAX_SAMPLES = 100  # Run with 100 samples for baseline inference
MAX_FILTERED_SAMPLES = None  # Maximum number of error-free samples to use (None = use all available)
MAX_ERRORS_FOR_FILTER = 0  # Maximum errors allowed for filtering (0 = error-free only)

def mount_google_drive():
    """Mount Google Drive in Colab."""
    if IN_COLAB:
        print("Mounting Google Drive...")
        drive.mount(DRIVE_MOUNT_PATH)
        os.makedirs(DRIVE_RESULTS_PATH, exist_ok=True)
        print(f"✓ Google Drive mounted at {DRIVE_MOUNT_PATH}")
        print(f"✓ Results will be saved to {DRIVE_RESULTS_PATH}")
        return True
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

def create_vllm_config_for_pipeline():
    """Create TrustScoreConfig with VLLM provider for all components."""
    
    # Span Tagger Config with VLLM
    span_tagger_config = SpanTaggerConfig(
        provider=LLMProvider.VLLM,
        model=VLLM_MODEL,
        model_path=VLLM_MODEL,  # HuggingFace model ID
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        device=DEVICE,
        torch_dtype="float16"
    )
    
    # Judge Configs with VLLM (all use same model for consistency)
    # VLLM uses model name, not model_path
    judge_configs = {
        "trust_judge_1": JudgeConfig(
            name="trust_judge_1",
            enabled=True,
            provider=LLMProvider.VLLM,
            model=VLLM_MODEL,  # VLLM uses model field
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        ),
        "bias_judge_1": JudgeConfig(
            name="bias_judge_1",
            enabled=True,
            provider=LLMProvider.VLLM,
            model=VLLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        ),
        "explain_judge_1": JudgeConfig(
            name="explain_judge_1",
            enabled=True,
            provider=LLMProvider.VLLM,
            model=VLLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        ),
    }
    
    # Create full config
    config = TrustScoreConfig(
        span_tagger=span_tagger_config,
        judges=judge_configs,
        aggregation_weights=AggregationWeights(
            trustworthiness=0.6,
            explainability=0.3,
            bias=0.1
        )
    )
    
    return config

def create_vllm_error_injector():
    """Create ErrorInjector with VLLM provider."""
    injector = ErrorInjector(
        provider=LLMProvider.VLLM,
        model=VLLM_MODEL,
        model_path=VLLM_MODEL,
        temperature=TEMPERATURE,  # Deterministic
        device=DEVICE
    )
    return injector

# Setup output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results/specificity_analysis_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to: {output_dir}")

# Initialize dual logging (console + file)
logger = None
try:
    logger = initialize_logging(output_dir, "execution.log")
    print(f"Logging initialized - output will be saved to: {os.path.join(output_dir, 'execution.log')}")
except Exception as e:
    print(f"Warning: Could not initialize file logging: {e}")
    print("Continuing with console-only output...")
    logger = None

# Mount Google Drive if in Colab (user can do this manually if needed)
# Don't auto-mount to avoid blocking
drive_mounted = False
if IN_COLAB and os.path.exists(DRIVE_MOUNT_PATH):
    drive_mounted = True
elif IN_COLAB:
    print("⚠ Google Drive not mounted. Run mount_google_drive() manually if needed.")

print("=" * 70)
print("SPECIFICITY ANALYSIS - STEP BY STEP")
print("=" * 70)
print(f"Random seed: {RANDOM_SEED}")
print(f"Temperature: {TEMPERATURE} (deterministic)")
print(f"Model: {VLLM_MODEL}")
print(f"Device: {DEVICE}")
if IN_COLAB:
    print(f"Running in Colab: {IN_COLAB}")
    print(f"Google Drive mounted: {drive_mounted}")
print("=" * 70)

# ============================================================================
# STEP 0: Sample observations from SummEval
# ============================================================================
print("\n" + "=" * 70)
print("STEP 0: Sampling Observations from SummEval")
print("=" * 70)

samples = load_and_sample_dataset(
    dataset_name="summeval",
    max_samples=MAX_SAMPLES,
    random_seed=RANDOM_SEED
)

print(f"\n✓ Loaded {len(samples)} samples")

# Save sampled dataset
samples_path = os.path.join(output_dir, "sampled_dataset.jsonl")
save_samples(samples, samples_path)
save_to_drive(samples_path)

# Inspect first sample
if samples:
    print("\nFirst sample preview:")
    print(f"  Sample ID (article): {samples[0]['sample_id']}")
    print(f"  Unique Dataset ID: {samples[0].get('unique_dataset_id', 'N/A')}")
    print(f"  Model: {samples[0]['model']}")
    print(f"  Prompt length: {len(samples[0]['prompt'])} chars")
    print(f"  Response length: {len(samples[0]['response'])} chars")
    print(f"  Response preview: {samples[0]['response'][:200]}...")

# ============================================================================
# STEP 1: Run TrustScore on Baseline (Original Responses)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Baseline TrustScore Inference with VLLM")
print("=" * 70)
print(f"Model: {VLLM_MODEL}")
print(f"Temperature: {TEMPERATURE}")
print("=" * 70)

baseline_path = os.path.join(output_dir, "baseline_results.jsonl")

# Create VLLM config for pipeline
vllm_config = create_vllm_config_for_pipeline()

# Run baseline inference with VLLM
baseline_results = run_baseline_inference(
    samples=samples,
    output_path=baseline_path,
    use_mock=False,  # Use real VLLM
    config=vllm_config
)

save_to_drive(baseline_path)
print(f"\n✓ Baseline results saved to {baseline_path}")

# Inspect first baseline result
if baseline_results and "error" not in baseline_results[0]:
    print(f"\n  First baseline result:")
    print(f"    TrustScore: {baseline_results[0].get('trust_score', 'N/A'):.3f}")
    print(f"    T: {baseline_results[0].get('agg_score_T', 'N/A'):.3f}")
    print(f"    E: {baseline_results[0].get('agg_score_E', 'N/A'):.3f}")
    print(f"    B: {baseline_results[0].get('agg_score_B', 'N/A'):.3f}")
    print(f"    Errors found: {baseline_results[0].get('num_errors', 0)}")

# ============================================================================
# STEP 1.5: Filter for Error-Free Samples
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1.5: Filtering for Error-Free Samples")
print("=" * 70)
print(f"Filtering criteria:")
print(f"  - Maximum errors allowed: {MAX_ERRORS_FOR_FILTER}")
print(f"  - Maximum samples to use: {MAX_FILTERED_SAMPLES or 'all available'}")

# Filter for error-free samples (especially important for T error injection)
error_free_ids = filter_error_free_samples(
    baseline_results_path=baseline_path,
    max_errors=MAX_ERRORS_FOR_FILTER,
    error_type_filter=None,  # Count all errors
    max_samples=MAX_FILTERED_SAMPLES
)

if not error_free_ids:
    print("\n⚠ WARNING: No error-free samples found!")
    print("  This may indicate:")
    print("  - All samples have errors")
    print("  - Consider increasing MAX_ERRORS_FOR_FILTER")
    print("  - Proceeding with all samples (may affect specificity analysis)")
    filtered_samples = samples
else:
    print(f"\n✓ Found {len(error_free_ids)} error-free samples")
    filtered_samples = filter_samples_by_ids(samples, set(error_free_ids))
    print(f"✓ Filtered to {len(filtered_samples)} samples for error injection")
    
    # Save filtered samples
    filtered_samples_path = os.path.join(output_dir, "filtered_samples.jsonl")
    save_samples(filtered_samples, filtered_samples_path)
    save_to_drive(filtered_samples_path)
    print(f"✓ Saved filtered samples to {filtered_samples_path}")

# For T error injection, use even stricter filtering (0 T errors specifically)
print("\n  Filtering for T error injection (0 T errors required)...")
t_error_free_ids = filter_error_free_samples(
    baseline_results_path=baseline_path,
    max_errors=0,
    error_type_filter="T",  # Only count T errors
    max_samples=MAX_FILTERED_SAMPLES
)

if not t_error_free_ids:
    print("  ⚠ WARNING: No samples with 0 T errors found!")
    print("  Using general error-free samples for T injection...")
    t_filtered_samples = filtered_samples
else:
    print(f"  ✓ Found {len(t_error_free_ids)} samples with 0 T errors")
    t_filtered_samples = filter_samples_by_ids(samples, set(t_error_free_ids))
    print(f"  ✓ Using {len(t_filtered_samples)} samples for T error injection")

# ============================================================================
# STEP 2: Run Error Injector for T/E/B/Placebo
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Error Injection with VLLM")
print("=" * 70)
print(f"Model: {VLLM_MODEL}")
print(f"Temperature: {TEMPERATURE}")
print("=" * 70)

# Initialize error injector with VLLM
try:
    injector = create_vllm_error_injector()
    print("✓ Error injector initialized with VLLM")
except Exception as e:
    print(f"⚠ Could not initialize VLLM error injector: {e}")
    print("This might require HuggingFace authentication.")
    print("For now, creating placeholder perturbed datasets...")
    injector = None

error_types = ["T", "B", "E", "PLACEBO"]
perturbed_datasets = {}

for error_type in error_types:
    print(f"\n  Creating {error_type}_perturbed dataset...")
    perturbed_path = os.path.join(output_dir, f"{error_type}_perturbed.jsonl")
    
    # Select appropriate samples for this error type
    # For T errors, use samples with 0 T errors specifically
    # For others, use general error-free samples
    if error_type == "T":
        samples_to_use = t_filtered_samples
        print(f"    Using {len(samples_to_use)} samples (filtered for 0 T errors)")
    else:
        samples_to_use = filtered_samples
        print(f"    Using {len(samples_to_use)} samples (filtered for error-free)")
    
    if injector:
        # Use real error injector with VLLM
        perturbed_samples = injector.create_perturbed_dataset(
            samples=samples_to_use,
            error_type=error_type
        )
    else:
        # Create placeholder (for testing structure)
        perturbed_samples = []
        for sample in tqdm(samples_to_use, desc=f"Creating {error_type}_perturbed (placeholder)", unit="sample"):
            perturbed = sample.copy()
            perturbed["error_type_injected"] = error_type
            # Ensure unique_dataset_id is preserved
            if "unique_dataset_id" not in perturbed:
                perturbed["unique_dataset_id"] = f"{perturbed.get('sample_id', 'unknown')}-{perturbed.get('model', 'unknown')}"
            if error_type == "PLACEBO":
                perturbed["response"] = sample["response"] + " \n"
                perturbed["error_subtype_injected"] = "placebo"
                perturbed["change_description"] = "Added trailing whitespace (format-only change, placeholder mode)"
            else:
                perturbed["response"] = sample["response"] + f" [MOCK_{error_type}_ERROR]"
                # Assign a default subtype based on error type
                default_subtypes = {
                    "T": "factual_error",
                    "B": "demographic_bias",
                    "E": "unclear_explanation"
                }
                perturbed["error_subtype_injected"] = default_subtypes.get(error_type, "unknown")
                perturbed["change_description"] = f"Added mock error marker [MOCK_{error_type}_ERROR] (placeholder mode)"
            perturbed_samples.append(perturbed)
    
    # Save perturbed dataset
    with open(perturbed_path, 'w', encoding='utf-8') as f:
        for sample in perturbed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    save_to_drive(perturbed_path)
    perturbed_datasets[error_type] = perturbed_samples
    print(f"  ✓ Saved {len(perturbed_samples)} samples to {perturbed_path}")
    
    # Inspect first perturbed sample
    if perturbed_samples:
        orig = perturbed_samples[0].get('original_response', 'N/A')
        pert = perturbed_samples[0]['response']
        print(f"    Original: {orig[:80]}..." if len(orig) > 80 else f"    Original: {orig}")
        print(f"    Perturbed: {pert[:80]}..." if len(pert) > 80 else f"    Perturbed: {pert}")
        print(f"    Changed: {orig != pert}")

# ============================================================================
# STEP 3: Run TrustScore on Perturbed Datasets (One by One)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: TrustScore Inference on Perturbed Datasets with VLLM")
print("=" * 70)

perturbed_results_paths = {}

for error_type in error_types:
    print(f"\n  Processing {error_type}_perturbed...")
    perturbed_path = os.path.join(output_dir, f"{error_type}_perturbed_results.jsonl")
    perturbed_results_paths[error_type] = perturbed_path
    
    results = run_perturbed_inference(
        perturbed_samples=perturbed_datasets[error_type],
        output_path=perturbed_path,
        error_type=error_type,
        use_mock=False,  # Use real VLLM
        config=vllm_config
    )
    
    save_to_drive(perturbed_path)
    print(f"  ✓ {error_type}_perturbed results saved to {perturbed_path}")

# ============================================================================
# STEP 4: Analysis (Run when ready)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Score Comparison Analysis")
print("=" * 70)
print("\nRunning score comparison analysis...")
print("=" * 70)

comparisons = {}

for error_type in tqdm(error_types, desc="Comparing scores", unit="error_type"):
    print(f"\n  Comparing {error_type}_perturbed with baseline...")
    try:
        comparison = compare_scores(
            baseline_results_path=baseline_path,
            perturbed_results_path=perturbed_results_paths[error_type],
            error_type=error_type
        )
        comparisons[error_type] = comparison
        print(f"  ✓ {error_type} comparison complete")
    except Exception as e:
        print(f"  ✗ Error comparing {error_type}: {str(e)}")
        comparisons[error_type] = {"error": str(e)}

# Generate final report
if comparisons:
    report_path = os.path.join(output_dir, "specificity_report.json")
    generate_report(comparisons, report_path)
    save_to_drive(report_path)
    print(f"\n✓ Final report saved to {report_path}")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)
print(f"\nAll results saved in: {output_dir}")
if IN_COLAB:
    print(f"Results also saved to Google Drive: {DRIVE_RESULTS_PATH}")
print("\nFiles created:")
print(f"  - sampled_dataset.jsonl")
print(f"  - baseline_results.jsonl")
for error_type in error_types:
    print(f"  - {error_type}_perturbed.jsonl")
    print(f"  - {error_type}_perturbed_results.jsonl")
print(f"\nConfigured with:")
print(f"  - Random seed: {RANDOM_SEED}")
print(f"  - Temperature: {TEMPERATURE} (deterministic)")
print(f"  - Model: {VLLM_MODEL}")
print(f"  - Device: {DEVICE}")

# Cleanup logging
try:
    print(f"\n✓ Execution complete. Log file saved to: {os.path.join(output_dir, 'execution.log')}")
except:
    pass
finally:
    if logger is not None:
        cleanup_logging(logger)
        
        # Save log file to Google Drive (after file is closed)
        log_file_path = os.path.join(output_dir, "execution.log")
        if os.path.exists(log_file_path):
            save_to_drive(log_file_path)
