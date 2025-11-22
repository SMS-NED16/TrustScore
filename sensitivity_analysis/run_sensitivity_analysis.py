"""
Sensitivity Analysis Pipeline

Verifies that TrustScore decreases monotonically as the number of errors
of a single type increases.
"""

import os
import json
import random
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datetime import datetime

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
from sensitivity_analysis.iterative_error_injector import IterativeErrorInjector
from sensitivity_analysis.sensitivity_inference import run_sensitivity_inference
from sensitivity_analysis.sensitivity_metrics import calculate_sensitivity_report
from specificity_analysis.dual_logger import initialize_logging, cleanup_logging

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

# VLLM Configuration (for Google Colab)
VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TEMPERATURE = 0.0  # Deterministic
MAX_TOKENS = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sensitivity Analysis Configuration
K_MAX = 5  # Maximum number of errors to inject (0, 1, 2, 3, 4, 5)

# Error subtypes to test
ERROR_SUBTYPES = {
    "T": ["factual_error", "hallucination"],  # Trustworthiness subtypes
    "B": ["gender_bias", "sycophancy_bias"],  # Bias subtypes
    "E": ["unclear_explanation", "missing_context"]  # Explainability subtypes
}

# Google Drive paths
DRIVE_MOUNT_PATH = "/content/drive"
DRIVE_RESULTS_BASE = "/content/drive/MyDrive/TrustScore_Results"

# ============================================================================
# CHECKPOINT/RESUME CONFIGURATION
# ============================================================================
# Set to a specific timestamp directory to resume from a checkpoint
# Example: "sensitivity_analysis_20251121_042854"
# Set to None to create a new run with a new timestamp
CHECKPOINT_DIR = "sensitivity_analysis_20251121_042854"  # Set to None for new run


def mount_google_drive():
    """Mount Google Drive in Colab."""
    if IN_COLAB:
        print("Mounting Google Drive...")
        drive.mount(DRIVE_MOUNT_PATH)
        os.makedirs(DRIVE_RESULTS_BASE, exist_ok=True)
        print(f"✓ Google Drive mounted at {DRIVE_MOUNT_PATH}")
        print(f"✓ Results base directory: {DRIVE_RESULTS_BASE}")
        return True
    return False


def get_drive_path(filename: str, timestamped_dir: str) -> str:
    """
    Construct Google Drive path with double-nested structure.
    
    Structure: DRIVE_RESULTS_BASE/timestamped_dir/timestamped_dir/filename
    This matches the structure created by save_to_drive().
    
    Args:
        filename: Name of the file (e.g., "sampled_dataset.jsonl")
        timestamped_dir: The timestamped directory name (e.g., "sensitivity_analysis_20251121_042854")
        
    Returns:
        Full Google Drive path
    """
    return os.path.join(DRIVE_RESULTS_BASE, timestamped_dir, timestamped_dir, filename)


def save_to_drive(local_path: str, drive_path: Optional[str] = None, timestamped_dir: Optional[str] = None):
    """Save file to Google Drive if in Colab and Drive is mounted."""
    if IN_COLAB and os.path.exists(DRIVE_MOUNT_PATH):
        try:
            if drive_path is None:
                # Auto-generate drive path from local path
                # Remove 'results/' prefix and replace with drive path
                if local_path.startswith("results/"):
                    rel_path = local_path.replace("results/", "")
                else:
                    rel_path = os.path.basename(local_path)
                
                # Use timestamped directory if provided
                if timestamped_dir:
                    drive_path = os.path.join(DRIVE_RESULTS_BASE, timestamped_dir, rel_path)
                else:
                    drive_path = os.path.join(DRIVE_RESULTS_BASE, "sensitivity_analysis", rel_path)

            drive_dir = os.path.dirname(drive_path)
            os.makedirs(drive_dir, exist_ok=True)

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
        model_path=VLLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        device=DEVICE,
        torch_dtype="float16"
    )

    # Judge Configs with VLLM
    judge_configs = {
        "trust_judge_1": JudgeConfig(
            name="trust_judge_1",
            enabled=True,
            provider=LLMProvider.VLLM,
            model=VLLM_MODEL,
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
    """Create IterativeErrorInjector with VLLM provider."""
    injector = IterativeErrorInjector(
        provider=LLMProvider.VLLM,
        model=VLLM_MODEL,
        model_path=VLLM_MODEL,
        temperature=TEMPERATURE,
        device=DEVICE
    )
    return injector


# Run sensitivity analysis pipeline on import
# Setup output directory with timestamp or checkpoint
if CHECKPOINT_DIR:
    # Resume from checkpoint
    output_dir = f"results/{CHECKPOINT_DIR}"
    drive_results_dir = CHECKPOINT_DIR
    print(f"🔄 RESUMING FROM CHECKPOINT: {CHECKPOINT_DIR}")
    
    # Validate checkpoint exists in Google Drive (double-nested structure)
    if IN_COLAB:
        checkpoint_drive_dir = os.path.join(DRIVE_RESULTS_BASE, CHECKPOINT_DIR, CHECKPOINT_DIR)
        if not os.path.exists(checkpoint_drive_dir):
            raise FileNotFoundError(
                f"Checkpoint directory not found in Google Drive: {checkpoint_drive_dir}\n"
                f"Expected structure: {DRIVE_RESULTS_BASE}/{CHECKPOINT_DIR}/{CHECKPOINT_DIR}/"
            )
        print(f"✓ Checkpoint directory found in Google Drive: {checkpoint_drive_dir}")
    else:
        # Fallback to local check if not in Colab
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {output_dir}")
        print(f"✓ Checkpoint directory found locally: {output_dir}")
else:
    # Create new run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/sensitivity_analysis_{timestamp}"
    drive_results_dir = f"sensitivity_analysis_{timestamp}"

os.makedirs(output_dir, exist_ok=True)

# Set up Google Drive path
if IN_COLAB:
    DRIVE_RESULTS_PATH = os.path.join(DRIVE_RESULTS_BASE, drive_results_dir, drive_results_dir)
else:
    DRIVE_RESULTS_PATH = None

print(f"Results will be saved to: {output_dir}")
if IN_COLAB and DRIVE_RESULTS_PATH:
    print(f"Google Drive path (double-nested): {DRIVE_RESULTS_PATH}")

# Initialize dual logging (console + file)
logger = None
try:
    logger = initialize_logging(output_dir, "execution.log")
    print(f"Logging initialized - output will be saved to: {os.path.join(output_dir, 'execution.log')}")
except Exception as e:
    print(f"Warning: Could not initialize file logging: {e}")
    print("Continuing with console-only output...")
    logger = None

# Mount Google Drive if in Colab
drive_mounted = False
if IN_COLAB and os.path.exists(DRIVE_MOUNT_PATH):
    drive_mounted = True
elif IN_COLAB:
    print("⚠ Google Drive not mounted. Run mount_google_drive() manually if needed.")

print("=" * 70)
print("SENSITIVITY ANALYSIS")
print("=" * 70)
print(f"Random seed: {RANDOM_SEED}")
print(f"Temperature: {TEMPERATURE} (deterministic)")
print(f"Model: {VLLM_MODEL}")
print(f"Device: {DEVICE}")
print(f"K_max: {K_MAX}")
print(f"Error subtypes: {ERROR_SUBTYPES}")
if IN_COLAB:
    print(f"Running in Colab: {IN_COLAB}")
    print(f"Google Drive mounted: {drive_mounted}")
print("=" * 70)

# ============================================================================
# STEP 1: Load samples (from checkpoint, specificity analysis, or generate new)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Loading Samples")
print("=" * 70)

samples = None

# Priority 1: Load from checkpoint directory (if resuming)
if CHECKPOINT_DIR:
    # Check Google Drive path (double-nested structure)
    if IN_COLAB:
        checkpoint_samples_path = get_drive_path("sampled_dataset.jsonl", CHECKPOINT_DIR)
        print(f"🔍 Checking for samples in Google Drive: {checkpoint_samples_path}")
        if os.path.exists(checkpoint_samples_path):
            print(f"✓ Found samples in Google Drive, loading...")
            samples = []
            with open(checkpoint_samples_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            print(f"✓ Loaded {len(samples)} samples from Google Drive checkpoint")
        else:
            print(f"⚠ Samples not found in Google Drive: {checkpoint_samples_path}")
    else:
        # Fallback to local if not in Colab
        checkpoint_samples_path = os.path.join(output_dir, "sampled_dataset.jsonl")
        if os.path.exists(checkpoint_samples_path):
            print(f"Loading samples from local checkpoint: {checkpoint_samples_path}")
            samples = []
            with open(checkpoint_samples_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            print(f"✓ Loaded {len(samples)} samples from local checkpoint")

# Priority 2: Load from specificity analysis
if samples is None:
    specificity_samples_path = "results/specificity_analysis/sampled_dataset.jsonl"
    if os.path.exists(specificity_samples_path):
        print(f"Loading samples from specificity analysis: {specificity_samples_path}")
        samples = []
        with open(specificity_samples_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"✓ Loaded {len(samples)} samples from specificity analysis")

# Priority 3: Generate new samples
if samples is None:
    print("No existing samples found. Generating new samples...")
    samples = load_and_sample_dataset(
        dataset_name="summeval",
        max_samples=100,
        random_seed=RANDOM_SEED
    )
    print(f"✓ Generated {len(samples)} samples")

# Save samples for reference (only if not resuming or if file doesn't exist)
samples_path = os.path.join(output_dir, "sampled_dataset.jsonl")
if not os.path.exists(samples_path):
    save_samples(samples, samples_path)
    save_to_drive(samples_path, timestamped_dir=drive_results_dir)

if samples:
    print("\nFirst sample preview:")
    sample = samples[0]
    print(f"  Sample ID (article): {sample.get('sample_id', 'unknown')}")
    print(f"  Unique Dataset ID: {sample.get('unique_dataset_id', 'unknown')}")
    print(f"  Model: {sample.get('model', 'unknown')}")
    print(f"  Prompt length: {len(sample.get('prompt', ''))} chars")
    print(f"  Response length: {len(sample.get('response', ''))} chars")
    print(f"  Response preview: {sample.get('response', '')[:100]}...")

# ============================================================================
# STEP 2: Initialize error injector and pipeline config
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Initializing Components")
print("=" * 70)

# Create VLLM config for pipeline
config = create_vllm_config_for_pipeline()

# Create iterative error injector
injector = create_vllm_error_injector()
print(f"✓ Iterative error injector initialized with VLLM")
print(f"✓ TrustScore pipeline config created")

# ============================================================================
# STEP 3: Generate k-error datasets for each dimension/subtype
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Generating k-Error Datasets")
print("=" * 70)

datasets_by_key: Dict[str, List[Dict[str, Any]]] = {}  # key = "{error_type}_{subtype}_k{k}"

for error_type, subtypes in ERROR_SUBTYPES.items():
    for subtype in subtypes:
        print(f"\nGenerating datasets for {error_type}_{subtype}...")
        
        for k in range(0, K_MAX + 1):
            try:
                dataset_filename = f"{error_type}_{subtype}_k{k}_perturbed.jsonl"
                dataset_path = os.path.join(output_dir, dataset_filename)
                
                # Check if dataset already exists in Google Drive (if checkpointing)
                dataset_exists = False
                drive_dataset_path = None
                
                if CHECKPOINT_DIR and IN_COLAB:
                    drive_dataset_path = get_drive_path(dataset_filename, drive_results_dir)
                    dataset_exists = os.path.exists(drive_dataset_path)
                    if dataset_exists:
                        print(f"  🔍 Found {error_type}_{subtype}_k{k} in Google Drive: {drive_dataset_path}")
                
                # Also check local path (for new runs or fallback)
                if not dataset_exists:
                    dataset_exists = os.path.exists(dataset_path)
                    if dataset_exists:
                        print(f"  🔍 Found {error_type}_{subtype}_k{k} locally: {dataset_path}")
                
                if dataset_exists:
                    print(f"  ⏭️  Skipping {error_type}_{subtype}_k{k} - dataset already exists")
                    # Load existing dataset for inference
                    load_path = drive_dataset_path if drive_dataset_path and os.path.exists(drive_dataset_path) else dataset_path
                    print(f"  📂 Loading from: {load_path}")
                    k_samples = []
                    with open(load_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                k_samples.append(json.loads(line))
                    datasets_by_key[f"{error_type}_{subtype}_k{k}"] = k_samples
                    print(f"  ✓ Loaded {len(k_samples)} samples from existing dataset")
                else:
                    print(f"  🔨 Creating {error_type}_{subtype}_k{k} dataset...")
                    
                    if k == 0:
                        # k=0: Use original responses (no errors)
                        k_samples = []
                        for sample in samples:
                            k_sample = sample.copy()
                            k_sample["error_type_injected"] = error_type
                            k_sample["error_subtype_injected"] = subtype
                            k_sample["k_errors_injected"] = 0
                            k_sample["original_response"] = sample["response"]
                            k_sample["change_descriptions"] = ["No errors injected (k=0)"]
                            k_samples.append(k_sample)
                        datasets_by_key[f"{error_type}_{subtype}_k{k}"] = k_samples
                    else:
                        # k>0: Inject k errors iteratively
                        k_samples = injector.create_k_error_dataset(
                            samples=samples,
                            error_type=error_type,
                            subtype=subtype,
                            k=k
                        )
                        datasets_by_key[f"{error_type}_{subtype}_k{k}"] = k_samples
                    
                    # Save dataset
                    save_samples(k_samples, dataset_path)
                    save_to_drive(dataset_path, timestamped_dir=drive_results_dir)
                    print(f"  ✓ Saved {len(k_samples)} samples to {dataset_path}")
                
            except Exception as e:
                print(f"  ✗ Error loading/creating {error_type}_{subtype}_k{k}: {str(e)}")
                import traceback
                traceback.print_exc()

# ============================================================================
# STEP 4: Run TrustScore inference on each k-error dataset
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Running TrustScore Inference")
print("=" * 70)

results_by_key: Dict[str, str] = {}  # key -> result_file_path

for error_type, subtypes in ERROR_SUBTYPES.items():
    for subtype in subtypes:
        for k in range(0, K_MAX + 1):
            key = f"{error_type}_{subtype}_k{k}"
            
            if key not in datasets_by_key:
                print(f"⚠ Skipping {key} - dataset not found")
                continue
            
            dataset = datasets_by_key[key]
            
            if not dataset:
                print(f"⚠ Skipping {key} - empty dataset")
                continue
            
            try:
                result_filename = f"{error_type}_{subtype}_k{k}_results.jsonl"
                result_path = os.path.join(output_dir, result_filename)
                
                # Check if results already exist in Google Drive (if checkpointing)
                result_exists = False
                drive_result_path = None
                
                if CHECKPOINT_DIR and IN_COLAB:
                    drive_result_path = get_drive_path(result_filename, drive_results_dir)
                    result_exists = os.path.exists(drive_result_path)
                    if result_exists:
                        print(f"  🔍 Found {key} results in Google Drive: {drive_result_path}")
                
                # Also check local path (for new runs or fallback)
                if not result_exists:
                    result_exists = os.path.exists(result_path)
                    if result_exists:
                        print(f"  🔍 Found {key} results locally: {result_path}")
                
                if result_exists:
                    print(f"  ⏭️  Skipping {key} - results already exist")
                    # Use the path where results were found
                    results_by_key[key] = drive_result_path if drive_result_path and os.path.exists(drive_result_path) else result_path
                else:
                    print(f"\n🔨 Processing {key} ({len(dataset)} samples)...")
                    
                    # Run TrustScore inference
                    run_sensitivity_inference(
                        samples=dataset,
                        output_path=result_path,
                        error_type=error_type,
                        subtype=subtype,
                        k=k,
                        config=config
                    )
                    
                    results_by_key[key] = result_path
                    save_to_drive(result_path, timestamped_dir=drive_results_dir)
                    
                    print(f"✓ Inference complete for {key}")
                
            except Exception as e:
                print(f"✗ Error running inference for {key}: {str(e)}")
                import traceback
                traceback.print_exc()

# ============================================================================
# STEP 5: Calculate monotonicity metrics and generate report
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Calculating Sensitivity Metrics")
print("=" * 70)

sensitivity_reports = {}

for error_type, subtypes in ERROR_SUBTYPES.items():
    for subtype in subtypes:
        try:
            print(f"\nCalculating sensitivity metrics for {error_type}_{subtype}...")
            
            report = calculate_sensitivity_report(
                results_dir=output_dir,
                error_type=error_type,
                subtype=subtype,
                k_max=K_MAX
            )
            
            sensitivity_reports[f"{error_type}_{subtype}"] = report
            
            # Print summary
            target_gauge = report.get("target_gauge", {})
            print(f"  Target dimension ({error_type}):")
            print(f"    Kendall's tau: {target_gauge.get('kendall_tau', 'N/A'):.3f} (p={target_gauge.get('kendall_pvalue', 'N/A'):.4f})")
            print(f"    Spearman's rho: {target_gauge.get('spearman_rho', 'N/A'):.3f} (p={target_gauge.get('spearman_pvalue', 'N/A'):.4f})")
            print(f"    Monotonic: {target_gauge.get('monotonic', False)}")
            
            mean_scores = report.get("mean_scores_by_k", {})
            print(f"  Mean scores by k: {mean_scores}")
            
        except Exception as e:
            print(f"✗ Error calculating metrics for {error_type}_{subtype}: {str(e)}")
            import traceback
            traceback.print_exc()

# Save sensitivity report
if sensitivity_reports:
    report_path = os.path.join(output_dir, "sensitivity_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(sensitivity_reports, f, indent=2, ensure_ascii=False)
    save_to_drive(report_path, timestamped_dir=drive_results_dir)
    print(f"\n✓ Sensitivity report saved to {report_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)
print(f"\nAll results saved in: {output_dir}")
if IN_COLAB and DRIVE_RESULTS_PATH:
    print(f"Results also saved to Google Drive: {DRIVE_RESULTS_PATH}")
print("\nFiles created:")
print(f"  - sampled_dataset.jsonl")
for error_type, subtypes in ERROR_SUBTYPES.items():
    for subtype in subtypes:
        for k in range(0, K_MAX + 1):
            print(f"  - {error_type}_{subtype}_k{k}_perturbed.jsonl")
            print(f"  - {error_type}_{subtype}_k{k}_results.jsonl")
print(f"  - sensitivity_report.json")
print(f"  - execution.log")
print(f"\nConfigured with:")
print(f"  - Random seed: {RANDOM_SEED}")
print(f"  - Temperature: {TEMPERATURE} (deterministic)")
print(f"  - Model: {VLLM_MODEL}")
print(f"  - Device: {DEVICE}")
print(f"  - K_max: {K_MAX}")
print(f"  - Error subtypes: {ERROR_SUBTYPES}")

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
            save_to_drive(log_file_path, timestamped_dir=drive_results_dir)

