# CI Calibration Analysis

This module evaluates whether TrustScore's reported confidence intervals (CIs) behave sensibly — that is, whether uncertainty decreases as you add more judges and whether the CIs are well-calibrated to observed variability.

## Overview

The CI calibration analysis tests:
1. **CI Width and Variance**: Whether CI width decreases roughly with 1/√J (where J is the number of judges)
2. **Coverage Check**: Whether 95% CIs actually cover the true value at the expected rate
3. **Correlation**: Whether CI width correlates with observed variability (r ≥ 0.5)
4. **Aggregation**: Whether ensemble averaging works correctly (width decreases with J)

## Experiment Setup

1. **Sample Selection**: Selects 3-5 summarization examples from SummEval (each example = 1 article + model-generated summary)
2. **Configuration**: Runs TrustScore pipeline for each example with:
   - Number of trustworthiness judges: J ∈ {1, 3, 5}
     - **1 judge**: LLaMA 3.1 8B
     - **3 judges**: LLaMA 3.1 8B, Mistral 7B, Qwen 7B
     - **5 judges**: 2× LLaMA 3.1 8B, 2× Mistral 7B, 1× Qwen 7B
   - Number of bias judges: 0
   - Number of explainability judges: 0
   - Span tagger temperature: 0.0 (deterministic for consistent spans)
   - Judge temperature: 0.7 (stochastic for variability)
   - Confidence level: 95% (configurable)
   - 5 repeats per (example, J) combination
3. **Total Runs**: ~3-5 examples × 3 judge counts × 5 repeats = 45-75 pipeline executions

## CI Levels Analyzed

The analysis checks CIs at multiple levels:

1. **Span-level CIs** (for each detected error):
   - Severity score CI (severity space)
   - Confidence CI (probability space [0-1])

2. **Category-level CIs** (T category):
   - T severity score CI (`agg_score_T_ci`) (severity space)
   - T confidence CI (`agg_confidence_T_ci`) (probability space [0-1])
   - Note: E and B CIs will be `None` (no spans), so skipped in analysis

3. **Final TrustScore CIs**:
   - Trust score CI (`trust_score_ci`) (severity space)
   - Trust confidence CI (`trust_confidence_ci`) (probability space [0-1])
   - Note: With only T errors, these should equal the T category CIs (sanity check)

## Usage

### Running in Google Colab

```python
# In a Google Colab notebook cell:

# 1. Install dependencies
!pip install vllm transformers datasets tqdm

# 2. Set HuggingFace token (if needed)
import os
os.environ["HF_TOKEN"] = "your_hf_token_here"  # Optional

# 3. Mount Google Drive (if not already mounted)
from google.colab import drive
drive.mount('/content/drive')

# 4. Run the analysis
from ci_calibration_analysis.run_ci_calibration import run_ci_calibration_analysis

output_dir = run_ci_calibration_analysis(
    summeval_path="datasets/raw/summeval/model_annotations.aligned.jsonl",
    output_dir="results",
    num_examples=5,
    judge_counts=[1, 3, 5],
    num_repeats=5,
    random_seed=42
)
```

### Running Locally

```python
from ci_calibration_analysis.run_ci_calibration import run_ci_calibration_analysis

# Run the analysis
output_dir = run_ci_calibration_analysis(
    summeval_path="datasets/raw/summeval/model_annotations.aligned.jsonl",
    output_dir="results",
    num_examples=5,
    judge_counts=[1, 3, 5],
    num_repeats=5,
    random_seed=42
)
```

Or from the command line:

```bash
python -m ci_calibration_analysis.run_ci_calibration \
    --summeval-path datasets/raw/summeval/model_annotations.aligned.jsonl \
    --output-dir results \
    --num-examples 5 \
    --judge-counts 1 3 5 \
    --num-repeats 5 \
    --random-seed 42
```

### Generating the Report

After running the analysis, generate a report from existing results:

```python
from ci_calibration_analysis.generate_ci_calibration_report import generate_ci_calibration_report

# Generate report from existing results
report_path = generate_ci_calibration_report(
    results_path=None,  # Auto-searches in results_dir
    results_dir="results",
    output_path=None,  # Auto-generates path
    confidence_level=0.95
)
```

Or from the command line:

```bash
python -m ci_calibration_analysis.generate_ci_calibration_report \
    --results-dir results \
    --confidence-level 0.95
```

## Output Files

Results are saved in `results/ci_calibration_results_<timestamp>/`:

- `samples_metadata.json`: Metadata about selected samples
- `calibration_results.jsonl`: All pipeline results (one JSON object per line)
- `execution.log`: Execution log with all console output
- `ci_calibration_report.json`: Generated analysis report (after running report generation)

Results are also automatically saved to Google Drive if running in Colab:
- `/content/drive/MyDrive/TrustScore_Results/ci_calibration_results_<timestamp>/`

## Report Structure

The generated report contains:

1. **Metadata**:
   - Results file path
   - Number of runs
   - Number of unique items
   - Judge counts tested
   - Confidence level

2. **Analyses** (by CI level):
   - `trust_score_severity`: Final TrustScore severity CI
   - `trust_score_confidence`: Final TrustScore confidence CI
   - `category_T_severity`: Category-level T severity CI
   - `category_T_confidence`: Category-level T confidence CI
   - `span_severity`: Span-level severity CI (aggregated)
   - `span_confidence`: Span-level confidence CI (aggregated)

   Each analysis includes:
   - `by_item_judge`: Per-item, per-judge-count statistics
   - `by_judge_count`: Aggregated statistics by judge count

3. **Summary**: Aggregated statistics by judge count

## Metrics Computed

For each (item, judge_count) combination:

1. **Mean CI Width**: Average (upper - lower) across repeats
2. **Observed Standard Deviation**: Std dev of point estimates across repeats
3. **Coverage**: Fraction of CIs containing the mean (using mean-as-truth method)
4. **Correlation**: Pearson/Spearman correlation between CI width and observed std

For each judge count:

1. **Mean CI Width**: Average across all items
2. **Mean Observed Std**: Average across all items
3. **Mean Coverage**: Average coverage across all items
4. **Correlation**: Overall correlation between CI width and observed std

## Interpretation Guide

- **Coverage ≈ 0.95**: CIs correctly sized
- **Width–Std r ≥ 0.5**: Intervals track true variability
- **Width decreases with J**: Ensemble averaging works
- **Coverage < 0.9**: CIs too narrow; multiply variance term by a small factor
- **Coverage > 1.0**: Intervals are conservative (acceptable)

## Google Colab Integration

The scripts automatically detect Colab environment and:
1. Mount Google Drive (if not already mounted)
2. Save all results to Google Drive in `/content/drive/MyDrive/TrustScore_Results/`
3. Support loading results from Drive when generating reports

## Configuration

Key configuration constants (in `run_ci_calibration.py`):

- `NUM_EXAMPLES`: Number of examples to use (3-5, default: 5)
- `JUDGE_COUNTS`: List of judge counts to test (default: [1, 3, 5])
- `NUM_REPEATS`: Repeats per (example, J) (default: 5)
- `SPAN_TAGGER_TEMPERATURE`: Temperature for span tagger (default: 0.0)
- `JUDGE_TEMPERATURE`: Temperature for judges (default: 0.7)
- `CONFIDENCE_LEVEL`: Confidence level for CIs (default: 0.95)
- `VLLM_MODEL`: Model to use (default: "meta-llama/Llama-3.1-8B-Instruct")

## Logging

All execution is logged to both console and `execution.log` file using the dual logger pattern from specificity analysis. This ensures:
- Full traceability of all runs
- Easy debugging if errors occur
- Complete record of configurations used

## Error Handling

The pipeline handles errors gracefully:
- Failed runs are logged as error records in `calibration_results.jsonl`
- Analysis continues even if some runs fail
- Report generation skips error records automatically

## Reproducibility

The analysis ensures reproducibility and controlled variability through:

1. **Deterministic Sample Selection**:
   - Samples are sorted by `unique_dataset_id` before random selection
   - Random seed (default: 42) ensures same samples are selected across runs
   - All random seeds (Python, NumPy, PyTorch) are set consistently

2. **Deterministic Span Tagger**:
   - Temperature set to 0.0 for deterministic generation
   - VLLM seed set to 42 when temperature=0
   - Ensures same spans are detected across all repeats

3. **Ensemble of Different Models**:
   - **1 judge**: LLaMA 3.1 8B only
   - **3 judges**: LLaMA 3.1 8B, Mistral 7B, Qwen 7B (different models for epistemic uncertainty)
   - **5 judges**: 2× LLaMA, 2× Mistral, 1× Qwen (hybrid ensemble)
   - All judges use temperature 0.7 for stochastic variability
   - **Deterministic seeds**: Each run uses a deterministic seed based on (random_seed, sample_idx, num_judges, repeat)
   - Seeds ensure reproducible variability: same configuration produces same results across runs
   - Each repeat produces different judge scores/confidences due to seeded randomness
   - Different models capture true epistemic uncertainty (model disagreement)

4. **PyTorch Determinism**:
   - `torch.backends.cudnn.deterministic = True`
   - `torch.backends.cudnn.benchmark = False`
   - All CUDA operations are deterministic

## Seed Generation Strategy

Judges use deterministic seeds for reproducible randomness:
- **Base seed formula**: `base_seed = random_seed * 1000000 + sample_idx * 10000 + num_judges * 1000 + repeat * 100`
- **Judge-specific seed**: `judge_seed = base_seed + span_hash + judge_idx`
  - Ensures each judge gets a unique seed for each span
  - Same (sample_idx, num_judges, repeat, judge_idx, span) → same seed → same output
- **Benefits**:
  - Fully reproducible across runs (same configuration → same results)
  - Variability across different runs (different seeds for different configurations)
  - Temperature=0.7 still introduces randomness, but seeded randomness

## Notes

- Span tagger uses temperature 0.0 with seed=42 to ensure consistent span detection across repeats
- Judges use temperature 0.7 with deterministic seeds (generated per run) to introduce reproducible variability in scoring
- Only T (trustworthiness) judges are enabled; E and B judges are disabled
- All CIs are analyzed in their respective spaces (severity vs probability)
- With only T errors, final TrustScore CIs should equal T category CIs (sanity check)
