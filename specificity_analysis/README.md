# TrustScore Specificity Analysis

This module evaluates TrustScore's specificity by injecting different error types and measuring corresponding score drops.

## Overview

The specificity analysis tests whether TrustScore correctly identifies errors in each dimension (Trustworthiness, Bias, Explainability) by:

1. **Baseline**: Running TrustScore on original LLM responses
2. **Perturbation**: Injecting specific error types (T, B, E) or placebo (format-only changes) into responses
3. **Re-evaluation**: Running TrustScore on perturbed responses
4. **Comparison**: Measuring score drops to validate specificity and ensure placebo has minimal effect

## Usage

### Quick Start (Full Pipeline)

```bash
# Run full analysis (with mock components for testing)
python specificity_analysis/run_full_analysis.py --use-mock --num-samples 10

# Run full analysis with real LLM calls
python specificity_analysis/run_full_analysis.py --num-samples 50 --api-key YOUR_API_KEY
```

### Step-by-Step Manual Execution (Recommended for Inspection)

For running step-by-step with inspection of intermediate results, especially on Google Colab:

```python
# Import the step-by-step script
from specificity_analysis.run_step_by_step import *

# Or run directly:
python specificity_analysis/run_step_by_step.py
```

**See `COLAB_SETUP.md` for complete Google Colab setup instructions.**

### Command Line Arguments

```
--dataset: Dataset to use ("summeval" or "cnn_dailymail") [default: summeval]
--num-samples: Number of samples to analyze [default: 50]
--use-mock: Use mock components (for testing)
--api-key: API key for LLM providers
--output-dir: Directory to save all results [default: results/specificity_analysis]
--skip-baseline: Skip baseline inference if results already exist
--skip-perturbation: Skip error injection if datasets already exist
```

### Step-by-Step Usage

```python
from specificity_analysis import *

# Step 0: Load dataset
samples = load_and_sample_dataset("summeval", max_samples=50)

# Step 1: Baseline inference
baseline_results = run_baseline_inference(samples, "results/baseline.jsonl")

# Step 2: Inject errors
injector = ErrorInjector()
t_perturbed = injector.create_perturbed_dataset(samples, "T")
b_perturbed = injector.create_perturbed_dataset(samples, "B")
e_perturbed = injector.create_perturbed_dataset(samples, "E")
placebo_perturbed = injector.create_perturbed_dataset(samples, "PLACEBO")

# Step 3: Run inference on perturbed
t_results = run_perturbed_inference(t_perturbed, "results/t_perturbed.jsonl", "T")
b_results = run_perturbed_inference(b_perturbed, "results/b_perturbed.jsonl", "B")
e_results = run_perturbed_inference(e_perturbed, "results/e_perturbed.jsonl", "E")
placebo_results = run_perturbed_inference(placebo_perturbed, "results/placebo_perturbed.jsonl", "PLACEBO")

# Step 4: Compare
t_comparison = compare_scores("results/baseline.jsonl", "results/t_perturbed.jsonl", "T")
b_comparison = compare_scores("results/baseline.jsonl", "results/b_perturbed.jsonl", "B")
e_comparison = compare_scores("results/baseline.jsonl", "results/e_perturbed.jsonl", "E")
placebo_comparison = compare_scores("results/baseline.jsonl", "results/placebo_perturbed.jsonl", "PLACEBO")

# Generate report
generate_report({"T": t_comparison, "B": b_comparison, "E": e_comparison, "PLACEBO": placebo_comparison}, "results/report.json")
```

## Output Files

The analysis produces the following files in the output directory:

- `sampled_dataset.jsonl`: Sampled observations from the dataset
- `baseline_results.jsonl`: TrustScore results on original responses
- `T_perturbed.jsonl`: Responses with Trustworthiness errors injected
- `B_perturbed.jsonl`: Responses with Bias errors injected
- `E_perturbed.jsonl`: Responses with Explainability errors injected
- `PLACEBO_perturbed.jsonl`: Responses with placebo (format-only) changes
- `T_perturbed_results.jsonl`: TrustScore results on T-perturbed responses
- `B_perturbed_results.jsonl`: TrustScore results on B-perturbed responses
- `E_perturbed_results.jsonl`: TrustScore results on E-perturbed responses
- `PLACEBO_perturbed_results.jsonl`: TrustScore results on placebo-perturbed responses
- `specificity_report.json`: Final comparison report with statistics

## Report Interpretation

The report contains:

1. **Summary**: Overview of error types tested
2. **Detailed Results**: For each error type (T, B, E):
   - Mean, median, std, min, max score drops for each dimension
   - Number of positive drops (where score decreased)
   - Total number of matched samples
3. **Specificity Analysis**: 
   - Target dimension drop (e.g., T drop when T errors are injected)
   - Other dimensions drops
   - `is_specific`: Boolean indicating if target dimension shows largest drop

### Success Criteria

A successful analysis shows:
- **Injecting T errors** → Largest drop in **T** score
- **Injecting B errors** → Largest drop in **B** score
- **Injecting E errors** → Largest drop in **E** score
- **Injecting PLACEBO (format-only)** → Minimal/no drop in any dimension (control condition)

This validates that TrustScore is specific to each error dimension and that score changes are due to actual errors, not just any perturbation.

## Error Injection Types

### Trustworthiness (T)
- Factual errors
- Hallucinations
- Inconsistencies
- Spelling/grammar errors

### Bias (B)
- Demographic bias
- Cultural bias
- Gender bias
- Political bias

### Explainability (E)
- Unclear explanations
- Missing context
- Overly complex language
- Unstated assumptions

### Placebo (PLACEBO)
- Format-only changes (whitespace additions)
- No meaningful content modification
- Serves as control condition to ensure score changes are due to actual errors

## Notes

- **VLLM Support**: The step-by-step script (`run_step_by_step.py`) uses VLLM for fast inference on Google Colab
- **Reproducibility**: Configured with random seed 42 and temperature 0.0 for deterministic results
- **Progress Tracking**: Uses `tqdm` for progress bars in all loops
- **Google Drive**: Automatically saves results to Google Drive when running in Colab
- **Error Injection**: Uses VLLM/LLaMA to modify responses minimally
- **Mock Mode**: Can be used for testing without actual LLM calls (see `run_full_analysis.py`)
- **Incremental Processing**: Results can be saved and compared incrementally using `--skip-baseline` and `--skip-perturbation` flags

## Google Colab Setup

For detailed instructions on running this on Google Colab, see **[COLAB_SETUP.md](COLAB_SETUP.md)**.

Quick summary:
1. Enable GPU runtime
2. Clone repository and install dependencies
3. Authenticate with HuggingFace (for LLaMA access)
4. Mount Google Drive (for saving results)
5. Run `run_step_by_step.py` or import from notebook

## Example Output

```
TRUSTSCORE SPECIFICITY ANALYSIS
======================================================================

[Step 0] Loading and sampling dataset...
✓ Loaded 50 samples

[Step 1] Running baseline TrustScore inference...
Processing 50 samples...

[Step 2] Creating perturbed datasets...
  Creating T_perturbed dataset...
  Creating B_perturbed dataset...
  Creating E_perturbed dataset...

[Step 3] Running TrustScore on perturbed datasets...
Processing T_perturbed samples...
Processing B_perturbed samples...
Processing E_perturbed samples...

[Step 4] Comparing scores...
Matched 50 samples for comparison

T Error Injection:
  Target dimension drop: 0.245
  Is specific: ✓ YES

B Error Injection:
  Target dimension drop: 0.312
  Is specific: ✓ YES

E Error Injection:
  Target dimension drop: 0.198
  Is specific: ✓ YES

✓ Report saved to results/specificity_analysis/specificity_report.json
```
