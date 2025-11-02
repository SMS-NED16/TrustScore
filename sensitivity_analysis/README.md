# Sensitivity Analysis

This module implements sensitivity analysis to verify that TrustScore decreases monotonically as the number of errors of a single type increases.

## Overview

For each dimension (Trustworthiness T, Bias B, Explainability E) and subtype (e.g., `factual_error`, `demographic_bias`), we:

1. Inject k = 0, 1, 2, 3, 4, 5 errors of the same subtype
2. Run TrustScore inference on each k-error dataset
3. Calculate monotonicity metrics (Kendall's tau, Spearman's rho) to verify that scores decrease as k increases

## Expected Results

- **Target dimension**: Should show negative correlation (scores decrease as k increases)
  - Kendall's tau < -0.5 and p < 0.05 indicates monotonic decrease
- **Off-target dimensions**: Should show near-zero correlation (scores remain flat)

This validates that TrustScore is sensitive to the number of errors and responds proportionally.

## Module Structure

- `iterative_error_injector.py`: Extends base ErrorInjector to inject k errors iteratively
- `sensitivity_inference.py`: Runs TrustScore inference on k-error datasets
- `sensitivity_metrics.py`: Calculates Kendall's tau and Spearman's rho correlations
- `run_sensitivity_analysis.py`: Main orchestrator script

## Usage

### Running the Analysis

```python
# In Google Colab or local environment
python sensitivity_analysis/run_sensitivity_analysis.py
```

### Configuration

Edit `run_sensitivity_analysis.py` to configure:

- `K_MAX`: Maximum number of errors to inject (default: 5)
- `ERROR_SUBTYPES`: Subtypes to test for each dimension
- `VLLM_MODEL`: Model to use for inference (default: "meta-llama/Llama-3.1-8B-Instruct")
- `TEMPERATURE`: Temperature for generation (default: 0.0 for deterministic)

## Output Files

Results are saved to `results/sensitivity_analysis/`:

- `sampled_dataset.jsonl`: Input samples (reused from specificity analysis if available)
- `{error_type}_{subtype}_k{k}_perturbed.jsonl`: Datasets with k errors injected
- `{error_type}_{subtype}_k{k}_results.jsonl`: TrustScore results for k-error datasets
- `sensitivity_report.json`: Final report with correlation metrics
- `execution.log`: Console output log file

If running in Google Colab with Drive mounted, files are also saved to:
`/content/drive/MyDrive/TrustScore_Results/sensitivity_analysis/`

## Report Format

The sensitivity report (`sensitivity_report.json`) contains:

```json
{
  "T_factual_error": {
    "dimension": "T",
    "subtype": "factual_error",
    "k_max": 5,
    "target_gauge": {
      "kendall_tau": -0.85,
      "kendall_pvalue": 0.001,
      "spearman_rho": -0.92,
      "spearman_pvalue": 0.0001,
      "monotonic": true
    },
    "off_target_gauges": {
      "E": {
        "kendall_tau": 0.05,
        "spearman_rho": 0.03
      },
      "B": {
        "kendall_tau": -0.02,
        "spearman_rho": 0.01
      }
    },
    "mean_scores_by_k": {
      "0": 0.0,
      "1": 0.5,
      "2": 1.2,
      "3": 2.1,
      "4": 3.0,
      "5": 3.8
    },
    "num_samples": 10
  }
}
```

## Error Injection

The iterative error injector (`IterativeErrorInjector`) extends the base `ErrorInjector` to:

1. Inject errors iteratively: Start with original response → inject 1 error → use that as base → inject another → repeat k times
2. Control subtype: All errors in a k-error response are of the same subtype
3. Track changes: Each injection step is logged for debugging

### Supported Subtypes

**Trustworthiness (T)**:
- `factual_error`: Incorrect facts, dates, names, numbers
- `hallucination`: Fabricated information not in source
- `spelling`: Spelling or capitalization errors
- `inconsistency`: Internal contradictions

**Bias (B)**:
- `demographic_bias`: Assumptions about age, race, ethnicity
- `cultural_bias`: Assumptions about cultural practices
- `gender_bias`: Gender stereotypes or assumptions
- `political_bias`: Partisan language or stereotypes

**Explainability (E)**:
- `unclear_explanation`: Vague or ambiguous language
- `missing_context`: Missing background information
- `overly_complex`: Unnecessary jargon or technical terms
- `assumption_not_stated`: Claims relying on unstated assumptions

## Dependencies

- `scipy.stats`: For Kendall's tau and Spearman's rho calculations
- `vllm`: For fast batched LLM inference (in Colab)
- Same dependencies as specificity analysis

## Notes

- **Sample reuse**: The pipeline tries to reuse samples from specificity analysis for consistency. If not found, it generates new samples.
- **Deterministic**: Uses temperature=0.0 for reproducible results
- **Dual logging**: All console output is saved to `execution.log` file
- **Google Drive**: Automatically saves results to Drive if mounted in Colab

