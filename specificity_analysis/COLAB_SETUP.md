# Google Colab Setup Instructions for Specificity Analysis

This guide will walk you through setting up and running the specificity analysis pipeline on Google Colab with VLLM.

## Prerequisites

1. Google Colab account (free tier works)
2. HuggingFace account (for accessing LLaMA models)
3. GPU runtime (required for VLLM)

## Step-by-Step Setup Instructions

### Step 1: Create a New Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Rename it to "TrustScore_Specificity_Analysis"

### Step 2: Enable GPU Runtime

1. Go to **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4 is fine, A100 is better)
3. Click **Save**

### Step 3: Clone and Install Dependencies

Run these cells in order:

```python
# Cell 1: Clone repository and install dependencies
!git clone https://github.com/SMS-NED16/TrustScore.git
%cd TrustScore

# Install required packages
!pip install -q torch transformers bitsandbytes accelerate vllm tqdm datasets huggingface_hub
!pip install -q pyyaml pydantic openai
```

### Step 4: Authenticate with HuggingFace

```python
# Cell 2: Authenticate with HuggingFace (for LLaMA model access)
from huggingface_hub import login

# You'll need to get a token from https://huggingface.co/settings/tokens
# Click "New token" → give it a name → copy the token
HF_TOKEN = "your_huggingface_token_here"  # Replace with your token

login(token=HF_TOKEN, add_to_git_credential=True)

# Also set as environment variable
import os
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN
```

### Step 5: Mount Google Drive

```python
# Cell 3: Mount Google Drive for saving results
from google.colab import drive
drive.mount('/content/drive')

# Create results directory in Drive
import os
os.makedirs('/content/drive/MyDrive/TrustScore_Results', exist_ok=True)
print("✓ Google Drive mounted and results directory created")
```

### Step 6: Navigate to Project Directory

```python
# Cell 4: Navigate to project
import sys
import os
sys.path.insert(0, '/content/TrustScore')

# Change to project directory
%cd /content/TrustScore

# Verify installation
from config.settings import LLMProvider
print(f"✓ TrustScore installed. Available providers: {[p.value for p in LLMProvider]}")
print(f"✓ Current directory: {os.getcwd()}")
```

### Step 7: Download/Check SummEval Dataset

```python
# Cell 5: Check if dataset exists, download if needed
import os

datasets_dir = "datasets/raw/summeval"
os.makedirs(datasets_dir, exist_ok=True)

summeval_file = os.path.join(datasets_dir, "model_annotations.aligned.jsonl")

if not os.path.exists(summeval_file):
    print("⚠ SummEval dataset not found!")
    print("\nTo download:")
    print("1. The dataset should be at: datasets/raw/summeval/model_annotations.aligned.jsonl")
    print("2. If you have the summeval zip file, extract it:")
    print("   !unzip /path/to/summeval.zip -d datasets/raw/")
    print("\n3. Or download from the SummEval repository:")
    print("   https://github.com/Yale-LILY/SummEval")
    print("\n4. Or use the preprocessed version if available")
    raise FileNotFoundError(f"Please download SummEval dataset to: {summeval_file}")
else:
    print(f"✓ SummEval dataset found at: {summeval_file}")
    print(f"  File size: {os.path.getsize(summeval_file) / (1024*1024):.2f} MB")
```

### Step 8: Run the Step-by-Step Pipeline

```python
# Cell 5: Run the step-by-step analysis
from specificity_analysis.run_step_by_step import *

# The script will automatically:
# 1. Sample 10 observations from SummEval
# 2. Create perturbed datasets (T/B/E/PLACEBO)
# 3. Run baseline TrustScore inference
# 4. Run TrustScore on all perturbed datasets
# 5. Save all results to both local and Google Drive

# All results will be saved automatically!
```

### Step 9: Inspect Results (Optional)

```python
# Cell 7: Inspect saved results
import json

# Check what files were created
import os
output_dir = "results/specificity_analysis"
files = os.listdir(output_dir)
print("Files created:")
for f in sorted(files):
    print(f"  - {f}")

# Load and inspect baseline results
with open(f"{output_dir}/baseline_results.jsonl", 'r') as f:
    baseline = [json.loads(line) for line in f]
    
print(f"\nBaseline results: {len(baseline)} samples")
if baseline and "error" not in baseline[0]:
    result = baseline[0]
    print(f"  TrustScore: {result['trust_score']:.3f}")
    print(f"  T: {result['agg_score_T']:.3f}")
    print(f"  E: {result['agg_score_E']:.3f}")
    print(f"  B: {result['agg_score_B']:.3f}")
```

### Step 10: Run Analysis (When Ready)

```python
# Cell 8: Run comparison analysis
from specificity_analysis.score_comparison import compare_scores, generate_report

output_dir = "results/specificity_analysis"
baseline_path = f"{output_dir}/baseline_results.jsonl"
error_types = ["T", "B", "E", "PLACEBO"]

comparisons = {}

for error_type in error_types:
    print(f"\nComparing {error_type}_perturbed...")
    perturbed_path = f"{output_dir}/{error_type}_perturbed_results.jsonl"
    
    try:
        comparison = compare_scores(
            baseline_results_path=baseline_path,
            perturbed_results_path=perturbed_path,
            error_type=error_type
        )
        comparisons[error_type] = comparison
        print(f"✓ {error_type} comparison complete")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        comparisons[error_type] = {"error": str(e)}

# Generate final report
if comparisons:
    report_path = f"{output_dir}/specificity_report.json"
    generate_report(comparisons, report_path)
    
    # Save to Drive
    import shutil
    drive_report_path = f"/content/drive/MyDrive/TrustScore_Results/specificity_report.json"
    shutil.copy2(report_path, drive_report_path)
    
    print(f"\n✓ Final report saved to:")
    print(f"  - Local: {report_path}")
    print(f"  - Drive: {drive_report_path}")
```

## Configuration Details

The script is configured with:
- **Random seed**: 42 (for reproducibility)
- **Temperature**: 0.0 (deterministic, greedy decoding)
- **Model**: `meta-llama/Llama-3.1-8B-Instruct` (can be changed in script)
- **Device**: Automatically uses CUDA if available
- **Google Drive**: Results automatically saved to `/content/drive/MyDrive/TrustScore_Results/`

## File Structure

Results will be saved in:
- **Local**: `results/specificity_analysis/`
- **Google Drive**: `/content/drive/MyDrive/TrustScore_Results/`

Files created:
- `sampled_dataset.jsonl` - Original 10 samples
- `baseline_results.jsonl` - TrustScore on original responses
- `T_perturbed.jsonl` - Responses with Trustworthiness errors
- `B_perturbed.jsonl` - Responses with Bias errors
- `E_perturbed.jsonl` - Responses with Explainability errors
- `PLACEBO_perturbed.jsonl` - Responses with placebo changes
- `T_perturbed_results.jsonl` - TrustScore on T-perturbed
- `B_perturbed_results.jsonl` - TrustScore on B-perturbed
- `E_perturbed_results.jsonl` - TrustScore on E-perturbed
- `PLACEBO_perturbed_results.jsonl` - TrustScore on placebo-perturbed
- `specificity_report.json` - Final analysis report

## Troubleshooting

### Issue: "Model not found" or "Permission denied"
- **Solution**: Make sure you've authenticated with HuggingFace and the model is accessible
- Check that your HF_TOKEN is set correctly

### Issue: "CUDA out of memory"
- **Solution**: 
  - Use a smaller model (e.g., 7B instead of 13B)
  - Reduce `max_model_len` in VLLMProvider
  - Use T4 GPU with lower `gpu_memory_utilization`

### Issue: "vLLM installation failed"
- **Solution**: Install vLLM separately:
  ```python
  !pip install vllm
  ```
  Or build from source if needed.

### Issue: Results not saving to Drive
- **Solution**: Make sure Drive is mounted:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

## Estimated Runtime

- **Step 0** (Sampling): ~30 seconds
- **Step 1** (Error Injection): ~5-10 minutes (depends on model speed)
- **Step 2** (Baseline Inference): ~10-20 minutes for 10 samples
- **Step 3** (Perturbed Inference): ~40-80 minutes total (4 datasets × 10 samples each)
- **Step 4** (Analysis): ~1 minute

**Total**: ~1-2 hours for 10 samples with VLLM

## Next Steps

1. After running, inspect results in Google Drive
2. Download results if needed: `File → Download` or use `files.download()`
3. Analyze the specificity report to validate TrustScore behavior
4. Increase sample size if needed (change `max_samples=10` to larger number)

## Notes

- VLLM loads the model once and reuses it across all components (efficient)
- Temperature 0.0 ensures deterministic results for reproducibility
- Random seed 42 ensures consistent sampling
- All intermediate results are saved so you can pause and resume

