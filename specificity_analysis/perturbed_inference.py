"""
Run TrustScore inference on perturbed datasets
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
from pipeline.orchestrator import TrustScorePipeline
from config.settings import TrustScoreConfig


def run_perturbed_inference(
    perturbed_samples: List[Dict[str, Any]],
    output_path: str,
    error_type: str,  # "T", "B", "E", or "PLACEBO"
    use_mock: bool = False,
    api_key: Optional[str] = None,
    config: Optional[TrustScoreConfig] = None
) -> List[Dict[str, Any]]:
    """
    Run TrustScore inference on perturbed responses.
    
    Args:
        perturbed_samples: List of samples with injected errors
        output_path: Path to save results
        error_type: Type of error injected ("T", "B", "E", or "PLACEBO")
        use_mock: Whether to use mock components
        api_key: API key for LLM providers
        config: Optional TrustScoreConfig to use custom configuration
        
    Returns:
        List of TrustScore results
    """
    print("=" * 70)
    print(f"Step 3: TrustScore Inference on {error_type}_perturbed Dataset")
    print("=" * 70)
    
    # Initialize pipeline with config if provided
    if config:
        pipeline = TrustScorePipeline(config=config, api_key=api_key, use_mock=use_mock)
    else:
        pipeline = TrustScorePipeline(api_key=api_key, use_mock=use_mock)
    
    # Create output directory and open file for incremental writing
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path includes a directory
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if file exists and load existing results for resuming
    existing_results = []
    existing_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_result = json.loads(line)
                        existing_results.append(existing_result)
                        existing_ids.add(existing_result.get("unique_dataset_id"))
            if existing_results:
                print(f"  Found {len(existing_results)} existing results. Will skip those and continue...")
        except Exception as e:
            print(f"  Warning: Could not load existing results: {e}. Starting fresh.")
            existing_results = []
            existing_ids = set()
    
    # Open file in append mode if resuming, write mode if starting fresh
    file_mode = 'a' if existing_results else 'w'
    results = existing_results.copy()
    
    with open(output_path, file_mode, encoding='utf-8') as f:
        for i, sample in enumerate(tqdm(perturbed_samples, desc=f"Processing {error_type}_perturbed", unit="sample")):
            
            # Skip if already processed
            unique_id = sample.get("unique_dataset_id", f"{sample.get('sample_id', 'unknown')}-{sample.get('model', 'unknown')}")
            if unique_id in existing_ids:
                tqdm.write(f"  Sample {i+1}: Skipping (already processed)")
                continue
            
            try:
                result = pipeline.process(
                    prompt=sample["prompt"],
                    response=sample["response"],
                    model=sample.get("model", "unknown"),
                    generated_on=datetime.now()
                )
                
                # Format result for storage
                result_data = {
                    "sample_id": sample.get("sample_id", f"sample_{i}"),  # Original article ID
                    "unique_dataset_id": unique_id,  # Unique identifier
                    "error_type_injected": error_type,
                    "perturbed": True,
                    "trust_score": result.summary.trust_score,  # Raw severity
                    "trust_quality": result.summary.trust_quality,  # Quality [0-100]
                    "agg_score_T": result.summary.agg_score_T,  # Raw severity
                    "agg_quality_T": result.summary.agg_quality_T,  # Quality [0-100]
                    "agg_score_E": result.summary.agg_score_E,  # Raw severity
                    "agg_quality_E": result.summary.agg_quality_E,  # Quality [0-100]
                    "agg_score_B": result.summary.agg_score_B,  # Raw severity
                    "agg_quality_B": result.summary.agg_quality_B,  # Quality [0-100]
                    "trust_score_ci": {
                        "lower": result.summary.trust_score_ci.lower,
                        "upper": result.summary.trust_score_ci.upper
                    },
                    "trust_quality_ci": {
                        "lower": result.summary.trust_quality_ci.lower,
                        "upper": result.summary.trust_quality_ci.upper
                    },
                    "agg_score_T_ci": {
                        "lower": result.summary.agg_score_T_ci.lower,
                        "upper": result.summary.agg_score_T_ci.upper
                    },
                    "agg_quality_T_ci": {
                        "lower": result.summary.agg_quality_T_ci.lower,
                        "upper": result.summary.agg_quality_T_ci.upper
                    },
                    "agg_score_E_ci": {
                        "lower": result.summary.agg_score_E_ci.lower,
                        "upper": result.summary.agg_score_E_ci.upper
                    },
                    "agg_quality_E_ci": {
                        "lower": result.summary.agg_quality_E_ci.lower,
                        "upper": result.summary.agg_quality_E_ci.upper
                    },
                    "agg_score_B_ci": {
                        "lower": result.summary.agg_score_B_ci.lower,
                        "upper": result.summary.agg_score_B_ci.upper
                    },
                    "agg_quality_B_ci": {
                        "lower": result.summary.agg_quality_B_ci.lower,
                        "upper": result.summary.agg_quality_B_ci.upper
                    },
                    "num_errors": len(result.errors),
                    "errors": {
                        error_id: {
                            "type": error.type.value,
                            "subtype": error.subtype,
                            "severity_score": error.severity_score,
                            "severity_bucket": error.severity_bucket.value,
                            "explanation": error.explanation  # Span tagger explanation
                        }
                        for error_id, error in result.errors.items()
                    },
                    # Include detailed span information if available
                    "spans": {
                        span_id: {
                            "start": span.start,
                            "end": span.end,
                            "type": span.type.value,
                            "subtype": span.subtype,
                            "explanation": span.explanation,  # Span tagger explanation
                            "severity_score": span.get_average_severity_score(),
                            "judge_count": len(span.analysis)
                        }
                        for span_id, span in (result.graded_spans.spans.items() if result.graded_spans else {}.items())
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                # Write immediately to file (incremental save)
                f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
                f.flush()  # Ensure data is written to disk immediately
                
                results.append(result_data)
                
                tqdm.write(f"  Sample {i+1}: TrustScore={result.summary.trust_score:.3f}, "
                          f"T={result.summary.agg_score_T:.3f}, "
                          f"E={result.summary.agg_score_E:.3f}, "
                          f"B={result.summary.agg_score_B:.3f}, "
                          f"Errors={len(result.errors)}")
                
            except Exception as e:
                tqdm.write(f"  Error processing sample {i+1}: {str(e)}")
                error_result = {
                    "sample_id": sample.get("sample_id", f"sample_{i}"),  # Original article ID
                    "unique_dataset_id": unique_id,  # Unique identifier
                    "error_type_injected": error_type,
                    "perturbed": True,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Write error result immediately
                f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                f.flush()
                
                results.append(error_result)
    
    print(f"\n✓ Perturbed inference complete! Results saved to {output_path}")
    print(f"  Processed {len(results)} samples (including {len(existing_results)} existing)")
    
    return results
