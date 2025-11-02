"""
Run TrustScore inference on k-error datasets for sensitivity analysis
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
from pipeline.orchestrator import TrustScorePipeline
from config.settings import TrustScoreConfig


def run_sensitivity_inference(
    samples: List[Dict[str, Any]],
    output_path: str,
    error_type: str,
    subtype: str,
    k: int,
    use_mock: bool = False,
    api_key: Optional[str] = None,
    config: Optional[TrustScoreConfig] = None
) -> List[Dict[str, Any]]:
    """
    Run TrustScore inference on samples with k errors injected.
    
    Args:
        samples: List of samples with 'prompt' and 'response' (k errors already injected)
        output_path: Path to save results
        error_type: Error type ("T", "B", or "E")
        subtype: Error subtype (e.g., "factual_error")
        k: Number of errors injected
        use_mock: Whether to use mock components (for testing)
        api_key: API key for LLM providers
        config: Optional TrustScoreConfig to use custom configuration
        
    Returns:
        List of TrustScore results
    """
    print("=" * 70)
    print(f"Sensitivity Analysis: TrustScore Inference for {error_type}_{subtype}_k{k}")
    print("=" * 70)
    
    # Initialize pipeline with config if provided
    if config:
        pipeline = TrustScorePipeline(config=config, api_key=api_key, use_mock=use_mock)
    else:
        pipeline = TrustScorePipeline(api_key=api_key, use_mock=use_mock)
    
    results = []
    
    for i, sample in enumerate(tqdm(samples, desc=f"Processing {error_type}_{subtype}_k{k} samples", unit="sample")):
        
        try:
            result = pipeline.process(
                prompt=sample["prompt"],
                response=sample["response"],
                model=sample.get("model", "unknown"),
                generated_on=datetime.now()
            )
            
            # Format result for storage
            result_data = {
                "sample_id": sample.get("sample_id", f"sample_{i}"),
                "unique_dataset_id": sample.get("unique_dataset_id", f"{sample.get('sample_id', 'unknown')}-{sample.get('model', 'unknown')}"),
                "error_type": error_type,
                "error_subtype": subtype,
                "k_errors_injected": k,
                "trust_score": result.summary.trust_score,
                "agg_score_T": result.summary.agg_score_T,
                "agg_score_E": result.summary.agg_score_E,
                "agg_score_B": result.summary.agg_score_B,
                "trust_score_ci": {
                    "lower": result.summary.trust_score_ci.lower,
                    "upper": result.summary.trust_score_ci.upper
                },
                "agg_score_T_ci": {
                    "lower": result.summary.agg_score_T_ci.lower,
                    "upper": result.summary.agg_score_T_ci.upper
                },
                "agg_score_E_ci": {
                    "lower": result.summary.agg_score_E_ci.lower,
                    "upper": result.summary.agg_score_E_ci.upper
                },
                "agg_score_B_ci": {
                    "lower": result.summary.agg_score_B_ci.lower,
                    "upper": result.summary.agg_score_B_ci.upper
                },
                "num_errors": len(result.errors),
                "errors": {
                    error_id: {
                        "type": error.type.value,
                        "subtype": error.subtype,
                        "severity_score": error.severity_score,
                        "severity_bucket": error.severity_bucket.value,
                        "explanation": error.explanation
                    }
                    for error_id, error in result.errors.items()
                },
                "spans": {
                    span_id: {
                        "start": span.start,
                        "end": span.end,
                        "type": span.type.value,
                        "subtype": span.subtype,
                        "explanation": span.explanation,
                        "severity_score": span.get_average_severity_score(),
                        "judge_count": len(span.analysis)
                    }
                    for span_id, span in (result.graded_spans.spans.items() if result.graded_spans else {}.items())
                },
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result_data)
            
            tqdm.write(f"  Sample {i+1}: TrustScore={result.summary.trust_score:.3f}, "
                      f"T={result.summary.agg_score_T:.3f}, "
                      f"E={result.summary.agg_score_E:.3f}, "
                      f"B={result.summary.agg_score_B:.3f}, "
                      f"Errors={len(result.errors)}")
            
        except Exception as e:
            tqdm.write(f"  Error processing sample {i+1}: {str(e)}")
            results.append({
                "sample_id": sample.get("sample_id", f"sample_{i}"),
                "unique_dataset_id": sample.get("unique_dataset_id", f"{sample.get('sample_id', 'unknown')}-{sample.get('model', 'unknown')}"),
                "error_type": error_type,
                "error_subtype": subtype,
                "k_errors_injected": k,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Sensitivity inference complete! Results saved to {output_path}")
    print(f"  Processed {len(results)} samples")
    
    return results

