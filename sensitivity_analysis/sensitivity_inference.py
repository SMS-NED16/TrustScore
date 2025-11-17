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
        batch_size = config.performance.max_batch_size
    else:
        pipeline = TrustScorePipeline(api_key=api_key, use_mock=use_mock)
        batch_size = 10  # Default batch size
    
    results = []
    
    # Process samples in batches for efficiency
    total_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(0, len(samples), batch_size), 
                         desc=f"Processing {error_type}_{subtype}_k{k} batches", 
                         unit="batch",
                         total=total_batches):
        batch = samples[batch_idx:batch_idx + batch_size]
        
        # Prepare batch inputs for process_batch
        batch_inputs = []
        for sample in batch:
            batch_inputs.append({
                'prompt': sample['prompt'],
                'response': sample['response'],
                'model': sample.get('model', 'unknown'),
                'generated_on': datetime.now()
            })
        
        try:
            # Process batch
            batch_results = pipeline.process_batch(batch_inputs)
            
            # Format results for storage
            for j, result in enumerate(batch_results):
                sample = batch[j]
                sample_idx = batch_idx + j
                
                if result is None:
                    # Handle failed processing
                    tqdm.write(f"  Error processing sample {sample_idx+1}: Processing failed")
                    results.append({
                        "sample_id": sample.get("sample_id", f"sample_{sample_idx}"),
                        "unique_dataset_id": sample.get("unique_dataset_id", f"{sample.get('sample_id', 'unknown')}-{sample.get('model', 'unknown')}"),
                        "error_type": error_type,
                        "error_subtype": subtype,
                        "k_errors_injected": k,
                        "error": "Processing failed",
                        "timestamp": datetime.now().isoformat()
                    })
                    continue
                
                # Format successful result
                result_data = {
                    "sample_id": sample.get("sample_id", f"sample_{sample_idx}"),
                    "unique_dataset_id": sample.get("unique_dataset_id", f"{sample.get('sample_id', 'unknown')}-{sample.get('model', 'unknown')}"),
                    "error_type": error_type,
                    "error_subtype": subtype,
                    "k_errors_injected": k,
                    "trust_score": result.summary.trust_score,
                    "trust_quality": result.summary.trust_quality,
                    "agg_score_T": result.summary.agg_score_T,
                    "agg_quality_T": result.summary.agg_quality_T,
                    "agg_score_E": result.summary.agg_score_E,
                    "agg_quality_E": result.summary.agg_quality_E,
                    "agg_score_B": result.summary.agg_score_B,
                    "agg_quality_B": result.summary.agg_quality_B,
                    "trust_score_ci": {
                        "lower": result.summary.trust_score_ci.lower,
                        "upper": result.summary.trust_score_ci.upper
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
                
                tqdm.write(f"  Sample {sample_idx+1}: TrustScore={result.summary.trust_score:.3f}, "
                          f"T={result.summary.agg_quality_T:.2f}, "
                          f"E={result.summary.agg_quality_E:.2f}, "
                          f"B={result.summary.agg_quality_B:.2f}, "
                          f"Errors={len(result.errors)}")
                
        except Exception as e:
            tqdm.write(f"  Error processing batch {batch_idx//batch_size + 1}: {str(e)}")
            # Add error results for failed batch
            for j in range(len(batch)):
                sample = batch[j]
                sample_idx = batch_idx + j
                results.append({
                    "sample_id": sample.get("sample_id", f"sample_{sample_idx}"),
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

