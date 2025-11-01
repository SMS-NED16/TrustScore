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
    
    results = []
    
    for i, sample in enumerate(tqdm(perturbed_samples, desc=f"Processing {error_type}_perturbed", unit="sample")):
        
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
                "error_type_injected": error_type,
                "perturbed": True,
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
                        "severity_bucket": error.severity_bucket.value
                    }
                    for error_id, error in result.errors.items()
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
                "error_type_injected": error_type,
                "perturbed": True,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Perturbed inference complete! Results saved to {output_path}")
    print(f"  Processed {len(results)} samples")
    
    return results
