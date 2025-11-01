"""
Run baseline TrustScore inference on original responses
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pipeline.orchestrator import TrustScorePipeline


def run_baseline_inference(
    samples: List[Dict[str, Any]],
    output_path: str,
    use_mock: bool = False,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Run TrustScore inference on original (unperturbed) responses.
    
    Args:
        samples: List of samples with 'prompt' and 'response'
        output_path: Path to save results
        use_mock: Whether to use mock components (for testing)
        api_key: API key for LLM providers
        
    Returns:
        List of TrustScore results
    """
    print("=" * 70)
    print("Step 1: Baseline TrustScore Inference")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = TrustScorePipeline(api_key=api_key, use_mock=use_mock)
    
    results = []
    
    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] Processing sample: {sample.get('sample_id', i)}")
        
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
                "baseline": True,
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
            
            print(f"  TrustScore: {result.summary.trust_score:.3f}")
            print(f"  T: {result.summary.agg_score_T:.3f}, "
                  f"E: {result.summary.agg_score_E:.3f}, "
                  f"B: {result.summary.agg_score_B:.3f}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results.append({
                "sample_id": sample.get("sample_id", f"sample_{i}"),
                "baseline": True,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Baseline inference complete! Results saved to {output_path}")
    print(f"  Processed {len(results)} samples")
    
    return results


if __name__ == "__main__":
    # Test with a small sample
    from .load_dataset import load_and_sample_dataset
    
    samples = load_and_sample_dataset(max_samples=3)
    results = run_baseline_inference(
        samples=samples,
        output_path="results/specificity_analysis/baseline_results.jsonl",
        use_mock=True  # Set to False for real inference
    )
